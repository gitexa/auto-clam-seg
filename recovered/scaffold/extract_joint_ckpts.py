"""Extract Stage 1 + Stage 2 ckpts from a joint training output for use
with the standard eval pipeline.

Joint trainer saves `best_checkpoint.pt` containing:
  - stage1_state_dict
  - stage2_state_dict
  - epoch, val_loss, config

eval_gars_cascade.py expects separate ckpt files for stage1 and
stage2_region; each one has its own `model_state_dict` and optional
`config` for the per-model class fields.

This script extracts the two into compatible files in the same dir:
  - stage1_extracted.pt   (model_state_dict = stage1_state_dict)
  - stage2_extracted.pt   (model_state_dict = stage2_state_dict)

Usage:
    python extract_joint_ckpts.py <joint_ckpt_dir>
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joint_dir", type=Path)
    args = ap.parse_args()

    ckpt_path = args.joint_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"missing {ckpt_path}")
        sys.exit(1)

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {})

    # Stage 1 ckpt — match the format `load_stage1` expects.
    s1_ckpt = {
        "model_state_dict": obj["stage1_state_dict"],
        # GraphTLSDetector defaults are fine; loader auto-detects.
        "config": {
            "model": {
                "in_dim": 1536, "hidden_dim": 256, "n_hops": 5,
                "gnn_type": "gatv2", "dropout": 0.1, "gat_heads": 4,
            },
        },
    }
    s1_out = args.joint_dir / "stage1_extracted.pt"
    torch.save(s1_ckpt, s1_out)
    print(f"wrote {s1_out}")

    # Stage 2 ckpt — match the format `load_stage2_region` expects.
    s2_ckpt = {
        "model_state_dict": obj["stage2_state_dict"],
        "config": {
            "model": {
                "uni_dim": 1536, "gat_dim": 256, "hidden_channels": 64,
                "n_classes": 3, "grid_n": 3, "rgb_pretrained": True,
                "freeze_rgb_encoder": False, "head_mode": "argmax",
            },
        },
    }
    s2_out = args.joint_dir / "stage2_extracted.pt"
    torch.save(s2_ckpt, s2_out)
    print(f"wrote {s2_out}")
    print(f"\nready for eval:")
    print(f"  stage1={s1_out}")
    print(f"  stage2_region={s2_out}")


if __name__ == "__main__":
    main()
