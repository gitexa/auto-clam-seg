"""G1 — strict-load gate for v3.36 NeighborhoodPixelDecoder.

Run:
    python verify_neighborhood_ckpt.py [--ckpt path/to/best_checkpoint.pt]

Without --ckpt, just builds the model from defaults and runs a dummy forward
to check the (B, 9, 1536) → (B, 3, 768, 768) shape contract + param count.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gars_neighborhood import NeighborhoodPixelDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeighborhoodPixelDecoder(
        in_dim=1536, bottleneck=512, hidden_channels=args.hidden,
        spatial_size=16, n_classes=3,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NeighborhoodPixelDecoder h={args.hidden}: {n_params:,} params")

    if args.ckpt:
        obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("model_state_dict", obj)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"strict=False load → missing={len(miss)} unexpected={len(unexp)}")
        # Now require strict.
        model.load_state_dict(sd, strict=True)
        print("strict=True OK")

    B = 2
    feats = torch.randn(B, 9, 1536, device=device)
    valid = torch.ones(B, 9, dtype=torch.bool, device=device)
    valid[0, 8] = False  # sanity: pad embed substitution path
    with torch.no_grad():
        y = model(feats, valid)
    print(f"forward: in {tuple(feats.shape)} → out {tuple(y.shape)}")
    assert y.shape == (B, 3, 768, 768), f"unexpected output shape {y.shape}"
    print("All OK")


if __name__ == "__main__":
    main()
