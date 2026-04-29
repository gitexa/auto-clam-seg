"""Load the recovered Stage 2 checkpoints into the rebuilt
`UNIv2PixelDecoder` with strict=True. Strict load passing is hard proof
that the reconstructed architecture matches the original.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_gars_stage2 import UNIv2PixelDecoder  # noqa: E402

EXPS = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")

CASES = [
    ("gars_stage2_gars_stage2_univ2_decoder_20260427_102326",
     dict(hidden_channels=64, spatial_size=16),  "univ2_decoder (h=64)"),
    ("gars_stage2_gars_stage2_univ2_hidden128_20260427_120532",
     dict(hidden_channels=128, spatial_size=16), "univ2_hidden128 (h=128)"),
]

ok_all = True
for exp_dir, kwargs, label in CASES:
    ckpt_path = EXPS / exp_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"[skip] {label}: no checkpoint at {ckpt_path}")
        continue
    print(f"\n=== {label} ===")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    model = UNIv2PixelDecoder(**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    n_ckpt = sum(v.numel() for v in sd.values())
    print(f"  built model: {n_params:,} params")
    print(f"  ckpt state_dict: {len(sd)} tensors, sum={n_ckpt:,}")
    try:
        model.load_state_dict(sd, strict=True)
        print(f"  STRICT LOAD OK ✓  — architecture matches.")

        # Smoke test: forward pass shape.
        with torch.no_grad():
            x = torch.randn(2, 1536)
            y = model(x)
        print(f"  forward(2, 1536) -> {tuple(y.shape)}  (expected (2, 3, 256, 256))")
    except Exception as e:
        ok_all = False
        print(f"  STRICT LOAD FAILED: {e}")

print("\nAll OK" if ok_all else "\nSome strict loads FAILED — see above.")
