"""Inspect the surviving Stage 1 GATv2 checkpoint to recover the exact
architecture (state_dict keys + shapes + parameter count)."""
from pathlib import Path

import torch

CKPT = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments/"
            "gars_stage1_gars_stage1_gatv2_3hop_v2_20260426_230650/best_checkpoint.pt")

obj = torch.load(CKPT, map_location="cpu", weights_only=False)
print("Top-level keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))

if isinstance(obj, dict) and "model" in obj:
    sd = obj["model"]
elif isinstance(obj, dict) and "state_dict" in obj:
    sd = obj["state_dict"]
elif isinstance(obj, dict) and any(k.endswith(".weight") for k in obj):
    sd = obj
else:
    sd = obj.get("model_state_dict", obj)

if isinstance(obj, dict):
    for k, v in obj.items():
        if not isinstance(v, dict) and not hasattr(v, "shape"):
            print(f"  meta {k}: {v!r}"[:200])

print("\nstate_dict shapes:")
total = 0
for k, v in sd.items():
    n = v.numel()
    total += n
    print(f"  {k:60s}  {tuple(v.shape)}  ({n})")
print(f"\nTOTAL params: {total:,}")
