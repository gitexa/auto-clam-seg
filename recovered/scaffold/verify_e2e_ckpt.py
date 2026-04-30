"""Load the recovered `qy3pj74h` GraphEnrichedDecoder checkpoint into the
rebuilt model with strict=True. Strict load passing is hard proof that
the architecture rebuild matches the original (10 152 322 params,
mDice=0.7202).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_gars_e2e import GraphEnrichedDecoder  # noqa: E402

CKPT = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments/"
            "gars_unified_gars_unified_gatv2_decoder_20260427_225200/"
            "best_checkpoint.pt")

print("=== GraphEnrichedDecoder vs qy3pj74h checkpoint ===")
obj = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = obj["model_state_dict"]
print(f"  ckpt val_metrics: {obj.get('val_metrics')}")
print(f"  ckpt epoch: {obj.get('epoch')}")
print(f"  ckpt state_dict: {len(sd)} tensors, "
      f"sum={sum(v.numel() for v in sd.values()):,}")

model = GraphEnrichedDecoder()
n_params = sum(p.numel() for p in model.parameters())
print(f"  built model: {n_params:,} params")

try:
    model.load_state_dict(sd, strict=True)
    print(f"  STRICT LOAD OK ✓  — architecture matches.")
    # Smoke test forward.
    with torch.no_grad():
        x = torch.randn(2, 1536)
        edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g = model.graph_context(x, edges)
        y = model.decode_patches(x, g)
    print(f"  forward(2 patches) -> graph={tuple(g.shape)}, "
          f"decode={tuple(y.shape)}  (expected (2,3,256,256))")
    print("\nAll OK")
except Exception as e:
    print(f"  STRICT LOAD FAILED: {e}")
    print("\nFAILED")
    sys.exit(1)
