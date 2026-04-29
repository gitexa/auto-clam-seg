"""Load every recovered Stage 1 checkpoint into the rebuilt
`GraphTLSDetector` with strict=True. Strict load passing is hard proof that
the reconstructed architecture matches the original.

We also print param counts to compare against the values printed in the
recovered output.log files.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_gars_stage1 import GraphTLSDetector  # noqa: E402

EXPS = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")

CASES = [
    ("gars_stage1_gars_stage1_gatv2_3hop_v2_20260426_230650",
     dict(n_hops=3, gnn_type="gatv2"), "GATv2 3-hop"),
    ("gars_stage1_gars_stage1_gcn3hop_v2_20260426_230650",
     dict(n_hops=3, gnn_type="gcn"), "GCN 3-hop (v2)"),
    ("gars_stage1_gars_stage1_no_gnn_v2_20260426_230650",
     dict(n_hops=0, gnn_type="gcn"), "no-GNN"),
    ("gars_stage1_gars_stage1_gcn3hop_512px_v2_20260426_230611",
     dict(n_hops=3, gnn_type="gcn"), "GCN 3-hop 512px"),
]

ok_all = True
for exp_dir, kwargs, label in CASES:
    ckpt_path = EXPS / exp_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"[skip] {label}: no checkpoint at {ckpt_path}")
        continue
    print(f"\n=== {label} ===")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    model = GraphTLSDetector(**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  built model: {n_params:,} params")
    print(f"  ckpt state_dict has {len(sd)} tensors, sum={sum(v.numel() for v in sd.values()):,}")
    try:
        model.load_state_dict(sd, strict=True)
        print(f"  STRICT LOAD OK ✓  — architecture matches.")
    except Exception as e:
        ok_all = False
        print(f"  STRICT LOAD FAILED: {e}")

print("\nAll OK" if ok_all else "\nSome strict loads FAILED — see above.")
