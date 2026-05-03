"""Build an augmented TLS patch cache that includes bg-only train slides.

The base cache `/home/ubuntu/local_data/tls_patch_dataset_min4096.pt` has
51 829 TLS-positive patches from 604 mask-having slides. This script
samples N random patches per bg-only train slide (167 slides) and
appends them with all-zero masks → augmented cache for v3.52.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gncaf_dataset import build_gncaf_split

BASE = Path("/home/ubuntu/local_data/tls_patch_dataset_min4096.pt")
OUT = Path("/home/ubuntu/local_data/tls_patch_dataset_min4096_with_bgneg.pt")
N_PER_BG_SLIDE = 12          # random bg patches per bg-only slide

bundle = torch.load(BASE, weights_only=False)
print(f"Base cache: {bundle['features'].shape[0]} patches, "
      f"{len(set(bundle['slide_ids']))} unique slides")

train_entries, _ = build_gncaf_split()
neg_entries = [e for e in train_entries if not e.get("mask_path") and e.get("zarr_path")]
print(f"bg-only train slides: {len(neg_entries)}")

rng = np.random.default_rng(42)

extra_features: list[torch.Tensor] = []
extra_masks: list[torch.Tensor] = []
extra_slide_ids: list[str] = []
extra_cancer_types: list[str] = []
extra_patch_idx: list[int] = []

for k, e in enumerate(neg_entries):
    z = zarr.open(e["zarr_path"], mode="r")
    feats = np.asarray(z["features"][:])
    n = feats.shape[0]
    if n == 0:
        continue
    n_take = min(N_PER_BG_SLIDE, n)
    idx = rng.choice(n, size=n_take, replace=False)
    extra_features.append(torch.from_numpy(feats[idx]).float())
    extra_masks.append(torch.zeros(n_take, 256, 256, dtype=torch.uint8))
    extra_slide_ids.extend([e["slide_id"]] * n_take)
    extra_cancer_types.extend([e["cancer_type"]] * n_take)
    extra_patch_idx.extend([int(i) for i in idx])
    if k % 30 == 0:
        print(f"  [{k+1}/{len(neg_entries)}] {e['slide_id'][:25]} +{n_take}")

print(f"\nExtra patches added: {sum(t.shape[0] for t in extra_features)}")

new_features = torch.cat([bundle["features"], torch.cat(extra_features)], dim=0)
new_masks = torch.cat([bundle["masks"], torch.cat(extra_masks)], dim=0)
new_slide_ids = list(bundle["slide_ids"]) + extra_slide_ids
new_cancer_types = list(bundle["cancer_types"]) + extra_cancer_types
new_patch_idx = torch.cat([bundle["patch_idx"], torch.tensor(extra_patch_idx, dtype=torch.int32)])

out = {
    "features": new_features,
    "masks": new_masks,
    "slide_ids": new_slide_ids,
    "cancer_types": new_cancer_types,
    "patch_idx": new_patch_idx,
}
torch.save(out, OUT)
print(f"\nWrote {OUT}: {new_features.shape[0]} patches, "
      f"{len(set(new_slide_ids))} unique slides")
