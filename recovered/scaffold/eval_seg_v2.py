"""seg_v2.0 per-slide inference — emits gncaf_agg.json compatible rows
so build_arch_comparison can pool seg_v2.0 alongside Cascade / GNCAF.

Reads each fold's best checkpoint from
  /home/ubuntu/ahaas-persistent-std-tcga/experiments/seg_v2.0_tls_only_5fold_31aaec0c/fold_{K}/best_checkpoint.pt
and writes per-slide rows to
  /home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_gncaf_eval_seg_v2_fullcohort_fold{K}_eval_shard0/gncaf_agg.json

Per-slide row schema (matches GNCAF eval):
  slide_id, cancer_type, gt_n_tls, gt_n_gc,
  n_tls_pred, n_gc_pred, tls_dice_grid, gc_dice_grid,
  gt_negative
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make profile-clam importable.
sys.path.insert(0, "/home/ubuntu/profile-clam")

from prepare_segmentation import (  # type: ignore
    build_slide_entries, build_mask_cache, create_splits,
    TLSSegmentationDataset, count_instances, count_from_center_heatmap,
    PATCH_SIZE, FEATURE_DIM,
)
from models.segmentation_decoder import GNNSegmentationDecoder  # type: ignore

import math
import torch.nn as nn


class GNNSegV2(GNNSegmentationDecoder):
    """seg_v2.0 saved 5-fold ckpts use all-5×5 depthwise decoder kernels —
    differs from current default (5 then 3). Override decoder + heads to
    match the saved state_dict shape."""

    def __init__(self, **kw):
        super().__init__(**kw)
        h = kw["hidden_dim"]
        n_blocks = int(math.log2(kw["upsample_factor"]))
        layers = []
        in_ch = h
        for i in range(n_blocks):
            out_ch = max(h // (2 ** (i + 1)), 32)
            layers.append(nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch))
            layers.append(nn.Conv2d(in_ch, out_ch, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            in_ch = out_ch
        self.decoder = nn.Sequential(*layers)
        self.head_tls = nn.Conv2d(in_ch, 1, 1)
        self.head_gc = nn.Conv2d(in_ch, 1, 1)
        self.center_head_tls = nn.Conv2d(in_ch, 1, 1)
        self.center_head_gc = nn.Conv2d(in_ch, 1, 1)
        self.offset_head = nn.Conv2d(in_ch, 2, 1)


import argparse

# Two seg_v2.0 5-fold ckpt sets are available:
#   tls_only_5fold     → simpler training recipe (TLS-only Dice term)
#   dual_tls_gc_5fold  → production recipe (dice_weight=2, gc_dice_weight=5,
#                         focal_weight=1) — better full-cohort metrics
SEG_V2_DIRS = {
    "seg_v2": Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments/seg_v2.0_tls_only_5fold_31aaec0c"),
    "seg_v2_dual": Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments/seg_v2.0_dual_tls_gc_5fold_57e12399"),
}
OUT_ROOT = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")

# Match train_segmentation.py defaults
HIDDEN_DIM = 384
GNN_LAYERS = 0
GNN_HEADS = 4
UPSAMPLE_FACTOR = 4   # The saved 5-fold ckpts trained with factor=4 (2 decoder blocks, 5×5 dw)
N_CLASSES = 3
DROPOUT = 0.1
SEED = 42
K_FOLDS = 5


def per_slide_metrics(model, sample, device):
    features = sample["features"].to(device)
    coords = sample["coords"].to(device)
    edge_index = sample["edge_index"]
    if edge_index is None:
        return None
    edge_index = edge_index.to(device)
    mask = sample["mask"].to(device).unsqueeze(0)  # (1,1,H,W)

    out = model(features, coords, edge_index, return_attention_weights=False)
    logits_tls = out["logits_tls"]
    logits_gc = out["logits_gc"]
    if mask.shape[2:] != logits_tls.shape[2:]:
        mask = F.interpolate(mask, size=logits_tls.shape[2:], mode="nearest")

    p_tls = (torch.sigmoid(logits_tls).squeeze().cpu().numpy() > 0.5).astype(float)
    p_gc = (torch.sigmoid(logits_gc).squeeze().cpu().numpy() > 0.5).astype(float)
    t = mask.squeeze().cpu().numpy()
    t_tls = (t >= 1).astype(float)
    t_gc = (t == 2).astype(float)

    def dice(p, t):
        if t.sum() == 0:
            return 1.0 if p.sum() == 0 else 0.0
        inter = float((p * t).sum())
        denom = float(p.sum() + t.sum())
        return (2.0 * inter) / (denom + 1e-8)

    tls_dice = dice(p_tls, t_tls)
    gc_dice = dice(p_gc, t_gc)

    # Instance counts via center heatmap (matches train-time eval).
    center_tls_pred = torch.sigmoid(out["center_tls"])[0, 0].cpu().numpy()
    center_gc_pred = torch.sigmoid(out["center_gc"])[0, 0].cpu().numpy()
    n_tls_pred = int(count_from_center_heatmap(center_tls_pred, semantic_pred=p_tls))
    n_gc_pred = int(count_from_center_heatmap(center_gc_pred, semantic_pred=p_gc))
    gt_n_tls = int(count_instances(t, class_id=None))   # any non-bg
    gt_n_gc = int(count_instances(t, class_id=2))
    gt_negative = bool(sample.get("_gt_negative", False))

    return {
        "slide_id": sample["slide_id"].split(".")[0],
        "cancer_type": sample["cancer_type"],
        "gt_n_tls": gt_n_tls,
        "gt_n_gc": gt_n_gc,
        "n_tls_pred": n_tls_pred,
        "n_gc_pred": n_gc_pred,
        "tls_dice_grid": tls_dice,
        "gc_dice_grid": gc_dice,
        "gt_negative": gt_negative,
    }


def eval_fold(fold_idx: int, val_entries, mask_dict, device, seg_v2_dir):
    ckpt_path = seg_v2_dir / f"fold_{fold_idx}" / "best_checkpoint.pt"
    print(f"  loading {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj["model_state_dict"]
    n_params = sum(v.numel() for v in sd.values())

    model = GNNSegV2(
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        gnn_layers=GNN_LAYERS, gnn_heads=GNN_HEADS,
        upsample_factor=UPSAMPLE_FACTOR, n_classes=N_CLASSES,
        dropout=DROPOUT, patch_size=PATCH_SIZE,
    )
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f"  Loaded seg_v2.0 fold {fold_idx} ({n_params:,} params, "
          f"epoch={obj.get('epoch')}, best_dice={float(obj.get('best_dice', 0)):.4f})")

    ds = TLSSegmentationDataset(val_entries, mask_dict, UPSAMPLE_FACTOR)
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            sample["_gt_negative"] = val_entries[i].get("mask_path") is None
            row = per_slide_metrics(model, sample, device)
            if row is not None:
                rows.append(row)
            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{len(ds)}] {row['slide_id'] if row else 'skip'} "
                      f"({(time.time()-t0)/(i+1):.1f}s/slide)")
    return rows, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(SEG_V2_DIRS.keys()), default="seg_v2_dual",
                        help="Which seg_v2.0 5-fold ckpt set to evaluate.")
    args = parser.parse_args()
    seg_v2_dir = SEG_V2_DIRS[args.variant]
    label = args.variant
    print(f"variant={args.variant}  seg_v2_dir={seg_v2_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Building entries + mask_dict (this is shared across folds)...")
    entries = build_slide_entries()
    mask_dict = build_mask_cache(entries, UPSAMPLE_FACTOR)
    folds, test_entries = create_splits(entries, K_FOLDS, SEED)
    print(f"Cohort: {len(entries)} slides, "
          f"{sum(1 for e in entries if e['mask_path'])} with mask, "
          f"{K_FOLDS} folds, test_entries={len(test_entries)}")

    for fold_idx in range(K_FOLDS):
        val_entries = folds[fold_idx]
        print(f"\n=== fold {fold_idx} ({len(val_entries)} val slides) ===")
        rows, n_params = eval_fold(fold_idx, val_entries, mask_dict, device, seg_v2_dir)

        out_dir = OUT_ROOT / f"gars_gncaf_eval_{label}_fullcohort_fold{fold_idx}_eval_shard0"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "approach": label,
            "fold": fold_idx,
            "n_params": n_params,
            "n_slides": len(rows),
            "n_gt_pos": sum(1 for r in rows if not r["gt_negative"]),
            "n_gt_neg": sum(1 for r in rows if r["gt_negative"]),
            "per_slide": rows,
        }
        (out_dir / "gncaf_agg.json").write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_dir / 'gncaf_agg.json'} ({len(rows)} slides)")


if __name__ == "__main__":
    main()
