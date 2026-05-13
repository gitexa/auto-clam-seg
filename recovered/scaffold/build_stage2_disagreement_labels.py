"""Strategy-1 helper: build per-patch (slide_id, patch_idx, label) labels
where label encodes whether Stage 2 SUCCESSFULLY segments Stage 1's selection.

For each Stage-1-selected patch on training slides:
  - load Stage 2's prediction tile (256x256, class 0/1/2)
  - load the per-patch HookNet GT tile (from tls_patch_dataset.pt)
  - compute per-tile pixel TLS Dice
  - label = 1 if patch was Stage-2-confirmed-positive (Stage 2 produced
    enough TLS pixels matching GT); label = 0 if Stage 2 disagreed
    (FP or under-segmented).

The output CSV has one row per Stage-1-selected patch:
  slide_id, patch_idx, s1_prob, s2_tls_pixels, s2_gc_pixels,
  gt_tls_pixels, gt_gc_pixels, tls_dice, gc_dice, label, label_reason

Used to fine-tune Stage 1 with the aux loss (see Strategy 1 in
plans/can-you-start-the-cuddly-minsky.md).

Run:
    python build_stage2_disagreement_labels.py \\
        --fold 0  # which fold's TRAIN set to label (folds 1-4 union)
        --out-csv stage2_disagreement_labels_for_fold0.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")

import prepare_segmentation as ps  # noqa: E402
from gncaf_dataset import slide_wsi_path  # noqa: E402
from eval_gars_cascade import (  # noqa: E402
    load_stage1, load_stage2_region, cascade_one_slide_region,
)
from eval_gars_gncaf_transunet import _load_features_and_graph  # noqa: E402


CKPT_STAGE1 = (
    "/home/ubuntu/ahaas-persistent-std-tcga/experiments/"
    "gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt"
)
CKPT_CASC_S2 = (
    "/home/ubuntu/ahaas-persistent-std-tcga/experiments/"
    "gars_region_v3.37_full_20260502_144124/best_checkpoint.pt"
)
PATCH_SIZE = 256
THRESHOLD = 0.5

# Label thresholds
TLS_DICE_POS = 0.5     # >= this → Stage 2 confirmed
TLS_DICE_NEG = 0.1     # <  this → Stage 2 disagreed
MIN_TLS_PX_POS = 100   # at least this many TLS pixels for an "agreement"


def per_tile_dice(pred_tile: np.ndarray, gt_tile: np.ndarray,
                  cls: int, eps: float = 1e-6) -> tuple[float, int, int]:
    """Return (Dice, #pred pixels, #gt pixels) for a binary class."""
    p = (pred_tile == cls).astype(np.uint8)
    t = (gt_tile == cls).astype(np.uint8)
    n_p = int(p.sum())
    n_t = int(t.sum())
    if n_t == 0:
        # Dice on empty target: 1 if pred also empty, 0 otherwise.
        return (1.0 if n_p == 0 else 0.0, n_p, n_t)
    inter = int((p * t).sum())
    return ((2.0 * inter + eps) / (n_p + n_t + eps), n_p, n_t)


def classify(tls_dice: float, gc_dice: float, n_tls_pred: int,
             gt_tls_px: int, gt_gc_px: int) -> tuple[int, str]:
    """Decide aux label. Returns (label, reason)."""
    # 1) Slide-level FP detection: GT has no TLS but Stage 2 fires TLS
    if gt_tls_px == 0 and gt_gc_px == 0:
        # Stage 1 selected a patch with no GT — by definition an FP target.
        if n_tls_pred >= MIN_TLS_PX_POS:
            return 0, "fp_s2_fired"
        # Stage 2 produced empty mask on Stage-1-selected patch with empty GT.
        # Stage 1 still wasted attention on this patch.
        return 0, "fp_s2_empty"

    # 2) GT has TLS or GC. Did Stage 2 capture it?
    overlap = max(tls_dice, gc_dice)
    if overlap >= TLS_DICE_POS and n_tls_pred >= MIN_TLS_PX_POS:
        return 1, "confirmed"
    if overlap < TLS_DICE_NEG:
        return 0, "missed"
    # 3) Ambiguous middle band — unlabelled. Caller filters these out.
    return -1, "ambiguous"


def build_labels(fold: int, out_csv: Path, limit_slides: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # TRAIN slides for the given fold: union of folds != fold.
    entries = ps.build_slide_entries()
    folds, _ = ps.create_splits(entries, k_folds=5, seed=42)
    train_entries: list[dict] = []
    for f in range(5):
        if f != fold:
            train_entries.extend(folds[f])
    if limit_slides:
        train_entries = train_entries[:limit_slides]
    print(f"Building labels for fold-{fold} training set: {len(train_entries)} slides")

    # Load per-tile GT cache.
    print("Loading TLS-patch cache for GT tiles...")
    from tls_patch_dataset import build_tls_patch_dataset  # noqa: E402
    bundle = build_tls_patch_dataset(
        cache_path="/home/ubuntu/local_data/tls_patch_dataset.pt"
    )
    bundle_masks = bundle["masks"]
    patch_mask_lookup: dict[str, dict[int, np.ndarray]] = {}
    for ci, sid in enumerate(bundle["slide_ids"]):
        short = sid.split(".")[0]
        pi = int(bundle["patch_idx"][ci])
        patch_mask_lookup.setdefault(short, {})[pi] = np.asarray(bundle_masks[ci])

    # Load cascade.
    print("Loading cascade...")
    stage1 = load_stage1(CKPT_STAGE1, device)
    stage2 = load_stage2_region(CKPT_CASC_S2, device)

    # Open CSV.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "slide_id", "patch_idx", "s1_prob",
            "s2_tls_pixels", "s2_gc_pixels",
            "gt_tls_pixels", "gt_gc_pixels",
            "tls_dice", "gc_dice", "label", "label_reason",
        ])

        # Process per-slide.
        t0 = time.time()
        for i_slide, entry in enumerate(train_entries):
            short_id = entry["slide_id"].split(".")[0]
            wsi_path = slide_wsi_path(entry)
            if not wsi_path or not Path(wsi_path).exists():
                continue
            feats_np, coords_np, edge_np = _load_features_and_graph(entry["zarr_path"])
            features = torch.from_numpy(feats_np)        # CPU
            coords = torch.from_numpy(coords_np).long()
            edge_index = torch.from_numpy(edge_np).long().to(device)

            # Stage 1 probs (need them per patch).
            with torch.no_grad():
                s1_logits = stage1(features.to(device), edge_index)
                s1_probs = torch.sigmoid(s1_logits).cpu().numpy()

            # Run full cascade Stage 2 — returns pred_tiles dict[idx → tile mask].
            try:
                _, _, pred_tiles = cascade_one_slide_region(
                    stage1, stage2, features, coords, edge_index,
                    threshold=THRESHOLD, s2_batch=8, device=device,
                    wsi_path=wsi_path, return_cell_probs=False,
                )
            except Exception as e:
                print(f"  [{i_slide+1}/{len(train_entries)}] {short_id}: cascade failed: {e}")
                continue

            slide_lookup = patch_mask_lookup.get(short_id, {})
            n_rows_this_slide = 0
            for pi, pred_tile in pred_tiles.items():
                gt = slide_lookup.get(int(pi))
                if gt is None:
                    # No GT cached for this patch → assume empty (slide-level GT-neg).
                    gt = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                else:
                    gt = np.asarray(gt)
                tls_dice, n_tls_p, n_tls_t = per_tile_dice(pred_tile, gt, 1)
                gc_dice, n_gc_p, n_gc_t = per_tile_dice(pred_tile, gt, 2)
                label, reason = classify(tls_dice, gc_dice, n_tls_p, n_tls_t, n_gc_t)
                if label == -1:
                    continue   # ambiguous — skip
                writer.writerow([
                    short_id, int(pi), float(s1_probs[int(pi)]),
                    n_tls_p, n_gc_p, n_tls_t, n_gc_t,
                    f"{tls_dice:.4f}", f"{gc_dice:.4f}", label, reason,
                ])
                n_rows_this_slide += 1
            fh.flush()

            if (i_slide + 1) % 5 == 0:
                print(f"  [{i_slide+1}/{len(train_entries)}] {short_id}: "
                      f"{n_rows_this_slide} rows ({(time.time()-t0)/(i_slide+1):.1f}s/slide)")

    print(f"\nWrote {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out-csv", type=Path,
                        default=Path("/home/ubuntu/auto-clam-seg/recovered/scaffold/"
                                     "stage2_disagreement_labels_for_fold0.csv"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_labels(args.fold, args.out_csv, limit_slides=args.limit)


if __name__ == "__main__":
    main()
