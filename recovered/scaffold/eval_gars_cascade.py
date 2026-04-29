"""GARS cascade eval: Stage 1 → threshold → Stage 2 → slide-level metrics.

Reproduces the recovered `gars_cascade_eval.log` and
`gars_cascade_low_thresh.log` outputs (114-slide val, 6 thresholds).

Pipeline per slide:
  1. Stage 1 (GraphTLSDetector): score every patch → TLS probability.
  2. Threshold at THR → select TLS-positive patch indices.
  3. Stage 2 (UNIv2PixelDecoder): for each selected patch, decode the
     UNI-v2 feature into a 256×256 3-class mask.
  4. Stitch per-patch predictions into a slide-level prediction grid at
     patch-grid resolution (one cell per patch). Compare to the cached
     mask at the same resolution → mDice / TLS dice / GC dice.
  5. Connected-component counting on the slide-level grid for TLS / GC
     instance counts → Spearman vs ground-truth counts.

Run:
    python eval_gars_cascade.py \
        --stage1 .../experiments/gars_stage1_..._gatv2_3hop_v2_.../best_checkpoint.pt \
        --stage2 .../experiments/gars_stage2_..._univ2_decoder_.../best_checkpoint.pt \
        --thresholds 0.05 0.10 0.20 0.30 0.40 0.50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/ubuntu/profile-clam")
from train_gars_stage1 import GraphTLSDetector  # noqa: E402
from train_gars_stage2 import UNIv2PixelDecoder  # noqa: E402
import prepare_segmentation as ps  # noqa: E402

PATCH_SIZE = 256


# ─── Checkpoint loaders ────────────────────────────────────────────────


def load_stage1(ckpt_path: str, device: torch.device) -> GraphTLSDetector:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    model = GraphTLSDetector(
        in_dim=1536,
        hidden_dim=cfg.get("hidden_dim", 256),
        n_hops=cfg.get("n_hops", 3),
        gnn_type=cfg.get("gnn_type", "gatv2"),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_stage2(ckpt_path: str, device: torch.device) -> UNIv2PixelDecoder:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    model = UNIv2PixelDecoder(
        in_dim=1536,
        bottleneck=cfg.get("bottleneck", 512),
        hidden_channels=cfg.get("hidden_channels", 64),
        spatial_size=cfg.get("spatial_size", 16),
        n_classes=3,
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


# ─── Per-slide cascade ─────────────────────────────────────────────────


@torch.no_grad()
def cascade_one_slide(
    stage1: GraphTLSDetector,
    stage2: UNIv2PixelDecoder,
    features: torch.Tensor,        # (N, 1536)
    coords: torch.Tensor,          # (N, 2) — patch top-left in slide px
    edge_index: torch.Tensor,      # (2, E)
    threshold: float,
    device: torch.device,
    s2_batch: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns slide-level prediction grids at patch resolution.

    Each cell of the grid is one patch's argmax class (0/1/2).
    Two grids are returned: the **patch-class** grid (argmax over all
    patches that survived Stage 1; non-selected patches set to 0) and a
    **fine-grained** grid that takes the most-frequent class within each
    patch's 256×256 mask. For dice metrics we use the fine-grained one
    averaged at patch resolution; for counting we use connected
    components on the fine-grained grid.
    """
    n = features.shape[0]
    grid_x = (coords[:, 0] / PATCH_SIZE).long()
    grid_y = (coords[:, 1] / PATCH_SIZE).long()
    grid_x -= grid_x.min()
    grid_y -= grid_y.min()
    H = int(grid_y.max().item()) + 1
    W = int(grid_x.max().item()) + 1

    # Stage 1 — score every patch
    s1_logits = stage1(features.to(device), edge_index.to(device))
    s1_probs = torch.sigmoid(s1_logits).cpu().numpy()
    selected = np.where(s1_probs > threshold)[0]

    # Stage 2 — decode selected patches in batches; keep argmax 256×256
    pred_class_per_patch = np.zeros(n, dtype=np.int64)        # 0 by default (bg)
    pred_argmax_tile_majority = np.zeros(n, dtype=np.int64)   # majority class in tile
    n_tls_px = np.zeros(n, dtype=np.int64)
    n_gc_px = np.zeros(n, dtype=np.int64)
    for s in range(0, len(selected), s2_batch):
        batch_idx = selected[s : s + s2_batch]
        feats = features[batch_idx].to(device)
        logits = stage2(feats)                           # (B, 3, 256, 256)
        argmax = logits.argmax(dim=1).cpu().numpy()      # (B, 256, 256)
        for b, i in enumerate(batch_idx):
            tile = argmax[b]
            n_tls_px[i] = int((tile == 1).sum())
            n_gc_px[i] = int((tile == 2).sum())
            pred_argmax_tile_majority[i] = int(np.bincount(tile.ravel(), minlength=3).argmax())
            # Patch-level summary: GC if any GC pixels, else TLS if any TLS, else bg.
            if n_gc_px[i] > 0:
                pred_class_per_patch[i] = 2
            elif n_tls_px[i] > 0:
                pred_class_per_patch[i] = 1

    # Stitch into patch grids
    grid_class = np.zeros((H, W), dtype=np.int64)
    grid_majority = np.zeros((H, W), dtype=np.int64)
    grid_n_tls = np.zeros((H, W), dtype=np.int64)
    grid_n_gc = np.zeros((H, W), dtype=np.int64)
    for i in range(n):
        gx, gy = int(grid_x[i]), int(grid_y[i])
        grid_class[gy, gx] = pred_class_per_patch[i]
        grid_majority[gy, gx] = pred_argmax_tile_majority[i]
        grid_n_tls[gy, gx] = n_tls_px[i]
        grid_n_gc[gy, gx] = n_gc_px[i]
    return grid_class, grid_majority, np.stack([grid_n_tls, grid_n_gc]), len(selected)


# ─── Slide-level metrics ───────────────────────────────────────────────


def patch_grid_from_mask_cache(
    cache: dict, coords: torch.Tensor, upsample_factor: int
) -> np.ndarray:
    """Reduce the upsampled mask to one class per patch.

    cache["mask"]: (H_up, W_up) values in {0, 1, 2}, where each patch
    occupies an upsample_factor × upsample_factor block. Reduce by
    plurality vote, with GC > TLS > bg precedence.
    """
    mask = cache["mask"].numpy() if hasattr(cache["mask"], "numpy") else cache["mask"]
    grid_x = (coords[:, 0] / PATCH_SIZE).long()
    grid_y = (coords[:, 1] / PATCH_SIZE).long()
    grid_x -= grid_x.min()
    grid_y -= grid_y.min()
    H = int(grid_y.max().item()) + 1
    W = int(grid_x.max().item()) + 1

    out = np.zeros((H, W), dtype=np.int64)
    u = upsample_factor
    for i in range(coords.shape[0]):
        gx, gy = int(grid_x[i]), int(grid_y[i])
        cell = mask[gy * u : (gy + 1) * u, gx * u : (gx + 1) * u]
        if (cell == 2).any():
            out[gy, gx] = 2
        elif (cell == 1).any():
            out[gy, gx] = 1
    return out


def dice_score(pred: np.ndarray, target: np.ndarray, cls: int, eps: float = 1e-6) -> float:
    p = (pred == cls)
    t = (target == cls)
    inter = float((p & t).sum())
    denom = float(p.sum() + t.sum())
    return (2 * inter + eps) / (denom + eps)


def count_components(grid: np.ndarray, cls: int) -> int:
    from scipy.ndimage import label
    _, n = label(grid == cls)
    return n


# ─── Main eval ─────────────────────────────────────────────────────────


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print(f"Stage 1: {args.stage1}")
    stage1 = load_stage1(args.stage1, device)
    print(f"Stage 2: {args.stage2}")
    stage2 = load_stage2(args.stage2, device)

    # Build val entries from the same split logic Stage 1 used.
    ps.set_seed(args.seed)
    entries = ps.build_slide_entries()
    splits = ps.create_splits(entries, k_folds=1, seed=args.seed)
    val_entries = splits[0]["val"]
    val_entries = [e for e in val_entries if e.get("mask_path") is not None]
    print(f"Val: {len(val_entries)} slides with masks\n")

    print(f"Loading mask cache (upsample_factor={args.upsample_factor})...")
    mask_dict = ps.build_mask_cache(val_entries, args.upsample_factor)

    rows = []
    for thr in args.thresholds:
        print(f"\n{'=' * 60}\nTHRESHOLD = {thr}\n{'=' * 60}")
        per_slide = []
        t_s1 = t_s2 = 0.0
        n_selected_total = n_total_total = 0
        for k, entry in enumerate(val_entries):
            short_id = entry["slide_id"].split(".")[0]
            cache = mask_dict.get(short_id)
            if cache is None:
                continue
            import zarr
            grp = zarr.open(entry["zarr_path"], mode="r")
            features = torch.from_numpy(grp["features"][:]).float()
            coords = torch.from_numpy(grp["coords"][:]).float()
            if "graph_edges_1hop" in grp:
                edge_index = torch.from_numpy(grp["graph_edges_1hop"][:]).long()
            elif "edge_index" in grp:
                edge_index = torch.from_numpy(grp["edge_index"][:]).long()
            else:
                continue

            t0 = time.time()
            grid_class, grid_majority, _, n_selected = cascade_one_slide(
                stage1, stage2, features, coords, edge_index, thr, device,
                s2_batch=args.s2_batch,
            )
            t_total = time.time() - t0

            target_grid = patch_grid_from_mask_cache(cache, coords, args.upsample_factor)
            tls_d = dice_score(grid_class, target_grid, 1)
            gc_d = dice_score(grid_class, target_grid, 2)
            mD = (tls_d + gc_d) / 2.0

            n_tls_pred = count_components(grid_class, 1)
            n_gc_pred = count_components(grid_class, 2)
            n_tls_true = count_components(target_grid, 1)
            n_gc_true = count_components(target_grid, 2)

            per_slide.append({
                "slide_id": short_id,
                "cancer_type": entry["cancer_type"],
                "tls_dice": tls_d,
                "gc_dice": gc_d,
                "mDice": mD,
                "n_tls_pred": n_tls_pred, "n_tls_true": n_tls_true,
                "n_gc_pred": n_gc_pred,  "n_gc_true": n_gc_true,
                "n_selected": n_selected,
                "n_total": features.shape[0],
                "t_total": t_total,
            })
            n_selected_total += n_selected
            n_total_total += features.shape[0]
            if (k + 1) % 20 == 0:
                print(f"  [{k + 1}/{len(val_entries)}] processed")

        if not per_slide:
            print("  No slides processed.")
            continue

        mD_mean = float(np.mean([r["mDice"] for r in per_slide]))
        tls_mean = float(np.mean([r["tls_dice"] for r in per_slide]))
        gc_mean = float(np.mean([r["gc_dice"] for r in per_slide]))
        tls_sp, _ = spearmanr([r["n_tls_true"] for r in per_slide],
                              [r["n_tls_pred"] for r in per_slide])
        gc_sp, _ = spearmanr([r["n_gc_true"] for r in per_slide],
                             [r["n_gc_pred"] for r in per_slide])
        per_slide_t = float(np.mean([r["t_total"] for r in per_slide]))
        sel_frac = n_selected_total / max(1, n_total_total)

        print(f"\n  Results ({len(per_slide)} slides, threshold={thr}):")
        print(f"  Segmentation:")
        print(f"    mDice:    {mD_mean:.4f}")
        print(f"    TLS dice: {tls_mean:.4f}")
        print(f"    GC dice:  {gc_mean:.4f}")
        print(f"  Counting:")
        print(f"    TLS sp:   {tls_sp:.3f}")
        print(f"    GC sp:    {gc_sp:.3f}")
        print(f"  Speed:")
        print(f"    Total:    {per_slide_t:.2f}s/slide")
        print(f"    Selected: {n_selected_total}/{n_total_total} patches "
              f"({100 * sel_frac:.1f}%)")

        for ct in sorted({r["cancer_type"] for r in per_slide}):
            sub = [r for r in per_slide if r["cancer_type"] == ct]
            print(f"  {ct} ({len(sub)} slides): "
                  f"TLS={np.mean([r['tls_dice'] for r in sub]):.3f} "
                  f"GC={np.mean([r['gc_dice'] for r in sub]):.3f}")

        rows.append({
            "threshold": thr,
            "mDice": mD_mean, "tls_dice": tls_mean, "gc_dice": gc_mean,
            "tls_sp": tls_sp, "gc_sp": gc_sp,
            "n_selected": n_selected_total, "n_total": n_total_total,
            "s_per_slide": per_slide_t,
        })

    print("\n\nSummary:")
    print(f"  {'thr':>6}  {'mDice':>7}  {'TLS d':>7}  {'GC d':>7}  "
          f"{'TLS sp':>7}  {'GC sp':>7}  {'sel%':>6}")
    for r in rows:
        sel_pct = 100 * r["n_selected"] / max(1, r["n_total"])
        print(f"  {r['threshold']:>6.2f}  {r['mDice']:>7.4f}  "
              f"{r['tls_dice']:>7.4f}  {r['gc_dice']:>7.4f}  "
              f"{r['tls_sp']:>7.3f}  {r['gc_sp']:>7.3f}  "
              f"{sel_pct:>5.2f}%")
    print("\nDone.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="Path to Stage 1 best_checkpoint.pt")
    ap.add_argument("--stage2", required=True, help="Path to Stage 2 best_checkpoint.pt")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--upsample_factor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s2_batch", type=int, default=64)
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
