"""GARS hybrid: patchcls (v3.22) selects TLS+GC patches → Stage 2 decoder.

Pipeline per slide:
  1. patchcls (GraphPatchClassifier) classifies every patch as bg/TLS/GC.
  2. Patches predicted as TLS or GC → Stage 2 decoder produces 256×256 mask.
  3. Patches predicted as bg → skipped (counted as bg in patch-grid).
  4. Slide-level metrics: patch-grid dice, pixel-aggregate dice over
     decoded patches, counting Spearman on connected components.

Replaces the binary Stage 1 → soft threshold step with a 3-class router.
Should keep patchcls's patch-grid lead and cascade's GC instance separation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import zarr
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, label
from scipy.stats import spearmanr

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gars_stage2 import UNIv2PixelDecoder
from train_gars_patchcls import (
    GraphPatchClassifier, build_patch_lookup, build_grid,
    patch_grid_from_class_per_patch, count_components_filtered,
    load_slide_features,
)
from eval_gars_cascade import load_stage2  # reuse loader

PATCH_SIZE = 256


def load_patchcls(ckpt_path: str, device: torch.device) -> GraphPatchClassifier:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    model = GraphPatchClassifier(
        in_dim=1536,
        hidden_dim=m.get("hidden_dim", 256),
        n_hops=m.get("n_hops", 5),
        gat_heads=m.get("gat_heads", 4),
        n_classes=3,
        dropout=m.get("dropout", 0.1),
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def hybrid_one_slide(patchcls, stage2, features, coords, edge_index,
                    device, s2_batch=64):
    n = features.shape[0]
    coords_np = coords.numpy() if torch.is_tensor(coords) else coords
    grid_x, grid_y, H, W = build_grid(coords_np)
    f = features.to(device); ei = edge_index.to(device)
    cls_logits = patchcls(f, ei)               # (N, 3)
    pred_class = cls_logits.argmax(dim=1).cpu().numpy()
    selected = np.where(pred_class > 0)[0]      # TLS or GC

    pred_tiles: dict[int, np.ndarray] = {}
    grid = np.zeros((H, W), dtype=np.int64)
    # First fill grid from patchcls (bg patches won't be touched by Stage 2).
    for k in range(n):
        grid[grid_y[k], grid_x[k]] = int(pred_class[k])
    # Now refine selected patches' grid cell with Stage 2 argmax tile (priority GC>TLS>bg).
    for s in range(0, len(selected), s2_batch):
        batch_idx = selected[s : s + s2_batch]
        feats = features[batch_idx].to(device)
        argmax = stage2(feats).argmax(dim=1).cpu().numpy().astype(np.uint8)
        for b, i in enumerate(batch_idx):
            tile = argmax[b]
            pred_tiles[int(i)] = tile
            n_gc = int((tile == 2).sum()); n_tls = int((tile == 1).sum())
            grid[grid_y[i], grid_x[i]] = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)
    return grid, len(selected), pred_tiles


@hydra.main(version_base=None, config_path="configs/hybrid", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"patchcls: {cfg.patchcls}")
    patchcls = load_patchcls(cfg.patchcls, device)
    print(f"Stage 2: {cfg.stage2}")
    stage2 = load_stage2(cfg.stage2, device)

    import prepare_segmentation as ps
    ps.set_seed(cfg.seed)
    entries = ps.build_slide_entries()
    folds_pair, _ = ps.create_splits(entries, k_folds=1, seed=cfg.seed)
    val_entries = [e for e in folds_pair[0] if e.get("mask_path")]
    print(f"Val: {len(val_entries)} slides\n")

    print("Loading patch cache for GT...")
    from tls_patch_dataset import build_tls_patch_dataset
    bundle = build_tls_patch_dataset(
        cache_path=cfg.patch_cache_path, verbose=False)
    patch_mask_lookup: dict[str, dict[int, np.ndarray]] = {}
    for ci, sid in enumerate(bundle["slide_ids"]):
        short = sid.split(".")[0]
        pi = int(bundle["patch_idx"][ci])
        patch_mask_lookup.setdefault(short, {})[pi] = np.asarray(bundle_masks := bundle["masks"][ci])

    import pandas as pd
    df = pd.read_csv(ps.META_CSV)
    gt_counts = {str(r["slide_id"]).split(".")[0]: (int(r["tls_num"]), int(r["gc_num"]))
                 for _, r in df.iterrows()}

    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != "disabled":
        import wandb
        run = wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=out_dir.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir), mode=cfg.wandb.mode,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        )

    per_slide = []
    n_selected_total = n_total_total = 0
    agg = {1: {"inter": 0, "denom": 0}, 2: {"inter": 0, "denom": 0}}
    for k, e in enumerate(val_entries):
        short = e["slide_id"].split(".")[0]
        f, c, ei = load_slide_features(e["zarr_path"])
        if ei is None:
            continue
        coords = c.numpy()

        t0 = time.time()
        grid_class, n_sel, pred_tiles = hybrid_one_slide(
            patchcls, stage2, f, c, ei, device, s2_batch=cfg.s2_batch)
        t_total = time.time() - t0

        # GT grid (priority GC>TLS>bg) from patch mask lookup.
        from train_gars_patchcls import patch_labels_from_mask_lookup
        slide_lookup = patch_mask_lookup.get(short, {})
        gt_labels = patch_labels_from_mask_lookup(coords, slide_lookup,
                                                  cfg.min_tls_pixels)
        target_grid = patch_grid_from_class_per_patch(gt_labels, coords)

        from eval_gars_cascade import dice_score
        tls_d = dice_score(grid_class, target_grid, 1)
        gc_d = dice_score(grid_class, target_grid, 2)

        # Pixel-aggregate over decoded patches.
        for i, tile in pred_tiles.items():
            gt = slide_lookup.get(int(i))
            gt = np.asarray(gt) if gt is not None else np.zeros((256, 256), np.uint8)
            for cls in (1, 2):
                p = (tile == cls); t = (gt == cls)
                agg[cls]["inter"] += int((p & t).sum())
                agg[cls]["denom"] += int(p.sum() + t.sum())

        min_size = int(cfg.get("min_component_size", 2))
        close_iters = int(cfg.get("closing_iters", 1))
        n_tls_pred = count_components_filtered(grid_class, 1, min_size, close_iters)
        n_gc_pred = count_components_filtered(grid_class, 2, min_size, close_iters)
        gt_n_tls, gt_n_gc = gt_counts.get(short, (0, 0))
        per_slide.append({
            "slide_id": short, "cancer_type": e["cancer_type"],
            "tls_dice": tls_d, "gc_dice": gc_d,
            "n_tls_pred": n_tls_pred, "n_gc_pred": n_gc_pred,
            "gt_n_tls": gt_n_tls, "gt_n_gc": gt_n_gc,
            "n_selected": n_sel, "n_total": f.shape[0],
            "t_total": t_total,
        })
        n_selected_total += n_sel; n_total_total += f.shape[0]
        if (k + 1) % 20 == 0:
            print(f"  [{k + 1}/{len(val_entries)}] processed")

    eps = 1e-6
    tls_pix = (2 * agg[1]["inter"] + eps) / (agg[1]["denom"] + eps)
    gc_pix = (2 * agg[2]["inter"] + eps) / (agg[2]["denom"] + eps)
    mD_pix = (tls_pix + gc_pix) / 2.0
    tls_grid = float(np.mean([r["tls_dice"] for r in per_slide]))
    gc_grid = float(np.mean([r["gc_dice"] for r in per_slide]))
    mD_grid = (tls_grid + gc_grid) / 2.0
    gt_t = [r["gt_n_tls"] for r in per_slide]
    gt_g = [r["gt_n_gc"] for r in per_slide]
    pr_t = [r["n_tls_pred"] for r in per_slide]
    pr_g = [r["n_gc_pred"] for r in per_slide]
    tls_sp, _ = spearmanr(gt_t, pr_t)
    gc_sp, _ = spearmanr(gt_g, pr_g) if any(gt_g) else (0.0, 0.0)
    from sklearn.metrics import mean_absolute_error
    tls_mae = float(mean_absolute_error(gt_t, pr_t))
    gc_mae = float(mean_absolute_error(gt_g, pr_g))
    sel_frac = n_selected_total / max(1, n_total_total)
    s_per = float(np.mean([r["t_total"] for r in per_slide]))
    print(f"\nResults ({len(per_slide)} slides):")
    print(f"  [patch-grid] mDice={mD_grid:.4f}  TLS={tls_grid:.4f}  GC={gc_grid:.4f}")
    print(f"  [pixel-agg]  mDice={mD_pix:.4f}  TLS={tls_pix:.4f}  GC={gc_pix:.4f}")
    print(f"  [counts vs gt]  TLS sp={tls_sp:.3f} mae={tls_mae:.2f}  "
          f"GC sp={gc_sp:.3f} mae={gc_mae:.2f}")
    print(f"  selected {n_selected_total}/{n_total_total} ({100 * sel_frac:.1f}%)  "
          f"{s_per:.2f}s/slide")

    rows = [{
        "patchcls": str(cfg.patchcls), "stage2": str(cfg.stage2),
        "mDice_grid": mD_grid, "tls_dice_grid": tls_grid, "gc_dice_grid": gc_grid,
        "mDice_pix": mD_pix, "tls_dice_pix": tls_pix, "gc_dice_pix": gc_pix,
        "tls_sp": tls_sp, "gc_sp": gc_sp,
        "tls_mae": tls_mae, "gc_mae": gc_mae,
        "n_selected": n_selected_total, "n_total": n_total_total,
        "s_per_slide": s_per,
    }]
    (out_dir / "hybrid_results.json").write_text(json.dumps(rows, indent=2))
    if run is not None:
        run.summary.update(rows[0])
        run.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
