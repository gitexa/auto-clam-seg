"""Slide-level GNCAF eval — same metric schema as eval_gars_cascade.py.

For each val slide, run GNCAF over **every** patch (not just sampled
targets) so the metrics aggregate per-pixel across the whole slide.

Outputs JSON in the same shape as cascade_results.json (single row,
no threshold sweep — GNCAF has no selection threshold):
    [{"mDice_pix":..., "tls_dice_pix":..., "gc_dice_pix":...,
      "tls_sp":..., "gc_sp":..., "tls_mae":..., "gc_mae":..., ...}]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import zarr
import tifffile
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, label
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gncaf_model import GNCAF
from gncaf_dataset import (
    PATCH_SIZE,
    _read_target_rgb_tile,
    _read_target_mask_tile,
    _normalise_rgb,
    _load_features_and_graph,
    slide_wsi_path,
    slide_mask_path,
    build_gncaf_split,
)
from eval_gars_cascade import (
    dice_score, count_components_filtered, count_components,
)
import prepare_segmentation as ps


def load_gncaf(ckpt_path: str, device: torch.device) -> GNCAF:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    model = GNCAF(
        in_features=1536, n_classes=3,
        dim=m.get("dim", 384),
        encoder_layers=m.get("encoder_layers", 12),
        encoder_heads=m.get("encoder_heads", 6),
        gcn_hops=m.get("gcn_hops", 3),
        fusion_heads=m.get("fusion_heads", 8),
        dropout=0.0,
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    print(f"  Loaded GNCAF strict-OK ({sum(p.numel() for p in model.parameters()):,} params)")
    return model.to(device).eval()


class _SlideTileDataset(Dataset):
    """Reads one (rgb, mask) pair per patch index for a single slide.

    Lives inside a DataLoader with N workers so WSI tile reads overlap
    with the GPU forward of the previous batch.
    """
    def __init__(self, wsi_path: str, mask_path: str, coords: np.ndarray):
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.coords = coords
        self._wsi_z = None
        self._mask_z = None

    def _open(self):
        if self._wsi_z is None:
            self._wsi_z = zarr.open(
                tifffile.imread(self.wsi_path, aszarr=True, level=0), mode="r")
            self._mask_z = zarr.open(
                tifffile.imread(self.mask_path, aszarr=True, level=0), mode="r")

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, i: int):
        self._open()
        x, y = int(self.coords[i, 0]), int(self.coords[i, 1])
        rgb = _read_target_rgb_tile(self._wsi_z, x, y)
        mask = _read_target_mask_tile(self._mask_z, x, y)
        return {
            "rgb": _normalise_rgb(rgb),                                  # (3, 256, 256)
            "mask": torch.from_numpy(mask.astype(np.int64)),             # (256, 256)
            "patch_idx": i,
        }


def _collate(batch):
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_idx": torch.tensor([b["patch_idx"] for b in batch], dtype=torch.long),
    }


@torch.no_grad()
def eval_one_slide(
    model: GNCAF, entry: dict, device: torch.device,
    batch_size: int = 16, num_workers: int = 8,
) -> dict[str, Any]:
    short = entry["slide_id"].split(".")[0]
    features_np, coords_np, edge_index_np = _load_features_and_graph(entry["zarr_path"])
    features = torch.from_numpy(features_np).to(device, non_blocking=True)
    edge_index = torch.from_numpy(edge_index_np).to(device, non_blocking=True)
    n = features.shape[0]

    wsi_path = slide_wsi_path(entry)
    mask_path = slide_mask_path(entry)
    if not (os.path.exists(wsi_path) and mask_path and os.path.exists(mask_path)):
        return {"_skip": True, "slide_id": short}

    # Pre-compute graph context once per slide (shared across all targets).
    node_context = model.context(features, edge_index)              # (N, dim)

    ds = _SlideTileDataset(wsi_path, mask_path, coords_np)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate,
                        pin_memory=True, persistent_workers=False,
                        prefetch_factor=4 if num_workers > 0 else None)

    grid_x = (coords_np[:, 0] // PATCH_SIZE).astype(np.int64)
    grid_y = (coords_np[:, 1] // PATCH_SIZE).astype(np.int64)
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    H_g, W_g = int(grid_y.max()) + 1, int(grid_x.max()) + 1

    pred_grid = np.zeros((H_g, W_g), dtype=np.int64)
    target_grid = np.zeros((H_g, W_g), dtype=np.int64)
    agg = {1: [0, 0], 2: [0, 0]}      # [intersection, denominator]

    t0 = time.time()
    for batch in loader:
        target_idx = batch["patch_idx"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            local = model.encoder(rgb)                              # (B, T, L)
            z_ctx = node_context[target_idx]                        # (B, L)
            fused = model.fusion(local, z_ctx)                      # (B, T, L)
            logits = model.decoder(fused)                           # (B, 3, 256, 256)
        argmax = logits.argmax(dim=1)                               # (B, 256, 256)

        # Per-pixel intersection + denom aggregation per class.
        for cls in (1, 2):
            p = (argmax == cls)
            t = (gt == cls)
            agg[cls][0] += int((p & t).sum().item())
            agg[cls][1] += int((p.sum() + t.sum()).item())

        # Per-patch class for grid (priority GC>TLS>bg).
        p_np = argmax.cpu().numpy()
        t_np = gt.cpu().numpy()
        idx_np = target_idx.cpu().numpy()
        for b in range(p_np.shape[0]):
            i = int(idx_np[b])
            tile_p = p_np[b]; tile_t = t_np[b]
            n_gc_p = int((tile_p == 2).sum()); n_tls_p = int((tile_p == 1).sum())
            n_gc_t = int((tile_t == 2).sum()); n_tls_t = int((tile_t == 1).sum())
            pred_grid[grid_y[i], grid_x[i]] = 2 if n_gc_p > 0 else (1 if n_tls_p > 0 else 0)
            target_grid[grid_y[i], grid_x[i]] = 2 if n_gc_t > 0 else (1 if n_tls_t > 0 else 0)
    t_total = time.time() - t0

    return {
        "slide_id": short,
        "cancer_type": entry["cancer_type"],
        "agg": agg,
        "pred_grid": pred_grid,
        "target_grid": target_grid,
        "n_total": n,
        "t_total": t_total,
    }


@hydra.main(version_base=None, config_path="configs/gncaf",
            config_name="eval_config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"checkpoint: {cfg.checkpoint}")
    model = load_gncaf(cfg.checkpoint, device)

    _train, val_entries = build_gncaf_split()
    val_entries = [e for e in val_entries if e.get("mask_path")]
    if cfg.get("limit_slides"):
        val_entries = val_entries[: int(cfg.limit_slides)]
    # Sharding: process every Nth slide starting at offset (for parallel runs).
    stride = int(cfg.get("slide_stride", 1) or 1)
    offset = int(cfg.get("slide_offset", 0) or 0)
    if stride > 1 or offset > 0:
        val_entries = val_entries[offset::stride]
        print(f"Sharded: offset={offset} stride={stride} → {len(val_entries)} slides")
    print(f"Val: {len(val_entries)} slides\n")

    # Counts ground truth from slide metadata (gold standard for sp/MAE).
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

    agg_total = {1: [0, 0], 2: [0, 0]}
    per_slide = []
    n_skipped = 0
    for k, e in enumerate(val_entries):
        r = eval_one_slide(model, e, device,
                           batch_size=int(cfg.batch_size),
                           num_workers=int(cfg.num_workers))
        if r.get("_skip"):
            n_skipped += 1; continue
        for cls in (1, 2):
            agg_total[cls][0] += r["agg"][cls][0]
            agg_total[cls][1] += r["agg"][cls][1]

        sid = r["slide_id"]
        gt_n_tls, gt_n_gc = gt_counts.get(sid, (
            count_components(r["target_grid"], 1),
            count_components(r["target_grid"], 2),
        ))
        n_tls_pred = count_components_filtered(r["pred_grid"], 1,
                                                int(cfg.min_component_size),
                                                int(cfg.closing_iters))
        n_gc_pred = count_components_filtered(r["pred_grid"], 2,
                                               int(cfg.min_component_size),
                                               int(cfg.closing_iters))
        # Patch-grid dice (per slide).
        p, tg = r["pred_grid"], r["target_grid"]
        tls_d_grid = dice_score(p, tg, 1)
        gc_d_grid = dice_score(p, tg, 2)

        per_slide.append({
            "slide_id": sid, "cancer_type": r["cancer_type"],
            "tls_dice_grid": tls_d_grid, "gc_dice_grid": gc_d_grid,
            "n_tls_pred": n_tls_pred, "n_gc_pred": n_gc_pred,
            "gt_n_tls": gt_n_tls, "gt_n_gc": gt_n_gc,
            "n_total": r["n_total"], "t_total": r["t_total"],
        })
        if (k + 1) % 5 == 0:
            print(f"  [{k + 1}/{len(val_entries)}] {sid} "
                  f"({r['t_total']:.1f}s, {r['n_total']} patches)")
    if not per_slide:
        print("No slides evaluated."); return

    eps = 1e-6
    tls_pix = (2 * agg_total[1][0] + eps) / (agg_total[1][1] + eps)
    gc_pix = (2 * agg_total[2][0] + eps) / (agg_total[2][1] + eps)
    mD_pix = (tls_pix + gc_pix) / 2.0
    tls_grid_m = float(np.mean([r["tls_dice_grid"] for r in per_slide]))
    gc_grid_m = float(np.mean([r["gc_dice_grid"] for r in per_slide]))
    mD_grid = (tls_grid_m + gc_grid_m) / 2.0
    gt_t = [r["gt_n_tls"] for r in per_slide]
    gt_g = [r["gt_n_gc"] for r in per_slide]
    pr_t = [r["n_tls_pred"] for r in per_slide]
    pr_g = [r["n_gc_pred"] for r in per_slide]
    tls_sp, _ = spearmanr(gt_t, pr_t)
    gc_sp, _ = spearmanr(gt_g, pr_g) if any(gt_g) else (0.0, 0.0)
    from sklearn.metrics import mean_absolute_error
    tls_mae = float(mean_absolute_error(gt_t, pr_t))
    gc_mae = float(mean_absolute_error(gt_g, pr_g))
    s_per = float(np.mean([r["t_total"] for r in per_slide]))
    n_total = int(np.sum([r["n_total"] for r in per_slide]))

    print(f"\nResults ({len(per_slide)} slides, {n_skipped} skipped):")
    print(f"  [patch-grid] mDice={mD_grid:.4f}  TLS={tls_grid_m:.4f}  GC={gc_grid_m:.4f}")
    print(f"  [pixel-agg]  mDice={mD_pix:.4f}  TLS={tls_pix:.4f}  GC={gc_pix:.4f}")
    print(f"  [counts]  TLS sp={tls_sp:.3f} mae={tls_mae:.2f}  GC sp={gc_sp:.3f} mae={gc_mae:.2f}")
    print(f"  {s_per:.1f}s/slide, {n_total:,} patches total")

    rows = [{
        "checkpoint": str(cfg.checkpoint),
        "mDice_grid": mD_grid, "tls_dice_grid": tls_grid_m, "gc_dice_grid": gc_grid_m,
        "mDice_pix": mD_pix, "tls_dice_pix": tls_pix, "gc_dice_pix": gc_pix,
        "tls_sp": tls_sp, "gc_sp": gc_sp,
        "tls_mae": tls_mae, "gc_mae": gc_mae,
        "n_slides": len(per_slide), "n_skipped": n_skipped,
        "n_patches_total": n_total,
        "s_per_slide": s_per,
        "min_component_size": int(cfg.min_component_size),
        "closing_iters": int(cfg.closing_iters),
        "slide_offset": int(cfg.get("slide_offset", 0) or 0),
        "slide_stride": int(cfg.get("slide_stride", 1) or 1),
    }]
    # Persist per-slide raw stats AND aggregates so a parallel-shard run
    # can be combined later: write _agg with the inter/denom totals.
    (out_dir / "gncaf_results.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "gncaf_agg.json").write_text(json.dumps({
        "tls_inter": agg_total[1][0], "tls_denom": agg_total[1][1],
        "gc_inter": agg_total[2][0], "gc_denom": agg_total[2][1],
        "per_slide": per_slide,
    }, indent=2))
    if run is not None:
        run.summary.update(rows[0])
        run.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
