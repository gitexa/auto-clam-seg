"""GARS cascade eval: Stage 1 → threshold → Stage 2 → slide-level metrics.

v3.2 — hydra config + wandb summary + config dump to results dir.

Pipeline per slide:
  1. Stage 1 (GraphTLSDetector) scores every patch.
  2. Threshold at THR → select TLS-positive patch indices.
  3. Stage 2 (UNIv2PixelDecoder) decodes each selected patch's UNI-v2
     feature into a 256×256 3-class mask (batched).
  4. Stitch per-patch argmax tiles into a slide-level patch grid.
  5. Compare against the cached HookNet mask reduced to the same
     patch-grid resolution → mDice / TLS dice / GC dice.
  6. Connected-component counting on the predicted grid → Spearman vs
     ground-truth instance counts.

Run:
    python eval_gars_cascade.py stage1=<path> stage2=<path>
    python eval_gars_cascade.py stage1=... stage2=... thresholds=[0.05,0.1]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/ubuntu/profile-clam")
from train_gars_stage1 import GraphTLSDetector  # noqa: E402
from train_gars_stage2 import UNIv2PixelDecoder  # noqa: E402

PATCH_SIZE = 256


def load_stage1(ckpt_path: str, device: torch.device) -> GraphTLSDetector:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    # Support both old (flat) and new (nested model.X) configs.
    m = cfg.get("model", cfg)
    model = GraphTLSDetector(
        in_dim=m.get("in_dim", 1536),
        hidden_dim=m.get("hidden_dim", 256),
        n_hops=m.get("n_hops", 3),
        gnn_type=m.get("gnn_type", "gatv2"),
        dropout=m.get("dropout", 0.1),
        gat_heads=m.get("gat_heads", 4),
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_stage2(ckpt_path: str, device: torch.device) -> UNIv2PixelDecoder:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    model = UNIv2PixelDecoder(
        in_dim=m.get("in_dim", 1536),
        bottleneck=m.get("bottleneck", 512),
        hidden_channels=m.get("hidden_channels", 64),
        spatial_size=m.get("spatial_size", 16),
        n_classes=m.get("n_classes", 3),
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def cascade_one_slide(stage1, stage2, features, coords, edge_index, threshold,
                      device, s2_batch=64):
    n = features.shape[0]
    grid_x = (coords[:, 0] / PATCH_SIZE).long()
    grid_y = (coords[:, 1] / PATCH_SIZE).long()
    grid_x -= grid_x.min()
    grid_y -= grid_y.min()
    H = int(grid_y.max().item()) + 1
    W = int(grid_x.max().item()) + 1

    s1_logits = stage1(features.to(device), edge_index.to(device))
    s1_probs = torch.sigmoid(s1_logits).cpu().numpy()
    selected = np.where(s1_probs > threshold)[0]

    pred_class_per_patch = np.zeros(n, dtype=np.int64)
    n_tls_px = np.zeros(n, dtype=np.int64)
    n_gc_px = np.zeros(n, dtype=np.int64)
    for s in range(0, len(selected), s2_batch):
        batch_idx = selected[s : s + s2_batch]
        feats = features[batch_idx].to(device)
        argmax = stage2(feats).argmax(dim=1).cpu().numpy()
        for b, i in enumerate(batch_idx):
            tile = argmax[b]
            n_tls_px[i] = int((tile == 1).sum())
            n_gc_px[i] = int((tile == 2).sum())
            pred_class_per_patch[i] = (
                2 if n_gc_px[i] > 0 else (1 if n_tls_px[i] > 0 else 0)
            )

    grid_class = np.zeros((H, W), dtype=np.int64)
    for i in range(n):
        gx, gy = int(grid_x[i]), int(grid_y[i])
        grid_class[gy, gx] = pred_class_per_patch[i]
    return grid_class, len(selected)


def patch_grid_from_mask_cache(cache, coords, upsample_factor):
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


def dice_score(pred, target, cls, eps=1e-6):
    p = (pred == cls); t = (target == cls)
    inter = float((p & t).sum())
    denom = float(p.sum() + t.sum())
    return (2 * inter + eps) / (denom + eps)


def count_components(grid, cls):
    from scipy.ndimage import label
    _, n = label(grid == cls)
    return n


@hydra.main(version_base=None, config_path="configs/cascade", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    import prepare_segmentation as ps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print(f"Stage 1: {cfg.stage1}")
    stage1 = load_stage1(cfg.stage1, device)
    print(f"Stage 2: {cfg.stage2}")
    stage2 = load_stage2(cfg.stage2, device)

    ps.set_seed(cfg.seed)
    entries = ps.build_slide_entries()
    folds_pair, _test = ps.create_splits(entries, k_folds=1, seed=cfg.seed)
    val_entries = folds_pair[0]
    val_entries = [e for e in val_entries if e.get("mask_path") is not None]
    print(f"Val: {len(val_entries)} slides with masks\n")

    print(f"Loading mask cache (upsample_factor={cfg.upsample_factor})...")
    mask_dict = ps.build_mask_cache(val_entries, cfg.upsample_factor)

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

    rows = []
    import zarr
    for thr in cfg.thresholds:
        print(f"\n{'=' * 60}\nTHRESHOLD = {thr}\n{'=' * 60}")
        per_slide = []
        n_selected_total = n_total_total = 0
        for k, entry in enumerate(val_entries):
            short_id = entry["slide_id"].split(".")[0]
            cache = mask_dict.get(short_id)
            if cache is None:
                continue
            grp = zarr.open(entry["zarr_path"], mode="r")
            features = torch.from_numpy(np.asarray(grp["features"][:])).float()
            coords = torch.from_numpy(np.asarray(grp["coords"][:])).float()
            if "graph_edges_1hop" in grp:
                edge_index = torch.from_numpy(np.asarray(grp["graph_edges_1hop"][:])).long()
            elif "edge_index" in grp:
                edge_index = torch.from_numpy(np.asarray(grp["edge_index"][:])).long()
            else:
                continue

            t0 = time.time()
            grid_class, n_selected = cascade_one_slide(
                stage1, stage2, features, coords, edge_index, thr, device,
                s2_batch=cfg.s2_batch,
            )
            t_total = time.time() - t0

            target_grid = patch_grid_from_mask_cache(cache, coords, cfg.upsample_factor)
            tls_d = dice_score(grid_class, target_grid, 1)
            gc_d = dice_score(grid_class, target_grid, 2)
            per_slide.append({
                "slide_id": short_id, "cancer_type": entry["cancer_type"],
                "tls_dice": tls_d, "gc_dice": gc_d, "mDice": (tls_d + gc_d) / 2.0,
                "n_tls_pred": count_components(grid_class, 1),
                "n_tls_true": count_components(target_grid, 1),
                "n_gc_pred": count_components(grid_class, 2),
                "n_gc_true": count_components(target_grid, 2),
                "n_selected": n_selected, "n_total": features.shape[0],
                "t_total": t_total,
            })
            n_selected_total += n_selected
            n_total_total += features.shape[0]
            if (k + 1) % 20 == 0:
                print(f"  [{k + 1}/{len(val_entries)}] processed")

        if not per_slide:
            continue
        mD = float(np.mean([r["mDice"] for r in per_slide]))
        tls_m = float(np.mean([r["tls_dice"] for r in per_slide]))
        gc_m = float(np.mean([r["gc_dice"] for r in per_slide]))
        tls_sp, _ = spearmanr([r["n_tls_true"] for r in per_slide],
                              [r["n_tls_pred"] for r in per_slide])
        gc_sp, _ = spearmanr([r["n_gc_true"] for r in per_slide],
                             [r["n_gc_pred"] for r in per_slide])
        per_slide_t = float(np.mean([r["t_total"] for r in per_slide]))
        sel_frac = n_selected_total / max(1, n_total_total)
        print(f"\n  Results ({len(per_slide)} slides, threshold={thr}):")
        print(f"    mDice={mD:.4f}  TLS={tls_m:.4f}  GC={gc_m:.4f}")
        print(f"    TLS sp={tls_sp:.3f}  GC sp={gc_sp:.3f}")
        print(f"    {per_slide_t:.2f}s/slide  selected {n_selected_total}/{n_total_total} "
              f"({100 * sel_frac:.1f}%)")
        for ct in sorted({r["cancer_type"] for r in per_slide}):
            sub = [r for r in per_slide if r["cancer_type"] == ct]
            print(f"    {ct} ({len(sub)}): TLS={np.mean([r['tls_dice'] for r in sub]):.3f} "
                  f"GC={np.mean([r['gc_dice'] for r in sub]):.3f}")
        rows.append({
            "threshold": thr, "mDice": mD, "tls_dice": tls_m, "gc_dice": gc_m,
            "tls_sp": tls_sp, "gc_sp": gc_sp, "n_selected": n_selected_total,
            "n_total": n_total_total, "s_per_slide": per_slide_t,
        })
        if run is not None:
            run.log({
                "threshold": thr, "mDice": mD, "tls_dice": tls_m, "gc_dice": gc_m,
                "tls_sp": tls_sp, "gc_sp": gc_sp, "selected_frac": sel_frac,
                "s_per_slide": per_slide_t,
            })

    print("\n\nSummary:")
    print(f"  {'thr':>6} {'mDice':>7} {'TLS d':>7} {'GC d':>7} {'TLS sp':>7} {'GC sp':>7} {'sel%':>6}")
    for r in rows:
        sel_pct = 100 * r["n_selected"] / max(1, r["n_total"])
        print(f"  {r['threshold']:>6.2f} {r['mDice']:>7.4f} {r['tls_dice']:>7.4f} "
              f"{r['gc_dice']:>7.4f} {r['tls_sp']:>7.3f} {r['gc_sp']:>7.3f} {sel_pct:>5.2f}%")

    (out_dir / "cascade_results.json").write_text(json.dumps(rows, indent=2))
    if run is not None:
        # Pick best mDice threshold for summary.
        if rows:
            best = max(rows, key=lambda r: r["mDice"])
            run.summary["best_mDice"] = best["mDice"]
            run.summary["best_threshold"] = best["threshold"]
            run.summary["best_gc_dice"] = best["gc_dice"]
        run.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
