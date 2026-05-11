"""Slide-level eval for the recovered TransUNet GNCAF (`gncaf_pixel_*`
checkpoints). Same metric schema as `eval_gars_gncaf.py` so the result
JSON can be compared directly against the cascade and v3.13/v3.37 numbers.

Run:
    python eval_gars_gncaf_transunet.py checkpoint=<best_checkpoint.pt> \
        batch_size=16 num_workers=8

Writes to `<run_dir>/gncaf_results.json` and `<run_dir>/gncaf_agg.json`
(the latter for shard-combine via `combine_gncaf_shards.py`).
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
import zarr
import tifffile
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gncaf_transunet_model import GNCAFPixelDecoder
from gncaf_dataset import (
    PATCH_SIZE,
    _read_target_rgb_tile, _read_target_mask_tile,
    _normalise_rgb, _load_features_and_graph,
    slide_wsi_path, slide_mask_path,
    build_gncaf_split,
)
from eval_gars_cascade import (
    dice_score, count_components_filtered, count_components,
)
import prepare_segmentation as ps


def load_gncaf_transunet(ckpt_path: str, device: torch.device):
    """Load GNCAF or GCUNet (v3.59) — auto-detected via `model_class`
    field in the checkpoint or by sniffing the state_dict for the
    GCUNetPixelDecoder-only `gcn.mlp.*` keys.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    state_dict = obj["model_state_dict"]
    model_class_name = obj.get("model_class") or m.get("model_class") or "gncaf"
    has_slide_head = any(k.startswith("slide_head.") for k in state_dict)
    is_strict = model_class_name == "strict" or any(k.endswith("gcn_layers.0.tau") or k.endswith(".q_proj.weight") and "gcn" in k for k in state_dict)
    if is_strict:
        from gncaf_strict_model import GNCAFStrict
        Model = GNCAFStrict
        cls_name = "GNCAFStrict"
        kwargs = dict()
        # Strict has gcn_hidden_dim + fusion_heads knobs not in vanilla cfg.
        if "gcn_hidden_dim" in m:
            kwargs["gcn_hidden_dim"] = int(m["gcn_hidden_dim"])
        if "fusion_heads" in m:
            kwargs["fusion_heads"] = int(m["fusion_heads"])
    elif model_class_name == "gcunet" or any(k.startswith("gcn.mlp.") for k in state_dict):
        from gcunet_model import GCUNetPixelDecoder
        Model = GCUNetPixelDecoder
        cls_name = "GCUNetPixelDecoder"
        kwargs = dict()
    elif any(k.startswith("patch_encoder.stem.") for k in state_dict):
        # Family-B reconstruction (lost custom-small architecture).
        # Infer hidden_dim, n_transformer_layers, and decoder norm type from the state_dict.
        from gncaf_family_b_model import GNCAFFamilyB
        hidden_dim = int(state_dict["patch_encoder.pos_embed"].shape[-1])
        n_tx_layers = sum(
            1 for k in state_dict
            if k.startswith("patch_encoder.transformer.layers.") and k.endswith(".self_attn.in_proj_weight")
        )
        feature_dim = int(state_dict["context_aggregator.feature_proj.0.weight"].shape[1])
        n_classes = int(state_dict["head_seg.weight"].shape[0])
        # Decoder norm: BN if running_mean keys present, else GN.
        decoder_norm = "bn" if "pixel_decoder.up1.1.running_mean" in state_dict else "gn"
        model = GNCAFFamilyB(
            hidden_dim=hidden_dim,
            n_transformer_layers=n_tx_layers,
            feature_dim=feature_dim,
            n_classes=n_classes,
            decoder_norm=decoder_norm,
        )
        model.load_state_dict(state_dict, strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded GNCAFFamilyB strict-OK ({n_params:,} params, "
              f"hidden_dim={hidden_dim}, n_tx_layers={n_tx_layers}, "
              f"decoder_norm={decoder_norm}, "
              f"epoch={obj.get('epoch')}, "
              f"saved best_mdice={float(obj.get('best_mdice', 0)):.4f})")
        return model.to(device).eval()
    else:
        Model = GNCAFPixelDecoder
        cls_name = "GNCAFPixelDecoder"
        # Detect dual-sigmoid head by looking for head_tls.weight in the state_dict.
        is_dual = any(k.startswith("head_tls.") or k.startswith("head_gc.") for k in state_dict)
        head_mode = "dual_sigmoid" if is_dual else "argmax"
        kwargs = dict(slide_aux_head=has_slide_head, head_mode=head_mode)
    model = Model(
        hidden_size=m.get("hidden_size", 768),
        n_classes=m.get("n_classes", 3),
        n_encoder_layers=m.get("n_encoder_layers", 6),
        n_heads=m.get("n_heads", 12),
        mlp_dim=m.get("mlp_dim", 3072),
        n_hops=m.get("n_hops", 3),
        n_fusion_layers=m.get("n_fusion_layers", 1),
        feature_dim=m.get("feature_dim", 1536),
        dropout=0.0,
        **kwargs,
    )
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {cls_name} strict-OK ({n_params:,} params, "
          f"epoch={obj.get('epoch')}, "
          f"saved best_mdice={float(obj.get('best_mdice', 0)):.4f})")
    return model.to(device).eval()


class _SlideTileDataset(Dataset):
    def __init__(self, wsi_path: str, mask_path: str | None, coords: np.ndarray):
        """If `mask_path` is None, the slide is GT-negative and every
        per-tile mask is returned as an all-zero tile (no TLS / no GC).
        """
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.coords = coords
        self._wsi_z = None
        self._mask_z = None

    def _open(self):
        if self._wsi_z is None:
            self._wsi_z = zarr.open(
                tifffile.imread(self.wsi_path, aszarr=True, level=0), mode="r")
            if self.mask_path is not None:
                self._mask_z = zarr.open(
                    tifffile.imread(self.mask_path, aszarr=True, level=0), mode="r")

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, i: int):
        self._open()
        x, y = int(self.coords[i, 0]), int(self.coords[i, 1])
        rgb = _read_target_rgb_tile(self._wsi_z, x, y)
        if self._mask_z is None:
            # GT-negative slide: synthesise a zero mask of the same shape
            # as a real one so the rest of the pipeline (Dice agg,
            # per-tile counts) treats every prediction as a false positive.
            mask = np.zeros_like(rgb[..., 0], dtype=np.uint8)
        else:
            mask = _read_target_mask_tile(self._mask_z, x, y)
        return {
            "rgb": _normalise_rgb(rgb),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "patch_idx": i,
        }


def _collate(batch):
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_idx": torch.tensor([b["patch_idx"] for b in batch], dtype=torch.long),
    }


@torch.no_grad()
def eval_one_slide(model: GNCAFPixelDecoder, entry: dict, device: torch.device,
                   batch_size: int = 16, num_workers: int = 8):
    short = entry["slide_id"].split(".")[0]
    features_np, coords_np, edge_index_np = _load_features_and_graph(entry["zarr_path"])
    features = torch.from_numpy(features_np).to(device, non_blocking=True)
    edge_index = torch.from_numpy(edge_index_np).to(device, non_blocking=True)
    n = features.shape[0]

    wsi_path = slide_wsi_path(entry)
    mask_path = slide_mask_path(entry)
    is_gt_negative = entry.get("mask_path") is None
    if not wsi_path or not os.path.exists(wsi_path):
        return {"_skip": True, "slide_id": short}
    if not is_gt_negative and (not mask_path or not os.path.exists(mask_path)):
        # Mask file declared but not on disk locally → skip rather than
        # silently mis-evaluate as negative.
        return {"_skip": True, "slide_id": short}
    if is_gt_negative:
        mask_path = None  # signal to _SlideTileDataset

    # Pre-compute graph context once per slide.
    is_family_b = hasattr(model, "context_aggregator") and hasattr(model, "patch_encoder")
    if is_family_b:
        node_context = model.context_aggregator(features, edge_index)  # (N, hidden_dim)
    else:
        node_context = model.gcn(features, edge_index)                   # (N, hidden)

    ds = _SlideTileDataset(wsi_path, mask_path, coords_np)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True, persistent_workers=False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    grid_x = (coords_np[:, 0] // PATCH_SIZE).astype(np.int64)
    grid_y = (coords_np[:, 1] // PATCH_SIZE).astype(np.int64)
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    H_g, W_g = int(grid_y.max()) + 1, int(grid_x.max()) + 1

    # For argmax models, the legacy 3-class grid is used (0/1/2 exclusive).
    # For dual-sigmoid models, TLS and GC are independent: keep separate
    # boolean grids to avoid penalising one head for the other firing.
    is_dual = (not is_family_b) and getattr(model, "head_mode", "argmax") == "dual_sigmoid"
    if is_dual:
        pred_tls_grid = np.zeros((H_g, W_g), dtype=bool)
        pred_gc_grid = np.zeros((H_g, W_g), dtype=bool)
        target_tls_grid = np.zeros((H_g, W_g), dtype=bool)
        target_gc_grid = np.zeros((H_g, W_g), dtype=bool)
    pred_grid = np.zeros((H_g, W_g), dtype=np.int64)
    target_grid = np.zeros((H_g, W_g), dtype=np.int64)
    agg = {1: [0, 0], 2: [0, 0]}

    t0 = time.time()
    for batch in loader:
        target_idx = batch["patch_idx"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            local_ctx = node_context[target_idx]
            tls_pred = None; gc_pred = None
            if is_family_b:
                tokens, skip32, skip64 = model.patch_encoder(rgb)
                fused = model.fusion(tokens, local_ctx)
                B = rgb.shape[0]
                feat = fused.transpose(1, 2).reshape(B, -1, 16, 16)
                x = model.pixel_decoder(feat, skip32, skip64)
                logits = model.head_seg(x)
                argmax = logits.argmax(dim=1)
            else:
                tokens, skips = model.encoder(rgb)
                ctx_t = local_ctx.unsqueeze(1)
                x = torch.cat([ctx_t, tokens], dim=1)
                for blk in model.fusion:
                    x = blk(x)
                x = x[:, 1:]
                x = x.transpose(1, 2).reshape(rgb.shape[0], -1, 16, 16)
                x = model.decoder(x, skips)
                if is_dual:
                    tls_logits = model.head_tls(x)
                    gc_logits = model.head_gc(x)
                    tls_pred = (torch.sigmoid(tls_logits) > 0.5).squeeze(1)
                    gc_pred = (torch.sigmoid(gc_logits) > 0.5).squeeze(1)
                    # Keep a 3-class argmax for legacy fields, but only used for
                    # n_tls_pred / n_gc_pred patch counting; class Dice below
                    # uses tls_pred / gc_pred *independently*.
                    argmax = torch.zeros_like(gt)
                    argmax = torch.where(tls_pred, torch.full_like(gt, 1), argmax)
                    argmax = torch.where(gc_pred, torch.full_like(gt, 2), argmax)
                else:
                    logits = model.head_seg(x)
                    argmax = logits.argmax(dim=1)

        if is_dual:
            # TLS Dice: target_tls = (gt >= 1)  (GC pixels are also TLS biologically).
            # GC Dice: target_gc = (gt == 2). Independent thresholds, no argmax collapse.
            t_tls = (gt >= 1)
            t_gc = (gt == 2)
            agg[1][0] += int((tls_pred & t_tls).sum().item())
            agg[1][1] += int((tls_pred.sum() + t_tls.sum()).item())
            agg[2][0] += int((gc_pred & t_gc).sum().item())
            agg[2][1] += int((gc_pred.sum() + t_gc.sum()).item())
        else:
            for cls in (1, 2):
                p = (argmax == cls); t = (gt == cls)
                agg[cls][0] += int((p & t).sum().item())
                agg[cls][1] += int((p.sum() + t.sum()).item())

        p_np = argmax.cpu().numpy(); t_np = gt.cpu().numpy()
        idx_np = target_idx.cpu().numpy()
        if is_dual:
            tls_pred_np = tls_pred.cpu().numpy()
            gc_pred_np = gc_pred.cpu().numpy()
        for b in range(p_np.shape[0]):
            i = int(idx_np[b])
            tile_p = p_np[b]; tile_t = t_np[b]
            n_gc_p = int((tile_p == 2).sum()); n_tls_p = int((tile_p == 1).sum())
            n_gc_t = int((tile_t == 2).sum()); n_tls_t = int((tile_t == 1).sum())
            pred_grid[grid_y[i], grid_x[i]] = 2 if n_gc_p > 0 else (1 if n_tls_p > 0 else 0)
            target_grid[grid_y[i], grid_x[i]] = 2 if n_gc_t > 0 else (1 if n_tls_t > 0 else 0)
            if is_dual:
                # Independent patch-grid flags: a patch is TLS-pred-positive
                # if any pixel from the TLS head fires (regardless of GC head),
                # and GC-pred-positive if any GC pixel fires. Target uses
                # GC ⊂ TLS semantics: TLS target = (gt >= 1).
                pred_tls_grid[grid_y[i], grid_x[i]] = bool(tls_pred_np[b].any())
                pred_gc_grid[grid_y[i], grid_x[i]] = bool(gc_pred_np[b].any())
                target_tls_grid[grid_y[i], grid_x[i]] = bool((tile_t >= 1).any())
                target_gc_grid[grid_y[i], grid_x[i]] = bool((tile_t == 2).any())
    t_total = time.time() - t0

    result = {
        "slide_id": short, "cancer_type": entry["cancer_type"],
        "agg": agg, "pred_grid": pred_grid, "target_grid": target_grid,
        "n_total": n, "t_total": t_total,
        "gt_negative": bool(is_gt_negative),
    }
    if is_dual:
        result["pred_tls_grid"] = pred_tls_grid
        result["pred_gc_grid"] = pred_gc_grid
        result["target_tls_grid"] = target_tls_grid
        result["target_gc_grid"] = target_gc_grid
        result["dual_sigmoid"] = True
    return result


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
    model = load_gncaf_transunet(cfg.checkpoint, device)

    _train, val_entries = build_gncaf_split(
        seed=cfg.seed,
        k_folds=cfg.get("k_folds", 5),
        fold_idx=cfg.get("fold_idx", 0),
    )
    # Full-cohort eval by default: include GT-negative slides so detection
    # CMs / PR-AUC / AUROC reflect the true cohort. Set
    # cfg.eval_positives_only=true to restore the legacy positive-only pool.
    if bool(cfg.get("eval_positives_only", False)):
        val_entries = [e for e in val_entries if e.get("mask_path")]
    n_neg = sum(1 for e in val_entries if not e.get("mask_path"))
    print(f"  cohort: {len(val_entries) - n_neg} GT-pos + {n_neg} GT-neg "
          f"(eval_positives_only={cfg.get('eval_positives_only', False)})")
    if cfg.get("limit_slides"):
        val_entries = val_entries[: int(cfg.limit_slides)]
        print(f"  limit_slides={cfg.limit_slides}")
    stride = int(cfg.get("slide_stride", 1) or 1)
    offset = int(cfg.get("slide_offset", 0) or 0)
    if stride > 1 or offset > 0:
        val_entries = val_entries[offset::stride]
        print(f"Sharded: offset={offset} stride={stride} → {len(val_entries)} slides")
    print(f"Val: {len(val_entries)} slides\n")

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
        if r.get("gt_negative"):
            gt_n_tls, gt_n_gc = 0, 0
        else:
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
        if r.get("dual_sigmoid"):
            # Dual-sigmoid: independent boolean grids per class, no argmax collapse.
            # TLS Dice uses target_tls = (gt >= 1) which includes GC pixels.
            ptg, ttg = r["pred_tls_grid"], r["target_tls_grid"]
            pgg, tgg = r["pred_gc_grid"], r["target_gc_grid"]
            inter_t = int((ptg & ttg).sum()); denom_t = int(ptg.sum() + ttg.sum())
            inter_g = int((pgg & tgg).sum()); denom_g = int(pgg.sum() + tgg.sum())
            tls_d_grid = (2 * inter_t / denom_t) if denom_t else (1.0 if ptg.sum() == 0 else 0.0)
            gc_d_grid = (2 * inter_g / denom_g) if denom_g else (1.0 if pgg.sum() == 0 else 0.0)
        else:
            p, tg = r["pred_grid"], r["target_grid"]
            tls_d_grid = dice_score(p, tg, 1)
            gc_d_grid = dice_score(p, tg, 2)

        per_slide.append({
            "slide_id": sid, "cancer_type": r["cancer_type"],
            "tls_dice_grid": tls_d_grid, "gc_dice_grid": gc_d_grid,
            "n_tls_pred": n_tls_pred, "n_gc_pred": n_gc_pred,
            "gt_n_tls": gt_n_tls, "gt_n_gc": gt_n_gc,
            "gt_negative": bool(r.get("gt_negative", False)),
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
    print(f"  [counts]     TLS sp={tls_sp:.3f} mae={tls_mae:.2f}  GC sp={gc_sp:.3f} mae={gc_mae:.2f}")
    print(f"  {s_per:.1f}s/slide, {n_total:,} patches total")

    rows = [{
        "checkpoint": str(cfg.checkpoint),
        "model": "GNCAFPixelDecoder (TransUNet R50+ViT6L)",
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
