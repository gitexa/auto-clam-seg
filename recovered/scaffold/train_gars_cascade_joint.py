"""End2end joint fine-tune of cascade Stage 1 + Stage 2 from their
sequential-training checkpoints.

Loads:
  - v3.8 Stage 1 (GraphTLSDetector) ckpt
  - v3.37 Stage 2 (RegionDecoder) ckpt
Unfreezes BOTH. Trains jointly with one optimizer at a small LR (typical
fine-tune ~1e-5) for `epochs` epochs. Stage 1 features flow through to
Stage 2 with autograd intact (no pre-computed cache, no .eval()).

Loss (per training step):
  Stage 2 pixel CE+Dice on the 3×3 windows around Stage-1-selected patches,
  same loss as the v3.37 baseline. Stage 1's per-patch BCE loss (against
  HookNet patch labels) is added optionally as `stage1_bce_weight`.

Iteration: SLIDE-level. Each step picks one slide, runs Stage 1 forward
over its full graph (one shot), selects patches with `threshold > 0.5`,
runs Stage 2 over the 3×3 windows for those patches, computes joint loss,
backprops, optimizer steps.

Run:
    python train_gars_cascade_joint.py \\
        +stage1_ckpt=<path> +stage2_ckpt=<path> \\
        +fold_idx=0 +k_folds=5 \\
        +epochs=10 +lr_stage1=1e-5 +lr_stage2=1e-5 \\
        +stage1_bce_weight=0.5 \\
        label=v3.11_joint_fold0
"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")

import prepare_segmentation as ps  # noqa: E402
from gncaf_dataset import slide_wsi_path, _read_target_rgb_tile, _normalise_rgb  # noqa: E402
from eval_gars_cascade import load_stage1, load_stage2_region  # noqa: E402
from eval_gars_gncaf_transunet import _load_features_and_graph  # noqa: E402
from region_decoder_model import extract_stage1_context  # noqa: E402
from train_gars_region import region_loss  # noqa: E402

PATCH = 256
G = 3  # 3x3 windows


def patch_labels_from_mask_grid(grid_mask: np.ndarray) -> np.ndarray:
    """Per-patch binary labels: 1 if any TLS/GC pixel in the 256-px patch."""
    return (grid_mask > 0).any(axis=(-1, -2)).astype(np.float32)


def build_windows_around_selected(grid_y: np.ndarray, grid_x: np.ndarray,
                                    selected: np.ndarray):
    """For each selected patch, return the 3x3 window's center-relative
    (gy_center, gx_center) coords."""
    return list(zip(grid_y[selected].tolist(), grid_x[selected].tolist()))


def joint_forward(stage1, stage2, entry, device, threshold=0.5,
                   max_windows=16, wsi_zarr_cache=None):
    """Run one slide forward, return loss-contributing logits + targets."""
    import zarr, tifffile
    feats_np, coords_np, edge_np = _load_features_and_graph(entry["zarr_path"])
    features = torch.from_numpy(feats_np).to(device)
    coords = torch.from_numpy(coords_np).long()
    edge_index = torch.from_numpy(edge_np).long().to(device)

    # Stage 1 forward — autograd intact
    s1_ctx = extract_stage1_context(stage1, features, edge_index)  # (N, 256)
    s1_logits = stage1.head(s1_ctx).squeeze(-1)  # (N,)
    s1_probs = torch.sigmoid(s1_logits)

    # Stage 1 patch labels: load HookNet mask for this slide
    mask_path = entry.get("mask_path")
    if mask_path is None or not Path(mask_path).exists():
        # GT-negative slide; no Stage 1 BCE loss
        s1_target = None
    else:
        mp = np.asarray(tifffile.imread(mask_path))
        # Per-patch label = any TLS/GC pixel in the patch
        n = coords_np.shape[0]
        s1_target = np.zeros(n, dtype=np.float32)
        for i in range(n):
            x = int(coords_np[i, 0]); y = int(coords_np[i, 1])
            if y >= mp.shape[0] or x >= mp.shape[1]:
                continue
            tile = mp[y:y + PATCH, x:x + PATCH]
            s1_target[i] = 1.0 if (tile > 0).any() else 0.0
        s1_target = torch.from_numpy(s1_target).to(device)

    # Stage 1 selection threshold
    selected_mask = s1_probs.detach() > threshold  # detach: hard threshold
    if selected_mask.sum() == 0:
        return None, None, s1_logits, s1_target

    selected = torch.where(selected_mask)[0].cpu().numpy()
    # Limit windows per slide
    if len(selected) > max_windows:
        selected = np.random.choice(selected, max_windows, replace=False)

    # Build 3x3 windows around each selected patch
    H_g = (coords_np[:, 1].max() // PATCH) + 1
    W_g = (coords_np[:, 0].max() // PATCH) + 1
    grid_y_np = (coords_np[:, 1] // PATCH).astype(np.int64)
    grid_x_np = (coords_np[:, 0] // PATCH).astype(np.int64)
    # Map (gy, gx) -> patch_idx
    grid2idx = {}
    for i in range(coords_np.shape[0]):
        grid2idx[(int(grid_y_np[i]), int(grid_x_np[i]))] = i

    wsi_path = slide_wsi_path(entry)
    if wsi_path is None or not Path(wsi_path).exists():
        return None, None, s1_logits, s1_target

    # Open WSI zarr (cache per slide)
    if wsi_zarr_cache is None or wsi_zarr_cache[0] != wsi_path:
        wsi_z = zarr.open(tifffile.imread(wsi_path, aszarr=True, level=0), mode="r")
    else:
        wsi_z = wsi_zarr_cache[1]
    new_cache = (wsi_path, wsi_z)

    # GT mask zarr if exists. For GT-NEGATIVE slides we have no per-pixel
    # supervision for Stage 2 — skip Stage 2 loss entirely (return None).
    gt_z = None
    if mask_path and Path(mask_path).exists():
        gt_z = zarr.open(tifffile.imread(mask_path, aszarr=True, level=0), mode="r")
    else:
        # GT-negative slide: Stage 1 BCE still applies (target = all 0), but
        # Stage 2 has nothing to supervise against. Return early.
        return None, None, s1_logits, s1_target

    n_w = len(selected)
    rgb_buf = np.zeros((n_w, G * G, 3, PATCH, PATCH), dtype=np.float32)
    uni_buf = np.zeros((n_w, G * G, feats_np.shape[1]), dtype=np.float32)
    gat_buf = torch.zeros((n_w, G * G, s1_ctx.shape[1]), device=device)
    val_buf = np.zeros((n_w, G * G), dtype=bool)
    mask_buf = np.full((n_w, 3 * PATCH, 3 * PATCH), -100, dtype=np.int64)

    for wi, center in enumerate(selected):
        cy = int(grid_y_np[center]); cx = int(grid_x_np[center])
        for k in range(G * G):
            dy, dx = divmod(k, G)
            gy = cy + dy - 1; gx = cx + dx - 1
            pi = grid2idx.get((gy, gx))
            if pi is None:
                continue
            val_buf[wi, k] = True
            uni_buf[wi, k] = feats_np[pi]
            gat_buf[wi, k] = s1_ctx[pi]  # gradient-bearing tensor
            x0 = int(coords_np[pi, 0]); y0 = int(coords_np[pi, 1])
            tile = _read_target_rgb_tile(wsi_z, x0, y0)
            rgb_buf[wi, k] = _normalise_rgb(tile).numpy()
            if gt_z is not None:
                # Clamp to slide bounds; tile may be partial near edges.
                gt_H, gt_W = gt_z.shape
                y_lo = min(y0, gt_H); y_hi = min(y0 + PATCH, gt_H)
                x_lo = min(x0, gt_W); x_hi = min(x0 + PATCH, gt_W)
                if y_hi <= y_lo or x_hi <= x_lo:
                    continue
                gt_tile = np.asarray(gt_z[y_lo:y_hi, x_lo:x_hi])
                h_use, w_use = gt_tile.shape
                pyy = dy * PATCH; pxx = dx * PATCH
                mask_buf[wi, pyy:pyy + h_use, pxx:pxx + w_use] = gt_tile

    rgb_t = torch.from_numpy(rgb_buf).to(device)
    uni_t = torch.from_numpy(uni_buf).to(device)
    val_t = torch.from_numpy(val_buf).to(device)
    mask_t = torch.from_numpy(mask_buf).to(device)

    s2_logits = stage2(rgb_t, uni_t, gat_buf, val_t)  # (B, 3, 768, 768)
    return s2_logits, mask_t, s1_logits, s1_target


def train_epoch(stage1, stage2, train_entries, optimizer, device, cfg,
                epoch_idx):
    stage1.train(); stage2.train()
    cw = torch.tensor(list(cfg.train.class_weights), device=device, dtype=torch.float32)
    s1_w = float(cfg.train.get("stage1_bce_weight", 0.5))
    n_iter = 0
    tot_loss = 0.0; tot_s2 = 0.0; tot_s1 = 0.0
    random.shuffle(train_entries)
    for slide_idx, entry in enumerate(train_entries):
        try:
            s2_logits, mask, s1_logits, s1_target = joint_forward(
                stage1, stage2, entry, device,
                threshold=cfg.train.threshold,
                max_windows=int(cfg.train.max_windows_per_slide),
            )
        except Exception as e:
            print(f"  skip slide {entry['slide_id'].split('.')[0]}: {e}")
            continue
        loss = 0.0
        loss_s2 = 0.0; loss_s1 = 0.0
        if s2_logits is not None and mask is not None:
            l, _ = region_loss(s2_logits, mask,
                                class_weights=cw,
                                gc_dice_weight=cfg.train.gc_dice_weight,
                                ignore_index=-100)
            loss_s2 = float(l.detach())
            loss = loss + l
        if s1_target is not None and s1_w > 0:
            ls1 = F.binary_cross_entropy_with_logits(
                s1_logits, s1_target,
                pos_weight=torch.tensor(cfg.train.s1_pos_weight, device=device),
            )
            loss_s1 = float(ls1.detach())
            loss = loss + s1_w * ls1
        if isinstance(loss, float):
            continue
        # Skip optimizer step on non-finite losses (bad slide / numerical
        # edge case) to avoid corrupting model params with NaN gradients.
        loss_val = float(loss.detach())
        if not np.isfinite(loss_val):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Also skip step if any grad is non-finite after backward.
        any_nan_grad = False
        for p in list(stage1.parameters()) + list(stage2.parameters()):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                any_nan_grad = True
                break
        if any_nan_grad:
            optimizer.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(stage1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(stage2.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss_val
        tot_s2 += loss_s2; tot_s1 += loss_s1
        n_iter += 1
        if (slide_idx + 1) % 50 == 0:
            print(f"  [E{epoch_idx} slide {slide_idx+1}/{len(train_entries)}] "
                  f"loss={tot_loss/max(1,n_iter):.4f} s2={tot_s2/max(1,n_iter):.4f} "
                  f"s1={tot_s1/max(1,n_iter):.4f}")
    return {"loss": tot_loss / max(1, n_iter),
            "s2_loss": tot_s2 / max(1, n_iter),
            "s1_loss": tot_s1 / max(1, n_iter),
            "n_iter": n_iter}


def val_epoch(stage1, stage2, val_entries, device, cfg):
    stage1.eval(); stage2.eval()
    cw = torch.tensor(list(cfg.train.class_weights), device=device, dtype=torch.float32)
    tot_loss = 0.0; n = 0
    with torch.no_grad():
        for entry in val_entries:
            try:
                s2_logits, mask, _, _ = joint_forward(
                    stage1, stage2, entry, device,
                    threshold=cfg.train.threshold,
                    max_windows=int(cfg.train.get("val_max_windows_per_slide", 16)),
                )
            except Exception:
                continue
            if s2_logits is None or mask is None:
                continue
            l, _ = region_loss(s2_logits, mask,
                                class_weights=cw,
                                gc_dice_weight=cfg.train.gc_dice_weight,
                                ignore_index=-100)
            lv = float(l.detach())
            if not np.isfinite(lv):
                continue
            tot_loss += lv; n += 1
    return {"val_loss": tot_loss / max(1, n), "n_slides_evald": n}


@hydra.main(version_base=None, config_path="configs/cascade",
            config_name="config_joint_v1")
def main(cfg: DictConfig):
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load both stages, UNFROZEN
    stage1 = load_stage1(str(cfg.stage1_ckpt), device)
    for p in stage1.parameters():
        p.requires_grad = True
    stage1.train()

    stage2 = load_stage2_region(str(cfg.stage2_ckpt), device)
    for p in stage2.parameters():
        p.requires_grad = True
    stage2.train()

    n_s1 = sum(p.numel() for p in stage1.parameters())
    n_s2 = sum(p.numel() for p in stage2.parameters())
    print(f"Stage 1 params (trainable): {n_s1:,}")
    print(f"Stage 2 params (trainable): {n_s2:,}")

    optimizer = AdamW([
        {"params": stage1.parameters(), "lr": float(cfg.train.lr_stage1)},
        {"params": stage2.parameters(), "lr": float(cfg.train.lr_stage2)},
    ], weight_decay=float(cfg.train.weight_decay))

    # Splits
    entries = ps.build_slide_entries()
    fold_idx = int(cfg.fold_idx)
    folds, _ = ps.create_splits(entries, k_folds=int(cfg.k_folds), seed=int(cfg.seed))
    val_entries = folds[fold_idx]
    train_entries = []
    for f in range(int(cfg.k_folds)):
        if f != fold_idx:
            train_entries.extend(folds[f])
    print(f"Fold {fold_idx}: train={len(train_entries)}, val={len(val_entries)}")

    # wandb
    wb = None
    if cfg.get("wandb", {}).get("enabled", False):
        import wandb
        wb = wandb.init(
            project=cfg.wandb.get("project", "tls-pixel-seg"),
            entity=cfg.wandb.get("entity"),
            name=out_dir.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir),
            mode=cfg.wandb.get("mode", "online"),
            tags=list(cfg.wandb.get("tags", [])),
        )
        print(f"wandb: synced run {wb.name}")

    best_val = float("inf"); best_epoch = -1
    for ep in range(1, int(cfg.train.epochs) + 1):
        t0 = time.time()
        tr = train_epoch(stage1, stage2, train_entries, optimizer, device, cfg,
                         epoch_idx=ep)
        tr_t = time.time() - t0
        t0 = time.time()
        va = val_epoch(stage1, stage2, val_entries, device, cfg)
        v_t = time.time() - t0
        is_best = va["val_loss"] < best_val
        marker = " BEST" if is_best else ""
        print(f"EPOCH ep={ep} train_loss={tr['loss']:.4f} s2={tr['s2_loss']:.4f} "
              f"s1={tr['s1_loss']:.4f} val_loss={va['val_loss']:.4f} "
              f"({tr_t:.0f}s/{v_t:.0f}s){marker}")
        if wb is not None:
            wb.log({
                "epoch": ep,
                "train/loss": tr["loss"],
                "train/s2_loss": tr["s2_loss"],
                "train/s1_loss": tr["s1_loss"],
                "val/loss": va["val_loss"],
                "val/n_slides": va["n_slides_evald"],
                "epoch_time/train_s": tr_t,
                "epoch_time/val_s": v_t,
                "is_best": int(is_best),
            }, step=ep)
        if is_best:
            best_val = va["val_loss"]; best_epoch = ep
            torch.save({
                "stage1_state_dict": stage1.state_dict(),
                "stage2_state_dict": stage2.state_dict(),
                "epoch": ep,
                "val_loss": va["val_loss"],
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, out_dir / "best_checkpoint.pt")

    print(f"Best at epoch {best_epoch}, val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
