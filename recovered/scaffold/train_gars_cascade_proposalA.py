"""End2end cascade training — Proposal A (stochastic-K joint from scratch).

Per END2END_POSTMORTEM Proposal A:
  - Both stages randomly initialized (NOT loaded from converged ckpts).
  - Slide-level joint forward: Stage 1 over full graph -> stochastic sampling
    of selected patches (Bernoulli with epsilon-explore noise on prob) ->
    Stage 2 over selected 3x3 windows.
  - Loss = w_s1 * BCE(s1, hooknet_patch_labels)
        + w_s2 * (CE+Dice)(s2, hooknet_tile_masks for selected windows)
        + w_ent * entropy_reg(avg(sigmoid(s1)))   # prevents collapse
  - No gradient through the sampler (stop_grad mask).
  - epsilon-explore annealed from 0.5 -> 0.05 over training.

Why this should work where v3.11 (joint fine-tune from ckpts) didn't:
  1. Stochastic selection means Stage 2 sees both Stage-1-low and
     Stage-1-high patches each epoch -> Stage 2 stays general.
  2. From-scratch init avoids the v3.46 / v3.11 "lateral drift in
     converged basin" failure mode.
  3. Stage 1 can't game the supervision because sampling is randomized.

Run:
    python train_gars_cascade_proposalA.py +fold_idx=0 +k_folds=5 \\
        +epochs=15 +eps_start=0.5 +eps_end=0.05 \\
        +w_ent=0.01 label=v3.12_proposalA_fold0
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
from train_gars_stage1 import GraphTLSDetector  # noqa: E402
from region_decoder_model import RegionDecoder, extract_stage1_context  # noqa: E402
from eval_gars_gncaf_transunet import _load_features_and_graph  # noqa: E402
from train_gars_region import region_loss  # noqa: E402

PATCH = 256
G = 3


def stochastic_select(s1_probs: torch.Tensor, eps_explore: float,
                       max_select: int) -> torch.Tensor:
    """Bernoulli-sample selection mask with epsilon-explore noise.

    p_i = clamp(sigmoid(s1) + eps_explore * uniform(-1, 1), [0.05, 0.95])
    select_i ~ Bernoulli(p_i)
    Then cap to max_select for compute budget.
    """
    n = s1_probs.shape[0]
    noise = (torch.rand_like(s1_probs) * 2 - 1) * eps_explore
    p = torch.clamp(s1_probs + noise, 0.05, 0.95)
    mask = torch.bernoulli(p).bool()
    selected = torch.where(mask)[0]
    if selected.numel() > max_select:
        idx = torch.randperm(selected.numel())[:max_select]
        selected = selected[idx]
    return selected


def joint_forward(stage1, stage2, entry, device, eps_explore=0.5,
                   max_windows=8):
    """Forward one slide: Stage 1 -> stochastic select -> Stage 2 windows.

    Returns: s2_logits, s2_target_mask, s1_logits, s1_target.
    """
    import zarr, tifffile
    feats_np, coords_np, edge_np = _load_features_and_graph(entry["zarr_path"])
    features = torch.from_numpy(feats_np).to(device)
    coords = torch.from_numpy(coords_np).long()
    edge_index = torch.from_numpy(edge_np).long().to(device)

    s1_ctx = extract_stage1_context(stage1, features, edge_index)
    s1_logits = stage1.head(s1_ctx).squeeze(-1)
    s1_probs = torch.sigmoid(s1_logits)

    # Stage 1 target from HookNet mask
    mask_path = entry.get("mask_path")
    if mask_path is None or not Path(mask_path).exists():
        s1_target = None
        gt_z = None
    else:
        mp = np.asarray(tifffile.imread(mask_path))
        n = coords_np.shape[0]
        s1_t = np.zeros(n, dtype=np.float32)
        for i in range(n):
            x = int(coords_np[i, 0]); y = int(coords_np[i, 1])
            if y >= mp.shape[0] or x >= mp.shape[1]:
                continue
            tile = mp[y:y + PATCH, x:x + PATCH]
            s1_t[i] = 1.0 if (tile > 0).any() else 0.0
        s1_target = torch.from_numpy(s1_t).to(device)
        gt_z = zarr.open(tifffile.imread(mask_path, aszarr=True, level=0), mode="r")

    # Stochastic-K selection (no grad through mask)
    with torch.no_grad():
        selected = stochastic_select(s1_probs, eps_explore, max_windows)
    if selected.numel() == 0 or gt_z is None:
        return None, None, s1_logits, s1_target

    selected_np = selected.cpu().numpy()
    grid_y_np = (coords_np[:, 1] // PATCH).astype(np.int64)
    grid_x_np = (coords_np[:, 0] // PATCH).astype(np.int64)
    grid2idx = {(int(grid_y_np[i]), int(grid_x_np[i])): i
                for i in range(coords_np.shape[0])}

    wsi_path = slide_wsi_path(entry)
    if wsi_path is None or not Path(wsi_path).exists():
        return None, None, s1_logits, s1_target
    wsi_z = zarr.open(tifffile.imread(wsi_path, aszarr=True, level=0), mode="r")

    n_w = len(selected_np)
    rgb_buf = np.zeros((n_w, G * G, 3, PATCH, PATCH), dtype=np.float32)
    uni_buf = np.zeros((n_w, G * G, feats_np.shape[1]), dtype=np.float32)
    gat_buf = torch.zeros((n_w, G * G, s1_ctx.shape[1]), device=device)
    val_buf = np.zeros((n_w, G * G), dtype=bool)
    mask_buf = np.full((n_w, 3 * PATCH, 3 * PATCH), -100, dtype=np.int64)

    for wi, center in enumerate(selected_np):
        cy = int(grid_y_np[center]); cx = int(grid_x_np[center])
        for k in range(G * G):
            dy, dx = divmod(k, G)
            gy = cy + dy - 1; gx = cx + dx - 1
            pi = grid2idx.get((gy, gx))
            if pi is None:
                continue
            val_buf[wi, k] = True
            uni_buf[wi, k] = feats_np[pi]
            gat_buf[wi, k] = s1_ctx[pi]
            x0 = int(coords_np[pi, 0]); y0 = int(coords_np[pi, 1])
            tile = _read_target_rgb_tile(wsi_z, x0, y0)
            rgb_buf[wi, k] = _normalise_rgb(tile).numpy()
            gt_H, gt_W = gt_z.shape
            y_hi = min(y0 + PATCH, gt_H); x_hi = min(x0 + PATCH, gt_W)
            if y_hi <= y0 or x_hi <= x0:
                continue
            gt_tile = np.asarray(gt_z[y0:y_hi, x0:x_hi])
            h_use, w_use = gt_tile.shape
            pyy = dy * PATCH; pxx = dx * PATCH
            mask_buf[wi, pyy:pyy + h_use, pxx:pxx + w_use] = gt_tile

    rgb_t = torch.from_numpy(rgb_buf).to(device)
    uni_t = torch.from_numpy(uni_buf).to(device)
    val_t = torch.from_numpy(val_buf).to(device)
    mask_t = torch.from_numpy(mask_buf).to(device)
    s2_logits = stage2(rgb_t, uni_t, gat_buf, val_t)
    return s2_logits, mask_t, s1_logits, s1_target


def entropy_reg(s1_probs_avg: torch.Tensor) -> torch.Tensor:
    """Penalize Stage 1 average prob drifting away from 0.5 (collapse).
    Entropy of Bernoulli with prob p = -p log p - (1-p) log(1-p), maxed at 0.5.
    Reg loss = -entropy (so we MAXIMIZE entropy of the average).
    """
    p = s1_probs_avg.clamp(1e-6, 1 - 1e-6)
    h = -(p * p.log() + (1 - p) * (1 - p).log())
    return -h.mean()


def train_epoch(stage1, stage2, train_entries, optimizer, device, cfg,
                eps_explore, epoch_idx):
    stage1.train(); stage2.train()
    cw = torch.tensor(list(cfg.train.class_weights), device=device, dtype=torch.float32)
    w_s1 = float(cfg.train.w_s1)
    w_ent = float(cfg.train.w_ent)
    n_iter = 0; tot_loss = 0.0; tot_s2 = 0.0; tot_s1 = 0.0; tot_ent = 0.0
    avg_s1_prob_acc = 0.0
    random.shuffle(train_entries)
    for slide_idx, entry in enumerate(train_entries):
        try:
            s2_logits, mask, s1_logits, s1_target = joint_forward(
                stage1, stage2, entry, device,
                eps_explore=eps_explore,
                max_windows=int(cfg.train.max_windows_per_slide),
            )
        except Exception as e:
            print(f"  skip {entry['slide_id'].split('.')[0]}: {e}")
            continue

        loss = 0.0
        loss_s2 = 0.0; loss_s1 = 0.0; loss_ent = 0.0
        if s2_logits is not None and mask is not None:
            l, _ = region_loss(s2_logits, mask, class_weights=cw,
                                gc_dice_weight=cfg.train.gc_dice_weight,
                                ignore_index=-100)
            loss_s2 = float(l.detach()); loss = loss + l

        if s1_target is not None and w_s1 > 0:
            ls1 = F.binary_cross_entropy_with_logits(
                s1_logits, s1_target,
                pos_weight=torch.tensor(cfg.train.s1_pos_weight, device=device),
            )
            loss_s1 = float(ls1.detach())
            loss = loss + w_s1 * ls1

        if w_ent > 0:
            avg_p = torch.sigmoid(s1_logits).mean()
            le = entropy_reg(avg_p)
            loss_ent = float(le.detach())
            loss = loss + w_ent * le
            avg_s1_prob_acc += float(avg_p.detach())

        if isinstance(loss, float):
            continue
        loss_val = float(loss.detach())
        if not np.isfinite(loss_val):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        any_nan_grad = False
        for p in list(stage1.parameters()) + list(stage2.parameters()):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                any_nan_grad = True; break
        if any_nan_grad:
            optimizer.zero_grad(set_to_none=True); continue
        torch.nn.utils.clip_grad_norm_(stage1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(stage2.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss_val; tot_s2 += loss_s2; tot_s1 += loss_s1; tot_ent += loss_ent
        n_iter += 1
        if (slide_idx + 1) % 50 == 0:
            print(f"  [E{epoch_idx} eps={eps_explore:.2f} slide {slide_idx+1}/{len(train_entries)}] "
                  f"loss={tot_loss/max(1,n_iter):.4f} s2={tot_s2/max(1,n_iter):.4f} "
                  f"s1={tot_s1/max(1,n_iter):.4f} ent={tot_ent/max(1,n_iter):.4f} "
                  f"avg_s1_prob={avg_s1_prob_acc/max(1,n_iter):.3f}")
    return {"loss": tot_loss/max(1,n_iter), "s2": tot_s2/max(1,n_iter),
            "s1": tot_s1/max(1,n_iter), "ent": tot_ent/max(1,n_iter),
            "avg_s1_prob": avg_s1_prob_acc/max(1,n_iter),
            "n_iter": n_iter}


def val_epoch(stage1, stage2, val_entries, device, cfg):
    """Val pass uses threshold=0.5 (inference-mode selection) — measures
    cascade quality, not training-time stochastic-K quality.
    """
    stage1.eval(); stage2.eval()
    cw = torch.tensor(list(cfg.train.class_weights), device=device, dtype=torch.float32)
    tot = 0.0; n = 0
    with torch.no_grad():
        for entry in val_entries:
            # For val we use deterministic threshold-based selection
            try:
                s2_logits, mask, _, _ = joint_forward(
                    stage1, stage2, entry, device,
                    eps_explore=0.0,  # zero noise -> deterministic from threshold
                    max_windows=int(cfg.train.get("val_max_windows_per_slide", 16)),
                )
            except Exception:
                continue
            if s2_logits is None or mask is None:
                continue
            l, _ = region_loss(s2_logits, mask, class_weights=cw,
                                gc_dice_weight=cfg.train.gc_dice_weight,
                                ignore_index=-100)
            lv = float(l.detach())
            if not np.isfinite(lv):
                continue
            tot += lv; n += 1
    return {"val_loss": tot/max(1,n), "n_eval": n}


@hydra.main(version_base=None, config_path="configs/cascade",
            config_name="config_proposalA")
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

    # Random-init both stages
    stage1 = GraphTLSDetector(
        in_dim=int(cfg.s1.in_dim),
        hidden_dim=int(cfg.s1.hidden_dim),
        n_hops=int(cfg.s1.n_hops),
        gnn_type=str(cfg.s1.gnn_type),
        dropout=float(cfg.s1.dropout),
        gat_heads=int(cfg.s1.gat_heads),
    ).to(device).train()
    stage2 = RegionDecoder(
        uni_dim=int(cfg.s2.uni_dim),
        gat_dim=int(cfg.s2.gat_dim),
        hidden_channels=int(cfg.s2.hidden_channels),
        n_classes=int(cfg.s2.n_classes),
        grid_n=int(cfg.s2.grid_n),
        rgb_pretrained=bool(cfg.s2.rgb_pretrained),
        freeze_rgb_encoder=bool(cfg.s2.freeze_rgb_encoder),
        head_mode="argmax",
    ).to(device).train()
    n_s1 = sum(p.numel() for p in stage1.parameters())
    n_s2 = sum(p.numel() for p in stage2.parameters())
    print(f"Stage 1: {n_s1:,} params; Stage 2: {n_s2:,} params (random init)")

    optimizer = AdamW([
        {"params": stage1.parameters(), "lr": float(cfg.train.lr_stage1)},
        {"params": stage2.parameters(), "lr": float(cfg.train.lr_stage2)},
    ], weight_decay=float(cfg.train.weight_decay))

    entries = ps.build_slide_entries()
    fold_idx = int(cfg.fold_idx)
    folds, _ = ps.create_splits(entries, k_folds=int(cfg.k_folds), seed=int(cfg.seed))
    val_entries = folds[fold_idx]
    train_entries = []
    for f in range(int(cfg.k_folds)):
        if f != fold_idx:
            train_entries.extend(folds[f])
    print(f"Fold {fold_idx}: train={len(train_entries)} val={len(val_entries)}")

    eps_start = float(cfg.train.eps_start)
    eps_end = float(cfg.train.eps_end)
    n_epochs = int(cfg.train.epochs)
    best_val = float("inf"); best_ep = -1
    for ep in range(1, n_epochs + 1):
        eps = eps_start + (eps_end - eps_start) * (ep - 1) / max(1, n_epochs - 1)
        t0 = time.time()
        tr = train_epoch(stage1, stage2, train_entries, optimizer, device,
                         cfg, eps_explore=eps, epoch_idx=ep)
        tr_t = time.time() - t0
        t0 = time.time()
        va = val_epoch(stage1, stage2, val_entries, device, cfg)
        v_t = time.time() - t0
        is_best = va["val_loss"] < best_val
        marker = " BEST" if is_best else ""
        print(f"EPOCH ep={ep} eps={eps:.3f} train_loss={tr['loss']:.4f} "
              f"s2={tr['s2']:.4f} s1={tr['s1']:.4f} ent={tr['ent']:.4f} "
              f"avg_p={tr['avg_s1_prob']:.3f} val_loss={va['val_loss']:.4f} "
              f"({tr_t:.0f}s/{v_t:.0f}s){marker}")
        if is_best:
            best_val = va["val_loss"]; best_ep = ep
            torch.save({
                "stage1_state_dict": stage1.state_dict(),
                "stage2_state_dict": stage2.state_dict(),
                "epoch": ep, "val_loss": va["val_loss"],
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, out_dir / "best_checkpoint.pt")
    print(f"Best at epoch {best_ep}, val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
