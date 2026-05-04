"""End-to-end cascade training (v4.0).

Joint optimization of Stage 1 (GraphTLSDetector) + Stage 2 (RegionDecoder).
Stage 2's segmentation loss is weighted by Stage 1's continuous score so
that gradient flows back into Stage 1, refining patch selection based on
how well Stage 2 actually segments those patches.

Forward (per slide):
  1. Stage 1 forward on full graph -> s1_logits (N,), s1_score = sigmoid.
  2. Top-K positive (or random if all-bg) patches by s1_score.
  3. For each picked patch i: build 3x3 region window centred on i,
     fetch RGB tiles + UNI features + Stage-1 graph context + GT mask.
  4. Stage 2 forward -> per-patch seg logits (K, 3, 768, 768).
  5. Loss = L_s1_bce + lambda_s2 * mean_k(s1_score_k * (CE_k + 0.5*Dice_k)).

Backprop through both stages updates Stage 1 (improves selection) AND
Stage 2 (improves decoding) jointly.
"""
from __future__ import annotations

import os, sys, time, json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gncaf_dataset import (slide_wsi_path, slide_mask_path, _read_target_rgb_tile,
                           _normalise_rgb, build_gncaf_split)
from region_decoder_model import RegionDecoder, extract_stage1_context
from train_gars_stage1 import GraphTLSDetector

PATCH_SIZE = 256
GRID_SIZE = 768
INVALID_MASK_VALUE = 255


def _load_features_and_graph(zarr_path: str):
    z = zarr.open(zarr_path, mode="r")
    features = np.asarray(z["features"][:])
    coords = np.asarray(z["coords"][:])
    if "graph_edges_1hop" in z:
        edge_index = np.asarray(z["graph_edges_1hop"][:])
    else:
        edge_index = np.asarray(z["edge_index"][:])
    return features, coords, edge_index


class E2ESlideDataset(Dataset):
    def __init__(self, entries, pos_lookup):
        self.entries = [e for e in entries if e.get("zarr_path")]
        self.pos_lookup = pos_lookup

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        short = entry["slide_id"].split(".")[0]
        features, coords, edge_index = _load_features_and_graph(entry["zarr_path"])
        n = features.shape[0]

        pos_set = self.pos_lookup.get(short, set())
        pos_label = np.zeros(n, dtype=np.float32)
        for i in pos_set:
            if 0 <= i < n:
                pos_label[i] = 1.0

        wsi_path = slide_wsi_path(entry)
        mask_path = slide_mask_path(entry)
        if not os.path.exists(wsi_path):
            return None
        is_negative = mask_path is None or not os.path.exists(mask_path)

        gx = (coords[:, 0] / PATCH_SIZE).astype(np.int64)
        gy = (coords[:, 1] / PATCH_SIZE).astype(np.int64)
        gx0 = gx - gx.min()
        gy0 = gy - gy.min()
        coord_to_idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(gy0, gx0))}

        return {
            "slide_id": short,
            "features": torch.from_numpy(features),
            "edge_index": torch.from_numpy(edge_index).long(),
            "coords": torch.from_numpy(coords).float(),
            "gy0": torch.from_numpy(gy0).long(),
            "gx0": torch.from_numpy(gx0).long(),
            "coord_to_idx": coord_to_idx,
            "pos_label": torch.from_numpy(pos_label),
            "wsi_path": wsi_path,
            "mask_path": mask_path,
            "is_negative": is_negative,
            "n_patches": n,
        }


def _collate_keep_first(batch):
    return [b for b in batch if b is not None]


def _build_window(sample, center_idx, mask_z, wsi_z, graph_ctx):
    """3x3 region window centred at `center_idx` (slide-local grid coord)."""
    coords = sample["coords"]
    gy0 = sample["gy0"]
    gx0 = sample["gx0"]
    coord_to_idx = sample["coord_to_idx"]
    features = sample["features"]
    is_negative = sample["is_negative"]
    gat_dim = graph_ctx.shape[1]

    cy, cx = int(gy0[center_idx].item()), int(gx0[center_idx].item())
    top_y, top_x = cy - 1, cx - 1

    rgb_tiles = torch.zeros(9, 3, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)
    uni_features = torch.zeros(9, features.shape[1], dtype=torch.float32)
    graph_ctx_w = torch.zeros(9, gat_dim, dtype=torch.float32)
    valid_mask = torch.zeros(9, dtype=torch.bool)
    mask = torch.full((GRID_SIZE, GRID_SIZE), INVALID_MASK_VALUE, dtype=torch.long)

    for k in range(9):
        dy, dx = divmod(k, 3)
        gy, gx = top_y + dy, top_x + dx
        pi = coord_to_idx.get((gy, gx))
        if pi is None:
            continue
        valid_mask[k] = True
        uni_features[k] = features[pi]
        graph_ctx_w[k] = graph_ctx[pi].cpu()
        x0 = int(coords[pi, 0].item())
        y0 = int(coords[pi, 1].item())
        rgb = _read_target_rgb_tile(wsi_z, x0, y0)
        rgb_tiles[k] = _normalise_rgb(rgb)
        y0p, x0p = dy * PATCH_SIZE, dx * PATCH_SIZE
        if is_negative or mask_z is None:
            tile = torch.zeros(PATCH_SIZE, PATCH_SIZE, dtype=torch.long)
        else:
            mh, mw = mask_z.shape
            ye, xe = min(y0 + PATCH_SIZE, mh), min(x0 + PATCH_SIZE, mw)
            tile_np = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            tile_np[: ye - y0, : xe - x0] = np.asarray(mask_z[y0:ye, x0:xe])
            tile = torch.from_numpy(tile_np.astype(np.int64))
        mask[y0p:y0p + PATCH_SIZE, x0p:x0p + PATCH_SIZE] = tile

    return {"rgb_tiles": rgb_tiles, "uni_features": uni_features,
            "graph_ctx": graph_ctx_w, "valid_mask": valid_mask, "mask": mask}


def soft_dice_loss(logits, target, ignore_index=INVALID_MASK_VALUE, eps=1e-6):
    valid = (target != ignore_index)
    target_clamped = torch.where(valid, target, torch.zeros_like(target))
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target_clamped, num_classes=3).permute(0, 3, 1, 2).float()
    target_oh = target_oh * valid.unsqueeze(1).float()
    per_class = []
    for c in (1, 2):
        inter = (probs[:, c] * target_oh[:, c]).sum(dim=(1, 2))
        denom = (probs[:, c] * valid.float()).sum(dim=(1, 2)) + target_oh[:, c].sum(dim=(1, 2))
        per_class.append((2 * inter + eps) / (denom + eps))
    return 1.0 - torch.stack(per_class).mean()


def per_patch_seg_loss(logits, mask, cw, dice_w=0.5):
    ce = F.cross_entropy(logits, mask, weight=cw, ignore_index=INVALID_MASK_VALUE,
                         reduction="mean")
    if dice_w > 0:
        return ce + dice_w * soft_dice_loss(logits, mask)
    return ce


@hydra.main(version_base=None, config_path="configs/e2e", config_name="config_v4_0")
def main(cfg: DictConfig):
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_entries, val_entries = build_gncaf_split()
    from train_gncaf import build_pos_lookup
    pos_lookup = build_pos_lookup()

    train_ds = E2ESlideDataset(train_entries, pos_lookup)
    val_ds = E2ESlideDataset(
        [e for e in val_entries if e.get("mask_path")], pos_lookup)
    print(f"Train: {len(train_ds)} slides, Val: {len(val_ds)} slides")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg.train.num_workers,
                              prefetch_factor=2, persistent_workers=True,
                              collate_fn=_collate_keep_first)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.train.num_workers,
                            prefetch_factor=2, persistent_workers=True,
                            collate_fn=_collate_keep_first)

    s1_ckpt = torch.load(cfg.stage1, map_location=device, weights_only=False)
    s1_cfg = s1_ckpt.get("config", {}).get("model", {}) if isinstance(s1_ckpt, dict) else {}
    stage1 = GraphTLSDetector(
        in_dim=s1_cfg.get("in_dim", 1536),
        hidden_dim=s1_cfg.get("hidden_dim", 256),
        n_hops=s1_cfg.get("n_hops", 5),
        gnn_type=s1_cfg.get("gnn_type", "gatv2"),
        dropout=s1_cfg.get("dropout", 0.1),
        gat_heads=s1_cfg.get("gat_heads", 4),
    ).to(device)
    msd = s1_ckpt.get("model_state_dict", s1_ckpt) if isinstance(s1_ckpt, dict) else s1_ckpt
    stage1.load_state_dict(msd, strict=False)
    print(f"Stage 1 loaded from {cfg.stage1}")

    stage2 = RegionDecoder(
        uni_dim=1536, gat_dim=cfg.model.get("gat_dim", 256),
        hidden_channels=cfg.model.get("hidden_channels", 64),
        n_classes=3, grid_n=3, rgb_pretrained=True,
        freeze_rgb_encoder=False,
    ).to(device)
    if cfg.get("stage2_init"):
        s2_ckpt = torch.load(cfg.stage2_init, map_location=device, weights_only=False)
        msd2 = s2_ckpt.get("model_state_dict", s2_ckpt) if isinstance(s2_ckpt, dict) else s2_ckpt
        stage2.load_state_dict(msd2, strict=False)
        print(f"Stage 2 seeded from {cfg.stage2_init}")

    s1_lr = float(cfg.train.s1_lr)
    s2_lr = float(cfg.train.s2_lr)
    opt = torch.optim.AdamW([
        {"params": stage1.parameters(), "lr": s1_lr},
        {"params": stage2.parameters(), "lr": s2_lr},
    ], weight_decay=cfg.train.weight_decay)

    cw = torch.tensor(cfg.train.class_weights, dtype=torch.float32, device=device)
    dice_w = float(cfg.train.get("dice_loss_weight", 0.5))
    lambda_s2 = float(cfg.train.get("lambda_s2", 1.0))
    K = int(cfg.train.top_k)
    e2e_mode = cfg.train.get("e2e_mode", "soft")

    wsi_cache, mask_cache = {}, {}

    def _wsi(p):
        if p not in wsi_cache:
            if len(wsi_cache) > 16:
                wsi_cache.clear()
            import tifffile
            wsi_cache[p] = zarr.open(tifffile.imread(p, aszarr=True, level=0), mode="r")
        return wsi_cache[p]

    def _mask(p):
        if p is None:
            return None
        if p not in mask_cache:
            if len(mask_cache) > 16:
                mask_cache.clear()
            import tifffile
            mask_cache[p] = zarr.open(tifffile.imread(p, aszarr=True, level=0), mode="r")
        return mask_cache[p]

    print(f"E2E config: K={K} mode={e2e_mode} s1_lr={s1_lr} s2_lr={s2_lr} "
          f"lambda_s2={lambda_s2} dice_w={dice_w}")
    print(f"  Stage 1 params: {sum(p.numel() for p in stage1.parameters()):,}")
    print(f"  Stage 2 params: {sum(p.numel() for p in stage2.parameters()):,}")

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

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
    best_mDice = -1.0
    best_epoch = -1
    patience_left = cfg.train.patience

    for ep in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        stage1.train()
        stage2.train()
        ep_l_s1, ep_l_s2, ep_n = 0.0, 0.0, 0
        for batch in train_loader:
            for s in batch:
                feats = s["features"].to(device, non_blocking=True)
                ei = s["edge_index"].to(device, non_blocking=True)
                pos_label = s["pos_label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    s1_logits = stage1(feats, ei).squeeze(-1)
                l_s1 = bce(s1_logits.float(), pos_label)

                with torch.no_grad():
                    s1_score_d = torch.sigmoid(s1_logits).detach()
                    n = s1_score_d.numel()
                    k = min(K, n)
                    top_idx = torch.topk(s1_score_d, k=k).indices

                graph_ctx = extract_stage1_context(stage1, feats, ei)

                wsi_z = _wsi(s["wsi_path"])
                mask_z = _mask(s["mask_path"])

                rgb_l, uni_l, ctx_l, val_l, mask_l = [], [], [], [], []
                for ti in top_idx.tolist():
                    w = _build_window(s, ti, mask_z, wsi_z, graph_ctx)
                    rgb_l.append(w["rgb_tiles"])
                    uni_l.append(w["uni_features"])
                    ctx_l.append(w["graph_ctx"])
                    val_l.append(w["valid_mask"])
                    mask_l.append(w["mask"])
                rgb = torch.stack(rgb_l).to(device, non_blocking=True)
                uni = torch.stack(uni_l).to(device, non_blocking=True)
                ctx = torch.stack(ctx_l).to(device, non_blocking=True)
                vmask = torch.stack(val_l).to(device, non_blocking=True)
                gt_mask = torch.stack(mask_l).to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    s2_logits = stage2(rgb, uni, ctx, vmask)

                s1_score_top = torch.sigmoid(s1_logits[top_idx])
                l_s2_per = []
                for k_i in range(s2_logits.shape[0]):
                    l_k = per_patch_seg_loss(s2_logits[k_i:k_i + 1].float(),
                                             gt_mask[k_i:k_i + 1], cw, dice_w=dice_w)
                    l_s2_per.append(l_k)
                l_s2_per = torch.stack(l_s2_per)
                if e2e_mode == "soft":
                    l_s2 = (s1_score_top * l_s2_per).mean()
                else:
                    l_s2 = l_s2_per.mean()

                l = l_s1 + lambda_s2 * l_s2
                opt.zero_grad(set_to_none=True)
                l.backward()
                if cfg.train.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(stage1.parameters()) + list(stage2.parameters()),
                        cfg.train.grad_clip,
                    )
                opt.step()

                ep_l_s1 += float(l_s1)
                ep_l_s2 += float(l_s2)
                ep_n += 1

        train_l_s1 = ep_l_s1 / max(1, ep_n)
        train_l_s2 = ep_l_s2 / max(1, ep_n)

        stage1.eval()
        stage2.eval()
        v_l_s1, v_n = 0.0, 0
        agg_inter = {1: 0.0, 2: 0.0}
        agg_pred = {1: 0.0, 2: 0.0}
        agg_targ = {1: 0.0, 2: 0.0}
        with torch.no_grad():
            for batch in val_loader:
                for s in batch:
                    feats = s["features"].to(device, non_blocking=True)
                    ei = s["edge_index"].to(device, non_blocking=True)
                    pos_label = s["pos_label"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s1_logits = stage1(feats, ei).squeeze(-1)
                    v_l_s1 += float(bce(s1_logits.float(), pos_label))

                    s1_score = torch.sigmoid(s1_logits)
                    k = min(K, s1_score.numel())
                    top_idx = torch.topk(s1_score, k=k).indices
                    graph_ctx = extract_stage1_context(stage1, feats, ei)

                    wsi_z = _wsi(s["wsi_path"])
                    mask_z = _mask(s["mask_path"])

                    rgb_l, uni_l, ctx_l, val_l, mask_l = [], [], [], [], []
                    for ti in top_idx.tolist():
                        w = _build_window(s, ti, mask_z, wsi_z, graph_ctx)
                        rgb_l.append(w["rgb_tiles"])
                        uni_l.append(w["uni_features"])
                        ctx_l.append(w["graph_ctx"])
                        val_l.append(w["valid_mask"])
                        mask_l.append(w["mask"])
                    rgb = torch.stack(rgb_l).to(device, non_blocking=True)
                    uni = torch.stack(uni_l).to(device, non_blocking=True)
                    ctx = torch.stack(ctx_l).to(device, non_blocking=True)
                    vmask = torch.stack(val_l).to(device, non_blocking=True)
                    gt_mask = torch.stack(mask_l).to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s2_logits = stage2(rgb, uni, ctx, vmask).float()
                    pred = s2_logits.argmax(dim=1)
                    valid = (gt_mask != INVALID_MASK_VALUE)
                    for c in (1, 2):
                        p = ((pred == c) & valid).float()
                        t = ((gt_mask == c) & valid).float()
                        agg_inter[c] += float((p * t).sum())
                        agg_pred[c] += float(p.sum())
                        agg_targ[c] += float(t.sum())
                    v_n += 1

        eps = 1e-6
        d_tls = (2 * agg_inter[1] + eps) / (agg_pred[1] + agg_targ[1] + eps)
        d_gc = (2 * agg_inter[2] + eps) / (agg_pred[2] + agg_targ[2] + eps)
        m_dice = (d_tls + d_gc) / 2

        msg = (f"EPOCH ep={ep} train_l_s1={train_l_s1:.4f} train_l_s2={train_l_s2:.4f} "
               f"val_dice_tls={d_tls:.4f} val_dice_gc={d_gc:.4f} mDice={m_dice:.4f} "
               f"train={time.time()-t0:.0f}s")
        if m_dice > best_mDice:
            best_mDice = m_dice
            best_epoch = ep
            patience_left = cfg.train.patience
            torch.save({
                "stage1_state_dict": stage1.state_dict(),
                "stage2_state_dict": stage2.state_dict(),
                "epoch": ep,
                "best_mDice": best_mDice,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, out_dir / "best_checkpoint.pt")
            msg += " BEST"
        else:
            patience_left -= 1
        print(msg, flush=True)
        if run:
            run.log({
                "epoch": ep, "train/l_s1": train_l_s1, "train/l_s2": train_l_s2,
                "val/dice_tls": d_tls, "val/dice_gc": d_gc, "val/mDice": m_dice,
                "best_mDice": best_mDice,
            }, step=ep)
        if patience_left <= 0:
            print(f"Early stopping at epoch {ep} (best ep{best_epoch} mDice={best_mDice:.4f})")
            break

    json.dump({"best_mDice": best_mDice, "best_epoch": best_epoch},
              open(out_dir / "final_results.json", "w"))
    print(f"Done. best mDice={best_mDice:.4f} at ep{best_epoch}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
