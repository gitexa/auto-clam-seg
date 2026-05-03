"""GNCAF training loop (recovered architecture, lost training script).

Per-slide forward:
    1. Pre-loaded UNI-v2 features + 4-conn edge_index (cheap, from zarr).
    2. Sample TLS-positive patches via pre-computed pos_idx_lookup from
       tls_patch_dataset cache (skips the slow centre-pixel scan that
       was the bottleneck in the original GNCAFSlideDataset).
    3. Read target RGB tiles from the WSI tif level-0 (windowed reads).
    4. Forward GNCAF: ViT encoder + GCN context + fusion + 4× upsample
       decoder → (K, 3, 256, 256) per-pixel logits.
    5. CE loss with class_weights (recovered v3.4b: [1, 5, 3]).

Hydra entry; outputs:
    out_dir/
      best_checkpoint.pt
      last.pt
      config.yaml
      train_gncaf.log
      final_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import tifffile
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gncaf_model import GNCAF
from gncaf_dataset import (
    PATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    _read_target_rgb_tile, _read_target_mask_tile, _normalise_rgb,
    _load_features_and_graph, slide_wsi_path, slide_mask_path,
    build_gncaf_split,
)
from tls_patch_dataset import build_tls_patch_dataset


def per_class_dice_batch(pred_argmax: torch.Tensor, target: torch.Tensor,
                         n_classes: int = 3, eps: float = 1e-6
                         ) -> dict[int, tuple[int, int]]:
    """Per-class (intersection, denom) over a batch."""
    out: dict[int, tuple[int, int]] = {}
    for c in range(n_classes):
        p = (pred_argmax == c)
        t = (target == c)
        out[c] = (int((p & t).sum()), int(p.sum() + t.sum()))
    return out


class GNCAFFastDataset(Dataset):
    """GNCAF dataset that uses precomputed pos_idx_lookup (from
    tls_patch_dataset cache) to skip the per-patch centre-pixel scan.
    """

    def __init__(self, entries, pos_lookup: dict[str, set[int]],
                 max_pos: int, bg_per_pos: float = 1.0, rng_seed: int = 0):
        self.entries = [e for e in entries
                        if e.get("zarr_path") and e.get("mask_path")]
        self.pos_lookup = pos_lookup
        self.max_pos = max_pos
        self.bg_per_pos = bg_per_pos
        self.rng_seed = rng_seed

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        short = entry["slide_id"].split(".")[0]
        rng = np.random.default_rng(self.rng_seed + idx + int(time.time() * 1000) % 1000)

        features, coords, edge_index = _load_features_and_graph(entry["zarr_path"])
        n = features.shape[0]

        wsi_path = slide_wsi_path(entry)
        mask_path = slide_mask_path(entry)
        if not (os.path.exists(wsi_path) and mask_path and os.path.exists(mask_path)):
            return None
        wsi_z = zarr.open(tifffile.imread(wsi_path, aszarr=True, level=0), mode="r")
        mask_z = zarr.open(tifffile.imread(mask_path, aszarr=True, level=0), mode="r")

        pos_set = self.pos_lookup.get(short, set())
        all_idx = np.arange(n)
        pos_idx = np.array([i for i in all_idx if i in pos_set], dtype=np.int64)
        neg_idx = np.array([i for i in all_idx if i not in pos_set], dtype=np.int64)

        n_pos = min(len(pos_idx), self.max_pos)
        n_bg = int(np.ceil(n_pos * self.bg_per_pos))
        n_bg = min(n_bg, len(neg_idx))
        chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.empty(0, dtype=np.int64)
        chosen_bg = rng.choice(neg_idx, size=n_bg, replace=False) if n_bg > 0 else np.empty(0, dtype=np.int64)
        target_idx = np.concatenate([chosen_pos, chosen_bg])
        if target_idx.size == 0:
            return None

        rgbs, masks = [], []
        for ti in target_idx:
            x, y = int(coords[ti, 0]), int(coords[ti, 1])
            rgb = _read_target_rgb_tile(wsi_z, x, y)
            m = _read_target_mask_tile(mask_z, x, y)
            rgbs.append(_normalise_rgb(rgb))
            masks.append(torch.from_numpy(m.astype(np.int64)))

        return {
            "features": torch.from_numpy(features),
            "edge_index": torch.from_numpy(edge_index),
            "target_idx": torch.from_numpy(target_idx),
            "target_rgb": torch.stack(rgbs),
            "target_mask": torch.stack(masks),
            "slide_id": short,
        }


def collate_keep_first(batch):
    return [b for b in batch if b is not None]


def build_pos_lookup() -> dict[str, set[int]]:
    print("Loading patch cache for pos_lookup...")
    bundle = build_tls_patch_dataset(
        cache_path="/home/ubuntu/local_data/tls_patch_dataset_min4096.pt",
        verbose=False)
    out: dict[str, set[int]] = {}
    for ci, sid in enumerate(bundle["slide_ids"]):
        short = sid.split(".")[0]
        out.setdefault(short, set()).add(int(bundle["patch_idx"][ci]))
    print(f"  pos_lookup: {len(out)} slides, "
          f"{sum(len(v) for v in out.values())} positive patches total")
    return out


@torch.no_grad()
def validate(model, val_loader, device, n_classes=3) -> dict:
    model.eval()
    n_slides = 0
    agg = {c: [0, 0] for c in range(n_classes)}
    total_loss, total_pix = 0.0, 0
    for batch in val_loader:
        for s in batch:
            features = s["features"].to(device, non_blocking=True)
            edge_index = s["edge_index"].to(device, non_blocking=True)
            target_idx = s["target_idx"].to(device, non_blocking=True)
            target_rgb = s["target_rgb"].to(device, non_blocking=True)
            target_mask = s["target_mask"].to(device, non_blocking=True)
            try:
                logits = model(target_rgb, features, target_idx, edge_index)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            loss = F.cross_entropy(logits, target_mask)
            total_loss += float(loss) * target_mask.numel()
            total_pix += target_mask.numel()
            argmax = logits.argmax(dim=1)
            for c, (i, d) in per_class_dice_batch(argmax, target_mask, n_classes).items():
                agg[c][0] += i; agg[c][1] += d
            n_slides += 1
    eps = 1e-6
    dice = {c: (2 * a[0] + eps) / (a[1] + eps) for c, a in agg.items()}
    return {
        "n_slides": n_slides,
        "loss": total_loss / max(1, total_pix),
        "dice_bg": dice[0], "dice_tls": dice[1], "dice_gc": dice[2],
        "mDice": (dice[1] + dice[2]) / 2.0,
    }


@hydra.main(version_base=None, config_path="configs/gncaf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_entries, val_entries = build_gncaf_split()
    pos_lookup = build_pos_lookup()
    train_ds = GNCAFFastDataset(train_entries, pos_lookup,
                                 max_pos=cfg.train.max_pos_per_slide,
                                 bg_per_pos=cfg.train.bg_per_pos,
                                 rng_seed=cfg.seed)
    val_ds = GNCAFFastDataset(val_entries, pos_lookup,
                               max_pos=cfg.train.max_pos_per_slide,
                               bg_per_pos=cfg.train.bg_per_pos,
                               rng_seed=cfg.seed + 1)
    print(f"Train: {len(train_ds)} slides, Val: {len(val_ds)} slides")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg.train.num_workers,
                              prefetch_factor=2, persistent_workers=True,
                              collate_fn=collate_keep_first)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.train.num_workers,
                            prefetch_factor=2, persistent_workers=True,
                            collate_fn=collate_keep_first)

    model = GNCAF(
        in_features=1536, dim=cfg.model.dim, n_classes=3,
        encoder_layers=cfg.model.encoder_layers,
        encoder_heads=cfg.model.encoder_heads,
        gcn_hops=cfg.model.gcn_hops,
        fusion_heads=cfg.model.fusion_heads,
        dropout=cfg.model.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNCAF params: {n_params:,}")

    cw = torch.tensor(cfg.train.class_weights, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                             weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 1e-2,
    )
    scaler = torch.cuda.amp.GradScaler() if cfg.train.amp else None

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

    best_mDice = -1.0; best_epoch = -1; patience_left = cfg.train.patience
    for ep in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss, ep_pix = 0.0, 0
        n_skip = 0
        for batch in train_loader:
            for s in batch:
                features = s["features"].to(device, non_blocking=True)
                edge_index = s["edge_index"].to(device, non_blocking=True)
                target_idx = s["target_idx"].to(device, non_blocking=True)
                target_rgb = s["target_rgb"].to(device, non_blocking=True)
                target_mask = s["target_mask"].to(device, non_blocking=True)
                try:
                    if scaler is not None:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            logits = model(target_rgb, features, target_idx, edge_index)
                            loss = F.cross_entropy(logits, target_mask, weight=cw)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(opt); scaler.update()
                    else:
                        logits = model(target_rgb, features, target_idx, edge_index)
                        loss = F.cross_entropy(logits, target_mask, weight=cw)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    ep_loss += float(loss) * target_mask.numel()
                    ep_pix += target_mask.numel()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); n_skip += 1
                    continue
        scheduler.step()
        train_t = time.time() - t0

        t0 = time.time()
        val_metrics = validate(model, val_loader, device)
        val_t = time.time() - t0
        train_loss = ep_loss / max(1, ep_pix)

        is_best = val_metrics["mDice"] > best_mDice
        msg = (f"EPOCH epoch={ep} train_loss={train_loss:.4f} "
               f"val_loss={val_metrics['loss']:.4f} "
               f"val_dice_tls={val_metrics['dice_tls']:.4f} "
               f"val_dice_gc={val_metrics['dice_gc']:.4f} "
               f"val_mDice={val_metrics['mDice']:.4f} "
               f"lr={opt.param_groups[0]['lr']:.2e} "
               f"train={train_t:.0f}s val={val_t:.0f}s "
               f"skipped_oom={n_skip}"
               + (" BEST" if is_best else ""))
        print(msg)
        if run is not None:
            run.log({
                "epoch": ep, "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_dice_tls": val_metrics["dice_tls"],
                "val_dice_gc": val_metrics["dice_gc"],
                "val_mDice": val_metrics["mDice"],
                "val_dice_bg": val_metrics["dice_bg"],
                "lr": opt.param_groups[0]["lr"],
            })

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": ep,
            "val_metrics": val_metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, out_dir / "last.pt")

        if is_best:
            best_mDice = val_metrics["mDice"]; best_epoch = ep; patience_left = cfg.train.patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep,
                "val_metrics": val_metrics,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, out_dir / "best_checkpoint.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {ep} (best ep{best_epoch} "
                      f"mDice={best_mDice:.4f})")
                break

    final = {"best_mDice": best_mDice, "best_epoch": best_epoch}
    (out_dir / "final_results.json").write_text(json.dumps(final, indent=2))
    if run is not None:
        run.summary["best_mDice"] = best_mDice
        run.summary["best_epoch"] = best_epoch
        run.finish()
    print(f"\nDone. best mDice={best_mDice:.4f} at ep{best_epoch}")


if __name__ == "__main__":
    main()
