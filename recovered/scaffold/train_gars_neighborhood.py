"""GARS v3.36 — NeighborhoodPixelDecoder (3×3 windows of UNI-v2 patches → 768×768 mask).

Per-cell SpatialBasis + spatial 3×3 tiling + shared decoder chain.
Re-uses `SpatialBasis`, `UpDoubleConv`, `make_warmup_cosine` from
`train_gars_stage2.py`.

Run:
    python train_gars_neighborhood.py
    WANDB_MODE=disabled python train_gars_neighborhood.py train.epochs=1 label=v3.36_smoke
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gars_stage2 import SpatialBasis, UpDoubleConv, make_warmup_cosine
from tls_neighborhood_dataset import (
    NeighborhoodTilesDataset,
    slide_level_split_windows,
    INVALID_MASK_VALUE,
)


# ─── Model ────────────────────────────────────────────────────────────


class NeighborhoodPixelDecoder(nn.Module):
    """Input (B, 9, 1536) features arranged in raster row-major over a
    3×3 neighborhood; output (B, 3, 768, 768) per-pixel logits.

    Each cell goes through `SpatialBasis` independently, then the 9
    feature maps are tiled into a single (B, C, 48, 48) grid that the
    decoder upsamples 2× four times → 768×768.
    """

    def __init__(
        self,
        in_dim: int = 1536,
        bottleneck: int = 512,
        hidden_channels: int = 128,
        spatial_size: int = 16,
        n_classes: int = 3,
        decoder_channels: tuple[int, int, int, int] = (64, 32, 16, 8),
        n_cells: int = 9,
        grid_n: int = 3,
    ):
        super().__init__()
        assert n_cells == grid_n * grid_n
        self.n_cells = n_cells
        self.grid_n = grid_n
        self.spatial_size = spatial_size
        self.hidden_channels = hidden_channels

        self.pad_embed = nn.Parameter(torch.zeros(in_dim))
        self.spatial_basis = SpatialBasis(in_dim, bottleneck, hidden_channels, spatial_size)
        c = hidden_channels
        d0, d1, d2, d3 = decoder_channels
        self.dec0 = UpDoubleConv(c, d0)
        self.dec1 = UpDoubleConv(d0, d1)
        self.dec2 = UpDoubleConv(d1, d2)
        self.dec3 = UpDoubleConv(d2, d3)
        self.seg_head = nn.Conv2d(d3, n_classes, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,           # (B, 9, in_dim)
        valid_mask: torch.Tensor | None,  # (B, 9) bool, optional
    ) -> torch.Tensor:
        b, k, d = features.shape
        assert k == self.n_cells
        if valid_mask is not None:
            # Replace invalid cells with the learned pad embedding.
            invalid = (~valid_mask).unsqueeze(-1)
            features = torch.where(invalid, self.pad_embed.expand_as(features), features)
        # Per-cell SpatialBasis: flatten to (b*k, d) → (b*k, C, S, S).
        z = features.reshape(b * k, d)
        z = self.spatial_basis(z)                  # (b*k, C, S, S)
        s = self.spatial_size; c = self.hidden_channels
        z = z.view(b, self.grid_n, self.grid_n, c, s, s)
        # Tile into spatial grid (B, C, n*S, n*S):
        z = z.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, self.grid_n * s, self.grid_n * s)
        # Decoder chain: 48 → 96 → 192 → 384 → 768
        z = self.dec0(z); z = self.dec1(z); z = self.dec2(z); z = self.dec3(z)
        return self.seg_head(z)


# ─── Loss ─────────────────────────────────────────────────────────────


def neighborhood_loss(
    logits: torch.Tensor,                  # (B, 3, 768, 768)
    targets: torch.Tensor,                 # (B, 768, 768) long, INVALID_MASK_VALUE for ignored
    class_weights: torch.Tensor,           # (3,)
    gc_dice_weight: float = 1.0,
    ignore_index: int = INVALID_MASK_VALUE,
    gc_class: int = 2,
    eps: float = 1e-6,
):
    ce = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=ignore_index)
    valid = (targets != ignore_index).float()                 # (B, 768, 768)
    probs = F.softmax(logits, dim=1)
    gc_p = probs[:, gc_class] * valid
    gc_t = (targets == gc_class).float() * valid
    inter = (gc_p * gc_t).sum(dim=(1, 2))
    union = gc_p.sum(dim=(1, 2)) + gc_t.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    return ce + gc_dice_weight * (1.0 - dice.mean()), {
        "ce": float(ce.detach()),
        "gc_dice": float(dice.mean().detach()),
    }


def per_class_dice_neighborhood(
    logits: torch.Tensor, targets: torch.Tensor,
    n_classes: int = 3, ignore_index: int = INVALID_MASK_VALUE, eps: float = 1e-6,
):
    pred = logits.argmax(dim=1)
    valid = (targets != ignore_index)
    out = []
    for c in range(n_classes):
        p = ((pred == c) & valid).float()
        t = ((targets == c) & valid).float()
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        out.append(((2 * inter + eps) / (denom + eps)).mean().item())
    return out


# ─── Train / val loop ─────────────────────────────────────────────────


def run_split(model, loader, device, optimizer, criterion_args, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    sums = [0.0, 0.0, 0.0]
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            logits = model(features, valid)
            loss, _ = neighborhood_loss(logits, mask, **criterion_args)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * features.shape[0]
            d = per_class_dice_neighborhood(logits.detach(), mask)
            for i in range(3):
                sums[i] += d[i] * features.shape[0]
            n += features.shape[0]
    return {
        "loss": total_loss / max(1, n),
        "dice_bg": sums[0] / max(1, n),
        "dice_tls": sums[1] / max(1, n),
        "dice_gc": sums[2] / max(1, n),
        "mDice": sum(sums) / max(1, n) / 3.0,
    }


# ─── Hydra entry ──────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="configs/neighborhood", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Patch cache (TLS-positive bundles).
    from tls_patch_dataset import build_tls_patch_dataset
    bundle = build_tls_patch_dataset(cache_path=cfg.data.cache_path)
    print(f"Loaded patch cache: {bundle['features'].shape[0]:,} positive patches")

    train_short, val_short, n_val_p = slide_level_split_windows(
        bundle, val_frac=cfg.train.val_frac, seed=cfg.seed,
    )
    print(f"Slide split: {len(train_short)} train, {len(val_short)} val "
          f"({n_val_p} val patients)")
    (out_dir / "val_slides.json").write_text(json.dumps(sorted(val_short), indent=2))
    (out_dir / "train_slides.json").write_text(json.dumps(sorted(train_short), indent=2))

    train_ds = NeighborhoodTilesDataset(
        bundle, slide_filter=train_short,
        max_windows_per_slide=cfg.data.max_windows_per_slide,
        tile_clusters=cfg.data.tile_clusters, stride=cfg.data.stride,
        seed=cfg.seed,
    )
    val_ds = NeighborhoodTilesDataset(
        bundle, slide_filter=val_short,
        max_windows_per_slide=cfg.data.max_windows_per_slide,
        tile_clusters=cfg.data.tile_clusters, stride=cfg.data.stride,
        seed=cfg.seed + 1,
    )
    print(f"Windows: {len(train_ds):,} train, {len(val_ds):,} val")

    nw = cfg.train.num_workers
    pf = cfg.train.get("prefetch_factor", 4) if nw > 0 else None
    pw = cfg.train.get("persistent_workers", True) if nw > 0 else False
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, prefetch_factor=pf, persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, prefetch_factor=pf, persistent_workers=pw,
    )

    model = NeighborhoodPixelDecoder(
        in_dim=cfg.model.in_dim, bottleneck=cfg.model.bottleneck,
        hidden_channels=cfg.model.hidden_channels,
        spatial_size=cfg.model.spatial_size, n_classes=cfg.model.n_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NeighborhoodPixelDecoder (h={cfg.model.hidden_channels}): {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = make_warmup_cosine(optimizer, cfg.train.warmup_epochs, cfg.train.epochs)
    cw = torch.tensor(list(cfg.train.class_weights), device=device, dtype=torch.float32)
    criterion_args = dict(class_weights=cw, gc_dice_weight=cfg.train.gc_dice_weight)

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

    best_m = -1.0
    best_epoch = -1
    epochs_since_best = 0
    last_va = None
    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr = run_split(model, train_loader, device, optimizer, criterion_args, train=True)
        train_t = time.time() - t0
        t0 = time.time()
        va = run_split(model, val_loader, device, optimizer, criterion_args, train=False)
        val_t = time.time() - t0
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        last_va = va

        is_best = va["mDice"] > best_m
        marker = "BEST" if is_best else ""
        print(
            f"EPOCH epoch={epoch} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_mDice={va['mDice']:.4f} val_tls={va['dice_tls']:.4f} val_gc={va['dice_gc']:.4f} "
            f"lr={lr:.2e} train={train_t:.0f}s val={val_t:.0f}s {marker}"
        )
        if run is not None:
            run.log({
                "epoch": epoch, "lr": lr,
                "train/loss": tr["loss"],
                "val/loss": va["loss"], "val/mDice": va["mDice"],
                "val/dice_tls": va["dice_tls"], "val/dice_gc": va["dice_gc"],
                "best_mDice": max(best_m, va["mDice"]),
            }, step=epoch)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "val_metrics": va,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, out_dir / "last.pt")

        if is_best:
            best_m, best_epoch = va["mDice"], epoch
            epochs_since_best = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_metrics": va,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, out_dir / "best_checkpoint.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.train.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best={best_epoch}, mDice={best_m:.4f})")
                break

    if last_va is not None:
        (out_dir / "final_results.json").write_text(json.dumps(last_va, indent=2))
    print(f"\nDone. Best mDice={best_m:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best_checkpoint.pt'}")
    if run is not None:
        run.summary["best_mDice"] = best_m
        run.summary["best_epoch"] = best_epoch
        run.finish()


if __name__ == "__main__":
    main()
