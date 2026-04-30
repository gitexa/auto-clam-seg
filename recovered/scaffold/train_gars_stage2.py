"""GARS Stage 2: UNIv2PixelDecoder — patch-level 3-class segmentation.

v3.1 — hydra config + wandb logging + config dump to results dir.

Architecture (verified strict-load against recovered checkpoints):
    spatial_basis: Linear(1536→bottleneck) → GELU → Linear(→S²·C) →
                   reshape (B, S, S, C) → LayerNorm(C) → permute (B, C, S, S)
    dec0..dec3:    each = Bilinear×2 + DoubleConv(in→out)
                   channel sequence (fixed): C → 64 → 32 → 16 → 8
                   spatial sequence: 16 → 32 → 64 → 128 → 256
    seg_head:      Conv2d(8, 3, 1)

Inputs (per-patch, batched):
    features:  (B, 1536)        UNI-v2 patch embedding
    mask:      (B, 256, 256)    3-class label (0=bg, 1=TLS, 2=GC)

Run:
    python train_gars_stage2.py
    python train_gars_stage2.py model=univ2_decoder_h128
    WANDB_MODE=disabled python train_gars_stage2.py train.epochs=1 label=smoke
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ─── Model ────────────────────────────────────────────────────────────


class SpatialBasis(nn.Module):
    def __init__(self, in_dim: int = 1536, bottleneck: int = 512,
                 hidden_channels: int = 64, spatial_size: int = 16) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.hidden_channels = hidden_channels
        self.proj = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, spatial_size * spatial_size * hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        b = z.shape[0]
        z = z.view(b, self.spatial_size, self.spatial_size, self.hidden_channels)
        z = self.norm(z)
        return z.permute(0, 3, 1, 2).contiguous()


class UpDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class UNIv2PixelDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        bottleneck: int = 512,
        hidden_channels: int = 64,
        spatial_size: int = 16,
        n_classes: int = 3,
        decoder_channels: tuple[int, int, int, int] = (64, 32, 16, 8),
    ) -> None:
        super().__init__()
        c = hidden_channels
        d0, d1, d2, d3 = decoder_channels
        self.spatial_basis = SpatialBasis(in_dim, bottleneck, c, spatial_size)
        self.dec0 = UpDoubleConv(c,  d0)
        self.dec1 = UpDoubleConv(d0, d1)
        self.dec2 = UpDoubleConv(d1, d2)
        self.dec3 = UpDoubleConv(d2, d3)
        self.seg_head = nn.Conv2d(d3, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spatial_basis(x)
        z = self.dec0(z); z = self.dec1(z); z = self.dec2(z); z = self.dec3(z)
        return self.seg_head(z)


# ─── Loss + metrics ───────────────────────────────────────────────────


def compute_loss(logits, targets, class_weights, gc_dice_weight=2.0,
                 gc_class=2, eps=1e-6):
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    probs = F.softmax(logits, dim=1)
    gc_p = probs[:, gc_class]
    gc_t = (targets == gc_class).float()
    inter = (gc_p * gc_t).sum(dim=(1, 2))
    union = gc_p.sum(dim=(1, 2)) + gc_t.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    return ce + gc_dice_weight * (1.0 - dice.mean())


def per_class_dice(logits, targets, n_classes=3, eps=1e-6):
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(n_classes):
        p = (pred == c).float()
        t = (targets == c).float()
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dices.append(((2 * inter + eps) / (denom + eps)).mean().item())
    return dices


# ─── Data ─────────────────────────────────────────────────────────────


class TLSPatchDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, masks: torch.Tensor):
        assert features.shape[0] == masks.shape[0]
        self.features = features
        self.masks = masks

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return {"features": self.features[idx], "mask": self.masks[idx].long()}


def slide_level_split(bundle, val_frac=0.157, seed=42):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tls_patch_dataset import patient_id_from_slide
    slide_ids = bundle["slide_ids"]
    patients = sorted({patient_id_from_slide(s) for s in slide_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val_p = max(1, int(val_frac * len(patients)))
    val_patients = set(patients[:n_val_p])
    is_val = np.array(
        [patient_id_from_slide(s) in val_patients for s in slide_ids],
        dtype=bool,
    )
    val_idx = np.where(is_val)[0]
    train_idx = np.where(~is_val)[0]
    return train_idx, val_idx, len(val_patients)


# ─── Scheduler ────────────────────────────────────────────────────────


def make_warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Loop ─────────────────────────────────────────────────────────────


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
            logits = model(features)
            loss = compute_loss(logits, mask, **criterion_args)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * features.shape[0]
            d = per_class_dice(logits.detach(), mask)
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


@hydra.main(version_base=None, config_path="configs/stage2", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tls_patch_dataset import build_tls_patch_dataset
    bundle = build_tls_patch_dataset(cache_path=cfg.cache_path)
    features = bundle["features"]
    masks = bundle["masks"]

    train_idx, val_idx, n_val_patients = slide_level_split(
        bundle, val_frac=cfg.train.val_frac, seed=cfg.seed,
    )
    train_ds = TLSPatchDataset(features[train_idx], masks[train_idx])
    val_ds = TLSPatchDataset(features[val_idx], masks[val_idx])
    print(f"Split: {len(train_ds)} train, {len(val_ds)} val patches "
          f"({n_val_patients} val patients)")

    # Save the slide IDs that ended up in val (deduped) — needed for
    # cascade eval reproducibility independent of numpy version drift.
    slide_ids = bundle["slide_ids"]
    val_slide_ids = sorted({slide_ids[int(i)] for i in val_idx})
    train_slide_ids = sorted({slide_ids[int(i)] for i in train_idx})
    (out_dir / "val_slides.json").write_text(json.dumps(val_slide_ids, indent=2))
    (out_dir / "train_slides.json").write_text(json.dumps(train_slide_ids, indent=2))

    nw = cfg.train.num_workers
    pf = cfg.train.get("prefetch_factor", 4) if nw > 0 else None
    pw = cfg.train.get("persistent_workers", True) if nw > 0 else False
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True,
        prefetch_factor=pf, persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        prefetch_factor=pf, persistent_workers=pw,
    )
    print(f"Batches/epoch: {len(train_loader)} train, {len(val_loader)} val")

    model = UNIv2PixelDecoder(
        in_dim=cfg.model.in_dim, bottleneck=cfg.model.bottleneck,
        hidden_channels=cfg.model.hidden_channels,
        spatial_size=cfg.model.spatial_size, n_classes=cfg.model.n_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 2 (UNIv2PixelDecoder, hidden={cfg.model.hidden_channels}): {n_params:,} params")

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
        # Always save last.pt (for resume / inference at any epoch).
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "val_metrics": va,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                out_dir / "best_checkpoint.pt",
            )
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
