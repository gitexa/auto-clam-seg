"""GARS Stage 2: UNIv2PixelDecoder — patch-level 3-class segmentation.

Scaffolded on 2026-04-29 from the recovered wandb config + log of run
`mn5jorov` (gars_stage2_univ2_decoder_20260427_102326). Architecture
shapes match the surviving `best_checkpoint.pt` exactly (verified by
loading state_dict with strict=True; see verify_stage2_ckpt.py).

Source code for the original training script was lost with the VM; this
is a re-implementation of the same model and training loop.

Inputs (per-patch, batched):
  features:  (B, 1536)        UNI-v2 patch embedding
  mask:      (B, 256, 256)    3-class label (0=bg, 1=TLS, 2=GC)

Output:
  logits:    (B, 3, 256, 256)

Run:
    python train_gars_stage2.py --label univ2_decoder
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ─── Model ────────────────────────────────────────────────────────────


class SpatialBasis(nn.Module):
    """Map a UNI-v2 patch embedding to a small spatial feature map.

    Linear(in_dim → bottleneck) → GELU → Linear(bottleneck → S²·C)
    → reshape to (B, S, S, C) → LayerNorm(C) → permute to (B, C, S, S).
    """

    def __init__(
        self,
        in_dim: int = 1536,
        bottleneck: int = 512,
        hidden_channels: int = 64,
        spatial_size: int = 16,
    ) -> None:
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
    """Bilinear ×2 upsample + (Conv-BN-ReLU)×2.

    Indices 0,1,2,3,4,5 in the recovered state_dict are
    [Conv, BN, ReLU, Conv, BN, ReLU] — index 0/3 are conv, 1/4 are BN,
    2/5 have no params (activation).
    """

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
    """SpatialBasis → 4× UpDoubleConv → 1×1 Conv segmentation head.

    With spatial_size=16 and hidden_channels=64, output is 256×256×3
    and the model has 9 303 075 parameters (matches mn5jorov checkpoint).
    With hidden_channels=128, 17 745 059 (matches cvrvn8yb).
    """

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
        self.dec0 = UpDoubleConv(c,  d0)   # S    → 2S    (c -> 64)
        self.dec1 = UpDoubleConv(d0, d1)   # 2S   → 4S    (64 -> 32)
        self.dec2 = UpDoubleConv(d1, d2)   # 4S   → 8S    (32 -> 16)
        self.dec3 = UpDoubleConv(d2, d3)   # 8S   → 16S   (16 -> 8)
        self.seg_head = nn.Conv2d(d3, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spatial_basis(x)
        z = self.dec0(z)
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        return self.seg_head(z)


# ─── Loss ─────────────────────────────────────────────────────────────


def compute_loss(
    logits: torch.Tensor,        # (B, C, H, W)
    targets: torch.Tensor,       # (B, H, W) long
    class_weights: torch.Tensor, # (C,)
    gc_dice_weight: float = 2.0,
    gc_class: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted CE + Dice loss on the GC class.

    Recovered loss-call signature (from the spatial8 crash trace) was
    `compute_loss(logits, masks, CLASS_WEIGHTS, GC_DICE_WEIGHT)`.
    """
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    probs = F.softmax(logits, dim=1)
    gc_p = probs[:, gc_class]                # (B, H, W)
    gc_t = (targets == gc_class).float()
    inter = (gc_p * gc_t).sum(dim=(1, 2))
    union = gc_p.sum(dim=(1, 2)) + gc_t.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    dice_loss = 1.0 - dice.mean()
    return ce + gc_dice_weight * dice_loss


# ─── Metrics ──────────────────────────────────────────────────────────


def per_class_dice(logits: torch.Tensor, targets: torch.Tensor, n_classes: int = 3, eps: float = 1e-6):
    pred = logits.argmax(dim=1)              # (B, H, W)
    dices = []
    for c in range(n_classes):
        p = (pred == c).float()
        t = (targets == c).float()
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dices.append(((2 * inter + eps) / (denom + eps)).mean().item())
    return dices  # [bg, tls, gc]


# ─── Data ─────────────────────────────────────────────────────────────


class TLSPatchDataset(torch.utils.data.Dataset):
    """In-memory TLS-patch dataset — features + 256×256 3-class masks.

    Produced by `build_tls_patch_dataset()` (below). Total memory for the
    full TCGA set: ~1.9 GB (162 MB features + 1.7 GB uint8 masks).
    """

    def __init__(self, features: torch.Tensor, masks: torch.Tensor):
        assert features.shape[0] == masks.shape[0]
        self.features = features
        self.masks = masks

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return {
            "features": self.features[idx],   # (1536,)
            "mask": self.masks[idx].long(),   # (256, 256), long for CE
        }


def build_tls_patch_dataset(
    *,
    patch_size: int = 256,
    cache_path: str | None = None,
    test_csv: str | None = None,
):
    """Walk all slides, find TLS-positive patches, return (features, masks).

    This is a re-implementation skeleton — the original lived in the
    `train_gars_stage2.py` we lost. Two valid options:

    1. Re-use `prepare_segmentation.load_mask_from_tif` with
       upsample_factor=patch_size to get per-patch native-resolution
       masks; iterate patches per slide, save (feat, mask_tile_256) for
       each patch where mask>0.

    2. Open the HookNet TIF directly with `tifffile` at its full mpp,
       crop per-patch 256×256 tiles using the slide's annotation→pixel
       scale.

    The recovered output.log says ~30 s/slide for option 1 with caching;
    final dataset is 26 364 patches (1 869 with GC) over 605 slides.

    Implement once, then use `cache_path` to memoise as a single .pt.
    """
    raise NotImplementedError(
        "Plug in your TLS-patch loader here. Recovered shape spec:\n"
        "  features: (N=26364, 1536) float32\n"
        "  masks:    (N=26364, 256, 256) uint8 in {0, 1, 2}\n"
        "Memory budget: ~1.9 GB. Use prepare_segmentation.build_slide_entries() "
        "+ prepare_segmentation.load_mask_from_tif(upsample_factor=patch_size)."
    )


# ─── Schedulers ───────────────────────────────────────────────────────


def make_warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
    """Linear warmup over `warmup_epochs` then cosine decay to 0."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Training loop ────────────────────────────────────────────────────


def run_split(model, loader, device, optimizer, criterion_args, train: bool):
    if train:
        model.train()
    else:
        model.eval()
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


def main():
    ap = argparse.ArgumentParser()
    # All defaults match the recovered mn5jorov config.yaml.
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--spatial_size", type=int, default=16)
    ap.add_argument("--hidden_channels", type=int, default=64)
    ap.add_argument("--bottleneck", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[1.0, 1.0, 3.0])
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--gc_dice_weight", type=float, default=2.0)
    ap.add_argument("--label", default="univ2_decoder")
    ap.add_argument("--cache_path", default=None,
                    help="Optional .pt cache for the (features, masks) tensors")
    ap.add_argument("--out_root", default="/home/ubuntu/ahaas-persistent-std-tcga/experiments")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─ Data ─
    print("Building TLS patch dataset (pre-loading features + masks)...")
    features, masks = build_tls_patch_dataset(cache_path=args.cache_path)
    n = features.shape[0]
    n_val = int(0.157 * n)  # 4124/26364 ≈ 0.157, the recovered split ratio
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = TLSPatchDataset(features[train_idx], masks[train_idx])
    val_ds = TLSPatchDataset(features[val_idx], masks[val_idx])
    print(f"Split: {len(train_ds)} train, {len(val_ds)} val patches")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Batches/epoch: {len(train_loader)} train, {len(val_loader)} val")

    # ─ Model + opt ─
    model = UNIv2PixelDecoder(
        in_dim=1536, bottleneck=args.bottleneck,
        hidden_channels=args.hidden_channels,
        spatial_size=args.spatial_size, n_classes=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 2 (UNIv2PixelDecoder, hidden={args.hidden_channels}): {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_warmup_cosine(optimizer, args.warmup_epochs, args.epochs, args.lr)
    cw = torch.tensor(args.class_weights, device=device, dtype=torch.float32)
    criterion_args = dict(class_weights=cw, gc_dice_weight=args.gc_dice_weight)

    # ─ Output dir ─
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"gars_stage2_{args.label}_{ts}"
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Run: {run_id}\nResults: {out_dir}\n")

    # ─ Loop ─
    best_m = -1.0
    best_epoch = -1
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_split(model, train_loader, device, optimizer, criterion_args, train=True)
        train_t = time.time() - t0
        t0 = time.time()
        va = run_split(model, val_loader, device, optimizer, criterion_args, train=False)
        val_t = time.time() - t0
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        is_best = va["mDice"] > best_m
        marker = "BEST" if is_best else ""
        print(
            f"EPOCH epoch={epoch} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_mDice={va['mDice']:.4f} val_tls={va['dice_tls']:.4f} val_gc={va['dice_gc']:.4f} "
            f"lr={lr:.2e} train={train_t:.0f}s val={val_t:.0f}s {marker}"
        )
        if is_best:
            best_m, best_epoch = va["mDice"], epoch
            epochs_since_best = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": va,
                    "config": vars(args),
                },
                out_dir / "best_checkpoint.pt",
            )
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best={best_epoch}, mDice={best_m:.4f})")
                break

    print(f"\nDone. Best mDice={best_m:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
