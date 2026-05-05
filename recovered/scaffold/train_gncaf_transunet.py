"""Train the TransUNet GNCAF (`GNCAFPixelDecoder`) — paper-faithful
architecture, intended to beat the recovered checkpoints' best
mDice=0.7143 on per-positives val.

Differences from `train_gncaf.py`:
- Uses `GNCAFPixelDecoder` (R50 + 6 ViT + GCN-3hop + 1 fusion + decoder)
  rather than the vanilla-ViT `GNCAF` re-implementation.
- The model expects `target_rgb`, `all_features`, `target_idx`,
  `edge_index` — same forward signature, so the existing
  `GNCAFFastDataset` + training loop transfers without changes.
- Loss: cross-entropy with optional class_weights + GC dice (matches
  `train_gncaf.neighborhood_loss`).

Run:
    python train_gncaf_transunet.py
    WANDB_MODE=disabled python train_gncaf_transunet.py train.epochs=1 label=smoke
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
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gncaf_transunet_model import GNCAFPixelDecoder, model_summary
from gncaf_dataset import build_gncaf_split
from train_gncaf import GNCAFFastDataset, build_pos_lookup, collate_keep_first


def per_class_dice_batch(pred_argmax: torch.Tensor, target: torch.Tensor,
                          n_classes: int = 3, eps: float = 1e-6):
    out = {}
    for c in range(n_classes):
        p = (pred_argmax == c)
        t = (target == c)
        out[c] = (int((p & t).sum()), int((p.sum() + t.sum())))
    return out


@torch.no_grad()
def validate(model, val_loader, device, n_classes=3):
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
                torch.cuda.empty_cache(); continue
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


@hydra.main(version_base=None, config_path="configs/gncaf",
            config_name="config_transunet_paper")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_entries, val_entries = build_gncaf_split(
        seed=cfg.seed,
        k_folds=cfg.get("k_folds", 5),
        fold_idx=cfg.get("fold_idx", 0),
    )
    pos_lookup = build_pos_lookup()

    train_ds = GNCAFFastDataset(
        train_entries, pos_lookup,
        max_pos=cfg.train.max_pos_per_slide,
        bg_per_pos=cfg.train.bg_per_pos,
        rng_seed=cfg.seed,
        include_negative_slides=cfg.train.get("include_negative_slides", False),
        neg_slide_targets=cfg.train.get("neg_slide_targets", 4),
    )
    val_ds = GNCAFFastDataset(
        val_entries, pos_lookup,
        max_pos=cfg.train.max_pos_per_slide,
        bg_per_pos=cfg.train.bg_per_pos,
        rng_seed=cfg.seed + 1,
    )
    print(f"Train: {len(train_ds)} slides, Val: {len(val_ds)} slides")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg.train.num_workers,
                              prefetch_factor=2, persistent_workers=True,
                              collate_fn=collate_keep_first)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.train.num_workers,
                            prefetch_factor=2, persistent_workers=True,
                            collate_fn=collate_keep_first)

    model = GNCAFPixelDecoder(
        hidden_size=cfg.model.hidden_size,
        n_classes=cfg.model.n_classes,
        n_encoder_layers=cfg.model.n_encoder_layers,
        n_heads=cfg.model.n_heads,
        mlp_dim=cfg.model.mlp_dim,
        n_hops=cfg.model.n_hops,
        n_fusion_layers=cfg.model.n_fusion_layers,
        feature_dim=cfg.model.feature_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    summary = model_summary(model)
    print(f"GNCAFPixelDecoder params: " +
          ", ".join(f"{k}={v:,}" for k, v in summary.items()))

    # Load ImageNet-pretrained R50 weights into the encoder trunk.
    if cfg.train.get("load_imagenet_r50", False):
        from gncaf_transunet_model import load_imagenet_r50_into_encoder
        loaded, skipped = load_imagenet_r50_into_encoder(model.encoder)
        print(f"  loaded ImageNet R50 weights: {loaded} loaded, {skipped} skipped")

    # Load ImageNet-pretrained ViT weights into the encoder transformer.
    if cfg.train.get("load_imagenet_vit", False):
        from gncaf_transunet_model import load_imagenet_vit_into_encoder
        loaded, skipped = load_imagenet_vit_into_encoder(model.encoder)
        print(f"  loaded ImageNet ViT weights: {loaded} loaded, {skipped} skipped")

    # Optionally freeze the R50 trunk (paper config: freeze_cnn=True).
    if cfg.train.get("freeze_cnn", True):
        for n, p in model.encoder.named_parameters():
            if any(n.startswith(s) for s in ("stem_conv", "layer1", "layer2", "layer3")):
                p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
        print(f"  froze R50 trunk ({n_frozen:,} params)")

    cw = torch.tensor(cfg.train.class_weights, dtype=torch.float32, device=device)

    def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                       n_classes: int = 3, eps: float = 1e-6) -> torch.Tensor:
        """Soft multiclass dice — averages over foreground classes (1, 2)."""
        probs = F.softmax(logits, dim=1)               # (B, C, H, W)
        target_oh = F.one_hot(target.clamp(0, n_classes - 1), n_classes)  # (B,H,W,C)
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        per_class_dice = []
        for c in (1, 2):
            inter = (probs[:, c] * target_oh[:, c]).sum(dim=(1, 2))
            denom = probs[:, c].sum(dim=(1, 2)) + target_oh[:, c].sum(dim=(1, 2))
            d = (2 * inter + eps) / (denom + eps)
            per_class_dice.append(d)
        return 1.0 - torch.stack(per_class_dice).mean()

    dice_w = float(cfg.train.get("dice_loss_weight", 0.0))

    def augment_pair(rgb: torch.Tensor, mask: torch.Tensor):
        """Mirror RGB+mask through random flip + 90° rotation. Per-batch."""
        if torch.rand(1).item() < 0.5:
            rgb = torch.flip(rgb, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() < 0.5:
            rgb = torch.flip(rgb, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            rgb = torch.rot90(rgb, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        return rgb, mask

    use_aug = bool(cfg.train.get("augment", False))
    print(f"  loss: CE(weight={cfg.train.class_weights}) + {dice_w}×Dice"
          f"  augment={use_aug}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 1e-2,
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.train.amp else None

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
                if use_aug:
                    target_rgb, target_mask = augment_pair(target_rgb, target_mask)
                try:
                    if scaler is not None:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(target_rgb, features, target_idx, edge_index)
                            loss = F.cross_entropy(logits, target_mask, weight=cw)
                            if dice_w > 0:
                                loss = loss + dice_w * soft_dice_loss(logits, target_mask)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        if cfg.train.get("grad_clip", 0) > 0:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in model.parameters() if p.requires_grad],
                                cfg.train.grad_clip,
                            )
                        scaler.step(opt); scaler.update()
                    else:
                        logits = model(target_rgb, features, target_idx, edge_index)
                        loss = F.cross_entropy(logits, target_mask, weight=cw)
                        if dice_w > 0:
                            loss = loss + dice_w * soft_dice_loss(logits, target_mask)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        if cfg.train.get("grad_clip", 0) > 0:
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in model.parameters() if p.requires_grad],
                                cfg.train.grad_clip,
                            )
                        opt.step()
                    ep_loss += float(loss) * target_mask.numel()
                    ep_pix += target_mask.numel()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); n_skip += 1; continue
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
               f"train={train_t:.0f}s val={val_t:.0f}s skipped_oom={n_skip}"
               + (" BEST" if is_best else ""))
        print(msg)
        if run is not None:
            run.log({
                "epoch": ep, "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/dice_tls": val_metrics["dice_tls"],
                "val/dice_gc": val_metrics["dice_gc"],
                "val/mDice": val_metrics["mDice"],
                "val/dice_bg": val_metrics["dice_bg"],
                "lr": opt.param_groups[0]["lr"],
            })

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": ep, "best_mdice": best_mDice,
            "val_metrics": val_metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, out_dir / "last.pt")

        if is_best:
            best_mDice = val_metrics["mDice"]; best_epoch = ep
            patience_left = cfg.train.patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep, "best_mdice": best_mDice,
                "val_metrics": val_metrics,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, out_dir / "best_checkpoint.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {ep} (best ep{best_epoch} mDice={best_mDice:.4f})")
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
