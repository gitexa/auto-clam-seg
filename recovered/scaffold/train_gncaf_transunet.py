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
    head_mode = getattr(model, "head_mode", "argmax")
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
            if head_mode == "dual_sigmoid":
                tls_logits, gc_logits = logits
                # Reconstruct argmax-equivalent for the same Dice metric.
                # Decision rule: GC if gc_logits > 0; else TLS if tls_logits > 0; else bg.
                tls_pred = (torch.sigmoid(tls_logits) > 0.5).squeeze(1)
                gc_pred = (torch.sigmoid(gc_logits) > 0.5).squeeze(1)
                argmax = torch.zeros_like(target_mask)
                argmax = torch.where(tls_pred, torch.full_like(target_mask, 1), argmax)
                argmax = torch.where(gc_pred, torch.full_like(target_mask, 2), argmax)
                # Use BCE sum as a stand-in loss for logging.
                t_tls = (target_mask >= 1).float().unsqueeze(1)
                t_gc = (target_mask == 2).float().unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(tls_logits, t_tls) + \
                       F.binary_cross_entropy_with_logits(gc_logits, t_gc)
            else:
                loss = F.cross_entropy(logits, target_mask)
                argmax = logits.argmax(dim=1)
            total_loss += float(loss) * target_mask.numel()
            total_pix += target_mask.numel()
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

    model_class_name = cfg.model.get("model_class", "gncaf")
    if model_class_name == "gcunet":
        from gcunet_model import GCUNetPixelDecoder
        Model = GCUNetPixelDecoder
    elif model_class_name == "strict":
        from gncaf_strict_model import GNCAFStrict
        Model = GNCAFStrict
    else:
        Model = GNCAFPixelDecoder
    slide_aux_head = bool(cfg.train.get("slide_aux_loss_weight", 0.0) > 0)
    model_kwargs = dict(
        hidden_size=cfg.model.hidden_size,
        n_classes=cfg.model.n_classes,
        n_encoder_layers=cfg.model.n_encoder_layers,
        n_heads=cfg.model.n_heads,
        mlp_dim=cfg.model.mlp_dim,
        n_hops=cfg.model.n_hops,
        n_fusion_layers=cfg.model.n_fusion_layers,
        feature_dim=cfg.model.feature_dim,
        dropout=cfg.model.dropout,
    )
    if slide_aux_head and Model is GNCAFPixelDecoder:
        model_kwargs["slide_aux_head"] = True
    head_mode = str(cfg.model.get("head_mode", "argmax"))
    if Model is GNCAFPixelDecoder:
        model_kwargs["head_mode"] = head_mode
    if model_class_name == "strict":
        # Pass paper-strict knobs (defaults already paper-faithful in
        # GNCAFStrict, but allow YAML overrides).
        if cfg.model.get("gcn_hidden_dim") is not None:
            model_kwargs["gcn_hidden_dim"] = int(cfg.model.gcn_hidden_dim)
        if cfg.model.get("fusion_heads") is not None:
            model_kwargs["fusion_heads"] = int(cfg.model.fusion_heads)
    model = Model(**model_kwargs).to(device)
    summary = model_summary(model)
    print(f"{type(model).__name__} params: " +
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
    slide_aux_w = float(cfg.train.get("slide_aux_loss_weight", 0.0))

    # v3.63 dual-sigmoid loss: independent BCE + Dice on each binary head.
    tls_pos_w = float(cfg.train.get("tls_pos_weight", 1.0))
    gc_pos_w = float(cfg.train.get("gc_pos_weight", 1.0))
    gc_dice_w = float(cfg.train.get("gc_dice_weight", 0.0))

    def binary_dice_loss(logits, target_bin, eps: float = 1e-6):
        """Binary soft-Dice on a single sigmoid head. logits/target both (B,1,H,W)."""
        probs = torch.sigmoid(logits)
        inter = (probs * target_bin).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
        return (1.0 - (2 * inter + eps) / (denom + eps)).mean()

    def dual_sigmoid_loss(logits_pair, target_mask):
        """v3.63 dual-sigmoid loss: BCE + Dice on each head, with per-class weights."""
        tls_logits, gc_logits = logits_pair
        # target_mask is (B,H,W) with values 0/1/2; biology says GC ⊂ TLS.
        target_tls = ((target_mask >= 1).float()).unsqueeze(1)  # (B,1,H,W)
        target_gc = ((target_mask == 2).float()).unsqueeze(1)
        loss_tls_bce = F.binary_cross_entropy_with_logits(
            tls_logits, target_tls,
            pos_weight=torch.tensor(tls_pos_w, device=tls_logits.device),
        )
        loss_gc_bce = F.binary_cross_entropy_with_logits(
            gc_logits, target_gc,
            pos_weight=torch.tensor(gc_pos_w, device=gc_logits.device),
        )
        loss = loss_tls_bce + loss_gc_bce
        if dice_w > 0:
            loss = loss + dice_w * binary_dice_loss(tls_logits, target_tls)
        if gc_dice_w > 0:
            loss = loss + gc_dice_w * binary_dice_loss(gc_logits, target_gc)
        return loss

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
    if head_mode == "dual_sigmoid":
        print(f"  loss: dual_sigmoid BCE(tls_pw={tls_pos_w}, gc_pw={gc_pos_w}) "
              f"+ {dice_w}×TLS_Dice + {gc_dice_w}×GC_Dice  augment={use_aug}")
    else:
        print(f"  loss: CE(weight={cfg.train.class_weights}) + {dice_w}×Dice"
              f"  augment={use_aug}")

    def compute_loss(logits, target_mask):
        if head_mode == "dual_sigmoid":
            return dual_sigmoid_loss(logits, target_mask)
        loss = F.cross_entropy(logits, target_mask, weight=cw)
        if dice_w > 0:
            loss = loss + dice_w * soft_dice_loss(logits, target_mask)
        return loss

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
                slide_label = s.get("slide_label")
                if slide_label is not None:
                    slide_label = slide_label.to(device, non_blocking=True)
                if use_aug:
                    target_rgb, target_mask = augment_pair(target_rgb, target_mask)
                try:
                    if scaler is not None:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            if getattr(model, "has_slide_head", False) and slide_label is not None and slide_aux_w > 0:
                                logits, slide_logit = model(target_rgb, features, target_idx, edge_index,
                                                            return_slide_logit=True)
                            else:
                                logits = model(target_rgb, features, target_idx, edge_index)
                                slide_logit = None
                            loss = compute_loss(logits, target_mask)
                            if slide_logit is not None:
                                loss = loss + slide_aux_w * F.binary_cross_entropy_with_logits(
                                    slide_logit, slide_label,
                                )
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
                        if getattr(model, "has_slide_head", False) and slide_label is not None and slide_aux_w > 0:
                            logits, slide_logit = model(target_rgb, features, target_idx, edge_index,
                                                        return_slide_logit=True)
                        else:
                            logits = model(target_rgb, features, target_idx, edge_index)
                            slide_logit = None
                        loss = compute_loss(logits, target_mask)
                        if slide_logit is not None:
                            loss = loss + slide_aux_w * F.binary_cross_entropy_with_logits(
                                slide_logit, slide_label,
                            )
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
            "model_class": model_class_name,
        }, out_dir / "last.pt")

        if is_best:
            best_mDice = val_metrics["mDice"]; best_epoch = ep
            patience_left = cfg.train.patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep, "best_mdice": best_mDice,
                "val_metrics": val_metrics,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "model_class": model_class_name,
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
