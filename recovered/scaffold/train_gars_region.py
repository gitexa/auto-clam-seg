"""GARS v3.37 — RegionDecoderCascade training.

Stage 1 frozen (v3.8 ckpt). New Stage 2 = RegionDecoder using RGB +
UNI + graph context. Decodes 3×3 patch regions selected by Stage 1.

Run:
    python train_gars_region.py
    WANDB_MODE=disabled python train_gars_region.py train.epochs=1 label=v3.37_smoke
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
from train_gars_stage1 import GraphTLSDetector
from train_gars_stage2 import make_warmup_cosine
from train_gars_neighborhood import (
    neighborhood_loss as region_loss,
    per_class_dice_neighborhood as per_class_dice_region,
)
from region_decoder_model import RegionDecoder
from region_dataset import RegionDataset
from tls_neighborhood_dataset import slide_level_split_windows


def load_stage1_frozen(ckpt_path: str, device: torch.device) -> GraphTLSDetector:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    model = GraphTLSDetector(
        in_dim=m.get("in_dim", 1536),
        hidden_dim=m.get("hidden_dim", 256),
        n_hops=m.get("n_hops", 5),
        gnn_type=m.get("gnn_type", "gatv2"),
        dropout=m.get("dropout", 0.1),
        gat_heads=m.get("gat_heads", 4),
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_split(model, loader, device, optimizer, criterion_args, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0; sums = [0.0, 0.0, 0.0]; n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            rgb = batch["rgb_tiles"].to(device, non_blocking=True)
            uni = batch["uni_features"].to(device, non_blocking=True)
            gat = batch["graph_ctx"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            logits = model(rgb, uni, gat, valid)
            loss, _ = region_loss(logits, mask, **criterion_args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * rgb.shape[0]
            d = per_class_dice_region(logits.detach(), mask)
            for i in range(3):
                sums[i] += d[i] * rgb.shape[0]
            n += rgb.shape[0]
    return {
        "loss": total_loss / max(1, n),
        "dice_bg": sums[0] / max(1, n),
        "dice_tls": sums[1] / max(1, n),
        "dice_gc": sums[2] / max(1, n),
        "mDice": sum(sums) / max(1, n) / 3.0,
    }


@hydra.main(version_base=None, config_path="configs/region", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Stage 1 frozen.
    stage1 = load_stage1_frozen(cfg.stage1, device)
    print(f"Loaded frozen Stage 1: {cfg.stage1}")

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

    train_ds = RegionDataset(
        bundle, stage1_model=stage1, stage1_device=device,
        slide_filter=train_short,
        max_windows_per_slide=cfg.data.max_windows_per_slide,
        tile_clusters=cfg.data.tile_clusters, stride=cfg.data.stride,
        seed=cfg.seed,
    )
    val_ds = RegionDataset(
        bundle, stage1_model=stage1, stage1_device=device,
        slide_filter=val_short,
        max_windows_per_slide=cfg.data.max_windows_per_slide,
        tile_clusters=cfg.data.tile_clusters, stride=cfg.data.stride,
        seed=cfg.seed + 1,
    )
    print(f"Windows: {len(train_ds):,} train, {len(val_ds):,} val")

    nw = cfg.train.num_workers
    pf = cfg.train.get("prefetch_factor", 2) if nw > 0 else None
    pw = cfg.train.get("persistent_workers", True) if nw > 0 else False
    # NOTE: the dataset runs Stage 1 on the GPU for graph context. For
    # multi-worker DataLoader, this would copy stage1 into each worker —
    # OK because num_workers=0 by default. With nw>0, ensure CUDA
    # safety: dataset's stage1_device should be CPU and the slide ctx
    # cache will warm up. To keep this simple we default nw=0.
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

    model = RegionDecoder(
        uni_dim=cfg.model.uni_dim,
        gat_dim=cfg.model.gat_dim,
        hidden_channels=cfg.model.hidden_channels,
        n_classes=cfg.model.n_classes,
        grid_n=cfg.model.grid_n,
        rgb_pretrained=cfg.model.rgb_pretrained,
        freeze_rgb_encoder=cfg.model.freeze_rgb_encoder,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RegionDecoder: {n_params:,} params ({n_trainable:,} trainable)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
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

    best_m = -1.0; best_epoch = -1
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
