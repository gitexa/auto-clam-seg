"""GARS v3.5: GraphEnrichedDecoder — single-pass end-to-end pipeline.

Architecture (verified strict-load against recovered checkpoint qy3pj74h):
    Graph branch (~790 K params):
      graph_proj:  Linear(1536, 256) + LayerNorm
      gat_layers:  3× GATv2Conv(256, 64, heads=4, concat=True)
      gat_norms:   3× LayerNorm(256)                           # residual + post-norm

    Spatial branch (~9.36 M params):
      spatial_proj: Linear(1792, 512) → GELU → Linear(512, 16²·64)
                    (input is concat([uni_v2_feat (1536), graph_feat (256)]))
      spatial_norm: LayerNorm(64)
      dec_blocks:   3× UpDoubleConv (64 → 32 → 16 → 8)
      final_up:     4th UpDoubleConv (8 → 8)
      seg_head:     Conv2d(8, 3, 1)

No discrete patch selection — every patch is decoded; the graph branch
modulates *what* features each patch's decoder sees. Background patches
that get a low-TLS graph context naturally produce low-TLS-confidence
masks.

Training: per slide, decode (all TLS-positive patches) ∪ (50 % bg
subsample), in `decode_batch_size` chunks. Per-patch loss is CE +
gc_dice_weight × Dice(GC), same as Stage 2.

Run:
    python train_gars_e2e.py                      # default config
    python train_gars_e2e.py train.bg_decode_ratio=0.25
    WANDB_MODE=disabled python train_gars_e2e.py train.epochs=1 label=v3.5_smoke
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
from torch_geometric.nn import GATv2Conv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gars_stage2 import compute_loss, per_class_dice  # reuse


def make_dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Flat decoder block matching recovered qy3pj74h key layout.

    Indices: 0=Upsample, 1=Conv2d, 2=BN, 3=ReLU, 4=Conv2d, 5=BN, 6=ReLU.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def make_final_up(ch: int) -> nn.Sequential:
    """Final upsample block matching recovered key layout: 0=Upsample,
    1=Conv2d, 2=BN, 3=ReLU."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
    )


# ─── Model ────────────────────────────────────────────────────────────


class GraphEnrichedDecoder(nn.Module):
    """End-to-end model: graph context fused into per-patch pixel decoder.

    Verified strict-load against recovered checkpoint `qy3pj74h`
    (10 152 322 params, mDice=0.7202).
    """

    def __init__(
        self,
        in_dim: int = 1536,
        graph_hidden_dim: int = 256,
        n_hops: int = 3,
        gat_heads: int = 4,
        spatial_bottleneck: int = 512,
        hidden_channels: int = 64,
        spatial_size: int = 16,
        n_classes: int = 3,
        decoder_channels: tuple[int, int, int] = (32, 16, 8),
    ) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.hidden_channels = hidden_channels

        # Graph branch.
        self.graph_proj = nn.Sequential(
            nn.Linear(in_dim, graph_hidden_dim),
            nn.LayerNorm(graph_hidden_dim),
        )
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(n_hops):
            assert graph_hidden_dim % gat_heads == 0
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=graph_hidden_dim,
                    out_channels=graph_hidden_dim // gat_heads,
                    heads=gat_heads,
                    concat=True,
                    dropout=0.0,
                )
            )
            self.gat_norms.append(nn.LayerNorm(graph_hidden_dim))

        # Spatial branch — input is concat([uni_v2 (1536), graph (256)]) = 1792.
        concat_dim = in_dim + graph_hidden_dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(concat_dim, spatial_bottleneck),
            nn.GELU(),
            nn.Linear(spatial_bottleneck,
                      spatial_size * spatial_size * hidden_channels),
        )
        self.spatial_norm = nn.LayerNorm(hidden_channels)

        # Decoder: hidden → 32 → 16 → 8, each block 2× upsample.
        d0, d1, d2 = decoder_channels
        self.dec_blocks = nn.ModuleList([
            make_dec_block(hidden_channels, d0),  # 16  → 32
            make_dec_block(d0, d1),                # 32  → 64
            make_dec_block(d1, d2),                # 64  → 128
        ])
        self.final_up = make_final_up(d2)         # 128 → 256
        self.seg_head = nn.Conv2d(d2, n_classes, kernel_size=1)

    def graph_context(
        self,
        features: torch.Tensor,  # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:
        g = self.graph_proj(features)
        for layer, norm in zip(self.gat_layers, self.gat_norms):
            g = norm(layer(g, edge_index) + g)
        return g  # (N, graph_hidden_dim)

    def decode_patches(
        self,
        features: torch.Tensor,  # (B, in_dim)
        graph_feat: torch.Tensor,  # (B, graph_hidden_dim)
    ) -> torch.Tensor:
        z = torch.cat([features, graph_feat], dim=-1)         # (B, 1792)
        z = self.spatial_proj(z)                              # (B, S²·C)
        b = z.shape[0]
        z = z.view(b, self.spatial_size, self.spatial_size,
                   self.hidden_channels)
        z = self.spatial_norm(z)
        z = z.permute(0, 3, 1, 2).contiguous()                # (B, C, S, S)
        for block in self.dec_blocks:
            z = block(z)
        z = self.final_up(z)
        return self.seg_head(z)                               # (B, n_classes, 256, 256)


# ─── Data ─────────────────────────────────────────────────────────────


def patch_labels(mask: torch.Tensor, coords: torch.Tensor,
                 patch_size: int, upsample_factor: int) -> torch.Tensor:
    """Binary TLS-positive label per patch, identical to Stage 1's rule."""
    grid_x = (coords[:, 0] / patch_size).long() - (coords[:, 0] / patch_size).long().min()
    grid_y = (coords[:, 1] / patch_size).long() - (coords[:, 1] / patch_size).long().min()
    m = (mask[0] > 0).float()
    u = upsample_factor
    H, W = m.shape
    labels = torch.zeros(coords.shape[0], dtype=torch.bool)
    for i, (gx, gy) in enumerate(zip(grid_x.tolist(), grid_y.tolist())):
        y0, y1 = gy * u, min((gy + 1) * u, H)
        x0, x1 = gx * u, min((gx + 1) * u, W)
        if y1 > y0 and x1 > x0 and m[y0:y1, x0:x1].any():
            labels[i] = True
    return labels


# ─── Training step ────────────────────────────────────────────────────


def run_one_slide(
    model: GraphEnrichedDecoder,
    batch: dict,
    device: torch.device,
    cw: torch.Tensor,
    cfg_train,
    train: bool,
    rng: np.random.Generator,
) -> dict:
    """One slide: graph forward (all patches) → sample patches → decode.

    Patch sampling per slide: keep all TLS-positive patches + bg_decode_ratio
    of bg patches, capped at max_patches_per_slide.
    """
    features = batch["features"].to(device, non_blocking=True)         # (N, 1536)
    coords = batch["coords"]                                            # (N, 2)
    edge_index = batch["edge_index"].to(device, non_blocking=True)
    mask_full = batch["mask"]                                           # (1, H, W)

    n = features.shape[0]
    # Graph forward over ALL patches (always — graph context needs full neighbourhood).
    graph_feat = model.graph_context(features, edge_index)              # (N, graph_dim)

    # Per-patch binary label (TLS-positive vs bg).
    labels_bool = patch_labels(mask_full, coords,
                               cfg_train.patch_size, cfg_train.upsample_factor)
    pos_idx = torch.where(labels_bool)[0].tolist()
    neg_idx = torch.where(~labels_bool)[0].tolist()
    rng.shuffle(neg_idx)
    n_neg_keep = int(cfg_train.bg_decode_ratio * len(neg_idx))
    selected = pos_idx + neg_idx[:n_neg_keep]
    rng.shuffle(selected)
    if len(selected) > cfg_train.max_patches_per_slide:
        selected = selected[: cfg_train.max_patches_per_slide]
    if not selected:
        return None  # slide has no TLS patches and bg_decode_ratio=0; skip

    # Build per-patch supervision (256x256 mask tile per selected patch).
    # Use the existing per-patch cache built by tls_patch_dataset for the
    # mask tiles, OR derive on the fly from the upsampled mask cache.
    upsampled = mask_full[0]                                            # (H, W) values 0/1/2
    grid_x = (coords[:, 0] / cfg_train.patch_size).long()
    grid_y = (coords[:, 1] / cfg_train.patch_size).long()
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    u = cfg_train.upsample_factor

    # Stitch per-patch masks at upsampled resolution.
    # Each patch covers a u × u cell of the upsampled mask. We need a 256×256
    # mask per patch — upsample the u×u cell to 256×256 via nearest-neighbour.
    patch_masks: list[torch.Tensor] = []
    patch_feats: list[torch.Tensor] = []
    patch_graphs: list[torch.Tensor] = []
    for i in selected:
        gx, gy = int(grid_x[i]), int(grid_y[i])
        cell = upsampled[gy * u : (gy + 1) * u, gx * u : (gx + 1) * u]   # (u, u)
        if cell.shape != (u, u):
            continue
        # Nearest-neighbour upsample u×u → 256×256.
        m = cell.long().unsqueeze(0).unsqueeze(0).float()
        m = F.interpolate(m, size=(256, 256), mode="nearest").squeeze().long()
        patch_masks.append(m)
        patch_feats.append(features[i])
        patch_graphs.append(graph_feat[i])

    if not patch_masks:
        return None

    feats_t = torch.stack(patch_feats, dim=0).to(device)
    graphs_t = torch.stack(patch_graphs, dim=0).to(device)
    masks_t = torch.stack(patch_masks, dim=0).to(device)

    # Decode in chunks. The graph features are shared across chunks, so
    # we accumulate per-chunk loss into a single scalar and do one
    # backward at the end (avoids the "backward second time" error).
    accumulated_loss: torch.Tensor | None = None
    total_loss_val = 0.0
    sums = [0.0, 0.0, 0.0]
    n_dec = 0
    bs = cfg_train.decode_batch_size
    for s in range(0, feats_t.shape[0], bs):
        f = feats_t[s : s + bs]
        g = graphs_t[s : s + bs]
        y = masks_t[s : s + bs]
        logits = model.decode_patches(f, g)
        loss = compute_loss(logits, y, class_weights=cw,
                            gc_dice_weight=cfg_train.gc_dice_weight)
        weighted = loss * f.shape[0]
        if train:
            accumulated_loss = weighted if accumulated_loss is None else accumulated_loss + weighted
        total_loss_val += float(loss.detach()) * f.shape[0]
        n_dec += f.shape[0]
        d = per_class_dice(logits.detach(), y)
        for j in range(3):
            sums[j] += d[j] * f.shape[0]
    if train and accumulated_loss is not None:
        (accumulated_loss / max(1, n_dec)).backward()

    return {
        "loss": total_loss_val / max(1, n_dec),
        "dice_bg": sums[0] / max(1, n_dec),
        "dice_tls": sums[1] / max(1, n_dec),
        "dice_gc": sums[2] / max(1, n_dec),
        "mDice": sum(sums) / max(1, n_dec) / 3.0,
        "n_decoded": n_dec,
        "n_total": n,
    }


def make_warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Main ─────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="configs/e2e", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    sys.path.insert(0, "/home/ubuntu/profile-clam")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare_segmentation as ps
    from stage_features_to_local import LOCAL_ROOT, local_zarr_dirs

    if cfg.use_local_ssd != "never":
        local_dirs = local_zarr_dirs()
        if all(Path(p).is_dir() and any(Path(p).iterdir()) for p in local_dirs.values()):
            ps.ZARR_DIRS = local_dirs
            print(f"Using locally-staged zarrs at {LOCAL_ROOT}")

    ps.set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building dataset...")
    entries = ps.build_slide_entries()
    folds_pair, _test = ps.create_splits(entries, k_folds=1, seed=cfg.seed)
    val_entries, train_entries = folds_pair[0], folds_pair[1]
    # Drop entries without masks — e2e supervision needs them.
    train_entries = [e for e in train_entries if e.get("mask_path")]
    val_entries = [e for e in val_entries if e.get("mask_path")]
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val (mask-positive only)")

    print("Building mask cache...")
    mask_dict = ps.build_mask_cache(train_entries + val_entries, cfg.train.upsample_factor)

    train_ds = ps.TLSSegmentationDataset(
        train_entries, mask_dict, cfg.train.upsample_factor, patch_size=cfg.train.patch_size,
    )
    val_ds = ps.TLSSegmentationDataset(
        val_entries, mask_dict, cfg.train.upsample_factor, patch_size=cfg.train.patch_size,
    )

    (out_dir / "val_slides.json").write_text(
        json.dumps([e["slide_id"] for e in val_entries], indent=2))
    (out_dir / "train_slides.json").write_text(
        json.dumps([e["slide_id"] for e in train_entries], indent=2))

    model = GraphEnrichedDecoder(
        in_dim=cfg.model.in_dim,
        graph_hidden_dim=cfg.model.graph_hidden_dim,
        n_hops=cfg.model.n_hops,
        gat_heads=cfg.model.gat_heads,
        spatial_bottleneck=cfg.model.spatial_bottleneck,
        hidden_channels=cfg.model.hidden_channels,
        spatial_size=cfg.model.spatial_size,
        n_classes=cfg.model.n_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GraphEnrichedDecoder: {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = make_warmup_cosine(optimizer, cfg.train.warmup_epochs, cfg.train.epochs)
    cw = torch.tensor(list(cfg.train.class_weights), device=device, dtype=torch.float32)

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
        # Train.
        t0 = time.time()
        model.train()
        sums_tr = [0.0, 0.0, 0.0]
        loss_sum = 0.0
        n_sum = 0
        n_decoded_total = 0
        for idx in range(len(train_ds)):
            optimizer.zero_grad()
            r = run_one_slide(model, train_ds[idx], device, cw, cfg.train,
                              train=True, rng=rng)
            if r is None:
                continue
            optimizer.step()
            loss_sum += r["loss"] * r["n_decoded"]
            n_sum += r["n_decoded"]
            n_decoded_total += r["n_decoded"]
            for j, k in enumerate(["dice_bg", "dice_tls", "dice_gc"]):
                sums_tr[j] += r[k] * r["n_decoded"]
        train_t = time.time() - t0
        tr_loss = loss_sum / max(1, n_sum)
        tr_mDice = sum(sums_tr) / max(1, n_sum) / 3.0
        tr_dice_tls = sums_tr[1] / max(1, n_sum)
        tr_dice_gc = sums_tr[2] / max(1, n_sum)

        # Val.
        t0 = time.time()
        model.eval()
        sums_va = [0.0, 0.0, 0.0]
        loss_sum = 0.0
        n_sum = 0
        with torch.no_grad():
            for idx in range(len(val_ds)):
                r = run_one_slide(model, val_ds[idx], device, cw, cfg.train,
                                  train=False, rng=rng)
                if r is None:
                    continue
                loss_sum += r["loss"] * r["n_decoded"]
                n_sum += r["n_decoded"]
                for j, k in enumerate(["dice_bg", "dice_tls", "dice_gc"]):
                    sums_va[j] += r[k] * r["n_decoded"]
        val_t = time.time() - t0
        va_loss = loss_sum / max(1, n_sum)
        va_mDice = sum(sums_va) / max(1, n_sum) / 3.0
        va_dice_tls = sums_va[1] / max(1, n_sum)
        va_dice_gc = sums_va[2] / max(1, n_sum)
        last_va = {"loss": va_loss, "mDice": va_mDice,
                   "dice_tls": va_dice_tls, "dice_gc": va_dice_gc}

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        is_best = va_mDice > best_m
        marker = "BEST" if is_best else ""
        print(
            f"EPOCH epoch={epoch} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_mDice={va_mDice:.4f} val_tls={va_dice_tls:.4f} val_gc={va_dice_gc:.4f} "
            f"train_decoded/slide={n_decoded_total/max(1,len(train_ds)):.0f} "
            f"lr={lr:.2e} train={train_t:.0f}s val={val_t:.0f}s {marker}"
        )
        if run is not None:
            run.log({
                "epoch": epoch, "lr": lr,
                "train/loss": tr_loss, "train/mDice": tr_mDice,
                "train/dice_tls": tr_dice_tls, "train/dice_gc": tr_dice_gc,
                "val/loss": va_loss, "val/mDice": va_mDice,
                "val/dice_tls": va_dice_tls, "val/dice_gc": va_dice_gc,
                "train/decoded_per_slide": n_decoded_total / max(1, len(train_ds)),
                "best_mDice": max(best_m, va_mDice),
            }, step=epoch)

        # Last checkpoint (always).
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "val_metrics": last_va,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, out_dir / "last.pt")

        if is_best:
            best_m, best_epoch = va_mDice, epoch
            epochs_since_best = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_metrics": last_va,
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
