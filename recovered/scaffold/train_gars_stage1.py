"""GARS Stage 1: GraphTLSDetector — patch-level TLS region proposal.

v3.0 — hydra config + wandb logging + config dump to results dir.

Architecture (verified strict-load against recovered checkpoints):
    proj:        Linear(1536 -> hidden_dim) + LayerNorm
    gnn_layers:  n_hops × {GATv2Conv | GCNConv}        (residual + post-norm)
    gnn_norms:   n_hops × LayerNorm
    head:        Linear(hidden_dim, 128) → GELU → Dropout → Linear(128, 1)

Inputs per slide (via prepare_segmentation.TLSSegmentationDataset):
    features:          (N, 1536)  UNI-v2 embeddings
    coords:            (N, 2)     patch top-left (x, y), slide native px
    graph_edges_1hop:  (2, E)     pre-computed 1-hop spatial edges
    mask:              (1, H, W)  3-class HookNet mask

Run:
    python train_gars_stage1.py                          # default config
    python train_gars_stage1.py model=gcn_3hop           # ablation
    python train_gars_stage1.py train.lr=5e-4 epochs=20  # tweak
    WANDB_MODE=disabled python train_gars_stage1.py epochs=1 label=smoke
"""
from __future__ import annotations

import json
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GATv2Conv, GCNConv


# ─── Model ────────────────────────────────────────────────────────────


class GraphTLSDetector(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 256,
        n_hops: int = 3,
        gnn_type: str = "gatv2",
        dropout: float = 0.1,
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_hops = n_hops
        self.gnn_type = gnn_type
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(n_hops):
            if gnn_type == "gatv2":
                assert hidden_dim % gat_heads == 0, "hidden_dim must be divisible by gat_heads"
                self.gnn_layers.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // gat_heads,
                        heads=gat_heads,
                        concat=True,
                        dropout=0.0,
                    )
                )
            elif gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"unknown gnn_type {gnn_type!r}")
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        x = self.proj(features)
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            assert edge_index is not None, "GNN layers need an edge_index"
            x = norm(layer(x, edge_index) + x)
        return self.head(x).squeeze(-1)


# ─── Patch labelling ──────────────────────────────────────────────────


def patch_labels_from_mask(
    mask: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    upsample_factor: int,
) -> torch.Tensor:
    grid_x = (coords[:, 0] / patch_size).long()
    grid_y = (coords[:, 1] / patch_size).long()
    grid_x -= grid_x.min()
    grid_y -= grid_y.min()
    m = (mask[0] > 0).float()
    u = upsample_factor
    H, W = m.shape
    labels = torch.zeros(coords.shape[0], dtype=torch.float32)
    for i, (gx, gy) in enumerate(zip(grid_x.tolist(), grid_y.tolist())):
        y0, y1 = gy * u, min((gy + 1) * u, H)
        x0, x1 = gx * u, min((gx + 1) * u, W)
        if y1 > y0 and x1 > x0:
            labels[i] = float(m[y0:y1, x0:x1].any())
    return labels


# ─── Metrics + loop ───────────────────────────────────────────────────


def confusion_at_threshold(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    pred = (torch.sigmoid(logits) > thr).int()
    tgt = target.int()
    tp = int(((pred == 1) & (tgt == 1)).sum())
    fp = int(((pred == 1) & (tgt == 0)).sum())
    fn = int(((pred == 0) & (tgt == 1)).sum())
    tn = int(((pred == 0) & (tgt == 0)).sum())
    return tp, fp, fn, tn


def f1_from_confusion(tp: int, fp: int, fn: int, tn: int):
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return rec, prec, f1


def _identity_collate(batch_list):
    """One slide per 'batch' — just unwrap the singleton list."""
    return batch_list[0]


def make_slide_loader(dataset, num_workers: int, prefetch_factor: int,
                      persistent_workers: bool, shuffle: bool):
    """Per-slide DataLoader with async prefetching.

    batch_size=1 because graph sizes vary per slide. With num_workers>0,
    slide N+1 loads from local SSD while GPU trains on slide N.
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=False,  # tensors are too varied; manual .to(device) below
        collate_fn=_identity_collate,
    )


def run_split(model, loader, optimizer, criterion, device, train: bool,
              upsample_factor: int, patch_size: int):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            coords = batch["coords"]
            edge_index = batch["edge_index"]
            edge_index = edge_index.to(device, non_blocking=True) if edge_index is not None else None
            target = patch_labels_from_mask(
                batch["mask"], coords, patch_size, upsample_factor
            ).to(device, non_blocking=True)
            logits = model(features, edge_index)
            loss = criterion(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
            a, b, c, d = confusion_at_threshold(logits.detach(), target)
            tp += a; fp += b; fn += c; tn += d
    rec, prec, f1 = f1_from_confusion(tp, fp, fn, tn)
    return {
        "loss": total_loss / max(1, n_batches),
        "recall": rec, "precision": prec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_selected": tp + fp,
        "n_total": tp + fp + fn + tn,
    }


@hydra.main(version_base=None, config_path="configs/stage1", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the resolved config at the top level of the run dir (hydra
    # also writes .hydra/config.yaml automatically; this is the convenient
    # discovery copy).
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    # Late imports (need profile-clam venv).
    sys.path.insert(0, "/home/ubuntu/profile-clam")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare_segmentation as ps
    from stage_features_to_local import LOCAL_ROOT, local_zarr_dirs

    # Local SSD redirect.
    if cfg.use_local_ssd != "never":
        local_dirs = local_zarr_dirs()
        all_present = all(Path(p).is_dir() and any(Path(p).iterdir())
                          for p in local_dirs.values())
        if all_present:
            ps.ZARR_DIRS = local_dirs
            print(f"Using locally-staged zarrs at {LOCAL_ROOT}")
        elif cfg.use_local_ssd == "always":
            raise RuntimeError(f"--use_local_ssd=always but local zarrs missing at {LOCAL_ROOT}")
        else:
            print(f"NFS zarrs in use (no local copy at {LOCAL_ROOT})")

    ps.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─ Data ─
    print("Building dataset...")
    entries = ps.build_slide_entries()
    if cfg.train.data_fraction < 1.0:
        entries = ps.subsample_entries(entries, cfg.train.data_fraction, cfg.seed)
    folds_pair, _test = ps.create_splits(entries, k_folds=cfg.train.k_folds, seed=cfg.seed)
    val_entries, train_entries = folds_pair[0], folds_pair[1]
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val")

    print("Building mask cache...")
    mask_dict = ps.build_mask_cache(train_entries + val_entries, cfg.train.upsample_factor)

    train_ds = ps.TLSSegmentationDataset(
        train_entries, mask_dict, cfg.train.upsample_factor, patch_size=cfg.train.patch_size,
    )
    val_ds = ps.TLSSegmentationDataset(
        val_entries, mask_dict, cfg.train.upsample_factor, patch_size=cfg.train.patch_size,
    )

    # Save val/train slide IDs so eval can reproduce the split exactly,
    # without depending on numpy.random.RandomState behavior across versions.
    (out_dir / "val_slides.json").write_text(json.dumps(
        [e["slide_id"] for e in val_entries], indent=2,
    ))
    (out_dir / "train_slides.json").write_text(json.dumps(
        [e["slide_id"] for e in train_entries], indent=2,
    ))

    nw = cfg.train.get("num_workers", 0)
    pf = cfg.train.get("prefetch_factor", 2)
    pw = cfg.train.get("persistent_workers", True)
    train_loader = make_slide_loader(train_ds, nw, pf, pw, shuffle=True)
    val_loader = make_slide_loader(val_ds, nw, pf, pw, shuffle=False)
    print(f"DataLoader: {nw} workers, prefetch={pf}, persistent={pw}")

    # ─ Model + opt ─
    model = GraphTLSDetector(
        in_dim=cfg.model.in_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_hops=cfg.model.n_hops,
        gnn_type=cfg.model.gnn_type,
        dropout=cfg.model.dropout,
        gat_heads=cfg.model.gat_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 ({cfg.model.gnn_type} {cfg.model.n_hops}-hop): {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=0.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.train.pos_weight, device=device))

    # ─ Wandb ─
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != "disabled":
        import wandb
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=out_dir.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir),
            mode=cfg.wandb.mode,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        )

    print(f"Training {len(train_entries)} slides, validating {len(val_entries)} slides\n")

    best_f1 = -1.0
    best_epoch = -1
    epochs_since_best = 0
    last_va = None
    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr = run_split(model, train_loader, optimizer, criterion, device, train=True,
                       upsample_factor=cfg.train.upsample_factor, patch_size=cfg.train.patch_size)
        train_t = time.time() - t0
        t0 = time.time()
        va = run_split(model, val_loader, optimizer, criterion, device, train=False,
                       upsample_factor=cfg.train.upsample_factor, patch_size=cfg.train.patch_size)
        val_t = time.time() - t0
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        last_va = va

        is_best = va["f1"] > best_f1
        marker = "BEST" if is_best else ""
        print(
            f"EPOCH epoch={epoch} train_loss={tr['loss']:.4f} "
            f"train_recall={tr['recall']:.3f} train_prec={tr['precision']:.3f} "
            f"train_f1={tr['f1']:.3f} "
            f"val_loss={va['loss']:.4f} val_recall={va['recall']:.3f} "
            f"val_prec={va['precision']:.3f} val_f1={va['f1']:.3f} "
            f"val_selected={va['n_selected']}/{va['n_total']} "
            f"lr={lr:.2e} train={train_t:.0f}s val={val_t:.0f}s {marker}"
        )
        if run is not None:
            # Match the recovered run's key set (x50tcqt6) so comparisons
            # are 1:1 in wandb.
            run.log({
                "epoch": epoch, "lr": lr,
                "train/loss": tr["loss"], "train/f1": tr["f1"],
                "train/recall": tr["recall"], "train/precision": tr["precision"],
                "val/loss": va["loss"], "val/f1": va["f1"],
                "val/recall": va["recall"], "val/precision": va["precision"],
                "val/tp": va["tp"], "val/fp": va["fp"],
                "val/fn": va["fn"], "val/tn": va["tn"],
                "val/n_selected": va["n_selected"], "val/n_total": va["n_total"],
                "best_f1": max(best_f1, va["f1"]),
            }, step=epoch)

        # Always save `last.pt` (for resume / inference at any epoch).
        last_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_metrics": va,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(last_payload, out_dir / "last.pt")

        if is_best:
            best_f1, best_epoch = va["f1"], epoch
            epochs_since_best = 0
            # Slim checkpoint (no optimizer state) — for inference.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": va,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                out_dir / "best_checkpoint.pt",
            )
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.train.patience:
                print(f"  Early stopping at epoch {epoch} (best={best_epoch}, f1={best_f1:.3f})")
                break

    if last_va is not None:
        (out_dir / "final_results.json").write_text(json.dumps(last_va, indent=2))
    print(f"\nDone. Best f1={best_f1:.3f} at epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best_checkpoint.pt'}")
    if run is not None:
        run.summary["best_f1"] = best_f1
        run.summary["best_epoch"] = best_epoch
        if last_va is not None:
            run.summary["final_recall"] = last_va["recall"]
            run.summary["final_precision"] = last_va["precision"]
            run.summary["final_f1"] = last_va["f1"]
        run.finish()


if __name__ == "__main__":
    main()
