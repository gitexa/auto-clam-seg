"""GARS Stage 1: GraphTLSDetector — patch-level TLS region proposal.

Scaffolded on 2026-04-29 from the recovered wandb config + log of run
`x50tcqt6` (gars_stage1_gatv2_3hop_v2_20260426_230650). Architecture
shapes match the surviving `best_checkpoint.pt` exactly (verified by
loading state_dict with strict=True; see verify_stage1_ckpt.py).

Source code for the original training script was lost with the VM; this
is a re-implementation of the same model and training loop.

Inputs per slide (via prepare_segmentation.TLSSegmentationDataset):
  features:          (N, 1536)  UNI-v2 patch embeddings
  coords:            (N, 2)     patch top-left (x, y) at slide native px
  graph_edges_1hop:  (2, E)     pre-computed 1-hop spatial edges (from zarr)
  mask:              (1, H, W)  3-class HookNet mask (0=bg, 1=TLS, 2=GC),
                                upsampled to upsample_factor × patch grid

Patch label: 1 if the patch's cell in the mask grid contains any positive
(TLS or GC) pixels; else 0.

Run:
    python train_gars_stage1.py  # config from CLI / env / hardcoded below
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GATv2Conv, GCNConv


# ─── Model ────────────────────────────────────────────────────────────


class GraphTLSDetector(nn.Module):
    """Per-patch binary TLS classifier with optional GNN context.

    Architecture (verified against checkpoint x50tcqt6 / GATv2 3-hop):
        proj:        Linear(1536 -> 256) + LayerNorm(256)
        gnn_layers:  n_hops × {GATv2Conv(256, 64, heads=4, concat=True)
                               | GCNConv(256, 256)}
        gnn_norms:   n_hops × LayerNorm(256)
        head:        Linear(256, 128) -> GELU -> Dropout(p) -> Linear(128, 1)

    n_hops=0 disables the GNN entirely (only proj + head).
    """

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
            x = norm(layer(x, edge_index) + x)  # residual + post-norm
        return self.head(x).squeeze(-1)


# ─── Patch label generation ───────────────────────────────────────────


def patch_labels_from_mask(
    mask: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
    upsample_factor: int,
) -> torch.Tensor:
    """For each patch, label = 1 if any TLS (mask>0) pixel falls inside it.

    mask: (1, H, W) values in {0, 1, 2} (bg / TLS / GC). H, W are the
        upsampled grid (upsample_factor patches-per-dim per native patch).
    coords: (N, 2) patch top-left in native pixels.
    """
    grid_x = (coords[:, 0] / patch_size).long()
    grid_y = (coords[:, 1] / patch_size).long()
    grid_x -= grid_x.min()
    grid_y -= grid_y.min()
    m = (mask[0] > 0).float()  # (H, W) — TLS or GC pixels

    # Each patch covers an upsample_factor × upsample_factor cell in the mask
    u = upsample_factor
    H, W = m.shape
    labels = torch.zeros(coords.shape[0], dtype=torch.float32)
    for i, (gx, gy) in enumerate(zip(grid_x.tolist(), grid_y.tolist())):
        y0, y1 = gy * u, min((gy + 1) * u, H)
        x0, x1 = gx * u, min((gx + 1) * u, W)
        if y1 > y0 and x1 > x0:
            labels[i] = float(m[y0:y1, x0:x1].any())
    return labels


# ─── Training ─────────────────────────────────────────────────────────


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


def run_split(model, dataset, optimizer, criterion, device, train: bool, upsample_factor: int, patch_size: int):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx in range(len(dataset)):
            batch = dataset[idx]
            features = batch["features"].to(device)                  # (N, 1536)
            coords = batch["coords"]                                  # (N, 2)
            edge_index = batch["edge_index"]
            edge_index = edge_index.to(device) if edge_index is not None else None
            target = patch_labels_from_mask(
                batch["mask"], coords, patch_size, upsample_factor
            ).to(device)

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
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_selected": tp + fp,
        "n_total": tp + fp + fn + tn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n_hops", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--gnn_type", choices=["gatv2", "gcn"], default="gatv2")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=256)  # data is at 256
    ap.add_argument("--upsample_factor", type=int, default=4)
    ap.add_argument("--pos_weight", type=float, default=5.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--k_folds", type=int, default=1)
    ap.add_argument("--data_fraction", type=float, default=1.0)
    ap.add_argument("--label", default="gars_stage1_gatv2_3hop")
    ap.add_argument("--out_root", default="/home/ubuntu/ahaas-persistent-std-tcga/experiments")
    args = ap.parse_args()

    sys.path.insert(0, "/home/ubuntu/profile-clam")
    import prepare_segmentation as ps  # local import — needs zarr / tifffile / scipy

    ps.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─ Build dataset ─
    print("Building dataset...")
    entries = ps.build_slide_entries()
    if args.data_fraction < 1.0:
        entries = ps.subsample_entries(entries, args.data_fraction, args.seed)
    splits = ps.create_splits(entries, k_folds=args.k_folds, seed=args.seed)
    train_entries, val_entries = splits[0]["train"], splits[0]["val"]
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val")

    print("Building mask cache...")
    mask_dict = ps.build_mask_cache(train_entries + val_entries, args.upsample_factor)

    train_ds = ps.TLSSegmentationDataset(
        train_entries, mask_dict, args.upsample_factor, patch_size=args.patch_size
    )
    val_ds = ps.TLSSegmentationDataset(
        val_entries, mask_dict, args.upsample_factor, patch_size=args.patch_size
    )

    # ─ Model + opt ─
    model = GraphTLSDetector(
        in_dim=1536,
        hidden_dim=args.hidden_dim,
        n_hops=args.n_hops,
        gnn_type=args.gnn_type,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 ({args.gnn_type} {args.n_hops}-hop): {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))

    # ─ Output dir ─
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"gars_stage1_{args.label}_{ts}"
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Run: {run_id}\nResults: {out_dir}")
    print(f"Training {len(train_entries)} slides, validating {len(val_entries)} slides\n")

    # ─ Loop ─
    best_f1 = -1.0
    best_epoch = -1
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_split(model, train_ds, optimizer, criterion, device, train=True,
                       upsample_factor=args.upsample_factor, patch_size=args.patch_size)
        train_t = time.time() - t0
        t0 = time.time()
        va = run_split(model, val_ds, optimizer, criterion, device, train=False,
                       upsample_factor=args.upsample_factor, patch_size=args.patch_size)
        val_t = time.time() - t0
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

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
        if is_best:
            best_f1, best_epoch = va["f1"], epoch
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
                      f"(best={best_epoch}, f1={best_f1:.3f})")
                break

    (out_dir / "final_results.json").write_text(json.dumps(va, indent=2))
    print(f"\nDone. Best f1={best_f1:.3f} at epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
