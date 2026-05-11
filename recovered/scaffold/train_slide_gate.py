"""A3 — train a small slide-level "is this slide TLS-positive" gate.

Reads UNI v2 features per slide from the local zarr root, mean-pools
them, trains a 2-layer MLP against the binary `tls_num > 0` label
from `df_summary_v10.csv`. Validates on fold 0; if recall ≥ 0.95 and
precision ≥ 0.85 at the chosen threshold, the gate is good enough to
retrofit on top of GNCAF predictions.

Usage:
    python train_slide_gate.py
    python train_slide_gate.py --eval_only --ckpt slide_gate.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_segmentation as ps
import zarr
from stage_features_to_local import local_zarr_dirs


class SlideGateModel(nn.Module):
    """Mean-pool UNI features, then 2-layer MLP → BCE logit."""

    def __init__(self, in_dim: int = 1536, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        # pooled_features: (B, in_dim)
        return self.head(pooled_features).squeeze(-1)


class SlidePoolDataset(Dataset):
    """Reads each slide's UNI features, mean-pools, returns (vec, label)."""

    def __init__(self, entries: list[dict], slide_label: dict[str, int]):
        # Pre-load mean-pooled features into memory (cheap: 1015 * 1536 * 4 = 6 MB).
        if all(Path(p).is_dir() for p in local_zarr_dirs().values()):
            ps.ZARR_DIRS = local_zarr_dirs()
        self.feats = []
        self.labels = []
        self.slide_ids = []
        for e in entries:
            short = e["slide_id"].split(".")[0]
            try:
                grp = zarr.open(e["zarr_path"], mode="r")
                f = np.asarray(grp["features"][:]).astype(np.float32)
                pooled = f.mean(axis=0)
            except Exception:
                continue
            self.feats.append(torch.from_numpy(pooled))
            self.labels.append(float(slide_label.get(short, 0)))
            self.slide_ids.append(short)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return {
            "feat": self.feats[i],
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
            "slide_id": self.slide_ids[i],
        }


def slide_label_lookup() -> dict[str, int]:
    df = pd.read_csv(ps.META_CSV)
    df["short"] = df["slide_id"].astype(str).str.split(".").str[0]
    return df.set_index("short")["tls_num"].apply(lambda v: 1 if v > 0 else 0).to_dict()


def collate(batch):
    return {
        "feat": torch.stack([b["feat"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "slide_id": [b["slide_id"] for b in batch],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold_idx", type=int, default=0)
    ap.add_argument("--out", default="/home/ubuntu/ahaas-persistent-std-tcga/experiments/slide_gate")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ps.set_seed(args.seed)
    entries = ps.build_slide_entries()
    all_folds, _test = ps.create_splits(entries, k_folds=5, seed=args.seed)
    val_entries = all_folds[args.fold_idx]
    train_entries = [s for i, f in enumerate(all_folds) if i != args.fold_idx for s in f]
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val (fold {args.fold_idx})")

    slide_lab = slide_label_lookup()
    n_pos = sum(1 for e in train_entries if slide_lab.get(e["slide_id"].split(".")[0], 0) == 1)
    n_neg = len(train_entries) - n_pos
    print(f"Train labels: {n_pos} TLS-positive / {n_neg} TLS-negative slides "
          f"({100*n_pos/len(train_entries):.0f}% positive)")

    print("Building train slide-pool dataset...")
    train_ds = SlidePoolDataset(train_entries, slide_lab)
    print(f"  loaded {len(train_ds)} slides")
    print("Building val slide-pool dataset...")
    val_ds = SlidePoolDataset(val_entries, slide_lab)
    print(f"  loaded {len(val_ds)} slides")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlideGateModel(in_dim=1536, hidden=args.hidden).to(device)
    print(f"SlideGateModel params: {sum(p.numel() for p in model.parameters()):,}")

    if args.eval_only:
        if not args.ckpt:
            args.ckpt = str(out_dir / "best.pt")
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
        model.eval()
        eval_and_save(model, val_ds, out_dir / "fold0_predictions.json", device)
        return

    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    print(f"BCE pos_weight = {pos_weight.item():.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0; n = 0
        for batch in train_loader:
            x = batch["feat"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            train_loss += float(loss) * x.shape[0]; n += x.shape[0]

        model.eval()
        all_logits = []; all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["feat"].to(device))
                all_logits.append(logits.cpu()); all_labels.append(batch["label"])
        logits = torch.cat(all_logits).numpy(); labels = torch.cat(all_labels).numpy()
        # Sweep thresholds; record best F1
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
        thrs = np.linspace(0.0, 1.0, 51)
        probs = 1.0 / (1.0 + np.exp(-logits))
        best = (None, -1.0)
        for tau in thrs:
            pred = (probs >= tau).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
            if f1 > best[1]:
                best = (tau, f1)
        tau_b, f1_b = best
        pred_b = (probs >= tau_b).astype(int)
        p_b, r_b, _, _ = precision_recall_fscore_support(labels, pred_b, average="binary", zero_division=0)
        try:
            auroc = roc_auc_score(labels, probs); ap = average_precision_score(labels, probs)
        except Exception:
            auroc = ap = float("nan")
        msg = (f"EPOCH {ep:2d}  train_loss={train_loss/max(1,n):.4f}  "
               f"val: AUROC={auroc:.3f}  AP={ap:.3f}  "
               f"best_F1={f1_b:.3f} (P={p_b:.3f} R={r_b:.3f}) @τ={tau_b:.2f}")
        is_best = f1_b > best_f1
        if is_best:
            best_f1 = f1_b
            torch.save({"model": model.state_dict(), "tau": float(tau_b), "f1": f1_b,
                        "p": float(p_b), "r": float(r_b), "auroc": float(auroc), "ap": float(ap)},
                       out_dir / "best.pt")
            msg += "  BEST"
        print(msg, flush=True)

    eval_and_save(model, val_ds, out_dir / "fold0_predictions.json", device)


def eval_and_save(model, val_ds, out_path, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            it = val_ds[i]
            logit = float(model(it["feat"].unsqueeze(0).to(device))[0])
            prob = 1.0 / (1.0 + np.exp(-logit))
            rows.append({"slide_id": it["slide_id"], "label": int(it["label"]),
                         "logit": logit, "prob": float(prob)})
    Path(out_path).write_text(json.dumps(rows, indent=2))
    print(f"Saved {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
