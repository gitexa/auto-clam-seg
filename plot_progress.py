#!/usr/bin/env python3
"""Plot experiment progress for autoresearch-clam-seg.

Generates two plots:
1. progress.png — experiment-level results (val_dice, det_auc, count_sp, bkt_bacc)
2. training.png — epoch-level training curves for the current/last run
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Benchmarks (targets to beat) ────────────────────────────────────────
BENCHMARKS = {
    "det_auc": 0.834,    # binary classification AUC
    "inst_sp": 0.774,    # count regression Spearman
    "bkt_bacc": 0.632,   # count regression bucket balanced acc
}
DICE_TARGET = 0.6  # aspirational target


def plot_experiments():
    """Plot experiment-level results from results.tsv."""
    tsv_path = BASE_DIR / "clam-worktree" / "results.tsv"
    if not tsv_path.exists():
        print("No results.tsv found")
        return

    df = pd.read_csv(tsv_path, sep="\t")
    if len(df) == 0:
        print("No experiments in results.tsv yet")
        return

    for col in ["val_dice", "det_auc", "inst_sp", "bkt_bacc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].str.strip().str.lower()

    metrics = [
        ("val_dice", "Val Dice", "#e74c3c", DICE_TARGET),
        ("det_auc", "Detection AUC", "#3498db", BENCHMARKS.get("det_auc")),
        ("inst_sp", "Count Spearman", "#2ecc71", BENCHMARKS.get("inst_sp")),
        ("bkt_bacc", "Bucket Bal.Acc", "#9b59b6", BENCHMARKS.get("bkt_bacc")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flat

    for ax, (col, label, color, benchmark) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        vals = df[col].values
        statuses = df["status"].values
        x = np.arange(len(df))

        # Plot all points
        for i, (xi, vi, st) in enumerate(zip(x, vals, statuses)):
            if pd.isna(vi):
                continue
            if st == "keep":
                ax.scatter(xi, vi, c=color, s=80, zorder=4,
                           edgecolors="black", linewidths=0.8)
            elif st == "discard":
                ax.scatter(xi, vi, c="#cccccc", s=30, zorder=2, alpha=0.6)
            elif st == "crash":
                ax.scatter(xi, vi, c="#e74c3c", s=30, zorder=2, marker="x")

        # Running best (kept only)
        kept_mask = df["status"] == "keep"
        if kept_mask.any():
            kept_vals = df.loc[kept_mask, col]
            running_best = kept_vals.cummax()
            ax.step(df.index[kept_mask], running_best, where="post",
                    color=color, linewidth=2, alpha=0.7, zorder=3)

        # Benchmark line
        if benchmark is not None:
            ax.axhline(y=benchmark, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(len(df) - 0.5, benchmark, f"benchmark={benchmark:.3f}",
                    ha="right", va="bottom", fontsize=8, color="gray")

        ax.set_xlabel("Experiment #", fontsize=10)
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # Annotate kept experiments
        for i, row in df[kept_mask].iterrows():
            desc = str(row.get("description", ""))
            if len(desc) > 35:
                desc = desc[:32] + "..."
            ax.annotate(desc, (i, row[col]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=7, color=color, alpha=0.8, rotation=20)

    n_total = len(df)
    n_kept = (df["status"] == "keep").sum()
    fig.suptitle(f"Autoresearch-CLAM-Seg: {n_total} Experiments, {n_kept} Kept",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = BASE_DIR / "progress.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def parse_epoch_log(log_path):
    """Parse EPOCH lines from run_stdout.log."""
    epochs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("EPOCH "):
                parts = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
                row = {}
                for k, v in parts.items():
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
                epochs.append(row)
    return pd.DataFrame(epochs)


def plot_training():
    """Plot epoch-level training curves from run_stdout.log."""
    log_path = BASE_DIR / "run_stdout.log"
    if not log_path.exists():
        print("No run_stdout.log found")
        return

    df = parse_epoch_log(log_path)
    if len(df) == 0:
        print("No EPOCH lines in log")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. Val Dice
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["val_dice"], color="#e74c3c", linewidth=1.5, label="val_dice")
    if "neg_dice" in df.columns:
        ax.plot(df["epoch"], df["neg_dice"], color="#e74c3c", linewidth=1, alpha=0.4,
                linestyle="--", label="neg_dice")
    ax.axhline(y=DICE_TARGET, color="gray", linestyle="--", alpha=0.5)
    best_idx = df["val_dice"].idxmax()
    ax.scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, "val_dice"],
               c="#e74c3c", s=80, zorder=5, edgecolors="black", marker="*")
    ax.set_ylabel("Dice", fontsize=11, fontweight="bold")
    ax.set_title(f"Best dice={df['val_dice'].max():.4f} (ep{int(df.loc[best_idx, 'epoch'])})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 2. Detection AUC
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["det_auc"], color="#3498db", linewidth=1.5)
    ax.axhline(y=BENCHMARKS["det_auc"], color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Detection AUC", fontsize=11, fontweight="bold")
    ax.set_title(f"Best det_auc={df['det_auc'].max():.4f}")
    ax.grid(True, alpha=0.2)

    # 3. Count Spearman
    ax = axes[0, 2]
    ax.plot(df["epoch"], df["count_sp"], color="#2ecc71", linewidth=1.5)
    ax.axhline(y=BENCHMARKS["inst_sp"], color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Count Spearman", fontsize=11, fontweight="bold")
    ax.set_title(f"Best count_sp={df['count_sp'].max():.4f}")
    ax.grid(True, alpha=0.2)

    # 4. Losses
    ax = axes[1, 0]
    for loss_key, label, color in [
        ("l_dice", "dice", "#e74c3c"),
        ("l_focal", "focal", "#f39c12"),
        ("l_center", "center", "#3498db"),
        ("l_offset", "offset", "#2ecc71"),
    ]:
        if loss_key in df.columns:
            ax.plot(df["epoch"], df[loss_key], label=label, linewidth=1.2, color=color)
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Training Losses (log scale)")
    ax.grid(True, alpha=0.2)

    # 5. Bucket balanced acc + cls0 recall
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["bkt_bacc"], color="#9b59b6", linewidth=1.5, label="bkt_bacc")
    if "cls0_rec" in df.columns:
        ax.plot(df["epoch"], df["cls0_rec"], color="#e67e22", linewidth=1.2,
                linestyle="--", label="cls0_recall (FP control)")
    ax.axhline(y=BENCHMARKS["bkt_bacc"], color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Balanced Acc", fontsize=11, fontweight="bold")
    ax.set_title(f"Best bkt_bacc={df['bkt_bacc'].max():.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 6. Learning rate
    ax = axes[1, 2]
    ax.plot(df["epoch"], df["lr"], color="#34495e", linewidth=1.5)
    ax.set_ylabel("Learning Rate", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_title("LR Schedule")
    ax.grid(True, alpha=0.2)

    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=10)

    # Extract run name from log
    run_name = ""
    with open(log_path) as f:
        for line in f:
            if line.startswith("Run: "):
                run_name = line.strip().split("Run: ", 1)[1]
                break

    fig.suptitle(f"Training Curves: {run_name} ({len(df)} epochs)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = BASE_DIR / "training.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_experiments()
    plot_training()
