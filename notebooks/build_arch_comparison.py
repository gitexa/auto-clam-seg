"""Build the architecture-comparison plots + summary table.

Inputs:
  * /home/ubuntu/auto-clam-seg/benchmark_5fold.csv   — fold-level metrics for all
    4 main approaches (Cascade, GNCAF v3.56, GNCAF v3.58, seg_v2.0).
  * gars_gncaf_eval_<v>_<fold>_eval_shard*/gncaf_agg.json  — per-slide rows for
    GNCAF v3.56, v3.58 (all 5 folds) and v3.59 fold-0 (single fold for now).

Outputs (under /home/ubuntu/auto-clam-seg/notebooks/architectures/):
  * fig_5fold_bars.png        5-fold mean±std bars (mDice_pix, TLS, GC) per approach
  * fig_perslide_dice.png     per-slide Dice violin/box for GNCAF (v3.56/v3.58/v3.59)
  * fig_count_scatter.png     gt vs pred counts scatter with Spearman/Pearson/R²
  * fig_buckets.png           bucketed precision/recall/F1 (TLS, GC) by gt count bin
  * fig_detection_cm.png      slide-level detection confusion matrices
  * fig_cancer_breakdown.png  per-cancer-type Dice (BLCA/KIRC/LUSC)
  * arch_summary.json         numeric companion to the markdown

Usage:
    python build_arch_comparison.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    confusion_matrix, mean_absolute_error, precision_score, recall_score, f1_score, r2_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)

REPO = Path("/home/ubuntu/auto-clam-seg")
EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")
OUT = REPO / "notebooks" / "architectures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})

APPROACH_COLORS = {
    "Cascade":         "#1f77b4",
    "Cascade v3.37":   "#1f77b4",
    "GNCAF v3.21":     "#ff7f0e",
    "GNCAF v3.58":     "#d62728",
    "GNCAF v3.59":     "#2ca02c",
    "GNCAF v3.56":     "#9467bd",
    "GNCAF v3.61":     "#bcbd22",
    "GNCAF v3.62":     "#17becf",
    "GNCAF v3.63":     "#8b0000",
    "GNCAF v3.64":     "#cc6677",
    "GNCAF v3.65":     "#aa3322",
    "Lost-0.7143":     "#e377c2",
    "seg_v2.0":        "#7f7f7f",
    "seg_v2.0 (dual)": "#525252",
}


# ───────────────────────── 5-fold benchmark ─────────────────────────

def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(REPO / "benchmark_5fold.csv")
    return df


def fig_5fold_bars(df: pd.DataFrame):
    metrics = [("mDice_pix", "mean Dice (pixel-agg)"),
               ("tls_dice_pix", "TLS Dice (pixel-agg)"),
               ("gc_dice_pix", "GC Dice (pixel-agg)")]
    approaches = ["Cascade", "GNCAF v3.58", "GNCAF v3.56", "seg_v2.0"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        means, stds = [], []
        for a in approaches:
            sub = df[df["approach"] == a][m].dropna()
            means.append(sub.mean())
            stds.append(sub.std(ddof=1))
        x = np.arange(len(approaches))
        colors = [APPROACH_COLORS.get(a, "#888") for a in approaches]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black")
        for b, mv in zip(bars, means):
            if not np.isnan(mv):
                ax.text(b.get_x() + b.get_width() / 2.0, mv + 0.01, f"{mv:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, rotation=15)
        ax.set_title(label)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.axhline(0.0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Dice (mean ± std across 5 folds)")
    fig.suptitle("5-fold cross-validation — pixel-Dice per architecture", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5fold_bars.png", bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── per-slide GNCAF rows ─────────────────────

def collect_perslide(label_prefix: str, folds: list[int]) -> pd.DataFrame:
    """Pool all per-slide rows across the requested fold dirs.

    Each fold contributes 4 shards. Slides are unique per fold (the
    cohort is patient-stratified) so deduping isn't needed within a
    single approach × fold set.

    Prefers fullcohort eval dirs (which include GT-negative slides)
    over the older 5fold/fold0 dirs (positive-only); if both exist for
    a given fold, the fullcohort one wins.
    """
    rows = []
    for fold in folds:
        # Order matters: fullcohort first so positive-only is only used as a fallback.
        patterns = [
            f"gars_gncaf_eval_{label_prefix}_fullcohort_fold{fold}_eval_shard*",
            f"gars_gncaf_eval_{label_prefix}_fullcohort_fold{fold}_shard*",
            f"gars_gncaf_eval_{label_prefix}_5fold_fold{fold}_eval_shard*",
            f"gars_gncaf_eval_{label_prefix}_fold{fold}_eval_shard*",
        ]
        dirs = None
        for pat in patterns:
            ds = sorted(EXP.glob(pat))
            if ds:
                dirs = ds
                break
        if not dirs:
            continue
        for d in dirs:
            agg_p = d / "gncaf_agg.json"
            if not agg_p.exists():
                continue
            o = json.loads(agg_p.read_text())
            for r in o["per_slide"]:
                r = dict(r)
                r["fold"] = fold
                r["approach"] = label_prefix
                # Default for legacy rows that pre-date the gt_negative field.
                r.setdefault("gt_negative", False)
                rows.append(r)
    return pd.DataFrame(rows)


def collect_perslide_cascade(label_prefixes: list[tuple[str, int]],
                              threshold: str = "0.5") -> pd.DataFrame:
    """Pool per-slide rows for cascade eval at a single threshold.

    `label_prefixes` is a list of `(prefix, fold)` tuples to glob —
    `gars_cascade_<prefix>_fold<F>_eval_shard*`. Cascade rows use a
    different schema (`tls_dice` / `gc_dice` without `_grid` suffix);
    we rename so all downstream plots can pool on common keys.
    """
    rows = []
    for prefix, fold in label_prefixes:
        pattern = f"gars_cascade_{prefix}_fold{fold}_eval_shard*"
        dirs = sorted(EXP.glob(pattern))
        for d in dirs:
            ps_path = d / "cascade_per_slide.json"
            if not ps_path.exists():
                continue
            obj = json.loads(ps_path.read_text())
            for r in obj.get(threshold, []):
                row = dict(r)
                row["tls_dice_grid"] = row.get("tls_dice", np.nan)
                row["gc_dice_grid"] = row.get("gc_dice", np.nan)
                row["fold"] = fold
                row["approach"] = "Cascade v3.37"
                row.setdefault("gt_negative", False)
                rows.append(row)
    return pd.DataFrame(rows)


def fig_perslide_dice(perslide: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    parts_data_tls = []
    parts_data_gc = []
    labels = []
    for k, df in perslide.items():
        parts_data_tls.append(df["tls_dice_grid"].dropna().values)
        parts_data_gc.append(df["gc_dice_grid"].dropna().values)
        labels.append(k)

    bp = axes[0].boxplot(parts_data_tls, labels=labels, widths=0.6,
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker="D", markerfacecolor="white",
                                        markeredgecolor="k", markersize=5))
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(APPROACH_COLORS.get(lbl, "#888"))
        patch.set_alpha(0.6)
    axes[0].set_title("Per-slide TLS Dice (patch-grid)")
    axes[0].set_ylabel("TLS Dice")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    bp = axes[1].boxplot(parts_data_gc, labels=labels, widths=0.6,
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker="D", markerfacecolor="white",
                                        markeredgecolor="k", markersize=5))
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(APPROACH_COLORS.get(lbl, "#888"))
        patch.set_alpha(0.6)
    axes[1].set_title("Per-slide GC Dice (patch-grid)")
    axes[1].set_ylabel("GC Dice")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Distribution of per-slide patch-grid Dice", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_perslide_dice.png", bbox_inches="tight")
    plt.close(fig)


def _corr_block(gt: np.ndarray, pred: np.ndarray) -> dict:
    if len(gt) < 3 or np.all(gt == 0):
        return {"sp": np.nan, "pe": np.nan, "r2": np.nan, "mae": np.nan}
    sp = spearmanr(gt, pred).correlation
    pe = pearsonr(gt, pred).statistic
    # R² of pred vs gt (not symmetric — measures how much pred explains gt).
    r2 = r2_score(gt, pred) if np.var(gt) > 0 else np.nan
    mae = mean_absolute_error(gt, pred)
    return {"sp": sp, "pe": pe, "r2": r2, "mae": mae}


def fig_count_scatter(perslide: dict[str, pd.DataFrame]):
    n = len(perslide)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    summary = {}
    for j, (k, df) in enumerate(perslide.items()):
        for i, (entity, gt_col, pred_col) in enumerate([
            ("TLS", "gt_n_tls", "n_tls_pred"),
            ("GC",  "gt_n_gc",  "n_gc_pred"),
        ]):
            ax = axes[i, j]
            gt = df[gt_col].astype(float).values
            pred = df[pred_col].astype(float).values
            stats = _corr_block(gt, pred)
            ax.scatter(gt, pred, s=10, alpha=0.5,
                       color=APPROACH_COLORS.get(k, "#888"))
            mx = max(gt.max(), pred.max(), 1.0)
            ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5)
            ax.set_title(f"{k}: {entity} count\n"
                         f"sp={stats['sp']:.2f}  pe={stats['pe']:.2f}  "
                         f"R²={stats['r2']:.2f}  MAE={stats['mae']:.1f}")
            ax.set_xlabel(f"GT n_{entity}")
            ax.set_ylabel(f"pred n_{entity}")
            ax.grid(linestyle=":", alpha=0.5)
            summary.setdefault(k, {})[entity] = stats
    fig.suptitle("Per-slide instance counts: predicted vs. gold-standard metadata", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_count_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return summary


# ───────────────────────── bucketed evaluation ──────────────────────

BUCKETS_TLS = [(0, 0, "0"), (1, 3, "1-3"), (4, 10, "4-10"), (11, 1_000_000, "11+")]
BUCKETS_TLS_3CLASS = [(0, 0, "0"), (1, 5, "1-5"), (6, 1_000_000, "6+")]
BUCKETS_GC = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 1_000_000, "4+")]
BUCKETS_GC_2CLASS = [(0, 0, "0"), (1, 1_000_000, "≥1")]


def dice_to_iou(d: float) -> float:
    """Per-slide Dice → IoU (Jaccard). IoU = D / (2 - D)."""
    if d is None or np.isnan(d):
        return float("nan")
    return float(d) / (2.0 - float(d))


def assign_bucket(n: int, buckets) -> str:
    for lo, hi, lbl in buckets:
        if lo <= n <= hi:
            return lbl
    return buckets[-1][2]


def bucket_metrics(df: pd.DataFrame, gt_col: str, pred_col: str, buckets) -> pd.DataFrame:
    rows = []
    bucket_labels = [b[2] for b in buckets]
    df = df.copy()
    df["gt_bin"] = df[gt_col].map(lambda n: assign_bucket(int(n), buckets))
    df["pred_bin"] = df[pred_col].map(lambda n: assign_bucket(int(n), buckets))
    for lbl in bucket_labels:
        sub = df[df["gt_bin"] == lbl]
        if len(sub) == 0:
            rows.append({"bin": lbl, "n_slides": 0,
                         "precision": np.nan, "recall": np.nan, "f1": np.nan,
                         "n_tp": 0, "n_pred_in_bin": int((df["pred_bin"] == lbl).sum())})
            continue
        tp = int(((df["gt_bin"] == lbl) & (df["pred_bin"] == lbl)).sum())
        pred_in = int((df["pred_bin"] == lbl).sum())
        gt_in = int((df["gt_bin"] == lbl).sum())
        prec = tp / pred_in if pred_in else np.nan
        rec  = tp / gt_in if gt_in else np.nan
        f1   = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
        rows.append({"bin": lbl, "n_slides": gt_in,
                     "precision": prec, "recall": rec, "f1": f1,
                     "n_tp": tp, "n_pred_in_bin": pred_in})
    return pd.DataFrame(rows)


def fig_buckets(perslide: dict[str, pd.DataFrame]):
    """Bucketed per-slide count classification per approach × entity."""
    fig, axes = plt.subplots(2, len(perslide), figsize=(4.5 * len(perslide), 7),
                             sharey=True)
    if len(perslide) == 1:
        axes = axes.reshape(2, 1)
    summary = {}
    for j, (k, df) in enumerate(perslide.items()):
        for i, (entity, gt_col, pred_col, buckets) in enumerate([
            ("TLS", "gt_n_tls", "n_tls_pred", BUCKETS_TLS),
            ("GC",  "gt_n_gc",  "n_gc_pred",  BUCKETS_GC),
        ]):
            ax = axes[i, j]
            bm = bucket_metrics(df, gt_col, pred_col, buckets)
            x = np.arange(len(bm))
            w = 0.27
            ax.bar(x - w, bm["precision"], width=w, label="precision",
                   color="#4477AA", edgecolor="black")
            ax.bar(x,     bm["recall"],    width=w, label="recall",
                   color="#EE6677", edgecolor="black")
            ax.bar(x + w, bm["f1"],        width=w, label="F1",
                   color="#228833", edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bm["bin"], bm["n_slides"])])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{k}: {entity} bin")
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            summary.setdefault(k, {})[entity] = bm.to_dict("records")
    axes[0, 0].set_ylabel("score")
    axes[1, 0].set_ylabel("score")
    fig.suptitle("Bucketed per-slide count classification (gt vs predicted instance count)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_buckets.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def fig_detection_cm(perslide: dict[str, pd.DataFrame]):
    """Slide-level binary detection: any TLS / GC vs none.

    Cohort = positives (gt_n > 0) + negatives (gt_negative=True or
    gt_n == 0). Titles surface the n_pos / n_neg split so the reader
    can see whether the GT-negative pool is included.
    """
    fig, axes = plt.subplots(2, len(perslide), figsize=(3.4 * len(perslide), 6.8))
    if len(perslide) == 1:
        axes = axes.reshape(2, 1)
    summary = {}
    for j, (k, df) in enumerate(perslide.items()):
        for i, (entity, gt_col, pred_col) in enumerate([
            ("TLS", "gt_n_tls", "n_tls_pred"),
            ("GC",  "gt_n_gc",  "n_gc_pred"),
        ]):
            ax = axes[i, j]
            gt = (df[gt_col].astype(int) > 0).astype(int).values
            pred = (df[pred_col].astype(int) > 0).astype(int).values
            n_pos = int(gt.sum()); n_neg = int((1 - gt).sum())
            cm = confusion_matrix(gt, pred, labels=[0, 1])
            prec = precision_score(gt, pred, zero_division=0)
            rec  = recall_score(gt, pred, zero_division=0)
            f1   = f1_score(gt, pred, zero_division=0)
            ax.imshow(cm, cmap="Blues")
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black",
                        fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred ≥1"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["gt 0", "gt ≥1"])
            ax.set_title(f"{k}: {entity}\n"
                         f"P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}\n"
                         f"(n+={n_pos}  n-={n_neg})",
                         fontsize=9)
            summary.setdefault(k, {})[entity] = {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                "precision": prec, "recall": rec, "f1": f1,
                "n_pos": n_pos, "n_neg": n_neg,
            }
    fig.suptitle("Slide-level detection (any-instance) confusion matrices "
                 "[full cohort: positives + GT-negatives]", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_detection_cm.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def fig_pr_roc(perslide: dict[str, pd.DataFrame]):
    """Slide-level binary detection PR + ROC curves using predicted instance
    counts as the ranking score.

    Per-slide we don't have a probabilistic detection score on disk; the
    eval pipeline thresholds at 0.5 then counts connected components.
    Using `n_pred` as the score means the curve slides over count
    thresholds (≥1, ≥2, …) — it gives an honest read on how well the
    model's count ranks GT-positive vs GT-negative slides.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    summary = {}
    for ent_idx, (entity, gt_col, pred_col) in enumerate([
        ("TLS", "gt_n_tls", "n_tls_pred"),
        ("GC",  "gt_n_gc",  "n_gc_pred"),
    ]):
        ax_roc = axes[ent_idx, 0]
        ax_pr  = axes[ent_idx, 1]
        for k, df in perslide.items():
            gt = (df[gt_col].astype(int) > 0).astype(int).values
            score = df[pred_col].astype(float).values
            # Edge case: TLS GT is all-positive in this cohort → AUROC undefined,
            # PR-AUC trivially = positive prevalence. Mark and skip.
            n_pos = int(gt.sum()); n_neg = int((1 - gt).sum())
            color = APPROACH_COLORS.get(k, "#888")
            block = {"n_pos": n_pos, "n_neg": n_neg}
            if n_pos == 0 or n_neg == 0:
                ax_roc.plot([], [], color=color,
                            label=f"{k}: AUROC=n/a  (gt all-{'pos' if n_neg == 0 else 'neg'})")
                # PR-AUC = prevalence at constant 1.
                ap = n_pos / max(1, n_pos + n_neg)
                ax_pr.plot([], [], color=color,
                           label=f"{k}: PR-AUC≈{ap:.2f} (trivial)")
                block["auroc"] = None
                block["pr_auc"] = ap
            else:
                fpr, tpr, _ = roc_curve(gt, score)
                prec, rec, _ = precision_recall_curve(gt, score)
                au = roc_auc_score(gt, score)
                ap = average_precision_score(gt, score)
                ax_roc.plot(fpr, tpr, color=color, lw=1.6,
                            label=f"{k}: AUROC={au:.3f}")
                ax_pr.plot(rec, prec, color=color, lw=1.6,
                           label=f"{k}: PR-AUC={ap:.3f}")
                block["auroc"] = float(au)
                block["pr_auc"] = float(ap)
            summary.setdefault(k, {})[entity] = block

        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{entity} detection — ROC (score = n_pred)")
        ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.grid(linestyle=":", alpha=0.5)
        ax_roc.legend(fontsize=8, loc="lower right")

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{entity} detection — PR (score = n_pred)")
        ax_pr.set_xlim(-0.02, 1.02); ax_pr.set_ylim(-0.02, 1.02)
        ax_pr.grid(linestyle=":", alpha=0.5)
        ax_pr.legend(fontsize=8, loc="lower left")

    fig.suptitle("Slide-level detection — ROC & PR curves "
                 "(predicted instance count as ranking score) "
                 "[full cohort: positives + GT-negatives]",
                 y=1.00, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pr_roc.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def fig_one_summary(perslide: dict[str, pd.DataFrame]):
    """One composite figure with per-slide Dice/IoU distributions (boxplots,
    new top rows), per-item Dice/IoU means with std error bars, bucketed
    P/R/F1, confusion matrices, and regression metrics for every
    architecture. Output: fig_one_summary.png.
    """
    arch_names = list(perslide.keys())
    n = len(arch_names)
    x = np.arange(n)

    # ── 12-row figure: 2 new boxplot rows on top, then existing layout ──
    fig = plt.figure(figsize=(max(14, 1.4 * n + 8), 33))
    gs = fig.add_gridspec(
        nrows=12, ncols=3,
        height_ratios=[
            1.2, 1.2,             # Row 0+1 — NEW per-slide Dice/IoU boxplots
            1.0, 1.0, 1.0,        # Row 2+3+4 — per-item Dice mean / IoU mean / bucket F1
            0.4, 1.2, 1.2, 0.4,   # Row 5 spacer / Row 6 TLS CM / Row 7 GC CM / Row 8 spacer
            1.2, 1.0, 1.0,        # Row 9+10+11 — regression
        ],
        hspace=0.6, wspace=0.32,
    )

    colors = [APPROACH_COLORS.get(k, "#888") for k in arch_names]

    # ─── NEW Row 0: per-slide TLS / GC / mean Dice boxplots (variability) ───
    def _per_slide_arrays(col):
        return [df[col].dropna().values for df in perslide.values()]

    def _per_slide_iou(col):
        return [df[col].dropna().map(dice_to_iou).values for df in perslide.values()]

    def _per_slide_mean_pair(col_a, col_b):
        out = []
        for df in perslide.values():
            a = df[col_a].dropna().values
            b = df[col_b].dropna().values
            n_min = min(len(a), len(b))
            if n_min:
                out.append((a[:n_min] + b[:n_min]) / 2.0)
            else:
                out.append(np.array([]))
        return out

    def _draw_box(ax, data_lists, title, ylabel, ylim=(0, 1.05)):
        bp = ax.boxplot(data_lists, widths=0.6, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="k", markersize=4))
        for patch, lbl in zip(bp["boxes"], arch_names):
            patch.set_facecolor(APPROACH_COLORS.get(lbl, "#888"))
            patch.set_alpha(0.6)
        ax.set_xticks(np.arange(1, len(arch_names) + 1))
        ax.set_xticklabels(arch_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(*ylim); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax_box_tls_d = fig.add_subplot(gs[0, 0])
    ax_box_gc_d  = fig.add_subplot(gs[0, 1])
    ax_box_m_d   = fig.add_subplot(gs[0, 2])
    _draw_box(ax_box_tls_d, _per_slide_arrays("tls_dice_grid"),
              "TLS Dice — per-slide distribution", "Dice")
    _draw_box(ax_box_gc_d,  _per_slide_arrays("gc_dice_grid"),
              "GC Dice — per-slide distribution", "Dice")
    _draw_box(ax_box_m_d,   _per_slide_mean_pair("tls_dice_grid", "gc_dice_grid"),
              "mean(TLS,GC) Dice — per-slide distribution", "Dice")

    # ─── NEW Row 1: per-slide TLS / GC / mean IoU boxplots ───
    ax_box_tls_i = fig.add_subplot(gs[1, 0])
    ax_box_gc_i  = fig.add_subplot(gs[1, 1])
    ax_box_m_i   = fig.add_subplot(gs[1, 2])

    def _per_slide_iou_pair():
        out = []
        for df in perslide.values():
            a = df["tls_dice_grid"].dropna().map(dice_to_iou).values
            b = df["gc_dice_grid"].dropna().map(dice_to_iou).values
            n_min = min(len(a), len(b))
            out.append(((a[:n_min] + b[:n_min]) / 2.0) if n_min else np.array([]))
        return out

    _draw_box(ax_box_tls_i, _per_slide_iou("tls_dice_grid"),
              "TLS IoU — per-slide distribution", "IoU")
    _draw_box(ax_box_gc_i,  _per_slide_iou("gc_dice_grid"),
              "GC IoU — per-slide distribution", "IoU")
    _draw_box(ax_box_m_i,   _per_slide_iou_pair(),
              "mean(TLS,GC) IoU — per-slide distribution", "IoU")

    # ─── Row 2 — per-item Dice (TLS, GC, mDice) — means with ±1 std error bars ───
    ax_dice_tls = fig.add_subplot(gs[2, 0])
    ax_dice_gc  = fig.add_subplot(gs[2, 1])
    ax_dice_m   = fig.add_subplot(gs[2, 2])
    tls_dice = np.array([np.nanmean(df["tls_dice_grid"].dropna()) for df in perslide.values()])
    gc_dice  = np.array([np.nanmean(df["gc_dice_grid"].dropna())  for df in perslide.values()])
    tls_dice_std = np.array([np.nanstd(df["tls_dice_grid"].dropna()) for df in perslide.values()])
    gc_dice_std  = np.array([np.nanstd(df["gc_dice_grid"].dropna())  for df in perslide.values()])
    m_dice   = (tls_dice + gc_dice) / 2.0
    # std of (a+b)/2 for paired samples = sqrt((var(a)+var(b)+2*cov)/4); approximate as mean of stds.
    m_dice_std = (tls_dice_std + gc_dice_std) / 2.0
    for ax, vals, errs, title in [
        (ax_dice_tls, tls_dice, tls_dice_std, "TLS Dice (per-slide mean ± std)"),
        (ax_dice_gc,  gc_dice,  gc_dice_std,  "GC Dice (per-slide mean ± std)"),
        (ax_dice_m,   m_dice,   m_dice_std,   "mDice ± std"),
    ]:
        ax.bar(x, vals, yerr=errs, color=colors, edgecolor="black",
               capsize=3, error_kw=dict(lw=0.8, ecolor="black"))
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(arch_names, rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel("Dice")
        ax.set_title(title); ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ─── Row 3 — per-item IoU (TLS, GC, mIoU) — means with ±1 std error bars ───
    ax_iou_tls = fig.add_subplot(gs[3, 0])
    ax_iou_gc  = fig.add_subplot(gs[3, 1])
    ax_iou_m   = fig.add_subplot(gs[3, 2])
    tls_iou_arrs = [df["tls_dice_grid"].dropna().map(dice_to_iou).values for df in perslide.values()]
    gc_iou_arrs  = [df["gc_dice_grid"].dropna().map(dice_to_iou).values  for df in perslide.values()]
    tls_iou_per = np.array([a.mean() if len(a) else np.nan for a in tls_iou_arrs])
    gc_iou_per  = np.array([a.mean() if len(a) else np.nan for a in gc_iou_arrs])
    tls_iou_std = np.array([a.std() if len(a) else 0.0 for a in tls_iou_arrs])
    gc_iou_std  = np.array([a.std() if len(a) else 0.0 for a in gc_iou_arrs])
    m_iou       = (tls_iou_per + gc_iou_per) / 2.0
    m_iou_std   = (tls_iou_std + gc_iou_std) / 2.0
    for ax, vals, errs, title in [
        (ax_iou_tls, tls_iou_per, tls_iou_std, "TLS IoU (per-slide mean ± std, derived)"),
        (ax_iou_gc,  gc_iou_per,  gc_iou_std,  "GC IoU (per-slide mean ± std, derived)"),
        (ax_iou_m,   m_iou,       m_iou_std,   "mIoU ± std"),
    ]:
        ax.bar(x, vals, yerr=errs, color=colors, edgecolor="black",
               capsize=3, error_kw=dict(lw=0.8, ecolor="black"))
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(arch_names, rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel("IoU")
        ax.set_title(title); ax.grid(axis="y", linestyle=":", alpha=0.5)

    # ─── Row 3 — bucketed P/R/F1: 3-class TLS, 4-class TLS, 2-class GC ───
    def _bucket_grouped_bars(ax, perslide_dict, gt_col, pred_col, buckets, title):
        """Grouped bars: F1 per bin × per architecture."""
        bins = [b[2] for b in buckets]
        width = 0.8 / max(1, len(perslide_dict))
        for j, (k, df) in enumerate(perslide_dict.items()):
            bm = bucket_metrics(df, gt_col, pred_col, buckets)
            offsets = (np.arange(len(bins)) - len(perslide_dict) / 2 + j + 0.5) * width
            f1s = bm["f1"].values
            ax.bar(np.arange(len(bins)) + offsets, f1s, width=width,
                   label=k, color=APPROACH_COLORS.get(k, "#888"),
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels([f"{b}\n(n={int(bm['n_slides'].iloc[i])})" for i, b in enumerate(bins)],
                           fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
        ax.set_title(title); ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    ax_b3 = fig.add_subplot(gs[4, 0])
    _bucket_grouped_bars(ax_b3, perslide, "gt_n_tls", "n_tls_pred",
                         BUCKETS_TLS_3CLASS, "TLS — 3-class bucket F1 {0, 1-5, 6+}")
    ax_b4 = fig.add_subplot(gs[4, 1])
    _bucket_grouped_bars(ax_b4, perslide, "gt_n_tls", "n_tls_pred",
                         BUCKETS_TLS, "TLS — 4-class bucket F1 {0, 1-3, 4-10, 11+}")
    ax_bgc = fig.add_subplot(gs[4, 2])
    _bucket_grouped_bars(ax_bgc, perslide, "gt_n_gc", "n_gc_pred",
                         BUCKETS_GC_2CLASS, "GC — 2-class bucket F1 {0, ≥1}")

    # ─── Spacer + Row 4: confusion matrices (TLS 4-class, top row; GC 2-class, bottom row) ───
    # Pre-compute CMs
    cms_tls_4 = []
    cms_gc_2 = []
    for k, df in perslide.items():
        gt4 = df["gt_n_tls"].astype(int).map(lambda v: assign_bucket(v, BUCKETS_TLS)).values
        pred4 = df["n_tls_pred"].astype(int).map(lambda v: assign_bucket(v, BUCKETS_TLS)).values
        labels4 = [b[2] for b in BUCKETS_TLS]
        cm4 = confusion_matrix(gt4, pred4, labels=labels4)
        cms_tls_4.append(cm4)

        gt2 = (df["gt_n_gc"].astype(int) > 0).astype(int).values
        pred2 = (df["n_gc_pred"].astype(int) > 0).astype(int).values
        cm2 = confusion_matrix(gt2, pred2, labels=[0, 1])
        cms_gc_2.append(cm2)

    # Row 5+6: TLS 4-class CMs (one per arch, side by side, 1 row across the figure)
    # Use a sub-gridspec for a strip of n CMs
    tls_cm_gs = gs[6, :].subgridspec(1, n, wspace=0.2)
    for j, (k, cm) in enumerate(zip(arch_names, cms_tls_4)):
        ax = fig.add_subplot(tls_cm_gs[0, j])
        # Row-normalise for readability (recall matrix)
        with np.errstate(invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            txt = f"{v}\n{cm_norm[r, c]:.2f}" if cm.sum() else ""
            ax.text(c, r, txt, ha="center", va="center", fontsize=6,
                    color="white" if cm_norm[r, c] > 0.5 else "black")
        ax.set_xticks(range(len(BUCKETS_TLS))); ax.set_yticks(range(len(BUCKETS_TLS)))
        ax.set_xticklabels([b[2] for b in BUCKETS_TLS], fontsize=6, rotation=45)
        ax.set_yticklabels([b[2] for b in BUCKETS_TLS], fontsize=6)
        ax.set_title(k, fontsize=8)
        if j == 0:
            ax.set_ylabel("gt bin", fontsize=8)
        ax.set_xlabel("pred bin", fontsize=7)
    fig.text(0.02, gs[6, 0].get_position(fig).y1 + 0.005,
             "TLS 4-class bucket confusion matrices (row-normalised)",
             fontsize=10, fontweight="bold")

    # Row 7: GC 2-class CMs
    gc_cm_gs = gs[7, :].subgridspec(1, n, wspace=0.2)
    for j, (k, cm) in enumerate(zip(arch_names, cms_gc_2)):
        ax = fig.add_subplot(gc_cm_gs[0, j])
        with np.errstate(invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_norm, cmap="Oranges", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}\n{cm_norm[r, c]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[r, c] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred ≥1"], fontsize=7)
        ax.set_yticklabels(["gt 0", "gt ≥1"], fontsize=7)
        # Add P / R / F1
        prec = precision_score((cm.sum(axis=1) > 0).astype(int) if False else
                               np.repeat([0, 1], cm.sum(axis=1)),
                               np.concatenate([np.repeat([0, 1], cm[r]) for r in range(2)]),
                               zero_division=0) if False else None
        gt_arr = np.concatenate([np.zeros(cm[0].sum(), int), np.ones(cm[1].sum(), int)])
        pr_arr = np.concatenate([np.zeros(cm[0, 0], int), np.ones(cm[0, 1], int),
                                  np.zeros(cm[1, 0], int), np.ones(cm[1, 1], int)])
        if len(gt_arr):
            p = precision_score(gt_arr, pr_arr, zero_division=0)
            r = recall_score(gt_arr, pr_arr, zero_division=0)
            f = f1_score(gt_arr, pr_arr, zero_division=0)
        else:
            p = r = f = 0.0
        ax.set_title(f"{k}\nP={p:.2f}  R={r:.2f}  F1={f:.2f}", fontsize=7)
    fig.text(0.02, gs[7, 0].get_position(fig).y1 + 0.005,
             "GC 2-class bucket confusion matrices (row-normalised)",
             fontsize=10, fontweight="bold")

    # ─── Row 7: Regression metrics (Spearman, Pearson, R², MAE) for TLS and GC counts ───
    def _regress(ax, perslide_dict, gt_col, pred_col, title, metric):
        vals = []
        for k, df in perslide_dict.items():
            stats = _corr_block(df[gt_col].astype(float).values, df[pred_col].astype(float).values)
            vals.append(stats.get(metric, float("nan")))
        ax.bar(x, vals, color=colors, edgecolor="black")
        for xi, v in zip(x, vals):
            if v is None or np.isnan(v):
                ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=7, color="red")
            else:
                ax.text(xi, v + 0.02 * (1 if v >= 0 else -1), f"{v:.2f}",
                        ha="center", va=("bottom" if v >= 0 else "top"), fontsize=7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(arch_names, rotation=30, ha="right")
        ax.set_title(title); ax.grid(axis="y", linestyle=":", alpha=0.5)
        # Auto y-limits: clip extreme R² for readability
        if metric == "r2":
            ymin = max(-2.0, min([v for v in vals if v is not None and not np.isnan(v)] + [-0.5]))
            ax.set_ylim(ymin, 1.05)
        elif metric in ("sp", "pe"):
            ax.set_ylim(-0.2, 1.05)
        elif metric == "mae":
            ax.set_ylim(0, max([v for v in vals if v is not None and not np.isnan(v)] + [1]) * 1.15)

    ax_sp_tls = fig.add_subplot(gs[9, 0])
    _regress(ax_sp_tls, perslide, "gt_n_tls", "n_tls_pred", "TLS Spearman ρ", "sp")
    ax_pe_tls = fig.add_subplot(gs[9, 1])
    _regress(ax_pe_tls, perslide, "gt_n_tls", "n_tls_pred", "TLS Pearson r", "pe")
    ax_r2_tls = fig.add_subplot(gs[9, 2])
    _regress(ax_r2_tls, perslide, "gt_n_tls", "n_tls_pred", "TLS R²", "r2")

    ax_sp_gc = fig.add_subplot(gs[10, 0])
    _regress(ax_sp_gc, perslide, "gt_n_gc", "n_gc_pred", "GC Spearman ρ", "sp")
    ax_pe_gc = fig.add_subplot(gs[10, 1])
    _regress(ax_pe_gc, perslide, "gt_n_gc", "n_gc_pred", "GC Pearson r", "pe")
    ax_r2_gc = fig.add_subplot(gs[10, 2])
    _regress(ax_r2_gc, perslide, "gt_n_gc", "n_gc_pred", "GC R²", "r2")

    ax_mae_tls = fig.add_subplot(gs[11, 0])
    _regress(ax_mae_tls, perslide, "gt_n_tls", "n_tls_pred", "TLS MAE (instances/slide)", "mae")
    ax_mae_gc = fig.add_subplot(gs[11, 1])
    _regress(ax_mae_gc, perslide, "gt_n_gc", "n_gc_pred", "GC MAE (instances/slide)", "mae")
    # Ranking summary panel
    ax_rank = fig.add_subplot(gs[11, 2])
    ax_rank.axis("off")
    # Overall composite rank: average rank across (mDice, mIoU, GC F1, TLS F1, TLS Sp, GC Sp, R²)
    rank_metrics = {}
    for j, (k, df) in enumerate(perslide.items()):
        rank_metrics[k] = {
            "mDice": (np.nanmean(df["tls_dice_grid"]) + np.nanmean(df["gc_dice_grid"])) / 2,
            "TLS Sp": _corr_block(df["gt_n_tls"].astype(float).values,
                                  df["n_tls_pred"].astype(float).values).get("sp", np.nan),
            "GC Sp":  _corr_block(df["gt_n_gc"].astype(float).values,
                                  df["n_gc_pred"].astype(float).values).get("sp", np.nan),
        }
    rdf = pd.DataFrame(rank_metrics).T  # rows = arch, cols = metrics
    # Rank each metric (higher = better; rank 1 best)
    ranks = rdf.rank(method="min", ascending=False, na_option="bottom")
    rdf["avg_rank"] = ranks.mean(axis=1)
    rdf = rdf.sort_values("avg_rank")
    # Render as a small summary table on the panel
    rows_to_show = [(name, f"{rdf.loc[name, 'mDice']:.3f}",
                     f"{rdf.loc[name, 'TLS Sp']:.2f}",
                     f"{rdf.loc[name, 'GC Sp']:.2f}",
                     f"{rdf.loc[name, 'avg_rank']:.1f}")
                    for name in rdf.index]
    table = ax_rank.table(
        cellText=rows_to_show,
        colLabels=["Architecture", "mDice", "TLS Sp", "GC Sp", "Avg rank"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.3)
    ax_rank.set_title("Composite ranking (lower avg rank = better)",
                       fontsize=10, fontweight="bold", pad=10)

    fig.suptitle(
        "GARS / GNCAF / seg_v2 — single-glance architecture summary "
        "(per-item Dice/IoU; bucketed F1 + CMs; regression metrics)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(OUT / "fig_one_summary.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return rdf


def fig_cancer_breakdown(perslide: dict[str, pd.DataFrame]):
    cancers = sorted({c for df in perslide.values()
                      for c in df["cancer_type"].unique()})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.8 / max(1, len(perslide))
    summary = {}
    for j, (entity, dice_col) in enumerate([("TLS", "tls_dice_grid"),
                                             ("GC",  "gc_dice_grid")]):
        ax = axes[j]
        for k_idx, (k, df) in enumerate(perslide.items()):
            means = []
            for c in cancers:
                v = df[df["cancer_type"] == c][dice_col].dropna().values
                means.append(np.nanmean(v) if len(v) else np.nan)
            x = np.arange(len(cancers)) + (k_idx - len(perslide) / 2 + 0.5) * width
            ax.bar(x, means, width=width, label=k, color=APPROACH_COLORS.get(k, "#888"),
                   edgecolor="black")
            summary.setdefault(k, {})[entity] = dict(zip(cancers, means))
        ax.set_xticks(np.arange(len(cancers)))
        ax.set_xticklabels(cancers)
        ax.set_title(f"{entity} mean Dice per cancer type")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Per-cancer-type breakdown (patch-grid Dice)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cancer_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    return summary


# ───────────────────────── main driver ──────────────────────────────

def main():
    print(f"Output dir: {OUT}")

    # 5-fold benchmark from CSV — covers Cascade, GNCAF v3.58/v3.56, seg_v2.0.
    df = load_benchmark()
    fig_5fold_bars(df)
    print("  saved fig_5fold_bars.png")

    # Per-slide rows. v3.56 / v3.58 have all 5 folds; v3.59 fold-0 only.
    perslide = {}
    # Cascade priority: fullcohort (with negatives) > 5fold per-slide (positives only) > legacy.
    cascade_label_folds = [
        # Phase 1: cascade fullcohort fold 0 (with GT-negatives).
        ("v3.37_fullcohort", 0),
        # Phase 2 (when ready): cascade folds 1-4 with negatives.
        ("cascade_fullcohort_5fold", 1),
        ("cascade_fullcohort_5fold", 2),
        ("cascade_fullcohort_5fold", 3),
        ("cascade_fullcohort_5fold", 4),
        # Legacy (positives-only) — kept only as a fallback for folds
        # not yet re-evaluated. Same (slide_id, fold) keys are deduped
        # with the fullcohort row taking priority.
        ("v3.37_perslide", 0),
    ]
    casc = collect_perslide_cascade(cascade_label_folds)
    # Dedupe (slide_id, fold) — fullcohort row wins over the older positive-only.
    if len(casc):
        casc = casc.drop_duplicates(subset=["slide_id", "fold"], keep="first")
        perslide["Cascade v3.37"] = casc
        nfolds = casc["fold"].nunique()
        nneg = int(casc["gt_negative"].sum())
        print(f"  v3.37 cascade per-slide rows: {len(casc)} ({nfolds} fold(s), {nneg} GT-neg)")
    perslide["GNCAF v3.58"] = collect_perslide("v3.58", folds=list(range(5)))
    perslide["GNCAF v3.56"] = collect_perslide("v3.56", folds=list(range(5)))
    v59 = collect_perslide("v3.59", folds=[0])
    if len(v59):
        perslide["GNCAF v3.59"] = v59
        print(f"  v3.59 per-slide rows: {len(v59)} (fold 0 only)")
    v21 = collect_perslide("v3.21", folds=[0])
    if len(v21):
        perslide["GNCAF v3.21"] = v21
        print(f"  v3.21 per-slide rows: {len(v21)} (fold 0 only)")
    v61 = collect_perslide("v3.61", folds=[0])
    if len(v61):
        perslide["GNCAF v3.61"] = v61
        print(f"  v3.61 per-slide rows: {len(v61)} (fold 0 only)")
    v62 = collect_perslide("v3.62", folds=[0])
    if len(v62):
        perslide["GNCAF v3.62"] = v62
        print(f"  v3.62 per-slide rows: {len(v62)} (fold 0 only)")
    v63 = collect_perslide("v3.63", folds=[0])
    if len(v63):
        perslide["GNCAF v3.63"] = v63
        print(f"  v3.63 per-slide rows: {len(v63)} (fold 0 only)")
    v64 = collect_perslide("v3.64", folds=[0])
    if len(v64):
        perslide["GNCAF v3.64"] = v64
        print(f"  v3.64 per-slide rows: {len(v64)} (fold 0 only)")
    v65 = collect_perslide("v3.65", folds=[0])
    if len(v65):
        perslide["GNCAF v3.65"] = v65
        print(f"  v3.65 per-slide rows: {len(v65)} (fold 0 only)")
    lost07 = collect_perslide("lost_0p7143", folds=[0])
    if len(lost07):
        perslide["Lost-0.7143"] = lost07
        print(f"  Lost-0.7143 per-slide rows: {len(lost07)} (fold 0 only)")
    sv2 = collect_perslide("seg_v2", folds=list(range(5)))
    if len(sv2):
        perslide["seg_v2.0"] = sv2
        print(f"  seg_v2.0 per-slide rows: {len(sv2)} ({sv2['fold'].nunique()} folds)")
    sv2_dual = collect_perslide("seg_v2_dual", folds=list(range(5)))
    if len(sv2_dual):
        perslide["seg_v2.0 (dual)"] = sv2_dual
        print(f"  seg_v2.0 (dual) per-slide rows: {len(sv2_dual)} ({sv2_dual['fold'].nunique()} folds)")
    fb_h128 = collect_perslide("family_b_frozen_gn_h128", folds=[0])
    if len(fb_h128):
        perslide["Family-B h=128 (lost)"] = fb_h128
        print(f"  Family-B h=128 per-slide rows: {len(fb_h128)} (fold 0 only)")
    fb_h256 = collect_perslide("family_b_demo_h256", folds=[0])
    if len(fb_h256):
        perslide["Family-B h=256 (lost demo)"] = fb_h256
        print(f"  Family-B h=256 per-slide rows: {len(fb_h256)} (fold 0 only)")
    for k, d in perslide.items():
        print(f"  {k}: {len(d)} per-slide rows")

    fig_perslide_dice(perslide)
    print("  saved fig_perslide_dice.png")

    corr = fig_count_scatter(perslide)
    print("  saved fig_count_scatter.png")

    bk = fig_buckets(perslide)
    print("  saved fig_buckets.png")

    cm = fig_detection_cm(perslide)
    print("  saved fig_detection_cm.png")

    pr_roc = fig_pr_roc(perslide)
    print("  saved fig_pr_roc.png")

    cb = fig_cancer_breakdown(perslide)
    print("  saved fig_cancer_breakdown.png")

    one_summary_rank = fig_one_summary(perslide)
    print("  saved fig_one_summary.png")

    # Persist a JSON companion to the markdown.
    summary = {
        "fold_means": {a: {m: float(df[df["approach"] == a][m].mean())
                           for m in ["mDice_pix", "tls_dice_pix", "gc_dice_pix"]}
                       for a in df["approach"].unique()},
        "fold_stds": {a: {m: float(df[df["approach"] == a][m].std(ddof=1))
                          for m in ["mDice_pix", "tls_dice_pix", "gc_dice_pix"]}
                      for a in df["approach"].unique()},
        "perslide_corr": corr,
        "perslide_buckets": bk,
        "detection_cm": cm,
        "detection_pr_roc": pr_roc,
        "cancer_breakdown": cb,
    }
    (OUT / "arch_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  saved arch_summary.json")


if __name__ == "__main__":
    main()
