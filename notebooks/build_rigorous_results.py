"""Rigorous per-experiment results aggregation.

For every evaluated architecture (cascade, GNCAF variants, seg_v2.0 variants,
ensembles), compute:

  Segmentation
    - TLS Dice (strict + union), GC Dice — per-slide mean ± std (95% CI),
      pixel-agg (cohort-level)
    - Same broken down by cancer type (BLCA / KIRC / LUSC)
  Regression (instance counts: pred vs gt metadata)
    - Spearman, Pearson, R², MAE for TLS and GC, plus 95% bootstrap CI
  Classification (slide-level binary detection)
    - Precision / Recall / F1 / balanced accuracy
    - AUROC and AUPRC (score = predicted instance count)
    - For TLS: positive = (gt_n_tls > 0)
    - For GC : positive = (gt_n_gc > 0)
  FP-rate metrics
    - TLS-FP = #{neg slide : n_tls_pred > 0} / #neg
    - GC-FP  = #{neg slide : n_gc_pred  > 0} / #neg
    - Mean predicted instances on negative slides
  Bucketed TLS detection
    - 3-class CM {0, 1-5, 6+}: P/R/F1 per bin
    - 4-class CM {0, 1-3, 4-10, 11+}: P/R/F1 per bin
    - 2-class GC CM {0, ≥1}: P/R/F1

Outputs:
  notebooks/architectures/rigorous_results/<arch_slug>.json       (per arch)
  notebooks/architectures/rigorous_results/_summary.json          (combined)
  notebooks/architectures/rigorous_results/_summary.md            (markdown table)
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    mean_absolute_error, r2_score,
)

EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")
OUT = Path("/home/ubuntu/auto-clam-seg/notebooks/architectures/rigorous_results")
OUT.mkdir(parents=True, exist_ok=True)


# ── Bucket definitions (match the build_arch_comparison conventions) ──
BUCKETS_TLS_3 = [(0, 0, "0"), (1, 5, "1-5"), (6, 1_000_000, "6+")]
BUCKETS_TLS_4 = [(0, 0, "0"), (1, 3, "1-3"), (4, 10, "4-10"), (11, 1_000_000, "11+")]
BUCKETS_GC_2  = [(0, 0, "0"), (1, 1_000_000, "≥1")]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def bucket_of(n: int, buckets):
    for lo, hi, lab in buckets:
        if lo <= n <= hi:
            return lab
    return buckets[-1][2]


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot=1000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for `stat` on a 1-D numeric array."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[b] = stat(sample)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(stat(values)), lo, hi


# ── Loaders ─────────────────────────────────────────────────────────────


def load_gncaf_rows(label_prefix: str, fold: int = 0) -> pd.DataFrame:
    """Load per-slide JSON rows from gncaf-style eval dirs.

    Auto-prefers union TLS Dice (tls_dice_grid_union) when present.
    """
    patterns = [
        f"gars_gncaf_eval_{label_prefix}_fullcohort_fold{fold}_eval_shard*",
        f"gars_gncaf_eval_{label_prefix}_fullcohort_fold{fold}_shard*",
    ]
    rows = []
    for pat in patterns:
        for d in sorted(EXP.glob(pat)):
            p = d / "gncaf_agg.json"
            if not p.exists():
                continue
            obj = json.loads(p.read_text())
            for r in obj.get("per_slide", []):
                r = dict(r)
                # Promote union if present (apples-to-apples)
                if "tls_dice_grid_union" in r:
                    r["tls_dice_grid"] = r["tls_dice_grid_union"]
                r.setdefault("gt_negative", False)
                rows.append(r)
        if rows:
            break
    return pd.DataFrame(rows)


def load_cascade_rows(label_prefix: str, fold: int = 0, threshold: str = "0.5") -> pd.DataFrame:
    """Load per-slide rows for cascade-style eval (cascade_per_slide.json).
    If multiple matching dirs exist (e.g. failed + successful run), use the
    LATEST one only (sorted by dir name → timestamp suffix sorts naturally).
    """
    rows = []
    pat = f"gars_cascade_{label_prefix}_fold{fold}_eval_shard*"
    dirs = sorted(EXP.glob(pat))
    if not dirs:
        pat2 = f"gars_cascade_{label_prefix}_fold{fold}_*"
        dirs = sorted(EXP.glob(pat2))
        # Dirs share a slide_id namespace (no shard split) — pick LATEST timestamp only.
        if dirs:
            dirs = [dirs[-1]]
    for d in dirs:
        p = d / "cascade_per_slide.json"
        if not p.exists():
            continue
        obj = json.loads(p.read_text())
        for r in obj.get(threshold, []):
            r = dict(r)
            # Rename to match the gncaf-style key naming for unified pipeline.
            r["tls_dice_grid"] = r.get("tls_dice_union", r.get("tls_dice", np.nan))
            r["gc_dice_grid"] = r.get("gc_dice", np.nan)
            r.setdefault("gt_negative", False)
            rows.append(r)
    return pd.DataFrame(rows)


# ── Metric computation per architecture ─────────────────────────────────


def compute_all_metrics(df: pd.DataFrame, arch_name: str) -> dict:
    """Compute the full rigorous metric panel for one architecture."""
    res = {"architecture": arch_name, "n_slides": len(df),
           "n_pos": int((~df["gt_negative"]).sum()),
           "n_neg": int(df["gt_negative"].sum())}

    pos = df[~df["gt_negative"]].copy()
    neg = df[df["gt_negative"]].copy()
    rng = np.random.default_rng(42)

    # ─── Segmentation (TLS/GC Dice per-slide on positives) ────────────
    seg = {}
    for ent, col in [("tls", "tls_dice_grid"), ("gc", "gc_dice_grid")]:
        vals = pos[col].dropna().astype(float).values
        m, lo, hi = bootstrap_ci(vals, rng=rng)
        # Derive per-slide IoU from Dice
        iou_vals = vals / (2.0 - vals + 1e-9)
        m_iou, lo_iou, hi_iou = bootstrap_ci(iou_vals, rng=rng)
        seg[f"{ent}_dice_mean"] = m
        seg[f"{ent}_dice_std"] = float(np.std(vals)) if len(vals) else float("nan")
        seg[f"{ent}_dice_ci95"] = [lo, hi]
        seg[f"{ent}_iou_mean"] = m_iou
        seg[f"{ent}_iou_ci95"] = [lo_iou, hi_iou]
    seg["mDice"] = (seg["tls_dice_mean"] + seg["gc_dice_mean"]) / 2.0
    seg["mIoU"] = (seg["tls_iou_mean"] + seg["gc_iou_mean"]) / 2.0
    res["segmentation"] = seg

    # ─── Per cancer type ──────────────────────────────────────────────
    by_cancer = {}
    for ct in sorted(pos["cancer_type"].dropna().unique()):
        sub = pos[pos["cancer_type"] == ct]
        by_cancer[ct] = {
            "n_slides": len(sub),
            "tls_dice_mean": float(sub["tls_dice_grid"].mean()),
            "tls_dice_std":  float(sub["tls_dice_grid"].std()) if len(sub) > 1 else 0.0,
            "gc_dice_mean":  float(sub["gc_dice_grid"].mean()),
            "gc_dice_std":   float(sub["gc_dice_grid"].std()) if len(sub) > 1 else 0.0,
            "mDice":         float((sub["tls_dice_grid"].mean() + sub["gc_dice_grid"].mean()) / 2.0),
        }
    res["by_cancer_type"] = by_cancer

    # ─── Regression (instance counts: pred vs gt metadata) ────────────
    regression = {}
    for ent, gt_col, pred_col in [
        ("tls", "gt_n_tls", "n_tls_pred"),
        ("gc",  "gt_n_gc",  "n_gc_pred"),
    ]:
        gt = df[gt_col].astype(float).values
        pr = df[pred_col].astype(float).values
        # Spearman / Pearson / R² / MAE
        def _block(g, p):
            if len(g) < 3 or np.all(g == 0):
                return {"spearman": np.nan, "pearson": np.nan,
                        "r2": np.nan, "mae": np.nan}
            sp = spearmanr(g, p).correlation
            pe = pearsonr(g, p).statistic
            r2 = r2_score(g, p) if np.var(g) > 0 else np.nan
            mae = mean_absolute_error(g, p)
            return {"spearman": float(sp), "pearson": float(pe),
                    "r2": float(r2), "mae": float(mae)}
        block = _block(gt, pr)
        # Bootstrap each metric
        boot = {k: [] for k in block}
        n = len(gt)
        for _ in range(500):
            idx = rng.integers(0, n, size=n)
            sub = _block(gt[idx], pr[idx])
            for k in boot:
                boot[k].append(sub[k])
        block_ci = {}
        for k, v in block.items():
            arr = np.array([x for x in boot[k] if not (isinstance(x, float) and math.isnan(x))])
            if len(arr):
                lo = float(np.percentile(arr, 2.5))
                hi = float(np.percentile(arr, 97.5))
            else:
                lo = hi = float("nan")
            block_ci[f"{k}_ci95"] = [lo, hi]
        block.update(block_ci)
        regression[ent] = block
    res["regression"] = regression

    # ─── Classification: slide-level TLS / GC detection ───────────────
    classification = {}
    for ent, gt_col, pred_col in [
        ("tls", "gt_n_tls", "n_tls_pred"),
        ("gc",  "gt_n_gc",  "n_gc_pred"),
    ]:
        gt_bin = (df[gt_col].astype(int).values > 0).astype(int)
        pr_bin = (df[pred_col].astype(int).values > 0).astype(int)
        pr_score = df[pred_col].astype(float).values   # ranking score for AUC/AUPRC
        cls = {
            "precision": float(precision_score(gt_bin, pr_bin, zero_division=0)),
            "recall":    float(recall_score(gt_bin, pr_bin, zero_division=0)),
            "f1":        float(f1_score(gt_bin, pr_bin, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(gt_bin, pr_bin)),
            "n_pos": int(gt_bin.sum()), "n_neg": int(len(gt_bin) - gt_bin.sum()),
            "tp": int(((gt_bin == 1) & (pr_bin == 1)).sum()),
            "fp": int(((gt_bin == 0) & (pr_bin == 1)).sum()),
            "fn": int(((gt_bin == 1) & (pr_bin == 0)).sum()),
            "tn": int(((gt_bin == 0) & (pr_bin == 0)).sum()),
        }
        # AUC / AUPRC only if both classes are present
        if gt_bin.sum() > 0 and gt_bin.sum() < len(gt_bin):
            try:
                cls["auroc"] = float(roc_auc_score(gt_bin, pr_score))
                cls["auprc"] = float(average_precision_score(gt_bin, pr_score))
            except Exception:
                cls["auroc"] = float("nan"); cls["auprc"] = float("nan")
        else:
            cls["auroc"] = float("nan"); cls["auprc"] = float("nan")
        classification[ent] = cls
    res["classification"] = classification

    # ─── FP-rate metrics on truly-negative slides ─────────────────────
    fp = {}
    if len(neg):
        for ent, col in [("tls", "n_tls_pred"), ("gc", "n_gc_pred")]:
            preds = neg[col].astype(int).values
            fp[f"{ent}_fp_rate"] = float((preds > 0).mean())
            fp[f"{ent}_fp_count"] = int((preds > 0).sum())
            fp[f"{ent}_mean_pred_per_neg"] = float(preds.mean())
        fp["n_neg_slides"] = int(len(neg))
    res["fp_rate"] = fp

    # ─── Bucketed TLS detection (3-class and 4-class) ─────────────────
    buckets = {}
    for cls_name, bks, gt_col, pred_col in [
        ("tls_3class", BUCKETS_TLS_3, "gt_n_tls", "n_tls_pred"),
        ("tls_4class", BUCKETS_TLS_4, "gt_n_tls", "n_tls_pred"),
        ("gc_2class",  BUCKETS_GC_2,  "gt_n_gc",  "n_gc_pred"),
    ]:
        labels = [b[2] for b in bks]
        gt_lab = df[gt_col].astype(int).map(lambda v: bucket_of(v, bks)).values
        pr_lab = df[pred_col].astype(int).map(lambda v: bucket_of(v, bks)).values
        cm = confusion_matrix(gt_lab, pr_lab, labels=labels)
        per_bin = {}
        for i, lab in enumerate(labels):
            tp = int(cm[i, i])
            fp_cnt = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            prec = tp / (tp + fp_cnt) if (tp + fp_cnt) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_bin[lab] = {
                "n_slides_gt": int(cm[i, :].sum()),
                "precision":   float(prec),
                "recall":      float(rec),
                "f1":          float(f1),
            }
        buckets[cls_name] = {
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "per_bin": per_bin,
            "macro_f1": float(np.mean([per_bin[l]["f1"] for l in labels])),
            "balanced_acc": float(balanced_accuracy_score(gt_lab, pr_lab)),
        }
    res["buckets"] = buckets

    return res


# ── Build registry of (arch_name, loader) ───────────────────────────────


def build_registry():
    return [
        ("Cascade v3.37",
         lambda: load_cascade_rows("cascade_union_fullcohort", fold=0)),
        ("GNCAF v3.58",
         lambda: load_gncaf_rows("v3.58_union", fold=0)),
        ("GNCAF v3.62 (paper-strict)",
         lambda: load_gncaf_rows("v3.62_union", fold=0)),
        ("GNCAF v3.63 (dual-σ, heavy)",
         lambda: load_gncaf_rows("v3.63", fold=0)),
        ("GNCAF v3.65 (dual-σ, simple)",
         lambda: load_gncaf_rows("v3.65", fold=0)),
        ("GNCAF v3.65 + Stage 1 gate",
         lambda: load_gncaf_rows("v3.65_gated", fold=0)),
        ("seg_v2.0 (tls_only)",
         lambda: load_gncaf_rows("seg_v2", fold=0)),
        ("seg_v2.0 (dual)",
         lambda: load_gncaf_rows("seg_v2_dual", fold=0)),
        ("seg_v2.0 (dual) + Stage 1 gate",
         lambda: load_gncaf_rows("seg_v2_dual_gated", fold=0)),
        ("Cascade v3.38 (dual-σ Stage 2)",
         lambda: load_cascade_rows("cascade_v3.38_fullcohort", fold=0)),
    ]


def main():
    summary = {}
    for arch_name, loader in build_registry():
        df = loader()
        if df.empty:
            print(f"  {arch_name}: no rows; skip")
            continue
        print(f"  {arch_name}: {len(df)} slides "
              f"({int((~df['gt_negative']).sum())} pos / "
              f"{int(df['gt_negative'].sum())} neg)")
        res = compute_all_metrics(df, arch_name)
        # Save per-arch JSON
        out_path = OUT / f"{slug(arch_name)}.json"
        out_path.write_text(json.dumps(res, indent=2, default=str))
        summary[arch_name] = res
    # Combined summary JSON
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved per-arch JSONs and _summary.json to {OUT}")

    # ─── Markdown headline table ───────────────────────────────────────
    md_lines = ["# Rigorous evaluation summary — fold-0 fullcohort\n"]
    md_lines.append("All numbers from per-slide rows under the union TLS Dice "
                    "semantic (GC ⊂ TLS). Bootstrap 95% CI on N=124 positive slides.\n")
    md_lines.append("| Architecture | mDice | TLS Dice [95% CI] | GC Dice [95% CI] "
                    "| TLS-FP | GC-FP | TLS det F1 | GC det F1 | "
                    "TLS Spearman | GC Spearman |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for arch_name, res in summary.items():
        s = res["segmentation"]; fp = res["fp_rate"]
        c = res["classification"]; r = res["regression"]
        tls_ci = s.get("tls_dice_ci95", [float("nan")] * 2)
        gc_ci = s.get("gc_dice_ci95", [float("nan")] * 2)
        row = (f"| {arch_name} "
               f"| {s['mDice']:.3f} "
               f"| {s['tls_dice_mean']:.3f} [{tls_ci[0]:.3f}, {tls_ci[1]:.3f}] "
               f"| {s['gc_dice_mean']:.3f} [{gc_ci[0]:.3f}, {gc_ci[1]:.3f}] "
               f"| {100 * fp.get('tls_fp_rate', float('nan')):.1f}% "
               f"| {100 * fp.get('gc_fp_rate', float('nan')):.1f}% "
               f"| {c['tls']['f1']:.3f} "
               f"| {c['gc']['f1']:.3f} "
               f"| {r['tls']['spearman']:.3f} "
               f"| {r['gc']['spearman']:.3f} |")
        md_lines.append(row)
    md_lines.append("\n## Definitions\n")
    md_lines.append("- **TLS Dice (union)**: target_tls = (gt ≥ 1) — GC pixels "
                    "count as TLS biologically. Per-slide mean over positives.")
    md_lines.append("- **TLS-FP rate**: fraction of GT-negative slides where "
                    "`n_tls_pred > 0` after post-processing (min_size=1, "
                    "closing_iters=0).")
    md_lines.append("- **TLS det F1**: slide-level binary F1 — positive = "
                    "(gt_n_tls > 0), pred-positive = (n_tls_pred > 0).")
    md_lines.append("- **TLS Spearman**: rank correlation between `n_tls_pred` "
                    "and metadata gold-standard `gt_n_tls` over all slides.")
    (OUT / "_summary.md").write_text("\n".join(md_lines))
    print(f"Saved markdown summary to {OUT}/_summary.md")


if __name__ == "__main__":
    main()
