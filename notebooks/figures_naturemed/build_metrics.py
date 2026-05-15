"""Compute the full metric panel for the Nature-Med figure set, from
the per-slide predictions CSV.

Metrics (per scope, per model, per cancer):
  - Slide-level regression (TLS, GC counts): ICC, Spearman, Pearson, R², MAE
  - Slide classification (3-class, 4-class): CM, P/R/F1 (macro+weighted)

Output:
  figures_naturemed/metrics_panelA.json   (full 165 slides, 5 models)
  figures_naturemed/metrics_panelB.json   (HookNet subset, 6 models)
  figures_naturemed/summary_table.csv     (long-form rows)
  figures_naturemed/summary_table.md      (markdown grid)

Run:
    python build_metrics.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    r2_score, confusion_matrix, precision_recall_fscore_support, mean_absolute_error,
)

ROOT = Path("/home/ubuntu/auto-clam-seg/figures_naturemed")

MODELS_A = ["Cascade v3.7", "Cascade v3.11", "GNCAF v3.62", "GNCAF v3.65", "seg_v2.0 (dual)"]
MODELS_B = MODELS_A + ["HookNet-TLS"]
COHORTS = ["BLCA", "KIRC", "LUSC"]


def icc(gt: np.ndarray, pr: np.ndarray) -> float | None:
    """Intraclass correlation coefficient (ICC(3,1), two-way mixed,
    single rater, absolute agreement). Use pingouin if available.
    """
    if len(gt) < 3:
        return None
    try:
        import pingouin as pg
        df = pd.DataFrame({
            "rater": ["gt"] * len(gt) + ["pred"] * len(pr),
            "target": list(range(len(gt))) * 2,
            "value": list(gt) + list(pr),
        })
        ic = pg.intraclass_corr(data=df, targets="target", raters="rater", ratings="value")
        # ICC3 (single fixed rater) absolute agreement = "ICC2" by Shrout-Fleiss
        for _, row in ic.iterrows():
            if row["Type"] == "ICC2":
                return float(row["ICC"])
        return float(ic["ICC"].iloc[0])
    except Exception:
        # Fallback: simple consistency ICC via formula (one-way random effects)
        gt = np.asarray(gt, dtype=float); pr = np.asarray(pr, dtype=float)
        if np.std(gt) == 0 and np.std(pr) == 0:
            return 1.0
        mean_per_subj = (gt + pr) / 2
        grand = mean_per_subj.mean()
        ms_between = ((mean_per_subj - grand) ** 2).sum() * 2 / max(len(gt) - 1, 1)
        ms_within = (((gt - mean_per_subj) ** 2 + (pr - mean_per_subj) ** 2)).sum() / max(len(gt), 1)
        denom = ms_between + ms_within
        return float((ms_between - ms_within) / max(denom, 1e-9))


def reg_metrics(gt: np.ndarray, pr: np.ndarray) -> dict:
    gt = np.asarray(gt, dtype=float); pr = np.asarray(pr, dtype=float)
    n = len(gt)
    if n < 3 or np.std(gt) == 0:
        return {"n": int(n), "r2": None, "spearman": None, "pearson": None,
                "icc": None, "mae": float(np.mean(np.abs(gt - pr))) if n else None}
    sp, _ = spearmanr(gt, pr); pe, _ = pearsonr(gt, pr)
    return {
        "n": int(n),
        "r2": float(r2_score(gt, pr)),
        "spearman": float(sp),
        "pearson": float(pe),
        "icc": icc(gt, pr),
        "mae": float(mean_absolute_error(gt, pr)),
    }


def predict_class(pred_count: float, thresholds_3: list, thresholds_4: list) -> tuple[int, int]:
    """Map a predicted count to the 3-class and 4-class label given
    cohort-derived thresholds."""
    c3 = 0
    for t in thresholds_3:
        if pred_count > t:
            c3 += 1
    c4 = 0
    for t in thresholds_4:
        if pred_count > t:
            c4 += 1
    return c3, c4


def derive_thresholds(df: pd.DataFrame) -> tuple[list, list]:
    """Derive count thresholds matching the df-defined classes from
    the gt_n_tls column. The 3-class label is 0/1/2; 4-class is 0/1/2/3.
    Use cutpoints that reproduce the GT class on as many rows as possible.
    """
    sub = df[["gt_n_tls", "tls_count_3class", "tls_count_4class"]].dropna()
    sub = sub.astype({"tls_count_3class": int, "tls_count_4class": int})
    # 3-class: 0=zero, 1=low, 2=high. Find the cutpoint between 1 and 2
    counts_for_class = {c: sub.loc[sub.tls_count_3class == c, "gt_n_tls"].values
                         for c in (0, 1, 2)}
    t3 = []
    if len(counts_for_class[1]) and len(counts_for_class[2]):
        # cut between class 1 and class 2
        c1_max = counts_for_class[1].max()
        c2_min = counts_for_class[2].min()
        t3 = [0, (c1_max + c2_min) / 2]
    else:
        t3 = [0, 10]

    # 4-class: 0/1/2/3
    counts4 = {c: sub.loc[sub.tls_count_4class == c, "gt_n_tls"].values for c in (0, 1, 2, 3)}
    t4 = [0]
    for c in (2, 3):
        if len(counts4[c - 1]) and len(counts4[c]):
            t4.append((counts4[c - 1].max() + counts4[c].min()) / 2)
        else:
            t4.append(t4[-1] + 5)
    return t3, t4


def cls_metrics(gt: np.ndarray, pr: np.ndarray, labels: list) -> dict:
    gt = np.asarray(gt); pr = np.asarray(pr)
    cm = confusion_matrix(gt, pr, labels=labels).tolist()
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(gt, pr, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(gt, pr, labels=labels, average="weighted", zero_division=0)
    p_pc, r_pc, f1_pc, support_pc = precision_recall_fscore_support(gt, pr, labels=labels, average=None, zero_division=0)
    per_class = [
        {"class": int(c), "p": float(p_pc[i]), "r": float(r_pc[i]),
         "f1": float(f1_pc[i]), "n": int(support_pc[i])}
        for i, c in enumerate(labels)
    ]
    return {
        "n": int(len(gt)), "confusion_matrix": cm,
        "macro": {"p": float(p_macro), "r": float(r_macro), "f1": float(f1_macro)},
        "weighted": {"p": float(p_w), "r": float(r_w), "f1": float(f1_w)},
        "per_class": per_class,
    }


def compute_for_subset(df: pd.DataFrame, models: list, scope: str, t3: list, t4: list,
                       cancer_filter: str | None = None) -> dict:
    if cancer_filter:
        df = df[df["cancer_type"] == cancer_filter]
    out = {"scope": scope, "cancer": cancer_filter or "ALL", "n_slides": int(len(df)), "models": {}}
    if len(df) == 0:
        return out
    for m in models:
        n_tls = df[f"{m}_n_tls"]; n_gc = df[f"{m}_n_gc"]
        mask = n_tls.notna() & n_gc.notna()
        sub = df[mask]
        if len(sub) < 3:
            out["models"][m] = None
            continue
        gt_t = sub["gt_n_tls"].values; pr_t = sub[f"{m}_n_tls"].values
        gt_g = sub["gt_n_gc"].values; pr_g = sub[f"{m}_n_gc"].values
        # Classification labels (from gt counts via the same thresholds we'll apply to pred)
        gt_c3 = sub["tls_count_3class"].dropna().astype(int)
        # Predict 3-class from pred counts
        pr_c3 = np.array([predict_class(p, t3, t4)[0] for p in pr_t])
        pr_c4 = np.array([predict_class(p, t3, t4)[1] for p in pr_t])
        # Restrict to rows where we have gt class labels
        valid_c = sub["tls_count_3class"].notna() & sub["tls_count_4class"].notna()
        gt_c3_v = sub.loc[valid_c, "tls_count_3class"].astype(int).values
        gt_c4_v = sub.loc[valid_c, "tls_count_4class"].astype(int).values
        pr_c3_v = pr_c3[valid_c.values]
        pr_c4_v = pr_c4[valid_c.values]

        out["models"][m] = {
            "n_pred": int(len(sub)),
            "reg_tls": reg_metrics(gt_t, pr_t),
            "reg_gc": reg_metrics(gt_g, pr_g),
            "cls3": cls_metrics(gt_c3_v, pr_c3_v, labels=[0, 1, 2]) if len(gt_c3_v) else None,
            "cls4": cls_metrics(gt_c4_v, pr_c4_v, labels=[0, 1, 2, 3]) if len(gt_c4_v) else None,
        }
    return out


def main():
    df = pd.read_csv(ROOT / "per_slide_predictions.csv")
    print(f"Loaded {len(df)} slides")

    # Derive class thresholds from the GT
    t3, t4 = derive_thresholds(df)
    print(f"3-class thresholds: {t3}")
    print(f"4-class thresholds: {t4}")

    # Panel A: 165 slides (our 5 models), no HookNet
    panelA = {"all": compute_for_subset(df, MODELS_A, "A_all_165", t3, t4)}
    for ct in COHORTS:
        panelA[ct] = compute_for_subset(df, MODELS_A, f"A_{ct}", t3, t4, cancer_filter=ct)

    # Panel B: HookNet's subset
    hk_mask = df["HookNet-TLS_present"] == 1
    dfB = df[hk_mask]
    panelB = {"all": compute_for_subset(dfB, MODELS_B, "B_all_hooknet", t3, t4)}
    for ct in COHORTS:
        panelB[ct] = compute_for_subset(dfB, MODELS_B, f"B_{ct}", t3, t4, cancer_filter=ct)

    out_a = ROOT / "metrics_panelA.json"
    out_b = ROOT / "metrics_panelB.json"
    out_a.write_text(json.dumps(panelA, indent=2))
    out_b.write_text(json.dumps(panelB, indent=2))
    print(f"Wrote {out_a}")
    print(f"Wrote {out_b}")
    print(f"Panel B n_slides = {panelB['all']['n_slides']}")

    # Long-form summary table
    rows = []
    for panel_name, panel in (("A", panelA), ("B", panelB)):
        for cancer_key, panel_data in panel.items():
            for m, mdata in (panel_data.get("models") or {}).items():
                if mdata is None: continue
                r = mdata["reg_tls"]; rg = mdata["reg_gc"]
                rows.append({
                    "panel": panel_name, "cancer": panel_data["cancer"],
                    "model": m, "n_slides": mdata["n_pred"],
                    "TLS_R2": r["r2"], "TLS_spearman": r["spearman"],
                    "TLS_pearson": r["pearson"], "TLS_ICC": r["icc"], "TLS_MAE": r["mae"],
                    "GC_R2": rg["r2"], "GC_spearman": rg["spearman"],
                    "GC_pearson": rg["pearson"], "GC_ICC": rg["icc"], "GC_MAE": rg["mae"],
                    "cls3_F1_macro": mdata["cls3"]["macro"]["f1"] if mdata["cls3"] else None,
                    "cls3_F1_weighted": mdata["cls3"]["weighted"]["f1"] if mdata["cls3"] else None,
                    "cls4_F1_macro": mdata["cls4"]["macro"]["f1"] if mdata["cls4"] else None,
                    "cls4_F1_weighted": mdata["cls4"]["weighted"]["f1"] if mdata["cls4"] else None,
                })
    sdf = pd.DataFrame(rows)
    sdf.to_csv(ROOT / "summary_table.csv", index=False)
    print(f"Wrote {ROOT / 'summary_table.csv'}: {len(sdf)} rows")

    # Markdown grid (overall A & B)
    md_lines = ["# Summary table (per-model, per-scope, per-cancer)\n"]
    for panel in ("A", "B"):
        for ct in ["ALL"] + COHORTS:
            sub = sdf[(sdf.panel == panel) & (sdf.cancer == ct)]
            if len(sub) == 0: continue
            md_lines.append(f"\n## Panel {panel} — {ct}\n")
            cols = ["model", "n_slides", "TLS_R2", "TLS_ICC", "TLS_spearman", "TLS_MAE",
                    "GC_R2", "GC_ICC", "GC_spearman", "GC_MAE",
                    "cls3_F1_macro", "cls4_F1_macro"]
            # Manual markdown table (avoid tabulate dependency)
            sub2 = sub[cols].copy()
            for c in cols:
                if sub2[c].dtype in (float, "float64"):
                    sub2[c] = sub2[c].apply(lambda v: "" if pd.isna(v) else f"{v:.3f}")
            md_lines.append("| " + " | ".join(cols) + " |")
            md_lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for _, row in sub2.iterrows():
                md_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    (ROOT / "summary_table.md").write_text("\n".join(md_lines))
    print(f"Wrote {ROOT / 'summary_table.md'}")


if __name__ == "__main__":
    main()
