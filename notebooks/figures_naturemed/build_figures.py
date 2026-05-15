"""Render publication-quality figures 01-07 from the metric JSONs.

Inputs:
  figures_naturemed/per_slide_predictions.csv
  figures_naturemed/metrics_panelA.json
  figures_naturemed/metrics_panelB.json

Outputs: figures_naturemed/fig01_*.png, fig02_*.png, ... (also .pdf where
applicable).

Run:
    python build_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

ROOT = Path("/home/ubuntu/auto-clam-seg/figures_naturemed")

COLORS = {
    "Cascade v3.7":   "#1f77b4",
    "Cascade v3.11":  "#0a3c5c",
    "GNCAF v3.62":    "#17becf",
    "GNCAF v3.65":    "#aa3322",
    "seg_v2.0 (dual)": "#525252",
    "HookNet-TLS":    "#000000",
}
MODELS_A = ["Cascade v3.7", "Cascade v3.11", "GNCAF v3.62", "GNCAF v3.65", "seg_v2.0 (dual)"]
MODELS_B = MODELS_A + ["HookNet-TLS"]


def load_panel(name: str) -> dict:
    return json.loads((ROOT / f"metrics_panel{name}.json").read_text())


# ───────────────────────── Fig 01 ─────────────────────────────
def fig01_regression():
    df = pd.read_csv(ROOT / "per_slide_predictions.csv")
    pA = load_panel("A")["all"]; pB = load_panel("B")["all"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = [
        (0, 0, df, MODELS_A, "TLS", pA, "Panel A (all 165 slides)"),
        (0, 1, df, MODELS_A, "GC", pA, "Panel A (all 165 slides)"),
        (1, 0, df[df["HookNet-TLS_present"] == 1], MODELS_B, "TLS", pB,
         f"Panel B (HookNet subset, n={pB['n_slides']})"),
        (1, 1, df[df["HookNet-TLS_present"] == 1], MODELS_B, "GC", pB,
         f"Panel B (HookNet subset, n={pB['n_slides']})"),
    ]
    for r, c, dsub, models, cls, panel, title in panels:
        ax = axes[r, c]
        if cls == "TLS":
            gt = dsub["gt_n_tls"].values
        else:
            gt = dsub["gt_n_gc"].values
        max_v = max(2, np.nanmax(gt) + 2)
        for m in models:
            pred_col = f"{m}_n_tls" if cls == "TLS" else f"{m}_n_gc"
            pr = dsub[pred_col].values
            mask = ~np.isnan(pr) & ~np.isnan(gt)
            if mask.sum() < 3: continue
            metr = panel["models"][m]
            if metr is None: continue
            reg = metr["reg_tls" if cls == "TLS" else "reg_gc"]
            r2 = reg["r2"]
            icc_v = reg.get("icc")
            label = f"{m}: R²={r2:.2f}"
            if icc_v is not None:
                label += f", ICC={icc_v:.2f}"
            ax.scatter(gt[mask], pr[mask], s=18, alpha=0.55, color=COLORS[m],
                       label=label, edgecolor="none")
            max_v = max(max_v, np.nanmax(pr[mask]) + 2)
        ax.plot([0, max_v], [0, max_v], "k--", linewidth=0.6, alpha=0.5)
        ax.set_xlim(-1, max_v); ax.set_ylim(-1, max_v)
        ax.set_xlabel(f"GT {cls} count"); ax.set_ylabel(f"Predicted {cls} count")
        ax.set_title(f"{title} — {cls}")
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle("Fig 01 — Slide-level count regression\n"
                 "Per-slide predicted vs GT (HookNet annotated instances). Metrics in legend.",
                 y=0.995, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig01_slide_regression.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig01")


# ───────────────────────── Fig 02/03 ─────────────────────────
def _classification_grid(cls_key: str, fig_num: int, title: str):
    pA = load_panel("A")["all"]; pB = load_panel("B")["all"]
    panels = [("Panel A — all 165 slides", MODELS_A, pA),
              (f"Panel B — HookNet subset (n={pB['n_slides']})", MODELS_B, pB)]
    # Build a grid: for each panel, one row of small CMs; below each CM, P/R/F1 bars
    fig, axes = plt.subplots(2, max(len(MODELS_A), len(MODELS_B)),
                              figsize=(15, 7.5))
    if axes.ndim == 1: axes = axes[None, :]
    n_cls = 3 if "3" in cls_key else 4

    for pi, (panel_title, models, pdata) in enumerate(panels):
        for ci, m in enumerate(models):
            ax = axes[pi, ci]
            md = pdata["models"].get(m)
            if md is None or md.get(cls_key) is None:
                ax.set_visible(False); continue
            cm = np.array(md[cls_key]["confusion_matrix"], dtype=float)
            cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    txt = ax.text(jj, ii, f"{int(cm[ii,jj])}", ha="center",
                                  va="center", fontsize=8,
                                  color="white" if cm_norm[ii,jj] > 0.5 else "black")
                    txt.set_path_effects([patheffects.withStroke(linewidth=1, foreground="white" if cm_norm[ii,jj] > 0.5 else "black", alpha=0)])
            ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
            ax.set_xticklabels(range(n_cls), fontsize=8)
            ax.set_yticklabels(range(n_cls), fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("GT")
            f1m = md[cls_key]["macro"]["f1"]
            f1w = md[cls_key]["weighted"]["f1"]
            ax.set_title(f"{m}\nF1 macro={f1m:.2f}  weighted={f1w:.2f}",
                         fontsize=8)
        # Hide unused columns in panel A
        for ci in range(len(models), axes.shape[1]):
            axes[pi, ci].set_visible(False)
        axes[pi, 0].set_ylabel(f"GT class\n{panel_title}", fontsize=9)

    fig.suptitle(f"Fig {fig_num:02d} — Slide-level {title} confusion matrices", y=1.00, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig{fig_num:02d}_slide_classification_{'3class' if '3' in cls_key else '4class'}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig{fig_num:02d}")


def fig02_3class():
    _classification_grid("cls3", 2, "3-class (tls_count_3class)")


def fig03_4class():
    _classification_grid("cls4", 3, "4-class (tls_count_4class)")


# ───────────────────────── Fig 04 ─────────────────────────
def fig04_per_entity():
    # Use slide-level count F1/precision/recall as proxy for entity-presence detection
    pA = load_panel("A")["all"]; pB = load_panel("B")["all"]
    # Build slide-presence detection P/R/F1 from per-slide preds (gt_n_X > 0 vs pred_n_X > 0)
    df = pd.read_csv(ROOT / "per_slide_predictions.csv")
    def presence_metrics(dsub, m, cls):
        col = f"{m}_n_tls" if cls == "TLS" else f"{m}_n_gc"
        gt = (dsub[f"gt_n_{cls.lower()}"] > 0).astype(int)
        pr = (dsub[col] > 0).astype(int)
        from sklearn.metrics import precision_recall_fscore_support
        valid = ~pd.isna(dsub[col])
        if valid.sum() < 3: return None
        p, r, f1, _ = precision_recall_fscore_support(gt[valid], pr[valid], average="binary", zero_division=0)
        return {"p": float(p), "r": float(r), "f1": float(f1), "n": int(valid.sum())}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_titles = [
        ("Panel A — all 165 slides", MODELS_A, df),
        (f"Panel B — HookNet subset", MODELS_B, df[df["HookNet-TLS_present"]==1]),
    ]
    for pi, (ptitle, models, dsub) in enumerate(panel_titles):
        for ci, cls in enumerate(("TLS", "GC")):
            ax = axes[pi, ci]
            x = np.arange(len(models))
            ps, rs, f1s = [], [], []
            for m in models:
                pm = presence_metrics(dsub, m, cls)
                ps.append(pm["p"] if pm else 0)
                rs.append(pm["r"] if pm else 0)
                f1s.append(pm["f1"] if pm else 0)
            width = 0.27
            ax.bar(x - width, ps, width=width, color="#a6cee3", label="Precision", edgecolor="black", linewidth=0.5)
            ax.bar(x,         rs, width=width, color="#1f78b4", label="Recall", edgecolor="black", linewidth=0.5)
            ax.bar(x + width, f1s, width=width, color="#08306b", label="F1", edgecolor="black", linewidth=0.5)
            for xi, (p, r, f1) in enumerate(zip(ps, rs, f1s)):
                ax.text(xi-width, p+0.02, f"{p:.2f}", ha="center", fontsize=7)
                ax.text(xi,       r+0.02, f"{r:.2f}", ha="center", fontsize=7)
                ax.text(xi+width, f1+0.02, f"{f1:.2f}", ha="center", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.set_title(f"{ptitle} — {cls} slide-presence detection")
            if pi == 0 and ci == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Fig 04 — Slide-level presence detection P/R/F1 (per-class)",
                 y=0.99, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig04_per_entity_presence.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig04")


# ───────────────────────── Fig 05 ─────────────────────────
def fig05_fp_fn_size():
    """Predicted-instance count distribution for FP slides (pred>0, GT=0)
    and FN slides (GT>0, pred=0). Uses per-slide CC counts.
    """
    df = pd.read_csv(ROOT / "per_slide_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ai, cls in enumerate(("TLS", "GC")):
        ax = axes[ai]
        # FP slides: gt=0, pred>0 -> show distribution of pred count
        data = []
        labels = []
        for m in MODELS_A + ["HookNet-TLS"]:
            col = f"{m}_n_{cls.lower()}"
            if col not in df: continue
            gt_col = f"gt_n_{cls.lower()}"
            mask = ~df[col].isna()
            sub = df[mask]
            fp = sub[(sub[gt_col] == 0) & (sub[col] > 0)][col].values
            data.append(fp if len(fp) else np.array([0]))
            labels.append(m)
        pos = np.arange(len(labels))
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                         showfliers=True, medianprops=dict(color="black"))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS.get(labels[i], "#aaa"))
            patch.set_alpha(0.7)
        ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"# predicted {cls} instances per FP slide\n(gt={cls}=0)")
        ax.set_title(f"FP {cls} burden per slide")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Fig 05 — FP burden (predicted-instance count on GT-negative slides)",
                 y=0.99, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig05_fp_fn_size.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig05")


# ───────────────────────── Fig 06 ─────────────────────────
def fig06_per_cancer():
    pA = load_panel("A"); pB = load_panel("B")
    cohorts = ["BLCA", "KIRC", "LUSC"]
    metrics = [("reg_tls", "r2", "TLS R²"), ("reg_gc", "r2", "GC R²"),
               ("reg_tls", "spearman", "TLS Spearman"),
               ("reg_gc", "spearman", "GC Spearman")]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(13, 14))
    for mi, (subkey, key, label) in enumerate(metrics):
        for pi, (panel_data, models, ptitle) in enumerate([
            (pA, MODELS_A, "Panel A (165 slides)"),
            (pB, MODELS_B, f"Panel B (HookNet subset, n={pB['all']['n_slides']})"),
        ]):
            ax = axes[mi, pi]
            heatmap = np.full((len(models), len(cohorts)), np.nan)
            for ri, m in enumerate(models):
                for ci, c in enumerate(cohorts):
                    md = panel_data.get(c, {}).get("models", {}).get(m)
                    if md is None: continue
                    v = md.get(subkey, {}).get(key)
                    if v is not None and np.isfinite(v):
                        heatmap[ri, ci] = max(min(v, 1), -1)
            im = ax.imshow(heatmap, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            for ri in range(len(models)):
                for ci in range(len(cohorts)):
                    v = heatmap[ri, ci]
                    txt = "" if np.isnan(v) else f"{v:.2f}"
                    ax.text(ci, ri, txt, ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(cohorts))); ax.set_xticklabels(cohorts)
            ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
            ax.set_title(f"{label} — {ptitle}")
    fig.suptitle("Fig 06 — Per-cancer-type metric breakdown", y=1.001, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig06_per_cancer.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig06")


# ───────────────────────── Fig 07 ─────────────────────────
def fig07_train_inference_time():
    # Hand-curated training time + measured inference seconds per slide
    df = pd.read_csv(ROOT / "per_slide_predictions.csv")
    train_h = {
        "Cascade v3.7": 5.0,       # v3.8 Stage 1 (~2h) + v3.37 Stage 2 (~3h)
        "Cascade v3.11": 12.0,     # v3.7 + 7h joint fine-tune
        "GNCAF v3.62": 6.0,        # GNCAF training
        "GNCAF v3.65": 6.0,
        "seg_v2.0 (dual)": 4.0,    # seg_v2 training
        "HookNet-TLS": float("nan"),  # external pretrained
    }
    inf_s = {}
    # Pull s_per_slide / t_total from per-slide JSONs if available
    import json as _json
    paths = {
        "Cascade v3.7": "gars_cascade_v3.7_5fold_prodpp_fold0_eval_*",
        "Cascade v3.11": "gars_cascade_v3.11_joint_fold0_eval_*",
        "GNCAF v3.62": "gars_gncaf_eval_v3.62_fullcohort_fold0*shard0*",
        "GNCAF v3.65": "gars_gncaf_eval_v3.65*fullcohort_fold0*shard0*",
        "seg_v2.0 (dual)": "gars_gncaf_eval_seg_v2_dual_fullcohort_fold0_eval_shard0*",
    }
    EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")
    for m, pat in paths.items():
        ds = sorted(EXP.glob(pat))
        if not ds: continue
        d = ds[-1]
        cp = d / "cascade_per_slide.json"
        gp = d / "gncaf_agg.json"
        if cp.exists():
            data = _json.loads(cp.read_text())
            xs = [s.get("t_total") for s in data.get("0.5", []) if s.get("t_total")]
            inf_s[m] = float(np.mean(xs)) if xs else None
        elif gp.exists():
            data = _json.loads(gp.read_text())
            xs = [s.get("t_total") for s in data.get("per_slide", []) if s.get("t_total")]
            inf_s[m] = float(np.mean(xs)) if xs else None
    inf_s["HookNet-TLS"] = 5 * 60.0  # ~5 min/slide measured

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # Training time
    ax = axes[0]
    models = list(train_h.keys())
    vals = [train_h[m] for m in models]
    bars = ax.bar(models, vals, color=[COLORS[m] for m in models], edgecolor="black")
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:.0f} h", ha="center", fontsize=8)
        else:
            ax.text(b.get_x()+b.get_width()/2, 0.5, "external", ha="center", fontsize=8)
    ax.set_ylabel("Training time (GPU-h)"); ax.set_title("Training compute")
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # Inference time
    ax = axes[1]
    ivals = [inf_s.get(m) for m in models]
    bars = ax.bar(models, [v or 0 for v in ivals], color=[COLORS[m] for m in models], edgecolor="black")
    for b, v in zip(bars, ivals):
        if v:
            ax.text(b.get_x()+b.get_width()/2, v+5, f"{v:.0f}s", ha="center", fontsize=8)
        else:
            ax.text(b.get_x()+b.get_width()/2, 1, "n/a", ha="center", fontsize=8)
    ax.set_ylabel("Inference time per slide (s)"); ax.set_title("Inference compute")
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_yscale("symlog", linthresh=10)
    fig.suptitle("Fig 07 — Training & inference compute (per slide)", y=1.02, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"fig07_train_inference_time.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig07")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    fig01_regression()
    fig02_3class()
    fig03_4class()
    fig04_per_entity()
    fig05_fp_fn_size()
    fig06_per_cancer()
    fig07_train_inference_time()


if __name__ == "__main__":
    main()
