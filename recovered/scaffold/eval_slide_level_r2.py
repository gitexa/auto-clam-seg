"""Slide-level R^2 for TLS and GC instance counts — head-to-head:
  HookNet vs our cascade v3.7 vs v3.10 (vs v3.11 / v3.12 / etc as added).

For each model and each slide:
  pred_count = # of predicted TLS or GC instances on the slide
  gt_count   = # of TLS or GC GT instances (from HookNet's masks_instance JSON)

Then compute R^2 (sklearn r2_score) and Spearman/Pearson across slides.

Run:
    python eval_slide_level_r2.py
"""
from __future__ import annotations
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score


INST_DIR = Path("/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/masks_instance")
HOOKNET_OUT = Path("/home/ubuntu/local_data/hooknet_out")
EXP_DIR = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")


def gt_counts(short_id: str) -> tuple[int, int]:
    """Return (n_TLS, n_GC) from HookNet's per-slide instance map."""
    p = INST_DIR / f"{short_id}_instance_map.json"
    if not p.exists():
        return 0, 0
    m = json.loads(p.read_text())
    n_tls = sum(1 for v in m.values() if v.get("class", "").lower().startswith("tls"))
    n_gc = sum(1 for v in m.values() if v.get("class", "").lower().startswith("gc"))
    return n_tls, n_gc


def hooknet_counts(short_id: str) -> tuple[int, int] | None:
    """Return (n_TLS, n_GC) from filtered polygon JSONs.
    Returns None if HookNet output for this slide isn't ready.
    """
    tls_p = HOOKNET_OUT / short_id / "images" / "filtered" / f"{short_id}_hooknettls_tls_filtered.json"
    gc_p = HOOKNET_OUT / short_id / "images" / "filtered" / f"{short_id}_hooknettls_gc_filtered.json"
    if not tls_p.exists() or not gc_p.exists():
        return None
    try:
        tls = json.loads(tls_p.read_text())
        gc = json.loads(gc_p.read_text())
        return len(tls), len(gc)
    except Exception:
        return None


def load_per_slide_json(eval_dir: Path) -> dict[str, dict]:
    p = eval_dir / "cascade_per_slide.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    if "0.5" not in d:
        return {}
    return {s["slide_id"]: s for s in d["0.5"]}


def find_eval_dir(pattern: str) -> Path | None:
    cands = sorted(EXP_DIR.glob(pattern))
    return cands[-1] if cands else None


def compute_r2(gt: np.ndarray, pred: np.ndarray) -> dict:
    if len(gt) < 3 or np.std(gt) == 0:
        return {"n": int(len(gt)), "r2": None, "spearman": None, "pearson": None, "mae": float(np.mean(np.abs(gt - pred)))}
    r2 = float(r2_score(gt, pred))
    sp, _ = spearmanr(gt, pred)
    pe, _ = pearsonr(gt, pred)
    return {"n": int(len(gt)), "r2": r2, "spearman": float(sp), "pearson": float(pe),
            "mae": float(np.mean(np.abs(gt - pred)))}


def collect_slide_pairs(slides: list[str], get_pred) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (gt_tls, gt_gc, pred_tls, pred_gc) numpy arrays."""
    gt_t, gt_g, pr_t, pr_g = [], [], [], []
    for sid in slides:
        gt = gt_counts(sid)
        pr = get_pred(sid)
        if pr is None:
            continue
        gt_t.append(gt[0]); gt_g.append(gt[1])
        pr_t.append(pr[0]); pr_g.append(pr[1])
    return np.array(gt_t), np.array(gt_g), np.array(pr_t), np.array(pr_g)


def main():
    import sys
    sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
    sys.path.insert(0, "/home/ubuntu/profile-clam")
    import prepare_segmentation as ps
    entries = ps.build_slide_entries()
    folds, _ = ps.create_splits(entries, k_folds=5, seed=42)
    val0 = [e["slide_id"].split(".")[0] for e in folds[0]]
    print(f"fold-0 val slides: {len(val0)}")

    # Build pred getters for each model
    def get_v3_7(sid):
        d = find_eval_dir("gars_cascade_v3.7_5fold_prodpp_fold0_eval_*")
        if d is None: return None
        per = load_per_slide_json(d)
        s = per.get(sid)
        if s is None: return None
        return s["n_tls_pred"], s["n_gc_pred"]

    def get_v3_10(sid):
        d = find_eval_dir("gars_cascade_v3.10_hardneg_fold0_eval_*")
        if d is None: return None
        per = load_per_slide_json(d)
        s = per.get(sid)
        if s is None: return None
        return s["n_tls_pred"], s["n_gc_pred"]

    def get_v3_11(sid):
        d = find_eval_dir("gars_cascade_v3.11_joint_fold0_eval_*")
        if d is None: return None
        per = load_per_slide_json(d)
        s = per.get(sid)
        if s is None: return None
        return s["n_tls_pred"], s["n_gc_pred"]

    def get_hooknet(sid):
        return hooknet_counts(sid)

    models = [
        ("Cascade v3.7", get_v3_7, "#1f77b4"),
        ("Cascade v3.10 (hi-prec)", get_v3_10, "#0a3c5c"),
        ("Cascade v3.11 (joint ft)", get_v3_11, "#5d8aaf"),
        ("HookNet-TLS", get_hooknet, "#000000"),
    ]

    summary = {}
    print(f"\n{'model':30s}  {'class':3s}  {'n':>4s}  {'R²':>7s}  {'spearman':>9s}  {'pearson':>8s}  {'MAE':>6s}")
    for name, getter, color in models:
        gt_t, gt_g, pr_t, pr_g = collect_slide_pairs(val0, getter)
        if len(gt_t) == 0:
            print(f"{name:30s}: NO DATA (skip)")
            continue
        tls = compute_r2(gt_t, pr_t)
        gc = compute_r2(gt_g, pr_g)
        summary[name] = {"tls": tls, "gc": gc, "n_slides": len(gt_t)}
        for cls_name, m in (("TLS", tls), ("GC", gc)):
            r2 = m["r2"]; sp = m["spearman"]; pe = m["pearson"]; mae = m["mae"]
            r2s = f"{r2:>7.3f}" if r2 is not None else "    n/a"
            sps = f"{sp:>9.3f}" if sp is not None else "      n/a"
            pes = f"{pe:>8.3f}" if pe is not None else "     n/a"
            print(f"{name:30s}  {cls_name:3s}  {m['n']:>4d}  {r2s}  {sps}  {pes}  {mae:>6.2f}")

    # Save summary JSON
    out_json = Path("/home/ubuntu/auto-clam-seg/slide_level_r2.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    # Scatter plot: pred vs gt per class, all models on one fig
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for i, cls in enumerate(("TLS", "GC")):
        ax = axes[i]
        for name, getter, color in models:
            gt_t, gt_g, pr_t, pr_g = collect_slide_pairs(val0, getter)
            if len(gt_t) == 0: continue
            gt = gt_t if cls == "TLS" else gt_g
            pr = pr_t if cls == "TLS" else pr_g
            r2 = summary.get(name, {}).get(cls.lower(), {}).get("r2")
            label = f"{name} (R²={r2:.2f})" if r2 is not None else name
            ax.scatter(gt, pr, s=14, alpha=0.5, color=color, label=label, edgecolor="none")
        # 1:1 line
        m = max([gt_counts(s)[0 if cls == "TLS" else 1] for s in val0]) + 2
        ax.plot([0, m], [0, m], "k--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel(f"GT {cls} count")
        ax.set_ylabel(f"Predicted {cls} count")
        ax.set_title(f"Slide-level {cls} count regression (fold-0 val)")
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Slide-level instance-count R²: HookNet vs our cascades",
                 y=1.01, fontsize=10)
    fig.tight_layout()
    out_png = Path("/home/ubuntu/auto-clam-seg/notebooks/architectures/fig_slide_r2.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
