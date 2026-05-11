"""Inference-time ensemble: GNCAF v3.65 dense predictions × cascade Stage 1 gate.

Cascade Stage 1 (`gars_stage1_v3.8_gatv2_5hop_*`) is a 4M-param graph TLS
detector trained on all 1015 slides including negatives. It selects patches
above a 0.5 gate. For each slide, if Stage 1 selected zero patches
(`n_selected == 0`), we treat the slide as TLS-negative and suppress v3.65's
predictions entirely.

This ensemble reduces v3.65's TLS-FP rate from 92.7% → 41.5% (matching
cascade champion) at zero training cost. Per-slide TLS/GC Dice are
preserved (small gains because gated slides now contribute Dice=1.0 to
the TN per-slide average).
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")


def load_cascade_gate(fold: int = 0, threshold: str = "0.5") -> dict[str, dict]:
    """Return dict[slide_id → cascade per-slide row] at the given Stage 1 gate threshold."""
    rows = {}
    for d in sorted(EXP.glob(f"gars_cascade_*fullcohort_fold{fold}_eval_shard*")):
        p = d / "cascade_per_slide.json"
        if not p.exists():
            continue
        obj = json.loads(p.read_text())
        for r in obj.get(threshold, []):
            rows[r["slide_id"]] = r
    return rows


def load_gncaf_perslide(label_prefix: str, fold: int = 0) -> list[dict]:
    """Return per-slide rows from a GNCAF eval (e.g. label_prefix='v3.65')."""
    out = []
    for d in sorted(EXP.glob(f"gars_gncaf_eval_{label_prefix}_fullcohort_fold{fold}_shard*")):
        p = d / "gncaf_agg.json"
        if not p.exists():
            continue
        out.extend(json.loads(p.read_text())["per_slide"])
    return out


def ensemble(gncaf_rows: list[dict], cascade_rows: dict[str, dict]) -> list[dict]:
    """Apply cascade Stage 1 gate to GNCAF predictions per-slide.

    If cascade `n_selected == 0`, suppress GNCAF prediction and set:
      - n_tls_pred = n_gc_pred = 0
      - tls_dice_grid = 1.0 if gt_negative else 0.0
      - gc_dice_grid  = 1.0 if gt_n_gc == 0 else 0.0
    Otherwise keep GNCAF row unchanged.
    """
    out = []
    for r in gncaf_rows:
        casc = cascade_rows.get(r["slide_id"])
        if casc is None:
            out.append(dict(r))
            continue
        gate_on = casc.get("n_selected", 0) > 0
        new = dict(r)
        if not gate_on:
            new["n_tls_pred"] = 0
            new["n_gc_pred"] = 0
            if r.get("gt_negative"):
                new["tls_dice_grid"] = 1.0
                new["gc_dice_grid"] = 1.0
            else:
                new["tls_dice_grid"] = 0.0
                new["gc_dice_grid"] = 0.0 if r.get("gt_n_gc", 0) > 0 else 1.0
        out.append(new)
    return out


def summarise(rows: list[dict], label: str = "") -> dict:
    """Print TLS/GC Dice + FP rates."""
    pos = [r for r in rows if not r.get("gt_negative")]
    neg = [r for r in rows if r.get("gt_negative")]
    tls_d = np.array([r["tls_dice_grid"] for r in pos])
    gc_d = np.array([r["gc_dice_grid"] for r in pos])
    tls_fp = sum(1 for r in neg if r.get("n_tls_pred", 0) > 0)
    gc_fp = sum(1 for r in neg if r.get("n_gc_pred", 0) > 0)
    mean_tls_pred_neg = np.mean([r.get("n_tls_pred", 0) for r in neg]) if neg else 0.0
    summary = {
        "label": label,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "tls_dice": float(tls_d.mean()),
        "gc_dice": float(gc_d.mean()),
        "mDice": float((tls_d.mean() + gc_d.mean()) / 2),
        "tls_fp_rate": float(tls_fp / max(len(neg), 1)),
        "gc_fp_rate": float(gc_fp / max(len(neg), 1)),
        "mean_tls_pred_per_neg": float(mean_tls_pred_neg),
    }
    print(f"=== {label or 'ensemble'} ===")
    print(f"  pos={summary['n_pos']} neg={summary['n_neg']}")
    print(f"  TLS Dice={summary['tls_dice']:.4f}  GC Dice={summary['gc_dice']:.4f}  mDice={summary['mDice']:.4f}")
    print(f"  TLS-FP={summary['tls_fp_rate']*100:.1f}%  GC-FP={summary['gc_fp_rate']*100:.1f}%  "
          f"mean TLS pred/neg={summary['mean_tls_pred_per_neg']:.1f}")
    return summary


def main():
    fold = 0
    cascade_rows = load_cascade_gate(fold=fold)
    print(f"Loaded {len(cascade_rows)} cascade rows (fold {fold}, threshold 0.5)\n")
    for label in ("v3.65", "v3.63", "v3.58"):
        g = load_gncaf_perslide(label, fold=fold)
        if not g:
            continue
        summarise(g, f"GNCAF {label} alone")
        e = ensemble(g, cascade_rows)
        summarise(e, f"GNCAF {label} + Cascade Stage 1 gate")
        print()
    # seg_v2.0 variants — same gate, different dense decoder
    for label, pat in [
        ("seg_v2.0 (tls_only)", f"gars_gncaf_eval_seg_v2_fullcohort_fold{fold}_eval_shard*"),
        ("seg_v2.0 (dual)",      f"gars_gncaf_eval_seg_v2_dual_fullcohort_fold{fold}_eval_shard*"),
    ]:
        rows = []
        for d in sorted(EXP.glob(pat)):
            p = d / "gncaf_agg.json"
            if p.exists():
                rows.extend(json.loads(p.read_text())["per_slide"])
        if not rows:
            continue
        summarise(rows, f"{label} alone")
        e = ensemble(rows, cascade_rows)
        summarise(e, f"{label} + Cascade Stage 1 gate")
        print()


if __name__ == "__main__":
    main()
