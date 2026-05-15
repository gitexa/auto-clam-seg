"""Collect per-slide predictions from each of the 6 models and compute
the comprehensive metric panel for the Nature-Medicine-quality figures.

Models:
  Cascade v3.7  (non-fine-tuned)
  Cascade v3.11 (joint fine-tune)
  GNCAF v3.62   (paper-strict)
  GNCAF v3.65   (dual-sigmoid)
  seg_v2.0      (dual)
  HookNet-TLS

Scopes:
  Panel A = 165 fold-0 val slides (our 5 models)
  Panel B = subset of slides HookNet completed (all 6 models)

Output:
  /home/ubuntu/auto-clam-seg/figures_naturemed/summary_table.csv
  /home/ubuntu/auto-clam-seg/figures_naturemed/per_slide_predictions.json
"""
from __future__ import annotations
import json
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")

import prepare_segmentation as ps  # noqa: E402

ROOT = Path("/home/ubuntu/auto-clam-seg/figures_naturemed")
ROOT.mkdir(parents=True, exist_ok=True)

EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")
HOOKNET_OUT = Path("/home/ubuntu/local_data/hooknet_out")
INST_DIR = Path("/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/masks_instance")
DF_PATH = "/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/clam_training/df_summary_v10.csv"

MODELS = [
    "Cascade v3.7",
    "Cascade v3.11",
    "GNCAF v3.62",
    "GNCAF v3.65",
    "seg_v2.0 (dual)",
    "HookNet-TLS",
]


def _latest(pattern: str) -> Path | None:
    cands = sorted(EXP.glob(pattern))
    return cands[-1] if cands else None


def _load_cascade(pattern: str) -> dict:
    d = _latest(pattern)
    if d is None:
        return {}
    p = d / "cascade_per_slide.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {s["slide_id"]: s for s in data.get("0.5", [])}


def _load_gncaf_shards(pattern: str) -> dict:
    """Pool per_slide entries across 4 shards."""
    out = {}
    for d in sorted(EXP.glob(pattern)):
        p = d / "gncaf_agg.json"
        if not p.exists():
            continue
        agg = json.loads(p.read_text())
        for r in agg.get("per_slide", []):
            out[r["slide_id"]] = r
    return out


def _load_hooknet(sid: str) -> dict | None:
    tls = HOOKNET_OUT / sid / "images" / "filtered" / f"{sid}_hooknettls_tls_filtered.json"
    gc = HOOKNET_OUT / sid / "images" / "filtered" / f"{sid}_hooknettls_gc_filtered.json"
    if not tls.exists() or not gc.exists():
        return None
    try:
        return {
            "n_tls_pred": len(json.loads(tls.read_text())),
            "n_gc_pred": len(json.loads(gc.read_text())),
        }
    except Exception:
        return None


def _gt_counts(sid: str) -> tuple[int, int]:
    p = INST_DIR / f"{sid}_instance_map.json"
    if not p.exists():
        return 0, 0
    m = json.loads(p.read_text())
    n_t = sum(1 for v in m.values() if v.get("class", "").lower().startswith("tls"))
    n_g = sum(1 for v in m.values() if v.get("class", "").lower().startswith("gc"))
    return n_t, n_g


def build_per_slide() -> pd.DataFrame:
    entries = ps.build_slide_entries()
    folds, _ = ps.create_splits(entries, k_folds=5, seed=42)
    val0 = folds[0]
    val_short = [e["slide_id"].split(".")[0] for e in val0]
    val_full = [e["slide_id"] for e in val0]
    val_ct = {e["slide_id"].split(".")[0]: e["cancer_type"] for e in val0}

    # Slide-level class labels from cohort metadata
    df_meta = pd.read_csv(DF_PATH)
    meta = df_meta.set_index("slide_id_short")
    tls3 = meta["tls_count_3class"].to_dict()
    tls4 = meta["tls_count_4class"].to_dict()

    # Per-model predictions
    print("Loading per-model predictions...")
    preds = {
        "Cascade v3.7":   _load_cascade("gars_cascade_v3.7_5fold_prodpp_fold0_eval_*"),
        "Cascade v3.11":  _load_cascade("gars_cascade_v3.11_joint_fold0_eval_*"),
        "GNCAF v3.62":    _load_gncaf_shards("gars_gncaf_eval_v3.62_fullcohort_fold0*shard*"),
        "GNCAF v3.65":    _load_gncaf_shards("gars_gncaf_eval_v3.65*fullcohort_fold0*shard*"),
        "seg_v2.0 (dual)": _load_gncaf_shards("gars_gncaf_eval_seg_v2_dual_fullcohort_fold0_eval_shard*"),
    }
    for m, d in preds.items():
        print(f"  {m}: {len(d)} slides")

    rows = []
    for sid in val_short:
        gt_t, gt_g = _gt_counts(sid)
        ct = val_ct.get(sid, "")
        row = {
            "slide_id": sid,
            "cancer_type": ct,
            "gt_n_tls": gt_t,
            "gt_n_gc": gt_g,
            "tls_count_3class": tls3.get(sid),
            "tls_count_4class": tls4.get(sid),
        }
        for m in MODELS:
            if m == "HookNet-TLS":
                hk = _load_hooknet(sid)
                row[f"{m}_n_tls"] = hk["n_tls_pred"] if hk else None
                row[f"{m}_n_gc"] = hk["n_gc_pred"] if hk else None
                row[f"{m}_present"] = 1 if hk is not None else 0
            else:
                s = preds.get(m, {}).get(sid)
                row[f"{m}_n_tls"] = s.get("n_tls_pred") if s else None
                row[f"{m}_n_gc"] = s.get("n_gc_pred") if s else None
                row[f"{m}_present"] = 1 if s is not None else 0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = build_per_slide()
    out_csv = ROOT / "per_slide_predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {len(df)} slides x {len(df.columns)} cols")
    print(df[[c for c in df.columns if "_present" in c]].sum())


if __name__ == "__main__":
    main()
