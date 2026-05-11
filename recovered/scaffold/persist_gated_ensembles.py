"""Persist cascade-Stage-1-gated ensembles as per-slide JSON files so they
appear in build_arch_comparison.py / fig_one_summary as new columns.

Writes to:
  experiments/gars_gncaf_eval_{label}_gated_fullcohort_fold0_eval_shard0/gncaf_agg.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_v365_cascade_gate import (
    load_cascade_gate,
    ensemble,
)

EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")


def collect(pattern: str) -> list[dict]:
    rows = []
    for d in sorted(EXP.glob(pattern)):
        p = d / "gncaf_agg.json"
        if p.exists():
            rows.extend(json.loads(p.read_text())["per_slide"])
    return rows


def write_gated(label_out: str, rows: list[dict], fold: int = 0):
    out_dir = EXP / f"gars_gncaf_eval_{label_out}_fullcohort_fold{fold}_eval_shard0"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "approach": label_out,
        "fold": fold,
        "n_slides": len(rows),
        "n_gt_pos": sum(1 for r in rows if not r.get("gt_negative")),
        "n_gt_neg": sum(1 for r in rows if r.get("gt_negative")),
        "per_slide": rows,
        "gated_by": "cascade_stage1_v3.8 (n_selected>0 at threshold 0.5)",
    }
    (out_dir / "gncaf_agg.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_dir / 'gncaf_agg.json'} ({len(rows)} slides)")


def main():
    fold = 0
    cascade_rows = load_cascade_gate(fold=fold)
    print(f"Loaded {len(cascade_rows)} cascade-gate rows (fold {fold})\n")

    targets = [
        ("v3.65_gated",          f"gars_gncaf_eval_v3.65_fullcohort_fold{fold}_shard*"),
        ("v3.63_gated",          f"gars_gncaf_eval_v3.63_fullcohort_fold{fold}_shard*"),
        ("seg_v2_dual_gated",    f"gars_gncaf_eval_seg_v2_dual_fullcohort_fold{fold}_eval_shard*"),
        ("seg_v2_gated",         f"gars_gncaf_eval_seg_v2_fullcohort_fold{fold}_eval_shard*"),
    ]
    for label_out, pat in targets:
        src = collect(pat)
        if not src:
            print(f"  {label_out}: no source rows; skip")
            continue
        gated = ensemble(src, cascade_rows)
        write_gated(label_out, gated, fold=fold)


if __name__ == "__main__":
    main()
