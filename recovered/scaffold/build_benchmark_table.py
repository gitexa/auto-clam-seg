"""Aggregate 5-fold CV results across all three approaches into a
publication-ready benchmark table.

Reads:
  - GNCAF v3.58 fold 0..4 from gars_gncaf_eval_v3.58_5fold_fold{K}_eval_shard*
    (combined into v3.58_5fold_fold{K}_combined.json)
  - GNCAF v3.56 fold 0..4 from gars_gncaf_eval_v3.56_5fold_fold{K}_eval_shard*
  - Cascade fold 0..4 from gars_cascade_cascade_5fold_fold{K}_eval_shard*
    (cascade eval CSVs differ in shape — handled separately)
  - seg_v2.0 from /home/ubuntu/ahaas-persistent-std-tcga/experiments/
    seg_v2.0_tls_only_5fold_31aaec0c/summary.json + per-fold incremental.json

Outputs:
  - benchmark_5fold.csv (publication supplement)
  - markdown table block (printed to stdout for paste into summary.md)
  - paired t-test p-values for cascade vs each baseline
"""
from __future__ import annotations

import json
import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

EXP_ROOT = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")


def _safe_mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0)


def _paired_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test on per-fold values. Returns (t-stat, two-sided p)."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    diff = a_arr - b_arr
    n = len(diff)
    mean = diff.mean()
    sd = diff.std(ddof=1) if n > 1 else 0.0
    if sd == 0:
        return float("nan"), float("nan")
    t = mean / (sd / math.sqrt(n))
    # Two-sided p via scipy.stats.t survival
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def collect_gncaf_fold(label_prefix: str) -> dict[str, Any] | None:
    """Read combined.json for a single fold's eval.

    Tries multiple candidate paths because the combiner script names
    differ between fold-0 (legacy) and 5fold runs.
    """
    candidates = [
        EXP_ROOT / f"{label_prefix}_combined.json",
    ]
    for cand in EXP_ROOT.glob(f"{label_prefix}*_combined.json"):
        candidates.append(cand)
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())
    return None


def collect_gncaf_5fold(label_template: str) -> dict[str, list[float]]:
    """Collect per-fold metrics from 5 GNCAF combined.json files.

    label_template like 'v3.58_5fold_fold{K}_eval_shard'.
    Fold 0 may be at the original path (e.g. v3.58_full_eval_shard).
    """
    out = {
        "mDice_pix": [], "tls_dice_pix": [], "gc_dice_pix": [],
        "mDice_grid": [], "tls_dice_grid": [], "gc_dice_grid": [],
        "tls_sp": [], "gc_sp": [], "tls_mae": [], "gc_mae": [],
        "s_per_slide": [], "n_slides": [],
    }
    for k in range(5):
        fold_label = label_template.replace("{K}", str(k))
        d = collect_gncaf_fold(fold_label)
        if d is None:
            print(f"  [warn] {fold_label}_combined.json not found, skipping fold {k}")
            continue
        for key in out:
            if key in d:
                out[key].append(d[key])
    return out


def collect_cascade_fold(fold_idx: int, label_template: str) -> dict[str, float] | None:
    """Cascade eval doesn't produce a single combined.json by default.
    Re-aggregate from the 4 shards' shard-level metric prints.
    """
    # eval_gars_cascade.py prints metrics inline at end of each shard.
    # Look for shard log files.
    out: dict[str, list[float]] = {}
    n_slides_total = 0
    for shard in range(4):
        # The eval writes hydra output dir gars_cascade_<label>_shard<S>_*
        label = label_template.replace("{K}", str(fold_idx)).replace("{S}", str(shard))
        for cand in EXP_ROOT.glob(f"gars_cascade_{label}_*"):
            log_path = next(cand.glob("*.log"), None)
            # Or use the file we tee'd to /tmp
            tmp_log = Path(f"/tmp/cascade_fold{fold_idx}_eval_shard{shard}.log")
            if tmp_log.exists():
                log_path = tmp_log
            if log_path is None or not log_path.exists():
                continue
            text = log_path.read_text()
            import re
            m = re.search(
                r"thr\s+mDice.*?\n\s*0\.50\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+([\d.]+)%",
                text, re.S,
            )
            n_match = re.search(r"Val:\s+(\d+)\s+slides", text)
            if not m:
                continue
            shard_metrics = {
                "mDice_pix": float(m.group(1)),
                "tls_dice_pix": float(m.group(2)),
                "gc_dice_pix": float(m.group(3)),
                "tls_sp": float(m.group(4)) if m.group(4) != "nan" else float("nan"),
                "gc_sp": float(m.group(5)) if m.group(5) != "nan" else float("nan"),
                "sel_pct": float(m.group(6)),
            }
            n = int(n_match.group(1)) if n_match else 0
            for key, v in shard_metrics.items():
                out.setdefault(key, []).append((v, n))
            n_slides_total += n
            break
    if not out or n_slides_total == 0:
        return None
    # Weighted by slide count per shard
    agg = {}
    for k, pairs in out.items():
        total_n = sum(n for _, n in pairs)
        if total_n == 0:
            agg[k] = float("nan")
        else:
            agg[k] = sum(v * n for v, n in pairs) / total_n
    agg["n_slides"] = n_slides_total
    return agg


def collect_cascade_5fold(label_template: str) -> dict[str, list[float]]:
    out = {
        "mDice_pix": [], "tls_dice_pix": [], "gc_dice_pix": [],
        "tls_sp": [], "gc_sp": [], "n_slides": [],
    }
    # Fold 0 is the existing v3.37 RegionDecoder cascade — known values
    # baked in from the apples-to-apples run on the same fold-0 124-slide
    # val. Source: summary.md 'champion' row + per-shard outputs.
    out["mDice_pix"].append(0.845)
    out["tls_dice_pix"].append(0.822)
    out["gc_dice_pix"].append(0.868)
    out["tls_sp"].append(0.876)
    out["gc_sp"].append(0.934)
    out["n_slides"].append(124)
    print("  cascade fold 0: using known v3.37 RegionDecoder values "
          "(mDice_pix=0.845, TLS=0.822, GC=0.868)")
    # Folds 1-4 from new chain
    for k in range(1, 5):
        d = collect_cascade_fold(k, label_template)
        if d is None:
            print(f"  [warn] cascade fold {k} not found, skipping")
            continue
        for key in out:
            if key in d:
                out[key].append(d[key])
    return out


def collect_seg_v2(seg_dir: Path) -> dict[str, list[float]]:
    """Per-fold metrics from seg_v2.0 summary.

    seg_v2.0 produces TLS pixel dice + TLS counting + GC count-based
    detection (no GC pixel dice — uses centroid heads). Fields available:
      val_metrics.dice           → tls_dice_pix
      val_metrics.count_spearman → tls_sp
      val_metrics.gc_count_spearman → gc_sp (counting only)
      val_metrics.gc_det_balanced_acc → gc detection acc

    GC pix dice: not available; report n/a in benchmark table for fairness.
    """
    summary = json.loads((seg_dir / "summary.json").read_text())
    folds = summary.get("folds", [])
    out = {
        "mDice_pix": [], "tls_dice_pix": [], "gc_dice_pix": [],
        "tls_sp": [], "gc_sp": [], "tls_mae": [], "gc_mae": [],
    }
    for f in folds:
        vm = f.get("val_metrics", {})
        if "dice" in vm:
            out["tls_dice_pix"].append(vm["dice"])
            # mDice for seg_v2.0 = TLS dice only (no GC pix)
            out["mDice_pix"].append(vm["dice"])
        if "count_spearman" in vm:
            out["tls_sp"].append(vm["count_spearman"])
        if "gc_count_spearman" in vm:
            out["gc_sp"].append(vm["gc_count_spearman"])
    return out


def _find_combined_json(name_patterns: list[str]) -> Path | None:
    """Try multiple candidate locations for a fold's combined.json."""
    for pat in name_patterns:
        for cand in EXP_ROOT.glob(pat):
            return cand
    return None


def fmt_mean_std(xs: list[float], decimals: int = 3) -> str:
    if not xs:
        return "n/a"
    m, s = _safe_mean_std(xs)
    if math.isnan(m):
        return "n/a"
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def main():
    print("===== 5-fold CV benchmark aggregator =====")
    print()

    # Cascade — uses both fold-0 from existing v3.37 + new fold 1-4
    cascade = collect_cascade_5fold("cascade_5fold_fold{K}_eval_shard{S}")

    # GNCAF v3.58 fold 0 lives at v3.58_combined.json (existing); fold 1-4
    # at v3.58_5fold_fold{K}_combined.json
    v358 = {k: [] for k in [
        "mDice_pix", "tls_dice_pix", "gc_dice_pix",
        "mDice_grid", "tls_dice_grid", "gc_dice_grid",
        "tls_sp", "gc_sp", "tls_mae", "gc_mae",
        "s_per_slide", "n_slides",
    ]}
    # Fold 0
    d0 = collect_gncaf_fold("v3.58")
    if d0 is not None:
        for k in v358:
            if k in d0 and isinstance(d0[k], (int, float)):
                v358[k].append(d0[k])
    # Folds 1..4
    for fold in range(1, 5):
        d = collect_gncaf_fold(f"v3.58_5fold_fold{fold}_eval_shard")
        if d is None:
            continue
        for k in v358:
            if k in d and isinstance(d[k], (int, float)):
                v358[k].append(d[k])

    # GNCAF v3.56 same pattern
    v356 = {k: [] for k in v358}
    d0 = collect_gncaf_fold("v3.56")
    if d0 is not None:
        for k in v356:
            if k in d0 and isinstance(d0[k], (int, float)):
                v356[k].append(d0[k])
    for fold in range(1, 5):
        d = collect_gncaf_fold(f"v3.56_5fold_fold{fold}_eval_shard")
        if d is None:
            continue
        for k in v356:
            if k in d and isinstance(d[k], (int, float)):
                v356[k].append(d[k])

    seg_v2 = collect_seg_v2(EXP_ROOT / "seg_v2.0_tls_only_5fold_31aaec0c")

    print(f"  cascade: {len(cascade['mDice_pix'])} folds collected")
    print(f"  GNCAF v3.58 (12L+aug): {len(v358['mDice_pix'])} folds collected")
    print(f"  GNCAF v3.56 (6L unfrozen): {len(v356['mDice_pix'])} folds collected")
    print(f"  seg_v2.0: {len(seg_v2['mDice_pix'])} folds collected")
    print()

    # ── Markdown table ───────────────────────────────
    print("## 5-fold CV benchmark (mean ± std across folds 0..4)")
    print()
    headers = ["Approach", "Params", "mDice_pix", "TLS pix", "GC pix",
               "TLS sp", "GC sp", "n folds"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")

    rows = [
        ("seg_v2.0 (no graph)", "~4M", seg_v2),
        ("GNCAF v3.58 (12L+aug)", "108M", v358),
        ("GNCAF v3.56 (6L unfrozen)", "65M", v356),
        ("**Cascade (Stage 1+2)**", "14.5M", cascade),
    ]
    for name, params, m in rows:
        cells = [
            name, params,
            fmt_mean_std(m.get("mDice_pix", []), 3),
            fmt_mean_std(m.get("tls_dice_pix", []), 3),
            fmt_mean_std(m.get("gc_dice_pix", []), 3),
            fmt_mean_std(m.get("tls_sp", []), 3),
            fmt_mean_std(m.get("gc_sp", []), 3),
            str(len(m.get("mDice_pix", []))),
        ]
        print("| " + " | ".join(cells) + " |")

    # ── Paired t-tests (cascade as reference) ────────
    print()
    print("### Paired t-tests vs cascade (per-fold mDice_pix)")
    for name, params, m in rows[:-1]:  # excluding cascade itself
        n_cascade = len(cascade["mDice_pix"])
        n_other = len(m["mDice_pix"])
        n = min(n_cascade, n_other)
        if n < 2:
            print(f"  {name}: insufficient folds (n={n})")
            continue
        t, p = _paired_t(cascade["mDice_pix"][:n], m["mDice_pix"][:n])
        print(f"  cascade vs {name}: t={t:.3f}, p={p:.4f} (n={n})")

    # ── Write CSV supplement ─────────────────────────
    out_csv = Path("/home/ubuntu/auto-clam-seg/benchmark_5fold.csv")
    import csv
    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["approach", "fold", "mDice_pix", "tls_dice_pix",
                    "gc_dice_pix", "tls_sp", "gc_sp", "tls_mae", "gc_mae",
                    "s_per_slide", "n_slides"])
        for name, params, m in rows:
            short = name.replace("**", "").split(" (")[0].strip()
            n_folds = len(m.get("mDice_pix", []))
            for k in range(n_folds):
                w.writerow([
                    short, k,
                    m.get("mDice_pix", [None] * n_folds)[k] if k < len(m.get("mDice_pix", [])) else "",
                    m.get("tls_dice_pix", [None] * n_folds)[k] if k < len(m.get("tls_dice_pix", [])) else "",
                    m.get("gc_dice_pix", [None] * n_folds)[k] if k < len(m.get("gc_dice_pix", [])) else "",
                    m.get("tls_sp", [None] * n_folds)[k] if k < len(m.get("tls_sp", [])) else "",
                    m.get("gc_sp", [None] * n_folds)[k] if k < len(m.get("gc_sp", [])) else "",
                    m.get("tls_mae", [None] * n_folds)[k] if k < len(m.get("tls_mae", [])) else "",
                    m.get("gc_mae", [None] * n_folds)[k] if k < len(m.get("gc_mae", [])) else "",
                    m.get("s_per_slide", [None] * n_folds)[k] if k < len(m.get("s_per_slide", [])) else "",
                    m.get("n_slides", [None] * n_folds)[k] if k < len(m.get("n_slides", [])) else "",
                ])
    print(f"\nWrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
