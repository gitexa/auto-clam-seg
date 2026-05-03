"""Combine 4 sharded GNCAF eval runs into one aggregate result.

Reads gncaf_agg.json from each shard's run dir, sums the
intersection/denominator counters, recomputes pixel-aggregate dice,
re-runs Spearman/MAE on the union of per-slide rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_prefix", default="v3.35_full_eval_shard",
                    help="run label prefix (will glob shard0..shardN)")
    ap.add_argument("--out", default=None,
                    help="output JSON; default = combined.json next to first shard")
    args = ap.parse_args()

    dirs = sorted(EXP.glob(f"gars_gncaf_eval_{args.label_prefix}*"))
    if not dirs:
        raise SystemExit(f"No shards under {EXP} matching {args.label_prefix}*")
    print(f"Found {len(dirs)} shard dirs:")
    for d in dirs:
        print(f"  {d}")

    inter_t = denom_t = inter_g = denom_g = 0
    per_slide_all = []
    s_per_slide_total = 0.0
    n_total_total = 0
    for d in dirs:
        agg_p = d / "gncaf_agg.json"
        if not agg_p.exists():
            print(f"  SKIP {d.name}: no gncaf_agg.json")
            continue
        agg = json.loads(agg_p.read_text())
        inter_t += agg["tls_inter"]; denom_t += agg["tls_denom"]
        inter_g += agg["gc_inter"]; denom_g += agg["gc_denom"]
        per_slide_all.extend(agg["per_slide"])

    if not per_slide_all:
        raise SystemExit("No per-slide data combined; aborting.")

    # Drop duplicates by slide_id (just in case shards overlap).
    seen = {}
    for r in per_slide_all:
        seen[r["slide_id"]] = r
    per_slide_all = list(seen.values())
    print(f"\nCombined {len(per_slide_all)} unique slides")

    eps = 1e-6
    tls_pix = (2 * inter_t + eps) / (denom_t + eps)
    gc_pix = (2 * inter_g + eps) / (denom_g + eps)
    mD_pix = (tls_pix + gc_pix) / 2.0

    tls_grid_m = float(np.mean([r["tls_dice_grid"] for r in per_slide_all]))
    gc_grid_m = float(np.mean([r["gc_dice_grid"] for r in per_slide_all]))
    mD_grid = (tls_grid_m + gc_grid_m) / 2.0

    gt_t = [r["gt_n_tls"] for r in per_slide_all]
    gt_g = [r["gt_n_gc"] for r in per_slide_all]
    pr_t = [r["n_tls_pred"] for r in per_slide_all]
    pr_g = [r["n_gc_pred"] for r in per_slide_all]
    tls_sp, _ = spearmanr(gt_t, pr_t)
    gc_sp, _ = spearmanr(gt_g, pr_g) if any(gt_g) else (0.0, 0.0)
    tls_mae = float(mean_absolute_error(gt_t, pr_t))
    gc_mae = float(mean_absolute_error(gt_g, pr_g))
    s_per = float(np.mean([r["t_total"] for r in per_slide_all]))
    n_total = int(np.sum([r["n_total"] for r in per_slide_all]))

    print(f"\nResults ({len(per_slide_all)} slides combined):")
    print(f"  [patch-grid] mDice={mD_grid:.4f}  TLS={tls_grid_m:.4f}  GC={gc_grid_m:.4f}")
    print(f"  [pixel-agg]  mDice={mD_pix:.4f}  TLS={tls_pix:.4f}  GC={gc_pix:.4f}")
    print(f"  [counts]  TLS sp={tls_sp:.3f} mae={tls_mae:.2f}  GC sp={gc_sp:.3f} mae={gc_mae:.2f}")
    print(f"  {s_per:.1f}s/slide (per shard), {n_total:,} patches total")

    out = {
        "n_slides": len(per_slide_all),
        "n_patches_total": n_total,
        "mDice_grid": mD_grid, "tls_dice_grid": tls_grid_m, "gc_dice_grid": gc_grid_m,
        "mDice_pix": mD_pix, "tls_dice_pix": tls_pix, "gc_dice_pix": gc_pix,
        "tls_sp": tls_sp, "gc_sp": gc_sp,
        "tls_mae": tls_mae, "gc_mae": gc_mae,
        "s_per_slide": s_per,
        "shards_combined": len(dirs),
    }
    if args.out:
        out_path = Path(args.out)
    else:
        # Use the label prefix to derive the combined filename so each
        # GNCAF retrain (v3.21, v3.35, v3.40, …) doesn't overwrite the
        # previous combined.json.
        # e.g. "v3.40_full_eval_shard" → "v3.40_combined.json"
        prefix = args.label_prefix.rstrip("_").rstrip("shard").rstrip("_")
        # If prefix ends with "_full_eval", strip that too for cleaner name.
        for suffix in ("_full_eval", "_eval"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break
        out_path = dirs[0].parent / f"{prefix}_combined.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
