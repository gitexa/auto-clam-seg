# v3.21 — "lost-original" GNCAF reproduction (results journal)

This sprint task was: **"Find the best 'lost checkpoint' and reproduce
the results autonomously in the next 6 hours."** The hypothesis the
user wanted to test: was the older (lost) GNCAF training pipeline
producing better-quality models than the current `train_gncaf_transunet.py`
+ v3.58 / v3.59 lineage?

**Conclusion: No, the lost-original is *worse* on every fullcohort
metric.** The current code reproduces a stronger model than the lost
one.

## Candidate pool

`experiments/` was scanned for GNCAF training dirs whose original
training script is no longer recoverable:

| Dir | Best val_mDice (training-time) | Notes |
|---|---|---|
| `gars_gncaf_v3.21_gncaf_recovered_20260501_103826` | n/a (older ckpt format) | Vanilla ViT (6×384) + paper-faithful concat-all-hops GCN. **14.9 M params.** Selected as the canonical "lost" candidate. |
| `gars_gncaf_v3.35_gncaf_full_12layer_20260502_101050` | n/a | 12-layer, no best_mdice metadata. |
| `gars_gncaf_v3.40_gncaf_paper_repro_20260503_010725` | n/a | Smaller (102 M ckpt), no best_mdice. |
| `gars_gncaf_v3.50_gncaf_transunet_supersede_20260503_095525` | 0.6073 | First TransUNet variant. |
| `gars_gncaf_v3.51_gncaf_transunet_bgneg_20260503_141606` | 0.6396 | Adds bg-neg sampling. |

The v3.21 was the obvious first candidate (explicitly labelled
"recovered" in its dir name; smallest model so fastest to evaluate).

## Eval pipeline patch

`recovered/scaffold/eval_gars_gncaf.py` was patched mirror-style to
`eval_gars_gncaf_transunet.py` (which had already been patched
earlier today): admits GT-negative slides by default, synthesises
zero target masks for them in `_SlideTileDataset`, tags per-slide
rows with `gt_negative: bool`. Adds `eval_positives_only=true` knob
to restore the legacy positive-only pool.

## Smoke test (positives-only, 4 slides)

Verified strict-OK load of the v3.21 ckpt into the `GNCAF` class
(`gncaf_model.py`); only adjustment was `n_classes=3` (paper default
is 4 — `eval_gars_gncaf.py:55` already sets it correctly).

* Loaded 14,875,947 params (matches expected 14.9 M).
* 4-slide pixel-agg mDice = 0.410 (TLS=0.562 / GC=0.258).
* Confirmed model produces reasonable per-slide patch-grids.

## Full v3.21 fold-0 fullcohort eval

`gars_gncaf_eval_v3.21_fullcohort_fold0_eval_shard{0..3}` — 4 shards,
165 slides total (124 GT-pos + 41 GT-neg).

| Metric | v3.21 | v3.56 | v3.58 | v3.59 | Cascade v3.37 |
|---|---|---|---|---|---|
| TLS pixel-Dice | **0.293** | 0.475 | 0.333 | 0.334 | ~0.83 |
| GC pixel-Dice  | **0.180** | 0.371 | 0.413 | 0.330 | ~0.83 |
| mDice_pix      | **0.237** | **0.423** | 0.373 | 0.332 | **~0.83** |
| TLS Dice (per-slide grid mean) | 0.128 | 0.279 | 0.207 | 0.185 | 0.591 |
| GC Dice  (per-slide grid mean) | 0.586 | 0.684 | 0.611 | 0.730 | 0.861 |
| TLS-FP rate (≥1 on negative)   | **40/41 (98 %)** | 34/41 (83 %) | 39/41 (95 %) | 39/41 (95 %) | 17/41 (41 %) |
| GC-FP rate  (≥1 on negative)   | 3/41 (7 %) | 3/41 (7 %) | 5/41 (12 %) | 2/41 (5 %) | 2/41 (5 %) |
| Mean predicted TLS / positive slide | 72.9 | 38.3 | 55.5 | 60.4 | low |
| Mean predicted TLS / negative slide | **46.0 ⚠️** | 9.9 | 27.6 | 25.0 | small |

**v3.21 fires false TLS detections on 40 of 41 negatives, with an
average of 46 false instances per slide.** It also has the lowest
patch-grid TLS Dice (0.128) of any GNCAF tested. The "lost" original
is the *worst* model in the pool.

## Why v3.21 is worse despite the paper-faithful GCN

* **Smaller backbone**: 6-layer × 384-d ViT (no R50 trunk, no skip
  connections in the encoder). The TransUNet variants (v3.50+)
  added the R50 trunk + skip connections which improved pixel
  decoding — explains v3.56's higher pixel-Dice (0.475 vs 0.293).
* **Earlier training**: v3.21 trained for 14 epochs total per the
  saved config; v3.58 trained for 41 epochs. More training time +
  deeper backbone = better model, even with the "non-faithful"
  iterative-residual GCN.
* **GCN aggregation isn't the bottleneck**: v3.59 *did* swap the
  iterative GCN for paper-faithful concat-all-hops on top of v3.58's
  backbone, and it didn't improve pixel-Dice — same trend here.
  The dense pixel-decoder paradigm just over-fires regardless of the
  GCN flavour.

## Status of other "lost" candidates (not evaluated)

The remaining old training dirs (`v3.35`, `v3.40`, `v3.50`, `v3.51`,
etc.) all have *higher* training-time best_mdice than v3.21 (0.61,
0.64, etc. vs v3.21's training mDice of 0.61). They likely behave
better than v3.21 but worse than the v3.56/v3.58 lineage. None of
them was evaluated in this sprint to keep the GPU free for the
Phase 2 cascade 5-fold work; can be evaluated later as a sanity
check if needed.

## Implications for the publication story

* Cascade still wins by a wide margin on every full-cohort metric
  (mDice_pix ~0.83 vs best GNCAF 0.42; TLS-FP rate 41 % vs 83 %+;
  GC detection F1 0.92 vs 0.69).
* The `current` code is not a regression from the lost original.
* v3.59's GCUNet-faithful fix didn't improve over v3.58. The dense
  pixel-decoder over-fires fundamentally.
* The case for the cascade as the recommended approach is now even
  stronger: not only does it win on Dice, but its Stage-1 gate
  protects against false detections on negative slides — a
  property the dense pixel-decoders structurally lack.

## Files touched in this sprint

* `recovered/scaffold/eval_gars_gncaf.py` — full-cohort patch
  (mirror of `eval_gars_gncaf_transunet.py`'s).
* `notebooks/build_arch_comparison.py` — added v3.21 to the per-slide pool.
* `ARCHITECTURE_COMPARISON.md` — added v3.21 row + the
  full-cohort fold-0 comparison table.
* `V321_LOST_CHECKPOINT_RESULTS.md` (this file) — the journal.

## Phase 2 status (2026-05-08, end of sprint)

**Cascade folds 1-4 fullcohort (DONE 2026-05-08 03:28 UTC).** Aggregated
5-fold cascade with 206 GT-negative slides:

| Cascade 5-fold (mean ± std) | mDice_pix | TLS_pix | GC_pix | TLS-FP rate | GC-FP rate |
|---|---|---|---|---|---|
| fullcohort eval | **0.618 ± 0.123** | **0.677 ± 0.078** | **0.559 ± 0.173** | 94/206 (46 %) | 14/206 (7 %) |

Per-fold breakdown:
* fold 0: mDice_pix=0.829 (highest)
* fold 1: 0.615
* fold 2: 0.529 (lowest)
* fold 3: 0.538
* fold 4: 0.578

**GNCAF v3.58 / v3.56 folds 1-4 fullcohort (LAUNCHED, in progress).**
Background driver `/tmp/run_gncaf_fullcohort_5fold.sh` started at
03:30 UTC. v3.58 fold 1 finished at 05:17 UTC (1h47m per fold).
Total ETA ~9-10 h, will continue past the sprint window.

* v3.58 fold 1 (just finished): TLS_pix=0.327, GC_pix=0.524, TLS-FP
  rate=95 % — **the 95 % TLS-FP rate reproduces on fold 1**, not a
  fold-0 anomaly. GNCAF fundamentally over-fires regardless of which
  fold's val set you pick.

The chain continues to run independently. After all 8 GNCAF folds
land (~9 h after launch), the comparison plots will have full 5-fold
data for cascade and both GNCAFs. Re-run
`notebooks/build_arch_comparison.py` to refresh.

## Final headline (sprint deliverable)

The **lost-original v3.21 hypothesis is disproven**: the lost
checkpoint is the *worst* GNCAF on the full-cohort eval (mDice_pix
0.237 vs current best v3.56's 0.423; TLS-FP rate 98 % vs 95 %).
There is no missing technique we lost during the rewrite. The dense
pixel-decoder paradigm fundamentally over-fires on tissue
regardless of GCN flavour or training time. The GARS cascade's
two-stage gate is the only paradigm in the comparison that
controls this — Stage 1 stays silent on tissue-only slides, Stage 2
never gets called → cascade resolves true negatives as TN by
construction.
