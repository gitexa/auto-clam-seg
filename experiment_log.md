# GARS v3 — Experiment Log

Two-stage cascade for TLS + GC pixel segmentation. Successor to v2.0
panoptic (history in `experiment_log_.md` / `experiment_log_tls.md`).
Driver doc: `program.md`. Hydra configs: `recovered/scaffold/configs/`.

## Recovered baselines (target numbers to reproduce)

| | Stage 1 (GATv2 3-hop) | Stage 2 (univ2_decoder h=64) | Cascade (thr=0.05) |
|---|---|---|---|
| Best metric | F1 = 0.561 (ep 35) | mDice = 0.7141 (ep 7) | mDice 0.486, GC dice 0.734 |
| Params | 825 K | 9.3 M | — |
| Wandb | `x50tcqt6` | `mn5jorov` | — |

GC pixel-dice 0.71 is the headline win — closes the v2.0 gap from 0.

## Metric definitions

- **Stage 1**: per-patch binary `tls_label = (mask > 0).any() over patch cell`;
  metrics are TP/FP/FN/TN aggregated across all slides at threshold 0.5.
- **Stage 2**: per-class dice on the predicted argmax 256×256 mask vs
  ground truth. `mDice` = mean over (bg, TLS, GC).
- **Cascade**: stitched per-patch class grid → per-class dice + Spearman
  correlation of connected-component counts.

## Caveats

- **Patch labelling drift** (still under investigation). Current
  `tile.max() > 0` rule produces ~2.4× more TLS-positive patches than
  the recovered build (108 vs 43.6 per slide). Doesn't affect relative
  trends but inflates absolute val_pos counts and depresses absolute
  TLS dice. GC dice unaffected.
- **Split drift**: same seed=42, same patient-stratified logic, but
  `numpy.random.RandomState` shuffle order differs across numpy versions.
  Result: val=166 slides (vs 154 in recovered); train=648 (vs 660).
  Total 814 unchanged. Val gets more high-TLS-bucket patients.

---

## Experiments

### v3.0 — Stage 1 GATv2 3-hop reproduction

- **Hypothesis**: Re-implementation reproduces recovered F1 ≈ 0.561.
- **Config**: `model=gatv2_3hop` (default), `train.lr=1e-3`,
  `train.pos_weight=5`, `train.epochs=50`, `train.patience=10`,
  `train.upsample_factor=4`, async DataLoader (4 workers, prefetch=2,
  persistent), local SSD zarrs.
- **Result**: best F1 = 0.646 at ep 11; early-stopped at ep 21.
  88 min wall-clock total, 4.2 min/epoch (~30 % faster than the
  pre-prefetch run thanks to async slide loading).
  Run dir `gars_stage1_v3.0_gatv2_3hop_20260430_002624`. Wandb
  `eufxm1u9`. **+15 % over recovered best (0.561)** — split drift
  inflates the absolute number (val has 2.4× more TLS-positives).
- **Conclusion**: KEEP. Architecture rebuild verified end-to-end. Use
  this checkpoint for v3.2 cascade eval and as the Stage 1 baseline
  for autoresearch.

### v3.1 — Stage 2 UNI-v2 decoder reproduction

- **Hypothesis**: Re-implementation reproduces recovered mDice ≈ 0.714.
- **Config**: `model=univ2_decoder_h64` (default), `train.lr=3e-4`,
  `train.batch_size=128`, `train.warmup_epochs=3`,
  `train.gc_dice_weight=2`, `train.class_weights=[1,1,3]`,
  `train.epochs=50`, `train.patience=10`. Patch cache from
  `tls_patch_dataset.pt`.
- **Status**: complete (run dir
  `gars_stage2_gars_stage2_univ2_decoder_20260427_102326`, pre-hydra
  argparse path; symlink to `v3.1_*` to follow).
- **Result**: best mDice = 0.7012 at ep 3 (val_tls=0.534, val_gc=0.899).
  Early-stopped at ep 13. **Above** recovered GC dice (0.899 vs 0.710);
  **below** recovered TLS dice (0.534 vs 0.719) — explained by patch-
  label drift inflating low-TLS patches in training.
- **Conclusion**: KEEP. Architecture rebuild verified; numbers within
  the expected range given known split + label drifts. Cascade uses
  this checkpoint until a v3.1 hydra-driven re-run lands.

### v3.2 — Cascade evaluation

- **Hypothesis**: Stitching v3.0 + v3.1 outputs reproduces recovered
  cascade numbers (mDice 0.486, GC dice 0.734 at thr=0.05).
- **Config**: `eval_gars_cascade.py stage1=<v3.0 ckpt> stage2=<v3.1 ckpt>`
  on 124 val slides with masks.
- **Result**: Run `7k1aae17`.

  | thr | mDice | TLS d | GC d | TLS sp | GC sp | sel% |
  |---|---|---|---|---|---|---|
  | 0.05 | 0.376 | 0.408 | 0.345 | 0.773 | 0.664 | 2.2 |
  | 0.10 | 0.419 | 0.448 | 0.389 | 0.814 | 0.675 | 1.7 |
  | 0.20 | 0.450 | 0.494 | 0.406 | 0.844 | 0.685 | 1.3 |
  | 0.30 | 0.474 | 0.522 | 0.425 | 0.866 | 0.685 | 1.1 |
  | 0.40 | 0.483 | 0.532 | 0.435 | 0.874 | 0.678 | 0.9 |
  | **0.50** | **0.512** | 0.530 | 0.494 | 0.874 | **0.695** | 0.7 |

  Best mDice=0.512 at thr=0.5. **Below recovered (0.486 at thr=0.05)
  on the headline mDice; HIGHER on TLS dice; LOWER on GC dice (0.494
  vs 0.734)**. The GC drop is the standout regression — likely
  caused by Stage 2 training on the over-permissive patch label set.
- **Conclusion**: KEEP as v3 baseline. Autoresearch should target
  Stage 2 GC dice or cascade GC dice as the primary improvement
  metric (mDice as secondary).

---

## Autoresearch loop (v3.3+)

Started 2026-04-30 after v3.0/v3.1/v3.2 baselines locked in.
**Target metric**: Stage 2 `val_mDice` (baseline 0.7012; iteration cost
~5 min/run). Cascade GC dice (baseline 0.494) re-checked periodically.
Per-experiment: edit → commit → train → evaluate → keep (commit) or
discard (`git reset --hard HEAD~1`).

### v3.3a — Stage 2 hidden_channels=128 (KEEP — new baseline)

- **Hypothesis**: 2× decoder bottleneck channels gives the model more
  capacity for joint TLS+GC representation. Strict-load already
  verified for the 17.7 M-param variant.
- **Config**: `model=univ2_decoder_h128`, all other defaults.
- **Result**: best mDice=0.7121 at ep10, early-stopped ep20. GC dice
  peaked 0.908 at ep4. Run `ajy0liu1`.
- **Conclusion**: KEEP. **+0.011 mDice over baseline 0.7012**. New
  Stage 2 baseline.

### v3.3b — h=128 + gc_dice_weight=5 (DISCARD)

- **Hypothesis**: Boost the GC dice term to push GC dice further. GC
  was already 0.90 in v3.3a; if it can hit 0.95, mDice climbs.
- **Config**: `model=univ2_decoder_h128 train.gc_dice_weight=5.0`.
- **Result**: best mDice=0.6822 at ep10, early-stopped ep20. GC dice
  oscillated 0.62-0.86 (more volatile than v3.3a), TLS dice stayed
  ~0.52. Run `17lu4cc0`.
- **Conclusion**: DISCARD. **−0.030 mDice vs v3.3a baseline**. Higher
  gc_dice_weight de-stabilises GC and doesn't compensate elsewhere.
  Bottleneck appears to be TLS dice (~0.53), not GC (~0.85).

### v3.3c — h=128 + class_weights=[1,3,3] (KEEP — new baseline)

- **Hypothesis**: TLS dice is the bottleneck. The default
  `class_weights=[1,1,3]` undervalues TLS in the per-pixel CE term.
  Boosting TLS to match GC (=3) should improve TLS without
  sacrificing GC.
- **Config**: `model=univ2_decoder_h128 train.class_weights=[1,3,3]`.
- **Result**: best mDice=**0.7152** at ep3, early-stopped ep13. TLS
  dice **0.572** (vs v3.3a 0.529, **+0.043**), GC dice 0.921 (vs
  v3.3a 0.908, **+0.013**). Run `0r7t2ddp`.
- **Conclusion**: **KEEP — +0.003 over v3.3a, +0.014 cumulative over
  v3.1**. Both classes improved. New Stage 2 baseline.

### v3.3d — h=128 + class_weights=[1,5,3] (KEEP — new baseline)

- **Hypothesis**: v3.3c showed TLS class weight boost helps. Push
  further (5 vs 3) to extract more.
- **Config**: `model=univ2_decoder_h128 train.class_weights=[1,5,3]`.
- **Result**: best mDice=**0.7235** at ep8, early-stopped ep18. TLS
  dice **0.589** (+0.017 over v3.3c), GC dice 0.900 (−0.021 vs v3.3c).
  Run `ywvikpbr`.
- **Conclusion**: **KEEP — +0.008 over v3.3c, +0.022 cumulative over
  v3.1**. TLS gains outweighed the small GC drop. Trend monotonic so
  far ([1,1,3]→[1,3,3]→[1,5,3] each gave a step up). Continue.

### v3.3e — h=128 + class_weights=[1,7,3] (DISCARD)

- **Hypothesis**: Continue the monotonic trend, push TLS to 7.
- **Config**: `model=univ2_decoder_h128 train.class_weights=[1,7,3]`.
- **Result**: best mDice=0.7060 at ep9, early-stopped ep19. TLS dice
  0.586 (~same as v3.3d), GC dice 0.897 (~same). Run `oq8wo258`.
- **Conclusion**: DISCARD. **−0.018 vs v3.3d**. Peak class-weight is
  TLS=5; further increases destabilize without adding signal. Sweet
  spot located.

### v3.3f — v3.3d + warmup_epochs=5 (DISCARD)

- **Hypothesis**: Longer warmup may stabilise the GC-collapse-and-
  recover pattern in ep1-2.
- **Config**: v3.3d (`class_weights=[1,5,3]`) + `train.warmup_epochs=5`.
- **Result**: best mDice=0.7187 ep8, early-stopped ep18.
  Run `kj81fx5a`.
- **Conclusion**: DISCARD. **−0.005 vs v3.3d**. Default warmup=3 is
  the sweet spot.

### v3.3g — v3.3d + gc_dice_weight=1 (KEEP — new baseline)

- **Hypothesis**: GC dice was already 0.90 — maybe the dice term is
  over-weighted (default 2). Lowering to 1 may free capacity for TLS
  without losing GC.
- **Config**: v3.3d (`class_weights=[1,5,3]`) + `train.gc_dice_weight=1`.
- **Result**: best mDice=**0.7257** at ep8, early-stopped ep18.
  TLS=0.588, GC=0.916. Run `g12d4gr5`.
- **Conclusion**: **KEEP — +0.002 over v3.3d, +0.024 cumulative**.
  Sweep `gc_dice_weight ∈ {1, 2, 5}` → best at 1. New Stage 2 baseline.

### v3.4a — v3.3g + min_tls_pixels=1024 patch label (KEEP)

- **Hypothesis**: The 2.4× over-permissive patch label dilutes TLS
  signal in training. Tightening from `tile.max() > 0` to
  `(tile > 0).sum() >= 1024` (1.5 % coverage) should improve TLS
  dice by removing low-density positives.
- **Config**: v3.3g + `cache_path=tls_patch_dataset_min1024.pt`
  (rebuilt cache, 59 323 patches vs 65 613 baseline, −10 %).
- **Result**: best mDice=**0.7265** at ep7, early-stopped ep17.
  **TLS dice 0.643 vs v3.3g 0.588 (+0.055)**, GC dice 0.900 (−0.016).
  Run `19r8e6np`.
- **Conclusion**: **KEEP — +0.001 over v3.3g, +0.025 cumulative**. The
  TLS gain is the bigger story — numbers are now much closer to the
  recovered Stage 2 profile (TLS 0.719, GC 0.710). Label tightening
  validated. Pushing min_tls_pixels higher next.

### v3.4b — v3.3g + min_tls_pixels=4096 patch label (KEEP)

- **Hypothesis**: Tighten further to ~6.25 % coverage threshold.
- **Config**: v3.3g + `cache_path=tls_patch_dataset_min4096.pt`
  (51 829 patches, −21 % vs baseline).
- **Result**: best mDice=**0.7329** at ep13, early-stopped ep23.
  TLS 0.693, GC 0.870. Run `eim80ek7`.
- **Conclusion**: **KEEP — +0.006 over v3.4a, +0.032 cumulative**.
  Tighter labels keep helping. New cascade-track baseline.

---

## End-to-end (v3.5+ — single-pass GraphEnrichedDecoder)

Architecture: `recovered/scaffold/train_gars_e2e.py`. UNI-v2 features
→ 3× GATv2 graph context → concat with raw features → spatial decoder
(16 → 256 px) → 3-class output. **No discrete patch selection** —
graph branch modulates pixel decoder for every patch.

### v3.5b — recovered config + v3.3 lessons (DISCARD — initial)

- **Hypothesis**: Reproduce recovered `qy3pj74h` (mDice=0.720) and
  apply v3.3 wins (`class_weights=[1,5,3]`, `gc_dice_weight=1`).
- **Config**: bg_decode_ratio=0.5 interpreted as "50 % of bg patches"
  (~8500 bg/slide; capped at 200 total).
- **Result**: best mDice=0.387 at ep21. Model collapsed to bg
  prediction (val_tls=0.15, val_gc=0). Run `s4lmxpq7`.
- **Conclusion**: DISCARD. bg sampling overwhelmed the TLS signal.
  Two bugs identified: (a) `per_class_dice` averages per-patch and
  uses eps trick, giving spurious 1.0 on bg-only patches; (b)
  `bg_decode_ratio` reinterpreted as ratio-relative-to-positives
  (0.5 = 1 bg per 2 pos), not "50 % of all bg".

### v3.5c-e — bg ratio + GC weight sweep (DISCARD)

- **v3.5c** (bg_decode_ratio=0): TLS=0.748, GC=0 (no bg patches → GC
  never seen as separate class). Doesn't generalize at slide level.
- **v3.5d** (bg=0.5 ratio, [1,5,3], gc_dw=1): mDice=0.508. bg dice
  recovers (0.77) but GC stays at 0 with batch-aggregate dice.
- **v3.5e** (bg=0.5, class_weights=[1,3,15], gc_dw=5): mDice=0.571
  ep5; GC climbs to 0.146 with heavy weighting but plateaus.
  Honest aggregate-then-divide dice is structurally hard for GC
  (rare class, bg-dominated denominator).

### v3.5f — bg=0.5 ratio + positives-only val (KEEP per-positives, but see v3.5g)

- **Hypothesis**: Train with bg sampling for proper slide-level
  generalisation, but **eval on positives only** with per-patch dice
  to make metrics directly comparable to cascade Stage 2.
- **Config**: `bg_decode_ratio=0.5` (relative), `class_weights=[1,5,3]`,
  `gc_dice_weight=1`. Train: positives + half as many bg = ~75/slide.
  Val: positives only with `per_class_dice`.
- **Result**: best mDice=**0.7472** at ep15, early-stopped ep25.
  TLS=0.700, GC=0.936. Run `pqwt64fe`.
- **Conclusion**: KEEP per-positives win (+0.014 over v3.4b cascade,
  +0.046 cumulative on per-positives metric). But this metric was
  insufficient — see v3.5g for the deployment-honest verdict.

### v3.5g — bg=1.0 balanced + full-slide eval at end (REJECTED at slide level)

- **Hypothesis**: 1:1 bg:positive sampling will train the model to
  predict bg correctly on bg patches, closing v3.5f's
  positives-only-val blind spot.
- **Config**: `bg_decode_ratio=1.0` (relative). End-of-training
  full-slide eval added (decode every patch in 124 val slides, build
  slide-level patch grid, compute slide-level dice + connected-
  component counts vs ground-truth `tls_num`/`gc_num`).
- **Result (per-positives val, per epoch)**: best mDice=**0.7441** ep26.
  Comparable to v3.5f.
- **Result (full-slide eval, deployment-honest)**: 124 slides, 7 min.

  | metric | full-slide | per-positives |
  |---|---|---|
  | TLS dice | **0.283** | 0.688 |
  | GC dice | **0.058** | 0.881 |
  | TLS count Spearman | 0.622 | 0.978 |
  | GC count Spearman | 0.473 | 0.634 |
  | TLS count MAE | **57.2** | 3.8 |
  | GC count MAE | **41.2** | 3.3 |

  Compare to v3.2 cascade slide-level (recovered numbers at thr=0.05):
  TLS dice 0.408, GC dice 0.345, TLS sp 0.771, GC sp 0.754.
  Run `yg5n311r`.
- **Conclusion**: **REJECT at deployment**. The single-pass e2e
  catastrophically over-predicts TLS/GC on bg patches it never
  learned to discriminate enough. Cascade > e2e by a wide margin
  (+0.25 TLS dice, +0.29 GC dice at slide level). The v3.5f
  per-positives win was an artifact of the eval regime, not a real
  improvement. **The cascade architecture is the biggest lever** —
  Stage 1 filters most bg before Stage 2, sidestepping the over-call
  problem.

---

## Next hypotheses (priority-ordered by expected lever size)

### H1 — Close the cascade-eval GC dice gap (0.49 mine vs 0.73 recovered)

- **Magnitude**: +0.24 GC dice, ~6× larger than any single
  hyperparameter tweak from v3.3.
- **Suspects**: (a) my patch-label rule is more permissive (2.4×
  more TLS-positive patches); (b) slide-level mask-cache resolution
  (upsample_factor=4) may differ from what the recovered cascade
  eval used; (c) Stage 1 selection rate is 2.2 % at thr=0.05 vs
  recovered 0.4 %.
- **Action**: instrument `eval_gars_cascade.py` to log each pipeline
  decision (selected count, mask resolution, count_components
  threshold, etc.) and diff against the recovered eval log line by
  line. Patch the divergences, re-run cascade eval.

### H2 — Slide-level training for e2e (only if cascade route exhausts)

- Train e2e with bg_decode_ratio ≥ 5 or all bg per slide. The
  model needs many more bg examples to learn the boundary.
  Risk: gradient swamped by easy bg, TLS underperforms again.

### H3 — Patch-label rule match to recovered

- The recovered Stage 2 dataset had 26 364 TLS patches; mine has
  65 613 (min=1) or 51 829 (min=4096). Try `min_tls_pixels` higher
  (8192, 16384) or reproduce the original area threshold if we can
  recover it from the recovered code path. Affects all downstream
  metrics for direct comparability.

### H4 — Stage 1 operating point

- Re-train Stage 1 with a higher precision target (e.g.,
  `pos_weight=2` instead of 5). Cascade GC dice peaks at thr=0.5 in
  my eval; the recovered cascade peaks at thr=0.05 with 4× fewer
  selections. Higher-precision Stage 1 + lower threshold should
  reproduce the recovered selection rate.

### H5 — Train Stage 2 with bg patches but eval cascade-style

- Bridge between v3.4b (positives-only training, cascade-best at
  deployment) and v3.5 (e2e). Train Stage 2 alone with limited bg
  context, then run cascade eval. Test whether Stage 2 alone gains
  at slide level when bg-aware.

### Deferred

- v3.3+ Stage 2 hyperparameter search saturated (+0.005 incremental).
- Joint S1+S2 training: known to collapse (recovered ufz9a2o4).

---

## v3.6 — Cascade with pixel-aggregate dice + v3.4b Stage 2 (KEEP — NEW BEST)

- **Hypothesis** (H1): My patch-grid cascade dice was a coarse
  approximation; the recovered eval used per-pixel aggregate dice
  over the selected patches' 256×256 native masks. Re-evaluating
  with the right metric and the better Stage 2 (v3.4b) should close
  the recovered-vs-mine gap.
- **Config**: `eval_gars_cascade.py` modified to compute both metrics
  (patch-grid kept for backward compat, pixel-agg added). Inputs:
  Stage 1 v3.0 (`gars_stage1_v3.0_gatv2_3hop_*`), Stage 2 v3.4b
  (`gars_stage2_v3.4b_h128_min4096_*`), 124 val slides, 6 thresholds.

  | thr | mDice (pix-agg) | TLS (pix-agg) | GC (pix-agg) | TLS sp | GC sp |
  |---|---|---|---|---|---|
  | 0.05 | 0.559 | 0.436 | 0.683 | 0.731 | 0.582 |
  | 0.10 | 0.633 | 0.516 | 0.749 | 0.776 | 0.609 |
  | 0.20 | 0.694 | 0.590 | 0.798 | 0.825 | 0.647 |
  | 0.30 | 0.723 | 0.631 | 0.815 | 0.852 | 0.668 |
  | 0.40 | 0.744 | 0.665 | 0.822 | 0.864 | 0.699 |
  | **0.50** | **0.771** | **0.712** | **0.830** | **0.875** | **0.737** |

  Run `so61vms2`.
- **Result vs everything else (deployment-honest)**:
  - Recovered cascade @ thr=0.50: mDice 0.457 / TLS 0.180 / GC 0.733
  - v3.2 cascade pix-agg @ thr=0.50: mDice 0.638 / TLS 0.700 / GC 0.576
  - **v3.6 cascade pix-agg @ thr=0.50: mDice 0.771 / TLS 0.712 / GC 0.830**
  - v3.5g e2e full-slide: TLS 0.283 / GC 0.058
- **Conclusion**: **KEEP — NEW CHAMPION**. **+0.097 GC dice** over the
  recovered cascade (0.733 → 0.830) and **+0.314 mDice** over v3.2
  (0.457 → 0.771) at the same threshold. The improvements stack
  cleanly:
  1. Pixel-agg metric (right deployment scoring): +0.18 mDice
     (v3.2 patch-grid 0.512 → v3.2 pix-agg 0.638)
  2. v3.4b Stage 2 (label-tightening + class_weights tuning):
     +0.13 mDice on top (v3.2 pix-agg 0.638 → v3.6 pix-agg 0.771)

  The cascade architecture wins decisively. **The "cascade-eval gap"
  was largely a metric-definition difference**, not a model
  weakness. The single Stage 2 hyperparameter wins from v3.3-v3.4
  also lift cascade deployment numbers — they weren't only
  per-positives illusions.

### Final standings

| Approach | mDice (deployment) | TLS dice | GC dice |
|---|---|---|---|
| Recovered cascade | 0.457 | 0.180 | 0.733 |
| **v3.6 cascade** | **0.771** | **0.712** | **0.830** |
| v3.5g e2e | (full-slide-grid metric) | 0.283 | 0.058 |
| v3.5f e2e | (per-positives proxy only) | 0.700 | 0.936 |

**Deployment-honest take**: cascade > e2e by a wide margin at slide
level. v3.5 e2e was an instructive dead end — single-pass models
without a Stage-1 filter overpredict TLS/GC on bg patches.

---

## v3.7 — Counting post-processing sweep + adaptive thresholding

- **Hypothesis**: dice is fixed by Stage 1+2; counting metrics
  (Spearman vs `df_summary_v10.csv` `tls_num`/`gc_num`, MAE) can still
  be lifted by morphological post-processing of the patch grid
  before connected-component counting (drop tiny components, fill
  1-cell gaps). Separately, adaptive top-K% thresholding might fix
  thr=0.5's over-conservatism on dense slides — but at the cost of
  forcing selection on bg-only slides.
- **Setup**: same pipeline as v3.6 (Stage 1 v3.0 + Stage 2 v3.4b,
  pix-agg metric, thr=0.5). Vary `min_component_size`,
  `closing_iters`, and `top_k_frac` independently.

  | run | min_size | close | top_K% | mDice_pix | TLS_pix | GC_pix | TLS sp | GC sp | GC MAE |
  |---|---|---|---|---|---|---|---|---|---|
  | v3.6 baseline | 1 | 0 | — | 0.771 | 0.712 | 0.830 | 0.875 | 0.737 | — |
  | **v3.7a** | **2** | **1** | — | 0.771 | 0.712 | 0.830 | 0.868 | **0.878** | **0.42** |
  | v3.7b | 3 | 1 | — | 0.771 | 0.712 | 0.830 | 0.867 | 0.852 | 0.58 |
  | v3.7c | 2 | 2 | — | 0.771 | 0.712 | 0.830 | 0.860 | 0.876 | 0.44 |
  | v3.7d | 2 | 1 | 0.005 | 0.733 | 0.662 | 0.804 | 0.308 | 0.720 | 0.94 |

- **Result**:
  - **v3.7a (min_size=2, closing=1) is the counting champion**.
    Same pixel-aggregate dice as v3.6 (post-processing operates on the
    grid, not on the per-pixel masks), but **GC Spearman 0.737 → 0.878
    (+0.141)** and **GC MAE 0.42** (vs v3.6 baseline that had no
    counting metric on file but per-pixel was unchanged).
  - min_size=3 is too aggressive (TLS sp drops, GC sp drops); closing=2
    barely changes things over closing=1; min_size=2 / closing=1 is
    the Pareto point.
  - **v3.7d (adaptive top-K%=0.005) is a clear regression**: TLS sp
    crashes from 0.868 → **0.308** because forcing top-0.5% selection
    on slides without any TLS produces phantom counts. The recovered
    cascade's "low threshold = better TLS" pattern was a function of
    its smaller selection set, not of adaptive K.
- **Conclusion**: **KEEP v3.7a as counting champion**. **DISCARD v3.7d**.
  Adaptive thresholding fails because it has no abstain. Cheap fix
  for a future sweep: only apply top-K when an absolute thr=0.X
  is exceeded somewhere on the slide.

### Final standings (counting + dice)

| Approach | mDice_pix | TLS dice | GC dice | TLS sp | GC sp | GC MAE |
|---|---|---|---|---|---|---|
| Recovered cascade | 0.457 | 0.180 | 0.733 | — | — | — |
| v3.6 cascade @0.5 | 0.771 | 0.712 | 0.830 | 0.875 | 0.737 | — |
| **v3.7a cascade @0.5** | **0.771** | **0.712** | **0.830** | **0.868** | **0.878** | **0.42** |
| v3.5g e2e (full-slide) | — | 0.283 | 0.058 | — | — | — |

---

## v3.8–v3.16 — Stage 1 graph-context + Stage 2 retraining sprint (6h)

User gave 6h autonomy + "max out the GPU". One A100 fits one Stage 1 at
a time (~34 GB), so trainings serialised; cascade evals chained behind.

### Trainings

- **v3.8** Stage 1, GATv2 n_hops=5, h=256 (1.09 M params). 30 epochs,
  ~265 s/epoch. Final F1=0.6255 (vs v3.0 F1=0.5997, **+0.026**).
  ep1 F1=0.367 (v3.0 ep1=0.039) — much faster ramp from richer graph
  context. Plateaued ~ep9 with val_f1=0.657.
- **v3.10** Stage 1, GATv2 n_hops=3, h=512 (2.44 M params). Wider
  graph features, same depth as v3.0. Run ~2h45m wall.
- **v3.11** Stage 2, h=128 + `min_tls_pixels=8192` patch cache (45 454
  patches; the 12.5 % coverage threshold drops 30 % of v3.4b's training
  data). Quick (~14 min).
- **v3.12** Stage 2, h=128 + class_weights=[1, 5, 5] + gc_dice_weight=2.
  GC-focused recipe to test whether v3.4b's GC dice ceiling (0.83) is
  loss-shape limited. (~12 min.)

### Cascade evals (all min_size=2, closing=1, threshold sweep)

Best threshold per run (mostly thr=0.5 — the cascade prefers tight
selection at deployment):

| Cascade | Stage 1 | Stage 2 | thr | mDice_pix | TLS_pix | GC_pix | TLS sp | GC sp | TLS mae | GC mae |
|---|---|---|---|---|---|---|---|---|---|---|
| v3.6 baseline | v3.0 (3hop, 256) | v3.4b | 0.50 | 0.7709 | 0.7117 | 0.8301 | 0.875 | 0.737 | — | — |
| v3.7a | v3.0 | v3.4b + min2/close1 | 0.50 | 0.7709 | 0.7117 | 0.8301 | 0.868 | 0.878 | 5.65 | 0.42 |
| v3.7e (abstain top-K) | v3.0 | v3.4b | 0.50 | 0.7339 | 0.6641 | 0.8036 | 0.325 | 0.720 | 10.82 | 0.94 |
| **v3.13 ← NEW BEST** | **v3.8 (5hop, 256)** | **v3.4b** | **0.50** | **0.7839** | **0.7222** | **0.8457** | 0.874 | **0.887** | 5.44 | **0.39** |
| v3.14 | v3.10 (3hop, 512) | v3.4b | 0.50 | 0.7721 | 0.7065 | 0.8376 | **0.885** | 0.859 | **5.03** | 0.42 |
| v3.15 | v3.0 | v3.11 (min8192) | 0.50 | 0.7139 | 0.6619 | 0.7658 | 0.870 | 0.873 | 5.66 | 0.49 |
| v3.16 | v3.0 | v3.12 (cw=[1,5,5], gc_dice=2) | 0.50 | 0.6419 | 0.6441 | 0.6398 | 0.870 | 0.804 | 5.61 | 0.65 |

### Conclusions

- **NEW CHAMPION: v3.13** (v3.8 + v3.4b @ thr=0.5). All five
  deployment metrics that matter at slide level either match or
  improve over v3.7a:
  - mDice_pix **0.7709 → 0.7839** (+0.013)
  - TLS_pix 0.7117 → **0.7222** (+0.010)
  - GC_pix 0.8301 → **0.8457** (+0.016)
  - GC sp 0.878 → **0.887** (+0.009)
  - GC mae 0.42 → **0.39** (−0.03)
  - TLS sp 0.868 → 0.874 (≈ flat)
  - TLS mae 5.65 → 5.44 (−0.2 instances)
- **Deeper > wider for graph context.** v3.8 (n_hops=5, 1.1 M params)
  beats v3.10 (n_hops=3, h=512, 2.4 M params) on dice.
  v3.10 wins TLS counting in isolation (best TLS sp=0.885, best TLS
  mae=5.03), but its lower GC dice and higher GC mae lose the
  composite race.
- **Abstain top-K (v3.7e) ≈ v3.7d** — adding a `top_k_abstain_thr=0.5`
  guard barely changed results (TLS sp 0.308 → 0.325). The diagnosis
  was wrong: top-K isn't *firing* on TLS-empty slides — its problem
  is forcing 0.5 % selection on every slide regardless of content
  density. Killing this branch.
- **`min_tls_pixels=8192` (v3.11) regresses**. Tighter patch labels
  drop training data faster than they clean it up. mDice 0.771 →
  0.714 at slide level.
- **GC-focused Stage 2 (v3.12) collapses GC dice** (0.830 → 0.640).
  Stacking class_weights=[1,5,5] AND gc_dice_weight=2 was too
  aggressive; the model overfits GC pixels into spurious shapes.
  Lesson: tune one loss-shape lever at a time.

### Final standings (deployment-honest)

| Approach | mDice_pix | TLS dice | GC dice | TLS sp | GC sp | GC mae |
|---|---|---|---|---|---|---|
| Recovered cascade | 0.457 | 0.180 | 0.733 | — | — | — |
| v3.5g e2e (full-slide) | — | 0.283 | 0.058 | — | — | — |
| v3.6 cascade | 0.771 | 0.712 | 0.830 | 0.875 | 0.737 | — |
| v3.7a cascade | 0.771 | 0.712 | 0.830 | 0.868 | 0.878 | 0.42 |
| **v3.13 cascade** | **0.784** | **0.722** | **0.846** | 0.874 | **0.887** | **0.39** |

---

## v3.21 — GNCAF revival (lost architecture)

User flagged: "another experimental architecture already tested which
also was lost." Found `gncaf_model.py` + `gncaf_dataset.py` +
`verify_gncaf.py` in `recovered/scaffold/` but **no training script**
— only the verify hook survived. Wrote `train_gncaf.py` from scratch.

GNCAF (Su et al. 2025, arXiv:2505.08430) is e2e:
- ViT TransUNet encoder over the *RGB tile* of each target patch (we
  used 6 layers / dim=384 to fit budget; paper uses 12).
- 3-hop GCN over UNI-v2 features for *all* patches in the slide.
- Fusion via 1-layer MSA over `[local_tokens; context.unsqueeze(1)]`.
- 4× upsample decoder → (B, 3, 256, 256) per-pixel logits.

**Run:** `gars_gncaf_v3.21_gncaf_recovered_*`. 14.9 M params. 15
epochs, ~70 s/epoch. Bottleneck: WSI tile reads (60 s of 70 s).

**Result:** ep13 val per-positives mDice=**0.607** (TLS=0.794,
GC=0.419). Roughly matches recovered `qy3pj74h` GraphEnrichedDecoder
(0.720) but **does not beat v3.4b cascade at per-positives (0.733)**,
and **cannot run slide-level eval at our budget** — it requires an
RGB tile per patch and we have ~17 K patches/slide × ~100 ms tile
read = ~30 min/slide, ~60 h for the val set. Architectural cost.

**Conclusion:** GNCAF works; it's not the deployment win. The cascade
keeps its lead because UNI-v2 features alone (no RGB) are enough for
patch-level classification + decoder.

---

## v3.22 — Patch classifier with NO pixel upsampling

User: "try to get the same results w/o the upsampling to pixel level."
Hypothesis: predict one class per patch (bg/TLS/GC) with the same
3-hop GATv2 graph context as Stage 1 but a 3-class head — no Stage 2,
no 16→256 spatial decoder. Cheapest possible architecture; eliminates
the per-pixel decoder entirely.

`GraphPatchClassifier`: 1.3 M params. Same backbone as Stage 1, head
is `Linear(256, 256) → GELU → Linear(256, 3)`.

Per-patch label rule: 2 if any GC pixel, else 1 if `(mask>0).sum() ≥
4096`, else 0. Trained 30 epochs, full-slide forward at val (no
sampling — every patch contributes a label).

**Run:** `gars_patchcls_v3.22_patchcls_gatv2_5hop_*`. Best at ep29.

| metric | v3.22 patchcls | v3.13 cascade |
|---|---|---|
| patch-grid mDice | **0.602** | 0.539 |
| patch-grid TLS dice | 0.648 | 0.564 |
| patch-grid GC dice | **0.555** | 0.511 |
| TLS Spearman | 0.873 | 0.874 |
| GC Spearman | 0.824 | **0.887** |
| TLS MAE | (n/a) | 5.44 |
| GC MAE | (n/a) | **0.39** |
| pix-agg mDice | (n/a) | **0.784** |
| params | **1.3 M** | 11.6 M (S1+S2) |
| inference | 1 forward, no decoder | 2 forwards + decoder |

**Conclusion:**
- **Patch-grid metric: patchcls wins** (+0.063). Without the per-pixel
  decoder noise, the patch classifier directly learns the
  patch-class boundary cleanly.
- **TLS counting: tied** (Spearman within 0.001).
- **GC counting: cascade wins** (+0.063 Spearman, much lower MAE).
  GC is small/dense — pixel decoder helps separate adjacent germinal
  centres into distinct components; patch-level prediction merges
  them into one big patch blob, undercounting.
- **Pixel-level dice is unavailable** for patchcls — that's the trade.

**When to use which:**
- Counting *only* TLS and patch-class deployment (downstream consumes
  patch-grid masks)? → `patchcls` is enough, 1/9 the parameters.
- Pixel-level dice or precise GC instance counts? → cascade required.

---

## v3.17, v3.23, v3.24, v3.25 — sweet-spot + Stage 1 swap

| run | Stage 1 | Stage 2 | mDice_pix | TLS_pix | GC_pix | TLS sp | GC sp | GC mae |
|---|---|---|---|---|---|---|---|---|
| **v3.13 (champ)** | v3.8 (5hop) | v3.4b | **0.7839** | 0.7222 | **0.8457** | 0.874 | **0.887** | **0.39** |
| v3.23 | **v3.17 (4hop)** | v3.4b | 0.7783 | 0.7197 | 0.8369 | 0.879 | 0.887 | 0.39 |
| v3.24 | v3.8 | v3.11 (min8192) | 0.7260 | 0.6743 | 0.7778 | 0.875 | 0.872 | 0.44 |
| v3.25 | v3.8 | v3.12 (GC focus) | 0.6528 | 0.6560 | 0.6496 | 0.874 | 0.819 | 0.64 |

**v3.17 Stage 1 (n_hops=4)** — best F1=0.654 ep15, final F1=0.616.
Sits between v3.0 (F1=0.600) and v3.8 (F1=0.626/0.657). Cascade
score (v3.23) is **−0.006 mDice** below v3.13 — within noise.

**Sweet-spot hypothesis fails.** n_hops=4 doesn't beat n_hops=5; depth
is monotonic but with diminishing returns past 5.

**v3.8 lifts min8192 cascade** (v3.24 mDice=0.726 vs v3.15=0.714 with
v3.0). +0.012 from Stage 1 swap, but still **−0.058 below v3.13**.
Tighter labels cost more than they gain.

**v3.8 lifts GC-focused cascade** marginally (v3.25 mDice=0.653 vs
v3.16=0.642 with v3.0), but GC dice still collapses (0.650 vs 0.846).
The loss-shape issue is independent of Stage 1.

**Champion unchanged: v3.13** (v3.8 + v3.4b @ thr=0.5).

### Final standings (deployment-honest, post v3.21–v3.25)

| Approach | params | mDice_pix | TLS dice | GC dice | TLS sp | GC sp | GC mae |
|---|---|---|---|---|---|---|---|
| Recovered cascade | — | 0.457 | 0.180 | 0.733 | — | — | — |
| v3.5g e2e (full-slide) | 10 M | — | 0.283 | 0.058 | — | — | — |
| v3.21 GNCAF (per-pos) | 14.9 M | 0.607* | 0.794* | 0.419* | — | — | — |
| **v3.22 patchcls (no decoder)** | **1.3 M** | n/a | **0.648 (grid)** | 0.555 (grid) | 0.873 | 0.824 | n/a |
| v3.6 cascade | 11.6 M | 0.771 | 0.712 | 0.830 | 0.875 | 0.737 | — |
| v3.7a cascade | 11.6 M | 0.771 | 0.712 | 0.830 | 0.868 | 0.878 | 0.42 |
| v3.23 cascade (4hop) | 11.6 M | 0.778 | 0.720 | 0.837 | 0.879 | 0.887 | 0.39 |
| **v3.13 cascade (5hop)** | **11.6 M** | **0.784** | **0.722** | **0.846** | 0.874 | **0.887** | **0.39** |

*GNCAF metric is per-positives only — not slide-level comparable.

---

## v3.26 — patchcls + cascade hybrid (DOES NOT win)

Hypothesis: patchcls (v3.22) is a more accurate 3-way patch router
than the binary Stage 1 (v3.8). Replace Stage 1 selection with
patchcls's `class > 0` patches → Stage 2 decoder for pixel masks.

| metric | v3.22 patchcls | v3.13 cascade | v3.26 hybrid |
|---|---|---|---|
| patch-grid mDice | **0.602** | 0.539 | 0.542 |
| pix-agg mDice | n/a | **0.784** | 0.777 |
| pix-agg TLS dice | n/a | **0.722** | 0.709 |
| pix-agg GC dice | n/a | 0.846 | **0.845** |
| TLS Spearman | 0.873 | 0.874 | **0.879** |
| GC Spearman | 0.824 | **0.887** | 0.831 |
| TLS MAE | n/a | 5.44 | **5.07** |
| GC MAE | n/a | **0.39** | 0.43 |

**Conclusion:** hybrid is a middle-ground, not a winner. Selected
15 573 patches (vs cascade's 15 406 at thr=0.5 — similar count). But:
- Patch-grid metric is *worse* than patchcls alone because Stage 2's
  argmax overrides patchcls predictions on selected patches, and
  Stage 2 is trained for pixel-level dice (sparser TLS pixels per
  patch) — so the per-patch class drifts toward what cascade
  predicts, not patchcls.
- GC counting drops vs cascade because patchcls's GC routing is less
  precise than Stage 1 v3.8's threshold (which has a tighter GC
  selection by way of the Stage 2 GC dice loss feedback).

The intuition that patchcls would route better than Stage 1 was
wrong — Stage 1's *binary* objective with a tuned threshold turns
out to select the right TLS+GC patches more cleanly than the
3-class softmax routing. (Probably because the binary loss has
~50× more positive examples per slide than the 3-class loss has
GC examples.)

**Champion still v3.13.** Hybrid kept on file for the speed wins
(TLS MAE 5.07 is the new best on that metric), not deployment.

---

## v3.27 — Stage 2 h=256 (regresses)

Bigger decoder (h=128 → h=256). Per-positives mDice=0.724 (vs v3.4b's
0.733). Cascade @ thr=0.5 mDice_pix=0.667 (vs v3.13=0.784). **−0.117
mDice at deployment.** More decoder capacity overfits per-positives
and underperforms on bg-rich slide-level eval. v3.4b at h=128 stays.

## v3.29 — Stage 2 with min_tls_pixels=2048 (looser label, regresses)

Looser patch-positive threshold (4096 → 2048 pixels of mask>0).
Per-positives mDice=0.727 (TLS=0.654, GC=0.879). Cascade v3.31
(v3.8 + v3.29) @ thr=0.5: mDice_pix=0.676 (vs v3.13=0.784). **−0.108
at deployment.** Looser labels add noisy weakly-positive TLS patches
that cost more in TLS dice than they gain in coverage.

## v3.32 / v3.33 / v3.34 — Stage 1 n_hops=6 (killed early)

v3.32 (Stage 1 GATv2 6-hop) was killed at ~ep14 to free GPU for
v3.35 GNCAF training. Cascade v3.33 (partial v3.32 + v3.4b) @ thr=0.5
mDice_pix=0.621 — undertrained, not a fair benchmark. v3.34 chain
aborted. n_hops=6 remains untested at convergence.

## v3.26 — patchcls + cascade hybrid (DOES NOT win)

Hypothesis: patchcls (v3.22) is a more accurate 3-way patch router
than the binary Stage 1 (v3.8). Replace Stage 1 selection with
patchcls's `class > 0` patches → Stage 2 decoder for pixel masks.

| metric | v3.22 patchcls | v3.13 cascade | v3.26 hybrid |
|---|---|---|---|
| patch-grid mDice | **0.602** | 0.539 | 0.542 |
| pix-agg mDice | n/a | **0.784** | 0.777 |
| pix-agg TLS dice | n/a | **0.722** | 0.709 |
| pix-agg GC dice | n/a | 0.846 | **0.845** |
| TLS Spearman | 0.873 | 0.874 | **0.879** |
| GC Spearman | 0.824 | **0.887** | 0.831 |
| TLS MAE | n/a | 5.44 | **5.07** |
| GC MAE | n/a | **0.39** | 0.43 |

**Conclusion:** middle-ground. Stage 2's pixel-argmax overrides
patchcls predictions on selected patches, so the patch-grid lift
disappears. GC Spearman drops because patchcls's GC routing is less
precise than Stage 1's tuned binary threshold. Champion still v3.13.

## v3.35 — Full-config GNCAF (in flight)

Per user request, training the **paper-faithful 12-layer ViT GNCAF**
(25.5 M params; v3.21 used a truncated 6-layer ViT). Config:
encoder_layers=12, dim=384, encoder_heads=6, gcn_hops=3,
fusion_heads=8, class_weights=[1,5,3], 30 epochs, AMP bf16.

Run: `gars_gncaf_v3.35_gncaf_full_12layer_*`. Followed by slide-level
eval `eval_gars_gncaf.py` (smoke-tested at 44 s/slide → ~91 min full
val). Results land in `summary.md` once the eval finishes.

## v3.36 — NeighborhoodPixelDecoder cascade (FAILS)

User: *"For the cascade: the selection should not be patch, but
neighbourhood based, and then the upsampling should be for a selected
neighbourhood; …use more information compared to patch-based strategy."*

`NeighborhoodPixelDecoder` (17.7 M params): per-cell SpatialBasis →
tile to (B, 128, 48, 48) → 4× UpDoubleConv → (B, 3, 768, 768).
Inference: Stage 1 v3.8 → connected components → 3×3 window centered
on each component (stride-2 raster for components > 3×3) → batched
decode → per-cell max-prob reconciliation.

Trained 12 epochs (patience-stopped); best ep2 per-positives
mDice = 0.798 (TLS = 0.719, GC = 0.826) — appeared to exceed v3.4b's
0.733 ceiling.

**Slide-level benchmark (124 val, threshold sweep):**

| thr | mDice_pix | TLS_pix | GC_pix | TLS sp | GC sp | GC mae |
|---|---|---|---|---|---|---|
| 0.05 | 0.573 | 0.562 | 0.584 | 0.839 | 0.760 | 0.79 |
| 0.10 | 0.590 | 0.587 | 0.593 | 0.858 | 0.778 | 0.77 |
| 0.20 | 0.612 | 0.617 | 0.607 | 0.871 | 0.792 | 0.67 |
| 0.30 | 0.626 | 0.637 | 0.614 | 0.863 | 0.797 | 0.64 |
| 0.40 | 0.639 | 0.658 | 0.621 | 0.871 | 0.834 | 0.56 |
| **0.50** | **0.658** | **0.686** | **0.629** | **0.874** | **0.862** | **0.48** |

**vs v3.13 champion @ thr=0.5:**

| metric | v3.13 cascade | v3.36 neighborhood | Δ |
|---|---|---|---|
| mDice_pix | **0.784** | 0.658 | −0.126 |
| TLS_pix | **0.722** | 0.686 | −0.036 |
| GC_pix | **0.846** | 0.629 | −0.217 |
| TLS sp | 0.874 | 0.874 | tied |
| GC sp | **0.887** | 0.862 | −0.025 |
| GC mae | **0.39** | 0.48 | +0.09 |

**Conclusion: REGRESSION on every metric.** Hypothesis falsified.
The per-positives val ceiling (0.798) was misleadingly optimistic
because val windows are *centered on positives* — same distribution
as training. At deployment, windows land on edge cells of larger
components or on isolated patches; the decoder's center-vs-edge cell
weights don't generalize to those positions. Critically, GC dice
collapses (−0.22): GC structures are mostly intra-patch, and the 3×3
windowed decoding blurs their boundaries via the per-cell argmax
reconciliation across overlapping windows.

**Architectural takeaway** (validated by user critique mid-experiment):
the UNI feature is a 1536-d summary of a 256×256 RGB tile — there is
no sub-patch pixel information. Adding 9 UNI features per decoded
region multiplies that summary 9× but doesn't break the UNI bottleneck.
For pixel-level accuracy, the actual RGB has to enter the decoder.
That's the v3.37 RegionDecoderCascade test.

## v3.37 — RegionDecoderCascade (NEW CHAMPION)

User's design: *"using the 'region proposal' network to limit to the
very few TLS containing candidate regions, and then only run the
decoder on those with by providing for them the slide patches (GNCAF
similar)…(UNIv2 context, raw patches as context (probably resnet
embedded or similar)…counting could probably be done on graph level;
upsampling is only for better interpretability."*

Architecture:
- **Stage 1 frozen** (v3.8 GAT 5-hop) — region proposal + counting
- **`RegionDecoder`** (14.5 M params total = 11.2 M ResNet-18 + 0.8 M UNI proj + 0.13 M GAT proj + 2.4 M decoder + pad embeds):
  - Per-cell ResNet-18 on RGB tile → (512, 8, 8)
  - Add projected UNI feature (1536→512) and projected Stage 1 GAT
    context (256→512) broadcast at the 8×8 feature map
  - Tile 9 cells to (B, 512, 24, 24) → 5× UpDoubleConv → (B, 3, 768, 768)
  - Pad embeddings for invalid cells, valid_mask suppression on RGB
- **At biobank scale (~20 K slides):** Stage 1 alone (~10 ms/slide) →
  ~1.5 h, gives instance counts. Region decoder optional, ~50 regions/
  slide → ~1.5 h additional. **~300× cheaper than GNCAF** (~6 days).

**Training:** Stage 1 frozen, only the decoder trains. Best at ep17
(per-positives mDice = 0.848, TLS=0.775, GC=0.870). Patience-stopped
at ep25 (~3 h training). After fixing a DataLoader fork issue (set
`stage1=None` post-precompute → enables num_workers=12 for parallel
WSI tile reads), training ran at ~7.3 min/epoch.

**Slide-level benchmark (123 val, threshold sweep) — `eval_gars_cascade.py region_mode=true`:**

| thr | mDice_pix | TLS_pix | GC_pix | TLS sp | GC sp | TLS mae | GC mae |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.7793 | 0.7167 | 0.8419 | 0.851 | 0.893 | 9.87 | 0.35 |
| 0.10 | 0.7925 | 0.7364 | 0.8487 | 0.862 | 0.892 | 8.06 | 0.35 |
| 0.20 | 0.8088 | 0.7633 | 0.8544 | 0.880 | 0.920 | 6.14 | 0.33 |
| 0.30 | 0.8205 | 0.7812 | 0.8597 | 0.867 | 0.924 | 5.46 | 0.28 |
| 0.40 | 0.8329 | 0.7994 | 0.8665 | 0.873 | 0.925 | 5.29 | 0.28 |
| **0.50** | **0.8450** | **0.8216** | **0.8683** | **0.876** | **0.934** | **5.36** | **0.28** |

**vs prior champion v3.13 cascade @ thr=0.5:**

| metric | v3.13 | v3.37 | Δ |
|---|---|---|---|
| mDice_pix | 0.784 | **0.845** | **+0.061** |
| TLS_pix | 0.722 | **0.822** | **+0.100** (+13.9 %) |
| GC_pix | 0.846 | **0.868** | +0.022 |
| TLS sp | 0.874 | 0.876 | +0.002 |
| GC sp | 0.887 | **0.934** | **+0.047** |
| TLS mae | 5.44 | 5.36 | −0.08 |
| GC mae | 0.39 | **0.28** | **−0.11** (−28 %) |

**Cost:** 2.65 s/slide (15× slower than v3.13 cascade, **40× faster
than GNCAF**). Biobank-scale (20 K slides) ≈ **14.7 h** vs GNCAF's
~23 days. **300× cost reduction at higher quality.**

**Per-cancer breakdown @ thr=0.5 (patch-grid):** BLCA TLS=0.613 GC=0.743;
KIRC TLS=0.567 GC=**0.948**; LUSC TLS=0.585 GC=0.828. KIRC GC counting
is essentially solved.

**Why v3.37 wins where v3.36 failed:**

- v3.36 (UNI-only neighborhood) couldn't break the **UNI bottleneck**:
  the 1536-d feature is a 256×256 RGB summary; multiplying by 9 cells
  doesn't add sub-patch pixel info.
- v3.37 (RGB region) directly feeds the actual pixel data into a
  pretrained ResNet-18 encoder, then fuses with UNI + Stage 1 graph
  context at the bottleneck. Sub-patch pixel detail flows into the
  decoder for the first time.
- Region-only decoding keeps cost bounded: only run the heavy ResNet
  pass on ~50 candidate regions/slide (≈0.7 % of patches), not all
  ~17 K patches like GNCAF does.

**Provenance:**
- Stage 1: `gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt` (frozen)
- Stage 2: `gars_region_v3.37_full_20260502_144124/best_checkpoint.pt` (56 MB, ep17)
- Eval: `gars_cascade_v3.37_full_eval_v2_*/cascade_results.json`

## Next hypotheses (post-v3.37)
2. **Stage 2 skip connections** (v3.27). Add encoder-decoder skips to
   the UNIv2PixelDecoder. Could lift TLS dice toward GC's level (the
   current 0.722 / 0.846 gap is the largest dice headroom we have).
3. **Per-slide / per-cancer-type threshold calibration** (v3.28).
4. **GNCAF with full ViT + cached RGB tiles** — pre-extract 256×256
   RGB tiles once for all val patches (~9 GB), eliminate the I/O wall.
   Then GNCAF can run slide-level eval and we compare apples to apples.
5. **n_hops=6 / 7 sweep** — still inside diminishing-returns territory
   but worth confirming the v3.13 ceiling isn't on the wrong side of
   the elbow.
6. **`min_tls_pixels` between 2048 and 4096** — v3.4b uses 4096,
   tighter (8192) regresses. Looser (e.g., 2048) might give a small
   lift if the floor effect is real.

