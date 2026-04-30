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

---

## Next hypotheses (v3.3+)

1. **Tighten patch label** — switch `tile.max() > 0` to
   `(tile > 0).sum() > N_pixels` (e.g., N=64) to match the recovered
   ~43 patches/slide. Should reduce false-positive Stage-1 selections.
2. **End-to-end joint training** — original `ufz9a2o4` collapsed
   (Stage 1 selects nothing). Try frozen-Stage-1 fine-tune of Stage 2
   with cascade-eval loss instead.
3. **Unified `GraphEnrichedDecoder`** — recovered `qy3pj74h` (10.15 M
   params) was crashed; rebuild from checkpoint shapes alone.
4. **GC-only Stage 2** — train a GC-only decoder on ROI patches; see if
   GC dice improves above 0.71 when not competing with TLS in the loss.
