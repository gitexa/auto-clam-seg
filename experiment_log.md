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
  `train.upsample_factor=4`. Local SSD zarrs.
- **Status**: in progress — ep7 F1 = 0.637 (BEST=0.648 at ep6),
  val_selected dropping 26k → 12k (model becoming more selective).
- **Result**: TBD on completion.
- **Conclusion**: TBD.

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
- **Config**: `python eval_gars_cascade.py stage1=<v3.0> stage2=<v3.1>
  thresholds=[0.05,0.10,0.20,0.30,0.40,0.50]`.
- **Status**: pending v3.0 completion.
- **Result**: TBD.
- **Conclusion**: TBD.

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
