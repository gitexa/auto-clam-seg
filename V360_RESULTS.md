# v3.60 — Multi-scale bipartite-graph cascade (results journal)

This file documents the v3.60 experiment: bipartite multi-scale graph
in Stage 1 of the GARS cascade, paired with the v3.37 RegionDecoder
in Stage 2. Started 2026-05-07 in autonomous mode (auto mode + GPU
sprint mode). The plan that motivated this work is in
`PLAN_multiscale_cascade_v360.md`.

## TL;DR (FINAL — 2026-05-07 06:00)

- Stage 1 multi-scale alone improves Stage 1 F1 by **+2.9 pts**
  (0.628 → 0.657, fold 0).
- But end-to-end cascade pixel-Dice **fails the decision gate
  (≥0.86)** in both Stage 2 variants tried:
  - frozen v3.37 Stage 2: mDice_pix = 0.816 (−0.029 vs 0.845 baseline)
  - paired v3.60 Stage 2 retrain: mDice_pix = **0.596 (−0.249)**
- Per the plan's decision rule, **5-fold compute is NOT committed**
  for v3.60. Multi-scale Stage 1 by itself does not buy enough at
  the cascade level to justify the extra zarr I/O and graph size.
- (My earlier note that frozen-S2 cascade was "0.527 mDice_pix" had
  the metric inverted — that was patch-grid mDice, not pixel-agg.)

## Architecture

### Bipartite graph

For each slide, build one heterogeneous graph with:

- **N_fine** ≈ 22k nodes — 256-px@20× patches (existing local SSD zarrs)
- **N_coarse** ≈ N_fine / 4 — 512-px@20× patches (trident zarrs at
  `…/representations_tif_trident/20x_512px_0px_overlap/features_uni_v2_grandqc_zarr`)
- Edges:
  - `fine-fine` (4-conn, from `graph_edges_1hop`)
  - `coarse-coarse` (4-conn, from coarse zarr `graph_edges_1hop`)
  - `coarse↔fine` containment edges (each 512-px patch ↔ its ≤4
    enclosing 256-px patches; bidirectional, two directed edges)

Containment is built by hashing fine coords by `(x, y)` and looking
up the four `(x_c + Δx, y_c + Δy)` for `Δ ∈ {0, 256}` (paper Eq.
position). Verified: avg 3.88 children per coarse node (boundary
patches have <4).

### Multi-scale Stage 1

`MultiScaleGraphTLSDetector` = `GraphTLSDetector` + a learnable
`nn.Embedding(2, hidden_dim)` "scale embedding" (zero-init) added
to projected features. Same 5-hop GATv2, same hyperparams; the
extra parameter count is negligible. Loss + metrics are computed on
**fine nodes only** (via `scale_mask == 0`); coarse nodes act as
context passers.

### Stage 2

Same `RegionDecoder` as v3.37. Per the plan's "isolate one variable
at a time" rule, we did NOT change the decoder for v3.60. Stage 1's
output 256-d graph context is fed exactly the way v3.37 expects.

## Results so far

### Stage 1 (fold 0)

| Variant                | val F1  | val rec | val prec |
|------------------------|---------|---------|----------|
| Single-scale (v3.37)   | 0.628   | 0.751   | 0.539    |
| Multi-scale (v3.60)    | **0.657** | 0.677 | 0.639    |

Multi-scale tilts the operating point toward higher precision /
slightly lower recall at the same threshold. Best epoch = 11 of 30.

### Cascade fold 0 — pixel-Dice (the gate metric)

Aggregated across the 4 eval shards (slide-weighted mean over 120
val slides; pixel-agg = mean of per-slide pixel-level Dice).

| Variant                                  | mDice_pix | TLS_pix | GC_pix | patch-grid mDice |
|------------------------------------------|-----------|---------|--------|------------------|
| v3.37 baseline (1-scale S1 + 1-scale S2) | **0.845** | 0.822   | 0.868  | n/a              |
| v3.60 multi-scale S1 + frozen v3.37 S2   | 0.816     | 0.839   | 0.793  | 0.527            |
| v3.60 multi-scale S1 + paired v3.60 S2   | **0.596** | 0.744   | **0.448** | 0.637         |

Two effects to highlight:

1. **Frozen v3.37 Stage 2 path** holds up surprisingly well in
   pixel-Dice (−0.029 vs baseline) and even slightly improves TLS
   pixel-Dice (+0.017). GC pixel-Dice drops by 0.075. Patch-grid
   mDice is much lower (0.527) but that just reflects
   distribution-shifted per-window outputs that the v3.37 decoder
   smooths over at the pixel level.

2. **Paired Stage 2 retrain crashes pixel-Dice** to 0.596 even
   though it lifts patch-grid mDice from 0.527 → 0.637. The
   neighborhood loss optimises per-window mDice; the retrained
   decoder exploits this, but produces overconfident small masks
   that miss most of the GC pixel area (GC pixel-Dice 0.448 vs
   0.793 frozen). This is a metric-misalignment failure, not a
   training bug — the loss simply doesn't reward pixel coverage
   on top of the (Stage-1-altered) selection distribution.

### Stage 2 paired retrain training curve (for the record)

`gars_region_v3.60_fold0_s2_paired_20260507_030713`

Best val_mDice (patch-grid) = **0.8549** at epoch 11; early-stopped
at epoch 19 (patience 8). Train loss collapsed from 0.90 → 0.11 over
the run, val loss climbed steadily — heavy training-side overfit but
val mDice stable, exactly the pattern that hurts pixel-coverage.

## Decision rule (from plan) — RESULT

- Gate: paired-Stage-2 v3.60 fold-0 mDice_pix ≥ 0.86. **Failed
  (0.596).** No 5-fold compute committed.
- Even the more lenient "frozen Stage 2" reading (0.816) does not
  beat the 0.845 baseline.

## Why this didn't work — hypotheses

The paired Stage 2 lifts the **patch-grid** metric it was trained
on (0.527 → 0.637) but loses on **pixel-Dice** (0.816 → 0.596).
Three plausible mechanisms, in decreasing order of confidence:

1. **GAT output distribution shift.** Adding a coarse-scale
   subgraph + scale embedding changes the marginal distribution of
   per-fine-node activations. The frozen v3.37 decoder absorbs the
   shift OK at pixel level (it was trained on a similar enough
   distribution from the v3.37 GAT). The retrained decoder
   re-fits per-window logits but loses the implicit pixel-coverage
   regularisation from the earlier curriculum.

2. **Coarse messages hurt small-object pixel localisation.** GCs
   are sub-512-px structures. Aggregating coarse-scale features
   into the fine-node representation may help "is this region
   positive" (Stage 1 F1 +2.9pt) but blur the per-pixel boundary
   the decoder needs. GC pixel-Dice drops the most (0.868 → 0.448
   in paired, 0.868 → 0.793 in frozen) — consistent with this.

3. **5-hop GAT was already saturated.** A 5-hop receptive field on
   a 4-conn 256-px graph reaches ±2.5 mm, already wider than a
   single 512-px patch. The bipartite topology may add no
   structural information that the existing graph couldn't reach.

## Status of work products

- ✅ Bipartite multi-scale data path implemented and verified
  (`multiscale_dataset.py`, `multiscale_stage1_model.py`).
- ✅ Multi-scale Stage 1 trainer
  (`train_gars_stage1_multiscale.py`) — works, ckpts saved with
  `model_class: "MultiScaleGraphTLSDetector"`.
- ✅ `eval_gars_cascade.py`, `train_gars_region.py`,
  `region_dataset.py` all auto-detect multi-scale Stage 1 ckpts and
  build the bipartite payload. Dropping slides without coarse zarr
  is graceful (warned, removed from windows).
- ✅ `run_chain_5fold_cascade_v360.sh` exists but is **NOT**
  launched per decision rule.
- 🔁 **Recommended next experiment** (not done): instead of paired
  Stage 2 retrain, try v3.60 with the **HeteroGAT (Option H2)**
  variant — separate GATv2Conv per edge type. If single-scale GAT
  saturated on a 4-conn graph, the typed edges may give the model
  a way to use coarse messages without diluting the fine-node
  representation. Defer.

## Implementation files (already on disk)

### New
- `recovered/scaffold/multiscale_dataset.py` — bipartite graph
  builder (`add_multi_scale`, `MultiScaleSlideDataset`,
  `build_multiscale_inputs_np` for the cascade-eval / region path).
- `recovered/scaffold/multiscale_stage1_model.py` —
  `MultiScaleGraphTLSDetector`.
- `recovered/scaffold/train_gars_stage1_multiscale.py` — multi-scale
  Stage 1 trainer (reuses v3.37 helpers + checkpoints with
  `model_class: "MultiScaleGraphTLSDetector"`).
- `recovered/scaffold/configs/stage1/model/gatv2_5hop_multiscale.yaml`
  — same hyperparams as `gatv2_5hop`.

### Modified
- `recovered/scaffold/eval_gars_cascade.py` — auto-detect multi-scale
  Stage 1 ckpt; build per-slide bipartite payload during cascade eval;
  slice fine-only ctx for Stage 2.
- `recovered/scaffold/train_gars_region.py` — `load_stage1_frozen`
  auto-detects multi-scale (presence of `scale_embed.weight` or
  `model_class == "MultiScaleGraphTLSDetector"`).
- `recovered/scaffold/region_dataset.py` — precompute branches on
  `_is_multi_scale`; per-slide bipartite payload built via
  `build_multiscale_inputs_np`; only fine-node ctx is cached. Slides
  whose coarse zarr is missing are dropped.

## Known risks

1. **Coarse zarr coverage**: confirmed 1007 slides covered (335 BLCA
   + 298 KIRC + 374 LUSC). The training slide split (480 train + 124
   val for fold 0) is fully within this set.
2. **GAT output distribution shift**: confirmed real and large (see
   above). The retraining experiment isolates this from the
   underlying topology gain.
3. **Stage 1 multi-scale may genuinely not improve cascade** even
   when paired with a fresh Stage 2 — 5-hop GAT on a 4-conn 256-px
   graph already reaches a 5-hop receptive field of ±2.5 mm in
   slide pixels, which is already wider than a single 512-px patch.
   The decision rule above is the gate.
