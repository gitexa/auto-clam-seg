# Next experiments — sprint 2026-05-13 handoff

The end2end cascade training direction (Strategies 1/2/3 from
`plans/can-you-start-the-cuddly-minsky.md`) is closed:
- Strategy 1 (v3.9 aux-loss Stage 1): NEGATIVE
- Strategy 2 (v3.46 Stage 2 retrained on v3.9 selections): STRONG NEGATIVE
- Strategy 3 (Gumbel-topK) would propagate the same misaligned signal — skipped
- GT cleanup hypothesis: REJECTED (cohort meta agrees with HookNet)
- Production v3.7 5-fold CV: mDice 0.658 ± 0.040

GPU is now idle. Queued experiments below, ranked by expected value. None
launched autonomously — each has either compute cost ≥10h or
implementation risk that warrants user direction.

## 1. Multi-scale Stage 1 (bipartite 256+512 px) — HIGHEST EV

**Hypothesis**: Stage 1's failures (LUSC TLS at 0.529 ± 0.024) come from
narrow context. A bipartite graph between 256-px nodes and their parent
512-px tiles gives Stage 1 broader morphological context.

**Architecture** (per user preference for bipartite/multi-scale):
- Two node sets: 256-px nodes (current) + 512-px parent nodes (4 children
  per parent at most).
- Bipartite edges: each 256-node ↔ its parent 512-node.
- 512-px features (1536-d UNI v2, on disk at
  `/home/ubuntu/ahaas-persistent-std-tcga/data/tcga-{blca,kirc,lusc}/representations_tif_trident/20x_512px_0px_overlap/features_uni_v2_grandqc_zarr/`).
- One bipartite GAT pass propagates 512→256 once before the existing
  GATv2-5hop runs on the 256 graph.

**Cost**: ~3-4h impl (bipartite layer + dataset cache for parent lookup) +
~2h fold-0 Stage 1 train + ~3h fold-0 Stage 2 retrain + ~10 min eval =
**~10h total**.

**Decision criterion**: Cascade fold-0 mDice > 0.737 (current fold-0
baseline). If yes → green-light 5-fold CV (~30h additional).

## 2. Hard-negative-only Stage 1 fine-tune — MEDIUM EV

**Hypothesis**: A *cleaner* version of Strategy 1 — push Stage 1 logits
DOWN on the 12,522 `fp_s2_fired` patches (no positive aux signal).
Avoids the "easy patches" pull mode that doomed v3.9.

**Concern**: 31.7% TLS-FP floor is annotation noise — pushing Stage 1
down on real lymphoid structures the cohort calls "TLS-negative" may
just teach Stage 1 to suppress real positives.

**Cost**: ~3h Stage 1 fine-tune + 10 min eval = ~3h. Use existing
`stage2_disagreement_labels_for_fold0.csv`, filter to
`reason='fp_s2_fired'` only, set `aux_loss_weight=0.3`. Single fold-0
proof-of-concept.

## 3. Stage 2 TTA (test-time augmentation) — LOW EV but CHEAPEST

**Hypothesis**: 4× flip/rotation Stage 2 inference → averaged logits →
~0.5–1 mDice gain "for free". Doesn't reduce TLS-FP (per Strategy C
analysis in `FP_REDUCTION_RESULTS.md`) but may improve segmentation
quality on TLS-positive slides.

**Cost**: ~30 min impl + ~25 min × 5 folds = **~2.5h**. No retraining.

## 4. Higher-resolution Stage 2 input — MEDIUM EV but HIGH RISK

**Hypothesis**: Current Stage 2 uses 256-px tiles upsampled to 768-px.
Direct 512-px tile input could preserve more detail.

**Cost**: Major surgery (~4h impl: new dataset, new positional grid,
re-validate post-proc) + ~5h Stage 2 retrain + ~10 min eval = **~10h**.
Risk: tile-clustering logic in RegionDecoder may break.

## 5. Pathologist re-annotation of the 13 confidently-FP slides — OUT OF SCOPE

The only path to push TLS-FP below the 31.7% annotation noise floor.
Domain-expert task, not a model change.

---

## Recommended path

1. If user has ~10h GPU budget: **Multi-scale Stage 1 (1)** — best
   alignment with stated arch preferences.
2. If user wants quick wins: **TTA (3)** — defensive, ~2.5h, likely
   gives +0.5 mDice for free.
3. If user wants to fully bury aux-loss family: **Hard-neg-only (2)** —
   ~3h to confirm negative.

Production v3.7 cascade is shippable as-is.
