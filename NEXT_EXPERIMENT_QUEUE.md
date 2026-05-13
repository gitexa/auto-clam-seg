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

## ~~2. Hard-negative-only Stage 1 fine-tune~~ — RUN, NEGATIVE on primary (2026-05-13)

**v3.10**: trained with aux loss on only the 12,522 fp_s2_fired patches
(no positive aux), `aux_loss_weight=0.3`. Result on fold 0:

- patch-grid mDice 0.667 (vs 0.737 baseline, −0.07)
- pixel-agg mDice 0.896 (vs 0.832, +0.06)
- Stage 1 fires on 58% fewer patches (sel% 0.5 → 0.21)
- Stage 1 own F1 dropped to 0.49 (vs ~0.56 baseline)

Pushes Stage 1 toward higher precision / lower recall — useful as a
**high-precision pre-screen variant** but doesn't beat v3.7 on the
primary patch-grid metric. Aux-loss family fully closed (no remaining
variants to try). See `FP_REDUCTION_RESULTS.md` Strategy 1b section.

## ~~3. Stage 2 TTA (test-time augmentation)~~ — RUN 2×, NEUTRAL (2026-05-13)

Implemented `tta=2x` (identity + horizontal flip averaging). Fold-0 result:
patch-grid +0.008 mDice (in line with TTA expectations) but pixel-agg
**−0.032 mDice**. Not a clean win — averaging blurs sharp pixel-precise
edges. 5-fold + 4×/8× TTA skipped. Knob preserved in
`eval_gars_cascade.py` for future use. See `FP_REDUCTION_RESULTS.md`
Strategy C section.

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
   alignment with stated arch preferences. Strongest remaining
   hypothesis.
2. If user wants quick wins: **TTA (3)** — defensive, ~2.5h, likely
   gives +0.5 mDice for free.

Aux-loss family fully buried (Strategies 1, 1b, 2 all negative on
patch-grid). End2end research direction closed; only architectural
changes remain.

Production v3.7 cascade is shippable as-is. v3.10 is a derived
**high-precision variant** for pre-screen deployment.
