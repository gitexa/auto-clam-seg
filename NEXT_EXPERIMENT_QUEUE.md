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

## ~~1. Multi-scale Stage 1 (bipartite 256+512 px)~~ — already run as v3.60, STRONGLY NEGATIVE (2026-05-13)

Infrastructure exists (`multiscale_stage1_model.py`, `multiscale_dataset.py`,
`train_gars_stage1_multiscale.py`). v3.60 was trained on May 7 with the
multi-scale Stage 1 (zero-init scale embed + GATv2-5hop over bipartite
graph) + paired Stage 2 retrain. Original eval (May 7) used old
post-proc; re-evaluated 2026-05-13 with production post-proc.

Fold-0 result vs v3.7 baseline:

- patch-grid mDice 0.682 (vs 0.737, **−0.055**)
- patch-grid GC   0.771 (vs 0.861, **−0.090**)
- pixel-agg mDice 0.610 (vs 0.832, **−0.222**)
- pixel-agg GC    0.479 (vs 0.849, **−0.370**)

Strongly negative. The bipartite information channel does not help and
appears to confuse the cascade's GC discrimination. ALL architectural
end2end variants tested are negative on patch-grid.

V3.7 single-scale cascade is genuinely the champion.

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

**All autonomously-testable hypotheses now exhausted (2026-05-13 sprint):**

- Aux-loss family (Strategies 1, 1b, 2): all negative on patch-grid
- TTA 2x: neutral
- Multi-scale Stage 1 (v3.60 re-evaluated with prod post-proc): **strongly negative**
- GT cleanup vs cohort metadata: rejected

The remaining unexplored ideas need user direction:

- **Pathologist re-annotation** of the 13 confidently-FP slides per
  fold. Only path to push TLS-FP below 31.7%. Domain-expert task.
- **Higher-resolution Stage 2** (input at 512 px instead of 256 px
  upsampled). Major surgery: tile-clustering logic, dataset rewrite.
  ~10h GPU + impl risk.
- **Stage 1 architectural rethink**: prior attempts (v3.60 bipartite,
  v3.38 dual-sigmoid Stage 2) all failed. Would need a genuinely new
  idea, not a variation on existing ones.

Production v3.7 cascade is shippable as-is. v3.10 is a derived
**high-precision variant** for pre-screen deployment. End2end and
multi-scale research directions both closed.
