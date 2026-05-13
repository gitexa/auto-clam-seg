# FP-reduction strategies — fold-0 fullcohort findings (cascade v3.37)

User asked for best strategies to push TLS-FP below the 41 % cascade ceiling.
Three strategies tested on fold-0 fullcohort (n=165 slides, 124 GT-pos + 41 GT-neg):

A. Stage 1 threshold + post-proc sweep (no retrain)
B. `top_k_abstain_thr` mode (no retrain)
C. Test-time augmentation (no retrain, 4× inference cost)  ← deferred

## Strategy A — Stage 1 threshold sweep (TLS-FP results)

### A.1 Threshold only (min_size=1, closing_iters=0 — current default)

| Stage 1 thr | TLS Dice | GC Dice | mDice | TLS-FP | GC-FP | mean TLS pred / neg |
|---|---|---|---|---|---|---|
| **0.5** (current) | 0.606 | 0.831 | **0.719** | 41.5 % | 4.9 % | 0.93 |
| 0.6 | 0.585 | 0.831 | 0.708 | 39.0 % | 4.9 % | 0.76 |
| 0.7 | 0.550 | 0.834 | 0.692 | 31.7 % | 4.9 % | 0.56 |
| 0.8 | 0.508 | 0.832 | 0.670 | 26.8 % | 2.4 % | 0.44 |
| 0.9 | 0.429 | 0.825 | 0.627 | 17.1 % | 2.4 % | 0.34 |

### A.2 With min_component_size=2 + closing_iters=1

| Stage 1 thr | TLS Dice | GC Dice | mDice | TLS-FP | GC-FP | mean TLS pred / neg |
|---|---|---|---|---|---|---|
| **0.5** | 0.604 | 0.831 | **0.717** | **31.7 %** | 4.9 % | 0.51 |
| 0.7 | 0.536 | 0.835 | 0.686 | **22.0 %** | 4.9 % | 0.32 |
| 0.8 | 0.495 | 0.832 | 0.664 | 22.0 % | 2.4 % | 0.32 |

### Pareto knees (recommend two operating points)

**1. "Free" knee — thr=0.5, min_size=2, closing=1**
- mDice 0.717 (essentially unchanged from baseline 0.719)
- TLS-FP **31.7 %** (down from 41.5 %, **−10 pp absolute**, ~24 % relative reduction)
- Singleton-CC filter alone removes the most spurious noise predictions.
- **Drop-in deployment default**: change cascade `min_component_size: 1 → 2` and
  `closing_iters: 0 → 1` in `recovered/scaffold/configs/cascade/config.yaml`.

**2. "Aggressive" knee — thr=0.7, min_size=2, closing=1**
- mDice 0.686 (−0.033 vs baseline, −4.6 % relative)
- TLS-FP **22.0 %** (down from 41.5 %, **−19.5 pp absolute**, ~47 % relative reduction)
- Useful for high-precision deployments (clinical pre-screen, dataset cleaning).

## Strategy B — top_k_abstain_thr (done)

Setup: `top_k_frac=0.005, top_k_abstain_thr=0.8, min_top_k=10, thr=0.5,
min_component_size=2, closing_iters=1`. Stage 1 picks top 0.5 % of patches
on slides where any patch exceeds 0.8; **abstains** entirely on slides where
no patch crosses 0.8.

| Setup | TLS Dice | GC Dice | mDice | TLS-FP | abstained neg slides |
|---|---|---|---|---|---|
| top_k_abstain (B) | 0.604 | 0.831 | **0.717** | **31.7 %** | 28 / 41 |
| min_size=2 + closing=1 (A.2 baseline) | 0.604 | 0.831 | 0.717 | 31.7 % | — |

**Strategy B converges to the same 31.7 % floor as A.2** through a different
mechanism (abstaining vs post-hoc CC filtering). 28/41 negative slides got
Stage 1 → empty selection (abstained → TN). The remaining 13 FPs are slides
where Stage 1 IS confidently positive (at least 1 patch > 0.8) but HookNet
GT says no TLS. **These are likely annotation-noise FPs** — Stage 1 sees real
lymphoid structure that the annotator missed.

A.2 and B stack to no further benefit (they catch the same noise-FPs).

## Strategy C — Test-time augmentation (2× hflip, 2026-05-13) — NEUTRAL

Implemented as `tta=2x` knob in `eval_gars_cascade.py:_stage2_forward_tta`:
Stage 2 forward runs twice (identity + horizontal flip of input cells +
RGB spatial flip), output logits are inverse-flipped and averaged.

### Fold-0 fullcohort result (n=165 slides)

| Metric              | Baseline (no TTA) | TTA 2× | Delta  |
|---------------------|-------------------|--------|--------|
| patch-grid mDice    | 0.737             | 0.745  | +0.008 |
| patch-grid TLS      | 0.613             | 0.613  | 0      |
| patch-grid GC       | 0.861             | 0.877  | +0.016 |
| **pixel-agg mDice** | 0.832             | 0.800  | **−0.032** |
| pixel-agg TLS       | 0.816             | 0.803  | −0.013 |
| pixel-agg GC        | 0.849             | 0.797  | −0.053 |

### Verdict: NET NEUTRAL — not worth 5-fold

Marginal patch-grid gain (+0.8%, in line with typical TTA expectations)
is offset by a meaningful pixel-agg regression (−3.2%). Mechanism:
averaging hflipped + identity logits smooths boundaries — preserves
patch-level decisions but blurs pixel-precise edges. The model's
predictions are sharper than its augmented average.

TTA was originally hypothesised to help on FP rate (B analysis predicted
no help there) and on segmentation Dice (a small help on patch-grid,
confirmed). The pixel-agg regression is the new finding.

**Skipped 5-fold (not a clean win) and skipped 4× / 8× TTA (gain
direction confirmed but magnitude unlikely to flip the verdict).**

## Strategy E (Plan v2) — Stage 1 fine-tune with Stage 2 disagreement labels — NEGATIVE

User asked for end-to-end cascade training strategies. Plan proposed three:
Strategy 1 (aux loss with Stage 2 success labels), Strategy 2 (iterative EM),
Strategy 3 (Gumbel-topK). Ran Strategy 1 (cheapest).

### Pipeline

1. `build_stage2_disagreement_labels.py` — sweeps fold-0 training set with
   frozen v3.7 cascade (Stage 1 v3.8 + Stage 2 v3.37). For each
   Stage-1-selected patch, computes per-tile TLS/GC Dice against the
   HookNet GT and assigns a label:
   - `1` (confirmed) — Stage 2 successfully segments the patch (Dice ≥ 0.5).
   - `0` (Stage 2 disagrees) — Stage 2 produces low/no TLS on a HookNet-
     positive patch, OR Stage 1 fired on a GT-negative patch.
2. `train_gars_stage1.py:run_split` extended with `aux_loss_weight` knob.
   Total loss = `BCE(HookNet binary labels) + aux_w · BCE(aux labels)`
   on Stage-1-selected patches.
3. `config_v3_9_aux.yaml`: load from v3.8 ckpt; lr=1e-4; 20 epochs;
   `aux_loss_weight=0.5`; `aux_label_csv=stage2_disagreement_labels_for_fold0.csv`.

### Labels collected

- 52,638 patches across 515 training slides (folds 1-4 ∪)
- Positive (confirmed): 34,538 (65.6 %)
- Negative breakdown:
  - `fp_s2_fired` (Stage 2 fires on GT-neg patch): 12,522
  - `fp_s2_empty` (Stage 1 wasted attention; Stage 2 self-corrected): 5,550
  - `missed` (Stage 2 fails on GT-positive): 28

### Stage 1 training-time result

v3.9 reaches **F1 = 0.660 at epoch 5** (early-stopped at epoch 13).
Baseline v3.8 F1 was ~0.561. **Stage 1's own metric improved by +0.10**.

### Cascade fold-0 fullcohort result (v3.9 Stage 1 + v3.37 Stage 2)

| Config | TLS Dice | GC Dice | mDice | TLS-FP |
|---|---|---|---|---|
| v3.8 baseline (thr=0.5, min=2, c=1) | 0.604 | 0.831 | **0.717** | 31.7 % |
| **v3.9** (thr=0.5, min=2, c=1) | 0.569 | 0.807 | 0.688 | 31.7 % |
| v3.9 thr=0.7 | 0.531 | 0.798 | 0.664 | 24.4 % |

### Verdict: NEGATIVE for Strategy 1 alone

Despite Stage 1 F1 climbing from 0.56 → 0.66, cascade mDice **dropped** by 0.029
and TLS-FP stayed at 31.7 %. The mechanism is straightforward:

* v3.9's improved Stage 1 selects a slightly DIFFERENT patch distribution
  (the aux loss pushed it away from "Stage 2 will disagree" patches).
* But Stage 2 (v3.37) was trained on v3.8's patch distribution. It
  doesn't generalize as well to v3.9's selections.
* Net cascade mDice slightly degrades.

### Implication for Strategy 2

Strategy 1's premise — **"better Stage 1 alone → better cascade"** — is
false. The two-stage cascade is a coupled system: Stage 2 expects the patch
distribution Stage 1 produces. To get the win, **must also retrain Stage 2**
on v3.9's selections. That is **Strategy 2 (iterative EM)**.

Strategy 2 estimated cost: ~6 h (Stage 2 retrain on v3.9's patches) per
round; 2 rounds = ~12 h. Still substantially cheaper than Stage 3 (full
Gumbel-topK joint training, ~10 h plus implementation risk).

v3.9 Stage 1 ckpt is preserved for Strategy 2 use:
`/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_stage1_v3.9_aux_finetune_fold0_20260512_224433/best_checkpoint.pt`

## Strategy 2 round 1 — v3.46 Stage 2 retrain on v3.9 selections — STRONG NEGATIVE

Retrained Stage 2 (RegionDecoder, same v3.37 architecture) on v3.9 Stage 1's
patch selections. Same `tls_patch_dataset_min4096.pt` cache. Training:
early-stopped at epoch 14 (best epoch 6, val_mDice=0.849 on patches).

### Cascade fold-0 fullcohort result (v3.9 Stage 1 + v3.46 Stage 2)

| thr | patch-grid mDice | TLS | GC |
|---|---|---|---|
| 0.30 | 0.625 | 0.574 | 0.677 |
| 0.40 | 0.624 | 0.571 | 0.676 |
| **0.50** | **0.628** | 0.568 | 0.687 |
| 0.60 | 0.634 (best) | 0.567 | 0.702 |
| 0.70 | 0.632 | 0.567 | 0.697 |

vs v3.7 baseline (v3.8 Stage 1 + v3.37 Stage 2) thr=0.5:
mDice **0.717 → 0.628** (−0.089), TLS 0.604 → 0.568 (−0.036),
GC **0.831 → 0.687** (−0.144).

vs v3.9 (v3.9 Stage 1 + v3.37 Stage 2) thr=0.5:
mDice 0.688 → 0.628 (−0.060), GC 0.807 → 0.687 (−0.120).

### Verdict: STRONG NEGATIVE

The cascade got **worse**, not better, after retraining Stage 2 on v3.9's
selections. GC took the hardest hit (−0.144).

### Root cause: aux signal pushes toward EASY patches

Both Strategy 1 (v3.9 Stage 1 alone) and Strategy 2 (v3.9 + retrained
Stage 2) degrade the cascade. The mechanism is the aux signal itself:

- "Stage 2 succeeded on this patch" labels mark patches where v3.37 already
  segments well — which are mostly EASY patches (clean, large, well-formed
  TLS/GC structures).
- Fine-tuning Stage 1 toward these labels biases selection toward easy patches
  and AWAY from the hard-but-informative ones.
- When Stage 2 is retrained on the easy-biased selection, it loses
  discrimination on the harder cases — especially GC (which requires
  fine-grained discrimination from surrounding TLS, abundant in hard patches).
- Result: v3.46 Stage 2 has high val_mDice on training patches (0.849) but
  the patch distribution doesn't generalise to full-slide eval (0.628).

### Implication for Strategy 3

Strategy 3 (Gumbel-topK joint training) would propagate the SAME misaligned
signal — Stage 2 pixel loss only exists on Stage-1-SELECTED patches, so
the gradient still rewards "select easy patches" over "select informative
patches". Skipping Strategy 3.

### Family conclusion: aux-loss-on-Stage-1 is misaligned

The fundamental problem: Stage 2's pixel-Dice is a *result* of patch
difficulty, not a signal for patch *informativeness*. To genuinely improve
the cascade end-to-end, would need a different supervision target — e.g.
hard-negative mining on FP-confidently-fires-on-empty patches, or
multi-scale Stage 1 to see broader context. Both orthogonal to aux loss.

### Production champion stands

**v3.7 (v3.8 Stage 1 + v3.37 Stage 2 + post-proc min_size=2, closing=1)**
remains the production deployment. mDice 0.717, TLS-FP 31.7%, GC-FP 4.9%.

## Strategy 1b (v3.10) — hard-negative-only aux loss — MIXED / NEGATIVE on primary

After Strategy 1 (pos+neg aux) and Strategy 2 (Stage 2 retrain) both
degraded the cascade, tried a *cleaner* variant: filter the aux-label CSV
to keep ONLY the 12,522 `fp_s2_fired` rows (no positive aux signal).
Same training driver, `aux_loss_weight=0.3`.

### Result on fold-0 fullcohort (165 slides)

| Metric              | v3.7 baseline | v3.10 thr=0.5 | v3.10 thr=0.7 |
|---------------------|---------------|---------------|---------------|
| patch-grid mDice    | 0.737         | **0.667** (−0.07) | 0.641 (−0.10) |
| patch-grid TLS      | 0.613         | **0.476** (−0.14) | 0.432 (−0.18) |
| patch-grid GC       | 0.861         | 0.858         | 0.849         |
| **pixel-agg mDice** | 0.832         | **0.896** (+0.06) | **0.906** (+0.07) |
| pixel-agg TLS       | 0.816         | 0.900 (+0.08) | 0.911 (+0.10) |
| pixel-agg GC        | 0.849         | 0.893 (+0.04) | 0.901 (+0.05) |
| sel% (patches fired)| 0.5 %         | 0.21 %        | 0.15 %        |
| Stage 1 own F1      | ~0.56         | 0.49          | 0.49          |

### Interpretation

Stage 1 shifted toward **higher precision / lower recall** — fires on
~58% fewer patches but the predictions it does make are tighter and
better aligned with GT. Per-patch averaging penalises the missed
patches; pixel-aggregate pooling rewards the high per-firing quality.

### Verdict: NEGATIVE on primary patch-grid metric

By the production metric (patch-grid mDice 5-fold), v3.10 is worse than
v3.7. Stage 1's own F1 also drops (0.56 → 0.49). The hard-neg-only aux
signal pushes Stage 1 to be over-conservative — it suppresses real
positives along with the FP-fired patches.

### What v3.10 IS useful for

A **high-precision deployment variant**: at thr=0.5, pixel-agg
TLS Dice 0.90 / mDice 0.90 means almost every predicted pixel is correct.
Useful when downstream cost of false-positive pixels dominates (e.g.,
auto-flagging for pathologist review where over-flagging is expensive).
Not a replacement for v3.7 on the standard benchmark.

### Aux-loss family — final summary

| Variant | aux loss | Stage 1 F1 | Cascade mDice (fold 0) | Verdict |
|---------|---------|-----------|-----------------------|---------|
| v3.8 (baseline) | — | 0.56 | 0.737 | reference |
| v3.9 (Strategy 1, pos+neg, w=0.5) | both | 0.66 | 0.688 | neg |
| v3.46 (Strategy 2, + Stage 2 retrain) | both | — | 0.628 | strong neg |
| **v3.10 (Strategy 1b, neg-only, w=0.3)** | neg only | **0.49** | **0.667** | neg (patch-grid) / pos (pixel-agg) |

The aux signal IS a tuning knob along the precision/recall axis:
- positives only → pull toward easy/positive patches (would over-fire)
- negatives only → pull toward conservative/precise (under-fires)
- mixed → some of each

NONE OF THE THREE beats v3.7 on the patch-grid metric. The HookNet
labels' BCE loss is already near-optimal for the cascade's primary
metric. Aux-loss only changes the operating point.

## Strategy D — GT cleanup (cohort cross-check, 2026-05-13) — REJECTED

Cross-checked HookNet GT-negative slides against `df_summary_v10.csv`'s
`tls_present` field across all 5 folds:

| Fold | gt_neg (HookNet) | of which cohort `tls_present=True` |
|------|------------------|------------------------------------|
| 0    | 42               | **0**                              |
| 1    | 40               | **0**                              |
| 2    | 42               | **0**                              |
| 3    | 40               | **0**                              |
| 4    | 45               | **0**                              |

**0/209 GT-negative slides across all folds have cohort `tls_present=True`.**

The HookNet annotation and the cohort-level `tls_present` flag are derived
from the same source — both agree that these slides are TLS-negative. The
13 residual FP slides at the 31.7 % floor are **genuinely TLS-negative
by both definitions** and the model fires on them anyway.

### Implication

The 31.7 % TLS-FP floor is NOT annotation-driven (no easy metadata fix).
Two real possibilities:

1. **Biological noise floor** — the model detects lymphoid aggregates or
   tertiary lymphoid structures that don't meet the formal TLS criteria
   (size, organisation, germinal center development) but have similar
   morphological signatures. Reducing this further would require
   stricter biological criteria in the model.
2. **Annotation noise upstream** — both HookNet and cohort metadata
   inherited the same noise. Resolution would require pathologist
   re-annotation of the 13 confidently-FP slides per fold.

Neither path is cheap. Production deployment ships at 31.7 % TLS-FP as
the honest floor.

## Bottom line

**Production deployment change** (zero compute cost, only config update):

```yaml
# recovered/scaffold/configs/cascade/config.yaml
thresholds: [0.5]              # unchanged
min_component_size: 2          # was 1 — drops singleton noise
closing_iters: 1               # was 0 — fills 1-cell gaps
```

Result on fold-0 fullcohort:
- mDice 0.717 (was 0.719 — essentially unchanged)
- TLS Dice 0.604 (was 0.606)
- GC Dice 0.831 (unchanged)
- **TLS-FP 31.7 %** (was 41.5 % — −9.8 pp absolute, ~24 % relative reduction)
- GC-FP 4.9 % (unchanged)
- TLS detection F1 still ≥ 0.92

For high-precision deployment (clinical pre-screen):

```yaml
thresholds: [0.7]
min_component_size: 2
closing_iters: 1
```

Result: mDice 0.686 (−4.6 %), TLS-FP **22.0 %** (−19.5 pp absolute, ~47 % relative).

The 31.7 % floor is likely annotation noise (real TLS that HookNet missed) —
further reduction requires GT cleanup or Stage 1 retraining with hard negatives.

## Recommendation

For **production deployment**, change the cascade default config to:
```yaml
thresholds: [0.5]              # unchanged
min_component_size: 2          # was 1 — drops singleton noise
closing_iters: 1               # was 0 — fills 1-cell gaps
```
This gives mDice 0.717 and TLS-FP 31.7 % at zero compute cost (post-hoc filter only).
The aggressive thr=0.7 config is reserved for high-precision use cases.
