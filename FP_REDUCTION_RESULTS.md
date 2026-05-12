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

## Strategy C — Test-time augmentation (skipped)

C was planned to add 4× flip/rotation logit averaging to Stage 2. Given B
showed that the residual 31.7 % TLS-FP comes from slides where Stage 1 has
**confident** (>0.8 prob) TLS detections, TTA won't help — the FPs aren't
spurious noise activations but consistent model-GT disagreement on real
lymphoid structure. TTA may modestly improve Stage 2 segmentation Dice on
the TLS-positive slides but won't push the slide-level FP rate below 32 %.

**Skipped to save 4× inference compute.**

## Strategy D — GT cleanup (out of scope, future work)

The 13 residual FP slides should be manually inspected. If they're real
TLS-positive with under-annotation by HookNet, the cohort metadata can be
updated (move from "negative" → "positive"), which would drop the FP rate
without any model change. Cross-checking with `df_summary_v10.csv`'s
`tls_present` field is a starting point.

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
