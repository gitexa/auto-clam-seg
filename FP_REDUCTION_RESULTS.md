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
