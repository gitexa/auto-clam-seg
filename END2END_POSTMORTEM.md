# End2end cascade training — postmortem + new-strategy proposal

Date: 2026-05-14

## TL;DR

The 2026-05-13 sprint ran every end2end cascade training strategy the
previous plan proposed (aux-loss positive+negative, hard-negative-only,
Stage 2 retrain on the aux-trained Stage 1, multi-scale Stage 1) plus a
TTA control. All variants are NEGATIVE on the production patch-grid mDice
metric vs v3.7 (fold-0 mDice 0.737):

| Variant | Hypothesis | Fold-0 result vs v3.7 | Verdict |
|---|---|---|---|
| **v3.9** Strategy 1 (aux pos+neg, w=0.5) | aux signal steers Stage 1 toward "useful" patches | mDice 0.737 → 0.688; Stage 1 F1 0.56 → 0.66 | **NEG** |
| **v3.10** Strategy 1b (hard-neg only, w=0.3) | only push DOWN on FP-fired patches | mDice 0.737 → 0.667 (patch-grid) / 0.832 → 0.896 (pixel-agg) | NEG patch / **POS pixel** |
| **v3.46** Strategy 2 (retrain Stage 2 on v3.9) | re-align Stage 2 to v3.9's selection | mDice 0.737 → 0.628; GC 0.86 → 0.69 | **STRONG NEG** |
| **v3.60** multi-scale Stage 1 + paired Stage 2 | bipartite 256+512 px context | mDice 0.737 → 0.682; pixel-agg GC 0.85 → 0.48 | **STRONG NEG** |
| TTA 2x (control) | flip-average Stage 2 inference | patch-grid +0.008; pixel-agg −0.032 | NEUTRAL |

This document is the postmortem on **why** none of these end2end recipes
worked, plus a proposal for the kind of recipe that should actually work
given the observed failure mechanism.

---

## 1. Failure-mode analysis

### v3.9 — Strategy 1 (aux loss with pos+neg disagreement labels)

**What changed**: fine-tune v3.8 Stage 1 with `aux_loss_weight = 0.5`,
applying BCE against per-patch "did v3.37 Stage 2 succeed on this patch"
labels (52,638 patches: 34,538 confirmed positives + 12,522 FP-fired
negatives + 5,550 wasted-attention negatives + 28 missed positives). The
v3.37 Stage 2 stays frozen at inference.

**Empirical signature**:

| | v3.8 (baseline) | v3.9 | Δ |
|---|---|---|---|
| Stage 1 own F1 | ~0.56 | 0.66 | **+0.10** |
| sel% (fold-0 fullcohort) | ~0.5% | ~0.5% | unchanged |
| Cascade patch-grid TLS | 0.613 | 0.569 | −0.04 |
| Cascade patch-grid GC | 0.861 | 0.807 | −0.05 |
| TLS-FP rate | 31.7% | 31.7% | unchanged |

**Mechanism**: Stage 1 improved on its own metric (F1) but the cascade
*regressed*. The aux labels mark patches v3.37 *already* segments well —
i.e., patches where Stage 2's behavior is most consistent and dominated
by easy structural features. Stage 1 shifted its selection distribution
toward those patches. v3.37 (frozen at inference) was originally trained
on v3.8's *wider* selection; restricting its inference-time inputs to a
narrower subset doesn't help — Stage 2's strength is generalization
across the hard cases too, and those are now under-represented.

The Stage-1-improved-but-cascade-worsened result is exactly the
diagnostic signature of distribution shift between Stage 2 at training
and inference time.

### v3.10 — Strategy 1b (hard-negative-only aux loss)

**What changed**: same as v3.9 but the aux CSV is filtered to **only**
the 12,522 `fp_s2_fired` patches (no positive aux signal),
`aux_loss_weight = 0.3`.

**Empirical signature**:

| | v3.8 | v3.10 | Δ |
|---|---|---|---|
| Stage 1 own F1 | ~0.56 | 0.49 | **−0.07** |
| sel% | ~0.5% | 0.21% | **−58%** (much more conservative) |
| Cascade patch-grid mDice | 0.737 | 0.667 | −0.07 |
| Cascade patch-grid TLS | 0.613 | 0.476 | **−0.14** |
| Cascade pixel-agg mDice | 0.832 | 0.896 | **+0.06** |
| Cascade pixel-agg TLS | 0.816 | 0.900 | **+0.08** |

**Mechanism**: pure FP-suppression pushed Stage 1 over-conservative — it
fires on only 0.21% of patches (vs 0.5% baseline). Recall collapses,
per-firing precision rises. This moves the cascade along the
precision/recall Pareto curve without expanding it: patch-grid (which
equally weights all positive slides) suffers because many positive
slides now get 0 predictions; pixel-agg (which pools all pixels) wins
because the few predictions made are very accurate per pixel.

v3.10 is **useful as a high-precision deployment variant** (pixel-agg
TLS Dice 0.90 means almost every predicted TLS pixel is correct) but
doesn't beat v3.7 on the primary cascade metric.

### v3.46 — Strategy 2 (retrain Stage 2 on v3.9's patch selections)

**What changed**: keep v3.9 Stage 1 (already fine-tuned with aux loss);
retrain Stage 2 from scratch (RegionDecoder, same v3.37 architecture and
hyperparameters) so it specializes to v3.9's patch distribution.

**Empirical signature**:

| | v3.7 (baseline) | v3.46 | Δ |
|---|---|---|---|
| Patch-grid mDice | 0.737 | 0.628 | −0.11 |
| Patch-grid TLS | 0.613 | 0.568 | −0.04 |
| Patch-grid **GC** | **0.861** | **0.687** | **−0.17** (catastrophic) |
| Pixel-agg mDice | 0.832 | 0.661 | −0.17 |
| Pixel-agg GC | 0.849 | 0.614 | **−0.24** |

**Mechanism**: Stage 2 over-fit to v3.9's "easy-patch" selection
distribution and lost discrimination on the hard cases that originally
taught it the GC-vs-TLS distinction. GC takes the biggest hit because
GC requires finer-grained pixel discrimination than TLS, and that
discrimination is learned from the harder patches that v3.9 no longer
selects.

This is the cleanest demonstration that the cascade is a **tightly
coupled system**: one-sided fine-tunes (Stage 1 → Stage 2) break the
joint optimum that sequential training had reached.

### v3.60 — multi-scale Stage 1 + paired Stage 2

**What changed**: replace Stage 1 with `MultiScaleGraphTLSDetector` —
the same GATv2-5hop body, but takes a combined fine (256-px) + coarse
(512-px UNI v2) feature graph with bipartite containment edges and a
learnable per-scale embedding (initialized to zero so the model starts
identical to v3.8). Paired Stage 2 retrained on v3.60's selections.

**Empirical signature**:

| | v3.7 | v3.60 | Δ |
|---|---|---|---|
| Patch-grid mDice | 0.737 | 0.682 | −0.06 |
| Patch-grid GC | 0.861 | 0.771 | −0.09 |
| Pixel-agg mDice | 0.832 | 0.610 | −0.22 |
| Pixel-agg GC | 0.849 | 0.479 | **−0.37** (catastrophic) |

**Mechanism**: the bipartite 256+512 px channel adds *capacity* but no
new *supervision signal* — the model has more parameters to fit the
same HookNet labels. The zero-init scale embed differentiates during
training, but apparently in a direction that hurts cascade Dice. This
is a different failure mode from v3.9/v3.46 (no aux signal involved) —
it's a capacity-without-signal problem.

The diagnostic is similar to v3.46 though: the *paired* Stage 2 was
retrained on v3.60's selections, and again over-fit to that distribution.
A multi-scale Stage 1 paired with v3.7's frozen Stage 2 might do better
(not tested, low priority given the strong negative).

---

## 2. Root cause synthesis

### Cause 1: the aux signal can be gamed by Stage 1

"Stage 2 succeeded on this patch" is **not a property of the patch** —
it is a property of the currently trained Stage 2 model. When Stage 1 is
fine-tuned to maximize aux-signal agreement, it learns to **pick
patches Stage 2 already handles**. This narrows the selection distribution
without improving the cascade's joint capability — Stage 2 (frozen) sees a
subset of its original training distribution and loses the cases it was
relying on for generalization.

This is a classical **distribution-shift exploitation** failure. The fix
cannot be in the aux signal itself; it has to come from changing what
Stage 1 *cannot* see during training.

### Cause 2: HookNet patch labels are already near-optimal

BCE on HookNet patch labels gives near-optimal patch selection for the
patch-grid mDice metric we care about. The aux signal can only displace
Stage 1 away from that optimum — it cannot add information the HookNet
labels don't already encode. The 31.7% TLS-FP residual is genuine model-GT
disagreement, not addressable by re-weighting the existing supervision.

This is why **none** of the aux-loss variants beat the baseline on
patch-grid: there is no information in "Stage 2 succeeded here" that
isn't already in "HookNet says this patch contains TLS." If anything, the
aux signal is *noisier* (it depends on Stage 2's stochasticity).

### Cause 3: the joint loss surface is non-convex on coupled fine-tuning

v3.46 demonstrated directly that retraining Stage 2 on a perturbed
Stage 1 collapses GC discrimination. The sequential-training joint
optimum (v3.8 + v3.37) is a narrow basin: leave it in *either* direction
(perturb Stage 1 or retrain Stage 2 on a perturbed Stage 1) and joint
capability falls off rapidly. The cascade architecture has more degrees
of freedom than the HookNet supervision can constrain, so the v3.7
solution sits at a fortunate but fragile equilibrium.

---

## 3. Why known end2end recipes don't apply here

**Strategy 3 (Gumbel-topK joint training)** would suffer the same Cause 1
problem. Gumbel-topK makes the selection differentiable, but the Stage 2
pixel loss still *only exists on the selected K patches*. Stage 1
retains the incentive to pick the K patches where Stage 2 already does
well — exactly v3.9's failure mode, just with smoother gradients. Adding
a Gumbel temperature schedule doesn't change that incentive — only the
differentiability of the sampling.

**REINFORCE / policy-gradient end2end** has the same issue: the reward
signal is Stage 2 quality on selected patches, which can be maximized
by selecting easy patches.

**Joint fine-tuning of both stages on the existing labels** would also
exhibit Cause 3: starting from (v3.8, v3.37) and unfreezing Stage 1 would
just let Stage 1 drift toward easy patches (Cause 1) and Stage 2 follow
(Cause 3 in reverse). It's likely to converge to a worse local optimum.

The literature recipes (Mask R-CNN style end-to-end, top-K differentiable
selection, RL with sparse rewards) all share the structural assumption
that *training-time and inference-time selection are the same*. For a
two-stage architecture where Stage 2's strength is *broad generalization*,
this assumption breaks the cascade.

---

## 4. Proposal A — "Stochastic-K joint training" (recommended)

The diagnosis points to a clear principle: **at training time Stage 2
must see a wider patch distribution than it sees at inference**. The
training-time selection cannot be a deterministic function of Stage 1's
current output (Cause 1), and joint training cannot start from sequential
local optima (Cause 3).

### Recipe

- **Initialization**: Stage 1 and Stage 2 both randomly initialized. Do
  NOT load v3.8 / v3.37 weights.
- **Optimizer**: one Adam optimizer over both parameter sets, common LR
  schedule (e.g. warm up 3 epochs, cosine decay over 30).
- **Training-time selection**: stochastic sampling with exploration noise.
  Each patch `i` is selected for Stage 2 with probability:
  ```
  p_i = clamp(sigmoid(s1_logit_i) + ε_explore · 𝒩(0, 1), [0.05, 0.95])
  select_i ~ Bernoulli(p_i)
  ```
  where `ε_explore` is annealed: 0.5 at epoch 0, 0.05 at the end of
  training. This guarantees Stage 2 sees both Stage-1-low and Stage-1-high
  patches in every epoch — Stage 2's gradient covers the full patch
  distribution.
- **Inference-time selection**: hard threshold at 0.5, unchanged from v3.7.
- **Loss**:
  ```
  L = w_s1 · BCE(s1_logits, hooknet_patch_labels)
    + w_s2 · (CE+Dice)(s2_logits[selected], hooknet_tile_masks[selected])
    + w_ent · entropy_reg(sigmoid(s1_logits))
  ```
  Recommended weights: `w_s1 = 1`, `w_s2 = 1`, `w_ent = 0.01` (just
  enough to prevent Stage 1 collapse to all-0 or all-1). Entropy reg is
  on the per-slide *average* probability, not per-patch — penalizes
  degenerate average sel% only.
- **No gradient through the sampler**. The Bernoulli mask is a stop-grad
  multiplier on the Stage 2 loss. Stage 1's gradient comes only from its
  own BCE term + entropy reg. This is a deliberate design choice: we
  want Stage 1 to learn HookNet labels, not to learn "which patches
  Stage 2 happens to like".

### Why it should work

- **No Cause 1**: the training-time selection is stochastically perturbed,
  so Stage 1 cannot game it. Stage 2 sees a representative cross-section
  of patches every epoch.
- **No Cause 2 violation**: Stage 1 is trained on HookNet labels (same as
  v3.8), inheriting the near-optimal signal. The entropy reg just keeps
  it from collapsing during joint training.
- **No Cause 3**: trained from scratch jointly, both stages co-adapt
  rather than one fine-tuning into the other's pre-trained basin.

### Predicted behavior

- Stage 1's converged F1 should match or slightly exceed v3.8's (~0.56).
  If it's much higher (~0.66 like v3.9), entropy reg failed.
- Stage 2's converged training Dice (on stochastically-selected patches)
  should match or exceed v3.37's on v3.8's deterministic selections —
  Stage 2 has effectively *more* training signal (more diverse patches).
- Inference-time cascade mDice should match v3.7's 0.737 fold-0 at
  minimum; ideally 1-3 pp better via Stage 2's broader generalization.

### Cost

- **Implementation**: ~3-4 h (new joint trainer, sampler, loss).
- **Compute**: ~12-15 h fold-0 (both stages from scratch jointly).
- **5-fold CV** (optional): +60 h.

### Failure modes

- Joint training instability: Stage 1 logits collapse early before Stage 2
  has learned anything. Mitigation: low w_ent → 0.005, larger
  ε_explore_start → 0.7.
- Training-inference gap: Stage 2 trained on noisier selections may
  produce noisier inference outputs. Mitigation: late-training fine-tune
  with ε_explore=0 to "settle" Stage 2 on Stage 1's actual selections.

---

## 5. Proposal B (cheaper fallback) — "Dense Stage 2 training"

If joint-from-scratch is too expensive, address only Cause 1 + Cause 3:
broaden Stage 2's training distribution without touching Stage 1.

### Recipe

- Keep v3.8 Stage 1 frozen, as in v3.7.
- Stage 2 trains on the **union of**:
  - (a) Stage 1's selected patches (current behavior)
  - (b) *Plus* a random extra fraction (e.g. 50%) of additional patches
    per slide, sampled uniformly from all patches in the slide
- Loss unchanged (pixel CE + Dice per HookNet tile).
- At inference: cascade gate as before (Stage 1 v3.8 selects, retrained
  Stage 2 segments).

### Why it should work

The "dense" supervision exposes Stage 2 to patches Stage 1 *wouldn't*
select — including (a) hard negatives (Stage 1 low-confidence but maybe
positive) and (b) clear negatives (Stage 1 confident-negative, mostly
tissue background). Stage 2 learns to handle both, so its inference-time
output on Stage-1-selected patches is more robust.

### Cost

- **Implementation**: ~1 h (dataset option + trainer flag).
- **Compute**: ~3-4 h Stage 2 retrain + 25 min eval.

### When to prefer B over A

- If A's failure-mode risk (joint instability) is unacceptable.
- If compute is tight (< 5 h available).
- If the goal is incremental improvement, not a fundamental
  re-architecting.

---

## 6. Decision criterion

When either proposal is implemented and run, the success gate is the
same as before: **cascade fold-0 patch-grid mDice > 0.737** (v3.7
baseline). If neither proposal beats 0.737, the cascade architecture is
at a genuine local optimum for this metric on this data, and further
gains require either:

- A different metric (e.g., pixel-agg, where v3.10 already wins)
- New annotation effort (pathologist review of the 13 high-confidence FP
  slides per fold)
- A new feature backbone (e.g., trident at higher resolution)

---

## 7. Critical files (for implementation, when this gets executed)

### Proposal A
- `recovered/scaffold/train_gars_cascade_joint.py` (NEW) — combined trainer
- `recovered/scaffold/configs/cascade/config_joint_v1.yaml` (NEW)
- `recovered/scaffold/region_decoder_model.py` — already supports being
  called with arbitrary patch sets; no change needed
- `recovered/scaffold/train_gars_stage1.py:167-214` — reuse loop pattern

### Proposal B
- `recovered/scaffold/region_dataset.py` — add `dense_random_extra_frac`
  option
- `recovered/scaffold/train_gars_region.py` — pass it through

---

## 8. Status

- This document records the state of the end2end research direction as
  of 2026-05-14.
- The four NEGATIVE variants (v3.9, v3.10, v3.46, v3.60) are
  fully-documented closed loops; no further sub-variants worth trying
  within the aux-loss family.
- Proposal A and B are unexecuted recommendations. The user decides if
  and when to allocate compute.
- v3.7 (v3.8 + v3.37 + production post-proc) remains the production
  champion. 5-fold CV: mDice 0.658 ± 0.040 (patch-grid). v3.10 is a
  derived high-precision pre-screen variant.

See also:
- `FP_REDUCTION_RESULTS.md` — full per-strategy result tables
- `V3.7_5FOLD_CV_RESULTS.md` — production champion 5-fold CV
- `V3.7_FP_5FOLD_ANALYSIS.md` — honest 38.8% 5-fold TLS-FP rate
- `NEXT_EXPERIMENT_QUEUE.md` — other unexplored research directions
