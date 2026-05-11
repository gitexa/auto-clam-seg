# A3 — slide-level gate retrofit (results)

This task: retrofit a small ABMIL/MLP slide-level "is this slide
TLS-positive" gate on top of GNCAF v3.58's predictions, and check
how much of the FP gap to the cascade we can close.

## Gate model

`recovered/scaffold/train_slide_gate.py` — `SlideGateModel`:

* Input: per-slide UNI v2 features `(N, 1536)`, mean-pooled to `(1536,)`.
* Architecture: `LayerNorm + Linear(1536→256) + GELU + Dropout(0.1) + Linear(256→1)`.
* **396 K params** (vs GNCAF's 108 M).
* Loss: `BCEWithLogitsLoss(pos_weight = n_neg/n_pos = 0.347)`.
* Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`.
* Trained 30 epochs on cohort folds 1-4 (648 slides; 481 TLS-pos / 167 TLS-neg), validated on fold 0 (166 slides).

## Slide-level gate quality (fold 0)

* **AUROC**: 0.74-0.77 (epoch-dependent; 0.765 at best).
* **AP**: 0.89-0.90.
* **Best F1 = 0.881** (P=0.839, R=0.927) at τ=0.22 (epoch 1).
* The gate is decent — slightly better than chance but far from
  perfect. Negative-slide recall (i.e. correctly identifying GT-neg
  slides as negative) is the relevant axis here, and the gate
  catches roughly 60-80 % of negatives at the operating points
  evaluated below.

## Effect on v3.58 fold-0 fullcohort FP rate

Apply the gate retroactively: for each slide with `gate_prob < τ`,
zero out `n_tls_pred` and `n_gc_pred` (the gate "overrules" GNCAF
on slides it classifies as TLS-negative).

| τ | n_gated | TLS-FP rate | GC-FP rate | TLS recall (pos slides) | GC recall (gc+ slides) |
|---|---|---|---|---|---|
| 0.10 | 20 | 71 % | 12 % | 93 % | 72 % |
| 0.22 (best F1) | 29 | 59 % | 10 % | 90 % | 69 % |
| 0.30 | 32 | 54 % | 10 % | 89 % | 69 % |
| 0.40 | 36 | **49 %** | **5 %** | 87 % | 67 % |
| 0.50 | 39 | 49 % | 5 % | 85 % | 67 % |
| 0.60 | 45 | 49 % | 5 % | 81 % | 67 % |
| 0.70 | 48 | **46 %** | **2 %** | 79 % | 67 % |

vs ungated:
* No gate: TLS-FP = 95 %, GC-FP = 12 %, TLS recall = 100 %, GC recall = ~88 %.

vs the cascade (no retrofit needed):
* Cascade v3.37 fold-0 fullcohort: **TLS-FP=41 %, GC-FP=5 %, TLS recall ~98 %**.

## Conclusion

The gate retrofit **does close most of the GNCAF FP gap** — TLS-FP
rate drops from 95 % to 46-49 % depending on τ. However:

1. **GNCAF + gate is still strictly worse than the native cascade**
   at every operating point. At τ=0.70 GNCAF + gate hits 46 % TLS-FP
   ≈ cascade's 41 %, but loses ~20 percentage points of TLS recall
   (79 % vs 98 %).
2. **GC behaviour improves modestly** — GC-FP drops from 12 % to
   2-5 %, comparable to cascade's 5 %.
3. The gate adds an extra inference dependency (~400K-param model
   running on every slide before GNCAF).

So: the GNCAF dense pixel-decoder + retrofit gate is a reasonable
*deployable* compromise (cuts FP by 2×, keeps most recall), but the
cascade remains structurally cleaner and stronger.

## What this rules in / out

* **Rules in**: a slide-level gate IS effective — the FP problem is
  partially a pure inference-time decision-rule issue, not solely a
  training-distribution issue. The cascade's gate is doing the work
  that GNCAF needs an external model for.
* **Rules out**: retrofitting a separate gate alone closes the gap
  to the cascade. To match cascade-level performance with a
  GNCAF-style decoder, you'd need a much better slide-level
  classifier (the current 0.77 AUROC is the bottleneck).

## Files

* `recovered/scaffold/train_slide_gate.py` — gate model + trainer.
* `gars_stage_slide_gate/best.pt` — trained ckpt (τ=0.22, F1=0.881).
* `gars_stage_slide_gate/fold0_predictions.json` — per-slide gate
  prob/logit on fold-0 val.
* This file (`A3_SLIDE_GATE_RESULTS.md`) — sprint journal.

## Path forward

The publication-ready answer to "how do you reduce GNCAF FP rate"
is now:

* **Native solution** = use the cascade (best Dice, best FP, best
  GC behaviour, 7× fewer params).
* **Retrofit solution** = if you must use a dense pixel-decoder
  (e.g. GNCAF-style for paper compliance), bolt a tiny slide-level
  classifier in front of it. Halves the FP rate at modest recall
  cost. Cheaper than retraining the dense decoder (B-track, which
  failed to help in v3.61).

The B-track v3.61 retrain (heavy negatives + slide-level aux loss
*inside* GNCAF) made things worse — the in-model auxiliary loss
fights the per-pixel objective. Keeping the gate as a separate
external model (this A3 retrofit) is the right design.
