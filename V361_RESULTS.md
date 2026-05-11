# v3.61 — heavy-negatives + slide-level auxiliary loss (results journal)

This sprint task was: try to reduce GNCAF v3.58's universal
false-positive rate on truly-negative slides (91 % of 5-fold
fullcohort negatives) by combining the two B-track interventions
from the FP-reduction plan into a single retrain:

* **B1 — `neg_slide_targets: 32`** (4× v3.58's 8): more negative-
  slide patches per epoch.
* **B2 — slide-level auxiliary BCE head**: a `LayerNorm + Linear(768→1)`
  head on the mean-pooled GCN node-context, trained against
  "this slide has any TLS" with `BCE × 0.5`.

Everything else identical to v3.58 (12-layer ViT TransUNet, AdamW
lr=5e-5, AMP, augmentation on, class weights `[1, 5, 3]`,
`dice_loss_weight=0.5`, 60 epochs / patience 12).

## Conclusion: gate failed

| Metric | v3.58 fold-0 fullcohort | v3.61 fold-0 fullcohort | Δ |
|---|---|---|---|
| **mDice_pix** (gate ≥ 0.40) | **0.373** | **0.270** | **−0.103** |
| TLS pixel-Dice | 0.333 | 0.310 | −0.023 |
| GC pixel-Dice | 0.413 | **0.230** | **−0.183** |
| **TLS-FP rate** (gate ≤ 60 %) | **95 %** | **90 %** | −5 pt |
| GC-FP rate | 12 % | 44 % | **+32 pt** |
| Mean predicted TLS / negative slide | 27.6 | 30.8 | +3.2 |
| Best val_mDice (training, on selected windows) | 0.6993 (epoch 41) | 0.6193 (epoch 18) | −0.080 |

Both gate conditions fail. **No 5-fold compute committed.**

## Why the intervention didn't work

1. **Slide-level head fights the per-pixel objective.** The auxiliary
   BCE on mean-pooled GCN context tells the model "this slide is
   TLS-positive → predict any TLS pixel". Combined with the heavy
   class weights `[1, 5, 3]`, the model learns to *over-predict* TLS
   on positive slides (mean predicted TLS / positive slide rose from
   55.5 → 66.5). The slide-level loss can be satisfied trivially by
   predicting *some* TLS pixel anywhere on the slide.
2. **Heavy negatives don't bite at the per-patch level.** With
   `neg_slide_targets=32`, the model sees more negative-slide patches
   but they are still individual 256-px tiles — the model has no
   way to know they came from a slide-level negative until the
   slide-level auxiliary loss tells it. And the auxiliary loss
   pulls *the wrong way* on positives (point 1).
3. **GC catastrophically over-predicted.** GC-FP rate jumped from
   12 % → 44 % and GC pixel-Dice halved (0.413 → 0.230). The
   slide-level "is this slide positive" signal is binary (TLS-
   positive vs not) so it doesn't disambiguate TLS from GC; the
   model leans on GC to satisfy the loss.
4. **Training-time val_mDice dropped 0.080.** Even on the selected
   per-window metric the v3.61 model is worse than v3.58. The
   added 0.5 × BCE term competes for capacity with the seg objective.

## What this rules out

* Pure no-architecture-change retraining with heavier negatives + an
  additive slide-level term **does not** fix GNCAF's FP behaviour.
* The slide-level auxiliary BCE in this form (mean-pool over GCN
  context) is the wrong supervision signal — it's correlated with
  "any TLS detected" rather than "should we predict TLS at all".
* Pushing `neg_slide_targets` from 8 to 32 alone is not enough to
  teach the model about slide-level absence at the per-patch
  decision rule.

## What's left (from the broader plan)

* **A1 / A2** (no-train post-proc + threshold sweep): never tried;
  could still help. Re-evaluating the existing v3.58 fold-0 outputs
  with `min_component_size ∈ {5, 10, 20}` and a per-class
  softmax-threshold sweep would cost no GPU compute.
* **A3** (slide-level gate retrofit, *separate* model): train a tiny
  ABMIL/MLP on UNI features → predict TLS-positive slide; apply at
  inference to suppress GNCAF outputs on slides classified
  negative. This is fundamentally different from B2 because the
  gate is **separate from training** — it doesn't compete with the
  segmentation loss for capacity.
* **B3** (hard-negative mining over the 188 v3.58 FP slides): not
  attempted; would need ~12 h GPU.

## Status of work products

* `recovered/scaffold/configs/gncaf/config_transunet_v3_61.yaml` —
  config preserved on disk for traceability.
* `recovered/scaffold/gncaf_transunet_model.py` — `GNCAFPixelDecoder`
  now accepts `slide_aux_head: bool` constructor kwarg + has a
  `forward(..., return_slide_logit=True)` path. Existing v3.58
  ckpts still load fine (the head is only constructed when the
  flag is True, and the eval auto-detects via `slide_head.weight`
  in the state_dict).
* `recovered/scaffold/train_gncaf_transunet.py` — slide-level BCE
  loss term wired in (gated on `slide_aux_loss_weight > 0`).
* `recovered/scaffold/train_gncaf.py` — `GNCAFFastDataset.__getitem__`
  now returns `slide_label: torch.tensor(0.0 if is_negative_slide else 1.0)`.
* `recovered/scaffold/eval_gars_gncaf_transunet.py` — auto-detects
  `slide_head.*` keys in state_dict, instantiates with
  `slide_aux_head=True`. Still uses argmax for inference (the slide
  logit isn't used at eval time).
* Trained ckpt at `gars_gncaf_v3.61_fold0_20260509_000015/best_checkpoint.pt`.

## Headline (sprint deliverable)

The B-track retrain hypothesis is disproven on fold 0. Neither
heavy negatives nor a slide-level auxiliary BCE — alone or
combined — fix the dense pixel-decoder's universal failure mode
on truly-negative slides. The cascade's two-stage gate remains
the only paradigm in the comparison that controls FPs structurally.
