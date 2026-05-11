# v3.59 — GCUNet-faithful GCN aggregation (results journal)

This is the rigour-fix experiment: replace the iterative-residual GCN
in `GNCAFPixelDecoder` with the paper-faithful **concat-all-hops + MLP**
aggregation from GCUNet Eq. 3, holding every other hyperparameter
identical to v3.58. See `recovered/scaffold/gcunet_model.py` and
`configs/gncaf/config_transunet_v3_59.yaml`.

## TL;DR

The architecture fix **does not improve** the GNCAF baseline at fold 0;
fold-0 pixel-Dice is **0.359** vs v3.58's **0.406** (−0.047). Per the
plan's decision rule, **5-fold compute is NOT committed**. The fix is
preserved on disk (`gcunet_model.py`, config, chain script) for future
reference.

## Architecture diff vs v3.58

| Block | v3.58 (`GCNContext`) | v3.59 (`GCNContextPaper`) |
|---|---|---|
| Forward | `for k in K: h = GeLU(GCN_k(x)); x = LN(x + drop(h))` — iterative residual updates, returns x⁽ᴷ⁾ only. | `hs=[x]; for k: h=GeLU(GCN_k(h)); hs.append(h); return MLP(CAT(hs))` — concat all hops then MLP, matching paper Eq. 3. |
| Output dim | hidden_dim | hidden_dim (after MLP projection from `(K+1)*hidden_dim`) |
| Param count | ~5.3M | ~5.9M (extra MLP) |
| Total model | 110.4M | **111.0M** (+0.6M) |

Everything else (R50 + 12 ViT encoder, 1 fusion block, UNet decoder,
loss, dataset, augmentation, LR schedule, AMP, 60 epochs, patience 12)
is bit-identical to v3.58.

## Training (fold 0)

`gars_gncaf_v3.59_fold0_20260507_055838/best_checkpoint.pt`

| Metric | v3.58 fold-0 | v3.59 fold-0 |
|---|---|---|
| Best val mDice (training-time, on selected windows) | 0.6993 (epoch 41) | **0.6970 (epoch 16)** |
| GC dice at peak | 0.412 | **0.584** |
| TLS dice at peak | 0.824 | 0.810 |
| Time to BEST | 41 epochs (~1.5 h) | **16 epochs (~30 min)** |
| Early-stop epoch | n/a (ran full 53) | 28 (patience 12 from BEST 16) |

The architectural fix made training **dramatically more sample-efficient**
(reaches v3.58's BEST in ~⅓ the epochs) and pushed the GC dice
substantially higher at peak. Two clean BEST jumps in the v3.59 run:
epoch 12 (val_mDice 0.41 → 0.60 in one epoch when GC kicked in) and
epoch 16 (peak 0.697). One transient collapse at epoch 17 (GC briefly
crashed to 0.08 then recovered).

## Eval (fold 0, slide-level pixel-Dice across 4 shards)

`gars_gncaf_eval_v3.59_fold0_eval_shard*_20260507_070806/`

Aggregated across 124 val slides:

| Metric | v3.58 fold-0 | v3.59 fold-0 |
|---|---|---|
| **mDice_pix (gate metric)** | **0.406** | 0.359 (−0.047) |
| TLS pixel-Dice | 0.357 | 0.354 |
| GC pixel-Dice | 0.455 | 0.365 (−0.090) |
| TLS counts MAE | n/a in CSV | n/a |
| GC counts MAE | n/a in CSV | n/a |

The GC pixel-Dice dropped despite the *training-time* GC dice being
higher in v3.59. This is the same pattern we saw with v3.60: the
training metric (per-window dice on selected positives) and the eval
metric (full-slide pixel coverage) diverge — the architecture fix
biases the model to be more confident on smaller GC instances at
training time but loses pixel coverage at inference.

## Decision rule (from plan) — RESULT

- Plan: "Fold-0 head-to-head: v3.59 vs v3.58 on same val. … (probably
  no-op). 5-fold paired t-test: cascade vs v3.59 — does the architecture
  fix change the publication conclusion?"
- Result: v3.59 fold-0 mDice_pix = 0.359, **lower** than v3.58 fold-0
  (0.406). The architectural fix does **not** change the cascade-vs-GNCAF
  conclusion (cascade fold-0 = 0.845, both GNCAF variants are far
  below). 5-fold compute is NOT committed.

## What's preserved on disk (not deleted)

- `recovered/scaffold/gcunet_model.py` — `GCNContextPaper` and
  `GCUNetPixelDecoder` (subclasses `GNCAFPixelDecoder`, swaps the gcn
  attribute).
- `recovered/scaffold/configs/gncaf/config_transunet_v3_59.yaml` —
  v3.59 config, identical to v3.58 except `model.model_class: gcunet`.
- `recovered/scaffold/run_chain_5fold_gcunet_v359.sh` — 5-fold chain
  ready to launch *if* the decision is later reversed (e.g. for a
  publication appendix that requires this row).
- `recovered/scaffold/train_gncaf_transunet.py` and
  `eval_gars_gncaf_transunet.py` — both auto-detect `gcunet` via
  `model_class` field or `gcn.mlp.*` keys in the state_dict; existing
  v3.58 ckpts continue to load via the original `GNCAFPixelDecoder`
  branch.

## Why this didn't change the publication conclusion

1. **Cascade beats both GNCAFs by a lot** at the fold level
   (0.845 vs 0.40–0.41 mDice_pix on fold 0). The architectural
   difference between v3.58 and v3.59 is much smaller than the
   gap between either GNCAF and the cascade.
2. **GCUNet's strength in the paper was likely Seg4/Seg3 class
   taxonomy** (4-way TLS subtypes) on TCGA-COAD, which we don't
   have annotations for. The pixel-Dice gap on our cohort isn't
   really about Eq. 3 vs iterative GCN — it's about the dense
   pixel-decoder approach paying full inference cost on every patch
   regardless of whether the patch contains tissue worth decoding.

## Status

- Task #41 completed. Implementation preserved; no 5-fold compute
  committed. Cascade remains the published champion.
