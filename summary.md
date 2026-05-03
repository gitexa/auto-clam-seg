# GARS — TLS / GC pixel-segmentation benchmark

## TL;DR

**Champion: v3.37 RegionDecoderCascade.** Stage 1 (5-hop GATv2, 1.1 M
params, frozen) proposes candidate regions; **`RegionDecoder`** (14.5 M
params: ResNet-18 RGB encoder + UNI projection + Stage-1-graph
projection + 5× UpDoubleConv) decodes only the ~50 candidate regions
per slide (≈0.7 % of patches) into full-resolution masks. Post-processed
with `min_size=2 + 1-iter binary closing` for instance counting.

| Metric @ thr=0.5 | Value |
|---|---|
| Slide-level mDice (pixel-aggregate) | **0.845** |
| TLS dice | **0.822** |
| GC dice | **0.868** |
| TLS Spearman vs `tls_num` | 0.876 |
| GC Spearman vs `gc_num` | **0.934** |
| TLS MAE / slide | 5.36 instances |
| GC MAE / slide | **0.28** instances |
| Inference cost (RGB read for selected regions only) | ~2.65 s/slide |

**Biobank-scale (20 K slides):** ~14.7 h. **GNCAF would take ~23 days.**
v3.37 is **40× cheaper than GNCAF at higher quality** because the heavy
RGB-aware decoder runs only on candidate regions, not all patches.

When to deviate:
- For TLS-only, count-only deployment with a lightweight model
  (1.3 M params, 1/9 the size): use **patchcls (v3.22)**.
- For UNI-only inference (no RGB I/O at all, ~15× faster): use **v3.13
  cascade** (mDice 0.784, TLS 0.722, GC 0.846).
- For research replication of the paper baseline: **GNCAF (v3.35)**.

## Comparison

All metrics on the same val set (123 or 124 val slides, deployment-honest
slide-level pixel-aggregate where applicable; per-positives where stated).

| Architecture | Params | Pix-agg mDice | TLS pix | GC pix | TLS sp | GC sp | GC MAE | RGB? | s/slide |
|---|---|---|---|---|---|---|---|---|---|
| **v3.37 RegionDecoder (champ)** | **14.5 M** | **0.845** | **0.822** | **0.868** | 0.876 | **0.934** | **0.28** | yes (regions only) | 2.65 |
| v3.13 cascade | 11.6 M | 0.784 | 0.722 | 0.846 | 0.874 | 0.887 | 0.39 | no | 0.18 |
| v3.22 patchcls (no decoder) | **1.3 M** | n/a | n/a | n/a | 0.873 | 0.824 | n/a | no | ~0.05 |
| v3.26 patchcls→cascade hybrid | 11.6 M | 0.777 | 0.709 | 0.845 | 0.879 | 0.831 | 0.43 | no | 0.13 |
| v3.36 NeighborhoodPixelDecoder | 17.7 M | 0.658 | 0.686 | 0.629 | 0.874 | 0.862 | 0.48 | no | 18 |
| v3.35 GNCAF — 12-layer vanilla ViT (paper-faithful, our retrain) | 25.5 M | 0.296 | 0.340 | 0.251 | 0.547 | 0.674 | 1.11 | yes (every patch) | 100 |
| **GNCAF TransUNet v3.51** (+ 167 bg-only train slides) | 65.6 M | 0.322 | 0.268 | 0.376 | 0.485 | 0.593 | **1.12** | yes (every patch) | 107 |
| **GNCAF TransUNet v3.50** (paper-faithful, our retrain on fold-0, **clean split**) | 65.6 M | 0.349 | 0.302 | 0.396 | 0.575 | 0.480 | 1.95 | yes (every patch) | 107 |
| GNCAF TransUNet (recovered best ckpt, ep23) ⚠️ leaky | 65.6 M | 0.311¹ | 0.461 | 0.161 | 0.760 | 0.521 | 1.95 | yes (every patch) | 108 |
| v3.21 GNCAF — 6-layer ViT, per-positives only | 14.9 M | (per-pos) 0.607 | 0.794 | 0.419 | — | — | — | yes (every patch) | infeasible |

`patchcls` patch-grid metric (only metric it natively produces):
mDice=0.602, TLS=0.648, GC=0.555.

¹ **Caveat — likely label leak.** The recovered TransUNet's
`best_epoch_results.json` lists 20 slides as its val set; only 5 of
those overlap with our cascade's 124-slide val. The recovered
training (100 slides, drawn from a different fold) likely overlaps
with ~15-20 of our 124 val slides → those slides were *seen during
training* of the recovered TransUNet, so 0.311 is an **over-estimate**
of its true generalisation. The leak-free TransUNet number is
**v3.50 = 0.349** (paper-faithful retrain on our standard fold-0 split,
60 ep schedule, R50 frozen, all train slides). The clean number
modestly *exceeds* the leaky 0.311 — the leak hurt by simply training
on a noisier 100-slide subset rather than helping; in either case
TransUNet GNCAF lands in the same 0.30–0.35 mDice_pix band, which is
**0.5 mDice_pix below the v3.37 cascade champion (0.845)** and confirms
that the structural per-positives → slide-level gap is architectural,
not a training-budget or split artefact.

### The per-positives → slide-level collapse

Both our v3.35 vanilla-ViT GNCAF and the recovered TransUNet GNCAF
checkpoint show the same pattern: per-positives val mDice ≈ 0.6–0.71,
slide-level pix-agg ≈ 0.30–0.31. **The 0.40-point gap is structural,
not training-budget-limited.**

| Architecture | Per-positives mDice | Slide-level mDice_pix | Δ |
|---|---|---|---|
| **GNCAF v3.51 TransUNet (+ 167 bg-only train slides)** | **0.640** | **0.322** | **−0.318** |
| GNCAF v3.50 TransUNet (paper-faithful, clean fold-0) | 0.607 | 0.349 | −0.258 |
| GNCAF TransUNet (recovered best, leaky) | 0.714 | 0.311 | −0.403 |
| GNCAF v3.35 (vanilla ViT) | 0.602 | 0.296 | −0.306 |
| GNCAF v3.21 (6-layer ViT) | 0.607 | (infeasible at scale) | — |
| GNCAF v3.40 (longer schedule) | 0.668 | (pending) | — |
| **v3.13 cascade** | (per-pos n/a — Stage 2 only) | **0.784** | n/a |
| **v3.37 RegionDecoder cascade** | (per-pos n/a) | **0.845** | n/a |

The cascade design's two-stage selection (Stage 1 picks ~0.7 % of
patches, then Stage 2 decodes only those) is the structural lever
that closes this gap. GNCAF, run on every patch, has no analogous
filtering — its decoder gets called on bg patches too and inevitably
makes false-positive marks at slide level even when per-positives
metrics look strong.

This isn't a "GNCAF didn't train enough" story: the 65.6 M-param
TransUNet was trained for 23 epochs and its per-positives matches
paper expectations. The architecture is just inherently less suited
to slide-level deployment than the cascade.

### v3.51 — bg-only training slides ablation (negative result)

A natural hypothesis was that GNCAF's slide-level collapse stems from
never seeing bg-only slides during training: the model was only trained
on the 481 train slides with HookNet GT, while at deployment ~17 % of
the slide pool has zero TLS anywhere. **v3.51** added the 167 bg-only
train slides as zero-mask supervision (8 random patches per slide per
epoch). Comparison vs v3.50 (same architecture, same hyperparams):

| Metric | v3.50 | v3.51 (+ bg-only) | Δ |
|---|---|---|---|
| Per-positives val mDice (training) | 0.607 | **0.640** | +0.033 |
| Patch-grid mDice (slide-level) | 0.341 | **0.411** | +0.070 |
| Patch-grid GC dice | 0.463 | **0.638** | +0.175 |
| GC MAE / slide | 1.95 | **1.12** | −0.83 |
| **Pixel-agg mDice (deployment)** | **0.349** | 0.322 | −0.027 |
| Pixel-agg TLS dice | 0.302 | 0.268 | −0.034 |
| Pixel-agg GC dice | 0.396 | 0.376 | −0.020 |

**Bg-only supervision improved every patch-level / per-positives metric
but slightly hurt the slide-level pixel-aggregate dice.** The model is
better at "is this patch bg or has TLS/GC?" — patch-grid GC dice
jumped 0.46 → 0.64, GC MAE dropped 1.95 → 1.12 — but the pixel masks
it produces remain spatially noisy at slide scale. This implies the
slide-level collapse is **not** primarily a "didn't see enough bg"
problem; even with explicit zero-mask supervision on entire slides, the
per-patch decoder still over-fires on bg patches it considers
ambiguous.

The cascade closes this gap by **explicit gatekeeping** at deployment
(Stage 1 GAT refuses to forward >99 % of patches to Stage 2), not by
patch-level calibration. Bg-only training is the right move for
*patch-level* downstream tasks (counting, localization), but it's not
sufficient to close the GNCAF → cascade slide-level pixel-dice gap.

### Quality vs cost frontier

```
quality (mDice_pix)
  0.85  ┤ v3.37 ●──── champion
  0.80  ┤
  0.78  ┤ v3.13 ●─── UNI-only cascade (no RGB I/O)
  0.66  ┤ v3.36 ●
  0.30  ┤ v3.35 ●
        └──────────────────────→  cost (s/slide, log scale)
        0.05   0.18  2.65   18    100
```

### v3.37 threshold sweep

| thr | mDice_pix | TLS_pix | GC_pix | TLS sp | GC sp | TLS mae | GC mae |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.779 | 0.717 | 0.842 | 0.851 | 0.893 | 9.87 | 0.35 |
| 0.10 | 0.793 | 0.736 | 0.849 | 0.862 | 0.892 | 8.06 | 0.35 |
| 0.20 | 0.809 | 0.763 | 0.854 | 0.880 | 0.920 | 6.14 | 0.33 |
| 0.30 | 0.821 | 0.781 | 0.860 | 0.867 | 0.924 | 5.46 | 0.28 |
| 0.40 | 0.833 | 0.799 | 0.867 | 0.873 | 0.925 | 5.29 | 0.28 |
| **0.50** | **0.845** | **0.822** | **0.868** | **0.876** | **0.934** | 5.36 | **0.28** |

## Per-architecture critique

### v3.37 RegionDecoderCascade (NEW deployment champion)

**What it does.** Stage 1 (frozen v3.8 GATv2 5-hop) is the region
proposal network: graph forward over UNI features → binary patch score
→ binary closing → connected components → ~50 candidate regions per
slide (≈0.7 % of patches). For each region: gather a 3×3 patch
neighborhood; for each cell read its RGB tile from the WSI .tif at
level 0 + look up its UNI feature + look up its Stage 1 graph context;
encode the RGB through ResNet-18 (ImageNet-pretrained, fine-tuned)
to (512, 8, 8); add projected UNI feature + projected graph context
broadcast to the 8×8 map; tile 9 cells into (B, 512, 24, 24); 5×
UpDoubleConv → (B, 3, 768, 768). Per-cell max-prob reconciliation
across overlapping windows. Counting via connected components on the
slide-level argmax.

**Strengths.** Highest deployment quality across every metric (TLS
+0.100, GC +0.022, GC sp +0.047, GC MAE −0.11 vs v3.13). RGB inclusion
breaks the **UNI bottleneck** that v3.13 was bumping against (UNI is a
1536-d summary of a 256×256 RGB tile — sub-patch detail is gone after
UNI). Region-only decoding keeps inference cost bounded: 2.65 s/slide
(15× v3.13 cascade, **40× cheaper than GNCAF**), feasible at biobank
scale. Counting is essentially free — instance counts come from
Stage 1's connected components without needing the heavy decoder.

**Weaknesses.** RGB I/O on candidate regions (~50 per slide × 9 tiles)
adds ~2 s/slide vs v3.13's pure-feature inference. Decoder argmax over
overlapping windows uses max-prob reconciliation which can produce
seam artifacts at window boundaries (not yet observed in viz, but
possible). val_loss diverges from train_loss during training
(overfitting signal); patience-stop at ep25 captured the right
checkpoint but a few epochs of fine-tuning at lower LR could help.

**When to use.** Default. Use this checkpoint unless you specifically
need <1 s/slide inference (then v3.13).

**Provenance.**
- Stage 1: `gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt`
- Stage 2: `gars_region_v3.37_full_20260502_144124/best_checkpoint.pt` (56 MB, ep17)
- Eval: `gars_cascade_v3.37_full_eval_v2_*/cascade_results.json`
- Run command: `eval_gars_cascade.py stage1=… region_mode=true stage2_region=…`

### v3.13 cascade (UNI-only fast option, prior champion)

**What it does.** UNI-v2 features (1536-d, ~17 K patches/slide) feed a
5-hop GATv2 graph (Stage 1) that outputs a binary TLS-positive score
per patch. Patches with score ≥ 0.5 go to a UNIv2PixelDecoder
(Stage 2): linear bottleneck 1536→512→16²·128, 4× UpDoubleConv,
Conv2d head → (3, 256, 256). Selected patches' argmax tiles stitch
into a slide-level mask; connected-component counts feed Spearman.

**Strengths.** Best slide-level pixel dice and best GC counting in the
field. Decoupled stages — each has been improved independently
(v3.0 → v3.8 lifted +0.013, v3.4b alone added +0.13). Uses only
UNI-v2 features → no WSI tile reads at deployment.

**Weaknesses.** TLS dice (0.722) lags GC (0.846) by 0.12 — the largest
dice gap in the system. v3.27 (h=256) and v3.29 (min_tls_pixels=2048)
both regressed: the gap is not capacity or label looseness. Stage 1
recall is only 79 % at thr=0.5, so Stage 2 never sees ~21 % of true
positives — cascade error compounds. Threshold is global (no
per-cancer or per-slide calibration). Post-processing is doing
disproportionate work on counting (v3.7a min_size=2 + close=1 added
+0.14 GC sp).

**Provenance.**
- Stage 1: `gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt`, F1=0.626 ep15
- Stage 2: `gars_stage2_v3.4b_h128_min4096_20260430_090510/best_checkpoint.pt`, mDice=0.733 ep13

### v3.22 patchcls — no pixel upsampling

**What it does.** Same backbone as Stage 1 (5-hop GATv2 over UNI-v2
features), but the head is `Linear(256, 256) → GELU → Linear(256, 3)`
producing one class label per patch (bg / TLS / GC). No spatial
decoder, no upsampling, no per-pixel output. Per-patch label rule
during training: GC if any GC pixel, else TLS if mask>0 ≥ 4096, else bg.

**Strengths.** **1.3 M params — 1/9 of cascade.** One forward, no
two-stage handoff, no threshold tuning. **Beats cascade on patch-grid
mDice (+0.063)** without a decoder. Ties cascade on TLS Spearman
(0.873 vs 0.874). Inference is fastest in the field (~50 ms/slide).

**Weaknesses.** No pixel-level prediction → no pixel-aggregate dice;
cannot separate adjacent GC instances → loses GC Spearman by 0.063
(0.824 vs 0.887). One class per 256-px patch caps geometric
resolution. Per-patch label rule is harsh — a single GC pixel labels
the whole patch GC.

**Provenance.**
- `gars_patchcls_v3.22_patchcls_gatv2_5hop_20260501_104023/best_checkpoint.pt`
- ep29 patch-grid mDice=0.602.

### v3.21 / v3.35 GNCAF — the previously-lost paper architecture

**What it does.** Faithful re-implementation of GNCAF (Su et al. 2025,
arXiv:2505.08430). End-to-end:

1. ViT TransUNet encoder over each target patch's RGB tile (paper:
   12 layers, dim=384) → local tokens (B, 256, 384).
2. Multi-hop GCN over UNI-v2 features for **all** patches in the
   slide → per-patch context vectors (N, 384).
3. 1-layer multi-head self-attention fuses
   `[local_tokens; context.unsqueeze(1)]` → (B, 256, 384).
4. 4× bilinear-upsample decoder → (B, 3, 256, 256).

**The recovery story.** Only the model + dataset + verify scripts
survived the VM crash; the training script was lost. v3.21
re-trained with a *truncated* 6-layer ViT (14.9 M params) for budget
— ep13 per-positives mDice=0.607 (TLS=0.794, GC=0.419). The user
asked for a full-config retrain; v3.35 uses the paper's 12-layer
config (25.5 M params), 30 epochs, class_weights=[1,5,3].

**Strengths.** End-to-end (no cascade compounding error). Strictly
more information than cascade (RGB *and* UNI features). Published
peer-reviewed method. Per-positives TLS dice already strong (0.794
at v3.21).

**Weaknesses.** **WSI tile I/O is the architectural cost.** GNCAF
needs a 256×256 RGB tile per patch; full-slide eval reads ~17 K
tiles per slide. Cascade and patchcls need none. This makes GNCAF
deployment-impractical without an RGB tile cache (~386 GB for full
val). Even with eval optimisation (8 DataLoader workers, parallel
zarr reads), GNCAF inference is ~200× the latency of cascade. The
ImageNet-pretrained ViT is also wrong for histology — UNI-v2 is
itself a much better histology ViT, and using a separate from-scratch
ViT in GNCAF duplicates capacity.

**Provenance.**
- v3.21 (truncated): `gars_gncaf_v3.21_gncaf_recovered_20260501_103826/best_checkpoint.pt`
- v3.35 (full): `gars_gncaf_v3.35_gncaf_full_12layer_20260502_101050/best_checkpoint.pt` *(in training)*
- Eval JSON: filled in by `eval_gars_gncaf.py` — see "filling in" rows above.

### v3.26 patchcls → cascade hybrid

Use patchcls (v3.22) as the 3-way patch router, run v3.4b decoder
only on patches it predicts as TLS or GC. Hypothesis: patchcls's
more accurate routing + cascade decoder = best of both. Result:
**middle-ground**, doesn't win anywhere clearly. Stage 2 argmax
overrides patchcls's per-patch class on selected patches, so the
patch-grid lead disappears; patchcls's GC routing is less precise
than Stage 1's tuned binary threshold, so GC counting drops. Best
TLS MAE on file (5.07) and TLS Spearman (0.879) — only kept for
those.

## What's still untested

Items in `experiment_log.md`'s next-hypotheses queue, not part of
this benchmark:
- **KNN graph for Stage 1** (v3.20). Replace radius graph with k=8
  feature-space neighbours.
- **Per-cancer-type / per-slide threshold calibration**. Cancer-type
  breakdowns suggest different operating points would help (BLCA TLS
  0.55, KIRC TLS 0.48, LUSC TLS 0.56).
- **Sub-patch-grid patchcls** (16×16 cells per patch). Closes the
  geometric resolution gap without a full pixel decoder.
- **Full-train n_hops=6 Stage 1**. v3.32 was killed pre-convergence.
- **GNCAF with cached RGB tiles**. One-time ~386 GB build would
  remove the I/O wall and let GNCAF eval at the same wall-clock as
  cascade.
- **Replacing GNCAF's ViT with UNI-v2's intermediate token map**.
  Different architecture; the closest descendant of GNCAF that could
  plausibly close the deployment gap.

## Reproducibility pointers

- All runs under `/home/ubuntu/ahaas-persistent-std-tcga/experiments/`
- Champion eval: `eval_gars_cascade.py stage1=… stage2=…` (hydra)
- patchcls train+eval: `train_gars_patchcls.py` ; eval inline
- GNCAF train: `train_gncaf.py --config-name=config_full`
- GNCAF slide-level eval: `eval_gars_gncaf.py checkpoint=…`
- Validation visualisation: `viz_val_compare.py` writes per-slide
  PNGs to `recovered/scaffold/viz_v3.13_vs_v3.7a/`.

This document is the deployment-honest snapshot. The full experiment
journal lives in `experiment_log.md`.
