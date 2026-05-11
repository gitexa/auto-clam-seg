# Architecture details — TLS / GC pixel segmentation on TCGA WSI

Comprehensive companion to `ARCHITECTURE_COMPARISON.md`. Covers:

1. Cohort definition and patient-stratified splits
2. TLS / GC ground-truth distribution per cohort
3. Per-architecture details (modules, dims, param counts)
4. Loss, optimizer, LR schedule, and full training hyperparameters
5. Evaluation pipeline and post-processing

All numbers are reproducible from `recovered/scaffold/configs/` and the
metadata CSV at
`/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/clam_training/df_summary_v10.csv`.

## 1. Cohort definition

The cohort comes from three TCGA projects: BLCA (bladder), KIRC (renal
clear cell), and LUSC (lung squamous). Slides are 20× WSI (`.tif` /
`.svs`); per-patch UNI v2 features (1536-d) are pre-extracted at
256-px@20× and stored as zarr.

Total slides: **1015** (1018 in metadata CSV; 3 dropped for missing
zarr features). Annotation status per HookNet (expert-annotated TLS /
GC masks):

| count | with HookNet mask | without (TLS-negative slide) |
|---|---|---|
| 1015 | 765 | 250 |

**Test split (170 patients, held out from all training/CV):**

* 201 slides (160 with mask + 41 GT-negative), 169 patients.
* Identified by `df_summary_v10_test.csv` and excluded by
  `prepare_segmentation.create_splits()`.
* No model has ever seen these slides during training or CV.
* Eval on test still pending (blocked on local TIF/SVS access).

**5-fold patient-stratified CV** (seed=42, deterministic):

| Fold | Slides | GT-pos | GT-neg | ~Patients |
|---|---|---|---|---|
| 0 | 166 | 124 | 42 | 160 |
| 1 | 158 | 118 | 40 | 156 |
| 2 | 163 | 121 | 42 | 153 |
| 3 | 162 | 122 | 40 | 151 |
| 4 | 165 | 120 | 45 | 151 |
| **Total CV** | **814** | **605** | **209** | ~771 |

All splits stratified at the patient (12-char `slide_id` prefix) level
so no patient leaks across folds.

## 2. TLS / GC distribution

From `df_summary_v10.csv` (`tls_num`, `gc_num` columns — counts per
slide of pathologist-annotated TLS / GC instances):

* **TLS-positive slides** (tls_num > 0): 767 / 1018 (75 %)
* **GC-positive slides** (gc_num > 0): 260 / 1018 (26 %)
* **TLS instances per slide**: mean 9.9 ± 17.6, median 4, max 226
* **GC instances per slide**:  mean 1.04 ± 3.40, median 0, max 61

**Per cancer type (whole cohort, before test split):**

| Cancer | Slides | TLS+ slides | GC+ slides | Mean tls_num | Mean gc_num |
|---|---|---|---|---|---|
| BLCA | 345 | 259 (75 %) | 124 (36 %) | 7.4 | 1.44 |
| KIRC | 299 | 166 (56 %) | 32 (11 %)  | 3.0 | 0.17 |
| LUSC | 374 | 342 (91 %) | 104 (28 %) | 17.9 | 1.37 |

Class skew implications:

* **GC is rare** — only 26 % of slides have any GC; this drives the
  weighted CE (class weights `[1.0, 5.0, 3.0]` for {bg, TLS, GC}) and
  the explicit GC-Dice term in cascade Stage 2's loss.
* **KIRC has the lowest TLS / GC density** — relevant for per-cancer
  Dice in `fig_cancer_breakdown.png`.

## 3. Architecture details

Each architecture's parameter counts were verified by instantiating
the model and summing `p.numel()` over `model.parameters()`.

### 3.1 Cascade v3.37 (champion)

Two stages with a hard gate.

![Cascade Stage 1 architecture](notebooks/architectures/arch_cascade_stage1.png)
![Cascade Stage 2 architecture](notebooks/architectures/arch_cascade_stage2.png)

#### Stage 1 — `GraphTLSDetector` (5-hop GATv2)

* **Input**: 1536-d UNI v2 features per patch (one node per 256×256
  patch); 4-conn `graph_edges_1hop` from the slide zarr.
* **Layers**:
  - `Linear(1536 → 256)` projection
  - 5 × `GATv2Conv(256 → 256, heads=4, concat=False, dropout=0.1)`
    with `LayerNorm(256)` + residual after each
  - `Linear(256 → 1)` head → per-patch logit
* **Params**: **1,090,049** (~1.1 M)
* **Output**: per-patch binary "is this patch TLS-or-GC positive?"
  logit. Used as a hard gate (threshold 0.5) for Stage 2.
* **File**: `recovered/scaffold/train_gars_stage1.py`,
  `configs/stage1/model/gatv2_5hop.yaml`.

#### Stage 2 — `RegionDecoder` (RGB + UNI + graph fusion)

* **Input** per region (3×3 patch window centred on a Stage-1-positive
  patch):
  - RGB tile (768×768) read from the WSI tif
  - UNI features for the 9 patches in the window (1536-d each)
  - Stage-1 graph context for the 9 patches (256-d each)
* **Layers**:
  - **RGB branch**: ResNet-18 encoder (ImageNet-pretrained, frozen
    by default) → 4-stage feature pyramid
  - **UNI projection**: `Linear(1536 → 64)` per patch, broadcast over
    the 256-px tile, concatenated with RGB features at the bottleneck
  - **Graph projection**: `Linear(256 → 64)` per patch, same broadcast
  - **5× UpDoubleConv decoder** (h=64): bilinear-upsample + 2× Conv3×3 + BN + ReLU
  - **Head**: `Conv2d(16 → 3)` for {bg, TLS, GC} per pixel
* **Params**: **14,506,627** (~14.5 M; 3.3 M trainable when
  `freeze_rgb_encoder=true`)
* **Decode resolution**: 768×768 region per inference (covers 9 patches).
* **File**: `recovered/scaffold/region_decoder_model.py`,
  `configs/region/model/region_h64.yaml`.

#### Cascade behaviour

On a slide:
1. Stage 1 produces a per-patch logit; sigmoid > 0.5 selects ~0.7 % of
   patches (typically tens of regions per slide).
2. Stage 2 decodes only those selected 3×3 regions at 768×768.
3. Component counts use `min_size=2, closing_iters=1` for production
   (default eval ran with `min_size=1, closing_iters=0` — the most
   permissive setting; published numbers in `summary.md` use the
   stricter post-proc).

### 3.2 GNCAF v3.58 / v3.56 — TransUNet + iterative-residual GCN

![GNCAF v3.58 architecture](notebooks/architectures/arch_gncaf_v358.png)

Three GNCAF variants, all sharing the same `GNCAFPixelDecoder` topology:

* **Encoder** (94,583,104 params): R50 trunk (stem + 3 ResNet stages,
  ImageNet-pretrained) + 12-layer ViT trunk operating on 16×16 tokens
  of 768-d.
* **GCN context** (2,958,336 params): `Linear(1536 → 768) + LN`, then
  3 × `GCNConv(768 → 768, add_self_loops=True, normalize=True)` with
  GeLU + LayerNorm + residual. Output: a per-node 768-d context vector.
* **Fusion block** (7,087,872 params): a single ViT-like block that
  prepends the per-target node's context as a `[CTX]` token to the
  256 image tokens; one MultiHeadAttention + MLP, then drop the CTX
  token.
* **Decoder** (3,465,072 params): 4-stage transposed-conv UNet decoder
  with R50 skip connections.
* **Head** (435 params): `Conv2d(16 → 3)` per pixel.
* **Total**: **108,094,819** (~108 M).

**v3.56 vs v3.58** — same architecture, different training (v3.58 turns
on RGB augmentation + uses LR=5e-5; v3.56 was the no-aug baseline).

Files: `recovered/scaffold/gncaf_transunet_model.py`,
`configs/gncaf/config_transunet_v3_56.yaml` and
`configs/gncaf/config_transunet_v3_58.yaml`.

### 3.3 GNCAF v3.59 — GCUNet-faithful (paper Eq. 3)

![GNCAF v3.59 architecture](notebooks/architectures/arch_gncaf_v359.png)

Identical to v3.58 *except* the GCN block:

* **GCN context** (5,905,920 params): `Linear(1536 → 768) + LN`, then
  3 × `GCNConv(768 → 768)` with GeLU; **all hops concatenated** to
  `(K+1) × 768 = 3072`-d, then projected back via
  `Linear(3072 → 768) + GeLU + Dropout + Linear(768 → 768) + LN`.
* **Total**: **111,042,403** (~111 M; +0.6 M vs v3.58 for the MLP).
* Implemented as `GCUNetPixelDecoder` (subclasses `GNCAFPixelDecoder`,
  swaps the `gcn` attribute).

Files: `recovered/scaffold/gcunet_model.py`,
`configs/gncaf/config_transunet_v3_59.yaml`.

### 3.4 seg_v2.0 — `GNNSegmentationDecoder` (older baseline, retained for context)

![seg_v2.0 architecture](notebooks/architectures/arch_seg_v2.png)

Pixel-level segmentation with a small CNN decoder fed by UNI v2
embeddings; uses dual *binary* sigmoid heads + center-heatmap regression
rather than the cascade's two-stage gate or GNCAF's TransUNet.

* **Input** per slide: per-patch UNI v2 features `(N, 1536)` + coords
  `(N, 2)` + 1-hop spatial graph `edge_index`.
* **Layers**:
  - Feature projection: `Linear(1536 → 384) + LN + GELU`
  - GNN context (configurable, **0 layers in published config** — kept
    in the architecture as `GATv2Conv(384 → 384, heads=4)` × N with
    residual + LN; the published v2.0 run sets `gnn_layers=0` so the
    GNN block is a no-op identity)
  - Scatter to 2D grid + on-grid spatial aggregation
    (`Conv2d(384, 384, k=5)` + BN + GELU)
  - CNN decoder: **3 upsample blocks** (`upsample_factor=8` ⇒ 2³),
    each = depthwise `Conv2d(in, in, k, groups=in)` + pointwise
    `Conv2d(in, out, 1)` + BN + GELU + bilinear `Upsample(2×)`,
    halving channels at each block (384 → 192 → 96 → 48 → 32-floored)
  - **5 prediction heads** (`Conv2d(in_ch → c, 1)` each):
    1. TLS binary sigmoid (TLS ≥ 1: TLS or GC pixel)
    2. GC binary sigmoid (independent — no class-conditional softmax)
    3. TLS center heatmap (Gaussian regression on instance centroids)
    4. GC center heatmap
    5. Offset (dy, dx) — disabled in published config (`offset_weight=0.0`)
* **Params (HIDDEN_DIM=384, GNN_LAYERS=0)**: **4,389,462** (~4.4 M)
* **Output resolution**: 8× upsampled patch grid (e.g. a 100×100
  patch grid → 800×800 pixel-prediction map).
* **Files**:
  `profile-clam/models/segmentation_decoder.py` (model),
  `profile-clam/train_segmentation.py` (trainer + config block at top),
  `profile-clam/prepare_segmentation.py` (panoptic targets + dataset).

#### Loss — `segmentation_loss` (dual-sigmoid + center heatmaps)

Six terms summed with config weights:

| Term | Formulation | Weight |
|---|---|---|
| TLS focal loss | `focal_loss_binary(logits_tls, targets≥1, α=0.75, γ=2.0)` | `focal_weight=1.0` |
| TLS Dice | `masked_dice_loss(logits_tls, targets≥1)` | `dice_weight=2.0` |
| GC focal loss | focal on logits_gc, **cropped to TLS-positive pixels only** (GC is 0.006 % of pixels at 4×; cropping to TLS lifts the rate to ~1 % so Dice is learnable) | `gc_focal_weight=1.0` |
| GC Dice | Dice on the cropped pixels (intersection / union with +1 smoothing) | `gc_dice_weight=5.0` |
| TLS center heatmap | MSE between predicted and Gaussian-blurred GT instance-center heatmap | `center_weight=2.0` |
| GC center heatmap  | Same, for GC instances | `gc_center_weight=2.0` |
| Offset regression | (dy, dx) per pixel | `offset_weight=0.0` (disabled) |

The dual-binary-sigmoid + center-heatmap formulation is **closer to a
panoptic-style segmentation** than the cascade's argmax 3-class output
and gives the model an explicit count-via-heatmap-peaks signal — that
plus the GC-cropping trick is what lets the model find very rare GC
instances at all. The benchmark CSV's "no GC dice" entry for seg_v2.0
reflects that GC is read out via centroid heads, not pixel classification.

#### Optimizer + LR + epochs (from `train_segmentation.py:39-86`)

| Knob | Value |
|---|---|
| Optimizer | `AdamW(lr=1e-4, weight_decay=1e-4)` |
| LR schedule | constant (no warmup, no cosine) |
| Epochs | 100 (early stop with patience 30, min 20 epochs) |
| Batch | 1 slide per step (slide-level dataset, full graph each step) |
| AMP | no |
| Augmentation | `AUGMENT=False` in published config (knobs exist for `coord_jitter=0.05`, `edge_drop_rate=0.1`, `patch_drop_rate=0.05`) |
| Hidden dim | 384 |
| GNN layers | **0** (the architecture supports up to N GATv2 layers, but the published v2.0 turns the GNN off — the slide-level "v2.0" baseline is essentially per-patch UNI features + spatial CNN decoder + center heads, no graph aggregation) |
| Upsample factor | **8** (vs cascade's 4) → finer 8× output grid for centroid localisation |
| Dropout | 0.1 |
| Class balance | handled by focal-loss alpha (α=0.75 TLS, α=0.90 GC inside the cropped GC focal) and the GC-Dice up-weight (×5) |
| Saw GT-negative slides? | YES — same `TLSSegmentationDataset` that admits all 1015 slides; missing-mask entries get an all-zero target |
| Seed | 42 |

#### Why it appears in the benchmark but not the per-slide plots

The 5-fold mDice_pix in `benchmark_5fold.csv` (0.552 ± 0.023) is the
**TLS-only pixel-Dice** — seg_v2.0 reports a single TLS-vs-bg metric
because the GC pathway uses centroid regression, not per-pixel
classification (so a per-pixel GC Dice is not directly comparable to
the cascade / GNCAF GC-Dice). The per-slide CMs / PR-AUC / AUROC in
`fig_*.png` are not generated for seg_v2.0 because the per-slide eval
JSON for it isn't on disk (it pre-dates the eval-pipeline harmonisation
that produced `cascade_per_slide.json` / `gncaf_agg.json`).

### 3.5 v3.60 multi-scale (closed)

Bipartite multi-scale graph extension of cascade Stage 1; documented
in `V360_RESULTS.md`. Did not pass the decision gate; not included
in the active comparison.

## 4. Training hyperparameters

Defaults from `configs/`. All architectures use **AdamW** unless noted.

### 4.1 Cascade Stage 1 (`train_gars_stage1.py`)

| Knob | Value |
|---|---|
| Loss | `BCEWithLogitsLoss(pos_weight=5.0)` per patch |
| Optimizer | `AdamW(lr=1.0e-3, weight_decay=1.0e-4)` |
| LR schedule | `CosineAnnealingLR(T_max=epochs, eta_min=0.0)` |
| Epochs | 50 (early stop with patience 10 on val F1) |
| Batch | 1 slide per step (slide-level dataset, full graph each step) |
| AMP | no |
| Augmentation | none (UNI features are pre-extracted) |
| Class balance | `pos_weight=5.0` in BCE |
| Targets | `patch_labels_from_mask(upsample_factor=4, patch_size=256)` — patch is "1" if any positive pixel falls in it |
| Saw GT-negative slides? | **YES** (TLSSegmentationDataset admits all 1015 slides; missing-mask entries get an all-zero target) |
| Seed | 42 |

### 4.2 Cascade Stage 2 — RegionDecoder (`train_gars_region.py`)

| Knob | Value |
|---|---|
| Loss | `CrossEntropy(weights=[1, 5, 3]) + 1.0 × (1 − soft_Dice_GC)` over 3 classes {bg, TLS, GC} |
| Optimizer | `AdamW(lr=3.0e-4, weight_decay=1.0e-4)` |
| LR schedule | warmup 3 epochs → cosine decay over (epochs − warmup) |
| Epochs | 30 (early stop with patience 8 on val mDice) |
| Batch size | 8 region windows (768×768 RGB + 9 UNI + 9 graph context) |
| AMP | no (fits in 40 GB) |
| Augmentation | none on regions (pre-cropped 768×768) |
| Class weights | `[1.0, 5.0, 3.0]` — TLS up-weighted 5×, GC up-weighted 3× |
| GC-Dice weight | `gc_dice_weight=1.0` (separate Dice term boosts the rarest class) |
| Stage 1 frozen | yes — Stage 1 ckpt is loaded with `requires_grad=False`; only RegionDecoder trains |
| ResNet-18 | ImageNet-pretrained; *not* frozen (`freeze_rgb_encoder=false`) |
| Window selection | up to 64 windows / slide, stride=2, drawn from TLS-positive patches |
| Saw GT-negative slides? | **NO** — `tls_patch_dataset` only collects TLS-positive patches; cascade gate (Stage 1) handles negatives. |
| Seed | 42 |

### 4.3 GNCAF v3.58 (`train_gncaf_transunet.py`, `config_transunet_v3_58.yaml`)

| Knob | Value |
|---|---|
| Loss | `CrossEntropy(weights=[1, 5, 3]) + 0.5 × soft_Dice (mean over 3 classes)` |
| Optimizer | `AdamW(lr=5.0e-5, weight_decay=1.0e-4, grad_clip=0.5)` |
| LR schedule | `CosineAnnealingLR(T_max=epochs, eta_min=5e-7)` (warmup 5 epochs linear ramp) |
| Epochs | 60 (early stop with patience 12 on val mDice) |
| Batch | 1 slide per step (heavy ViT decoder; per-step OOM-guarded with `max_pos_per_slide=12`, `bg_per_pos=1.0`) |
| AMP | yes (`bfloat16`) |
| Augmentation | yes (`augment: true`) — RGB flips + 90° rotations on the target tile |
| Class weights | `[1.0, 5.0, 3.0]` |
| Dice weight | `dice_loss_weight=0.5` |
| ImageNet weights | R50 + ViT both loaded from timm |
| `freeze_cnn` | false (CNN trunk is fine-tuned) |
| `include_negative_slides` | **true**, `neg_slide_targets=8` (8 patches per negative slide per batch) |
| Seed | 42 |

### 4.4 GNCAF v3.56 — same as v3.58 with `augment: false`

### 4.5 GNCAF v3.59 (paper-faithful) — same as v3.58 with `model_class: gcunet`

Architecture changes only the GCN block (concat-all-hops + MLP);
training hyperparameters bit-identical to v3.58.

## 5. Evaluation pipeline

* **Cascade**: `eval_gars_cascade.py region_mode=true` — Stage 1
  selects, Stage 2 decodes selected windows, post-process with
  configurable `min_component_size` / `closing_iters`. Produces
  per-slide rows in `cascade_per_slide.json` (one entry per threshold).
* **GNCAF**: `eval_gars_gncaf_transunet.py` — runs the full TransUNet
  on every patch, then aggregates per-slide patch-grid + pixel-agg
  Dice. Per-slide rows in `gncaf_agg.json["per_slide"]`.
* **Sharding**: 4 parallel shards with `slide_offset=N slide_stride=4`
  for both eval scripts (lets us run all four shards on a single A100).
* **Cohort selection**: as of the patch landed today, both eval
  scripts admit GT-negative slides by default
  (`eval_positives_only=False`); set the flag to `True` to restore
  the legacy positive-only pool.
* **Pixel-Dice aggregation**: per-slide Dice on the patch-grid
  (`{bg, TLS, GC}`) + an intersection / denom counter aggregated
  across all selected patches' 256×256 native masks → one
  `mDice_pix` per fold.
* **Detection metrics** (`build_arch_comparison.py`): per-slide
  `n_pred` (CCs in the predicted grid) is the ranking score; AUROC,
  PR-AUC, P/R/F1, and bucketed P/R/F1 are computed against
  `gt_n > 0` (binary detection) and against count bins.

## 6. Reproducibility

* Seed `42` for all dataset splits and model init.
* Patient-stratified k-fold split is deterministic given seed +
  `df_summary_v10_test.csv`.
* Hydra outputs every run's resolved `config.yaml` next to the
  checkpoint, so any reported number can be regenerated exactly by
  the same command.
* `chain_lib.sh` orchestrates the multi-fold runs with marker files
  (`<label>.done` / `.failed`) so a chain can resume after
  interruption.
