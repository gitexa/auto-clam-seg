# Champion model registry — sync to another VM and run on all TCGA slides

Last updated: 2026-05-10. Each row is the best checkpoint of its
architecture category measured on the **5-fold full-cohort eval**
(165 slides per fold, includes 41 GT-negatives) — except where noted
that the metric is fold-0 only or on a smaller pool.

## TL;DR — files to rsync

| Category | mDice_pix (fold-0 fullcohort) | Checkpoint | Config | Size |
|---|---|---|---|---|
| **🏆 Cascade — Stage 1 (gate)** | — (TLS-detect only) | `experiments/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt` | `experiments/gars_stage1_v3.8_gatv2_5hop_20260501_043445/.hydra/config.yaml` | 6.0 M |
| **🏆 Cascade — Stage 2 (RegionDecoder)** | **0.829** | `experiments/gars_region_v3.37_full_20260502_144124/best_checkpoint.pt` | `experiments/gars_region_v3.37_full_20260502_144124/.hydra/config.yaml` | 56 M |
| **GNCAF v3.58 (line-B prod)** | 0.382 | `experiments/gars_gncaf_v3.58_gncaf_transunet_vit12_aug_20260504_204647/best_checkpoint.pt` | `experiments/gars_gncaf_v3.58_gncaf_transunet_vit12_aug_20260504_204647/.hydra/config.yaml` | 414 M |
| **GNCAF v3.63 (dual-sigmoid heads, fold 0)** | 0.434 (mDice); GC=**0.669** | `experiments/gars_gncaf_v3.63_dual_sigmoid_20260510_175021/best_checkpoint.pt` | `experiments/gars_gncaf_v3.63_dual_sigmoid_20260510_175021/.hydra/config.yaml` | 432 M |
| **seg_v2.0 (no-graph baseline)** | 0.579 (TLS Dice only) | `experiments/seg_v2.0_tls_only_5fold_31aaec0c/fold_{0..4}/best_checkpoint.pt` | n/a — config in code (`profile-clam/train_segmentation.py` constants) | 52 M × 5 folds |
| **🥈 seg_v2.0 (dual TLS+GC, 5-fold)** | **0.622 ± 0.027** (TLS 0.561 / GC **0.682**) | `experiments/seg_v2.0_dual_tls_gc_5fold_57e12399/fold_{0..4}/best_checkpoint.pt` | `experiments/seg_v2.0_dual_tls_gc_5fold_57e12399/pipeline_config.json` | 17 M × 5 folds |
| **seg_v2.0 (final, all-folds for TCGA-wide deploy)** | n/a (val=train; saved best_dice 0.946 is overfit) | `experiments/seg_v2.0_final_tls_gc_all_folds_c6c32fc3/fold_0/best_checkpoint.pt` | `experiments/seg_v2.0_final_tls_gc_all_folds_c6c32fc3/pipeline_config.json` | 52 M |
| **Family-B `frozen_gn_h128`** (lost paper-faithful, fold-0 ckpt) | 0.7707 (lost-original training metric — NOT yet validated full-cohort) | `experiments/gncaf_pixel_pixel_gncaf_frozen_gn_h128_c088b44f/best_checkpoint.pt` | n/a — saved config is empty | 16 M |
| **Family-B `demo` (h=256)** (lost) | 0.7804 (lost-original) | `experiments/gncaf_pixel_demo_7cfcd812/best_checkpoint.pt` | n/a | 34 M |
| **Lost 0.7143 line-B baseline** | 0.298 fold-0 fullcohort (recovered) | `experiments/gncaf_pixel_100slides_lr3e4_gcn3hop_fusion_2983e563/best_checkpoint.pt` | `experiments/gncaf_pixel_100slides_lr3e4_gcn3hop_fusion_2983e563/config.yaml` | 252 M |

`experiments/` = `/home/ubuntu/ahaas-persistent-std-tcga/experiments/`.

## What you need on the destination VM

For ALL models:

* **UNI v2 features at 20×/256-px** for the cohort you want to score:
  `/home/ubuntu/ahaas-persistent-std-tcga/data/tcga-{blca,kirc,lusc}/representations_tif_trident/20x_256px_0px_overlap/`
  (one zarr per slide, `*_complete.zarr`).
* **WSI .tif files** at the same path layout (only for the Cascade
  Stage 2 RegionDecoder + the GNCAF/Family-B variants — they read
  256-px RGB tiles per target patch). Stage 1 of the cascade and
  seg_v2.0 are **feature-only** (no WSI reads needed).
* **HookNet TLS/GC ground-truth masks** (only if you want metrics):
  `/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/masks_tls_gc/`.

For each model, also clone the relevant code repo:

| Code | Repo path | Used by |
|---|---|---|
| `auto-clam-seg/recovered/scaffold/` | this repo | Cascade, GNCAF v3.58, Family-B, lost 0.7143 |
| `profile-clam/` | `~/profile-clam/` | seg_v2.0 only (`models/segmentation_decoder.py` + `prepare_segmentation.py` + `train_segmentation.py`) |

## Per-category run instructions

### 1. Cascade (champion — recommended for production runs)

**Two stages**: Stage 1 GraphTLSDetector gates which slides are
non-zero; Stage 2 RegionDecoder produces TLS/GC masks for the gated slides.

```bash
cd auto-clam-seg/recovered/scaffold
WANDB_MODE=disabled python eval_gars_cascade.py \
    stage1_checkpoint=/path/to/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt \
    stage2_checkpoint=/path/to/gars_region_v3.37_full_20260502_144124/best_checkpoint.pt \
    fold_idx=0   # or use limit_slides=N for spot-check
```

Outputs `cascade_per_slide.json` with per-slide rows (slide_id,
gt_n_tls, gt_n_gc, n_tls_pred, n_gc_pred, tls_dice, gc_dice).

Reads: zarr features, WSI tifs, HookNet masks. Writes nothing
besides metrics + per-slide JSON unless you enable mask export.

### 2. GNCAF v3.58 (best line-B GNCAF, 5-fold-trained)

```bash
cd auto-clam-seg/recovered/scaffold
WANDB_MODE=disabled python eval_gars_gncaf_transunet.py \
    checkpoint=/path/to/gars_gncaf_v3.58_gncaf_transunet_vit12_aug_20260504_204647/best_checkpoint.pt \
    fold_idx=0
```

5-fold versions also exist:
`gars_gncaf_v3.58_5fold_fold{1..4}_*/best_checkpoint.pt`.

### 3. seg_v2.0 (no-graph baseline) — TWO 5-fold variants + 1 production model

```bash
cd auto-clam-seg/recovered/scaffold
# Default: dual_tls_gc 5-fold (the production-recipe variant — recommended)
python eval_seg_v2.py --variant seg_v2_dual
# Alt: tls_only 5-fold
python eval_seg_v2.py --variant seg_v2
```

Outputs per-slide JSON in
`gars_gncaf_eval_seg_v2{,_dual}_fullcohort_fold{0..4}_eval_shard0/gncaf_agg.json`.

**Production model for full-TCGA deployment** (val=train, NOT a CV
metric — only use for inference):

```python
import torch
from eval_seg_v2 import GNNSegV2
m = GNNSegV2(feature_dim=1536, hidden_dim=384, gnn_layers=0, gnn_heads=4,
             upsample_factor=4, n_classes=3, dropout=0.1, patch_size=256)
sd = torch.load("seg_v2.0_final_tls_gc_all_folds_c6c32fc3/fold_0/best_checkpoint.pt",
                map_location="cpu", weights_only=False)["model_state_dict"]
m.load_state_dict(sd, strict=True)
```

This is the model trained on ALL folds combined — the right ckpt to
score new TCGA slides outside the 5-fold CV.

### 4. Family-B reconstructed (`frozen_gn_h128` or `demo`)

```bash
cd auto-clam-seg/recovered/scaffold
WANDB_MODE=disabled python eval_gars_gncaf_transunet.py \
    checkpoint=/path/to/gncaf_pixel_pixel_gncaf_frozen_gn_h128_c088b44f/best_checkpoint.pt \
    fold_idx=0
```

The eval auto-detects Family-B from the state_dict
(`patch_encoder.stem.0.weight` key) and instantiates
`GNCAFFamilyB` with the right `hidden_dim`/`n_transformer_layers`/
`decoder_norm`. Same call works for the `demo` (h=256) ckpt.

### 5. Lost 0.7143 line-B baseline (history reference)

Same call pattern as v3.58 — eval auto-detects via the saved
`gcn.context_mlp.*` keys and loads `GCUNetPixelDecoder`.

## Minimal rsync set (copy these to a new VM)

For *one full inference loop* (Cascade only, since it's the champion):

```bash
# checkpoints
SRC=/home/ubuntu/ahaas-persistent-std-tcga/experiments
rsync -av \
  $SRC/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt  $SRC/gars_stage1_v3.8_gatv2_5hop_20260501_043445/.hydra \
  $SRC/gars_region_v3.37_full_20260502_144124/best_checkpoint.pt       $SRC/gars_region_v3.37_full_20260502_144124/.hydra \
  destination:/path/to/checkpoints/

# code
rsync -av --exclude='.git' --exclude='wandb' \
  /home/ubuntu/auto-clam-seg/recovered/scaffold/ \
  destination:/path/to/auto-clam-seg/recovered/scaffold/

# python deps (off the .venv pyproject)
rsync -av /home/ubuntu/profile-clam/.venv  destination:/path/to/.venv
# OR rebuild: pip install torch torch-geometric tifffile zarr hydra-core wandb numpy scikit-image scipy scikit-learn timm pyyaml omegaconf tqdm
```

Total cascade-only payload: **~62 MB checkpoints + ~3 MB code**.

## Notes on portability

* Every checkpoint is plain `torch.save({...})` — no model-class
  pickling, just `state_dict`. Code on the destination must match
  the architecture class names declared in
  `auto-clam-seg/recovered/scaffold/eval_gars_*.py:load_*`.
* All eval scripts auto-detect the right model class from the
  state_dict — you don't need to pass model-class flags.
* `eval_positives_only=true` will restrict to GT-positive slides
  (legacy mode). Default `false` includes the 250 negative slides
  for honest TLS/GC FP metrics.
* For "all TCGA slides" inference (not just the held-out folds),
  pass `slide_offset` / `slide_stride` for parallelism, and either
  drop the fold-filter via a custom config or set `fold_idx` per
  shard.
