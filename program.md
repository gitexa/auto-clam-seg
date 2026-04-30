# GARS v3 — TLS + GC pixel segmentation on TCGA

Successor to the v2.0 dual-sigmoid panoptic work (history in
`experiment_log_.md` / `program_.md`). Reproduction of GARS recovered
from wandb after the Apr 2026 VM crash, and the platform for new
experiments going forward.

## Goal

A two-stage cascade that beats the GNCAF baseline on GC pixel-dice while
running ~300× faster:

| | GNCAF (paper) | GARS v3 |
|---|---|---|
| mDice | 0.688 | 0.714 |
| GC dice | 0.625 | 0.710 |
| Inference | 5–10 min/slide | ~1.5 s/slide |
| Params | 57 M | 10.1 M |

## Architecture (verified strict-load against recovered checkpoints)

- **Stage 1 — `GraphTLSDetector`** (~825 K params, GATv2 3-hop variant):
  graph TLS gate over UNI-v2 patch embeddings. Selects ~0.5 % of patches
  as TLS-positive.
  - `proj`: `Linear(1536→256) + LayerNorm`
  - `gnn_layers`: `n × {GATv2Conv(256, 64, heads=4) | GCNConv(256, 256)}` with residual + post-norm
  - `head`: `Linear(256, 128) → GELU → Dropout → Linear(128, 1)`
- **Stage 2 — `UNIv2PixelDecoder`** (~9.3 M params, hidden=64): per-patch
  3-class segmentation (bg / TLS / GC) at native 256 × 256 resolution.
  - `spatial_basis`: `Linear(1536→512) → GELU → Linear(→16²·C) → reshape → LayerNorm`
  - `dec0..dec3`: `Bilinear×2 + DoubleConv(in→out)`, fixed channel
    sequence `C → 64 → 32 → 16 → 8`, spatial `16 → 32 → 64 → 128 → 256`
  - `seg_head`: `Conv2d(8, 3, 1)`
- **Cascade**: Stage 1 scores all patches → threshold → Stage 2 decodes
  selected patches → stitch per-patch argmax tiles into a slide-level grid.

Reference papers in `paper/`: `GNCAF.pdf`, `GCUNET.pdf`.

## Data

- **Patch features**: UNI-v2 (1536-d) at 20× / 256 px in zarr v3 format
  (BLCA + KIRC + LUSC). Staged to local SSD at
  `/home/ubuntu/local_data/zarr/`.
- **Masks**: HookNet annotations, 767 multi-class TIFs (0=bg, 1=TLS,
  2=GC), staged to `/home/ubuntu/local_data/hooknet_masks_tls_gc/`.
- **Splits**: `prepare_segmentation.create_splits` — patient-stratified
  5-fold by (cancer_type, tls_bucket); 814 slides post test-exclusion;
  for `k_folds=1` fold 0 is val (~166 slides), folds 1-4 are train.
- **Stage 2 patch cache**: `/home/ubuntu/local_data/tls_patch_dataset.pt`
  (~4.7 GB), built once by `tls_patch_dataset.py`.

## Configs (Hydra)

All training scripts use Hydra. Configs live under
`recovered/scaffold/configs/`:

```
configs/
├── stage1/{config,model/{gatv2_3hop,gcn_3hop,no_gnn},train/stage1}.yaml
├── stage2/{config,model/{univ2_decoder_h64,univ2_decoder_h128},train/stage2}.yaml
└── cascade/config.yaml
```

Override on the CLI:

```bash
python train_gars_stage1.py model=gcn_3hop train.lr=5e-4 label=v3.0_lr5e4
python train_gars_stage2.py model=univ2_decoder_h128
```

Each run dumps the resolved config to `<out_dir>/config.yaml` and Hydra's
auto-dump to `<out_dir>/.hydra/{config,hydra,overrides}.yaml`.

## Wandb

All training scripts log to `wandb` project `tls-pixel-seg`. Disable
with `WANDB_MODE=disabled` (e.g. for smoke tests).

## Quick-start

```bash
cd /home/ubuntu/auto-clam-seg/recovered/scaffold

# 1. (One-time) verify model architecture matches recovered checkpoints.
python verify_stage1_ckpt.py     # All OK — 824k/626k/427k params
python verify_stage2_ckpt.py     # All OK — 9.3M/17.7M params

# 2. (One-time) stage features + masks to local SSD.
python stage_features_to_local.py     # ~10 min (107 GB zarr)
python tls_patch_dataset.py            # ~20 min (builds 4.7 GB cache)

# 3. Reproduce baselines.
python train_gars_stage1.py            # v3.0 (default: GATv2 3-hop)
python train_gars_stage2.py            # v3.1 (default: hidden=64)
python eval_gars_cascade.py \
    stage1=/path/to/stage1/best_checkpoint.pt \
    stage2=/path/to/stage2/best_checkpoint.pt
```

## Reference targets (from recovered runs)

- **Stage 1 GATv2 3-hop**: best F1 = 0.561 at ep 35
- **Stage 2 univ2_decoder (h=64)**: best mDice = 0.7141 at ep 7
- **Cascade (thr=0.05)**: mDice = 0.486, TLS dice = 0.237, GC dice = 0.734,
  TLS sp = 0.771, GC sp = 0.754, ~0.4 % patches selected, ~0.41 s/slide

## References

- `paper/GNCAF.pdf`, `paper/GCUNET.pdf` — baselines being compared against
- `recovered/README.md` — recovery summary (what was recovered from wandb)
- `recovered/scaffold/README.md` — architecture details + open decisions
- `notebooks/RECONSTRUCTION_apr13_apr27.md` — full timeline of phases 1–8
- `notebooks/verify_patch_mask_alignment.ipynb` — alignment checks
- `experiment_log_.md`, `experiment_log_tls.md` — v2.x history
- `program_.md`, `program_tls.md` — v2.x driver docs
