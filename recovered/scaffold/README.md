# GARS — re-implementation scaffolds

Scaffolded 2026-04-29 from the wandb-recovered configs + logs. The
original training source is gone — these files are a re-implementation
that reproduces the same `GraphTLSDetector` (Stage 1) and
`UNIv2PixelDecoder` (Stage 2) architectures.

## Files

| File | Purpose |
|---|---|
| `train_gars_stage1.py` | Stage 1: GraphTLSDetector training (per-slide patch classification) |
| `verify_stage1_ckpt.py` | Loads every recovered Stage 1 checkpoint with `strict=True` |
| `train_gars_stage2.py` | Stage 2: UNIv2PixelDecoder training (per-patch 3-class segmentation) |
| `verify_stage2_ckpt.py` | Loads every recovered Stage 2 checkpoint with `strict=True` |
| `tls_patch_dataset.py` | Stage 2 data loader: stage masks to local SSD, walk slides in parallel, build & cache the in-memory (features, masks) tensors |
| `eval_gars_cascade.py` | Cascade eval: Stage 1 → threshold → Stage 2 → slide-level mDice / TLS dice / GC dice + counting metrics, sweep over thresholds |

## Architecture (verified exact)

```
GraphTLSDetector(
  proj:        Linear(1536, 256) → LayerNorm(256)
  gnn_layers:  n_hops × {GATv2Conv(256, 64, heads=4, concat=True)
                         | GCNConv(256, 256)}            # residual + post-norm
  gnn_norms:   n_hops × LayerNorm(256)
  head:        Linear(256, 128) → GELU → Dropout(p) → Linear(128, 1)
)
```

Param counts match the surviving checkpoints exactly:

| Variant | Reconstructed | Checkpoint | strict=True |
|---|---|---|---|
| GATv2 3-hop | 824,833 | 824,833 | ✓ |
| GCN 3-hop (256 / 512 px) | 625,921 | 625,921 | ✓ |
| no-GNN MLP | 427,009 | 427,009 | ✓ |

## Recovered hyperparameters (from `x50tcqt6` config.yaml)

```
lr:           1e-3
weight_decay: 1e-4
seed:         42
epochs:       50
patience:     10
n_hops:       3
gnn_type:     gatv2
hidden_dim:   256
dropout:      0.1
patch_size:   256
pos_weight:   5
```

Loss = `BCEWithLogitsLoss(pos_weight=5)` (class imbalance: ~0.18 % TLS
patches per slide). Optimizer = AdamW; LR schedule = `CosineAnnealingLR`
over 50 epochs (matches the lr trajectory in the recovered log:
9.99e-04 → 2.54e-05 across ep 1 → 45).

## Running

```bash
# Reproduce the GATv2 3-hop run (target F1 ≈ 0.561):
python train_gars_stage1.py \
    --gnn_type gatv2 --n_hops 3 --hidden_dim 256 \
    --lr 1e-3 --epochs 50 --patience 10 \
    --pos_weight 5 --weight_decay 1e-4 \
    --label gatv2_3hop_recovered
```

Needs the `profile-clam` venv (`scipy`, `zarr`, `tifffile`, `pandas`,
`scikit-learn`, `torch`, `torch_geometric`) — re-create from
`requirements.txt` saved alongside any recovered run.

## Verification

```bash
python verify_stage1_ckpt.py
```

Output should end with `All OK` — every recovered checkpoint loads into
the rebuilt model with `strict=True`. Run this **before** any further
work to confirm your env still loads the originals.

## Open decisions worth confirming before re-running

These were not in the recovered config and were inferred from logs /
checkpoint shapes — easy to flip if you remember them differently:

1. **Residual + post-norm** in the GNN block (`norm(layer(x, edge_index) + x)`).
   Could also be pre-norm or no residual; checkpoint doesn't pin this.
2. **Patch label** = `1` if any TLS *or* GC pixel falls in the patch's mask
   cell. Could be TLS-only (mask == 1) — adjust in `patch_labels_from_mask`.
3. **Optimizer** = AdamW (vs Adam). Both compatible with the saved
   checkpoint; AdamW with `weight_decay=1e-4` is the standard pairing.
4. **Edge source** = `graph_edges_1hop` from the zarr (pre-computed, used
   by `prepare_segmentation.TLSSegmentationDataset`). This is a 4-connected
   spatial grid; the n_hops parameter just stacks more GATv2 layers, each
   extending receptive field by 1 hop on the same 1-hop edge set.

## Stage 2 — UNIv2PixelDecoder

Architecture (verified exact against `mn5jorov` and `cvrvn8yb`):

```
UNIv2PixelDecoder(
  spatial_basis: Linear(1536, 512) → GELU → Linear(512, 16²·C)
                 → reshape (B, 16, 16, C) → LayerNorm(C) → permute (B, C, 16, 16)
  dec0..dec3:    each = Bilinear ×2 + DoubleConv(in→out)
                 channel sequence: C → 64 → 32 → 16 → 8
                 spatial sequence: 16 → 32 → 64 → 128 → 256
  seg_head:      Conv2d(8, 3, 1) → 3-class logits (bg, TLS, GC)
)
```

`hidden_channels` (the SpatialBasis output dim) is independent of the
fixed decoder progression `(64, 32, 16, 8)`. dec0 is the only block whose
input channels change with `hidden_channels` — when `C ≠ 64`, dec0 maps
`C → 64`; otherwise it's `64 → 64`.

| Variant | Reconstructed params | Checkpoint params | strict=True |
|---|---|---|---|
| univ2_decoder (h=64) | 9,302,587 | 9,303,075 (+488 BN buffers) | ✓ |
| univ2_hidden128 (h=128) | 17,744,571 | 17,745,059 (+488 BN buffers) | ✓ |

Recovered hyperparameters (from `mn5jorov` config.yaml):

```
lr:              3e-4
weight_decay:    1e-4
seed:            42
epochs:          50
patience:        10
batch_size:      128
spatial_size:    16
hidden_channels: 64
class_weights:   [1, 1, 3]
gc_dice_weight:  2.0
warmup_epochs:   3
```

Loss = `CrossEntropy(weight=[1,1,3]) + 2.0 · DiceLoss(GC)` (recovered from
the `compute_loss(logits, masks, CLASS_WEIGHTS, GC_DICE_WEIGHT)` signature
in the spatial-ablation crash trace). Optimizer = AdamW; LR schedule =
linear warmup over 3 epochs then cosine decay to 0.

### Open decisions for Stage 2

1. **Activation in DoubleConv**: ReLU vs GELU — checkpoint can't
   distinguish (no params on activation). The recovered loss values
   (train_loss 3.10 → 1.22 over 17 epochs) match either. Default: ReLU.
2. **Activation in SpatialBasis** (between the two Linears): GELU
   assumed; index 1 in the `proj` Sequential has no params, so any
   parameter-less activation works.

### Stage 2 — TLS-patch data loader

`tls_patch_dataset.py` reproduces the recovered build pipeline:

1. **Stages mask TIFs to local SSD** at first run
   (`/home/ubuntu/local_data/hooknet_masks_tls_gc/`). The HookNet TIFs
   live on NFS by default; one copy + per-patch reads from `/dev/vda1`
   gives roughly an order of magnitude speedup. Idempotent — second
   run skips already-copied files. This matches the recovered log
   line `Found 767 masks on local SSD`.
2. **Excludes test patients** (170, from `df_summary_v10_test.csv` —
   matches `Loaded 170 test patients to exclude`). Patient ID is the
   first 12 chars of the slide ID (TCGA-XX-XXXX).
3. **Walks slides in parallel** (`ProcessPoolExecutor`, default 8
   workers). Per slide: open the zarr (features + coords), open the
   mask TIF at level 0 (native resolution, scale-down detection if
   the mask is at a coarser MPP), crop a 256×256 mask tile per patch,
   keep patches where `tile.max() > 0` (i.e., any TLS or GC pixel).
4. **Concatenates and caches** the assembled tensors as a single .pt
   at `/home/ubuntu/local_data/tls_patch_dataset.pt`. Subsequent runs
   skip directly to the load.

Output bundle:

```
features:     torch.float32,  (N, 1536)
masks:        torch.uint8,    (N, 256, 256)        # values in {0, 1, 2}
slide_ids:    list[str],      length N
cancer_types: list[str],      length N             # {BLCA, KIRC, LUSC}
patch_idx:    torch.int32,    (N,)                 # within-slide index
```

Memory: ~1.9 GB — fits in RAM for the full TCGA training set.

`slide_level_split()` in `train_gars_stage2.py` partitions by
**patient** (not patch) to avoid leak across the train/val boundary.

### Stage 2 — running

```bash
# First run builds the cache (~30 s/slide × 605 slides ≈ 5 min);
# every subsequent run loads it in ~1 s.
python tls_patch_dataset.py

# Reproduce the mn5jorov result (target mDice ≈ 0.714):
python train_gars_stage2.py \
    --hidden_channels 64 --spatial_size 16 \
    --lr 3e-4 --epochs 50 --patience 10 --batch_size 128 \
    --warmup_epochs 3 --gc_dice_weight 2.0 \
    --class_weights 1 1 3 --label univ2_decoder_recovered
```

### Stage 2 — verification

```bash
python verify_stage2_ckpt.py
```

Output should end with `All OK` and confirm `forward(2, 1536) ->
(2, 3, 256, 256)` for both variants.

## Cascade evaluation

`eval_gars_cascade.py` reproduces `gars_cascade_eval.log` and
`gars_cascade_low_thresh.log`. Per slide:

1. Stage 1 scores every patch → TLS probability.
2. Threshold at `THR` → select TLS-positive patch indices.
3. Stage 2 decodes each selected patch's UNI-v2 feature into a
   256×256 3-class mask (batched, default 64 patches/batch).
4. Stitch per-patch predictions into a slide-level patch-grid (one
   class per patch with GC > TLS > bg precedence).
5. Compare against the cached HookNet mask reduced to the same
   patch-grid resolution → mDice / TLS dice / GC dice.
6. Connected-component counting on the predicted grid → Spearman vs
   ground-truth instance counts.

```bash
python eval_gars_cascade.py \
    --stage1 /home/ubuntu/ahaas-persistent-std-tcga/experiments/\
gars_stage1_gars_stage1_gatv2_3hop_v2_20260426_230650/best_checkpoint.pt \
    --stage2 /home/ubuntu/ahaas-persistent-std-tcga/experiments/\
gars_stage2_gars_stage2_univ2_decoder_20260427_102326/best_checkpoint.pt \
    --thresholds 0.05 0.10 0.20 0.30 0.40 0.50
```

Target numbers (from recovered `gars_cascade_low_thresh.log` at thr=0.05):
mDice 0.486, TLS dice 0.237, GC dice 0.734, TLS sp 0.771, GC sp 0.754,
~0.4% of patches selected, ~0.41 s/slide.

## Next scaffolds

End-to-end joint and the unified `GraphEnrichedDecoder` are the
remaining pieces. The unified run `qy3pj74h` is crashed in wandb
(no `output.log`/`config.yaml` recovered), so its architecture would
have to be inferred purely from checkpoint shapes (~10.15 M params).
