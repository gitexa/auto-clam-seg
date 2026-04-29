# GARS Stage 1 — re-implementation scaffold

Scaffolded 2026-04-29 from the wandb-recovered config + log of run
`x50tcqt6` (GATv2 3-hop, F1 = 0.561 at epoch 35). The original training
source is gone — these files are a re-implementation that reproduces the
same `GraphTLSDetector` architecture.

## Files

| File | Purpose |
|---|---|
| `train_gars_stage1.py` | Training script: model + data path + loop |
| `verify_stage1_ckpt.py` | Loads every recovered checkpoint with `strict=True` into the rebuilt model — proof the rebuild matches the original |

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

## Next scaffolds

Stage 2 `UNIv2PixelDecoder` (run `mn5jorov`, mDice = 0.714) is the next
highest-value rebuild — it's the model that closed the GC pixel-dice gap
from 0 → 0.71. Ping me when you want it.
