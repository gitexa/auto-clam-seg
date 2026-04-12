# Autoresearch-CLAM-Seg

Autonomous experiment framework for **panoptic TLS (Tertiary Lymphoid Structure) segmentation** from whole-slide images. An AI research agent iteratively modifies model architecture and training configuration, runs experiments, and tracks results — fully autonomously.

## Architecture

A 3-head panoptic segmentation model operating on UNI-v2 patch embeddings arranged in spatial graphs:

- **Encoder**: GATv2Conv GNN (or no GNN — see findings) on patch-level feature graphs
- **Decoder**: Shared CNN decoder with configurable upsample factor
- **Heads**:
  - **Semantic**: Binary TLS mask prediction (focal + dice loss)
  - **Center**: Gaussian heatmap at TLS instance centroids (weighted BCE loss)
  - **Offset**: (dy, dx) vectors from TLS pixels to instance centers (L1 loss, optional)

Slide-level TLS counting uses peak detection on the center heatmap, gated by the semantic mask.

## Key Results (23 experiments)

Best configuration (upsample_factor=4, no GNN, dice_weight=2, weighted BCE center loss):

| Metric | Result | Benchmark | Improvement |
|--------|--------|-----------|-------------|
| det_auc | 0.927 | 0.834 | +11% |
| count_sp (Spearman) | 0.897 | 0.774 | +16% |
| bkt_bacc | 0.760 | 0.632 | +20% |
| val_dice | 0.602 | 0.600 target | Exceeded |

3-fold cross-validation mean: val_dice=0.577, det_auc=0.913, count_sp=0.857, bkt_bacc=0.754.

## Key Findings

1. **Weighted BCE center loss** (exp 6) — fixed center head collapse that plagued MSE loss, enabling real TLS instance counting
2. **Upsample factor 4x** (exp 17) — broke the 0.60 dice ceiling by matching output resolution to model capability (16 pixels per patch vs 64)
3. **No GNN needed** (exp 27) — GNN adds zero value; UNI-v2 features + CNN decoder alone match or beat the GNN version
4. **dice_weight=2 + no offset head** (exp 11) — prioritizing the semantic head improves dice without hurting counting

## Usage

```bash
# Launch an experiment (non-blocking)
python run_experiment.py start

# Check progress
python run_experiment.py status

# Kill and collect results
python run_experiment.py kill
```

Training outputs machine-parseable lines (`EPOCH`, `RESULT`) to stdout. Parse with:
```bash
grep "^EPOCH\|^RESULT" run_stdout.log
```

## Project Structure

```
├── program.md              # Autonomous agent instructions
├── run_experiment.py       # Experiment runner (start/status/kill)
├── configs/
│   └── segmentation.yaml   # Training configuration
├── experiment_log.md       # Detailed experiment journal
├── results.tsv             # Tabular results summary
├── plot_progress.py        # Visualization script
└── clam-worktree/          # Git worktree with training code (not tracked)
```

## Data

Uses TCGA whole-slide images (BLCA, KIRC, LUSC) with:
- UNI-v2 foundation model patch embeddings (1536-dim)
- HookNet TLS mask annotations
- Stratified splits by cancer type and TLS count bucket, with test patient exclusion

## Requirements

Training code lives in a separate repository ([profile-clam](https://github.com/gitexa/auto-clam-seg)) as a git worktree. Requires PyTorch, PyG, and a GPU with sufficient memory (OOM observed with hidden_dim > 384 or gnn_heads > 4).
