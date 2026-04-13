# Autoresearch-CLAM-Seg: Autonomous TLS + GC Panoptic Segmentation

You are an autonomous research agent. Your job is to improve a panoptic TLS + GC segmentation model by running experiments in a loop — modifying training code and model architecture, evaluating results, and keeping or discarding changes. The main aim is to achieve very good TLS and GC predictions aggregated on slide-level for clinical analysis, avoiding false positives and strong deviations (predicted count vs true count per slide). **Counting accuracy is the primary clinical metric.**

---

## Setup (once, at start of session)

Do all of this automatically — do NOT ask the human for confirmation:

1. **Create run tag**: Use today's date (e.g. `apr11`). If branch exists, append `-2`.
2. **Create git worktree**:
   ```bash
   cd /home/ubuntu/profile-clam
   git worktree add /home/ubuntu/autoresearch-clam-seg/clam-worktree autoresearch/seg-<tag>
   ```
3. **Read these files** for full context:
   - This file (`program.md`)
   - `clam-worktree/train_segmentation.py` — config constants, training loop, optimizer, scheduler, augmentation
   - `clam-worktree/models/segmentation_decoder.py` — model architecture, CNN decoder, loss functions, heads
   - `clam-worktree/prepare_segmentation.py` — data loading, dataset, splits, metrics, mask caching
   - `experiment_log.md` — your experiment journal. Update after each experiment.
4. **Verify mask cache**: Check `/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/mask_cache/all_masks_panoptic_tls_gc.pt` exists (or let it build).
5. **Initialize `results.tsv`** if empty (header row only).
6. **Immediately start** the experiment loop.

---

## Architecture

Multi-head panoptic model: Linear projection + optional spatial aggregation + CNN decoder + heads:
- **Semantic head**: 3-class softmax (background=0, TLS=1, GC=2) — multi-class focal + per-class dice loss
- **TLS Center head**: Gaussian heatmap at TLS instance centroids (weighted BCE)
- **GC Center head**: Gaussian heatmap at GC instance centroids (weighted BCE)
- **Offset head**: (dy, dx) vectors (currently disabled, offset_weight=0)

Counting: peak detection on class-specific center heatmaps gated by semantic mask.

### Best architecture (from 50 TLS-only experiments):
- No GNN (proven unnecessary — UNI-v2 features are locally sufficient - might be different for TLS+GC)
- hidden_dim=384, upsample_factor=4
- 5x5 depthwise separable conv decoder
- 5x5 grid spatial aggregation (Conv2d on scattered grid, helps counting)
- Bilinear upsampling

---

## Data

- **Patch features**: UNI-v2 embeddings (1536-d) from 256px patches at 20x, in zarr format
- **Masks**: `/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/masks_tls_gc/` — 767 multi-class TIF files (0=bg, 1=TLS, 2=GC)
- **Metadata**: `df_summary_v10.csv` — has tls_num, gc_num, gc_present columns
- **Stats**: 260/767 slides have GC (~34%), GC is ~6-7% of TLS pixels, mean 1.04 GC instances/slide
- **Splits**: 5-fold stratified by (cancer_type, tls_bucket), 201 test slides excluded

---

## What you CAN modify

In `clam-worktree/`:
- `train_segmentation.py` — config constants, training loop, optimizer, scheduler, augmentation, evaluation
- `models/segmentation_decoder.py` — model architecture, CNN decoder, loss functions, heads, task formulation
- `prepare_segmentation.py` — data loading, mask caching, metrics (for structural changes like adding GC)

---

## Benchmarks (targets to beat)

| Task | Metric | Value |
|------|--------|-------|
| Binary classification | AUC | 0.834 |
| Binary classification | balanced_acc | 0.783 |
| TLS count regression | Spearman | 0.774 |
| TLS count regression | R² | 0.589 |
| TLS count regression | bucket_balanced_acc | 0.632 |

### Best TLS-only results (50 experiments):
| Metric | Best 1-fold | 3-fold mean | Config |
|--------|------------|-------------|--------|
| val_dice_tls | 0.609 | 0.574 | 5x5 dw sep, h384, no GNN |
| det_auc | 0.947 | 0.888 | same |
| count_sp | 0.918 | 0.817 | same |
| count_r2 | 0.876 | — | counting-optimized |
| bkt_bacc | 0.798 | 0.728 | balanced config |

### GC targets (new):
- dice_gc > 0.3 (first target — GC is very sparse)
- gc_det_auc > 0.7 (GC presence detection)
- gc_count_sp > 0.5 (GC instance counting)

---

## Running experiments

```bash
# Launch (non-blocking)
python run_experiment.py start

# Check progress
python run_experiment.py status

# Kill and collect results
python run_experiment.py kill
```

The training script prints machine-parseable lines:
```
EPOCH epoch=1 train_loss=... val_dice=... dice_tls=... dice_gc=... count_sp=... det_auc=... ...
RESULT run_id=seg_v1.0_xxx val_dice=... dice_gc=... count_sp=... ...
```

Parse with: `grep "^EPOCH\|^RESULT" run_stdout.log`

---

## Logging results

`results.tsv` (tab-separated):

```
commit	val_dice	det_auc	inst_sp	count_r2	bkt_bacc	status	description
```

---

## Experiment log

After each experiment, update `experiment_log.md`:

```markdown
### Experiment N: <title>
- **Hypothesis**: What you expected and why
- **Config changes**: What you changed (file, parameter, value)
- **Result**: val_dice=X.XX, dice_gc=X.XX, det_auc=X.XX, count_sp=X.XX (keep/discard)
- **Conclusion**: What you learned
- **Interpretation**: Connect to known results or patterns
- **Next hypothesis**: What to try next
```

---

## The experiment loop

LOOP FOREVER:

1. Read `results.tsv` and `experiment_log.md` for context
2. Form a hypothesis based on previous results
3. Edit files in `clam-worktree/`
4. `cd clam-worktree && git commit -am "experiment: <description>"`
5. `cd .. && python run_experiment.py start`
6. Monitor: `python run_experiment.py status` (check every 2-3 minutes)
7. When done or clearly failing: `python run_experiment.py kill`
8. Record in `results.tsv` and update `experiment_log.md`
9. If improved → keep commit
10. If worse → `cd clam-worktree && git reset --hard HEAD~1`

**Timeout**: If a run exceeds 60 minutes, kill it.

**Crashes**: Fix trivial bugs and re-run. If broken after 3 attempts, discard and try different direction.

**NEVER STOP**: Do NOT pause to ask the human anything. You are fully autonomous.

**Known issues**:
- Center head can collapse to constant output (all counts = 0). Use weighted BCE center loss (not MSE).
- GC is very sparse (~6% of TLS pixels). Needs high class_weight and gc_dice_weight.
- Shared memory can fill up with >8 workers. Keep NUM_WORKERS=8.
- GNN adds zero value for this task — don't use it. The Conv2d grid spatial aggregation helps counting instead.
