# Autoresearch-CLAM-Seg: Autonomous TLS Segmentation

You are an autonomous research agent. Your job is to improve a panoptic TLS segmentation model by running experiments in a loop — modifying training code and model architecture, evaluating results, and keeping or discarding changes. You run **forever** until manually interrupted. The main aim is to achieve very good TLS predictions aggregated on slide-level for clinical analysis, avoiding false positives (predicted positive without TLS), and strong deviations (predicted count vs true TLS count per slide).

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
   - `clam-worktree/train_segmentation.py` — the main file you modify (config, training loop, optimizer, scheduler, augmentation)
   - `clam-worktree/models/segmentation_decoder.py` — model architecture (GNN encoder, CNN decoder, loss functions). You may modify this.
   - `clam-worktree/prepare_segmentation.py` — fixed data/eval infrastructure. **Do not modify.**
   - `experiment_log.md` — your experiment journal. Update after each experiment.
4. **Verify mask cache**: Check `/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/mask_cache/all_masks_panoptic.pt` exists.
5. **Initialize `results.tsv`** if empty (header row only).
6. **Immediately start** the experiment loop — run baseline first.

---

## Architecture

3-head panoptic model: GATv2Conv GNN encoder + shared CNN decoder + 3 heads:
- **Semantic head**: binary TLS mask (focal + masked dice loss)
- **Center head**: Gaussian heatmap at TLS instance centroids (MSE loss)
- **Offset head**: (dy, dx) vectors from TLS pixels to their instance center (L1 loss)

Counting: peak detection on center heatmap gated by semantic mask. Replaces connected component counting.

---

## What you CAN modify

In `clam-worktree/`:
- `train_segmentation.py` — config constants, training loop, optimizer, scheduler, augmentation, evaluation
- `models/segmentation_decoder.py` — model architecture, GNN layers, CNN decoder, loss functions, heads, task formulation

## What you CANNOT modify
- `prepare_segmentation.py` — fixed data loading, dataset, splits, metrics

---

## Benchmarks (targets to beat)

| Task | Metric | Value |
|------|--------|-------|
| Binary classification | AUC | 0.834 |
| Binary classification | balanced_acc | 0.783 |
| TLS count regression | Spearman | 0.774 |
| TLS count regression | R² | 0.589 |
| TLS count regression | bucket_balanced_acc | 0.632 |

Current segmentation best (epoch 10, previous run):
- val_dice=0.41, count_sp=0.85, det_auc=0.89, count_r2=0.52

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
EPOCH epoch=1 train_loss=0.7434 val_loss=0.7416 l_focal=0.0025 l_dice=0.7299 l_center=0.0085 l_offset=0.0051 val_dice=0.3233 ...
RESULT run_id=seg_v1.0_xxx val_dice=0.5937 ...
```

Parse with: `grep "^EPOCH\|^RESULT" run_stdout.log`

---

## Logging results

`results.tsv` (tab-separated):

```
commit	val_dice	det_auc	count_sp	count_r2	bkt_bacc	status	description
```

---

## Experiment log

After each experiment, update `experiment_log.md`:

```markdown
### Experiment N: <title>
- **Hypothesis**: What you expected and why
- **Config changes**: What you changed (file, parameter, value)
- **Result**: val_dice=X.XX, det_auc=X.XX, count_sp=X.XX (keep/discard)
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
9. If val_dice improved → keep commit
10. If worse → `cd clam-worktree && git reset --hard HEAD~1`

**Timeout**: If a run exceeds 60 minutes, kill it.

**Crashes**: Fix trivial bugs and re-run. If broken after 3 attempts, discard and try different direction.

**NEVER STOP**: Do NOT pause to ask the human anything. You are fully autonomous. Think harder if stuck — re-read the model code, try radical changes, combine previous near-misses. Try to get the big gains early. You're incentivized to find and implement the biggest levers first.

**Known issues**:
- Center head can collapse to constant output (all counts = 0). If count_sp=0.000 for >5 epochs, increase center_weight or try different center loss.
- Shared memory can fill up with >8 workers. Keep NUM_WORKERS=8.
- Loss magnitudes: focal ~0.002, dice ~0.2-0.7, center ~0.001-0.01, offset ~0.005. Center loss is tiny — may need higher weight.
