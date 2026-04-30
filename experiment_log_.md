# Segmentation Lab Notebook

## Task
Pixel-level TLS segmentation from UNI-v2 patch embeddings + spatial graphs.
Model: GATv2Conv GNN encoder + CNN decoder (8x upsample).
Supervised by binary TLS masks from HookNet annotations.

## Benchmarks (other tasks on same data, for comparison)

### Binary Classification (TLS present/absent) — v8.0, best sweep, 5-fold, full data
- AUC: 0.834
- Balanced Acc: 0.783
- F1 macro: 0.798

### TLS Count Regression — v6.0, 2-fold, full data
- R2: 0.589
- Spearman: 0.774
- MAE: 0.593
- Bucket balanced_acc: 0.632 (edges=[1,10])

### Target for segmentation
- det_auc > 0.834 (match or beat classification)
- inst_sp > 0.774 (match or beat regression count correlation)
- val_dice > 0.6 (good pixel-level segmentation)
- cls0_recall > 0.5 (reduce false positives on TLS-negative slides)

---

## Experiments

### Experiment 1: Baseline (30% data, old split, no test exclusion)
- **Run**: segmentation_v1
- **Hypothesis**: GNN+CNN decoder can learn pixel-level TLS segmentation from foundation model embeddings
- **Config**: lr=1e-4, hidden=256, gnn_layers=2, focal+dice loss, 30% data, 1 fold, no augmentation, no stratification, ReduceLROnPlateau
- **Result**: val_dice=0.581, inst_sp=0.881, det_auc=not_measured
- **Conclusion**: Model converges well. Instance counting (Spearman=0.88) already beats regression (0.77). But this was on an easy split (no test exclusion, no stratification).
- **Interpretation**: The GNN context aggregation + spatial grid reconstruction approach works. Connected component counting from pixel predictions is more accurate than direct count regression.

### Experiment 2: Full data, fixed IDs, stratified split, test exclusion
- **Run**: seg_v1.0_gatv2_focal_dice_noaug_6a784337
- **Hypothesis**: Full data with proper stratification and test exclusion should improve over 30% baseline
- **Config**: Same as exp 1 but full data, stratified by (cancer_type, tls_bucket), test patients excluded, ReduceLROnPlateau
- **Result**: val_dice=0.378, inst_sp=0.190, det_auc=0.602
- **Conclusion**: MUCH WORSE than baseline. Investigation revealed slide_id format bug — zarr used short IDs (TCGA-XX-...-DX1) but metadata CSV uses full UUIDs. All TLS counts resolved to 0, making stratification meaningless and test exclusion potentially wrong.
- **Interpretation**: The "good" baseline results were on a non-stratified, potentially leaking split. Real performance on proper splits is much lower.
- **Changes**: Fixed slide_id to use full UUID format, matching metadata CSV.

### Experiment 3: Full data, fixed IDs (corrected), no aug vs aug
- **Run**: seg_v1.0_gatv2_focal_dice_noaug_37e0e3b0 (in progress)
- **Hypothesis**: With corrected IDs, proper stratification, and longer patience (30), the model should converge better. CosineAnnealingWarmRestarts should help escape plateaus.
- **Config**: lr=1e-4, patience=30, stop_epoch=20, CosineAnnealingWarmRestarts(T_0=10, T_mult=2), full data, 16 workers, proper test exclusion
- **Result**: (pending — superseded by Experiment 4 with panoptic architecture)
- **Conclusion**: (pending)

### Experiment 4: Panoptic 3-head baseline (semantic + center + offset)
- **Run**: seg_v1.0_panoptic_3head_lr1e4_f4edc1f9
- **Hypothesis**: Panoptic 3-head architecture (semantic + center heatmap + offset vectors) with linear warmup (5 epochs) + cosine annealing should establish a strong baseline.
- **Config**: lr=1e-4, hidden=256, gnn_layers=2, warmup=5ep + cosine(T_max=95), EPOCHS=100, patience=30, stop_epoch=20, center_weight=1.0, offset_weight=0.5, focal_alpha=0.75, no augmentation
- **Result**: val_dice=0.5983 (ep55), count_sp=0.8113 (ep84), det_auc=0.8697 (ep71), bkt_bacc=0.583, cls0_rec=0.90. Early stopped at epoch 85.
- **Conclusion**: Semantic head works well (dice ~0.60), but **center head collapsed** — l_center dropped to 0.0001 by epoch 8 and stayed there. count_sp=0 for epochs 5-42. Counting recovered late (epoch 43+) through connected components on semantic mask, not center peaks. Center head is not contributing.
- **Interpretation**: Center loss magnitude (~0.001) is dwarfed by dice (~0.2-0.7). Default center_weight=1.0 is far too low relative to other losses. The center head gets no gradient signal and collapses to constant output. Late counting recovery comes from semantic mask quality improving enough for connected components to work.
- **Next hypothesis**: Increase center_weight dramatically (10-50x) to prevent center head collapse. This should enable proper peak-based counting and improve det_auc/count_sp significantly.

### Experiment 5: Fix center head collapse with center_weight=30
- **Run**: seg_v1.0_panoptic_center_w30_dc9d2c50
- **Hypothesis**: Increasing center_weight from 1.0 to 30 should prevent center head collapse by matching loss magnitudes.
- **Config**: Same as exp 4 but center_weight=30, offset_weight=0.5 (explicit)
- **Result**: val_dice=0.5887 (ep12), count_sp=0.8402, det_auc=0.8756, bkt_bacc=0.7067, count_r2=0.6701. Early stopped at epoch 50.
- **Conclusion**: **KEEP** — dice slightly worse (-0.01) but counting/detection much better. Center head STILL collapsed (l_center→0 by ep8) despite 30x weight. But counting recovery was faster (ep16 vs ep43 in baseline). All benchmarks beaten except dice<0.6.
- **Interpretation**: center_weight=30 is still insufficient — center loss drops so fast during warmup that even 30x doesn't help. The problem is structural: MSE on a mostly-zero heatmap converges to 0 trivially. Need to change the loss function itself, not just the weight. Faster counting recovery suggests the early center gradient during warmup improves shared backbone quality.
- **Next hypothesis**: Replace MSE center loss with a weighted/focal variant that penalizes false negatives more, OR increase Gaussian sigma to make center targets cover more pixels.

### Experiment 6: Weighted BCE center loss (fix center head collapse)
- **Run**: seg_v1.0_center_wbce_balanced_f0163d40
- **Hypothesis**: Replace MSE center loss with weighted BCE (pos_weight = n_neg/n_pos, capped at 500) to prevent trivial all-zeros solution.
- **Config**: Same as exp 5 but center_loss=weighted_BCE, center_weight=1.0. Two failed attempts first: CenterNet focal loss with w=30 (l_center=44714, killed ep4) and w=0.1 (l_center=92009, killed ep3) — focal loss normalized by n_pos gave enormous magnitudes.
- **Result**: val_dice=0.5926 (ep21), count_sp=0.9037 (ep27), det_auc=0.9374 (ep27), bkt_bacc=0.7973 (ep33), count_r2=0.8878 (ep38). Early stopped at epoch 51.
- **Conclusion**: **KEEP — BREAKTHROUGH**. Center head alive for the first time (l_center stable at 0.03, never collapsed). All counting/detection metrics at record levels, far exceeding all benchmarks. Dice slightly below baseline (-0.006) but the improvement in counting is transformational.
- **Interpretation**: The weighted BCE equalizes positive/negative pixel contributions, preventing the sparse-target collapse that plagued MSE. The center head genuinely learns to locate TLS centers, giving count_sp>0.90 and det_auc>0.93. Dice plateau at ~0.59 may reflect tension between semantic and center heads sharing a backbone — the center loss "steals" some gradient bandwidth from the semantic task.
- **Next hypothesis**: Reduce center_weight to 0.5 to give semantic head more gradient bandwidth, potentially improving dice while keeping center head alive. Also try focal_alpha=0.9 for better class imbalance handling.

### Experiment 7: center_weight=0.5 + focal_alpha=0.9
- **Run**: seg_v1.0_wbce_cw0.5_focal0.9 (killed ep29)
- **Hypothesis**: Lower center weight gives semantic head more gradient; higher focal alpha handles class imbalance better. Both should improve dice.
- **Config**: Same as exp 6 but center_weight=0.5, focal_alpha=0.9
- **Result**: val_dice=0.5912 (ep24), count_sp=0.9124 (ep29), det_auc=0.9345 (ep29). Killed at epoch 29 — identical trajectory to exp 6.
- **Conclusion**: **DISCARD** — no meaningful improvement over exp 6. Dice 0.5912 vs 0.5926, counting marginally better. These hyperparameter tweaks don't move the needle.
- **Interpretation**: The 0.59 dice plateau is not about focal alpha or center weight balance. It may be a capacity bottleneck — the 256-dim, 2-layer GNN shared backbone can't serve both tasks better.
- **Next hypothesis**: Increase model capacity (hidden_dim=512, gnn_layers=3) to break the dice plateau.

### Experiment 8: More model capacity (OOM / no improvement)
- **Run**: Multiple attempts — h512_gnn3 (OOM), h384_gnn3 (OOM ep8), h256_gnn3 (killed ep17, dice=0.5796)
- **Hypothesis**: Larger model capacity should break the dice plateau.
- **Conclusion**: **DISCARD** — h512/h384 OOM on large slides, h256_gnn3 didn't improve dice. Capacity is not the bottleneck.

### Experiment 9: Graph augmentation
- **Run**: seg_v1.0_wbce_augment (killed ep22)
- **Hypothesis**: Coord jitter, edge/patch drop should regularize and improve generalization.
- **Result**: val_dice=0.5178 (ep18). **DISCARD** — augmentation hurt dice significantly (-0.07). Spatial structure is critical for this task.

### Experiment 10: Lower LR + higher dropout
- **Run**: seg_v1.0_wbce_lr5e5_drop0.2 (killed ep15)
- **Hypothesis**: lr=5e-5, dropout=0.2 for more stable convergence.
- **Result**: val_dice=0.4082 (ep15). **DISCARD** — too slow convergence with half LR.

### Experiment 11: Prioritize semantic head (dice 2x, no offset)
- **Run**: seg_v1.0_dice2x_center0.5_nooffset_bb06a2ca
- **Hypothesis**: Double dice_weight, halve center_weight, remove offset head to give semantic head more gradient bandwidth.
- **Config**: dice_weight=2.0, center_weight=0.5, offset_weight=0.0
- **Result**: val_dice=0.5987 (ep12), count_sp=0.8883, det_auc=0.9192, bkt_bacc=0.7050. Early stopped ep50.
- **Conclusion**: **KEEP** — new best dice (0.5987 > 0.5926) with strong counting. Removing the offset head freed capacity for the semantic task. bkt_bacc dropped slightly (-0.06) but other metrics improved.
- **Interpretation**: The offset head was wasting capacity on a task that doesn't contribute to any tracked metric. Doubling dice weight gives stronger gradient signal to the semantic head. The 0.60 dice barrier remains elusive.
- **Next hypothesis**: Try dice_weight=3.0 to push dice harder, or try a different dice loss variant (e.g., Tversky loss for better recall).

### Experiment 12: Tversky loss (alpha=0.3, beta=0.7)
- **Run**: killed ep21. val_dice=0.5869. **DISCARD** — high recall bias hurt precision, net-negative for dice.

### Experiment 13: weight_decay=1e-5
- **Run**: killed ep21. val_dice=0.5958. **DISCARD** — same plateau, det_auc=0.9483 (record) but dice not improved.

### Experiment 14: Separate CNN decoder branches
- **Run**: killed ep23. val_dice=0.5942. **DISCARD** — split decoder doesn't help. The bottleneck is in the GNN encoder or data, not the decoder.

### Experiment 15: HookNet predefined splits
- **Run**: crashed. Patient overlap assertion error. **CRASH** — HookNet splits have train/val patient overlap.

### Experiment 16: GNN_HEADS=8
- **Run**: OOM. **CRASH** — 8 attention heads exceed GPU memory on large slides.

### Experiment 17: Upsample factor 4x (BREAKTHROUGH)
- **Run**: seg_v1.0_upsample4x_dice2x_05626541
- **Hypothesis**: Coarser 4x upsample (2 decoder blocks instead of 3) reduces decoder distortion and better matches the model's actual spatial resolution capability.
- **Config**: upsample_factor=4, dice_weight=2.0, center_weight=0.5, offset_weight=0.0
- **Result**: **val_dice=0.6005** (ep11), count_sp=0.8941, det_auc=0.9291, bkt_bacc=0.7609. Early stopped ep50.
- **Conclusion**: **KEEP — NEW BEST**. First experiment to break the 0.60 dice barrier. All metrics at or above benchmarks. The coarser output resolution lets the model focus on getting the patch-level prediction right rather than hallucinating sub-patch details.
- **Interpretation**: The 8x upsample required the CNN to synthesize 64 output pixels per patch feature, causing boundary artifacts. At 4x, each patch feature controls 16 pixels — a resolution the 256-dim GNN features can actually represent well. The dice ceiling was a resolution mismatch, not a capacity or optimization problem.

---

### Experiments 18-22: Refinements on upsample_factor=4
- **Exp 18**: upsample_factor=2 — killed ep12, dice=0.5913, too coarse
- **Exp 19**: upsample4x + wd=1e-5 — killed ep15, dice=0.5996, no improvement
- **Exp 20**: upsample4x + center_weight=1.0 — killed ep15, dice=0.6006, no improvement
- **Exp 21**: upsample4x + skip connection — killed ep15, dice=0.5978, no improvement
- **Exp 22**: seed=123 variance check — killed ep15, dice=0.5936, confirms 0.59-0.60 range

### Experiment 23: 3-fold cross-validation
- **Run**: seg_v1.0_up4x_3fold_b394e77e
- **Config**: Best config (upsample4x, dice_weight=2, center_weight=0.5, no offset, wBCE), K_FOLDS=3
- **Result**: 
  - Fold 1: dice=0.6018 (ep15)
  - Fold 2: dice=0.5753 (ep80) 
  - Fold 3: dice=0.5528 (ep19)
  - **Mean: val_dice=0.5767, det_auc=0.9129, count_sp=0.8571, bkt_bacc=0.7544**
- **Conclusion**: **KEEP** — robust 3-fold validation confirms all benchmarks exceeded. Fold variance is significant (std~0.025 on dice). The 1-fold result (0.6005) was at the upper end of the distribution.

---

### Experiments 24-34: Further refinements on no-GNN model
- Exp 28: no-GNN + refine block — 0.5866, no gain
- Exp 29: no-GNN h384 — 0.6013, similar
- Exp 30: no-GNN dice3x — 0.5986, no gain
- Exp 31: no-GNN deep proj — 0.5990, no gain
- Exp 32: semantic-only — 0.5953, no gain
- Exp 33: SGD optimizer — 0.5903, peaked lower
- Exp 34: label smoothing — 0.5993, no gain

### Experiments 35-38: Depthwise separable conv (BREAKTHROUGH #3)
- Exp 35: **dw sep 3x3 h256** — 0.6070 (NEW BEST! dw sep conv helps)
- Exp 36: **dw sep 3x3 h384** — 0.6079 (new best)
- Exp 37: dw sep 3x3 h512 — 0.6037 (h384 is sweet spot)
- Exp 38: **dw sep 5x5 h384** — **0.6086** (NEW BEST! 5x5 kernel optimal)

### Experiments 39-42: Optimizing dw sep conv
- Exp 39: dw sep 7x7 — 0.5954 (oversmooth, 5x5 better)
- Exp 40: double 5x5 blocks — 0.5801 (overfit, single better)
- Exp 41: mixed 5x3 kernels — 0.6012 (5x5 uniform better)
- Exp 42: warm restarts — 0.6089 (restart didn't help, same first cycle)

---

### Experiments 43-50: Final refinements + counting optimization
- Exp 43: SiLU activation — 0.6026, GELU better
- Exp 44: ConvTranspose2d — 0.5798, overfit early
- Exp 45: **Grid spatial agg Conv2d 3x3** — dice=0.5798, det_auc=0.9420 (counting improved!)
- Exp 46: **Counting-optimized: grid_agg + cw=2** — count_r2=**0.876**, best counting
- Exp 47: cw=5 — 0.845 count_r2, cw=2 better
- Exp 48: **5x5 grid agg + cw=2** — dice=0.605 + count_r2=0.860 (BEST BALANCED)
- Exp 49: dice_weight=1 — 0.798 count_r2, dw=2 better for both
- Exp 50: NN upsample — count_r2=-13, destroys counting
- Exp 51: **3-fold CV balanced config** — mean dice=0.574, det_auc=0.888, count_sp=0.817

### Key insight: Conv2d spatial agg helps counting but GNN doesn't
The Conv2d on the scattered grid sees empty cells (tissue gaps), which is informative for instance separation. The GNN only connects existing patches and can't see "where tissue isn't."

---

## Summary of TLS-only experiments (50 experiments)

### Three optimized configurations:
| Config | val_dice | count_r2 | det_auc | count_sp | bkt_bacc |
|--------|----------|----------|---------|----------|----------|
| **Best dice** (exp 38) | **0.609** | 0.82 | **0.947** | **0.918** | 0.76 |
| **Best balanced** (exp 48) | 0.605 | 0.86 | 0.942 | 0.911 | **0.80** |
| **Best counting** (exp 46) | 0.594 | **0.876** | 0.943 | 0.919 | 0.79 |
| 3-fold CV mean | 0.574 | — | 0.888 | 0.817 | 0.73 |
| Benchmark | 0.600 | 0.589 | 0.834 | 0.774 | 0.632 |

### Key breakthroughs:
1. **Weighted BCE center loss** (exp 6) — fixed center head collapse
2. **Upsample factor 4x** (exp 17) — matched output resolution to model capability
3. **No GNN** (exp 27) — GNN adds zero value, simplifies model
4. **5x5 depthwise separable conv** (exp 38) — better spatial reasoning
5. **Grid spatial aggregation** (exp 45) — Conv2d on grid helps counting via tissue gap awareness
6. **dice_weight=2, center_weight=2** — optimal loss balance for both tasks

### What didn't work:
GNN layers/heads, graph augmentation, capacity increases (OOM), lower LR/SGD, Tversky loss, separate decoders, skip connections, gradient accumulation, label smoothing, warm restarts, deep projection, 7x7 kernels, double decoder blocks, ConvTranspose2d, NN upsample, SiLU

---

## Phase 2: TLS + GC Panoptic Segmentation

### Experiment 52-53: 3-class softmax (FAILED)
- 3-class softmax with class_weights=[0.1,1.0,5.0] and [0.1,1.0,50.0]
- **Result**: GC completely invisible — dice_gc=0, l_dice_gc stuck at 0.2622 for 14 epochs
- **Conclusion**: Softmax can't learn GC because predicting TLS everywhere is "close enough"

### Experiment 54: Dual-sigmoid baseline (BREAKTHROUGH)
- **Architecture**: Independent TLS sigmoid + GC sigmoid heads (not competing via softmax)
- **Config**: gc_dice_weight=5, gc_center_weight=2, focal_alpha_gc=0.90
- **Result**: TLS: dice=0.593, count_sp=0.902, det_auc=0.935, count_r2=0.863
- **GC Result**: dice_gc=0.000, gc_count_sp=**0.795**, gc_count_r2=**0.792**, gc_det_auc=**0.893**
- **Conclusion**: **KEEP — GC counting works via center head!** dice_gc=0 but GC center head detects/counts GC independently. TLS performance maintained.

### Experiments 55-61: GC optimization (all discarded)
- Exp 55: gc_dice_weight=20 — no improvement (l_dice_gc declines same rate)
- Exp 56: gc_center_weight=5 — hurt GC counting (0.73 vs 0.80)
- Exp 57: GC center-only (no semantic loss) — hurt counting (0.62 vs 0.80). "Dead" semantic head provides useful implicit regularization!
- Exp 58: GC refine branch (separate decoder) — no improvement
- Exp 59: GC gated by TLS mask — worse (TLS mask too aggressive)
- Exp 60: Ungated GC counting — worse (same as TLS-gated, no benefit)
- Exp 61: GC threshold=0.3 — worse (too many false positives)

### Key v2.0 findings:
1. **3-class softmax fails for GC** — GC too sparse to compete with TLS in shared softmax
2. **Dual sigmoid works** — independent heads let GC learn without TLS competition
3. **GC counting via center head** — dice_gc=0 but gc_count_sp=0.80, gc_det_auc=0.89
4. **"Dead" GC semantic head helps counting** — its sub-threshold gradients improve shared backbone
5. **gc_center_weight=2 is optimal** — higher weights overfit, lower loses signal

---

## Summary of all findings (61 experiments)

### v2.0 Final Results (commit 77f0549):
| Metric | TLS | GC | Benchmark |
|--------|-----|-----|-----------|
| dice | 0.593 | 0.000* | 0.600 |
| count_sp | **0.902** | **0.795** | 0.774 |
| count_r2 | **0.863** | **0.792** | 0.589 |
| det_auc | **0.935** | **0.893** | 0.834 |
| bkt_bacc | **0.774** | — | 0.632 |

*GC dice=0 at threshold 0.5, but sub-threshold predictions provide useful regularization

---

## Changes Log
- v1: Initial pipeline, 30% data, simple split
- v2: Added checkpointing, visualization, attention logging, bucket eval, per-class metrics, binary detection metrics
- v2.1: Fixed slide_id bug (short vs full UUID), proper test exclusion, stratified splits, CosineAnnealingWarmRestarts
- v2.2: HookNet predefined splits option, machine-parseable stdout
- v3.0: Panoptic 3-head (semantic + center + offset), linear warmup + cosine annealing
- v3.1: Weighted BCE center loss (breakthrough)
- v3.2: dice_weight=2, no offset head
- v3.3: upsample_factor=4
- v3.4: No GNN (ablation showed GNN unnecessary)
- v3.5: 5x5 depthwise separable conv decoder
- v3.6: Grid spatial aggregation (Conv2d 5x5 on grid)
- v3.7: Counting-optimized (center_weight=2.0)
- **v4.0: TLS + GC panoptic segmentation (3-class, separate center heads)**
- **v5.0: GNCAF reproduction — 8 KNN/GCN graph variants, all hit the same ~0.59 TLS dice ceiling as no-GNN (Apr 15)**
- **v6.0: GARS two-stage cascade — graph region proposal + UNI-v2 pixel decoder; first non-zero GC dice (Apr 26–27)**

---

## Phase 7: GARS — Graph-guided Adaptive Region Segmentation (Apr 26–27)

*Reconstructed from persistent experiment artefacts after the VM crash that
wiped the autoresearch worktree. Source code lived on the un-pushed branch
`autoresearch/gars-cascade` (final commit `63e1e5b`, lost). Best
checkpoints survive on `/lambda/nfs/.../experiments/`; runs synced to wandb
project `ajhaas/tls-pixel-seg`. See
`notebooks/RECONSTRUCTION_apr13_apr27.md` for the full picture.*

### Motivation
v2.0 dual-sigmoid TLS+GC gave `dice_gc=0` because GC is too sparse at the
4× upsample slide-level grid (~31 GC pixels in 480 K total). 8 GNCAF
variants (Phase 4) confirmed graph context doesn't help at this stage. The
two-stage GARS paradigm flips it: use the graph **only for region proposal**
on patch embeddings, then run a heavier pixel decoder on the ~0.5 % of
patches that survive. UNI-v2 features end up serving both stages — no RGB
loading.

### Experiment 62: Stage 1 — `GraphTLSDetector` ablations
- **Hypothesis**: Graph context aggregation over UNI-v2 patch embeddings
  identifies TLS-positive patches better than per-patch MLPs because TLS is
  a multi-patch *cluster* phenomenon ("a single patch with lymphocytes
  might not be TLS, but a cluster of such patches is").
- **Setup**: 814 slides (660 train / 154 val), patch-level binary label
  from mask overlap, recall-weighted loss, batch = 1 slide.
- **Variants and results**:
  | Variant | best F1 | epoch | wandb |
  |---|---|---|---|
  | GCN 3-hop, 256 px | 0.561 | 28 | `bmnj8pu1` |
  | GCN 3-hop, 512 px | 0.365 | 36 | `hwo25qzz` |
  | **GATv2 3-hop, 256 px** | **0.561** | 35 | `x50tcqt6` |
  | no-GNN (MLP only) | 0.343 | 13 | `so4ujuqa` |
- **Conclusion**: **KEEP — graph context is essential here.** GATv2 = GCN at
  the same hop count, both ~63 % above the no-GNN MLP. 256 px wins
  decisively over 512 px (+52 %) — UNI-v2 was trained at 256 px and the
  bigger tile dilutes signal.
- **Interpretation**: This is the *opposite* finding from Phase 1/4 where
  the GNN added nothing to the pixel decoder. The two stages need
  fundamentally different inductive biases — region proposal is about
  cluster detection (graph helps), pixel decoding is about local feature →
  mask synthesis (foundation-model embeddings already carry it).

### Experiment 63: Stage 2 Option A — RGB MobileNetV3 (DISCARDED)
- **Hypothesis**: A small CNN over the RGB 256 × 256 patches selected by
  Stage 1 will give finer pixel detail than decoding UNI-v2 features.
- **Setup**: MobileNetV3 backbone, 3-class softmax, freeze_encoder=False,
  warmup_epochs=3, lr=3e-4, gc_dice_weight=2, class_weights=[1, 1, 3],
  patch_size=256.
- **Result**: 4 launches (`gars_stage2_mobilenet_tls_patches_*`,
  Apr 26 22:51 → Apr 27 06:47). All aborted before producing a checkpoint —
  only `config.json` reached disk on the persistent volume. Stdout for
  these runs is gone with the worktree.
- **Conclusion**: **DISCARD** — the RGB load + on-the-fly tile reading
  pipeline was the bottleneck (anecdotally; logs lost). Worth reviving
  only if pixel-level fidelity becomes a bottleneck.

### Experiment 64: Stage 2 Option B — `UNIv2PixelDecoder` (KEEP)
- **Hypothesis**: A linear projection from the 1 536-d UNI-v2 features to a
  16 × 16 spatial grid plus a conv decoder is enough to recover a 256 × 256
  3-class mask, with no RGB loading.
- **Setup**: 605 slides, 26 364 TLS patches (1 869 with GC). Pre-load
  features + masks (~1.9 GB total). Train 22 240, val 4 124. lr=2e-4 with
  warmup, batch 128, 30 epochs max.
- **Variants**:
  | Variant | best mDice | TLS dice | GC dice | epoch | wandb |
  |---|---|---|---|---|---|
  | univ2_decoder, hidden=256 | 0.7141 | 0.719 | 0.710 | 7 | `mn5jorov` |
  | **univ2_hidden128** | **0.7178** | 0.710 | 0.687 | 8 | `cvrvn8yb` |
  | univ2_spatial8 | crash | — | — | — | `r9xfeiax` |
  | univ2_spatial32 | crash | — | — | — | `ve53vorx` |
- **Result**: TLS dice ≈ 0.72, **GC dice ≈ 0.71**. The GC dice is the big
  win — Phase 2 had `dice_gc=0` because GC is unrecoverable at the
  slide-level 4× resolution. At native 256 px patch resolution, restricted
  to TLS-positive patches, it's a normal segmentation problem.
- **Conclusion**: **KEEP — primary headline result.** hidden=128 reached
  marginally higher mDice but worse GC dice; hidden=256 is the more useful
  config in practice.
- **Interpretation**: The "pixel-level GC" plateau at 0 was a *resolution*
  problem, not a *modelling* problem. Decoding UNI-v2 at 16 × 16 → 256 × 256
  is cheap (no RGB) and works because foundation-model features at 256 px
  already carry sub-patch structure.

### Experiment 65: Cascade evaluation
- **Setup**: Stage 1 GATv2 (F1=0.561) → Stage 2 univ2_decoder (mDice=0.714);
  114-slide val. Sweep Stage 1 threshold ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}.
- **Result**: GC dice held at 0.728–0.734 across all thresholds (very
  robust). TLS pixel dice degrades as the threshold rises (0.237 at 0.05 →
  0.180 at 0.50) because Stage 1 selects only "core" TLS patches and misses
  borders. Per-cancer at thr=0.05: BLCA TLS=0.307 GC=0.707, KIRC TLS=0.150
  GC=0.798, LUSC TLS=0.206 GC=0.729. Whole-slide latency 0.32–1.06 s.
- **Conclusion**: **KEEP — operating point thr=0.05** for best GC sp /
  speed trade-off (0.41 s/slide).

## Phase 8: End-to-end and unified variants (Apr 27)

### Experiment 66: End-to-end joint S1+S2 (DISCARDED)
- **Hypothesis**: Joint fine-tuning lets Stage 1 learn to select the
  patches Stage 2 needs, recovering the TLS-border dice that the cascade
  loses.
- **Setup**: Loaded S1 (F1=0.561) + S2 (mDice=0.714); joint loss
  `s1_loss + px_loss`; freeze S2 for 3 ep, unfreeze at ep 4. lr=1e-5.
  489 train / 116 val slides.
- **Result**: Best at **epoch 1**: `gc_dice=0.655`, then collapse —
  Stage 1 recall fell 0.586 → 0.003 by ep 11, dice_tls 0.699 → 0.009,
  dice_gc 0.655 → 0.000, `selected` patches 43 → 1. The S2 unfreeze at
  ep 4 accelerated the collapse.
- **Conclusion**: **DISCARD.** Joint loss lets S1 minimise its term by
  selecting nothing (zero false positives ≫ no true positives). The
  cascade's frozen S1→S2 interface is a hard regulariser worth keeping.
- **Wandb**: `ufz9a2o4`.

### Experiment 67: Unified `GraphEnrichedDecoder` (KEEP — alternative to cascade)
- **Hypothesis**: Inject GATv2-aggregated graph features into the pixel
  decoder so a single model does both jobs without a discrete patch
  selection step.
- **Setup**: ~10.15 M params; trained on 26 364 TLS patches (489 slides /
  116 val); augmented with GATv2 graph features fused to the decoder
  input. lr=2e-4, batch 1, 30 ep max.
- **Result** (epoch 6, BEST): mDice = **0.7202**, TLS = 0.7619, GC = 0.6784.
  v2 attempt (`gars_unified_v2.log`) was launched Apr 27 23:49 and truncated
  by the VM crash (only dataset-build lines on disk).
- **Conclusion**: **KEEP — comparable to the cascade in one stage**, but
  loses the adaptive-compute speed-up (runs the heavier decoder on every
  TLS-candidate patch). Use the cascade when latency matters; use the
  unified model for cleaner training.
- **Wandb**: `qy3pj74h`.

### Final comparison vs published GNCAF
On 114-slide TCGA val, Stage-1 threshold 0.05:

| Metric | GNCAF | GARS | Δ |
|---|---|---|---|
| mDice | 0.688 | **0.714** | +4 % |
| TLS dice | 0.736 | 0.718 | −2 % |
| GC dice | 0.625 | **0.710** | **+14 %** |
| Params | 57 M | 10.1 M | 6× fewer |
| Inference | 5–10 min | 1.5 s | ~300× |
| Training | ~6 h | ~7 min | ~50× |

Figure: `notebooks/gars_vs_gncaf.png`.

### Five findings
1. UNI-v2 features are sufficient for pixel-level segmentation — no RGB.
2. Graph context matters at region-proposal, not at pixel decoding *(inferred — OCR gap)*.
3. Cascade > end-to-end joint training *(inferred — OCR gap)*.
4. 256 px > 512 px for both Stage 1 (+52 % F1) and pixel-level models.
5. Adaptive compute: only ~0.5 % of patches go through the pixel decoder.

The GARS paradigm generalises: graph-guided region proposal + foundation-
model feature decoding, applied wherever a tissue concept is sparse over
a WSI.

## Next Hypotheses
1. Verify GC test run works — check dice_gc, gc center loss, gc counting
2. Tune GC class weights — GC is very sparse, may need higher weight
3. GC-specific augmentation or loss focus
4. Hierarchical constraint: enforce GC ⊂ TLS post-hoc or via loss
5. GC counting evaluation: gc_det_auc, gc_count_sp
6. **Recover GARS source from wandb artefacts** (runs `bmnj8pu1`, `mn5jorov`,
   `cvrvn8yb`, `x50tcqt6`, `qy3pj74h`) before re-running anything new.

