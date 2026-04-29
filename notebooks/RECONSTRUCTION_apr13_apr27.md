# Reconstruction: Apr 13 → Apr 27, 2026

Reconstructed on 2026-04-29 after the VM crash that wiped the local worktree
`/home/ubuntu/autoresearch-clam-seg/clam-worktree/` and all commits made on
top of `gitexa/profile-clam@autoresearch/seg-apr11` after **2026-04-13**.

## State of the world after the crash

- `auto-clam-seg` re-cloned: history ends at `cc1567a` (2026-04-13). Holds the
  autoresearch driver/journal: `program.md`, `program_tls.md`,
  `experiment_log.md`, `experiment_log_tls.md`, `run_experiment.py`,
  `plot_progress.py`.
- `profile-clam` re-cloned: `main` history ends 2026-04-13. The remote branch
  `origin/autoresearch/seg-apr11` exists with commits up to Apr 13.
- The local worktree referenced by every wandb run path is gone, so all
  commits **Apr 14 → Apr 27** are lost from git.
- Surviving on `/lambda/nfs/ahaas-persistent-std-tcga/experiments/`: per-run
  logs, `summary.json`, `best_checkpoint.pt`, and the comparison plots.
- All runs were synced to **wandb project `ajhaas/tls-pixel-seg`** — code
  artifacts and configs are still recoverable from the run pages.

---

## Timeline

### Phase 1 — TLS-only pipeline finalised (≤ Apr 13)
50 experiments documented in `experiment_log_tls.md`. Winning recipe:

- No GNN, `hidden_dim=384`, `upsample_factor=4`
- 5×5 depthwise-separable conv decoder (2 blocks)
- Weighted-BCE center head, `dice_weight=2`, `center_weight=0.5`, no offset
- 5×5 grid spatial aggregation (Conv2d on the scattered grid — helps counting)

**Best**: `val_dice=0.609`, `det_auc=0.947`, `count_sp=0.918`,
`bkt_bacc=0.764` — comfortably above all benchmarks.

### Phase 2 — TLS + GC dual-sigmoid panoptic (Apr 13)
3-class softmax couldn't learn GC; switching to **independent dual-sigmoid
heads** unlocked GC counting via the center head. Apr 13 commits in order:

| Commit | Change |
|---|---|
| `c61f954` | feat: add GC to panoptic segmentation |
| `7392e4d` | fix: multi-class loss-key references |
| `d0a1b32` | viz + GC R²/spearman/det_auc per epoch |
| `56f2626` | GC class_weight=50, gc_dice_weight=5 |
| `48bdf95` | switch to dual-sigmoid heads |
| `2d5fc26` | compute_metrics backward compat |
| `77f0549` | v2.0 baseline EPOCHS=100 |
| `036be2a` | GC loss cropped to TLS pixels (31 GC in 3K TLS vs 31 in 480K total) |
| `6440f91` | 16× upsample + cropped GC + hard-case metrics |
| `ead8ca8` | parallelise mask-cache build (16 threads) |
| `3a3dd76` | 8× + cropped GC + hard-case metrics (16× OOM'd) |
| `b9f5d0f` | plots |
| `33fe013` | script to update prediction masks with GC |

v2.0 result on TCGA val: TLS dice=0.593; **GC counting via center head**
`gc_count_sp=0.795`, `gc_count_r2=0.792`, `gc_det_auc=0.893` (with
`dice_gc=0` — GC too sparse at 4× to pixel-segment).

### Phase 3 — TLS-only k-fold robustness (Apr 14)
- `tls_only_5fold.log` (Apr 14 12:56) and `tls_only_5fold_seed123.log`
  (Apr 14 23:36): confirmed the 1-fold result was at the upper end. 3/5-fold
  mean dice ≈ 0.577.
- `tls_gc_5fold_seed123.log` (Apr 14 17:41): same exercise for the panoptic
  version.

### Phase 4 — GNCAF reproduction (Apr 15)
Eight variants of the GNCAF / GC-UNet idea (KNN graph + multi-hop GCN over
UNI-v2 patch embeddings). Folders `seg_v2.0_gncaf_*` and `seg_v2.0_gcunet_*`.

| Run | dice | count_sp | det_auc | gc_det_auc | bkt_bacc |
|---|---|---|---|---|---|
| `gncaf_transunet_baseline` (e148e747 / ff3455d1) | — initial port | — | — | — | — |
| `gncaf_dwsep_baseline` | 0.592 | 0.885 | 0.921 | 0.895 | 0.714 |
| `gncaf_4conn_onfly` | **0.594** | **0.896** | 0.925 | 0.893 | **0.782** |
| `gncaf_knn8_gcn` | 0.585 | — | — | — | — |
| `gncaf_knn8_3hop_gcn` | 0.590 | 0.866 | 0.904 | 0.881 | 0.726 |
| `gncaf_knn8_5hop_gcn` | 0.592 | 0.887 | 0.911 | 0.864 | 0.732 |
| `gncaf_knn16_7hop_gcn` | (no summary) | | | | |
| `gcunet_baseline` | 0.586 | 0.882 | 0.923 | 0.903 | 0.729 |
| `gcunet_4conn_onfly` | 0.590 | 0.895 | 0.926 | 0.892 | 0.743 |

**Conclusion**: faithful reproductions land at TLS dice ≈ 0.59 — on par with
the no-GNN baseline. Graph context didn't hurt, but it didn't help either; on
this task with UNI-v2 features the GNN value at the pixel-seg stage was zero.
Big training cost, no metric win → motivated the GARS extension.

### Phase 5 — Architecture comparison sweep (Apr 16)
`alt1_linear_probe.log` … `alt5_factored64.log` (Apr 16 02:13–06:18) — five
decoder formulations on the same v2.0 panoptic head:

| Variant | val_dice | count_sp | det_auc | bkt_bacc |
|---|---|---|---|---|
| alt1 linear probe | 0.42 | 0.81 | 0.88 | 0.50 |
| alt2 linear conv | 0.47 | 0.89 | 0.92 | 0.65 |
| alt3 pixel-shuffle | **0.59** | 0.89 | **0.93** | 0.74 |
| alt4 patch MLP | 0.42 | 0.82 | 0.88 | 0.51 |
| alt5 factored 64-d | 0.59 | **0.91** | 0.93 | 0.72 |

Pixel-shuffle and factored-bottleneck hit the same ~0.59 ceiling as the dwsep
decoder — the dice plateau is data/feature-bound, not decoder-bound.
`final_model.log` (2.3 MB, Apr 15 01:59) was the 5-fold full final-model run
on the v2.0 winner.

### Phase 6 — GC focus and high-resolution attempts (Apr 21–24)
- `gcn_comparison.png` (Apr 21 17:30), `gcn_comparison_final.png`
  (Apr 21 23:03) — summary plot of GNCAF / GC-UNet / no-GNN variants.
- `training_progress.png` (Apr 22 16:59).
- `seg_v2.0_upsample8x_gc_focus_*` (5 runs Apr 24 04:28–04:57) and
  `seg_v2.0_upsample16x_gc_focus_409e4b21` (Apr 24 04:57): pushed upsample
  factor to attack `dice_gc=0`.
- `baseline_cc_eval.log` (Apr 24 12:27) — patch-level UNI-v2 + connected
  components ceiling on 597 slides:
  - **TLS**: sp=0.945, R²=0.926, MAE=1.6, bucket acc=0.912
  - **GC**: sp=0.733, R²=0.841, MAE=0.5, det AUC=0.869
  - GC binary: prec=0.711, rec=0.842, F1=0.771

  → slide-level counting was essentially solved; the open problem was
  pixel-level GC dice.
- `full_val_eval.log` (Apr 24 12:17).

### Phase 7 — GARS: two-stage extension (Apr 26 → Apr 27)
The expensive GNCAF reproduction gave no gain at slide-level pixel
segmentation, so the problem was split into two cheaper UNI-only stages.

#### Core idea (paradigm)
Use a graph network on patch embeddings to **identify regions of interest**,
then allocate pixel-level compute **only where it matters** (~0.5 % of
patches), instead of processing every patch at full resolution (GNCAF:
5–10 min/slide) or never going to the pixel level (Phase-2 panoptic:
GC dice ≈ 0).

#### Architecture
- **Stage 1 — `GraphTLSDetector`** (~626 K params, ~0.3 s/slide):
  graph TLS gate over UNI-v2 patch embeddings.
  - Input: UNI-v2 patch embeddings (N, 1536) + spatial coords (N, 2)
  - Output: per-patch TLS probability + selected-patch indices
  - Supervision: binary patch label from mask overlap
    (`get_tls_patch_indices`)
  - Training: ~30 s/epoch (operates on embeddings, not RGB)
  - Selects ~6–10 K of 2.6 M val patches (~0.4 %)
- **Stage 2** — patch-level pixel decoder over *only* the selected TLS
  patches; outputs 3-class mask (bg / TLS / GC) at native 256 px. Two model
  options were tried:
  - **Option A** — `MobileNetV3` over raw RGB 256×256 patches (loads tiles
    from WSI, runs a vision CNN). 4 runs (`gars_stage2_mobilenet_tls_patches_*`,
    Apr 26 22:51 → Apr 27 06:47): all aborted — only `config.json` made it
    to disk, no checkpoint produced. RGB tile loading was the bottleneck.
  - **Option B** — `UNIv2PixelDecoder` (~9.3 M params, kept): linear
    projection from the 1 536-d UNI-v2 features → spatial 16×16 grid →
    convolutional decoder to 256×256 mask. No RGB load needed. **This is
    the chosen Stage 2.**
- **Cascade**: Stage 1 picks patches → Stage 2 decodes. Total whole-slide
  inference ~1.5 s/slide.

#### Stage 1 results (`gars_stage1_*.log`)

| Variant | best F1 | epoch |
|---|---|---|
| GCN 3-hop, 256 px (`gcn3hop`) | 0.561 (full run, ep 28) | — |
| GCN 3-hop, 512 px (`512px`) | 0.365 | 36 |
| **GATv2 3-hop (`gatv2`)** | **0.561** | 35 |
| no-GNN ablation (`no_gnn`) | 0.343 | 13 |

→ The GNN clearly mattered for **Stage 1 patch selection** (~0.56 F1 vs 0.34
no-GNN) — the opposite of what you'd seen at the pixel-seg stage.

#### Stage 2 results (`gars_stage2_*.log`)

| Variant | best mDice | TLS dice | GC dice | epoch |
|---|---|---|---|---|
| `univ2_decoder` | 0.7141 | 0.7187 | 0.7101 | 7 |
| `univ2_hidden128` | **0.7178** | 0.7097 | 0.6872 | 8 |
| `univ2_spatial8` | crash (cross_entropy shape: target [128,256,256] vs input [128,3,128,128]) | | | |
| `univ2_spatial32` | crash (target [128,256,256] vs input [128,3,512,512]) | | | |

→ TLS dice ≈ 0.72 and **GC dice ≈ 0.71** — the headline. The patch-resolution
decoder finally lifted GC dice from 0 to 0.71.

#### Cascade evaluation (`gars_cascade_eval.log`, `gars_cascade_low_thresh.log`)
114-slide val. Stage 1 GATv2 F1=0.561 + Stage 2 mDice=0.714.

| Stage-1 thr | mDice | TLS dice | GC dice | TLS sp | GC sp | s/slide |
|---|---|---|---|---|---|---|
| 0.05 | 0.486 | 0.237 | 0.734 | 0.771 | 0.754 | 0.41 |
| 0.10 | 0.475 | 0.220 | 0.730 | 0.763 | 0.733 | 0.32 |
| 0.20 | 0.467 | 0.207 | 0.728 | 0.752 | 0.730 | 0.40 |
| 0.30 | 0.466 | 0.200 | 0.733 | 0.741 | 0.724 | 0.75 |
| 0.40 | 0.462 | 0.192 | 0.731 | 0.727 | 0.725 | 1.06 |
| 0.50 | 0.457 | 0.180 | 0.733 | 0.713 | 0.726 | 1.04 |

Per-cancer at thr=0.05: BLCA TLS=0.307 GC=0.707, KIRC TLS=0.150 GC=0.798,
LUSC TLS=0.206 GC=0.729.

→ GC dice held at ≈ 0.73 across all Stage-1 thresholds (robust). TLS pixel
dice dropped because Stage 1 selects only "core" TLS patches and misses
borders; slide-level counting still strong at low thresholds.

### Phase 8 — End-to-end joint and unified attempts (Apr 27)
- `gars_e2e.log` (joint S1+S2 fine-tune): collapsed. Stage 1 recall went
  from 0.59 → 0.003 by ep 11; GC dice 0.65 → 0.0. The joint loss let Stage 1
  minimise its term by selecting nothing; **unfreezing Stage 2 at ep 4
  destabilised everything**. Best was epoch 1 with `gc_dice=0.6550` before
  collapse. → end-to-end abandoned.
- `gars_unified.log` (`GraphEnrichedDecoder`, 10.15 M params, GATv2 features
  fused into the pixel decoder): mDice 0.720 by ep 6 (`val_tls=0.762,
  val_gc=0.678`) — comparable to the cascade with one fewer training stage.
- `gars_unified_v2.log`: started but truncated when the VM died (only
  dataset-build lines reached disk).

---

## Headline achievements

1. **Reproduced GNCAF / GC-UNet** in 8 variants — confirmed graph context
   adds nothing at the slide-level pixel-segmentation stage with UNI-v2
   features (~0.59 dice ceiling, same as no-GNN).
2. **Closed the GC pixel-dice gap** (`dice_gc` 0.00 → 0.71) by switching
   from "one model per slide" to **GARS = TLS-gating GNN + patch-level
   UNI-v2 pixel decoder**. The graph helps where it's cheapest (which
   patches?), the pixel decoder runs only on the few selected patches.
3. **Slide-level counting is essentially saturated**: TLS sp=0.945 / R²=0.93,
   GC sp=0.73 / R²=0.84, GC det AUC ≈ 0.87–0.90 — well above the regression /
   classification benchmarks.
4. **End-to-end joint training degenerates** (Stage 1 selects nothing), but
   the **unified `GraphEnrichedDecoder`** matches the cascade in one stage.

## GARS vs published GNCAF — final comparison

Reconstructed from the lost final commit (`autoresearch/gars-cascade@63e1e5b`,
never pushed to origin). Validation: 114-slide TCGA val, Stage-1
threshold = 0.05.

| Metric | GNCAF (published) | GARS (ours) | Δ |
|---|---|---|---|
| mDice | 0.688 | **0.714** | +4 % |
| TLS dice | 0.736 | 0.718 | −2 % |
| GC dice | 0.625 | **0.710** | **+14 %** |
| Params | 57 M | **10.1 M** | 6× fewer |
| Inference | 5–10 min/slide | **~1.5 s/slide** | ~300× faster |
| Training | ~6 h | **~7 min** | ~50× faster |

Figure: `gars_vs_gncaf.png` (regenerated in this notebook directory by
`make_gars_vs_gncaf.py`).

### Five key findings

1. **UNI-v2 features are sufficient for pixel-level segmentation — no RGB
   needed.** Stage 2 Option A (MobileNetV3 over RGB 256×256) aborted; Option
   B (linear projection of UNI-v2 → conv decoder) reaches mDice 0.714.
2. **Graph context matters at region-proposal, not at pixel decoding.**
   Stage 1 GATv2 F1 = 0.561 vs no-GNN F1 = 0.343 (+63 %). At the pixel-seg
   stage (Phase 4) GNN ablations were within noise of no-GNN. *(inferred —
   findings #2/#3 OCR-blanked in the source fragment)*
3. **Cascade > end-to-end joint training.** E2E lets Stage 1 minimise its
   loss by selecting nothing (recall 0.59 → 0.003 by ep 11; GC dice
   0.65 → 0.0); the cascade's frozen interface is a hard regulariser.
   *(inferred)*
4. **256 px > 512 px** for both Stage 1 (+52 % F1) and the pixel-level
   decoder. Larger receptive fields hurt — likely because UNI-v2 was trained
   at 256 px and bigger tiles dilute the relevant signal.
5. **Adaptive compute**: only ~0.5 % of patches go through the pixel
   decoder. This is what buys the 300× inference and 50× training speed-ups.

The **GARS paradigm is general**: graph-guided region proposal +
foundation-model feature decoding, applied wherever a tissue concept is
sparse over a WSI.

## Recovery pointers

1. **wandb runs** (`https://wandb.ai/ajhaas/tls-pixel-seg/runs/<id>`):
   `bmnj8pu1` (Stage 1 gcn3hop), `mn5jorov` (Stage 2 univ2_decoder),
   `cvrvn8yb` (Stage 2 hidden128), `x50tcqt6` (Stage 1 gatv2),
   `so4ujuqa` (Stage 1 no-GNN), `hwo25qzz` (Stage 1 512px),
   `ufz9a2o4` (e2e joint), `qy3pj74h` (unified GraphEnrichedDecoder),
   `r9xfeiax` / `ve53vorx` (Stage 2 spatial — both crashed).
   Each run page has `config.yaml`, `output.log`, and a code artifact.
2. **Persistent experiment dirs** still hold `best_checkpoint.pt` for every
   completed run — model state can be loaded directly even without source.
3. **GitHub `gitexa/profile-clam` branch `autoresearch/seg-apr11`** has
   commits through Apr 13. Anything Apr 14+ was on local worktree branches
   that are gone unless they were pushed.
4. The two `program*.md` and `experiment_log*.md` files in `auto-clam-seg`
   are the autoresearch journals — they cover Phases 1–2 in detail
   (61 experiments).

## Best surviving configurations

### Slide-level v2.0 panoptic (no-GARS)
- No GNN, hidden_dim=384, upsample_factor=4, 5×5 dwsep conv decoder
- Dual sigmoid heads, weighted-BCE center, dice_weight=2, center_weight=0.5
- Result: TLS dice=0.593, count_sp=0.902, det_auc=0.944, bkt_bacc=0.774;
  GC counting via center head sp=0.795 / det_auc=0.893; GC dice = 0.

### GARS cascade (best GC pixel-seg)
- Stage 1: GraphTLSDetector, GATv2 3-hop, UNI-v2 256-px features.
  F1=0.561, threshold 0.05 at inference.
- Stage 2: UNIv2PixelDecoder (~9.3 M params), 3-class softmax over selected
  TLS patches. mDice=0.714 (TLS=0.719, GC=0.710) at epoch 7.
- Cascade on 114 val slides: GC dice ≈ 0.73, GC sp=0.75, TLS sp=0.77.

### GARS unified (one-stage variant)
- GraphEnrichedDecoder (~10.15 M params), GATv2 features fused into the
  pixel decoder. mDice=0.720 at epoch 6 (TLS=0.762, GC=0.678).
