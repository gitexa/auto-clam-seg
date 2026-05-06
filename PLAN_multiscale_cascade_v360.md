# Plan: Multi-scale bipartite graph for cascade (Part B priority) +
# GCUNet-faithful GNCAF rigour fix (Part A deferred)

## Context

The user has set two priorities. The HIGH-priority change is a
**multi-scale bipartite graph cascade (Part B)** that exploits the
already-extracted 512×512@20× UNI v2 features to add coarse-scale
context to the cascade. The LOWER-priority fix is the **GCUNet-faithful
GCN aggregation (Part A)** that corrects an architectural deviation
in our existing GNCAF reproduction. Per user direction:

- **Cascade improvement comes first** (Part B), GNCAF rigour fix later.
- **Bipartite graph** (B2), not feature concatenation.
- 512-px features already exist; **no extraction needed**.

User rationale (caught from feedback): structurally separating the
multi-scale information into the graph topology is preferred over
collapsing it into a per-node feature vector.

---

## Part B (PRIORITY) — Multi-scale bipartite graph cascade

### Idea

Today's cascade Stage 1 (GraphTLSDetector, 5-hop GATv2) operates on a
single-scale graph: nodes are 256×256@20× patches, edges are
4-connectivity, features are 1536-d UNI v2. Stage 2 (RegionDecoder)
decodes only Stage-1-positive 3×3 regions of those patches.

**Proposal**: build a bipartite multi-scale graph with two node
classes — fine (256-px@20×) and coarse (512-px@20× ≈ 256-px@10×) —
connected by containment edges. Stage 1 GAT learns over the union
graph; coarse-scale messages flow up-down through containment edges,
giving each fine node access to a wider effective field of view that
roughly matches the GCUNet paper's 1 µm/pixel resolution.

### Data sources (verified)

- **Fine (256-px @ 20×)**: `/home/ubuntu/local_data/zarr/{blca,kirc,lusc}/<slide>_complete.zarr`
  - `coords` (N₂₅₆, 2), `features` (N₂₅₆, 1536), `graph_edges_1hop`, `graph_edges_2hop`
- **Coarse (512-px @ 20×, trident)**: `/home/ubuntu/ahaas-persistent-std-tcga/data/tcga-{blca,kirc,lusc}/representations_tif_trident/20x_512px_0px_overlap/features_uni_v2_grandqc_zarr/<slide>_complete.zarr`
  - `coords` (N₅₁₂, 2), `features` (N₅₁₂, 1536), `graph_edges_1hop`, `graph_edges_2hop`
  - Slide coverage: 335 BLCA + 298 KIRC + 374 LUSC = 1007 slides (full cohort)
  - N₅₁₂ ≈ N₂₅₆ / 4 (e.g. 5,569 vs 21,950 for the sample slide we inspected)

### Bipartite graph construction

For a slide with N₂₅₆ fine nodes and N₅₁₂ coarse nodes:

1. **Fine sub-graph** (existing): 4-conn between 256-nodes, edges from
   `local_data/zarr/<...>/graph_edges_1hop`.
2. **Coarse sub-graph** (new): 4-conn between 512-nodes, edges from
   `trident/.../graph_edges_1hop`.
3. **Containment edges** (new): each coarse node c at (x_c, y_c)
   connects bidirectionally to its (up to 4) fine children — fine
   nodes at (x, y) where `x_c ≤ x < x_c + 512` AND `y_c ≤ y < y_c + 512`.
   Computed once per slide from coords.

Total nodes per slide: N₂₅₆ + N₅₁₂.
Edge types: `fine-fine` (within-scale), `coarse-coarse`
(within-scale), `coarse-fine` (containment, undirected → 2 directed
edges per pair).

### Stage 1 model: heterogeneous GATv2

Two options:

**Option H1 — single edge_index, one node feature space**:
- Project both fine (1536) and coarse (1536) features into the same
  hidden dim h via separate Linear layers (so the GAT sees
  type-aware initial features).
- Concat all edges into one `edge_index` tensor; ignore edge type
  in the GAT (treat all edges uniformly).
- Trainable scale embedding (1 vec per scale) added to projected
  features so the GAT can distinguish.
- 5-hop GAT propagates over the union graph; output dim same as
  before. Stage 1 prediction head only attends to fine-node outputs
  (since the per-patch task is at the 256-px granularity).

**Option H2 — typed-edge GAT (HeteroGAT)**:
- `torch_geometric.nn.HeteroConv` with separate `GATv2Conv` per edge
  type (fine-fine, coarse-coarse, coarse-fine).
- Cleaner type semantics, more parameters.
- Slower per-step due to 3 GATv2 instances per layer.

Recommendation: **start with H1** as a simpler first cut; it's
strictly more powerful than the current single-scale Stage 1 and
catches the multi-scale signal without the complexity of HeteroGAT.
If H1 shows promise, **upgrade to H2** for the publication run.

### Stage 1 prediction head

Output binary "is this 256-px patch positive?" only for fine nodes.
The coarse-node outputs are discarded after the GAT layers
(training-time supervision: only fine-node BCE against the existing
patch-level labels).

### Stage 2 (RegionDecoder)

Two paths:

1. **Reuse v3.37 RegionDecoder unchanged** — it operates on per-patch
   3×3 windows of 256-px features + RGB tiles + Stage-1 graph context.
   Just feed it Stage 1's improved fine-node graph_ctx; the rest is
   unchanged.

2. **Add coarse-scale context to RegionDecoder** — for each selected
   region, also fetch the parent 512-node's UNI features and inject
   into the fusion path. More change, possibly more gain.

Recommendation: **start with (1)** (reuse v3.37 RegionDecoder as-is).
This isolates the Stage 1 multi-scale upgrade as the only variable.
If multi-scale Stage 1 helps, we'll know it's the graph-topology
change, not Stage 2 changes. Add (2) only if (1) shows promise.

### Implementation files

| File | Change |
|---|---|
| `recovered/scaffold/multiscale_dataset.py` | **NEW** — loader that reads BOTH zarrs (fine + coarse), builds containment edges, returns combined `(features, edge_index, scale_mask, fine_mask)` per slide. |
| `recovered/scaffold/train_gars_stage1.py` | thread a `multi_scale: bool` knob; when true, use the multi-scale dataset and add a scale-embedding to the GAT. |
| `recovered/scaffold/configs/stage1/model/gatv2_5hop_multiscale.yaml` | **NEW** — same hyperparams as `gatv2_5hop` plus a `scale_embed_dim` knob. |
| `recovered/scaffold/configs/stage1/train/stage1_multiscale.yaml` | **NEW** — `multi_scale: true` |
| `recovered/scaffold/configs/cascade/v3_60_multiscale.yaml` | **NEW** — eval config pointing at multi-scale Stage 1 + v3.37 Stage 2. |
| `recovered/scaffold/run_chain_5fold_cascade_v360.sh` | **NEW** — 4-fold Stage 1 multi-scale + cascade eval (Stage 2 is the existing v3.37 ckpts so no Stage 2 retrain). |

Reuse:
- `chain_lib.sh` marker-based step orchestration
- Existing `eval_gars_cascade.py` (with the `region_mode` flag) — feed it the new Stage 1 ckpt + the existing v3.37 Stage 2 ckpt, evaluate per fold.
- Existing fold-aligned splits via `build_gncaf_split` (already exposes `seed`, `k_folds`, `fold_idx`).

### Validation (Part B)

| Metric | v3.37 cascade (1-scale) | v3.60 cascade (multi-scale) |
|---|---|---|
| mDice_pix fold 0 | 0.845 | TBD |
| 5-fold mDice_pix | 0.649 ± 0.110 | TBD |
| Stage 1 F1 | TBD | TBD |
| s/slide | 2.65 | likely +20–50% (extra zarr read + larger graph) |

**Decision rule**: if v3.60 fold-0 mDice_pix ≥ 0.86 (≥ +1.5pt over
v3.37), commit 5-fold; otherwise document as no-op and stop. The fair
comparison is fold 0 with the existing v3.37 ckpts vs fold 0 with the
multi-scale Stage 1 + the same v3.37 Stage 2.

### Compute budget (Part B)

| Phase | Hours |
|---|---|
| Multi-scale dataset + Stage 1 model dev | ~3h |
| Smoke test (CPU forward + 1 epoch) | ~1h |
| Stage 1 fold-0 multi-scale training | ~1.5h |
| Cascade fold-0 eval (4 shards) | ~30 min |
| **Decision gate** | — |
| If pass: folds 1-4 Stage 1 + 4 cascade evals | ~10h |
| **Total to 5-fold (if decision-gate passes)** | **~15h** |

Stage 2 retrain is **not** part of this — we reuse the existing v3.37
fold-1..4 Stage 2 ckpts.

---

## Part A (DEFERRED) — GCUNet-faithful GCN multi-hop aggregation

After Part B lands. Already documented in detail; key points:

### Mismatch
- **Paper (GCUNet Eq. 3)**: `MLP(CAT([x⁽⁰⁾; x⁽¹⁾; …; x⁽ᴷ⁾]))`
- **Our `GCNContext`** (`gncaf_transunet_model.py:297-302`): iterative
  residual updates, returns only x⁽ᴷ⁾.

### Fix
New file `recovered/scaffold/gcunet_model.py` with `GCNContextPaper`
(concat-all-hops + MLP) and `GCUNetPixelDecoder` wiring. Train v3.59
with same hyperparams as v3.58 plus `model_class: gcunet`. Fold-0
smoke first (~3.5h), then 5-fold (~15h) if results are sensible.

This is a rigour fix for the **GNCAF** column of our 5-fold benchmark,
not a cascade improvement. It probably doesn't change conclusions
(structural cascade-vs-GNCAF gap is at the decoder/selection level,
not the GCN), but is the right thing to do for publication-quality
disclosure.

---

## Critical files

### Part B (priority)
- `recovered/scaffold/multiscale_dataset.py` — NEW (multi-scale data loader)
- `recovered/scaffold/train_gars_stage1.py` — multi-scale flag
- `recovered/scaffold/configs/stage1/model/gatv2_5hop_multiscale.yaml` — NEW
- `recovered/scaffold/configs/stage1/train/stage1_multiscale.yaml` — NEW
- `recovered/scaffold/configs/cascade/v3_60_multiscale.yaml` — NEW
- `recovered/scaffold/run_chain_5fold_cascade_v360.sh` — NEW
- `recovered/scaffold/build_benchmark_table.py` — add v3.60 row
- `summary.md` — append v3.60 result + bipartite graph description

### Part A (deferred)
- `recovered/scaffold/gcunet_model.py` — NEW (`GCNContextPaper`)
- `recovered/scaffold/configs/gncaf/config_transunet_v3_59.yaml` — NEW
- `recovered/scaffold/run_chain_5fold_gcunet_v359.sh` — NEW

### Reuse for both
- `chain_lib.sh` (marker-based, 350+ selftest passes in this session)
- `eval_gars_cascade.py` / `eval_gars_gncaf_transunet.py` (model-agnostic)
- `combine_gncaf_shards.py`
- `build_gncaf_split` (fold splits already plumbed)

## Verification

### Part B
1. **Bipartite graph correctness**: for one sample slide, manually
   verify each 512-px coord (x, y) has its 4 expected fine children
   among the 256-px coords. Print containment edge counts; should
   roughly equal 4 × N₅₁₂ minus boundary cases.
2. **Stage 1 fold-0 forward**: random forward pass through the
   multi-scale GAT; output shape (N₂₅₆,) for fine-node logits.
3. **Reproducibility**: train fold 0 with seed=42; val slide-IDs match
   existing fold-0 v3.8 cascade (since `build_gncaf_split` is unchanged).
4. **Fold-0 cascade head-to-head**: v3.60 fold-0 (multi-scale Stage 1 +
   existing v3.37 Stage 2) vs v3.37 fold-0 baseline (0.845 mDice_pix).
5. **5-fold paired t-test**: if 5-fold completes, compare per-fold
   mDice_pix against the existing v3.37/cascade 5-fold (already in
   `benchmark_5fold.csv`). Paired t with n=5.

### Part A
1. **Strict shape check**: `GCUNetPixelDecoder` output shape on
   random tensors.
2. **Fold-0 head-to-head**: v3.59 vs v3.58 on same val.
3. **5-fold paired t-test**: cascade vs v3.59 — does the
   architecture fix change the publication conclusion?

## Risk / fallback

### Part B
- **Multi-scale doesn't help cascade**: 5-hop GATv2 may already
  capture broad context implicitly via the 1-hop graph. Possible the
  explicit coarse-scale doesn't add new information. If v3.60 fold-0
  ≤ v3.37 (0.845), document as no-op and stop before 5-fold.
- **Bipartite graph causes OOM**: N₂₅₆ + N₅₁₂ ≈ 27K nodes per slide
  (vs 22K before) plus extra edges. Should still fit on 40GB but
  worth checking. If OOM, reduce `max_pos_per_slide` or drop edges.
- **Stage 2 v3.37 isn't compatible** with new Stage 1 graph_ctx
  shape: v3.37's `extract_stage1_context` may need a tweak for the
  new GAT output. Verify in smoke test.

### Part A
- Same risks documented earlier (most likely no-op).

## Out of scope

- Re-extracting features at UNI v1 (1024-d) — not aligned with
  current direction.
- Implementing GCUNet's Seg4/Seg3 class taxonomy — needs different
  GT annotations than HookNet provides.
- Test-set evaluation — still blocked on missing local TIFs/SVS.
- v4 e2e cascade variants — already shown negative.
- Stage 2 retraining for multi-scale (Part B path 2) — defer to a
  follow-up if Stage 1 multi-scale shows promise.

## Open questions for the user (none blocking)

User has answered the major design questions:
- Target paper: GCUNet ✓
- Scope: cascade first (Part B), GNCAF fix later (Part A) ✓
- Multi-scale design: bipartite graph, not feature concat ✓
- 512-px features: trident path, no extraction ✓

Plan is ready to execute. User stated "I want to do something else
first" — implementation will start when they direct.
