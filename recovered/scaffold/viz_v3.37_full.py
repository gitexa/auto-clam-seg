"""Per-slide validation visualisation using v3.37 RegionDecoderCascade.

For each cancer type (BLCA, KIRC, LUSC), pick 5 slides from the
intersection of (v6.0 fold-0 val) ∩ (cascade val) and produce
**three** figures per slide:

1. `<slide_id>_overview.png`  : thumbnail · seg mask · overlay (1×3 fig)
2. `<slide_id>_tls.png`        : 2 TLS crops, each as
                                 [centre patch RGB | seg | overlay]
                                 [1-hop neighbourhood RGB | seg | overlay]
3. `<slide_id>_gc.png`         : same as (2) but for GC

The 1-hop neighbourhood is the centre patch + its 8 surrounding cells
(3×3 = 768×768 px). Cells the cascade Stage 1 selected get a thick
green border; non-selected get a thin grey border. The numeric Stage 1
prob is shown on the centre cell. Cells decoded by Stage 2 show their
predictions; undecoded cells show bg.

Outputs to `viz_v3.37_validation/{CANCER}/`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile
import torch
import zarr
from PIL import Image
from scipy.ndimage import binary_closing as _bclose, label as _label

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_segmentation as ps  # noqa: E402
from gncaf_dataset import slide_wsi_path  # noqa: E402
from eval_gars_cascade import (  # noqa: E402
    load_stage1, load_stage2_region, cascade_one_slide_region, PATCH_SIZE,
)

V8_CKPT = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt"
V37_CKPT = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_region_v3.37_full_20260502_144124/best_checkpoint.pt"
V6_SPLITS = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/v6.0_tls_regression_best_5fold/v6.0_tls_regression_best_5fold_sweep_0aabc07fe386_/s_42/splits_0.csv"
OUT_ROOT = Path("/home/ubuntu/auto-clam-seg/recovered/scaffold/viz_v3.37_validation")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# RGBA palette: bg = transparent, TLS = blue, GC = red
PALETTE = np.array([
    [0, 0, 0, 0],
    [30, 100, 255, 200],
    [220, 30, 30, 220],
], dtype=np.uint8)
N_PER_CANCER = 5
N_TLS_CROPS = 10            # max TLS components per slide
N_GC_CROPS = 10             # max GC  components per slide
THUMB_LONG_MIN = 1500
STAGE1_THR = 0.5            # green-border threshold


# ─── Slide selection ──────────────────────────────────────────────────


def select_slides() -> pd.DataFrame:
    print("Loading v6.0 fold-0 val list...")
    v6 = pd.read_csv(V6_SPLITS)
    v6_val_short = {s.split(".")[0] for s in v6["val"].dropna().tolist()}
    print(f"  v6.0 fold-0 val: {len(v6_val_short)} unique slide ids")

    print("Loading cascade val (fold-0)...")
    ps.set_seed(42)
    entries = ps.build_slide_entries()
    folds_pair, _ = ps.create_splits(entries, k_folds=1, seed=42)
    cascade_val = [e for e in folds_pair[0] if e.get("mask_path") is not None]
    print(f"  cascade val: {len(cascade_val)} slides with masks")

    df = pd.read_csv(ps.META_CSV)
    counts = {str(r["slide_id"]).split(".")[0]: (int(r["tls_num"]), int(r["gc_num"]))
              for _, r in df.iterrows()}

    rows = []
    for e in cascade_val:
        short = e["slide_id"].split(".")[0]
        if short not in v6_val_short:
            continue
        wsi = slide_wsi_path(e)
        if not wsi or not Path(wsi).exists():
            continue
        tn, gn = counts.get(short, (0, 0))
        rows.append({
            "slide_id": short, "full_slide_id": e["slide_id"],
            "cancer_type": e["cancer_type"].upper(),
            "wsi_path": wsi, "zarr_path": e["zarr_path"],
            "mask_path": e.get("mask_path"),
            "tls_num": tn, "gc_num": gn,
            "rank_score": tn + 5 * gn,
        })
    df_pool = pd.DataFrame(rows)
    print(f"  intersection: {len(df_pool)} slides")
    out = []
    for ct in ("BLCA", "KIRC", "LUSC"):
        sub = df_pool[df_pool["cancer_type"] == ct].sort_values(
            "rank_score", ascending=False).head(N_PER_CANCER)
        out.append(sub)
        print(f"  {ct}: {len(sub)} picked")
    selected = pd.concat(out, ignore_index=True)
    selected.to_csv(OUT_ROOT / "selected_slides.csv", index=False)
    return selected


# ─── Thumbnail ────────────────────────────────────────────────────────


def read_thumbnail(wsi_path: str, target_long_min: int = THUMB_LONG_MIN
                    ) -> tuple[np.ndarray, int, int, int]:
    """Pick the smallest pyramid level whose long axis ≥ target_long_min,
    return (rgb, level, slide_H_pix, slide_W_pix). Slide_H/W are level-0 dims.
    """
    with tifffile.TiffFile(wsi_path) as t:
        slide_H, slide_W = t.pages[0].shape[:2]
        n_levels = len(t.pages)
        chosen = 0
        for lvl in range(n_levels - 1, -1, -1):
            shp = t.pages[lvl].shape
            if max(shp[0], shp[1]) >= target_long_min:
                chosen = lvl
                break
    z = zarr.open(tifffile.imread(wsi_path, aszarr=True, level=chosen), mode="r")
    rgb = np.asarray(z[:])
    return rgb, chosen, slide_H, slide_W


# ─── Stage 1 forward (per slide) ──────────────────────────────────────


@torch.no_grad()
def stage1_probs(stage1, features, edge_index, device):
    logits = stage1(features.to(device), edge_index.to(device))
    return torch.sigmoid(logits).cpu().numpy()


# ─── Slide-level mask aligned to thumbnail ───────────────────────────


def render_grid_rgba(grid_class: np.ndarray) -> np.ndarray:
    return PALETTE[grid_class]


def patch_grid_argmax(coords_np: np.ndarray, cell_probs: dict
                      ) -> tuple[np.ndarray, int, int, int, int, int, int]:
    """Build (H_grid, W_grid) argmax. Returns also the patch-pixel
    extent in level-0 coords (gy_min*256, gx_min*256, gy_max*256+256,
    gx_max*256+256).
    """
    grid_x = (coords_np[:, 0] // PATCH_SIZE).astype(np.int64)
    grid_y = (coords_np[:, 1] // PATCH_SIZE).astype(np.int64)
    gx_min = int(grid_x.min()); gy_min = int(grid_y.min())
    grid_x -= gx_min; grid_y -= gy_min
    H = int(grid_y.max()) + 1
    W = int(grid_x.max()) + 1
    grid = np.zeros((H, W), dtype=np.uint8)
    for (gy, gx), cp in cell_probs.items():
        if 0 <= gy < H and 0 <= gx < W:
            ta = cp.argmax(axis=0)
            n_gc = int((ta == 2).sum()); n_tls = int((ta == 1).sum())
            grid[gy, gx] = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)
    pix_y0 = gy_min * PATCH_SIZE
    pix_x0 = gx_min * PATCH_SIZE
    pix_y1 = (gy_min + H) * PATCH_SIZE
    pix_x1 = (gx_min + W) * PATCH_SIZE
    return grid, gy_min, gx_min, H, W, pix_y0, pix_x0


def thumb_seg_aligned(grid_argmax: np.ndarray,
                       gy_min: int, gx_min: int,
                       slide_H: int, slide_W: int,
                       thumb_h: int, thumb_w: int) -> np.ndarray:
    """Place the patch-grid mask in a thumbnail-resolution canvas at
    the correct offset (so it aligns with the WSI thumbnail spatially).

    The thumbnail is a downsampled view of the **whole** slide (level-0
    H_pix × W_pix). The patch grid covers only the patches' extent
    starting at (gy_min*256, gx_min*256). To overlay the mask onto the
    thumbnail correctly, we need to render the mask at thumbnail
    resolution AT the patch offset.
    """
    dy = slide_H / thumb_h
    dx = slide_W / thumb_w
    H_grid, W_grid = grid_argmax.shape
    # Patch-grid pixel extent in thumbnail coords:
    ty0 = int(round(gy_min * PATCH_SIZE / dy))
    tx0 = int(round(gx_min * PATCH_SIZE / dx))
    ty1 = int(round((gy_min + H_grid) * PATCH_SIZE / dy))
    tx1 = int(round((gx_min + W_grid) * PATCH_SIZE / dx))
    ty1 = min(ty1, thumb_h); tx1 = min(tx1, thumb_w)
    sub_h = max(0, ty1 - ty0); sub_w = max(0, tx1 - tx0)
    if sub_h == 0 or sub_w == 0:
        return np.zeros((thumb_h, thumb_w, 4), dtype=np.uint8)
    # Resize patch-grid mask (NEAREST) into the sub-region.
    rgba = render_grid_rgba(grid_argmax)
    pil = Image.fromarray(rgba, mode="RGBA").resize(
        (sub_w, sub_h), Image.Resampling.NEAREST)
    out = np.zeros((thumb_h, thumb_w, 4), dtype=np.uint8)
    out[ty0:ty0 + sub_h, tx0:tx0 + sub_w] = np.asarray(pil)
    return out


def overlay_alpha(base_rgb: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
    a = mask_rgba[:, :, 3:4].astype(np.float32) / 255.0
    o = base_rgb.astype(np.float32) * (1 - a) + \
        mask_rgba[:, :, :3].astype(np.float32) * a
    return o.astype(np.uint8)


# ─── Component selection ─────────────────────────────────────────────


def get_components_with_bbox(grid: np.ndarray, cls: int,
                              min_size: int = 2,
                              closing_iters: int = 1):
    """Plain connected components on the patch-grid mask. Used for GC
    and for the slide-level overview. TLS components in the per-slide
    figures use `get_components_graph` instead, which respects the
    actual graph topology + the GAT-derived score.
    """
    binary = (grid == cls)
    if closing_iters > 0:
        binary = _bclose(binary, iterations=closing_iters)
    lab, n = _label(binary)
    out = []
    for c in range(1, n + 1):
        ys, xs = np.where(lab == c)
        if ys.size < min_size:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        out.append({
            "size": int(ys.size),
            "bbox": (y0, x0, y1, x1),
            "centroid": ((y0 + y1) // 2, (x0 + x1) // 2),
            "n_subcomps": 1,
        })
    out.sort(key=lambda d: -d["size"])
    return out


def get_components_graph(s1_probs: np.ndarray, edge_index_np: np.ndarray,
                         patch_to_grid: dict[int, tuple[int, int]],
                         s1_thr_high: float = STAGE1_THR,
                         s1_thr_low: float = 0.3,
                         min_size: int = 2):
    """Connected components on the **graph topology** of `edge_index_np`
    (2, E), with two-threshold gating.

    Two cells are unioned iff:
      - There is a graph edge between them, AND
      - Both have s1 ≥ `s1_thr_low`  (the LOW threshold is the
        "candidate" gate; LOW should be lower than HIGH).

    A returned component must contain at least one cell with
    s1 ≥ `s1_thr_high` (the "core" gate). The centroid is set to the
    cell with the highest s1 score, so the 1-hop view is centred on
    the component's most confident point.

    Why this matches the user's "feature- and spatial-relationship-
    driven" framing: the graph edges already encode spatial adjacency,
    and the GAT-trained Stage 1 score already encodes feature context
    (it was *produced by* a 5-hop GATv2 over UNI features). So the
    score IS the model's neighbourhood-aware judgement; we just use
    the graph topology to grow components rather than imposing a
    purely-spatial 4-conn grid label step.
    """
    n_nodes = int(s1_probs.shape[0])
    is_core = s1_probs >= s1_thr_high
    is_cand = s1_probs >= s1_thr_low

    parent = list(range(n_nodes))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    src = edge_index_np[0]
    dst = edge_index_np[1]
    for s_, d_ in zip(src, dst):
        s_, d_ = int(s_), int(d_)
        if is_cand[s_] and is_cand[d_]:
            union(s_, d_)

    groups: dict[int, list[int]] = {}
    for i in range(n_nodes):
        if is_cand[i]:
            groups.setdefault(find(i), []).append(i)

    out = []
    for members in groups.values():
        cores_in = [m for m in members if is_core[m]]
        if not cores_in:
            continue
        members_in_grid = [m for m in members if m in patch_to_grid]
        if len(members_in_grid) < min_size:
            continue
        gys = [patch_to_grid[m][0] for m in members_in_grid]
        gxs = [patch_to_grid[m][1] for m in members_in_grid]
        y0, y1 = min(gys), max(gys) + 1
        x0, x1 = min(gxs), max(gxs) + 1
        # Centroid = highest-s1 core cell.
        best = max(cores_in, key=lambda m: float(s1_probs[m]))
        cy, cx = patch_to_grid[best] if best in patch_to_grid else \
                  ((y0 + y1) // 2, (x0 + x1) // 2)
        n_core = len(cores_in)
        n_cand = len(members)
        out.append({
            "size": n_cand,
            "n_core": n_core,
            "bbox": (y0, x0, y1, x1),
            "centroid": (cy, cx),
        })
    # Sort by core-cell count, then by total size — most "confident" components first.
    out.sort(key=lambda d: (-d["n_core"], -d["size"]))
    return out


# ─── Crop renderers ──────────────────────────────────────────────────


def read_centre_patch(wsi_z, gy_in_grid: int, gx_in_grid: int,
                       gy_min: int, gx_min: int,
                       slide_H: int, slide_W: int):
    """Read the (256, 256, 3) RGB tile of the centre patch + return
    its (256, 256) seg argmax (or zeros if not decoded) at the cell
    indexed by gy_in_grid, gx_in_grid (within the patch grid)."""
    pix_y = (gy_min + gy_in_grid) * PATCH_SIZE
    pix_x = (gx_min + gx_in_grid) * PATCH_SIZE
    py1 = min(slide_H, pix_y + PATCH_SIZE)
    px1 = min(slide_W, pix_x + PATCH_SIZE)
    rgb = np.asarray(wsi_z[pix_y:py1, pix_x:px1])
    if rgb.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        pad_y = PATCH_SIZE - rgb.shape[0]; pad_x = PATCH_SIZE - rgb.shape[1]
        rgb = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), constant_values=255)
    return rgb


def read_1hop_window(wsi_z, gy_in_grid: int, gx_in_grid: int,
                     gy_min: int, gx_min: int,
                     H_grid: int, W_grid: int,
                     slide_H: int, slide_W: int,
                     cell_probs: dict, s1_probs: np.ndarray,
                     coord_to_idx: dict):
    """Read a 3×3 RGB tile (768×768 px) centred on (gy_in_grid, gx_in_grid),
    plus per-cell info: which cells are valid (have a patch), s1 probs,
    s2 argmax (from cell_probs).

    Returns:
        rgb (768, 768, 3) uint8
        seg (768, 768) uint8 — class argmax per pixel; bg where cell
                                wasn't decoded by Stage 2
        overlay (768, 768, 3) uint8
        valid (3, 3) bool — does this cell have a patch on the slide?
        s1_per_cell (3, 3) float — Stage 1 prob (-1 if no patch)
        s2_decoded (3, 3) bool — was Stage 2 run on this cell?
    """
    P = PATCH_SIZE
    rgb = np.full((3 * P, 3 * P, 3), 255, dtype=np.uint8)
    seg = np.zeros((3 * P, 3 * P), dtype=np.uint8)
    valid = np.zeros((3, 3), dtype=bool)
    s1_pc = -np.ones((3, 3), dtype=np.float32)
    s2_pc = np.zeros((3, 3), dtype=bool)
    for di in range(3):
        for dj in range(3):
            gy = gy_in_grid + (di - 1)
            gx = gx_in_grid + (dj - 1)
            if not (0 <= gy < H_grid and 0 <= gx < W_grid):
                continue
            patch_idx = coord_to_idx.get((gy, gx))
            if patch_idx is None:
                continue
            valid[di, dj] = True
            s1_pc[di, dj] = float(s1_probs[patch_idx])
            pix_y = (gy_min + gy) * P
            pix_x = (gx_min + gx) * P
            py1 = min(slide_H, pix_y + P)
            px1 = min(slide_W, pix_x + P)
            tile = np.asarray(wsi_z[pix_y:py1, pix_x:px1])
            h = tile.shape[0]; w = tile.shape[1]
            rgb[di * P:di * P + h, dj * P:dj * P + w] = tile
            cp = cell_probs.get((gy, gx))
            if cp is not None:
                s2_pc[di, dj] = True
                tile_argmax = cp.argmax(axis=0).astype(np.uint8)
                seg[di * P:(di + 1) * P, dj * P:(dj + 1) * P] = tile_argmax
    overlay = overlay_alpha(rgb, render_grid_rgba(seg))
    return rgb, seg, overlay, valid, s1_pc, s2_pc


# ─── Figure builders ─────────────────────────────────────────────────


def fig_overview(thumb: np.ndarray, thumb_seg_rgba: np.ndarray,
                  thumb_overlay: np.ndarray, title: str, out_path: Path):
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.05)
    for col, (img, t) in enumerate([
        (thumb, "thumbnail"),
        (thumb_seg_rgba, "segmentation mask"),
        (thumb_overlay, "overlay"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(t, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [
        mpatches.Patch(color="#1e64ff", label="TLS"),
        mpatches.Patch(color="#dc1e1e", label="GC"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.97), fontsize=10, ncol=2)
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _draw_cell_borders(ax, valid: np.ndarray, s1_pc: np.ndarray,
                        s2_pc: np.ndarray, cell_px: int,
                        annotate_scores: bool = True,
                        s1_smoothed_pc: np.ndarray | None = None):
    """Draw the per-cell border on a 3×3 view.

    If `s1_smoothed_pc` is provided, the cell's selection state is
    judged from the SMOOTHED score (≥STAGE1_THR → green) and the label
    shows both `raw → smoothed` so the spatial reasoning is visible.
    """
    for di in range(3):
        for dj in range(3):
            x0 = dj * cell_px; y0 = di * cell_px
            if not valid[di, dj]:
                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), cell_px, cell_px,
                    linewidth=0, facecolor="lightgrey", alpha=0.35,
                ))
                continue
            score_raw = float(s1_pc[di, dj])
            score_smooth = (float(s1_smoothed_pc[di, dj])
                             if s1_smoothed_pc is not None else score_raw)
            is_centre = (di, dj) == (1, 1)
            sel_score = score_smooth                  # judge from smoothed
            if sel_score >= STAGE1_THR:
                color = "#22aa22"
                lw = 5.0 if is_centre else 3.0
            else:
                color = "#888888"
                lw = 3.0 if is_centre else 1.0
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), cell_px, cell_px,
                linewidth=lw, edgecolor=color, facecolor="none",
            ))
            if annotate_scores:
                weight = "bold" if is_centre else "normal"
                fontsize = 9 if is_centre else 7
                if s1_smoothed_pc is not None:
                    label = f"raw {score_raw:.2f}\nsmt {score_smooth:.2f}"
                else:
                    label = f"s1={score_raw:.2f}"
                ax.text(x0 + 6, y0 + 22,
                        label,
                        color="black", fontsize=fontsize, weight=weight,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="none", alpha=0.85))


def fig_crops(class_label: str, comps: list, wsi_z, slide_H: int,
               slide_W: int, gy_min: int, gx_min: int, H_grid: int, W_grid: int,
               cell_probs: dict, s1_probs: np.ndarray, coord_to_idx: dict,
               title: str, out_path: Path):
    """One row per component (up to 10), 3 columns [RGB | seg | overlay]
    of the 1-hop 3×3 neighbourhood. All cells annotated with their
    Stage-1 prob; centre cell rendered with a thicker border + bold
    label.
    """
    n = len(comps)
    if n == 0:
        # Single placeholder figure.
        fig = plt.figure(figsize=(15, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f"No {class_label} components found",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title, fontsize=12)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    height_per_row = 4.5
    fig = plt.figure(figsize=(15, height_per_row * n + 0.5))
    gs = fig.add_gridspec(n, 3, hspace=0.20, wspace=0.04)

    for r, comp in enumerate(comps):
        cy, cx = comp["centroid"]
        rgb_n, seg_n, ovr_n, valid, s1_pc, s2_pc = read_1hop_window(
            wsi_z, cy, cx, gy_min, gx_min, H_grid, W_grid,
            slide_H, slide_W, cell_probs, s1_probs, coord_to_idx,
        )
        centre_score = float(s1_probs[coord_to_idx[(cy, cx)]]) \
            if (cy, cx) in coord_to_idx else -1.0
        for col, (img, t, annotate) in enumerate([
            (rgb_n, f"{class_label} #{r + 1} — 1-hop RGB (768×768, "
                    f"size={comp['size']} cells, centre s1={centre_score:.2f})", True),
            (render_grid_rgba(seg_n), "1-hop seg", False),
            (ovr_n, "1-hop overlay", False),
        ]):
            ax = fig.add_subplot(gs[r, col])
            ax.imshow(img)
            ax.set_title(t, fontsize=9)
            _draw_cell_borders(ax, valid, s1_pc, s2_pc,
                               cell_px=PATCH_SIZE, annotate_scores=annotate)
            ax.set_xticks([]); ax.set_yticks([])

    handles = [
        mpatches.Patch(color="#1e64ff", label="TLS pixels"),
        mpatches.Patch(color="#dc1e1e", label="GC pixels"),
        mpatches.Patch(facecolor="none", edgecolor="#22aa22", linewidth=3,
                       label=f"Stage-1-selected (s1≥{STAGE1_THR})"),
        mpatches.Patch(facecolor="none", edgecolor="#888888", linewidth=1,
                       label="Stage-1 not selected"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.998), fontsize=9, ncol=4)
    fig.suptitle(title + f" (showing {n} component{'s' if n != 1 else ''})",
                 fontsize=12, y=0.999)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Per-slide pipeline ──────────────────────────────────────────────


def predict_slide(stage1, stage2_region, entry: dict, device,
                   threshold: float = 0.5, s2_batch: int = 2):
    grp = zarr.open(entry["zarr_path"], mode="r")
    features = torch.from_numpy(np.asarray(grp["features"][:])).float()
    coords = torch.from_numpy(np.asarray(grp["coords"][:])).float()
    if "graph_edges_1hop" in grp:
        edge_index = torch.from_numpy(np.asarray(grp["graph_edges_1hop"][:])).long()
    elif "edge_index" in grp:
        edge_index = torch.from_numpy(np.asarray(grp["edge_index"][:])).long()
    else:
        raise ValueError("no edge_index in zarr")
    s1_probs_arr = stage1_probs(stage1, features, edge_index, device)
    grid_class, n_sel, pred_tiles, cell_probs = cascade_one_slide_region(
        stage1, stage2_region, features, coords, edge_index,
        threshold, device, wsi_path=entry["wsi_path"],
        s2_batch=s2_batch, min_component_size=2, closing_iters=1,
        return_cell_probs=True,
    )
    return {
        "grid_class": grid_class, "pred_tiles": pred_tiles,
        "cell_probs": cell_probs,
        "coords": coords.numpy(),
        "edge_index": edge_index.numpy(),
        "n_sel": n_sel,
        "s1_probs": s1_probs_arr,
    }


def viz_one_slide(entry: dict, pred: dict, out_dir: Path,
                   s1_thr_high: float = STAGE1_THR,
                   s1_thr_low: float = 0.3) -> None:
    short = entry["slide_id"]
    p_overview = out_dir / f"{short}_overview.png"
    p_tls = out_dir / f"{short}_tls.png"
    p_gc = out_dir / f"{short}_gc.png"
    if all(p.exists() for p in (p_overview, p_tls, p_gc)):
        print(f"  skip — all 3 PNGs exist")
        return

    coords_np = pred["coords"]
    grid_argmax, gy_min, gx_min, H_grid, W_grid, _, _ = patch_grid_argmax(
        coords_np, pred["cell_probs"])
    coord_to_idx = {(int(coords_np[i, 1] // PATCH_SIZE) - gy_min,
                     int(coords_np[i, 0] // PATCH_SIZE) - gx_min): i
                    for i in range(coords_np.shape[0])}
    # Inverse map for graph-based component finder.
    patch_to_grid = {i: (int(coords_np[i, 1] // PATCH_SIZE) - gy_min,
                          int(coords_np[i, 0] // PATCH_SIZE) - gx_min)
                     for i in range(coords_np.shape[0])}

    # Thumbnail.
    thumb, level, slide_H, slide_W = read_thumbnail(entry["wsi_path"])
    thumb_h, thumb_w = thumb.shape[:2]
    thumb_seg_rgba = thumb_seg_aligned(grid_argmax, gy_min, gx_min,
                                        slide_H, slide_W, thumb_h, thumb_w)
    thumb_overlay = overlay_alpha(thumb, thumb_seg_rgba)

    title = (f"{entry['cancer_type']} — {short}  "
             f"(GT: TLS={entry['tls_num']}, GC={entry['gc_num']}; "
             f"Stage-1 selected={pred['n_sel']}, level={level})")
    fig_overview(thumb, thumb_seg_rgba, thumb_overlay, title, p_overview)
    print(f"  wrote {p_overview.name}")

    # Open WSI handle for crop reads.
    wsi_z = zarr.open(tifffile.imread(entry["wsi_path"], aszarr=True, level=0),
                      mode="r")
    # TLS: graph-topology component finder. The Stage 1 GAT score is
    # already feature-and-spatial-aware; we just gate at two
    # thresholds and grow components along the actual graph edges.
    tls_comps = get_components_graph(
        s1_probs=pred["s1_probs"], edge_index_np=pred["edge_index"],
        patch_to_grid=patch_to_grid,
        s1_thr_high=s1_thr_high, s1_thr_low=s1_thr_low, min_size=2,
    )[:N_TLS_CROPS]
    # GC: keep the per-pixel-mask connected components; smoothing
    # doesn't apply directly since GC is a multi-class output of
    # Stage 2, not a Stage-1 probability score.
    gc_comps = get_components_with_bbox(
        grid_argmax, cls=2, min_size=2,
    )[:N_GC_CROPS]

    fig_crops("TLS", tls_comps, wsi_z, slide_H, slide_W, gy_min, gx_min,
              H_grid, W_grid, pred["cell_probs"], pred["s1_probs"],
              coord_to_idx,
              title=f"{entry['cancer_type']} — {short} — TLS crops "
                    f"(GT TLS={entry['tls_num']})",
              out_path=p_tls)
    print(f"  wrote {p_tls.name}")
    fig_crops("GC", gc_comps, wsi_z, slide_H, slide_W, gy_min, gx_min,
              H_grid, W_grid, pred["cell_probs"], pred["s1_probs"],
              coord_to_idx,
              title=f"{entry['cancer_type']} — {short} — GC crops "
                    f"(GT GC={entry['gc_num']})",
              out_path=p_gc)
    print(f"  wrote {p_gc.name}")


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    selected = select_slides()
    print(f"\nLoading models...")
    stage1 = load_stage1(V8_CKPT, device)
    stage2_region = load_stage2_region(V37_CKPT, device)
    print(f"\nGenerating viz for {len(selected)} slides (3 PNGs each = "
          f"{len(selected) * 3} files)...\n")
    for i, row in selected.iterrows():
        out_dir = OUT_ROOT / row["cancer_type"]
        out_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "slide_id": row["slide_id"],
            "full_slide_id": row["full_slide_id"],
            "cancer_type": row["cancer_type"],
            "wsi_path": row["wsi_path"],
            "zarr_path": row["zarr_path"],
            "mask_path": row.get("mask_path"),
            "tls_num": row["tls_num"], "gc_num": row["gc_num"],
        }
        print(f"[{i + 1}/{len(selected)}] {row['cancer_type']} {row['slide_id']}")
        t0 = time.time()
        try:
            pred = predict_slide(stage1, stage2_region, entry, device)
            viz_one_slide(entry, pred, out_dir)
            print(f"  done in {time.time() - t0:.1f}s "
                  f"(selected {pred['n_sel']} patches)")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
    print(f"\nAll done — {OUT_ROOT}")


if __name__ == "__main__":
    main()
