"""Validation visualisation: compare v3.13 (champion) vs v3.7a (baseline).

For each picked val slide, render a panel:
  - GT mask (HookNet) downsampled
  - v3.13 prediction (slide-level patch grid) with bboxes around CC instances
  - v3.7a prediction (slide-level patch grid) with bboxes
  - 4 zoomed crops showing per-patch native-res predictions

Saves one PNG per slide to <out_dir>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import zarr
from scipy.ndimage import binary_closing, label

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_gars_cascade import load_stage1, load_stage2, cascade_one_slide
import prepare_segmentation as ps
from tls_patch_dataset import build_tls_patch_dataset

PATCH_SIZE = 256
EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")
V0_CKPT = EXP / "gars_stage1_v3.0_gatv2_3hop_20260430_002624/best_checkpoint.pt"
V8_CKPT = EXP / "gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt"
V4B_CKPT = EXP / "gars_stage2_v3.4b_h128_min4096_20260430_090510/best_checkpoint.pt"
OUT_DIR = Path("/home/ubuntu/auto-clam-seg/recovered/scaffold/viz_v3.13_vs_v3.7a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Class palette (RGBA): bg = transparent, TLS = blue, GC = red.
PALETTE = np.array([
    [0, 0, 0, 0],          # bg
    [30, 100, 255, 200],   # TLS
    [220, 30, 30, 220],    # GC
], dtype=np.uint8)


def build_grid(coords):
    grid_x = (coords[:, 0] / PATCH_SIZE).astype(np.int64)
    grid_y = (coords[:, 1] / PATCH_SIZE).astype(np.int64)
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    H, W = int(grid_y.max()) + 1, int(grid_x.max()) + 1
    return grid_x, grid_y, H, W


def render_grid(grid_class):
    """Render a (H, W) class grid as RGBA."""
    out = PALETTE[grid_class]
    return out


def native_pred_image(coords, pred_tiles, downsample=4):
    """Stitch per-patch native 256×256 argmax tiles into a slide-scale image,
    downsampled by the given factor to keep it manageable. Patches not in
    pred_tiles are background (class 0)."""
    grid_x, grid_y, H_g, W_g = build_grid(coords)
    cell = PATCH_SIZE // downsample
    H, W = H_g * cell, W_g * cell
    img = np.zeros((H, W), dtype=np.uint8)
    for k, (gx, gy) in enumerate(zip(grid_x, grid_y)):
        tile = pred_tiles.get(int(k))
        if tile is None:
            continue
        sub = tile[::downsample, ::downsample].astype(np.uint8)
        img[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = sub
    return img, H_g, W_g, cell


def gt_native_image(coords, slide_lookup, downsample=4):
    grid_x, grid_y, H_g, W_g = build_grid(coords)
    cell = PATCH_SIZE // downsample
    H, W = H_g * cell, W_g * cell
    img = np.zeros((H, W), dtype=np.uint8)
    for k, (gx, gy) in enumerate(zip(grid_x, grid_y)):
        tile = slide_lookup.get(int(k))
        if tile is None:
            continue
        sub = np.asarray(tile)[::downsample, ::downsample].astype(np.uint8)
        img[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = sub
    return img, H_g, W_g, cell


def get_instance_bboxes(grid_class, cls, min_size=2, closing_iters=1, cell=64):
    """Return bboxes (x, y, w, h) in *image pixels* (post-downsample) for
    each connected component of `cls` in the grid."""
    binary = (grid_class == cls)
    if closing_iters > 0:
        binary = binary_closing(binary, iterations=closing_iters)
    lab, n = label(binary)
    bboxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.size < min_size:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        bboxes.append((x0 * cell, y0 * cell, (x1 - x0) * cell, (y1 - y0) * cell))
    return bboxes


def overlay_panel(ax, img_class, bboxes_per_class, title):
    rgba = render_grid(img_class)
    ax.imshow(rgba, interpolation="nearest")
    for cls, color in [(1, "blue"), (2, "red")]:
        for (x, y, w, h) in bboxes_per_class.get(cls, []):
            rect = mpatches.Rectangle((x, y), w, h, linewidth=1.0,
                                      edgecolor=color, facecolor="none")
            ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def crop_patch_compare(ax, gt_tile, pred_tile_a, pred_tile_b, idx):
    """3-strip view of a single patch: GT | v3.7a | v3.13."""
    gt_rgba = render_grid(gt_tile)
    a_rgba = render_grid(pred_tile_a)
    b_rgba = render_grid(pred_tile_b)
    strip = np.concatenate([gt_rgba, a_rgba, b_rgba], axis=1)
    ax.imshow(strip)
    ax.axvline(256, color="white", lw=0.7); ax.axvline(512, color="white", lw=0.7)
    ax.text(128, -10, "GT", ha="center", fontsize=8)
    ax.text(384, -10, "v3.7a", ha="center", fontsize=8)
    ax.text(640, -10, "v3.13", ha="center", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"patch #{idx}", fontsize=8)


def viz_slide(short_id, cancer_type, gt_n_tls, gt_n_gc,
              cascade_a, cascade_b, patch_lookup, save_path):
    """cascade_X = (grid_class, n_selected, pred_tiles, n_tls, n_gc) tuple."""
    grid_a, _, tiles_a, na_t, na_g = cascade_a
    grid_b, _, tiles_b, nb_t, nb_g = cascade_b
    H_g, W_g = grid_a.shape

    # GT grid: from the patch-grid mask cache (use TLS-positive priority).
    grid_gt = np.zeros_like(grid_a)
    for k, tile in patch_lookup.items():
        # Find this patch's grid coord — patch_lookup keys are global patch
        # indices; we need the grid_class location. We have the same grid
        # construction as cascade.
        pass
    # Easier path: derive GT grid from per-patch tiles directly.
    # We'll just paint each patch's class as "GC if any GC, TLS if any TLS, else bg"
    # but we need (grid_x, grid_y) per patch. Pull from patch_lookup keys
    # via the slide's coords (passed in from caller — we'll attach below).
    # Actually let's accept a precomputed grid_gt instead.
    raise NotImplementedError  # see viz_slide_v2


def viz_slide_v2(short_id, cancer_type, gt_n_tls, gt_n_gc,
                 grid_gt, cascade_a, cascade_b,
                 native_a, native_b, native_gt, cell,
                 crops, save_path):
    grid_a, _, _, na_t, na_g = cascade_a
    grid_b, _, _, nb_t, nb_g = cascade_b

    # Compute bboxes in (downsampled native-image) pixels.
    bb_a = {1: get_instance_bboxes(grid_a, 1, cell=cell),
            2: get_instance_bboxes(grid_a, 2, cell=cell)}
    bb_b = {1: get_instance_bboxes(grid_b, 1, cell=cell),
            2: get_instance_bboxes(grid_b, 2, cell=cell)}
    bb_gt = {1: get_instance_bboxes(grid_gt, 1, cell=cell),
             2: get_instance_bboxes(grid_gt, 2, cell=cell)}

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[3, 3, 3], hspace=0.30, wspace=0.18)

    ax_gt = fig.add_subplot(gs[0:2, 0])
    ax_b = fig.add_subplot(gs[0:2, 1])
    ax_a = fig.add_subplot(gs[0:2, 2])
    ax_legend = fig.add_subplot(gs[0:2, 3])

    overlay = render_grid(native_gt)
    ax_gt.imshow(overlay)
    for c, color in [(1, "blue"), (2, "red")]:
        for (x, y, w, h) in bb_gt[c]:
            ax_gt.add_patch(mpatches.Rectangle((x, y), w, h, lw=1.0,
                                               edgecolor=color, facecolor="none"))
    ax_gt.set_title(f"GT — TLS={gt_n_tls} GC={gt_n_gc}", fontsize=11)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])

    ax_b.imshow(render_grid(native_b))
    for c, color in [(1, "blue"), (2, "red")]:
        for (x, y, w, h) in bb_b[c]:
            ax_b.add_patch(mpatches.Rectangle((x, y), w, h, lw=1.0,
                                              edgecolor=color, facecolor="none"))
    ax_b.set_title(f"v3.7a (baseline) — TLS={len(bb_b[1])} GC={len(bb_b[2])}",
                   fontsize=11)
    ax_b.set_xticks([]); ax_b.set_yticks([])

    ax_a.imshow(render_grid(native_a))
    for c, color in [(1, "blue"), (2, "red")]:
        for (x, y, w, h) in bb_a[c]:
            ax_a.add_patch(mpatches.Rectangle((x, y), w, h, lw=1.0,
                                              edgecolor=color, facecolor="none"))
    ax_a.set_title(f"v3.13 (champion) — TLS={len(bb_a[1])} GC={len(bb_a[2])}",
                   fontsize=11)
    ax_a.set_xticks([]); ax_a.set_yticks([])

    ax_legend.axis("off")
    handles = [
        mpatches.Patch(color="#1e64ff", label="TLS pixels"),
        mpatches.Patch(color="#dc1e1e", label="GC pixels"),
        mpatches.Patch(facecolor="none", edgecolor="blue", label="TLS instance bbox"),
        mpatches.Patch(facecolor="none", edgecolor="red", label="GC instance bbox"),
    ]
    ax_legend.legend(handles=handles, loc="upper left", fontsize=10)
    ax_legend.set_title(f"{short_id}\n{cancer_type}", fontsize=11)

    # Detail crops row
    for j, (idx, gt_tile, pa, pb) in enumerate(crops[:4]):
        ax = fig.add_subplot(gs[2, j])
        crop_patch_compare(ax, gt_tile, pb, pa, idx)

    fig.suptitle(
        f"{short_id} — cascade comparison "
        f"(v3.13 = Stage 1 v3.8 n_hops=5 + Stage 2 v3.4b @ thr=0.5,  "
        f"v3.7a = Stage 1 v3.0 n_hops=3 + same Stage 2)",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both Stage 1's and the shared Stage 2.
    stage1_a = load_stage1(str(V8_CKPT), device)   # champion
    stage1_b = load_stage1(str(V0_CKPT), device)   # baseline
    stage2 = load_stage2(str(V4B_CKPT), device)

    # Val split.
    ps.set_seed(42)
    entries = ps.build_slide_entries()
    fp, _ = ps.create_splits(entries, k_folds=1, seed=42)
    val_entries = [e for e in fp[0] if e.get("mask_path")]

    # Counts from metadata.
    import pandas as pd
    df = pd.read_csv(ps.META_CSV)
    counts = {str(r["slide_id"]).split(".")[0]: (int(r["tls_num"]), int(r["gc_num"]))
              for _, r in df.iterrows()}

    # Patch-cache GT (native 256×256 per patch).
    print("Loading patch cache...")
    bundle = build_tls_patch_dataset(
        cache_path="/home/ubuntu/local_data/tls_patch_dataset_min4096.pt")
    bundle_masks = bundle["masks"]
    patch_mask_lookup = {}
    for ci, sid in enumerate(bundle["slide_ids"]):
        short = sid.split(".")[0]
        pi = int(bundle["patch_idx"][ci])
        patch_mask_lookup.setdefault(short, {})[pi] = np.asarray(bundle_masks[ci])

    # Pick interesting slides.
    targets = [
        "TCGA-22-1011-01Z-00-DX1",   # LUSC, 57 TLS / 15 GC
        "TCGA-34-8454-01Z-00-DX1",   # LUSC, 107 TLS / 13 GC
        "TCGA-85-6561-01Z-00-DX1",   # LUSC, 46 TLS / 13 GC
        "TCGA-GU-A42P-01Z-00-DX2",   # BLCA, 26 TLS / 11 GC
        "TCGA-B0-4945-01Z-00-DX1",   # KIRC, 1 TLS / 0 GC (control)
        "TCGA-FD-A62P-01Z-00-DX1",   # BLCA, 1 TLS / 0 GC (control)
    ]
    by_id = {e["slide_id"].split(".")[0]: e for e in val_entries}
    THRESHOLD = 0.5

    for sid in targets:
        e = by_id.get(sid)
        if e is None:
            print(f"skip {sid}: not in val")
            continue
        print(f"== {sid} ({e['cancer_type']}) ==")
        grp = zarr.open(e["zarr_path"], mode="r")
        features = torch.from_numpy(np.asarray(grp["features"][:])).float()
        coords = torch.from_numpy(np.asarray(grp["coords"][:])).float()
        edge_index = torch.from_numpy(np.asarray(grp["graph_edges_1hop"][:])).long()
        coords_np = coords.numpy()

        # Champion (v3.13) and baseline (v3.7a) cascade outputs.
        with torch.no_grad():
            grid_a, ns_a, tiles_a = cascade_one_slide(
                stage1_a, stage2, features, coords, edge_index, THRESHOLD,
                device, s2_batch=64)
            grid_b, ns_b, tiles_b = cascade_one_slide(
                stage1_b, stage2, features, coords, edge_index, THRESHOLD,
                device, s2_batch=64)

        # GT grid: derive from patch lookup directly (same indexing as cascade).
        grid_x = (coords_np[:, 0] / PATCH_SIZE).astype(np.int64)
        grid_y = (coords_np[:, 1] / PATCH_SIZE).astype(np.int64)
        grid_x -= grid_x.min(); grid_y -= grid_y.min()
        H_g, W_g = int(grid_y.max()) + 1, int(grid_x.max()) + 1
        slide_lookup = patch_mask_lookup.get(sid, {})
        grid_gt = np.zeros((H_g, W_g), dtype=np.int64)
        for k, (gx, gy) in enumerate(zip(grid_x, grid_y)):
            t = slide_lookup.get(int(k))
            if t is None:
                continue
            t = np.asarray(t)
            if (t == 2).any():
                grid_gt[gy, gx] = 2
            elif (t == 1).any():
                grid_gt[gy, gx] = 1

        # Native-resolution slide images (downsampled).
        native_a, _, _, cell = native_pred_image(coords_np, tiles_a, downsample=8)
        native_b, _, _, _ = native_pred_image(coords_np, tiles_b, downsample=8)
        native_gt, _, _, _ = gt_native_image(coords_np, slide_lookup, downsample=8)

        # Pick crops where the two models disagree (within selected patches).
        common = set(tiles_a.keys()) & set(tiles_b.keys())
        diffs = []
        for k in common:
            ta = tiles_a[int(k)]; tb = tiles_b[int(k)]
            d = int((ta != tb).sum())
            if d > 100:  # ignore tiny disagreements
                diffs.append((d, int(k)))
        diffs.sort(reverse=True)
        crops = []
        for _, k in diffs[:4]:
            gt_tile = slide_lookup.get(k, np.zeros((256, 256), dtype=np.uint8))
            crops.append((k, np.asarray(gt_tile),
                          tiles_a[int(k)], tiles_b[int(k)]))
        # If we don't have 4 disagreement crops, top up with any selected patches.
        if len(crops) < 4:
            for k in list(common)[:4 - len(crops)]:
                gt_tile = slide_lookup.get(k, np.zeros((256, 256), dtype=np.uint8))
                crops.append((k, np.asarray(gt_tile),
                              tiles_a[int(k)], tiles_b[int(k)]))

        gt_n_tls, gt_n_gc = counts.get(sid, (-1, -1))
        save_path = OUT_DIR / f"{sid}.png"
        viz_slide_v2(
            sid, e["cancer_type"], gt_n_tls, gt_n_gc, grid_gt,
            cascade_a=(grid_a, ns_a, tiles_a,
                       len(get_instance_bboxes(grid_a, 1)),
                       len(get_instance_bboxes(grid_a, 2))),
            cascade_b=(grid_b, ns_b, tiles_b,
                       len(get_instance_bboxes(grid_b, 1)),
                       len(get_instance_bboxes(grid_b, 2))),
            native_a=native_a, native_b=native_b, native_gt=native_gt, cell=cell,
            crops=crops, save_path=save_path,
        )

    print(f"\nDone — {OUT_DIR}")


if __name__ == "__main__":
    main()
