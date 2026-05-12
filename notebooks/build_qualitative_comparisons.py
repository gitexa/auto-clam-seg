"""Qualitative per-slide segmentation comparison images.

For 5 slides per cancer type (BLCA / KIRC / LUSC, 15 total) from fold-0 val,
render side-by-side panels:

    WSI thumbnail | GT mask | Cascade v3.37 | GNCAF v3.65 | seg_v2.0 (dual)

Each prediction is rendered at the patch-grid level (1 cell per 256-px patch)
with class colors: bg=transparent, TLS=red, GC=yellow. All overlays are at the
thumbnail resolution (WSI level 7).

Output: notebooks/architectures/qualitative_comparisons/<cancer>_<slide_id>.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import zarr
from scipy import ndimage

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")

import prepare_segmentation as ps
from gncaf_dataset import slide_wsi_path, slide_mask_path
from eval_gars_cascade import load_stage1, load_stage2_region, cascade_one_slide_region
from eval_gars_gncaf_transunet import load_gncaf_transunet, eval_one_slide as gncaf_eval_one_slide
from eval_gars_gncaf_transunet import _load_features_and_graph

PATCH_SIZE = 256
OUT = Path("/home/ubuntu/auto-clam-seg/notebooks/architectures/qualitative_comparisons")
OUT.mkdir(parents=True, exist_ok=True)

# Production ckpts per CHAMPION_MODEL_REGISTRY.md
CKPT_STAGE1 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt"
CKPT_CASC_S2 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_region_v3.37_full_20260502_144124/best_checkpoint.pt"
CKPT_GNCAF = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_gncaf_v3.65_dual_sigmoid_simple_loss_20260511_011243/best_checkpoint.pt"
CKPT_SEG_V2 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/seg_v2.0_dual_tls_gc_5fold_57e12399/fold_0/best_checkpoint.pt"


def patch_grid_to_thumbnail(grid: np.ndarray, thumb_shape: tuple[int, int],
                            slide_h_pix: int, slide_w_pix: int) -> np.ndarray:
    """Upsample (H_grid, W_grid) class grid → (H_thumb, W_thumb) nearest-neighbor."""
    h_g, w_g = grid.shape
    # Each grid cell covers PATCH_SIZE × PATCH_SIZE level-0 pixels.
    # Slide pix → thumb pix scale.
    h_t, w_t = thumb_shape
    # Resample with nearest neighbor: each grid cell maps to a rectangle on thumb.
    out = np.zeros((h_t, w_t), dtype=np.int64)
    for gy in range(h_g):
        y0 = int(gy * PATCH_SIZE / slide_h_pix * h_t)
        y1 = int((gy + 1) * PATCH_SIZE / slide_h_pix * h_t)
        for gx in range(w_g):
            x0 = int(gx * PATCH_SIZE / slide_w_pix * w_t)
            x1 = int((gx + 1) * PATCH_SIZE / slide_w_pix * w_t)
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] = grid[gy, gx]
    return out


def load_thumbnail(wsi_path: str, level: int = 7) -> np.ndarray:
    """Return RGB thumbnail at the given pyramid level."""
    with tifffile.TiffFile(wsi_path) as tif:
        if level >= len(tif.pages):
            level = len(tif.pages) - 1
        thumb = tif.pages[level].asarray()
    return thumb


def get_slide_dims(wsi_path: str) -> tuple[int, int]:
    """Return (H, W) of level 0."""
    with tifffile.TiffFile(wsi_path) as tif:
        return tif.pages[0].shape[:2]


def load_gt_grid(entry: dict, slide_h: int, slide_w: int,
                 coords: np.ndarray) -> np.ndarray:
    """Build a patch-grid GT (0=bg, 1=TLS, 2=GC) from the HookNet mask tif.

    Returns (H_grid, W_grid) int array.
    """
    grid_y = (coords[:, 1] // PATCH_SIZE).astype(np.int64)
    grid_x = (coords[:, 0] // PATCH_SIZE).astype(np.int64)
    H_g = int(grid_y.max()) + 1
    W_g = int(grid_x.max()) + 1
    out = np.zeros((H_g, W_g), dtype=np.int64)
    mask_path = slide_mask_path(entry)
    if mask_path is None or not Path(mask_path).exists():
        return out
    # Read mask at level 0 in patch-size chunks
    mask_z = zarr.open(tifffile.imread(mask_path, aszarr=True, level=0), mode="r")
    for i in range(len(coords)):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        tile = np.asarray(mask_z[y:y + PATCH_SIZE, x:x + PATCH_SIZE])
        n_gc = int((tile == 2).sum())
        n_tls = int((tile == 1).sum())
        cls = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)
        out[grid_y[i], grid_x[i]] = cls
    return out


# ── Cascade inference (returns per-patch grid 0/1/2) ────────────────────


def cascade_predict(stage1, stage2, entry: dict, device: torch.device) -> np.ndarray:
    """Run cascade Stage 1 + Stage 2 on a single slide, return per-patch grid."""
    feats_np, coords_np, edge_np = _load_features_and_graph(entry["zarr_path"])
    # cascade_one_slide_region wants features as a CPU torch tensor —
    # the function calls `.to(device)` for Stage 1 forward AND `.numpy()`
    # for per-cell UNI lookup. CPU tensor satisfies both.
    features = torch.from_numpy(feats_np)
    coords = torch.from_numpy(coords_np).long()
    edge_index = torch.from_numpy(edge_np).long().to(device)
    grid_class, _, _ = cascade_one_slide_region(
        stage1, stage2, features, coords, edge_index,
        threshold=0.5, s2_batch=8, device=device, wsi_path=slide_wsi_path(entry),
        return_cell_probs=False,
    )
    return grid_class  # (H_grid, W_grid)


# ── GNCAF inference (returns per-patch grid 0/1/2) ──────────────────────


def gncaf_predict(model, entry: dict, device: torch.device,
                  batch_size: int = 16, num_workers: int = 4) -> np.ndarray:
    """Run GNCAF on a single slide and return its per-patch class grid."""
    result = gncaf_eval_one_slide(model, entry, device, batch_size=batch_size,
                                   num_workers=num_workers)
    return result["pred_grid"]  # (H_grid, W_grid)


# ── seg_v2.0 inference (returns per-patch grid 0/1/2) ──────────────────


def seg_v2_predict(model, entry: dict, device: torch.device) -> np.ndarray:
    """Run seg_v2.0 (dual) on a single slide; max-pool sub-patch
    predictions to per-patch grid."""
    grp = zarr.open(entry["zarr_path"], mode="r")
    features = torch.from_numpy(grp["features"][:]).float().to(device)
    coords = torch.from_numpy(grp["coords"][:]).float().to(device)
    edge_index = None
    if "graph_edges_1hop" in grp:
        edge_index = torch.from_numpy(grp["graph_edges_1hop"][:]).long().to(device)
    elif "edge_index" in grp:
        edge_index = torch.from_numpy(grp["edge_index"][:]).long().to(device)
    if edge_index is None:
        # No graph → can't run
        return np.zeros((1, 1), dtype=np.int64)
    with torch.no_grad():
        out = model(features, coords, edge_index, return_attention_weights=False)
    p_tls = (torch.sigmoid(out["logits_tls"]).squeeze().cpu().numpy() > 0.5)
    p_gc = (torch.sigmoid(out["logits_gc"]).squeeze().cpu().numpy() > 0.5)
    # Map sub-patch resolution to per-patch grid via max-pooling.
    coords_np = grp["coords"][:]
    grid_y = (coords_np[:, 1] // PATCH_SIZE).astype(np.int64)
    grid_x = (coords_np[:, 0] // PATCH_SIZE).astype(np.int64)
    H_g = int(grid_y.max()) + 1
    W_g = int(grid_x.max()) + 1
    upsample = 4   # seg_v2.0 dual uses upsample_factor=4
    grid = np.zeros((H_g, W_g), dtype=np.int64)
    for gy in range(H_g):
        for gx in range(W_g):
            y0 = gy * upsample; y1 = (gy + 1) * upsample
            x0 = gx * upsample; x1 = (gx + 1) * upsample
            if y1 > p_tls.shape[0] or x1 > p_tls.shape[1]:
                continue
            tls = bool(p_tls[y0:y1, x0:x1].any())
            gc = bool(p_gc[y0:y1, x0:x1].any())
            grid[gy, gx] = 2 if gc else (1 if tls else 0)
    return grid


# ── Visualization ───────────────────────────────────────────────────────


CMAP = {0: (0, 0, 0, 0), 1: (255, 60, 60, 110), 2: (255, 220, 0, 200)}   # bg, TLS, GC


def colorise_grid(grid_class: np.ndarray) -> np.ndarray:
    """Convert (H, W) integer-class array to (H, W, 4) RGBA uint8."""
    h, w = grid_class.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, (r, g, b, a) in CMAP.items():
        m = (grid_class == cls)
        out[m] = (r, g, b, a)
    return out


def render_panel(thumb_rgb: np.ndarray, grids: dict[str, np.ndarray],
                 slide_h_pix: int, slide_w_pix: int, slide_id: str,
                 cancer_type: str, out_path: Path):
    """Render thumbnail + N model overlay panels side by side."""
    titles = ["WSI thumbnail"] + list(grids.keys())
    n = len(titles)
    h_t, w_t = thumb_rgb.shape[:2]
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5 * (h_t / w_t * (3.2 / 1.0))))
    if n == 1:
        axes = [axes]

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(thumb_rgb)

    axes[0].set_title(f"{cancer_type} {slide_id}", fontsize=9)

    for i, (label, grid) in enumerate(grids.items(), start=1):
        ax = axes[i]
        overlay = colorise_grid(patch_grid_to_thumbnail(
            grid, (h_t, w_t), slide_h_pix, slide_w_pix
        ))
        ax.imshow(overlay)
        # Add stats to title
        n_tls = int((grid == 1).sum())
        n_gc = int((grid == 2).sum())
        ax.set_title(f"{label}\nTLS patches={n_tls}  GC patches={n_gc}", fontsize=9)

    fig.suptitle(f"TLS (red) + GC (yellow) per-patch predictions — {cancer_type} {slide_id}",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────


def select_slides(entries: list[dict], n_per_cancer: int = 5,
                  cancers=("BLCA", "KIRC", "LUSC")) -> list[dict]:
    """Pick N TLS-positive slides per cancer type, deterministic."""
    selected = []
    for ct in cancers:
        pool = [e for e in entries
                if e["cancer_type"] == ct and e.get("mask_path") is not None]
        # Sort by slide_id for determinism, take first N
        pool.sort(key=lambda e: e["slide_id"])
        selected.extend(pool[:n_per_cancer])
    return selected


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Fold-0 val entries
    entries = ps.build_slide_entries()
    folds, _ = ps.create_splits(entries, k_folds=5, seed=42)
    val_entries = folds[0]
    print(f"Fold-0 val: {len(val_entries)} slides")

    slides = select_slides(val_entries, n_per_cancer=5)
    print(f"Selected {len(slides)} slides for visualization "
          f"({sum(1 for s in slides if s['cancer_type']=='BLCA')} BLCA, "
          f"{sum(1 for s in slides if s['cancer_type']=='KIRC')} KIRC, "
          f"{sum(1 for s in slides if s['cancer_type']=='LUSC')} LUSC)")

    # Load models
    print("Loading models...")
    stage1 = load_stage1(CKPT_STAGE1, device)
    stage2 = load_stage2_region(CKPT_CASC_S2, device)
    gncaf = load_gncaf_transunet(CKPT_GNCAF, device)
    # seg_v2.0 (dual)
    sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
    from eval_seg_v2 import GNNSegV2, FEATURE_DIM, PATCH_SIZE as SV2_PATCH
    seg_v2_obj = torch.load(CKPT_SEG_V2, map_location="cpu", weights_only=False)
    seg_v2 = GNNSegV2(feature_dim=FEATURE_DIM, hidden_dim=384, gnn_layers=0,
                     gnn_heads=4, upsample_factor=4, n_classes=3, dropout=0.1,
                     patch_size=SV2_PATCH).to(device).eval()
    seg_v2.load_state_dict(seg_v2_obj["model_state_dict"], strict=True)

    # Run per slide
    for entry in slides:
        short_id = entry["slide_id"].split(".")[0]
        ct = entry["cancer_type"]
        out_path = OUT / f"{ct}_{short_id}.png"
        wsi_path = slide_wsi_path(entry)
        if not wsi_path or not Path(wsi_path).exists():
            print(f"  {short_id}: WSI missing, skip")
            continue

        print(f"\n[{ct}] {short_id}")
        thumb = load_thumbnail(wsi_path, level=7)
        slide_h, slide_w = get_slide_dims(wsi_path)

        # GT
        grp = zarr.open(entry["zarr_path"], mode="r")
        coords = grp["coords"][:]
        gt_grid = load_gt_grid(entry, slide_h, slide_w, coords)

        # Cascade
        try:
            casc_grid = cascade_predict(stage1, stage2, entry, device)
        except Exception as e:
            print(f"  cascade failed: {e}")
            casc_grid = np.zeros_like(gt_grid)

        # GNCAF
        try:
            gncaf_grid = gncaf_predict(gncaf, entry, device,
                                       batch_size=16, num_workers=4)
        except Exception as e:
            print(f"  GNCAF failed: {e}")
            gncaf_grid = np.zeros_like(gt_grid)

        # seg_v2.0 (dual)
        try:
            sv2_grid = seg_v2_predict(seg_v2, entry, device)
        except Exception as e:
            print(f"  seg_v2.0 failed: {e}")
            sv2_grid = np.zeros_like(gt_grid)

        grids = {
            "Ground truth": gt_grid,
            "Cascade v3.37": casc_grid,
            "GNCAF v3.65": gncaf_grid,
            "seg_v2.0 (dual)": sv2_grid,
        }
        render_panel(thumb, grids, slide_h, slide_w, short_id, ct, out_path)


if __name__ == "__main__":
    main()
