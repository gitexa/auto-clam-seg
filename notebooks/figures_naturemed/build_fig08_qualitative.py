"""Fig 08 — Qualitative per-slide instance crops (Nature-Med style).

For each of 9 slides (3 per cohort × {low, mid, high} GT TLS count):
  Row 0: full-slide overlays for 6 models
  Rows 1..N: per-instance crops at WSI level-2 (limit 20 instances/slide)
    Columns: WSI crop | GT mask | v3.11 | GNCAF v3.65 | seg_v2.0 (dual) | HookNet

Output: figures_naturemed/qualitative/<cancer>_<slide_id>.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
import torch
import zarr
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, "/home/ubuntu/auto-clam-seg/notebooks")

import prepare_segmentation as ps  # noqa: E402
from gncaf_dataset import slide_wsi_path, slide_mask_path  # noqa: E402
from eval_gars_cascade import load_stage1, load_stage2_region, cascade_one_slide_region  # noqa: E402
from eval_gars_gncaf_transunet import load_gncaf_transunet, eval_one_slide as gncaf_eval_one_slide  # noqa: E402
from eval_gars_gncaf_transunet import _load_features_and_graph  # noqa: E402

PATCH = 256
CROP_LEVEL = 2  # ~4 µm/px
ROOT = Path("/home/ubuntu/auto-clam-seg/figures_naturemed/qualitative")
ROOT.mkdir(parents=True, exist_ok=True)

# Model ckpts
CKPT_S1_V37 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_stage1_v3.8_gatv2_5hop_20260501_043445/best_checkpoint.pt"
CKPT_S2_V37 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_region_v3.37_full_20260502_144124/best_checkpoint.pt"
CKPT_S1_V11 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_cascade_v3.11_joint_v2_fold0_20260514_070326/stage1_extracted.pt"
CKPT_S2_V11 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_cascade_v3.11_joint_v2_fold0_20260514_070326/stage2_extracted.pt"
CKPT_GNCAF65 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/gars_gncaf_v3.65_dual_sigmoid_simple_loss_20260511_011243/best_checkpoint.pt"
CKPT_SEG_V2 = "/home/ubuntu/ahaas-persistent-std-tcga/experiments/seg_v2.0_dual_tls_gc_5fold_57e12399/fold_0/best_checkpoint.pt"
HOOKNET_OUT = Path("/home/ubuntu/local_data/hooknet_out")

CMAP = {0: (0, 0, 0, 0), 1: (255, 60, 60, 110), 2: (255, 220, 0, 200)}


def colorise(grid):
    h, w = grid.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, (r, g, b, a) in CMAP.items():
        m = (grid == cls); out[m] = (r, g, b, a)
    return out


def patch_grid_to_full(grid, target_h, target_w, slide_h_pix, slide_w_pix):
    """Upsample grid to (target_h, target_w) at requested aspect."""
    h_g, w_g = grid.shape
    out = np.zeros((target_h, target_w), dtype=np.int64)
    for gy in range(h_g):
        y0 = int(gy * PATCH / slide_h_pix * target_h)
        y1 = int((gy + 1) * PATCH / slide_h_pix * target_h)
        for gx in range(w_g):
            x0 = int(gx * PATCH / slide_w_pix * target_w)
            x1 = int((gx + 1) * PATCH / slide_w_pix * target_w)
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] = grid[gy, gx]
    return out


def load_thumbnail(wsi, level=7):
    with tifffile.TiffFile(wsi) as tf:
        lv = min(level, len(tf.pages) - 1)
        return tf.pages[lv].asarray()


def crop_wsi_region(wsi, gy0, gx0, gy1, gx1, level=CROP_LEVEL):
    stride = 1 << level
    y0_l0 = gy0 * PATCH; y1_l0 = (gy1 + 1) * PATCH
    x0_l0 = gx0 * PATCH; x1_l0 = (gx1 + 1) * PATCH
    y0, y1 = y0_l0 // stride, y1_l0 // stride
    x0, x1 = x0_l0 // stride, x1_l0 // stride
    wz = zarr.open(tifffile.imread(wsi, aszarr=True, level=level), mode="r")
    H, W = wz.shape[:2]
    y0 = max(0, y0); x0 = max(0, x0)
    y1 = min(H, y1); x1 = min(W, x1)
    arr = np.asarray(wz[y0:y1, x0:x1])
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def gt_grid(entry, coords):
    """Build patch-grid GT (0/1/2) from HookNet mask tif."""
    grid_y = (coords[:, 1] // PATCH).astype(np.int64)
    grid_x = (coords[:, 0] // PATCH).astype(np.int64)
    H_g = int(grid_y.max()) + 1; W_g = int(grid_x.max()) + 1
    out = np.zeros((H_g, W_g), dtype=np.int64)
    mp = slide_mask_path(entry)
    if mp is None or not Path(mp).exists():
        return out
    mz = zarr.open(tifffile.imread(mp, aszarr=True, level=0), mode="r")
    for i in range(len(coords)):
        x = int(coords[i, 0]); y = int(coords[i, 1])
        tile = np.asarray(mz[y:y + PATCH, x:x + PATCH])
        n_gc = int((tile == 2).sum()); n_tls = int((tile == 1).sum())
        out[grid_y[i], grid_x[i]] = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)
    return out


def cascade_grid(stage1, stage2, entry, device):
    feats, coords, edges = _load_features_and_graph(entry["zarr_path"])
    features = torch.from_numpy(feats); coords_t = torch.from_numpy(coords).long()
    edge_index = torch.from_numpy(edges).long().to(device)
    grid_class, _, _ = cascade_one_slide_region(
        stage1, stage2, features, coords_t, edge_index,
        threshold=0.5, s2_batch=8, device=device, wsi_path=slide_wsi_path(entry),
        min_component_size=2, closing_iters=1, return_cell_probs=False,
    )
    return grid_class


def gncaf_grid(model, entry, device):
    return gncaf_eval_one_slide(model, entry, device, batch_size=16, num_workers=4)["pred_grid"]


def seg_v2_grid(model, entry, device):
    grp = zarr.open(entry["zarr_path"], mode="r")
    features = torch.from_numpy(grp["features"][:]).float().to(device)
    coords = torch.from_numpy(grp["coords"][:]).float().to(device)
    ei = None
    if "graph_edges_1hop" in grp:
        ei = torch.from_numpy(grp["graph_edges_1hop"][:]).long().to(device)
    if ei is None:
        return np.zeros((1, 1), dtype=np.int64)
    with torch.no_grad():
        out = model(features, coords, ei, return_attention_weights=False)
    p_tls = torch.sigmoid(out["logits_tls"]).squeeze().cpu().numpy() > 0.5
    p_gc = torch.sigmoid(out["logits_gc"]).squeeze().cpu().numpy() > 0.5
    coords_np = np.asarray(grp["coords"][:])
    grid_y = (coords_np[:, 1] // PATCH).astype(np.int64)
    grid_x = (coords_np[:, 0] // PATCH).astype(np.int64)
    H_g = int(grid_y.max()) + 1; W_g = int(grid_x.max()) + 1
    upsample = 4
    grid = np.zeros((H_g, W_g), dtype=np.int64)
    for gy in range(H_g):
        for gx in range(W_g):
            y0 = gy * upsample; y1 = (gy + 1) * upsample
            x0 = gx * upsample; x1 = (gx + 1) * upsample
            if y1 > p_tls.shape[0] or x1 > p_tls.shape[1]: continue
            tls = bool(p_tls[y0:y1, x0:x1].any()); gc = bool(p_gc[y0:y1, x0:x1].any())
            grid[gy, gx] = 2 if gc else (1 if tls else 0)
    return grid


def hooknet_grid(sid, slide_h_l0, slide_w_l0):
    """Rasterize HookNet polygons to patch grid (256-px cells)."""
    tls_p = HOOKNET_OUT / sid / "images/filtered" / f"{sid}_hooknettls_tls_filtered.json"
    gc_p = HOOKNET_OUT / sid / "images/filtered" / f"{sid}_hooknettls_gc_filtered.json"
    H_g = (slide_h_l0 + PATCH - 1) // PATCH
    W_g = (slide_w_l0 + PATCH - 1) // PATCH
    grid = np.zeros((H_g, W_g), dtype=np.int64)
    for fp, cls in [(tls_p, 1), (gc_p, 2)]:
        if not fp.exists(): continue
        polys = json.loads(fp.read_text())
        img = Image.new("L", (W_g, H_g), 0)
        draw = ImageDraw.Draw(img)
        for p in polys:
            coords = p.get("coordinates", [])
            if not coords or len(coords) < 3: continue
            pts = [(c[0] / PATCH, c[1] / PATCH) for c in coords]
            draw.polygon(pts, fill=1)
        mask = np.asarray(img)
        grid[(mask > 0) & (grid == 0)] = cls
        grid[(mask > 0) & (grid == 1) & (cls == 2)] = 2  # GC overrides TLS
    return grid


def gt_instances(grid, margin=2, max_inst=20):
    fg = (grid > 0).astype(np.uint8)
    if fg.sum() == 0: return []
    lbl, n = ndimage.label(fg)
    bboxes = []
    for k in range(1, n + 1):
        mask = (lbl == k); size = int(mask.sum())
        if size == 0: continue
        ys, xs = np.where(mask)
        gy0 = max(0, int(ys.min()) - margin); gx0 = max(0, int(xs.min()) - margin)
        gy1 = min(grid.shape[0]-1, int(ys.max()) + margin); gx1 = min(grid.shape[1]-1, int(xs.max()) + margin)
        bboxes.append((size, gy0, gx0, gy1, gx1))
    bboxes.sort(reverse=True, key=lambda t: t[0])
    return [b[1:] for b in bboxes[:max_inst]]


def crop_grid(grid, bbox):
    gy0, gx0, gy1, gx1 = bbox
    return grid[gy0:gy1+1, gx0:gx1+1]


def render_slide(slide_id, cancer, entry, predictions, gt_g, thumb, slide_h, slide_w, out_path):
    """predictions: dict[name -> per-patch grid] from each model."""
    model_names = list(predictions.keys())
    # Top row: thumbnail | GT | each model overlay
    titles = ["WSI thumbnail", "Ground truth"] + model_names
    n_cols = len(titles)
    bboxes = gt_instances(gt_g, margin=2, max_inst=20)
    n_rows = 1 + len(bboxes)
    h_t, w_t = thumb.shape[:2]
    aspect = h_t / w_t
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.4 * n_cols, 2.4 * aspect * (1 + 0.7 * len(bboxes))),
                              squeeze=False)
    # Row 0 — full-slide views
    for c, ax in enumerate(axes[0]):
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(thumb)
    axes[0, 0].set_title(f"{cancer} {slide_id}", fontsize=8)
    axes[0, 1].set_title(f"Ground truth\nTLS={(gt_g==1).sum()} GC={(gt_g==2).sum()}", fontsize=8)
    gt_overlay = colorise(patch_grid_to_full(gt_g, h_t, w_t, slide_h, slide_w))
    axes[0, 1].imshow(gt_overlay)
    for i, name in enumerate(model_names, start=2):
        ax = axes[0, i]
        grid = predictions[name]
        overlay = colorise(patch_grid_to_full(grid, h_t, w_t, slide_h, slide_w))
        ax.imshow(overlay)
        ax.set_title(f"{name}\nTLS={(grid==1).sum()} GC={(grid==2).sum()}", fontsize=8)
    # Rows 1..N — per-instance crops
    wsi = slide_wsi_path(entry)
    for r, bbox in enumerate(bboxes, start=1):
        gy0, gx0, gy1, gx1 = bbox
        try:
            region = crop_wsi_region(wsi, gy0, gx0, gy1, gx1, level=CROP_LEVEL)
        except Exception:
            region = thumb[:32, :32]
        ch, cw = region.shape[:2]
        for c, ax in enumerate(axes[r]):
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(region)
        # Title for column 0
        physical_um = (gy1 - gy0 + 1) * PATCH * 0.5
        axes[r, 0].set_title(f"Instance {r}\n~{physical_um:.0f} µm", fontsize=8)
        # GT crop col 1
        g_crop = crop_grid(gt_g, bbox)
        cell_px = max(1, PATCH // (2 ** CROP_LEVEL))
        target_h = max(1, g_crop.shape[0] * cell_px); target_w = max(1, g_crop.shape[1] * cell_px)
        ov = colorise(patch_grid_to_full(g_crop, target_h, target_w,
                                          g_crop.shape[0] * PATCH, g_crop.shape[1] * PATCH))
        axes[r, 1].imshow(region, extent=(0, target_w, target_h, 0))
        axes[r, 1].imshow(ov, extent=(0, target_w, target_h, 0))
        axes[r, 1].set_title(f"TLS={(g_crop==1).sum()} GC={(g_crop==2).sum()}", fontsize=8)
        # Model crops col 2..
        for i, name in enumerate(model_names, start=2):
            pad_grid = predictions[name]
            # Pad if smaller than bbox extent
            need_h = bbox[2] + 1; need_w = bbox[3] + 1
            if pad_grid.shape[0] < need_h or pad_grid.shape[1] < need_w:
                pad_grid = np.pad(pad_grid, ((0, max(0, need_h - pad_grid.shape[0])),
                                              (0, max(0, need_w - pad_grid.shape[1]))))
            pg_crop = crop_grid(pad_grid, bbox)
            if pg_crop.shape[0] == 0 or pg_crop.shape[1] == 0:
                axes[r, i].set_title("(oob)", fontsize=7); continue
            ov2 = colorise(patch_grid_to_full(pg_crop, target_h, target_w,
                                                pg_crop.shape[0] * PATCH, pg_crop.shape[1] * PATCH))
            axes[r, i].imshow(region, extent=(0, target_w, target_h, 0))
            axes[r, i].imshow(ov2, extent=(0, target_w, target_h, 0))
            axes[r, i].set_title(f"TLS={(pg_crop==1).sum()} GC={(pg_crop==2).sum()}", fontsize=7)
    fig.suptitle(f"Fig 08 — Qualitative comparison — {cancer} {slide_id}\n"
                 f"Row 0 full slide; rows 1-{len(bboxes)} per-GT-instance crops at WSI L{CROP_LEVEL}.",
                 y=1.00, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path} ({len(bboxes)} instances)")


def pick_slides():
    """Pick 3 slides per cohort: low (gt_n_tls=0 or low), mid, high."""
    df = pd.read_csv("/home/ubuntu/auto-clam-seg/figures_naturemed/per_slide_predictions.csv")
    out = []
    for ct in ("BLCA", "KIRC", "LUSC"):
        sub = df[df.cancer_type == ct].sort_values("gt_n_tls")
        if len(sub) == 0: continue
        # low (smallest non-zero TLS or first available)
        non_zero = sub[sub.gt_n_tls > 0]
        if len(non_zero) >= 3:
            low = non_zero.iloc[0]
            mid = non_zero.iloc[len(non_zero) // 2]
            high = non_zero.iloc[-1]
        else:
            low = sub.iloc[0]; mid = sub.iloc[len(sub) // 2]; high = sub.iloc[-1]
        for label, row in [("low", low), ("mid", mid), ("high", high)]:
            out.append((ct, label, row["slide_id"], int(row["gt_n_tls"])))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading models...")
    stage1_v37 = load_stage1(CKPT_S1_V37, device)
    stage2_v37 = load_stage2_region(CKPT_S2_V37, device)
    stage1_v11 = load_stage1(CKPT_S1_V11, device)
    stage2_v11 = load_stage2_region(CKPT_S2_V11, device)
    gncaf65 = load_gncaf_transunet(CKPT_GNCAF65, device)
    # seg_v2.0 (dual)
    from eval_seg_v2 import GNNSegV2, FEATURE_DIM, PATCH_SIZE as SV2_PATCH
    seg_v2_obj = torch.load(CKPT_SEG_V2, map_location="cpu", weights_only=False)
    seg_v2 = GNNSegV2(feature_dim=FEATURE_DIM, hidden_dim=384, gnn_layers=0, gnn_heads=4,
                       upsample_factor=4, n_classes=3, dropout=0.1, patch_size=SV2_PATCH).to(device).eval()
    seg_v2.load_state_dict(seg_v2_obj["model_state_dict"], strict=True)

    slides = pick_slides()
    print(f"Selected {len(slides)} slides")
    entries = ps.build_slide_entries()
    entry_by_short = {e["slide_id"].split(".")[0]: e for e in entries}
    for ct, level, sid, gt_t in slides:
        e = entry_by_short.get(sid)
        if e is None: print(f"  {sid}: missing entry, skip"); continue
        wsi = slide_wsi_path(e)
        if not wsi or not Path(wsi).exists(): print(f"  {sid}: missing WSI"); continue
        print(f"\n[{ct} {level}] {sid} (GT TLS={gt_t})")
        thumb = load_thumbnail(wsi, level=7)
        with tifffile.TiffFile(wsi) as tf:
            slide_h, slide_w = tf.pages[0].shape[:2]
        # GT grid
        grp = zarr.open(e["zarr_path"], mode="r")
        coords = grp["coords"][:]
        gt_g = gt_grid(e, coords)
        # Predictions
        preds = {}
        try: preds["Cascade v3.7"] = cascade_grid(stage1_v37, stage2_v37, e, device)
        except Exception as ex: print(f"  v3.7 fail: {ex}"); preds["Cascade v3.7"] = np.zeros_like(gt_g)
        try: preds["Cascade v3.11"] = cascade_grid(stage1_v11, stage2_v11, e, device)
        except Exception as ex: print(f"  v3.11 fail: {ex}"); preds["Cascade v3.11"] = np.zeros_like(gt_g)
        try: preds["GNCAF v3.65"] = gncaf_grid(gncaf65, e, device)
        except Exception as ex: print(f"  GNCAF fail: {ex}"); preds["GNCAF v3.65"] = np.zeros_like(gt_g)
        try: preds["seg_v2.0 (dual)"] = seg_v2_grid(seg_v2, e, device)
        except Exception as ex: print(f"  seg_v2 fail: {ex}"); preds["seg_v2.0 (dual)"] = np.zeros_like(gt_g)
        try: preds["HookNet-TLS"] = hooknet_grid(sid, slide_h, slide_w)
        except Exception as ex: print(f"  HookNet fail: {ex}"); preds["HookNet-TLS"] = np.zeros_like(gt_g)

        out_path = ROOT / f"{ct}_{level}_{sid}.png"
        render_slide(sid, ct, e, preds, gt_g, thumb, slide_h, slide_w, out_path)


if __name__ == "__main__":
    main()
