"""Instance-level object-detection F1 — HookNet-style metric (pixel-level).

Matches at WSI pyramid level `match_level` (default 2 = 4 px/cell at level 0)
to balance fidelity vs compute. Stage 2's per-patch pixel-level predictions
(`pred_tiles` from `cascade_one_slide_region`) are pasted into a level-0
mask, downsampled to `match_level`, and matched against the GT instance
mask at the same level.

For each slide:
  1. Cascade -> `pred_tiles: {patch_idx: (256, 256) uint8 class mask}` at L0.
  2. Build a sparse level-0 prediction mask, fill in Stage 2 tiles at their
     coords.
  3. Downsample to `match_level` by stride (preserves class IDs).
  4. Load GT instance mask at L0, downsample to `match_level`.
  5. For each class c in {1, 2}: extract pred CCs of class c (scipy.label);
     for each GT instance of class c (instance_id -> class via JSON map),
     compute IoU vs each pred CC. Greedy match at IoU >= threshold.
  6. TP/FP/FN per class -> per-slide -> per-cohort -> overall P/R/F1.

Run:
    python eval_instance_f1.py \\
        stage1=<path> region_mode=true stage2_region=<path> \\
        fold_idx=0 k_folds=5 \\
        +iou_threshold=0.5 +match_level=2 \\
        label=v3.7_instance_f1_fold0_l2
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import tifffile
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import ndimage

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")

import prepare_segmentation as ps  # noqa: E402
from gncaf_dataset import slide_wsi_path  # noqa: E402
from eval_gars_cascade import (  # noqa: E402
    load_stage1, load_stage2_region, cascade_one_slide_region,
)
from eval_gars_gncaf_transunet import _load_features_and_graph  # noqa: E402

PATCH = 256
INST_DIR = Path("/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/masks_instance")


def _grid_components(grid: np.ndarray, cls: int):
    mask = (grid == cls).astype(np.uint8)
    lbl, n = ndimage.label(mask)
    return lbl, n


def match_instances(gt_inst_grid: np.ndarray, gt_inst_class: np.ndarray,
                    gt_inst_ids: np.ndarray,
                    pred_grid: np.ndarray, target_class: int,
                    iou_threshold: float = 0.5):
    """Greedy matching of pred CCs to GT instances for a single class."""
    cls_mask = (gt_inst_class == target_class)
    gt_ids_cls = gt_inst_ids[cls_mask]
    gt_masks = []
    for gid in gt_ids_cls:
        gm = (gt_inst_grid == int(gid))
        if gm.any():
            gt_masks.append((int(gid), gm))

    pred_lbl, n_pred = _grid_components(pred_grid, target_class)
    pred_masks = []
    for pid in range(1, n_pred + 1):
        pm = (pred_lbl == pid)
        if pm.any():
            pred_masks.append((pid, pm))

    if not gt_masks and not pred_masks:
        return 0, 0, 0, []
    if not gt_masks:
        return 0, len(pred_masks), 0, []
    if not pred_masks:
        return 0, 0, len(gt_masks), []

    iou_mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for gi, (_, gm) in enumerate(gt_masks):
        for pi, (_, pm) in enumerate(pred_masks):
            inter = int(np.logical_and(gm, pm).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(gm, pm).sum())
            iou_mat[gi, pi] = inter / max(union, 1)

    matched_gt = set()
    matched_pred = set()
    matches = []
    while True:
        gi, pi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        v = iou_mat[gi, pi]
        if v < iou_threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            iou_mat[gi, pi] = 0
            continue
        matched_gt.add(int(gi))
        matched_pred.add(int(pi))
        matches.append(float(v))
        iou_mat[gi, pi] = 0

    tp = len(matches)
    fn = len(gt_masks) - tp
    fp = len(pred_masks) - tp
    return tp, fp, fn, matches


def match_at_level(gt_masks: list, pred_masks: list,
                    iou_threshold: float = 0.5):
    """Greedy IoU matching of pixel-level pred CC masks against pixel-level
    GT instance masks. Returns (tp, fp, fn, ious).
    """
    if not gt_masks and not pred_masks:
        return 0, 0, 0, []
    if not gt_masks:
        return 0, len(pred_masks), 0, []
    if not pred_masks:
        return 0, 0, len(gt_masks), []

    iou_mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for gi, gm in enumerate(gt_masks):
        for pi, pm in enumerate(pred_masks):
            inter = int(np.logical_and(gm, pm).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(gm, pm).sum())
            iou_mat[gi, pi] = inter / max(union, 1)

    matched_gt = set(); matched_pred = set(); matches = []
    while True:
        gi, pi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        v = iou_mat[gi, pi]
        if v < iou_threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            iou_mat[gi, pi] = 0
            continue
        matched_gt.add(int(gi)); matched_pred.add(int(pi))
        matches.append(float(v))
        iou_mat[gi, pi] = 0
    tp = len(matches)
    fn = len(gt_masks) - tp
    fp = len(pred_masks) - tp
    return tp, fp, fn, matches


def load_gt_inst_downsampled(short_id: str, match_level: int):
    """Read GT instance mask + class map, downsample to match_level via stride.

    Returns (gt_inst_l: int32 array, id_to_class: dict[int, int]) or
    (None, None) if no GT mask exists for this slide.
    """
    inst_tif = INST_DIR / f"{short_id}_instance_mask.tif"
    inst_json = INST_DIR / f"{short_id}_instance_map.json"
    if not inst_tif.exists() or not inst_json.exists():
        return None, None
    inst_map = json.loads(inst_json.read_text())
    id_to_class = {}
    for k, v in inst_map.items():
        cls = v["class"].lower()
        if cls.startswith("gc"):
            id_to_class[int(k)] = 2
        elif cls.startswith("tls"):
            id_to_class[int(k)] = 1

    mask_l0 = tifffile.imread(inst_tif, level=0)  # uint16
    stride = 1 << match_level  # 2^level
    if stride == 1:
        return mask_l0.astype(np.int32), id_to_class
    return mask_l0[::stride, ::stride].astype(np.int32), id_to_class


def build_pred_downsampled(pred_tiles: dict, coords_np: np.ndarray,
                            H_l0: int, W_l0: int, match_level: int) -> np.ndarray:
    """Stitch per-patch pred tiles (256x256 at L0) into a downsampled
    full-slide prediction mask at match_level.
    """
    stride = 1 << match_level
    H = (H_l0 + stride - 1) // stride
    W = (W_l0 + stride - 1) // stride
    pred = np.zeros((H, W), dtype=np.uint8)
    tile_l = PATCH // stride  # tile size at match level
    for pi, tile in pred_tiles.items():
        x_l0 = int(coords_np[pi, 0]); y_l0 = int(coords_np[pi, 1])
        y_l = y_l0 // stride; x_l = x_l0 // stride
        # Downsample the tile by stride
        tile_ds = tile[::stride, ::stride] if stride > 1 else tile
        # Clip
        h_use = min(tile_l, H - y_l)
        w_use = min(tile_l, W - x_l)
        if h_use <= 0 or w_use <= 0:
            continue
        pred[y_l:y_l + h_use, x_l:x_l + w_use] = tile_ds[:h_use, :w_use]
    return pred


def load_instance_cache(cache_dir: Path, short_id: str, H: int, W: int):
    p = cache_dir / f"{short_id}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    patch_y = d["patch_y"]; patch_x = d["patch_x"]
    # Cache may have coords beyond cascade's grid bounds — size to fit
    # whichever is bigger, then crop to (H, W) below in caller.
    if patch_y.size:
        H_eff = max(H, int(patch_y.max()) + 1)
        W_eff = max(W, int(patch_x.max()) + 1)
    else:
        H_eff, W_eff = H, W
    inst_grid = np.zeros((H_eff, W_eff), dtype=np.int32)
    cls_grid = np.zeros((H_eff, W_eff), dtype=np.uint8)
    if patch_y.size:
        inst_grid[patch_y, patch_x] = d["dominant_instance_id"]
        cls_grid[patch_y, patch_x] = d["dominant_class"]
    # Pad/crop to (H, W) to match cascade grid
    if (H_eff, W_eff) != (H, W):
        pad_h = max(0, H - H_eff); pad_w = max(0, W - W_eff)
        inst_grid = np.pad(inst_grid, ((0, pad_h), (0, pad_w)))[:H, :W]
        cls_grid = np.pad(cls_grid, ((0, pad_h), (0, pad_w)))[:H, :W]
    return {
        "inst_grid": inst_grid,
        "cls_grid": cls_grid,
        "instance_id": d["instance_id"],
        "instance_class": d["instance_class"],
        "instance_area_patches": d["instance_area_patches"],
    }


@hydra.main(version_base=None, config_path="configs/cascade", config_name="config")
def main(cfg: DictConfig):
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    cache_dir = Path(cfg.get("instance_cache_dir",
                              "/home/ubuntu/local_data/instance_patch_cache"))
    iou_thr = float(cfg.get("iou_threshold", 0.5))
    match_level = int(cfg.get("match_level", 2))  # WSI pyramid level for IoU matching
    print(f"match_level = {match_level} (stride {1 << match_level} from L0)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading cascade...")
    stage1 = load_stage1(str(cfg.stage1), device)
    stage2 = load_stage2_region(str(cfg.stage2_region), device)

    entries = ps.build_slide_entries()
    fold_idx = int(cfg.fold_idx)
    folds_pair, _ = ps.create_splits(entries, k_folds=int(cfg.k_folds), seed=int(cfg.seed))
    val_entries = folds_pair[fold_idx]
    if cfg.get("limit_slides"):
        val_entries = val_entries[: int(cfg.limit_slides)]
    print(f"Fold {fold_idx}: {len(val_entries)} slides")

    thr_field = cfg.thresholds
    if hasattr(thr_field, "__iter__") and not isinstance(thr_field, (str, bytes)):
        threshold = float(list(thr_field)[0])
    else:
        threshold = float(thr_field)
    min_size = int(cfg.get("min_component_size", 2))
    closing_it = int(cfg.get("closing_iters", 1))

    metrics = {}
    per_slide = []

    t0 = time.time()
    for i, entry in enumerate(val_entries):
        short_id = entry["slide_id"].split(".")[0]
        wsi_p = slide_wsi_path(entry)
        if not wsi_p or not Path(wsi_p).exists():
            continue

        feats_np, coords_np, edge_np = _load_features_and_graph(entry["zarr_path"])
        features = torch.from_numpy(feats_np)
        coords = torch.from_numpy(coords_np).long()
        edge_index = torch.from_numpy(edge_np).long().to(device)
        try:
            grid_class, _, pred_tiles = cascade_one_slide_region(
                stage1, stage2, features, coords, edge_index,
                threshold=threshold, s2_batch=8, device=device,
                wsi_path=wsi_p,
                min_component_size=min_size, closing_iters=closing_it,
                return_cell_probs=False,
            )
        except Exception as e:
            print(f"  {short_id}: cascade fail: {e}")
            continue

        # Slide level-0 dims from WSI tif
        try:
            with tifffile.TiffFile(wsi_p) as tf:
                H_l0, W_l0 = tf.pages[0].shape[:2]
        except Exception:
            H_l0 = (grid_class.shape[0] + 1) * PATCH
            W_l0 = (grid_class.shape[1] + 1) * PATCH

        # Build pred mask at match_level by downsampling pred_tiles
        pred_l = build_pred_downsampled(pred_tiles, coords_np, H_l0, W_l0,
                                         match_level)

        # Load GT instance mask at match_level (None if GT-negative slide)
        gt_inst_l, gt_id_to_cls = load_gt_inst_downsampled(short_id, match_level)
        if gt_inst_l is None:
            gt_inst_l = np.zeros_like(pred_l, dtype=np.int32)
            gt_id_to_cls = {}
        # Align shapes (GT might differ slightly from pred due to slide
        # dim rounding)
        H_m = min(pred_l.shape[0], gt_inst_l.shape[0])
        W_m = min(pred_l.shape[1], gt_inst_l.shape[1])
        pred_l = pred_l[:H_m, :W_m]
        gt_inst_l = gt_inst_l[:H_m, :W_m]

        ct = entry["cancer_type"]
        bucket = metrics.setdefault(ct, {"tls": [0, 0, 0], "gc": [0, 0, 0]})

        for cls_name, target_cls in (("tls", 1), ("gc", 2)):
            # GT instance masks of this class
            gt_inst_ids_cls = [iid for iid, cls in gt_id_to_cls.items()
                                if cls == target_cls]
            gt_masks = []
            for gid in gt_inst_ids_cls:
                gm = (gt_inst_l == gid)
                if gm.any():
                    gt_masks.append(gm)

            # Predicted instances = CCs of (pred_l == target_cls)
            pred_lbl, n_pred = ndimage.label(pred_l == target_cls)
            pred_masks = [pred_lbl == pid for pid in range(1, n_pred + 1)]

            tp, fp, fn, _ious = match_at_level(gt_masks, pred_masks,
                                                iou_threshold=iou_thr)
            bucket[cls_name][0] += tp
            bucket[cls_name][1] += fp
            bucket[cls_name][2] += fn
            per_slide.append({
                "slide_id": short_id, "cancer_type": ct, "class": cls_name,
                "tp": tp, "fp": fp, "fn": fn,
                "n_gt_inst": len(gt_masks),
                "n_pred_inst": n_pred,
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(val_entries) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(val_entries)}] {short_id}: "
                  f"{rate:.2f}/s, ETA {eta:.1f}min")

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1v = 2 * p * r / max(p + r, 1e-9)
        return p, r, f1v

    overall = {"tls": [0, 0, 0], "gc": [0, 0, 0]}
    for ct, vals in metrics.items():
        for cls_name in ("tls", "gc"):
            for k in range(3):
                overall[cls_name][k] += vals[cls_name][k]

    print(f"\n========== Results (IoU >= {iou_thr:.2f}) ==========")
    print(f"{'cohort':10s}  {'class':4s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  "
          f"{'P':>6s}  {'R':>6s}  {'F1':>6s}")
    for ct, vals in sorted(metrics.items()):
        for cls_name in ("tls", "gc"):
            tp, fp, fn = vals[cls_name]
            p, r, f1v = prf(tp, fp, fn)
            print(f"{ct:10s}  {cls_name:4s}  {tp:>5d}  {fp:>5d}  {fn:>5d}  "
                  f"{p:>6.3f}  {r:>6.3f}  {f1v:>6.3f}")
    print()
    for cls_name in ("tls", "gc"):
        tp, fp, fn = overall[cls_name]
        p, r, f1v = prf(tp, fp, fn)
        print(f"OVERALL    {cls_name:4s}  {tp:>5d}  {fp:>5d}  {fn:>5d}  "
              f"{p:>6.3f}  {r:>6.3f}  {f1v:>6.3f}")

    summary = {
        "fold_idx": fold_idx, "k_folds": int(cfg.k_folds),
        "threshold": threshold, "iou_threshold": iou_thr,
        "min_size": min_size, "closing_iters": closing_it,
        "per_cancer": {ct: {cls: {"tp": v[cls][0], "fp": v[cls][1], "fn": v[cls][2],
                                   "p": prf(*v[cls])[0], "r": prf(*v[cls])[1],
                                   "f1": prf(*v[cls])[2]}
                              for cls in ("tls", "gc")}
                       for ct, v in metrics.items()},
        "overall": {cls: {"tp": overall[cls][0], "fp": overall[cls][1], "fn": overall[cls][2],
                          "p": prf(*overall[cls])[0], "r": prf(*overall[cls])[1],
                          "f1": prf(*overall[cls])[2]} for cls in ("tls", "gc")},
    }
    (out_dir / "instance_f1.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_slide.json").write_text(json.dumps(per_slide, indent=2))
    print(f"\nWrote {out_dir}/instance_f1.json")


if __name__ == "__main__":
    main()
