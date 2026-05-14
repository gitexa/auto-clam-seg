"""Compute instance-F1 @ IoU>=0.5 for HookNet-TLS outputs on our fold-0
val slides. Matches the methodology used in eval_instance_f1.py for our
cascade (match at WSI level-3, ~8 µm/px).

For each slide that has a completed HookNet output:
  1. Load filtered TLS polygons + GC polygons from
     <out_dir>/<slide_id>/images/filtered/<slide>_hooknettls_*_filtered.json
  2. Rasterize polygons at level-3 (stride=8 in level-0 coords) per class.
  3. Read GT instance mask + map at level-3.
  4. For each class c in {TLS=1, GC=2}: extract pred CCs of c, IoU-match
     against GT instances of c, compute TP/FP/FN.
  5. Aggregate per-cohort and overall.

Run:
    python eval_hooknet_instance_f1.py \\
        --hooknet-out /home/ubuntu/local_data/hooknet_out \\
        --fold 0 --match-level 3 --iou 0.5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")

import prepare_segmentation as ps  # noqa: E402


INST_DIR = Path("/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/masks_instance")


def rasterize_polygons_to_level(polygons: list, slide_h_l0: int, slide_w_l0: int,
                                  match_level: int) -> np.ndarray:
    """Render filled polygons (level-0 coords) onto a level-`match_level`
    binary mask via PIL.
    """
    from PIL import Image, ImageDraw
    stride = 1 << match_level
    H = (slide_h_l0 + stride - 1) // stride
    W = (slide_w_l0 + stride - 1) // stride
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for p in polygons:
        coords = p.get("coordinates", [])
        if not coords:
            continue
        # Convert level-0 (x, y) to level-N (x, y); PIL expects tuples
        pts = [(c[0] / stride, c[1] / stride) for c in coords]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
    return np.asarray(img, dtype=np.uint8)


def load_gt_inst_downsampled(short_id: str, match_level: int):
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
    mask_l0 = tifffile.imread(inst_tif, level=0)
    stride = 1 << match_level
    if stride == 1:
        return mask_l0.astype(np.int32), id_to_class
    return mask_l0[::stride, ::stride].astype(np.int32), id_to_class


def slide_dims_l0(short_id: str, hooknet_out: Path) -> tuple[int, int] | None:
    """Get level-0 dims for the slide from its preprocessed pyramid TIF
    (which we save as <out>/<sid>.tif then delete; instead use the
    HookNet prediction mask which has same dims).
    """
    pred = hooknet_out / short_id / "images" / f"{short_id}_hooknettls.tif"
    if pred.exists():
        with tifffile.TiffFile(pred) as tf:
            return tf.pages[0].shape[:2]
    # Else fall back to GT instance mask
    inst = INST_DIR / f"{short_id}_instance_mask.tif"
    if inst.exists():
        with tifffile.TiffFile(inst) as tf:
            return tf.pages[0].shape[:2]
    return None


def match_at_level(gt_masks, pred_masks, iou_threshold=0.5):
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
    return tp, len(pred_masks) - tp, len(gt_masks) - tp, matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hooknet-out", type=Path, default=Path("/home/ubuntu/local_data/hooknet_out"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--match-level", type=int, default=3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("/home/ubuntu/auto-clam-seg/hooknet_instance_f1.json"))
    args = ap.parse_args()

    # Resolve fold-0 val slides
    entries = ps.build_slide_entries()
    folds, _ = ps.create_splits(entries, k_folds=5, seed=42)
    val_entries = folds[args.fold]
    print(f"fold {args.fold}: {len(val_entries)} slides")

    metrics = {}
    per_slide = []
    t0 = time.time()
    n_done = 0; n_skip = 0
    for i, entry in enumerate(val_entries):
        short = entry["slide_id"].split(".")[0]
        ct = entry["cancer_type"]
        # Skip if HookNet output not ready
        pp_xml = args.hooknet_out / short / "images" / "post-processed" / f"{short}_hooknettls_post_processed.xml"
        if not pp_xml.exists():
            n_skip += 1
            continue
        # Load filtered polygons
        tls_p = args.hooknet_out / short / "images" / "filtered" / f"{short}_hooknettls_tls_filtered.json"
        gc_p = args.hooknet_out / short / "images" / "filtered" / f"{short}_hooknettls_gc_filtered.json"
        try:
            tls_polys = json.loads(tls_p.read_text()) if tls_p.exists() else []
            gc_polys = json.loads(gc_p.read_text()) if gc_p.exists() else []
        except Exception as e:
            print(f"  {short}: load fail {e}")
            continue

        # Slide level-0 dims
        dims = slide_dims_l0(short, args.hooknet_out)
        if dims is None:
            print(f"  {short}: dims fail")
            continue
        H_l0, W_l0 = dims

        # GT
        gt_inst_l, gt_id_to_cls = load_gt_inst_downsampled(short, args.match_level)
        if gt_inst_l is None:
            gt_inst_l = np.zeros((1, 1), dtype=np.int32); gt_id_to_cls = {}

        bucket = metrics.setdefault(ct, {"tls": [0, 0, 0], "gc": [0, 0, 0]})

        for cls_name, target_cls, polys in (("tls", 1, tls_polys), ("gc", 2, gc_polys)):
            # Rasterize HookNet polygons
            pred_mask = rasterize_polygons_to_level(polys, H_l0, W_l0, args.match_level)
            # Align with GT
            H_m = min(pred_mask.shape[0], gt_inst_l.shape[0])
            W_m = min(pred_mask.shape[1], gt_inst_l.shape[1])
            pred_mask = pred_mask[:H_m, :W_m]
            gt_l = gt_inst_l[:H_m, :W_m]
            # Predicted CCs
            pred_lbl, n_pred = ndimage.label(pred_mask > 0)
            pred_masks = [pred_lbl == pid for pid in range(1, n_pred + 1)]
            # GT instances of this class
            gt_masks = []
            for gid, cls in gt_id_to_cls.items():
                if cls == target_cls:
                    gm = (gt_l == gid)
                    if gm.any():
                        gt_masks.append(gm)
            tp, fp, fn, _ = match_at_level(gt_masks, pred_masks, iou_threshold=args.iou)
            bucket[cls_name][0] += tp
            bucket[cls_name][1] += fp
            bucket[cls_name][2] += fn
            per_slide.append({
                "slide_id": short, "cancer_type": ct, "class": cls_name,
                "tp": tp, "fp": fp, "fn": fn,
                "n_gt_inst": len(gt_masks),
                "n_pred_inst": n_pred,
            })
        n_done += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(val_entries)}] done={n_done} skip={n_skip} "
                  f"({(time.time()-t0)/max(n_done,1):.1f}s/slide)")

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return p, r, f1

    overall = {"tls": [0, 0, 0], "gc": [0, 0, 0]}
    for ct, vals in metrics.items():
        for cls_name in ("tls", "gc"):
            for k in range(3):
                overall[cls_name][k] += vals[cls_name][k]

    print(f"\n=== HookNet instance F1 (IoU >= {args.iou}, match L{args.match_level}) ===")
    print(f"slides with HookNet output: {n_done}, skipped (not ready): {n_skip}")
    print(f"{'cohort':10s}  {'class':4s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  "
          f"{'P':>6s}  {'R':>6s}  {'F1':>6s}")
    for ct, vals in sorted(metrics.items()):
        for cls_name in ("tls", "gc"):
            tp, fp, fn = vals[cls_name]
            p, r, f1 = prf(tp, fp, fn)
            print(f"{ct:10s}  {cls_name:4s}  {tp:>5d}  {fp:>5d}  {fn:>5d}  "
                  f"{p:>6.3f}  {r:>6.3f}  {f1:>6.3f}")
    print()
    for cls_name in ("tls", "gc"):
        tp, fp, fn = overall[cls_name]
        p, r, f1 = prf(tp, fp, fn)
        print(f"OVERALL    {cls_name:4s}  {tp:>5d}  {fp:>5d}  {fn:>5d}  "
              f"{p:>6.3f}  {r:>6.3f}  {f1:>6.3f}")

    summary = {
        "fold": args.fold, "match_level": args.match_level, "iou_threshold": args.iou,
        "n_slides_evaluated": n_done, "n_skipped": n_skip,
        "per_cancer": {ct: {cls: {"tp": v[cls][0], "fp": v[cls][1], "fn": v[cls][2],
                                   "p": prf(*v[cls])[0], "r": prf(*v[cls])[1],
                                   "f1": prf(*v[cls])[2]}
                              for cls in ("tls", "gc")}
                       for ct, v in metrics.items()},
        "overall": {cls: {"tp": overall[cls][0], "fp": overall[cls][1], "fn": overall[cls][2],
                          "p": prf(*overall[cls])[0], "r": prf(*overall[cls])[1],
                          "f1": prf(*overall[cls])[2]} for cls in ("tls", "gc")},
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
