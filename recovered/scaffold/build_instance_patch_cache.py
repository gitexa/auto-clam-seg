"""Pre-compute per-slide instance-at-patch-grid cache.

Reads HookNet's per-slide `<slide>_instance_mask.tif` (level-0 uint16) +
`<slide>_instance_map.json` (instance_id → class), and for each
256-px patch in the slide records:

  * dominant_instance_id: which GT instance has the most pixels in this patch
    (0 = no instance / bg-only)
  * dominant_class:       0 = bg, 1 = TLS, 2 = GC
  * patch_y, patch_x:     grid indices (y // 256, x // 256)

Output (per slide): /home/ubuntu/local_data/instance_patch_cache/<short_id>.npz
Cache is fast to load and small (~tens of KB per slide).

Run:
    python build_instance_patch_cache.py [--limit N]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr

sys.path.insert(0, "/home/ubuntu/auto-clam-seg/recovered/scaffold")
sys.path.insert(0, "/home/ubuntu/profile-clam")

import prepare_segmentation as ps  # noqa: E402

PATCH = 256
INST_DIR = Path("/home/ubuntu/ahaas-persistent-std-tcga/data/metadata/annotations/hooknet/masks_instance")
OUT_DIR = Path("/home/ubuntu/local_data/instance_patch_cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def class_to_id(cls: str) -> int:
    if cls.lower().startswith("gc"):
        return 2
    if cls.lower().startswith("tls"):
        return 1
    return 0


def build_one(entry: dict) -> bool:
    short_id = entry["slide_id"].split(".")[0]
    out_path = OUT_DIR / f"{short_id}.npz"
    if out_path.exists():
        return True

    inst_tif = INST_DIR / f"{short_id}_instance_mask.tif"
    inst_json = INST_DIR / f"{short_id}_instance_map.json"
    if not inst_tif.exists() or not inst_json.exists():
        # Slide is GT-negative — write an "empty" cache so the eval can detect it
        np.savez_compressed(out_path,
                            patch_y=np.zeros(0, dtype=np.int32),
                            patch_x=np.zeros(0, dtype=np.int32),
                            dominant_instance_id=np.zeros(0, dtype=np.int32),
                            dominant_class=np.zeros(0, dtype=np.uint8),
                            instance_id=np.zeros(0, dtype=np.int32),
                            instance_class=np.zeros(0, dtype=np.uint8),
                            instance_area_patches=np.zeros(0, dtype=np.int32))
        return True

    inst_map = json.loads(inst_json.read_text())
    id_to_class = {int(k): class_to_id(v["class"]) for k, v in inst_map.items()}

    # Load level-0 instance mask
    mask = tifffile.imread(inst_tif, level=0)  # uint16 (~5 GB)
    H, W = mask.shape

    # Read slide's patch coords from the zarr
    grp = zarr.open(entry["zarr_path"], mode="r")
    coords = np.asarray(grp["coords"][:])  # (N, 2) level-0 (x, y)

    n = coords.shape[0]
    dominant_inst = np.zeros(n, dtype=np.int32)
    dominant_cls = np.zeros(n, dtype=np.uint8)
    patch_y = (coords[:, 1] // PATCH).astype(np.int32)
    patch_x = (coords[:, 0] // PATCH).astype(np.int32)

    for i in range(n):
        y0 = int(coords[i, 1]); x0 = int(coords[i, 0])
        if y0 >= H or x0 >= W:
            continue
        y1 = min(H, y0 + PATCH); x1 = min(W, x0 + PATCH)
        tile = mask[y0:y1, x0:x1]
        if tile.size == 0:
            continue
        nz = tile[tile > 0]
        if nz.size == 0:
            continue
        # Majority instance ID in this patch (over non-zero pixels)
        ids, counts = np.unique(nz, return_counts=True)
        winner = int(ids[counts.argmax()])
        dominant_inst[i] = winner
        dominant_cls[i] = id_to_class.get(winner, 0)

    # Per-instance summary: list of (id, class, n_patches)
    inst_ids = np.array(sorted(id_to_class.keys()), dtype=np.int32)
    inst_classes = np.array([id_to_class[i] for i in inst_ids], dtype=np.uint8)
    inst_area_patches = np.array(
        [int((dominant_inst == i).sum()) for i in inst_ids],
        dtype=np.int32,
    )

    np.savez_compressed(
        out_path,
        patch_y=patch_y,
        patch_x=patch_x,
        dominant_instance_id=dominant_inst,
        dominant_class=dominant_cls,
        instance_id=inst_ids,
        instance_class=inst_classes,
        instance_area_patches=inst_area_patches,
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    entries = ps.build_slide_entries()
    if args.limit:
        entries = entries[args.start:args.start + args.limit]
    print(f"building cache for {len(entries)} slides")
    t0 = time.time()
    done = 0
    failed = 0
    for i, e in enumerate(entries):
        try:
            build_one(e)
            done += 1
        except Exception as ex:
            print(f"  [{i+1}/{len(entries)}] {e['slide_id'].split('.')[0]} FAILED: {ex}")
            failed += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(entries) - i - 1) / rate
            print(f"  [{i+1}/{len(entries)}] done={done} failed={failed} "
                  f"({rate:.2f}/s, ETA {eta/60:.1f} min)")
    print(f"DONE: {done} cached, {failed} failed, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
