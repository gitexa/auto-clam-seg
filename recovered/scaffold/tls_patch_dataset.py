"""Stage 2 data loader: build the in-memory TLS-patch dataset.

Reproduces the recovered output.log line:
    "Building TLS patch dataset (pre-loading features + masks)..."
    "Loaded 170 test patients to exclude"
    "Found 767 masks on local SSD"
    "Total: 605 slides, 26,364 TLS patches (1,869 with GC)"
    "Memory: 162MB features + 1728MB masks = 1890MB total"

Strategy:
  1. Stage masks to local SSD on first run (~767 small TIFs).
     The HookNet TIFs live on NFS by default; copying once to `/dev/vda1`
     gives ~10× faster per-patch crop reads during dataset build.
  2. Walk slides in parallel: per slide, open the zarr (features +
     coords) and the mask TIF, crop a 256×256 mask tile per patch from
     the highest-resolution mask page. Keep patches where the tile has
     any TLS or GC pixel.
  3. Save the assembled (features, masks, slide_ids, patch_idx_in_slide)
     to a single .pt cache. Subsequent runs skip directly to load.

Output shapes (single .pt cache):
    features:    (N, 1536)        float32
    masks:       (N, 256, 256)    uint8 in {0, 1, 2}
    slide_ids:   (N,)             list of TCGA slide UUIDs
    patch_idx:   (N,)             int32 — index of each patch within its slide
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
import zarr

sys.path.insert(0, "/home/ubuntu/profile-clam")
import prepare_segmentation as ps  # noqa: E402

DEFAULT_LOCAL_MASK_DIR = "/home/ubuntu/local_data/hooknet_masks_tls_gc"
DEFAULT_CACHE_PATH = "/home/ubuntu/local_data/tls_patch_dataset.pt"
PATCH_SIZE = 256


# ─── Mask staging to local SSD ─────────────────────────────────────────


def stage_masks_to_local_ssd(
    src_dir: str = ps.MASK_DIR,
    dst_dir: str = DEFAULT_LOCAL_MASK_DIR,
    verbose: bool = True,
) -> str:
    """Copy mask TIFs from NFS to local SSD on first run; idempotent.

    Returns the local mask directory path.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    src_tifs = sorted(src.glob("*.tif"))
    n_copied = 0
    n_skipped = 0
    for src_tif in src_tifs:
        dst_tif = dst / src_tif.name
        if dst_tif.exists() and dst_tif.stat().st_size == src_tif.stat().st_size:
            n_skipped += 1
            continue
        shutil.copy2(src_tif, dst_tif)
        n_copied += 1
    if verbose:
        print(f"  Found {len(src_tifs)} masks on local SSD "
              f"({n_copied} newly copied, {n_skipped} reused) at {dst}")
    return str(dst)


# ─── Test-patient exclusion ────────────────────────────────────────────


def load_test_patient_ids(test_csv: str = ps.TEST_CSV) -> set[str]:
    """Recovered log: 'Loaded 170 test patients to exclude'."""
    df = pd.read_csv(test_csv)
    # The metadata CSV has either a `patient_id` or a `case_id` column;
    # both spellings appear in the codebase.
    for col in ("case_submitter_id", "patient_id", "case_id"):
        if col in df.columns:
            return set(df[col].astype(str).tolist())
    raise KeyError(f"No patient/case column in {test_csv} (have: {list(df.columns)})")


def patient_id_from_slide(slide_id: str) -> str:
    """TCGA slide_id starts with `TCGA-XX-XXXX-...`; first 12 chars are the
    case_submitter_id (TCGA-XX-XXXX)."""
    return slide_id[:12]


# ─── Per-slide patch extraction (worker) ───────────────────────────────


def _process_one_slide(args: tuple[dict, str]) -> dict[str, Any] | None:
    """Worker: open one slide's zarr + mask, return TLS-positive patches.

    Each TLS-positive patch contributes one (feature, 256×256 mask tile).
    """
    entry, local_mask_dir = args
    if entry["mask_path"] is None:
        return None
    short_id = entry["slide_id"].split(".")[0]
    local_tif = Path(local_mask_dir) / Path(entry["mask_path"]).name
    mask_path = str(local_tif) if local_tif.exists() else entry["mask_path"]

    grp = zarr.open(entry["zarr_path"], mode="r")
    features = np.asarray(grp["features"][:], dtype=np.float32)         # (N, 1536)
    coords = np.asarray(grp["coords"][:], dtype=np.float32)             # (N, 2)
    coords_i = coords.astype(np.int64)
    n = features.shape[0]

    # Memory-efficient mask access: open the level-0 TIF page as a zarr
    # view so per-patch crops are windowed reads (≪ 1 MB each) instead
    # of loading the whole 3-7 GB page.
    tif = tifffile.TiffFile(mask_path)
    full_h, full_w = tif.pages[0].shape[:2]
    z0 = zarr.open(tifffile.imread(mask_path, aszarr=True, level=0), mode="r")

    # Slide coords vs mask coords: usually 1:1 (level-0 alignment), but
    # if the mask TIF is at a coarser MPP we scale down.
    coord_max_x = float(coords[:, 0].max()) + PATCH_SIZE
    coord_max_y = float(coords[:, 1].max()) + PATCH_SIZE
    scale = max(1.0, coord_max_x / full_w, coord_max_y / full_h)
    if scale > 1.0:
        coords_m = (coords / scale).astype(np.int64)
        psize_m = max(1, int(PATCH_SIZE / scale))
    else:
        coords_m = coords_i
        psize_m = PATCH_SIZE

    feats_kept: list[np.ndarray] = []
    masks_kept: list[np.ndarray] = []
    idx_kept: list[int] = []
    for i in range(n):
        x, y = int(coords_m[i, 0]), int(coords_m[i, 1])
        x1 = min(x + psize_m, full_w)
        y1 = min(y + psize_m, full_h)
        if x1 <= x or y1 <= y:
            continue
        tile = np.asarray(z0[y:y1, x:x1])  # windowed read
        if tile.max() == 0:
            continue
        if tile.shape != (PATCH_SIZE, PATCH_SIZE):
            t = torch.from_numpy(tile.astype(np.uint8)).unsqueeze(0).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(
                t, size=(PATCH_SIZE, PATCH_SIZE), mode="nearest"
            )
            tile = t.squeeze().to(torch.uint8).numpy()
        feats_kept.append(features[i])
        masks_kept.append(tile.astype(np.uint8))
        idx_kept.append(i)

    if not feats_kept:
        return None
    return {
        "slide_id": entry["slide_id"],
        "cancer_type": entry["cancer_type"],
        "features": np.stack(feats_kept),                # (k, 1536)
        "masks": np.stack(masks_kept),                   # (k, 256, 256)
        "patch_idx": np.array(idx_kept, dtype=np.int32), # (k,)
        "n_total": n,
        "n_with_gc": int(sum(int((m == 2).any()) for m in masks_kept)),
    }


# ─── Main entry point ──────────────────────────────────────────────────


def build_tls_patch_dataset(
    cache_path: str = DEFAULT_CACHE_PATH,
    local_mask_dir: str = DEFAULT_LOCAL_MASK_DIR,
    n_workers: int = 8,
    skip_test_patients: bool = True,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict[str, torch.Tensor | list]:
    """Build (or load from cache) the in-memory TLS-patch dataset.

    Returns a dict with keys: features, masks, slide_ids, cancer_types,
    patch_idx. Save it as a single .pt to skip the slow build next time.
    """
    cache_p = Path(cache_path)
    if cache_p.exists() and not rebuild:
        if verbose:
            print(f"Loading TLS-patch cache from {cache_p} "
                  f"({cache_p.stat().st_size / 1e9:.2f} GB)")
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    if verbose:
        print("Building TLS patch dataset (pre-loading features + masks)...")
    if skip_test_patients:
        test_patients = load_test_patient_ids()
        if verbose:
            print(f"  Loaded {len(test_patients)} test patients to exclude")
    else:
        test_patients = set()

    local_mask_dir = stage_masks_to_local_ssd(dst_dir=local_mask_dir, verbose=verbose)

    entries = ps.build_slide_entries()
    entries = [
        e for e in entries
        if e.get("mask_path") and e.get("zarr_path")  # both non-empty
        and patient_id_from_slide(e["slide_id"]) not in test_patients
    ]
    if verbose:
        print(f"  {len(entries)} slides with both mask and features")

    args_list = [(e, local_mask_dir) for e in entries]
    results: list[dict[str, Any]] = []
    n_done = 0
    n_total_patches = 0
    n_gc_patches = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_process_one_slide, a) for a in args_list]
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if res is None:
                continue
            results.append(res)
            n_total_patches += res["features"].shape[0]
            n_gc_patches += res["n_with_gc"]
            if verbose and n_done % 100 == 0:
                print(f"  [{n_done}] {n_total_patches:,} TLS patches "
                      f"({n_gc_patches:,} with GC)")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Built in {elapsed:.0f}s "
              f"({elapsed / max(1, len(entries)):.1f}s/slide)")
        print(f"Total: {len(results)} slides, "
              f"{n_total_patches:,} TLS patches ({n_gc_patches:,} with GC)")

    # Concatenate.
    features = np.concatenate([r["features"] for r in results], axis=0)
    masks = np.concatenate([r["masks"] for r in results], axis=0)
    slide_ids: list[str] = []
    cancer_types: list[str] = []
    patch_idx: list[int] = []
    for r in results:
        slide_ids.extend([r["slide_id"]] * r["features"].shape[0])
        cancer_types.extend([r["cancer_type"]] * r["features"].shape[0])
        patch_idx.extend(r["patch_idx"].tolist())

    bundle = {
        "features": torch.from_numpy(features),                    # float32
        "masks": torch.from_numpy(masks),                          # uint8
        "slide_ids": slide_ids,
        "cancer_types": cancer_types,
        "patch_idx": torch.tensor(patch_idx, dtype=torch.int32),
    }
    if verbose:
        print(f"Memory: {features.nbytes / 1e6:.0f}MB features + "
              f"{masks.nbytes / 1e6:.0f}MB masks = "
              f"{(features.nbytes + masks.nbytes) / 1e6:.0f}MB total")

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_p)
    if verbose:
        print(f"  Saved cache to {cache_p}")
    return bundle


if __name__ == "__main__":
    # Run once to build the cache.
    bundle = build_tls_patch_dataset()
    print(f"\nDone. features={tuple(bundle['features'].shape)} "
          f"masks={tuple(bundle['masks'].shape)}")
