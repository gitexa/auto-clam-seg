"""3×3 neighborhood tiles for v3.36 NeighborhoodPixelDecoder.

Each sample is a 3×3 window of patches (raster row-major, cells 0..8)
centered on a TLS-positive patch (or stride-2 raster across larger
clusters of positives). For each window we yield:

    features:   (9, 1536) float32 — UNI-v2 per cell, zeros for invalid
    mask:       (768, 768) uint8  — stitched native masks; 255 in
                                     invalid cells (CE ignore_index)
    valid_mask: (9,) bool         — 0 where the cell has no patch on
                                     the slide grid

Uses the existing `tls_patch_dataset_min4096.pt` bundle as the source of
TLS-positive masks. Non-positive neighbor cells get zeros for the mask
(matches v3.4b's "patch is bg if mask>0 < min_tls_pixels" labeling
assumption — small label noise is accepted; same as v3.4b training).

Features for non-positive neighbor cells are read lazily from the
slide's zarr, with a per-worker LRU cache (max 8 slides → ~800 MB
worst case worker memory).
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from scipy.ndimage import label as scipy_label
from torch.utils.data import Dataset

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_segmentation as ps  # noqa: E402

PATCH_SIZE = 256
INVALID_MASK_VALUE = 255  # used as CE ignore_index in the trainer


# ─── per-slide grid bookkeeping ───────────────────────────────────────


def _slide_zarr_path(slide_id: str) -> str | None:
    """Locate the per-slide zarr (UNI features) on local SSD; fall back
    to the prepare_segmentation entry list."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stage_features_to_local import local_zarr_dirs
    if all(Path(p).is_dir() for p in local_zarr_dirs().values()):
        ps.ZARR_DIRS = local_zarr_dirs()
    entries = ps.build_slide_entries()
    for e in entries:
        if e["slide_id"].split(".")[0] == slide_id.split(".")[0]:
            return e.get("zarr_path")
    return None


def _build_slide_grid(coords: np.ndarray):
    grid_x = (coords[:, 0] // PATCH_SIZE).astype(np.int64)
    grid_y = (coords[:, 1] // PATCH_SIZE).astype(np.int64)
    gx_min, gy_min = int(grid_x.min()), int(grid_y.min())
    grid_x -= gx_min; grid_y -= gy_min
    H = int(grid_y.max()) + 1
    W = int(grid_x.max()) + 1
    coord_to_idx: dict[tuple[int, int], int] = {
        (int(gy), int(gx)): i for i, (gy, gx) in enumerate(zip(grid_y, grid_x))
    }
    return coord_to_idx, H, W, grid_y, grid_x


# ─── window enumeration per slide ─────────────────────────────────────


def _enumerate_windows(
    pos_grid_cells: list[tuple[int, int]],
    H: int, W: int,
    tile_clusters: bool = True,
    stride: int = 2,
) -> list[tuple[int, int]]:
    """Enumerate (top_left_y, top_left_x) of 3×3 windows for one slide.

    1. One window centered on each TLS-positive patch (clipped to bounds).
    2. If `tile_clusters`, additionally raster-scan stride-K windows over
       the bbox of any 4-conn component bigger than 3×3.
    """
    if not pos_grid_cells:
        return []

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []

    def add(top_y: int, top_x: int):
        top_y = max(0, min(H - 3, top_y))
        top_x = max(0, min(W - 3, top_x))
        key = (top_y, top_x)
        if key not in seen:
            seen.add(key); out.append(key)

    # Center one window on each TLS-positive cell.
    for gy, gx in pos_grid_cells:
        add(gy - 1, gx - 1)

    if tile_clusters:
        # Build a binary grid and connected components (4-conn).
        grid = np.zeros((H, W), dtype=np.uint8)
        for gy, gx in pos_grid_cells:
            grid[gy, gx] = 1
        labels, n = scipy_label(grid)
        for c in range(1, n + 1):
            ys, xs = np.where(labels == c)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            if (y1 - y0) <= 3 and (x1 - x0) <= 3:
                continue  # already covered by center-on-positive
            for top_y in range(max(0, y0 - 1), max(y0, y1 - 2) + 1, stride):
                for top_x in range(max(0, x0 - 1), max(x0, x1 - 2) + 1, stride):
                    add(top_y, top_x)
    return out


# ─── dataset ──────────────────────────────────────────────────────────


class NeighborhoodTilesDataset(Dataset):
    """Lazy 3×3 neighborhood dataset over a TLS-positive patch cache.

    Args:
        bundle: dict from `tls_patch_dataset.build_tls_patch_dataset(...)`
        slide_filter: optional set of slide_ids to keep (for train/val split)
        max_windows_per_slide: subsample uniformly if exceeded
        tile_clusters: if True, also tile stride-2 windows over big
            connected components of positives
        stride: stride for cluster tiling
        zarr_lru: per-worker LRU size for slide zarr handles
        seed: rng seed for window subsampling
    """

    def __init__(
        self,
        bundle: dict[str, Any],
        slide_filter: set[str] | None = None,
        max_windows_per_slide: int = 64,
        tile_clusters: bool = True,
        stride: int = 2,
        zarr_lru: int = 8,
        seed: int = 42,
    ):
        self._bundle_features = bundle["features"]      # torch.Tensor (N_pos, 1536)
        self._bundle_masks = bundle["masks"]            # torch.Tensor (N_pos, 256, 256)
        self._bundle_slide_ids = bundle["slide_ids"]    # list[str], len N_pos
        self._bundle_patch_idx = bundle["patch_idx"]    # torch.Tensor (N_pos,)
        self._zarr_lru = zarr_lru
        self._seed = seed

        # Per-slide bookkeeping built once at init from the bundle.
        # cache_rows_by_slide: short_slide_id → list[(cache_row, patch_idx_in_slide)]
        cache_rows_by_slide: dict[str, list[tuple[int, int]]] = {}
        for ci, sid in enumerate(self._bundle_slide_ids):
            short = sid.split(".")[0]
            if slide_filter is not None and short not in slide_filter:
                continue
            cache_rows_by_slide.setdefault(short, []).append(
                (ci, int(self._bundle_patch_idx[ci]))
            )
        # Per-slide zarr coord lookup.
        self._slide_meta: dict[str, dict] = {}
        rng = np.random.default_rng(seed)
        windows: list[tuple[str, int, int]] = []         # (short_id, top_y, top_x)
        for short, rows in cache_rows_by_slide.items():
            zpath = _slide_zarr_path(short)
            if zpath is None:
                continue
            try:
                z = zarr.open(zpath, mode="r")
                coords = np.asarray(z["coords"][:])
            except Exception:
                continue
            coord_to_idx, H, W, grid_y, grid_x = _build_slide_grid(coords)
            # cache_row → grid_cell, for fast positive-cell mask lookup
            cache_row_to_cell: dict[int, tuple[int, int]] = {}
            cell_to_cache_row: dict[tuple[int, int], int] = {}
            for ci, pidx in rows:
                if 0 <= pidx < coords.shape[0]:
                    gy = int(grid_y[pidx]); gx = int(grid_x[pidx])
                    cache_row_to_cell[ci] = (gy, gx)
                    cell_to_cache_row[(gy, gx)] = ci
            self._slide_meta[short] = {
                "zarr_path": zpath,
                "coord_to_idx": coord_to_idx,
                "cell_to_cache_row": cell_to_cache_row,
                "H": H, "W": W,
            }
            pos_cells = list(cell_to_cache_row.keys())
            tops = _enumerate_windows(
                pos_cells, H, W, tile_clusters=tile_clusters, stride=stride
            )
            if max_windows_per_slide and len(tops) > max_windows_per_slide:
                idx = rng.choice(len(tops), size=max_windows_per_slide, replace=False)
                tops = [tops[int(i)] for i in idx]
            windows.extend((short, ty, tx) for (ty, tx) in tops)
        self.windows = windows
        # Per-worker zarr handle cache (lazy, opened on first access).
        self._zarr_handles: OrderedDict[str, zarr.Array] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _get_zarr(self, short: str):
        if short in self._zarr_handles:
            self._zarr_handles.move_to_end(short)
            return self._zarr_handles[short]
        zpath = self._slide_meta[short]["zarr_path"]
        z = zarr.open(zpath, mode="r")
        self._zarr_handles[short] = z
        if len(self._zarr_handles) > self._zarr_lru:
            self._zarr_handles.popitem(last=False)
        return z

    def __getitem__(self, idx: int):
        short, top_y, top_x = self.windows[idx]
        meta = self._slide_meta[short]
        coord_to_idx = meta["coord_to_idx"]
        cell_to_cache_row = meta["cell_to_cache_row"]
        z = self._get_zarr(short)
        z_features = z["features"]

        features = np.zeros((9, 1536), dtype=np.float32)
        valid_mask = np.zeros(9, dtype=bool)
        mask = np.full((768, 768), INVALID_MASK_VALUE, dtype=np.uint8)

        # Bulk-read all needed zarr indices in one shot to amortise read overhead.
        cell_zarr_idx: dict[int, int] = {}    # cell_k → zarr index (if valid)
        for k in range(9):
            dy, dx = divmod(k, 3)
            gy, gx = top_y + dy, top_x + dx
            patch_idx = coord_to_idx.get((gy, gx))
            if patch_idx is None:
                continue
            valid_mask[k] = True
            cell_zarr_idx[k] = patch_idx
        if cell_zarr_idx:
            ks = list(cell_zarr_idx.keys())
            zarr_idxs = [cell_zarr_idx[k] for k in ks]
            # zarr v3 supports arbitrary index slicing; fall back to per-cell if not.
            try:
                feats = np.asarray(z_features.get_orthogonal_selection((zarr_idxs,)))
            except Exception:
                feats = np.stack([np.asarray(z_features[i]) for i in zarr_idxs], axis=0)
            for ki, k in enumerate(ks):
                features[k] = feats[ki]

        # Stitch masks: use cache for positive cells, zeros for non-positive (still valid),
        # 255 for invalid cells (already initialised).
        for k in range(9):
            dy, dx = divmod(k, 3)
            gy, gx = top_y + dy, top_x + dx
            if not valid_mask[k]:
                continue
            y0 = dy * PATCH_SIZE; y1 = y0 + PATCH_SIZE
            x0 = dx * PATCH_SIZE; x1 = x0 + PATCH_SIZE
            cache_row = cell_to_cache_row.get((gy, gx))
            if cache_row is not None:
                tile = self._bundle_masks[cache_row].numpy().astype(np.uint8)
            else:
                tile = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            mask[y0:y1, x0:x1] = tile

        return {
            "features": torch.from_numpy(features),                     # (9, 1536)
            "mask": torch.from_numpy(mask).long(),                       # (768, 768) long
            "valid_mask": torch.from_numpy(valid_mask),                  # (9,) bool
        }


def slide_level_split_windows(
    bundle: dict[str, Any], val_frac: float = 0.157, seed: int = 42,
    k_folds: int = 1, fold_idx: int = 0,
) -> tuple[set[str], set[str], int]:
    """Patient-level split over the unique slide_ids in the bundle.

    Two modes:
      * Legacy (default, k_folds=1, fold_idx=0): random shuffle of
        patients, take val_frac as val.
      * Fold-based (k_folds>1 OR fold_idx>0): use the same patient-
        stratified 5-fold partition as `ps.create_splits` (seed=42 by
        default — must match Stage 1 / GNCAF), then pick fold_idx as
        val. This is the only way to align Stage 2 windows with Stage
        1's val set for cross-validation.

    Returns (train_short_ids, val_short_ids, n_val_patients).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tls_patch_dataset import patient_id_from_slide

    if k_folds > 1 or fold_idx > 0:
        # Aligned fold split — call ps.create_splits exactly as Stage 1 does.
        entries = ps.build_slide_entries()
        all_folds, _test = ps.create_splits(entries, k_folds=5, seed=seed)
        if fold_idx < 0 or fold_idx >= len(all_folds):
            raise ValueError(f"fold_idx={fold_idx} out of range")
        val_entries = all_folds[fold_idx]
        train_entries = [s for i, f in enumerate(all_folds) if i != fold_idx for s in f]

        bundle_slides = set(bundle["slide_ids"])
        val: set[str] = set(); train: set[str] = set()
        for e in val_entries:
            if e["slide_id"] in bundle_slides:
                val.add(e["slide_id"].split(".")[0])
        for e in train_entries:
            if e["slide_id"] in bundle_slides:
                train.add(e["slide_id"].split(".")[0])
        n_val_p = len({"-".join(e["slide_id"].split("-")[:3])
                       for e in val_entries if e["slide_id"] in bundle_slides})
        print(f"  Fold-aligned split: {len(train)} train, {len(val)} val "
              f"slides ({n_val_p} val patients) [k_folds={k_folds}, fold_idx={fold_idx}]")
        return train, val, n_val_p

    # Legacy path
    short_ids = sorted({s.split(".")[0] for s in bundle["slide_ids"]})
    full_ids = sorted({s for s in bundle["slide_ids"]})
    patients = sorted({patient_id_from_slide(s) for s in full_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val_p = max(1, int(val_frac * len(patients)))
    val_patients = set(patients[:n_val_p])
    train: set[str] = set(); val: set[str] = set()
    for s in full_ids:
        short = s.split(".")[0]
        if patient_id_from_slide(s) in val_patients:
            val.add(short)
        else:
            train.add(short)
    return train, val, len(val_patients)
