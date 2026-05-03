"""GNCAF slide-level dataset: WSI tiles + UNI features + 4-conn graph + masks.

One dataset item = one slide. Returns:
    features:    (N, 1536)        UNI-v2 features for every patch in the slide
    edge_index:  (2, E)           pre-computed 1-hop spatial edges (4-conn)
    target_idx:  (K,)             which N nodes were sampled as targets
    target_rgb:  (K, 3, 256, 256) RGB tiles for those targets, ImageNet-normalised
    target_mask: (K, 256, 256)    HookNet 3-class mask tiles in {0=bg, 1=TLS, 2=GC}
    slide_id:    str
    cancer_type: str

K is sampled per-slide: by default we take all TLS-positive patches plus an
equal number of random negatives, capped by `max_targets`.

Data sources (all on local SSD after staging):
    WSI tiles:  /home/ubuntu/ahaas-persistent-std-tcga/slides_tif_/data/drive2/
                alex/tcga/slides_v2/tcga-{ct}/{slide_id}.tif
    Features:   /home/ubuntu/local_data/zarr/{ct}/{slide_id}_complete.zarr
    Masks:      /home/ubuntu/local_data/hooknet_masks_tls_gc/{slide_id_short}_tls_gc_mask.tif

Patch coordinates from the zarr are at slide native resolution (level 0),
which matches WSI level 0 and mask level 0 — same coordinate space throughout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import zarr

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_segmentation as ps  # noqa: E402
from stage_features_to_local import LOCAL_ROOT as LOCAL_ZARR_ROOT, local_zarr_dirs  # noqa: E402
from tls_patch_dataset import (  # noqa: E402
    DEFAULT_LOCAL_MASK_DIR,
    load_test_patient_ids,
    patient_id_from_slide,
)

PATCH_SIZE = 256
WSI_ROOT = "/home/ubuntu/ahaas-persistent-std-tcga/slides_tif_/data/drive2/alex/tcga/slides_v2"

# ImageNet normalisation (TransUNet's encoder is typically pretrained on ImageNet).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─── Slide path resolution ─────────────────────────────────────────────


def slide_wsi_path(entry: dict) -> str:
    """Map an entry's slide_id + cancer_type to the WSI .tif on local SSD."""
    return os.path.join(
        WSI_ROOT, f"tcga-{entry['cancer_type'].lower()}",
        f"{entry['slide_id']}.tif",
    )


def slide_mask_path(entry: dict, local_mask_dir: str = DEFAULT_LOCAL_MASK_DIR) -> str | None:
    """Local mask TIF if available, else fall back to NFS."""
    if not entry.get("mask_path"):
        return None
    local = Path(local_mask_dir) / Path(entry["mask_path"]).name
    return str(local) if local.exists() else entry["mask_path"]


# ─── Per-slide loaders (called from __getitem__) ───────────────────────


def _load_features_and_graph(zarr_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load UNI features, coords, and 1-hop edge_index from a zarr v3 store."""
    g = zarr.open(zarr_path, mode="r")
    features = np.asarray(g["features"][:], dtype=np.float32)         # (N, 1536)
    coords = np.asarray(g["coords"][:], dtype=np.int64)               # (N, 2)
    if "graph_edges_1hop" in g:
        edge_index = np.asarray(g["graph_edges_1hop"][:], dtype=np.int64)
    elif "edge_index" in g:
        edge_index = np.asarray(g["edge_index"][:], dtype=np.int64)
    else:
        # Build 4-connectivity graph from coords on the fly.
        edge_index = _four_conn_edges(coords).astype(np.int64)
    return features, coords, edge_index


def _four_conn_edges(coords: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Construct 4-connectivity edges between patches at adjacent grid cells."""
    cell = coords // patch_size  # (N, 2) integer grid coords
    cell -= cell.min(axis=0, keepdims=True)
    pos_to_idx: dict[tuple[int, int], int] = {(int(x), int(y)): i for i, (x, y) in enumerate(cell)}
    src: list[int] = []
    dst: list[int] = []
    for i, (x, y) in enumerate(cell):
        x, y = int(x), int(y)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None:
                src.append(i); dst.append(j)
    return np.stack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)])


def _read_target_rgb_tile(wsi_zarr: zarr.Array, x: int, y: int,
                          patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Crop a patch_size × patch_size RGB tile from the WSI level-0 zarr view."""
    h, w = wsi_zarr.shape[:2]
    x1, y1 = min(x + patch_size, w), min(y + patch_size, h)
    tile = np.asarray(wsi_zarr[y:y1, x:x1])           # (h, w, 3)
    if tile.shape[:2] != (patch_size, patch_size):
        # Pad with white for edge tiles.
        out = np.full((patch_size, patch_size, 3), 255, dtype=tile.dtype)
        out[: tile.shape[0], : tile.shape[1]] = tile
        tile = out
    return tile


def _read_target_mask_tile(mask_zarr: zarr.Array, x: int, y: int,
                           patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Crop the patch_size × patch_size mask tile from the level-0 mask zarr view."""
    h, w = mask_zarr.shape
    x1, y1 = min(x + patch_size, w), min(y + patch_size, h)
    tile = np.asarray(mask_zarr[y:y1, x:x1])
    if tile.shape != (patch_size, patch_size):
        out = np.zeros((patch_size, patch_size), dtype=tile.dtype)
        out[: tile.shape[0], : tile.shape[1]] = tile
        tile = out
    return tile


def _normalise_rgb(tile: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) → float32 (3, H, W), ImageNet-normalised."""
    t = torch.from_numpy(tile).float().permute(2, 0, 1) / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD


# ─── Dataset ───────────────────────────────────────────────────────────


class GNCAFSlideDataset(torch.utils.data.Dataset):
    """One sample = one slide. Sampling lives inside __getitem__.

    Args:
        entries:        list of dicts from prepare_segmentation.build_slide_entries()
        max_targets:    cap on (TLS-positive + sampled-negative) targets per slide
        neg_per_pos:    sample this many negatives for every TLS-positive patch
                        (capped at max_targets total)
        rng_seed:       per-call seed for reproducibility (None → unseeded)
    """

    def __init__(
        self,
        entries: list[dict],
        max_targets: int = 16,
        neg_per_pos: float = 1.0,
        rng_seed: int | None = None,
        include_negative_slides: bool = False,
    ) -> None:
        if include_negative_slides:
            self.entries = [e for e in entries if e.get("zarr_path")]
        else:
            self.entries = [e for e in entries if e.get("zarr_path") and e.get("mask_path")]
        self.max_targets = max_targets
        self.neg_per_pos = neg_per_pos
        self.rng_seed = rng_seed

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]

        # Slide features + graph (cheap: zarrs on local SSD).
        features, coords, edge_index = _load_features_and_graph(entry["zarr_path"])
        n = features.shape[0]

        # Open WSI as zarr-view for windowed reads. Mask may be missing for
        # bg-only slides — handled below.
        wsi_path = slide_wsi_path(entry)
        mask_path = slide_mask_path(entry)
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"Missing WSI for {entry['slide_id']}: wsi={wsi_path}")
        is_negative_slide = mask_path is None or not os.path.exists(mask_path)
        wsi_z = zarr.open(tifffile.imread(wsi_path, aszarr=True, level=0), mode="r")
        mask_z = None if is_negative_slide else zarr.open(
            tifffile.imread(mask_path, aszarr=True, level=0), mode="r")

        rng = np.random.default_rng(self.rng_seed)

        if is_negative_slide:
            # No GT mask: every patch is bg. Sample max_targets random patches;
            # target_mask is all-zero (bg class) for each.
            n_target = min(self.max_targets, n)
            target_idx = rng.choice(n, size=n_target, replace=False).astype(np.int64)
        else:
            # Determine which patches are TLS-positive by sampling the mask centre.
            centre_y = (coords[:, 1] + PATCH_SIZE // 2).astype(np.int64)
            centre_x = (coords[:, 0] + PATCH_SIZE // 2).astype(np.int64)
            mh, mw = mask_z.shape
            centre_y = np.clip(centre_y, 0, mh - 1)
            centre_x = np.clip(centre_x, 0, mw - 1)
            centre_label = np.asarray(
                [int(mask_z[int(y), int(x)]) for y, x in zip(centre_y, centre_x)]
            )
            pos_mask = centre_label > 0
            pos_idx = np.where(pos_mask)[0]
            neg_idx = np.where(~pos_mask)[0]

            n_pos = min(len(pos_idx), self.max_targets)
            n_neg = min(len(neg_idx), int(np.ceil(n_pos * self.neg_per_pos)),
                        self.max_targets - n_pos)
            if n_pos > 0:
                chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
            else:
                chosen_pos = np.empty(0, dtype=np.int64)
            if n_neg > 0:
                chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            else:
                chosen_neg = np.empty(0, dtype=np.int64)
            target_idx = np.concatenate([chosen_pos, chosen_neg])

            if target_idx.size == 0:
                target_idx = np.array([rng.integers(0, n)], dtype=np.int64)

        # Read each target's RGB + mask tile at slide-native pixels.
        target_rgbs = []
        target_masks = []
        for ti in target_idx:
            x, y = int(coords[ti, 0]), int(coords[ti, 1])
            rgb = _read_target_rgb_tile(wsi_z, x, y)
            target_rgbs.append(_normalise_rgb(rgb))
            if mask_z is None:
                target_masks.append(torch.zeros(PATCH_SIZE, PATCH_SIZE, dtype=torch.int64))
            else:
                mask = _read_target_mask_tile(mask_z, x, y)
                target_masks.append(torch.from_numpy(mask.astype(np.int64)))

        return {
            "features": torch.from_numpy(features),                   # (N, 1536)
            "edge_index": torch.from_numpy(edge_index),                # (2, E)
            "target_idx": torch.from_numpy(target_idx),                # (K,)
            "target_rgb": torch.stack(target_rgbs),                    # (K, 3, 256, 256)
            "target_mask": torch.stack(target_masks),                  # (K, 256, 256)
            "slide_id": entry["slide_id"],
            "cancer_type": entry["cancer_type"],
        }


# ─── Top-level builder ─────────────────────────────────────────────────


def build_gncaf_split(
    seed: int = 42, k_folds: int = 1, use_local_ssd: bool = True,
    skip_test_patients: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Apply our standard test-patient exclusion + 5-fold patient-stratified
    split, then return (train_entries, val_entries).
    """
    if use_local_ssd and all(Path(p).is_dir() for p in local_zarr_dirs().values()):
        ps.ZARR_DIRS = local_zarr_dirs()
        print(f"Using locally-staged zarrs at {LOCAL_ZARR_ROOT}")
    ps.set_seed(seed)
    entries = ps.build_slide_entries()
    if skip_test_patients:
        # create_splits already excludes test patients via TEST_CSV; just call it.
        pass
    folds_pair, _test = ps.create_splits(entries, k_folds=k_folds, seed=seed)
    val_entries, train_entries = folds_pair[0], folds_pair[1]
    return train_entries, val_entries


if __name__ == "__main__":
    import time
    train, val = build_gncaf_split()
    print(f"\nGNCAF dataset: {len(train)} train slides, {len(val)} val slides")
    ds = GNCAFSlideDataset(train, max_targets=8, neg_per_pos=1.0, rng_seed=0)
    print(f"Filtered (with zarr+mask): {len(ds)}")
    if len(ds) == 0:
        print("No usable slides!")
    else:
        t0 = time.time()
        sample = ds[0]
        dt = time.time() - t0
        print(f"\nLoaded slide 0 ({sample['slide_id'][:30]}) in {dt:.1f}s:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k}: {v}")
