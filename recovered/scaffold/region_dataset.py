"""v3.37 — region dataset: adds RGB tiles + Stage 1 graph context to
v3.36's neighborhood window enumeration.

Each sample is a 3×3 patch region centred on a TLS-positive component.
For each cell:
    rgb_tile:    (3, 256, 256) ImageNet-normalised float32 — read from WSI .tif
    uni_feature: (1536,) UNI-v2 features — from slide zarr
    graph_ctx:   (gat_dim,) Stage 1 GAT outputs — computed via stage1.proj+gat hops
    mask:        (256, 256) uint8 — per-cell native mask (cache or zeros)
    valid:       bool
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import zarr
from scipy.ndimage import label as scipy_label
from torch.utils.data import Dataset

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_segmentation as ps  # noqa: E402
from gncaf_dataset import (  # noqa: E402
    _read_target_rgb_tile, _normalise_rgb,
    slide_wsi_path,
)
from tls_neighborhood_dataset import (  # noqa: E402
    _slide_zarr_path, _build_slide_grid, _enumerate_windows,
    INVALID_MASK_VALUE, PATCH_SIZE,
)
from region_decoder_model import extract_stage1_context  # noqa: E402


class RegionDataset(Dataset):
    """3×3 region tiles with RGB + UNI + graph context.

    Args:
        bundle: dict from `tls_patch_dataset.build_tls_patch_dataset(...)`
        stage1_model: GraphTLSDetector (frozen, .eval()) on `stage1_device`
        stage1_device: device to run the per-slide Stage 1 forward on
        slide_filter: optional set of short slide_ids to keep (train/val split)
        max_windows_per_slide: subsample uniformly if exceeded
        tile_clusters: also tile stride-2 windows over big TLS clusters
        stride: stride for cluster tiling
        zarr_lru, wsi_lru, ctx_lru: per-worker LRU sizes
        seed: rng seed for window subsampling
    """

    def __init__(
        self,
        bundle: dict[str, Any],
        stage1_model,
        stage1_device: torch.device,
        slide_filter: set[str] | None = None,
        max_windows_per_slide: int = 64,
        tile_clusters: bool = True,
        stride: int = 2,
        zarr_lru: int = 32,
        wsi_lru: int = 8,
        ctx_lru: int = 1024,         # precompute fits all train slides; effectively unbounded
        seed: int = 42,
        precompute_graph_ctx: bool = True,
    ):
        self._bundle_features = bundle["features"]
        self._bundle_masks = bundle["masks"]
        self._bundle_slide_ids = bundle["slide_ids"]
        self._bundle_patch_idx = bundle["patch_idx"]
        self._stage1 = stage1_model
        self._stage1_device = stage1_device
        self._zarr_lru = zarr_lru
        self._wsi_lru = wsi_lru
        self._ctx_lru = ctx_lru
        self._seed = seed

        cache_rows_by_slide: dict[str, list[tuple[int, int]]] = {}
        for ci, sid in enumerate(self._bundle_slide_ids):
            short = sid.split(".")[0]
            if slide_filter is not None and short not in slide_filter:
                continue
            cache_rows_by_slide.setdefault(short, []).append(
                (ci, int(self._bundle_patch_idx[ci]))
            )

        self._slide_meta: dict[str, dict] = {}
        rng = np.random.default_rng(seed)
        windows: list[tuple[str, int, int]] = []
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tls_neighborhood_dataset import _slide_zarr_path  # noqa
        # Resolve full zarr+WSI paths via prepare_segmentation entries.
        from gncaf_dataset import slide_mask_path
        from stage_features_to_local import local_zarr_dirs
        if all(Path(p).is_dir() for p in local_zarr_dirs().values()):
            ps.ZARR_DIRS = local_zarr_dirs()
        entries_by_short = {e["slide_id"].split(".")[0]: e for e in ps.build_slide_entries()}

        for short, rows in cache_rows_by_slide.items():
            entry = entries_by_short.get(short)
            if entry is None:
                continue
            zpath = entry.get("zarr_path")
            wsi_p = slide_wsi_path(entry)
            if not zpath or not (wsi_p and os.path.exists(wsi_p)):
                continue
            try:
                z = zarr.open(zpath, mode="r")
                coords = np.asarray(z["coords"][:])
                if "graph_edges_1hop" in z:
                    edges = np.asarray(z["graph_edges_1hop"][:])
                elif "edge_index" in z:
                    edges = np.asarray(z["edge_index"][:])
                else:
                    continue
            except Exception:
                continue
            coord_to_idx, H, W, grid_y, grid_x = _build_slide_grid(coords)
            cell_to_cache_row: dict[tuple[int, int], int] = {}
            for ci, pidx in rows:
                if 0 <= pidx < coords.shape[0]:
                    gy = int(grid_y[pidx]); gx = int(grid_x[pidx])
                    cell_to_cache_row[(gy, gx)] = ci
            self._slide_meta[short] = {
                "zarr_path": zpath,
                "wsi_path": wsi_p,
                "edge_index": edges,
                "coords": coords,
                "coord_to_idx": coord_to_idx,
                "cell_to_cache_row": cell_to_cache_row,
                "H": H, "W": W,
                "cancer_type": entry["cancer_type"],
                "slide_id_full": entry["slide_id"],
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
        self._zarr_handles: OrderedDict[str, Any] = OrderedDict()
        self._wsi_handles: OrderedDict[str, Any] = OrderedDict()
        self._ctx_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        if precompute_graph_ctx and self._slide_meta:
            is_multi = bool(getattr(self._stage1, "_is_multi_scale", False))
            print(f"  Pre-computing Stage 1 graph context for {len(self._slide_meta)} slides "
                  f"(multi_scale={is_multi})...")
            t0 = time.time()
            n_skipped_ms = 0
            with torch.no_grad():
                for short, meta in self._slide_meta.items():
                    z = self._zarr(short)
                    feats_np = np.asarray(z["features"][:]).astype(np.float32)
                    if is_multi:
                        from multiscale_dataset import build_multiscale_inputs_np
                        ms = build_multiscale_inputs_np(
                            feats_np, meta["coords"], meta["edge_index"],
                            meta["cancer_type"], meta["slide_id_full"],
                        )
                        if ms is None:
                            # Coarse zarr missing — skip; window selection
                            # would still proceed but ctx would be wrong.
                            # Drop this slide so the trainer doesn't see it.
                            n_skipped_ms += 1
                            continue
                        feats_combined, edges_combined, scale_mask, n_fine = ms
                        feats_t = torch.from_numpy(feats_combined).to(self._stage1_device)
                        edges_t = torch.from_numpy(edges_combined).long().to(self._stage1_device)
                        smask_t = torch.from_numpy(scale_mask).to(self._stage1_device)
                        ctx_all = self._stage1.extract_context(feats_t, edges_t, smask_t)
                        ctx = ctx_all[:n_fine].cpu().numpy()
                    else:
                        feats = torch.from_numpy(feats_np)
                        edges = torch.from_numpy(meta["edge_index"]).long()
                        ctx = extract_stage1_context(
                            self._stage1, feats.to(self._stage1_device),
                            edges.to(self._stage1_device),
                        ).cpu().numpy()
                    self._ctx_cache[short] = ctx
            if is_multi and n_skipped_ms:
                print(f"  WARN: {n_skipped_ms} slides missing coarse zarr — dropping them")
                # Also drop their windows.
                kept_shorts = set(self._ctx_cache.keys())
                self.windows = [w for w in windows if w[0] in kept_shorts]
                for s in list(self._slide_meta.keys()):
                    if s not in kept_shorts:
                        self._slide_meta.pop(s)
            print(f"  done in {time.time() - t0:.1f}s "
                  f"(~{ctx.shape[1]}d × {sum(c.shape[0] for c in self._ctx_cache.values()):,} patches)")
            # IMPORTANT: drop the CUDA Stage 1 reference so DataLoader workers
            # can fork without cloning a CUDA context (which deadlocks with
            # num_workers > 0). Graph context is fully precomputed; workers
            # only need numpy arrays + zarr + WSI handles.
            self._stage1 = None
            self._stage1_device = None
            # Also clear any stale zarr handle (workers re-open lazily).
            self._zarr_handles.clear()

    def __len__(self) -> int:
        return len(self.windows)

    def _zarr(self, short: str):
        if short in self._zarr_handles:
            self._zarr_handles.move_to_end(short)
            return self._zarr_handles[short]
        z = zarr.open(self._slide_meta[short]["zarr_path"], mode="r")
        self._zarr_handles[short] = z
        if len(self._zarr_handles) > self._zarr_lru:
            self._zarr_handles.popitem(last=False)
        return z

    def _wsi(self, short: str):
        if short in self._wsi_handles:
            self._wsi_handles.move_to_end(short)
            return self._wsi_handles[short]
        w = zarr.open(tifffile.imread(self._slide_meta[short]["wsi_path"],
                                       aszarr=True, level=0), mode="r")
        self._wsi_handles[short] = w
        if len(self._wsi_handles) > self._wsi_lru:
            self._wsi_handles.popitem(last=False)
        return w

    def _graph_ctx(self, short: str) -> np.ndarray:
        if short in self._ctx_cache:
            self._ctx_cache.move_to_end(short)
            return self._ctx_cache[short]
        meta = self._slide_meta[short]
        z = self._zarr(short)
        feats_np = np.asarray(z["features"][:]).astype(np.float32)
        is_multi = bool(getattr(self._stage1, "_is_multi_scale", False))
        with torch.no_grad():
            if is_multi:
                from multiscale_dataset import build_multiscale_inputs_np
                ms = build_multiscale_inputs_np(
                    feats_np, meta["coords"], meta["edge_index"],
                    meta["cancer_type"], meta["slide_id_full"],
                )
                if ms is None:
                    raise RuntimeError(
                        f"multi-scale Stage 1 set but coarse zarr missing for {short}"
                    )
                feats_combined, edges_combined, scale_mask, n_fine = ms
                feats_t = torch.from_numpy(feats_combined).to(self._stage1_device)
                edges_t = torch.from_numpy(edges_combined).long().to(self._stage1_device)
                smask_t = torch.from_numpy(scale_mask).to(self._stage1_device)
                ctx = self._stage1.extract_context(feats_t, edges_t, smask_t)[:n_fine].cpu().numpy()
            else:
                feats = torch.from_numpy(feats_np)
                edges = torch.from_numpy(meta["edge_index"]).long()
                ctx = extract_stage1_context(
                    self._stage1, feats.to(self._stage1_device),
                    edges.to(self._stage1_device),
                ).cpu().numpy()
        self._ctx_cache[short] = ctx
        if len(self._ctx_cache) > self._ctx_lru:
            self._ctx_cache.popitem(last=False)
        return ctx

    def __getitem__(self, idx: int):
        short, top_y, top_x = self.windows[idx]
        meta = self._slide_meta[short]
        coord_to_idx = meta["coord_to_idx"]
        cell_to_cache_row = meta["cell_to_cache_row"]
        coords = meta["coords"]
        z = self._zarr(short)
        wsi_z = self._wsi(short)
        ctx = self._graph_ctx(short)        # (N, gat_dim)

        gat_dim = ctx.shape[1]
        rgb_tiles = np.zeros((9, 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        uni_features = np.zeros((9, 1536), dtype=np.float32)
        graph_ctx = np.zeros((9, gat_dim), dtype=np.float32)
        valid_mask = np.zeros(9, dtype=bool)
        mask = np.full((768, 768), INVALID_MASK_VALUE, dtype=np.uint8)

        # Resolve all 9 cells.
        cell_to_zarr: dict[int, int] = {}
        for k in range(9):
            dy, dx = divmod(k, 3)
            gy = top_y + dy; gx = top_x + dx
            patch_idx = coord_to_idx.get((gy, gx))
            if patch_idx is None:
                continue
            valid_mask[k] = True
            cell_to_zarr[k] = patch_idx

        if cell_to_zarr:
            ks = list(cell_to_zarr.keys())
            zidxs = [cell_to_zarr[k] for k in ks]
            try:
                feats = np.asarray(z["features"].get_orthogonal_selection((zidxs,)))
            except Exception:
                feats = np.stack([np.asarray(z["features"][i]) for i in zidxs], axis=0)
            for ki, k in enumerate(ks):
                uni_features[k] = feats[ki]
                graph_ctx[k] = ctx[zidxs[ki]]
                # RGB tile: read via WSI zarr at level-0 coord.
                pi = zidxs[ki]
                x0, y0 = int(coords[pi, 0]), int(coords[pi, 1])
                rgb = _read_target_rgb_tile(wsi_z, x0, y0)         # (256, 256, 3) uint8
                rgb_tiles[k] = _normalise_rgb(rgb).numpy()         # (3, 256, 256)
                # Mask: cache for positive cells, zeros otherwise.
                dy, dx = divmod(k, 3)
                y0p = dy * PATCH_SIZE; y1p = y0p + PATCH_SIZE
                x0p = dx * PATCH_SIZE; x1p = x0p + PATCH_SIZE
                cache_row = cell_to_cache_row.get((top_y + dy, top_x + dx))
                if cache_row is not None:
                    tile = self._bundle_masks[cache_row].numpy().astype(np.uint8)
                else:
                    tile = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                mask[y0p:y1p, x0p:x1p] = tile

        return {
            "rgb_tiles": torch.from_numpy(rgb_tiles),                  # (9, 3, 256, 256)
            "uni_features": torch.from_numpy(uni_features),            # (9, 1536)
            "graph_ctx": torch.from_numpy(graph_ctx),                  # (9, gat_dim)
            "mask": torch.from_numpy(mask).long(),                     # (768, 768)
            "valid_mask": torch.from_numpy(valid_mask),                # (9,) bool
        }
