"""Multi-scale bipartite-graph wrapper for Stage 1 (v3.60).

Combines 256-px@20x patches (existing local SSD zarrs / NFS trident
20x_256px) with 512-px@20x patches (trident 20x_512px, ~256-px@10x
effective FoV) into one bipartite graph per slide:

    fine nodes  : 256-px patches  (N_256 nodes)
    coarse nodes: 512-px patches  (N_512 nodes ≈ N_256 / 4)

    edges:
      fine-fine     : 4-conn from local zarr's `graph_edges_1hop`
      coarse-coarse : 4-conn from trident 512-px zarr's `graph_edges_1hop`
      coarse-fine   : containment (each 512-patch ↔ its ≤4 enclosing 256-patches)

The result is fed to a heterogeneous Stage 1 GAT
(`MultiScaleGraphTLSDetector`) that uses a scale embedding to let
the same GATv2Conv handle both node types.

Output of `__getitem__` (one slide):
    features      : (N_256 + N_512, 1536) float32
    edge_index    : (2, E_total) int64 — concat of all three edge types
    scale_mask    : (N_256 + N_512,) int64 — 0 = fine, 1 = coarse
    fine_to_global: (N_256,) int64 — indices into features for fine nodes
                                     (also: torch.arange(N_256))
    coarse_to_global: (N_512,) int64 — coarse-node indices
                                       (= torch.arange(N_256, N_256+N_512))
    coords        : (N_256, 2) float32 — fine-node coords (for label compute)
    mask, center_*, offset_*, slide_id, cancer_type — passthrough from base.

Containment edges: for each coarse node c at (x_c, y_c) (slide-native
pixels), connect to fine nodes f at (x, y) where
    x_c ≤ x < x_c + 512  AND  y_c ≤ y < y_c + 512.
At 20x with 0px overlap a 512-px patch contains exactly 4 of the 256-px
positions: (x_c, y_c), (x_c+256, y_c), (x_c, y_c+256), (x_c+256, y_c+256).
Some may be missing if they failed tissue filtering — we add an edge
for each that exists. Edges are made bidirectional in `edge_index`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/ubuntu/profile-clam")

PATCH_SIZE_FINE = 256
PATCH_SIZE_COARSE = 512


# ─── 512-px zarr roots (trident) ──────────────────────────────────────


COARSE_ZARR_DIRS = {
    "blca": "/home/ubuntu/ahaas-persistent-std-tcga/data/tcga-blca/"
            "representations_tif_trident/20x_512px_0px_overlap/features_uni_v2_grandqc_zarr",
    "kirc": "/home/ubuntu/ahaas-persistent-std-tcga/data/tcga-kirc/"
            "representations_tif_trident/20x_512px_0px_overlap/features_uni_v2_grandqc_zarr",
    "lusc": "/home/ubuntu/ahaas-persistent-std-tcga/data/tcga-lusc/"
            "representations_tif_trident/20x_512px_0px_overlap/features_uni_v2_grandqc_zarr",
}


def coarse_zarr_path(cancer_type: str, slide_id: str) -> str:
    """Return the full path to a slide's 512-px zarr."""
    ct = cancer_type.lower()
    if ct not in COARSE_ZARR_DIRS:
        raise KeyError(f"unknown cancer_type {ct!r}")
    return os.path.join(COARSE_ZARR_DIRS[ct], f"{slide_id}_complete.zarr")


def _build_containment_edges(
    fine_coords: np.ndarray,    # (N_256, 2) float32
    coarse_coords: np.ndarray,  # (N_512, 2) float32
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bipartite containment edges between coarse and fine nodes.

    Returns (coarse_idx, fine_idx) arrays of equal length E. Each
    coarse node connects to every fine node it spatially contains
    (typically 4, fewer at the slide boundary or where 256-px patches
    were filtered out).

    Both index arrays are LOCAL to their own scale (i.e. 0..N_512-1
    and 0..N_256-1). The caller offsets them into a combined index
    space when packing edge_index.
    """
    # Hash fine coords by their (x, y) integer position for O(1) lookup.
    # Fine patches are at multiples of 256 (with offset that doesn't
    # matter — we just look up exact coords).
    fine_lookup: dict[tuple[int, int], int] = {}
    for i, (x, y) in enumerate(fine_coords):
        fine_lookup[(int(x), int(y))] = i

    coarse_idx_list: list[int] = []
    fine_idx_list: list[int] = []
    for c, (xc, yc) in enumerate(coarse_coords):
        xci, yci = int(xc), int(yc)
        for dx in (0, PATCH_SIZE_FINE):
            for dy in (0, PATCH_SIZE_FINE):
                f_idx = fine_lookup.get((xci + dx, yci + dy))
                if f_idx is not None:
                    coarse_idx_list.append(c)
                    fine_idx_list.append(f_idx)
    return (
        np.asarray(coarse_idx_list, dtype=np.int64),
        np.asarray(fine_idx_list, dtype=np.int64),
    )


def _load_coarse_zarr(cancer_type: str, slide_id: str
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load (features, coords, edge_index) for a slide's 512-px zarr.

    Returns None if the zarr doesn't exist for this slide. Each
    field is a numpy array; tensorisation happens at the caller.
    """
    path = coarse_zarr_path(cancer_type, slide_id)
    if not os.path.isdir(path):
        return None
    z = zarr.open(path, mode="r")
    feats = np.asarray(z["features"][:], dtype=np.float32)
    coords = np.asarray(z["coords"][:], dtype=np.float32)
    if "graph_edges_1hop" in z:
        ei = np.asarray(z["graph_edges_1hop"][:], dtype=np.int64)
    elif "edge_index" in z:
        ei = np.asarray(z["edge_index"][:], dtype=np.int64)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    return feats, coords, ei


def add_multi_scale(sample: dict, cancer_type: str, slide_id: str) -> dict:
    """Augment a single-slide sample dict (from TLSSegmentationDataset
    or similar) with coarse-scale features + bipartite edges.

    Modifies and returns the sample. If the coarse zarr is missing
    for this slide, returns the sample unchanged with a `multi_scale=False`
    marker so the caller can decide to skip or fall back.
    """
    fine_features = sample["features"]                  # (N_256, 1536) tensor
    fine_coords = sample["coords"]                       # (N_256, 2) tensor
    fine_edge_index = sample.get("edge_index")           # (2, E_fine) tensor or None

    coarse = _load_coarse_zarr(cancer_type, slide_id)
    if coarse is None:
        # Fallback: single-scale only. Mark the sample so the model
        # knows to use just the fine subgraph for this slide.
        sample["multi_scale"] = False
        sample["scale_mask"] = torch.zeros(fine_features.shape[0], dtype=torch.long)
        return sample

    coarse_features_np, coarse_coords_np, coarse_ei_np = coarse

    n_fine = fine_features.shape[0]
    n_coarse = coarse_features_np.shape[0]

    coarse_features = torch.from_numpy(coarse_features_np)
    coarse_coords = torch.from_numpy(coarse_coords_np)
    coarse_ei = torch.from_numpy(coarse_ei_np).long()

    # Containment edges (numpy → tensor).
    c_idx, f_idx = _build_containment_edges(
        fine_coords.numpy(), coarse_coords.numpy()
    )
    cont_c = torch.from_numpy(c_idx).long()
    cont_f = torch.from_numpy(f_idx).long()

    # Pack everything into a unified (N_fine + N_coarse) node space.
    features = torch.cat([fine_features, coarse_features], dim=0)
    scale_mask = torch.cat([
        torch.zeros(n_fine, dtype=torch.long),
        torch.ones(n_coarse, dtype=torch.long),
    ])

    # Build combined edge_index. Coarse-coarse and containment are
    # offset by N_fine.
    edges = []
    if fine_edge_index is not None and fine_edge_index.numel() > 0:
        edges.append(fine_edge_index.long())
    if coarse_ei.numel() > 0:
        edges.append(coarse_ei + n_fine)
    if cont_c.numel() > 0:
        # Two directed edges per containment relation (bidirectional).
        c_global = cont_c + n_fine
        f_global = cont_f
        edges.append(torch.stack([c_global, f_global]))
        edges.append(torch.stack([f_global, c_global]))
    edge_index = torch.cat(edges, dim=1) if edges else torch.zeros(2, 0, dtype=torch.long)

    sample["features"] = features
    sample["edge_index"] = edge_index
    sample["scale_mask"] = scale_mask
    sample["multi_scale"] = True
    sample["n_fine"] = n_fine
    sample["n_coarse"] = n_coarse
    # Do NOT touch sample["coords"] — it stays at fine-only for label
    # computation by `patch_labels_from_mask`.
    return sample


class MultiScaleSlideDataset(Dataset):
    """Wraps a `TLSSegmentationDataset` and adds coarse-scale 512-px
    nodes + containment edges per slide.

    The base sample's `features` / `edge_index` are replaced with the
    multi-scale combined versions; `scale_mask` and `n_fine`/`n_coarse`
    are added. All other fields (mask, center heatmaps, slide_id, etc.)
    pass through unchanged.

    Loss / label computation in the trainer must use ONLY fine nodes
    (slice via `scale_mask == 0`).
    """

    def __init__(self, base_dataset: Dataset):
        self._base = base_dataset
        # Mirror entries list for cancer_type lookup at __getitem__ time.
        self._entries = getattr(base_dataset, "entries", None)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[idx]
        slide_id_full = sample["slide_id"]
        cancer_type = sample.get("cancer_type") or sample.get(
            "cancer", None
        )
        if cancer_type is None and self._entries is not None:
            cancer_type = self._entries[idx]["cancer_type"]
        if cancer_type is None:
            # Last resort: skip multi-scale augmentation for this slide.
            sample["multi_scale"] = False
            sample["scale_mask"] = torch.zeros(
                sample["features"].shape[0], dtype=torch.long
            )
            return sample
        return add_multi_scale(sample, cancer_type, slide_id_full)
