"""GARS cascade eval: Stage 1 → threshold → Stage 2 → slide-level metrics.

v3.2 — hydra config + wandb summary + config dump to results dir.

Pipeline per slide:
  1. Stage 1 (GraphTLSDetector) scores every patch.
  2. Threshold at THR → select TLS-positive patch indices.
  3. Stage 2 (UNIv2PixelDecoder) decodes each selected patch's UNI-v2
     feature into a 256×256 3-class mask (batched).
  4. Stitch per-patch argmax tiles into a slide-level patch grid.
  5. Compare against the cached HookNet mask reduced to the same
     patch-grid resolution → mDice / TLS dice / GC dice.
  6. Connected-component counting on the predicted grid → Spearman vs
     ground-truth instance counts.

Run:
    python eval_gars_cascade.py stage1=<path> stage2=<path>
    python eval_gars_cascade.py stage1=... stage2=... thresholds=[0.05,0.1]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/ubuntu/profile-clam")
from train_gars_stage1 import GraphTLSDetector  # noqa: E402
from train_gars_stage2 import UNIv2PixelDecoder  # noqa: E402

PATCH_SIZE = 256
NEIGHBORHOOD_GRID = 3   # 3×3 windows for v3.36 NeighborhoodPixelDecoder


def load_stage2_neighborhood(ckpt_path: str, device: torch.device):
    """Load v3.36 NeighborhoodPixelDecoder with strict=True."""
    from train_gars_neighborhood import NeighborhoodPixelDecoder
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    model = NeighborhoodPixelDecoder(
        in_dim=m.get("in_dim", 1536),
        bottleneck=m.get("bottleneck", 512),
        hidden_channels=m.get("hidden_channels", 128),
        spatial_size=m.get("spatial_size", 16),
        n_classes=m.get("n_classes", 3),
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_stage2_region(ckpt_path: str, device: torch.device):
    """Load v3.37 RegionDecoder (RGB+UNI+graph) with strict=True."""
    from region_decoder_model import RegionDecoder
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    model = RegionDecoder(
        uni_dim=m.get("uni_dim", 1536),
        gat_dim=m.get("gat_dim", 256),
        hidden_channels=m.get("hidden_channels", 64),
        n_classes=m.get("n_classes", 3),
        grid_n=m.get("grid_n", 3),
        rgb_pretrained=False,           # weights overridden by strict-load anyway
        freeze_rgb_encoder=False,
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_stage1(ckpt_path: str, device: torch.device):
    """Load a Stage 1 checkpoint. Auto-detects multi-scale checkpoints
    (which contain a `scale_embed.weight` key) and instantiates the
    appropriate model class.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    state_dict = obj["model_state_dict"]
    is_multi_scale = (
        obj.get("model_class") == "MultiScaleGraphTLSDetector"
        or "scale_embed.weight" in state_dict
    )
    if is_multi_scale:
        from multiscale_stage1_model import MultiScaleGraphTLSDetector
        model = MultiScaleGraphTLSDetector(
            in_dim=m.get("in_dim", 1536),
            hidden_dim=m.get("hidden_dim", 256),
            n_hops=m.get("n_hops", 5),
            gnn_type=m.get("gnn_type", "gatv2"),
            dropout=m.get("dropout", 0.1),
            gat_heads=m.get("gat_heads", 4),
        )
    else:
        model = GraphTLSDetector(
            in_dim=m.get("in_dim", 1536),
            hidden_dim=m.get("hidden_dim", 256),
            n_hops=m.get("n_hops", 3),
            gnn_type=m.get("gnn_type", "gatv2"),
            dropout=m.get("dropout", 0.1),
            gat_heads=m.get("gat_heads", 4),
        )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    model._is_multi_scale = is_multi_scale
    return model


def load_stage2(ckpt_path: str, device: torch.device) -> UNIv2PixelDecoder:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) or {}
    m = cfg.get("model", cfg)
    model = UNIv2PixelDecoder(
        in_dim=m.get("in_dim", 1536),
        bottleneck=m.get("bottleneck", 512),
        hidden_channels=m.get("hidden_channels", 64),
        spatial_size=m.get("spatial_size", 16),
        n_classes=m.get("n_classes", 3),
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def cascade_one_slide(stage1, stage2, features, coords, edge_index, threshold,
                      device, s2_batch=64, top_k_frac: float | None = None,
                      min_top_k: int = 1, top_k_abstain_thr: float | None = None):
    """Returns:
        grid_class:    (H_grid, W_grid) patch-level argmax (priority GC>TLS>bg)
        selected:      indices Stage 1 picked
        pred_tiles:    {patch_idx: (256, 256) np.uint8 argmax mask} for selected

    Selection rule:
        - If `top_k_frac` is None, use absolute `threshold` on Stage 1 prob.
        - Else, select top-K% of patches by Stage 1 prob (per-slide
          adaptive threshold). Useful when slides vary in TLS density.
        - If `top_k_abstain_thr` is also set, top-K is only used when at
          least one patch's prob exceeds it; otherwise fall back to
          absolute `threshold`. Avoids forcing top-K on TLS-empty slides.
    """
    n = features.shape[0]
    grid_x = (coords[:, 0] / PATCH_SIZE).long()
    grid_y = (coords[:, 1] / PATCH_SIZE).long()
    grid_x -= grid_x.min()
    grid_y -= grid_y.min()
    H = int(grid_y.max().item()) + 1
    W = int(grid_x.max().item()) + 1

    s1_logits = stage1(features.to(device), edge_index.to(device))
    s1_probs = torch.sigmoid(s1_logits).cpu().numpy()
    use_topk = top_k_frac is not None
    if use_topk and top_k_abstain_thr is not None:
        use_topk = bool((s1_probs > top_k_abstain_thr).any())
    if use_topk:
        k = max(min_top_k, int(top_k_frac * n))
        selected = np.argpartition(-s1_probs, min(k, n - 1))[:k]
    else:
        selected = np.where(s1_probs > threshold)[0]

    pred_class_per_patch = np.zeros(n, dtype=np.int64)
    pred_tiles: dict[int, np.ndarray] = {}
    for s in range(0, len(selected), s2_batch):
        batch_idx = selected[s : s + s2_batch]
        feats = features[batch_idx].to(device)
        argmax = stage2(feats).argmax(dim=1).cpu().numpy().astype(np.uint8)
        for b, i in enumerate(batch_idx):
            tile = argmax[b]
            pred_tiles[int(i)] = tile
            n_gc = int((tile == 2).sum())
            n_tls = int((tile == 1).sum())
            pred_class_per_patch[i] = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)

    grid_class = np.zeros((H, W), dtype=np.int64)
    for i in range(n):
        gx, gy = int(grid_x[i]), int(grid_y[i])
        grid_class[gy, gx] = pred_class_per_patch[i]
    return grid_class, len(selected), pred_tiles


@torch.no_grad()
def cascade_one_slide_neighborhood(stage1, stage2_nbhd, features, coords,
                                    edge_index, threshold, device,
                                    s2_batch=16,
                                    min_component_size=2,
                                    closing_iters=1):
    """Neighborhood-cascade variant of cascade_one_slide.

    Stage 1 score → binary closing → connected components → 3×3 windows
    centered on each component (stride-2 raster for components > 3×3).
    Each window decoded by NeighborhoodPixelDecoder → 768×768 probs.
    Slide-level prob accumulator with max-prob reconciliation.

    Returns the same triple as `cascade_one_slide`:
        grid_class, len(selected), pred_tiles (256×256 per Stage-1-positive patch)
    """
    from scipy.ndimage import binary_closing as _bclose, label as _label
    n = features.shape[0]
    grid_x = (coords[:, 0] / PATCH_SIZE).long()
    grid_y = (coords[:, 1] / PATCH_SIZE).long()
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    H = int(grid_y.max().item()) + 1
    W = int(grid_x.max().item()) + 1
    grid_x_np = grid_x.numpy(); grid_y_np = grid_y.numpy()
    coord_to_idx = {(int(grid_y_np[i]), int(grid_x_np[i])): i for i in range(n)}

    # Stage 1 selection.
    s1_logits = stage1(features.to(device), edge_index.to(device))
    s1_probs = torch.sigmoid(s1_logits).cpu().numpy()
    selected = np.where(s1_probs > threshold)[0]
    selected_grid = np.zeros((H, W), dtype=np.uint8)
    for i in selected:
        selected_grid[int(grid_y_np[i]), int(grid_x_np[i])] = 1

    # Connected components on (selected_grid + closing).
    if closing_iters > 0:
        binary = _bclose(selected_grid.astype(bool), iterations=closing_iters)
    else:
        binary = selected_grid.astype(bool)
    lab, n_comp = _label(binary)

    # Enumerate 3×3 windows.
    windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    G = NEIGHBORHOOD_GRID

    def add_window(top_y: int, top_x: int):
        top_y = max(0, min(H - G, top_y))
        top_x = max(0, min(W - G, top_x))
        key = (top_y, top_x)
        if key not in seen:
            seen.add(key); windows.append(key)

    for c in range(1, n_comp + 1):
        ys, xs = np.where(lab == c)
        if ys.size < min_component_size:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if (y1 - y0) <= G and (x1 - x0) <= G:
            cy = (y0 + y1) // 2; cx = (x0 + x1) // 2
            add_window(cy - G // 2, cx - G // 2)
        else:
            for ty in range(max(0, y0 - 1), max(y0, y1 - (G - 1)) + 1, 2):
                for tx in range(max(0, x0 - 1), max(x0, x1 - (G - 1)) + 1, 2):
                    add_window(ty, tx)
    if not windows:
        # No candidate regions — return empty result like cascade_one_slide.
        return np.zeros((H, W), dtype=np.int64), 0, {}

    # Build (n_w, 9, 1536) feature batch + (n_w, 9) valid_mask + window meta.
    n_w = len(windows)
    win_features = np.zeros((n_w, G * G, features.shape[1]), dtype=np.float32)
    win_valid = np.zeros((n_w, G * G), dtype=bool)
    win_cells_patchidx: list[list[int | None]] = [[None] * (G * G) for _ in range(n_w)]
    feats_np = features.numpy() if torch.is_tensor(features) else features
    for wi, (ty, tx) in enumerate(windows):
        for k in range(G * G):
            dy, dx = divmod(k, G)
            gy = ty + dy; gx = tx + dx
            pi = coord_to_idx.get((gy, gx))
            if pi is not None:
                win_features[wi, k] = feats_np[pi]
                win_valid[wi, k] = True
                win_cells_patchidx[wi][k] = pi

    # Per-cell prob dict (max-prob reconciliation over overlapping windows).
    # Avoids the 30 GB slide-wide accumulator we had originally.
    cell_probs: dict[tuple[int, int], np.ndarray] = {}

    feats_t = torch.from_numpy(win_features).to(device)
    valid_t = torch.from_numpy(win_valid).to(device)
    for s in range(0, n_w, s2_batch):
        f_b = feats_t[s : s + s2_batch]
        v_b = valid_t[s : s + s2_batch]
        logits = stage2_nbhd(f_b, v_b)                              # (B, 3, 768, 768)
        probs = torch.softmax(logits, dim=1).cpu().numpy()          # (B, 3, 768, 768)
        for b, wi in enumerate(range(s, s + f_b.shape[0])):
            ty, tx = windows[wi]
            for k in range(G * G):
                dy, dx = divmod(k, G)
                gy = ty + dy; gx = tx + dx
                if not (0 <= gy < H and 0 <= gx < W):
                    continue
                if not bool(win_valid[wi, k]):
                    continue
                cell = probs[b, :, dy * PATCH_SIZE:(dy + 1) * PATCH_SIZE,
                                  dx * PATCH_SIZE:(dx + 1) * PATCH_SIZE]
                existing = cell_probs.get((gy, gx))
                if existing is None:
                    cell_probs[(gy, gx)] = cell.copy()
                else:
                    np.maximum(existing, cell, out=existing)

    # Build pred_tiles for Stage-1-positive patches (matches per-patch eval API).
    pred_tiles: dict[int, np.ndarray] = {}
    pred_class_per_patch = np.zeros(n, dtype=np.int64)
    for i in selected:
        gy = int(grid_y_np[i]); gx = int(grid_x_np[i])
        cp = cell_probs.get((gy, gx))
        if cp is None:
            tile = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        else:
            tile = cp.argmax(axis=0).astype(np.uint8)
        pred_tiles[int(i)] = tile
        n_gc = int((tile == 2).sum()); n_tls = int((tile == 1).sum())
        pred_class_per_patch[i] = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)

    grid_class = np.zeros((H, W), dtype=np.int64)
    for i in range(n):
        grid_class[int(grid_y_np[i]), int(grid_x_np[i])] = pred_class_per_patch[i]
    return grid_class, len(selected), pred_tiles


@torch.no_grad()
def cascade_one_slide_region(stage1, stage2_region, features, coords, edge_index,
                             threshold, device, wsi_path, s2_batch=8,
                             min_component_size=2, closing_iters=1,
                             return_cell_probs: bool = False,
                             multiscale_payload: dict | None = None):
    """v3.37 region-cascade: Stage 1 → connected components → 3×3 windows
    → RegionDecoder (RGB + UNI + graph). RGB tiles read from WSI .tif
    on demand (only for selected regions, not the full slide).

    `features`, `coords`, and `edge_index` are FINE-only (256-px). When
    using a multi-scale Stage 1, pass `multiscale_payload` =
    {features_bi, edge_index_bi, scale_mask, n_fine} from
    `multiscale_dataset.add_multi_scale` so Stage 1 can run over the
    bipartite graph; outputs are sliced back to fine for selection
    and Stage 2.
    """
    from scipy.ndimage import binary_closing as _bclose, label as _label
    import tifffile as _tifffile
    import zarr  # noqa: F811
    from gncaf_dataset import _read_target_rgb_tile, _normalise_rgb
    from region_decoder_model import extract_stage1_context

    n = features.shape[0]
    grid_x = (coords[:, 0] / PATCH_SIZE).long()
    grid_y = (coords[:, 1] / PATCH_SIZE).long()
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    H = int(grid_y.max().item()) + 1
    W = int(grid_x.max().item()) + 1
    grid_x_np = grid_x.numpy(); grid_y_np = grid_y.numpy()
    coord_to_idx = {(int(grid_y_np[i]), int(grid_x_np[i])): i for i in range(n)}

    # Stage 1 selection + graph context (single forward; head + before-head).
    if multiscale_payload is not None:
        feats_dev = multiscale_payload["features"].to(device)
        ei_dev = multiscale_payload["edge_index"].to(device)
        scale_mask_dev = multiscale_payload["scale_mask"].to(device)
        n_fine = int(multiscale_payload["n_fine"])
        assert n_fine == n, f"n_fine mismatch: {n_fine} vs {n}"
        s1_logits_all = stage1(feats_dev, ei_dev, scale_mask_dev)
        graph_ctx_all = stage1.extract_context(feats_dev, ei_dev, scale_mask_dev)
        s1_logits = s1_logits_all[:n_fine]
        graph_ctx_t = graph_ctx_all[:n_fine]
    else:
        feats_dev = features.to(device); ei_dev = edge_index.to(device)
        s1_logits = stage1(feats_dev, ei_dev)
        graph_ctx_t = extract_stage1_context(stage1, feats_dev, ei_dev)   # (N, gat_dim)
    s1_probs = torch.sigmoid(s1_logits).cpu().numpy()

    selected = np.where(s1_probs > threshold)[0]
    selected_grid = np.zeros((H, W), dtype=np.uint8)
    for i in selected:
        selected_grid[int(grid_y_np[i]), int(grid_x_np[i])] = 1
    if closing_iters > 0:
        binary = _bclose(selected_grid.astype(bool), iterations=closing_iters)
    else:
        binary = selected_grid.astype(bool)
    lab, n_comp = _label(binary)

    G = NEIGHBORHOOD_GRID
    windows: list[tuple[int, int]] = []; seen: set[tuple[int, int]] = set()

    def add_window(top_y: int, top_x: int):
        top_y = max(0, min(H - G, top_y))
        top_x = max(0, min(W - G, top_x))
        key = (top_y, top_x)
        if key not in seen:
            seen.add(key); windows.append(key)

    for c in range(1, n_comp + 1):
        ys, xs = np.where(lab == c)
        if ys.size < min_component_size:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if (y1 - y0) <= G and (x1 - x0) <= G:
            cy = (y0 + y1) // 2; cx = (x0 + x1) // 2
            add_window(cy - G // 2, cx - G // 2)
        else:
            for ty in range(max(0, y0 - 1), max(y0, y1 - (G - 1)) + 1, 2):
                for tx in range(max(0, x0 - 1), max(x0, x1 - (G - 1)) + 1, 2):
                    add_window(ty, tx)
    if not windows:
        return np.zeros((H, W), dtype=np.int64), 0, {}

    # Open WSI handle once.
    wsi_z = zarr.open(_tifffile.imread(wsi_path, aszarr=True, level=0), mode="r")
    feats_np = features.numpy() if torch.is_tensor(features) else features
    coords_np = coords.numpy() if torch.is_tensor(coords) else coords
    ctx_np = graph_ctx_t.cpu().numpy()

    n_w = len(windows)
    rgb_buf = np.zeros((n_w, G * G, 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    uni_buf = np.zeros((n_w, G * G, feats_np.shape[1]), dtype=np.float32)
    gat_buf = np.zeros((n_w, G * G, ctx_np.shape[1]), dtype=np.float32)
    val_buf = np.zeros((n_w, G * G), dtype=bool)

    for wi, (ty, tx) in enumerate(windows):
        for k in range(G * G):
            dy, dx = divmod(k, G)
            gy = ty + dy; gx = tx + dx
            pi = coord_to_idx.get((gy, gx))
            if pi is None:
                continue
            val_buf[wi, k] = True
            uni_buf[wi, k] = feats_np[pi]
            gat_buf[wi, k] = ctx_np[pi]
            x0p = int(coords_np[pi, 0]); y0p = int(coords_np[pi, 1])
            tile = _read_target_rgb_tile(wsi_z, x0p, y0p)            # (256, 256, 3)
            rgb_buf[wi, k] = _normalise_rgb(tile).numpy()             # (3, 256, 256)

    cell_probs: dict[tuple[int, int], np.ndarray] = {}

    rgb_t = torch.from_numpy(rgb_buf).to(device)
    uni_t = torch.from_numpy(uni_buf).to(device)
    gat_t = torch.from_numpy(gat_buf).to(device)
    val_t = torch.from_numpy(val_buf).to(device)
    for s in range(0, n_w, s2_batch):
        b_rgb = rgb_t[s : s + s2_batch]; b_uni = uni_t[s : s + s2_batch]
        b_gat = gat_t[s : s + s2_batch]; b_val = val_t[s : s + s2_batch]
        logits = stage2_region(b_rgb, b_uni, b_gat, b_val)
        probs = torch.softmax(logits, dim=1).cpu().numpy()           # (B, 3, 768, 768)
        for b, wi in enumerate(range(s, s + b_rgb.shape[0])):
            ty, tx = windows[wi]
            for k in range(G * G):
                dy, dx = divmod(k, G)
                gy = ty + dy; gx = tx + dx
                if not (0 <= gy < H and 0 <= gx < W):
                    continue
                if not bool(val_buf[wi, k]):
                    continue
                cell = probs[b, :, dy * PATCH_SIZE:(dy + 1) * PATCH_SIZE,
                                  dx * PATCH_SIZE:(dx + 1) * PATCH_SIZE]
                existing = cell_probs.get((gy, gx))
                if existing is None:
                    cell_probs[(gy, gx)] = cell.copy()
                else:
                    np.maximum(existing, cell, out=existing)

    pred_tiles: dict[int, np.ndarray] = {}
    pred_class_per_patch = np.zeros(n, dtype=np.int64)
    for i in selected:
        gy = int(grid_y_np[i]); gx = int(grid_x_np[i])
        cp = cell_probs.get((gy, gx))
        if cp is None:
            tile = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        else:
            tile = cp.argmax(axis=0).astype(np.uint8)
        pred_tiles[int(i)] = tile
        n_gc = int((tile == 2).sum()); n_tls = int((tile == 1).sum())
        pred_class_per_patch[i] = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)
    grid_class = np.zeros((H, W), dtype=np.int64)
    for i in range(n):
        grid_class[int(grid_y_np[i]), int(grid_x_np[i])] = pred_class_per_patch[i]
    if return_cell_probs:
        return grid_class, len(selected), pred_tiles, cell_probs
    return grid_class, len(selected), pred_tiles


def patch_grid_from_mask_cache(cache, coords, upsample_factor):
    mask = cache["mask"].numpy() if hasattr(cache["mask"], "numpy") else cache["mask"]
    grid_x = (coords[:, 0] / PATCH_SIZE).long()
    grid_y = (coords[:, 1] / PATCH_SIZE).long()
    grid_x -= grid_x.min()
    grid_y -= grid_y.min()
    H = int(grid_y.max().item()) + 1
    W = int(grid_x.max().item()) + 1
    out = np.zeros((H, W), dtype=np.int64)
    u = upsample_factor
    for i in range(coords.shape[0]):
        gx, gy = int(grid_x[i]), int(grid_y[i])
        cell = mask[gy * u : (gy + 1) * u, gx * u : (gx + 1) * u]
        if (cell == 2).any():
            out[gy, gx] = 2
        elif (cell == 1).any():
            out[gy, gx] = 1
    return out


def dice_score(pred, target, cls, eps=1e-6):
    p = (pred == cls); t = (target == cls)
    inter = float((p & t).sum())
    denom = float(p.sum() + t.sum())
    return (2 * inter + eps) / (denom + eps)


def count_components(grid, cls):
    from scipy.ndimage import label
    _, n = label(grid == cls)
    return n


def count_components_filtered(grid, cls, min_size: int = 1, closing_iters: int = 0):
    """Connected-component count with min-size filter and optional
    binary closing (fills 1-cell gaps so adjacent positive patches
    that should be one instance aren't fragmented).
    """
    from scipy.ndimage import label, binary_closing
    binary = (grid == cls)
    if closing_iters > 0:
        binary = binary_closing(binary, iterations=closing_iters)
    labels, n = label(binary)
    if min_size > 1 and n > 0:
        sizes = np.bincount(labels.ravel())[1:]  # skip background (0)
        n = int((sizes >= min_size).sum())
    return n


@hydra.main(version_base=None, config_path="configs/cascade", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    import prepare_segmentation as ps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print(f"Stage 1: {cfg.stage1}")
    stage1 = load_stage1(cfg.stage1, device)
    neighborhood_mode = bool(cfg.get("neighborhood_mode", False))
    region_mode = bool(cfg.get("region_mode", False))
    if neighborhood_mode and region_mode:
        raise ValueError("neighborhood_mode and region_mode are mutually exclusive")
    if region_mode:
        rgn_ckpt = cfg.get("stage2_region")
        if not rgn_ckpt:
            raise ValueError("region_mode=true requires stage2_region=<ckpt>")
        print(f"Stage 2 (region): {rgn_ckpt}")
        stage2 = load_stage2_region(rgn_ckpt, device)
    elif neighborhood_mode:
        nbhd_ckpt = cfg.get("stage2_neighborhood")
        if not nbhd_ckpt:
            raise ValueError("neighborhood_mode=true requires stage2_neighborhood=<ckpt>")
        print(f"Stage 2 (neighborhood): {nbhd_ckpt}")
        stage2 = load_stage2_neighborhood(nbhd_ckpt, device)
    else:
        print(f"Stage 2: {cfg.stage2}")
        stage2 = load_stage2(cfg.stage2, device)

    ps.set_seed(cfg.seed)
    entries = ps.build_slide_entries()
    fold_idx = int(cfg.get("fold_idx", 0))
    k_folds = int(cfg.get("k_folds", 1))
    # Full-cohort evaluation: include GT-negative slides (mask_path=None)
    # so that the slide-level detection CMs / PR-AUC / AUROC reflect the
    # true cohort. The legacy "drop mask_path is None" filters are gated
    # behind cfg.eval_positives_only (default False).
    positives_only = bool(cfg.get("eval_positives_only", False))
    def _maybe_filter(es):
        if positives_only:
            return [e for e in es if e.get("mask_path") is not None]
        return list(es)
    if cfg.get("use_test_split", False):
        _folds, test_entries = ps.create_splits(entries, k_folds=1, seed=cfg.seed)
        val_entries = _maybe_filter(test_entries)
        print(f"Using TEST split: {len(val_entries)} slides "
              f"({sum(1 for e in val_entries if e.get('mask_path') is not None)} with masks)")
    elif k_folds == 1 and fold_idx == 0:
        folds_pair, _test = ps.create_splits(entries, k_folds=1, seed=cfg.seed)
        val_entries = _maybe_filter(folds_pair[0])
    else:
        all_folds, _test = ps.create_splits(entries, k_folds=5, seed=cfg.seed)
        if fold_idx < 0 or fold_idx >= len(all_folds):
            raise ValueError(f"fold_idx={fold_idx} out of range")
        val_entries = _maybe_filter(all_folds[fold_idx])
        n_neg = sum(1 for e in val_entries if e.get("mask_path") is None)
        print(f"Using fold {fold_idx} as val "
              f"(seed={cfg.seed}, k_folds={k_folds}, "
              f"{len(val_entries) - n_neg} mask slides + {n_neg} GT-negatives)")
    if cfg.get("limit_slides"):
        val_entries = val_entries[: int(cfg.limit_slides)]
        print(f"  limit_slides={cfg.limit_slides}")
    offset = int(cfg.get("slide_offset", 0))
    stride = int(cfg.get("slide_stride", 1))
    if stride > 1:
        val_entries = val_entries[offset::stride]
        print(f"Sharded: offset={offset} stride={stride} → {len(val_entries)} slides")
    print(f"Val: {len(val_entries)} slides with masks\n")

    print(f"Loading mask cache (upsample_factor={cfg.upsample_factor})...")
    mask_dict = ps.build_mask_cache(val_entries, cfg.upsample_factor)

    # Ground-truth instance counts (per slide) for the counting Spearman.
    import pandas as pd
    meta_df = pd.read_csv(ps.META_CSV)
    gt_counts: dict[str, tuple[int, int]] = {}
    for _, row in meta_df.iterrows():
        sid = str(row.get("slide_id", ""))
        if sid:
            gt_counts[sid.split(".")[0]] = (
                int(row.get("tls_num", 0)),
                int(row.get("gc_num", 0)),
            )

    # Native-resolution per-patch GT masks for per-pixel slide-level dice.
    print("Loading TLS-patch cache (native 256×256 GT masks for pixel dice)...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tls_patch_dataset import build_tls_patch_dataset
    patch_cache_path = cfg.get("patch_cache_path",
                               "/home/ubuntu/local_data/tls_patch_dataset.pt")
    bundle = build_tls_patch_dataset(cache_path=patch_cache_path)
    bundle_masks = bundle["masks"]
    patch_mask_lookup: dict[str, dict[int, np.ndarray]] = {}
    for ci, sid in enumerate(bundle["slide_ids"]):
        short = sid.split(".")[0]
        pi = int(bundle["patch_idx"][ci])
        patch_mask_lookup.setdefault(short, {})[pi] = np.asarray(bundle_masks[ci])

    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != "disabled":
        import wandb
        run = wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=out_dir.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir), mode=cfg.wandb.mode,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        )

    rows = []
    per_slide_by_thr: dict[float, list[dict]] = {}
    import zarr
    for thr in cfg.thresholds:
        print(f"\n{'=' * 60}\nTHRESHOLD = {thr}\n{'=' * 60}")
        per_slide = []
        n_selected_total = n_total_total = 0
        # Per-pixel aggregate dice over selected patches' 256×256 masks
        # (matches the recovered cascade_eval.log style — high GC, low TLS).
        # Aggregate intersection + union per class across all selected
        # patches in val, then divide once at the end.
        agg = {1: {"inter": 0, "denom": 0}, 2: {"inter": 0, "denom": 0}}
        for k, entry in enumerate(val_entries):
            short_id = entry["slide_id"].split(".")[0]
            cache = mask_dict.get(short_id)
            is_gt_negative = (cache is None) or (entry.get("mask_path") is None)
            grp = zarr.open(entry["zarr_path"], mode="r")
            features = torch.from_numpy(np.asarray(grp["features"][:])).float()
            coords = torch.from_numpy(np.asarray(grp["coords"][:])).float()
            if "graph_edges_1hop" in grp:
                edge_index = torch.from_numpy(np.asarray(grp["graph_edges_1hop"][:])).long()
            elif "edge_index" in grp:
                edge_index = torch.from_numpy(np.asarray(grp["edge_index"][:])).long()
            else:
                continue

            t0 = time.time()
            # Build multi-scale payload if Stage 1 is multi-scale.
            multiscale_payload = None
            if getattr(stage1, "_is_multi_scale", False):
                from multiscale_dataset import _load_coarse_zarr, _build_containment_edges
                cz = _load_coarse_zarr(entry["cancer_type"], entry["slide_id"])
                if cz is None:
                    print(f"  skip {short_id}: no coarse 512-px zarr")
                    continue
                coarse_features_np, coarse_coords_np, coarse_ei_np = cz
                n_fine = features.shape[0]
                n_coarse = coarse_features_np.shape[0]
                feats_bi = torch.cat([
                    features,
                    torch.from_numpy(coarse_features_np),
                ], dim=0)
                # Edges: fine-fine + coarse-coarse (offset) + bidirectional containment.
                c_idx, f_idx = _build_containment_edges(coords.numpy(), coarse_coords_np)
                edges = [edge_index.long()]
                if coarse_ei_np.size:
                    edges.append(torch.from_numpy(coarse_ei_np).long() + n_fine)
                if c_idx.size:
                    c_global = torch.from_numpy(c_idx).long() + n_fine
                    f_global = torch.from_numpy(f_idx).long()
                    edges.append(torch.stack([c_global, f_global]))
                    edges.append(torch.stack([f_global, c_global]))
                ei_bi = torch.cat(edges, dim=1)
                scale_mask = torch.cat([
                    torch.zeros(n_fine, dtype=torch.long),
                    torch.ones(n_coarse, dtype=torch.long),
                ])
                multiscale_payload = {
                    "features": feats_bi, "edge_index": ei_bi,
                    "scale_mask": scale_mask, "n_fine": n_fine,
                }

            if region_mode:
                from gncaf_dataset import slide_wsi_path as _wsi_path
                wsi_p = _wsi_path(entry)
                if not wsi_p or not Path(wsi_p).exists():
                    print(f"  skip {short_id}: missing WSI tif")
                    continue
                grid_class, n_selected, pred_tiles = cascade_one_slide_region(
                    stage1, stage2, features, coords, edge_index, thr, device,
                    wsi_path=wsi_p,
                    s2_batch=int(cfg.get("region_s2_batch", 8)),
                    min_component_size=int(cfg.get("min_component_size", 2)),
                    closing_iters=int(cfg.get("closing_iters", 1)),
                    multiscale_payload=multiscale_payload,
                )
            elif neighborhood_mode:
                grid_class, n_selected, pred_tiles = cascade_one_slide_neighborhood(
                    stage1, stage2, features, coords, edge_index, thr, device,
                    s2_batch=int(cfg.get("nbhd_s2_batch", 16)),
                    min_component_size=int(cfg.get("min_component_size", 2)),
                    closing_iters=int(cfg.get("closing_iters", 1)),
                )
            else:
                top_k_frac = cfg.get("top_k_frac", None)
                top_k_abstain_thr = cfg.get("top_k_abstain_thr", None)
                grid_class, n_selected, pred_tiles = cascade_one_slide(
                    stage1, stage2, features, coords, edge_index, thr, device,
                    s2_batch=cfg.s2_batch,
                    top_k_frac=top_k_frac,
                    min_top_k=int(cfg.get("min_top_k", 10)),
                    top_k_abstain_thr=top_k_abstain_thr,
                )
            t_total = time.time() - t0

            # Patch-grid dice (coarse, what we had before — kept for backward compat).
            if is_gt_negative:
                # No annotation file ⇒ slide is GT-negative (no TLS, no GC).
                # Synthesise an all-zero patch grid the right shape so the
                # rest of the pipeline scores predictions as FPs cleanly.
                target_grid = np.zeros_like(grid_class, dtype=np.int64)
            else:
                target_grid = patch_grid_from_mask_cache(cache, coords, cfg.upsample_factor)
            tls_d = dice_score(grid_class, target_grid, 1)
            gc_d = dice_score(grid_class, target_grid, 2)

            # Per-pixel aggregate dice over selected patches' native masks.
            # GT-negatives have no per-patch GT lookup → every selected
            # tile gets a zero GT (all selections are FPs).
            slide_lookup = patch_mask_lookup.get(short_id, {})
            for i, tile in pred_tiles.items():
                gt = slide_lookup.get(int(i))
                if gt is None:
                    gt = np.zeros((256, 256), dtype=np.uint8)
                else:
                    gt = np.asarray(gt)
                for cls in (1, 2):
                    p = (tile == cls)
                    t = (gt == cls)
                    agg[cls]["inter"] += int((p & t).sum())
                    agg[cls]["denom"] += int(p.sum() + t.sum())

            min_size = int(cfg.get("min_component_size", 1))
            close_iters = int(cfg.get("closing_iters", 0))
            n_tls_pred = count_components_filtered(grid_class, 1, min_size, close_iters)
            n_gc_pred = count_components_filtered(grid_class, 2, min_size, close_iters)
            if is_gt_negative:
                gt_n_tls, gt_n_gc = 0, 0
            else:
                gt_n_tls, gt_n_gc = gt_counts.get(short_id, (
                    count_components(target_grid, 1),
                    count_components(target_grid, 2),
                ))
            per_slide.append({
                "slide_id": short_id, "cancer_type": entry["cancer_type"],
                "tls_dice": tls_d, "gc_dice": gc_d, "mDice": (tls_d + gc_d) / 2.0,
                "n_tls_pred": n_tls_pred,
                "n_tls_true": count_components(target_grid, 1),
                "n_gc_pred": n_gc_pred,
                "n_gc_true": count_components(target_grid, 2),
                # Metadata ground-truth (gold standard for counting).
                "gt_n_tls": gt_n_tls, "gt_n_gc": gt_n_gc,
                "gt_negative": bool(is_gt_negative),
                "n_selected": n_selected, "n_total": features.shape[0],
                "t_total": t_total,
            })
            n_selected_total += n_selected
            n_total_total += features.shape[0]
            if (k + 1) % 20 == 0:
                print(f"  [{k + 1}/{len(val_entries)}] processed")

        if not per_slide:
            continue
        mD = float(np.mean([r["mDice"] for r in per_slide]))
        tls_m = float(np.mean([r["tls_dice"] for r in per_slide]))
        gc_m = float(np.mean([r["gc_dice"] for r in per_slide]))
        # Per-pixel aggregate dice over selected patches' 256×256 native masks.
        eps = 1e-6
        tls_pix = (2 * agg[1]["inter"] + eps) / (agg[1]["denom"] + eps)
        gc_pix = (2 * agg[2]["inter"] + eps) / (agg[2]["denom"] + eps)
        mD_pix = (tls_pix + gc_pix) / 2.0
        # Metadata ground truth (gold standard).
        gt_tls = [r["gt_n_tls"] for r in per_slide]
        gt_gc = [r["gt_n_gc"] for r in per_slide]
        pred_tls = [r["n_tls_pred"] for r in per_slide]
        pred_gc = [r["n_gc_pred"] for r in per_slide]
        tls_sp, _ = spearmanr(gt_tls, pred_tls)
        gc_sp, _ = spearmanr(gt_gc, pred_gc) if any(gt_gc) else (0.0, 0.0)
        from sklearn.metrics import mean_absolute_error
        tls_mae = float(mean_absolute_error(gt_tls, pred_tls))
        gc_mae = float(mean_absolute_error(gt_gc, pred_gc))
        per_slide_t = float(np.mean([r["t_total"] for r in per_slide]))
        sel_frac = n_selected_total / max(1, n_total_total)
        print(f"\n  Results ({len(per_slide)} slides, threshold={thr}):")
        print(f"    [patch-grid] mDice={mD:.4f}  TLS={tls_m:.4f}  GC={gc_m:.4f}")
        print(f"    [pixel-agg]  mDice={mD_pix:.4f}  TLS={tls_pix:.4f}  GC={gc_pix:.4f}")
        print(f"    [counts vs gt]  TLS sp={tls_sp:.3f} mae={tls_mae:.2f}  "
              f"GC sp={gc_sp:.3f} mae={gc_mae:.2f}")
        print(f"    [post-proc] min_size={min_size}, closing_iters={close_iters}")
        print(f"    {per_slide_t:.2f}s/slide  selected {n_selected_total}/{n_total_total} "
              f"({100 * sel_frac:.1f}%)")
        for ct in sorted({r["cancer_type"] for r in per_slide}):
            sub = [r for r in per_slide if r["cancer_type"] == ct]
            print(f"    {ct} ({len(sub)}): TLS={np.mean([r['tls_dice'] for r in sub]):.3f} "
                  f"GC={np.mean([r['gc_dice'] for r in sub]):.3f}")
        rows.append({
            "threshold": thr, "mDice": mD, "tls_dice": tls_m, "gc_dice": gc_m,
            "mDice_pix": mD_pix, "tls_dice_pix": tls_pix, "gc_dice_pix": gc_pix,
            "tls_sp": tls_sp, "gc_sp": gc_sp,
            "tls_mae": tls_mae, "gc_mae": gc_mae,
            "min_size": min_size, "closing_iters": close_iters,
            "n_selected": n_selected_total,
            "n_total": n_total_total, "s_per_slide": per_slide_t,
        })
        per_slide_by_thr[float(thr)] = per_slide
        if run is not None:
            run.log({
                "threshold": thr, "mDice": mD, "tls_dice": tls_m, "gc_dice": gc_m,
                "tls_sp": tls_sp, "gc_sp": gc_sp, "selected_frac": sel_frac,
                "s_per_slide": per_slide_t,
            })

    print("\n\nSummary:")
    print(f"  {'thr':>6} {'mDice':>7} {'TLS d':>7} {'GC d':>7} {'TLS sp':>7} {'GC sp':>7} {'sel%':>6}")
    for r in rows:
        sel_pct = 100 * r["n_selected"] / max(1, r["n_total"])
        print(f"  {r['threshold']:>6.2f} {r['mDice']:>7.4f} {r['tls_dice']:>7.4f} "
              f"{r['gc_dice']:>7.4f} {r['tls_sp']:>7.3f} {r['gc_sp']:>7.3f} {sel_pct:>5.2f}%")

    (out_dir / "cascade_results.json").write_text(json.dumps(rows, indent=2))
    if per_slide_by_thr:
        (out_dir / "cascade_per_slide.json").write_text(
            json.dumps(per_slide_by_thr, indent=2, default=str)
        )
    if run is not None:
        # Pick best mDice threshold for summary.
        if rows:
            best = max(rows, key=lambda r: r["mDice"])
            run.summary["best_mDice"] = best["mDice"]
            run.summary["best_threshold"] = best["threshold"]
            run.summary["best_gc_dice"] = best["gc_dice"]
        run.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
