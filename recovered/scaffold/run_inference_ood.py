"""GARS — out-of-distribution slide inference.

Self-contained inference for a folder of WSI slides. Designed to run
on another machine with minimal setup:

    Required (copy to target machine):
      - this script
      - gncaf_transunet_model.py        (for GNCAF inference)
      - region_decoder_model.py         (for v3.37 cascade Stage 2)
      - train_gars_stage1.py            (for v3.37 cascade Stage 1)
      - eval_gars_cascade.py            (provides cascade_one_slide_region)
      - the model checkpoints (.pt files)
      - requirements: torch, torch_geometric, tifffile, zarr, imagecodecs,
        timm (optional), scipy, scikit-learn, numpy, pandas

    Required per-slide on the target machine:
      - WSI .tif/.svs file
      - precomputed UNI-v2 features in a zarr store with `features` and
        `coords` and `graph_edges_1hop` (or we build edges from coords)

Usage:
    python run_inference_ood.py \\
        --slides_dir /data/ood_wsis/ \\
        --features_dir /data/ood_uni_features/ \\
        --out_dir ./predictions/ \\
        --model cascade \\
        --stage1_ckpt path/to/stage1.pt \\
        --stage2_region_ckpt path/to/region.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
import tifffile
from scipy.ndimage import binary_closing, label

PATCH_SIZE = 256


# ─── Per-slide IO ─────────────────────────────────────────────────────


def open_wsi(path: str, level: int = 0):
    """Open a TIFF/SVS slide as a zarr-backed array at given level."""
    return zarr.open(tifffile.imread(path, aszarr=True, level=level), mode="r")


def read_rgb_tile(wsi_z, x: int, y: int, size: int = 256) -> np.ndarray:
    h, w = wsi_z.shape[:2]
    x1, y1 = min(x + size, w), min(y + size, h)
    tile = np.asarray(wsi_z[y:y1, x:x1])
    if tile.shape[:2] != (size, size):
        out = np.full((size, size, 3), 255, dtype=tile.dtype)
        out[: tile.shape[0], : tile.shape[1]] = tile
        tile = out
    return tile


# ImageNet normalisation, used by both models.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalise_rgb(tile: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(tile).float().permute(2, 0, 1) / 255.0
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


def load_features_and_graph(zarr_path: str):
    g = zarr.open(zarr_path, mode="r")
    features = np.asarray(g["features"][:], dtype=np.float32)
    coords = np.asarray(g["coords"][:], dtype=np.int64)
    if "graph_edges_1hop" in g:
        edges = np.asarray(g["graph_edges_1hop"][:], dtype=np.int64)
    elif "edge_index" in g:
        edges = np.asarray(g["edge_index"][:], dtype=np.int64)
    else:
        edges = build_4conn_edges(coords)
    return features, coords, edges


def build_4conn_edges(coords: np.ndarray) -> np.ndarray:
    """Build 4-connectivity (N, S, E, W) edges from patch coords."""
    cell = coords // PATCH_SIZE
    cell -= cell.min(axis=0, keepdims=True)
    pos_to_idx: dict[tuple[int, int], int] = {
        (int(c[0]), int(c[1])): i for i, c in enumerate(cell)}
    src, dst = [], []
    for i, (cx, cy) in enumerate(cell):
        cx, cy = int(cx), int(cy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((cx + dx, cy + dy))
            if j is not None:
                src.append(i); dst.append(j)
    return np.stack([np.array(src, dtype=np.int64),
                     np.array(dst, dtype=np.int64)])


# ─── Cascade (v3.37) inference ────────────────────────────────────────


def run_cascade(
    stage1, stage2_region, features, coords, edge_index,
    wsi_path: str, device: torch.device,
    threshold: float = 0.5, s2_batch: int = 8,
    min_component_size: int = 2, closing_iters: int = 1,
):
    """v3.37 RegionDecoderCascade inference. Returns:
        slide_argmax: (H_g*256, W_g*256) uint8 — pixel-level prediction
        instance_counts: dict with 'tls' and 'gc' instance counts
        per_class_counts: dict with patch-grid component counts
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_gars_cascade import cascade_one_slide_region
    feats = torch.from_numpy(features).float()
    co = torch.from_numpy(coords).float()
    ei = torch.from_numpy(edge_index).long()
    grid_class, n_sel, pred_tiles, cell_probs = cascade_one_slide_region(
        stage1, stage2_region, feats, co, ei,
        threshold, device, wsi_path=wsi_path, s2_batch=s2_batch,
        min_component_size=min_component_size, closing_iters=closing_iters,
        return_cell_probs=True,
    )
    H, W = grid_class.shape
    # Build slide-pixel prediction from cell_probs.
    slide_argmax = np.zeros((H * PATCH_SIZE, W * PATCH_SIZE), dtype=np.uint8)
    for (gy, gx), cp in cell_probs.items():
        slide_argmax[gy * PATCH_SIZE:(gy + 1) * PATCH_SIZE,
                     gx * PATCH_SIZE:(gx + 1) * PATCH_SIZE] = cp.argmax(axis=0).astype(np.uint8)
    counts = {}
    for cls_name, cls_id in (("tls", 1), ("gc", 2)):
        binary = (grid_class == cls_id)
        if closing_iters > 0:
            binary = binary_closing(binary, iterations=closing_iters)
        lab, _ = label(binary)
        sizes = np.bincount(lab.ravel())[1:] if lab.max() else np.array([])
        n = int((sizes >= min_component_size).sum()) if sizes.size else 0
        counts[cls_name] = n
    return slide_argmax, counts, n_sel


# ─── GNCAF (TransUNet) inference ─────────────────────────────────────


@torch.no_grad()
def run_gncaf(
    model, features, coords, edge_index,
    wsi_path: str, device: torch.device,
    batch_size: int = 16, min_component_size: int = 2, closing_iters: int = 1,
):
    """GNCAFPixelDecoder inference over all patches in the slide.
    Returns the same triple as run_cascade.
    """
    feats = torch.from_numpy(features).float().to(device)
    ei = torch.from_numpy(edge_index).long().to(device)
    n = features.shape[0]
    # Pre-compute graph context once.
    node_ctx = model.gcn(feats, ei)
    wsi_z = open_wsi(wsi_path, level=0)
    grid_x = (coords[:, 0] // PATCH_SIZE).astype(np.int64)
    grid_y = (coords[:, 1] // PATCH_SIZE).astype(np.int64)
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    H = int(grid_y.max()) + 1; W = int(grid_x.max()) + 1
    pred_grid = np.zeros((H, W), dtype=np.uint8)
    slide_argmax = np.zeros((H * PATCH_SIZE, W * PATCH_SIZE), dtype=np.uint8)

    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        rgb_batch = []
        for i in idx:
            tile = read_rgb_tile(wsi_z, int(coords[i, 0]), int(coords[i, 1]))
            rgb_batch.append(normalise_rgb(tile))
        rgb_t = torch.stack(rgb_batch).to(device)
        target_idx = torch.tensor(idx, dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens, skips = model.encoder(rgb_t)
            local_ctx = node_ctx[target_idx]
            x = torch.cat([local_ctx.unsqueeze(1), tokens], dim=1)
            for blk in model.fusion:
                x = blk(x)
            x = x[:, 1:].transpose(1, 2).reshape(rgb_t.shape[0], -1, 16, 16)
            x = model.decoder(x, skips)
            logits = model.head_seg(x)
        argmax = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for k, i in enumerate(idx):
            tile = argmax[k]
            slide_argmax[grid_y[i] * PATCH_SIZE:(grid_y[i] + 1) * PATCH_SIZE,
                         grid_x[i] * PATCH_SIZE:(grid_x[i] + 1) * PATCH_SIZE] = tile
            n_gc = int((tile == 2).sum()); n_tls = int((tile == 1).sum())
            pred_grid[grid_y[i], grid_x[i]] = 2 if n_gc > 0 else (1 if n_tls > 0 else 0)
    counts = {}
    for cls_name, cls_id in (("tls", 1), ("gc", 2)):
        binary = (pred_grid == cls_id)
        if closing_iters > 0:
            binary = binary_closing(binary, iterations=closing_iters)
        lab, _ = label(binary)
        sizes = np.bincount(lab.ravel())[1:] if lab.max() else np.array([])
        m = int((sizes >= min_component_size).sum()) if sizes.size else 0
        counts[cls_name] = m
    return slide_argmax, counts, n


# ─── Model loading ────────────────────────────────────────────────────


def load_cascade_models(stage1_ckpt: str, stage2_region_ckpt: str,
                         device: torch.device):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_gars_cascade import load_stage1, load_stage2_region
    s1 = load_stage1(stage1_ckpt, device)
    s2 = load_stage2_region(stage2_region_ckpt, device)
    return s1, s2


def load_gncaf_model(ckpt: str, device: torch.device):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gncaf_transunet_model import GNCAFPixelDecoder
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}).get("model", {})
    model = GNCAFPixelDecoder(
        hidden_size=cfg.get("hidden_size", 768),
        n_classes=cfg.get("n_classes", 3),
        n_encoder_layers=cfg.get("n_encoder_layers", 6),
        n_heads=cfg.get("n_heads", 12),
        mlp_dim=cfg.get("mlp_dim", 3072),
        n_hops=cfg.get("n_hops", 3),
        n_fusion_layers=cfg.get("n_fusion_layers", 1),
        feature_dim=cfg.get("feature_dim", 1536),
        dropout=0.0,
    )
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device).eval()


# ─── Main inference loop ─────────────────────────────────────────────


def find_features_path(features_dir: str, slide_basename: str) -> str | None:
    """Look for `<slide_basename>.zarr` or `<slide_basename>_complete.zarr`
    under features_dir. Returns the first match or None."""
    fdir = Path(features_dir)
    candidates = [
        fdir / f"{slide_basename}.zarr",
        fdir / f"{slide_basename}_complete.zarr",
    ]
    candidates += list(fdir.glob(f"{slide_basename}*.zarr"))
    for c in candidates:
        if c.is_dir():
            return str(c)
    return None


def main():
    ap = argparse.ArgumentParser(
        description="GARS OOD inference — run cascade or GNCAF on a folder of WSIs."
    )
    ap.add_argument("--slides_dir", required=True,
                    help="Directory containing .tif/.svs slides")
    ap.add_argument("--features_dir", required=True,
                    help="Directory containing precomputed UNI-v2 zarrs (one per slide)")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for predicted masks + counts CSV")
    ap.add_argument("--model", choices=("cascade", "gncaf"), default="cascade",
                    help="Which architecture to run")
    ap.add_argument("--stage1_ckpt",
                    help="(cascade) Stage 1 checkpoint .pt")
    ap.add_argument("--stage2_region_ckpt",
                    help="(cascade) Stage 2 region-decoder checkpoint .pt")
    ap.add_argument("--gncaf_ckpt",
                    help="(gncaf) GNCAFPixelDecoder checkpoint .pt")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Stage 1 threshold (cascade only)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--save_masks", action="store_true",
                    help="Save predicted masks as TIF (large files)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    if args.model == "cascade":
        if not (args.stage1_ckpt and args.stage2_region_ckpt):
            ap.error("--model cascade requires --stage1_ckpt and --stage2_region_ckpt")
        s1, s2 = load_cascade_models(args.stage1_ckpt, args.stage2_region_ckpt, device)
        print(f"Loaded cascade: stage1 + stage2_region")
    else:
        if not args.gncaf_ckpt:
            ap.error("--model gncaf requires --gncaf_ckpt")
        gncaf = load_gncaf_model(args.gncaf_ckpt, device)
        print(f"Loaded GNCAFPixelDecoder ({sum(p.numel() for p in gncaf.parameters()):,} params)")

    slides_dir = Path(args.slides_dir)
    slide_paths = sorted(list(slides_dir.glob("*.tif")) + list(slides_dir.glob("*.svs")))
    if not slide_paths:
        print(f"No slides found in {slides_dir}"); return
    print(f"Found {len(slide_paths)} slides\n")

    rows = []
    for i, slide_path in enumerate(slide_paths):
        slide_id = slide_path.stem
        print(f"[{i + 1}/{len(slide_paths)}] {slide_id}")
        feats_path = find_features_path(args.features_dir, slide_id)
        if feats_path is None:
            print(f"  SKIP — no UNI features under {args.features_dir}")
            rows.append({"slide_id": slide_id, "status": "no_features"})
            continue
        try:
            features, coords, edges = load_features_and_graph(feats_path)
            t0 = time.time()
            if args.model == "cascade":
                slide_argmax, counts, n_sel = run_cascade(
                    s1, s2, features, coords, edges,
                    str(slide_path), device,
                    threshold=args.threshold, s2_batch=args.batch_size,
                )
            else:
                slide_argmax, counts, n_sel = run_gncaf(
                    gncaf, features, coords, edges,
                    str(slide_path), device, batch_size=args.batch_size,
                )
            t = time.time() - t0
            row = {
                "slide_id": slide_id,
                "n_patches": int(features.shape[0]),
                "n_decoded": int(n_sel),
                "tls_count": counts["tls"],
                "gc_count": counts["gc"],
                "s_per_slide": float(t),
                "status": "ok",
            }
            rows.append(row)
            print(f"  TLS={counts['tls']}, GC={counts['gc']}, decoded={n_sel}, t={t:.1f}s")
            if args.save_masks:
                mask_out = out_dir / f"{slide_id}_pred.tif"
                tifffile.imwrite(str(mask_out), slide_argmax, compression="zlib")
                print(f"  saved {mask_out.name}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}"); traceback.print_exc()
            rows.append({"slide_id": slide_id, "status": f"error:{e}"})

    # Counts CSV.
    import csv
    csv_path = out_dir / "counts.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "slide_id", "n_patches", "n_decoded", "tls_count", "gc_count",
            "s_per_slide", "status"])
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})
    print(f"\nWrote {csv_path}")

    # JSON summary.
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "model": args.model,
        "n_slides": len(slide_paths),
        "n_processed": sum(1 for r in rows if r.get("status") == "ok"),
        "rows": rows,
    }, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
