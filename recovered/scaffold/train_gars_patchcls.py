"""3-class patch classifier with graph context — NO pixel-level upsampling.

Replaces the cascade (Stage 1 + Stage 2 decoder). One forward pass per slide
classifies every patch into {bg=0, TLS=1, GC=2} from its UNI-v2 feature
plus 1-hop graph context. The decoder + 16→256 upsampling is gone.

Per-slide training:
    1. UNI-v2 features (N, 1536) loaded from zarr
    2. 1-hop edge_index from zarr
    3. Forward: GATv2 × n_hops over features → (N, hidden) → Linear → (N, 3)
    4. Per-patch class label: 2 if any GC pixels, else 1 if (TLS>=N_pix), else 0
    5. CE loss over all N patches (no per-pixel)
    6. Sample bg patches at bg_per_pos to balance.

Per-slide validation: same forward; evaluate patch-grid dice (the only
metric that doesn't need pixel masks) + counting Spearman on connected
components of the predicted class grid.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, label
from scipy.stats import spearmanr
from torch_geometric.nn import GATv2Conv

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_segmentation as ps
from tls_patch_dataset import build_tls_patch_dataset

PATCH_SIZE = 256


class GraphPatchClassifier(nn.Module):
    """Per-patch 3-class classifier with k-hop GATv2 context aggregation.

    Same graph backbone as Stage 1 (`GraphTLSDetector`) but outputs 3
    logits per patch instead of 1 (binary TLS positive).
    """

    def __init__(self, in_dim=1536, hidden_dim=256, n_hops=3,
                 n_classes=3, gat_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm0 = nn.LayerNorm(hidden_dim)
        head_dim = hidden_dim // gat_heads
        assert hidden_dim % gat_heads == 0
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, head_dim, heads=gat_heads,
                      concat=True, dropout=dropout)
            for _ in range(n_hops)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_hops)
        ])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features, edge_index):
        x = self.norm0(self.proj(features))
        for gat, n in zip(self.gat_layers, self.gat_norms):
            x = n(gat(x, edge_index) + x)
        return self.head(x)


def patch_labels_from_mask_lookup(coords, slide_lookup, min_tls_pixels=4096):
    """Per-patch class label from precomputed native 256×256 patch masks.

    Class 2 if any GC pixel; else 1 if (mask > 0).sum() >= min_tls_pixels;
    else 0 (background).
    """
    n = coords.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for k, t in slide_lookup.items():
        t = np.asarray(t)
        if (t == 2).any():
            out[int(k)] = 2
        elif int((t == 1).sum()) >= min_tls_pixels:
            out[int(k)] = 1
    return out


def build_grid(coords):
    grid_x = (coords[:, 0] / PATCH_SIZE).astype(np.int64)
    grid_y = (coords[:, 1] / PATCH_SIZE).astype(np.int64)
    grid_x -= grid_x.min(); grid_y -= grid_y.min()
    H, W = int(grid_y.max()) + 1, int(grid_x.max()) + 1
    return grid_x, grid_y, H, W


def patch_grid_from_class_per_patch(class_per_patch, coords):
    grid_x, grid_y, H, W = build_grid(coords)
    grid = np.zeros((H, W), dtype=np.int64)
    for k, (gx, gy) in enumerate(zip(grid_x, grid_y)):
        grid[gy, gx] = int(class_per_patch[k])
    return grid


def dice_score(pred, target, cls, eps=1e-6):
    p = (pred == cls); t = (target == cls)
    inter = float((p & t).sum())
    denom = float(p.sum() + t.sum())
    return (2 * inter + eps) / (denom + eps)


def count_components_filtered(grid, cls, min_size=2, closing_iters=1):
    binary = (grid == cls)
    if closing_iters > 0:
        binary = binary_closing(binary, iterations=closing_iters)
    lab, n = label(binary)
    if min_size > 1 and n > 0:
        sizes = np.bincount(lab.ravel())[1:]
        n = int((sizes >= min_size).sum())
    return n


def load_slide_features(zarr_path):
    g = zarr.open(zarr_path, mode="r")
    f = torch.from_numpy(np.asarray(g["features"][:])).float()
    c = torch.from_numpy(np.asarray(g["coords"][:])).float()
    if "graph_edges_1hop" in g:
        e = torch.from_numpy(np.asarray(g["graph_edges_1hop"][:])).long()
    elif "edge_index" in g:
        e = torch.from_numpy(np.asarray(g["edge_index"][:])).long()
    else:
        e = None
    return f, c, e


def build_patch_lookup():
    print("Loading patch cache for per-patch labels...")
    bundle = build_tls_patch_dataset(
        cache_path="/home/ubuntu/local_data/tls_patch_dataset_min4096.pt",
        verbose=False)
    bundle_masks = bundle["masks"]
    out: dict[str, dict[int, np.ndarray]] = {}
    for ci, sid in enumerate(bundle["slide_ids"]):
        short = sid.split(".")[0]
        pi = int(bundle["patch_idx"][ci])
        out.setdefault(short, {})[pi] = np.asarray(bundle_masks[ci])
    return out


@torch.no_grad()
def validate(model, val_entries, patch_lookup, device, gt_counts,
             min_tls_pixels=4096, post_min_size=2, post_close=1):
    model.eval()
    agg = {1: [0, 0], 2: [0, 0]}
    per_slide = []
    for e in val_entries:
        sid = e["slide_id"].split(".")[0]
        cache = patch_lookup.get(sid, {})
        f, c, ei = load_slide_features(e["zarr_path"])
        if ei is None:
            continue
        coords = c.numpy()
        labels = patch_labels_from_mask_lookup(coords, cache, min_tls_pixels)
        target_grid = patch_grid_from_class_per_patch(labels, coords)
        f = f.to(device); ei = ei.to(device)
        logits = model(f, ei)
        pred = logits.argmax(dim=1).cpu().numpy()
        pred_grid = patch_grid_from_class_per_patch(pred, coords)
        for cls in (1, 2):
            p = (pred_grid == cls); t = (target_grid == cls)
            agg[cls][0] += int((p & t).sum())
            agg[cls][1] += int(p.sum() + t.sum())
        n_tls_pred = count_components_filtered(pred_grid, 1, post_min_size, post_close)
        n_gc_pred = count_components_filtered(pred_grid, 2, post_min_size, post_close)
        gt = gt_counts.get(sid, (0, 0))
        per_slide.append({
            "sid": sid,
            "n_tls_pred": n_tls_pred, "n_gc_pred": n_gc_pred,
            "gt_n_tls": gt[0], "gt_n_gc": gt[1],
        })
    eps = 1e-6
    tls_d = (2 * agg[1][0] + eps) / (agg[1][1] + eps)
    gc_d = (2 * agg[2][0] + eps) / (agg[2][1] + eps)
    gt_t = [r["gt_n_tls"] for r in per_slide]
    gt_g = [r["gt_n_gc"] for r in per_slide]
    pr_t = [r["n_tls_pred"] for r in per_slide]
    pr_g = [r["n_gc_pred"] for r in per_slide]
    tls_sp, _ = spearmanr(gt_t, pr_t) if len(gt_t) else (0.0, 0.0)
    gc_sp, _ = spearmanr(gt_g, pr_g) if any(gt_g) else (0.0, 0.0)
    from sklearn.metrics import mean_absolute_error
    return {
        "tls_dice_grid": tls_d, "gc_dice_grid": gc_d,
        "mDice_grid": (tls_d + gc_d) / 2.0,
        "tls_sp": tls_sp, "gc_sp": gc_sp,
        "tls_mae": float(mean_absolute_error(gt_t, pr_t)),
        "gc_mae": float(mean_absolute_error(gt_g, pr_g)),
        "n_slides": len(per_slide),
    }


@hydra.main(version_base=None, config_path="configs/patchcls", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ps.set_seed(cfg.seed)
    entries = ps.build_slide_entries()
    fp, _ = ps.create_splits(entries, k_folds=1, seed=cfg.seed)
    val_entries = [e for e in fp[0] if e.get("mask_path")]
    train_entries = [e for e in fp[1] if e.get("mask_path")]
    print(f"Train: {len(train_entries)} slides, Val: {len(val_entries)} slides")

    patch_lookup = build_patch_lookup()

    import pandas as pd
    df = pd.read_csv(ps.META_CSV)
    gt_counts = {str(r["slide_id"]).split(".")[0]: (int(r["tls_num"]), int(r["gc_num"]))
                 for _, r in df.iterrows()}

    model = GraphPatchClassifier(
        in_dim=1536, hidden_dim=cfg.model.hidden_dim,
        n_hops=cfg.model.n_hops, n_classes=3,
        gat_heads=cfg.model.gat_heads, dropout=cfg.model.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GraphPatchClassifier params: {n_params:,}")

    cw = torch.tensor(cfg.train.class_weights, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 1e-2,
    )

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

    best_mDice = -1.0; best_epoch = -1; patience_left = cfg.train.patience
    for ep in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0; ep_n = 0
        np.random.shuffle(train_entries)
        for e in train_entries:
            sid = e["slide_id"].split(".")[0]
            cache = patch_lookup.get(sid, {})
            f, c, ei = load_slide_features(e["zarr_path"])
            if ei is None:
                continue
            coords = c.numpy()
            labels = patch_labels_from_mask_lookup(coords, cache, cfg.train.min_tls_pixels)
            f = f.to(device); ei = ei.to(device)
            logits = model(f, ei)
            target = torch.from_numpy(labels).long().to(device)
            loss = F.cross_entropy(logits, target, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss); ep_n += 1
        scheduler.step()
        train_t = time.time() - t0

        t0 = time.time()
        val = validate(model, val_entries, patch_lookup, device, gt_counts,
                       min_tls_pixels=cfg.train.min_tls_pixels)
        val_t = time.time() - t0

        is_best = val["mDice_grid"] > best_mDice
        msg = (f"EPOCH epoch={ep} train_loss={ep_loss / max(1,ep_n):.4f} "
               f"val_TLS_grid={val['tls_dice_grid']:.4f} "
               f"val_GC_grid={val['gc_dice_grid']:.4f} "
               f"val_mDice_grid={val['mDice_grid']:.4f} "
               f"val_TLS_sp={val['tls_sp']:.3f} "
               f"val_GC_sp={val['gc_sp']:.3f} "
               f"lr={opt.param_groups[0]['lr']:.2e} "
               f"train={train_t:.0f}s val={val_t:.0f}s"
               + (" BEST" if is_best else ""))
        print(msg)
        if run is not None:
            run.log({"epoch": ep, **val,
                     "train_loss": ep_loss / max(1, ep_n),
                     "lr": opt.param_groups[0]["lr"]})

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": ep, "val": val,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, out_dir / "last.pt")
        if is_best:
            best_mDice = val["mDice_grid"]; best_epoch = ep
            patience_left = cfg.train.patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep, "val": val,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, out_dir / "best_checkpoint.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop ep{ep} (best ep{best_epoch} mDice={best_mDice:.4f})")
                break

    final = {"best_mDice_grid": best_mDice, "best_epoch": best_epoch}
    (out_dir / "final_results.json").write_text(json.dumps(final, indent=2))
    if run is not None:
        run.summary["best_mDice_grid"] = best_mDice
        run.summary["best_epoch"] = best_epoch
        run.finish()
    print(f"Done. best mDice_grid={best_mDice:.4f} ep{best_epoch}")


if __name__ == "__main__":
    main()
