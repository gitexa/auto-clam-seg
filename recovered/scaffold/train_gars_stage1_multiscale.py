"""GARS Stage 1 v3.60 — multi-scale bipartite graph TLS detector.

Same training/eval flow as `train_gars_stage1.py` but uses
`MultiScaleSlideDataset` + `MultiScaleGraphTLSDetector` so each slide
becomes a bipartite graph of fine (256-px@20×) and coarse (512-px@20×)
nodes. The classifier head produces a logit per node; loss + metrics
are computed on FINE nodes only (256-px patch labels).

Hydra config: `configs/stage1` (reuses base) plus the
`train.multi_scale: true` knob and (optionally) a multi-scale model
file `model: gatv2_5hop_multiscale`.

Run:
    python train_gars_stage1_multiscale.py train.multi_scale=true \
      train.fold_idx=0 train.k_folds=5 \
      label=cascade_5fold_fold0_s1_multiscale
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/ubuntu/profile-clam")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_gars_stage1 import (  # reuse helpers
    _identity_collate,
    confusion_at_threshold,
    f1_from_confusion,
    make_slide_loader,
    patch_labels_from_mask,
)
from multiscale_dataset import MultiScaleSlideDataset
from multiscale_stage1_model import MultiScaleGraphTLSDetector


def run_split_multiscale(model, loader, optimizer, criterion, device,
                          train: bool, upsample_factor: int, patch_size: int):
    """Train/eval one epoch on multi-scale slides. Computes loss and
    confusion ONLY on fine nodes (scale_mask == 0).
    """
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            edge_index = batch["edge_index"]
            edge_index = edge_index.to(device, non_blocking=True) if edge_index is not None else None
            scale_mask = batch["scale_mask"].to(device, non_blocking=True)

            target = patch_labels_from_mask(
                batch["mask"], batch["coords"], patch_size, upsample_factor
            ).to(device, non_blocking=True)

            logits_all = model(features, edge_index, scale_mask)
            # Slice to fine nodes for loss / metrics.
            fine_logits = logits_all[scale_mask == 0]
            assert fine_logits.shape[0] == target.shape[0], (
                f"fine logits {fine_logits.shape} != target {target.shape}"
            )
            loss = criterion(fine_logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
            a, b, c, d = confusion_at_threshold(fine_logits.detach(), target)
            tp += a; fp += b; fn += c; tn += d
    rec, prec, f1 = f1_from_confusion(tp, fp, fn, tn)
    return {
        "loss": total_loss / max(1, n_batches),
        "recall": rec, "precision": prec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_selected": tp + fp,
        "n_total": tp + fp + fn + tn,
    }


@hydra.main(version_base=None, config_path="configs/stage1", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output dir: {out_dir}")

    import prepare_segmentation as ps
    from stage_features_to_local import LOCAL_ROOT, local_zarr_dirs

    if cfg.use_local_ssd != "never":
        local_dirs = local_zarr_dirs()
        all_present = all(Path(p).is_dir() and any(Path(p).iterdir())
                          for p in local_dirs.values())
        if all_present:
            ps.ZARR_DIRS = local_dirs
            print(f"Using locally-staged 256-px zarrs at {LOCAL_ROOT}")
        elif cfg.use_local_ssd == "always":
            raise RuntimeError(f"--use_local_ssd=always but local zarrs missing at {LOCAL_ROOT}")
        else:
            print(f"NFS 256-px zarrs in use (no local copy at {LOCAL_ROOT})")

    ps.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building dataset...")
    entries = ps.build_slide_entries()
    if cfg.train.data_fraction < 1.0:
        entries = ps.subsample_entries(entries, cfg.train.data_fraction, cfg.seed)
    fold_idx = int(cfg.train.get("fold_idx", 0))
    if cfg.train.k_folds == 1 and fold_idx == 0:
        folds_pair, _test = ps.create_splits(entries, k_folds=1, seed=cfg.seed)
        val_entries, train_entries = folds_pair[0], folds_pair[1]
    else:
        all_folds, _test = ps.create_splits(entries, k_folds=5, seed=cfg.seed)
        if fold_idx < 0 or fold_idx >= len(all_folds):
            raise ValueError(f"fold_idx={fold_idx} out of range")
        val_entries = all_folds[fold_idx]
        train_entries = [s for i, f in enumerate(all_folds) if i != fold_idx for s in f]
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val "
          f"(fold_idx={fold_idx}, k_folds={cfg.train.k_folds}, seed={cfg.seed})")

    print("Building mask cache...")
    mask_dict = ps.build_mask_cache(train_entries + val_entries, cfg.train.upsample_factor)

    base_train_ds = ps.TLSSegmentationDataset(
        train_entries, mask_dict, cfg.train.upsample_factor, patch_size=cfg.train.patch_size,
    )
    base_val_ds = ps.TLSSegmentationDataset(
        val_entries, mask_dict, cfg.train.upsample_factor, patch_size=cfg.train.patch_size,
    )
    train_ds = MultiScaleSlideDataset(base_train_ds)
    val_ds = MultiScaleSlideDataset(base_val_ds)

    (out_dir / "val_slides.json").write_text(json.dumps(
        [e["slide_id"] for e in val_entries], indent=2,
    ))
    (out_dir / "train_slides.json").write_text(json.dumps(
        [e["slide_id"] for e in train_entries], indent=2,
    ))

    nw = cfg.train.get("num_workers", 0)
    pf = cfg.train.get("prefetch_factor", 2)
    pw = cfg.train.get("persistent_workers", True)
    train_loader = make_slide_loader(train_ds, nw, pf, pw, shuffle=True)
    val_loader = make_slide_loader(val_ds, nw, pf, pw, shuffle=False)
    print(f"DataLoader: {nw} workers, prefetch={pf}, persistent={pw}")

    model = MultiScaleGraphTLSDetector(
        in_dim=cfg.model.in_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_hops=cfg.model.n_hops,
        gnn_type=cfg.model.gnn_type,
        dropout=cfg.model.dropout,
        gat_heads=cfg.model.gat_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 multi-scale ({cfg.model.gnn_type} {cfg.model.n_hops}-hop): "
          f"{n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=0.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.train.pos_weight, device=device))

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

    print(f"Training {len(train_entries)} slides, validating {len(val_entries)} slides\n")

    best_f1 = -1.0
    best_epoch = -1
    epochs_since_best = 0
    last_va = None
    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr = run_split_multiscale(model, train_loader, optimizer, criterion, device,
                                   train=True,
                                   upsample_factor=cfg.train.upsample_factor,
                                   patch_size=cfg.train.patch_size)
        train_t = time.time() - t0
        t0 = time.time()
        va = run_split_multiscale(model, val_loader, optimizer, criterion, device,
                                   train=False,
                                   upsample_factor=cfg.train.upsample_factor,
                                   patch_size=cfg.train.patch_size)
        val_t = time.time() - t0
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        last_va = va

        is_best = va["f1"] > best_f1
        marker = "BEST" if is_best else ""
        print(
            f"EPOCH epoch={epoch} train_loss={tr['loss']:.4f} "
            f"train_recall={tr['recall']:.3f} train_prec={tr['precision']:.3f} "
            f"train_f1={tr['f1']:.3f} "
            f"val_loss={va['loss']:.4f} val_recall={va['recall']:.3f} "
            f"val_prec={va['precision']:.3f} val_f1={va['f1']:.3f} "
            f"val_selected={va['n_selected']}/{va['n_total']} "
            f"lr={lr:.2e} train={train_t:.0f}s val={val_t:.0f}s {marker}",
            flush=True,
        )
        if run is not None:
            run.log({
                "epoch": epoch, "lr": lr,
                "train/loss": tr["loss"], "train/f1": tr["f1"],
                "train/recall": tr["recall"], "train/precision": tr["precision"],
                "val/loss": va["loss"], "val/f1": va["f1"],
                "val/recall": va["recall"], "val/precision": va["precision"],
                "val/tp": va["tp"], "val/fp": va["fp"],
                "val/fn": va["fn"], "val/tn": va["tn"],
                "val/n_selected": va["n_selected"], "val/n_total": va["n_total"],
                "best_f1": max(best_f1, va["f1"]),
            }, step=epoch)

        last_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_metrics": va,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_class": "MultiScaleGraphTLSDetector",
        }
        torch.save(last_payload, out_dir / "last.pt")

        if is_best:
            best_f1 = va["f1"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(last_payload, out_dir / "best_checkpoint.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.train.patience:
                print(f"Early stopping at epoch {epoch} (best={best_epoch}, f1={best_f1:.3f})",
                      flush=True)
                break

    print(f"Done. Best f1={best_f1:.3f} at epoch {best_epoch}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
