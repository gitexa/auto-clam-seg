"""Verify the GNCAF reconstruction:
  1. Synthetic forward pass — shape correctness without touching disk.
  2. Real-data forward pass on one slide — confirms data path + model
     wiring agree on tensor shapes and dtypes.
  3. Backward + optimizer step — confirms the loss has gradients.

Run:
    /home/ubuntu/profile-clam/.venv/bin/python verify_gncaf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gncaf_dataset import GNCAFSlideDataset, build_gncaf_split  # noqa: E402
from gncaf_model import GNCAF, model_summary  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 3  # bg, TLS, GC (matches our HookNet labels)


def synthetic_forward() -> None:
    print("=== 1. Synthetic forward (no disk) ===")
    model = GNCAF(in_features=1536, dim=384, n_classes=N_CLASSES).to(DEVICE)
    summary = model_summary(model)
    print({k: f"{v:,}" for k, v in summary.items()})

    B, N, E = 4, 200, 600
    target_rgb = torch.randn(B, 3, 256, 256, device=DEVICE)
    all_features = torch.randn(N, 1536, device=DEVICE)
    target_idx = torch.randint(0, N, (B,), device=DEVICE)
    edge_index = torch.randint(0, N, (2, E), device=DEVICE)
    y = model(target_rgb, all_features, target_idx, edge_index)
    expected = (B, N_CLASSES, 256, 256)
    assert tuple(y.shape) == expected, f"Got {tuple(y.shape)}, expected {expected}"
    print(f"  output shape OK: {tuple(y.shape)}\n")


def real_data_forward(max_targets: int = 4, train_step: bool = True) -> None:
    print("=== 2. Real-data forward (one slide) ===")
    train, _val = build_gncaf_split()
    ds = GNCAFSlideDataset(train, max_targets=max_targets, neg_per_pos=1.0,
                            rng_seed=0)
    print(f"  Slides with WSI+mask+features: {len(ds)}")
    if len(ds) == 0:
        print("  No usable slides!"); return

    t0 = time.time()
    sample = ds[0]
    print(f"  Loaded slide '{sample['slide_id'][:30]}' in {time.time() - t0:.1f}s")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {tuple(v.shape)} {v.dtype}")

    model = GNCAF(in_features=1536, dim=384, n_classes=N_CLASSES).to(DEVICE)

    target_rgb = sample["target_rgb"].to(DEVICE)
    all_features = sample["features"].to(DEVICE)
    target_idx = sample["target_idx"].to(DEVICE)
    edge_index = sample["edge_index"].to(DEVICE)
    target_mask = sample["target_mask"].to(DEVICE)

    t0 = time.time()
    logits = model(target_rgb, all_features, target_idx, edge_index)
    fwd_t = time.time() - t0
    print(f"\n  forward: {tuple(logits.shape)} in {fwd_t:.2f}s")
    expected = (target_rgb.shape[0], N_CLASSES, 256, 256)
    assert tuple(logits.shape) == expected, f"Got {tuple(logits.shape)}, expected {expected}"

    if train_step:
        loss = F.cross_entropy(logits, target_mask)
        print(f"  CE loss: {float(loss):.4f}")
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        t0 = time.time()
        opt.zero_grad()
        loss.backward()
        opt.step()
        bwd_t = time.time() - t0
        # Confirm grads flowed.
        n_with_grad = sum(1 for p in model.parameters()
                           if p.grad is not None and p.grad.abs().sum() > 0)
        n_total = sum(1 for _ in model.parameters())
        print(f"  backward + step: {bwd_t:.2f}s")
        print(f"  params with non-zero grad: {n_with_grad}/{n_total}")
        assert n_with_grad == n_total, "some params received no gradient — check forward path"
    print()


def main() -> None:
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    synthetic_forward()
    real_data_forward(max_targets=4)
    print("All checks passed ✓")


if __name__ == "__main__":
    main()
