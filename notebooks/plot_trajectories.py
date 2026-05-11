"""Plot training trajectories across all available experiments.

Sources:
- /tmp/v3_*_train.log — stdout EPOCH lines (parsed)
- experiments/seg_v2.0_*/fold_0/incremental.json — per-epoch val metrics
- experiments/gncaf_pixel_*/best_checkpoint.pt — final best_mdice (lost ckpts; no trajectory)
- experiments/gars_gncaf_v3.*/wandb/latest-run/files/wandb-summary.json — final summary
"""
from __future__ import annotations
import json
import re
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

EXP = Path("/home/ubuntu/ahaas-persistent-std-tcga/experiments")
OUT = Path("/home/ubuntu/auto-clam-seg/notebooks/architectures")
OUT.mkdir(parents=True, exist_ok=True)

# Epoch-line regex from train_gncaf_transunet.py stdout
EPOCH_RE = re.compile(
    r"EPOCH epoch=(\d+).*?val_dice_tls=([0-9.]+).*?val_dice_gc=([0-9.]+).*?val_mDice=([0-9.]+)"
)


def parse_stdout_log(path: Path):
    """Parse trajectory from /tmp/v3_*_train.log."""
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "tls": float(m.group(2)),
                "gc": float(m.group(3)),
                "mdice": float(m.group(4)),
            })
    return rows if rows else None


def parse_seg_v2_incremental(d: Path):
    """Parse trajectory from seg_v2.0/fold_0/incremental.json.
    Field names: val_dice (TLS), val_iou; GC dice may not be saved per epoch.
    """
    p = d / "fold_0" / "incremental.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    rows = []
    for ep in j.get("epochs", []):
        tls = float(ep.get("val_dice", 0.0) or 0.0)
        gc = float(ep.get("dice_gc", ep.get("val_dice_gc", 0.0)) or 0.0)
        rows.append({
            "epoch": int(ep["epoch"]) + 1,
            "tls": tls,
            "gc": gc,
            "mdice": tls,  # seg_v2's val_dice is essentially TLS Dice
        })
    return rows


def parse_v3_65_from_ckpt(d: Path):
    """v3.65 stdout log is empty; use last.pt epoch + best_mdice as 1 point."""
    p = d / "last.pt"
    if not p.exists():
        return None
    o = torch.load(p, map_location="cpu", weights_only=False)
    vm = o.get("val_metrics", {})
    return [{
        "epoch": int(o.get("epoch", 0)),
        "tls": float(vm.get("dice_tls", 0)),
        "gc": float(vm.get("dice_gc", 0)),
        "mdice": float(vm.get("mDice", 0)),
    }]


def get_lost_ckpts():
    """Final best_mdice from saved ckpts (no trajectory available)."""
    results = []
    for d in sorted(EXP.glob("gncaf_pixel_*")):
        ck = d / "best_checkpoint.pt"
        if not ck.exists():
            continue
        try:
            o = torch.load(ck, map_location="cpu", weights_only=False)
            md = o.get("best_mdice")
            ep = o.get("epoch")
            if md is None or md < 0.55:  # filter low ones for plot readability
                continue
            results.append({
                "name": d.name.replace("gncaf_pixel_", "").rstrip("_")[:50],
                "epoch": int(ep) if ep is not None else 0,
                "mdice": float(md),
            })
        except Exception:
            continue
    return results


def main():
    # ── Per-epoch trajectories ──────────────────────────────────────────
    trajectories = {}
    for label, p in [
        ("GNCAF v3.63 (dual-σ, heavy)",  Path("/tmp/v3_63_train.log")),
        ("GNCAF v3.64 (dual-σ, tls_pw=1)", Path("/tmp/v3_64_train.log")),
        ("GNCAF v3.65 (dual-σ, simple)", Path("/tmp/v3_65_train.log")),
    ]:
        r = parse_stdout_log(p)
        if r:
            trajectories[label] = r
    # seg_v2 variants
    for label, d in [
        ("seg_v2.0 (tls_only)",  EXP / "seg_v2.0_tls_only_5fold_31aaec0c"),
        ("seg_v2.0 (dual)",      EXP / "seg_v2.0_dual_tls_gc_5fold_57e12399"),
    ]:
        r = parse_seg_v2_incremental(d)
        if r:
            trajectories[label] = r
    # v3.65 ckpt fallback ONLY if stdout log didn't have epochs (e.g. early in run)
    if "GNCAF v3.65 (dual-σ, simple)" not in trajectories:
        v365_dirs = sorted(EXP.glob("gars_gncaf_v3.65_*"))
        if v365_dirs:
            r = parse_v3_65_from_ckpt(v365_dirs[-1])
            if r:
                trajectories["GNCAF v3.65 (dual-σ, simple)"] = r

    print(f"Per-epoch trajectories: {len(trajectories)} runs")
    for k, v in trajectories.items():
        print(f"  {k}: {len(v)} epochs, best_mDice={max(r['mdice'] for r in v):.4f}")

    # ── Lost-ckpt final points (no trajectory) ───────────────────────────
    lost = get_lost_ckpts()
    print(f"Lost ckpts (final-only): {len(lost)}")

    # ── Figure: 3 panels (mDice, TLS, GC) ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    palette = {
        "GNCAF v3.63 (dual-σ, heavy)":          "#8b0000",
        "GNCAF v3.64 (dual-σ, tls_pw=1)":       "#cc6677",
        "GNCAF v3.65 (dual-σ, simple)":         "#aa3322",
        "GNCAF v3.65 (dual-σ, simple-loss)":    "#aa3322",
        "seg_v2.0 (tls_only)":                  "#7f7f7f",
        "seg_v2.0 (dual)":                       "#525252",
    }
    line_styles = {
        "GNCAF v3.63 (dual-σ, heavy)":          "-",
        "GNCAF v3.64 (dual-σ, tls_pw=1)":       "--",
        "GNCAF v3.65 (dual-σ, simple)":         "-",
        "GNCAF v3.65 (dual-σ, simple-loss)":    "-",
        "seg_v2.0 (tls_only)":                  "-",
        "seg_v2.0 (dual)":                       "-",
    }

    for ax, key, title in [
        (axes[0], "mdice", "val mDice"),
        (axes[1], "tls",   "val TLS Dice"),
        (axes[2], "gc",    "val GC Dice"),
    ]:
        for label, rows in trajectories.items():
            xs = [r["epoch"] for r in rows]
            ys = [r[key]    for r in rows]
            ax.plot(xs, ys,
                    color=palette.get(label, "#888"),
                    linestyle=line_styles.get(label, "-"),
                    linewidth=1.8, label=label, marker="o", markersize=3,
                    alpha=0.85)
            # Mark best epoch
            best_idx = int(np.argmax(ys))
            ax.scatter([xs[best_idx]], [ys[best_idx]], s=80,
                       color=palette.get(label, "#888"),
                       edgecolor="black", linewidth=1.2, zorder=10)
        # Cascade and lost ckpts as horizontal lines/markers (mDice only)
        if key == "mdice":
            # cascade champion line at train-time-equivalent
            ax.axhline(0.649, color="#1f77b4", linestyle="-.", linewidth=1.5,
                       alpha=0.7, label="Cascade v3.37 (full-cohort eval)")
            # Lost ckpts as scatter on right
            if lost:
                # Sort by best mdice descending
                lost_s = sorted(lost, key=lambda x: -x["mdice"])[:10]
                for r in lost_s:
                    ax.scatter([r["epoch"]], [r["mdice"]],
                               s=40, marker="x", color="#999",
                               alpha=0.6, linewidth=1.5)
                # One legend entry for all lost
                ax.scatter([], [], s=40, marker="x", color="#999",
                           alpha=0.6, linewidth=1.5,
                           label="Lost-original ckpts (top 10, no trajectory)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.02, 1.02)
        if key == "mdice":
            ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(
        "Training trajectories — GNCAF / seg_v2 variants (per-epoch val) + lost-ckpt final points",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    out = OUT / "fig_trajectories.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
