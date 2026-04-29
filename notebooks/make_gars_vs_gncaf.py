"""Comparison figure: GARS (our two-stage UNI-v2 cascade) vs published GNCAF.

Data is reconstructed from the lost gars-cascade branch (commit 63e1e5b on
autoresearch/gars-cascade — never pushed to origin). Source numbers come from
the final comparison the user produced before the VM died.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/ubuntu/auto-clam-seg/notebooks/gars_vs_gncaf.png")

metrics = ["mDice", "TLS dice", "GC dice"]
gncaf = [0.688, 0.736, 0.625]
gars = [0.714, 0.718, 0.710]

cost_labels = ["Params (M)", "Inference (s/slide)", "Training (min)"]
gncaf_cost = [57.0, 450.0, 360.0]   # 5-10 min/slide -> 7.5 min = 450 s; 6 h = 360 min
gars_cost = [10.1, 1.5, 7.0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Left: dice metrics
ax = axes[0]
x = np.arange(len(metrics))
w = 0.36
bars_g = ax.bar(x - w / 2, gncaf, w, label="GNCAF (published, ~57 M)",
                color="#9aa0a6", edgecolor="black")
bars_o = ax.bar(x + w / 2, gars, w, label="GARS (ours, 10.1 M)",
                color="#1f77b4", edgecolor="black")
for rect, val in zip(bars_g, gncaf):
    ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
for rect, val in zip(bars_o, gars):
    ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
deltas = [g - n for g, n in zip(gars, gncaf)]
for xi, d in zip(x, deltas):
    pct = 100 * d / max(1e-6, abs(gncaf[list(x).index(xi)]))
    sign = "+" if d > 0 else ""
    color = "#188038" if d > 0 else "#c5221f"
    ax.text(xi, max(gars[list(x).index(xi)], gncaf[list(x).index(xi)]) + 0.06,
            f"{sign}{pct:.0f}%", ha="center", fontsize=10,
            color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Dice (114-slide val, threshold=0.05)")
ax.set_ylim(0, 0.92)
ax.set_title("Segmentation quality")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle=":", alpha=0.5)

# Right: cost — use log scale because the dynamic range is huge
ax = axes[1]
x = np.arange(len(cost_labels))
bars_g = ax.bar(x - w / 2, gncaf_cost, w, label="GNCAF",
                color="#9aa0a6", edgecolor="black")
bars_o = ax.bar(x + w / 2, gars_cost, w, label="GARS",
                color="#1f77b4", edgecolor="black")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(cost_labels)
ax.set_ylabel("log scale")
ax.set_title("Compute cost (lower is better)")
ax.legend(loc="upper right")
ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)
ratios = [n / o for n, o in zip(gncaf_cost, gars_cost)]
for xi, n, o, r in zip(x, gncaf_cost, gars_cost, ratios):
    label = f"{n:.0f}" if n >= 10 else f"{n:.1f}"
    ax.text(xi - w / 2, n * 1.18, label, ha="center", fontsize=9)
    label = f"{o:.0f}" if o >= 10 else f"{o:.1f}"
    ax.text(xi + w / 2, o * 1.18, label, ha="center", fontsize=9, fontweight="bold")
    ax.text(xi, max(n, o) * 3.0, f"{r:.0f}× fewer",
            ha="center", fontsize=10, color="#188038", fontweight="bold")

fig.suptitle("GARS vs GNCAF — TLS + GC pixel segmentation on TCGA",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"Saved: {OUT}")
