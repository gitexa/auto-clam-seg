"""Draw block-diagram figures for each architecture in the comparison.

Outputs `arch_<name>.png` under /home/ubuntu/auto-clam-seg/notebooks/architectures/.

Each figure is a top-down flow with coloured blocks for module class
(input data / projection / GNN / fusion / decoder / heads). Same colour
palette across all diagrams so they're directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("/home/ubuntu/auto-clam-seg/notebooks/architectures")
OUT.mkdir(parents=True, exist_ok=True)

# Colour palette per module class
COL = {
    "input":   "#cfe2f3",   # light blue
    "proj":    "#d9ead3",   # light green
    "gnn":     "#fce5cd",   # light orange
    "fusion":  "#f4cccc",   # light red
    "rgb":     "#ead1dc",   # light pink
    "decoder": "#d9d2e9",   # light purple
    "head":    "#fff2cc",   # light yellow
    "gate":    "#cccccc",   # grey
}


def block(ax, x, y, w, h, label, kind="proj", fontsize=8, lh=1.0):
    """Draw a labelled rounded box centred at (x, y)."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=COL.get(kind, "#eeeeee"), edgecolor="black", linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            linespacing=lh, wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, fontsize=7, ls="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=12,
        linewidth=1.0, color="black", linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, label,
                ha="left", va="center", fontsize=fontsize, color="#333")


def legend(ax, items):
    handles = [mpatches.Patch(facecolor=COL[k], edgecolor="black", label=lbl)
               for k, lbl in items]
    ax.legend(handles=handles, loc="lower center", ncol=len(items),
              bbox_to_anchor=(0.5, -0.05), fontsize=7, frameon=False)


def _setup(figsize=(7, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    return fig, ax


# ─────────────────────── Cascade Stage 1 ────────────────────────────

def fig_cascade_stage1():
    fig, ax = _setup((7, 7.5))
    ax.set_ylim(0, 10)
    ax.set_title("Cascade — Stage 1: GraphTLSDetector (GATv2 5-hop)\n"
                 "1.09 M params · BCEWithLogitsLoss(pos_weight=5) per patch",
                 fontsize=10)

    block(ax, 5, 9, 6, 0.7,
          "Per-patch UNI v2 features  (N, 1536)\n+ 4-conn graph_edges_1hop  (2, E)",
          "input", fontsize=8)
    arrow(ax, 5, 8.6, 5, 8.0)
    block(ax, 5, 7.7, 5, 0.6, "Linear(1536 → 256) projection", "proj")
    arrow(ax, 5, 7.4, 5, 6.95)

    # 5x GATv2Conv stack
    block(ax, 5, 6.6, 5, 0.6, "GATv2Conv(256 → 256, heads=4)\n+ LayerNorm + residual",
          "gnn", lh=1.1)
    arrow(ax, 5, 6.3, 5, 5.85)
    block(ax, 5, 5.5, 5, 0.6, "GATv2Conv(256 → 256, heads=4)\n+ LayerNorm + residual",
          "gnn", lh=1.1)
    arrow(ax, 5, 5.2, 5, 4.75)
    ax.text(5, 4.5, "× 5 hops total", ha="center", va="center", fontsize=8,
            color="#666", style="italic")
    arrow(ax, 5, 4.3, 5, 3.85)
    block(ax, 5, 3.55, 5, 0.6, "GATv2Conv(256 → 256, heads=4)\n+ LayerNorm + residual",
          "gnn", lh=1.1)
    arrow(ax, 5, 3.2, 5, 2.75)
    block(ax, 5, 2.45, 4, 0.55, "Linear(256 → 1) head", "head")
    arrow(ax, 5, 2.15, 5, 1.75)
    block(ax, 5, 1.45, 5.5, 0.5,
          "Per-patch logit  →  sigmoid > 0.5 selects ~0.7 % of patches", "gate")
    arrow(ax, 5, 1.15, 5, 0.7, "to Stage 2", fontsize=8)

    legend(ax, [("input", "input"), ("proj", "projection"), ("gnn", "GNN"),
                ("head", "head"), ("gate", "gate to Stage 2")])
    fig.savefig(OUT / "arch_cascade_stage1.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ─────────────────────── Cascade Stage 2 ────────────────────────────

def fig_cascade_stage2():
    fig, ax = _setup((9.5, 8.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.set_title("Cascade — Stage 2: RegionDecoder (RGB + UNI + graph fusion)\n"
                 "14.5 M params · CE([1,5,3]) + 1.0 × (1 − soft_Dice_GC) on 768×768 region",
                 fontsize=10)

    # Three input streams: RGB tile, UNI features, Stage-1 graph context
    block(ax, 2.0, 9.7, 3.2, 0.7,
          "RGB tile 768×768\n(read from WSI tif)", "input", lh=1.1)
    block(ax, 6.0, 9.7, 3.2, 0.7,
          "UNI features for 9 patches\n(9, 1536)", "input", lh=1.1)
    block(ax, 10.0, 9.7, 3.2, 0.7,
          "Stage 1 graph context\n(9, 256)", "input", lh=1.1)

    arrow(ax, 2.0, 9.3, 2.0, 8.7)
    arrow(ax, 6.0, 9.3, 6.0, 8.7)
    arrow(ax, 10.0, 9.3, 10.0, 8.7)

    # Per-stream encoders
    block(ax, 2.0, 8.3, 3.4, 0.6,
          "ResNet-18 RGB encoder (frozen)", "rgb")
    block(ax, 6.0, 8.3, 3.4, 0.6,
          "Linear(1536 → 64) per patch", "proj")
    block(ax, 10.0, 8.3, 3.4, 0.6,
          "Linear(256 → 64) per patch", "proj")

    arrow(ax, 2.0, 8.0, 2.0, 7.5)
    arrow(ax, 6.0, 8.0, 6.0, 7.5)
    arrow(ax, 10.0, 8.0, 10.0, 7.5)

    # Broadcast to spatial grid
    block(ax, 2.0, 7.2, 3.4, 0.55,
          "4-stage feature pyramid", "rgb")
    block(ax, 6.0, 7.2, 3.4, 0.55,
          "Broadcast to 256-px tile", "proj")
    block(ax, 10.0, 7.2, 3.4, 0.55,
          "Broadcast to 256-px tile", "proj")

    # Concat at bottleneck
    arrow(ax, 2.0, 6.9, 5.5, 6.3)
    arrow(ax, 6.0, 6.9, 6.0, 6.3)
    arrow(ax, 10.0, 6.9, 6.5, 6.3)

    block(ax, 6.0, 5.9, 6.0, 0.7,
          "Concat (RGB + UNI + Stage-1 graph) at bottleneck",
          "fusion")
    arrow(ax, 6.0, 5.55, 6.0, 5.1)

    # Decoder
    block(ax, 6.0, 4.8, 6.0, 0.6, "5× UpDoubleConv decoder (h=64)",
          "decoder")
    arrow(ax, 6.0, 4.5, 6.0, 4.05)
    block(ax, 6.0, 3.75, 4, 0.55, "Conv2d(16 → 3) head", "head")
    arrow(ax, 6.0, 3.45, 6.0, 3.0)
    block(ax, 6.0, 2.75, 6.5, 0.6,
          "Per-pixel {bg, TLS, GC} logits  (B, 3, 768, 768)",
          "head")
    arrow(ax, 6.0, 2.45, 6.0, 1.95)
    block(ax, 6.0, 1.65, 7, 0.55,
          "argmax → 768×768 mask  →  CC count w/ min_size=2, closing_iters=1",
          "gate")
    legend(ax, [("input", "input"), ("rgb", "RGB CNN"), ("proj", "projection"),
                ("fusion", "fusion"), ("decoder", "decoder"),
                ("head", "head"), ("gate", "post-proc")])
    fig.savefig(OUT / "arch_cascade_stage2.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ─────────────────────── GNCAF v3.58 ────────────────────────────────

def fig_gncaf_v358():
    fig, ax = _setup((9.5, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_title("GNCAF v3.58 — TransUNet + 3-hop iterative-residual GCN\n"
                 "108 M params · CE([1,5,3]) + 0.5 × soft_Dice  ·  augmented",
                 fontsize=10)

    # Inputs
    block(ax, 2.5, 11.0, 3.5, 0.7,
          "Target RGB tile  (B, 3, 256, 256)", "input")
    block(ax, 6.5, 11.0, 3.5, 0.7,
          "All-patch UNI features  (N, 1536)", "input")
    block(ax, 10.0, 11.0, 3.0, 0.7,
          "Spatial graph  (2, E)", "input")

    arrow(ax, 2.5, 10.6, 2.5, 10.0)
    arrow(ax, 6.5, 10.6, 6.5, 10.0)
    arrow(ax, 10.0, 10.6, 9.0, 10.0)

    # RGB encoder branch
    block(ax, 2.5, 9.6, 3.6, 0.6,
          "R50 trunk  (stem + 3 stages, ImageNet)", "rgb")
    arrow(ax, 2.5, 9.3, 2.5, 8.85)
    block(ax, 2.5, 8.55, 3.6, 0.6,
          "12-layer ViT trunk\n(16×16 tokens, 768-d)", "rgb", lh=1.05)
    arrow(ax, 2.5, 8.2, 2.5, 7.75)
    block(ax, 2.5, 7.45, 3.6, 0.55,
          "Tokens (B, 256, 768) + skips", "rgb")

    # GCN branch
    block(ax, 7.5, 9.5, 4.0, 0.55,
          "Linear(1536 → 768) + LN", "proj")
    arrow(ax, 7.5, 9.2, 7.5, 8.7)
    block(ax, 7.5, 8.4, 4.0, 0.6,
          "GCNConv(768 → 768) + GeLU\n+ LayerNorm + residual", "gnn", lh=1.1)
    arrow(ax, 7.5, 8.05, 7.5, 7.55)
    ax.text(7.5, 7.35, "× 3 hops (iterative residual)", ha="center", va="center",
            fontsize=8, color="#666", style="italic")
    arrow(ax, 7.5, 7.2, 7.5, 6.7)
    block(ax, 7.5, 6.4, 4.0, 0.55,
          "Per-node 768-d context  (N, 768)", "gnn")

    # Pull target node's context
    arrow(ax, 7.5, 6.1, 6.0, 5.55, ls="--")
    arrow(ax, 2.5, 7.15, 4.5, 5.55, ls="--")

    # Fusion: concat CTX token to image tokens
    block(ax, 5.0, 5.25, 5.5, 0.6,
          "[CTX] token + 256 image tokens", "fusion")
    arrow(ax, 5.0, 4.95, 5.0, 4.5)
    block(ax, 5.0, 4.2, 5.0, 0.65,
          "FusionBlock (MultiHeadAttention + MLP)\nthen drop CTX token",
          "fusion", lh=1.1)
    arrow(ax, 5.0, 3.85, 5.0, 3.4)

    # Decoder + head
    block(ax, 5.0, 3.1, 5.0, 0.55,
          "Reshape 256 → (B, 768, 16, 16)", "proj")
    arrow(ax, 5.0, 2.83, 5.0, 2.4)
    block(ax, 5.0, 2.1, 5.0, 0.6,
          "TransUNet decoder (4 stages, R50 skips)", "decoder")
    arrow(ax, 5.0, 1.8, 5.0, 1.35)
    block(ax, 5.0, 1.05, 5.0, 0.55,
          "Conv2d(16 → 3) → (B, 3, 256, 256)", "head")

    legend(ax, [("input", "input"), ("rgb", "RGB encoder"),
                ("proj", "projection"), ("gnn", "GCN context"),
                ("fusion", "fusion"), ("decoder", "decoder"),
                ("head", "head")])
    fig.savefig(OUT / "arch_gncaf_v358.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ─────────────────────── GNCAF v3.59 (GCUNet-faithful) ──────────────

def fig_gncaf_v359():
    fig, ax = _setup((9.5, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_title("GNCAF v3.59 — GCUNet-faithful GCN (paper Eq. 3, concat-all-hops)\n"
                 "111 M params · everything else identical to v3.58",
                 fontsize=10)

    # Inputs (same)
    block(ax, 2.5, 11.0, 3.5, 0.7,
          "Target RGB tile  (B, 3, 256, 256)", "input")
    block(ax, 6.5, 11.0, 3.5, 0.7,
          "All-patch UNI features  (N, 1536)", "input")
    block(ax, 10.0, 11.0, 3.0, 0.7,
          "Spatial graph  (2, E)", "input")

    arrow(ax, 2.5, 10.6, 2.5, 10.0)
    arrow(ax, 6.5, 10.6, 6.5, 10.0)
    arrow(ax, 10.0, 10.6, 9.0, 10.0)

    # RGB encoder branch (same as v3.58)
    block(ax, 2.5, 9.6, 3.6, 0.6,
          "R50 trunk  (stem + 3 stages, ImageNet)", "rgb")
    arrow(ax, 2.5, 9.3, 2.5, 8.85)
    block(ax, 2.5, 8.55, 3.6, 0.6,
          "12-layer ViT trunk\n(16×16 tokens, 768-d)", "rgb", lh=1.05)
    arrow(ax, 2.5, 8.2, 2.5, 7.75)
    block(ax, 2.5, 7.45, 3.6, 0.55,
          "Tokens (B, 256, 768) + skips", "rgb")

    # GCN branch — DIFFERENT (concat all hops)
    block(ax, 7.5, 9.5, 4.0, 0.55,
          "Linear(1536 → 768) + LN  → x⁽⁰⁾", "proj")
    arrow(ax, 7.5, 9.2, 7.5, 8.75)
    block(ax, 7.5, 8.45, 4.0, 0.6,
          "GCNConv(768 → 768) + GeLU\nx⁽¹⁾, x⁽²⁾, x⁽³⁾  (no residual)",
          "gnn", lh=1.1)
    arrow(ax, 7.5, 8.1, 7.5, 7.6)
    block(ax, 7.5, 7.3, 5.0, 0.6,
          "CONCAT(x⁽⁰⁾, x⁽¹⁾, x⁽²⁾, x⁽³⁾) → (N, 4·768)",
          "gnn")
    arrow(ax, 7.5, 7.0, 7.5, 6.5)
    block(ax, 7.5, 6.2, 5.0, 0.65,
          "MLP(3072 → 768 → 768) + LN\n(per Eq. 3 — paper-faithful)",
          "gnn", lh=1.1)
    arrow(ax, 7.5, 5.85, 7.5, 5.4)
    block(ax, 7.5, 5.1, 4.0, 0.55,
          "Per-node 768-d context", "gnn")

    # Pull target node's context
    arrow(ax, 7.5, 4.85, 6.0, 4.35, ls="--")
    arrow(ax, 2.5, 7.15, 4.5, 4.35, ls="--")

    # Fusion + decoder + head (same as v3.58, more compact)
    block(ax, 5.0, 4.05, 5.5, 0.55,
          "[CTX] + 256 image tokens", "fusion")
    arrow(ax, 5.0, 3.78, 5.0, 3.35)
    block(ax, 5.0, 3.05, 5.0, 0.6,
          "FusionBlock (Attn + MLP) → drop CTX", "fusion")
    arrow(ax, 5.0, 2.75, 5.0, 2.3)
    block(ax, 5.0, 2.0, 5.0, 0.55,
          "TransUNet decoder (4 stages)", "decoder")
    arrow(ax, 5.0, 1.73, 5.0, 1.3)
    block(ax, 5.0, 1.0, 5.0, 0.55,
          "Conv2d(16 → 3) → (B, 3, 256, 256)", "head")

    legend(ax, [("input", "input"), ("rgb", "RGB encoder"),
                ("proj", "projection"), ("gnn", "GCN (Eq. 3)"),
                ("fusion", "fusion"), ("decoder", "decoder"),
                ("head", "head")])
    fig.savefig(OUT / "arch_gncaf_v359.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ─────────────────────── seg_v2.0 ───────────────────────────────────

def fig_seg_v2():
    fig, ax = _setup((9, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_title("seg_v2.0 — GNNSegmentationDecoder (centroid + dual-sigmoid)\n"
                 "4.4 M params · TLS focal+Dice + GC focal+Dice (cropped) + 2× center heatmap",
                 fontsize=10)

    # Inputs
    block(ax, 6.0, 11.3, 8.0, 0.7,
          "Per-patch UNI v2 (N, 1536)  +  coords (N, 2)  +  1-hop spatial graph (2, E)",
          "input")
    arrow(ax, 6.0, 10.95, 6.0, 10.5)

    # Feature projection
    block(ax, 6.0, 10.2, 5.5, 0.55,
          "Feature proj: Linear(1536 → 384) + LN + GELU", "proj")
    arrow(ax, 6.0, 9.95, 6.0, 9.5)

    # GNN (no-op in v2.0)
    block(ax, 6.0, 9.2, 5.5, 0.6,
          "GNN context  (gnn_layers=0  →  no-op identity)\n(architecture supports GATv2(384, h=4) × N + residual)",
          "gnn", lh=1.1)
    arrow(ax, 6.0, 8.85, 6.0, 8.4)

    # Scatter to grid + on-grid agg
    block(ax, 6.0, 8.1, 5.5, 0.6,
          "Scatter to (1, 384, H, W) patch grid", "proj")
    arrow(ax, 6.0, 7.8, 6.0, 7.35)
    block(ax, 6.0, 7.05, 5.5, 0.6,
          "Conv2d(384, 384, k=5) + BN + GELU  (on-grid spatial agg)",
          "fusion")
    arrow(ax, 6.0, 6.75, 6.0, 6.3)

    # CNN decoder: 3 upsample blocks (8x)
    block(ax, 6.0, 5.95, 6.5, 0.7,
          "3 × UpsampleBlock\n[depthwise Conv + 1×1 Conv + BN + GELU + Upsample(2×)]\n384 → 192 → 96 → 48",
          "decoder", lh=1.1)
    arrow(ax, 6.0, 5.55, 6.0, 5.0)

    # 5 heads
    block(ax, 6.0, 4.7, 6.5, 0.5,
          "Pixel features (1, 48, 8H, 8W)", "decoder")

    arrow(ax, 6.0, 4.45, 1.5, 3.85)
    arrow(ax, 6.0, 4.45, 4.0, 3.85)
    arrow(ax, 6.0, 4.45, 6.5, 3.85)
    arrow(ax, 6.0, 4.45, 9.0, 3.85)
    arrow(ax, 6.0, 4.45, 11.5, 3.85)

    block(ax, 1.5, 3.55, 1.7, 0.55, "TLS\nsigmoid head", "head", lh=1.1)
    block(ax, 4.0, 3.55, 1.7, 0.55, "GC\nsigmoid head\n(independent)", "head", lh=1.0, fontsize=7)
    block(ax, 6.5, 3.55, 1.7, 0.55, "TLS center\nheatmap head", "head", lh=1.1)
    block(ax, 9.0, 3.55, 1.7, 0.55, "GC center\nheatmap head", "head", lh=1.1)
    block(ax, 11.5, 3.55, 1.6, 0.55, "Offset (dy, dx)\n(disabled)", "head", lh=1.1, fontsize=7)

    # Loss summary
    block(ax, 6.0, 2.5, 11.0, 0.65,
          "segmentation_loss = 1·focal_TLS + 2·dice_TLS + 1·focal_GC* + 5·dice_GC* + 2·center_TLS + 2·center_GC\n"
          "(* GC focal & Dice are cropped to TLS-positive pixels — GC is 0.006 % of pixels at full res)",
          "fusion", lh=1.1, fontsize=7.5)
    arrow(ax, 6.0, 2.15, 6.0, 1.7)
    block(ax, 6.0, 1.4, 7.0, 0.55,
          "Per-pixel TLS + GC mask + instance centroids (8× up grid)", "head")

    legend(ax, [("input", "input"), ("proj", "projection"),
                ("gnn", "GNN (off)"), ("fusion", "spatial agg / loss"),
                ("decoder", "CNN decoder"), ("head", "heads")])
    fig.savefig(OUT / "arch_seg_v2.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main():
    fig_cascade_stage1()
    fig_cascade_stage2()
    fig_gncaf_v358()
    fig_gncaf_v359()
    fig_seg_v2()
    print("Saved 5 architecture diagrams to", OUT)


if __name__ == "__main__":
    main()
