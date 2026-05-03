"""GNCAF — GNN-based Neighboring Context Aggregation Framework.

Faithful re-implementation of:
  Lei Su et al. (2025), "GNCAF: A GNN-based Neighboring Context Aggregation
  Framework for Tertiary Lymphoid Structures Semantic Segmentation in WSI."
  arXiv:2505.08430.

Architecture (paper §2):
  • Patch encoder: ViT (TransUNet-style) on the 256×256 RGB target patch
      → local tokens z_local ∈ R^{B × T × L}, T = (H/p)² (= 16² = 256 with p=16)
  • Context aggregator: K-hop GCN over UNI features for ALL patches in the slide
      → x⁽ᵏ⁾ = σ(Ã x⁽ᵏ⁻¹⁾ W⁽ᵏ⁻¹⁾)
      → z_context_i = MLP(concat[x⁽⁰⁾, …, x⁽ᴷ⁾])  (per node)
  • Fusion: 1-layer MSA over [z_local + pos ; z_context.unsqueeze(1)]
      → drop the context token → (B, T, L)
  • Decoder: spatial-grid reshape + 4× (bilinear up + DoubleConv) → (B, C, 256, 256)
  • Loss: cross-entropy

Defaults:
  L = 384  (ViT-S dim — paper used TransUNet R50+ViT-B which is bigger)
  K = 3    (paper default)
  patch_size = 16  → T = 256 tokens
  fusion_heads = 8 (paper default)
  in_features = 1536 (UNI-v2 in our zarrs; paper used UNI 1024-d)

Inputs:
    target_rgb:    (B, 3, H, W)  float32, ImageNet-normalised
    all_features:  (N, F)        float32  — UNI features for every patch in slide
    target_idx:    (B,)          long     — index into all_features
    edge_index:    (2, E)        long     — 4-connectivity adjacency

Output:
    logits:        (B, C, H, W)  float32

The model is decoupled from the data loader: feed it whatever target_rgb you
have. See gncaf_dataset.py for the slide-level loader that materialises these
tensors from WSI .tif + zarr features + 4-connectivity edges.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ─── ViT encoder (TransUNet-style, minimal) ───────────────────────────


class ViTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNetEncoder(nn.Module):
    """Small ViT encoder. 256×256 input → (B, T, dim) tokens (T=256, dim=384)."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        dim: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.dim = dim

        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size,
                                     stride=patch_size)
        n_tokens = self.grid ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [ViTBlock(dim, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) → tokens (B, T, dim)
        x = self.patch_embed(x)            # (B, dim, h, w)
        x = x.flatten(2).transpose(1, 2)   # (B, T, dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ─── Multi-hop GCN context aggregator ─────────────────────────────────


class GCNContextAggregator(nn.Module):
    """K-hop GCN with concat-then-MLP across hops.

    z_context_i = MLP( concat[x⁽⁰⁾_i, x⁽¹⁾_i, …, x⁽ᴷ⁾_i] ) ∈ R^L
    """

    def __init__(self, in_features: int = 1536, hidden_dim: int = 384,
                 n_hops: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_hops = n_hops
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for i in range(n_hops):
            in_d = in_features if i == 0 else hidden_dim
            self.layers.append(GCNConv(in_d, hidden_dim, add_self_loops=True,
                                       normalize=True))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # x⁽⁰⁾ has dim in_features, x⁽¹⁾..x⁽ᴷ⁾ each hidden_dim.
        cat_dim = in_features + n_hops * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hops = [x]
        h = x
        for layer in self.layers:
            h = F.gelu(layer(h, edge_index))
            h = self.dropout(h)
            hops.append(h)
        cat = torch.cat(hops, dim=-1)
        return self.mlp(cat)


# ─── Fusion block (1-layer MSA over [local; context]) ─────────────────


class FusionBlock(nn.Module):
    """LayerNorm → MultiheadAttention(self) over T+1 tokens; drop the
    context token at the end and return (B, T, L)."""

    def __init__(self, dim: int = 384, n_heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True,
                                         dropout=dropout)

    def forward(self, local_tokens: torch.Tensor,
                z_context: torch.Tensor) -> torch.Tensor:
        # local_tokens: (B, T, L); z_context: (B, L)
        ctx = z_context.unsqueeze(1)               # (B, 1, L)
        x = torch.cat([local_tokens, ctx], dim=1)  # (B, T+1, L)
        x = self.norm(x)
        out, _ = self.mha(x, x, x, need_weights=False)
        return out[:, :-1]                         # drop context token


# ─── Segmentation decoder (4× upsample) ───────────────────────────────


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SegDecoder(nn.Module):
    """Token grid (B, T, L) → mask (B, C, 256, 256) via 4× bilinear upsamples.

    With T = 256 (16×16 spatial), L = 384, n_classes = C:
        16×16×384 → 32×32×192 → 64×64×96 → 128×128×48 → 256×256×24 → C
    """

    def __init__(self, in_dim: int = 384, n_classes: int = 4,
                 grid: int = 16,
                 channels: tuple[int, int, int, int] = (192, 96, 48, 24)) -> None:
        super().__init__()
        self.grid = grid
        ch_in = in_dim
        blocks = []
        for c in channels:
            blocks.append(_DoubleConv(ch_in, c))
            ch_in = c
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(channels[-1], n_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, L = tokens.shape
        assert T == self.grid ** 2, f"T={T} ≠ grid²={self.grid ** 2}"
        x = tokens.transpose(1, 2).reshape(B, L, self.grid, self.grid)
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear",
                              align_corners=False)
            x = blk(x)
        return self.head(x)


# ─── End-to-end GNCAF ─────────────────────────────────────────────────


class GNCAF(nn.Module):
    """TransUNet+GNCAF (paper Fig. 2)."""

    def __init__(
        self,
        in_features: int = 1536,
        dim: int = 384,
        n_classes: int = 4,
        encoder_layers: int = 12,
        encoder_heads: int = 6,
        gcn_hops: int = 3,
        fusion_heads: int = 8,
        patch_size: int = 16,
        img_size: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TransUNetEncoder(
            img_size=img_size, patch_size=patch_size, dim=dim,
            n_layers=encoder_layers, n_heads=encoder_heads,
        )
        self.context = GCNContextAggregator(
            in_features=in_features, hidden_dim=dim, n_hops=gcn_hops,
            dropout=dropout,
        )
        self.fusion = FusionBlock(dim=dim, n_heads=fusion_heads, dropout=dropout)
        self.decoder = SegDecoder(
            in_dim=dim, n_classes=n_classes,
            grid=img_size // patch_size,
        )

    def forward(
        self,
        target_rgb: torch.Tensor,    # (B, 3, H, W)
        all_features: torch.Tensor,  # (N, F)
        target_idx: torch.Tensor,    # (B,)
        edge_index: torch.Tensor,    # (2, E)
    ) -> torch.Tensor:
        local_tokens = self.encoder(target_rgb)             # (B, T, L)
        node_context = self.context(all_features, edge_index)  # (N, L)
        z_context = node_context[target_idx]                # (B, L)
        fused = self.fusion(local_tokens, z_context)        # (B, T, L)
        return self.decoder(fused)                           # (B, C, H, W)


def _module_param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def model_summary(model: GNCAF) -> dict[str, int]:
    return {
        "encoder": _module_param_count(model.encoder),
        "context": _module_param_count(model.context),
        "fusion": _module_param_count(model.fusion),
        "decoder": _module_param_count(model.decoder),
        "total": _module_param_count(model),
    }


if __name__ == "__main__":
    # Quick shape + param sanity check.
    m = GNCAF(in_features=1536, dim=384, n_classes=4)
    print({k: f"{v:,}" for k, v in model_summary(m).items()})

    B, N, E = 2, 100, 200
    target_rgb = torch.randn(B, 3, 256, 256)
    all_features = torch.randn(N, 1536)
    target_idx = torch.tensor([0, 1])
    edge_index = torch.randint(0, N, (2, E))
    y = m(target_rgb, all_features, target_idx, edge_index)
    print("output shape:", tuple(y.shape))
