"""GCUNet-faithful GCN aggregation (paper Eq. 3) — v3.59.

Difference from `gncaf_transunet_model.GCNContext` (v3.58 and earlier):

  * Eq. 3 (paper): h_v = MLP( CAT([x⁽⁰⁾_v ; x⁽¹⁾_v ; … ; x⁽ᴷ⁾_v]) )
  * v3.58 GCNContext: iterative residual updates, returns only x⁽ᴷ⁾.

Everything else (TransUNet encoder, Fusion block, decoder, head) is
inherited unchanged from `gncaf_transunet_model.GNCAFPixelDecoder`,
so this is a drop-in swap of the GCN block only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from gncaf_transunet_model import GNCAFPixelDecoder


class GCNContextPaper(nn.Module):
    """Concat-all-hops + MLP, matching GCUNet Eq. 3."""

    def __init__(
        self,
        in_features: int = 1536,
        hidden_dim: int = 768,
        n_hops: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
            for _ in range(n_hops)
        ])
        # Per Eq. 3 we concat x⁽⁰⁾, x⁽¹⁾, …, x⁽ᴷ⁾ ∈ ℝ^hidden_dim then
        # project back to hidden_dim with a small MLP.
        cat_dim = (n_hops + 1) * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.n_hops = n_hops

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # (N, hidden_dim) = x⁽⁰⁾
        hs = [x]
        h = x
        for layer in self.gcn_layers:
            h = F.gelu(layer(h, edge_index))   # x⁽ᵏ⁾
            hs.append(h)
        cat = torch.cat(hs, dim=-1)            # (N, (K+1)·hidden_dim)
        return self.mlp(cat)                   # (N, hidden_dim)


class GCUNetPixelDecoder(GNCAFPixelDecoder):
    """v3.59 = v3.58 + GCNContextPaper (Eq. 3) replacing GCNContext.

    The constructor signature is identical so existing trainer/eval
    code can swap classes without any other changes.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        n_classes: int = 3,
        n_encoder_layers: int = 12,
        n_heads: int = 12,
        mlp_dim: int = 3072,
        n_hops: int = 3,
        n_fusion_layers: int = 1,
        feature_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__(
            hidden_size=hidden_size, n_classes=n_classes,
            n_encoder_layers=n_encoder_layers, n_heads=n_heads,
            mlp_dim=mlp_dim, n_hops=n_hops, n_fusion_layers=n_fusion_layers,
            feature_dim=feature_dim, dropout=dropout,
        )
        # Swap the GCN block for the paper-faithful one. Same I/O shape
        # so the rest of the pipeline (fusion, decoder, head) is unchanged.
        self.gcn = GCNContextPaper(
            in_features=feature_dim, hidden_dim=hidden_size,
            n_hops=n_hops, dropout=dropout,
        )


if __name__ == "__main__":
    m = GCUNetPixelDecoder().eval()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"GCUNetPixelDecoder params: {n_params:,}")
    B, N, E = 2, 200, 600
    rgb = torch.randn(B, 3, 256, 256)
    feats = torch.randn(N, 1536)
    tidx = torch.randint(0, N, (B,))
    ei = torch.randint(0, N, (2, E))
    y = m(rgb, feats, tidx, ei)
    print(f"forward OK: {tuple(y.shape)}")
