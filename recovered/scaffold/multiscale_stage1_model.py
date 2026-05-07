"""Multi-scale Stage 1 (v3.60) — heterogeneous bipartite GAT.

Same structure as `train_gars_stage1.GraphTLSDetector` but takes a
combined fine+coarse feature tensor + a `scale_mask` (0=fine, 1=coarse)
and adds a learned scale embedding before the GAT layers.

Forward signature:
    features:     (N_fine + N_coarse, in_dim) UNI v2 patch embeddings
    edge_index:   (2, E)                       union of fine-fine, coarse-coarse,
                                                and bidirectional containment edges
    scale_mask:   (N_fine + N_coarse,)         0 for fine, 1 for coarse

Output:
    logits:       (N_fine + N_coarse,)         per-node TLS-positive logit.
                                                Caller masks with scale_mask==0
                                                for loss / cascade selection.

Architecture preserves the `GraphTLSDetector` parameter shapes for
`proj`, `gnn_layers`, `gnn_norms`, and `head`, so a multi-scale
checkpoint cannot be strict-loaded into the single-scale class
(scale_embed is the new param block) but the GAT body matches.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv


class MultiScaleGraphTLSDetector(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 256,
        n_hops: int = 5,
        gnn_type: str = "gatv2",
        dropout: float = 0.1,
        gat_heads: int = 4,
        n_scales: int = 2,
        scale_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_hops = n_hops
        self.gnn_type = gnn_type
        self.n_scales = n_scales
        # Default scale embedding dim = hidden_dim so it adds cleanly to
        # the projected features. We use Embedding (lookup), not Linear,
        # so the scale signal is a simple bias-like vector per scale.
        self.scale_embed_dim = scale_embed_dim or hidden_dim
        assert self.scale_embed_dim == hidden_dim, \
            "scale_embed_dim must equal hidden_dim (added pre-GAT)"

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.scale_embed = nn.Embedding(n_scales, hidden_dim)
        # Initialize scale embedding to zeros so the model starts
        # behaving exactly like the single-scale baseline; it learns
        # a per-scale bias only as the data warrants.
        nn.init.zeros_(self.scale_embed.weight)

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(n_hops):
            if gnn_type == "gatv2":
                assert hidden_dim % gat_heads == 0, \
                    "hidden_dim must be divisible by gat_heads"
                self.gnn_layers.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // gat_heads,
                        heads=gat_heads,
                        concat=True,
                        dropout=0.0,
                    )
                )
            elif gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"unknown gnn_type {gnn_type!r}")
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        scale_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.proj(features)                      # (N, h)
        x = x + self.scale_embed(scale_mask)         # add per-scale bias
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            x = norm(layer(x, edge_index) + x)
        return self.head(x).squeeze(-1)              # (N,)

    def extract_context(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        scale_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the post-GAT pre-head representation. For cascade
        Stage 2 use; mirrors the single-scale `extract_stage1_context`.
        """
        x = self.proj(features)
        x = x + self.scale_embed(scale_mask)
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            x = norm(layer(x, edge_index) + x)
        return x  # (N, hidden_dim)


def smoke():
    """Minimal forward / extract_context check on random tensors."""
    m = MultiScaleGraphTLSDetector(in_dim=1536, hidden_dim=256, n_hops=5,
                                   gnn_type="gatv2", gat_heads=4, n_scales=2)
    n_fine, n_coarse = 100, 25
    n_total = n_fine + n_coarse
    feats = torch.randn(n_total, 1536)
    scale_mask = torch.cat([
        torch.zeros(n_fine, dtype=torch.long),
        torch.ones(n_coarse, dtype=torch.long),
    ])
    # Random edges across all node types.
    src = torch.randint(0, n_total, (300,))
    dst = torch.randint(0, n_total, (300,))
    edge_index = torch.stack([src, dst]).long()
    logits = m(feats, edge_index, scale_mask)
    ctx = m.extract_context(feats, edge_index, scale_mask)
    assert logits.shape == (n_total,)
    assert ctx.shape == (n_total, 256)
    print(f"forward: logits {tuple(logits.shape)}; ctx {tuple(ctx.shape)} OK")


if __name__ == "__main__":
    smoke()
