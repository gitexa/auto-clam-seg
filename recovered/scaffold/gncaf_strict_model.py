"""GNCAF / GCUNet — paper-strict reimplementation.

Reference:
  Su et al. 2025 (GNCAF, arXiv:2505.08430)
  Su et al. 2024 (GCUNet, arXiv:2412.06129)

Differences from `gncaf_transunet_model.GNCAFPixelDecoder`:

* **GCN block**: hidden_dim **128** (not 768; per GCUNet §4.2).
* **GCN aggregation**: **softmax-attention with learnable temperature**
  (per GCUNet §4.2: "Softmax is employed in the aggregation function
  of the GCN layer, with the temperature constant initialized as a
  learnable parameter set to 1"). Implemented as a GAT-like edge-
  softmax rather than `torch_geometric.nn.GCNConv`'s symmetric
  Laplacian.
* **Multi-hop aggregation**: concat-all-hops + MLP per Eq. 3,
  `z_c = MLP(CAT([x⁽⁰⁾; …; x⁽ᴷ⁾]))`. Output projected to
  encoder hidden (768) so the fusion ctx token is dim-matched.
* **Fusion block**: **8 heads** (per GNCAF §3.1) MSA only — no MLP.
  Per GNCAF Eq. 4: `z_i^(ℓ) = MSA^(ℓ)(LN(z_i^(0)))`.
* **Training**: plain per-pixel cross-entropy (Eq. 6), no class
  weights, no Dice. Knobs in `train_gncaf_transunet.py` allow this
  by setting `class_weights=[1,1,1]` and `dice_loss_weight=0.0`.

Architecture remains: TransUNet R50 + 12 ViT encoder, GCN context
+ MSA fusion, TransUNet decoder, 3-class output head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax as edge_softmax

from gncaf_transunet_model import (
    TransUNetEncoder, TransUNetDecoder, GNCAFPixelDecoder,
)


# ─── softmax-attention GCN (paper §4.2) ───────────────────────────────


class SoftmaxAttentionGCNLayer(MessagePassing):
    """One layer of GAT-like attention with a learnable scalar temperature τ.

    For each edge (i, j) (j is neighbour of i):
        score_ij = (q_i · k_j) / τ
        α_ij    = softmax_j (score_ij)        # over neighbours of i
        h_i'    = Σ_j α_ij · v_j              # weighted aggregation

    `softmax_j` uses `torch_geometric.utils.softmax` so per-node
    normalisation is correct on a sparse edge list.

    Shape preserved: (N, dim) → (N, dim).
    """

    def __init__(self, dim: int):
        super().__init__(aggr="add", node_dim=0)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # Learnable scalar temperature, init=1 per paper.
        self.tau = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Add self-loops so each node always has its own message.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, q=q, k=k, v=v, size=None)

    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor,
                index: torch.Tensor, ptr, size_i: int) -> torch.Tensor:
        # Inner-product score per edge, divided by learnable τ.
        # Clamp τ from below so it can't go to zero (would explode).
        tau = torch.clamp(self.tau, min=1e-3)
        score = (q_i * k_j).sum(dim=-1) / tau           # (E,)
        alpha = edge_softmax(score, index, ptr, size_i) # softmax over neighbours of i
        return v_j * alpha.unsqueeze(-1)


# ─── GCN context, paper-strict ────────────────────────────────────────


class GCNContextStrict(nn.Module):
    """Per GCUNet §3.3 + §4.2:

    * input: per-node UNI features (N, in_features=1024 for UNI v1; 1536 for UNI v2)
    * proj: Linear(in_features → hidden_dim=128) + LayerNorm
    * 3 × SoftmaxAttentionGCNLayer(128) — softmax aggregation, learnable τ
    * concat all (K+1) hops → (N, (K+1)·128)
    * context_mlp: MLP((K+1)·128 → out_dim=768) so the per-node context
      can be fed directly as the [CTX] token of the fusion block.

    `out_dim` is the encoder's hidden dim (768) — matches paper where
    z_c is L-dimensional and L = encoder hidden.
    """

    def __init__(
        self,
        in_features: int = 1536,
        hidden_dim: int = 128,
        n_hops: int = 3,
        out_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gcn_layers = nn.ModuleList([
            SoftmaxAttentionGCNLayer(hidden_dim) for _ in range(n_hops)
        ])
        cat_dim = (n_hops + 1) * hidden_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(cat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.LayerNorm(out_dim),
        )
        self.n_hops = n_hops

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)              # (N, hidden_dim)
        hs = [x]
        h = x
        for layer in self.gcn_layers:
            h = F.gelu(layer(h, edge_index))
            hs.append(h)
        cat = torch.cat(hs, dim=-1)   # (N, (K+1)·hidden_dim)
        return self.context_mlp(cat)  # (N, out_dim)


# ─── Fusion block, paper-strict (8 heads, MSA only, NO MLP) ──────────


class FusionBlockStrict(nn.Module):
    """GNCAF Eq. 4: `z_i^(ℓ) = MSA^(ℓ)(LN(z_i^(0)))`. Single MSA, no MLP.

    Operates on the [CTX]-prepended sequence: input shape (B, T+1, dim).
    Caller drops the CTX token afterward.
    """

    def __init__(self, dim: int = 768, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return x + a


# ─── Top-level paper-strict model ─────────────────────────────────────


class GNCAFStrict(nn.Module):
    """Paper-strict GNCAF / GCUNet. Same forward signature as
    `GNCAFPixelDecoder` so the trainer + eval pipeline can call it
    interchangeably.
    """

    def __init__(
        self,
        hidden_size: int = 768,         # encoder + fusion hidden
        n_classes: int = 3,
        n_encoder_layers: int = 12,     # paper: 12-layer attention
        n_heads: int = 12,              # encoder heads
        mlp_dim: int = 3072,
        n_hops: int = 3,
        n_fusion_layers: int = 1,
        feature_dim: int = 1536,         # UNI v2 in our zarrs
        dropout: float = 0.1,
        gcn_hidden_dim: int = 128,       # paper §4.2: "hidden feature dimension is set to 128"
        fusion_heads: int = 8,           # paper GNCAF §3.1: "1 layer, 8 heads"
    ):
        super().__init__()
        self.encoder = TransUNetEncoder(
            hidden_size=hidden_size, n_layers=n_encoder_layers,
            n_heads=n_heads, mlp_dim=mlp_dim, dropout=dropout,
        )
        self.gcn = GCNContextStrict(
            in_features=feature_dim, hidden_dim=gcn_hidden_dim,
            n_hops=n_hops, out_dim=hidden_size, dropout=dropout,
        )
        self.fusion = nn.ModuleList([
            FusionBlockStrict(dim=hidden_size, n_heads=fusion_heads, dropout=dropout)
            for _ in range(n_fusion_layers)
        ])
        self.decoder = TransUNetDecoder(in_dim=hidden_size)
        self.head_seg = nn.Conv2d(16, n_classes, 3, padding=1)

    def forward(
        self,
        target_rgb: torch.Tensor,
        all_features: torch.Tensor,
        target_idx: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        tokens, skips = self.encoder(target_rgb)               # (B, 256, 768)
        node_ctx = self.gcn(all_features, edge_index)          # (N, 768)
        local_ctx = node_ctx[target_idx]                       # (B, 768)

        ctx_t = local_ctx.unsqueeze(1)                          # (B, 1, 768)
        x = torch.cat([ctx_t, tokens], dim=1)                  # (B, 257, 768)
        for blk in self.fusion:
            x = blk(x)
        x = x[:, 1:]                                            # (B, 256, 768)

        b, t, d = x.shape
        x = x.transpose(1, 2).reshape(b, d, 16, 16)
        x = self.decoder(x, skips)                              # (B, 16, 256, 256)
        return self.head_seg(x)                                 # (B, 3, 256, 256)


def _module_param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def model_summary_strict(m: GNCAFStrict) -> dict:
    return {
        "encoder": _module_param_count(m.encoder),
        "gcn":     _module_param_count(m.gcn),
        "fusion":  _module_param_count(m.fusion),
        "decoder": _module_param_count(m.decoder),
        "head":    _module_param_count(m.head_seg),
        "total":   _module_param_count(m),
    }


if __name__ == "__main__":
    m = GNCAFStrict().eval()
    print({k: f"{v:,}" for k, v in model_summary_strict(m).items()})
    B, N, E = 2, 200, 600
    rgb = torch.randn(B, 3, 256, 256)
    feats = torch.randn(N, 1536)
    tidx = torch.randint(0, N, (B,))
    ei = torch.randint(0, N, (2, E))
    y = m(rgb, feats, tidx, ei)
    print(f"forward OK: {tuple(y.shape)}")
