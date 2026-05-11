"""GNCAF Family-B reconstruction.

Reconstructed from state_dict shapes of the lost
`gncaf_pixel_pixel_gncaf_frozen_gn_h128_c088b44f` checkpoint
(4.1 M params, best_mDice=0.7707) and the matching `demo`,
`55slides_patient_split`, `589slides` ckpts.

State_dict signature: top-level modules are
    patch_encoder, context_aggregator, fusion, pixel_decoder, head_seg
(NOT encoder/gcn/decoder — that's our line-B Family-A).

Architecture (hidden_dim parameterizes everything):
- patch_encoder: ResNet stem + 3 stages of BasicBlocks (64→128→256)
  + 1×1 proj to hidden_dim + N transformer encoder layers + pos_embed
- context_aggregator: feature_proj(1536→hidden) + 3× GCNConv(hidden)
  + concat-all-hops MLP (4·hidden → hidden) — paper-faithful Eq. 3
- fusion: 1× self-attention layer (MSA-only, no MLP) — paper-faithful Eq. 4
- pixel_decoder: 4-stage U with ConvTranspose2d up + Conv2d, with
  skip connections at the first 2 stages (matches up1 in=128 + skip 128 → conv1 in=256)
- head_seg: Conv2d(16, n_classes, 1×1)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class BasicBlock(nn.Module):
    """ResNet BasicBlock — matches torchvision BasicBlock state_dict layout."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class FamilyBPatchEncoder(nn.Module):
    """ResNet-18-style stem + 3 stages of 2 BasicBlocks + N transformer layers."""

    def __init__(self, hidden_dim: int = 128, n_transformer_layers: int = 2,
                 mlp_ratio: int = 4, n_heads: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Stem: conv 7×7 stride 2 → BN → ReLU → MaxPool 3×3 stride 2
        # Output: 256/4 = 64×64 spatial
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Stage 1: 64→64, no downsample → 64×64
        self.layer1 = nn.Sequential(BasicBlock(64, 64, stride=1), BasicBlock(64, 64, stride=1))
        # Stage 2: 64→128, stride 2 → 32×32
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128, stride=1))
        # Stage 3: 128→256, stride 2 → 16×16
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256, stride=1))
        # 1×1 proj to hidden_dim
        self.proj = nn.Conv2d(256, hidden_dim, 1)
        # 16×16 = 256 token grid
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        # Transformer encoder
        if n_heads is None:
            # default heuristic: 4 heads if hidden=128, 4 if hidden=256
            n_heads = 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=mlp_ratio * hidden_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 3, 256, 256)
        feat_64 = self.layer1(self.stem(x))     # (B, 64, 64, 64)
        feat_32 = self.layer2(feat_64)          # (B, 128, 32, 32)
        feat_16 = self.layer3(feat_32)          # (B, 256, 16, 16)
        tokens = self.proj(feat_16)             # (B, hidden, 16, 16)
        B, C, H, W = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, 256, hidden)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)        # (B, 256, hidden)
        return tokens, feat_32, feat_64


class FamilyBContextAggregator(nn.Module):
    """Paper-faithful concat-all-hops GCN aggregator.

    feature_proj: Linear(in=feature_dim, out=hidden) + LayerNorm(hidden)
    gcn_layers: 3× GCNConv(hidden, hidden)
    context_mlp: Linear((K+1)·hidden, hidden) + LayerNorm(hidden)
        where K=n_hops=3 → cat dim = 4·hidden
    """

    def __init__(self, feature_dim: int = 1536, hidden_dim: int = 128,
                 n_hops: int = 3, dropout: float = 0.0):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
            for _ in range(n_hops)
        ])
        cat_dim = (n_hops + 1) * hidden_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x: (N_nodes, feature_dim); edge_index: (2, N_edges)
        x = self.feature_proj(x)
        hs = [x]
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            hs.append(x)
        cat = torch.cat(hs, dim=-1)
        return self.context_mlp(cat)


class FamilyBFusion(nn.Module):
    """Paper-faithful MSA-only fusion (Eq. 4): self-attention over [ctx; detail] tokens.

    norm + multi-head self-attention; no MLP. Single layer.
    Input: tokens (B, 256, hidden); ctx (B, hidden) — one ctx per batch element.
    Output: (B, 256, hidden) — fused detail tokens.
    """

    def __init__(self, hidden_dim: int = 128, n_heads: int | None = None):
        super().__init__()
        if n_heads is None:
            n_heads = 4
        self.layers = nn.ModuleList([_FusionLayer(hidden_dim, n_heads)])

    def forward(self, detail_tokens, ctx_vec):
        # detail_tokens: (B, 256, hidden); ctx_vec: (B, hidden)
        for layer in self.layers:
            detail_tokens = layer(detail_tokens, ctx_vec)
        return detail_tokens


class _FusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)

    def forward(self, detail, ctx):
        # detail: (B, 256, H); ctx: (B, H)
        ctx_t = ctx.unsqueeze(1)                                        # (B, 1, H)
        seq = torch.cat([ctx_t, detail], dim=1)                          # (B, 257, H)
        seq_n = self.norm(seq)
        attn_out, _ = self.attn(seq_n, seq_n, seq_n, need_weights=False)
        seq = seq + attn_out                                             # residual
        return seq[:, 1:, :]                                             # drop ctx → (B, 256, H)


class FamilyBPixelDecoder(nn.Module):
    """4-stage U-decoder reconstructed from state_dict shapes.

    up1: ConvTranspose(hidden→hidden, k=2, s=2) → 16×16 → 32×32
    conv1: Conv(hidden + 128 skip → hidden, k=3, p=1)  [skip from layer2]
    up2: ConvTranspose(hidden→hidden//2)              → 32×32 → 64×64
    conv2: Conv(hidden//2 + 64 skip → hidden//2)       [skip from layer1]
    up3: ConvTranspose(hidden//2→hidden//4)           → 64×64 → 128×128
    conv3: Conv(hidden//4 → hidden//4) [no skip]
    up4: ConvTranspose(hidden//4→hidden//8)           → 128×128 → 256×256
    conv4: Conv(hidden//8 → hidden//8) [no skip]
    """

    def __init__(self, hidden_dim: int = 128, gn_groups: int = 8, norm: str = "gn"):
        """norm: 'gn' (GroupNorm — frozen_gn_h128) or 'bn' (BatchNorm — demo)."""
        super().__init__()
        h = hidden_dim
        h2 = h // 2
        h4 = h // 4
        h8 = h // 8
        if norm == "gn":
            def n(c): return nn.GroupNorm(min(gn_groups, c), c)
        elif norm == "bn":
            def n(c): return nn.BatchNorm2d(c)
        else:
            raise ValueError(f"unknown norm: {norm}")
        # Stage 1: 16×16 → 32×32, skip from layer2 (128 ch)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(h, h, 2, 2), n(h))
        self.conv1 = nn.Sequential(nn.Conv2d(h + 128, h, 3, padding=1), n(h))
        # Stage 2: 32×32 → 64×64, skip from layer1 (64 ch)
        self.up2 = nn.Sequential(nn.ConvTranspose2d(h, h2, 2, 2), n(h2))
        self.conv2 = nn.Sequential(nn.Conv2d(h2 + 64, h2, 3, padding=1), n(h2))
        # Stage 3: 64×64 → 128×128 (no skip)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(h2, h4, 2, 2), n(h4))
        self.conv3 = nn.Sequential(nn.Conv2d(h4, h4, 3, padding=1), n(h4))
        # Stage 4: 128×128 → 256×256 (no skip)
        self.up4 = nn.Sequential(nn.ConvTranspose2d(h4, h8, 2, 2), n(h8))
        self.conv4 = nn.Sequential(nn.Conv2d(h8, h8, 3, padding=1), n(h8))

    def forward(self, x, skip32, skip64):
        # x: (B, hidden, 16, 16); skip32: (B, 128, 32, 32); skip64: (B, 64, 64, 64)
        x = F.relu(self.up1[1](self.up1[0](x)), inplace=True)         # (B, hidden, 32, 32)
        x = torch.cat([x, skip32], dim=1)
        x = F.relu(self.conv1[1](self.conv1[0](x)), inplace=True)     # (B, hidden, 32, 32)
        x = F.relu(self.up2[1](self.up2[0](x)), inplace=True)         # (B, h2, 64, 64)
        x = torch.cat([x, skip64], dim=1)
        x = F.relu(self.conv2[1](self.conv2[0](x)), inplace=True)
        x = F.relu(self.up3[1](self.up3[0](x)), inplace=True)         # (B, h4, 128, 128)
        x = F.relu(self.conv3[1](self.conv3[0](x)), inplace=True)
        x = F.relu(self.up4[1](self.up4[0](x)), inplace=True)         # (B, h8, 256, 256)
        x = F.relu(self.conv4[1](self.conv4[0](x)), inplace=True)
        return x


class GNCAFFamilyB(nn.Module):
    """Lost Family-B reconstruction.

    Args:
        hidden_dim: 128 (frozen_gn_h128, 4.1 M) or 256 (demo, 8.8 M).
        n_transformer_layers: 2 (frozen_gn_h128) or 4 (demo).
        feature_dim: 1536 (UNI v2).
        n_classes: 3 ({BG, TLS, GC}).
        n_hops: 3 (paper).
    """

    def __init__(self, hidden_dim: int = 128, n_transformer_layers: int = 2,
                 feature_dim: int = 1536, n_classes: int = 3,
                 n_hops: int = 3, n_heads: int | None = None,
                 decoder_norm: str = "gn"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_encoder = FamilyBPatchEncoder(hidden_dim, n_transformer_layers, n_heads=n_heads)
        self.context_aggregator = FamilyBContextAggregator(feature_dim, hidden_dim, n_hops=n_hops)
        self.fusion = FamilyBFusion(hidden_dim, n_heads=n_heads)
        self.pixel_decoder = FamilyBPixelDecoder(hidden_dim, norm=decoder_norm)
        self.head_seg = nn.Conv2d(hidden_dim // 8, n_classes, 1)

    def forward(self, patches, node_features, edge_index, target_idx):
        """
        patches:        (B, 3, 256, 256) — raw RGB patches for B target nodes
        node_features:  (N_nodes, feature_dim) — UNI features for ALL slide nodes
        edge_index:     (2, N_edges) — kNN graph over slide nodes
        target_idx:     (B,) — node indices in node_features for the B target patches
        """
        # 1. Patch encoder → detail tokens + 2 skip levels
        tokens, skip32, skip64 = self.patch_encoder(patches)  # (B, 256, H), (B, 128, 32, 32), (B, 64, 64, 64)
        # 2. Context aggregator → per-node context vectors over the slide graph
        ctx_all = self.context_aggregator(node_features, edge_index)  # (N, H)
        ctx = ctx_all[target_idx]                                       # (B, H)
        # 3. Fusion: MSA over [ctx; tokens]
        fused = self.fusion(tokens, ctx)                                # (B, 256, H)
        # 4. Reshape to 16×16 spatial and feed to decoder
        B, _, H = fused.shape
        feat = fused.transpose(1, 2).reshape(B, H, 16, 16)
        x = self.pixel_decoder(feat, skip32, skip64)
        return self.head_seg(x)
