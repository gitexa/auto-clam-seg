"""TransUNet GNCAF (`GNCAFPixelDecoder`) — reverse-engineered from
state_dict shapes of the existing `gncaf_pixel_*` checkpoints on
`/lambda/nfs/.../experiments/`.

Architecture (verified from
`gncaf_pixel_100slides_lr3e4_gcn3hop_fusion_2983e563/best_checkpoint.pt`,
ep23, mDice=0.714):

- **encoder.stem_conv**: Conv2d(3, 64, 7×7, stride=2) + BN(64)
- **encoder.maxpool**: 3×3 stride=2 (no params)
- **encoder.layer1**: 3 Bottleneck blocks (256 ch, /4 spatial)
- **encoder.layer2**: 4 Bottleneck blocks (512 ch, /8 spatial)
- **encoder.layer3**: 6 Bottleneck blocks (1024 ch, /16 spatial)
- **encoder.patch_embed**: Conv2d(1024 → 768, 1×1)
- **encoder.pos_embed**: (1, 256, 768) — 16×16 token grid
- **encoder.blocks**: 6× ViTBlock(768, 12-head, mlp=3072)
- **encoder.norm**: LayerNorm(768)
- **gcn.proj**: Linear(1536→768) + LayerNorm(768)
- **gcn.gcn_layers**: 3× GCNConv(768→768)
- **gcn.gcn_norms**: 3× LayerNorm(768)
- **fusion**: 1× FusionBlock(768, 12, 3072) — prepend ctx token, attn,
  mlp, drop the ctx token
- **decoder.head**: Conv2d(768→512 1×1) + BN(512)
- **decoder.blocks**: 4× DecoderBlock — Conv2d 3×3 + BN. Channels:
  - blocks.0: 1024 → 256  (input = 512 from head + 512 skip from layer2)
  - blocks.1: 512 → 128  (input = 256 from prev + 256 skip from layer1)
  - blocks.2: 192 → 64   (input = 128 from prev + 64  skip from stem)
  - blocks.3: 64 → 16     (input = 64  from prev, no skip)
- **head_seg**: Conv2d(16 → n_classes 3×3)

Spatial flow: 256×256 → /2 (stem) → /4 (layer1) → /8 (layer2)
→ /16 (layer3 → ViT) → /8 (block0) → /4 (block1) → /2 (block2) → /1 (block3) → head.

65.6 M params total at hidden_size=768 / 6 ViT layers / 3 GCN hops /
1 fusion block.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─── Bottleneck (R50) ─────────────────────────────────────────────────


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes * Bottleneck.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * Bottleneck.expansion),
        )
    layers = [Bottleneck(inplanes, planes, stride, downsample)]
    inplanes = planes * Bottleneck.expansion
    for _ in range(1, blocks):
        layers.append(Bottleneck(inplanes, planes))
    return nn.Sequential(*layers)


# ─── ViT block ────────────────────────────────────────────────────────


class ViTBlock(nn.Module):
    def __init__(self, dim: int = 768, n_heads: int = 12,
                 mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        # mlp is Sequential[Linear, GELU, Dropout, Linear, Dropout].
        # state_dict shows weights at indices 0 and 3 only.
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),     # 0
            nn.GELU(),                    # 1 (no params)
            nn.Dropout(dropout),          # 2 (no params)
            nn.Linear(mlp_dim, dim),     # 3
            nn.Dropout(dropout),          # 4 (no params)
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


# ─── TransUNet encoder (R50 + 6 ViT) ─────────────────────────────────


class TransUNetEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768, n_layers: int = 6,
                 n_heads: int = 12, mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = _make_layer(64, 64, blocks=3, stride=1)        # /4 → 256ch
        self.layer2 = _make_layer(256, 128, blocks=4, stride=2)      # /8 → 512ch
        self.layer3 = _make_layer(512, 256, blocks=6, stride=2)      # /16 → 1024ch
        self.patch_embed = nn.Conv2d(1024, hidden_size, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))
        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, n_heads, mlp_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        # Stem (skip-1: pre-maxpool, /2 spatial, 64ch)
        s_stem = F.relu(self.stem_conv(x))                  # (B, 64, 128, 128) for 256-input
        x = self.maxpool(s_stem)                            # (B, 64, 64, 64)
        s_layer1 = self.layer1(x)                           # (B, 256, 64, 64)
        s_layer2 = self.layer2(s_layer1)                    # (B, 512, 32, 32)
        x = self.layer3(s_layer2)                           # (B, 1024, 16, 16)
        x = self.patch_embed(x)                             # (B, 768, 16, 16)
        # Tokenise: (B, T=256, 768)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)               # (B, 256, 768)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens, (s_stem, s_layer1, s_layer2)


def load_imagenet_r50_into_encoder(encoder: "TransUNetEncoder") -> tuple[int, int]:
    """Load torchvision ImageNet-pretrained ResNet50 weights into a
    TransUNetEncoder's R50 trunk (stem + layer1/2/3). Returns
    (n_loaded, n_skipped) for diagnostics. layer4 + ViT not touched.
    """
    import torchvision
    src = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    ).state_dict()
    target = encoder.state_dict()

    def _copy(src_key: str, dst_key: str) -> bool:
        if src_key in src and dst_key in target and src[src_key].shape == target[dst_key].shape:
            target[dst_key] = src[src_key].clone()
            return True
        return False

    loaded = skipped = 0
    # stem: conv1.* → stem_conv.0.*; bn1.* → stem_conv.1.*
    for src_k, dst_k in [
        ("conv1.weight", "stem_conv.0.weight"),
        ("bn1.weight", "stem_conv.1.weight"),
        ("bn1.bias", "stem_conv.1.bias"),
        ("bn1.running_mean", "stem_conv.1.running_mean"),
        ("bn1.running_var", "stem_conv.1.running_var"),
        ("bn1.num_batches_tracked", "stem_conv.1.num_batches_tracked"),
    ]:
        if _copy(src_k, dst_k):
            loaded += 1
        else:
            skipped += 1

    # layers 1-3 share key shape
    for layer_name in ("layer1", "layer2", "layer3"):
        for src_k in [k for k in src if k.startswith(f"{layer_name}.")]:
            dst_k = src_k  # same name
            if _copy(src_k, dst_k):
                loaded += 1
            else:
                skipped += 1

    encoder.load_state_dict(target, strict=False)
    return loaded, skipped


# ─── GCN context ──────────────────────────────────────────────────────


class GCNContext(nn.Module):
    """3-hop GCN with concat-then-norm aggregation. Output shape (N, hidden)."""

    def __init__(self, in_features: int = 1536, hidden_dim: int = 768,
                 n_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        # State_dict keys: gcn.proj.0.{weight,bias} (Linear), gcn.proj.1.{weight,bias} (LN)
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
            for _ in range(n_hops)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_hops)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for layer, norm in zip(self.gcn_layers, self.gcn_norms):
            h = F.gelu(layer(x, edge_index))
            x = norm(x + self.dropout(h))
        return x


# ─── Fusion block ─────────────────────────────────────────────────────


class FusionBlock(nn.Module):
    """Same shape as a ViT block but operates on a sequence with a
    prepended context token. Calling code drops the ctx token after."""

    def __init__(self, dim: int = 768, n_heads: int = 12,
                 mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


# ─── Decoder ──────────────────────────────────────────────────────────


class _DecoderBlock(nn.Module):
    """Conv 3×3 + BN + ReLU. The bilinear upsample is done by the caller
    so the conv sees skip-concatenated input at the new resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class TransUNetDecoder(nn.Module):
    def __init__(self, in_dim: int = 768):
        super().__init__()
        # head reduces ViT 768 → 512.
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 512, 1),
            nn.BatchNorm2d(512),
        )
        # Skip channels: layer2=512 (/8), layer1=256 (/4), stem=64 (/2), then /1 (no skip).
        self.blocks = nn.ModuleList([
            _DecoderBlock(512 + 512, 256),     # /8: 512+layer2 → 256
            _DecoderBlock(256 + 256, 128),     # /4: 256+layer1 → 128
            _DecoderBlock(128 + 64,  64),      # /2: 128+stem   → 64
            _DecoderBlock(64,         16),      # /1: 64         → 16
        ])

    def forward(self, x: torch.Tensor, skips: tuple) -> torch.Tensor:
        s_stem, s_layer1, s_layer2 = skips
        x = F.relu(self.head(x))                         # (B, 512, 16, 16)
        # /16 → /8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s_layer2], dim=1)              # (B, 1024, 32, 32)
        x = self.blocks[0](x)                             # (B, 256, 32, 32)
        # /8 → /4
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s_layer1], dim=1)              # (B, 512, 64, 64)
        x = self.blocks[1](x)                             # (B, 128, 64, 64)
        # /4 → /2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s_stem], dim=1)                # (B, 192, 128, 128)
        x = self.blocks[2](x)                             # (B, 64, 128, 128)
        # /2 → /1
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.blocks[3](x)                             # (B, 16, 256, 256)
        return x


# ─── Whole model ──────────────────────────────────────────────────────


class GNCAFPixelDecoder(nn.Module):
    """TransUNet R50 + 6 ViT + GCN-3hop + 1 fusion + TransUNet decoder.

    Forward signature matches the recovered training script:
        target_rgb:    (B, 3, 256, 256)  ImageNet-normalised float32
        all_features:  (N, 1536)         UNI-v2 features for every patch in slide
        target_idx:    (B,)              indices of the target patches in `all_features`
        edge_index:    (2, E)            graph edges
    Output: (B, n_classes, 256, 256) logits.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        n_classes: int = 3,
        n_encoder_layers: int = 6,
        n_heads: int = 12,
        mlp_dim: int = 3072,
        n_hops: int = 3,
        n_fusion_layers: int = 1,
        feature_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransUNetEncoder(
            hidden_size=hidden_size, n_layers=n_encoder_layers,
            n_heads=n_heads, mlp_dim=mlp_dim, dropout=dropout,
        )
        self.gcn = GCNContext(
            in_features=feature_dim, hidden_dim=hidden_size,
            n_hops=n_hops, dropout=dropout,
        )
        self.fusion = nn.ModuleList([
            FusionBlock(hidden_size, n_heads, mlp_dim, dropout)
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
        tokens, skips = self.encoder(target_rgb)               # (B, 256, 768), skip tuple
        node_ctx = self.gcn(all_features, edge_index)          # (N, 768)
        local_ctx = node_ctx[target_idx]                       # (B, 768)

        # Prepend ctx token, fuse, drop ctx token.
        ctx_t = local_ctx.unsqueeze(1)                          # (B, 1, 768)
        x = torch.cat([ctx_t, tokens], dim=1)                  # (B, 257, 768)
        for blk in self.fusion:
            x = blk(x)
        x = x[:, 1:]                                            # (B, 256, 768) — drop ctx

        # Spatial reshape + decoder + head.
        b, t, d = x.shape
        x = x.transpose(1, 2).reshape(b, d, 16, 16)
        x = self.decoder(x, skips)                              # (B, 16, 256, 256)
        return self.head_seg(x)                                 # (B, 3, 256, 256)


def model_summary(m: GNCAFPixelDecoder) -> dict:
    return {
        "encoder": sum(p.numel() for p in m.encoder.parameters()),
        "gcn":     sum(p.numel() for p in m.gcn.parameters()),
        "fusion":  sum(p.numel() for p in m.fusion.parameters()),
        "decoder": sum(p.numel() for p in m.decoder.parameters()),
        "head":    sum(p.numel() for p in m.head_seg.parameters()),
        "total":   sum(p.numel() for p in m.parameters()),
    }


if __name__ == "__main__":
    m = GNCAFPixelDecoder().eval()
    print({k: f"{v:,}" for k, v in model_summary(m).items()})
    B, N, E = 2, 200, 600
    rgb = torch.randn(B, 3, 256, 256)
    feats = torch.randn(N, 1536)
    tidx = torch.randint(0, N, (B,))
    ei = torch.randint(0, N, (2, E))
    y = m(rgb, feats, tidx, ei)
    print(f"forward OK: {tuple(y.shape)}")
