"""v3.37 — RegionDecoder: RGB + UNI-v2 + GAT context fused decoder over a
3×3 (or k×k) patch region. Stage 1 stays unchanged; this replaces Stage 2.

The user's framing: at biobank scale, run Stage 1 on every slide for
counting (cheap graph forward, ~10 ms/slide). Run THIS heavy decoder
ONLY on candidate regions Stage 1 surfaces — typically <1 % of patches —
so per-slide cost stays bounded and the RGB ceiling becomes accessible.

Cost target on A100: <0.3 s/slide for 50–100 candidate regions per slide.
At biobank scale (20 k slides) → ~1.5 h total. ~300× cheaper than GNCAF.

Forward signature:
    rgb_tiles:    (B, P, 3, 256, 256)  uint8 ImageNet-normalised
    uni_features: (B, P, 1536)         UNI-v2 features
    graph_ctx:    (B, P, gat_dim)      Stage 1 GAT outputs (residual+norm)
    valid_mask:   (B, P) bool          P = grid_n*grid_n cells per region

Output:
    logits: (B, 3, grid_n*256, grid_n*256)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gars_stage2 import UpDoubleConv


# ─── Stage 1 graph-context extraction ─────────────────────────────────


def extract_stage1_context(stage1, features: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
    """Run Stage 1 (GraphTLSDetector) forward up to before the head;
    return the per-patch graph context (N, hidden_dim) — the same tensor
    Stage 1 feeds to its scoring head.
    """
    x = stage1.proj(features)
    for layer, norm in zip(stage1.gnn_layers, stage1.gnn_norms):
        x = norm(layer(x, edge_index) + x)
    return x   # (N, hidden_dim)


# ─── RGB encoder (ResNet-18, last spatial map at 1/32) ────────────────


class RGBEncoder(nn.Module):
    """Truncated ResNet-18: input (3, 256, 256) → output (512, 8, 8).

    Returns multi-scale features for FPN-style fusion in the decoder:
        c1: (64,  64, 64)   after layer1
        c2: (128, 32, 32)   after layer2
        c3: (256, 16, 16)   after layer3
        c4: (512,  8,  8)   after layer4
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        rn = resnet18(weights=weights)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
        x = self.stem(x)               # (64, 64, 64)
        c1 = self.layer1(x)            # (64, 64, 64)
        c2 = self.layer2(c1)           # (128, 32, 32)
        c3 = self.layer3(c2)           # (256, 16, 16)
        c4 = self.layer4(c3)           # (512,  8,  8)
        return c1, c2, c3, c4


# ─── Region decoder ───────────────────────────────────────────────────


class RegionDecoder(nn.Module):
    """k×k region decoder: per-patch RGB + UNI + GAT → joint pixel mask.

    Per cell: ResNet-18 → (512, 8, 8) → fuse with UNI (Linear 1536→512)
    + graph context (Linear gat_dim→512), all broadcast-added to the
    8×8 feature map. Cells tile spatially into (B, 512, 8k, 8k); decoder
    upsamples 2× five times to reach (B, 3, 256k, 256k).

    Pad cells (invalid) get a learned (3, 256, 256) RGB embedding (trained
    via valid_mask suppression), and learned UNI / graph pad embeddings
    for the per-cell projections.
    """

    def __init__(
        self,
        uni_dim: int = 1536,
        gat_dim: int = 256,
        hidden_channels: int = 64,
        n_classes: int = 3,
        grid_n: int = 3,
        rgb_pretrained: bool = True,
        freeze_rgb_encoder: bool = False,
        head_mode: str = "argmax",
    ):
        super().__init__()
        self.head_mode = head_mode
        self.grid_n = grid_n
        self.n_cells = grid_n * grid_n
        self.fuse_dim = 512

        self.rgb_encoder = RGBEncoder(pretrained=rgb_pretrained)
        if freeze_rgb_encoder:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False
        self.uni_proj = nn.Sequential(
            nn.Linear(uni_dim, self.fuse_dim),
            nn.GELU(),
            nn.LayerNorm(self.fuse_dim),
        )
        self.gat_proj = nn.Sequential(
            nn.Linear(gat_dim, self.fuse_dim),
            nn.GELU(),
            nn.LayerNorm(self.fuse_dim),
        )
        # Learned pad embeddings for invalid cells.
        self.uni_pad = nn.Parameter(torch.zeros(uni_dim))
        self.gat_pad = nn.Parameter(torch.zeros(gat_dim))
        # Decoder. 8k → 16k → 32k → 64k → 128k → 256k for grid_n=k.
        # Channel sequence: 512 → 256 → 128 → 64 → 32 → hidden_channels.
        self.dec0 = UpDoubleConv(self.fuse_dim, 256)
        self.dec1 = UpDoubleConv(256, 128)
        self.dec2 = UpDoubleConv(128, 64)
        self.dec3 = UpDoubleConv(64, 32)
        self.dec4 = UpDoubleConv(32, hidden_channels)
        if head_mode == "dual_sigmoid":
            # v3.38: independent binary heads for TLS (target>=1) and GC (target==2).
            # Biology: GC ⊂ TLS; argmax forces them to compete for the same pixel.
            self.head_tls = nn.Conv2d(hidden_channels, 1, kernel_size=1)
            self.head_gc = nn.Conv2d(hidden_channels, 1, kernel_size=1)
            self.seg_head = None
        else:
            self.seg_head = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)
            self.head_tls = None
            self.head_gc = None

    def forward(
        self,
        rgb_tiles: torch.Tensor,        # (B, P, 3, 256, 256)
        uni_features: torch.Tensor,     # (B, P, uni_dim)
        graph_ctx: torch.Tensor,        # (B, P, gat_dim)
        valid_mask: torch.Tensor,       # (B, P) bool
    ) -> torch.Tensor:
        b, p = uni_features.shape[:2]
        assert p == self.n_cells, f"P={p} != grid_n²={self.n_cells}"

        # Substitute pad embeddings on invalid cells.
        invalid = (~valid_mask)                              # (B, P)
        if invalid.any():
            uni_features = torch.where(
                invalid.unsqueeze(-1),
                self.uni_pad.expand_as(uni_features), uni_features
            )
            graph_ctx = torch.where(
                invalid.unsqueeze(-1),
                self.gat_pad.expand_as(graph_ctx), graph_ctx
            )

        # ResNet on each cell. Flatten (B*P, 3, 256, 256).
        rgb_flat = rgb_tiles.reshape(b * p, *rgb_tiles.shape[-3:])
        # For invalid cells, zero out the RGB so the encoder produces a
        # consistent zero-tile feature (which the decoder learns to handle
        # via the same valid_mask path used in v3.36).
        if invalid.any():
            inv_flat = invalid.reshape(b * p, 1, 1, 1)
            rgb_flat = torch.where(inv_flat, torch.zeros_like(rgb_flat), rgb_flat)
        _c1, _c2, _c3, c4 = self.rgb_encoder(rgb_flat)        # (B*P, 512, 8, 8)

        # Per-cell feature fusion: c4 + projected UNI broadcast + projected graph broadcast.
        uni_proj = self.uni_proj(uni_features)               # (B, P, fuse_dim)
        gat_proj = self.gat_proj(graph_ctx)                  # (B, P, fuse_dim)
        uni_proj = uni_proj.reshape(b * p, self.fuse_dim, 1, 1)
        gat_proj = gat_proj.reshape(b * p, self.fuse_dim, 1, 1)
        fused = c4 + uni_proj + gat_proj                     # (B*P, fuse_dim, 8, 8)
        # Suppress invalid-cell features to zero (decoder handles via context).
        if invalid.any():
            inv_flat_2d = invalid.reshape(b * p, 1, 1, 1).float()
            fused = fused * (1.0 - inv_flat_2d)

        # Reshape into (B, P, fuse_dim, 8, 8) → tile into 8k × 8k.
        z = fused.view(b, self.grid_n, self.grid_n, self.fuse_dim, 8, 8)
        z = z.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            b, self.fuse_dim, self.grid_n * 8, self.grid_n * 8
        )
        # Decoder. 8·k → 16k → 32k → 64k → 128k → 256k.
        z = self.dec0(z)            # 16k
        z = self.dec1(z)            # 32k
        z = self.dec2(z)            # 64k
        z = self.dec3(z)            # 128k
        z = self.dec4(z)            # 256k
        if self.head_mode == "dual_sigmoid":
            tls_logits = self.head_tls(z)
            gc_logits = self.head_gc(z)
            return (tls_logits, gc_logits)
        return self.seg_head(z)     # (B, n_classes, 256·k, 256·k)


def model_summary(model: RegionDecoder) -> dict[str, int]:
    parts = {
        "rgb_encoder": sum(p.numel() for p in model.rgb_encoder.parameters()),
        "uni_proj":    sum(p.numel() for p in model.uni_proj.parameters()),
        "gat_proj":    sum(p.numel() for p in model.gat_proj.parameters()),
        "decoder":     sum(p.numel()
                            for n, p in model.named_parameters()
                            if n.startswith(("dec0", "dec1", "dec2", "dec3", "dec4", "seg_head"))),
    }
    parts["pad_embeds"] = model.uni_pad.numel() + model.gat_pad.numel()
    parts["total"] = sum(p.numel() for p in model.parameters())
    return parts


if __name__ == "__main__":
    m = RegionDecoder()
    print({k: f"{v:,}" for k, v in model_summary(m).items()})
    B, P = 2, 9
    rgb = torch.randn(B, P, 3, 256, 256)
    uni = torch.randn(B, P, 1536)
    gat = torch.randn(B, P, 256)
    valid = torch.ones(B, P, dtype=torch.bool); valid[0, 8] = False
    y = m(rgb, uni, gat, valid)
    print(f"out: {tuple(y.shape)}")  # → (2, 3, 768, 768)
