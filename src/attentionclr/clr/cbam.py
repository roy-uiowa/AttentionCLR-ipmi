"""Convolutional Block Attention Module (CBAM).

Woo et al., ECCV 2018. https://arxiv.org/abs/1807.06521

Channel attention: MLP over (avg-pool, max-pool) of the feature map → sigmoid mask.
Spatial attention: 7x7 conv over (channel-avg, channel-max) → sigmoid mask.

Both are multiplicative residual gates: output = input * channel_mask * spatial_mask.
Shapes are preserved exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention: shared 2-layer MLP on avg- and max-pooled feature maps."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)  # never collapse to 0 for tiny channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # The MLP is shared between avg and max branches (parameter sharing is
        # the key design choice of CBAM channel attention).
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention: 7x7 conv over channel-wise avg and max."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        # 2 input channels (avg, max) -> 1 output (spatial mask)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    """Full CBAM module: channel attention then spatial attention, both multiplicative."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)  # broadcast (B,C,1,1) * (B,C,H,W)
        x = x * self.spatial_attn(x)  # broadcast (B,1,H,W) * (B,C,H,W)
        return x
