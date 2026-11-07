"""ResNeXt-50 encoder with CBAM inserted after each residual stage.

Topology (torchvision resnext50_32x4d):
    stem       -> (B,  64, 125, 125)
    layer1     -> (B, 256,  63,  63)   + CBAM(256)
    layer2     -> (B, 512,  32,  32)   + CBAM(512)
    layer3     -> (B, 1024, 16,  16)   + CBAM(1024)
    layer4     -> (B, 2048,  8,   8)   + CBAM(2048)
    avgpool    -> (B, 2048,  1,   1)
    flatten    -> (B, 2048)

The CBAM is inserted *after* each stage's output, not inside the residual block,
so the residual identity path is untouched. This is the standard CBAM-in-ResNet
placement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d

from attentionclr.clr.cbam import CBAM


class ResNeXt50CBAMEncoder(nn.Module):
    """ResNeXt-50 + CBAM encoder. forward() returns the 2048-d avgpool feature."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        cbam_reduction: int = 16,
    ) -> None:
        super().__init__()
        backbone = resnext50_32x4d(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        if in_channels != 3:
            # Replace the stem conv if the channel count differs. Not used for MGG
            # (3-channel RGB) but kept for parity with the earlier supervised model.
            old = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )

        # Extract the stages, add CBAM after each, and rebuild a sequential.
        # Stage output channel counts for ResNeXt-50: 256, 512, 1024, 2048.
        stage_channels = [256, 512, 1024, 2048]
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = nn.Sequential(backbone.layer1, CBAM(stage_channels[0], cbam_reduction))
        self.layer2 = nn.Sequential(backbone.layer2, CBAM(stage_channels[1], cbam_reduction))
        self.layer3 = nn.Sequential(backbone.layer3, CBAM(stage_channels[2], cbam_reduction))
        self.layer4 = nn.Sequential(backbone.layer4, CBAM(stage_channels[3], cbam_reduction))
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d((1,1))

        self.feature_dim = 2048
        self.in_channels = in_channels

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (B, 2048) feature vector (after avgpool + flatten)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
