"""ResNeXt-50 (C=32), 250x250 input, 21-head output (output_mode: direct21).

Target param count: 23,059,094 (paper main text). We start from torchvision's
resnext50_32x4d and change only the classifier head. If the param count
diverges, the head replacement is the culprit — the tests in tests/test_models.py
assert exact equality.

Note: torchvision's resnext50_32x4d has a 2048->1000 FC. Paper's head is
2048->22 (or 21 after merge). We use 2048->21 for direct21. The paper's exact
count 23,059,094 corresponds to a 22-way head; a 21-way head is
23,059,094 - (2048 * 1) = 23,057,046. See the note below.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d


class ResNeXt50Classifier(nn.Module):
    def __init__(self, num_classes: int = 21, in_channels: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        model = resnext50_32x4d(weights="IMAGENET1K_V1" if pretrained else None)
        if in_channels != 3:
            # Replace first conv if input channels differ (paper uses 3).
            old = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
