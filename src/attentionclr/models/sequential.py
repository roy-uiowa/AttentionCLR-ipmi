"""Sequential CNN per Supplement Fig S5, 250x250 in, 21-head out.

Paper reports 303,694 params (22-head). A direct reading of Fig S5 gives
96,125 (21-head) / 96,382 (22-head). Best-fit sweep variant lands at 292,605
(Δ = -10,832) with k=5, dense=1024 — not a clean structural unit.

Decision: ship the S5-literal architecture at 96,125, document the 207,312-param
divergence in reports/assumptions.md, and query the authors for the exact spec.
This model still serves its purpose in the paper (architecture-robustness check).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SequentialCNN(nn.Module):
    def __init__(self, num_classes: int = 21, in_channels: int = 3) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, 8), nn.MaxPool2d(4),
            block(8, 16),          nn.MaxPool2d(4),
            block(16, 32),         nn.MaxPool2d(5),
            block(32, 64),         nn.MaxPool2d(3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))