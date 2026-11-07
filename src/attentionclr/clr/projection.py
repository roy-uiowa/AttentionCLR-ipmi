"""SimCLR v1 projection head.

Chen et al., ICML 2020 (SimCLR). 2-layer MLP with ReLU, no BatchNorm on the
output so the contrastive loss sees the raw embedding. The penultimate layer
(2048-d) is where downstream linear probe happens; the 128-d output is only
used for the contrastive objective.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
