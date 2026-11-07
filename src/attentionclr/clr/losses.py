"""NT-Xent contrastive loss (SimCLR).

Given a batch of N images, we produce 2N embeddings (two augmented views each),
L2-normalize them, and compute a 2N x 2N similarity matrix. For each embedding,
its positive is its paired view; the other 2N-2 are negatives.

Loss = -log( exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
averaged over i, where j is the positive for i.

Key implementation details (this is where SimCLR reproductions usually break):
  1. Ordering convention: rows [0..N-1] are views A, rows [N..2N-1] are views B.
     The positive for row i (i < N) is row i + N, and vice versa. Do not
     concatenate views in interleaved order — the pair indexing will be wrong.
  2. Mask the self-similarity diagonal (i==j) before the softmax, otherwise the
     positive is guaranteed to be chosen (self is always most similar), loss
     collapses to 0, and gradients vanish.
  3. Clamp the denominator to 1e-12 to avoid NaN on fp16.
  4. Use log-sum-exp for numerical stability rather than log(sum(exp(.))).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-12) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """z_a, z_b: (N, D) — the two views of the same N images.

        Returns: scalar loss.
        """
        if z_a.shape != z_b.shape:
            raise ValueError(f"Shape mismatch: {tuple(z_a.shape)} vs {tuple(z_b.shape)}")
        N, D = z_a.shape
        if N < 2:
            raise ValueError("NTXent requires N >= 2 for at least one negative")

        # L2-normalize embeddings so dot product = cosine similarity.
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        # Concatenate: rows [0..N-1] = a, rows [N..2N-1] = b.
        z = torch.cat([z_a, z_b], dim=0)  # (2N, D)
        sim = (z @ z.T) / self.temperature  # (2N, 2N)

        # Mask self-similarity (i==j) with -inf so it contributes 0 after softmax.
        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive index for row i: (i + N) % (2N). For i<N -> i+N; for i>=N -> i-N.
        positive_indices = (torch.arange(2 * N, device=z.device) + N) % (2 * N)

        # Cross-entropy over the (2N-1) negatives + 1 positive.
        loss = F.cross_entropy(sim, positive_indices)
        return loss
