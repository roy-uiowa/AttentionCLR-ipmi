"""SmoothGrad: average saliency over noisy copies of the input.

Smilkov et al., https://arxiv.org/abs/1706.03825 (paper ref 40).

Saliency here = |d(score_target)/d(input)|, summed over channels, then averaged
over the noise samples. Higher = more important pixel (for the target class).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _saliency_no_grad_placeholder() -> None:
    pass


def smoothgrad(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor | None = None,
    n_samples: int = 50,
    noise_std: float = 0.10,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SmoothGrad saliency for a batch.

    Args:
        model: eval-mode classifier, forward -> (B, C) logits.
        x: (B, 3, H, W) input batch, already normalized.
        target: (B,) target class indices. Defaults to argmax(model(x)).
        n_samples: number of noisy forward+backward passes.
        noise_std: Gaussian noise std (in normalized-input units).
        seed: for reproducibility.

    Returns:
        saliency: (B, H, W) non-negative, channel-summed, noise-averaged.
        predicted: (B,) the target class used (so caller can label the plot).
    """
    model.eval()
    x = x.detach()
    B, C, H, W = x.shape
    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1)
    target = target.detach()

    g = torch.Generator(device=x.device).manual_seed(seed)
    saliency = torch.zeros(B, H, W, device=x.device)
    for _ in range(n_samples):
        noise = torch.randn(x.shape, device=x.device, generator=g) * noise_std
        x_noisy = (x + noise).requires_grad_(True)
        logits = model(x_noisy)
        score = logits.gather(1, target.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, x_noisy, retain_graph=False)[0]
        saliency += grads.abs().sum(dim=1)
    saliency /= n_samples

    # Normalize per-image to [0, 1] for visualization (preserve relative importance).
    sal_min = saliency.amin(dim=(1, 2), keepdim=True)
    sal_max = saliency.amax(dim=(1, 2), keepdim=True)
    saliency = (saliency - sal_min) / (sal_max - sal_min).clamp(min=1e-8)
    return saliency, target
