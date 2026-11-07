"""Grad-CAM: weighted sum of conv activations, weighted by gradient.

Selvaraju et al., https://arxiv.org/abs/1610.02391 (paper ref 41).

Steps:
  1. Forward hook captures A = activation of the target conv layer: (B, C, h, w).
  2. Backward hook captures dScore/dA: (B, C, h, w).
  3. Weights w_c = spatial-mean of dScore/dA over (h, w).
  4. CAM = ReLU(sum_c w_c * A_c).
  5. Upsample to input size and normalize to [0, 1].

The target conv layer for ResNeXt-50 is model.model.layer4[-1].conv3.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from attentionclr.explain.hooks import GradCAMHook


def gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Grad-CAM for a batch.

    Args:
        model: eval-mode classifier.
        target_layer: the conv layer whose activations define the CAM.
        x: (B, 3, H, W) input batch.
        target: (B,) target class. Defaults to argmax(model(x)).

    Returns:
        cam: (B, H, W) in [0, 1], upsampled to input size.
        predicted: (B,) target class used.
    """
    model.eval()
    x = x.detach()
    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1)
    target = target.detach()

    hook = GradCAMHook(target_layer)
    try:
        x = x.requires_grad_(True)
        logits = model(x)
        score = logits.gather(1, target.view(-1, 1)).sum()
        score.backward()

        A = hook.store.activation            # (B, C, h, w)
        G = hook.store.gradient              # (B, C, h, w)
        if A is None or G is None:
            raise RuntimeError("Grad-CAM hook did not capture activation/gradient")
        # Weights: global-average-pool the gradient.
        w = G.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (w * A).sum(dim=1, keepdim=True).clamp(min=0.0)  # (B, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)                 # (B, H, W)
        # Per-image normalization to [0, 1].
        lo = cam.amin(dim=(1, 2), keepdim=True)
        hi = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - lo) / (hi - lo).clamp(min=1e-8)
    finally:
        hook.remove()
    return cam.detach(), target


def resnext50_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Locate ResNeXt-50's last conv layer inside our ResNeXt50Classifier wrapper.

    Our wrapper: ResNeXt50Classifier.model = torchvision's ResNeXt; .layer4 is the
    last residual block; [-1].conv3 is the last 1x1 conv. Output: (B, 2048, 8, 8)
    at 250x250 input (stride 32).
    """
    return model.model.layer4[-1].conv3
