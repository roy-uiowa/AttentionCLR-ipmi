"""Forward/backward hooks for Grad-CAM and penultimate feature extraction."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ActivationGradient:
    """Holds the activation and its gradient for one forward+backward pass."""

    activation: torch.Tensor | None = None
    gradient: torch.Tensor | None = None

    def clear(self) -> None:
        self.activation = None
        self.gradient = None


class GradCAMHook:
    """Attach to a conv layer. Stores activation in forward, grad in backward."""

    def __init__(self, layer: nn.Module) -> None:
        self.store = ActivationGradient()
        self._fwd = layer.register_forward_hook(self._forward_hook)
        self._bwd = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.store.activation = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.store.gradient = grad_output[0].detach()

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


class FeatureHook:
    """Capture the output of a layer (e.g. avgpool -> 2048-d) as a flat feature vector."""

    def __init__(self, layer: nn.Module) -> None:
        self.features: torch.Tensor | None = None
        self._fwd = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.features = output.detach()

    def remove(self) -> None:
        self._fwd.remove()
