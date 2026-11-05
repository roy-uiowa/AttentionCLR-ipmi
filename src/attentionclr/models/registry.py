"""Name -> model constructor."""
from __future__ import annotations

from attentionclr.models.resnext50 import ResNeXt50Classifier
from attentionclr.models.sequential import SequentialCNN

_REGISTRY = {
    "resnext50": ResNeXt50Classifier,
    "sequential": SequentialCNN,
}


def build_model(name: str, num_classes: int = 21, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes, **kwargs)
