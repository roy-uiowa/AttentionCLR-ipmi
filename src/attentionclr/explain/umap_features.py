"""UMAP of the penultimate feature vector, colored by true class (paper Fig 6)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from attentionclr.explain.hooks import FeatureHook


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    hook_layer: torch.nn.Module,
    device: str = "cuda",
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate features and labels for every batch in `loader`.

    Returns:
        features: (N, D) float32, D = feature dim of `hook_layer` output.
        labels:   (N,) int64.
    """
    model.eval().to(device)
    hook = FeatureHook(hook_layer)
    feats, labels = [], []
    n = 0
    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)                    # hook captures penultimate
            f = hook.features               # (B, D) or (B, D, 1, 1)
            if f.dim() > 2:
                f = torch.flatten(f, 1)
            feats.append(f.cpu())
            labels.append(y)
            n += x.size(0)
            if max_samples is not None and n >= max_samples:
                break
    finally:
        hook.remove()
    F = torch.cat(feats, dim=0).numpy()
    Y = torch.cat(labels, dim=0).numpy()
    if max_samples is not None:
        F, Y = F[:max_samples], Y[:max_samples]
    return F, Y


def umap_2d(
    features: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """Project (N, D) features to (N, 2) with UMAP. Returns the 2D embedding."""
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
    )
    return reducer.fit_transform(features)
