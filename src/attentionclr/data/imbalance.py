"""Class-imbalance handling: on-the-fly oversample and cap-downsample.

Decision B: on_the_fly_oversample_to is default; cap_downsample is config option.
`max_samples_per_epoch` caps the total sampler size to keep epoch time manageable
(assumption, documented). When set, the effective per-class target is reduced
proportionally so the largest class still dominates by the intended ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import WeightedRandomSampler


@dataclass(frozen=True)
class ImbalanceConfig:
    mode: str
    target_per_class: int
    max_samples_per_epoch: int | None = None


def make_sampler(
    labels: np.ndarray,
    cfg: ImbalanceConfig,
    seed: int = 0,
    num_samples: int | None = None,
) -> WeightedRandomSampler | None:
    if cfg.mode == "none":
        return None
    if cfg.mode not in {"on_the_fly_oversample_to", "cap_downsample"}:
        raise ValueError(f"Unknown imbalance mode: {cfg.mode}")

    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(classes.tolist(), counts.tolist()))

    if cfg.mode == "on_the_fly_oversample_to":
        per_class_weight = {
            c: (cfg.target_per_class / class_counts[c]) if class_counts[c] < cfg.target_per_class else 1.0
            for c in class_counts
        }
    else:  # cap_downsample
        per_class_weight = {
            c: (cfg.target_per_class / class_counts[c]) if class_counts[c] > cfg.target_per_class else 1.0
            for c in class_counts
        }

    weights = np.array([per_class_weight[int(l)] for l in labels], dtype=np.float64)
    weights = weights / weights.sum()

    if num_samples is None:
        if cfg.mode == "on_the_fly_oversample_to":
            num_samples = sum(max(cfg.target_per_class, class_counts[c]) for c in class_counts)
        else:
            num_samples = sum(min(cfg.target_per_class, class_counts[c]) for c in class_counts)
        if cfg.max_samples_per_epoch is not None:
            num_samples = min(num_samples, cfg.max_samples_per_epoch)

    import torch
    gen = torch.Generator()
    gen.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights.tolist(), num_samples=num_samples, replacement=True, generator=gen
    )