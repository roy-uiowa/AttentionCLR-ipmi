"""SimCLR augmentation pipeline for MGG-stained bone marrow cells.

Two independently augmented views per image, designed to preserve cell
morphology (nucleus shape, cytoplasm granule color) while destroying
nuisance variation (position, orientation, exposure).

Design constraints from MGG biology:
  - Color is diagnostically meaningful (eosin pink, baso blue). NEVER grayscale.
  - Hue jitter is mild (±0.05): simulates stain-run variation, preserves category.
  - Crop scale (0.5, 1.0): ensures partial-cell and full-cell views.
  - Rotation ±180°: cells have no canonical orientation.
  - Blur/erasing small enough to preserve the nucleus.

The two views are independently sampled — this is what makes the contrastive
task non-trivial. SimCLR with a single random transform + its inverse collapses
to predicting the inverse, not learning invariant features.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# Standard ImageNet mean/std. The supervised ResNeXt-50 baseline used these
# (mean=0.5, std=0.5 in data/transforms.py), but for SSL pretraining the choice
# of normalization matters less than for supervised transfer — the projection
# head and NT-Xent are scale-invariant after L2-normalize. We use ImageNet stats
# here because they match the downstream fine-tune target and any future
# ImageNet transfer baseline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SimCLRAugmentConfig:
    """Configuration for SimCLR augmentation. All magnitudes are opinionated
    defaults informed by MGG biology; override only with a documented reason."""
    image_size: int = 250
    # Random resized crop
    crop_scale: tuple[float, float] = (0.5, 1.0)
    crop_ratio: tuple[float, float] = (0.75, 1.33)
    # Geometric
    rotation_deg: float = 180.0
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    # Color (mild)
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    # Blur / noise / erasing
    blur_p: float = 0.5
    blur_kernel: tuple[int, int] = (3, 7)
    blur_sigma: tuple[float, float] = (0.1, 2.0)
    gaussian_noise_p: float = 0.3
    gaussian_noise_std: float = 15.0 / 255.0
    erasing_p: float = 0.25
    erasing_area: tuple[float, float] = (0.02, 0.10)
    # Normalization
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


class SimCLRAugment:
    """One-view augmentation. Call twice with independent RNG for two views.

    Uses torchvision.transforms.v2, which supports batched tensors and is
    compatible with PIL Image inputs (the format our dataset yields).
    """

    def __init__(self, cfg: SimCLRAugmentConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or SimCLRAugmentConfig()
        c = self.cfg
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

        from torchvision.transforms import v2

        self.transform = v2.Compose([
            v2.Resize((c.image_size, c.image_size), antialias=True),
            v2.RandomResizedCrop(
                size=(c.image_size, c.image_size),
                scale=c.crop_scale,
                ratio=c.crop_ratio,
                antialias=True,
            ),
            v2.RandomRotation(degrees=c.rotation_deg),
            v2.RandomHorizontalFlip(p=c.hflip_p),
            v2.RandomVerticalFlip(p=c.vflip_p),
            v2.ColorJitter(
                brightness=c.brightness,
                contrast=c.contrast,
                saturation=c.saturation,
                hue=c.hue,
            ),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=c.blur_p),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def _apply_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise with probability p to a normalized tensor."""
        if torch.rand(1, generator=self._rng).item() >= self.cfg.gaussian_noise_p:
            return x
        noise = torch.randn(x.shape, generator=self._rng) * self.cfg.gaussian_noise_std
        return (x + noise).clamp(-3.0, 3.0)  # after normalization, valid range is ~[-2.7, 2.7]
    
    def _apply_erasing(self, x: torch.Tensor) -> torch.Tensor:
        """Random erasing with small area; forces the model not to rely on a
        single spatial region. Area 2-10% of image."""
        if torch.rand(1, generator=self._rng).item() >= self.cfg.erasing_p:
            return x
        _, H, W = x.shape
        lo, hi = self.cfg.erasing_area
        area_frac = lo + (hi - lo) * torch.rand(1, generator=self._rng).item()
        area = area_frac * H * W
        aspect = 0.3 + (3.3 - 0.3) * torch.rand(1, generator=self._rng).item()
        h = int(round((area * aspect) ** 0.5))
        w = int(round((area / aspect) ** 0.5))
        if h >= H or w >= W:
            return x
        top = int(torch.randint(0, H - h, (1,), generator=self._rng).item())
        left = int(torch.randint(0, W - w, (1,), generator=self._rng).item())
        x = x.clone()
        x[:, top:top + h, left:left + w] = 0.0
        return x

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply augmentation to a PIL image. Returns a normalized (3, H, W) tensor.
    
        torchvision v2 stochastic transforms read from torch's *global* RNG and do
        not accept a per-instance generator. To make this class deterministic given
        its own self._rng (and avoid leaking state into the caller), we save the
        global RNG state, seed it from our RNG, run the transforms, then restore
        the previous state.
        """
        # Save global torch RNG state (the v2 transforms use this, not our _rng).
        global_state = torch.random.get_rng_state()
        try:
            # Draw a fresh 64-bit seed from our _rng and use it as the global seed.
            seed = int(torch.randint(0, 2**63 - 1, (1,), generator=self._rng).item())
            torch.manual_seed(seed)
            x = self.transform(img)  # (3, H, W), float32 in [0, 1]
        finally:
            torch.random.set_rng_state(global_state)
    
        # Normalize, noise, erasing: these use self._rng directly, not global state.
        mean = torch.tensor(self.cfg.mean, dtype=x.dtype).view(3, 1, 1)
        std = torch.tensor(self.cfg.std, dtype=x.dtype).view(3, 1, 1)
        x = (x - mean) / std
        x = self._apply_gaussian_noise(x)
        x = self._apply_erasing(x)
        return x


class SimCLRPairAugment:
    """Returns two independent views of the same image.

    The two views share the augmentation config but draw independent random
    samples. This is what makes the contrastive task non-trivial.
    """

    def __init__(self, cfg: SimCLRAugmentConfig | None = None, base_seed: int | None = None) -> None:
        self.cfg = cfg or SimCLRAugmentConfig()
        # Two independent RNG streams with independent seeds, so the two views
        # are truly uncorrelated.
        seed_a = None if base_seed is None else base_seed
        seed_b = None if base_seed is None else base_seed + 1
        self.view_a = SimCLRAugment(self.cfg, seed=seed_a)
        self.view_b = SimCLRAugment(self.cfg, seed=seed_b)

    def __call__(self, img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.view_a(img), self.view_b(img)


class SimCLRTwoViewDataset:
    """Wraps an existing dataset (that yields (img_tensor_or_pil, label)) and
    returns (view_a, view_b) pairs, discarding labels.

    The dataset __getitem__ returns (img, label). We assume img is a PIL Image
    (our BMDataset yields PIL because the transform is applied inside __getitem__).
    Actually, BMDataset applies the transform in __getitem__, so if we want raw
    PIL we need to instantiate BMDataset with transform=None. This wrapper
    enforces that by re-applying the transform.
    """

    def __init__(self, base_dataset, augment: SimCLRPairAugment) -> None:
        self.base = base_dataset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _label = self.base[idx]  # label discarded
        if isinstance(img, torch.Tensor):
            raise TypeError(
                "SimCLRTwoViewDataset expects a dataset yielding PIL images. "
                "Instantiate BMDataset with transform=None."
            )
        va, vb = self.augment(img)
        return va, vb
