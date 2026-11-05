"""Train/val transform builders. Val transform is resize+normalize only."""
from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms

from attentionclr.data.augment import (
    GeometricConfig,
    StainConfig,
    augment_image,
)


def val_transform(size: int = 250) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class TrainTransform:
    """Callable: PIL -> augmented tensor.

    Applies geometric + Macenko stain augmentation in numpy/PIL space, then
    ToTensor + Normalize. Deterministic given the numpy Generator.

    Note: because the augmentation operates on the full image, we resize to the
    target size *before* augmenting, so shifts/shears/rotations operate in the
    final coordinate frame (matches the paper's 250x250 training patches).
    """

    def __init__(
        self,
        size: int = 250,
        geo_cfg: GeometricConfig | None = None,
        stain_cfg: StainConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.geo_cfg = geo_cfg or GeometricConfig()
        self.stain_cfg = stain_cfg or StainConfig()
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def _get_rng(self) -> np.random.Generator:
        # Lazy per-worker init so DataLoader workers each get their own stream.
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    def __call__(self, img: Image.Image) -> "torch.Tensor":  # noqa: F821
        import torch

        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        arr = augment_image(arr, self.geo_cfg, self.stain_cfg, self._get_rng(), apply_stain=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        return t
