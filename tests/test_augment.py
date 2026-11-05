"""Augmentation shape/dtype tests; no correctness assertions (visual check is separate)."""
import numpy as np

from attentionclr.data.augment import GeometricConfig, StainConfig, augment_image


def test_geometric_preserves_shape_and_dtype():
    img = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = augment_image(img, GeometricConfig(), StainConfig(), rng, apply_stain=False)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_stain_preserves_shape_and_dtype():
    # Uniform image: Macenko decomposition may fail (low variance); the
    # augmenter should return the input unchanged rather than crash.
    img = np.full((250, 250, 3), 200, dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = augment_image(img, GeometricConfig(), StainConfig(), rng, apply_stain=True)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_augment_deterministic_given_rng():
    img = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    a = augment_image(img, GeometricConfig(), StainConfig(), np.random.default_rng(42))
    b = augment_image(img, GeometricConfig(), StainConfig(), np.random.default_rng(42))
    assert np.array_equal(a, b)
