"""Tests for the SimCLR augmentation pipeline. These catch:
  - Shape mismatches between the two views.
  - The two views being accidentally identical (SimCLR would collapse).
  - Determinism given a seed (for reproducibility).
  - The augmentation destroying the image (too aggressive crop/blur).
"""
import torch
import numpy as np
import pytest
from PIL import Image

from attentionclr.clr.augment import (
    SimCLRAugment,
    SimCLRAugmentConfig,
    SimCLRPairAugment,
    SimCLRTwoViewDataset,
)


def _random_pil(h=250, w=250, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_single_view_shape():
    aug = SimCLRAugment(seed=0)
    x = aug(_random_pil())
    assert x.shape == (3, 250, 250)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_pair_shapes_and_dtype():
    aug = SimCLRPairAugment(base_seed=0)
    va, vb = aug(_random_pil())
    assert va.shape == (3, 250, 250)
    assert vb.shape == (3, 250, 250)
    assert va.dtype == vb.dtype == torch.float32


def test_two_views_are_not_identical():
    """The core SimCLR invariant: two views of the same image must differ.
    If they're identical, NT-Xent has zero gradients from this pair and the
    encoder learns nothing."""
    aug = SimCLRPairAugment(base_seed=0)
    va, vb = aug(_random_pil())
    assert not torch.allclose(va, vb), (
        "Two views are identical — check that the two SimCLRAugment instances "
        "have independent seeds and the transforms are stochastic."
    )


def test_pair_deterministic_with_seed():
    """Same base_seed + same input => same views. Required for reproducibility."""
    aug1 = SimCLRPairAugment(base_seed=42)
    aug2 = SimCLRPairAugment(base_seed=42)
    img = _random_pil(seed=7)
    va1, vb1 = aug1(img)
    va2, vb2 = aug2(img)
    assert torch.allclose(va1, va2)
    assert torch.allclose(vb1, vb2)


def test_different_base_seeds_differ():
    """Different base_seed => different views. Sanity check that the seed
    actually controls the RNG."""
    img = _random_pil(seed=7)
    va1, _ = SimCLRPairAugment(base_seed=0)(img)
    va2, _ = SimCLRPairAugment(base_seed=1)(img)
    assert not torch.allclose(va1, va2)


def test_views_have_non_trivial_content():
    """The augmentation should not collapse the image to a constant (e.g.,
    a very aggressive crop landing outside the cell, or extreme blur).
    Assert the per-view std is above a threshold."""
    aug = SimCLRPairAugment(base_seed=0)
    va, vb = aug(_random_pil())
    assert va.std().item() > 0.05, f"view A nearly constant: std={va.std().item():.4f}"
    assert vb.std().item() > 0.05, f"view B nearly constant: std={vb.std().item():.4f}"


def test_no_grayscale():
    """The augmentation must preserve color (eosin vs. basophil distinction).
    Assert that the two views, on a colored input, differ in a way that isn't
    just luminance — i.e., the color channels are not all equal."""
    # Create a strongly colored image: red half, blue half.
    arr = np.zeros((250, 250, 3), dtype=np.uint8)
    arr[:125, :, 0] = 200  # top half red
    arr[125:, :, 2] = 200  # bottom half blue
    img = Image.fromarray(arr, mode="RGB")
    aug = SimCLRPairAugment(base_seed=0)
    va, vb = aug(img)
    # After color jitter, channels are not equal — assert the per-view
    # R and B channels are not equal (i.e., not grayscale).
    assert not torch.allclose(va[0], va[2], atol=0.1), "view A looks grayscale"
    assert not torch.allclose(vb[0], vb[2], atol=0.1), "view B looks grayscale"


def test_pair_works_on_real_cell_patch():
    """Integration test: load one cell from the manifest, run the augmenter,
    verify shapes and finiteness."""
    pytest.importorskip("pyarrow")
    from pathlib import Path
    from attentionclr.utils.io import read_manifest

    manifest_path = Path("data/processed/manifest.parquet")
    if not manifest_path.exists():
        pytest.skip("manifest not present; run `make data` first")
    manifest = read_manifest(manifest_path).head(1)
    img_path = Path("data/bone_marrow_cell_dataset") / manifest.iloc[0]["rel_path"]
    if not img_path.exists():
        pytest.skip("dataset not present")
    img = Image.open(img_path).convert("RGB")
    aug = SimCLRPairAugment(base_seed=0)
    va, vb = aug(img)
    assert va.shape == (3, 250, 250)
    assert vb.shape == (3, 250, 250)
    assert torch.isfinite(va).all() and torch.isfinite(vb).all()
    assert not torch.allclose(va, vb)
