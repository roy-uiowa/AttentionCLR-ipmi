"""Shape and range tests for SmoothGrad, Grad-CAM, UMAP."""
import numpy as np
import pytest
import torch

from attentionclr.explain.gradcam import gradcam, resnext50_target_layer
from attentionclr.explain.smoothgrad import smoothgrad
from attentionclr.models.resnext50 import ResNeXt50Classifier


@pytest.fixture(scope="module")
def model():
    m = ResNeXt50Classifier(num_classes=21, pretrained=False).eval().cpu()
    return m


@pytest.fixture
def x_batch():
    torch.manual_seed(0)
    return torch.randn(2, 3, 250, 250)


def test_smoothgrad_shape_and_range(model, x_batch):
    sal, target = smoothgrad(model, x_batch, n_samples=4, noise_std=0.1, seed=0)
    assert sal.shape == (2, 250, 250)
    assert torch.all(sal >= 0)
    assert torch.all(sal <= 1)
    assert target.shape == (2,)


def test_smoothgrad_deterministic_with_seed(model, x_batch):
    sal_a, _ = smoothgrad(model, x_batch, n_samples=4, seed=42)
    sal_b, _ = smoothgrad(model, x_batch, n_samples=4, seed=42)
    assert torch.allclose(sal_a, sal_b)


def test_gradcam_shape_and_range(model, x_batch):
    layer = resnext50_target_layer(model)
    cam, target = gradcam(model, layer, x_batch)
    assert cam.shape == (2, 250, 250)
    assert torch.all(cam >= 0)
    assert torch.all(cam <= 1)
    assert target.shape == (2,)


def test_umap_2d_runs():
    pytest.importorskip("umap")
    from attentionclr.explain.umap_features import umap_2d
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((200, 64)).astype(np.float32)
    emb = umap_2d(feats, n_neighbors=10, min_dist=0.1)
    assert emb.shape == (200, 2)
