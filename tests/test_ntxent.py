"""NT-Xent correctness tests.

If any of these fail, do not train. The loss is either collapsing to 0
(no gradients) or computing the wrong contrast pairs, and 24h of pretraining
will produce an unusable encoder.
"""
import math
import torch
import pytest

from attentionclr.clr.losses import NTXentLoss


def test_loss_identity_views_is_low():
    """If z_a == z_b (perfect positive agreement), loss should be near zero."""
    torch.manual_seed(0)
    loss_fn = NTXentLoss(temperature=0.1)
    z_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_b = z_a.clone()
    loss = loss_fn(z_a, z_b).item()
    assert loss < 0.5, f"identity-view loss too high: {loss:.4f}"


def test_loss_orthonormal_embeddings_equals_log_2n_minus_1():
    """With perfectly orthonormal embeddings, off-diagonal similarities are
    exactly 0, so the softmax is uniform over (2N-1) classes, and the loss is
    exactly log(2N-1). This is the *exact* reference value — no tolerance
    beyond fp noise.

    This is the strongest indexing/masking test: if the pair indexing is wrong
    or the diagonal isn't masked, the loss differs from log(2N-1) by >>1e-3.
    """
    torch.manual_seed(0)
    N, D = 8, 64   # D >= 2N
    loss_fn = NTXentLoss(temperature=0.1)
    Q, _ = torch.linalg.qr(torch.randn(D, D))   # orthonormal D x D
    z_a = Q[:N, :]      # (N, D)
    z_b = Q[N:2*N, :]   # (N, D), orthonormal to z_a and to each other
    loss = loss_fn(z_a, z_b).item()
    expected = math.log(2 * N - 1)
    assert abs(loss - expected) < 1e-3, (
        f"orthonormal loss {loss:.6f} != log(2N-1)={expected:.6f}; "
        f"the contrast pair indexing or the diagonal mask is wrong."
    )


def test_loss_random_init_in_plausible_range():
    """Random Gaussian embeddings have spread similarities, so the loss is
    above the orthonormal log(2N-1) baseline (peakier softmax at tau=0.1).
    Plausible range: [log(2N-1), log(2N-1) + 2].
    """
    torch.manual_seed(0)
    N, D = 8, 64
    loss_fn = NTXentLoss(temperature=0.1)
    loss = loss_fn(torch.randn(N, D), torch.randn(N, D)).item()
    lo = math.log(2 * N - 1)
    hi = lo + 2.0
    assert lo <= loss <= hi, f"random-init loss {loss:.4f} outside [{lo:.3f}, {hi:.3f}]"


def test_loss_symmetric_in_inputs():
    torch.manual_seed(1)
    loss_fn = NTXentLoss(temperature=0.1)
    z_a = torch.randn(8, 32)
    z_b = torch.randn(8, 32)
    assert torch.allclose(loss_fn(z_a, z_b), loss_fn(z_b, z_a))


def test_loss_changes_when_views_change():
    torch.manual_seed(2)
    loss_fn = NTXentLoss(temperature=0.1)
    z_a = torch.randn(8, 32)
    z_b = torch.randn(8, 32)
    z_b_perturbed = z_b + 0.5 * torch.randn_like(z_b)
    assert abs(loss_fn(z_a, z_b).item() - loss_fn(z_a, z_b_perturbed).item()) > 1e-3


def test_loss_backward_populates_gradients():
    import torch.nn as nn
    torch.manual_seed(3)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 64))
    loss_fn = NTXentLoss(temperature=0.1)
    z_a = encoder(torch.randn(4, 3, 8, 8))
    z_b = encoder(torch.randn(4, 3, 8, 8))
    loss = loss_fn(z_a, z_b)
    loss.backward()
    assert encoder[-1].weight.grad is not None
    assert torch.isfinite(encoder[-1].weight.grad).all()


def test_loss_shape_mismatch_raises():
    loss_fn = NTXentLoss()
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(torch.randn(4, 16), torch.randn(5, 16))


def test_loss_n_equals_one_raises():
    loss_fn = NTXentLoss()
    with pytest.raises(ValueError, match="N >= 2"):
        loss_fn(torch.randn(1, 16), torch.randn(1, 16))