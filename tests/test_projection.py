"""Projection head shape and NO-BatchNorm-on-output test."""
import torch
from attentionclr.clr.projection import SimCLRProjectionHead


def test_projection_shape():
    m = SimCLRProjectionHead(in_dim=2048, hidden_dim=2048, out_dim=128)
    y = m(torch.randn(4, 2048))
    assert y.shape == (4, 128)


def test_projection_has_no_norm_on_output():
    """SimCLR v1 specifically avoids BatchNorm on the projection output.
    Assert the final layer is a bare Linear."""
    m = SimCLRProjectionHead()
    # The last module in the Sequential should be nn.Linear (not BN, not ReLU).
    last = list(m.net.modules())[-1]
    assert isinstance(last, torch.nn.Linear), type(last)
