"""Shape and range tests for CBAM. These catch the 'CBAM silently returns a
different shape' class of bugs that would propagate to the encoder."""
import torch
import pytest

from attentionclr.clr.cbam import ChannelAttention, SpatialAttention, CBAM


@pytest.mark.parametrize("channels,H,W", [(64, 32, 32), (256, 63, 63), (2048, 8, 8)])
def test_channel_attention_shape_and_range(channels, H, W):
    m = ChannelAttention(channels).eval()
    x = torch.randn(2, channels, H, W)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, channels, 1, 1), y.shape
    assert torch.all(y >= 0) and torch.all(y <= 1), (y.min(), y.max())


def test_spatial_attention_shape_and_range():
    m = SpatialAttention().eval()
    x = torch.randn(2, 256, 63, 63)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 1, 63, 63), y.shape
    assert torch.all(y >= 0) and torch.all(y <= 1)


def test_cbam_preserves_shape():
    for channels, H, W in [(64, 32, 32), (256, 63, 63), (2048, 8, 8)]:
        m = CBAM(channels).eval()
        x = torch.randn(2, channels, H, W)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape, (channels, y.shape)


def test_cbam_is_multiplicative_gate():
    """CBAM output must be input * (mask in [0,1]), i.e. per-element amplitude
    is not increased, and masked-out channels/positions become 0."""
    m = CBAM(16).eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        y = m(x)
    # Element-wise, |y| <= |x| (allow small epsilon for numerical).
    assert torch.all(y.abs() <= x.abs() + 1e-5)


def test_cbam_reduction_never_zero():
    """For very small channels the reduction must not produce a 0-channel conv."""
    m = ChannelAttention(channels=8, reduction=16).eval()  # 8//16 = 0 -> clamp to 1
    x = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 8, 1, 1)
