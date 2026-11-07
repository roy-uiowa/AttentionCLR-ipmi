"""Shape and param-count tests for the ResNeXt50+CBAM encoder."""
import torch
import pytest

from attentionclr.clr.encoder import ResNeXt50CBAMEncoder, count_trainable_params


def test_encoder_forward_shape():
    m = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 250, 250))
    assert out.shape == (2, 2048), out.shape


def test_encoder_forward_features_matches_forward():
    m = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False).eval()
    x = torch.randn(2, 3, 250, 250)
    with torch.no_grad():
        a = m(x)
        b = m.forward_features(x)
    assert torch.allclose(a, b)


def test_encoder_stage_shapes():
    """Manually walk the stages to pin the intermediate spatial sizes.

    ResNeXt-50 stem: conv 7x7 stride 2 (250->125) + maxpool 3x3 stride 2 (125->63).
    layer1 keeps 63x63; layer2/3/4 each halve the spatial size.
    """
    m = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False).eval()
    x = torch.zeros(1, 3, 250, 250)
    with torch.no_grad():
        s = m.stem(x)
        l1 = m.layer1(s)
        l2 = m.layer2(l1)
        l3 = m.layer3(l2)
        l4 = m.layer4(l3)
        p = m.avgpool(l4)
    assert s.shape  == (1,   64,  63,  63), s.shape
    assert l1.shape == (1,  256,  63,  63), l1.shape
    assert l2.shape == (1,  512,  32,  32), l2.shape
    assert l3.shape == (1, 1024,  16,  16), l3.shape
    assert l4.shape == (1, 2048,   8,   8), l4.shape
    assert p.shape  == (1, 2048,   1,   1), p.shape


def test_encoder_param_count():
    """Pin the exact encoder param count.

    torchvision resnext50_32x4d with default 1000-head fc: 25,028,904
    - remove fc (encoder is trunk-only):                 -2,049,000
    = trunk:                                              22,979,904
    + CBAM (4 stages, reduction 16):                       +696,712
    = 23,676,616
    """
    m = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False)
    n = count_trainable_params(m)
    print(f"\nResNeXt50+CBAM encoder params: {n:,}")
    assert n == 23_676_616, (
        f"Encoder param count changed: got {n:,}, expected 23,676,616. "
        f"If this is intentional, update the test AND reports/assumptions.md."
    )


def test_encoder_gradient_flows():
    m = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False)
    x = torch.randn(1, 3, 250, 250)
    out = m(x)
    out.sum().backward()
    assert m.stem[0].weight.grad is not None
    cbam_layer4_conv = m.layer4[1].spatial_attn.conv
    assert cbam_layer4_conv.weight.grad is not None