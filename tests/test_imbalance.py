import numpy as np
from attentionclr.data.imbalance import ImbalanceConfig, make_sampler


def test_oversample_mode_weights_minority_up():
    labels = np.array([0] * 1000 + [1] * 10)  # class 1 is tiny
    sampler = make_sampler(labels, ImbalanceConfig(mode="on_the_fly_oversample_to", target_per_class=500))
    assert sampler is not None
    # The sampler's weights should be higher for class 1.
    w = np.array(sampler.weights)
    assert w[labels == 1].mean() > w[labels == 0].mean()


def test_cap_mode_weights_majority_down():
    labels = np.array([0] * 1000 + [1] * 10)
    sampler = make_sampler(labels, ImbalanceConfig(mode="cap_downsample", target_per_class=500))
    assert sampler is not None


def test_none_mode_returns_none():
    labels = np.array([0, 1])
    assert make_sampler(labels, ImbalanceConfig(mode="none", target_per_class=1)) is None
