"""Split protocol tests: stratification, no leakage, determinism."""
import numpy as np
import pandas as pd
import pytest

from attentionclr.data.splits import fivefold, holdout_70_30, holdout_80_20


@pytest.fixture
def toy_manifest() -> pd.DataFrame:
    # 10 classes x 20 images = 200 rows
    labels = np.repeat([f"c{i}" for i in range(10)], 20)
    return pd.DataFrame({"label": labels, "image_id": range(len(labels))})


def test_fivefold_no_overlap(toy_manifest):
    folds = fivefold(toy_manifest, seed=0, n_splits=5)
    assert len(folds) == 5
    for f in folds:
        assert len(set(f.train_idx) & set(f.val_idx)) == 0
        assert set(f.train_idx) | set(f.val_idx) == set(range(len(toy_manifest)))


def test_fivefold_val_fraction_approx_20pct(toy_manifest):
    folds = fivefold(toy_manifest, seed=0, n_splits=5)
    for f in folds:
        frac = len(f.val_idx) / len(toy_manifest)
        assert 0.15 <= frac <= 0.25, frac


def test_fivefold_stratified(toy_manifest):
    folds = fivefold(toy_manifest, seed=0, n_splits=5)
    for f in folds:
        val_labels = toy_manifest.iloc[f.val_idx]["label"]
        # each class should have ~4 images in each val fold
        counts = val_labels.value_counts()
        assert (counts >= 3).all() and (counts <= 5).all(), counts.to_dict()


def test_fivefold_deterministic(toy_manifest):
    a = fivefold(toy_manifest, seed=42)
    b = fivefold(toy_manifest, seed=42)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.train_idx, fb.train_idx)
        assert np.array_equal(fa.val_idx, fb.val_idx)


def test_holdout_80_20_fraction(toy_manifest):
    f = holdout_80_20(toy_manifest, seed=0)
    assert abs(len(f.train_idx) / len(toy_manifest) - 0.8) < 0.02


def test_holdout_70_30_fraction(toy_manifest):
    f = holdout_70_30(toy_manifest, seed=0)
    assert abs(len(f.train_idx) / len(toy_manifest) - 0.7) < 0.02