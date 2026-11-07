"""Metrics regression tests: strict and tolerant precision/recall.

These tests exist because the tolerant-recall formula had a bug (summing the
wrong axis of the confusion matrix, producing recall > 1.0) that went undetected
until 5-fold evaluation. The property test is the guard against that class of bug.
"""
import numpy as np
import pytest

from attentionclr.eval.metrics import compute_metrics, strict_metrics


def test_strict_recall_on_identity_predictions():
    """All correct predictions => strict P = strict R = 1.0 for all classes."""
    y = np.array([0, 0, 1, 1, 2, 2])
    class_names = ["A", "B", "C"]
    m = strict_metrics(y, y, class_names)
    for cls in class_names:
        assert m.per_class[cls].precision == 1.0
        assert m.per_class[cls].recall == 1.0
        assert m.per_class[cls].support == 2

def test_strict_recall_on_all_wrong_v2():
    """Cyclic misprediction: every true image is predicted as the next class.
    With 3 classes, this gives recall 0 for every class."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 1, 2, 2, 0, 0])   # A->B, B->C, C->A
    class_names = ["A", "B", "C"]
    m = strict_metrics(y_true, y_pred, class_names)
    for cls in class_names:
        assert m.per_class[cls].recall == 0.0
        assert m.per_class[cls].precision == 0.0


def test_tolerant_recall_bounded_by_one():
    """Hand-constructed case: A misclassified as B, B misclassified as A.
    With tolerance pair (A, B), tolerant recall for A and B should both be 1.0.
    Strict recall for A and B should both be 0.0."""
    class_names = ["A", "B", "C"]
    # 10 A's predicted as B; 10 B's predicted as A; 10 C's predicted as C
    y_true = np.array([0] * 10 + [1] * 10 + [2] * 10)
    y_pred = np.array([1] * 10 + [0] * 10 + [2] * 10)

    strict = strict_metrics(y_true, y_pred, class_names)
    assert strict.per_class["A"].recall == 0.0
    assert strict.per_class["B"].recall == 0.0
    assert strict.per_class["C"].recall == 1.0

    tolerant = compute_metrics(y_true, y_pred, class_names, tolerance_pairs=[("A", "B")])
    assert tolerant.per_class["A"].recall == 1.0
    assert tolerant.per_class["B"].recall == 1.0
    assert tolerant.per_class["C"].recall == 1.0


def test_tolerant_recall_never_exceeds_one():
    """Property test: for random y_true/y_pred and random tolerance pairs,
    every precision and recall must be in [0, 1]. This is the direct guard
    against the row/column-sum bug."""
    rng = np.random.default_rng(0)
    C = 6
    class_names = [f"c{i}" for i in range(C)]
    for trial in range(50):
        y_true = rng.integers(0, C, size=500)
        y_pred = rng.integers(0, C, size=500)
        pairs = [("c0", "c1"), ("c2", "c3"), ("c4", "c5")]
        tol = compute_metrics(y_true, y_pred, class_names, tolerance_pairs=pairs)
        for cls, m in tol.per_class.items():
            assert 0.0 <= m.recall <= 1.0, f"trial {trial} {cls} recall={m.recall}"
            assert 0.0 <= m.precision <= 1.0, f"trial {trial} {cls} precision={m.precision}"


def test_strict_tolerant_identical_with_empty_pairs():
    """With no tolerance pairs, tolerant == strict (T is the identity)."""
    rng = np.random.default_rng(1)
    C = 5
    class_names = [f"c{i}" for i in range(C)]
    y_true = rng.integers(0, C, size=200)
    y_pred = rng.integers(0, C, size=200)
    strict = strict_metrics(y_true, y_pred, class_names)
    tolerant = compute_metrics(y_true, y_pred, class_names, tolerance_pairs=[])
    for cls in class_names:
        assert strict.per_class[cls].precision == pytest.approx(tolerant.per_class[cls].precision)
        assert strict.per_class[cls].recall == pytest.approx(tolerant.per_class[cls].recall)


def test_tolerant_precision_lifted_by_partner_predictions():
    """For the paper's canonical band<->segmented case: true segmented neutrophils
    predicted as band neutrophils should count as TP for band neutrophils under
    the tolerance pair (band, segmented)."""
    class_names = ["band", "seg", "other"]
    # 5 true band predicted as seg; 5 true seg predicted as band; 5 true other correct.
    y_true = np.array([0] * 5 + [1] * 5 + [2] * 5)
    y_pred = np.array([1] * 5 + [0] * 5 + [2] * 5)

    strict = strict_metrics(y_true, y_pred, class_names)
    assert strict.per_class["band"].recall == 0.0
    assert strict.per_class["band"].precision == 0.0

    tolerant = compute_metrics(y_true, y_pred, class_names, tolerance_pairs=[("band", "seg")])
    # Tolerance pair lifts both precision and recall to 1.0 for band and seg:
    # - band precision: 5 true-seg predicted as band are now TP (tolerated)
    # - band recall: 5 true-band predicted as seg are now TP (tolerated)
    assert tolerant.per_class["band"].recall == 1.0
    assert tolerant.per_class["band"].precision == 1.0
    assert tolerant.per_class["seg"].recall == 1.0
    assert tolerant.per_class["seg"].precision == 1.0