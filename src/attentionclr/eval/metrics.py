"""Strict and tolerant precision/recall, per Matek 2021 Table 1.

Strict: prediction must equal true label.
Tolerant: prediction of a paired class counts as correct (symmetric).

Precision_tolerant is defined analogously: for class C, TP is the count of
predictions of C on images whose true label is C or a tolerance partner of C.
Documented in reports/assumptions.md; matches the paper's "tolerance classes"
concept but the exact formula is inferred from Fig 3A.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    support: int


@dataclass
class MetricsResult:
    per_class: dict[str, ClassMetrics]
    macro_precision: float
    macro_recall: float
    per_class_support: dict[str, int]


def _build_tolerance_matrix(
    class_names: list[str], pairs: list[tuple[str, str]]
) -> np.ndarray:
    """Return a (C, C) boolean matrix: T[i, j] = True iff i == j or (i, j) is tolerant."""
    C = len(class_names)
    idx = {c: i for i, c in enumerate(class_names)}
    T = np.eye(C, dtype=bool)
    for a, b in pairs:
        if a not in idx or b not in idx:
            raise ValueError(f"Tolerance pair references unknown class: ({a}, {b})")
        T[idx[a], idx[b]] = True
        T[idx[b], idx[a]] = True
    return T


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    tolerance_pairs: list[tuple[str, str]] | None = None,
) -> MetricsResult:
    """Compute strict + tolerant precision/recall. Both returned in one result,
    with the strict values under _strict and tolerant under _tolerant naming
    (this module returns tolerant by default; see strict_metrics wrapper)."""
    C = len(class_names)
    T = _build_tolerance_matrix(class_names, tolerance_pairs or [])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    per_class: dict[str, ClassMetrics] = {}
    for i, name in enumerate(class_names):
        support = cm[i].sum()
        tp = cm[i, i]
        pred_pos = cm[:, i].sum()

        # Strict
        prec_s = tp / pred_pos if pred_pos > 0 else 0.0
        rec_s = tp / support if support > 0 else 0.0

        # Tolerant: consider all (true, pred) pairs allowed by T
        tp_t = 0
        for true_c in range(C):
            if not T[i, true_c]:
                continue
            for pred_c in range(C):
                if T[i, pred_c] or pred_c == i:
                    tp_t += cm[true_c, pred_c] if pred_c == i else 0
        # Count predictions of class i that are "correct" (true is i or tolerant)
        tp_t = sum(
            cm[j, i]
            for j in range(C)
            if T[i, j]  # true j is tolerant-compatible with predicted i
        )
        # Tolerant precision: TP_t over all predictions of i whose true is in tolerance(i)
        pred_pos_t = sum(cm[j, i] for j in range(C) if T[i, j])
        prec_t = tp_t / pred_pos_t if pred_pos_t > 0 else 0.0
        # Tolerant recall: TP_t over support
        rec_t = tp_t / support if support > 0 else 0.0

        per_class[name] = ClassMetrics(
            precision=prec_t, recall=rec_t, support=int(support)
        )

    return MetricsResult(
        per_class=per_class,
        macro_precision=float(np.mean([m.precision for m in per_class.values()])),
        macro_recall=float(np.mean([m.recall for m in per_class.values()])),
        per_class_support={k: v.support for k, v in per_class.items()},
    )


def strict_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> MetricsResult:
    """Strict = tolerant with an empty pair list (only diagonal is True)."""
    return compute_metrics(y_true, y_pred, class_names, tolerance_pairs=[])
