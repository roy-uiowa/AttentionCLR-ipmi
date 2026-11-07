"""Strict and tolerant precision/recall, per Matek 2021 Table 1.

Strict: prediction must equal true label.
Tolerant: prediction of a tolerance-partner class counts as correct (symmetric).

Definitions (inferred from Fig 3A; to be validated against one published number):
  support[i]       = cm[i, :].sum()                             # true-class count
  pred_pos[i]      = cm[:, i].sum()                             # all predictions of i
  tp_tol[i]        = sum_{j: T[i,j]} cm[j, i]                   # TP: true j tolerant with i
  fp_tol[i]        = pred_pos[i] - tp_tol[i]                    # FP: true j NOT tolerant
  precision_tol[i] = tp_tol[i] / pred_pos[i]
  recall_tol[i]    = tp_tol[i] / support[i]
Strict is the special case T = identity.
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


def _build_tolerance_matrix(class_names: list[str], pairs: list[tuple[str, str]]) -> np.ndarray:
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
    """Strict and tolerant precision/recall from a confusion matrix.

    Tolerant recall of class i counts true-i images predicted as i OR as any
    tolerance partner of i  (row sum over tolerance columns).
    Tolerant precision of class i counts predictions of i whose true label is
    i OR a tolerance partner of i  (column sum over tolerance rows).
    Both are bounded by 1.0 by construction.
    """
    C = len(class_names)
    T = _build_tolerance_matrix(class_names, tolerance_pairs or [])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))

    per_class: dict[str, ClassMetrics] = {}
    for i, name in enumerate(class_names):
        support = int(cm[i].sum())
        pred_pos = int(cm[:, i].sum())

        # precision numerator: column i, tolerance rows
        tp_for_prec = int(sum(cm[j, i] for j in range(C) if T[i, j]))
        # recall numerator: row i, tolerance columns
        tp_for_rec = int(sum(cm[i, k] for k in range(C) if T[i, k]))

        prec = tp_for_prec / pred_pos if pred_pos > 0 else 0.0
        rec = tp_for_rec / support if support > 0 else 0.0
        per_class[name] = ClassMetrics(precision=prec, recall=rec, support=support)

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