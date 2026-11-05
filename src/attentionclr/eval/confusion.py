"""Row-normalized confusion matrix, paper-style (Figure 2)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Network prediction")
    ax.set_ylabel("Examiner label")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Relative frequency")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Also write raw + normalized counts as CSV for downstream analysis.
    np.savetxt(path.with_suffix(".csv"), cm_norm, delimiter=",", fmt="%.6f")
