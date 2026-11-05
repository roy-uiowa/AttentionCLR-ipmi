"""Split protocols from Matek 2021.

  - fivefold:     stratified 5-fold CV, ~20% per class per fold (Methods)
  - holdout_80_20: single stratified 80/20 split (Methods)
  - holdout_70_30: single stratified 70/30 split (robustness vs Krappe, Supp S6)

All splitters take a manifest DataFrame and return index arrays (into the
manifest, not into the filesystem). Deterministic given (manifest, seed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class Fold:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def _stratified_kfold_indices(
    labels: pd.Series, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros(len(labels))  # placeholder; sklearn only needs the labels
    return [(tr, va) for tr, va in skf.split(X, labels.to_numpy())]


def fivefold(df: pd.DataFrame, seed: int = 1337, n_splits: int = 5) -> list[Fold]:
    return [
        Fold(fold=i, train_idx=tr, val_idx=va)
        for i, (tr, va) in enumerate(
            _stratified_kfold_indices(df["label"], n_splits, seed)
        )
    ]


def holdout(
    df: pd.DataFrame, train_frac: float, seed: int = 1337
) -> Fold:
    """Stratified single split. train_frac is the fraction used for training.

    Note: the paper calls these "80/20" and "70/30". The *test* fraction is
    1 - train_frac. We use train_frac explicitly so there is no ambiguity.
    """
    idx = np.arange(len(df))
    tr, va = train_test_split(
        idx,
        train_size=train_frac,
        stratify=df["label"].to_numpy(),
        random_state=seed,
    )
    return Fold(fold=0, train_idx=np.sort(tr), val_idx=np.sort(va))


# Convenience wrappers matching the paper's naming.
def holdout_80_20(df: pd.DataFrame, seed: int = 1337) -> Fold:
    return holdout(df, train_frac=0.8, seed=seed)


def holdout_70_30(df: pd.DataFrame, seed: int = 1337) -> Fold:
    return holdout(df, train_frac=0.7, seed=seed)