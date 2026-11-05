"""Matek et al. 2021, Table 1, "Images*" column. Do not edit without re-checking the paper.

Two corrupt files were removed from the local copy (see known_corrupt.txt);
verify_dataset.py subtracts those from EXPECTED_COUNTS when checking.
"""
from __future__ import annotations

EXPECTED_COUNTS: dict[str, int] = {
    "Abnormal eosinophils": 8,
    "Artefacts": 19630,
    "Basophils": 441,
    "Blasts": 11973,
    "Erythroblasts": 27395,
    "Eosinophils": 5883,
    "Faggot cells": 47,
    "Hairy cells": 409,
    "Smudge cells": 42,
    "Immature lymphocytes": 65,
    "Lymphocytes": 26242,
    "Metamyelocytes": 3055,
    "Monocytes": 4040,
    "Myelocytes": 6557,
    "Band neutrophils": 9968,
    "Segmented neutrophils": 29424,
    "Not identifiable": 3538,
    "Other cells": 294,
    "Proerythroblasts": 2740,
    "Plasma cells": 7629,
    "Promyelocytes": 11994,
}

EXPECTED_TOTAL: int = 171_374

# Canonical class order, sorted alphabetically by Table 1 name. Used to assign
# stable integer indices. Do NOT reorder; checkpoints embed this order.
CLASS_NAMES: list[str] = sorted(EXPECTED_COUNTS.keys())
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: name for name, i in CLASS_TO_IDX.items()}
NUM_CLASSES: int = len(CLASS_NAMES)

assert NUM_CLASSES == 21, NUM_CLASSES
