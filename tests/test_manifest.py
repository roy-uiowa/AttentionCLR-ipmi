"""Dataset integrity. Skipped if the raw dataset isn't present."""
from pathlib import Path

import pytest

from attentionclr.utils.io import load_yaml, read_manifest

DATA_ROOT = Path("data/bone_marrow_cell_dataset")
MANIFEST = Path("data/processed/manifest.parquet")
LABEL_MAP = Path("data/processed/label_map.yaml")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT.exists() and MANIFEST.exists()),
    reason="raw dataset or manifest not present; run `make data` first",
)


def test_manifest_has_expected_columns():
    df = read_manifest(MANIFEST)
    assert set(df.columns) >= {"image_id", "rel_path", "folder_code", "label", "file_size"}


def test_all_manifest_files_exist():
    df = read_manifest(MANIFEST)
    missing = [r for r in df["rel_path"] if not (DATA_ROOT / r).exists()]
    assert not missing, f"{len(missing)} manifest rows point to missing files (first 5: {missing[:5]})"


def test_no_duplicate_image_ids():
    df = read_manifest(MANIFEST)
    assert not df["image_id"].duplicated().any()


def test_labels_are_subset_of_label_map_values():
    df = read_manifest(MANIFEST)
    allowed = set(load_yaml(LABEL_MAP).values())
    assert set(df["label"]) <= allowed


def test_counts_match_table1():
    # Delegates to the same source-of-truth dict as scripts/verify_dataset.py.
    from scripts.verify_dataset import EXPECTED_COUNTS, EXPECTED_TOTAL
    df = read_manifest(MANIFEST)
    actual = df["label"].value_counts().to_dict()
    assert len(df) == EXPECTED_TOTAL
    assert actual == EXPECTED_COUNTS