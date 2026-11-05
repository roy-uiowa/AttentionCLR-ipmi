"""Build and read the dataset manifest.

The manifest is the single source of truth downstream: one row per image, with
canonical class name, split-ready fields, and a stable path relative to data_root.
The raw dataset tree is never written to.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from attentionclr.utils.io import load_yaml, write_manifest
from attentionclr.utils.logging import get_logger

log = get_logger(__name__)

# Image extensions we accept. Anything else is flagged, not silently skipped.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestConfig:
    data_root: Path
    label_map_path: Path
    out_path: Path


def _iter_images(class_dir: Path):
    """Yield (path,) for every image under class_dir, at any depth.

    Handles both flat classes (ABE/ABE_00001.jpg) and sharded classes
    (ART/0001-1000/ART_00001.jpg) without assuming a particular layout.
    """
    for p in sorted(class_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _sha1_of_paths(paths: list[Path], root: Path) -> str:
    """Content hash: relative paths + file sizes + mtimes.

    We deliberately do NOT hash image bytes (slow, and the raw dataset is
    immutable). This is a structural fingerprint: catches added/removed/renamed
    files and resized-on-disk files. Good enough for "did the dataset change?".
    """
    h = hashlib.sha1()
    for p in sorted(paths):
        rel = str(p.relative_to(root))
        st = p.stat()
        h.update(f"{rel}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def build_manifest(cfg: ManifestConfig) -> pd.DataFrame:
    """Scan data_root, apply label_map, return a manifest DataFrame.

    Columns:
        image_id     : stable string (code + zero-padded index within class)
        rel_path     : path relative to data_root (portable)
        folder_code  : e.g. "NGS"
        label        : canonical Table 1 class name (e.g. "Segmented neutrophils")
        file_size    : bytes
    """
    label_map: dict[str, str] = load_yaml(cfg.label_map_path)
    if not label_map:
        raise ValueError(f"Empty label map: {cfg.label_map_path}")

    data_root = cfg.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    rows: list[dict[str, Any]] = []
    unknown_dirs: list[str] = []

    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        code = class_dir.name
        if code not in label_map:
            unknown_dirs.append(code)
            continue
        label = label_map[code]
        images = list(_iter_images(class_dir))
        # Stable, class-local ordinal. Independent of shard layout.
        for i, p in enumerate(images, start=1):
            rows.append(
                {
                    "image_id": f"{code}_{i:05d}",
                    "rel_path": str(p.relative_to(data_root)),
                    "folder_code": code,
                    "label": label,
                    "file_size": p.stat().st_size,
                }
            )
        log.info("%-4s %-28s %6d images", code, label, len(images))

    if unknown_dirs:
        raise ValueError(
            f"Folders not in label_map: {unknown_dirs}. "
            f"Add them to {cfg.label_map_path} or remove them."
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under {data_root}")

    # Duplicate detection: same rel_path twice would mean a bug; same image_id
    # twice would break downstream joins.
    if df["rel_path"].duplicated().any():
        dupes = df.loc[df["rel_path"].duplicated(), "rel_path"].tolist()[:5]
        raise ValueError(f"Duplicate rel_path entries (first 5): {dupes}")
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values — bug in ordinal assignment")

    df = df.sort_values(["folder_code", "image_id"]).reset_index(drop=True)
    df.attrs["dataset_hash"] = _sha1_of_paths(
        [data_root / r for r in df["rel_path"]], data_root
    )
    df.attrs["data_root"] = str(data_root)
    return df


def write_manifest_and_hash(df: pd.DataFrame, out_path: Path) -> Path:
    write_manifest(df, out_path)
    hash_path = out_path.parent / "dataset_hash.txt"
    hash_path.write_text(df.attrs["dataset_hash"] + "\n", encoding="utf-8")
    log.info("Wrote manifest: %s", out_path)
    log.info("Dataset hash : %s", df.attrs["dataset_hash"])
    return hash_path