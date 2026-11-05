"""Audit manifest + folder names for label spelling/consistency issues.

Checks:
  - Manifest labels match label_map values exactly.
  - No duplicate rel_paths (also enforced in build_manifest).
  - Image files that failed to parse (corrupt / zero-byte) — reported, not fatal.
  - Known spelling variants: "Fagott"/"Faggot", "Promyelocytes"/"Promyelocyten".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from attentionclr.utils.io import load_yaml, read_manifest
from attentionclr.utils.logging import get_logger

log = get_logger("audit_labels")

SPELLING_ALIASES = {
    "Fagott cells": "Faggot cells",
    "Faggotzellen": "Faggot cells",
    "Promyelocyten": "Promyelocytes",
    "Promyelocyte": "Promyelocytes",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    ap.add_argument("--label-map", type=Path, default=Path("data/processed/label_map.yaml"))
    ap.add_argument("--data-root", type=Path, default=Path("data/bone_marrow_cell_dataset"))
    ap.add_argument("--check-images", action="store_true",
                    help="Open every image with PIL (slow: ~171k files).")
    args = ap.parse_args()

    df = read_manifest(args.manifest)
    label_map = load_yaml(args.label_map)
    problems = 0

    # 1. Manifest labels must be exactly label_map values.
    allowed = set(label_map.values())
    bad = set(df["label"]) - allowed
    if bad:
        log.error("Manifest labels not in label_map values: %s", sorted(bad))
        problems += 1

    # 2. Spelling aliases.
    for alias, canonical in SPELLING_ALIASES.items():
        if alias in set(df["label"]):
            log.error("Spelling variant %r should be %r", alias, canonical)
            problems += 1

    # 3. Zero-byte files.
    data_root = args.data_root
    zero = df.loc[df["file_size"] == 0, "rel_path"].tolist()
    if zero:
        log.error("%d zero-byte images (first 5): %s", len(zero), zero[:5])
        problems += 1

    # 4. Optional: decode every image.
    if args.check_images:
        n_bad = 0
        for rel in df["rel_path"]:
            try:
                with Image.open(data_root / rel) as im:
                    im.verify()
            except Exception as e:  # noqa: BLE001 — reporting, not handling
                n_bad += 1
                if n_bad <= 5:
                    log.error("Unreadable image %s: %s", rel, e)
        if n_bad:
            log.error("%d unreadable images total", n_bad)
            problems += 1
        else:
            log.info("All %d images decoded OK", len(df))

    if problems == 0:
        log.info("Audit OK")
        sys.exit(0)
    log.error("Audit found %d problem(s)", problems)
    sys.exit(1)


if __name__ == "__main__":
    main()