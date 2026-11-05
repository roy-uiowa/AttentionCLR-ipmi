"""Build data/processed/manifest.parquet + dataset_hash.txt from the raw tree."""
from __future__ import annotations

import argparse
from pathlib import Path

from attentionclr.data.manifest import (
    ManifestConfig,
    build_manifest,
    write_manifest_and_hash,
)
from attentionclr.utils.logging import get_logger

log = get_logger("build_manifest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/bone_marrow_cell_dataset"))
    ap.add_argument("--label-map", type=Path, default=Path("data/processed/label_map.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/manifest.parquet"))
    args = ap.parse_args()

    cfg = ManifestConfig(
        data_root=args.root,
        label_map_path=args.label_map,
        out_path=args.out,
    )
    df = build_manifest(cfg)
    write_manifest_and_hash(df, cfg.out_path)

    log.info("Total images: %d across %d classes", len(df), df["label"].nunique())
    per_class = df["label"].value_counts().sort_index()
    log.info("Per-class counts:\n%s", per_class.to_string())


if __name__ == "__main__":
    main()