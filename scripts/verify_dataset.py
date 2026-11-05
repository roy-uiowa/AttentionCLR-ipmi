"""Assert that data/processed/manifest.parquet matches Matek 2021 Table 1,
respecting known_corrupt.txt (removed files)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from attentionclr.data.table1 import EXPECTED_COUNTS, EXPECTED_TOTAL
from attentionclr.utils.io import read_manifest
from attentionclr.utils.logging import get_logger

log = get_logger("verify_dataset")


def _load_known_corrupt(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def verify(manifest_path: Path, known_corrupt_path: Path) -> int:
    if not manifest_path.exists():
        log.error(
            "Manifest not found: %s\n"
            "Run `make data` first (it builds the manifest and then verifies it).",
            manifest_path,
        )
        return 2

    df = read_manifest(manifest_path)
    actual = df["label"].value_counts().to_dict()

    known = _load_known_corrupt(known_corrupt_path)
    # Subtract known-corrupt rows from expected counts, per label.
    expected = dict(EXPECTED_COUNTS)
    if known:
        missing_in_manifest = known - set(df["rel_path"])
        if missing_in_manifest:
            log.warning(
                "%d known_corrupt entries not found in manifest (fine if already removed): %s",
                len(missing_in_manifest), sorted(missing_in_manifest)[:5],
            )
        # For each known-corrupt path, infer its class from the folder code
        # by matching against the manifest's code->label map. Since the file is
        # absent from the manifest, we infer from path structure instead.
        for rel in known:
            code = rel.split("/")[0]
            # Load label map lazily to avoid a circular import
            from attentionclr.utils.io import load_yaml
            label_map = load_yaml(Path("data/processed/label_map.yaml"))
            label = label_map.get(code)
            if label is None:
                log.warning("known_corrupt entry %s has unknown folder code %s", rel, code)
                continue
            expected[label] = expected.get(label, 0) - 1

    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    mismatches = {k: (expected[k], actual.get(k)) for k in expected if actual.get(k) != expected[k]}
    expected_total = sum(expected.values())
    ok = not (missing or extra or mismatches) and len(df) == expected_total

    if missing:
        log.error("Missing classes in manifest: %s", sorted(missing))
    if extra:
        log.error("Unexpected classes in manifest: %s", sorted(extra))
    if mismatches:
        log.error("Count mismatches (expected, actual):")
        for k, (e, a) in sorted(mismatches.items()):
            log.error("  %-28s expected %6d, got %s", k, e, a)
    if len(df) != expected_total:
        log.error("Total images: expected %d, got %d", expected_total, len(df))

    if ok:
        delta = EXPECTED_TOTAL - expected_total
        log.info(
            "OK: %d images, %d classes, all counts match Table 1%s.",
            len(df), len(actual),
            f" minus {delta} known-corrupt file(s)" if delta else "",
        )
        return 0
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    ap.add_argument(
        "--known-corrupt", type=Path, default=Path("data/processed/known_corrupt.txt")
    )
    args = ap.parse_args()
    sys.exit(verify(args.manifest, args.known_corrupt))


if __name__ == "__main__":
    main()