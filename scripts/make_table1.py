"""Regenerate Table 1 (strict + tolerant P/R, mean ± SD across folds) from eval outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from attentionclr.data.table1 import CLASS_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--out", type=Path, default=Path("reports/table1.csv"))
    args = ap.parse_args()

    # Load every fold's metrics.json.
    metrics_per_fold = []
    for metrics_file in sorted(args.runs.glob("fold*/eval/metrics.json")):
        metrics_per_fold.append(json.loads(metrics_file.read_text()))
    if not metrics_per_fold:
        raise SystemExit(f"No metrics found under {args.runs}")

    rows = []
    for cls in CLASS_NAMES:
        def _collect(key, field):
            return np.array([m[key][cls][field] for m in metrics_per_fold])
        sp = _collect("strict", "precision"); sr = _collect("strict", "recall")
        tp = _collect("tolerant", "precision"); tr = _collect("tolerant", "recall")
        support = metrics_per_fold[0]["strict"][cls]["support"]
        rows.append({
            "class": cls,
            "tolerant_precision_mean": tp.mean(), "tolerant_precision_std": tp.std(ddof=1) if len(tp) > 1 else 0.0,
            "tolerant_recall_mean": tr.mean(), "tolerant_recall_std": tr.std(ddof=1) if len(tr) > 1 else 0.0,
            "strict_precision_mean": sp.mean(), "strict_precision_std": sp.std(ddof=1) if len(sp) > 1 else 0.0,
            "strict_recall_mean": sr.mean(), "strict_recall_std": sr.std(ddof=1) if len(sr) > 1 else 0.0,
            "support": support,
        })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
