"""Evaluate a trained checkpoint on its validation fold; write metrics + confusion matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from attentionclr.data.datamodule import BMDataModule, DataConfig
from attentionclr.data.imbalance import ImbalanceConfig
from attentionclr.data.splits import fivefold
from attentionclr.data.table1 import CLASS_NAMES
from attentionclr.eval.metrics import compute_metrics, strict_metrics
from attentionclr.eval.confusion import save_confusion
from attentionclr.models.registry import build_model
from attentionclr.utils.config import load_config, merge_subconfig
from attentionclr.utils.io import load_yaml, read_manifest
from attentionclr.utils.logging import get_logger
from attentionclr.train.litmodule import ClassifierModule

log = get_logger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    cfg = load_config(args.config)
    cfg["model"] = merge_subconfig(cfg, "model")
    if isinstance(cfg.get("augment"), str):
        cfg["augment"] = merge_subconfig(cfg, "augment")

    manifest = read_manifest(Path(cfg["paths"]["manifest"]))
    folds = fivefold(manifest, seed=cfg.get("seed", 1337), n_splits=cfg["training"]["folds"])
    fold = folds[args.fold]

    # Build the val DataLoader only (no training sampler needed).
    from dataclasses import fields
    from attentionclr.data.augment import GeometricConfig, StainConfig
    geo_allowed = {f.name for f in fields(GeometricConfig)}
    a = cfg["augment"]
    dm = BMDataModule(
        DataConfig(
            data_root=Path(cfg["paths"]["data_root"]),
            manifest=Path(cfg["paths"]["manifest"]),
            image_size=cfg.get("image_size", 250),
            num_workers=cfg.get("num_workers", 4),
            batch_size=cfg["training"].get("batch_size", 64),
            geo=GeometricConfig(**{k: v for k, v in a.items() if k in geo_allowed}),
            stain=StainConfig(a.get("stain_method", "macenko"), a.get("stain_sigma", 0.05)),
            imbalance=ImbalanceConfig(
                cfg["training"]["imbalance"]["mode"],
                cfg["training"]["imbalance"]["target_per_class"],
                cfg["training"]["imbalance"].get("max_samples_per_epoch"),
            ),
            seed=cfg.get("seed", 1337),
        ),
        fold=fold,
    )
    dm.setup()
    val_loader = dm.val_dataloader()

    model = build_model(cfg["model"]["name"], num_classes=cfg["model"].get("num_classes", 21))
    # state = torch.load(args.checkpoint, map_location="cpu")
    # model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    # model.eval().cuda()
    lit = ClassifierModule.load_from_checkpoint(
        args.checkpoint,
        model=model,
        lr=cfg["training"].get("lr", 3e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    model = lit.model.eval().cuda()

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.cuda())
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_true.append(y.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    tol_pairs = load_yaml(Path(cfg["eval"]["tolerance_classes"])).get("tolerance_pairs", []) or []
    strict = strict_metrics(y_true, y_pred, CLASS_NAMES)
    tolerant = compute_metrics(y_true, y_pred, CLASS_NAMES, tolerance_pairs=[tuple(p) for p in tol_pairs])
    
    fold_run_dir = args.run_dir / f"fold{args.fold}"
    out_dir = fold_run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "fold": args.fold,
        "strict": {k: {"precision": v.precision, "recall": v.recall, "support": v.support} for k, v in strict.per_class.items()},
        "tolerant": {k: {"precision": v.precision, "recall": v.recall, "support": v.support} for k, v in tolerant.per_class.items()},
        "macro_strict": {"precision": strict.macro_precision, "recall": strict.macro_recall},
        "macro_tolerant": {"precision": tolerant.macro_precision, "recall": tolerant.macro_recall},
    }, indent=2), encoding="utf-8")
    save_confusion(y_true, y_pred, CLASS_NAMES, out_dir / "confusion_strict.png")
    log.info("Wrote metrics + confusion to %s", out_dir)
    log.info("Macro strict: P=%.3f R=%.3f", strict.macro_precision, strict.macro_recall)
    log.info("Macro tolerant: P=%.3f R=%.3f", tolerant.macro_precision, tolerant.macro_recall)


if __name__ == "__main__":
    main()
