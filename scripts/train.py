"""Train a single fold. `--smoke` runs 2 epochs on 10 train + 5 val batches."""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from attentionclr.data.augment import GeometricConfig, StainConfig
from attentionclr.data.datamodule import BMDataModule, DataConfig
from attentionclr.data.imbalance import ImbalanceConfig
from attentionclr.data.splits import fivefold
from attentionclr.models.registry import build_model
from attentionclr.train.litmodule import ClassifierModule
from attentionclr.train.trainer import make_trainer
from attentionclr.utils.config import load_config, merge_subconfig
from attentionclr.utils.io import read_manifest
from attentionclr.utils.logging import get_logger
from attentionclr.utils.provenance import capture_provenance
from attentionclr.utils.seed import seed_everything

log = get_logger("train")


def _geometric_cfg_from(a: dict) -> GeometricConfig:
    allowed = {f.name for f in fields(GeometricConfig)}
    return GeometricConfig(**{k: v for k, v in a.items() if k in allowed})


def _stain_cfg_from(a: dict) -> StainConfig:
    return StainConfig(method=a.get("stain_method", "macenko"), sigma=a.get("stain_sigma", 0.05))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    # A100 Tensor-Core hint. Changes fp behavior; documented in assumptions.md.
    torch.set_float32_matmul_precision("medium")

    cfg = load_config(args.config)
    cfg["model"] = merge_subconfig(cfg, "model")
    cfg["eval"] = merge_subconfig(cfg, "eval")
    if isinstance(cfg.get("augment"), str):
        cfg["augment"] = merge_subconfig(cfg, "augment")
    for k in ("paths", "training", "augment", "model"):
        if k not in cfg:
            raise SystemExit(f"Config missing required section {k!r} after merge.")

    seed = cfg.get("seed", 1337)
    seed_everything(seed)

    manifest_path = Path(cfg["paths"]["manifest"])
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run `make data`.")
    manifest = read_manifest(manifest_path)

    folds = fivefold(manifest, seed=seed, n_splits=cfg["training"]["folds"])
    fold = folds[args.fold]

    augment = cfg["augment"]
    imb = cfg["training"]["imbalance"]
    data_cfg = DataConfig(
        data_root=Path(cfg["paths"]["data_root"]),
        manifest=manifest_path,
        image_size=cfg.get("image_size", 250),
        num_workers=cfg.get("num_workers", 4),
        batch_size=cfg["training"].get("batch_size", 64),
        geo=_geometric_cfg_from(augment),
        stain=_stain_cfg_from(augment),
        imbalance=ImbalanceConfig(
            mode=imb["mode"],
            target_per_class=imb["target_per_class"],
            max_samples_per_epoch=imb.get("max_samples_per_epoch"),
        ),
        seed=seed,
    )
    dm = BMDataModule(data_cfg, fold)

    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"].get("num_classes", 21),
        pretrained=cfg["model"].get("pretrained", False),
    )
    lit = ClassifierModule(
        model,
        lr=cfg["training"].get("lr", 3e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    run_dir = args.run_dir or Path(cfg["paths"]["artifacts"]) / "runs" / f"fold{args.fold}"
    if args.smoke:
        run_dir = run_dir / "smoke"
    capture_provenance(cfg, run_dir)

    trainer = make_trainer(
        max_epochs=2 if args.smoke else cfg["training"]["epochs"],
        run_dir=run_dir,
        devices=cfg["training"].get("devices", "auto"),
        precision=cfg["training"].get("precision", 32),
        log_every_n_steps=1 if args.smoke else 20,
        limit_train_batches=10 if args.smoke else None,
        limit_val_batches=5 if args.smoke else None,
        enable_model_summary=False,
    )
    trainer.fit(lit, datamodule=dm)
    log.info("Run dir: %s", run_dir)


if __name__ == "__main__":
    main()