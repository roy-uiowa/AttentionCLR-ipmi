"""SimCLR pretraining entry point.

Usage:
    # Smoke (50 batches, 2 epochs, ~5 min):
    python scripts/pretrain_simclr.py --config configs/clr/pretrain_resnext50_cbam.yaml \
        --smoke

    # Full pretrain (200 epochs):
    python scripts/pretrain_simclr.py --config configs/clr/pretrain_resnext50_cbam.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml

from attentionclr.clr.augment import SimCLRAugmentConfig, SimCLRPairAugment, SimCLRTwoViewDataset
from attentionclr.clr.encoder import ResNeXt50CBAMEncoder
from attentionclr.clr.litmodule import SimCLRModule
from attentionclr.clr.projection import SimCLRProjectionHead
from attentionclr.data.dataset import BMDataset
from attentionclr.data.splits import fivefold
from attentionclr.utils.config import load_config
from attentionclr.utils.io import read_manifest
from attentionclr.utils.logging import get_logger
from attentionclr.utils.provenance import capture_provenance
from attentionclr.utils.seed import seed_everything

log = get_logger("pretrain_simclr")


def build_dataloaders(cfg, fold, split: str, batch_size: int, num_workers: int):
    manifest = read_manifest(Path(cfg["paths"]["manifest"]))
    folds = fivefold(manifest, seed=cfg.get("seed", 42), n_splits=5)
    f = folds[fold]

    if split == "train":
        indices = f.train_idx.tolist()
    elif split == "val":
        indices = f.val_idx.tolist()
    else:
        raise ValueError(f"Unknown split: {split}")

    # BMDataset with transform=None yields raw PIL images (required by SimCLRTwoViewDataset).
    base_ds = BMDataset(
        manifest=manifest,
        data_root=Path(cfg["paths"]["data_root"]),
        indices=indices,
        transform=None,  # critical: SimCLRTwoViewDataset applies the augmentation
    )

    aug_cfg = SimCLRAugmentConfig(**cfg["augment"])
    augment = SimCLRPairAugment(cfg=aug_cfg, base_seed=cfg.get("seed", 42))
    two_view_ds = SimCLRTwoViewDataset(base_ds, augment)

    # For pretraining, we use a plain shuffle sampler. No class balancing
    # (labels are discarded, and SimCLR benefits from a balanced but random
    # batch rather than class-stratified sampling — the contrastive task is
    # "same image vs. different image", not "same class vs. different class").
    loader = torch.utils.data.DataLoader(
        two_view_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # NT-Xent needs 2N-1 >= 3 => N >= 2; drop_last avoids tail batches of 1
        persistent_workers=num_workers > 0,
    )
    return loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 50 batches")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--precision", type=str, default=None, help="Override precision (e.g. 32)")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    # Build encoder + projection head.
    encoder = ResNeXt50CBAMEncoder(
        in_channels=3,
        pretrained=cfg["encoder"]["pretrained"],
        cbam_reduction=cfg["encoder"]["cbam_reduction"],
    )
    projection = SimCLRProjectionHead(
        in_dim=cfg["projection"]["in_dim"],
        hidden_dim=cfg["projection"]["hidden_dim"],
        out_dim=cfg["projection"]["out_dim"],
    )

    max_epochs = 2 if args.smoke else cfg["training"]["epochs"]
    lit = SimCLRModule(
        encoder=encoder,
        projection_head=projection,
        temperature=cfg["loss"]["temperature"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_frac=cfg["training"]["warmup_frac"],
        max_epochs=max_epochs,
    )

    train_loader = build_dataloaders(
        cfg,
        fold=cfg["data"]["fold"],
        split=cfg["data"]["split"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    run_dir = args.run_dir or Path(cfg["paths"]["artifacts"]) / "runs" / "simclr" / "pretrain"
    if args.smoke:
        run_dir = run_dir / "smoke"
    capture_provenance(cfg, run_dir)

    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="last",
        save_last=True,
        save_top_k=0,
        every_n_epochs=cfg["training"].get("checkpoint_every_n_epochs", 10),
    )

    precision = args.precision or cfg["training"].get("precision", "16-mixed")
    log.info("Building Trainer (precision=%s, max_epochs=%d)", precision, max_epochs)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        default_root_dir=str(run_dir),
        accelerator="auto",
        devices=cfg["training"].get("devices", "auto"),
        precision=precision,
        log_every_n_steps=cfg["training"].get("log_every_n_steps", 20),
        limit_train_batches=50 if args.smoke else None,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[checkpoint],
    )

    trainer.fit(lit, train_dataloaders=train_loader)
    log.info("Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
