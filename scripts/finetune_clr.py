"""Fine-tune a SimCLR-pretrained encoder end-to-end on labeled data.

Usage:
    # Full fine-tune at 100% labels, fold 0:
    python scripts/finetune_clr.py \
        --config configs/clr/pretrain_resnext50_cbam.yaml \
        --checkpoint artifacts/runs/simclr/pretrain/full200/checkpoints/last.ckpt \
        --fold 0 \
        --run-dir artifacts/runs/finetune/full170_fold0 \
        --label-fraction 1.0

    # Fine-tune at 10% labels (the SSL benefit regime):
    python scripts/finetune_clr.py \
        --config configs/clr/pretrain_resnext50_cbam.yaml \
        --checkpoint artifacts/runs/simclr/pretrain/full200/checkpoints/last.ckpt \
        --fold 0 \
        --run-dir artifacts/runs/finetune/full170_fold0_frac10 \
        --label-fraction 0.1

    # Frozen-encoder baseline (linear-probe-equivalent with fine-tune pipeline):
    python scripts/finetune_clr.py ... --frozen-encoder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attentionclr.clr.encoder import ResNeXt50CBAMEncoder
from attentionclr.data.datamodule import BMDataModule, DataConfig
from attentionclr.data.imbalance import ImbalanceConfig
from attentionclr.data.splits import fivefold
from attentionclr.data.table1 import CLASS_NAMES, NUM_CLASSES
from attentionclr.utils.config import load_config
from attentionclr.utils.io import read_manifest
from attentionclr.utils.logging import get_logger
from attentionclr.utils.provenance import capture_provenance
from attentionclr.utils.seed import seed_everything

log = get_logger("finetune_clr")


class FineTuneModule(L.LightningModule):
    """Encoder (from SimCLR) + a 21-class head, trained end-to-end on labels.

    The encoder can be frozen (`--frozen-encoder`), in which case only the
    head is trained (equivalent to a linear probe with the SSL augmentation).
    When unfrozen, the optimizer uses two param groups: a lower LR for the
    encoder (to prevent catastrophic forgetting) and a higher LR for the head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        frozen_encoder: bool = False,
        head_lr: float = 1e-3,
        encoder_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.frozen_encoder = frozen_encoder
        self.head = nn.Linear(encoder.feature_dim, num_classes)
        self.head_lr = head_lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.save_hyperparameters(ignore=["encoder"])

        if frozen_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return self.head(feats)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        if self.frozen_encoder:
            param_groups = [
                {"params": self.head.parameters(), "lr": self.head_lr},
            ]
        else:
            # Discriminative LRs: encoder at encoder_lr, head at head_lr.
            param_groups = [
                {"params": self.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.head.parameters(), "lr": self.head_lr},
            ]
        opt = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # Cosine schedule with linear warmup.
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else None
        if total_steps is None or total_steps < 1:
            return opt
        warmup_steps = max(1, int(total_steps * 0.05))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + float(torch.cos(torch.tensor(torch.pi * progress))))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def load_encoder_from_simclr(ckpt_path: Path) -> ResNeXt50CBAMEncoder:
    """Extract the encoder from a SimCLR Lightning checkpoint (ignoring projection head)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    encoder = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False)
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        log.warning("Encoder load: %d missing, %d unexpected", len(missing), len(unexpected))
    log.info("Loaded SimCLR encoder from %s", ckpt_path)
    return encoder


def build_datamodule(cfg, fold, batch_size, num_workers, label_fraction, seed=42):
    """Build the DataModule for fold-N train, optionally subsampling labels."""
    manifest = read_manifest(Path(cfg["paths"]["manifest"]))
    folds = fivefold(manifest, seed=seed, n_splits=5)
    f = folds[fold]

    # If label_fraction < 1.0, subsample the training indices (stratified by class).
    if label_fraction < 1.0:
        train_df = manifest.iloc[f.train_idx].reset_index(drop=True)
        n_per_class = (train_df["label"].value_counts() * label_fraction).round().astype(int)
        n_per_class = n_per_class.clip(lower=2)  # ensure at least 2 per class

        rng = np.random.default_rng(seed)
        sampled_idx = []
        for cls, n in n_per_class.items():
            cls_positions = train_df.index[train_df["label"] == cls].tolist()
            sampled = rng.choice(cls_positions, size=min(n, len(cls_positions)), replace=False)
            sampled_idx.extend(sampled.tolist())
        # Map local sampled positions back to manifest indices
        local_to_manifest = {i: int(f.train_idx[i]) for i in range(len(f.train_idx))}
        sampled_manifest_idx = np.array([local_to_manifest[i] for i in sorted(sampled_idx)])
        log.info("Subsampled %d train samples (%.1f%% of fold-%d train, stratified)",
                 len(sampled_manifest_idx), label_fraction * 100, fold)
    else:
        sampled_manifest_idx = np.array(f.train_idx)

    # Build a temporary Fold with the (possibly subsampled) train indices and the
    # original val indices. This reuses the existing BMDataModule without changes.
    from attentionclr.data.splits import Fold
    fold_with_frac = Fold(fold=fold, train_idx=sampled_manifest_idx, val_idx=f.val_idx)

    return BMDataModule(
        DataConfig(
            data_root=Path(cfg["paths"]["data_root"]),
            manifest=Path(cfg["paths"]["manifest"]),
            image_size=cfg.get("image_size", 250),
            num_workers=num_workers,
            batch_size=batch_size,
            imbalance=ImbalanceConfig("none", 0, None),  # no oversampling for fine-tune
            seed=seed,
        ),
        fold=fold_with_frac,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="SimCLR checkpoint (.ckpt) to initialize the encoder from. "
                         "Pass an empty string to train from random init.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--frozen-encoder", action="store_true",
                    help="Freeze the encoder; only train the head (linear-probe equivalent).")
    ap.add_argument("--epochs", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    # Load encoder from SimCLR, or build fresh if --checkpoint is empty.
    if args.checkpoint and str(args.checkpoint):
        encoder = load_encoder_from_simclr(args.checkpoint)
        init_source = str(args.checkpoint)
    else:
        encoder = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False)
        init_source = "random"

    lit = FineTuneModule(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        frozen_encoder=args.frozen_encoder,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    dm = build_datamodule(
        cfg, fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_fraction=args.label_fraction,
        seed=cfg.get("seed", 42),
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    capture_provenance({
        "config": cfg,
        "checkpoint": init_source,
        "fold": args.fold,
        "label_fraction": args.label_fraction,
        "frozen_encoder": args.frozen_encoder,
    }, args.run_dir)

    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(args.run_dir / "checkpoints"),
        filename="last",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )
    # Also save best checkpoint by val accuracy.
    best_cb = ModelCheckpoint(
        dirpath=str(args.run_dir / "checkpoints"),
        filename="best",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=str(args.run_dir),
        accelerator="auto",
        devices=1,
        precision=args.precision,
        log_every_n_steps=20,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[checkpoint_cb, best_cb],
    )
    trainer.fit(lit, datamodule=dm)

    # Reload best checkpoint for final evaluation.
    best_path = args.run_dir / "checkpoints" / "best.ckpt"
    final = FineTuneModule.load_from_checkpoint(
        best_path, encoder=encoder, frozen_encoder=args.frozen_encoder,
        head_lr=args.head_lr, encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
    )
    final.eval().cuda()

    # Final val accuracy on the best checkpoint.
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            pred = final(x.cuda()).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    final_acc = correct / total
    log.info("Best checkpoint val accuracy: %.4f", final_acc)

    (args.run_dir / "metrics.json").write_text(json.dumps({
        "fold": args.fold,
        "label_fraction": args.label_fraction,
        "frozen_encoder": args.frozen_encoder,
        "checkpoint": init_source,
        "n_train": len(dm.train_ds),
        "n_val": len(dm.val_ds),
        "epochs": args.epochs,
        "best_val_acc": final_acc,
    }, indent=2), encoding="utf-8")
    log.info("Wrote %s", args.run_dir / "metrics.json")


if __name__ == "__main__":
    main()
