"""Linear probe on a frozen SimCLR encoder.

Loads the encoder from a SimCLR checkpoint, freezes it, attaches a single
Linear(2048, 21) head, trains the head on fold-0 train (labels), evaluates
on fold-0 val. This is the standard SSL representation-quality metric.

Usage:
    python scripts/linear_probe.py \
        --config configs/clr/pretrain_resnext50_cbam.yaml \
        --checkpoint artifacts/runs/simclr/pretrain/probe50/checkpoints/last.ckpt \
        --fold 0 \
        --run-dir artifacts/runs/linear_probe/probe50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attentionclr.clr.encoder import ResNeXt50CBAMEncoder
from attentionclr.data.dataset import BMDataset
from attentionclr.data.splits import fivefold
from attentionclr.data.table1 import CLASS_NAMES, NUM_CLASSES
from attentionclr.data.transforms import val_transform
from attentionclr.utils.config import load_config
from attentionclr.utils.io import read_manifest
from attentionclr.utils.logging import get_logger
from attentionclr.utils.seed import seed_everything

log = get_logger("linear_probe")


def load_encoder(ckpt_path: str) -> nn.Module:
    """Load encoder state_dict from a SimCLR Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    encoder = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False)
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        log.warning("Encoder load: %d missing, %d unexpected", len(missing), len(unexpected))
    log.info("Loaded encoder from %s", ckpt_path)
    return encoder


def extract_features(encoder, loader, device, max_samples: int | None = None):
    encoder.eval()
    feats, labels = [], []
    n = 0
    with torch.no_grad():
        for x, y in loader:
            f = encoder(x.to(device, non_blocking=True))
            feats.append(f.cpu())
            labels.append(y)
            n += x.size(0)
            if max_samples is not None and n >= max_samples:
                break
    F = torch.cat(feats, dim=0)
    Y = torch.cat(labels, dim=0)
    if max_samples is not None:
        F, Y = F[:max_samples], Y[:max_samples]
    return F, Y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap training samples (None = use full fold-0 train)")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.config)
    manifest = read_manifest(Path(cfg["paths"]["manifest"]))
    folds = fivefold(manifest, seed=42, n_splits=5)
    f = folds[args.fold]

    # Extract features once; then train a linear head on them.
    train_ds = BMDataset(
        manifest=manifest,
        data_root=Path(cfg["paths"]["data_root"]),
        indices=f.train_idx.tolist(),
        transform=val_transform(250),
    )
    val_ds = BMDataset(
        manifest=manifest,
        data_root=Path(cfg["paths"]["data_root"]),
        indices=f.val_idx.tolist(),
        transform=val_transform(250),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=8)

    encoder = load_encoder(str(args.checkpoint)).to(device)

    log.info("Extracting train features ...")
    F_train, Y_train = extract_features(encoder, train_loader, device, max_samples=args.max_train)
    log.info("Extracting val features ...")
    F_val, Y_val = extract_features(encoder, val_loader, device)

    log.info("Train features: %s, Val features: %s", tuple(F_train.shape), tuple(F_val.shape))

    # Train a Linear head on the frozen features.
    head = nn.Linear(2048, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = F_train.size(0)

    best_acc = 0.0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if idx.numel() < 2:
                continue
            xb = F_train[idx].to(device, non_blocking=True)
            yb = Y_train[idx].to(device, non_blocking=True)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, F_val.size(0), 512):
                xb = F_val[i:i + 512].to(device)
                yb = Y_val[i:i + 512].to(device)
                pred = head(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / total
        best_acc = max(best_acc, acc)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            log.info("Epoch %2d: loss=%.4f  val_acc=%.4f", epoch, total_loss / max(1, n_batches), acc)

    log.info("Best val accuracy: %.4f", best_acc)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "metrics.json").write_text(json.dumps({
        "fold": args.fold,
        "best_val_acc": best_acc,
        "n_train": int(F_train.size(0)),
        "n_val": int(F_val.size(0)),
        "epochs": args.epochs,
        "checkpoint": str(args.checkpoint),
    }, indent=2), encoding="utf-8")
    torch.save(head.state_dict(), args.run_dir / "head.pt")
    log.info("Wrote %s", args.run_dir / "metrics.json")


if __name__ == "__main__":
    main()
