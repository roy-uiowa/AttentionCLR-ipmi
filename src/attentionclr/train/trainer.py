"""Trainer wrapper: Lightning Trainer construction. Git provenance moved to utils.git."""
from __future__ import annotations

from pathlib import Path

import lightning as L

from attentionclr.utils.git import git_sha_and_dirty  # re-export for back-compat


def make_trainer(
    max_epochs: int,
    run_dir: Path,
    devices: int | str = "auto",
    precision: str | int = 32,
    log_every_n_steps: int = 20,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    enable_model_summary: bool = False,
) -> L.Trainer:
    run_dir.mkdir(parents=True, exist_ok=True)
    return L.Trainer(
        max_epochs=max_epochs,
        default_root_dir=str(run_dir),
        devices=devices,
        accelerator="auto",
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=enable_model_summary,
    )