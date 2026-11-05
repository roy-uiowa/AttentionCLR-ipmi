"""End-to-end smoke: 2 epochs on a 200-image subset, CPU, <2 min."""
import pytest

pytest.importorskip("lightning")

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from attentionclr.data.datamodule import BMDataModule, DataConfig
from attentionclr.data.splits import Fold
from attentionclr.models.registry import build_model
from attentionclr.train.litmodule import ClassifierModule


@pytest.mark.skipif(
    not Path("data/bone_marrow_cell_dataset").exists(),
    reason="raw dataset not present",
)
def test_smoke_train_runs(tmp_path):
    from attentionclr.utils.io import read_manifest
    manifest = read_manifest(Path("data/processed/manifest.parquet")).head(200).reset_index(drop=True)
    # Subsample to classes with at least 2 examples so validation is non-empty.
    keep = manifest["label"].value_counts()
    keep = keep[keep >= 2].index
    manifest = manifest[manifest["label"].isin(keep)].reset_index(drop=True)

    dm = BMDataModule(
        DataConfig(
            data_root=Path("data/bone_marrow_cell_dataset"),
            manifest=Path("data/processed/manifest.parquet"),
            num_workers=0,
        ),
        fold=Fold(fold=0, train_idx=np.arange(len(manifest)), val_idx=np.arange(len(manifest))),
    )
    dm.setup()
    dm.set_batch_size(8)
    dm.train_ds.indices = list(range(min(40, len(dm.train_ds))))
    dm.val_ds.indices = list(range(min(20, len(dm.val_ds))))

    model = build_model("resnext50", num_classes=21, pretrained=False)
    lit = ClassifierModule(model, lr=1e-3)
    import lightning as L
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(lit, datamodule=dm)
