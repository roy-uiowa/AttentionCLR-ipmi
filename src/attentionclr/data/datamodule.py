"""LightningDataModule over manifest + split."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

from attentionclr.data.augment import GeometricConfig, StainConfig
from attentionclr.data.dataset import BMDataset
from attentionclr.data.imbalance import ImbalanceConfig, make_sampler
from attentionclr.data.splits import Fold
from attentionclr.data.table1 import CLASS_TO_IDX
from attentionclr.data.transforms import TrainTransform, val_transform
from attentionclr.utils.io import read_manifest


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    manifest: Path
    image_size: int = 250
    num_workers: int = 4
    batch_size: int = 64
    geo: GeometricConfig = field(default_factory=GeometricConfig)
    stain: StainConfig = field(default_factory=StainConfig)
    imbalance: ImbalanceConfig = field(
        default_factory=lambda: ImbalanceConfig("on_the_fly_oversample_to", 25_000, 171_372)
    )
    seed: int = 1337


class BMDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig, fold: Fold) -> None:
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self._manifest = None
        self._train_sampler = None

    def setup(self, stage: str | None = None) -> None:
        if self._manifest is None:
            self._manifest = read_manifest(self.cfg.manifest)

        train_labels = (
            self._manifest.iloc[self.fold.train_idx]["label"].map(CLASS_TO_IDX).to_numpy()
        )
        self._train_sampler = make_sampler(train_labels, self.cfg.imbalance, seed=self.cfg.seed)

        self.train_ds = BMDataset(
            self._manifest,
            data_root=self.cfg.data_root,
            indices=self.fold.train_idx.tolist(),
            transform=TrainTransform(
                size=self.cfg.image_size,
                geo_cfg=self.cfg.geo,
                stain_cfg=self.cfg.stain,
                seed=self.cfg.seed,
            ),
        )
        self.val_ds = BMDataset(
            self._manifest,
            data_root=self.cfg.data_root,
            indices=self.fold.val_idx.tolist(),
            transform=val_transform(self.cfg.image_size),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            sampler=self._train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )