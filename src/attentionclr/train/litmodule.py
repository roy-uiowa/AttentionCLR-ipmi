"""LightningModule wrapping a classifier with CE loss + accuracy."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MulticlassAccuracy

from attentionclr.data.table1 import NUM_CLASSES


class ClassifierModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_acc = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc if stage == "train" else self.val_acc
        acc(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # Placeholder: AdamW with config lr. See configs/train/*.yaml (ASSUMED).
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
