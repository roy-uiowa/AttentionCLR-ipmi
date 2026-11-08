"""SimCLR LightningModule: encoder + projection head + NT-Xent loss.

Data flow per training step:
  (view_a, view_b) -> encoder -> (z_a, z_b)  [2048-d features]
                   -> projection_head -> (h_a, h_b)  [128-d embeddings]
                   -> NT-Xent loss

The encoder's 2048-d features are the representation that downstream
linear probe / fine-tune uses. The 128-d embeddings are only for the
contrastive objective.
"""
from __future__ import annotations

import torch
import lightning as L

from attentionclr.clr.encoder import ResNeXt50CBAMEncoder
from attentionclr.clr.losses import NTXentLoss
from attentionclr.clr.projection import SimCLRProjectionHead


class SimCLRModule(L.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        projection_head: torch.nn.Module,
        temperature: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_frac: float = 0.1,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.loss_fn = NTXentLoss(temperature=temperature)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.max_epochs = max_epochs
        # Save hyperparameters for checkpoint reloading. Note: we don't save the
        # encoder/projection as hparams (they're submodules); Lightning handles
        # state_dict correctly, and load_from_checkpoint(encoder=..., ...) reconstructs.
        self.save_hyperparameters(ignore=["encoder", "projection_head"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 2048-d features (for linear probe / fine-tune)."""
        return self.encoder(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        va, vb = batch
        z_a = self.encoder(va)
        z_b = self.encoder(vb)
        h_a = self.projection_head(z_a)
        h_b = self.projection_head(z_b)
        loss = self.loss_fn(h_a, h_b)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    # def validation_step(self, batch, _):
    #     # We don't have labels; validation is only a sanity check that the loss
    #     # on held-out data isn't collapsing. The real check is linear probe at
    #     # Phase 6, which uses labels on the frozen encoder.
    #     return self._shared_step(batch, "val")

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.projection_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # Cosine schedule with linear warmup over warmup_frac of training.
        # We need a rough total step count; Lightning's CosineAnnealingLR
        # with T_max = total_steps is the standard approach. We use a
        # LambdaLR with warmup + cosine decay, computed from self.trainer.
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else None
        if total_steps is None or total_steps < 1:
            # Fall back to constant LR if trainer isn't yet attached.
            return optimizer

        warmup_steps = max(1, int(total_steps * self.warmup_frac))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
