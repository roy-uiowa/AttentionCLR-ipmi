"""Second smoke test: same pipeline, tiny encoder. On a small model, 20 batches
of batch 8 IS enough signal to observe loss decrease. This test proves the
training loop learns, independent of the encoder's scale."""
import pytest
import torch
import torch.nn as nn

from attentionclr.clr.litmodule import SimCLRModule
from attentionclr.clr.projection import SimCLRProjectionHead


class TinyEncoder(nn.Module):
    """A ~10k-param CNN with the same 2048-d output shape as ResNeXt+CBAM."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 2048)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))

@pytest.mark.slow
def test_simclr_tiny_encoder_loss_decreases():
    """20 batches of batch 8 with a tiny encoder: the loss must decrease
    by ≥0.1. This is the "pipeline learns" check that the ResNeXt test can't
    provide at small scale."""
    from pathlib import Path
    if not Path("data/processed/manifest.parquet").exists():
        pytest.skip("manifest not present")
    from attentionclr.clr.augment import SimCLRAugmentConfig, SimCLRPairAugment, SimCLRTwoViewDataset
    from attentionclr.data.dataset import BMDataset
    from attentionclr.data.splits import fivefold
    from attentionclr.utils.io import read_manifest

    manifest = read_manifest(Path("data/processed/manifest.parquet"))
    folds = fivefold(manifest, seed=42, n_splits=5)
    indices = folds[0].train_idx.tolist()[:200]
    base_ds = BMDataset(
        manifest=manifest,
        data_root=Path("data/bone_marrow_cell_dataset"),
        indices=indices,
        transform=None,
    )
    aug = SimCLRPairAugment(cfg=SimCLRAugmentConfig(image_size=250), base_seed=42)
    two_view_ds = SimCLRTwoViewDataset(base_ds, aug)
    loader = torch.utils.data.DataLoader(two_view_ds, batch_size=8, shuffle=True, num_workers=0)

    torch.manual_seed(0)
    encoder = TinyEncoder()
    projection = SimCLRProjectionHead(in_dim=2048, hidden_dim=2048, out_dim=128)
    lit = SimCLRModule(
        encoder=encoder, projection_head=projection,
        temperature=0.1, lr=3e-3, weight_decay=1e-4,
        warmup_frac=0.0, max_epochs=1,
    )
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=3e-3, weight_decay=1e-4,
    )
    lit.train()
    losses = []
    for i, (va, vb) in enumerate(loader):
        if i >= 20:
            break
        optimizer.zero_grad()
        h_a = projection(encoder(va))
        h_b = projection(encoder(vb))
        loss = lit.loss_fn(h_a, h_b)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(l)).item() for l in losses)
    assert losses[0] - losses[-1] > 0.1, (
        f"Tiny-encoder loss did not decrease by 0.1: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )
    
