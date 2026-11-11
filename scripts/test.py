#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:57:51 2026

@author: troy3
"""

# import torch
# from attentionclr.clr.encoder import ResNeXt50CBAMEncoder
# from attentionclr.data.datamodule import BMDataModule, DataConfig
# from attentionclr.data.splits import fivefold
# from attentionclr.data.transforms import TrainTransform, val_transform
# from attentionclr.utils.io import read_manifest
# from pathlib import Path

# manifest = read_manifest(Path("data/processed/manifest.parquet"))
# folds = fivefold(manifest, seed=42, n_splits=5)
# f = folds[0]

# # Same DataConfig as the fine-tune
# dm = BMDataModule(
#     DataConfig(
#         data_root=Path("data/bone_marrow_cell_dataset"),
#         manifest=Path("data/processed/manifest.parquet"),
#         image_size=250,
#         batch_size=64,
#         num_workers=0,
#     ),
#     fold=f,
# )
# dm.setup()

# # Print first train and first val batch stats
# train_x, train_y = next(iter(dm.train_dataloader()))
# val_x, val_y = next(iter(dm.val_dataloader()))
# print(f"train batch: {train_x.shape}, labels {train_y[:8].tolist()}")
# print(f"train_x stats: mean={train_x.mean():.4f}, std={train_x.std():.4f}, "
#       f"min={train_x.min():.4f}, max={train_x.max():.4f}")
# print(f"val batch: {val_x.shape}, labels {val_y[:8].tolist()}")
# print(f"val_x stats: mean={val_x.mean():.4f}, std={val_x.std():.4f}, "
#       f"min={val_x.min():.4f}, max={val_x.max():.4f}")
# print(f"val_ds length: {len(dm.val_ds)}")

#%%
# from attentionclr.data.splits import fivefold
# from attentionclr.utils.io import read_manifest
# from attentionclr.data.table1 import IDX_TO_CLASS
# import pandas as pd
# from pathlib import Path

# manifest = read_manifest(Path("data/processed/manifest.parquet"))
# folds = fivefold(manifest, seed=42, n_splits=5)
# f = folds[0]

# print(f"Train indices: {len(f.train_idx)}, Val indices: {len(f.val_idx)}")
# print(f"First 16 val indices: {f.val_idx[:16].tolist()}")
# print(f"Val labels (first 16): {manifest.iloc[f.val_idx[:16]]['label'].tolist()}")
# print(f"Val label distribution: {manifest.iloc[f.val_idx]['label'].value_counts().to_dict()}")

#%%
# from attentionclr.data.splits import fivefold
# from attentionclr.utils.io import read_manifest
# from pathlib import Path

# manifest = read_manifest(Path("data/processed/manifest.parquet"))
# folds = fivefold(manifest, seed=42, n_splits=5)
# print(f"folds[0].val_idx[:16] = {folds[0].val_idx[:16].tolist()}")
# print(f"folds[1].val_idx[:16] = {folds[1].val_idx[:16].tolist()}")

#%%
# from attentionclr.data.splits import fivefold
# from attentionclr.utils.io import read_manifest
# from pathlib import Path
# import numpy as np

# manifest = read_manifest(Path("data/processed/manifest.parquet"))
# folds = fivefold(manifest, seed=42, n_splits=5)

# for i in range(5):
#     v = folds[i].val_idx
#     is_sorted = bool(np.all(np.diff(v) > 0))
#     print(f"fold {i}: sorted={is_sorted}, first 8={v[:8].tolist()}, "
#           f"last 8={v[-8:].tolist()}, min={v.min()}, max={v.max()}")

#%%
import torch
from attentionclr.clr.encoder import ResNeXt50CBAMEncoder
from attentionclr.data.dataset import BMDataset
from attentionclr.data.transforms import val_transform
from attentionclr.data.splits import fivefold
from attentionclr.utils.io import read_manifest
from pathlib import Path

# The fine-tune checkpoint is a FineTuneModule; encoder is in state_dict under 'encoder.'
ckpt = torch.load("artifacts/runs/finetune/full170_frac100_fold0/checkpoints/last.ckpt",
                  map_location="cpu", weights_only=False)
enc_sd = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
encoder = ResNeXt50CBAMEncoder(in_channels=3, pretrained=False)
encoder.load_state_dict(enc_sd, strict=False)
encoder = encoder.eval().cuda()

manifest = read_manifest(Path("data/processed/manifest.parquet"))
folds = fivefold(manifest, seed=42, n_splits=5)
ds = BMDataset(manifest, Path("data/bone_marrow_cell_dataset"),
               indices=folds[0].val_idx.tolist()[:256],
               transform=val_transform(250))
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

with torch.no_grad():
    feats = []
    for x, _ in loader:
        feats.append(encoder(x.cuda()).cpu())
F = torch.cat(feats, dim=0)

print(f"After fine-tune — feature stats (compare to probe50: per-dim std 0.21, rank 255):")
print(f"  per-dim std: {F.std(dim=0).mean().item():.4f}")
print(f"  overall std: {F.std().item():.4f}")
print(f"  per-sample norm std: {F.norm(dim=1).std().item():.4f}")
print(f"  effective rank: {torch.linalg.matrix_rank(F - F.mean(0, keepdim=True)).item()}")
from attentionclr.data.splits import fivefold
from attentionclr.utils.io import read_manifest
from pathlib import Path
import numpy as np

manifest = read_manifest(Path("data/processed/manifest.parquet"))
folds = fivefold(manifest, seed=42, n_splits=5)

for i in range(5):
    v = folds[i].val_idx
    is_sorted = bool(np.all(np.diff(v) > 0))
    print(f"fold {i}: sorted={is_sorted}, first 8={v[:8].tolist()}, "
          f"last 8={v[-8:].tolist()}, min={v.min()}, max={v.max()}")