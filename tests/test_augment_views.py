import torch
from attentionclr.clr.augment import SimCLRPairAugment, SimCLRAugmentConfig
from PIL import Image
from pathlib import Path
import pandas as pd

df = pd.read_parquet("data/processed/manifest.parquet")
print(df.columns)
aug = SimCLRPairAugment(SimCLRAugmentConfig(image_size=250), base_seed=42)
paths = [Path("data/bone_marrow_cell_dataset") / p for p in df["rel_path"].head(8)]
imgs = [Image.open(p).convert("RGB") for p in paths]
print("Mean |view_A - view_B| per image (should be > 1.0 for real diversity):")
for i, img in enumerate(imgs):
    va, vb = aug(img)
    print(f"  img{i}: {(va-vb).abs().mean().item():.3f}")
