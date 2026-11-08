import pandas as pd
import torch
from pathlib import Path
from PIL import Image

from attentionclr.clr.augment import SimCLRPairAugment, SimCLRAugmentConfig

df = pd.read_parquet("data/processed/manifest.parquet")
print("Columns:", df.columns.tolist())

# One image per class, first 8 classes. drop_duplicates is the simplest way
# to do "first row of each group" and preserves all columns.
sample = df.drop_duplicates(subset="label", keep="first").head(8).reset_index(drop=True)

paths = [Path("data/bone_marrow_cell_dataset") / p for p in sample["rel_path"]]
labels = sample["label"].tolist()

aug = SimCLRPairAugment(SimCLRAugmentConfig(image_size=250), base_seed=42)

print("\nMean |view_A - view_B| per image:")
print("(> 1.0 = real diversity; < 0.5 = views near-identical)")
print()
for i, (path, label) in enumerate(zip(paths, labels)):
    img = Image.open(path).convert("RGB")
    va, vb = aug(img)
    diff = (va - vb).abs()
    print(
        f"  {i}: {label:<25s}  mean_diff={diff.mean().item():.3f}  "
        f"max_diff={diff.max().item():.3f}  "
        f"view_A_std={va.std().item():.3f}  view_B_std={vb.std().item():.3f}"
    )
