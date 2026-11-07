import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch
from attentionclr.clr.augment import SimCLRPairAugment

# Load a real cell
manifest_path = Path("data/processed/manifest.parquet")
import pandas as pd
manifest = pd.read_parquet(manifest_path)
img_path = Path("data/bone_marrow_cell_dataset") / manifest.iloc[0]["rel_path"]
img = Image.open(img_path).convert("RGB")

aug = SimCLRPairAugment(base_seed=0)
va, vb = aug(img)

# Denormalize for display
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def denorm(t):
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(denorm(va)); axes[1].set_title("View A"); axes[1].axis("off")
axes[2].imshow(denorm(vb)); axes[2].set_title("View B"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("artifacts/explain/simclr_aug_sample.png", dpi=150)
