"""torch Dataset over the manifest with transform + target plumbing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from attentionclr.data.table1 import CLASS_TO_IDX


class BMDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        data_root: str | Path,
        indices: list[int] | None = None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.indices = list(range(len(self.manifest))) if indices is None else list(indices)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.manifest.iloc[self.indices[i]]
        with Image.open(self.data_root / row["rel_path"]) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        target = CLASS_TO_IDX[row["label"]]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return im, target
