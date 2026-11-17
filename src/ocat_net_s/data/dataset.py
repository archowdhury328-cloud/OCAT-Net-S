"""Image dataset over a manifest slice."""

from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ocat_net_s.data.transforms import corrupt, eval_geometry_transform
from ocat_net_s.utils.common import safe_join


def load_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, transform, corruption: tuple[str, str] | None = None, normalize=None):
        self.df, self.root, self.transform = df.reset_index(drop=True), root, transform
        self.corruption, self.normalize = corruption, normalize
        self.geom = eval_geometry_transform() if corruption else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        im = load_rgb(safe_join(self.root, r["rel_path"]))
        if self.corruption:  # corrupt AFTER deterministic resize/crop, BEFORE tensor conversion + normalisation
            im = corrupt(self.geom(im), *self.corruption, r["image_id"])
        return {"image": self.transform(im), "label": torch.tensor(int(r["label"])), "image_id": r["image_id"]}


def pil_images(df: pd.DataFrame, root: str) -> list[Image.Image]:
    g = eval_geometry_transform()
    return [g(load_rgb(safe_join(root, p))) for p in df["rel_path"]]
