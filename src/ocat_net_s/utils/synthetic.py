"""SYNTHETIC images for software tests ONLY. Never used for, or reported as, scientific results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_synthetic_dataset(root, n_per_class: int = 30, size: int = 72, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    root = Path(root)
    for name, base in (("Normal", (140, 90, 90)), ("Oral Cancer", (170, 60, 60))):
        (root / name).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            a = np.clip(rng.normal(base, 25, (size, size, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(a).save(root / name / f"{name.split()[0].lower()}_{i + 1}.png")
    (root / "SYNTHETIC_DATA_DO_NOT_REPORT").write_text("software tests only\n")
