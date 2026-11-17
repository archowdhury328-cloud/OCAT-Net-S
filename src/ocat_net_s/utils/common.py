"""Hashing, seeding, safe paths, environment record."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np

NORMAL, CANCER = 0, 1
LABEL_NAMES = {NORMAL: "Normal", CANCER: "Oral Cancer"}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def safe_join(root: str | Path, rel: str | Path) -> Path:
    r = Path(root).resolve()
    full = (r / rel).resolve()
    if r != full and r not in full.parents:
        raise ValueError(f"path escapes data root: {rel}")
    return full


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def environment_record() -> dict[str, Any]:
    import torch

    rec: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 2))
    for mod in ("torchvision", "timm", "numpy", "pandas", "scipy", "sklearn", "PIL"):
        try:
            rec[mod] = getattr(__import__(mod), "__version__", "unknown")
        except ImportError:
            rec[mod] = None
    return rec
