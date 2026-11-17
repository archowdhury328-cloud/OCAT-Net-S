"""Declarative transform pipelines (paper Table 2 + preprocessing text) and synthetic corruptions.

Training: RandomResizedCrop(224, scale .70-1, ratio .75-1.33, bicubic) -> HFlip(.5) -> Rotation(+-15, p=1, fill 0)
-> ColorJitter(.4,.4,.4,.1) -> GaussianBlur(3x3, sigma .1-2, p=.5) -> AutoAugment(ImageNet) -> ToTensor ->
Normalize(train-fold stats) -> RandomErasing(p=.25, scale .02-.33, ratio .3-3.3, random value).
ORDER is an assumption (paper lists operations, not order). Eval: Resize(256) -> CenterCrop(224) -> ToTensor ->
Normalize. Interpolation for eval resize: bicubic (assumption). No TTA.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

from ocat_net_s.utils.common import sha256_json


def build_eval_transform(mean, std) -> T.Compose:
    return T.Compose([T.Resize(256, interpolation=IM.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])


def eval_geometry_transform() -> T.Compose:
    """Deterministic resize+crop returning a PIL image (used for norm stats and corruptions)."""
    return T.Compose([T.Resize(256, interpolation=IM.BICUBIC), T.CenterCrop(224)])


def build_train_transform(mean, std) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.70, 1.0), ratio=(0.75, 1.33), interpolation=IM.BICUBIC),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(15, expand=False, fill=0),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"),
    ])  # fmt: skip


def compute_norm_stats(images: list[Image.Image]) -> tuple[list[float], list[float]]:
    """Per-channel mean/std over deterministic-preprocessed images of the OPTIMISATION-TRAIN partition only."""
    n, s, ss = 0, torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    to_t = T.ToTensor()
    for im in images:
        x = to_t(im).double()
        n += x.shape[1] * x.shape[2]
        s += x.sum(dim=(1, 2))
        ss += (x**2).sum(dim=(1, 2))
    if n == 0:
        raise ValueError("no images for normalisation statistics")
    mean = s / n
    std = (ss / n - mean**2).clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()


# ---------------- synthetic corruptions (paper Table 9). Domain: uint8 RGB after resize+crop, before normalisation.
CORRUPTION_LEVELS = {
    "gaussian_noise": {"low": 0.01, "medium": 0.05, "high": 0.10},  # sigma on [0,1] scale
    "gaussian_blur": {"low": 3, "medium": 5, "high": 7},  # kernel size; sigma ASSUMED (cv2 rule below)
    "jpeg": {"low": 70, "medium": 50, "high": 30},  # quality, PIL/libjpeg
}


def _blur_sigma(k: int) -> float:
    return 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # OpenCV default sigma-from-kernel rule


def corrupt(img: Image.Image, kind: str, level: str, sample_id: str) -> Image.Image:
    """Reproducible corruption; randomness depends only on (sample_id, kind, level) so it is shared across models."""
    p = CORRUPTION_LEVELS[kind][level]
    if kind == "gaussian_noise":
        seed = int(sha256_json([sample_id, kind, level])[:8], 16)
        a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        a = np.clip(a + np.random.default_rng(seed).normal(0, p, a.shape).astype(np.float32), 0, 1)
        return Image.fromarray((a * 255).round().astype(np.uint8))
    if kind == "gaussian_blur":
        t = T.functional.gaussian_blur(T.functional.pil_to_tensor(img), [p, p], [_blur_sigma(p)] * 2)
        return T.functional.to_pil_image(t)  # reflect padding (torchvision default)
    if kind == "jpeg":
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=int(p))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"unknown corruption {kind}")
