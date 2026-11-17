"""Label smoothing, MixUp, CutMix on SOFT targets. Smoothing is applied exactly once (to the one-hot targets,
before mixing; mixing is linear so mixed targets stay correctly smoothed). Targets are never hardened.

Interpretation (assumption A-MIX1): per batch, MixUp fires with p=0.5 and CutMix fires with p=0.5 independently;
if both fire one is chosen uniformly at random (so they are never stacked). alpha: MixUp 0.4, CutMix 1.0.
"""

from __future__ import annotations

import numpy as np
import torch


def smooth_onehot(y: torch.Tensor, num_classes: int = 2, eps: float = 0.10) -> torch.Tensor:
    oh = torch.zeros(y.shape[0], num_classes, device=y.device)
    oh.scatter_(1, y.view(-1, 1), 1.0)
    return oh * (1 - eps) + eps / num_classes


def mixup(x: torch.Tensor, t: torch.Tensor, alpha: float, rng: np.random.Generator):
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device, generator=None)
    return lam * x + (1 - lam) * x[perm], lam * t + (1 - lam) * t[perm], lam


def cutmix(x: torch.Tensor, t: torch.Tensor, alpha: float, rng: np.random.Generator):
    """Box area sets the weight: lam is recomputed from the CLIPPED patch area."""
    lam0 = float(rng.beta(alpha, alpha))
    _, _, h, w = x.shape
    cut = np.sqrt(1.0 - lam0)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    perm = torch.randperm(x.shape[0], device=x.device)
    xm = x.clone()
    xm[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1.0 - float((y2 - y1) * (x2 - x1)) / (h * w)  # actual kept-area fraction
    return xm, lam * t + (1 - lam) * t[perm], lam


def apply_batch_mix(x, t, rng: np.random.Generator, p_mixup=0.5, p_cutmix=0.5, a_mixup=0.4, a_cutmix=1.0):
    do_m, do_c = rng.random() < p_mixup, rng.random() < p_cutmix
    if do_m and do_c:
        do_m, do_c = (True, False) if rng.random() < 0.5 else (False, True)
    if do_m:
        return (*mixup(x, t, a_mixup, rng)[:2], "mixup")
    if do_c:
        return (*cutmix(x, t, a_cutmix, rng)[:2], "cutmix")
    return x, t, "none"
