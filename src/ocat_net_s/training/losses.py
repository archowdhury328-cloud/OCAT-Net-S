"""Weighted focal loss with soft targets.

FL = (1/B) sum_i sum_c  -w_c * (1 - p_ic)^gamma * q_ic * log p_ic ,   p = softmax(z), q = (mixed/smoothed) soft target.
Class weights are indexed by class (Normal 0.40, Oral Cancer 0.60); mean over the batch (not weight-normalised).
The convention is an assumption: the paper only says smoothed soft targets are used before focal-loss calculation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_focal_loss(logits: torch.Tensor, soft_targets: torch.Tensor, weights=(0.40, 0.60), gamma: float = 2.0):
    logp = F.log_softmax(logits.float(), dim=1)
    p = logp.exp()
    w = torch.as_tensor(weights, dtype=logp.dtype, device=logp.device).view(1, -1)
    return -(w * (1 - p).pow(gamma) * soft_targets * logp).sum(dim=1).mean()


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor, weights=None):
    logp = F.log_softmax(logits.float(), dim=1)
    w = 1.0 if weights is None else torch.as_tensor(weights, dtype=logp.dtype, device=logp.device).view(1, -1)
    return -(w * soft_targets * logp).sum(dim=1).mean()
