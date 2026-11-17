"""AdamW with layer-wise LR decay groups, warm-up + cosine schedule, EMA."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from ocat_net_s.models.ocat import param_group_name

# Table 4 (= 2e-4 * 0.75^k)
GROUP_LR = {
    "head": 2.000e-4,
    "stage4": 1.500e-4,
    "stage3": 1.125e-4,
    "stage2": 8.438e-5,
    "stage1": 6.328e-5,
    "stem": 4.746e-5,
}


def _no_decay(name: str, p: torch.nn.Parameter) -> bool:
    # bias, norm parameters (1-D), relative-position table and the gate scale lambda
    return p.ndim <= 1 or name.endswith(".bias") or "rel_table" in name or name.endswith(".lam")


def build_param_groups(model: nn.Module, weight_decay: float = 0.05, group_lr: dict | None = None) -> list[dict]:
    group_lr = group_lr or GROUP_LR
    buckets: dict[tuple[str, bool], list] = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            buckets.setdefault((param_group_name(n), _no_decay(n, p)), []).append(p)
    groups = []
    for (g, nd), ps in sorted(buckets.items()):
        groups.append(
            {
                "params": ps,
                "lr": group_lr[g],
                "base_lr": group_lr[g],
                "weight_decay": 0.0 if nd else weight_decay,
                "name": f"{g}{'/no_decay' if nd else ''}",
            }
        )
    return groups


def lr_at(step: int, total: int, warmup: int, base: float, min_lr: float = 1e-6) -> float:
    """Linear warm-up from 0 to base, then cosine annealing (no restart) to min_lr."""
    if step < warmup:
        return base * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + (base - min_lr) * 0.5 * (1 + math.cos(math.pi * prog))


def set_lrs(opt: torch.optim.Optimizer, step: int, total: int, warmup: int, min_lr: float = 1e-6) -> None:
    for g in opt.param_groups:
        g["lr"] = lr_at(step, total, warmup, g["base_lr"], min_lr)


class EMA:
    """Parameter EMA updated after every optimizer step; buffers (BN statistics) are copied, not averaged."""

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, m in zip(self.module.parameters(), model.parameters(), strict=True):
            e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
        for e, m in zip(self.module.buffers(), model.buffers(), strict=True):
            e.copy_(m)
