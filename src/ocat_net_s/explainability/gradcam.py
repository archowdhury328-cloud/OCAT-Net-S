"""Grad-CAM for any NCHW feature layer. Target = logit of class ``target`` (default 1 = Oral Cancer) from the
2-class head (raw logit, not softmax). Qualitative only: no lesion masks exist, so no localisation claim is made.
For OCAT-Net-S use ``model.stage4`` (7x7x256, NCHW); the CNN-style baselines use their last conv stage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def gradcam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target: int = 1, out_size: int = 224):
    """Returns (cam (B,1,out,out) in [0,1], logits). Raw CAM before normalisation is also returned in ``raw``."""
    store: dict[str, torch.Tensor] = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(False)
            logits = model(x)
            act = store["a"]
            (grad,) = torch.autograd.grad(logits[:, target].sum(), act)
    finally:
        h1.remove()
    if act.ndim != 4:
        raise ValueError(f"expected NCHW activation, got {tuple(act.shape)}; reshape tokens first")
    w = grad.mean(dim=(2, 3), keepdim=True)
    raw = F.relu((w * act).sum(1, keepdim=True)).detach()
    cam = F.interpolate(raw, size=(out_size, out_size), mode="bilinear", align_corners=False)
    mx = cam.amax(dim=(2, 3), keepdim=True)
    return torch.where(mx > 0, cam / mx.clamp_min(1e-12), torch.zeros_like(cam)), {
        "raw": raw,
        "logits": logits.detach(),
    }


def select_examples(y: list[int], pred: list[int], ids: list[str], per_cell: int = 2) -> dict[str, list[str]]:
    """Predefined selection rule: first ``per_cell`` image_ids (sorted by id hash order) in each of TP/TN/FP/FN."""
    cells = {"TP": [], "TN": [], "FP": [], "FN": []}
    for i, (t, p) in enumerate(zip(y, pred, strict=True)):
        cells[("T" if t == p else "F") + ("P" if p == 1 else "N")].append(ids[i])
    return {k: sorted(v)[:per_cell] for k, v in cells.items()}
