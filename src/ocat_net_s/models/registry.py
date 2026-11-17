"""Model registry: OCAT-Net-S + 9 ablations + 10 quantitative baselines + ConvNeXt-B (qualitative comparator).

timm identifiers verified with timm 1.0.28 (`timm.list_models`). Where a paper name could map to several variants,
the chosen one is listed in docs/assumptions_and_deviations.md (e.g. TinyViT-5M -> tiny_vit_5m_224,
EfficientViT-M4 -> efficientvit_m4 [the MIT 'M' family], MaxViT-Tiny -> maxvit_tiny_tf_224).
Pretraining: no pretrained OCAT-Net-S checkpoint exists in timm (custom architecture) -> from-scratch default;
`load_partial_checkpoint` only transfers shape-matching keys and logs everything else.
"""

from __future__ import annotations

import dataclasses
import logging

import timm
import torch
from torch import nn

from ocat_net_s.models.ocat import OCATConfig, OCATNetS

log = logging.getLogger(__name__)

BASELINES = {
    "resnet50": "resnet50",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121": "densenet121",
    "deit_s16": "deit_small_patch16_224",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "maxvit_tiny": "maxvit_tiny_tf_224",
    "efficientvit_m4": "efficientvit_m4",
    "tinyvit_5m": "tiny_vit_5m_224",
}
QUALITATIVE = {"convnext_b": "convnext_base"}

ABLATIONS = {
    "full": {},
    "cnn_only": {"late_block": "mbconv_se"},  # ASSUMPTION: TFSG stages -> MBConv-SE blocks (same widths/depths)
    "mhsa": {"late_block": "mhsa"},
    "no_se": {"se_in_stage2": False},
    "no_cpe": {"cpe": False},
}
# training-side ablations (no model change): handled by training config flags
TRAINING_ABLATIONS = {"no_mixup_cutmix": {"mix": False}, "ce_loss": {"loss": "ce"}, "no_label_smoothing": {"smoothing": 0.0},
                      "no_ema": {"ema": False}}  # fmt: skip

ALL_EXPERIMENT_MODELS = ["ocat_full"] + list(BASELINES)


def build_model(name: str, pretrained: bool = False, num_classes: int = 2, **ocat_overrides) -> nn.Module:
    """name: ``ocat`` (+ ablation key via ``ablation=``), or a baseline / qualitative key."""
    if name == "ocat":
        ab = ocat_overrides.pop("ablation", "full")
        cfg = dataclasses.replace(OCATConfig(), **{**ABLATIONS[ab], **ocat_overrides})
        return OCATNetS(cfg)
    table = {**BASELINES, **QUALITATIVE}
    if name not in table:
        raise KeyError(f"unknown model {name}; options: ocat, {sorted(table)}")
    return timm.create_model(table[name], pretrained=pretrained, num_classes=num_classes)


def load_partial_checkpoint(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Load only keys whose names and shapes match; return and log missing / unexpected / shape-mismatched keys."""
    own = model.state_dict()
    ok, mismatched, unexpected = {}, [], []
    for k, v in state.items():
        if k not in own:
            unexpected.append(k)
        elif own[k].shape != v.shape:
            mismatched.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(own[k].shape)}")
        else:
            ok[k] = v
    missing = [k for k in own if k not in ok]
    model.load_state_dict(ok, strict=False)
    rep = {"loaded": list(ok), "missing": missing, "unexpected": unexpected, "shape_mismatched": mismatched}
    log.warning("partial load: %d loaded, %d missing, %d unexpected, %d mismatched",
                len(ok), len(missing), len(unexpected), len(mismatched))  # fmt: skip
    return rep
