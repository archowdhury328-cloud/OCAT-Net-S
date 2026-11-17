"""OCAT-Net-S (paper Fig. 3 / Table 3). Tensors between blocks are NCHW; attention blocks convert internally.

Hierarchy (verified against Table 3): stem 112x112x32 -> 56x56x64 -> S1 2xMBConv -> 28x28x96 -> S2 3xMBConv-SE
-> 14x14x160 (+CPE) -> S3 4xTFSG (5 heads, 7x7 windows) -> 7x7x256 (+CPE) -> S4 3xTFSG (8 heads, 1 global 7x7
window) -> GAP -> Linear(256, 2). DropPath constant per stage: 0 / .05 / .15 / .20.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ocat_net_s.models.tfsg import WindowFreqAttention


@dataclass(frozen=True)
class OCATConfig:
    dims: tuple[int, ...] = (32, 64, 96, 160, 256)  # stem, S1, S2, S3, S4
    depths: tuple[int, ...] = (2, 3, 4, 3)
    heads: tuple[int, int] = (5, 8)
    window: int = 7
    drop_path: tuple[float, ...] = (0.0, 0.05, 0.15, 0.20)
    expansion: int = 4
    se_reduction: int = 4
    c_hf: int = 16
    mlp_ratio: int = 4  # paper text: C -> 4C -> C
    num_classes: int = 2
    # --- ablation / assumption switches ---
    late_block: str = "tfsg"  # tfsg | mhsa (gate off, same dims) | mbconv_se (CNN-only ASSUMPTION)
    se_in_stage2: bool = True
    cpe: bool = True
    cpe_mode: str = "stage"  # stage (assumption A-CPE1) | block
    gate_per_head: bool = True
    lam_init: float = 0.1
    lam_learnable: bool = True
    extras: dict = field(default_factory=dict, compare=False)

    def validate(self) -> None:
        if self.late_block not in ("tfsg", "mhsa", "mbconv_se"):
            raise ValueError("late_block must be tfsg|mhsa|mbconv_se")
        if self.cpe_mode not in ("stage", "block"):
            raise ValueError("cpe_mode must be stage|block")
        if len(self.dims) != 5 or len(self.depths) != 4 or len(self.drop_path) != 4:
            raise ValueError("dims(5)/depths(4)/drop_path(4) length mismatch")


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # NCHW, normalise over C
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(
            0, 3, 1, 2
        )


class SE(nn.Module):
    """GAP -> FC(rC -> rC/red) -> ReLU -> FC(rC/red -> rC) -> sigmoid. The paper's prose says W2 maps rC to rC/4,
    but Fig. 5 and the channel-wise rescale require rC/4 -> rC (typo resolved, see deviations D2)."""

    def __init__(self, ch: int, reduction: int):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction)
        self.fc2 = nn.Linear(ch // reduction, ch)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(v.mean(dim=(2, 3))))))
        return v * s[:, :, None, None]


class MBConv(nn.Module):
    def __init__(self, dim: int, r: int = 4, se_reduction: int | None = None, drop_path: float = 0.0):
        super().__init__()
        ch = dim * r
        self.norm = LayerNorm2d(dim)
        self.expand = nn.Conv2d(dim, ch, 1)
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.se = SE(ch, se_reduction) if se_reduction else nn.Identity()
        self.project = nn.Conv2d(ch, dim, 1)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = F.gelu(self.expand(self.norm(x)))
        v = F.gelu(self.dw(v))
        return x + self.drop_path(self.project(self.se(v)))


class CPE(nn.Module):
    """Convolutional positional encoding: residual 3x3 depthwise conv."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dw(x)


def window_partition(x: torch.Tensor, w: int) -> torch.Tensor:
    """(B,H,W,C) -> (B*nW, w*w, C)."""
    b, h, wd, c = x.shape
    if h % w or wd % w:
        raise ValueError(f"feature map {h}x{wd} not divisible by window {w} (OCAT-Net-S expects 224x224 input)")
    x = x.view(b, h // w, w, wd // w, w, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, w * w, c)


def window_reverse(win: torch.Tensor, w: int, h: int, wd: int) -> torch.Tensor:
    b = win.shape[0] // ((h // w) * (wd // w))
    x = win.view(b, h // w, wd // w, w, w, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, wd, -1)


class TFSGBlock(nn.Module):
    """CPE (optional, block mode) -> LN -> window attention w/ DCT gate -> +res -> LN -> MLP(4C) -> +res."""

    def __init__(self, dim, heads, window, drop_path, cfg: OCATConfig, use_gate: bool = True, block_cpe: bool = False):
        super().__init__()
        self.window = window
        self.cpe = CPE(dim) if block_cpe else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowFreqAttention(
            dim,
            heads,
            window,
            cfg.c_hf,
            use_gate=use_gate,
            gate_per_head=cfg.gate_per_head,
            lam_init=cfg.lam_init,
            lam_learnable=cfg.lam_learnable,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, cfg.mlp_ratio * dim), nn.GELU(), nn.Linear(cfg.mlp_ratio * dim, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # NCHW
        x = self.cpe(x)
        x = x.permute(0, 2, 3, 1)
        b, h, w, _ = x.shape
        y = window_reverse(self.attn(window_partition(self.norm1(x), self.window)), self.window, h, w)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2)


def _down(cin: int, cout: int) -> nn.Module:
    return nn.Conv2d(cin, cout, 3, stride=2, padding=1)


class OCATNetS(nn.Module):
    def __init__(self, cfg: OCATConfig = OCATConfig()):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d, dp = cfg.dims, cfg.drop_path
        self.stem = nn.Sequential(nn.Conv2d(3, d[0], 3, stride=2, padding=1), nn.BatchNorm2d(d[0]), nn.GELU())
        self.down1 = _down(d[0], d[1])
        self.stage1 = nn.Sequential(*[MBConv(d[1], cfg.expansion, None, dp[0]) for _ in range(cfg.depths[0])])
        self.down2 = _down(d[1], d[2])
        se = cfg.se_reduction if cfg.se_in_stage2 else None
        self.stage2 = nn.Sequential(*[MBConv(d[2], cfg.expansion, se, dp[1]) for _ in range(cfg.depths[1])])
        self.down3 = nn.Sequential(
            _down(d[2], d[3]), CPE(d[3]) if cfg.cpe and cfg.cpe_mode == "stage" else nn.Identity()
        )
        self.stage3 = self._late_stage(d[3], cfg.heads[0], cfg.depths[2], dp[2])
        self.down4 = nn.Sequential(
            _down(d[3], d[4]), CPE(d[4]) if cfg.cpe and cfg.cpe_mode == "stage" else nn.Identity()
        )
        self.stage4 = self._late_stage(d[4], cfg.heads[1], cfg.depths[3], dp[3])
        self.head = nn.Linear(d[4], cfg.num_classes)
        self.apply(self._init)

    def _late_stage(self, dim, heads, depth, dp) -> nn.Sequential:
        cfg = self.cfg
        if cfg.late_block == "mbconv_se":
            return nn.Sequential(*[MBConv(dim, cfg.expansion, cfg.se_reduction, dp) for _ in range(depth)])
        block_cpe = cfg.cpe and cfg.cpe_mode == "block"
        return nn.Sequential(
            *[TFSGBlock(dim, heads, cfg.window, dp, cfg, cfg.late_block == "tfsg", block_cpe) for _ in range(depth)]
        )

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = {}
        f["stem"] = self.stem(x)
        f["stage1"] = self.stage1(self.down1(f["stem"]))
        f["stage2"] = self.stage2(self.down2(f["stage1"]))
        f["stage3"] = self.stage3(self.down3(f["stage2"]))
        f["stage4"] = self.stage4(self.down4(f["stage3"]))
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)["stage4"].mean(dim=(2, 3)))


def param_group_name(param_name: str) -> str:
    """Map a parameter to its LLRD group (Table 4): stem | stage1..4 | head. Transition k joins stage k-1+1."""
    top = param_name.split(".")[0]
    return {
        "stem": "stem", "down1": "stage1", "stage1": "stage1", "down2": "stage2", "stage2": "stage2",
        "down3": "stage3", "stage3": "stage3", "down4": "stage4", "stage4": "stage4", "head": "head",
    }[top]  # fmt: skip
