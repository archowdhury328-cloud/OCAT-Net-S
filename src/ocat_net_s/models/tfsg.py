"""Texture-Frequency Spatial Gating (TFSG) attention, per paper Eq. for A_w.

A_w = Softmax( (Q K^T / sqrt(d_k)) * (1 + lambda * G_w) + B ) V ,  G_w[p,q] = sigmoid(e_p + e_q)

Everything the paper leaves open is an explicit, configurable ASSUMPTION (docs/assumptions_and_deviations.md):
  T1 DCT: orthonormal DCT-II along the CHANNEL axis of each (layer-normed) window token.
  T2 High-frequency set: the C_HF highest-index coefficients (k = C-C_HF .. C-1), magnitude |.|.
  T3 e = W_C(|DCT_hf|), W_C: Linear(C_HF -> heads) (gate per head; ``gate_per_head=False`` shares one gate).
  T4 sigmoid is applied to the pairwise sum (e_p + e_q), as in the text (the displayed equation omits it).
  T5 lambda: learnable scalar, init ``lam_init`` (0.1), unconstrained sign; ``lam_learnable=False`` fixes it.
  T6 Relative position bias: Swin-style learned table over (2w-1)^2 offsets, per head.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def dct_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n x n): y = D @ x  (matches scipy.fft.dct(norm='ortho'))."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    d[0] = d[0] / math.sqrt(2.0)
    return d.to(dtype)


def dct_reference(x: torch.Tensor) -> torch.Tensor:
    """Slow explicit-sum orthonormal DCT-II over the last axis (for tests only)."""
    n = x.shape[-1]
    out = torch.zeros_like(x, dtype=torch.float64)
    xd = x.to(torch.float64)
    for k in range(n):
        s = sum(xd[..., i] * math.cos(math.pi * (2 * i + 1) * k / (2 * n)) for i in range(n))
        out[..., k] = s * math.sqrt((1.0 if k else 0.5) * 2.0 / n)
    return out.to(x.dtype)


class FreqGate(nn.Module):
    """x (Bw, n, C) -> pairwise gate G (Bw, h|1, n, n) in (0,1)."""

    def __init__(self, dim: int, heads: int, c_hf: int = 16, per_head: bool = True):
        super().__init__()
        if c_hf > dim:
            raise ValueError(f"c_hf={c_hf} > channels={dim}")
        self.c_hf = c_hf
        self.register_buffer("dct", dct_matrix(dim), persistent=False)
        self.w_c = nn.Linear(c_hf, heads if per_head else 1)

    def descriptor(self, x: torch.Tensor) -> torch.Tensor:
        coeff = x @ self.dct.to(x.dtype).T  # (Bw, n, C): DCT-II along channels
        return coeff[..., -self.c_hf :].abs()  # (Bw, n, C_HF) high-frequency magnitudes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.w_c(self.descriptor(x)).permute(0, 2, 1)  # (Bw, h|1, n)
        return torch.sigmoid(e.unsqueeze(-1) + e.unsqueeze(-2))  # (Bw, h|1, n, n)


class WindowFreqAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window: int,
        c_hf: int = 16,
        use_gate: bool = True,
        gate_per_head: bool = True,
        lam_init: float = 0.1,
        lam_learnable: bool = True,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.dim, self.heads, self.window = dim, heads, window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * window - 1) + rel[..., 1], persistent=False)
        self.use_gate = use_gate
        self.gate = FreqGate(dim, heads, c_hf, gate_per_head) if use_gate else None
        lam = torch.tensor(float(lam_init))
        if lam_learnable:
            self.lam = nn.Parameter(lam)
        else:
            self.register_buffer("lam", lam)

    def rel_bias(self) -> torch.Tensor:
        n = self.window**2
        return self.rel_table[self.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1)  # (h, n, n)

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        bw, n, c = x.shape
        q, k, v = self.qkv(x).view(bw, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-2, -1)) * self.scale  # (Bw, h, n, n)
        gate = None
        if self.use_gate:
            gate = self.gate(x)
            logits = logits * (1 + self.lam * gate)
        logits = logits + self.rel_bias().unsqueeze(0)
        attn = logits.softmax(dim=-1)
        out = self.proj((attn @ v).transpose(1, 2).reshape(bw, n, c))
        return (out, {"attn": attn, "gate": gate}) if return_parts else out


def attention_reference(att: WindowFreqAttention, x: torch.Tensor) -> torch.Tensor:
    """Loop-based float64 reference of the same computation (small tensors, tests only)."""
    bw, n, c = x.shape
    h, d = att.heads, c // att.heads
    xd = x.double()
    W = att.qkv.weight.double()
    bq = att.qkv.bias.double()
    out = torch.zeros(bw, n, c, dtype=torch.float64)
    bias = att.rel_bias().double()
    for b in range(bw):
        qkv = xd[b] @ W.T + bq
        heads_out = []
        gate_e = None
        if att.use_gate:
            D = dct_matrix(c, torch.float64)
            hf = torch.stack([(xd[b][t] @ D.T)[-att.gate.c_hf :].abs() for t in range(n)])
            gate_e = hf @ att.gate.w_c.weight.double().T + att.gate.w_c.bias.double()  # (n, h|1)
        for hh in range(h):
            q, k, v = (qkv[:, j * c + hh * d : j * c + (hh + 1) * d] for j in range(3))
            logit = torch.zeros(n, n, dtype=torch.float64)
            for p in range(n):
                for r in range(n):
                    s = (q[p] * k[r]).sum() * att.scale
                    if att.use_gate:
                        ee = gate_e[:, hh if gate_e.shape[1] > 1 else 0]
                        s = s * (1 + att.lam.double() * torch.sigmoid(ee[p] + ee[r]))
                    logit[p, r] = s + bias[hh, p, r]
            heads_out.append(torch.softmax(logit, dim=-1) @ v)
        out[b] = torch.cat(heads_out, dim=-1) @ att.proj.weight.double().T + att.proj.bias.double()
    return out.to(x.dtype)
