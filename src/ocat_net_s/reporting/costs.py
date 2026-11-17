"""Measured parameters, MACs (fvcore), serialized size, latency/throughput. Never copies published numbers."""

from __future__ import annotations

import io
import time

import numpy as np
import torch
from torch import nn


def param_counts(m: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in m.parameters()),
        "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
    }


def stage_report(model: nn.Module, size: int = 224) -> list[dict]:
    """Per-stage output shapes and parameter counts from a REAL forward pass (OCAT-Net-S)."""
    model.eval()
    with torch.no_grad():
        feats = model.forward_features(torch.zeros(1, 3, size, size))
    rows = []
    for name, f in feats.items():
        mods = [name] if name in ("stem",) else [name]
        n = sum(p.numel() for p in getattr(model, mods[0]).parameters())
        rows.append({"stage": name, "output_shape": list(f.shape[1:]), "params": n})
    return rows


def macs_fvcore(model: nn.Module, size: int = 224) -> dict:
    """fvcore: 1 MAC = 1 flop. Unsupported ops (listed) are NOT counted -- e.g. softmax, GELU, DCT einsum variants."""
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    fa = FlopCountAnalysis(model.cpu(), torch.zeros(1, 3, size, size))
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return {
        "macs_g": fa.total() / 1e9,
        "convention": "fvcore, 1 MAC = 1 flop",
        "unsupported_ops": dict(fa.unsupported_ops()),
    }


def serialized_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2**20


@torch.no_grad()
def latency(model: nn.Module, device="cpu", batch=1, size=224, warmup=5, iters=30) -> dict:
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.rand(batch, 3, size, size, device=dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        model(x)
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    r = {"device": str(dev), "batch": batch, "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
         "ms_per_image": float(np.median(a) / batch), "iters": iters, "precision": "fp32"}  # fmt: skip
    if dev.type == "cuda":
        r["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return r
