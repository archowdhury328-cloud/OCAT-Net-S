"""Training loop: AdamW + LLRD, warm-up/cosine, focal loss on soft targets, MixUp/CutMix, EMA, AMP, resume.

Live-model and EMA validation are both logged and labelled; checkpoint selection uses ``select`` (default EMA val AUC,
assumption A-TR1). The outer test fold is never touched here.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ocat_net_s.training.losses import soft_cross_entropy, soft_focal_loss
from ocat_net_s.training.mix import apply_batch_mix, smooth_onehot
from ocat_net_s.training.optim import EMA, build_param_groups, set_lrs


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    clip_norm: float = 1.0
    smoothing: float = 0.10
    loss: str = "focal"  # focal | ce
    focal_gamma: float = 2.0
    class_weights: tuple[float, float] = (0.40, 0.60)
    mix: bool = True
    ema: bool = True
    ema_decay: float = 0.9998
    amp: bool = False
    select: str = "ema_val_auc"  # ema_val_auc | live_val_auc
    num_workers: int = 0
    seed: int = 42
    groups_lr: bool = True  # False -> single lr (used for baselines) with ``base_lr``
    base_lr: float = 2e-4

    def validate(self) -> None:
        if self.loss not in ("focal", "ce"):
            raise ValueError("loss must be focal|ce")
        if self.select not in ("ema_val_auc", "live_val_auc"):
            raise ValueError("select must be ema_val_auc|live_val_auc")
        if self.select == "ema_val_auc" and not self.ema:
            raise ValueError("select=ema_val_auc requires ema=True")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    ids, ys, zs = [], [], []
    for b in loader:
        zs.append(model(b["image"].to(device)).float().cpu())
        ys.append(b["label"])
        ids.extend(b["image_id"])
    return ids, torch.cat(ys).numpy(), torch.cat(zs).numpy()


def _val_auc(model, loader, device) -> float:
    _, y, z = predict_logits(model, loader, device)
    return float(roc_auc_score(y, z[:, 1] - z[:, 0])) if len(np.unique(y)) > 1 else float("nan")


def fit(model: nn.Module, train_ds, val_loader: DataLoader, cfg: TrainConfig, out_dir, device="cpu", resume=True,
        meta: dict | None = None) -> list[dict]:  # fmt: skip
    cfg.validate()
    device = torch.device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.to(device)
    if cfg.groups_lr:
        groups = build_param_groups(model, cfg.weight_decay)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.base_lr,
                   "base_lr": cfg.base_lr, "weight_decay": cfg.weight_decay, "name": "all"}]  # fmt: skip
    opt = torch.optim.AdamW(groups, betas=cfg.betas, eps=cfg.eps)
    ema = EMA(model, cfg.ema_decay) if cfg.ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g, drop_last=False,
                        num_workers=cfg.num_workers)  # fmt: skip
    steps_per_epoch = len(loader)  # ACTUAL optimisation-train size (not the paper's 31), see deviations E
    total, warm = steps_per_epoch * cfg.epochs, steps_per_epoch * cfg.warmup_epochs
    history, start_epoch, step, best = [], 0, 0, -np.inf
    last = out / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)  # our own trusted checkpoint
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if ema:
            ema.module.load_state_dict(ck["ema"])
        history, start_epoch, step, best = ck["history"], ck["epoch"] + 1, ck["step"], ck["best"]
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        random.setstate(ck["py_rng"])
    for epoch in range(start_epoch, cfg.epochs):
        t0, run, nb, mixes = time.time(), 0.0, 0, {"mixup": 0, "cutmix": 0, "none": 0}
        model.train()
        for b in loader:
            x, y = b["image"].to(device), b["label"].to(device)
            t = smooth_onehot(y, 2, cfg.smoothing)  # smoothing applied exactly once
            if cfg.mix:
                x, t, kind = apply_batch_mix(x, t, rng)
                mixes[kind] += 1
            set_lrs(opt, step, total, warm, cfg.min_lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                z = model(x)
            if cfg.loss == "focal":
                loss = soft_focal_loss(z, t, cfg.class_weights, cfg.focal_gamma)
            else:
                loss = soft_cross_entropy(z, t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # clip AFTER unscale
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            run += float(loss.detach())
            nb += 1
            step += 1
        rec = {"epoch": epoch, "train_loss": run / max(nb, 1), "live_val_auc": _val_auc(model, val_loader, device),
               "lrs": {gr["name"]: gr["lr"] for gr in opt.param_groups}, "mix_batches": mixes,
               "steps_per_epoch": steps_per_epoch, "seconds": round(time.time() - t0, 2)}  # fmt: skip
        if ema:
            rec["ema_val_auc"] = _val_auc(ema.module, val_loader, device)
        score = rec[cfg.select]
        rec["selected_as_best"] = bool(np.isnan(best) or score > best or best == -np.inf)
        if rec["selected_as_best"]:
            best = score
            torch.save({"model": (ema.module if cfg.select == "ema_val_auc" else model).state_dict(),
                        "epoch": epoch, "metric": cfg.select, "value": best, "meta": meta or {}}, out / "best.pt")  # fmt: skip
        history.append(rec)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.module.state_dict() if ema else None, "history": history, "epoch": epoch,
                    "step": step, "best": best, "cfg": asdict(cfg), "np_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "py_rng": random.getstate(), "meta": meta or {}}, last)  # fmt: skip
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    return history


def load_best(model: nn.Module, out_dir, device="cpu") -> dict:
    ck = torch.load(Path(out_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return {k: ck[k] for k in ("epoch", "metric", "value", "meta")}
