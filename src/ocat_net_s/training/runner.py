"""One experiment unit = (model/ablation, outer fold, seed). Order inside a unit:
inner group-disjoint val split -> norm stats from optimisation-train only -> train -> select checkpoint on VAL ->
fit temperature + thresholds on VAL -> predict VAL and TEST once -> save sample-level predictions + metrics.
The outer test fold is only read at the very end for prediction (never for any fitting/selection).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ocat_net_s.calibration.calibrate import brier, ece, fit_temperature, nll, positive_prob, select_thresholds
from ocat_net_s.data.dataset import ImageDataset, pil_images
from ocat_net_s.data.manifest import folds_hash, inner_val_split, verify_folds
from ocat_net_s.data.transforms import build_eval_transform, build_train_transform, compute_norm_stats
from ocat_net_s.evaluation.metrics import binary_metrics
from ocat_net_s.models.registry import ABLATIONS, TRAINING_ABLATIONS, build_model
from ocat_net_s.training.trainer import TrainConfig, fit, load_best, predict_logits
from ocat_net_s.utils.common import environment_record, write_json


@dataclass
class RunSpec:
    manifest_csv: str
    data_root: str
    out_dir: str
    model: str = "ocat"  # ocat | baseline key | convnext_b
    ablation: str = "full"  # only for model == ocat
    fold: int = 0
    seed: int = 42
    inner_val_frac: float = 0.10  # ASSUMPTION A-D2
    pretrained: bool = False  # OCAT: no checkpoint exists; baselines: set True for ImageNet init
    device: str = "cpu"
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)

    @property
    def run_id(self) -> str:
        tag = self.ablation if self.model == "ocat" else "base"
        return f"{self.model}-{tag}-f{self.fold}-s{self.seed}"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _loader(ds, bs, nw=0):
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)


def run_unit(spec: RunSpec) -> dict:
    df = pd.read_csv(spec.manifest_csv, keep_default_na=False)
    verify_folds(df)
    test_df = df[df.fold == spec.fold]
    opt_train, val = inner_val_split(df[df.fold != spec.fold], spec.inner_val_frac, seed=spec.fold)
    if set(opt_train.group_id) & set(val.group_id) or set(test_df.group_id) & (
        set(opt_train.group_id) | set(val.group_id)
    ):
        raise RuntimeError("group leakage between optimisation-train / validation / test")
    out = Path(spec.out_dir) / spec.run_id
    mean, std = compute_norm_stats(pil_images(opt_train, spec.data_root))
    ev_t, tr_t = build_eval_transform(mean, std), build_train_transform(mean, std)
    train_ds = ImageDataset(opt_train, spec.data_root, tr_t)
    val_loader = _loader(ImageDataset(val, spec.data_root, ev_t), spec.train.batch_size, spec.train.num_workers)
    if spec.model == "ocat":
        tcfg = dataclasses.replace(spec.train, **TRAINING_ABLATIONS.get(spec.ablation, {}))
        net = build_model("ocat", ablation=spec.ablation if spec.ablation in ABLATIONS else "full")
    else:
        tcfg = dataclasses.replace(spec.train, groups_lr=False)  # baselines: single lr, same schedule (assumption)
        net = build_model(spec.model, pretrained=spec.pretrained)
    meta = {"folds_hash": folds_hash(df), "norm_mean": mean, "norm_std": std, "git": _git_commit(),
            "model": spec.model, "ablation": spec.ablation, "fold": spec.fold, "seed": spec.seed}  # fmt: skip
    write_json(out / "run_config.json", {"spec": dataclasses.asdict(spec), "train": dataclasses.asdict(tcfg),
                                         "meta": meta, "env": environment_record(),
                                         "n": {"opt_train": len(opt_train), "val": len(val), "test": len(test_df)}})  # fmt: skip
    dev = torch.device(spec.device)
    fit(net, train_ds, val_loader, tcfg, out, dev, meta=meta)
    best = load_best(net, out, dev)
    net.to(dev).eval()
    v_ids, v_y, v_z = predict_logits(net, val_loader, dev)
    temp = fit_temperature(v_z, v_y)
    v_p = positive_prob(v_z, temp)
    thr = select_thresholds(v_y, v_p)
    te_loader = _loader(ImageDataset(test_df, spec.data_root, ev_t), spec.train.batch_size, spec.train.num_workers)
    t_ids, t_y, t_z = predict_logits(net, te_loader, dev)
    t_p = positive_prob(t_z, temp)
    pred = pd.DataFrame({"image_id": t_ids, "y_true": t_y, "logit_normal": t_z[:, 0], "logit_cancer": t_z[:, 1],
                         "prob_cancer": t_p, "model": spec.run_id, "fold": spec.fold, "seed": spec.seed,
                         "checkpoint": f"best@epoch{best['epoch']}:{best['metric']}", "temperature": temp,
                         "threshold_f1": thr["f1_optimal"]})  # fmt: skip
    pred["pred_f1"] = (pred.prob_cancer >= thr["f1_optimal"]).astype(int)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions" / "test.csv", index=False)
    res = {"run_id": spec.run_id, "temperature": temp, "thresholds": thr,
           "test": {k: binary_metrics(t_y, t_p, v) for k, v in thr.items() if not np.isnan(v)},
           "test_calibration": {"brier": brier(t_y, t_p), "ece15": ece(t_y, t_p), "nll": nll(t_y, t_p)},
           "val_calibration": {"brier": brier(v_y, v_p), "ece15": ece(v_y, v_p), "nll": nll(v_y, v_p)}, "best": best}  # fmt: skip
    write_json(out / "metrics.json", res)
    if spec.model == "ocat":
        from ocat_net_s.inference.bundle import save_bundle

        save_bundle(
            out / "bundle",
            net.cpu(),
            mean,
            std,
            temp,
            thr["f1_optimal"],
            {"folds_hash": meta["folds_hash"], "git": meta["git"], "fold": spec.fold, "seed": spec.seed},
        )
    (out / "DONE").write_text(json.dumps({"git": meta["git"]}))
    return res


def make_grid(models: list[tuple[str, str]], folds=range(5), seeds=(42, 123, 2024, 7, 1337)) -> list[dict]:
    """Experiment grid [(model, ablation)] x folds x seeds. Proposed + 10 baselines = 275; 8 ablations = 200 more."""
    return [{"model": m, "ablation": a, "fold": f, "seed": s, "run_id": f"{m}-{a if m == 'ocat' else 'base'}-f{f}-s{s}"}
            for m, a in models for f in folds for s in seeds]  # fmt: skip


def status(grid: list[dict], runs_dir: str) -> pd.DataFrame:
    g = pd.DataFrame(grid)
    g["done"] = [(Path(runs_dir) / r / "DONE").exists() for r in g.run_id]
    return g
