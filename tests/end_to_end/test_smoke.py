"""End-to-end on SYNTHETIC images (software correctness only; metrics are meaningless)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ocat_net_s.data.dataset import ImageDataset
from ocat_net_s.data.manifest import assign_outer_folds, assign_proxy_groups, build_manifest
from ocat_net_s.data.transforms import build_eval_transform, build_train_transform
from ocat_net_s.explainability.gradcam import gradcam
from ocat_net_s.inference.bundle import load_bundle, predict_files
from ocat_net_s.models.ocat import OCATConfig, OCATNetS
from ocat_net_s.training.runner import RunSpec, make_grid, run_unit, status
from ocat_net_s.training.trainer import TrainConfig, fit
from ocat_net_s.utils.synthetic import make_synthetic_dataset


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    t = tmp_path_factory.mktemp("e2e")
    make_synthetic_dataset(t / "d", 30)
    df, _ = build_manifest(t / "d")
    assign_outer_folds(assign_proxy_groups(df, 3)).to_csv(t / "m.csv", index=False)
    return t


@pytest.fixture(scope="module")
def run(env):
    tc = TrainConfig(epochs=2, batch_size=8, warmup_epochs=1)
    spec = RunSpec(str(env / "m.csv"), str(env / "d"), str(env / "o"), train=tc, fold=1, seed=7)
    return spec, run_unit(spec)


def test_run_unit_outputs_and_no_test_fold_leakage(env, run):
    spec, res = run
    out = env / "o" / spec.run_id
    cfg = json.loads((out / "run_config.json").read_text())
    n = cfg["n"]
    assert n["opt_train"] + n["val"] + n["test"] == 60
    pred = pd.read_csv(out / "predictions" / "test.csv")
    assert {
        "image_id",
        "y_true",
        "logit_cancer",
        "prob_cancer",
        "model",
        "fold",
        "seed",
        "checkpoint",
        "temperature",
        "threshold_f1",
    } <= set(pred.columns)
    man = pd.read_csv(env / "m.csv")
    assert set(pred.image_id) == set(man[man.fold == 1].image_id)  # exactly the outer test fold
    assert res["temperature"] > 0 and 0 <= res["test_calibration"]["ece15"] <= 1
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and all({"live_val_auc", "ema_val_auc", "lrs"} <= set(h) for h in hist)
    assert (out / "DONE").exists()


def test_metrics_reproducible_from_saved_predictions(env, run):
    from ocat_net_s.evaluation.metrics import binary_metrics

    spec, res = run
    pred = pd.read_csv(env / "o" / spec.run_id / "predictions" / "test.csv")
    thr = res["thresholds"]["f1_optimal"]
    m = binary_metrics(pred.y_true, pred.prob_cancer, thr)
    assert m["accuracy"] == pytest.approx(res["test"]["f1_optimal"]["accuracy"])


def test_bundle_roundtrip_equals_pipeline_probability(env, run):
    spec, _ = run
    out = env / "o" / spec.run_id
    b = load_bundle(out / "bundle")
    man = pd.read_csv(env / "m.csv")
    row = man[man.fold == 1].iloc[0]
    r = predict_files(b, [env / "d" / row.rel_path])[0]
    pred = pd.read_csv(out / "predictions" / "test.csv").set_index("image_id")
    assert r["prob_oral_cancer"] == pytest.approx(pred.loc[row.image_id, "prob_cancer"], abs=1e-4)
    assert r["predicted_class"] in ("Normal", "Oral Cancer") and "not a diagnostic" in r["note"]


def test_bundle_tamper_and_bad_inputs(env, run, tmp_path):
    spec, _ = run
    import shutil

    shutil.copytree(env / "o" / spec.run_id / "bundle", tmp_path / "b")
    b = load_bundle(tmp_path / "b")
    txt = tmp_path / "x.txt"
    txt.write_text("no")
    with pytest.raises(ValueError):
        predict_files(b, [txt])
    w = tmp_path / "b" / "weights.pt"
    w.write_bytes(w.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "b")


def test_resume_continues_and_is_idempotent(env):
    df = pd.read_csv(env / "m.csv", keep_default_na=False)
    tr = ImageDataset(df[df.fold != 0], str(env / "d"), build_train_transform((0.5,) * 3, (0.25,) * 3))
    va = DataLoader(
        ImageDataset(df[df.fold == 0], str(env / "d"), build_eval_transform((0.5,) * 3, (0.25,) * 3)), batch_size=8
    )
    small = OCATConfig(dims=(8, 8, 16, 40, 64), depths=(1, 1, 1, 1), heads=(5, 8), c_hf=8)
    out = env / "resume"
    cfg = TrainConfig(epochs=1, batch_size=16, warmup_epochs=0)
    fit(OCATNetS(small), tr, va, cfg, out)
    h = fit(OCATNetS(small), tr, va, TrainConfig(epochs=2, batch_size=16, warmup_epochs=0), out)
    assert [r["epoch"] for r in h] == [0, 1]
    assert len((out / "history.jsonl").read_text().splitlines()) == 2
    ck = torch.load(out / "last.pt", weights_only=False)
    assert {"model", "opt", "scaler", "ema", "np_rng", "torch_rng", "meta", "step"} <= set(ck) and ck["step"] > 0


def test_gradcam_on_ocat_stage4():
    m = OCATNetS(OCATConfig(dims=(8, 8, 16, 40, 64), depths=(1, 1, 1, 1))).eval()
    cam, aux = gradcam(m, m.stage4, torch.rand(2, 3, 224, 224))
    assert cam.shape == (2, 1, 224, 224) and torch.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    assert aux["raw"].shape[-2:] == (7, 7)


def test_grid_counts_match_workload_estimate(tmp_path):
    from ocat_net_s.models.registry import BASELINES

    primary = make_grid([("ocat", "full")] + [(b, "base") for b in BASELINES])
    assert len(primary) == 275
    abl = make_grid(
        [
            ("ocat", k)
            for k in (
                "cnn_only",
                "mhsa",
                "no_se",
                "no_cpe",
                "no_mixup_cutmix",
                "ce_loss",
                "no_label_smoothing",
                "no_ema",
            )
        ]
    )
    assert len(abl) == 200 and len({g["run_id"] for g in primary + abl}) == 475
    assert not status(primary, str(tmp_path)).done.any()
    assert np.isfinite(1.0)
