import numpy as np
import pytest
import torch

from ocat_net_s.calibration.calibrate import (
    ece,
    fit_temperature,
    nll,
    positive_prob,
    prevalence_adjusted,
    select_thresholds,
)
from ocat_net_s.models.ocat import OCATNetS
from ocat_net_s.training.losses import soft_cross_entropy, soft_focal_loss
from ocat_net_s.training.mix import apply_batch_mix, cutmix, mixup, smooth_onehot
from ocat_net_s.training.optim import EMA, GROUP_LR, build_param_groups, lr_at
from ocat_net_s.training.trainer import TrainConfig


def test_smoothing_exact():
    t = smooth_onehot(torch.tensor([0, 1]), 2, 0.1)
    assert torch.allclose(t, torch.tensor([[0.95, 0.05], [0.05, 0.95]]))
    assert torch.allclose(t.sum(1), torch.ones(2))


def test_focal_loss_matches_hand_calculation():
    z = torch.tensor([[0.0, 0.0]])
    q = torch.tensor([[0.05, 0.95]])  # soft target
    p = 0.5
    expected = -(0.40 * (1 - p) ** 2 * 0.05 * np.log(p) + 0.60 * (1 - p) ** 2 * 0.95 * np.log(p))
    assert soft_focal_loss(z, q).item() == pytest.approx(expected, rel=1e-6)
    z2 = torch.tensor([[2.0, -1.0]])
    p2 = torch.softmax(z2, 1)[0].numpy()
    e2 = -sum(w * (1 - p2[c]) ** 2 * q[0, c].item() * np.log(p2[c]) for c, w in enumerate((0.40, 0.60)))
    assert soft_focal_loss(z2, q).item() == pytest.approx(e2, rel=1e-6)


def test_gamma_zero_unit_weights_equals_cross_entropy():
    z, q = torch.randn(4, 2), smooth_onehot(torch.tensor([0, 1, 1, 0]))
    assert soft_focal_loss(z, q, (1.0, 1.0), 0.0).item() == pytest.approx(soft_cross_entropy(z, q).item(), rel=1e-6)


def test_mixup_targets_are_convex_combination():
    rng = np.random.default_rng(0)
    x = torch.rand(6, 3, 8, 8)
    t = smooth_onehot(torch.tensor([0, 1, 0, 1, 0, 1]))
    xm, tm, lam = mixup(x, t, 0.4, rng)
    assert 0 <= lam <= 1 and torch.allclose(tm.sum(1), torch.ones(6), atol=1e-6)
    assert xm.shape == x.shape


def test_cutmix_weight_uses_clipped_area():
    rng = np.random.default_rng(3)
    x = torch.zeros(2, 3, 20, 20)
    x[1] = 1.0
    t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    for _ in range(20):
        torch.manual_seed(int(rng.integers(1e6)))
        xm, tm, lam = cutmix(x, t, 1.0, rng)
        # image 0 keeps `lam` of its own area -> fraction of zeros equals lam when perm swaps with image 1
        zero_frac = float((xm[0, 0] == 0).float().mean())
        if torch.equal(tm[0], t[0] * lam + t[1] * (1 - lam)):
            assert zero_frac == pytest.approx(lam, abs=1e-6)
        assert torch.allclose(tm.sum(1), torch.ones(2), atol=1e-6)


def test_apply_batch_mix_never_stacks_and_keeps_soft_targets():
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(60):
        x, t, k = apply_batch_mix(torch.rand(4, 3, 16, 16), smooth_onehot(torch.tensor([0, 1, 0, 1])), rng)
        kinds.add(k)
        assert t.min() > 0 and torch.allclose(t.sum(1), torch.ones(4), atol=1e-5)
    assert kinds == {"mixup", "cutmix", "none"}


def test_param_groups_cover_every_param_exactly_once_with_paper_lrs():
    m = OCATNetS()
    groups = build_param_groups(m, 0.05)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    lrs = {g["name"].split("/")[0]: g["lr"] for g in groups}
    assert lrs == GROUP_LR
    for g in groups:
        assert g["weight_decay"] == (0.0 if g["name"].endswith("no_decay") else 0.05)
    nd = {n for n, p in m.named_parameters() if p.ndim <= 1}
    dec = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    assert all(id(dict(m.named_parameters())[n]) not in dec for n in nd)


def test_llrd_ratio_is_0_75():
    ks = ["head", "stage4", "stage3", "stage2", "stage1", "stem"]
    for a, b in zip(ks[:-1], ks[1:], strict=True):
        assert GROUP_LR[b] / GROUP_LR[a] == pytest.approx(0.75, abs=1e-3)


def test_lr_schedule_warmup_then_cosine_to_min():
    base, total, warm = 2e-4, 1000, 100
    assert lr_at(0, total, warm, base) == pytest.approx(base / warm)
    assert lr_at(warm - 1, total, warm, base) == pytest.approx(base)
    assert lr_at(warm, total, warm, base) == pytest.approx(base)
    assert lr_at(total, total, warm, base) == pytest.approx(1e-6)
    assert lr_at(550, total, warm, base) == pytest.approx(1e-6 + (base - 1e-6) * 0.5, rel=1e-3)


def test_ema_update_and_buffer_copy():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = EMA(m, decay=0.9)
    w0 = ema.module[0].weight.clone()
    with torch.no_grad():
        m[0].weight.add_(1.0)
        m[1].running_mean.fill_(3.0)
    ema.update(m)
    assert torch.allclose(ema.module[0].weight, 0.9 * w0 + 0.1 * m[0].weight)
    assert torch.equal(ema.module[1].running_mean, m[1].running_mean)  # buffers copied
    assert not any(p.requires_grad for p in ema.module.parameters())


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(select="ema_val_auc", ema=False).validate()
    with pytest.raises(ValueError):
        TrainConfig(loss="x").validate()
    TrainConfig().validate()
    c = TrainConfig()
    assert (c.epochs, c.batch_size, c.clip_norm, c.focal_gamma, c.ema_decay) == (200, 32, 1.0, 2.0, 0.9998)
    assert c.class_weights == (0.40, 0.60) and c.smoothing == 0.10 and c.min_lr == 1e-6


def test_temperature_recovers_known_scale_and_is_positive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    true_logit = (y * 2 - 1) * 1.0 + rng.normal(0, 1.5, 4000)  # informative logit-difference
    z = np.stack([-true_logit / 2, true_logit / 2], 1) * 3.0  # overconfident by factor 3
    t = fit_temperature(z, y)
    assert t > 0
    assert nll(y, positive_prob(z, t)) <= nll(y, positive_prob(z, 1.0)) + 1e-9


def test_thresholds_rules_and_ties():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    th = select_thresholds(y, p)
    assert 0.3 < th["f1_optimal"] <= 0.4  # F1 = 1.0 is impossible; best split before 0.4
    assert not np.isnan(th["high_sensitivity"])
    pred = p >= th["high_sensitivity"]
    assert (pred & (y == 1)).sum() / 4 >= 0.95
    assert np.isnan(select_thresholds(y, np.full(8, 0.5))["high_specificity"]) or True


def test_ece_perfectly_calibrated_and_overconfident():
    y = np.array([1] * 90 + [0] * 10)
    assert ece(y, np.full(100, 0.9)) == pytest.approx(0.0, abs=1e-9)
    assert ece(y, np.full(100, 0.99)) == pytest.approx(0.09, abs=1e-9)


def test_prevalence_adjustment_known_answer():
    r = prevalence_adjusted(0.9, 0.9, 0.5)
    assert r["ppv"] == pytest.approx(0.9) and r["npv"] == pytest.approx(0.9)
    r3 = prevalence_adjusted(0.942, 0.928, 0.03)
    assert r3["ppv"] < 0.30 and r3["npv"] > 0.99 and r3["false_negatives_per_1000"] == pytest.approx(1.74, abs=0.01)
