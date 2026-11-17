import dataclasses

import numpy as np
import pytest
import scipy.fft
import torch

from ocat_net_s.models.ocat import (
    OCATConfig,
    OCATNetS,
    TFSGBlock,
    window_partition,
    window_reverse,
)
from ocat_net_s.models.registry import ABLATIONS, build_model, load_partial_checkpoint
from ocat_net_s.models.tfsg import WindowFreqAttention, attention_reference, dct_matrix, dct_reference
from ocat_net_s.reporting import costs


@pytest.fixture(scope="module")
def model():
    return OCATNetS().eval()


def test_stage_shapes_match_paper_table3(model):
    f = model.forward_features(torch.rand(2, 3, 224, 224))
    assert tuple(f["stem"].shape[1:]) == (32, 112, 112)
    assert tuple(f["stage1"].shape[1:]) == (64, 56, 56)
    assert tuple(f["stage2"].shape[1:]) == (96, 28, 28)
    assert tuple(f["stage3"].shape[1:]) == (160, 14, 14)
    assert tuple(f["stage4"].shape[1:]) == (256, 7, 7)
    assert model(torch.rand(2, 3, 224, 224)).shape == (2, 2)


def test_block_counts_heads_windows_droppath(model):
    assert [len(s) for s in (model.stage1, model.stage2, model.stage3, model.stage4)] == [2, 3, 4, 3]
    assert all(isinstance(b, TFSGBlock) for b in list(model.stage3) + list(model.stage4))
    assert model.stage3[0].attn.heads == 5 and model.stage4[0].attn.heads == 8
    assert model.stage3[0].attn.window == 7 and model.stage4[0].attn.window == 7
    assert model.stage1[0].drop_path.p == 0.0 and model.stage2[0].drop_path.p == 0.05
    assert model.stage3[0].drop_path.p == 0.15 and model.stage4[0].drop_path.p == 0.20


def test_se_restores_expanded_channels(model):
    se = model.stage2[0].se
    assert se.fc1.in_features == 96 * 4 and se.fc1.out_features == 96
    assert se.fc2.in_features == 96 and se.fc2.out_features == 96 * 4


def test_non_224_input_raises_clear_error(model):
    with pytest.raises(ValueError, match="window"):
        model(torch.rand(1, 3, 192, 192))


def test_window_roundtrip():
    x = torch.rand(2, 14, 14, 8)
    assert torch.equal(window_reverse(window_partition(x, 7), 7, 14, 14), x)


def test_dct_matrix_matches_scipy_and_reference():
    x = torch.randn(3, 5, 12, dtype=torch.float64)
    ref = torch.tensor(scipy.fft.dct(x.numpy(), type=2, norm="ortho", axis=-1))
    assert torch.allclose(x @ dct_matrix(12, torch.float64).T, ref, atol=1e-10)
    assert torch.allclose(dct_reference(x), ref, atol=1e-10)
    d = dct_matrix(16, torch.float64)
    assert torch.allclose(d @ d.T, torch.eye(16, dtype=torch.float64), atol=1e-10)  # orthonormal


@pytest.mark.parametrize("per_head", [True, False])
def test_optimized_attention_matches_loop_reference(per_head):
    torch.manual_seed(0)
    att = WindowFreqAttention(8, 2, 3, c_hf=4, gate_per_head=per_head, lam_init=0.7)
    x = torch.randn(2, 9, 8)
    assert torch.allclose(att(x), attention_reference(att, x), atol=1e-4)


def test_lambda_zero_equals_ungated_attention():
    torch.manual_seed(1)
    gated = WindowFreqAttention(8, 2, 3, c_hf=4, lam_init=0.0, lam_learnable=False)
    plain = WindowFreqAttention(8, 2, 3, use_gate=False)
    plain.load_state_dict({k: v for k, v in gated.state_dict().items() if "gate" not in k}, strict=False)
    x = torch.randn(2, 9, 8)
    assert torch.allclose(gated(x), plain(x), atol=1e-6)


def test_gate_changes_output_and_gradients_flow():
    torch.manual_seed(2)
    att = WindowFreqAttention(8, 2, 3, c_hf=4, lam_init=0.5)
    x = torch.randn(2, 9, 8, requires_grad=True)
    out, parts = att(x, return_parts=True)
    assert parts["attn"].shape == (2, 2, 9, 9) and parts["gate"].shape == (2, 2, 9, 9)
    assert torch.allclose(parts["attn"].sum(-1), torch.ones(2, 2, 9), atol=1e-5)
    assert ((parts["gate"] > 0) & (parts["gate"] < 1)).all()
    out.sum().backward()
    assert x.grad.abs().sum() > 0
    assert att.gate.w_c.weight.grad.abs().sum() > 0  # gate path not detached
    assert att.lam.grad is not None and att.lam.grad.abs() > 0
    assert torch.isfinite(out).all()


def test_ablation_param_counts_and_configs():
    base = sum(p.numel() for p in build_model("ocat").parameters())
    counts = {k: sum(p.numel() for p in build_model("ocat", ablation=k).parameters()) for k in ABLATIONS}
    assert counts["full"] == base
    assert counts["no_se"] < base and counts["no_cpe"] < base
    assert abs(counts["mhsa"] - base) < 5000  # only gate projection + lambda removed
    assert counts["cnn_only"] != base
    cfg = OCATConfig()
    assert dataclasses.replace(cfg, mlp_ratio=2) != cfg


def test_mlp_ratio_documented_alternative_is_near_reported_size():
    n = sum(p.numel() for p in OCATNetS(OCATConfig(mlp_ratio=2)).parameters()) / 1e6
    assert 3.4 < n < 3.7  # evidence only; default stays 4 per the paper's text


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        OCATNetS(OCATConfig(late_block="bad"))


def test_stage_report_and_costs(model):
    rows = costs.stage_report(model)
    assert [r["stage"] for r in rows] == ["stem", "stage1", "stage2", "stage3", "stage4"]
    assert costs.param_counts(model)["total"] > 1_000_000
    assert costs.latency(OCATNetS(), batch=1, warmup=0, iters=2)["median_ms"] > 0


def test_partial_checkpoint_logs_all_key_classes():
    m = build_model("ocat", ablation="no_cpe")
    sd = build_model("ocat").state_dict()
    sd["stem.0.weight"] = torch.zeros(5, 3, 3, 3)  # shape mismatch
    sd["bogus.key"] = torch.zeros(1)
    rep = load_partial_checkpoint(m, sd)
    assert "bogus.key" in rep["unexpected"] and any("stem.0.weight" in s for s in rep["shape_mismatched"])
    assert "stem.0.weight" in rep["missing"] and len(rep["loaded"]) > 100


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("nope")


def test_baselines_instantiate_with_two_classes():
    for k in ("resnet50", "mobilenetv3_large", "tinyvit_5m", "efficientvit_m4"):
        m = build_model(k).eval()
        assert m(torch.rand(1, 3, 224, 224)).shape == (1, 2)
    assert np.isfinite(0.0)
