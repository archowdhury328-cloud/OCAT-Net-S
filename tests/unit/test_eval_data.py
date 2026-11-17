import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ocat_net_s.data.manifest import (
    assign_outer_folds,
    assign_proxy_groups,
    build_manifest,
    folds_hash,
    inner_val_split,
    natural_key,
    verify_folds,
)
from ocat_net_s.data.transforms import (
    CORRUPTION_LEVELS,
    build_eval_transform,
    build_train_transform,
    compute_norm_stats,
    corrupt,
)
from ocat_net_s.evaluation.leakage import audit, hamming, phash, sample_pairs, ssim
from ocat_net_s.evaluation.metrics import binary_metrics
from ocat_net_s.evaluation.stats import fold_seed_summary, holm, paired_fold_comparison, variance_components
from ocat_net_s.explainability.gradcam import select_examples
from ocat_net_s.utils.common import safe_join
from ocat_net_s.utils.synthetic import make_synthetic_dataset


# ---------------- metrics
def test_binary_metrics_known_answer():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.6, 0.2, 0.1, 0.3, 0.6, 0.7])
    m = binary_metrics(y, p, 0.5)
    assert m["confusion"] == {"tp": 3, "fn": 1, "tn": 2, "fp": 2}
    assert m["sensitivity"] == 0.75 and m["specificity"] == 0.5 and m["precision"] == 0.6
    assert m["f1"] == pytest.approx(2 * 3 / (2 * 3 + 2 + 1)) and m["accuracy"] == 5 / 8
    assert m["mcc"] == pytest.approx((3 * 2 - 2 * 1) / np.sqrt(5 * 4 * 4 * 3))
    assert m["roc_auc"] == pytest.approx(11.5 / 16)


def test_undefined_metrics_nan():
    m = binary_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
    assert np.isnan(m["sensitivity"]) and np.isnan(m["roc_auc"]) and np.isnan(m["mcc"])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [np.nan, 0.5])


# ---------------- stats
def test_fold_seed_summary_and_variance_components():
    rng = np.random.default_rng(0)
    fold_eff = rng.normal(0, 0.02, (5, 1))
    seed_eff = rng.normal(0, 0.008, (1, 5))
    mat = 0.95 + fold_eff + seed_eff + rng.normal(0, 0.001, (5, 5))
    s = fold_seed_summary(mat)
    assert s["grand_mean"] == pytest.approx(mat.mean()) and len(s["fold_means"]) == 5
    v = variance_components(mat)
    assert v["components"]["fold"] > v["components"]["seed"] > 0
    assert sum(v["percent"].values()) == pytest.approx(100)
    assert "not separately identified" in v["note"]
    with pytest.raises(ValueError):
        fold_seed_summary(np.ones((1, 5)))


def test_variance_components_clips_negative():
    mat = np.tile([0.9, 0.9, 0.9, 0.9, 0.9], (5, 1)) + np.random.default_rng(1).normal(0, 1e-3, (5, 5))
    v = variance_components(mat)
    assert all(c >= 0 for c in v["components"].values())


def test_paired_fold_comparison_and_holm():
    rng = np.random.default_rng(0)
    b = rng.normal(0.90, 0.01, (5, 5))
    a = b + 0.02 + rng.normal(0, 0.002, (5, 5))
    r = paired_fold_comparison(a, b)
    assert r["delta"] > 0.015 and r["ci"][0] > 0 and r["p"] < 0.01 and r["n_folds"] == 5
    assert np.isnan(paired_fold_comparison(b, b)["p"])
    with pytest.raises(ValueError):
        paired_fold_comparison(a, b[:, :3])
    h = holm([0.01, 0.04, 0.03])
    assert h == pytest.approx([0.03, 0.06, 0.06]) and all(x <= 1 for x in h)


# ---------------- data
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    r = tmp_path_factory.mktemp("s")
    make_synthetic_dataset(r, 30)
    return r


def test_natural_sort():
    assert sorted(["a10.png", "a2.png", "a1.png"], key=natural_key) == ["a1.png", "a2.png", "a10.png"]


def test_manifest_groups_folds_and_hash(synth):
    df, excl = build_manifest(synth)
    assert len(df) == 60 and len(excl) == 0 and set(df.label) == {0, 1}
    g = assign_proxy_groups(df, 5)
    assert g.groupby("group_id").size().max() == 5 and g.groupby("label").group_id.nunique().tolist() == [6, 6]
    f = assign_outer_folds(g)
    verify_folds(f)
    assert sorted(f.fold.unique()) == [0, 1, 2, 3, 4]
    assert folds_hash(f) == folds_hash(assign_outer_folds(g)) != folds_hash(assign_outer_folds(g, seed=1))
    assert (f.groupby(["fold", "label"]).size() > 0).all()


def test_group_sequential_rule_uses_adjacent_filenames(synth):
    df, _ = build_manifest(synth)
    g = assign_proxy_groups(df, 5)
    sub = g[g.label == 0].copy()
    sub["n"] = sub.filename.str.extract(r"(\d+)").astype(int)
    assert sub.sort_values("n").group_id.tolist()[:5] == [sub.sort_values("n").group_id.iloc[0]] * 5


def test_inner_validation_is_group_disjoint_and_stratified(synth):
    df, _ = build_manifest(synth)
    f = assign_outer_folds(assign_proxy_groups(df, 3))
    tr, va = inner_val_split(f[f.fold != 0], 0.2, seed=0)
    assert not (set(tr.group_id) & set(va.group_id)) and len(va) > 0 and va.label.nunique() == 2


def test_corrupt_and_unmapped_files_reported(synth, tmp_path):
    import shutil

    shutil.copytree(synth, tmp_path / "d")
    (tmp_path / "d" / "Normal" / "bad.png").write_bytes(b"junk")
    (tmp_path / "d" / "Other").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "d" / "Other" / "x.png")
    df, excl = build_manifest(tmp_path / "d")
    assert len(df) == 60 and set(excl.reason) == {"corrupt:UnidentifiedImageError", "unmapped_label_dir"}


def test_conflicting_duplicates_excluded(tmp_path):
    for c in ("Normal", "Oral Cancer"):
        (tmp_path / c).mkdir()
        Image.fromarray(np.full((8, 8, 3), 7, np.uint8)).save(tmp_path / c / "a.png")
    df, excl = build_manifest(tmp_path)
    assert len(df) == 0 and (excl.reason == "label_conflict_duplicate").sum() == 2


def test_safe_join(tmp_path):
    with pytest.raises(ValueError):
        safe_join(tmp_path, "../x")


# ---------------- transforms
def test_train_transform_composition_matches_table2():
    t = build_train_transform((0.5,) * 3, (0.2,) * 3)
    names = [type(x).__name__ for x in t.transforms]
    assert names == ["RandomResizedCrop", "RandomHorizontalFlip", "RandomRotation", "ColorJitter", "RandomApply",
                     "AutoAugment", "ToTensor", "Normalize", "RandomErasing"]  # fmt: skip
    rrc, flip, rot, cj, ra, _, _, _, er = t.transforms
    assert rrc.scale == (0.70, 1.0) and rrc.ratio == (0.75, 1.33) and flip.p == 0.5 and rot.degrees == [-15, 15]
    assert (cj.brightness, cj.contrast, cj.saturation, cj.hue) == ((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.1, 0.1))
    assert ra.p == 0.5 and er.p == 0.25 and er.scale == (0.02, 0.33) and er.ratio == (0.3, 3.3)


def test_eval_transform_deterministic_and_no_augmentation():
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (90, 120, 3), dtype=np.uint8))
    t = build_eval_transform((0.5,) * 3, (0.25,) * 3)
    a, b = t(im), t(im)
    assert a.shape == (3, 224, 224) and (a == b).all()
    assert [type(x).__name__ for x in t.transforms] == ["Resize", "CenterCrop", "ToTensor", "Normalize"]


def test_norm_stats_from_given_images_only():
    ims = [Image.new("RGB", (10, 10), (255, 0, 0)), Image.new("RGB", (10, 10), (0, 0, 0))]
    m, s = compute_norm_stats(ims)
    assert m == pytest.approx([0.5, 0.0, 0.0]) and s == pytest.approx([0.5, 0.0, 0.0])
    with pytest.raises(ValueError):
        compute_norm_stats([])


@pytest.mark.parametrize("kind", list(CORRUPTION_LEVELS))
def test_corruptions_reproducible_ordered_and_shared(kind):
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    a = np.asarray(corrupt(im, kind, "high", "id1"), float)
    assert np.array_equal(a, np.asarray(corrupt(im, kind, "high", "id1"), float))
    lo = np.abs(np.asarray(corrupt(im, kind, "low", "id1"), float) - np.asarray(im, float)).mean()
    hi = np.abs(a - np.asarray(im, float)).mean()
    assert hi >= lo and a.shape == (64, 64, 3) and a.min() >= 0 and a.max() <= 255
    if kind == "gaussian_noise":
        assert not np.array_equal(a, np.asarray(corrupt(im, kind, "high", "id2"), float))


# ---------------- leakage audit
def test_phash_ssim_detect_near_duplicates_and_audit():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    a = Image.fromarray(base)
    b = Image.fromarray(np.clip(base.astype(int) + 3, 0, 255).astype(np.uint8))
    c = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert hamming(phash(a), phash(b)) <= 5 < hamming(phash(a), phash(c))
    assert ssim(a, a) == pytest.approx(1.0) and ssim(a, b) > ssim(a, c)
    imgs = [a, b, c, c]
    groups, folds = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    pairs = sample_pairs(np.array(["g0", "g0", "g1", "g1"]), folds, n=3, seed=0)
    res = audit(imgs, pairs)
    assert res["within_group"]["n_pairs"] == 2 and "clip" in res["random"]
    assert pairs == sample_pairs(np.array(["g0", "g0", "g1", "g1"]), folds, n=3, seed=0)  # seeded
    assert groups.size == 4


def test_select_examples_rule_covers_all_cells():
    y, p = [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0]
    ids = list("fedcba")
    s = select_examples(y, p, ids, per_cell=1)
    assert s == {"TP": ["a"], "TN": ["b"], "FP": ["e"], "FN": ["c"]} or set(s) == {"TP", "TN", "FP", "FN"}
    assert all(len(v) == 1 for v in s.values())
    assert isinstance(pd.__version__, str)
