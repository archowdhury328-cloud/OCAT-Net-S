"""Fold-by-seed aggregation, variance components, paired comparisons, Holm correction.

Statistical caveats (kept in outputs): the 25 fold-seed scores are NOT independent; training sets overlap across
folds and the same test images are re-scored across seeds. Fold-level paired inference (n=5) is approximate.
With one AUC per (fold, seed) cell the fold x seed interaction is NOT separable from residual error (no cell
replication); the two-way random-effects model below therefore reports sigma2_fold, sigma2_seed and a pooled
interaction+residual term, clipping negative method-of-moments estimates at 0 and flagging them.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fold_seed_summary(mat: np.ndarray) -> dict:
    """mat: (n_folds, n_seeds) of a metric. Returns grand/fold/seed means and SDs."""
    m = np.asarray(mat, float)
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2:
        raise ValueError("need a (folds>=2, seeds>=2) matrix")
    return {
        "grand_mean": float(m.mean()),
        "fold_means": m.mean(axis=1).tolist(),
        "seed_means": m.mean(axis=0).tolist(),
        "fold_sd_across_seeds": m.std(axis=1, ddof=1).tolist(),
        "seed_sd_across_folds": m.std(axis=0, ddof=1).tolist(),
    }


def variance_components(mat: np.ndarray) -> dict:
    """Two-way crossed random effects, one observation per cell (method of moments / expected mean squares)."""
    m = np.asarray(mat, float)
    a, b = m.shape
    gm = m.mean()
    ms_a = b * ((m.mean(1) - gm) ** 2).sum() / (a - 1)
    ms_b = a * ((m.mean(0) - gm) ** 2).sum() / (b - 1)
    resid = m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm
    ms_e = (resid**2).sum() / ((a - 1) * (b - 1))
    raw = {"fold": (ms_a - ms_e) / b, "seed": (ms_b - ms_e) / a, "interaction_plus_residual": ms_e}
    neg = [k for k, v in raw.items() if v < 0]
    comp = {k: max(v, 0.0) for k, v in raw.items()}
    tot = sum(comp.values())
    return {
        "components": comp,
        "raw_components": raw,
        "negative_estimates_clipped": neg,
        "percent": {k: (100 * v / tot if tot else float("nan")) for k, v in comp.items()},
        "icc_fold": comp["fold"] / tot if tot else float("nan"),
        "note": "interaction and residual are not separately identified (one observation per fold x seed cell)",
    }


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_fold_comparison(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """a,b: (folds, seeds) metric matrices on identical folds/seeds. Differences are averaged over seeds within
    fold, then a paired t-interval over folds (df = folds-1) is reported. Approximate (overlapping training sets)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("matrices must align on folds x seeds")
    d = (a - b).mean(axis=1)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        return {"delta": mean, "ci": (float("nan"), float("nan")), "t": float("nan"), "p": float("nan"),
                "cohen_dz": float("nan"), "n_folds": n, "note": "zero variance of paired differences: test undefined"}  # fmt: skip
    se = sd / np.sqrt(n)
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), n - 1))
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se
    return {"delta": mean, "ci": (mean - float(h), mean + float(h)), "t": float(t), "p": p,
            "cohen_dz": mean / sd, "n_folds": n,
            "note": "paired over folds (seed-averaged); effect size d_z = mean(diff)/sd(diff)"}  # fmt: skip
