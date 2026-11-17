"""Validation-only temperature scaling, thresholds, calibration metrics, prevalence-adjusted PPV/NPV.

P(Oral Cancer) = softmax(z / T)[1], T > 0 fitted by minimising NLL on VALIDATION logits (LBFGS on log T).
Thresholds (predefined; paper criteria not recoverable -> assumption A-CAL1), chosen on validation probabilities only:
  f1_optimal      argmax F1 over candidate thresholds; ties -> threshold closest to 0.5.
  high_sensitivity among thresholds with sensitivity >= 0.95 choose max specificity; ties -> larger threshold.
  high_specificity among thresholds with specificity >= 0.95 choose max sensitivity; ties -> smaller threshold.
Decision: predict Oral Cancer if prob >= threshold. ECE: 15 equal-width bins on top-label confidence in [0.5, 1].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> float:
    z = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def positive_prob(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = torch.as_tensor(logits, dtype=torch.float64) / temperature
    return z.softmax(dim=1)[:, 1].numpy()


def _counts(y, p, thr):
    pred = p >= thr
    tp, fn = int(((y == 1) & pred).sum()), int(((y == 1) & ~pred).sum())
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    return tp, fn, tn, fp


def _candidates(p: np.ndarray) -> np.ndarray:
    u = np.unique(p)
    if len(u) < 2:
        return np.array([0.5])
    return np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])


def select_thresholds(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for thr in _candidates(p):
        tp, fn, tn, fp = _counts(y, p, thr)
        se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        rows.append((thr, se, sp, 2 * tp / max(2 * tp + fp + fn, 1)))
    r = np.array(rows)
    f1c = r[np.isclose(r[:, 3], r[:, 3].max())]
    out = {"f1_optimal": float(f1c[np.argmin(np.abs(f1c[:, 0] - 0.5)), 0])}
    hs = r[r[:, 1] >= target]
    out["high_sensitivity"] = float(hs[np.lexsort((-hs[:, 0], -hs[:, 2]))[0], 0]) if len(hs) else float("nan")
    hp = r[r[:, 2] >= target]
    out["high_specificity"] = float(hp[np.lexsort((hp[:, 0], -hp[:, 1]))[0], 0]) if len(hp) else float("nan")
    return out


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def nll(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins: int = 15) -> float:
    y, p = np.asarray(y), np.asarray(p)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def prevalence_adjusted(sens: float, spec: float, prevalence: float) -> dict[str, float]:
    """Scenario analysis ONLY (assumed prevalence, same sens/spec); not a measured screening cohort."""
    tp, fn = prevalence * sens, prevalence * (1 - sens)
    tn, fp = (1 - prevalence) * spec, (1 - prevalence) * (1 - spec)
    return {
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "false_negatives_per_1000": 1000 * fn,
    }
