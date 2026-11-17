"""Metrics from sample-level predictions. Positive = Oral Cancer (1). Undefined ratios -> NaN, never 0."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _d(a, b):
    return float(a / b) if b else float("nan")


def binary_metrics(y, prob, threshold: float = 0.5) -> dict:
    y, prob = np.asarray(y).astype(int), np.asarray(prob, float)
    if not np.isfinite(prob).all():
        raise ValueError("non-finite probabilities/scores")
    pred = (prob >= threshold).astype(int)
    tp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    sens, spec = _d(tp, tp + fn), _d(tn, tn + fp)
    den = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    one = len(np.unique(y)) < 2
    return {
        "n": int(len(y)),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "accuracy": _d(tp + tn, len(y)),
        "sensitivity": sens,
        "specificity": spec,
        "precision": _d(tp, tp + fp),
        "f1": _d(2 * tp, 2 * tp + fp + fn),
        "npv": _d(tn, tn + fn),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "mcc": (tp * tn - fp * fn) / den if den else float("nan"),
        "roc_auc": float("nan") if one else float(roc_auc_score(y, prob)),
        "average_precision": float("nan") if one else float(average_precision_score(y, prob)),
    }
