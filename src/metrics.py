"""
Calibration and skill metrics used throughout the IPL pipeline.

ECE (Expected Calibration Error) and Brier Skill Score are the primary
evaluation metrics alongside AUC, following the emphasis on well-calibrated
probability outputs rather than raw accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error: weighted mean absolute difference between
    predicted probability and observed frequency across equal-width bins.

    Returns a value in [0, 1]; lower is better-calibrated.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        err += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(err)


def brier_skill_score(
    y: np.ndarray,
    p: np.ndarray,
    p_clim: np.ndarray,
) -> float:
    """
    Brier Skill Score relative to a reference climatology forecast.

    BSS = 1 - BS(model) / BS(climatology).
    BSS = 0 means no improvement over climatology; BSS = 1 is perfect.
    Negative values indicate the model is worse than climatology.

    DEF-M03: requires an explicit p_clim (training base-rate) to avoid
    leaking test-set label distribution into the reference forecast.
    """
    return float(1 - brier_score_loss(y, p) / brier_score_loss(y, p_clim))
