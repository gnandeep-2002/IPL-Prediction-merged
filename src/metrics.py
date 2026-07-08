"""
Calibration and skill metrics used throughout the IPL pipeline.

ECE (Expected Calibration Error) and Brier Skill Score are the primary
evaluation metrics alongside AUC, following the emphasis on well-calibrated
probability outputs rather than raw accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """
    Per-bin calibration data for a reliability diagram, using the same
    equal-width binning ece() reduces to a single scalar.

    Returns a list of dicts, one per NON-EMPTY bin (empty bins are omitted --
    nothing to plot), each with:
        pred_mean : mean predicted probability of samples in the bin
        obs_freq  : observed win frequency (mean of y_true) in the bin
        count     : number of samples in the bin

    A perfectly calibrated model has pred_mean == obs_freq in every bin,
    i.e. every point falls on the y=x diagonal on a reliability diagram.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        out.append({
            "pred_mean": float(y_prob[m].mean()),
            "obs_freq": float(y_true[m].mean()),
            "count": int(m.sum()),
        })
    return out


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error: weighted mean absolute difference between
    predicted probability and observed frequency across equal-width bins.

    Returns a value in [0, 1]; lower is better-calibrated. Built on top of
    calibration_bins() so the scalar ECE and the per-bin reliability-diagram
    data can never disagree with each other.
    """
    bins = calibration_bins(y_true, y_prob, n_bins)
    total = len(y_prob)
    if total == 0:
        return 0.0
    return float(sum((b["count"] / total) * abs(b["obs_freq"] - b["pred_mean"]) for b in bins))


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
