from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob <= hi if hi >= 1.0 else y_prob < hi)
        if m.sum() == 0:
            continue
        out.append({
            "pred_mean": float(y_prob[m].mean()),
            "obs_freq": float(y_true[m].mean()),
            "count": int(m.sum()),
        })
    return out


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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
    return float(1 - brier_score_loss(y, p) / brier_score_loss(y, p_clim))
