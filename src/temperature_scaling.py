"""
Temperature scaling for probabilistic classifiers.

Ported from project_hrishav/calibration.py. A single-parameter calibration
method (Guo et al. 2017): given uncalibrated probabilities p, find a scalar
T > 0 such that the rescaled probabilities

    logit(p)    = log(p / (1 - p))
    p_cal       = sigmoid(logit(p) / T)

minimise the negative log-likelihood on a held-out validation set.

T > 1 -> predictions are too confident (softens them).
T < 1 -> predictions are too cautious (sharpens them).
T = 1 -> no change.

Kept as a separate, standalone utility from src/models.py's isotonic
CalibratedClassifierCV approach -- they solve different problems (isotonic
needs k-fold CV and is more flexible; temperature scaling is a single
scalar fit on a val set and cannot overfit) and are not meant to share
an abstraction.

Note: the one code change vs. the original is `compare_calibration`'s ECE
call, which now uses src.metrics.ece (already ported and tested) instead
of project_hrishav's own evaluate.expected_calibration_error, so this
module has no dependency on unported hrishav code.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """
    Fit a single scalar temperature T on validation probabilities.

    Usage
    -----
        scaler = TemperatureScaler()
        scaler.fit(val_probs, val_labels)
        cal_probs = scaler.transform(test_probs)
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        probs_val = np.asarray(probs_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        logits = _logit(probs_val)

        def _nll(T: float) -> float:
            if T <= 0:
                return 1e18
            p = _sigmoid(logits / T)
            p = np.clip(p, _EPS, 1.0 - _EPS)
            return -np.mean(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p))

        result = minimize_scalar(_nll, bounds=(0.05, 20.0), method="bounded",
                                  options={"xatol": 1e-4})
        self.T = float(result.x)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        return _sigmoid(_logit(probs) / self.T)

    def fit_transform(self, probs_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        self.fit(probs_val, y_val)
        return self.transform(probs_val)


def compare_calibration(
    name: str,
    probs_val: np.ndarray, y_val: np.ndarray,
    probs_test: np.ndarray, y_test: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Fit temperature on val, apply to test, return before/after metrics.

    Returns a dict with: T, brier_raw, brier_cal, logloss_raw, logloss_cal,
    auc_raw, auc_cal, ece_raw, ece_cal.
    """
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

    from src.metrics import ece

    scaler = TemperatureScaler().fit(probs_val, y_val)
    cal_test = scaler.transform(probs_test)

    return {
        "model": name,
        "T": round(scaler.T, 4),
        "brier_raw": round(brier_score_loss(y_test, probs_test), 4),
        "brier_cal": round(brier_score_loss(y_test, cal_test), 4),
        "logloss_raw": round(log_loss(y_test, np.clip(probs_test, _EPS, 1 - _EPS)), 4),
        "logloss_cal": round(log_loss(y_test, np.clip(cal_test, _EPS, 1 - _EPS)), 4),
        "auc_raw": round(roc_auc_score(y_test, probs_test), 4),
        "auc_cal": round(roc_auc_score(y_test, cal_test), 4),
        "ece_raw": round(ece(np.asarray(y_test), np.asarray(probs_test), n_bins=n_bins), 4),
        "ece_cal": round(ece(np.asarray(y_test), np.asarray(cal_test), n_bins=n_bins), 4),
    }
