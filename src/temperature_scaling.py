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
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

    from src.metrics import calibration_bins, ece

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
        "bins_raw": calibration_bins(np.asarray(y_test), np.asarray(probs_test), n_bins=n_bins),
        "bins_cal": calibration_bins(np.asarray(y_test), np.asarray(cal_test), n_bins=n_bins),
    }
