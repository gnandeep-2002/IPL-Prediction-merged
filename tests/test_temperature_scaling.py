"""
Tests for src/temperature_scaling.py (ported from project_hrishav/calibration.py).
"""
import numpy as np
import pytest

from src.temperature_scaling import TemperatureScaler, compare_calibration


def test_identity_when_already_calibrated():
    """If probs are already well-calibrated, T should stay close to 1."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    scaler = TemperatureScaler().fit(p, y)
    assert scaler.T == pytest.approx(1.0, abs=0.3)


def test_transform_preserves_ranking_auc_unchanged():
    """Temperature scaling must preserve ROC-AUC (monotonic transform)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    logits = rng.normal(size=500)
    p_raw = 1 / (1 + np.exp(-logits / 3.0))  # overconfident-looking probs
    y = (rng.uniform(size=500) < 1 / (1 + np.exp(-logits))).astype(int)

    scaler = TemperatureScaler().fit(p_raw, y)
    p_cal = scaler.transform(p_raw)

    assert roc_auc_score(y, p_raw) == pytest.approx(roc_auc_score(y, p_cal), abs=1e-9)


def test_overconfident_model_gets_softened():
    """An overconfident model (T should be > 1) should have its extreme
    probabilities pulled toward 0.5 after transform."""
    rng = np.random.default_rng(2)
    logits = rng.normal(size=2000) * 3  # exaggerated confidence
    true_p = 1 / (1 + np.exp(-logits / 3.0))  # true probabilities are much less extreme
    p_raw = 1 / (1 + np.exp(-logits))  # overconfident reported probabilities
    y = (rng.uniform(size=2000) < true_p).astype(int)

    scaler = TemperatureScaler().fit(p_raw, y)
    assert scaler.T > 1.0

    p_cal = scaler.transform(p_raw)
    # calibrated probabilities should be less extreme on average
    assert np.abs(p_cal - 0.5).mean() < np.abs(p_raw - 0.5).mean()


def test_compare_calibration_returns_expected_keys():
    rng = np.random.default_rng(3)
    p_val = rng.uniform(0.05, 0.95, 300)
    y_val = (rng.uniform(size=300) < p_val).astype(int)
    p_test = rng.uniform(0.05, 0.95, 300)
    y_test = (rng.uniform(size=300) < p_test).astype(int)

    result = compare_calibration("dummy", p_val, y_val, p_test, y_test)
    expected_keys = {"model", "T", "brier_raw", "brier_cal", "logloss_raw",
                      "logloss_cal", "auc_raw", "auc_cal", "ece_raw", "ece_cal"}
    assert expected_keys.issubset(result.keys())
    assert result["model"] == "dummy"
