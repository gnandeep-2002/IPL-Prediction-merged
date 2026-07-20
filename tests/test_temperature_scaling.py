import numpy as np
import pytest

from src.temperature_scaling import TemperatureScaler, compare_calibration


def test_identity_when_already_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    scaler = TemperatureScaler().fit(p, y)
    assert scaler.T == pytest.approx(1.0, abs=0.3)


def test_transform_preserves_ranking_auc_unchanged():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    logits = rng.normal(size=500)
    p_raw = 1 / (1 + np.exp(-logits / 3.0))
    y = (rng.uniform(size=500) < 1 / (1 + np.exp(-logits))).astype(int)

    scaler = TemperatureScaler().fit(p_raw, y)
    p_cal = scaler.transform(p_raw)

    assert roc_auc_score(y, p_raw) == pytest.approx(roc_auc_score(y, p_cal), abs=1e-9)


def test_overconfident_model_gets_softened():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=2000) * 3
    true_p = 1 / (1 + np.exp(-logits / 3.0))
    p_raw = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=2000) < true_p).astype(int)

    scaler = TemperatureScaler().fit(p_raw, y)
    assert scaler.T > 1.0

    p_cal = scaler.transform(p_raw)
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


def test_compare_calibration_bins_are_consistent_with_scalar_ece():
    from src.metrics import ece

    rng = np.random.default_rng(6)
    p_val = rng.uniform(0.05, 0.95, 400)
    y_val = (rng.uniform(size=400) < p_val).astype(int)
    p_test = rng.uniform(0.05, 0.95, 400)
    y_test = (rng.uniform(size=400) < p_test).astype(int)

    result = compare_calibration("dummy", p_val, y_val, p_test, y_test, n_bins=10)

    assert "bins_raw" in result and "bins_cal" in result
    assert len(result["bins_raw"]) > 0
    assert len(result["bins_cal"]) > 0
    for b in result["bins_raw"] + result["bins_cal"]:
        assert {"pred_mean", "obs_freq", "count"}.issubset(b.keys())

    manual_ece_raw = sum((b["count"] / 400) * abs(b["obs_freq"] - b["pred_mean"]) for b in result["bins_raw"])
    assert manual_ece_raw == pytest.approx(result["ece_raw"], abs=1e-3)
