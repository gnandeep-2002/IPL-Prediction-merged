"""
Tests for src/pipeline.py's calibration-method comparison
(compare_calibration_methods_dyn2) using a small synthetic dynamic
2nd-innings dataframe -- no real dataset needed.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import compare_calibration_methods_dyn2, DYN2


def _make_df2(n=2000, seed=0):
    """Synthetic dynamic 2nd-innings dataframe spanning 2008-2025, with a
    genuine (if noisy) relationship between features and chasing_wins so
    the fitted models aren't just noise."""
    rng = np.random.default_rng(seed)
    years = rng.integers(2008, 2026, n)
    runs_needed = rng.uniform(0, 200, n)
    balls_remaining = rng.uniform(1, 120, n)
    wkts_remaining = rng.integers(0, 11, n)
    crr = rng.uniform(4, 12, n)
    rrr = runs_needed / np.clip(balls_remaining, 1, None) * 6
    elo_adv = rng.normal(0, 100, n)
    phase = rng.integers(0, 3, n)

    # chasing team more likely to win with a lower required rate, more
    # wickets in hand, and a stronger Elo -- gives the model real signal.
    logit = (-0.05 * rrr) + (0.3 * wkts_remaining) + (0.01 * elo_adv) + rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-logit))
    chasing_wins = (rng.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "match_id": np.arange(n),
        "year": years,
        "runs_needed": runs_needed,
        "balls_remaining": balls_remaining,
        "wkts_remaining": wkts_remaining,
        "crr": crr,
        "rrr": rrr,
        "elo_adv": elo_adv,
        "phase": phase,
        "chasing_wins": chasing_wins,
    })


class TestCompareCalibrationMethodsDyn2:

    def test_returns_temperature_and_isotonic_keys(self):
        df2 = _make_df2()
        result = compare_calibration_methods_dyn2(df2)
        assert "temperature" in result and "isotonic" in result

    def test_isotonic_result_has_expected_fields(self):
        df2 = _make_df2()
        result = compare_calibration_methods_dyn2(df2)
        iso = result["isotonic"]
        assert {"brier", "auc", "ece", "bins"}.issubset(iso.keys())
        assert 0.0 <= iso["brier"] <= 1.0
        assert 0.0 <= iso["auc"] <= 1.0

    def test_temperature_result_has_bins_for_reliability_diagram(self):
        df2 = _make_df2()
        result = compare_calibration_methods_dyn2(df2)
        temp = result["temperature"]
        assert "bins_raw" in temp and "bins_cal" in temp
        assert len(temp["bins_cal"]) > 0

    def test_uses_only_dyn2_feature_columns(self):
        """Sanity check: the function must not require any columns beyond
        DYN2 + year + chasing_wins + match_id. match_id became a REQUIRED
        column with the DEF-001 fix: calibration folds are grouped by match
        so one match's deliveries are never split across the base-model and
        calibration partitions."""
        df2 = _make_df2()
        required = set(DYN2) | {"year", "chasing_wins", "match_id"}
        assert required.issubset(df2.columns)
        # Should not raise even if we drop everything else.
        minimal = df2[list(required)]
        compare_calibration_methods_dyn2(minimal)
