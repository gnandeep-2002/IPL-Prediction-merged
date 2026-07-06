"""
Tests for calibration metric functions (src.metrics).
"""
import numpy as np
import pytest

from src.metrics import brier_skill_score, ece


def _clim(y_true):
    """Constant climatology array (training base-rate approximation)."""
    return np.full(len(y_true), y_true.mean())


class TestECE:

    def test_perfect_calibration_returns_zero(self):
        """When predicted probability equals actual frequency, ECE = 0."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.full(10, 0.5)
        assert ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictor_has_low_ece(self):
        """Model that always predicts the correct answer should have near-zero ECE."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.99, 0.99, 0.99, 0.01, 0.01, 0.01])
        assert ece(y_true, y_prob) < 0.05

    def test_overconfident_model_has_high_ece(self):
        """Model that always predicts extreme probabilities on uncertain data."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.where(y_true == 1, 0.99, 0.01)
        flip = np.random.choice(200, 100, replace=False)
        y_true[flip] = 1 - y_true[flip]
        assert ece(y_true, y_prob) > 0.30

    def test_ece_range(self):
        """ECE must always be in [0, 1]."""
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.uniform(0, 1, 500)
        result = ece(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_ece_returns_scalar(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3])
        result = ece(y_true, y_prob)
        assert isinstance(result, float)

    def test_ece_bins_argument(self):
        """Different n_bins should still produce a valid result."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        for n in [5, 10, 20]:
            r = ece(y_true, y_prob, n_bins=n)
            assert 0.0 <= r <= 1.0


class TestBrierSkillScore:

    def test_perfect_model_returns_one(self):
        """A perfect model (BSS = 1) when predictions match labels exactly."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        assert brier_skill_score(y_true, y_prob, _clim(y_true)) == pytest.approx(1.0, abs=1e-9)

    def test_climatology_returns_zero(self):
        """Predicting the base rate for every sample = BSS of 0."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1])
        climate = np.full(len(y_true), y_true.mean())
        assert brier_skill_score(y_true, climate, climate) == pytest.approx(0.0, abs=1e-9)

    def test_worse_than_climate_is_negative(self):
        """A model worse than climatology should have negative BSS."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.05, 0.95, 0.05, 0.95, 0.05, 0.95])
        assert brier_skill_score(y_true, y_prob, _clim(y_true)) < 0.0

    def test_better_than_climate_is_positive(self):
        """A model better than climatology should have positive BSS."""
        np.random.seed(1)
        y_true = np.random.randint(0, 2, 300)
        y_prob = np.clip(y_true * 0.7 + np.random.normal(0, 0.1, 300), 0, 1)
        assert brier_skill_score(y_true, y_prob, _clim(y_true)) > 0.0

    def test_bss_range(self):
        """BSS can go below -1 for adversarial predictions but perfect = 1."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        bss = brier_skill_score(y_true, y_prob, _clim(y_true))
        assert bss < 0.0

    def test_bss_symmetry(self):
        """Predictions p and 1-p should give the same BSS on labels y and 1-y."""
        np.random.seed(7)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        bss1 = brier_skill_score(y_true, y_prob, _clim(y_true))
        bss2 = brier_skill_score(1 - y_true, 1 - y_prob, _clim(1 - y_true))
        assert bss1 == pytest.approx(bss2, abs=1e-9)

    def test_explicit_climatology_different_from_test_mean(self):
        """
        DEF-M03: The 3-arg form allows passing a training climatology that differs
        from the test-set marginal, which avoids test-set label leakage.
        """
        y_true     = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_prob     = np.array([0.7, 0.65, 0.6, 0.4, 0.35, 0.3, 0.35, 0.4])
        train_clim = np.full(len(y_true), 0.5)
        bss_train  = brier_skill_score(y_true, y_prob, train_clim)
        test_clim  = _clim(y_true)
        bss_test   = brier_skill_score(y_true, y_prob, test_clim)
        assert bss_train != pytest.approx(bss_test, abs=1e-6)
