import numpy as np
import pytest

from src.pipeline import binary_log_loss, classification_metrics, _regression_row


class TestBinaryLogLoss:

    def test_matches_manual_formula(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.2, 0.6, 0.4])
        expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        assert binary_log_loss(y, p) == pytest.approx(expected)

    def test_finite_for_exact_zero_and_one_probabilities(self):
        val = binary_log_loss(np.array([1, 0]), np.array([0.0, 1.0]))
        assert np.isfinite(val)
        assert val > 30

    def test_perfect_predictions_score_near_zero(self):
        assert binary_log_loss(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-12)


class TestClassificationMetricsExtended:

    Y = np.array([0, 1, 1, 0])
    P = np.array([0.2, 0.8, 0.6, 0.6])

    def test_confusion_counts_exact(self):
        cm = classification_metrics(self.Y, self.P)["confusion"]
        assert cm == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}

    def test_confusion_counts_sum_to_n(self):
        cm = classification_metrics(self.Y, self.P)["confusion"]
        assert sum(cm.values()) == len(self.Y)

    def test_precision_recall_f1_exact(self):
        m = classification_metrics(self.Y, self.P)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(0.8)

    def test_specificity_fpr_fnr_exact(self):
        m = classification_metrics(self.Y, self.P)
        assert m["specificity"] == pytest.approx(0.5)
        assert m["fpr"] == pytest.approx(0.5)
        assert m["fnr"] == pytest.approx(0.0)

    def test_fpr_fnr_are_complements_of_specificity_and_recall(self):
        m = classification_metrics(self.Y, self.P)
        assert m["fpr"] == pytest.approx(1.0 - m["specificity"])
        assert m["fnr"] == pytest.approx(1.0 - m["recall"])

    def test_zero_division_guard_when_no_positive_predictions(self):
        m = classification_metrics(np.array([0, 1, 1]), np.array([0.4, 0.4, 0.4]))
        assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
        assert m["confusion"]["tp"] == 0 and m["confusion"]["fp"] == 0
        assert m["specificity"] == 1.0
        assert m["fnr"] == 1.0

    def test_zero_division_guard_when_no_actual_negatives(self):
        m = classification_metrics(np.array([1, 1]), np.array([0.9, 0.2]))
        assert m["specificity"] == 0.0 and m["fpr"] == 0.0
        assert m["auc"] == 0.5

    def test_log_loss_key_present_and_matches_helper(self):
        m = classification_metrics(self.Y, self.P)
        assert m["log_loss"] == pytest.approx(binary_log_loss(self.Y, self.P))

    def test_preexisting_keys_and_values_unchanged(self):
        m = classification_metrics(self.Y, self.P)
        assert {"brier", "auc", "acc", "ece"}.issubset(m.keys())
        assert "bss" not in m
        assert m["acc"] == pytest.approx(0.75)

        with_ref = classification_metrics(self.Y, self.P, p_ref=np.full(4, 0.5))
        assert "bss" in with_ref

        no_ece = classification_metrics(self.Y, self.P, with_ece=False)
        assert "ece" not in no_ece
        assert "log_loss" in no_ece


class TestRegressionRow:

    def test_reports_all_four_metrics_consistently(self):
        y = np.array([100.0, 150.0, 200.0])
        pred = np.array([110.0, 140.0, 190.0])
        row = _regression_row(y, pred)
        assert set(row.keys()) == {"MAE", "RMSE", "R2", "n"}
        assert row["MAE"] == pytest.approx(10.0)
        assert row["RMSE"] == pytest.approx(10.0)
        assert row["n"] == 3
