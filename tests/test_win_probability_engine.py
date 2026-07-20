import numpy as np
import torch

from src.transformer_model import IPLTransformer, INPUT_DIM
from src.win_probability_engine import WinProbabilityEngine, _make_over_labels


def test_predict_returns_correct_shapes():
    torch.manual_seed(0)
    model = IPLTransformer()
    engine = WinProbabilityEngine(model, mc_samples=5)
    features = np.random.randn(15, INPUT_DIM).astype(np.float32)
    result = engine.predict(features)
    assert result.win_prob_mean.shape == (15,)
    assert result.win_prob_std.shape == (15,)
    assert len(result.over_labels) == 15


def test_mc_dropout_produces_nonzero_variance():
    torch.manual_seed(0)
    model = IPLTransformer()
    engine = WinProbabilityEngine(model, mc_samples=20)
    features = np.random.randn(10, INPUT_DIM).astype(np.float32)
    result = engine.predict(features)
    assert (result.win_prob_std > 0).any()


def test_model_returns_to_eval_mode_after_predict():
    torch.manual_seed(0)
    model = IPLTransformer()
    engine = WinProbabilityEngine(model, mc_samples=3)
    features = np.random.randn(5, INPUT_DIM).astype(np.float32)
    engine.predict(features)
    assert not model.training


def test_ci_bounds_contain_mean_and_are_clipped():
    torch.manual_seed(0)
    model = IPLTransformer()
    engine = WinProbabilityEngine(model, mc_samples=15)
    features = np.random.randn(8, INPUT_DIM).astype(np.float32)
    result = engine.predict(features)
    assert (result.ci_lower <= result.win_prob_mean + 1e-9).all()
    assert (result.win_prob_mean - 1e-9 <= result.ci_upper).all()
    assert (result.ci_lower >= 0).all() and (result.ci_upper <= 1).all()


def test_update_matches_last_element_of_predict():
    torch.manual_seed(0)
    model = IPLTransformer()
    engine = WinProbabilityEngine(model, mc_samples=5)
    features = np.random.randn(12, INPUT_DIM).astype(np.float32)

    torch.manual_seed(1)
    mean, lo, hi = engine.update(features)
    assert lo <= mean <= hi


def test_make_over_labels_format():
    labels = _make_over_labels(8)
    assert labels[0] == "0.1"
    assert labels[5] == "0.6"
    assert labels[6] == "1.1"


def test_predict_match_handles_none_innings():
    torch.manual_seed(0)
    model = IPLTransformer()
    engine = WinProbabilityEngine(model, mc_samples=3)
    features = np.random.randn(6, INPUT_DIM).astype(np.float32)
    match_result = engine.predict_match(features, None, "Team A", "Team B")
    assert match_result.innings1 is not None
    assert match_result.innings2 is None
    assert match_result.batting_team1 == "Team A"
