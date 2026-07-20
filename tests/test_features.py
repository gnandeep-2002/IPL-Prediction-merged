import numpy as np
import pandas as pd
import pytest

from src.data import normalise
from src.features import (
    balls_remaining,
    compute_form_h2h,
    compute_h2h_beta,
    crr,
    get_enc,
    phase,
    rrr,
    runs_needed,
)


def test_name_map_known():
    assert normalise("Delhi Daredevils") == "Delhi Capitals"
    assert normalise("Kings XI Punjab") == "Punjab Kings"
    assert normalise("Royal Challengers Bangalore") == "Royal Challengers Bengaluru"
    assert normalise("Rising Pune Supergiant") == "Rising Pune Supergiants"


def test_name_map_deccan_chargers():
    assert normalise("Deccan Chargers") == "Deccan Chargers"


def test_name_map_pune_warriors():
    assert normalise("Pune Warriors") == "Pune Warriors"


def test_name_map_passthrough():
    assert normalise("Mumbai Indians") == "Mumbai Indians"
    assert normalise("Chennai Super Kings") == "Chennai Super Kings"


def test_balls_remaining_mid_innings():
    assert balls_remaining(60) == 60

def test_balls_remaining_start():
    assert balls_remaining(0) == 120

def test_balls_remaining_end():
    assert balls_remaining(120) == 1

def test_balls_remaining_never_zero():
    for tb in range(0, 125):
        assert balls_remaining(tb) >= 1


def test_runs_needed_normal():
    assert runs_needed(180, 100) == 80

def test_runs_needed_already_won():
    assert runs_needed(180, 190) == 0

def test_runs_needed_exactly_at_target():
    assert runs_needed(180, 180) == 0


def test_crr_basic():
    assert crr(60, 60) == pytest.approx(6.0)

def test_crr_zero_balls():
    assert crr(0, 0) == pytest.approx(0.0)


def test_rrr_basic():
    assert rrr(60, 60) == pytest.approx(6.0)

def test_rrr_zero_balls():
    assert rrr(10, 0) == pytest.approx(60.0)


def test_target_enc_known_player():
    enc_map = {"Virat Kohli": 35.2, "MS Dhoni": 28.7}
    assert get_enc(enc_map, 20.0, "Virat Kohli") == 35.2

def test_target_enc_unseen_player():
    enc_map = {"Virat Kohli": 35.2}
    assert get_enc(enc_map, 20.0, "Unknown Player") == 20.0

def test_target_enc_empty_map():
    assert get_enc({}, 19.5, "Anyone") == 19.5


def test_phase_powerplay():
    for o in range(0, 7):
        assert phase(o) == 0, f"over {o} should be powerplay (phase 0)"

def test_phase_middle():
    for o in range(7, 16):
        assert phase(o) == 1, f"over {o} should be middle (phase 1)"

def test_phase_death():
    for o in range(16, 20):
        assert phase(o) == 2, f"over {o} should be death (phase 2)"


def _make_matches(wins_pattern):
    rows = [
        {"match_id": i, "team1": "MI", "team2": "CSK", "team1_win": w, "year": 2015}
        for i, w in enumerate(wins_pattern)
    ]
    return pd.DataFrame(rows)


def test_form_first_match_is_default():
    df = _make_matches([1, 0, 1])
    form1, form2, h2h, _, _ = compute_form_h2h(df, window=5)
    assert form1[0] == pytest.approx(0.5)
    assert form2[0] == pytest.approx(0.5)


def test_h2h_first_match_is_default():
    df = _make_matches([1, 0])
    _, _, h2h, _, _ = compute_form_h2h(df, window=5)
    assert h2h[0] == pytest.approx(0.5)


def test_h2h_no_lookahead():
    df = _make_matches([1, 1, 1, 0, 0])
    _, _, h2h_rate, _, _ = compute_form_h2h(df, window=5)
    assert h2h_rate[3] == pytest.approx(1.0, abs=1e-6)


def test_form_walk_forward_no_leakage():
    df = _make_matches([0, 0, 0, 1, 1])
    form1, *_ = compute_form_h2h(df, window=5)
    assert form1[3] == pytest.approx(0.0)


def test_state_dicts_returned():
    df = _make_matches([1, 0, 1])
    form1, form2, h2h_rate, tw, h2h = compute_form_h2h(df, window=5)
    assert isinstance(tw, dict)
    assert isinstance(h2h, dict)
    assert "MI" in tw


def test_h2h_beta_prior_shrinkage():
    df = _make_matches([1, 1, 1, 1, 1])
    betas = compute_h2h_beta(df)
    assert betas[4] == pytest.approx(6 / 8, abs=1e-6)
    assert betas[4] < 1.0


def test_h2h_beta_prior_mean_on_empty():
    df = _make_matches([1])
    betas = compute_h2h_beta(df)
    assert betas[0] == pytest.approx(0.5, abs=1e-6)
