"""
Tests for IPL pipeline feature engineering logic (src.features, src.data).
"""
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


# ── NAME_MAP normalisation ────────────────────────────────────────────────────

def test_name_map_known():
    assert normalise("Delhi Daredevils") == "Delhi Capitals"
    assert normalise("Kings XI Punjab") == "Punjab Kings"
    assert normalise("Royal Challengers Bangalore") == "Royal Challengers Bengaluru"
    assert normalise("Rising Pune Supergiant") == "Rising Pune Supergiants"


def test_name_map_deccan_chargers():
    """DEF-008 (supersedes DEF-L01): Deccan Chargers must NOT map to
    Sunrisers Hyderabad -- they are distinct franchises (Deccan folded after
    2012; SRH entered 2013 as a new franchise), so each keeps its own
    Elo/form/H2H history."""
    assert normalise("Deccan Chargers") == "Deccan Chargers"


def test_name_map_pune_warriors():
    """Pune Warriors (defunct 2013) must map to itself (no successor)."""
    assert normalise("Pune Warriors") == "Pune Warriors"


def test_name_map_passthrough():
    assert normalise("Mumbai Indians") == "Mumbai Indians"
    assert normalise("Chennai Super Kings") == "Chennai Super Kings"


# ── balls_remaining ───────────────────────────────────────────────────────────

def test_balls_remaining_mid_innings():
    assert balls_remaining(60) == 60

def test_balls_remaining_start():
    assert balls_remaining(0) == 120

def test_balls_remaining_end():
    assert balls_remaining(120) == 1

def test_balls_remaining_never_zero():
    for tb in range(0, 125):
        assert balls_remaining(tb) >= 1


# ── runs_needed (2nd innings) ─────────────────────────────────────────────────

def test_runs_needed_normal():
    assert runs_needed(180, 100) == 80

def test_runs_needed_already_won():
    assert runs_needed(180, 190) == 0

def test_runs_needed_exactly_at_target():
    assert runs_needed(180, 180) == 0


# ── current run rate ──────────────────────────────────────────────────────────

def test_crr_basic():
    assert crr(60, 60) == pytest.approx(6.0)

def test_crr_zero_balls():
    assert crr(0, 0) == pytest.approx(0.0)


# ── required run rate ─────────────────────────────────────────────────────────

def test_rrr_basic():
    assert rrr(60, 60) == pytest.approx(6.0)

def test_rrr_zero_balls():
    assert rrr(10, 0) == pytest.approx(60.0)


# ── target encoding fallback ──────────────────────────────────────────────────

def test_target_enc_known_player():
    enc_map = {"Virat Kohli": 35.2, "MS Dhoni": 28.7}
    assert get_enc(enc_map, 20.0, "Virat Kohli") == 35.2

def test_target_enc_unseen_player():
    enc_map = {"Virat Kohli": 35.2}
    assert get_enc(enc_map, 20.0, "Unknown Player") == 20.0

def test_target_enc_empty_map():
    assert get_enc({}, 19.5, "Anyone") == 19.5


# ── phase encoding ────────────────────────────────────────────────────────────

def test_phase_powerplay():
    for o in range(0, 7):
        assert phase(o) == 0, f"over {o} should be powerplay (phase 0)"

def test_phase_middle():
    for o in range(7, 16):
        assert phase(o) == 1, f"over {o} should be middle (phase 1)"

def test_phase_death():
    for o in range(16, 20):
        assert phase(o) == 2, f"over {o} should be death (phase 2)"


# ── compute_form_h2h ──────────────────────────────────────────────────────────

def _make_matches(wins_pattern):
    """Build a minimal match_df: MI vs CSK, wins per pattern."""
    rows = [
        {"match_id": i, "team1": "MI", "team2": "CSK", "team1_win": w, "year": 2015}
        for i, w in enumerate(wins_pattern)
    ]
    return pd.DataFrame(rows)


def test_form_first_match_is_default():
    """DEF-M04: first match has no history — form must return default 0.5."""
    df = _make_matches([1, 0, 1])
    form1, form2, h2h, _, _ = compute_form_h2h(df, window=5)
    assert form1[0] == pytest.approx(0.5)
    assert form2[0] == pytest.approx(0.5)


def test_h2h_first_match_is_default():
    """DEF-M04: with no prior H2H matches, raw H2H defaults to 0.5."""
    df = _make_matches([1, 0])
    _, _, h2h, _, _ = compute_form_h2h(df, window=5)
    assert h2h[0] == pytest.approx(0.5)


def test_h2h_no_lookahead():
    """DEF-M04: H2H rate at match i must not include results from match i onward."""
    df = _make_matches([1, 1, 1, 0, 0])
    _, _, h2h_rate, _, _ = compute_form_h2h(df, window=5)
    # After 3 MI wins, at match 3: raw h2h = 3/3 = 1.0
    assert h2h_rate[3] == pytest.approx(1.0, abs=1e-6)


def test_form_walk_forward_no_leakage():
    """DEF-M04: form at match i uses only matches 0..(i-1)."""
    df = _make_matches([0, 0, 0, 1, 1])
    form1, *_ = compute_form_h2h(df, window=5)
    # MI won 0 of first 3, so form1[3] = 0.0
    assert form1[3] == pytest.approx(0.0)


def test_state_dicts_returned():
    """compute_form_h2h must return state dicts (tw, h2h) for walk-forward eval."""
    df = _make_matches([1, 0, 1])
    form1, form2, h2h_rate, tw, h2h = compute_form_h2h(df, window=5)
    assert isinstance(tw, dict)
    assert isinstance(h2h, dict)
    assert "MI" in tw


# ── compute_h2h_beta ──────────────────────────────────────────────────────────

def test_h2h_beta_prior_shrinkage():
    """
    DEF-L03: Beta(2,2) prior shrinks H2H toward 0.5.
    With 4 wins and 0 losses: raw=1.0, smoothed=(4+2)/(4+4)=0.75.
    """
    df = _make_matches([1, 1, 1, 1, 1])
    betas = compute_h2h_beta(df)
    # At match 4 (5th game): 4 prior H2H wins by MI => (4+2)/(4+4) = 0.75
    assert betas[4] == pytest.approx(6 / 8, abs=1e-6)
    assert betas[4] < 1.0


def test_h2h_beta_prior_mean_on_empty():
    """With no prior history, Beta(2,2) posterior mean = alpha/(alpha+beta) = 0.5."""
    df = _make_matches([1])
    betas = compute_h2h_beta(df)
    # First match: no history => (0+2)/(0+2+2) = 0.5
    assert betas[0] == pytest.approx(0.5, abs=1e-6)
