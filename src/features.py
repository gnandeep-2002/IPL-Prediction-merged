"""
Feature engineering for the IPL win probability and score prediction pipeline.

All walk-forward functions compute features using only past match data —
no future information is ever used (no lookahead leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Scalar helpers used in dynamic (in-game) feature construction ─────────────

def balls_remaining(team_balls: int, total_balls: int = 120) -> int:
    """Balls left in the innings; clipped to at least 1 to avoid division by zero."""
    return max(total_balls - team_balls, 1)


def runs_needed(target: int, team_runs: int) -> int:
    """Runs the chasing team still needs; clipped to 0 once target is passed."""
    return max(target - team_runs, 0)


def crr(team_runs: float, team_balls: float) -> float:
    """Current run rate (runs per over) from team-level ball-by-ball state."""
    return team_runs / max(team_balls, 1) * 6


def rrr(runs_needed_val: float, balls_remaining_val: float) -> float:
    """Required run rate (runs per over) for the chasing team."""
    return runs_needed_val / max(balls_remaining_val, 1) * 6


def phase(over: int) -> int:
    """
    Match phase based on over number (0-indexed).

    Returns
    -------
    0 : Powerplay  (overs 0–6)
    1 : Middle     (overs 7–15)
    2 : Death      (overs 16–19)
    """
    return int(over > 6) + int(over > 15)


def get_enc(enc_map: dict, global_mean: float, name: str) -> float:
    """Target-encoding lookup with fallback to global mean for unseen values."""
    return enc_map.get(name, global_mean)


# ── Match-level walk-forward features ────────────────────────────────────────

def compute_form_h2h(
    match_df: pd.DataFrame,
    window: int = 5,
) -> tuple[list, list, list, dict, dict]:
    """
    Walk-forward computation of recent team form and head-to-head rate.

    Iterates through matches in order; each match sees only results from
    strictly prior matches (no lookahead).

    Parameters
    ----------
    match_df : pd.DataFrame
        Must contain columns: team1, team2, team1_win.
        Must already be sorted chronologically.
    window : int
        Number of recent matches used for form computation.

    Returns
    -------
    form1, form2 : list[float]
        Rolling win rate for team1 / team2 in their last `window` matches.
    h2h_rate : list[float]
        Raw head-to-head win rate for team1 vs team2 (default 0.5 if no history).
    tw : dict
        Accumulated per-team result history (used for walk-forward 2026 eval).
    h2h : dict
        Accumulated per-pair H2H counts (used for walk-forward 2026 eval).
    """
    tw: dict = {}
    h2h: dict = {}
    form1, form2, h2h_rate = [], [], []

    for _, r in match_df.iterrows():
        t1, t2, out = r["team1"], r["team2"], r["team1_win"]
        h1 = tw.get(t1, [])
        h2 = tw.get(t2, [])
        form1.append(np.mean(h1[-window:]) if h1 else 0.5)
        form2.append(np.mean(h2[-window:]) if h2 else 0.5)
        key = frozenset([t1, t2])
        if key not in h2h:
            h2h[key] = {t1: 0, t2: 0, "n": 0}
        e = h2h[key]
        h2h_rate.append(e[t1] / e["n"] if e["n"] > 0 else 0.5)
        tw.setdefault(t1, []).append(out)
        tw.setdefault(t2, []).append(1 - out)
        e[t1 if out else t2] += 1
        e["n"] += 1

    return form1, form2, h2h_rate, tw, h2h


def compute_h2h_beta(
    match_df: pd.DataFrame,
    alpha: float = 2.0,
    beta_param: float = 2.0,
) -> list[float]:
    """
    Bayesian Beta-prior H2H win rate (Papers 47, 51).

    Raw H2H win rates are unreliable for sparse matchups. A symmetric
    Beta(alpha, beta_param) prior shrinks extreme rates toward 0.5.
    With alpha=beta_param=2, the posterior mean is (wins+2)/(n+4).

    Computed in match order with no lookahead.
    """
    h2h: dict = {}
    col = []
    for _, r in match_df.iterrows():
        t1, t2 = r["team1"], r["team2"]
        key = frozenset([t1, t2])
        if key not in h2h:
            h2h[key] = {t1: 0, t2: 0, "n": 0}
        e = h2h[key]
        col.append((e[t1] + alpha) / (e["n"] + alpha + beta_param))
        out = r["team1_win"]
        e[t1 if out else t2] += 1
        e["n"] += 1
    return col
