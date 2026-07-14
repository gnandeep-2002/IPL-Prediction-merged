"""
24-dim per-ball game-state feature vector for the alternative Transformer
model, adapted from project_hrishav/feature_engineering.py's
game_state_vector()/build_game_state_matrix() to run on project_gagan's
ipl_data.xlsx ball-by-ball schema instead of Cricsheet-derived columns.

hrishav's original row-by-row builder assumed data_loader.py had already
computed cumulative per-ball state (score_before, wickets_before,
partnership_runs, batter_sr_innings, etc.) while parsing the raw JSON.
ipl_data.xlsx does not carry those columns, so this module computes them
here with vectorised pandas groupby/cumsum operations (adapting the
approach, not just relocating the same columns).

Feature layout (indices 0-23; index 1 changed vs. the original -- DEF-005):
    0  over / 20
    1  legal_balls_before / 120  (balls bowled BEFORE this delivery -- the
       feature vector is strictly pre-ball, so it must not encode whether
       the imminent delivery is legal; legal_balls_total is still computed
       as a validation column, see DEF-005)
    2  innings==2 flag
    3-5 phase one-hot (0=Powerplay, 1=Middle, 2=Death)
    6  score_before / 250
    7  wickets_before / 10
    8  run_rate, clipped at 15/over
    9  runs in the last completed over / 30
    10 wickets in the last 5 overs (last 30 LEGAL balls, DEF-006) / 5
    11 partnership_runs / 150
    12 partnership_balls / 60
    13 batter_sr_innings / 200
    14 batter_balls_innings / 120
    15 bowler_econ_innings, clipped at 20/over
    16 bowler_wkts_innings / 10
    17 boundary_rate so far this innings
    18 dot_rate so far this innings
    19 runs_required / 200 (innings 2 only)
    20 balls_remaining / 120 (innings 2 only)
    21 required_rr, clipped at 36/over (innings 2 only)
    22 toss_won_bat (1 if the toss winner chose to bat)
    23 innings==2 flag (duplicate of index 2, kept for parity with the original)

The build is decomposed into _add_* helpers that each attach one group of
intermediate columns. They MUST run in the order build_game_state_matrix
calls them: later steps consume earlier columns (legal_balls_before feeds
the run rate, wicket window, partnership, batter/bowler, and chase-state
steps), and every step assumes the clean RangeIndex established by the
initial sort/reset (re-established by the merge in _add_last_over_runs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import phase_vec

GAME_STATE_DIM = 24


def _add_legal_ball_state(d: pd.DataFrame) -> pd.DataFrame:
    """Legal-ball accounting plus pre-ball score/wickets/run-rate state."""
    # NOTE: hrishav's original convention (Cricsheet data) treats no-balls as
    # legal ("over still advances") and only excludes wides. gagan's own
    # ipl_data.xlsx uses a DIFFERENT convention: team_balls excludes both
    # wides AND no-balls (verified empirically: team_balls increments iff
    # extras_wides==0 AND extras_noballs==0, with zero exceptions across all
    # 273,503 deliveries). Matching gagan's convention here -- not hrishav's
    # -- keeps legal_balls_total consistent with team_balls/team_wicket/
    # crr/rrr as used elsewhere in this merged pipeline (src/pipeline.py).
    d["is_legal"] = (d["extras_wides"] == 0) & (d["extras_noballs"] == 0)
    d["phase"] = phase_vec(d["over"])

    grp_inn = d.groupby(["match_id", "innings"])

    # legal_balls_total is a POSITION counter (balls bowled so far, including
    # this one) -- deliberately matching gagan's own team_balls semantics
    # (see the is_legal comment above), so it can be compared 1:1 against it.
    # DEF-005: legal_balls_total is kept ONLY as a validation column (tests
    # cross-check it against team_balls); the feature matrix uses
    # legal_balls_before, because a pre-ball state must not know whether the
    # imminent delivery will be legal.
    # legal_balls_before is the "before this ball" count, used everywhere a
    # ratio needs to pair with score_before/wickets_before (both "before this
    # ball") without mixing a pre-ball numerator against a ball-inclusive
    # denominator -- that mismatch was caught by scripts/validate_game_state.py
    # (run_rate for ball 2 of an innings coming out as score_before/2*6
    # instead of score_before/1*6, i.e. silently halved early in an innings).
    d["legal_balls_total"] = grp_inn["is_legal"].cumsum()
    d["legal_balls_before"] = d["legal_balls_total"] - d["is_legal"].astype(int)
    d["score_before"] = grp_inn["runs_total"].cumsum() - d["runs_total"]
    d["wickets_before"] = grp_inn["is_wicket"].cumsum() - d["is_wicket"]
    d["run_rate"] = d["score_before"] / d["legal_balls_before"].clip(lower=1) * 6
    return d


def _add_last_over_runs(d: pd.DataFrame) -> pd.DataFrame:
    """Runs in the last completed over: sum of runs_total in the previous over.
    (The merge rebuilds the frame; row order and the RangeIndex are preserved.)"""
    prev_over_map = d.groupby(["match_id", "innings", "over"])["runs_total"].sum().groupby(
        level=[0, 1]).shift(1).fillna(0.0)
    d = d.merge(
        prev_over_map.rename("runs_last_over").reset_index(),
        on=["match_id", "innings", "over"], how="left",
    )
    d["runs_last_over"] = d["runs_last_over"].fillna(0.0)
    return d


def _add_wicket_window(d: pd.DataFrame) -> pd.DataFrame:
    """Wickets in the last 5 overs = wickets that fell during the span of the
    last 30 LEGAL balls before this one (DEF-006: a fixed 30-ROW window
    would shrink below five overs whenever the span contains wides or
    no-balls). legal_balls_before is non-decreasing within an innings, so
    searchsorted finds the first row of the 30-legal-ball span directly.
    (d has a fresh RangeIndex -- see the sort/reset at the top of
    build_game_state_matrix -- so group index values are positional.)"""
    wk_window = np.zeros(len(d))
    for _, grp in d.groupby(["match_id", "innings"]):
        lb = grp["legal_balls_before"].to_numpy()
        wk = grp["wickets_before"].to_numpy()
        span_start = np.searchsorted(lb, lb - 30, side="left")
        wk_window[grp.index.to_numpy()] = wk - wk[span_start]
    d["wk_last_5_overs"] = wk_window
    return d


def _add_partnership_state(d: pd.DataFrame) -> pd.DataFrame:
    """Partnership: cumulative runs/balls since the last wicket in this innings."""
    wicket_group_id = d.groupby(["match_id", "innings"])["is_wicket"].cumsum().shift(1).fillna(0)
    d["_partnership_grp"] = wicket_group_id
    part_grp = d.groupby(["match_id", "innings", "_partnership_grp"])
    d["partnership_runs"] = part_grp["runs_total"].cumsum() - d["runs_total"]
    d["partnership_balls"] = part_grp["is_legal"].cumsum() - d["is_legal"].astype(int)
    return d


def _add_batter_bowler_state(d: pd.DataFrame) -> pd.DataFrame:
    """In-innings batter strike rate/balls and bowler economy/wickets, all
    'before this ball'."""
    bat_grp = d.groupby(["match_id", "innings", "batter"])
    bat_runs_before = bat_grp["runs_batter"].cumsum() - d["runs_batter"]
    bat_legal_before = bat_grp["is_legal"].cumsum() - d["is_legal"].astype(int)
    d["batter_balls_innings"] = bat_legal_before
    d["batter_sr_innings"] = (bat_runs_before / bat_legal_before.clip(lower=1) * 100).where(
        bat_legal_before > 0, 0.0)

    bowl_grp = d.groupby(["match_id", "innings", "bowler"])
    bowl_runs_before = bowl_grp["runs_total"].cumsum() - d["runs_total"]
    bowl_legal_before = bowl_grp["is_legal"].cumsum() - d["is_legal"].astype(int)
    bowl_wkts_before = bowl_grp["is_wicket"].cumsum() - d["is_wicket"]
    d["bowler_wkts_innings"] = bowl_wkts_before
    d["bowler_econ_innings"] = (bowl_runs_before / bowl_legal_before.clip(lower=1) * 6).where(
        bowl_legal_before > 0, 0.0)
    return d


def _add_boundary_dot_state(d: pd.DataFrame) -> pd.DataFrame:
    """Boundaries/dots so far this innings (before this ball)."""
    d["_is_boundary"] = d["runs_batter"].isin([4, 6]).astype(int)
    d["_is_dot"] = ((d["runs_batter"] == 0) & (d["runs_extras"] == 0)).astype(int)
    grp_inn = d.groupby(["match_id", "innings"])
    d["boundaries_before"] = grp_inn["_is_boundary"].cumsum() - d["_is_boundary"]
    d["dots_before"] = grp_inn["_is_dot"].cumsum() - d["_is_dot"]
    return d


def _add_chase_state(d: pd.DataFrame) -> pd.DataFrame:
    """2nd-innings chase state (0 for innings 1) plus the toss flag.
    balls_remaining uses legal_balls_before for the same reason run_rate
    does in _add_legal_ball_state: it must pair with runs_required, which
    is computed from score_before."""
    inn2 = d["innings"] == 2
    d["runs_required"] = np.where(inn2, (d["runs_target"] - d["score_before"]).clip(lower=0), 0.0)
    d["balls_remaining"] = np.where(inn2, (120 - d["legal_balls_before"]).clip(lower=1), 0.0)
    d["required_rr"] = np.where(inn2, d["runs_required"] / np.clip(d["balls_remaining"], 1, None) * 6, 0.0)

    # Toss: did the toss winner choose to bat (match-level, broadcast per ball).
    d["toss_won_bat"] = (d["toss_decision"] == "bat").astype(float)
    return d


def _assemble_matrix(d: pd.DataFrame) -> np.ndarray:
    """Stack the intermediate columns into the (N, 24) feature matrix."""
    lb = d["legal_balls_before"].clip(lower=1)
    inn2_f = (d["innings"] == 2).astype(float)
    phase = d["phase"]

    X = np.stack([
        d["over"] / 20.0,
        d["legal_balls_before"] / 120.0,
        inn2_f,
        (phase == 0).astype(float),
        (phase == 1).astype(float),
        (phase == 2).astype(float),
        d["score_before"] / 250.0,
        d["wickets_before"] / 10.0,
        (d["run_rate"] / 15.0).clip(upper=1.0),
        d["runs_last_over"] / 30.0,
        d["wk_last_5_overs"] / 5.0,
        d["partnership_runs"] / 150.0,
        d["partnership_balls"] / 60.0,
        d["batter_sr_innings"] / 200.0,
        d["batter_balls_innings"] / 120.0,
        (d["bowler_econ_innings"] / 20.0).clip(upper=1.0),
        d["bowler_wkts_innings"] / 10.0,
        d["boundaries_before"] / lb,
        d["dots_before"] / lb,
        d["runs_required"] / 200.0,
        d["balls_remaining"] / 120.0,
        (d["required_rr"] / 36.0).clip(upper=1.0),
        d["toss_won_bat"],
        inn2_f,
    ], axis=1).astype(np.float32)

    assert X.shape[1] == GAME_STATE_DIM
    return X


def build_game_state_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute the (N_balls, 24) game-state matrix for every ball in df
    (project_gagan's cleaned Ball-by-Ball dataframe, one row per delivery).

    Returns the feature matrix plus the input df with the intermediate
    per-ball columns attached (useful for building the win/next-ball/score
    labels alongside the features).

    Each _add_* step attaches one group of columns; see the module
    docstring for why their order is load-bearing.
    """
    d = df.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True).copy()

    d = _add_legal_ball_state(d)
    d = _add_last_over_runs(d)
    d = _add_wicket_window(d)
    d = _add_partnership_state(d)
    d = _add_batter_bowler_state(d)
    d = _add_boundary_dot_state(d)
    d = _add_chase_state(d)

    return _assemble_matrix(d), d
