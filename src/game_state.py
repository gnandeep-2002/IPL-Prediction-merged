from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import phase_vec

GAME_STATE_DIM = 24


def _add_legal_ball_state(d: pd.DataFrame) -> pd.DataFrame:
    d["is_legal"] = (d["extras_wides"] == 0) & (d["extras_noballs"] == 0)
    d["phase"] = phase_vec(d["over"])

    grp_inn = d.groupby(["match_id", "innings"])

    d["legal_balls_total"] = grp_inn["is_legal"].cumsum()
    d["legal_balls_before"] = d["legal_balls_total"] - d["is_legal"].astype(int)
    d["score_before"] = grp_inn["runs_total"].cumsum() - d["runs_total"]
    d["wickets_before"] = grp_inn["is_wicket"].cumsum() - d["is_wicket"]
    d["run_rate"] = d["score_before"] / d["legal_balls_before"].clip(lower=1) * 6
    return d


def _add_last_over_runs(d: pd.DataFrame) -> pd.DataFrame:
    prev_over_map = d.groupby(["match_id", "innings", "over"])["runs_total"].sum().groupby(
        level=[0, 1]).shift(1).fillna(0.0)
    d = d.merge(
        prev_over_map.rename("runs_last_over").reset_index(),
        on=["match_id", "innings", "over"], how="left",
    )
    d["runs_last_over"] = d["runs_last_over"].fillna(0.0)
    return d


def _add_wicket_window(d: pd.DataFrame) -> pd.DataFrame:
    wk_window = np.zeros(len(d))
    for _, grp in d.groupby(["match_id", "innings"]):
        lb = grp["legal_balls_before"].to_numpy()
        wk = grp["wickets_before"].to_numpy()
        span_start = np.searchsorted(lb, lb - 30, side="left")
        wk_window[grp.index.to_numpy()] = wk - wk[span_start]
    d["wk_last_5_overs"] = wk_window
    return d


def _add_partnership_state(d: pd.DataFrame) -> pd.DataFrame:
    wicket_group_id = d.groupby(["match_id", "innings"])["is_wicket"].cumsum().shift(1).fillna(0)
    d["_partnership_grp"] = wicket_group_id
    part_grp = d.groupby(["match_id", "innings", "_partnership_grp"])
    d["partnership_runs"] = part_grp["runs_total"].cumsum() - d["runs_total"]
    d["partnership_balls"] = part_grp["is_legal"].cumsum() - d["is_legal"].astype(int)
    return d


def _add_batter_bowler_state(d: pd.DataFrame) -> pd.DataFrame:
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
    d["_is_boundary"] = d["runs_batter"].isin([4, 6]).astype(int)
    d["_is_dot"] = ((d["runs_batter"] == 0) & (d["runs_extras"] == 0)).astype(int)
    grp_inn = d.groupby(["match_id", "innings"])
    d["boundaries_before"] = grp_inn["_is_boundary"].cumsum() - d["_is_boundary"]
    d["dots_before"] = grp_inn["_is_dot"].cumsum() - d["_is_dot"]
    return d


def _add_chase_state(d: pd.DataFrame) -> pd.DataFrame:
    inn2 = d["innings"] == 2
    d["runs_required"] = np.where(inn2, (d["runs_target"] - d["score_before"]).clip(lower=0), 0.0)
    d["balls_remaining"] = np.where(inn2, (120 - d["legal_balls_before"]).clip(lower=1), 0.0)
    d["required_rr"] = np.where(inn2, d["runs_required"] / np.clip(d["balls_remaining"], 1, None) * 6, 0.0)

    d["toss_won_bat"] = (d["toss_decision"] == "bat").astype(float)
    return d


def _assemble_matrix(d: pd.DataFrame) -> np.ndarray:
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
    d = df.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True).copy()

    d = _add_legal_ball_state(d)
    d = _add_last_over_runs(d)
    d = _add_wicket_window(d)
    d = _add_partnership_state(d)
    d = _add_batter_bowler_state(d)
    d = _add_boundary_dot_state(d)
    d = _add_chase_state(d)

    return _assemble_matrix(d), d
