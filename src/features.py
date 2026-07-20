from __future__ import annotations

import numpy as np
import pandas as pd


def balls_remaining(team_balls: int, total_balls: int = 120) -> int:
    return max(total_balls - team_balls, 1)


def runs_needed(target: int, team_runs: int) -> int:
    return max(target - team_runs, 0)


def crr(team_runs: float, team_balls: float) -> float:
    return team_runs / max(team_balls, 1) * 6


def rrr(runs_needed_val: float, balls_remaining_val: float) -> float:
    return runs_needed_val / max(balls_remaining_val, 1) * 6


def phase(over: int) -> int:
    return int(over > 6) + int(over > 15)


def phase_vec(over_col: pd.Series) -> pd.Series:
    return (over_col > 6).astype(int) + (over_col > 15).astype(int)


def get_enc(enc_map: dict, global_mean: float, name: str) -> float:
    return enc_map.get(name, global_mean)


def compute_form_h2h(
    match_df: pd.DataFrame,
    window: int = 5,
) -> tuple[list, list, list, dict, dict]:
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
