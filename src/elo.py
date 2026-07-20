from __future__ import annotations

import pandas as pd


def _update(elo: dict[str, float], t1: str, t2: str, team1_win: int, K: int, init: int) -> tuple[float, float]:
    e1 = elo.get(t1, float(init))
    e2 = elo.get(t2, float(init))
    exp1 = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))
    elo[t1] = e1 + K * (team1_win - exp1)
    elo[t2] = e2 + K * ((1 - team1_win) - (1 - exp1))
    return e1, e2


def compute_elo(
    match_df: pd.DataFrame,
    K: int = 32,
    init: int = 1500,
) -> tuple[pd.DataFrame, dict[str, float]]:
    elo: dict[str, float] = {}
    elo1_list, elo2_list = [], []

    for _, row in match_df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        e1, e2 = _update(elo, t1, t2, row["team1_win"], K, init)
        elo1_list.append(e1)
        elo2_list.append(e2)

    out = match_df.copy()
    out["elo1"] = elo1_list
    out["elo2"] = elo2_list
    out["elo_diff"] = out["elo1"] - out["elo2"]
    return out, elo


def compute_elo_history(
    match_df: pd.DataFrame,
    K: int = 32,
    init: int = 1500,
) -> dict[str, list[dict]]:
    elo: dict[str, float] = {}
    history: dict[str, list[dict]] = {}

    for _, row in match_df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        _update(elo, t1, t2, row["team1_win"], K, init)

        for team in (t1, t2):
            history.setdefault(team, []).append({
                "match_id": int(row["match_id"]),
                "year": int(row["year"]),
                "elo": float(elo[team]),
            })

    return history
