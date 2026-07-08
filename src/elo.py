"""
Elo rating system for IPL teams.

Ratings are updated chronologically after each match (no lookahead).
The standard chess K-factor (K=32) and initial rating (1500) are used
throughout, consistent with Papers 3, 17, and 47 in the reference list.
"""
from __future__ import annotations

import pandas as pd


def compute_elo(
    match_df: pd.DataFrame,
    K: int = 32,
    init: int = 1500,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute pre-match Elo ratings for every row in match_df.

    match_df must be sorted chronologically before calling (by match_id or date).
    Each team starts at `init`; ratings are updated using the standard
    Elo update rule after each match.

    Returns
    -------
    match_df_with_elo : pd.DataFrame
        Input DataFrame with three columns added:
        ``elo1`` (team1 pre-match rating), ``elo2`` (team2 pre-match rating),
        ``elo_diff`` (elo1 - elo2).
    final_elo : dict[str, float]
        End-of-season ratings for every team, for use in walk-forward
        evaluation on the 2026 external test set.
    """
    elo: dict[str, float] = {}
    elo1_list, elo2_list = [], []

    for _, row in match_df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        e1 = elo.get(t1, float(init))
        e2 = elo.get(t2, float(init))
        exp1 = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))
        elo1_list.append(e1)
        elo2_list.append(e2)
        outcome = row["team1_win"]
        elo[t1] = e1 + K * (outcome - exp1)
        elo[t2] = e2 + K * ((1 - outcome) - (1 - exp1))

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
    """
    Like compute_elo(), but returns each team's full chronological rating
    trajectory instead of only the final scalar -- for plotting Elo over
    time rather than just using it as a single pre-match feature.

    match_df must already have normalised team names (src/data.py's
    NAME_MAP is applied upstream in load_ball_by_ball(), before a raw
    franchise name like "Deccan Chargers" ever reaches this function) --
    otherwise a defunct franchise's history would incorrectly appear as a
    separate team from its successor instead of one continuous trajectory.

    Returns
    -------
    history : dict[str, list[dict]]
        {team_name: [{"match_id": ..., "year": ..., "elo": ...}, ...]},
        one entry per match the team played, in chronological order. `elo`
        is the team's rating AFTER that match's result is applied (so the
        first entry already reflects the team's first-ever match).
    """
    elo: dict[str, float] = {}
    history: dict[str, list[dict]] = {}

    for _, row in match_df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        e1 = elo.get(t1, float(init))
        e2 = elo.get(t2, float(init))
        exp1 = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))
        outcome = row["team1_win"]
        elo[t1] = e1 + K * (outcome - exp1)
        elo[t2] = e2 + K * ((1 - outcome) - (1 - exp1))

        for team in (t1, t2):
            history.setdefault(team, []).append({
                "match_id": int(row["match_id"]),
                "year": int(row["year"]),
                "elo": float(elo[team]),
            })

    return history
