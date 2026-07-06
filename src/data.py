"""
Data loading, cleaning, and team name normalisation for the IPL pipeline.

NAME_MAP covers historical franchise renames and defunct teams so that
all seasons (2008–2026) use consistent team identifiers.
ABBREV maps the short codes used in the 2026 external dataset back to
the same full names.
"""
from __future__ import annotations

import pandas as pd

NAME_MAP: dict[str, str] = {
    "Delhi Daredevils":             "Delhi Capitals",
    "Kings XI Punjab":              "Punjab Kings",
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
    "Rising Pune Supergiant":       "Rising Pune Supergiants",
    # defunct franchises — DEF-L01
    "Deccan Chargers":              "Sunrisers Hyderabad",
    "Pune Warriors":                "Pune Warriors",
}

ABBREV: dict[str, str] = {
    "SRH":  "Sunrisers Hyderabad",
    "RCB":  "Royal Challengers Bengaluru",
    "KKR":  "Kolkata Knight Riders",
    "MI":   "Mumbai Indians",
    "CSK":  "Chennai Super Kings",
    "RR":   "Rajasthan Royals",
    "GT":   "Gujarat Titans",
    "PBKS": "Punjab Kings",
    "LSG":  "Lucknow Super Giants",
    "DC":   "Delhi Capitals",
}


def normalise(name: str) -> str:
    """Return the canonical team name, mapping old/defunct franchise names."""
    return NAME_MAP.get(name, name)


def load_ball_by_ball(xlsx_path: str) -> pd.DataFrame:
    """
    Load and clean the 'Ball by Ball' sheet: normalise team names, keep only
    standard completed innings (1/2, no DLS/no-result), and add
    ``batting_wins`` (1 if the batting team won that match).

    Ported from project_gagan's original pipeline source, cell 4.
    """
    raw = pd.read_excel(xlsx_path, sheet_name="Ball by Ball")

    for col in ["batting_team", "bowling_team", "match_winner", "toss_winner"]:
        if col in raw.columns:
            raw[col] = raw[col].replace(NAME_MAP)

    df = raw[raw["innings"].isin([1, 2]) & raw["result_type"].isna()].copy()
    df["batting_wins"] = (df["batting_team"] == df["match_winner"]).astype(int)
    return df


def build_match_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball rows into one row per match with team1 (bat-first),
    team2 (bowl-first), scores, toss features, and the team1_win label.

    Ported from project_gagan's original pipeline source, cell 6.
    """
    match_df = (
        df[df["innings"] == 1]
        .groupby("match_id")
        .agg(
            team1=("batting_team", "first"),
            team2=("bowling_team", "first"),
            winner=("match_winner", "first"),
            year=("year", "first"),
            venue=("venue", "first"),
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            score1=("team_runs", "last"),
        )
        .reset_index()
        .sort_values("match_id")
        .reset_index(drop=True)
    )
    score2_map = df[df["innings"] == 2].groupby("match_id")["team_runs"].last()
    match_df["score2"] = match_df["match_id"].map(score2_map)
    match_df["team1_win"] = (match_df["team1"] == match_df["winner"]).astype(int)
    match_df["toss_bat_first"] = (
        (match_df["toss_winner"] == match_df["team1"])
        & (match_df["toss_decision"] == "bat")
    ).astype(int)
    match_df["toss_field_first"] = (
        (match_df["toss_winner"] == match_df["team2"])
        & (match_df["toss_decision"] == "field")
    ).astype(int)
    return match_df
