from __future__ import annotations

import pandas as pd

NAME_MAP: dict[str, str] = {
    "Delhi Daredevils":             "Delhi Capitals",
    "Kings XI Punjab":              "Punjab Kings",
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
    "Rising Pune Supergiant":       "Rising Pune Supergiants",
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
    return NAME_MAP.get(name, name)


def load_ball_by_ball(xlsx_path: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name="Ball by Ball")

    for col in ["batting_team", "bowling_team", "match_winner", "toss_winner"]:
        if col in raw.columns:
            raw[col] = raw[col].replace(NAME_MAP)

    df = raw[raw["innings"].isin([1, 2]) & raw["result_type"].isna()].copy()
    df["batting_wins"] = (df["batting_team"] == df["match_winner"]).astype(int)
    return df


def build_match_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["match_id", "innings", "over", "ball"], kind="mergesort")

    null_dates = df["date"].isna()
    if null_dates.any():
        bad = sorted(df.loc[null_dates, "match_id"].unique().tolist())
        raise ValueError(f"delivery rows with missing dates in matches: {bad}")

    dates_per_match = df.groupby("match_id")["date"].nunique()
    if (dates_per_match > 1).any():
        bad = dates_per_match[dates_per_match > 1].index.tolist()
        raise ValueError(f"matches with more than one date in the source data: {bad}")

    match_df = (
        df[df["innings"] == 1]
        .groupby("match_id")
        .agg(
            team1=("batting_team", "first"),
            team2=("bowling_team", "first"),
            winner=("match_winner", "first"),
            year=("year", "first"),
            date=("date", "first"),
            venue=("venue", "first"),
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            score1=("team_runs", "last"),
        )
        .reset_index()
        .sort_values("match_id")
        .reset_index(drop=True)
    )
    match_df["date"] = pd.to_datetime(match_df["date"], errors="coerce")
    if match_df["date"].isna().any():
        bad = match_df.loc[match_df["date"].isna(), "match_id"].tolist()
        raise ValueError(f"matches with missing/unparseable dates: {bad}")

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
