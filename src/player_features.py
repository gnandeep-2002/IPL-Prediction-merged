"""
Player-level rolling batting/bowling stats and batter-vs-bowler matchup
features, computed with no lookahead (season Y only ever sees data from
seasons < Y).

Ported from project_hrishav/feature_engineering.py's
compute_rolling_player_stats() / compute_matchup_features(), adapted from
Cricsheet's column schema to project_gagan's ipl_data.xlsx ball-by-ball
schema (per the approved decision: keep gagan's xlsx loader only, adapt
hrishav's player-level feature computation to run on it).

STATUS (DEF-010): this is an ADDITIVE, optional feature set -- it is NOT
consumed by run_all.py's default pipeline or by the alternative
Transformer (which uses learned player embeddings instead). It is kept as
a tested reference implementation for future integration work (see the
defect report's improvement priority 6: add player availability/rolling
form only after the evaluation protocol is reliable). If it is still
unconsumed once that work lands, archive or remove it.

Column mapping vs. the original (Cricsheet -> ipl_data.xlsx):
    batter_runs   -> runs_batter
    total_runs    -> runs_total
    extra_runs    -> runs_extras
    is_wide       -> extras_wides > 0
    season_year   -> year
    phase         -> (over > 6) + (over > 15)   [same formula as src/features.phase]

Unlike the original, this module has no pickle-caching layer -- gagan's
pipeline does not re-run this on every process launch the way hrishav's
run_all.py did, so caching was not needed (kept out per the "minimum code
that solves the problem" constraint).
"""
from __future__ import annotations

import pandas as pd

from src.features import phase_vec

BOWLER_WICKET_KINDS = {"caught", "bowled", "lbw", "stumped", "caught and bowled", "hit wicket"}


def _add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_wide"] = df["extras_wides"] > 0
    df["phase"] = phase_vec(df["over"])
    return df


def compute_rolling_player_stats(df: pd.DataFrame) -> dict[int, dict[str, dict]]:
    """
    Returns {year: {player_name: {bat_runs, bat_sr, ..., bowl_econ, ...}}}.

    For year Y, stats are derived from all balls with year < Y. The first
    year has all-zero features (no history available).
    """
    df = _add_helper_columns(df)
    season_stats: dict[int, dict] = {}

    # Walk-forward: year Y sees exactly the balls from seasons < Y. A direct
    # filter states that invariant explicitly and avoids the old quadratic
    # rebuild (pd.concat of an ever-growing frame inside the loop); the
    # aggregations are all order-insensitive groupby sums/counts, so the
    # results are identical.
    for year in sorted(df["year"].unique()):
        history = df[df["year"] < year]
        season_stats[year] = {} if history.empty else _aggregate_player_stats(history)

    return season_stats


def _aggregate_player_stats(df: pd.DataFrame) -> dict[str, dict]:
    stats: dict[str, dict] = {}

    # ---- Batting stats ----
    bat_legal = df[~df["is_wide"]]
    bat_g = bat_legal.groupby("batter")

    bat_runs = bat_g["runs_batter"].sum()
    bat_balls = bat_g["runs_batter"].count()
    bat_inns = bat_legal.groupby(["batter", "match_id"])["innings"].nunique().groupby("batter").count()
    bat_4s = (bat_legal["runs_batter"] == 4).groupby(bat_legal["batter"]).sum()
    bat_6s = (bat_legal["runs_batter"] == 6).groupby(bat_legal["batter"]).sum()
    bat_outs = df[df["is_wicket"] == 1].groupby("player_out").size()

    def _phase_sr(phase_id: int) -> pd.Series:
        p = bat_legal[bat_legal["phase"] == phase_id]
        r = p.groupby("batter")["runs_batter"].sum()
        b = p.groupby("batter")["runs_batter"].count()
        return (r / b * 100).fillna(0.0)

    pp_sr, mid_sr, dth_sr = _phase_sr(0), _phase_sr(1), _phase_sr(2)

    all_batters = bat_runs.index.union(bat_outs.index)
    for player in all_batters:
        r = bat_runs.get(player, 0)
        b = bat_balls.get(player, 0)
        inn = bat_inns.get(player, 0)
        outs = bat_outs.get(player, 0)

        stats.setdefault(player, {}).update({
            "bat_innings": int(inn),
            "bat_runs": int(r),
            "bat_balls": int(b),
            "bat_avg": round(r / outs, 4) if outs > 0 else float(r),
            "bat_sr": round(r / b * 100, 4) if b > 0 else 0.0,
            "bat_4s": int(bat_4s.get(player, 0)),
            "bat_6s": int(bat_6s.get(player, 0)),
            "bat_boundary_rt": round((bat_4s.get(player, 0) + bat_6s.get(player, 0)) / b, 4) if b > 0 else 0.0,
            "bat_pp_sr": round(pp_sr.get(player, 0.0), 4),
            "bat_mid_sr": round(mid_sr.get(player, 0.0), 4),
            "bat_death_sr": round(dth_sr.get(player, 0.0), 4),
        })

    # ---- Bowling stats ----
    bowl_legal = df[~df["is_wide"]]
    bowler_wk_mask = (df["is_wicket"] == 1) & df["wicket_kind"].isin(BOWLER_WICKET_KINDS)
    bowl_wk = df[bowler_wk_mask].groupby("bowler").size()

    bowl_g = bowl_legal.groupby("bowler")
    bowl_runs = bowl_g["runs_total"].sum()
    bowl_balls = bowl_g["runs_total"].count()
    bowl_dots = (bowl_legal["runs_batter"] == 0).groupby(bowl_legal["bowler"]).sum()

    def _bowl_phase_econ(phase_id: int) -> pd.Series:
        p = bowl_legal[bowl_legal["phase"] == phase_id]
        r = p.groupby("bowler")["runs_total"].sum()
        b = p.groupby("bowler")["runs_total"].count()
        return (r / b * 6).fillna(0.0)

    pp_econ, mid_econ, dth_econ = _bowl_phase_econ(0), _bowl_phase_econ(1), _bowl_phase_econ(2)

    for player in bowl_runs.index:
        r = bowl_runs.get(player, 0)
        b = bowl_balls.get(player, 0)
        wk = bowl_wk.get(player, 0)
        dots = bowl_dots.get(player, 0)

        stats.setdefault(player, {}).update({
            "bowl_balls": int(b),
            "bowl_runs": int(r),
            "bowl_wkts": int(wk),
            "bowl_econ": round(r / b * 6, 4) if b > 0 else 0.0,
            "bowl_avg": round(r / wk, 4) if wk > 0 else (float(r) if r else 0.0),
            "bowl_sr": round(b / wk, 4) if wk > 0 else 0.0,
            "bowl_dot_rt": round(dots / b, 4) if b > 0 else 0.0,
            "bowl_pp_econ": round(pp_econ.get(player, 0.0), 4),
            "bowl_mid_econ": round(mid_econ.get(player, 0.0), 4),
            "bowl_dth_econ": round(dth_econ.get(player, 0.0), 4),
        })

    return stats


def compute_matchup_features(df: pd.DataFrame) -> dict[int, dict[tuple, dict]]:
    """
    For each year, head-to-head batter-vs-bowler stats from prior years.

    Returns {year: {(batter, bowler): {balls, runs, dismissals, boundary_rt, dot_rt, matchup_sr}}}
    """
    df = _add_helper_columns(df)
    matchup_stats: dict[int, dict] = {}

    # Same walk-forward-by-filter pattern as compute_rolling_player_stats:
    # "history strictly before year Y", stated directly, no quadratic concat.
    for year in sorted(df["year"].unique()):
        history = df[df["year"] < year]
        matchup_stats[year] = {} if history.empty else _aggregate_matchup(history)

    return matchup_stats


def _aggregate_matchup(df: pd.DataFrame) -> dict[tuple, dict]:
    legal = df[~df["is_wide"]].copy()
    legal["is_boundary"] = legal["runs_batter"].isin([4, 6]).astype(int)
    legal["is_dot"] = ((legal["runs_batter"] == 0) & (legal["runs_extras"] == 0)).astype(int)

    wk_df = df[(df["is_wicket"] == 1) & df["wicket_kind"].isin(BOWLER_WICKET_KINDS)].copy()

    g = legal.groupby(["batter", "bowler"])
    balls = g["runs_batter"].count().rename("balls")
    runs = g["runs_batter"].sum().rename("runs")
    bnd = g["is_boundary"].sum().rename("boundaries")
    dot = g["is_dot"].sum().rename("dots")
    dism = wk_df.groupby(["player_out", "bowler"]).size().rename("dismissals")
    dism.index.names = ["batter", "bowler"]

    combined = pd.concat([balls, runs, bnd, dot], axis=1).fillna(0)
    combined["dismissals"] = dism.reindex(combined.index).fillna(0).astype(int)
    combined["boundary_rt"] = (combined["boundaries"] / combined["balls"]).round(4)
    combined["dot_rt"] = (combined["dots"] / combined["balls"]).round(4)
    combined["matchup_sr"] = (combined["runs"] / combined["balls"] * 100).round(4)

    return {(batter, bowler): row.to_dict() for (batter, bowler), row in combined.iterrows()}
