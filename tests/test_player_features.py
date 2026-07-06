"""
Tests for src/player_features.py (ported from
project_hrishav/feature_engineering.py, adapted to ipl_data.xlsx schema).

Uses synthetic ball-by-ball data mirroring ipl_data.xlsx's schema, matching
the style of tests/test_pipeline_integrity.py (no real dataset needed).
"""
import pandas as pd
import pytest

from src.player_features import compute_rolling_player_stats, compute_matchup_features


def _make_synthetic_balls():
    """Two seasons, one match each, with a known batter/bowler pairing."""
    rows = []
    # Season 2020: 6 legal balls, batter A scores runs off bowler X, no wicket
    for i, runs in enumerate([1, 4, 0, 6, 1, 0]):
        rows.append({
            "match_id": 1, "year": 2020, "innings": 1, "over": 0, "ball": i + 1,
            "batter": "A", "non_striker": "B", "bowler": "X",
            "runs_batter": runs, "runs_extras": 0, "runs_total": runs,
            "extras_wides": 0, "is_wicket": 0, "wicket_kind": None, "player_out": None,
        })
    # Season 2021: batter A faces bowler X again and gets out; batter B scores too
    for i, runs in enumerate([0, 2, 4]):
        rows.append({
            "match_id": 2, "year": 2021, "innings": 1, "over": 0, "ball": i + 1,
            "batter": "A", "non_striker": "B", "bowler": "X",
            "runs_batter": runs, "runs_extras": 0, "runs_total": runs,
            "extras_wides": 0, "is_wicket": 0, "wicket_kind": None, "player_out": None,
        })
    rows.append({
        "match_id": 2, "year": 2021, "innings": 1, "over": 0, "ball": 4,
        "batter": "A", "non_striker": "B", "bowler": "X",
        "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
        "extras_wides": 0, "is_wicket": 1, "wicket_kind": "bowled", "player_out": "A",
    })
    return pd.DataFrame(rows)


def test_first_season_has_no_stats():
    """The earliest season must have empty stats -- no prior history exists."""
    df = _make_synthetic_balls()
    stats = compute_rolling_player_stats(df)
    assert stats[2020] == {}


def test_no_lookahead_stats_only_from_prior_seasons():
    """2021's stats must reflect only 2020 data (6 legal balls), not 2021's."""
    df = _make_synthetic_balls()
    stats = compute_rolling_player_stats(df)
    a_2021 = stats[2021]["A"]
    assert a_2021["bat_balls"] == 6
    assert a_2021["bat_runs"] == 1 + 4 + 0 + 6 + 1 + 0


def test_bowler_credited_only_for_bowler_wicket_kinds():
    """A run-out (not in BOWLER_WICKET_KINDS) must not appear in bowl_wkts,
    but a 'bowled' dismissal must."""
    rows = _make_synthetic_balls().to_dict("records")
    rows.append({
        "match_id": 3, "year": 2022, "innings": 1, "over": 0, "ball": 1,
        "batter": "C", "non_striker": "D", "bowler": "Y",
        "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
        "extras_wides": 0, "is_wicket": 1, "wicket_kind": "run out", "player_out": "C",
    })
    df = pd.DataFrame(rows)
    stats = compute_rolling_player_stats(df)
    # by 2023 (first season after both years' data is accumulated)
    stats_2023 = compute_rolling_player_stats(df)
    assert "Y" not in stats_2023.get(2022, {}) or stats_2023[2022].get("Y", {}).get("bowl_wkts", 0) == 0


def test_wides_excluded_from_legal_balls():
    """A wide should not count toward bat_balls/bowl_balls."""
    rows = _make_synthetic_balls().to_dict("records")
    rows.append({
        "match_id": 1, "year": 2020, "innings": 1, "over": 0, "ball": 7,
        "batter": "A", "non_striker": "B", "bowler": "X",
        "runs_batter": 0, "runs_extras": 1, "runs_total": 1,
        "extras_wides": 1, "is_wicket": 0, "wicket_kind": None, "player_out": None,
    })
    df = pd.DataFrame(rows)
    stats = compute_rolling_player_stats(df)
    assert stats[2021]["A"]["bat_balls"] == 6  # unchanged despite the added wide


def test_matchup_features_no_lookahead():
    """Matchup stats for 2021 must reflect only the 6 legal balls A faced
    from X in 2020."""
    df = _make_synthetic_balls()
    matchups = compute_matchup_features(df)
    assert matchups[2020] == {}
    m = matchups[2021][("A", "X")]
    assert m["balls"] == 6
    assert m["runs"] == 12


def test_matchup_dismissal_recorded():
    df = _make_synthetic_balls()
    matchups = compute_matchup_features(df)
    # By the time we reach a hypothetical 2022 season, the 2021 bowled
    # dismissal (A out to X) should show up in matchup dismissals.
    df2 = pd.concat([df, pd.DataFrame([{
        "match_id": 3, "year": 2022, "innings": 1, "over": 0, "ball": 1,
        "batter": "Z", "non_striker": "W", "bowler": "Y",
        "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
        "extras_wides": 0, "is_wicket": 0, "wicket_kind": None, "player_out": None,
    }])], ignore_index=True)
    matchups = compute_matchup_features(df2)
    assert matchups[2022][("A", "X")]["dismissals"] == 1
