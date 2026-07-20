import pandas as pd
import pytest

from src.player_features import compute_rolling_player_stats, compute_matchup_features


def _make_synthetic_balls():
    rows = []
    for i, runs in enumerate([1, 4, 0, 6, 1, 0]):
        rows.append({
            "match_id": 1, "year": 2020, "innings": 1, "over": 0, "ball": i + 1,
            "batter": "A", "non_striker": "B", "bowler": "X",
            "runs_batter": runs, "runs_extras": 0, "runs_total": runs,
            "extras_wides": 0, "is_wicket": 0, "wicket_kind": None, "player_out": None,
        })
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
    df = _make_synthetic_balls()
    stats = compute_rolling_player_stats(df)
    assert stats[2020] == {}


def test_no_lookahead_stats_only_from_prior_seasons():
    df = _make_synthetic_balls()
    stats = compute_rolling_player_stats(df)
    a_2021 = stats[2021]["A"]
    assert a_2021["bat_balls"] == 6
    assert a_2021["bat_runs"] == 1 + 4 + 0 + 6 + 1 + 0


def test_bowler_credited_only_for_bowler_wicket_kinds():
    rows = _make_synthetic_balls().to_dict("records")
    rows.append({
        "match_id": 3, "year": 2022, "innings": 1, "over": 0, "ball": 1,
        "batter": "C", "non_striker": "D", "bowler": "Y",
        "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
        "extras_wides": 0, "is_wicket": 1, "wicket_kind": "run out", "player_out": "C",
    })
    df = pd.DataFrame(rows)
    stats = compute_rolling_player_stats(df)
    stats_2023 = compute_rolling_player_stats(df)
    assert "Y" not in stats_2023.get(2022, {}) or stats_2023[2022].get("Y", {}).get("bowl_wkts", 0) == 0


def test_wides_excluded_from_legal_balls():
    rows = _make_synthetic_balls().to_dict("records")
    rows.append({
        "match_id": 1, "year": 2020, "innings": 1, "over": 0, "ball": 7,
        "batter": "A", "non_striker": "B", "bowler": "X",
        "runs_batter": 0, "runs_extras": 1, "runs_total": 1,
        "extras_wides": 1, "is_wicket": 0, "wicket_kind": None, "player_out": None,
    })
    df = pd.DataFrame(rows)
    stats = compute_rolling_player_stats(df)
    assert stats[2021]["A"]["bat_balls"] == 6


def test_matchup_features_no_lookahead():
    df = _make_synthetic_balls()
    matchups = compute_matchup_features(df)
    assert matchups[2020] == {}
    m = matchups[2021][("A", "X")]
    assert m["balls"] == 6
    assert m["runs"] == 12


def test_matchup_dismissal_recorded():
    df = _make_synthetic_balls()
    matchups = compute_matchup_features(df)
    df2 = pd.concat([df, pd.DataFrame([{
        "match_id": 3, "year": 2022, "innings": 1, "over": 0, "ball": 1,
        "batter": "Z", "non_striker": "W", "bowler": "Y",
        "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
        "extras_wides": 0, "is_wicket": 0, "wicket_kind": None, "player_out": None,
    }])], ignore_index=True)
    matchups = compute_matchup_features(df2)
    assert matchups[2022][("A", "X")]["dismissals"] == 1
