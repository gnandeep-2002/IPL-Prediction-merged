import os

import numpy as np
import pandas as pd
import pytest

from src.data import load_ball_by_ball
from src.features import phase as gagan_phase
from src.game_state import build_game_state_matrix

DATA_PATH = "data/raw/ipl_data.xlsx"
pytestmark = pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f"{DATA_PATH} not found")


@pytest.fixture(scope="module")
def raw_df():
    return load_ball_by_ball(DATA_PATH)


@pytest.fixture(scope="module")
def sample_matches(raw_df):
    match_ids_sorted = sorted(raw_df["match_id"].unique())
    has_noball = raw_df[raw_df["extras_noballs"] > 0]["match_id"].iloc[0]
    has_wide = raw_df[raw_df["extras_wides"] > 0]["match_id"].iloc[0]
    picks = {
        match_ids_sorted[0],
        has_noball,
        has_wide,
        match_ids_sorted[len(match_ids_sorted) // 2],
        match_ids_sorted[-1],
    }
    return {mid: raw_df[raw_df["match_id"] == mid] for mid in picks}


def test_legal_balls_total_matches_gagans_own_team_balls(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        mismatches = (d["legal_balls_total"] != d["team_balls"]).sum()
        assert mismatches == 0, f"match {mid}: {mismatches} legal_balls_total/team_balls mismatches"


def test_first_ball_of_innings_has_zero_prior_state(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        for (m, inn), grp in d.groupby(["match_id", "innings"]):
            first = grp.sort_values(["over", "ball"]).iloc[0]
            assert first["score_before"] == 0, f"match {mid} inn {inn}: first ball score_before != 0"
            assert first["wickets_before"] == 0, f"match {mid} inn {inn}: first ball wickets_before != 0"


def test_score_before_matches_shifted_team_runs(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        for (m, inn), grp in d.groupby(["match_id", "innings"]):
            grp = grp.sort_values(["over", "ball"])
            expected = grp["team_runs"].shift(1).fillna(0)
            assert np.allclose(grp["score_before"].values, expected.values), (
                f"match {mid} inn {inn}: score_before does not match shifted team_runs")


def test_wickets_before_matches_shifted_team_wicket(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        for (m, inn), grp in d.groupby(["match_id", "innings"]):
            grp = grp.sort_values(["over", "ball"])
            expected = grp["team_wicket"].shift(1).fillna(0)
            assert np.allclose(grp["wickets_before"].values, expected.values), (
                f"match {mid} inn {inn}: wickets_before does not match shifted team_wicket")


def test_cumulative_stats_are_monotonic_and_bounded(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        for (m, inn), grp in d.groupby(["match_id", "innings"]):
            grp = grp.sort_values(["over", "ball"])
            assert (grp["legal_balls_total"].diff().dropna() >= 0).all(), f"match {mid} inn {inn}: legal_balls_total decreased"
            assert (grp["score_before"].diff().dropna() >= 0).all(), f"match {mid} inn {inn}: score_before decreased"
            assert (grp["wickets_before"].diff().dropna() >= 0).all(), f"match {mid} inn {inn}: wickets_before decreased"
            assert (grp["wickets_before"] <= 10).all(), f"match {mid} inn {inn}: wickets_before exceeded 10"


def test_phase_matches_gagans_own_phase_function(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        expected_phase = d["over"].apply(gagan_phase)
        assert (d["phase"].values == expected_phase.values).all(), f"match {mid}: phase mismatch vs src.features.phase()"


@pytest.mark.parametrize("over,expected_phase", [(0, 0), (6, 0), (7, 1), (15, 1), (16, 2), (19, 2)])
def test_phase_boundaries_exact(sample_matches, over, expected_phase):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        at_over = d[d["over"] == over]
        if at_over.empty:
            continue
        assert (at_over["phase"] == expected_phase).all(), f"match {mid}: over {over} should be phase {expected_phase}"


def test_run_rate_uses_before_ball_balls_not_including_ball(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        for (m, inn), grp in d.groupby(["match_id", "innings"]):
            grp = grp.sort_values(["over", "ball"])
            prev_runs = grp["team_runs"].shift(1).fillna(0)
            prev_balls = grp["team_balls"].shift(1).fillna(0).clip(lower=1)
            expected_run_rate = (prev_runs / prev_balls * 6).clip(upper=15.0) / 15.0
            actual_run_rate = grp["run_rate"].clip(upper=15.0) / 15.0
            assert np.allclose(actual_run_rate.values, expected_run_rate.values, atol=1e-6), (
                f"match {mid} inn {inn}: run_rate does not match crr computed from the previous ball's "
                f"team_runs/team_balls (before-ball semantics)")


def test_runs_required_and_balls_remaining_use_before_ball_state(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        for (m, inn), grp in d.groupby(["match_id", "innings"]):
            if inn != 2 or "runs_target" not in grp.columns or grp["runs_target"].isna().all():
                continue
            grp = grp.sort_values(["over", "ball"])
            target = grp["runs_target"].iloc[0]
            prev_runs = grp["team_runs"].shift(1).fillna(0)
            expected_runs_required = (target - prev_runs).clip(lower=0)
            assert np.allclose(grp["runs_required"].values, expected_runs_required.values, atol=1e-6), (
                f"match {mid} inn {inn}: runs_required does not match (target - previous ball's team_runs)")


def test_toss_won_bat_matches_toss_decision_column(sample_matches):
    for mid, sub in sample_matches.items():
        _, d = build_game_state_matrix(sub)
        for (m, inn), grp in d.groupby(["match_id", "innings"]):
            expected = float(grp["toss_decision"].iloc[0] == "bat")
            assert (grp["toss_won_bat"] == expected).all(), f"match {mid} inn {inn}: toss_won_bat mismatch"


def test_bounded_features_stay_in_unit_interval(sample_matches):
    clipped_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 17, 18, 20, 21, 22, 23]
    for mid, sub in sample_matches.items():
        X, _ = build_game_state_matrix(sub)
        for col in clipped_cols:
            assert X[:, col].min() >= -1e-6, f"match {mid}: feature idx {col} went below 0"
            assert X[:, col].max() <= 1.0 + 1e-6, f"match {mid}: feature idx {col} exceeded 1"
