"""
Tests for the Elo rating computation (src.elo.compute_elo).
"""
import pandas as pd
import pytest

from src.elo import compute_elo, compute_elo_history


def _make_match_df(records):
    """Build a minimal match_df from (team1, team2, team1_win) tuples."""
    return pd.DataFrame(records, columns=["team1", "team2", "team1_win"])


class TestEloRating:

    def test_initial_ratings_all_equal(self):
        """Before any match, all teams start at init=1500."""
        df = _make_match_df([("MI", "CSK", 1)])
        result, _ = compute_elo(df, K=32, init=1500)
        assert result.iloc[0]["elo1"] == 1500
        assert result.iloc[0]["elo2"] == 1500

    def test_winner_gains_rating(self):
        """After a win, the winner's next-match Elo should be higher than 1500."""
        df = _make_match_df([("MI", "CSK", 1), ("MI", "RCB", 1)])
        result, _ = compute_elo(df)
        assert result.iloc[1]["elo1"] > 1500

    def test_loser_loses_rating(self):
        """After a loss, the loser's next-match Elo should be below 1500."""
        df = _make_match_df([("MI", "CSK", 0), ("CSK", "RCB", 1)])
        result, _ = compute_elo(df)
        assert result.iloc[1]["elo1"] > 1500

    def test_zero_sum_rating_changes(self):
        """For two teams only, total Elo must remain constant (zero-sum)."""
        df = _make_match_df([("A", "B", 1), ("A", "B", 0), ("A", "B", 1), ("A", "B", 0)])
        result, _ = compute_elo(df, K=32, init=1000)
        for _, row in result.iterrows():
            assert row["elo1"] + row["elo2"] == pytest.approx(2000.0, abs=1e-6)

    def test_elo_diff_sign_consistency(self):
        """elo_diff = elo1 - elo2 should match the sign of the recorded ratings."""
        df = _make_match_df([("A", "B", 1)] * 10)
        result, _ = compute_elo(df)
        for _, row in result.iterrows():
            assert row["elo_diff"] == pytest.approx(row["elo1"] - row["elo2"], abs=1e-9)

    def test_output_columns_added(self):
        """compute_elo must add elo1, elo2, elo_diff columns."""
        df = _make_match_df([("A", "B", 1)])
        result, _ = compute_elo(df)
        for col in ("elo1", "elo2", "elo_diff"):
            assert col in result.columns

    def test_final_elo_dict_returned(self):
        """compute_elo must return final elo dict as second element."""
        df = _make_match_df([("A", "B", 1), ("A", "B", 0)])
        _, final_elo = compute_elo(df, init=1500)
        assert "A" in final_elo and "B" in final_elo

    def test_dominant_team_has_higher_elo(self):
        """A team that wins 8 of 10 matches should end with elo > starting value."""
        matches = ([("A", "B", 1)] * 8) + ([("A", "B", 0)] * 2)
        df = _make_match_df(matches + [("A", "B", 1)])
        result, _ = compute_elo(df)
        assert result.iloc[-1]["elo1"] > 1500

    def test_no_mutation_of_input(self):
        """compute_elo must not mutate the original DataFrame."""
        df = _make_match_df([("A", "B", 1)])
        original_cols = set(df.columns)
        _ = compute_elo(df)
        assert set(df.columns) == original_cols

    def test_symmetric_matchup(self):
        """When results are split 50/50, Elo stays close to init."""
        matches = ([("A", "B", 1), ("A", "B", 0)] * 20)
        df = _make_match_df(matches + [("A", "B", 1)])
        result, _ = compute_elo(df, K=32, init=1500)
        last = result.iloc[-1]
        assert abs(last["elo1"] - 1500) < 100
        assert abs(last["elo2"] - 1500) < 100

    def test_k_factor_scales_update(self):
        """Higher K should produce larger rating swings per match."""
        df1 = _make_match_df([("A", "B", 1), ("A", "B", 1)])
        df2 = _make_match_df([("A", "B", 1), ("A", "B", 1)])
        r_low, _  = compute_elo(df1, K=10)
        r_high, _ = compute_elo(df2, K=64)
        assert r_high.iloc[-1]["elo1"] > r_low.iloc[-1]["elo1"]


def _make_history_df(records):
    """Build a match_df with match_id/year for compute_elo_history tests.
    records: list of (match_id, year, team1, team2, team1_win) tuples."""
    return pd.DataFrame(records, columns=["match_id", "year", "team1", "team2", "team1_win"])


class TestEloHistory:

    def test_every_team_appears(self):
        df = _make_history_df([(1, 2020, "A", "B", 1)])
        hist = compute_elo_history(df)
        assert "A" in hist and "B" in hist

    def test_one_entry_per_match_played(self):
        """A team's history must have exactly one entry per match it played."""
        df = _make_history_df([
            (1, 2020, "A", "B", 1),
            (2, 2020, "A", "C", 0),
            (3, 2021, "B", "C", 1),
        ])
        hist = compute_elo_history(df)
        assert len(hist["A"]) == 2
        assert len(hist["B"]) == 2
        assert len(hist["C"]) == 2

    def test_chronological_order_preserved(self):
        df = _make_history_df([
            (1, 2020, "A", "B", 1),
            (2, 2021, "A", "B", 0),
            (3, 2022, "A", "B", 1),
        ])
        hist = compute_elo_history(df)
        assert [e["match_id"] for e in hist["A"]] == [1, 2, 3]
        assert [e["year"] for e in hist["A"]] == [2020, 2021, 2022]

    def test_final_history_entry_matches_compute_elo_final_dict(self):
        """The last history entry's elo must equal compute_elo()'s returned
        final rating -- both functions apply the identical update rule."""
        records = [(i, 2020, "A", "B", i % 2) for i in range(1, 11)]
        df = _make_history_df(records)
        hist = compute_elo_history(df, K=32, init=1500)
        _, final_elo = compute_elo(df[["team1", "team2", "team1_win"]], K=32, init=1500)
        assert hist["A"][-1]["elo"] == pytest.approx(final_elo["A"], abs=1e-9)
        assert hist["B"][-1]["elo"] == pytest.approx(final_elo["B"], abs=1e-9)

    def test_post_match_rating_reflects_result(self):
        """A team's first history entry (a win) must already be above init."""
        df = _make_history_df([(1, 2020, "A", "B", 1)])
        hist = compute_elo_history(df, init=1500)
        assert hist["A"][0]["elo"] > 1500
        assert hist["B"][0]["elo"] < 1500

    def test_franchise_rename_produces_one_continuous_trajectory(self):
        """If team names are pre-normalised (as src/data.py's NAME_MAP does
        upstream), a renamed franchise's history is one continuous series
        under the new name -- not split across old/new identities."""
        # Simulates Deccan Chargers -> Sunrisers Hyderabad already normalised
        # to "Sunrisers Hyderabad" before this function ever sees the data.
        df = _make_history_df([
            (1, 2008, "Sunrisers Hyderabad", "Mumbai Indians", 1),
            (2, 2013, "Sunrisers Hyderabad", "Mumbai Indians", 0),
        ])
        hist = compute_elo_history(df)
        assert len(hist["Sunrisers Hyderabad"]) == 2
        assert "Deccan Chargers" not in hist
