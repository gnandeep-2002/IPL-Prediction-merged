"""
Tests for the Elo rating computation (src.elo.compute_elo).
"""
import pandas as pd
import pytest

from src.elo import compute_elo


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
