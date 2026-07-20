import json
import os

import numpy as np
import pandas as pd
import pytest

from src.pipeline import match_grouped_cv, horizon_snapshot, apply_label_corrections
from src.game_state import build_game_state_matrix
from src.data import build_match_table
from src.dashboard_export import update_dashboard_data


class TestMatchGroupedCV:

    def test_no_match_is_split_across_fold_sides(self):
        match_ids = np.repeat(np.arange(20), 10)
        for train_idx, test_idx in match_grouped_cv(match_ids):
            overlap = set(match_ids[train_idx]) & set(match_ids[test_idx])
            assert overlap == set(), f"matches on both fold sides: {overlap}"

    def test_every_row_is_held_out_exactly_once(self):
        match_ids = np.repeat(np.arange(17), 7)
        held_out = np.concatenate([te for _, te in match_grouped_cv(match_ids)])
        assert sorted(held_out) == list(range(len(match_ids)))


class TestHorizonSnapshot:

    def _df(self):
        rows = []
        for b in range(1, 61):
            rows.append({"match_id": 1, "team_balls": b, "team_runs": 2 * b})
        rows.insert(30, {"match_id": 1, "team_balls": 30, "team_runs": 61})
        for b in range(1, 26):
            rows.append({"match_id": 2, "team_balls": b, "team_runs": b})
        df = pd.DataFrame(rows)
        seq = df.groupby("match_id").cumcount()
        df["over"], df["ball"] = seq // 6, seq % 6 + 1
        return df

    def test_one_row_per_innings_at_exactly_h_overs(self):
        snap = horizon_snapshot(self._df(), 5)
        assert list(snap["match_id"]) == [1]
        assert (snap["team_balls"] == 30).all()

    def test_last_row_wins_when_wides_duplicate_the_ball_count(self):
        snap = horizon_snapshot(self._df(), 5)
        assert snap["team_runs"].iloc[0] == 61

    def test_ended_innings_are_excluded_not_trivially_scored(self):
        snap10 = horizon_snapshot(self._df(), 10)
        assert 2 not in set(snap10["match_id"])

    def test_snapshot_is_independent_of_input_row_order(self):
        scrambled = self._df().sample(frac=1.0, random_state=7)
        snap = horizon_snapshot(scrambled, 5)
        assert list(snap["match_id"]) == [1]
        assert snap["team_runs"].iloc[0] == 61


class TestApplyLabelCorrections:

    def _pm(self):
        return pd.DataFrame({
            "match_id": [1, 2, 3],
            "match_winner": ["MI", np.nan, "CSK"],
        })

    def test_correction_is_applied_and_logged(self, tmp_path):
        path = tmp_path / "label_corrections.csv"
        pd.DataFrame([{"match_id": 2, "column": "match_winner", "value": "KKR",
                       "reason": "missing in source", "source": "unit test"}]).to_csv(path, index=False)
        pm = self._pm()
        applied = apply_label_corrections(pm, str(path))
        assert pm.loc[pm.match_id == 2, "match_winner"].iloc[0] == "KKR"
        assert len(applied) == 1
        assert applied[0]["old"] is None and applied[0]["new"] == "KKR"
        assert applied[0]["source"] == "unit test"

    def test_unknown_match_id_raises(self, tmp_path):
        path = tmp_path / "label_corrections.csv"
        pd.DataFrame([{"match_id": 99, "column": "match_winner", "value": "KKR",
                       "reason": "stale", "source": "unit test"}]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="99"):
            apply_label_corrections(self._pm(), str(path))

    def test_missing_file_is_a_noop(self, tmp_path):
        pm = self._pm()
        assert apply_label_corrections(pm, str(tmp_path / "absent.csv")) == []
        assert pm["match_winner"].isna().sum() == 1


def _synthetic_innings(specs):
    rows = []
    for i, (wide, wkt) in enumerate(specs):
        rows.append({
            "match_id": 1, "innings": 1, "over": i // 6, "ball": i % 6 + 1,
            "extras_wides": 1 if wide else 0, "extras_noballs": 0,
            "runs_batter": 0, "runs_extras": 1 if wide else 0,
            "runs_total": 1 if wide else 0, "is_wicket": 1 if wkt else 0,
            "batter": "A", "non_striker": "B", "bowler": "C",
            "runs_target": np.nan, "toss_decision": "bat",
        })
    return pd.DataFrame(rows)


class TestWicketWindowUsesLegalBalls:

    def test_wides_do_not_shrink_the_five_over_window(self):
        specs = [(False, True)] + [(True, False)] * 5 + [(False, False)] * 31
        _, d = build_game_state_matrix(_synthetic_innings(specs))
        assert d.loc[31, "wk_last_5_overs"] == 1
        assert d.loc[35, "legal_balls_before"] == 30
        assert d.loc[35, "wk_last_5_overs"] == 1
        assert d.loc[36, "legal_balls_before"] == 31
        assert d.loc[36, "wk_last_5_overs"] == 0

    def test_current_ball_wicket_is_not_counted(self):
        specs = [(False, False)] * 3 + [(False, True)]
        _, d = build_game_state_matrix(_synthetic_innings(specs))
        assert d.loc[3, "wk_last_5_overs"] == 0

    def test_feature_index_1_is_pre_ball(self):
        specs = [(False, False), (True, False), (False, False)]
        X, d = build_game_state_matrix(_synthetic_innings(specs))
        assert np.allclose(X[:, 1] * 120.0, d["legal_balls_before"].values)


class TestBuildMatchTable:

    def _balls(self, date="2021-04-01", date2=None):
        rows = []
        for inn in (1, 2):
            for b in range(1, 7):
                rows.append({
                    "match_id": 7, "innings": inn, "over": 0, "ball": b,
                    "batting_team": "MI" if inn == 1 else "CSK",
                    "bowling_team": "CSK" if inn == 1 else "MI",
                    "match_winner": "MI", "year": 2021,
                    "date": (date2 if (date2 and b > 3) else date),
                    "venue": "Wankhede", "toss_winner": "MI",
                    "toss_decision": "bat", "team_runs": b * 2,
                })
        return pd.DataFrame(rows)

    def test_score1_correct_even_from_scrambled_row_order(self):
        scrambled = self._balls().sample(frac=1.0, random_state=0)
        match_df = build_match_table(scrambled)
        assert match_df["score1"].iloc[0] == 12
        assert match_df["team1"].iloc[0] == "MI"

    def test_date_is_retained_and_parsed(self):
        match_df = build_match_table(self._balls())
        assert pd.api.types.is_datetime64_any_dtype(match_df["date"])
        assert match_df["date"].iloc[0] == pd.Timestamp("2021-04-01")

    def test_conflicting_dates_within_a_match_raise(self):
        with pytest.raises(ValueError, match="more than one date"):
            build_match_table(self._balls(date2="2021-04-02"))

    def test_partially_missing_dates_raise(self):
        balls = self._balls()
        balls.loc[balls.index[1:], "date"] = None
        with pytest.raises(ValueError, match="missing dates"):
            build_match_table(balls)


class TestEngineFromCheckpoint:

    def _save_ckpt(self, tmp_path, torch, registry):
        from src.transformer_model import IPLTransformer
        model = IPLTransformer()
        path = tmp_path / "ckpt.pt"
        torch.save({"state_dict": model.state_dict(),
                    "player_registry": registry, "embed_seed": 42}, path)
        return str(path)

    def test_from_checkpoint_roundtrip(self, tmp_path):
        torch = pytest.importorskip("torch")
        from src.transformer_model import INPUT_DIM
        from src.win_probability_engine import WinProbabilityEngine

        path = self._save_ckpt(tmp_path, torch, {"A": 1})
        engine = WinProbabilityEngine.from_checkpoint(path, mc_samples=3)
        feats = np.random.default_rng(0).normal(size=(10, INPUT_DIM)).astype(np.float32)
        result = engine.predict(feats)
        assert result.win_prob_mean.shape == (10,)
        assert (result.win_prob_mean >= 0).all() and (result.win_prob_mean <= 1).all()

    def test_deliveries_to_prediction_roundtrip(self, tmp_path):
        torch = pytest.importorskip("torch")
        from src.win_probability_engine import WinProbabilityEngine

        df = _synthetic_innings([(False, False)] * 12)
        path = self._save_ckpt(tmp_path, torch, {"A": 1, "B": 2, "C": 3})
        engine = WinProbabilityEngine.from_checkpoint(path, mc_samples=3)

        feats = engine.features_from_deliveries(df)
        assert set(feats) == {(1, 1)}
        assert feats[(1, 1)].shape == (12, 120)
        result = engine.predict(feats[(1, 1)])
        assert result.win_prob_mean.shape == (12,)
        assert (result.win_prob_mean >= 0).all() and (result.win_prob_mean <= 1).all()

    def test_engine_without_metadata_rejects_raw_deliveries(self):
        torch = pytest.importorskip("torch")
        from src.transformer_model import IPLTransformer
        from src.win_probability_engine import WinProbabilityEngine

        engine = WinProbabilityEngine(IPLTransformer(), mc_samples=3)
        with pytest.raises(ValueError, match="from_checkpoint"):
            engine.features_from_deliveries(_synthetic_innings([(False, False)] * 3))

    @pytest.mark.skipif(not os.path.exists("data/raw/ipl_data.xlsx"),
                        reason="data/raw/ipl_data.xlsx not found")
    def test_real_data_to_prediction_roundtrip(self, tmp_path):
        torch = pytest.importorskip("torch")
        from src.data import load_ball_by_ball
        from src.alt_transformer_data import build_player_registry
        from src.win_probability_engine import WinProbabilityEngine

        df = load_ball_by_ball("data/raw/ipl_data.xlsx")
        mid = df["match_id"].iloc[0]
        sub = df[df["match_id"] == mid]

        path = self._save_ckpt(tmp_path, torch, build_player_registry(sub))
        engine = WinProbabilityEngine.from_checkpoint(path, mc_samples=3)
        feats = engine.features_from_deliveries(sub)

        assert (mid, 1) in feats and (mid, 2) in feats
        n_inn1 = int((sub["innings"] == 1).sum())
        assert feats[(mid, 1)].shape == (n_inn1, 120)
        result = engine.predict(feats[(mid, 1)])
        assert result.win_prob_mean.shape == (n_inn1,)
        assert (result.win_prob_mean >= 0).all() and (result.win_prob_mean <= 1).all()


class TestDashboardExportRobustness:

    def test_semicolon_after_blob_on_same_line_does_not_break_parsing(self, tmp_path):
        html_path = tmp_path / "dashboard.html"
        html_path.write_text(
            "<script>\nconst DATA = " + json.dumps({"a": 1})
            + "; console.log(DATA);\n</script>"
        )
        update_dashboard_data({"b": 2}, str(html_path))
        text = html_path.read_text()
        start = text.index("const DATA = ") + len("const DATA = ")
        data, _ = json.JSONDecoder().raw_decode(text, start)
        assert data == {"a": 1, "b": 2}
        assert "console.log(DATA);" in text

    def test_json_artifact_is_written_and_summary_returned(self, tmp_path):
        html_path = tmp_path / "dashboard.html"
        html_path.write_text("<script>\nconst DATA = " + json.dumps({"keep": 1}) + ";\n</script>")
        artifact = tmp_path / "dashboard_data.json"

        summary = update_dashboard_data({"new": {"x": np.float64(1.5)}},
                                        str(html_path), json_artifact_path=str(artifact))
        assert summary == {"updated": ["new"], "retained": ["keep"]}
        blob = json.loads(artifact.read_text())
        assert blob == {"keep": 1, "new": {"x": 1.5}}
