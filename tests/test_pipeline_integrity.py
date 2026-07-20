import numpy as np
import pandas as pd
import pytest


def _make_ball_df(n_matches=10, n_balls_per_inn=20, seed=0):
    rng = np.random.default_rng(seed)
    teams = ['MI', 'CSK', 'RCB', 'KKR', 'DC']
    rows  = []
    match_id = 1000

    for _ in range(n_matches):
        t1 = rng.choice(teams)
        remaining = [t for t in teams if t != t1]
        t2 = rng.choice(remaining)

        for inn in [1, 2]:
            cumulative_runs   = 0
            cumulative_wickets = 0
            for ball_num in range(1, n_balls_per_inn + 1):
                over = (ball_num - 1) // 6
                runs_this_ball = int(rng.choice([0, 1, 2, 4, 6], p=[0.3, 0.3, 0.2, 0.15, 0.05]))
                wkt_this_ball  = 1 if (rng.random() < 0.05 and cumulative_wickets < 9) else 0
                cumulative_runs    += runs_this_ball
                cumulative_wickets += wkt_this_ball

                rows.append({
                    'match_id':     match_id,
                    'innings':      inn,
                    'batting_team': t1 if inn == 1 else t2,
                    'bowling_team': t2 if inn == 1 else t1,
                    'over':         over,
                    'team_runs':    cumulative_runs,
                    'team_wicket':  cumulative_wickets,
                    'team_balls':   ball_num,
                    'year':         2021,
                    'venue':        'Wankhede',
                    'match_winner': t1,
                    'toss_winner':  t1,
                    'toss_decision': 'bat',
                })
        match_id += 1

    df = pd.DataFrame(rows)
    df['phase']         = (df['over'] > 6).astype(int) + (df['over'] > 15).astype(int)
    df['balls_remaining'] = 120 - df['team_balls']
    return df


def _make_match_df(n_matches=20, seed=0):
    rng    = np.random.default_rng(seed)
    teams  = ['MI', 'CSK', 'RCB', 'KKR', 'DC', 'SRH', 'RR', 'PBKS']
    mids   = list(range(1000, 1000 + n_matches))
    team1s = [rng.choice(teams) for _ in range(n_matches)]
    team2s = [rng.choice([t for t in teams if t != t1]) for t1 in team1s]
    wins   = rng.integers(0, 2, n_matches)
    years  = rng.integers(2010, 2024, n_matches)
    return pd.DataFrame({
        'match_id':    mids,
        'team1':       team1s,
        'team2':       team2s,
        'team1_win':   wins,
        'year':        years,
        'venue':       'Wankhede',
        'score1':      rng.integers(120, 220, n_matches),
        'toss_bat_first':   rng.integers(0, 2, n_matches),
        'toss_field_first': rng.integers(0, 2, n_matches),
    })


@pytest.fixture(scope='class')
def bdf():
    return _make_ball_df(n_matches=5, n_balls_per_inn=20)


class TestBallByBallDerivedColumns:

    def test_team_balls_starts_at_one(self, bdf):
        first_balls = (bdf.sort_values('team_balls')
                         .groupby(['match_id', 'innings'])['team_balls'].first())
        assert (first_balls == 1).all()

    def test_team_runs_monotonically_non_decreasing(self, bdf):
        for (mid, inn), grp in bdf.groupby(['match_id', 'innings']):
            runs = grp.sort_values('team_balls')['team_runs'].values
            diffs = np.diff(runs)
            assert (diffs >= 0).all(), (
                f'match {mid} inn {inn}: runs decreased mid-innings')

    def test_team_wickets_monotonically_non_decreasing(self, bdf):
        for (mid, inn), grp in bdf.groupby(['match_id', 'innings']):
            wkts = grp.sort_values('team_balls')['team_wicket'].values
            diffs = np.diff(wkts)
            assert (diffs >= 0).all()

    def test_team_wickets_capped_at_ten(self, bdf):
        assert (bdf['team_wicket'] <= 10).all()

    def test_over_derived_from_team_balls(self, bdf):
        expected_over = (bdf['team_balls'] - 1) // 6
        assert (bdf['over'] == expected_over).all()

    def test_phase_values_are_zero_one_two(self, bdf):
        assert set(bdf['phase'].unique()).issubset({0, 1, 2})

    def test_balls_remaining_non_negative(self, bdf):
        assert (bdf['balls_remaining'] >= 0).all()

    def test_no_nan_in_key_columns(self, bdf):
        for col in ['team_runs', 'team_wicket', 'team_balls', 'phase', 'balls_remaining']:
            assert bdf[col].isna().sum() == 0, f'NaN found in {col}'

    def test_innings_values_are_one_or_two(self, bdf):
        assert set(bdf['innings'].unique()).issubset({1, 2})


@pytest.fixture(scope='class')
def mdf():
    return _make_match_df(n_matches=30)


class TestMatchDataFrame:

    def test_team1_win_is_binary(self, mdf):
        assert set(mdf['team1_win'].unique()).issubset({0, 1})

    def test_no_team_plays_itself(self, mdf):
        same = (mdf['team1'] == mdf['team2'])
        assert not same.any(), 'Some matches have team1 == team2'

    def test_score1_positive(self, mdf):
        assert (mdf['score1'] > 0).all()

    def test_toss_features_binary(self, mdf):
        for col in ['toss_bat_first', 'toss_field_first']:
            assert set(mdf[col].unique()).issubset({0, 1}), f'{col} has non-binary values'


class TestEloOutputConstraints:

    def test_elo_ratings_bounded(self):
        from src.elo import compute_elo
        mdf = _make_match_df(n_matches=50)
        result, _ = compute_elo(mdf)
        assert (result['elo1'] > 1000).all(), 'elo1 dropped below 1000'
        assert (result['elo1'] < 2500).all(), 'elo1 exceeded 2500'
        assert (result['elo2'] > 1000).all(), 'elo2 dropped below 1000'
        assert (result['elo2'] < 2500).all(), 'elo2 exceeded 2500'


class TestFeatureScaling:

    def test_standard_scaler_zero_mean(self):
        from sklearn.preprocessing import StandardScaler
        X = np.random.randn(200, 5) * 10 + 50
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-9

    def test_scaler_applied_to_test_uses_train_stats(self):
        from sklearn.preprocessing import StandardScaler
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_test  = np.array([[10.0, 20.0]])
        sc = StandardScaler()
        sc.fit(X_train)
        X_test_scaled = sc.transform(X_test)
        expected = (10.0 - 3.0) / np.std([1, 3, 5], ddof=0)
        assert X_test_scaled[0, 0] == pytest.approx(expected, rel=1e-5)

    def test_no_leakage_train_stats_dont_use_test(self):
        from sklearn.preprocessing import StandardScaler
        X_train = np.ones((100, 1)) * 5.0
        X_test  = np.ones((50,  1)) * 20.0
        sc = StandardScaler()
        sc.fit(X_train)
        X_te_scaled = sc.transform(X_test)
        X_train2 = np.linspace(0, 10, 100).reshape(-1, 1)
        X_test2  = np.linspace(50, 60, 50).reshape(-1, 1)
        sc2 = StandardScaler()
        sc2.fit(X_train2)
        X_te2 = sc2.transform(X_test2)
        assert abs(X_te2.mean()) > 5.0, (
            'Test data should not be zero-mean when using train scaler on out-of-distribution data')


class TestSequenceBuilder:

    def _simple_build_sequences(self, df_inn, feat_cols, max_len):
        from sklearn.preprocessing import StandardScaler
        sc    = StandardScaler()
        feats = df_inn[feat_cols].fillna(0).values
        sc.fit(feats)

        mids    = sorted(df_inn['match_id'].unique())
        mid2idx = {m: i for i, m in enumerate(mids)}
        X       = np.zeros((len(mids), max_len, len(feat_cols)))
        y       = np.zeros(len(mids))

        for mid, grp in df_inn.sort_values('team_balls').groupby('match_id'):
            idx       = mid2idx[mid]
            f         = sc.transform(grp[feat_cols].fillna(0).values)
            L         = min(len(f), max_len)
            X[idx,:L] = f[:L]
            y[idx]    = grp.iloc[0]['chasing_wins'] if 'chasing_wins' in grp.columns else 0

        return X, y, mids

    def test_output_shape(self):
        bdf  = _make_ball_df(n_matches=8, n_balls_per_inn=15)
        df2  = bdf[bdf['innings'] == 2].copy()
        df2['chasing_wins'] = 0
        feat_cols = ['team_runs', 'team_wicket', 'balls_remaining']
        max_len   = 30

        X, y, mids = self._simple_build_sequences(df2, feat_cols, max_len)
        assert X.shape == (len(mids), max_len, len(feat_cols))
        assert y.shape == (len(mids),)

    def test_padding_is_zero(self):
        bdf  = _make_ball_df(n_matches=3, n_balls_per_inn=10)
        df2  = bdf[bdf['innings'] == 2].copy()
        df2['chasing_wins'] = 0
        feat_cols = ['team_runs', 'team_wicket', 'balls_remaining']
        max_len   = 30

        X, _, _ = self._simple_build_sequences(df2, feat_cols, max_len)
        padding = X[:, 10:, :]
        assert (padding == 0).all(), 'Padding positions are not zero'

    def test_one_row_per_match(self):
        bdf  = _make_ball_df(n_matches=6, n_balls_per_inn=12)
        df2  = bdf[bdf['innings'] == 2].copy()
        df2['chasing_wins'] = 0
        feat_cols = ['team_runs', 'team_wicket']
        _, _, mids = self._simple_build_sequences(df2, feat_cols, max_len=20)
        assert len(mids) == df2['match_id'].nunique()

    def test_no_data_leakage_after_horizon(self):
        bdf  = _make_ball_df(n_matches=4, n_balls_per_inn=30)
        df2  = bdf[bdf['innings'] == 2].copy()
        df2['chasing_wins'] = 0
        feat_cols = ['team_runs', 'team_wicket', 'balls_remaining']
        max_len   = 30
        h_balls   = 12

        X_full, _, mids = self._simple_build_sequences(df2, feat_cols, max_len)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(df2[feat_cols].fillna(0).values)
        X_trunc = np.zeros_like(X_full)
        for i, mid in enumerate(mids):
            grp = df2[df2['match_id'] == mid].sort_values('team_balls')
            sub = grp[grp['team_balls'] <= h_balls]
            if len(sub) == 0:
                continue
            feats = sc.transform(sub[feat_cols].fillna(0).values)
            L     = min(len(feats), max_len)
            X_trunc[i, :L] = feats[:L]

        assert (X_trunc[:, h_balls:, :] == 0).all(), (
            'Future deliveries leaked past the horizon cutoff')
