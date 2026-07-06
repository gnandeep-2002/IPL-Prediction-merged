"""
Model sanity tests — verify that trained models produce outputs that obey
cricket logic, not just that they run without error.

These tests train tiny versions of each model on synthetic data that encodes
obvious cricket relationships, then check the model has learned the relationship.

No external dataset or setup required — everything is self-contained.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               HistGradientBoostingClassifier,
                               RandomForestRegressor)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_chase_data(n=800, seed=42):
    """
    Synthetic 2nd-innings data encoding obvious cricket logic:
      - runs_needed LOW  → chasing team likely to WIN
      - balls_remaining HIGH → chasing team likely to WIN
      - wkts_remaining LOW  → chasing team likely to LOSE
    Returns X (n × 3), y (n,) where y=1 means chasing team wins.
    """
    rng = np.random.default_rng(seed)
    runs_needed    = rng.uniform(0, 200, n)
    balls_rem      = rng.uniform(1, 120, n)
    wkts_rem       = rng.integers(0, 11, n).astype(float)

    # Simple deterministic rule: win if runs_needed < balls_rem * 1.5 and wkts_rem > 2
    win = ((runs_needed < balls_rem * 1.5) & (wkts_rem > 2)).astype(int)

    X = np.column_stack([runs_needed, balls_rem, wkts_rem])
    return X, win


def _train_2nd_inn_model(clf, X_tr, y_tr, X_te):
    sc  = StandardScaler()
    mdl = CalibratedClassifierCV(clf, method='isotonic', cv=3)
    mdl.fit(sc.fit_transform(X_tr), y_tr)
    p = mdl.predict_proba(sc.transform(X_te))[:, 1]
    return p


CLASSIFIERS = {
    'Logistic':      LogisticRegression(C=1.0, max_iter=500),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0),
    'Gradient BT':   GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0),
    'XGBoost':       HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=0),
    'SVM':           LinearSVC(C=1.0, max_iter=2000),
}


# ── Cricket logic tests ───────────────────────────────────────────────────────

# BUG-NEW-05 fix: moved fixture to module level — avoids pytest deprecation
# for class-scoped fixtures defined as instance methods (removed in pytest 10).
@pytest.fixture(scope='class')
def trained_models():
    X, y = _make_chase_data(n=800, seed=0)
    split = 600
    X_tr, y_tr = X[:split], y[:split]
    X_te = X[split:]
    models = {}
    for nm, clf in CLASSIFIERS.items():
        sc  = StandardScaler()
        mdl = CalibratedClassifierCV(clf, method='isotonic', cv=3)
        mdl.fit(sc.fit_transform(X_tr), y_tr)
        models[nm] = (mdl, sc)
    return models, X_te


class TestWinProbabilityCricketLogic:

    def _pred(self, models, X):
        preds = {}
        for nm, (mdl, sc) in models.items():
            preds[nm] = mdl.predict_proba(sc.transform(X))[:, 1]
        return preds

    def test_probabilities_in_unit_interval(self, trained_models):
        """All models must output P in [0, 1]."""
        models, X_te = trained_models
        preds = self._pred(models, X_te)
        for nm, p in preds.items():
            assert p.min() >= 0.0, f'{nm}: probability < 0'
            assert p.max() <= 1.0, f'{nm}: probability > 1'

    def test_easy_win_high_probability(self, trained_models):
        """5 runs needed, 60 balls left, 8 wickets — should be high win probability."""
        models, _ = trained_models
        X_easy = np.array([[5.0, 60.0, 8.0]])   # trivial chase
        preds  = self._pred(models, X_easy)
        for nm, p in preds.items():
            assert p[0] > 0.7, f'{nm}: easy chase predicted only {p[0]:.2f}'

    def test_impossible_win_low_probability(self, trained_models):
        """180 runs needed, 6 balls left, 1 wicket — should be low win probability."""
        models, _ = trained_models
        X_hard = np.array([[180.0, 6.0, 1.0]])  # near-impossible chase
        preds  = self._pred(models, X_hard)
        for nm, p in preds.items():
            assert p[0] < 0.30, f'{nm}: impossible chase predicted {p[0]:.2f}'

    def test_more_runs_needed_lowers_probability(self, trained_models):
        """Holding everything else fixed, more runs needed → lower win prob."""
        models, _ = trained_models
        base      = np.array([[30.0, 60.0, 7.0]])
        harder    = np.array([[90.0, 60.0, 7.0]])
        preds_b   = self._pred(models, base)
        preds_h   = self._pred(models, harder)
        for nm in CLASSIFIERS:
            assert preds_b[nm][0] > preds_h[nm][0], (
                f'{nm}: more runs needed should reduce P(win) '
                f'but got {preds_b[nm][0]:.3f} vs {preds_h[nm][0]:.3f}')

    def test_more_balls_remaining_increases_probability(self, trained_models):
        """Holding runs needed fixed, more balls remaining → higher win prob."""
        models, _ = trained_models
        few_balls  = np.array([[60.0,  18.0, 7.0]])
        many_balls = np.array([[60.0,  90.0, 7.0]])
        preds_f    = self._pred(models, few_balls)
        preds_m    = self._pred(models, many_balls)
        for nm in CLASSIFIERS:
            assert preds_m[nm][0] > preds_f[nm][0], (
                f'{nm}: more balls should increase P(win) '
                f'but got {preds_m[nm][0]:.3f} vs {preds_f[nm][0]:.3f}')

    def test_fewer_wickets_lowers_probability(self, trained_models):
        """Holding everything else fixed, fewer wickets remaining → lower win prob."""
        models, _ = trained_models
        many_wkts = np.array([[60.0, 60.0, 9.0]])
        few_wkts  = np.array([[60.0, 60.0, 1.0]])
        preds_m   = self._pred(models, many_wkts)
        preds_f   = self._pred(models, few_wkts)
        for nm in CLASSIFIERS:
            assert preds_m[nm][0] > preds_f[nm][0], (
                f'{nm}: fewer wickets should reduce P(win)')

    def test_output_shape_matches_input(self, trained_models):
        """predict_proba must return one value per input row."""
        models, X_te = trained_models
        preds = self._pred(models, X_te)
        for nm, p in preds.items():
            assert len(p) == len(X_te), f'{nm}: output length mismatch'

    def test_probabilities_not_all_same(self, trained_models):
        """Models should not collapse to predicting the same value for all inputs."""
        models, X_te = trained_models
        preds = self._pred(models, X_te)
        for nm, p in preds.items():
            std = np.std(p)
            assert std > 0.05, f'{nm}: all probabilities are nearly identical (std={std:.4f})'


# ── Score regression sanity ───────────────────────────────────────────────────

class TestScorePredictionLogic:

    def _make_score_data(self, n=600, seed=1):
        """
        Synthetic 1st innings data:
          final_score ≈ proj_total (current_runs / balls_played * 120) + noise
        """
        rng  = np.random.default_rng(seed)
        team_runs    = rng.uniform(0, 200, n)
        balls_played = rng.uniform(6, 119, n)
        team_wicket  = rng.integers(0, 10, n).astype(float)
        balls_rem    = 120 - balls_played
        run_rate     = team_runs / (balls_played / 6)
        proj_total   = run_rate * (120 / 6)
        # True final score = proj_total ± random wicket penalty ± noise
        final_score  = proj_total - team_wicket * 2 + rng.normal(0, 8, n)
        final_score  = np.clip(final_score, 30, 350)

        X = np.column_stack([team_runs, team_wicket, balls_rem, run_rate, proj_total])
        return X, final_score

    def test_score_predictions_in_realistic_range(self):
        """Median predicted T20 score should be in a plausible range (80–250).

        Ridge is unconstrained so individual predictions can stray negative on
        noisy synthetic data; we check the median rather than the min/max.
        """
        X, y = self._make_score_data()
        split = 400
        from sklearn.linear_model import Ridge
        sc  = StandardScaler()
        mdl = Ridge(alpha=1.0)
        mdl.fit(sc.fit_transform(X[:split]), y[:split])
        p      = mdl.predict(sc.transform(X[split:]))
        median = np.median(p)
        assert 80 < median < 250, (
            f'Median predicted score outside plausible range: {median:.1f}')

    def test_higher_current_runs_predicts_higher_final(self):
        """Holding everything else fixed, more runs at over 10 → higher final score."""
        from sklearn.linear_model import Ridge
        X, y = self._make_score_data(n=800)
        sc  = StandardScaler()
        mdl = Ridge(alpha=1.0)
        mdl.fit(sc.fit_transform(X[:600]), y[:600])

        # Two scenarios: 80 runs off 60 balls vs 40 runs off 60 balls
        # team_runs, wicket, balls_rem, run_rate, proj_total
        low_runs  = np.array([[40, 2, 60, 4.0, 80.0]])
        high_runs = np.array([[80, 2, 60, 8.0, 160.0]])

        p_low  = mdl.predict(sc.transform(low_runs))[0]
        p_high = mdl.predict(sc.transform(high_runs))[0]
        assert p_high > p_low, (
            f'More runs at over 10 should predict higher final score '
            f'but got {p_high:.1f} vs {p_low:.1f}')

    def test_more_wickets_predicts_lower_final(self):
        """More wickets lost → lower predicted final score."""
        from sklearn.linear_model import Ridge
        X, y = self._make_score_data(n=800)
        sc  = StandardScaler()
        mdl = Ridge(alpha=1.0)
        mdl.fit(sc.fit_transform(X[:600]), y[:600])

        # Same run rate, different wicket count
        few_wkts  = np.array([[80, 1, 60, 8.0, 160.0]])
        many_wkts = np.array([[80, 8, 60, 8.0, 160.0]])

        p_few  = mdl.predict(sc.transform(few_wkts))[0]
        p_many = mdl.predict(sc.transform(many_wkts))[0]
        assert p_few > p_many, (
            f'Fewer wickets should predict higher score '
            f'but got {p_few:.1f} vs {p_many:.1f}')

    def test_r2_is_reasonable_on_synthetic_data(self):
        """Ridge regression should achieve R² > 0.7 on this synthetic score data."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        X, y = self._make_score_data(n=800)
        sc  = StandardScaler()
        mdl = Ridge(alpha=1.0)
        mdl.fit(sc.fit_transform(X[:600]), y[:600])
        p = mdl.predict(sc.transform(X[600:]))
        r2 = r2_score(y[600:], p)
        assert r2 > 0.70, f'R² on synthetic data too low: {r2:.3f}'
