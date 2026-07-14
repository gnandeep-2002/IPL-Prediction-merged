"""
Model zoo builder for the IPL final-score prediction pipeline.

SVR uses LinearSVR for scalability on delivery-level data (~95k rows).

'XGBoost' label uses sklearn HistGradientBoosting as a drop-in equivalent —
see requirements.txt note on native xgboost installation.

DEF-010: the former make_zoo() (five calibrated CLASSIFIERS) was removed --
it had no consumers anywhere in the pipeline. The win-probability
classifiers actually used are built inline in src/pipeline.py
(calibrated LogisticRegression/GradientBoosting with match-grouped
calibration folds).
"""
from __future__ import annotations

SEED: int = 42

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR


def make_score_zoo() -> dict:
    """
    Return a fresh dict of five regressors for final innings score prediction.

    Evaluated at the predeclared over horizons in src/pipeline.py's
    SCORE_HORIZONS (5, 10, 15, 18) plus pre-match (DEF-002: the horizons
    named here are now genuinely what train_score_zoo_and_save reports).
    Metrics: MAE (runs), RMSE, R², n.
    """
    return {
        "Linear": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=SEED
        ),
        "Gradient BT": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=SEED
        ),
        "XGBoost": HistGradientBoostingRegressor(
            max_iter=100, max_depth=5, random_state=SEED
        ),
        # random_state pins liblinear's data shuffling -- without it SVR was
        # the one zoo member whose metrics drifted between identical runs.
        "SVR": LinearSVR(C=1.0, max_iter=3000, random_state=SEED),
    }
