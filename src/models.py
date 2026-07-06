"""
Model zoo builders for the IPL win probability and score prediction pipeline.

All classifiers are wrapped in CalibratedClassifierCV (isotonic regression,
3-fold CV) so that predict_proba outputs are well-calibrated probabilities.
SVM uses LinearSVC/LinearSVR for scalability on delivery-level data (~95k rows).

'XGBoost' label uses sklearn HistGradientBoosting as a drop-in equivalent —
see requirements.txt note on native xgboost installation.
"""
from __future__ import annotations

SEED: int = 42

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR


def make_zoo() -> dict:
    """
    Return a fresh dict of five calibrated classifiers for win probability.

    Keys match the display names used in all result tables and figures.
    Call this function each time you need a new set of untrained models
    (do not reuse a trained zoo for a new split).
    """
    return {
        "Logistic": CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=500, random_state=SEED),
            method="isotonic",
            cv=3,
        ),
        "Random Forest": CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED),
            method="isotonic",
            cv=3,
        ),
        "Gradient BT": CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
            method="isotonic",
            cv=3,
        ),
        "XGBoost": CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=SEED),
            method="isotonic",
            cv=3,
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=3000, random_state=SEED),
            method="isotonic",
            cv=3,
        ),
    }


def make_score_zoo() -> dict:
    """
    Return a fresh dict of five regressors for final innings score prediction.

    Evaluated at over horizons 5, 10, 15, 18 and pre-match.
    Metric: MAE (runs), RMSE, R².
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
        "SVR": LinearSVR(C=1.0, max_iter=3000),
    }
