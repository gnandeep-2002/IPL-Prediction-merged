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
        "SVR": LinearSVR(C=1.0, max_iter=3000, random_state=SEED),
    }
