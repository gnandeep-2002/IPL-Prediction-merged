"""
SHAP explainability for calibrated classifiers.

Ported from project_gagan's original pipeline source, cell 24. DEF-H02 fix preserved: explains
the CALIBRATED model's predict_proba output (via permutation explainer),
not the raw uncalibrated base estimator -- CalibratedClassifierCV's isotonic
layer would otherwise be invisible to the explanation.
"""
from __future__ import annotations

import numpy as np

SEED = 42


def shap_importance(model, X_background: np.ndarray, X_explain: np.ndarray,
                     feature_names: list[str], n_background: int = 80,
                     n_explain: int = 80) -> dict:
    """
    Compute mean |SHAP value| per feature for a calibrated classifier's
    positive-class probability.

    Parameters
    ----------
    model : a fitted classifier with predict_proba (e.g. CalibratedClassifierCV)
    X_background : array used to build the permutation background distribution
    X_explain : array of samples to explain
    feature_names : names matching the columns of X_background/X_explain

    Returns
    -------
    dict with keys 'shap_values' (array, n_explain x n_features) and
    'importance' (dict feature_name -> mean |SHAP value|, sorted descending).
    """
    import shap

    bg = shap.sample(X_background, min(n_background, len(X_background)), random_state=SEED)

    def _prob(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.Explainer(_prob, bg, algorithm="permutation")
    n = min(n_explain, len(X_explain))
    values = explainer(X_explain[:n]).values

    mean_abs = np.abs(values).mean(axis=0)
    importance = dict(
        sorted(zip(feature_names, mean_abs), key=lambda kv: -kv[1])
    )
    return {"shap_values": values, "importance": importance}
