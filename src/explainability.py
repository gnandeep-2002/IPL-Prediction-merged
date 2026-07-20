from __future__ import annotations

import numpy as np

SEED = 42


def shap_importance(model, X_background: np.ndarray, X_explain: np.ndarray,
                     feature_names: list[str], n_background: int = 80,
                     n_explain: int = 80, silent: bool = True) -> dict:
    import shap

    bg = shap.sample(X_background, min(n_background, len(X_background)), random_state=SEED)

    def _prob(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.Explainer(_prob, bg, algorithm="permutation", seed=SEED)
    n = min(n_explain, len(X_explain))
    values = explainer(X_explain[:n], silent=silent).values

    mean_abs = np.abs(values).mean(axis=0)
    importance = dict(
        sorted(zip(feature_names, mean_abs), key=lambda kv: -kv[1])
    )
    return {"shap_values": values, "importance": importance}
