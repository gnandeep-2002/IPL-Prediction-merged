"""
DEF-C02 — Artefact integrity smoke tests.

DEF-C02: The canonical score pipeline (sklearn regressors evaluated by
         src/pipeline.py) must be saved as ipl_score_pipeline.pkl via joblib.

Note: project_gagan's supplementary MLP pipeline file (ipl_score_bundle.pkl,
Keras model) was not ported into merged_project — it is not part of the
main pipeline's reported results (see DEEP_COMPARISON.md Table 5). The
DEF-C01 bundle tests that covered that artefact were dropped accordingly.
"""
import os
import pytest
import joblib

PIPELINE_PATH = 'models/ipl_score_pipeline.pkl'


@pytest.mark.skipif(
    not os.path.exists(PIPELINE_PATH),
    reason=(
        f'{PIPELINE_PATH} not found. '
        'Run `python3 src/pipeline.py` first to generate it.'
    ),
)
def test_score_pipeline_loads():
    """ipl_score_pipeline.pkl must load and contain inn1/inn2/pm zoo + scalers."""
    pipeline = joblib.load(PIPELINE_PATH)
    for key in ('inn1_zoo', 'inn2_zoo', 'pm_zoo', 'sc_inn1', 'sc_inn2', 'sc_pre'):
        assert key in pipeline, f'ipl_score_pipeline.pkl missing key: {key}'


@pytest.mark.skipif(
    not os.path.exists(PIPELINE_PATH),
    reason=f'{PIPELINE_PATH} not found — run src/pipeline.py first',
)
def test_score_pipeline_inn1_models_predict():
    """Inn1 models in the pipeline must return a scalar prediction for one sample."""
    import numpy as np
    pipeline = joblib.load(PIPELINE_PATH)
    sc       = pipeline['sc_inn1']
    # 7 features: team_runs, team_wicket, balls_remaining, run_rate, proj_total, elo_adv, phase
    sample   = np.array([[80, 3, 60, 8.0, 160.0, 20.0, 1]])
    X        = sc.transform(sample)
    for nm, mdl in pipeline['inn1_zoo'].items():
        pred = mdl.predict(X)[0]
        assert 30 < pred < 350, f'Inn1 {nm} predicted unrealistic score: {pred:.0f}'
