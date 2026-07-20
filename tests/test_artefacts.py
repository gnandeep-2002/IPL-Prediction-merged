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
    pipeline = joblib.load(PIPELINE_PATH)
    for key in ('inn1_zoo', 'inn2_zoo', 'pm_zoo', 'sc_inn1', 'sc_inn2', 'sc_pre'):
        assert key in pipeline, f'ipl_score_pipeline.pkl missing key: {key}'


@pytest.mark.skipif(
    not os.path.exists(PIPELINE_PATH),
    reason=f'{PIPELINE_PATH} not found — run src/pipeline.py first',
)
def test_score_pipeline_inn1_models_predict():
    import numpy as np
    pipeline = joblib.load(PIPELINE_PATH)
    sc       = pipeline['sc_inn1']
    sample   = np.array([[80, 3, 60, 8.0, 160.0, 20.0, 1]])
    X        = sc.transform(sample)
    for nm, mdl in pipeline['inn1_zoo'].items():
        pred = mdl.predict(X)[0]
        assert 30 < pred < 350, f'Inn1 {nm} predicted unrealistic score: {pred:.0f}'
