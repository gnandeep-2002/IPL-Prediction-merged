# IPL Score & Win-Probability Prediction -- Merged Project

This project merges `project_gagan` and `project_hrishav` (two independent
IPL prediction codebases in this workspace) into a single project, keeping
whichever implementation performs better per component, per the decisions
in `../COMPARISON_REPORT.md` and `../DEEP_COMPARISON.md`. Those two files
document the full investigation (every metric verified live, every
hyperparameter cited to source); this README only summarises the outcome
and explains what came from where.

**`project_gagan/` and `project_hrishav/` are untouched** -- they remain
in the workspace as reference. All merge work lives here.

## What's the default path?

**First**, place the raw dataset at `data/raw/ipl_data.xlsx` — it's not
committed to this repo (see `.gitignore`). See "Where the raw data comes
from" below for provenance and how to regenerate an equivalent file from
scratch if you don't have the original.

```bash
pip install -r requirements.txt
python3 run_all.py
```

This runs project_gagan's pipeline end-to-end: Elo/form/H2H feature
engineering, the pre-match win-probability model, the dynamic 2nd/1st
innings models, phase-specific evaluation, the score regression zoo, the
locked-2026 external holdout evaluation, SHAP explainability, and the
2026 tournament simulation. Run `pytest tests/` to verify (128 tests).

**Why gagan as the default:** its dynamic (in-game) win-probability model
has a verified AUC of 0.878 vs. hrishav's 0.807 on their respective test
sets, plus phase-specific evaluation reaching AUC 0.953 in the death
overs. It's also the only side with score regression, a real test suite,
and explainability tooling. See `COMPARISON_REPORT.md` §2 for the full
component-by-component reasoning.

## What came from where

| Component | Source | File(s) |
|---|---|---|
| Data loading, team-name normalisation | gagan | `src/data.py` |
| Elo ratings | gagan | `src/elo.py` |
| Team-level features (form, H2H, Beta-smoothed H2H) | gagan | `src/features.py` |
| Calibration metrics (ECE, Brier Skill Score) | gagan | `src/metrics.py` |
| Win-probability & score model zoos | gagan | `src/models.py` |
| Pre-match/dynamic pipeline, 2026 holdout eval | gagan | `src/pipeline.py`, `run_all.py` |
| SHAP explainability | gagan | `src/explainability.py` |
| Tournament simulation | gagan | `src/tournament.py` |
| Temperature-scaling calibration | hrishav | `src/temperature_scaling.py` |
| Player-level rolling stats + matchup features | hrishav (adapted to gagan's schema) | `src/player_features.py` |
| 24-dim game-state vector | hrishav (adapted to gagan's schema) | `src/game_state.py` |
| Causal Transformer + embedding table (secondary model) | hrishav | `src/transformer_model.py`, `src/alt_transformer_*.py`, `run_alt_transformer.py` |
| MC-Dropout win-probability engine | hrishav | `src/win_probability_engine.py` |
| Test suite (base) | gagan | `tests/test_elo.py`, `test_features.py`, `test_metrics.py`, `test_model_sanity.py`, `test_pipeline_integrity.py`, `test_artefacts.py` |
| Test suite (new, for ported hrishav code) | this merge | `test_temperature_scaling.py`, `test_player_features.py`, `test_alt_transformer.py`, `test_win_probability_engine.py` |

**Not ported** (with reasons): hrishav's PSM Impact Player analysis
(needs a data field gagan's dataset doesn't have), gagan's LSTM (beaten by
Logistic Regression on every reported metric), hrishav's GraphSAGE GNN
(hrishav's own ablation found no benefit over random embeddings), hrishav's
Optuna-tuned baselines and native XGBoost (not the approved default path;
native XGBoost also needs Homebrew's `libomp`, which isn't always
available). Full detail in `docs/known_limitations.md`.

## Honest caveats -- read before quoting a number

- **Pre-match win prediction is close to a coin flip** (LOSO AUC ~0.51).
  The 2026 holdout's 64.9% accuracy beats the majority-class baseline by
  only 1.4 points and may not reflect genuine skill over just 74 matches.
- **Pre-match score regression is worse than predicting the mean**
  (negative R² for every model). In-game score regression is much better.
- **The alternative Transformer (`run_alt_transformer.py`) is not the
  recommended model.** It's kept as a clearly-labeled secondary path for
  comparison -- hrishav's own statistical testing found it doesn't beat a
  simple calibrated Logistic Regression, and it's trained here on
  different data than hrishav's original numbers, so don't compare its
  output to hrishav's headline Brier/AUC figures.

Full detail, including two unresolved metric discrepancies inherited from
hrishav's own reporting, is in `docs/known_limitations.md`.

## Running the alternative model

```bash
python3 run_alt_transformer.py --epochs 8
```

Trains the ported causal Transformer on gagan's data (~1.5-2 minutes for
8 epochs) and reports Brier/AUC at the final ball of each innings, purely
for comparison against the default path above.

## Project layout

```
merged_project/
├── README.md                    (this file)
├── requirements.txt
├── run_all.py                   entry point: default pipeline
├── run_alt_transformer.py       entry point: secondary Transformer model
├── data/
│   ├── raw/ipl_data.xlsx              (gagan's source, copied in)
│   └── external_2026/*.csv            (gagan's locked 2026 holdout)
├── data_pipeline/                separate raw-data reference tool, see below -- not used by run_all.py
├── models/ipl_score_pipeline.pkl      (generated by run_all.py)
├── scripts/validate_game_state.py     one-off cross-check vs. real data (see tests/test_game_state.py)
├── src/                          see "What came from where" above
├── tests/
└── docs/
    └── known_limitations.md
```

## Where the raw data comes from

`data/raw/ipl_data.xlsx` ultimately traces back to Cricsheet ball-by-ball
JSON, but the file used here has been separately built and tweaked to
fit what these models need (filtered to standard completed matches,
0-indexed overs, derived ML columns already joined in). `data_pipeline/`
contains the original raw JSON-to-Excel conversion step for provenance --
it produces a different, more raw file (see
`data_pipeline/README.md` for exactly how it differs) and is **not** a
way to regenerate or update `data/raw/ipl_data.xlsx`. It plays no part in
`run_all.py` or any other default path.
