# IPL Win Probability & Score Prediction

This is our MCM practicum project at DCU — an attempt to answer a deceptively
simple question: *given the state of an IPL cricket match at any point, what's
the probability each team wins, and what score are they likely to end up
with?*

We approach it two ways. The main path is a set of calibrated,
team-level models (logistic regression / gradient-boosted trees) that predict
win probability both before a ball is bowled and dynamically as the match
unfolds, plus a small "zoo" of regressors for projecting final score. The
second, more experimental path is a ball-by-ball Transformer that consumes
individual deliveries and player embeddings directly. The team-level path is
the one we'd actually recommend trusting — the Transformer is there for
comparison and to see how far a sequence model gets with limited data.

Everything is evaluated honestly, including where the models fall short. The
pre-match model's internal AUC sits around 0.51 (barely better than a coin
flip), and we say so in the run output rather than hiding it. If you're
looking for a project that claims to have "solved" cricket prediction, this
isn't it — it's a calibration and evaluation exercise as much as a prediction
one.

## What's actually in here

- **Pre-match model** — predicts the winner before a ball is bowled, using
  Elo rating difference, recent form, head-to-head history, and toss.
- **Dynamic in-game models** — separate models for 1st and 2nd innings that
  update win probability as the match progresses (runs needed, run rate,
  wickets in hand, over phase, etc.).
- **Score regression zoo** — Ridge, Random Forest, Gradient Boosting,
  HistGBT, and Linear SVR, evaluated at fixed 5/10/15/18-over horizons.
- **External 2026 holdout evaluation** — the 2026 season is held out
  entirely and never touched during training, so the accuracy figure it
  produces is a genuine out-of-sample check, not a fitted one.
- **Alternative path: ball-by-ball Transformer** — `src/transformer_model.py`
  + `src/win_probability_engine.py`, with Monte-Carlo dropout for
  uncertainty bands on the win probability curve. Player embeddings are
  fixed random init, not pretrained — see the docstring in
  `transformer_model.py` for why.
- **Explainability** — SHAP importances on the pre-match model.
- **Calibration diagnostics** — reliability bins, isotonic vs. temperature
  scaling comparison.
- **Elo trajectories** per team, 2008–2025.
- **Tournament simulation** — predicted vs. actual 2026 points table.
- **A results dashboard** (`dashboard/index.html`) that gets rewritten with
  fresh numbers every time you run the pipeline.

## Getting the data

The raw dataset isn't checked into this repo (it's a few hundred MB of
ball-by-ball data and just doesn't belong in git). Here's how to get it:

1. Download IPL match data from [Cricsheet](https://cricsheet.org/matches/)
   (the JSON format, all seasons you want, 2008 onward for the full range
   this project expects).
2. Drop the `.json` files into `data_pipeline/cricsheet_raw/`.
3. From `data_pipeline/`, run:
   ```bash
   python convert_to_excel.py
   ```
   This flattens everything into one `ipl_data.xlsx` with two sheets
   (`Ball by Ball` and `Match Info`). Takes about 30 seconds.
4. Move (or copy) the resulting file to `data/raw/ipl_data.xlsx` — that's
   where the pipeline expects to find it.

The `data/external_2026/` CSVs (the 2026 holdout set) are already included,
since that's a small, fixed evaluation set rather than something regenerated
from raw data.

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or, if you'd rather install it as a package:

```bash
pip install -e ".[test]"
```

## Running it

The main entry point trains and evaluates everything, prints a full report
to the console, and rewrites the dashboard:

```bash
python run_all.py
```

Add `--verbose` if you want to see library warnings and the SHAP progress
bar instead of the cleaned-up default output. A copy of the report is also
saved to `reports/run_summary.txt`, and the dashboard data is written to
`reports/dashboard_data.json` alongside the HTML. Open
`dashboard/index.html` in a browser afterwards to see it visually.

To train the alternative Transformer path separately:

```bash
python run_alt_transformer.py --epochs 8
```

This saves a checkpoint to `models/alt_transformer.pt`, which
`WinProbabilityEngine.from_checkpoint()` can reload later.

## Tests

```bash
pytest
```

The test suite covers feature engineering, Elo, game-state construction, the
score/win-probability pipeline end to end, calibration methods, dashboard
export, and a handful of regression tests for defects we found and fixed
along the way (`test_defect_fixes.py`). There's also
`scripts/validate_game_state.py`, a standalone sanity check for the
game-state matrix used by the Transformer path.

## Project layout

```
src/                  Core library: data loading, features, models, pipeline
data_pipeline/         Cricsheet JSON -> ipl_data.xlsx conversion
data/raw/               Where ipl_data.xlsx goes (not committed)
data/external_2026/    Fixed 2026 holdout set (committed)
models/                Saved model artifacts
reports/               Run summaries + dashboard JSON snapshot
dashboard/             Static HTML dashboard, data-driven
scripts/               Standalone validation/debugging scripts
tests/                 pytest suite
run_all.py             Main pipeline entry point
run_alt_transformer.py Transformer training entry point
```

## A couple of honest caveats

- The pre-match model's skill is weak (AUC ~0.51 internally). Any headline
  accuracy number on the 2026 holdout should be read next to the McNemar
  test against a naive majority-class baseline in the run output, not in
  isolation — with only ~74 matches, a single good (or bad) season can move
  the number more than genuine skill does.
- Pre-match score regression R² is expected to be *negative* — worse than
  predicting the mean. That's not a bug; there just isn't enough signal
  before a ball is bowled to project a final score, and we'd rather report
  that than quietly drop the metric.
- The Transformer's player embeddings are randomly initialised, not learned
  from a pretraining objective (no GNN pretraining, despite that being the
  more obvious approach). Treat its output as illustrative of the
  architecture, not as a stronger model than the calibrated LogReg/GBT path.

## Context

Built by Gnandeep and Hrishav as our MCM practicum at DCU, supervised by
Andrew McCarren. The project proposal and literature review live in
`../docs/` if you want the academic framing behind the modelling choices
here.
