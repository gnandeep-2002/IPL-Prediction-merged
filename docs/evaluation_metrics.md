# Evaluation Metrics Reference

What every metric reported by `run_all.py` measures, why it is included,
and how to read it. Metrics marked **(new)** were added in the evaluation
extension of 2026-07-14; everything else predates it. Nothing about the
models, features, splits, or calibration changed when the new metrics were
added — they are computed from the same predictions the pipeline already
made.

**Positive-class conventions** (fixed throughout the codebase):

| Evaluation | y = 1 means |
|---|---|
| Pre-match (internal, sections 1) | team1 — the bat-first side — wins |
| 2026 external holdout (section 5) | the bat-first side wins |
| Dynamic 2nd innings | the chasing side wins |
| Dynamic 1st innings | the defending side wins |

All threshold-based metrics (accuracy, precision, recall, F1, confusion
counts) use the same 0.5 probability cutoff.

## Probabilistic metrics (classification)

- **Brier score** — mean squared error between the predicted probability
  and the 0/1 outcome. Lower is better; 0.25 is the score of always
  predicting 0.5. It is a *proper scoring rule*: the model minimises it by
  reporting its true beliefs, which is why it is the primary headline
  metric for the win-probability models.
- **Log loss (new)** — the negative log-likelihood per prediction:
  `-mean(y·log p + (1−y)·log(1−p))`. Lower is better; ln 2 ≈ 0.693 is the
  score of always predicting 0.5. Also a proper scoring rule, but unlike
  Brier it punishes *confident wrongness* almost unboundedly — a model that
  says 99% and is wrong takes a far bigger hit than under Brier. Reported
  alongside Brier because the two together distinguish "slightly
  miscalibrated everywhere" (Brier and log loss move together) from
  "occasionally confidently wrong" (log loss blows up first). It is also
  the standard comparison metric in the forecasting literature.
  *Implementation note:* probabilities are clipped to [1e-15, 1−1e-15]
  before the log — isotonic calibration can output exact 0/1, and a single
  such miss would otherwise make the metric infinite. Values near or above
  0.693 mean the model is no more informative than a coin flip.
- **Brier Skill Score (BSS)** — 1 − Brier/Brier_climatology, where
  climatology always predicts the training-window base rate. Positive =
  more skilful than the base-rate forecast; 0 = no better; negative =
  actively worse. This is the number that keeps the pre-match model honest
  (it is negative there).
- **ROC-AUC** — probability that a randomly chosen positive case gets a
  higher predicted probability than a randomly chosen negative one.
  Measures *discrimination only* (ranking), completely insensitive to
  calibration. 0.5 = chance.
- **ECE (Expected Calibration Error)** — average gap between predicted
  probability and observed frequency across 10 probability bins,
  bin-weighted. Measures *calibration only*. Note it is unstable on small
  samples (the 74-match 2026 holdout spreads thinly over 10 bins — read
  that ECE with caution).

## Threshold metrics at 0.5 (classification)

- **Accuracy** — fraction of correct win/lose calls. Simple but blind to
  *which* errors are made and misleading under class imbalance (a 63.5%
  base rate makes 63.5% accuracy trivially achievable).
- **Confusion matrix (new)** — the four raw counts behind every threshold
  metric: TN, FP, FN, TP (reported in that order). Useful because a single
  aggregate can hide degenerate behaviour — e.g. the climatology baseline's
  row shows TP = FP = 0: it *never* predicts the positive class, which no
  accuracy figure would reveal on its own.
- **Precision (new)** — TP/(TP+FP): when the model calls a positive (e.g.
  "the chasing side will win"), how often is it right? The metric to watch
  when false alarms are costly.
- **Recall / Sensitivity / TPR (new)** — TP/(TP+FN): of the actual positive
  outcomes, how many did the model call? The metric to watch when misses
  are costly. "Recall", "sensitivity", and "true positive rate" are three
  names for the same quantity; the code key is `recall`.
- **Specificity / TNR (new)** — TN/(TN+FP): of the actual *negative*
  outcomes, how many did the model correctly leave alone? Sensitivity and
  specificity together describe both sides of the decision — a model can
  buy high sensitivity by calling everything positive, and specificity is
  what exposes that trade.
- **False Positive Rate (new)** — FP/(FP+TN) = 1 − specificity: how often
  a true negative gets incorrectly flagged positive. This is the x-axis of
  the ROC curve, so reporting it makes the AUC number inspectable at the
  operating threshold.
- **False Negative Rate (new)** — FN/(FN+TP) = 1 − recall: how often a
  true positive is missed. FPR and FNR are the two error rates a threshold
  trades against each other; at 0.5 they show where this model actually
  sits on that trade-off.
- **F1 (new)** — harmonic mean of precision and recall; a single balanced
  summary of the two, more informative than accuracy when classes are
  imbalanced. All of the ratio metrics above use the convention that an
  undefined ratio (empty denominator, e.g. no positive predictions or no
  positive cases) reports 0 rather than erroring — the climatology
  baseline exercises this. Note the complements (FPR = 1 − specificity,
  FNR = 1 − recall) hold whenever the denominators are non-empty; when a
  denominator is empty both members of the pair report 0.

Interpretation guidance for this project specifically: for the pre-match
model, expect precision/recall near the base rate and F1 adding little
information — the model has no demonstrated skill (see
`docs/known_limitations.md`). For the dynamic models the threshold metrics
are healthier, but the probabilistic metrics remain the primary ones: the
product's output is a *probability trajectory*, not a hard call at 0.5.

## Regression metrics (score prediction)

Reported identically at every stage — pre-match and each of the fixed
5/10/15/18-over horizons (one snapshot per innings, per DEF-002) — via a
single shared row builder, so no horizon can silently drop a metric:

- **MAE** — mean absolute error, in runs. The most interpretable: "off by
  N runs on average". Insensitive to outliers.
- **RMSE** — root mean squared error, in runs. Punishes large misses more
  than MAE; RMSE ≫ MAE signals occasional badly-missed innings rather than
  uniform error.
- **R²** — fraction of variance in final scores explained; 0 = no better
  than predicting the mean score, negative = worse than that. Negative
  pre-match R² is expected and documented.
- **n** — number of innings snapshots the row was computed from (horizons
  exclude innings that had already ended).

## Where the metrics appear

Every metric above is reported consistently in all four outputs: the
console/`reports/run_summary.txt` report, the dashboard's embedded data
(`dashboard/index.html`), the standalone artifact
`reports/dashboard_data.json`, and — for classification — the "full
classification metric set" table on the dashboard. The whole evaluation is
deterministic (SEED=42 end to end): two consecutive `run_all.py` runs
differ only in the report's timestamp line.
