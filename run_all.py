"""
Run the merged IPL pipeline end-to-end: internal evaluation (train 2008-2020,
test 2021-2025), phase-specific evaluation, score regression, and the locked
2026 external holdout evaluation.

Usage: python3 run_all.py            (clean report)
       python3 run_all.py --verbose  (full detail, including library warnings)

Structure: one run_*_section() function per numbered report section, plus
export_dashboard() for the final DEF-011 push. main() is a thin orchestrator
that threads each section's results to the sections (and the dashboard
export) that consume them -- the sections are deliberately NOT independent:
e.g. the pre-match models feed SHAP (section 6) and the reliability bins
(section 8), and the external evaluation feeds the tournament simulation
(section 7).
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.pipeline import (
    load_and_prepare, train_pre_match_internal, build_dynamic_2nd,
    build_dynamic_1st, train_dynamic_internal, phase_specific_eval,
    train_score_zoo_and_save, retrain_pre_match_full, evaluate_2026_pre_match,
    compare_calibration_methods_dyn2, split_by_year,
    PRE_FEAT, DYN2,
)
from src.explainability import shap_importance
from src.tournament import actual_points_table, predicted_points_table, compare_tables
from src.elo import compute_elo_history
from src.metrics import calibration_bins
from src.dashboard_export import update_dashboard_data

DATA_XLSX = "data/raw/ipl_data.xlsx"
PM26_CSV = "data/external_2026/prematch_dataset.csv"
IG26_CSV = "data/external_2026/ingame_dataset.csv"
SCORE_PIPELINE_PATH = "models/ipl_score_pipeline.pkl"
DASHBOARD_HTML_PATH = "dashboard/index.html"
DASHBOARD_JSON_PATH = "reports/dashboard_data.json"
REPORT_PATH = "reports/run_summary.txt"

# Confirmed false positive: numpy 2.0's `matmul` raises spurious divide-by-zero
# / overflow / invalid-value RuntimeWarnings on macOS's Accelerate BLAS backend
# even for exactly-finite, exactly-zero results (reproduced directly against
# `X @ np.zeros(...)` outside of any model code -- see conversation history).
# Not a data or modelling issue, so only this exact warning text is silenced,
# never RuntimeWarning as a whole.
_BENIGN_MATMUL_WARNING = r".*encountered in matmul"

WIDTH = 78


class Report:
    """Buffers a clean, aligned run report to stdout and to REPORT_PATH.
    Formatting only -- orchestration lives in the run_*_section functions."""

    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger

    def header(self, matches: int, deliveries: int) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.info("=" * WIDTH)
        self.log.info("IPL Win-Probability & Score Prediction -- run_all.py")
        self.log.info(f"Run: {now}    Matches: {matches:,}    Deliveries: {deliveries:,}")
        self.log.info("=" * WIDTH)

    def section(self, n: int, title: str) -> None:
        label = f" {n}. {title} "
        self.log.info("")
        self.log.info(label.center(WIDTH, "-"))

    def row(self, label: str, **metrics: float | int | str) -> None:
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self.log.info(f"  {label:<20s} {'  '.join(parts)}")

    def line(self, text: str) -> None:
        self.log.info(f"  {text}")

    def note(self, text: str) -> None:
        import textwrap
        lines = textwrap.wrap(text, width=WIDTH - 10)
        for i, chunk in enumerate(lines):
            prefix = "note: " if i == 0 else "      "
            self.log.info(f"    {prefix}{chunk}")

    def footer(self, dashboard_path: str, report_path: str) -> None:
        self.log.info("")
        self.log.info("=" * WIDTH)
        self.log.info(f"Dashboard written to {dashboard_path}")
        self.log.info(f"Report saved to {report_path}")
        self.log.info("Done.")


def _setup(verbose: bool) -> Report:
    if verbose:
        warnings.simplefilter("always")
    else:
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=_BENIGN_MATMUL_WARNING)

    Path("reports").mkdir(exist_ok=True)
    logger = logging.getLogger("run_all")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(REPORT_PATH, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return Report(logger)


def run_pre_match_section(r: Report, match_df: pd.DataFrame) -> dict:
    r.section(1, "PRE-MATCH MODEL (internal: train <=2020, test >2020)")
    pre = train_pre_match_internal(match_df)
    r.row("Climatology", Brier=pre["climatology"]["brier"], AUC=pre["climatology"]["auc"])
    r.row("Cal. LR", Brier=pre["cal_lr"]["brier"], AUC=pre["cal_lr"]["auc"],
          Acc=pre["cal_lr"]["acc"], BSS=pre["cal_lr"]["bss"], ECE=pre["cal_lr"]["ece"])
    r.row("Cal. GBT", Brier=pre["cal_gbt"]["brier"], AUC=pre["cal_gbt"]["auc"],
          Acc=pre["cal_gbt"]["acc"], BSS=pre["cal_gbt"]["bss"], ECE=pre["cal_gbt"]["ece"])
    return pre


def run_dynamic_section(r: Report, df: pd.DataFrame, match_df: pd.DataFrame) -> tuple:
    r.section(2, "DYNAMIC 2ND/1ST-INNINGS MODELS (internal)")
    df2 = build_dynamic_2nd(df, match_df)
    df1 = build_dynamic_1st(df, match_df)
    dyn = train_dynamic_internal(df2, df1)
    r.row("Dynamic 2nd", Brier=dyn["dyn2"]["brier"], AUC=dyn["dyn2"]["auc"],
          Acc=dyn["dyn2"]["acc"], ECE=dyn["dyn2"]["ece"])
    r.row("Dynamic 1st", Brier=dyn["dyn1"]["brier"], AUC=dyn["dyn1"]["auc"], Acc=dyn["dyn1"]["acc"])
    return df1, df2, dyn


def run_phase_section(r: Report, dyn: dict) -> dict:
    r.section(3, "PHASE-SPECIFIC EVALUATION (2nd innings)")
    phases = phase_specific_eval(dyn["train2"], dyn["test2"], dyn["dsc2"])
    for name, m in phases.items():
        r.row(name, AUC=m["auc"], Brier=m["brier"], n=m["n"])
    return phases


def run_score_section(r: Report, df1: pd.DataFrame, df2: pd.DataFrame,
                      match_df: pd.DataFrame) -> dict:
    r.section(4, "SCORE REGRESSION ZOO (fixed horizons; saved to models/ipl_score_pipeline.pkl)")
    score_metrics = train_score_zoo_and_save(df1, df2, match_df, SCORE_PIPELINE_PATH)
    for stage_name, stage_metrics in score_metrics.items():
        r.line(f"{stage_name}:")
        for model_name, m in stage_metrics.items():
            r.log.info(f"    {model_name:<16s} MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  "
                       f"R2={m['R2']:.3f}  n={m['n']}")
    r.note("innings-stage metrics are reported at the predeclared 5/10/15/18-over "
           "horizons only (one snapshot per innings, DEF-002) -- aggregate "
           "all-delivery metrics were dominated by trivially-easy late-innings "
           "states and are no longer reported.")
    r.note("pre-match score R2 is expected to be negative (worse than predicting "
           "the mean) -- this mirrors project_gagan's own finding and is not a "
           "bug. See docs/known_limitations.md.")
    return score_metrics


def run_external_eval_section(r: Report, match_df: pd.DataFrame) -> dict:
    r.section(5, "EXTERNAL 2026 HOLDOUT EVALUATION (locked, never used above)")
    r.line("Retraining pre-match model on full 2008-2025 data first...")
    pre_model_full, sc_pre_full = retrain_pre_match_full(match_df)
    ext = evaluate_2026_pre_match(match_df, pre_model_full, sc_pre_full, PM26_CSV)
    for c in ext["corrections_applied"]:
        r.note(f"label correction applied (data/external_2026/label_corrections.csv): "
               f"match {c['match_id']} {c['column']}: {c['old']!r} -> {c['new']!r}. "
               f"Source: {c['source']}")
    r.line(f"Pre-match accuracy: {ext['accuracy']:.1%}  ({ext['k']}/{ext['n']})")
    r.line(f"95% CI (Clopper-Pearson): [{ext['ci'][0]:.1%}, {ext['ci'][1]:.1%}]")
    r.line(f"p-value vs 50/50 (exact binomial): {ext['p_value']:.3f}")
    r.line(f"Naive majority-class baseline (always pick the {ext['naive_side'].replace('_', '-')} side): "
           f"{ext['naive_baseline_acc']:.1%}")
    r.line(f"p-value vs naive baseline (exact McNemar, "
           f"{ext['mcnemar']['model_only_correct']}/{ext['mcnemar']['naive_only_correct']} discordant): "
           f"{ext['p_value_vs_naive']:.3f}")
    r.note("significance against 50/50 does not establish improvement over the "
           "naive majority baseline (DEF-009) -- the McNemar line above is the "
           "relevant comparison. Pre-match AUC is ~0.51 on internal LOSO "
           "(near-random); the 2026 accuracy figure may reflect variance over "
           "74 matches rather than genuine model skill. See docs/known_limitations.md.")
    return ext


def run_explainability_section(r: Report, pre: dict, match_df: pd.DataFrame,
                               verbose: bool) -> dict:
    r.section(6, "SHAP EXPLAINABILITY (pre-match Cal. GBT model)")
    train_m, test_m = split_by_year(match_df)
    X_tr = pre["sc"].transform(train_m[PRE_FEAT])
    X_te = pre["sc"].transform(test_m[PRE_FEAT])
    shap_result = shap_importance(pre["cal_gbt_model"], X_tr, X_te, PRE_FEAT, silent=not verbose)
    for feat, val in shap_result["importance"].items():
        r.log.info(f"  {feat:<18s} mean|SHAP|={val:.4f}")
    return shap_result


def run_tournament_section(r: Report, ext: dict) -> tuple:
    r.section(7, "2026 TOURNAMENT SIMULATION (actual vs. predicted points table)")
    ig26 = pd.read_csv(IG26_CSV)
    table_actual = actual_points_table(ext["pm26"], ig26)
    table_pred = predicted_points_table(ext["pre_df"])
    cmp = compare_tables(table_actual, table_pred)
    r.line(f"Actual table topper   : {cmp['actual_topper']}")
    r.line(f"Predicted table topper: {cmp['predicted_topper']}  "
           f"({'CORRECT' if cmp['topper_correct'] else 'WRONG'})")
    r.line(f"Top-4 overlap: {cmp['top4_overlap_count']}/4  {cmp['top4_overlap']}")
    return table_actual, table_pred, cmp


def run_reliability_section(r: Report, pre: dict, dyn: dict) -> tuple:
    r.section(8, "CALIBRATION RELIABILITY (pre-match Cal. GBT + dynamic 2nd-innings)")
    bins_pm_gbt = calibration_bins(pre["y_te"], pre["p_gbt"])
    p_dyn2_test = dyn["dyn2_model"].predict_proba(dyn["dsc2"].transform(dyn["test2"][DYN2]))[:, 1]
    bins_dyn2 = calibration_bins(dyn["test2"]["chasing_wins"].values, p_dyn2_test)
    r.line(f"Pre-match Cal. GBT: {len(bins_pm_gbt)} populated bins (see dashboard for the reliability diagram)")
    r.line(f"Dynamic 2nd innings: {len(bins_dyn2)} populated bins (see dashboard for the reliability diagram)")
    return bins_pm_gbt, bins_dyn2


def run_elo_section(r: Report, match_df: pd.DataFrame) -> dict:
    r.section(9, "TEAM ELO TRAJECTORIES (2008-2025)")
    elo_history = compute_elo_history(match_df)
    r.line(f"Computed Elo history for {len(elo_history)} teams "
           f"({sum(len(v) for v in elo_history.values())} match-level data points total)")
    return elo_history


def run_calibration_section(r: Report, df2: pd.DataFrame) -> dict:
    r.section(10, "TEMPERATURE SCALING vs. ISOTONIC CALIBRATION (dynamic 2nd innings)")
    calib_compare = compare_calibration_methods_dyn2(df2)
    t = calib_compare["temperature"]
    iso = calib_compare["isotonic"]
    r.line(f"Temperature-scaled (T={t['T']}): Brier {t['brier_raw']}->{t['brier_cal']}  "
           f"AUC {t['auc_raw']}->{t['auc_cal']}  ECE {t['ece_raw']}->{t['ece_cal']}")
    r.line(f"Isotonic (default):              Brier {iso['brier']}  AUC {iso['auc']}  ECE {iso['ece']}")
    r.note("both methods start from the same uncalibrated LogisticRegression; "
           "isotonic is this pipeline's default (see src/models.py) -- "
           "temperature scaling is shown for comparison only, not as a replacement.")
    return calib_compare


def export_dashboard(r: Report, df: pd.DataFrame, match_df: pd.DataFrame,
                     pre: dict, dyn: dict, phases: dict, score_metrics: dict,
                     ext: dict, shap_result: dict, tournament: tuple,
                     bins_pm_gbt: list, bins_dyn2: list, elo_history: dict,
                     calib_compare: dict) -> None:
    # DEF-011: every section run_all computes is pushed, so the dashboard can
    # no longer drift out of sync with the run report for computed numbers.
    # (Sections listed as "retained" below are hand-curated visualisations
    # -- trajectories, notable matches, etc. -- with no run_all counterpart.)
    table_actual, table_pred, cmp = tournament
    dash = update_dashboard_data({
        "overview": {"matches": len(match_df), "deliveries": len(df),
                     "seasons": f"{int(match_df['year'].min())}-{int(match_df['year'].max())}"},
        "pre_match": {k: pre[k] for k in ("climatology", "cal_lr", "cal_gbt")},
        "dynamic": {"dyn2": dyn["dyn2"], "dyn1": dyn["dyn1"]},
        "phases": phases,
        "score_regression": score_metrics,
        "external_2026": {"accuracy": ext["accuracy"], "k": ext["k"], "n": ext["n"],
                          "ci_lo": ext["ci"][0], "ci_hi": ext["ci"][1],
                          "p_value": ext["p_value"],
                          "p_value_vs_naive": ext["p_value_vs_naive"],
                          "naive_baseline_acc": ext["naive_baseline_acc"]},
        "shap": shap_result["importance"],
        "tournament": {"actual_table": table_actual.to_dict("records"),
                       "predicted_table": table_pred.to_dict("records"),
                       **{k: cmp[k] for k in ("actual_topper", "predicted_topper",
                                              "topper_correct", "top4_overlap",
                                              "top4_overlap_count")}},
        "pre_match_reliability": {"cal_gbt": bins_pm_gbt},
        "dyn2_reliability": {"isotonic": bins_dyn2},
        "elo_history": elo_history,
        "calibration_comparison": calib_compare,
    }, DASHBOARD_HTML_PATH, json_artifact_path=DASHBOARD_JSON_PATH)
    r.line("")
    r.line(f"Dashboard sections updated : {', '.join(dash['updated'])}")
    r.line(f"Dashboard sections retained: {', '.join(dash['retained'])}")
    r.line(f"Dashboard data artifact    : {DASHBOARD_JSON_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                         help="show full detail, including library warnings and the SHAP progress bar")
    args = parser.parse_args()

    r = _setup(args.verbose)

    df, match_df = load_and_prepare(DATA_XLSX)
    r.header(len(match_df), len(df))

    pre = run_pre_match_section(r, match_df)
    df1, df2, dyn = run_dynamic_section(r, df, match_df)
    phases = run_phase_section(r, dyn)
    score_metrics = run_score_section(r, df1, df2, match_df)
    ext = run_external_eval_section(r, match_df)
    shap_result = run_explainability_section(r, pre, match_df, verbose=args.verbose)
    tournament = run_tournament_section(r, ext)
    bins_pm_gbt, bins_dyn2 = run_reliability_section(r, pre, dyn)
    elo_history = run_elo_section(r, match_df)
    calib_compare = run_calibration_section(r, df2)

    export_dashboard(r, df, match_df, pre, dyn, phases, score_metrics, ext,
                     shap_result, tournament, bins_pm_gbt, bins_dyn2,
                     elo_history, calib_compare)

    r.footer(DASHBOARD_HTML_PATH, REPORT_PATH)


if __name__ == "__main__":
    sys.exit(main())
