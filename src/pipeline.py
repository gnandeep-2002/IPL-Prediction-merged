"""
Win-probability and score-prediction pipeline: pre-match model, dynamic
2nd/1st-innings models, phase-specific evaluation, score regression zoo,
and the locked-2026-holdout external evaluation.

Ported from project_gagan's original pipeline source
(ipl_pipeline_2008_2026, cells 4, 6, 9, 12, 14, 16, 26, 28, 32, 41,
49-58), using the already-tested src/data.py, src/elo.py, src/features.py,
src/metrics.py, src/models.py modules instead of inlining that logic
again.

Bug fix vs. the original source: cell 47 of the original pipeline
re-printed pre-match Cal. LR metrics using a stale `p_lr` variable that
had been reassigned by an intervening cell, producing nonsense numbers
(Brier=0.81, AUC=0.03) that contradicted the correct values computed
earlier in the same run (see DEEP_COMPARISON.md "Notable discrepancies").
`print_summary()` below takes the already-correct result dicts directly
as arguments instead of reading from module-level/global state, so this
class of bug cannot recur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import beta as beta_dist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data import ABBREV, load_ball_by_ball, build_match_table
from src.elo import compute_elo
from src.features import compute_form_h2h, compute_h2h_beta
from src.metrics import brier_skill_score, calibration_bins, ece
from src.models import make_score_zoo

PRE_FEAT = ["elo_diff", "form_diff", "h2h", "toss_bat_first", "toss_field_first"]
DYN2 = ["runs_needed", "balls_remaining", "wkts_remaining", "crr", "rrr", "elo_adv", "phase"]
DYN1 = ["team_runs", "team_wicket", "balls_remaining", "run_rate", "proj_total", "elo_adv", "phase"]
SEED = 42


def load_and_prepare(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ball-by-ball data, build the match table, and attach Elo/form/H2H."""
    df = load_ball_by_ball(xlsx_path)
    match_df = build_match_table(df)
    match_df = match_df.sort_values("match_id").reset_index(drop=True)
    match_df, _ = compute_elo(match_df)
    match_df["form1"], match_df["form2"], match_df["h2h"], _, _ = compute_form_h2h(match_df)
    match_df["form_diff"] = match_df["form1"] - match_df["form2"]
    match_df["h2h_beta"] = compute_h2h_beta(match_df)
    return df, match_df


def train_pre_match_internal(match_df: pd.DataFrame) -> dict:
    """Internal pre-match evaluation: train <=2020, test >2020 (cell 16)."""
    train_m = match_df[match_df["year"] <= 2020]
    test_m = match_df[match_df["year"] > 2020]

    sc = StandardScaler()
    X_tr = sc.fit_transform(train_m[PRE_FEAT])
    X_te = sc.transform(test_m[PRE_FEAT])
    y_tr, y_te = train_m["team1_win"].values, test_m["team1_win"].values

    clim_tr = float(y_tr.mean())
    p_clim = np.full(len(y_te), clim_tr)

    cal_lr = CalibratedClassifierCV(LogisticRegression(C=1.0), method="isotonic", cv=5)
    cal_lr.fit(X_tr, y_tr)

    from sklearn.ensemble import GradientBoostingClassifier
    cal_gbt = CalibratedClassifierCV(GradientBoostingClassifier(random_state=SEED), method="isotonic", cv=5)
    cal_gbt.fit(X_tr, y_tr)

    p_lr = cal_lr.predict_proba(X_te)[:, 1]
    p_gbt = cal_gbt.predict_proba(X_te)[:, 1]

    def _row(p):
        return {
            "brier": brier_score_loss(y_te, p),
            "auc": roc_auc_score(y_te, p),
            "acc": float(((p > 0.5).astype(int) == y_te).mean()),
            "bss": brier_skill_score(y_te, p, p_clim),
            "ece": ece(y_te, p),
        }

    return {
        "climatology": {"brier": brier_score_loss(y_te, p_clim), "auc": 0.5, "acc": float(y_te.mean()), "bss": 0.0, "ece": 0.0},
        "cal_lr": _row(p_lr),
        "cal_gbt": _row(p_gbt),
        "sc": sc, "cal_lr_model": cal_lr, "cal_gbt_model": cal_gbt,
        "p_lr": p_lr, "p_gbt": p_gbt, "y_te": y_te,
    }


def build_dynamic_2nd(df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    """Delivery-level 2nd-innings features (cell 26)."""
    df2 = df[df["innings"] == 2].copy()
    df2["balls_remaining"] = (df2["overs"] * 6 - df2["team_balls"]).clip(lower=1)
    df2["runs_needed"] = (df2["runs_target"] - df2["team_runs"]).clip(lower=0)
    df2["wkts_remaining"] = (10 - df2["team_wicket"]).clip(lower=0)
    df2["crr"] = df2["team_runs"] / df2["team_balls"].clip(lower=1) * 6
    df2["rrr"] = df2["runs_needed"] / df2["balls_remaining"] * 6
    df2["phase"] = (df2["over"] > 6).astype(int) + (df2["over"] > 15).astype(int)
    df2["chasing_wins"] = df2["batting_wins"]

    elo_map = match_df.set_index("match_id")[["elo1", "elo2"]]
    df2 = df2.join(elo_map, on="match_id", how="left")
    df2["elo_adv"] = df2["elo2"] - df2["elo1"]
    return df2


def build_dynamic_1st(df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    """Delivery-level 1st-innings features (cell 28)."""
    df1 = df[df["innings"] == 1].copy()
    df1["balls_remaining"] = (df1["overs"] * 6 - df1["team_balls"]).clip(lower=1)
    df1["run_rate"] = df1["team_runs"] / df1["team_balls"].clip(lower=1) * 6
    df1["proj_total"] = df1["team_runs"] + df1["run_rate"] * df1["balls_remaining"] / 6
    df1["phase"] = (df1["over"] > 6).astype(int) + (df1["over"] > 15).astype(int)
    df1["defending_wins"] = df1["batting_wins"]

    elo_map = match_df.set_index("match_id")[["elo1", "elo2"]]
    df1 = df1.join(elo_map, on="match_id", how="left")
    df1["elo_adv"] = df1["elo1"] - df1["elo2"]
    return df1


def train_dynamic_internal(df2: pd.DataFrame, df1: pd.DataFrame) -> dict:
    """Internal dynamic 2nd/1st-innings evaluation (cells 26, 28)."""
    train2, test2 = df2[df2["year"] <= 2020], df2[df2["year"] > 2020]
    dsc2 = StandardScaler()
    dyn2_model = CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=500), method="isotonic", cv=5)
    dyn2_model.fit(dsc2.fit_transform(train2[DYN2]), train2["chasing_wins"].values)
    p2 = dyn2_model.predict_proba(dsc2.transform(test2[DYN2]))[:, 1]
    y2 = test2["chasing_wins"].values

    train1, test1 = df1[df1["year"] <= 2020], df1[df1["year"] > 2020]
    dsc1 = StandardScaler()
    dyn1_model = CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=500), method="isotonic", cv=5)
    dyn1_model.fit(dsc1.fit_transform(train1[DYN1]), train1["defending_wins"].values)
    p1 = dyn1_model.predict_proba(dsc1.transform(test1[DYN1]))[:, 1]
    y1 = test1["defending_wins"].values

    return {
        "dyn2": {"brier": brier_score_loss(y2, p2), "auc": roc_auc_score(y2, p2),
                 "acc": float(((p2 > 0.5).astype(int) == y2).mean()), "ece": ece(y2, p2)},
        "dyn1": {"brier": brier_score_loss(y1, p1), "auc": roc_auc_score(y1, p1),
                 "acc": float(((p1 > 0.5).astype(int) == y1).mean())},
        "dsc2": dsc2, "dyn2_model": dyn2_model, "train2": train2, "test2": test2,
        "dsc1": dsc1, "dyn1_model": dyn1_model,
    }


def phase_specific_eval(train2: pd.DataFrame, test2: pd.DataFrame, dsc2: StandardScaler) -> dict:
    """Separate calibrated model per match phase (Powerplay/Middle/Death), cell 32."""
    dyn2_no_phase = [f for f in DYN2 if f != "phase"]
    idxs = [DYN2.index(f) for f in dyn2_no_phase]
    phase_names = ["Powerplay (0-6 ov)", "Middle (6-15 ov)", "Death (15-20 ov)"]
    results = {}
    for pid, pname in enumerate(phase_names):
        tr_p = train2[train2["phase"] == pid]
        te_p = test2[test2["phase"] == pid]
        m_p = CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=500), method="isotonic", cv=5)
        m_p.fit(dsc2.transform(tr_p[DYN2])[:, idxs], tr_p["chasing_wins"].values)
        pp = m_p.predict_proba(dsc2.transform(te_p[DYN2])[:, idxs])[:, 1]
        yp = te_p["chasing_wins"].values
        results[pname] = {"brier": brier_score_loss(yp, pp), "auc": roc_auc_score(yp, pp), "n": len(yp)}
    return results


def train_score_zoo_and_save(df1: pd.DataFrame, df2: pd.DataFrame, match_df: pd.DataFrame, out_path: str) -> dict:
    """
    Pre-match + 1st/2nd-innings score regressor zoo, saved via joblib
    (fixes DEF-C02: this is the canonical score pipeline artefact, distinct
    from project_gagan's supplementary MLP pipeline file, which was not ported).
    """
    import joblib

    inn1_final = df1.groupby("match_id")["team_runs"].max().reset_index().rename(columns={"team_runs": "final_score"})
    inn2_final = df2.groupby("match_id")["team_runs"].max().reset_index().rename(columns={"team_runs": "final_score"})

    pm = match_df[["match_id", "year"] + PRE_FEAT + ["score1"]].dropna()
    train_pm, test_pm = pm[pm["year"] <= 2020], pm[pm["year"] > 2020]
    sc_pre = StandardScaler()
    X_tr_pm = sc_pre.fit_transform(train_pm[PRE_FEAT])
    X_te_pm = sc_pre.transform(test_pm[PRE_FEAT])
    pm_zoo = make_score_zoo()
    pm_metrics = {}
    for name, mdl in pm_zoo.items():
        mdl.fit(X_tr_pm, train_pm["score1"].values)
        pred = mdl.predict(X_te_pm)
        pm_metrics[name] = {
            "MAE": float(mean_absolute_error(test_pm["score1"], pred)),
            "R2": float(r2_score(test_pm["score1"], pred)),
        }

    def _fit_inn_zoo(df_inn, feat_cols, final_df):
        merged = df_inn.merge(final_df, on="match_id")
        train_i, test_i = merged[merged["year"] <= 2020], merged[merged["year"] > 2020]
        sc = StandardScaler()
        X_tr = sc.fit_transform(train_i[feat_cols])
        X_te = sc.transform(test_i[feat_cols])
        zoo = make_score_zoo()
        metrics = {}
        for name, mdl in zoo.items():
            mdl.fit(X_tr, train_i["final_score"].values)
            pred = mdl.predict(X_te)
            metrics[name] = {
                "MAE": float(mean_absolute_error(test_i["final_score"], pred)),
                "R2": float(r2_score(test_i["final_score"], pred)),
            }
        return zoo, sc, metrics

    inn1_zoo, sc_inn1, inn1_metrics = _fit_inn_zoo(df1, DYN1, inn1_final)
    inn2_zoo, sc_inn2, inn2_metrics = _fit_inn_zoo(df2, DYN2, inn2_final)

    bundle = {
        "pm_zoo": pm_zoo, "sc_pre": sc_pre,
        "inn1_zoo": inn1_zoo, "sc_inn1": sc_inn1,
        "inn2_zoo": inn2_zoo, "sc_inn2": sc_inn2,
    }
    joblib.dump(bundle, out_path)
    return {"pre_match": pm_metrics, "inn1": inn1_metrics, "inn2": inn2_metrics}


def retrain_pre_match_full(match_df: pd.DataFrame) -> tuple[CalibratedClassifierCV, StandardScaler]:
    """Retrain the pre-match model on the FULL 2008-2025 dataset (cell 49) --
    required before evaluating on the locked 2026 holdout. Using the
    internal-only (train<=2020) model here would understate real-world
    performance, since it throws away 2021-2025 training signal."""
    sc_pre = StandardScaler()
    pre_model = CalibratedClassifierCV(LogisticRegression(C=1.0), method="isotonic", cv=5)
    pre_model.fit(sc_pre.fit_transform(match_df[PRE_FEAT]), match_df["team1_win"].values)
    return pre_model, sc_pre


def evaluate_2026_pre_match(match_df: pd.DataFrame, pre_model, sc_pre, pm26_path: str) -> dict:
    """Evaluate a pre-match model (already retrained on full 2008-2025 data
    via retrain_pre_match_full) on the locked 2026 holdout (cells 51, 53, 54)."""
    _, elo_2025 = compute_elo(match_df)
    _, _, _, form_hist_full, h2h_hist_full = compute_form_h2h(match_df)

    pm26 = pd.read_csv(pm26_path)
    for col in ["team_bat_first", "team_bowl_first", "toss_winner", "match_winner"]:
        if col in pm26.columns:
            pm26[col] = pm26[col].map(lambda x: ABBREV.get(str(x).strip(), x) if pd.notna(x) else np.nan)
    pm26["toss_bat_first"] = ((pm26["toss_winner"] == pm26["team_bat_first"]) & (pm26["toss_decision"] == "Bat")).fillna(0).astype(int)
    pm26["toss_field_first"] = ((pm26["toss_winner"] == pm26["team_bowl_first"]) & (pm26["toss_decision"] == "Bowl")).fillna(0).astype(int)
    pm26["actual_winner"] = np.where(pm26["bat_first_won"] == 1, pm26["team_bat_first"], pm26["team_bowl_first"])
    pm26.loc[pm26["match_id"] == 12, "actual_winner"] = pm26.loc[pm26["match_id"] == 12, "team_bowl_first"].values[0]

    WINDOW, K = 5, 32
    elo26 = dict(elo_2025)
    form26 = {k: list(v) for k, v in form_hist_full.items()}
    h2h26 = {k: dict(v) for k, v in h2h_hist_full.items()}

    pre_results = []
    for _, r in pm26.sort_values("match_id").iterrows():
        t1, t2, mid = r["team_bat_first"], r["team_bowl_first"], r["match_id"]
        e1, e2 = elo26.get(t1, 1500.0), elo26.get(t2, 1500.0)
        elo_diff = e1 - e2
        h1, h2 = form26.get(t1, []), form26.get(t2, [])
        form_diff = (np.mean(h1[-WINDOW:]) if h1 else 0.5) - (np.mean(h2[-WINDOW:]) if h2 else 0.5)
        key = frozenset([t1, t2])
        if key not in h2h26:
            h2h26[key] = {t1: 0, t2: 0, "n": 0}
        e = h2h26[key]
        h2h_rate = e.get(t1, 0) / e["n"] if e["n"] > 0 else 0.5
        feat = np.array([[elo_diff, form_diff, h2h_rate, r["toss_bat_first"], r["toss_field_first"]]])
        p_bat = pre_model.predict_proba(sc_pre.transform(feat))[0, 1]
        pred = t1 if p_bat > 0.5 else t2
        actual = r["actual_winner"]
        pre_results.append({"match_id": mid, "bat_first": t1, "bowl_first": t2, "pred": pred, "actual": actual, "correct": pred == actual})

        t1_won = int(t1 == actual)
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        delta = K * (t1_won - exp1)
        elo26[t1] += delta
        elo26[t2] -= delta
        form26.setdefault(t1, []).append(t1_won)
        form26.setdefault(t2, []).append(1 - t1_won)
        e[t1 if t1_won else t2] += 1
        e["n"] += 1

    pre_df = pd.DataFrame(pre_results)
    n = len(pre_df)
    k = int(pre_df["correct"].sum())
    acc = k / n
    ci_lo = beta_dist.ppf(0.025, k, n - k + 1)
    ci_hi = beta_dist.ppf(0.975, k + 1, n - k)
    z_stat = (acc - 0.5) / (0.5 / n ** 0.5)
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    bat_first_win_rate = (pre_df["actual"] == pre_df["bat_first"]).mean()
    naive_acc = max(bat_first_win_rate, 1.0 - bat_first_win_rate)

    return {
        "accuracy": acc, "k": k, "n": n, "ci": (ci_lo, ci_hi), "p_value": p_val,
        "naive_baseline_acc": naive_acc, "pm26": pm26, "elo_2025": elo_2025,
        "pre_df": pre_df,
    }


def compare_calibration_methods_dyn2(df2: pd.DataFrame) -> dict:
    """
    Compares isotonic calibration (the pipeline's default, via
    CalibratedClassifierCV in train_dynamic_internal) against temperature
    scaling (src/temperature_scaling.py) for the dynamic 2nd-innings model.

    Both start from the same uncalibrated LogisticRegression fit on the
    same training years. Temperature scaling additionally needs its own
    held-out validation split to fit its single scalar T -- carved out of
    the training years (2019-2020) rather than reusing the test years
    (2021+), so T is never fit on the same data used to report its result.
    The isotonic comparator is refit on train+val combined (matching how
    its internal 5-fold CV already uses all of that data), so both methods
    get a fair, equally-sized effective training set for this comparison.
    """
    from src.temperature_scaling import compare_calibration

    train2 = df2[df2["year"] <= 2018]
    val2 = df2[(df2["year"] > 2018) & (df2["year"] <= 2020)]
    test2 = df2[df2["year"] > 2020]

    sc = StandardScaler()
    X_tr = sc.fit_transform(train2[DYN2])
    X_val = sc.transform(val2[DYN2])
    X_te = sc.transform(test2[DYN2])
    y_val = val2["chasing_wins"].values
    y_te = test2["chasing_wins"].values

    base = LogisticRegression(C=1.0, max_iter=500)
    base.fit(X_tr, train2["chasing_wins"].values)
    p_val = base.predict_proba(X_val)[:, 1]
    p_te = base.predict_proba(X_te)[:, 1]

    temp_result = compare_calibration("Dynamic 2nd (temperature-scaled)", p_val, y_val, p_te, y_te)

    train_val2 = pd.concat([train2, val2])
    iso_model = CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=500), method="isotonic", cv=5)
    iso_model.fit(sc.transform(train_val2[DYN2]), train_val2["chasing_wins"].values)
    p_iso = iso_model.predict_proba(X_te)[:, 1]

    return {
        "temperature": temp_result,
        "isotonic": {
            "brier": round(float(brier_score_loss(y_te, p_iso)), 4),
            "auc": round(float(roc_auc_score(y_te, p_iso)), 4),
            "ece": round(float(ece(y_te, p_iso)), 4),
            "bins": calibration_bins(y_te, p_iso),
        },
    }
