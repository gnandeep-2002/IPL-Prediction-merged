from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import beta as beta_dist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data import ABBREV, load_ball_by_ball, build_match_table
from src.elo import compute_elo
from src.features import compute_form_h2h, compute_h2h_beta, phase_vec
from src.metrics import brier_skill_score, calibration_bins, ece
from src.models import make_score_zoo

PRE_FEAT = ["elo_diff", "form_diff", "h2h", "toss_bat_first", "toss_field_first"]
DYN2 = ["runs_needed", "balls_remaining", "wkts_remaining", "crr", "rrr", "elo_adv", "phase"]
DYN1 = ["team_runs", "team_wicket", "balls_remaining", "run_rate", "proj_total", "elo_adv", "phase"]
SEED = 42

SCORE_HORIZONS = (5, 10, 15, 18)


def match_grouped_cv(match_ids, n_splits: int = 5) -> list:
    groups = np.asarray(match_ids)
    dummy_X = np.zeros((len(groups), 1))
    return list(GroupKFold(n_splits=n_splits).split(dummy_X, groups=groups))


def split_by_year(df: pd.DataFrame, train_end_year: int = 2020) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["year"] <= train_end_year], df[df["year"] > train_end_year]


def binary_log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def classification_metrics(y, p, p_ref=None, with_ece: bool = True) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pred = (p > 0.5).astype(int)

    row = {
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5,
        "acc": float((pred == y).mean()),
    }
    if p_ref is not None:
        row["bss"] = brier_skill_score(y, p, p_ref)
    if with_ece:
        row["ece"] = ece(y, p)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    row["log_loss"] = binary_log_loss(y, p)
    row["precision"] = precision
    row["recall"] = recall
    row["specificity"] = specificity
    row["fpr"] = fpr
    row["fnr"] = fnr
    row["f1"] = f1
    row["confusion"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return row


def load_and_prepare(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_ball_by_ball(xlsx_path)
    match_df = build_match_table(df)
    match_df = match_df.sort_values(["date", "match_id"]).reset_index(drop=True)
    match_df, _ = compute_elo(match_df)
    match_df["form1"], match_df["form2"], match_df["h2h"], _, _ = compute_form_h2h(match_df)
    match_df["form_diff"] = match_df["form1"] - match_df["form2"]
    match_df["h2h_beta"] = compute_h2h_beta(match_df)
    return df, match_df


def train_pre_match_internal(match_df: pd.DataFrame) -> dict:
    train_m, test_m = split_by_year(match_df)

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

    clim_extra = classification_metrics(y_te, p_clim)
    climatology = {"brier": brier_score_loss(y_te, p_clim), "auc": 0.5,
                   "bss": 0.0, "positive_rate": float(y_te.mean()),
                   **{k: clim_extra[k] for k in ("acc", "ece", "log_loss", "precision",
                                                 "recall", "specificity", "fpr", "fnr",
                                                 "f1", "confusion")}}

    return {
        "climatology": climatology,
        "cal_lr": classification_metrics(y_te, p_lr, p_ref=p_clim),
        "cal_gbt": classification_metrics(y_te, p_gbt, p_ref=p_clim),
        "sc": sc, "cal_lr_model": cal_lr, "cal_gbt_model": cal_gbt,
        "p_lr": p_lr, "p_gbt": p_gbt, "y_te": y_te,
    }


def build_dynamic_2nd(df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    df2 = df[df["innings"] == 2].copy()
    df2["balls_remaining"] = (df2["overs"] * 6 - df2["team_balls"]).clip(lower=1)
    df2["runs_needed"] = (df2["runs_target"] - df2["team_runs"]).clip(lower=0)
    df2["wkts_remaining"] = (10 - df2["team_wicket"]).clip(lower=0)
    df2["crr"] = df2["team_runs"] / df2["team_balls"].clip(lower=1) * 6
    df2["rrr"] = df2["runs_needed"] / df2["balls_remaining"] * 6
    df2["phase"] = phase_vec(df2["over"])
    df2["chasing_wins"] = df2["batting_wins"]

    elo_map = match_df.set_index("match_id")[["elo1", "elo2"]]
    df2 = df2.join(elo_map, on="match_id", how="left")
    df2["elo_adv"] = df2["elo2"] - df2["elo1"]
    return df2


def build_dynamic_1st(df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    df1 = df[df["innings"] == 1].copy()
    df1["balls_remaining"] = (df1["overs"] * 6 - df1["team_balls"]).clip(lower=1)
    df1["run_rate"] = df1["team_runs"] / df1["team_balls"].clip(lower=1) * 6
    df1["proj_total"] = df1["team_runs"] + df1["run_rate"] * df1["balls_remaining"] / 6
    df1["phase"] = phase_vec(df1["over"])
    df1["defending_wins"] = df1["batting_wins"]

    elo_map = match_df.set_index("match_id")[["elo1", "elo2"]]
    df1 = df1.join(elo_map, on="match_id", how="left")
    df1["elo_adv"] = df1["elo1"] - df1["elo2"]
    return df1


def train_dynamic_internal(df2: pd.DataFrame, df1: pd.DataFrame) -> dict:
    train2, test2 = split_by_year(df2)
    dsc2 = StandardScaler()
    dyn2_model = CalibratedClassifierCV(
        LogisticRegression(C=1.0, max_iter=500), method="isotonic",
        cv=match_grouped_cv(train2["match_id"]))
    dyn2_model.fit(dsc2.fit_transform(train2[DYN2]), train2["chasing_wins"].values)
    p2 = dyn2_model.predict_proba(dsc2.transform(test2[DYN2]))[:, 1]
    y2 = test2["chasing_wins"].values

    train1, test1 = split_by_year(df1)
    dsc1 = StandardScaler()
    dyn1_model = CalibratedClassifierCV(
        LogisticRegression(C=1.0, max_iter=500), method="isotonic",
        cv=match_grouped_cv(train1["match_id"]))
    dyn1_model.fit(dsc1.fit_transform(train1[DYN1]), train1["defending_wins"].values)
    p1 = dyn1_model.predict_proba(dsc1.transform(test1[DYN1]))[:, 1]
    y1 = test1["defending_wins"].values

    return {
        "dyn2": classification_metrics(y2, p2),
        "dyn1": classification_metrics(y1, p1),
        "dsc2": dsc2, "dyn2_model": dyn2_model, "train2": train2, "test2": test2,
        "dsc1": dsc1, "dyn1_model": dyn1_model,
    }


def phase_specific_eval(train2: pd.DataFrame, test2: pd.DataFrame, dsc2: StandardScaler) -> dict:
    dyn2_no_phase = [f for f in DYN2 if f != "phase"]
    idxs = [DYN2.index(f) for f in dyn2_no_phase]
    phase_names = ["Powerplay (0-6 ov)", "Middle (6-15 ov)", "Death (15-20 ov)"]
    results = {}
    for pid, pname in enumerate(phase_names):
        tr_p = train2[train2["phase"] == pid]
        te_p = test2[test2["phase"] == pid]
        m_p = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=500), method="isotonic",
            cv=match_grouped_cv(tr_p["match_id"]))
        m_p.fit(dsc2.transform(tr_p[DYN2])[:, idxs], tr_p["chasing_wins"].values)
        pp = m_p.predict_proba(dsc2.transform(te_p[DYN2])[:, idxs])[:, 1]
        yp = te_p["chasing_wins"].values
        results[pname] = {**classification_metrics(yp, pp), "n": len(yp)}
    return results


def horizon_snapshot(df_inn: pd.DataFrame, overs: int) -> pd.DataFrame:
    snap = df_inn[df_inn["team_balls"] == overs * 6]
    snap = snap.sort_values(["match_id", "over", "ball"], kind="mergesort")
    return snap.groupby("match_id").tail(1)


def _regression_row(y_true, pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(pred)) ** 2))),
        "R2": float(r2_score(y_true, pred)),
        "n": int(len(y_true)),
    }


def train_score_zoo_and_save(df1: pd.DataFrame, df2: pd.DataFrame, match_df: pd.DataFrame, out_path: str) -> dict:
    import joblib

    inn1_final = df1.groupby("match_id")["team_runs"].max().reset_index().rename(columns={"team_runs": "final_score"})
    inn2_final = df2.groupby("match_id")["team_runs"].max().reset_index().rename(columns={"team_runs": "final_score"})

    pm = match_df[["match_id", "year"] + PRE_FEAT + ["score1"]].dropna()
    train_pm, test_pm = split_by_year(pm)
    sc_pre = StandardScaler()
    X_tr_pm = sc_pre.fit_transform(train_pm[PRE_FEAT])
    X_te_pm = sc_pre.transform(test_pm[PRE_FEAT])
    pm_zoo = make_score_zoo()
    pm_metrics = {}
    for name, mdl in pm_zoo.items():
        mdl.fit(X_tr_pm, train_pm["score1"].values)
        pm_metrics[name] = _regression_row(test_pm["score1"], mdl.predict(X_te_pm))

    def _fit_inn_zoo(df_inn, feat_cols, final_df, stage_label):
        merged = df_inn.merge(final_df, on="match_id")
        train_i, test_i = split_by_year(merged)
        sc = StandardScaler()
        X_tr = sc.fit_transform(train_i[feat_cols])
        zoo = make_score_zoo()
        for name, mdl in zoo.items():
            mdl.fit(X_tr, train_i["final_score"].values)
        metrics = {}
        for h in SCORE_HORIZONS:
            te_h = horizon_snapshot(test_i, h)
            X_h = sc.transform(te_h[feat_cols])
            metrics[f"{stage_label} @{h}ov"] = {
                name: _regression_row(te_h["final_score"], mdl.predict(X_h))
                for name, mdl in zoo.items()
            }
        return zoo, sc, metrics

    inn1_zoo, sc_inn1, inn1_metrics = _fit_inn_zoo(df1, DYN1, inn1_final, "inn1")
    inn2_zoo, sc_inn2, inn2_metrics = _fit_inn_zoo(df2, DYN2, inn2_final, "inn2")

    bundle = {
        "pm_zoo": pm_zoo, "sc_pre": sc_pre,
        "inn1_zoo": inn1_zoo, "sc_inn1": sc_inn1,
        "inn2_zoo": inn2_zoo, "sc_inn2": sc_inn2,
        "score_horizons": list(SCORE_HORIZONS),
        "feature_cols": {"pre_match": PRE_FEAT, "inn1": DYN1, "inn2": DYN2},
    }
    joblib.dump(bundle, out_path)
    return {"pre_match": pm_metrics, **inn1_metrics, **inn2_metrics}


def retrain_pre_match_full(match_df: pd.DataFrame) -> tuple[CalibratedClassifierCV, StandardScaler]:
    sc_pre = StandardScaler()
    pre_model = CalibratedClassifierCV(LogisticRegression(C=1.0), method="isotonic", cv=5)
    pre_model.fit(sc_pre.fit_transform(match_df[PRE_FEAT]), match_df["team1_win"].values)
    return pre_model, sc_pre


def apply_label_corrections(pm26: pd.DataFrame, corrections_path: str) -> list[dict]:
    if not os.path.exists(corrections_path):
        return []
    corr = pd.read_csv(corrections_path)
    applied = []
    for _, c in corr.iterrows():
        mask = pm26["match_id"] == c["match_id"]
        if not mask.any():
            raise ValueError(
                f"label_corrections: match_id {c['match_id']} not present in the "
                f"external dataset -- the correction table is stale, refusing to continue")
        if c["column"] not in pm26.columns:
            raise ValueError(f"label_corrections: unknown column {c['column']!r}")
        old = pm26.loc[mask, c["column"]].iloc[0]
        pm26.loc[mask, c["column"]] = c["value"]
        applied.append({
            "match_id": int(c["match_id"]), "column": str(c["column"]),
            "old": None if pd.isna(old) else old, "new": c["value"],
            "reason": str(c["reason"]), "source": str(c["source"]),
        })
    return applied


def evaluate_2026_pre_match(match_df: pd.DataFrame, pre_model, sc_pre, pm26_path: str) -> dict:
    _, elo_2025 = compute_elo(match_df)
    _, _, _, form_hist_full, h2h_hist_full = compute_form_h2h(match_df)

    pm26 = pd.read_csv(pm26_path)
    corrections = apply_label_corrections(
        pm26, os.path.join(os.path.dirname(pm26_path), "label_corrections.csv"))
    pm26["_sort_date"] = pd.to_datetime(pm26["date"], format="mixed")
    if pm26["_sort_date"].isna().any():
        bad = pm26.loc[pm26["_sort_date"].isna(), "match_id"].tolist()
        raise ValueError(f"external dataset has unparseable dates for matches: {bad}")
    for col in ["team_bat_first", "team_bowl_first", "toss_winner", "match_winner"]:
        if col in pm26.columns:
            pm26[col] = pm26[col].map(lambda x: ABBREV.get(str(x).strip(), x) if pd.notna(x) else np.nan)
    pm26["toss_bat_first"] = ((pm26["toss_winner"] == pm26["team_bat_first"]) & (pm26["toss_decision"] == "Bat")).fillna(0).astype(int)
    pm26["toss_field_first"] = ((pm26["toss_winner"] == pm26["team_bowl_first"]) & (pm26["toss_decision"] == "Bowl")).fillna(0).astype(int)

    pm26["actual_winner"] = pm26["match_winner"]
    if pm26["actual_winner"].isna().any():
        bad = pm26.loc[pm26["actual_winner"].isna(), "match_id"].tolist()
        raise ValueError(
            f"external dataset has matches with no winner: {bad}. Add a "
            f"provenance-documented row to label_corrections.csv instead of "
            f"overriding in code.")
    implied = np.where(pm26["bat_first_won"] == 1, pm26["team_bat_first"], pm26["team_bowl_first"])
    inconsistent = pm26.loc[pm26["actual_winner"] != implied, "match_id"].tolist()
    if inconsistent:
        raise ValueError(f"match_winner disagrees with bat_first_won for matches: {inconsistent}")

    WINDOW, K = 5, 32
    elo26 = dict(elo_2025)
    form26 = {k: list(v) for k, v in form_hist_full.items()}
    h2h26 = {k: dict(v) for k, v in h2h_hist_full.items()}

    pre_results = []
    for _, r in pm26.sort_values(["_sort_date", "match_id"]).iterrows():
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
        feat = pd.DataFrame(
            [[elo_diff, form_diff, h2h_rate, r["toss_bat_first"], r["toss_field_first"]]],
            columns=PRE_FEAT,
        )
        p_bat = pre_model.predict_proba(sc_pre.transform(feat))[0, 1]
        pred = t1 if p_bat > 0.5 else t2
        actual = r["actual_winner"]
        pre_results.append({"match_id": mid, "bat_first": t1, "bowl_first": t2,
                            "p_bat": float(p_bat),
                            "pred": pred, "actual": actual, "correct": pred == actual})

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
    p_val = float(stats.binomtest(k, n, 0.5).pvalue)
    bat_first_win_rate = (pre_df["actual"] == pre_df["bat_first"]).mean()
    naive_side = "bat_first" if bat_first_win_rate >= 0.5 else "bowl_first"
    naive_acc = max(bat_first_win_rate, 1.0 - bat_first_win_rate)
    naive_correct = (pre_df["actual"] == pre_df[naive_side])
    model_only = int((pre_df["correct"] & ~naive_correct).sum())
    naive_only = int((~pre_df["correct"] & naive_correct).sum())
    discordant = model_only + naive_only
    p_val_vs_naive = (
        float(stats.binomtest(model_only, discordant, 0.5).pvalue) if discordant else 1.0)

    y_true = (pre_df["actual"] == pre_df["bat_first"]).astype(int).values
    p_clim = np.full(n, float(match_df["team1_win"].mean()))
    ext_classification = classification_metrics(y_true, pre_df["p_bat"].values, p_ref=p_clim)

    return {
        "accuracy": acc, "k": k, "n": n, "ci": (ci_lo, ci_hi), "p_value": p_val,
        "naive_baseline_acc": naive_acc, "naive_side": naive_side,
        "p_value_vs_naive": p_val_vs_naive,
        "mcnemar": {"model_only_correct": model_only, "naive_only_correct": naive_only},
        "classification": ext_classification,
        "corrections_applied": corrections,
        "pm26": pm26, "elo_2025": elo_2025,
        "pre_df": pre_df,
    }


def compare_calibration_methods_dyn2(df2: pd.DataFrame) -> dict:
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
    iso_model = CalibratedClassifierCV(
        LogisticRegression(C=1.0, max_iter=500), method="isotonic",
        cv=match_grouped_cv(train_val2["match_id"]))
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
