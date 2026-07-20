from __future__ import annotations

import pandas as pd

from src.data import ABBREV


def actual_points_table(pm26: pd.DataFrame, ig26: pd.DataFrame) -> pd.DataFrame:
    teams_2026 = list(ABBREV.values())
    pts = {t: 0 for t in teams_2026}
    wins = {t: 0 for t in teams_2026}
    losses = {t: 0 for t in teams_2026}
    nrr_rf = {t: 0.0 for t in teams_2026}
    nrr_bf = {t: 0 for t in teams_2026}
    nrr_ra = {t: 0.0 for t in teams_2026}
    nrr_ba = {t: 0 for t in teams_2026}

    for _, r in pm26.iterrows():
        t1, t2, winner = r["team_bat_first"], r["team_bowl_first"], r["actual_winner"]
        loser = t2 if winner == t1 else t1
        pts[winner] += 2
        wins[winner] += 1
        losses[loser] += 1
        mid = r["match_id"]
        m1 = ig26[(ig26["match_id"] == mid) & (ig26["innings"] == 1)].sort_values("team_balls")
        m2 = ig26[(ig26["match_id"] == mid) & (ig26["innings"] == 2)].sort_values("team_balls")
        if len(m1) == 0 or len(m2) == 0:
            continue
        l1, l2 = m1.iloc[-1], m2.iloc[-1]
        s1, b1, w1 = l1["team_runs"], int(l1["team_balls"]), int(l1["team_wicket"])
        s2, b2, w2 = l2["team_runs"], int(l2["team_balls"]), int(l2["team_wicket"])
        b1e = 120 if w1 >= 10 else b1
        b2e = 120 if w2 >= 10 else b2
        nrr_rf[t1] += s1; nrr_bf[t1] += b1e; nrr_ra[t1] += s2; nrr_ba[t1] += b2e
        nrr_rf[t2] += s2; nrr_bf[t2] += b2e; nrr_ra[t2] += s1; nrr_ba[t2] += b1e

    def nrr(t):
        f = nrr_rf[t] / nrr_bf[t] * 6 if nrr_bf[t] > 0 else 0
        a = nrr_ra[t] / nrr_ba[t] * 6 if nrr_ba[t] > 0 else 0
        return f - a

    table_df = pd.DataFrame(
        [{"Team": t, "P": wins[t] + losses[t], "W": wins[t], "L": losses[t],
          "Pts": pts[t], "NRR": round(nrr(t), 3)} for t in teams_2026]
    )
    table_df = table_df.sort_values(["Pts", "NRR"], ascending=False).reset_index(drop=True)
    table_df.index += 1
    return table_df


def predicted_points_table(pre_df: pd.DataFrame) -> pd.DataFrame:
    teams_2026 = list(ABBREV.values())
    pts_p = {t: 0 for t in teams_2026}
    wins_p = {t: 0 for t in teams_2026}
    losses_p = {t: 0 for t in teams_2026}
    for _, r in pre_df.iterrows():
        t1, t2 = r["bat_first"], r["bowl_first"]
        w = r["pred"]
        l = t2 if w == t1 else t1
        pts_p[w] += 2
        wins_p[w] += 1
        losses_p[l] += 1

    pred_table = pd.DataFrame(
        [{"Team": t, "P": wins_p[t] + losses_p[t], "W_pred": wins_p[t],
          "L_pred": losses_p[t], "Pts_pred": pts_p[t]} for t in teams_2026]
    ).sort_values("Pts_pred", ascending=False).reset_index(drop=True)
    pred_table.index += 1
    return pred_table


def compare_tables(table_df: pd.DataFrame, pred_table: pd.DataFrame) -> dict:
    top4_actual = set(table_df.head(4)["Team"])
    top4_pred = set(pred_table.head(4)["Team"])
    overlap = top4_actual & top4_pred
    return {
        "actual_topper": table_df.iloc[0]["Team"],
        "predicted_topper": pred_table.iloc[0]["Team"],
        "topper_correct": table_df.iloc[0]["Team"] == pred_table.iloc[0]["Team"],
        "top4_actual": sorted(top4_actual),
        "top4_predicted": sorted(top4_pred),
        "top4_overlap": sorted(overlap),
        "top4_overlap_count": len(overlap),
    }
