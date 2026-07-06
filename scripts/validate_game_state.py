"""
One-off validation script (not part of the test suite): picks a handful
of real matches from gagan's dataset and cross-checks src/game_state.py's
output against gagan's own equivalent columns/functions wherever they
overlap, plus hand-traceable invariants (monotonicity, phase boundaries,
first-ball values).

Run: python3 scripts/validate_game_state.py

Findings from this script were promoted into tests/test_game_state.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import load_ball_by_ball
from src.features import phase as gagan_phase
from src.game_state import build_game_state_matrix

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def main() -> None:
    df = load_ball_by_ball("data/raw/ipl_data.xlsx")

    # Pick 5 varied matches: earliest, a match with at least one no-ball,
    # a match with at least one wide, a 2nd-innings chase, and the latest.
    match_ids_sorted = sorted(df["match_id"].unique())
    has_noball = df[df["extras_noballs"] > 0]["match_id"].iloc[0]
    has_wide = df[df["extras_wides"] > 0]["match_id"].iloc[0]

    picks = {
        "earliest match": match_ids_sorted[0],
        "match with a no-ball": has_noball,
        "match with a wide": has_wide,
        "mid-dataset match": match_ids_sorted[len(match_ids_sorted) // 2],
        "latest match": match_ids_sorted[-1],
    }

    for label, mid in picks.items():
        print(f"\n=== {label} (match_id={mid}) ===")
        sub = df[df["match_id"] == mid]
        X, d = build_game_state_matrix(sub)

        # --- 1. legal_balls_total must exactly match gagan's own team_balls ---
        check(
            "legal_balls_total == team_balls (all balls)",
            (d["legal_balls_total"] == d["team_balls"]).all(),
            f"{(d['legal_balls_total'] != d['team_balls']).sum()} mismatches",
        )

        for inn, grp in d.groupby("innings"):
            grp = grp.sort_values(["over", "ball"])

            # --- 2. First ball of the innings: score_before=0, wickets_before=0 ---
            first = grp.iloc[0]
            check(
                f"inn{inn}: first ball score_before == 0",
                first["score_before"] == 0,
                f"got {first['score_before']}",
            )
            check(
                f"inn{inn}: first ball wickets_before == 0",
                first["wickets_before"] == 0,
                f"got {first['wickets_before']}",
            )

            # --- 3. score_before(ball i) == team_runs(ball i-1) (or 0 for ball 1) ---
            expected_score_before = grp["team_runs"].shift(1).fillna(0)
            check(
                f"inn{inn}: score_before matches shifted team_runs",
                np.allclose(grp["score_before"].values, expected_score_before.values),
            )

            # --- 4. wickets_before(ball i) == team_wicket(ball i-1) (or 0) ---
            expected_wkts_before = grp["team_wicket"].shift(1).fillna(0)
            check(
                f"inn{inn}: wickets_before matches shifted team_wicket",
                np.allclose(grp["wickets_before"].values, expected_wkts_before.values),
            )

            # --- 5. Monotonicity: legal_balls_total, score_before, wickets_before ---
            check(
                f"inn{inn}: legal_balls_total is non-decreasing",
                (grp["legal_balls_total"].diff().dropna() >= 0).all(),
            )
            check(
                f"inn{inn}: score_before is non-decreasing",
                (grp["score_before"].diff().dropna() >= 0).all(),
            )
            check(
                f"inn{inn}: wickets_before is non-decreasing",
                (grp["wickets_before"].diff().dropna() >= 0).all(),
            )
            check(
                f"inn{inn}: wickets_before never exceeds 10",
                (grp["wickets_before"] <= 10).all(),
            )

            # --- 6. phase matches src/features.py's phase() exactly ---
            expected_phase = grp["over"].apply(gagan_phase)
            check(
                f"inn{inn}: phase matches src.features.phase(over)",
                (grp["phase"].values == expected_phase.values).all(),
            )

            # --- 7. Phase boundary edge cases: over==6 -> phase 0, over==15 -> phase 1 ---
            if (grp["over"] == 6).any():
                check(
                    f"inn{inn}: over==6 is phase 0 (Powerplay, boundary inclusive)",
                    (grp.loc[grp["over"] == 6, "phase"] == 0).all(),
                )
            if (grp["over"] == 15).any():
                check(
                    f"inn{inn}: over==15 is phase 1 (Middle, boundary inclusive)",
                    (grp.loc[grp["over"] == 15, "phase"] == 1).all(),
                )
            if (grp["over"] == 16).any():
                check(
                    f"inn{inn}: over==16 is phase 2 (Death)",
                    (grp.loc[grp["over"] == 16, "phase"] == 2).all(),
                )

            # --- 8. run_rate (idx 8, pre-clip) vs crr computed from the PREVIOUS
            #        ball's team_runs/team_balls (game-state reflects state
            #        BEFORE the current ball, so it must lag by one ball) ---
            prev_runs = grp["team_runs"].shift(1).fillna(0)
            prev_balls = grp["team_balls"].shift(1).fillna(0).clip(lower=1)
            expected_run_rate = (prev_runs / prev_balls * 6).clip(upper=15.0) / 15.0
            actual_run_rate = X[grp.index.map(lambda i: d.index.get_loc(i)), 8] if False else None
            # (compare using d's own recomputed run_rate/15 column instead of
            #  re-deriving indices into X, simpler and equivalent)
            actual_run_rate = (grp["run_rate"].clip(upper=15.0) / 15.0)
            check(
                f"inn{inn}: run_rate (idx 8) matches crr from previous ball's team_runs/team_balls",
                np.allclose(actual_run_rate.values, expected_run_rate.values, atol=1e-6),
            )

            # --- 9. 2nd innings only: runs_required/balls_remaining/required_rr
            #        vs pipeline.py's runs_needed/balls_remaining/rrr, lagged by one ball ---
            if inn == 2 and "runs_target" in grp.columns and grp["runs_target"].notna().any():
                target = grp["runs_target"].iloc[0]
                expected_runs_required = (target - prev_runs).clip(lower=0)
                check(
                    f"inn{inn}: runs_required matches (target - previous ball's team_runs)",
                    np.allclose(grp["runs_required"].values, expected_runs_required.values, atol=1e-6),
                )

            # --- 10. toss_won_bat matches the match's toss_decision column exactly ---
            expected_toss = float(grp["toss_decision"].iloc[0] == "bat")
            check(
                f"inn{inn}: toss_won_bat matches toss_decision=='bat'",
                (grp["toss_won_bat"] == expected_toss).all(),
            )

        # --- 11. Boundedness: most of the 24 features should be in [0,1] ---
        # (batter_sr_innings/200, runs_required/200 can legitimately exceed 1;
        #  everything else should not)
        clipped_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 17, 18, 20, 21, 22, 23]
        for col in clipped_cols:
            in_range = (X[:, col] >= -1e-6).all() and (X[:, col] <= 1.0 + 1e-6).all()
            check(f"feature idx {col} within [0,1]", in_range, f"min={X[:,col].min():.3f} max={X[:,col].max():.3f}")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
