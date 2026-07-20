import os
import json
import glob
import pandas as pd

RAW_DIR  = "cricsheet_raw"
OUT_FILE = "ipl_data.xlsx"


def extract_extras(extras: dict) -> dict:
    return {
        "extras_wides":   extras.get("wides",   0),
        "extras_noballs": extras.get("noballs", 0),
        "extras_byes":    extras.get("byes",    0),
        "extras_legbyes": extras.get("legbyes", 0),
        "extras_penalty": extras.get("penalty", 0),
    }

def extract_wicket(ball: dict) -> dict:
    wickets = ball.get("wickets", [])
    if not wickets:
        return {"is_wicket": 0, "wicket_kind": "", "player_out": "", "fielder": ""}
    w = wickets[0]
    fielders = w.get("fielders", [])
    fielder_name = ", ".join(f.get("name", "") for f in fielders)
    return {
        "is_wicket":   1,
        "wicket_kind": w.get("kind", ""),
        "player_out":  w.get("player_out", ""),
        "fielder":     fielder_name,
    }


files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
print(f"Processing {len(files)} match files...")

ball_rows  = []
match_rows = []

for i, fp in enumerate(files, 1):
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    info     = data["info"]
    match_id = os.path.splitext(os.path.basename(fp))[0]

    teams   = info.get("teams", ["", ""])
    dates   = info.get("dates", [""])
    date    = dates[0]
    season  = info.get("season", date[:4])
    venue   = info.get("venue", "")

    toss         = info.get("toss", {})
    toss_winner  = toss.get("winner", "")
    toss_decision = toss.get("decision", "")

    outcome      = info.get("outcome", {})
    match_winner = outcome.get("winner", outcome.get("result", ""))
    by           = outcome.get("by", {})
    win_runs     = by.get("runs", "")
    win_wickets  = by.get("wickets", "")
    win_margin   = win_runs if win_runs != "" else (f"{win_wickets} wkts" if win_wickets != "" else "")

    pom_list     = info.get("player_of_match", [])
    player_of_match = ", ".join(pom_list)

    match_rows.append({
        "match_id":        match_id,
        "season":          season,
        "date":            date,
        "venue":           venue,
        "team1":           teams[0] if len(teams) > 0 else "",
        "team2":           teams[1] if len(teams) > 1 else "",
        "toss_winner":     toss_winner,
        "toss_decision":   toss_decision,
        "match_winner":    match_winner,
        "win_margin":      win_margin,
        "player_of_match": player_of_match,
    })

    for inn_idx, innings in enumerate(data.get("innings", []), 1):
        batting_team = innings.get("team", "")
        bowling_team = next((t for t in teams if t != batting_team), "")

        for over_obj in innings.get("overs", []):
            over_num = over_obj.get("over", 0)

            for ball_idx, ball in enumerate(over_obj.get("deliveries", []), 1):
                runs  = ball.get("runs", {})
                row = {
                    "match_id":      match_id,
                    "season":        season,
                    "date":          date,
                    "venue":         venue,
                    "team1":         teams[0] if len(teams) > 0 else "",
                    "team2":         teams[1] if len(teams) > 1 else "",
                    "match_winner":  match_winner,
                    "innings":       inn_idx,
                    "batting_team":  batting_team,
                    "bowling_team":  bowling_team,
                    "over":          over_num + 1,
                    "ball":          ball_idx,
                    "batter":        ball.get("batter", ""),
                    "non_striker":   ball.get("non_striker", ""),
                    "bowler":        ball.get("bowler", ""),
                    "runs_batter":   runs.get("batter", 0),
                    "runs_extras":   runs.get("extras", 0),
                    "runs_total":    runs.get("total", 0),
                    **extract_extras(ball.get("extras", {})),
                    **extract_wicket(ball),
                }
                ball_rows.append(row)

    if i % 100 == 0 or i == len(files):
        print(f"  [{i}/{len(files)}] {len(ball_rows):,} deliveries collected...")


print("\nBuilding DataFrames...")
df_balls  = pd.DataFrame(ball_rows)
df_matches = pd.DataFrame(match_rows)

df_balls = df_balls.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)
df_matches = df_matches.sort_values("date").reset_index(drop=True)

print(f"Ball-by-ball rows : {len(df_balls):,}")
print(f"Match summary rows: {len(df_matches):,}")


print(f"\nWriting to {OUT_FILE} (this may take ~30 seconds)...")

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    df_balls.to_excel(writer, sheet_name="Ball by Ball", index=False)
    df_matches.to_excel(writer, sheet_name="Match Info", index=False)

    for sheet_name, df in [("Ball by Ball", df_balls), ("Match Info", df_matches)]:
        ws = writer.sheets[sheet_name]
        for col_idx, col in enumerate(df.columns, 1):
            max_len = max(len(str(col)), df[col].astype(str).str.len().max())
            ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 2, 40)

size_mb = os.path.getsize(OUT_FILE) / 1024 / 1024
print(f"\nDone! {OUT_FILE} ({size_mb:.1f} MB)")
print(f"  Sheet 'Ball by Ball' : {len(df_balls):,} rows × {len(df_balls.columns)} columns")
print(f"  Sheet 'Match Info'   : {len(df_matches):,} rows × {len(df_matches.columns)} columns")
