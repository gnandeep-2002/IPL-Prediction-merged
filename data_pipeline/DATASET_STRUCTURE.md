# IPL Dataset Structure — `ipl_data.xlsx`

## Overview

| | |
|---|---|
| **File** | `ipl_data.xlsx` |
| **Seasons** | 2008 – 2026 |
| **Matches** | 1,243 |
| **Total Deliveries** | 295,732 |
| **Sheets** | Ball by Ball, Match Info |

---

## Two Sheets

| Sheet | Rows | Columns | One row = |
|---|---|---|---|
| **Ball by Ball** | 295,732 | 27 | One delivery |
| **Match Info** | 1,243 | 11 | One match |

---

## Sheet 1 — Ball by Ball (295,732 × 27)

### Group A — Match Identifiers

| Column | Type | Example | Notes |
|---|---|---|---|
| `match_id` | string | `1082591` | Cricsheet unique match ID |
| `season` | string | `2017` or `"2007/08"` | Raw Cricsheet season label |
| `date` | string | `2017-04-05` | Match date |
| `venue` | string | `Rajiv Gandhi International Stadium` | Ground name |
| `team1` | string | `Sunrisers Hyderabad` | First team listed |
| `team2` | string | `Royal Challengers Bangalore` | Second team listed |

---

### Group B — Match Result

| Column | Type | Example | Notes |
|---|---|---|---|
| `match_winner` | string | `Sunrisers Hyderabad` | Winning team name, or `"tie"` / `"no result"` |

---

### Group C — Delivery Position

| Column | Type | Example | Notes |
|---|---|---|---|
| `innings` | int | `1`, `2`, `3`, `4` | 1 & 2 = normal innings; 3 & 4 = Super Over |
| `batting_team` | string | `Sunrisers Hyderabad` | Team currently batting |
| `bowling_team` | string | `Royal Challengers Bangalore` | Team currently bowling |
| `over` | int | `1` – `20` | **1-indexed** — Over 1 = `1`, Over 20 = `20` |
| `ball` | int | `1`, `2`, `3`… | Ball counter within the over (includes wides & no-balls) |

---

### Group D — Players on that Delivery

| Column | Type | Example |
|---|---|---|
| `batter` | string | `DA Warner` |
| `non_striker` | string | `S Dhawan` |
| `bowler` | string | `TS Mills` |

---

### Group E — Runs

| Column | Type | Example | Meaning |
|---|---|---|---|
| `runs_batter` | int | `4` | Runs credited to the batter |
| `runs_extras` | int | `0` | Extras on this ball |
| `runs_total` | int | `4` | Total runs off this ball |
| `extras_wides` | int | `0` or `1+` | Wide runs |
| `extras_noballs` | int | `0` or `1+` | No-ball runs |
| `extras_byes` | int | `0` or `1+` | Bye runs |
| `extras_legbyes` | int | `0` or `1+` | Leg-bye runs |
| `extras_penalty` | int | `0` or `5` | Penalty runs |

---

### Group F — Wicket

| Column | Type | Example | Nulls |
|---|---|---|---|
| `is_wicket` | int | `0` or `1` | Never null |
| `wicket_kind` | string | `caught`, `bowled`, `run out` | Null when no wicket |
| `player_out` | string | `DA Warner` | Null when no wicket |
| `fielder` | string | `Mandeep Singh` | Null when no wicket or no fielder |

---

## Sheet 2 — Match Info (1,243 × 11)

One summary row per match.

| Column | Type | Example | Notes |
|---|---|---|---|
| `match_id` | string | `335982` | Cricsheet unique match ID |
| `season` | string | `2007/08`, `2025` | Raw Cricsheet season label |
| `date` | string | `2008-04-18` | Match date |
| `venue` | string | `M Chinnaswamy Stadium` | Ground name |
| `team1` | string | `Royal Challengers Bangalore` | First team |
| `team2` | string | `Kolkata Knight Riders` | Second team |
| `toss_winner` | string | `Royal Challengers Bangalore` | Toss winner |
| `toss_decision` | string | `field` | `bat` or `field` |
| `match_winner` | string | `Kolkata Knight Riders` | Winning team |
| `win_margin` | mixed | `140` or `9 wkts` | Margin of victory |
| `player_of_match` | string | `BB McCullum` | Player of the match |

---

## Important Things to Know

| # | Thing | Detail |
|---|---|---|
| 1 | **`over` is 1-indexed** | Over 1 = `1`, end of powerplay (Over 6) = `6`, last over = `20` |
| 2 | **Super Overs** | `innings` values of `3` or `4` are Super Overs. Filter to `innings <= 2` for standard analysis |
| 3 | **`season` raw format** | Some seasons stored as `"2007/08"`, `"2009/10"`, `"2020/21"` — normalize if needed |

---

## Null Value Guide

| Column | Null % | Why |
|---|---|---|
| `wicket_kind` | ~95% | Null = no wicket on that ball (expected) |
| `player_out` | ~95% | Null = no wicket on that ball (expected) |
| `fielder` | ~96% | Null = no fielder involved or no wicket |

---

## Data Source

All columns derived directly from Cricsheet JSON files in `cricsheet_raw/`.
