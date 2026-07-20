# IPL Win-Probability & Score Prediction — Output Metrics

Full metrics extracted from a fresh, from-scratch run of the pipeline.

- **Run timestamp:** 2026-07-17 08:56:01
- **Environment:** fresh `.venv` (Python 3.9.6), dependencies installed from `requirements.txt`
  (numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1, joblib 1.5.3, openpyxl 3.1.5,
  shap 0.49.1, torch 2.8.0, pytest 8.4.2)
- **Data:** `data/raw/ipl_data.xlsx` — **1,146 matches**, **273,503 deliveries**, seasons **2008–2025**
- **Commands run:** `python run_all.py` then `python run_alt_transformer.py --epochs 8`
- **Sources:** `reports/run_summary.txt`, `reports/dashboard_data.json`, transformer training stdout

---

## 1. Pre-Match Model (internal: train ≤2020, test >2020)

Positive class throughout this section: **team1 (the bat-first side) wins**. Threshold metrics use a 0.5 cutoff.

| Model | Brier | AUC | Accuracy | BSS | ECE | LogLoss | Precision | Recall | Specificity | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Climatology (baseline) | 0.2508 | 0.5000 | 0.4813 | 0.0000 | 0.0000 | 0.6947 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| Calibrated LogReg (Cal. LR) | 0.2641 | 0.4408 | 0.4640 | -0.0533 | 0.1183 | 0.7222 | 0.2979 | 0.0838 | 0.8167 | 0.1308 | 0.1833 | 0.9162 |
| Calibrated GBT (Cal. GBT) | 0.2563 | 0.4667 | 0.4986 | -0.0220 | 0.0568 | 0.7065 | 0.4103 | 0.0958 | 0.8722 | 0.1553 | 0.1278 | 0.9042 |

Confusion matrices (TN / FP / FN / TP):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Climatology | 180 | 0 | 167 | 0 |
| Cal. LR | 147 | 33 | 153 | 14 |
| Cal. GBT | 157 | 23 | 151 | 16 |

> Pre-match skill is weak by design (AUC ≈ 0.44–0.51, near-random) — this is called out explicitly in the project's own caveats, not a bug.

---

## 2. Dynamic 2nd/1st-Innings Models (internal)

| Model | Brier | AUC | Accuracy | ECE | LogLoss | Precision | Recall | Specificity | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dynamic 2nd innings | 0.1450 | 0.8771 | 0.7912 | 0.0413 | 0.4502 | 0.8053 | 0.7619 | 0.8198 | 0.7830 | 0.1802 | 0.2381 |
| Dynamic 1st innings | 0.2204 | 0.7012 | 0.6389 | — | 0.6320 | 0.6184 | 0.6655 | 0.6139 | 0.6411 | 0.3861 | 0.3345 |

Confusion matrices (TN / FP / FN / TP):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Dynamic 2nd | 16,727 | 3,677 | 4,753 | 15,213 |
| Dynamic 1st | 13,689 | 8,608 | 7,013 | 13,952 |

---

## 3. Phase-Specific Evaluation (2nd innings)

| Phase | AUC | Brier | LogLoss | n (deliveries) |
|---|---|---|---|---|
| Powerplay (0–6 ov) | 0.8197 | 0.1786 | 0.7596 | 15,098 |
| Middle (6–15 ov) | 0.8849 | 0.1393 | 0.4410 | 18,889 |
| Death (15–20 ov) | 0.9532 | 0.0844 | 0.3616 | 6,383 |

---

## 4. Score Regression Zoo

Fixed horizons only (5/10/15/18-over snapshots, per innings-stage). Model artifact saved to `models/ipl_score_pipeline.pkl`. Pre-match R² is expected to be negative (no signal before a ball is bowled) — reported honestly, not suppressed.

### pre_match (n=347)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 30.81 | 38.96 | -0.2135 |
| Random Forest | 32.27 | 40.38 | -0.3032 |
| Gradient BT | 32.70 | 40.94 | -0.3402 |
| XGBoost | 32.71 | 40.69 | -0.3234 |
| SVR | 32.92 | 41.49 | -0.3764 |

### inn1 @5ov (n=347)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 24.46 | 30.60 | 0.2516 |
| Random Forest | 24.81 | 31.47 | 0.2082 |
| Gradient BT | 25.95 | 32.54 | 0.1535 |
| XGBoost | 26.44 | 33.23 | 0.1172 |
| SVR | 24.03 | 30.09 | 0.2763 |

### inn1 @10ov (n=347)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 19.28 | 23.94 | 0.5420 |
| Random Forest | 20.68 | 27.49 | 0.3957 |
| Gradient BT | 19.66 | 25.11 | 0.4960 |
| XGBoost | 19.99 | 25.99 | 0.4600 |
| SVR | 19.41 | 24.07 | 0.5366 |

### inn1 @15ov (n=345)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 12.23 | 15.65 | 0.7991 |
| Random Forest | 12.34 | 16.66 | 0.7723 |
| Gradient BT | 13.05 | 17.38 | 0.7522 |
| XGBoost | 12.76 | 17.19 | 0.7578 |
| SVR | 12.13 | 15.55 | 0.8018 |

### inn1 @18ov (n=338)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 8.42 | 10.76 | 0.8967 |
| Random Forest | 8.42 | 11.60 | 0.8800 |
| Gradient BT | 9.29 | 13.09 | 0.8473 |
| XGBoost | 9.44 | 13.09 | 0.8472 |
| SVR | 7.74 | 9.92 | 0.9123 |

### inn2 @5ov (n=347)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 17.56 | 21.54 | 0.5630 |
| Random Forest | 16.50 | 22.73 | 0.5136 |
| Gradient BT | 16.28 | 22.42 | 0.5266 |
| XGBoost | 16.35 | 22.18 | 0.5368 |
| SVR | 14.57 | 19.83 | 0.6295 |

### inn2 @10ov (n=343)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 17.53 | 21.78 | 0.5336 |
| Random Forest | 13.53 | 18.31 | 0.6703 |
| Gradient BT | 12.89 | 17.59 | 0.6956 |
| XGBoost | 13.19 | 17.95 | 0.6832 |
| SVR | 15.12 | 19.90 | 0.6104 |

### inn2 @15ov (n=326)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 19.90 | 25.03 | 0.2969 |
| Random Forest | 10.27 | 14.19 | 0.7739 |
| Gradient BT | 9.76 | 13.24 | 0.8033 |
| XGBoost | 9.26 | 12.77 | 0.8171 |
| SVR | 18.02 | 22.61 | 0.4262 |

### inn2 @18ov (n=266)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 23.60 | 29.49 | -0.0051 |
| Random Forest | 8.17 | 11.69 | 0.8420 |
| Gradient BT | 7.43 | 10.43 | 0.8743 |
| XGBoost | 7.16 | 9.98 | 0.8849 |
| SVR | 21.50 | 26.49 | 0.1890 |

---

## 5. External 2026 Holdout Evaluation (locked, never used in sections 1–4)

Pre-match model retrained on the full 2008–2025 dataset first. One label correction applied (`data/external_2026/label_corrections.csv`): match 12 `match_winner`: `None` → `'PBKS'`.

| Metric | Value |
|---|---|
| Accuracy | 63.51% (47/74) |
| 95% CI (Clopper–Pearson) | [51.51%, 74.40%] |
| p-value vs. 50/50 (exact binomial) | 0.0265 |
| Naive majority-class baseline (always pick bowl-first side) | 63.51% |
| p-value vs. naive baseline (exact McNemar, 1/1 discordant) | 1.000 |
| Brier | 0.2390 |
| LogLoss | 0.6710 |
| AUC | 0.5229 |
| BSS | 0.0061 |
| ECE | 0.0969 |
| Precision | 0.5000 |
| Recall (Sensitivity) | 0.0370 |
| Specificity | 0.9787 |
| F1 | 0.0690 |
| FPR | 0.0213 |
| FNR | 0.9630 |
| Confusion matrix | TN 46, FP 1, FN 26, TP 1 |

> Significance against 50/50 does not establish improvement over the naive majority baseline (see McNemar p=1.000, above). Internal pre-match AUC is ~0.44–0.51 (near-random); the 74-match 2026 accuracy figure may reflect sample variance rather than genuine skill.

---

## 6. SHAP Explainability (pre-match Cal. GBT model)

Mean |SHAP| feature importance:

| Feature | mean\|SHAP\| |
|---|---|
| elo_diff | 0.0289 |
| h2h | 0.0108 |
| form_diff | 0.0105 |
| toss_bat_first | 0.0006 |
| toss_field_first | 0.0006 |

---

## 7. 2026 Tournament Simulation (actual vs. predicted points table)

- **Actual table topper:** Royal Challengers Bengaluru
- **Predicted table topper:** Royal Challengers Bengaluru — **CORRECT**
- **Top-4 overlap:** 3/4 — Gujarat Titans, Punjab Kings, Royal Challengers Bengaluru

### Actual points table

| Team | P | W | L | Pts | NRR |
|---|---|---|---|---|---|
| Royal Challengers Bengaluru | 16 | 11 | 5 | 22 | 1.064 |
| Gujarat Titans | 17 | 10 | 7 | 20 | 0.254 |
| Sunrisers Hyderabad | 15 | 9 | 6 | 18 | 0.334 |
| Punjab Kings | 14 | 8 | 6 | 16 | 0.304 |
| Rajasthan Royals | 16 | 8 | 8 | 16 | 0.265 |
| Delhi Capitals | 14 | 7 | 7 | 14 | -0.660 |
| Kolkata Knight Riders | 14 | 6 | 8 | 12 | -0.112 |
| Chennai Super Kings | 14 | 6 | 8 | 12 | -0.339 |
| Lucknow Super Giants | 14 | 5 | 9 | 10 | -0.762 |
| Mumbai Indians | 14 | 4 | 10 | 8 | -0.561 |

### Predicted points table

| Team | P | W (pred) | L (pred) | Pts (pred) |
|---|---|---|---|---|
| Royal Challengers Bengaluru | 16 | 10 | 6 | 20 |
| Gujarat Titans | 17 | 10 | 7 | 20 |
| Punjab Kings | 14 | 10 | 4 | 20 |
| Delhi Capitals | 14 | 9 | 5 | 18 |
| Kolkata Knight Riders | 14 | 7 | 7 | 14 |
| Mumbai Indians | 14 | 7 | 7 | 14 |
| Lucknow Super Giants | 14 | 6 | 8 | 12 |
| Sunrisers Hyderabad | 15 | 5 | 10 | 10 |
| Chennai Super Kings | 14 | 5 | 9 | 10 |
| Rajasthan Royals | 16 | 5 | 11 | 10 |

---

## 8. Calibration Reliability (pre-match Cal. GBT + dynamic 2nd innings)

- Pre-match Cal. GBT: 5 populated bins
- Dynamic 2nd innings (isotonic): 10 populated bins

### Pre-match Cal. GBT reliability bins

| Predicted mean | Observed freq | Count |
|---|---|---|
| 0.2599 | 0.8000 | 5 |
| 0.3717 | 0.4634 | 41 |
| 0.4544 | 0.4885 | 262 |
| 0.5064 | 0.4118 | 34 |
| 0.6152 | 0.4000 | 5 |

### Dynamic 2nd-innings (isotonic) reliability bins

| Predicted mean | Observed freq | Count |
|---|---|---|
| 0.0251 | 0.0653 | 9,385 |
| 0.1570 | 0.2199 | 2,478 |
| 0.2511 | 0.2749 | 3,325 |
| 0.3490 | 0.4026 | 4,575 |
| 0.4493 | 0.4886 | 1,717 |
| 0.5411 | 0.6076 | 4,472 |
| 0.6529 | 0.7203 | 2,192 |
| 0.7439 | 0.8045 | 3,529 |
| 0.8493 | 0.8671 | 2,717 |
| 0.9626 | 0.9569 | 5,980 |

---

## 9. Team Elo Trajectories (2008–2025)

Elo history computed for **15 teams**, **2,292 match-level data points** total. Starting Elo for each franchise is 1,500 ± seeding offset; table below shows the first and last recorded rating per team.

| Team | Data points | First Elo (year) | Last Elo (year) |
|---|---|---|---|
| Kolkata Knight Riders | 259 | 1516.00 (2008) | 1534.20 (2025) |
| Royal Challengers Bengaluru | 264 | 1484.00 (2008) | 1616.18 (2025) |
| Chennai Super Kings | 250 | 1516.00 (2008) | 1468.46 (2025) |
| Punjab Kings | 258 | 1484.00 (2008) | 1535.21 (2025) |
| Rajasthan Royals | 229 | 1484.00 (2008) | 1475.92 (2025) |
| Delhi Capitals | 258 | 1516.00 (2008) | 1520.72 (2025) |
| Mumbai Indians | 273 | 1483.26 (2008) | 1526.79 (2025) |
| Deccan Chargers | 75 | 1484.74 (2008) | 1424.69 (2012) |
| Kochi Tuskers Kerala | 14 | 1483.86 (2011) | 1476.99 (2011) |
| Pune Warriors | 45 | 1513.32 (2011) | 1351.09 (2013) |
| Sunrisers Hyderabad | 191 | 1509.78 (2013) | 1511.72 (2025) |
| Rising Pune Supergiants | 30 | 1521.86 (2016) | 1556.49 (2017) |
| Gujarat Lions | 29 | 1513.95 (2016) | 1453.74 (2017) |
| Lucknow Super Giants | 57 | 1484.00 (2022) | 1500.14 (2025) |
| Gujarat Titans | 60 | 1516.00 (2022) | 1547.67 (2025) |

---

## 10. Temperature Scaling vs. Isotonic Calibration (dynamic 2nd innings)

Both methods start from the same uncalibrated `LogisticRegression`. Isotonic is the pipeline's default; temperature scaling is shown for comparison only.

| Method | Brier (raw → cal) | AUC (raw → cal) | ECE (raw → cal) |
|---|---|---|---|
| Temperature-scaled (T=1.0639) | 0.1457 → 0.1453 | 0.8772 → 0.8772 | 0.0457 → 0.0450 |
| Isotonic (default) | 0.1450 (final) | 0.8771 (final) | 0.0415 (final) |

---

## 11. Alternative Path: Ball-by-Ball Transformer (`run_alt_transformer.py --epochs 8`)

Player embeddings are fixed random init (not GNN-pretrained). Win probability evaluated at the final ball of each innings.

| Item | Value |
|---|---|
| Unique players in embedding table | 766 |
| Train innings (years 2008–2018) | 1,372 |
| Val innings (years 2019–2020) | 226 |
| Test innings (years 2021–2025) | 694 |
| Epochs | 8 |
| Best validation Brier | 0.1264 |
| Test Brier | 0.1387 |
| Test AUC | 0.8926 |
| Test n | 694 |

Checkpoint saved to `models/alt_transformer.pt`. Per the project's own README, the team-level calibrated LogReg/GBT path (sections 1–2 above) is the recommended win-probability model; the Transformer is included for architectural comparison.

---

## Appendix: Run Overview

| Field | Value |
|---|---|
| Matches | 1,146 |
| Deliveries | 273,503 |
| Seasons | 2008–2025 |
| Dashboard rewritten | `dashboard/index.html` |
| Text report | `reports/run_summary.txt` |
| JSON data artifact | `reports/dashboard_data.json` |
| Transformer checkpoint | `models/alt_transformer.pt` |
