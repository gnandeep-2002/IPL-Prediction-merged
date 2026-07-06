# IPL & Sports Probabilistic ML Prediction — Full Paper Reference

> Compiled from research discussed across this conversation.  
> Papers are grouped into six categories covering all 51 sources.

---

## Table of Contents

1. [IPL / Cricket — General ML Prediction](#1-ipl--cricket--general-ml-prediction)
2. [IPL / Cricket — Deep Learning Score Prediction](#2-ipl--cricket--deep-learning-score-prediction)
3. [Cricket — Win/Score Prediction (Extended)](#3-cricket--winscore-prediction-extended)
4. [IPL — Auction, Player Price & Fantasy](#4-ipl--auction-player-price--fantasy)
5. [General Sports — ML Prediction (Other Sports)](#5-general-sports--ml-prediction-other-sports)
6. [Probabilistic / Bayesian — Football (Soccer)](#6-probabilistic--bayesian--football-soccer)
7. [Probabilistic / Bayesian — Cricket](#7-probabilistic--bayesian--cricket)
8. [Probabilistic / Bayesian — Other Sports](#8-probabilistic--bayesian--other-sports)

---

## 1. IPL / Cricket — General ML Prediction

---

### Paper 1: Forecasting IPL Score Using Machine Learning Techniques

| Field | Detail |
|---|---|
| **Source** | IEEE Xplore |
| **URL** | https://ieeexplore.ieee.org/document/10882211/ |
| **Type** | IEEE Conference Publication |

**Summary**  
This paper employs various machine learning techniques to predict the cricket scores of IPL matches. It aims to provide valuable insights to sporting teams, fans, and analysts about score outcomes given the chaotic and dynamic nature of T20 cricket.

**Key Approach**
- Applies multiple ML algorithms to the score forecasting problem
- Uses historical IPL ball-by-ball and match-level data as input
- Focuses on providing actionable insights beyond raw accuracy numbers

**Significance**  
One of the few papers to focus purely on score (not winner) forecasting with multiple ML methods compared side-by-side on IPL data.

---

### Paper 2: Prediction of IPL Match Outcome Using Machine Learning Techniques

| Field | Detail |
|---|---|
| **Source** | ResearchGate / arXiv (2110.01395) |
| **URL** | https://arxiv.org/pdf/2110.01395 |
| **URL (ResearchGate)** | https://www.researchgate.net/publication/355061139 |
| **Type** | Research Article |

**Summary**  
Proposes a model for predicting IPL match outcomes. Introduces novel player-rating indices — a Batting Index and a Bowling Index — rather than relying on raw averages. Adds team form and team strength as features beyond traditional toss and venue variables.

**Algorithms Used**
- Support Vector Machine (SVM)
- Random Forest Classifier (RFC)
- Logistic Regression
- K-Nearest Neighbor (KNN)
- Naive Bayes (also tested)

**Key Results**
- Random Forest achieved the highest accuracy: **88.10%**
- Novel Batting Index and Bowling Index outperformed raw averages as features

**Features Used**
- Team composition
- Batting and bowling averages per player
- Team success in previous matches
- Toss outcome
- Venue / Day-Night status
- Win probability batting first at a given venue against a specific team

**Significance**  
One of the most cited IPL ML papers; introduced the idea of computed performance indices as features, moving beyond raw historical averages.

---

### Paper 3: Cricket Match Prediction using Machine Learning

| Field | Detail |
|---|---|
| **Source** | IJARSCT |
| **URL** | https://ijarsct.co.in/Paper9073.pdf |
| **Type** | Journal Article |

**Summary**  
Proposes a three-method model: (1) first-innings score prediction, (2) IPL match win percentage, and (3) ODI match win percentage, creating a unified web application for cricket enthusiasts.

**Algorithms Used**
- Decision Tree Classifier
- Random Forest Classifier
- Lasso Regression
- Logistic Regression

**Key Approach**
- Ball-by-ball data ingested in real time
- Simultaneous prediction of score and winner
- Web application interface for live use

**Referenced Work**
- Cites Kumash Kapadia's feature selection work (IG, Correlation-based, ReliefF, Wrapper methods)
- Cites Prasad Thorat et al. on cricket score prediction

---

### Paper 4: Analysis and Predictions of Winning Indian Premier League Match Using Machine Learning Algorithm

| Field | Detail |
|---|---|
| **Source** | IEEE Xplore |
| **URL** | https://ieeexplore.ieee.org/document/10134747/ |
| **Type** | IEEE Conference Publication |

**Summary**  
Focuses on predicting the win probability ball-by-ball during the second innings. Also ranks bowlers and batters based on their match performances.

**Key Approach**
- Second-innings win probability updated after every ball
- Player ranking system derived from the same model outputs
- Uses the large amount of statistical information embedded in IPL data

**Significance**  
One of the few IPL papers to address live, in-match win probability at ball level rather than pre-match prediction only.

---

### Paper 5: Prediction of IPL Match Outcome Using Machine Learning Techniques

| Field | Detail |
|---|---|
| **Source** | arXiv 2110.01395 (also on ResearchGate — same paper as #2 above) |
| **URL** | https://www.researchgate.net/publication/355061139 |

*(See Paper 2 for full details — this is the same paper indexed separately.)*

---

### Paper 6: Prediction on IPL Data Using Machine Learning Techniques in R Package

| Field | Detail |
|---|---|
| **Source** | ICTACT Journals on Soft Computing |
| **Authors** | G. Sudhamathy and G. Raja Meenakshi |
| **URL** | https://ictactjournals.in/paper/IJSC_Vol_11_Iss_1_Paper_2_2199_2204.pdf |
| **Type** | Journal Article |

**Summary**  
Implements ML algorithms entirely in R and uses the last ten years of IPL history to predict match winners. Uses confusion matrices for evaluation and visualizes decision trees.

**Algorithms Used**
- Decision Tree
- Naive Bayes
- K-Nearest Neighbour
- Random Forest

**Key Results**
- Random Forest performed best (highest accuracy, lowest error)
- KKR predicted as most likely winner based on 10-year history
- Evaluated using accuracy, precision, recall, sensitivity, and error rate

---

### Paper 7: IPL Team Winning Prediction using Machine Learning

| Field | Detail |
|---|---|
| **Source** | IJARCCE (2025) |
| **Authors** | Labana Milendra, Rohit S, Jithin C, Dr. G Paavai Anand |
| **DOI** | 10.17148/IJARCCE.2025.141139 |
| **URL** | https://ijarcce.com/papers/ipl-team-winning-prediction-using-machine-learning/ |
| **Year** | 2025 |

**Summary**  
Applies data-driven ML on the most recent IPL dataset (2008–2024), making it one of the most current IPL ML papers available.

**Algorithms Tested**
- Logistic Regression
- Random Forest
- XGBoost

**Dataset**
- IPL seasons 2008–2024
- Features: team stats, player performance, venue details, toss results

**Keywords**  
Machine Learning, IPL Prediction, Sports Analytics, XGBoost, Cricket Match Outcome, Data-Driven Decision Making

---

### Paper 8: Analysing Long Short Term Memory Models for Cricket Match Outcome Prediction

| Field | Detail |
|---|---|
| **Source** | arXiv 2011.02122 |
| **URL** | https://arxiv.org/pdf/2011.02122 |
| **Type** | arXiv Preprint |

**Summary**  
A key paper in applying sequential deep learning to cricket. Uses a single LSTM model to represent the entire match and generate win predictions after every ball — contrasting with approaches that train one model per over.

**Key Approach**
- Single LSTM covers the full match (not a separate model per over)
- Gives predictions after every ball, not just before the match
- Uses sequential/time-series nature of ball-by-ball data

**Referenced Comparisons**
- Shah et al. (Logistic Regression on ODI ICC data): 81% accuracy
- Daniel et al. (Decision Tree + XGBoost on IPL): 94.8% accuracy
- Jhanwar et al. (balanced ODI dataset): 97% accuracy
- Jhawar et al. (Random Forest per over, ODI): 75.68% average accuracy across overs

**Significance**  
Demonstrates that a single sequence model can match or beat the per-over model ensemble approach, while being far more computationally efficient.

---

### Paper 9: Cricket Analysis and Prediction of Projected Score and Winner using Machine Learning

| Field | Detail |
|---|---|
| **Source** | IJARCCE |
| **Authors** | Apurva Lawate, Nomesh Katare, Salil Hoskeri, Santosh Takle, Prof. Supriya B. Jadhav |
| **DOI** | 10.17148/IJARCCE.2021.10223 |
| **URL** | https://ijarcce.com/papers/cricket-analysis-and-prediction-of-projected-score-and-winner-using-machine-learning/ |
| **Year** | 2021 |

**Summary**  
Predicts the first-innings projected score while the match is in progress, then uses that projected score to also predict the winner.

**Algorithm**
- Linear Regression for score prediction

**Key Innovation**
- Emphasis on the **last 5 overs** as the primary feature window — not considered in prior models
- Features: wickets in last 5 overs, runs in last 5 overs, total score, wickets at current ball, overs bowled
- Model explains ~75.226% of data variance (R²)

**Dataset**  
IPL 2008–2019

---

## 2. IPL / Cricket — Deep Learning Score Prediction

---

### Paper 10: IPL Score Predictor using Deep Learning

| Field | Detail |
|---|---|
| **Source** | GitHub |
| **Author** | Vinayak0042 |
| **URL** | https://github.com/Vinayak0042/IPL-Score-Predictor-using-Deep-Learning |
| **Type** | Project / Repository |

**Summary**  
A project using a feedforward neural network to predict the total score the batting team will achieve, given live match inputs.

**Inputs**
- Venue
- Batting team
- Bowling team
- Current striker
- Bowler

**Architecture**
- Standard feedforward (dense) neural network layers
- Trained on historical per-ball IPL data

**Dataset**  
Comprehensive historical IPL match data (venue, teams, players, outcomes)

---

### Paper 11: IPL Score Prediction Using Deep Learning (Video/Tutorial)

| Field | Detail |
|---|---|
| **Source** | GeeksforGeeks |
| **URL** | https://www.geeksforgeeks.org/videos/ipl-score-prediction-using-deep-learning/ |
| **Type** | Educational walkthrough |

**Summary**  
Step-by-step tutorial for building an IPL score predictor using deep learning, covering the full pipeline from data collection to deployment.

**Pipeline Steps**
1. Data Collection — historical IPL matches from Kaggle or scraped sources
2. Preprocessing — missing values, categorical encoding
3. Feature Engineering — run rate, wickets in hand, batsmen strike rate
4. Model Building — Keras/TensorFlow neural network
5. Evaluation — standard regression metrics

**Key Features Used**  
Runs scored, wickets lost, current over, run rate, batsman-specific variables

---

### Paper 12: IPL Score Prediction & Analysis

| Field | Detail |
|---|---|
| **Source** | IJFMR |
| **URL** | https://www.ijfmr.com/papers/2023/6/8241.pdf |
| **Year** | 2023 |

**Summary**  
Reviews both shallow ML and deep learning for IPL score prediction, with particular focus on RNN and LSTM architectures that capture the temporal/sequential nature of ball-by-ball data.

**Key Methods Reviewed**
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTMs)
- Convolutional Neural Networks (CNNs) combined with LSTMs (from 2020 study)

**Significance**  
Establishes that deep learning, especially recurrent architectures, is better suited to cricket score prediction than static ML because of the time-series nature of cricket data. Relevant for fantasy cricket, analytics, and strategy.

---

### Paper 13: Cricket Score Prediction Using Deep Learning

| Field | Detail |
|---|---|
| **Source** | IJRASET |
| **URL** | https://www.ijraset.com/research-paper/cricket-score-prediction-using-deep-learning |
| **Year** | 2025 |

**Summary**  
Builds a multilayer dense neural network to forecast the first-innings final score, with deployment via both a Jupyter widget and a Flask web app.

**Architecture**
- Multilayer dense (fully-connected) neural network
- Built in Python with TensorFlow and Keras

**Feature Engineering**
- Balls remaining
- Wickets in hand
- Current run rate
- Venue, batting/bowling teams, batsman, bowler (encoded)

**Evaluation Metrics**
- Mean Absolute Error (MAE)
- R² Score

**Deployment**
- Jupyter notebook widget for exploratory analysis
- Flask web application for real-time user predictions

---

### Paper 14: IPL Score Prediction using Deep Learning

| Field | Detail |
|---|---|
| **Source** | GeeksforGeeks |
| **URL** | https://www.geeksforgeeks.org/deep-learning/ipl-score-prediction-using-deep-learning/ |
| **Year** | October 2025 |

**Summary**  
Implements a regression neural network for score prediction with an emphasis on robust loss functions to handle IPL score outliers.

**Architecture Details**
- Dense layers (fully connected)
- Output layer: **linear activation** (regression)
- Loss function: **Huber Loss** (combines robustness of MAE for outliers with MSE smoothness near the mean)

**Significance**  
One of the few IPL papers to explicitly justify its loss function choice — Huber loss is more appropriate than pure MSE when outlier innings (e.g., 250+ scores) would otherwise dominate training.

---

### Paper 15: IPL Score Prediction using Deep Learning + Ensemble Techniques

| Field | Detail |
|---|---|
| **Source** | GitHub |
| **Author** | arindal1 |
| **URL** | https://github.com/arindal1/IPL-score-pred |
| **Type** | Project / Repository |

**Summary**  
Experiments with multiple neural network architectures and ensemble configurations, training on historical IPL data.

**Evaluation Metrics**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-Squared (R²)

**Results**
- Final model R² ≈ 0.592 on test set
- Explains ~59.2% of variance in IPL scores

**Files**
- `ipl_dataset.csv` — training/evaluation data
- `IPLScore.ipynb` — full annotated notebook

---

## 3. Cricket — Win/Score Prediction (Extended)

---

### Paper 16: Live Cricket Predictions for Runs and Win Using Machine Learning

| Field | Detail |
|---|---|
| **Source** | ResearchGate |
| **URL** | https://www.researchgate.net/publication/384795727 |
| **Year** | 2024 |

**Summary**  
Two separate models: (1) a runs predictor for first innings end score, and (2) a live win probability updater for the second innings. Uses ball-by-ball data and models relative team strength from player career stats.

**Dataset**  
Cricksheet ball-by-ball data for international matches 2005–2020

**Models & Results**

| Task | Best Model | Score |
|---|---|---|
| First-innings run prediction | XGBoost | R² = 0.99, RMSE = 5.85 |
| Win prediction (accuracy) | Random Forest | ~99.99% accuracy |
| Win probability calibration | Logistic Regression | Best calibrated probabilities |

**Key Insight**  
Random Forest wins on raw accuracy but Logistic Regression is better calibrated for probability — an important distinction for probabilistic applications.

---

### Paper 17: A Systematic Review of Machine Learning in Sports Betting

| Field | Detail |
|---|---|
| **Source** | arXiv 2410.21484 |
| **URL** | https://arxiv.org/pdf/2410.21484 |
| **Type** | Survey / Review Paper |

**Summary**  
Broad review of ML techniques applied across sports betting. Covers cricket in the context of Vistro et al.'s IPL predictions.

**Covered Cricket Work (Vistro et al., 2019)**
- IPL data 2008–2017
- Decision Tree, Random Forest, XGBoost
- Features: player performance, weather, venue
- Decision Tree: 76.9% → 94.87% (after tuning)
- Random Forest: 71% → 80% (after tuning)
- XGBoost: 94.23% (no tuning)

**Also Covers**  
Jayanth et al. (2018) — SVM with linear, poly, and RBF kernels on cricket outcomes; Linear Regression vs Current Run Rate for scoring

---

### Paper 18: XGBoosting Cricket: Enhancing Predictive Modeling for T20 Match Results

| Field | Detail |
|---|---|
| **Source** | SN Computer Science (Springer) |
| **URL** | https://link.springer.com/article/10.1007/s42979-024-03385-0 |
| **Year** | 2024 |

**Summary**  
Builds an XGBoost pipeline as the primary method for T20 outcome prediction, and reviews the prior art in the area.

**Referenced Related Work**
- Rodrigues et al. — multiple random forest regression for squad analysis (2019)
- Manivannan et al. — CNN + feature encoding for match outcome (2019)
- Hatharasinghe & Poravi — data mining and ML in cricket (2019)
- Sudhamathy & Meenakshi — IPL in R package (2020)
- Weeraddana & Premaratne — XGBoost for cricket outcome (2021)
- Dhonge et al. — IPL score and win prediction with ML (2021)
- Pansare et al. — XGBoost regression for score (2022)

---

### Paper 19: Cricket Data Analytics: Forecasting T20 Match Winners Through Machine Learning

| Field | Detail |
|---|---|
| **Source** | SAGE / ACM (International Journal of Knowledge-based and Intelligent Engineering Systems) |
| **Authors** | Sanjay Chakraborty, Arnab Mondal, Aritra Bhattacharjee, Ankush Mallick, Riju Santra, Saikat Maity, Lopamudra Dey |
| **URL** | https://journals.sagepub.com/doi/abs/10.3233/KES-230060 |
| **Year** | 2024 |

**Summary**  
Pre-match T20 winner forecasting using an ensemble of five ML classifiers on team performance and rankings data, validated on the 2022 T20 World Cup.

**Algorithms Used**
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- XGBoost

**Best Result**  
Random Forest: **84.06% accuracy**

**Validation**  
Post-case study on T20 World Cup 2022 (England as predicted winner — correct)

---

### Paper 20: Win Probability Prediction for IPL Match Using Various Machine Learning Methods

| Field | Detail |
|---|---|
| **Source** | International Journal of Electrical and Computer Systems (IJECS) |
| **URL** | https://www.computersciencejournals.com/ijecs/article/view/94/5-2-3 |

**Summary**  
Specifically focuses on outputting **win probability values** (not just labels) using probabilistic models, with toss outcome and powerplay performance as primary features.

**Models & Results**

| Model | Accuracy |
|---|---|
| Naive Bayes | 63% |
| Logistic Regression | 82% |
| Random Forest | 96% |

**Key Point**  
Explicitly falls under supervised machine learning, and chooses Naive Bayes and Logistic Regression precisely because they output probabilities — a rare choice in IPL papers that usually optimize for classification accuracy only.

---

### Paper 21: Sports Prediction for Cricket Match Using Grid Search and Extreme Gradient Boosting Classifier

| Field | Detail |
|---|---|
| **Source** | Springer |
| **URL** | https://link.springer.com/chapter/10.1007/978-981-97-8160-7_13 |
| **Year** | 2025 |

**Summary**  
Applies XGBoost with grid search hyperparameter tuning as the core prediction method for cricket outcomes, building on Lamsal & Choudhary's IPL classification work (arXiv 1809) and Vistro et al.'s data analytics approach.

**Approach**
- Grid search over XGBoost hyperparameter space
- Classification of match winner
- Comparison against baseline methods from prior literature

---

## 4. IPL — Auction, Player Price & Fantasy

---

### Paper 22: Prediction of Player Price in IPL Auction Using Machine Learning Regression Algorithms

| Field | Detail |
|---|---|
| **Source** | IEEE CONECCT 2020 / Semantic Scholar |
| **Authors** | Jhansi Rani P., Apurva Kulkarni, Aditya V. Kamath, Aadith Menon, Prajwal Dhatwalia, Rishabh D |
| **DOI** | IEEE CONECCT 2020 |
| **URL** | https://ieeexplore.ieee.org/document/9198668/ |
| **Year** | 2020 |

**Summary**  
Predicts the price at which a player is sold at the IPL auction, using their past performance statistics and accounting for inflation.

**Algorithms Tested**
- Decision Tree Regressor
- K-Nearest Neighbors (KNN)
- Linear Regression
- Stochastic Logistic Regression
- Random Forest Regressor
- Support Vector Regression (SVR)

**Best Results**
- Batsmen: SVR performs best
- Bowlers: Linear Regression performs best
- Results produced within 3 seconds — suitable for real-time auction use

**Features Used**  
Runs, balls, innings, wickets, matches played, inflation factor

**Special Feature**  
Inflation mapping to budget — accounts for the fact that base prices change season to season with inflation.

---

### Paper 23: ipl_auction_predictor

| Field | Detail |
|---|---|
| **Source** | GitHub |
| **Author** | shiv6146 |
| **URL** | https://github.com/shiv6146/ipl_auction_predictor |
| **Type** | Project / Repository |

**Summary**  
Fetches live player stats and predicts both auction price and the most likely team to bid for a player.

**Models Used**
- Regression models — predict auction price
- Classification models — predict potential bidding team

**Data Sources**  
Historical auction data + player statistics scraped from web sources

---

### Paper 24: Multiple Regression to Calculate IPL Player Auction Price Using Player Performance Attributes

| Field | Detail |
|---|---|
| **Source** | Springer |
| **URL** | https://link.springer.com/chapter/10.1007/978-981-96-4679-1_28 |

**Summary**  
Builds two regression models that go beyond raw stats — introducing composite metrics like Eigen-factor score, runs-above-average, and Value-Rank combinations.

**Two Models**
- **VRW** (Value-Rank-Wins)
- **VRRWE** (Value-Rank-RAA-Wins-EFScore)

**Dataset**  
659 players from IPL 2008–2022 (at least 1 season played)

**Key Check**  
Whether the actual auction price was higher or lower than what the model would predict — a fair-value assessment tool.

---

### Paper 25: Players Selling Price Prediction in IPL (Analytics Vidhya)

| Field | Detail |
|---|---|
| **Source** | Analytics Vidhya |
| **URL** | https://www.analyticsvidhya.com/blog/2021/07/players-selling-price-prediction-in-ipl-lets-see-if-machine-learning-can-help/ |
| **Year** | July 2021 |

**Summary**  
Tutorial-style walkthrough predicting player selling prices from 2008–2011 IPL auction data (130 players, 26 features).

**Approach**
- Feature selection from 26 columns to most predictive subset
- Regression modelling on sold price
- Covers multi-format performance (ODI, Test) feeding into IPL auction price

---

### Paper 26: IPL Cricket Fantasy Team Prediction for Dream11 using Machine Learning

| Field | Detail |
|---|---|
| **Source** | ACM Digital Library |
| **URL** | https://dl.acm.org/doi/pdf/10.1145/3698062.3698091 |

**Summary**  
Predicts individual player fantasy points for Dream11, then uses those predictions to select the optimal 11-player fantasy team including captain/vice-captain selection.

**Models for Batsmen Prediction**
- Linear Regression
- Decision Tree
- Random Forest
- Support Vector Regression (SVR)
- Best: **Random Forest** (evaluated via RMSE, R², MAE)

**Team Selection Rules**
- At least 1 player per role (batter, bowler, all-rounder, wicket-keeper)
- Maximum 10 players from one team
- Top 11 by predicted fantasy points selected
- Captain and vice-captain = highest two predicted scorers

**Performance**  
Model correctly predicted at least 7 top performers **85% of the time**

---

## 5. General Sports — ML Prediction (Other Sports)

---

### Paper 27: From Players to Champions: A Generalizable ML Approach for FIFA World Cup Match Prediction

| Field | Detail |
|---|---|
| **Source** | arXiv 2505.01902 |
| **URL** | https://arxiv.org/html/2505.01902v1 |
| **Year** | May 2025 |

**Summary**  
Integrates team-level historical data with detailed player-level statistics for World Cup prediction, addressing the unique challenges of tournament structure and national team assembly from diverse clubs.

**Features Used**
- Goals, assists, passing accuracy, tackles (player-level)
- Team rankings, historical tournament data

**Compared Against**
- TGM Research: SVM and neural networks on 2022 World Cup (team-level only)
- Oxford mathematical model (probabilistic, but coarser granularity)

**Novel Contribution**  
Graph-based methods for modeling player interaction within teams — a new direction in sports analytics

---

### Paper 28: Forecasting NCAA Basketball Outcomes with Deep Learning

| Field | Detail |
|---|---|
| **Source** | arXiv 2508.02725 |
| **URL** | https://arxiv.org/pdf/2508.02725 |

**Summary**  
Generates probabilistic predictions for every possible matchup in the NCAA basketball tournament, with explicit focus on calibration metrics (not just accuracy).

**Key Design Choices**
- Advanced feature engineering for tournament context
- Neural architectures evaluated on probabilistic calibration (not just discriminative accuracy)
- Emphasis on managing unbalanced datasets

**Significance**  
One of the clearest sports papers to frame the problem as probabilistic prediction and calibration, rather than binary classification accuracy — directly relevant to IPL probabilistic modeling.

---

### Paper 29: Integration of ML XGBoost and SHAP Models for NBA Game Outcome Prediction

| Field | Detail |
|---|---|
| **Source** | PMC/NCBI |
| **URL** | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11265715/ |

**Summary**  
Real-time in-game NBA outcome prediction using XGBoost, with SHAP values for interpretability. Identifies which features matter most at different stages of the game.

**Key Findings**
- XGBoost highly effective for NBA game outcome prediction
- Field goal %, defensive rebounds, turnovers — important throughout
- First half: assists most critical
- Second half: offensive rebounds and 3-point % most critical

**Significance**  
SHAP integration is a major contribution — provides explainability beyond "the model says X" and directly supports strategic coaching decisions.

---

### Paper 30: A Machine Learning Framework for Sport Result Prediction

| Field | Detail |
|---|---|
| **Source** | ScienceDirect |
| **URL** | https://www.sciencedirect.com/science/article/pii/S2210832717301485 |
| **Year** | 2017 |

**Summary**  
Critical review of ML literature for sport result prediction, focusing specifically on Artificial Neural Network (ANN) applications. Identifies data sources, learning methodologies, and evaluation challenges.

**Topics Covered**
- Historical match results, player performance indicators, opposition information as features
- Review of ANN configurations for different sports
- Evaluation methodology critique
- Challenges: small datasets, class imbalance, non-stationarity of team performance

---

### Paper 31: Machine Learning for Sports Betting: Should Model Selection Use Calibration?

| Field | Detail |
|---|---|
| **Source** | arXiv 2303.06021 |
| **URL** | https://arxiv.org/pdf/2303.06021 |

**Summary**  
Tests the hypothesis that for probabilistic decision problems (betting), **calibration is more important than accuracy** as a model selection criterion.

**Experiment**
- NBA data over multiple seasons
- Betting experiments on a single hold-out season
- Compared accuracy-based vs calibration-based model selection

**Results**

| Selection Criterion | Average ROI |
|---|---|
| Accuracy-based | **−35.17%** |
| Calibration-based | **+34.69%** |

**Key Metric**  
Classwise Expected Calibration Error (ECE) — measures how well predicted probabilities match empirical win rates

**Significance**  
Fundamental insight for any probabilistic sports model: a model that says "70% chance of winning" should be right 70% of the time. Raw accuracy does not guarantee this. Directly applicable to IPL probabilistic modeling.

---

### Paper 32: Predicting Sport Event Outcomes Using Deep Learning (1D CNN + Transformer)

| Field | Detail |
|---|---|
| **Source** | PMC/NCBI |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC12453701/ |

**Summary**  
Hybrid deep learning framework combining 1D CNN and Transformer architecture for sports outcome prediction (applied to European soccer data).

**Architecture**
- **1D CNN**: captures local spatial patterns in structured match data
- **Transformer**: models long-range dependencies via self-attention
- Combined: uncovers nuanced feature interactions

**Dataset**  
European Soccer Dataset (Win/Draw/Defeat labels)

**Significance**  
Demonstrates that hybrid architectures combining local-feature extraction (CNN) with sequence-level context (Transformer) outperform either alone — a technique transferable to cricket's ball-by-ball sequences.

---

### Paper 33: NCAA Bracket Prediction Using Machine Learning and Combinatorial Fusion Analysis

| Field | Detail |
|---|---|
| **Source** | arXiv 2603.10916 |
| **URL** | https://arxiv.org/pdf/2603.10916 |

**Summary**  
Applies ML models and Combinatorial Fusion Analysis (CFA) to bracket prediction for the NCAA men's basketball tournament.

**Key Observation**  
"Sports prediction has an inherently unpredictable nature. Factors such as player injuries, team chemistry, coaching strategies, weather, momentum swings, and underdog performances all complicate prediction tasks."

**Significance**  
Honest framing of the inherent limitations in sports prediction — directly applicable to IPL where "upsets" are common.

---

### Paper 34: Machine Learning Models for DOTA 2 Outcomes Prediction

| Field | Detail |
|---|---|
| **Source** | arXiv 2106.01782 |
| **URL** | https://arxiv.org/pdf/2106.01782 |

**Summary**  
Real-time MOBA game outcome prediction using a novel multi-forward-steps prediction method.

**Models Compared**
- Linear Regression (LR)
- Neural Networks (NN)
- Long Short-Term Memory (LSTM)

**Novel Contribution**  
Built a custom Python server using Game State Integration (GSI) to collect real-time game data — analogous to ball-by-ball data collection in cricket.

---

### Paper 35: Evaluating One-Shot Tournament Predictions

| Field | Detail |
|---|---|
| **Source** | arXiv 1912.07364 |
| **URL** | https://arxiv.org/pdf/1912.07364 |

**Summary**  
Evaluates methodology for predicting entire tournament brackets at once (pre-tournament), focusing on scoring rules and evaluation frameworks.

**Referenced Methods**
- Bradley-Terry paired comparison model
- Ratings-based Poisson simulation (Dyta & Clarke for World Cup)
- Constantinou & Fenton scoring rules for probabilistic football forecasts
- Ekstrøm (2018) on evaluation metrics

---

## 6. Probabilistic / Bayesian — Football (Soccer)

---

### Paper 36: Bayesian Weighted Discrete-Time Dynamic Models for Association Football Prediction

| Field | Detail |
|---|---|
| **Source** | arXiv 2508.05891 |
| **URL** | https://arxiv.org/html/2508.05891v1 |
| **Year** | 2025 |

**Summary**  
State-of-the-art extension of the entire Poisson football modeling lineage, moving to time-varying team strength parameters and implemented in the R package *footBayes*.

**Model Lineage**
| Year | Authors | Contribution |
|---|---|---|
| 1982 | Maher | Double Poisson — independent team goal counts |
| 1997 | Dixon & Coles | Added score correlation + dependence parameter |
| 2003 | Karlis & Ntzoufras | Bivariate Poisson — positive goal dependencies |
| 2011 | Ntzoufras | Bayesian extension of bivariate Poisson |
| 2025 | This paper | Discrete-time dynamic model — time-varying abilities |

**Key Innovation**  
Relaxes the assumption that team offensive/defensive abilities are static — allows them to evolve over the season.

**Software**  
R package: `footBayes` (free, open source)

---

### Paper 37: Bayesian Hierarchical Models for the Prediction of Volleyball Results

| Field | Detail |
|---|---|
| **Source** | arXiv 1911.08791 |
| **URL** | https://arxiv.org/pdf/1911.08791 |

**Summary**  
Applies Bayesian hierarchical Poisson models to volleyball — the same mathematical framework as football score modeling, but for a different sport.

**Model Structure**
- Two conditionally independent Poisson variables (one per team's score)
- Bayesian hierarchical structure for team ability parameters
- Predictions generated via **posterior predictive distribution**

**Significance**  
Shows the Poisson-based Bayesian framework generalizes across sports — relevant for designing a cricket equivalent.

---

### Paper 38: Predicting the Outcome of Sports Competitions Using Poisson Methods

| Field | Detail |
|---|---|
| **Source** | IACIS Journal |
| **URL** | https://iacis.org/iis/2024/1_iis_2024_188-198.pdf |
| **Year** | 2024 |

**Summary**  
Enhances the standard Poisson Distribution model for sports prediction and benchmarks it against alternatives.

**Referenced Work**
- Louzada & Saraiva (2016) — Poisson regression for football scores
- Moyeed & Shahtahmassebi (2016) — Generalized Poisson difference distribution + Bayesian modelling
- Penn & Donnelly (2022) — Double Poisson model
- Raval (2020) — Can Poisson model EPL outcomes?

---

### Paper 39: A Bayesian Approach to Predict Performance in Football: A Case Study

| Field | Detail |
|---|---|
| **Source** | Frontiers in Sports and Active Living |
| **URL** | https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1486928/full |
| **Year** | 2025 |

**Summary**  
Case-study application of Bayesian methods to football performance prediction, situating the work within the full Bayesian sports modeling lineage.

**Key Referenced Models**
- Koopman & Lit — dynamic bivariate Poisson (EPL)
- Olivieri Filho et al. — Bayesian approach for English Championship
- Santos-Fernandez, Wu & Mengersen — "Bayesian statistics meets sports: a comprehensive review" (JQAS 2019)

---

### Paper 40: A Bayesian Approach to Predict Football Matches with Changed Home Advantage (COVID-19)

| Field | Detail |
|---|---|
| **Source** | PMC/NCBI |
| **URL** | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8947042/ |

**Summary**  
Quantifies how COVID-19 behind-closed-doors matches changed home advantage in four major European leagues, and uses that insight to improve ML prediction models.

**Method**
- Bayesian hierarchical Poisson model for goal-scoring
- Estimates changed home advantage as a latent parameter
- Feeds estimated parameters as extra features into downstream ML models
- Verified statistical decrease in home advantage across EPL, Bundesliga, La Liga, Serie A

**Key Innovation**  
Bayesian model feeds into ML model as a feature extractor — a hybrid Bayesian + ML pipeline, directly relevant to IPL toss/venue effects.

---

### Paper 41: Combining Historical Data and Bookmakers' Odds in Modelling Football Scores

| Field | Detail |
|---|---|
| **Source** | arXiv 1802.08848 |
| **URL** | https://arxiv.org/pdf/1802.08848 |

**Summary**  
Builds a hierarchical Bayesian Poisson model whose scoring rates combine two information sources: bookmaker odds and historical match results.

**Method**
- Inverse betting odds converted to probabilities
- Bookmakers' scoring rates derived via Skellam distribution
- Convex combination with historical rates in Poisson model

**Results**  
Good predictive accuracy for top four European leagues; bookmaker information relevant and complementary to historical data.

**Referenced Work**
- Baio & Blangiardo (2010) — Bayesian hierarchical model for football
- Karlis & Ntzoufras (2003) — Bivariate Poisson
- Karlis & Ntzoufras (2009) — Skellam distribution for goal difference

---

### Paper 42: Alternative Ranking Measures to Predict International Football Results

| Field | Detail |
|---|---|
| **Source** | arXiv 2405.10247 |
| **URL** | https://arxiv.org/pdf/2405.10247 |

**Summary**  
Tests alternative team ranking measures (beyond FIFA rankings) as predictors for international football outcomes, building on the classic Bradley-Terry paired comparison model.

**Foundational Model Referenced**
- Bradley & Terry (1952) — Paired Comparisons Model
- Leonard (1977) — Bayesian approach to Bradley-Terry
- Ntzoufras (2011) — Bayesian Modeling Using WinBUGS

---

### Paper 43: Bayesian State-Space Models for Modelling and Prediction of EPL Football Results

| Field | Detail |
|---|---|
| **Source** | Oxford Academic (Journal of the Royal Statistical Society, Series C) |
| **URL** | https://academic.oup.com/jrsssc/article/74/3/717/7929974 |
| **Year** | 2025 |

**Summary**  
Proposes bivariate Bayesian sequential state-space models for EPL prediction, allowing for overdispersion and positive correlation between home and away scores.

**Models Compared**
- Bivariate Bayesian sequential state-space model
- Koopman & Lit (2019) dynamic score-driven time series
- Six weighted likelihood methods (with tested decay parameter)

**Key Finding**  
Bivariate model captures both overdispersion and positive correlation between scores — important for matches where both teams score heavily (or both score little).

---

### Paper 44: Predictive Analysis and Modelling Football Results Using ML for EPL

| Field | Detail |
|---|---|
| **Source** | ScienceDirect |
| **URL** | https://www.sciencedirect.com/science/article/abs/pii/S0169207018300116 |
| **Year** | 2018 |

**Summary**  
Applies ML to EPL prediction and reviews the evolution from Dixon-Coles bivariate Poisson through Bayesian dynamic generalized linear models with MCMC, up to ML-based approaches.

**Historical Approaches Reviewed**
- Dixon & Coles (1997) — bivariate Poisson
- Crowder et al. (2002) — stochastic process approximation
- Bayesian dynamic GLM + MCMC for retrospective analysis

---

## 7. Probabilistic / Bayesian — Cricket

---

### Paper 45: A Probabilistic Approach to Identifying Run Scoring Advantage in the Order of Playing Cricket

| Field | Detail |
|---|---|
| **Source** | arXiv 2007.05894 |
| **URL** | https://arxiv.org/pdf/2007.05894 |

**Summary**  
Uses Bayesian posterior inference to identify whether batting first or second gives a scoring advantage, and proposes a revised target score to equalize win probabilities.

**Model**
Uses Bayes' rule to derive:

```
P(Win | Score > Xf, BatFirst) = P(Score > Xf, BatFirst | Win) × P(Win) / P(Score > Xf, BatFirst)
```

Applied symmetrically for second innings, comparing posterior win probabilities to recommend fairer target revisions.

**Application**  
ODI cricket, but methodology directly applicable to T20/IPL where toss advantage is debated.

---

### Paper 46: A Bayesian Stochastic Model for Team Performance Evaluation in T20 Cricket

| Field | Detail |
|---|---|
| **Source** | GitHub/Blog (srisai85) |
| **URL** | https://srisai85.github.io/T20Cricket/Bayesian_analysis_of_T20Cricket.html |
| **Year** | 2016 |

**Summary**  
Investigates whether power hitting or batting consistency leads to higher T20 series win rates, using a Bayesian simulation framework.

**Method**
- 1,000 simulations per 7-match series
- Repeated random sampling to generate probability distributions
- Bayesian posterior updating of prior beliefs about batting performance
- Full innings simulated (20 overs) per game

**Conclusion**  
Power hitting is the superior contributor to higher T20 win percentages over consistency.

**Key Distinction from Frequentist Methods**  
Paper explicitly discusses how Bayesian approach treats parameters as having probability distributions rather than fixed constants — a fundamental philosophical distinction.

---

### Paper 47: Predicting Cricket Outcomes using Bayesian Priors

| Field | Detail |
|---|---|
| **Source** | ResearchGate |
| **URL** | https://www.researchgate.net/publication/359389832 |
| **Year** | 2022 |

**Summary**  
Incorporates stratified survey sampling with Bayesian priors on player performance against specific opponents, tested on IPL 2020 and the 2023 ICC World Cup.

**Approach**
- Player performance history used as Bayesian priors
- Priors conditioned on specific opposition (not just career average)
- Simulation used to generate win probability for all team matchups
- Probabilities output for each participating team

**IPL 2020 Results**  
Correctly predicted the top three finishers, including winner Mumbai Indians

---

### Paper 48: Application of Probabilistic Methods to Predict Individual Match Results in the IPL

| Field | Detail |
|---|---|
| **Source** | Academia.edu |
| **URL** | https://www.academia.edu/23680866 |

**Summary**  
Surveys four generic modeling categories (empirical, dynamic systems, statistical, AI) for cricket prediction, then focuses on probabilistic classification models, building the COP (Cricket Outcome Predictor) tool.

**Algorithms**
- Naive Bayes
- Support Vector Machines
- Random Forest

**Tool Output**  
Win/loss probability — not just a label, making this one of the explicitly probabilistic IPL papers

**Referenced Probabilistic Cricket Work**
- Swartz et al. (2009) — discrete probabilistic model for cricket simulation
- Baker & Scarf (2006) — likelihood functions for annual sporting contests
- Preston & Thomas (2002) — probabilities of victory in cricket

---

### Paper 49: In-Game Win Prediction Models for Cricket

| Field | Detail |
|---|---|
| **Source** | Springer (book chapter) |
| **URL** | https://link.springer.com/chapter/10.1007/978-3-031-67871-4_11 |

**Summary**  
The most technically sophisticated Bayesian cricket paper in this list. Extends the Asif & McHale (2016) dynamic logistic regression framework for IPL win prediction into a full Bayesian setting.

**Two Specific Advances Over Asif & McHale**
1. **Power priors** — integrates historical IPL data as informative priors, weighted by recency
2. **Gaussian Process smoothing** — allows model coefficients to vary smoothly across the match (instead of one fixed coefficient for all overs), capturing the difference between powerplay and death-overs dynamics

**Data**  
Ball-by-ball data from hundreds of IPL matches

**Evaluation**  
Cross-validation across multiple IPL seasons

**Significance**  
The closest existing paper to the full probabilistic Bayesian in-game model for IPL described in the 1,000-word methodology summary in this conversation.

---

## 8. Probabilistic / Bayesian — Other Sports

---

### Paper 50: Research on Dynamic Analysis and Prediction Model of Tennis Match Based on Bayesian Probability and AHP

| Field | Detail |
|---|---|
| **Source** | arXiv 2407.07116 |
| **URL** | https://arxiv.org/pdf/2407.07116 |

**Summary**  
Applies Bayesian probability and Analytic Hierarchy Process (AHP) to tennis match prediction, validated on the 2023 Wimbledon Gentlemen's Final (Alcaraz vs Djokovic).

**Model Components**

| Component | Method | Output |
|---|---|---|
| Serve win probability | Bayesian + logistic regression multi-class | P(win first serve) = 0.6734 |
| "Momentum" model | AHP unsupervised ranking | Momentum score → win rate correlation |
| Psychological factors | Trend analysis | Significant impact on match result |
| Generalisation | Tested on women's tennis | Confirmed |

---

### Paper 51: Modeling the Probability of a Batter/Pitcher Matchup Event: A Bayesian Approach (Baseball)

| Field | Detail |
|---|---|
| **Source** | PMC/NCBI |
| **URL** | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6192592/ |

**Summary**  
Develops a Bayesian hierarchical log5 model to estimate the probability of specific matchup outcomes between a batter and pitcher, using limited head-to-head data.

**Model**
- Combines standard log5 model with generalized log5 model
- Bayesian hierarchical approach handles small sample sizes via shrinkage toward prior
- Does not require platoon configuration assumptions unlike the generalized log5

**Key Idea**  
When a batter has a special rivalry record against a specific pitcher, that head-to-head record is more predictive than the batter's overall average — Bayesian shrinkage combines both appropriately.

**Analogy to IPL**  
Directly analogous to modeling batsman-vs-bowler head-to-head matchup probabilities in IPL — e.g., Bumrah vs Kohli career matchup probability conditioned on limited data.

---

## Key Themes Across All 51 Sources

### Algorithms by Frequency

| Algorithm | Papers Using It |
|---|---|
| Random Forest | 1, 2, 3, 6, 7, 16, 17, 18, 19, 20, 25, 26 |
| Logistic Regression | 2, 7, 16, 19, 20, 27, 49, 50 |
| XGBoost | 7, 17, 18, 19, 21 |
| SVM / SVR | 2, 17, 22, 48 |
| LSTM / RNN | 8, 12, 34 |
| Naive Bayes | 6, 20, 48 |
| Bayesian Hierarchical Poisson | 36, 37, 38, 39, 40, 41, 43, 44 |
| Gaussian Process | 49 |
| Neural Network / Deep Learning | 10, 11, 13, 14, 15, 28, 32 |
| Decision Tree | 3, 6, 17 |
| KNN | 2, 6, 22 |

### Evaluation Metrics Used

| Metric | Type | Used For |
|---|---|---|
| Accuracy | Classification | Winner prediction |
| Precision / Recall / F1 | Classification | Winner prediction |
| MAE | Regression | Score prediction |
| RMSE | Regression | Score prediction |
| R² | Regression | Score prediction |
| Brier Score | Probabilistic | Calibration |
| Expected Calibration Error (ECE) | Probabilistic | Calibration |
| Return on Investment (ROI) | Economic | Betting applications |

### What Separates Probabilistic from Standard ML Papers

| Dimension | Standard ML Papers (1–26) | Probabilistic / Bayesian Papers (36–51) |
|---|---|---|
| Output | Win/loss label | Probability distribution P(win) |
| Uncertainty | None — point estimate | Explicit posterior uncertainty |
| Team strength | Engineered feature (average) | Latent parameter with prior |
| New teams/players | Out-of-distribution problem | Shrinkage toward prior handles it |
| Evaluation | Accuracy | Calibration + accuracy |
| Temporal dynamics | Static feature table | Dynamic model (updates ball-by-ball) |

---

## Methodology: Building a Probabilistic ML Model to Predict IPL Winners

> This section synthesises the key insights from the 51 papers above into a single coherent architecture for an IPL win-probability model.

Combining the cricket-specific Bayesian work with the broader sports-modeling literature suggests a fairly clear architecture: rather than treating "win/loss" as a single classification problem solved by Random Forest or XGBoost, the goal is to **estimate a calibrated posterior probability that evolves as the match unfolds**, built on a layered model rather than one black-box classifier.

---

### Step 1 — Frame the Problem Correctly

The single biggest shift from the standard ML papers (Random Forest, XGBoost, SVM) is moving from "predict a label" to "predict a probability distribution and update it." This is exactly what the dynamic logistic regression framework (Asif & McHale 2016) and its Bayesian extension (**Paper 49**) do for IPL win prediction — they model win probability as a function that updates ball-by-ball, rather than a static pre-match prediction.

Adopt this framing from the start:

> Your target is not just `P(Team A wins)` at time zero. It is `P(Team A wins | state at ball t)` for every `t`, with pre-match prediction as the `t = 0` special case.

---

### Step 2 — Build a Hierarchical "Team Strength" Layer First

Before any match-specific features, every football Poisson lineage (**Papers 36–44**) starts the same way: each team gets latent "ability" parameters (attack/defence strength in football; batting/bowling strength in cricket) estimated from historical results, with a prior that lets new or sparse-data teams shrink toward a league average.

For IPL, this maps naturally onto:

- **Team-level latent strength** — batting strength, bowling strength — estimated hierarchically across seasons so that a team's current strength borrows information from its historical strength rather than being estimated from a single season's small sample.
- **Player-level latent ability** — aggregated up into team strength. This mirrors how **Paper 47** (Bayesian Priors) conditioned predictions on individual player history against specific opponents, and how **Paper 51** (baseball batter/pitcher) used a hierarchical log5 model to estimate head-to-head probabilities from limited data.

This hierarchical structure is the main ingredient missing from the vanilla Random Forest/XGBoost IPL papers (**Papers 1–9**): those treat each match as an independent row of static features, so a team's "strength" gets baked into engineered averages rather than being a properly estimated, uncertainty-aware parameter that improves as more data arrives.

---

### Step 3 — Choose the Right Likelihood for the Outcome

Football models score as goals via Poisson/bivariate-Poisson because goals are counts (**Papers 36, 37, 38, 41**). Cricket's natural analogue is messier — wickets and balls remaining interact non-linearly — so cricket papers instead model **win/loss directly via a logistic (Bernoulli) likelihood**, with the linear predictor built from match-state features plus the team-strength parameters from Step 2.

**Match-state features to include:**
- Runs needed
- Balls remaining
- Wickets in hand
- Required run rate vs. current run rate
- Phase of match (Powerplay / Middle overs / Death overs)

This is the approach in **Paper 49** (in-game IPL Bayesian model), and it is the right choice: a logistic-regression-style win-probability model with Bayesian inference replacing point-estimate logistic regression, giving a full posterior over the coefficients rather than a single number.

---

### Step 4 — Make It Dynamic, Not Static

A key insight from **Paper 49** is the addition of **Gaussian Process (GP) smoothing over the dynamic coefficients**. Instead of fitting one fixed coefficient for "runs required per ball" across the whole match, the coefficients themselves are allowed to vary smoothly as the match progresses — early overs behave differently from death overs.

For your model, this means:

- Do **not** fit one static classifier on a flattened feature table.
- Fit coefficients (or a small neural correction layer) that vary by match phase, with a GP or spline prior enforcing smooth — not abrupt — changes over balls/overs.
- **Paper 8** (LSTM) shows a related approach: a single sequential model that implicitly learns how match-state features interact differently at different points in an innings.

---

### Step 5 — Borrow Strength Across Seasons with Power Priors

Because IPL only has ~15 seasons and ~70 matches per season, naive year-by-year fitting is data-starved. The Bayesian football models solve this with **power priors and weighted-likelihood approaches** (**Papers 40, 41, 43**) — letting older seasons inform the current model but down-weighting their influence the further back they go.

Apply the same idea to IPL:

- Weight historical seasons by recency when updating team-strength priors.
- A decay parameter (optimised on held-out seasons) controls how fast past information fades.
- This is similar to what **Paper 43** (EPL state-space) explicitly tested: decay-weighted historical data vs. full dynamic state-space models.

---

### Step 6 — Layer in Auxiliary Information

The football literature shows that **combining historical data with bookmaker odds** improves calibration over historical data alone (**Paper 41**). For IPL, the analogous auxiliary signals are:

| Signal | How to Use |
|---|---|
| Toss outcome | Informative prior / covariate (venue-specific toss win-rate) |
| Venue-specific scoring rates | Prior on first-innings expected score at that ground |
| Head-to-head team history | Prior on latent matchup strength |
| Player injury/availability | Binary flag adjusting team strength parameter |
| Betting market odds (if available) | Informative prior via Skellam/inverse-odds conversion (Paper 41) |

These should enter as **informative priors or additional covariates**, not dummy-coded features the way plain ML papers treat them.

---

### Step 7 — Practical Implementation Pipeline

```
1. DATA
   └── Ball-by-ball IPL data (Cricsheet), 2008–present
       + Player career stats per season
       + Venue/toss metadata

2. PRIORS
   └── Hierarchical priors on team & player strength
       └── Shrinkage toward league mean for new/sparse entities
       └── Power priors weighting older seasons less (decay parameter)

3. MODEL
   └── Bayesian dynamic logistic regression
       └── Win probability = f(match-state features, team-strength params)
       └── Coefficients smoothed over time via Gaussian Process
       └── Bernoulli likelihood for win/loss outcome

4. INFERENCE
   └── MCMC (Stan / PyMC) for full posterior — best for accuracy
   └── Variational inference / NumPyro for speed at scale
   └── This replaces point-estimate logistic regression from Papers 2, 3, 16

5. CALIBRATION CHECK
   └── Evaluate using Brier Score and Expected Calibration Error (ECE)
   └── NOT just accuracy (see Paper 31 — calibration-first selection
       raised ROI from −35% to +34% on NBA data)
   └── A "60% confident" prediction should be right 60% of the time

6. VALIDATION
   └── Leave-one-season-out cross-validation
   └── NOT random train/test splits (match order and team evolution matter)
   └── Mirrors the EPL state-space study (Paper 43) methodology
```

---

### Why This Beats the Plain ML Papers

| Dimension | Random Forest / XGBoost Papers | Proposed Probabilistic Model |
|---|---|---|
| Output | Win/loss label | Full posterior P(win) at every ball |
| Uncertainty | None | Credible intervals on win probability |
| Team/player strength | Engineered feature (average) | Latent hierarchical parameter |
| New players or teams | Out-of-distribution problem | Shrinkage prior handles it |
| Match dynamics | Static — one row per match | Dynamic — updates ball-by-ball |
| Evaluation | Accuracy (~88–94% claimed) | Calibration + Brier Score |
| Interpretability | Black box | SHAP (Paper 29) or posterior coefficient inspection |
| Cross-validation | Random split | Leave-one-season-out |

The Random Forest/XGBoost IPL papers (**Papers 1–9**) get good point accuracy but report no calibration, no uncertainty, and treat each match as independent — so a "94% accuracy" claim can hide badly miscalibrated confidence and will not gracefully handle new players or rare venue/team combinations.

A hierarchical Bayesian dynamic model:
- Naturally handles small-sample teams/players via **shrinkage**
- Updates its **uncertainty** as the match progresses
- Gives an **honest probability** rather than a label
- Is directly usable for win-probability broadcast graphics, fantasy cricket, in-game strategy, and betting applications

---

### Recommended Libraries and Tools

| Task | Tool |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Bayesian modelling | `PyMC`, `Stan` (via `cmdstanpy`), `NumPyro` |
| Variational inference (fast) | `NumPyro` + JAX |
| Gaussian Processes | `GPyTorch`, `PyMC GP module` |
| ML baselines | `scikit-learn`, `XGBoost`, `LightGBM` |
| Calibration evaluation | `sklearn.calibration`, manual ECE computation |
| SHAP interpretability | `shap` library |
| Visualisation | `matplotlib`, `plotly` (for live win-probability graphs) |

---

*Last updated: June 2026 | Compiled from IEEE Xplore, arXiv, ResearchGate, ACM, Springer, ScienceDirect, PMC/NCBI, and GitHub*
