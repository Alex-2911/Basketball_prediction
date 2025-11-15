# Basketball Prediction

End-to-end NBA betting pipeline that:

- Scrapes game data from [basketball-reference.com](https://www.basketball-reference.com)  
- Joins it with **closing odds** from an odds API  
- Trains **LightGBM** models to predict home win probabilities  
- Calibrates probabilities with **Isotonic Regression**  
- Runs a **grid search** over betting parameters  
- Builds a **daily bet shortlist** and updates a **live bet log + ROI stats**

The codebase is structured by season (e.g. `2025/`, `2026/`) and can run via **GitHub Actions** on a daily schedule.

---

## 2026 Pipeline Overview

The 2026 branch is implemented as a 5-step pipeline:

### 1️⃣ Script 1 – Previous Game Day Scraper & Parser  
**Goal:** Fetch and parse data for the **previous NBA game day**.

- Scrapes box scores and standings from Basketball-Reference using `requests` / `Selenium` + `BeautifulSoup`
- Cleans HTML tables (header rows, duplicate columns, etc.)
- Updates the season-long stats dataset in `2026/output/Gathering_Data/...`

**Output (examples):**

- `nba_games_YYYY-MM-DD.csv` (raw game stats)
- Rolling / aggregated statistics used later by the model

---

### 2️⃣ Script 2 – Next Game Day Scraper  
**Goal:** Collect **upcoming games** for the next game day.

- Scrapes the NBA schedule and upcoming matchups
- Normalizes team names / codes to a common 3-letter format
- Saves the slate for the next game day in the “Next_Game” folder

**Output (example):**

- `2026/output/Gathering_Data/Next_Game/nba_games_predict_YYYY-MM-DD.csv`  
  → Contains `home_team`, `away_team`, initial `home_team_prob` features, placeholder odds, etc.

---

### 3️⃣ Script 3 – LightGBM Model & Predictions  
**Goal:** Predict **home win probabilities** for upcoming games.

- Loads historical features + labels from Script 1
- Trains a **LightGBM** classifier on past seasons / games
- Generates **raw home win probabilities** for each upcoming game
- Merges them into a daily combined file

**Output (example):**

- `2026/output/LightGBM/combined_nba_predictions_acc_YYYY-MM-DD.csv`  
  with columns like:
  - `home_team`, `away_team`
  - `home_team_prob` (model probability)
  - `odds_1`, `odds_2` (closing odds from API)
  - `result` (once the game is finished)
  - `date`

---

### 4️⃣ Script 4 – Betting Statistics & Accuracy  
**Goal:** Track **historical performance** of the model and betting rules.

- Merges predictions with **actual results**
- Computes accuracy, ROI, and breakdowns (e.g. home favorites vs. dogs)
- Maintains a historical combined file for backtesting Script 5

**Output (examples):**

- Updated `combined_nba_predictions_acc_YYYY-MM-DD.csv`
- Daily Excel summary:  
  `2026/output/LightGBM/Kelly/betting_summary_YYYY-MM-DD.xlsx`

---

### 5️⃣ Script 5 – Isotonic Calibration, Grid Search & Shortlist  
**Goal:** Turn model predictions into a **practical betting strategy**.

1. **Load combined historical predictions**  
   - `combined_nba_predictions_acc_YYYY-MM-DD.csv`
2. **Ensure / create key columns**  
   - `game_date` (from `date`)  
   - `home_team_won` (1 if `result == home_team`, else 0 / NaN for future)  
   - `pred_home_win_proba` (from `home_team_prob`)  
   - `closing_home_odds` (from `odds_1`)  
   - `closing_away_odds` (from `odds_2`)
3. **Merge additional inputs**
   - Today’s upcoming games from `nba_games_predict_YYYY-MM-DD.csv`
   - Home win context from `home_win_rates_sorted_YYYY-MM-DD.csv` (e.g. last-20-games home win rate per team)
4. **Split into**
   - `df_past`: completed games (results known)  
   - `df_future`: upcoming games (today & tomorrow)
5. **Calibrate probabilities**
   - Fit **Isotonic Regression** on `df_past`
   - Add `iso_proba_home_win` for both past and future games
   - Log **Brier score** and **log-loss** before/after calibration
6. **Grid search betting strategies**
   - Parameters:  
     `min_home_win_rate`, `min_odds`, `max_odds`, `min_iso_proba`
   - Backtest each combination on `df_past` with flat stakes
   - Pick best combo by **ROI per bet**, tie-breaking by number of bets
7. **Save results**
   - Full backtest grid:  
     `2026/output/LightGBM/Kelly/nba_grid_search_results_YYYY-MM-DD.csv`
   - Full calibrated predictions:  
     `2026/output/LightGBM/Kelly/combined_nba_predictions_iso_YYYY-MM-DD.csv`
8. **Build today’s bet shortlist**
   - Apply best strategy to `df_future` (today & tomorrow)
   - Filter by:
     - `home_win_rate` (from home win rate file)
     - `closing_home_odds` (odds window)
     - `iso_proba_home_win` (min calibrated edge)
   - Compute flat stake and expected value
   - Write shortlist to:  
     `2026/output/LightGBM/bet_shortlist_YYYY-MM-DD.csv`
   - Update / append to live bet log files in the `Kelly/` folder

If there are **no upcoming games** or **no bets passing the filters**, Script 5 still completes successfully and just logs an empty shortlist for that day.

---

## Run Order (Local)

1. **Script 1** – Previous game day scraping & parsing  
2. **Script 2** – Next game day scraping  
3. **Script 3** – LightGBM training & predictions  
4. **Script 4** – Betting statistics & accuracy update  
5. **Script 5** – Isotonic calibration, grid search & bet shortlist

On GitHub Actions, this sequence is orchestrated automatically for the current season.

---

## Installation

Install all required libraries:

```bash
pip install -r requirements.txt
