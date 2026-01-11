# 🏀 NBA Prediction & Betting Automation (2026)

This repository automates the complete **end-to-end NBA prediction pipeline**, including:

- Daily **scraping of historical & upcoming games**  
- **Machine-learning predictions** using LightGBM  
- **Isotonic regression calibration**  
- **Grid search for optimal betting parameters**  
- **Daily bet shortlist generation**  
- Fully automated via **GitHub Actions**

All outputs are saved under:  
`2026/output/LightGBM/`

---

## 🔧 Pipeline Overview (Fully Automated Daily)

The 2026 workflow consists of five Python scripts executed automatically in sequence:

---

## **1️⃣ Script 1 — Load Previous Game Day**
**File:** `1_get_data_previous_game_day_2026.py`  
**Purpose:**  
- Scrapes and parses **yesterday’s NBA box scores**  
- Updates the historical dataset  
- Cleans and stores structured statistics  

**Output:** Updated historical data.

---

## **2️⃣ Script 2 — Load Next Game Day**
**File:** `2_get_data_next_game_day_2026.py`  
**Purpose:**  
- Scrapes **upcoming NBA games**  
- Collects schedule, matchups, opening odds  
- Prepares the input for predictions  

**Output:**  
`nba_games_predict_YYYY-MM-DD.csv`

---

## **3️⃣ Script 3 — Predict Next Game Day (LightGBM)**
**File:** `3_predict_games_hybrid_2026.py`  
**Purpose:**  
- Creates ML features from rolling team stats  
- Uses LightGBM to predict **home win probabilities**  
- Merges predictions with historical data  

**Output:**  
`combined_nba_predictions_acc_YYYY-MM-DD.csv`

---

## **4️⃣ Script 4 — Merge Results & Betting Statistics**
**File:** `4_calculate_betting_statistics_2026.py`  
**Purpose:**  
- Merges actual NBA results once games are finished  
- Updates model accuracy and betting metrics  
- Maintains the master combined prediction file  

**Outputs:**  
- Updated combined CSV  
- Daily bet logs  
- Daily betting summary sheets

---

## **5️⃣ Script 5 — Isotonic Calibration, Grid Search & Shortlist**
**File:** `5_isotonic_based_betting_strategy_2026.py`  
**Purpose:**  
- Fits **Isotonic Regression** to calibrate predictions  
- Loads **today’s predictions** and **home win rate data**  
- Runs a **grid search over 256 parameter combinations**:
  - Minimum home win rate  
  - Minimum & maximum odds  
  - Minimum calibrated win probability  
- Selects the **best strategy** by ROI per bet  
- Applies it to **today/tomorrow’s games**  
- Generates the final **bet shortlist**  

**Outputs:**  
- `Kelly/nba_grid_search_results_YYYY-MM-DD.csv`  
- `Kelly/combined_nba_predictions_iso_YYYY-MM-DD.csv`  
- `bet_shortlist_YYYY-MM-DD.csv`

If no bets pass the filters, the shortlist may be empty.

---

## 📁 Repository Structure

```text
2026/
├── src/
│   ├── 1_get_data_previous_game_day_2026.py
│   ├── 2_get_data_next_game_day_2026.py
│   ├── 3_predict_games_hybrid_2026.py
│   ├── 4_calculate_betting_statistics_2026.py
│   ├── 5_isotonic_based_betting_strategy_2026.py
│   └── nba_utils_2026.py
│
└── output/
    └── LightGBM/
        ├── Kelly/
        ├── bet logs (bet_log_YYYY-MM-DD.csv)
        ├── bet summaries (betting_summary_YYYY-MM-DD.xlsx)
        ├── predictions (combined_*.csv)
        ├── home_win_rates_sorted_*.csv)
        └── nba_games_predict_*.csv
```

---

## 📊 KPI Definitions & Data Sources

This repo separates **historical model performance**, **strategy simulations**, and **real placed bets** to avoid mixing simulated and real-world results.

### ✅ Windowed Historical Results (Last 200 Games)
**Source:** `combined_nba_predictions_*` (outputs under `2026/output/LightGBM/`)  
**Used for:**
- Overall accuracy
- Calibration metrics (Brier, LogLoss, ECE, Slope)
- Home win rates
- Strategy filter coverage

### ✅ Strategy / Local Params (Simulated, Windowed)
**Source:** `local_matched_games_YYYY-MM-DD.csv` (last-200 window)  
**Used for:**
- Local Matched Games table
- Simulated bankroll (Last 200 games)
- Sharpe ratio and drawdown for the strategy

### ✅ Placed & Settled Bets (Real, 2026 YTD)
**Source:** `bet_log_flat_live.csv` (settled using `combined_*` results by date + home + away)  
**Rules:**
- Only settled rows (win + pnl available) are counted
- Deduping is stable on `(date, home, away)`
- Missing stake/odds rows are discarded

**Used for:**
- Bankroll (2026 YTD)
- Settled bets overview and table

### ✅ Active Filters & Params Used
**Source:** `metrics_snapshot.json` (`params_used`, `params_used_type`)  
**Notes:**
- Active filters are rendered from **params_used** (LOCAL or GLOBAL)
- `strategy_params.txt` is used only as a fallback

This ensures **simulation ≠ reality**, while keeping both views consistent and auditable.
