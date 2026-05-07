# 🏀 NBA Prediction & Betting Automation (2026)

This repository automates the complete **end-to-end NBA prediction pipeline**, including:

- Daily **scraping of historical & upcoming games**  
- **Machine-learning predictions** using LightGBM  
- **Isotonic regression calibration**  
- **Grid search for optimal betting parameters**  
- **Daily bet shortlist generation**  
- Fully automated via **GitHub Actions**

All outputs are saved under:  
`2026/LightGBM/`

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
**Source:** `combined_nba_predictions_*` (outputs under `2026/LightGBM/`)  
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


## Probability chain (live-safe + OOS calibration)

The betting pipeline now uses an explicit probability chain with separate historical and live columns:

- `home_team_prob`: raw LightGBM probability.
- `prob_iso`: in-sample isotonic (reference only).
- `prob_iso_oos_time`: walk-forward time OOS isotonic for played rows (`PROB_COL_HIST`).
- `prob_live_oos_proxy`: live proxy calibrator built only from OOS-labeled history.
- `prob_live_safe_pre_clip`: live-safe probability after market-gap guards + continuous shrink.
- `prob_base`: pre-clip EV-driving value.
- `prob_used`: final clipped value used by EV and bet filtering (`PROB_COL_LIVE`).

Runtime logs include a line like:

`[LIVE OOS PROXY] ready=True train_rows=312 win_rate=0.5417`

and debug flags such as `model_market_gap_flag`, `live_underdog_upscale_guard_triggered`, and `live_shrink_triggered` are persisted into combined outputs, shortlist files, and live bet logs for full traceability.

## Output schemas and verification

The pipeline enforces canonical snake_case CSV headers for downstream compatibility.

- `2026/output/LightGBM/combined_nba_predictions_acc_<DATE>.csv`
- `2026/output/LightGBM/Kelly/combined_nba_predictions_iso_<DATE>.csv`

Expected leading columns:
`home_team,away_team,home_team_prob,odds_1,odds_2,result,date,accuracy,prob_iso_insample,prob_iso_oos_time,prob_live_oos_proxy,prob_live_safe_pre_clip,prob_base,prob_live_safe,prob_used`

- `2026/output/LightGBM/Kelly/bet_shortlist_<DATE>.csv` always exists and always has a single header row.
- If no games qualify, shortlist contains header only (no blank lines).
- `2026/output/LightGBM/bet_log_flat_live.csv` is continuously updated from the latest combined predictions and shortlist.

### Automated verification

Run this check locally (also executed in CI after scripts 4 and 5):

```bash
python 2026/scripts/verify_outputs.py
```

The verifier checks:

1. Latest ACC and ISO files exist and share the same run date.
2. ACC/ISO headers are single-line and match the expected schema.
3. Latest shortlist exists and matches expected schema.
4. Ledger exists and is not older than latest shortlist date when shortlist rows are present.

---

## Hoops Insight Agent Chat MVP

The dashboard includes an **Agent Chat** panel, but it is intentionally not a pure frontend chatbot. The static page never stores or sends OpenAI/GitHub secrets from browser code. Instead, it only sends the user's question to a backend/serverless endpoint that owns secrets and enforces tool permissions.

```text
Hoops Insight dashboard
  -> Agent Chat panel
  -> backend/serverless API
  -> LLM + allowlisted tools
  -> GitHub repo outputs / workflow artifacts
  -> answer returned to dashboard
```

### Frontend contract

Configure the deployed dashboard with either:

```html
<meta name="hoops-agent-api" content="https://your-agent.example.com/api/agent">
```

or set `window.HOOPS_AGENT_API_URL` before the dashboard script runs.

When configured, the dashboard sends a read-only request shaped like:

```json
{
  "question": "Explain today's canonical bets and near misses.",
  "capability": "read_only",
  "context": {
    "dashboard_state_url": "public/data/dashboard_state.json",
    "metrics_url": "public/data/metrics_snapshot.json",
    "agent_manifest_url": "public/data/agent_manifest.json"
  }
}
```

If no endpoint is configured, the Agent Chat panel remains visible as a safe planning/stub UI and will not send the question anywhere.

### Read-only v1 sources

`node scripts/build_dashboard_assets.js` now publishes `web/public/data/agent_manifest.json` plus optional latest agent-readable outputs when they exist:

- `combined_latest.csv`
- `local_matched_games_latest.csv`
- `metrics_snapshot.json`
- `bet_log_flat_live.csv`
- `stage1_daily_snapshot_latest.csv` / `.json`
- `setup_profitability_scan_latest.csv` / summary JSON
- `script11_watchlist_history_latest.csv` / summary JSON
- `actual_bets_manual.csv` if present

### Backend guardrails

The backend should hold `OPENAI_API_KEY`, `GITHUB_TOKEN`, repo allowlists, and workflow allowlists. The recommended first version is read-only and may answer questions about today's board, canonical vs. near-miss candidates, setup-profitability candidates, no-bet discipline, manual bet logs, and Steadivus lessons.

Allowed for v1:

- Read latest CSV/JSON outputs and dashboard assets.
- Fetch GitHub workflow artifacts/logs for known workflows.
- Return structured analysis fields such as `canonical_signal`, `near_miss`, `vibe_candidate`, `skip_reason`, `suggested_stake_class`, and `steadivus_lesson`.

Not allowed:

- Place bets or access betting accounts.
- Run arbitrary shell commands.
- Store API keys or GitHub tokens in React/Vite/browser code.
- Push code, mutate historical outputs, or trigger write actions without explicit confirmation.
