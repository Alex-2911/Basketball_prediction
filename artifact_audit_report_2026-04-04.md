# Artifact Audit Report (generated 2026-04-04 UTC)

Repository inspected: `Basketball_prediction` (current local branch: `work`).

## Commands used
- `python` filesystem enumeration for dated artifact families and gap analysis.
- `python` content scan for `strategy_params` embedded dates and `bet_log_flat_live.csv` max date.
- `git branch --show-current`.

## Scope inspected
- `2026/output/LightGBM/`
- `2026/output/LightGBM/Kelly/`
- `web/public/data/`
- repository-wide fallback search for specified filename families.

## A) Artifact family presence/history

### 1) combined_nba_predictions_acc_YYYY-MM-DD.csv
- Exists: **yes**
- Dated files found: **173**
- Earliest date: **2025-05-04**
- Latest 10 dates:
  - 2026-03-17
  - 2026-03-19
  - 2026-03-21
  - 2026-03-23
  - 2026-03-26
  - 2026-03-28
  - 2026-03-30
  - 2026-04-01
  - 2026-04-03
  - 2026-04-04
- Gaps: **yes** (163 missing dates between earliest and latest)
- Paths:
  - `2025/output/LightGBM/1_2025_Prediction/combined_nba_predictions_acc_*.csv`
  - `2026/output/LightGBM/combined_nba_predictions_acc_*.csv`

### 2) combined_nba_predictions_iso_YYYY-MM-DD.csv
- Exists: **yes**
- Dated files found: **117**
- Earliest date: **2025-11-15**
- Latest 10 dates:
  - 2026-03-17
  - 2026-03-19
  - 2026-03-21
  - 2026-03-23
  - 2026-03-26
  - 2026-03-28
  - 2026-03-30
  - 2026-04-01
  - 2026-04-03
  - 2026-04-04
- Gaps: **yes** (24 missing dates between earliest and latest)
- Paths:
  - `2026/output/LightGBM/Kelly/combined_nba_predictions_iso_*.csv`

### 3) local_matched_games_YYYY-MM-DD.csv
- Exists: **yes**
- Dated files found: **5**
- Earliest date: **2026-01-05**
- Latest 10 dates (all available):
  - 2026-01-05
  - 2026-01-06
  - 2026-01-08
  - 2026-01-16
  - 2026-01-17
- Gaps: **yes** (sparse historical availability)
- Paths:
  - `2026/output/LightGBM/local_matched_games_*.csv`

### 4) strategy_params_YYYY-MM-DD.json
- Exists: **no**
- Dated files found: **0**

### 5) strategy_params_YYYY-MM-DD.txt
- Exists: **no**
- Dated files found: **0**

### 6) Undated strategy_params.*
- `2026/output/LightGBM/strategy_params.txt` (exists)
- `web/public/data/strategy_params.json` (exists)

### 7) bet_log_flat_live.csv
- Exists: **yes**
- Paths:
  - `2026/bet_log/bet_log_flat_live.csv`
  - `2026/output/LightGBM/bet_log_flat_live.csv`
  - `web/public/data/bet_log_flat_live.csv`

### 8) metrics_snapshot.json
- Exists: **yes**
- Paths:
  - `2026/output/LightGBM/metrics_snapshot.json`
  - `web/public/data/metrics_snapshot.json`

## B) local_matched_games-specific conclusion
- Historically dated `local_matched_games_YYYY-MM-DD.csv` files **do exist**, but only **5** were found and all are in a narrow January 2026 window.
- This is effectively sparse history (not a continuous snapshot timeline).
- If hoops-insight snapshot resolver expects broad/continuous historical snapshots, current availability likely contradicts that requirement.

## C) strategy_params-specific conclusion
- Dated strategy params files found:
  - `strategy_params_YYYY-MM-DD.json`: none
  - `strategy_params_YYYY-MM-DD.txt`: none
- Undated strategy params files found:
  - `2026/output/LightGBM/strategy_params.txt`
  - `web/public/data/strategy_params.json`
- Embedded date in undated files:
  - `as_of_date=2026-04-03` (TXT)
  - `"as_of_date": "2026-04-03"` (JSON)

## D) bet_log_flat_live.csv max date
- Max date detected in all three copies: **2026-03-15**.

## E) Final yes/no conclusions
- Required dated artifacts all present? **No**.
- Failure caused by missing dated local_matched files? **Partly yes** (sparse history: only 5 dates).
- Failure also caused by missing dated strategy_params files? **Yes** (none found).
- Upstream artifacts that must be produced for complete snapshot resolution:
  1. `strategy_params_YYYY-MM-DD.json` and/or `strategy_params_YYYY-MM-DD.txt` for each snapshot date.
  2. A complete historical series of `local_matched_games_YYYY-MM-DD.csv` aligned to the same snapshot dates.
  3. (Consistency check) ensure corresponding dated `combined_nba_predictions_acc_YYYY-MM-DD.csv` and `Kelly/combined_nba_predictions_iso_YYYY-MM-DD.csv` exist for each snapshot date used by resolver.
