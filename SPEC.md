# Basketball Prediction Logic Specification (2026)

## 1. Purpose

This repository implements an end-to-end NBA data, prediction, calibration, and betting-automation pipeline for the 2026 season.

The system’s goals are to:

- collect historical NBA results and upcoming fixtures,
- train a model to estimate home-team win probability,
- calibrate and adjust probabilities for live betting,
- generate a shortlist of qualifying bets,
- maintain a settled bet ledger,
- publish dashboard-ready artifacts.

The repository is organized around a five-script daily pipeline plus dashboard and validation helpers.

---

## 2. High-Level Flow

Canonical daily flow:

1. Scrape previous game-day results.
2. Scrape next game-day schedule.
3. Train/predict next-day games with LightGBM and odds.
4. Merge actual results into cumulative prediction history and compute betting statistics.
5. Build isotonic-calibrated probabilities, apply live-safety logic, produce a shortlist, and update the bet ledger.

Primary output tree:

- `2026/output/LightGBM`
- `2026/output/LightGBM/Kelly`
- `web/public/data`

---

## 3. Core Data Domains

### 3.1 Historical game statistics
Produced by Script 1 and consumed by later scripts.

Typical fields include:

- `team`
- `team_opp`
- `date`
- `season`
- `won`
- team stats, opponent-mirrored stats, and rolling-feature columns

### 3.2 Upcoming games
Produced by Script 2.

Required fields:

- `home_team`
- `away_team`
- `game_date`

### 3.3 Combined prediction history
Maintained by Scripts 3 and 4, enriched by Script 5.

Canonical columns include:

- `home_team`
- `away_team`
- `home_team_prob`
- `odds_1`
- `odds_2`
- `result`
- `date`
- `accuracy`

Probability-chain fields:

- `prob_iso`
- `prob_iso_insample`
- `prob_iso_oos_time`
- `prob_live_oos_proxy`
- `prob_live_safe_pre_clip`
- `prob_base`
- `prob_live_safe`
- `prob_used`

### 3.4 Shortlist
Script 5 emits a shortlist containing only bets that survive filtering and safety logic.

Required fields:

- `game_date`
- `home_team`
- `away_team`
- `home_team_prob`
- `prob_iso`
- `prob_iso_oos_time`
- `prob_live_oos_proxy`
- `prob_live_safe_pre_clip`
- `prob_base`
- `prob_used`
- `odds_1`
- `market_implied_p_raw`
- `market_implied_p_devig`
- `model_market_gap`
- `model_market_gap_flag`
- `live_underdog_upscale_guard_triggered`
- `live_shrink_triggered`
- `live_oos_proxy_ready`
- `live_oos_proxy_train_rows`
- `live_oos_proxy_bin_n`
- `live_oos_proxy_bin_winrate`
- `blocked_by`
- `home_win_rate`
- `EV_€_per_100`

### 3.5 Ledger
The live bet ledger tracks settled bets, stake, outcome, and P&L.

Core fields:

- `date`
- `home_team`
- `away_team`
- `stake`
- `odds`
- `pick`
- `status`
- `won`
- `pnl`
- `prob_used`
- `prob_base`
- `prob_live_oos_proxy`
- `prob_live_safe_pre_clip`
- `market_implied_p_raw`
- `market_implied_p_devig`
- `model_market_gap`
- `model_market_gap_flag`
- `live_underdog_upscale_guard_triggered`
- `live_shrink_triggered`
- `live_oos_proxy_ready`
- `live_oos_proxy_train_rows`
- `live_oos_proxy_bin_n`
- `live_oos_proxy_bin_winrate`
- `blocked_by`
- `ev_per_100`
- `created_at_utc`
- `settled_at_utc`
- `source`

---

## 4. Script-by-Script Specification

### 4.1 Script 1: Previous game-day scrape
**File:** `2026/src/1_get_data_previous_game_day_2026.py`

Responsibilities:

- fetch current-month NBA schedule page,
- locate previous completed game day,
- fetch each box-score HTML page,
- validate HTML to reject anti-bot junk,
- parse team box-score tables,
- flatten and normalize stats,
- append home/away rows to historical dataset,
- save daily historical snapshot.

Rules:

- invalid/blocked HTML must not be persisted as box-score data,
- parsing must tolerate Basketball-Reference table quirks and hidden comments,
- valid local box scores should be reused,
- invalid local copies should refresh via requests and then Selenium fallback.

Outputs:

- daily `nba_games_<DATE>.csv`,
- updated schedule HTML in standings storage,
- updated box-score HTML in scores storage.

### 4.2 Script 2: Next game-day scrape
**File:** `2026/src/2_get_data_next_game_day_2026.py`

Responsibilities:

- load schedule HTML for relevant month,
- identify next game day on/after anchor date,
- extract all matchups for that date,
- normalize team names to internal 3-letter codes,
- persist upcoming fixtures to CSV.

Rules:

- must never block waiting for user input,
- if no games are found, still write empty CSV with correct schema,
- if current month has no next game day, continue checking later months.

Output:

- `games_df_<DATE>.csv` in next-game directory.

### 4.3 Script 3: Prediction and odds merge
**File:** `2026/src/3_predict_games_hybrid_2026.py`

Responsibilities:

- load latest historical stats file,
- construct rolling features,
- scale features with MinMax scaling,
- create next-game targets,
- train LightGBM on historical data,
- predict home-win probabilities for upcoming games,
- fetch bookmaker odds from The Odds API,
- merge predictions and odds into daily prediction file.

Key rules:

- `ODDS_API_KEY` must be present,
- fail fast if odds key is missing,
- normalize team names/abbreviations before odds lookup,
- map Phoenix/Charlotte naming differences,
- fallback to latest available games file if today’s file is missing.

Outputs:

- `nba_games_predict_<DATE>.csv`,
- cumulative prediction artifacts for downstream steps.

### 4.4 Script 4: Betting statistics and cumulative predictions
**File:** `2026/src/4_calculate_betting_statistics_2026.py`

Responsibilities:

- load latest prediction file,
- merge actual outcomes into historical prediction snapshots,
- rebuild cumulative combined predictions file,
- normalize combined schema,
- preserve canonical column ordering,
- update accuracy and derived metrics,
- support deduped upsert behavior for game rows.

Important logic:

- game identity = `date + home_team + away_team`,
- row completeness prefers more complete records in dedupe,
- cumulative state can be reconstructed from prior snapshots.

Outputs:

- `combined_nba_predictions_acc_<DATE>.csv`,
- related summaries and historical snapshots.

### 4.5 Script 5: Calibration, safety, shortlist, and metrics
**File:** `2026/src/5_Isotonic_based_betting_strategy_2026.py`

Responsibilities:

- load combined prediction history,
- compute isotonic calibration,
- build time-aware OOS probability chain,
- build live OOS proxy,
- apply market-aware safety guards,
- run parameter search/strategy selection,
- generate shortlist and local matched-game outputs,
- write strategy parameters and metrics snapshots,
- validate dashboard-ready artifact consistency.

#### Probability chain semantics

1. `home_team_prob` — raw LightGBM probability.
2. `prob_iso` — in-sample isotonic reference probability.
3. `prob_iso_oos_time` — time-aware OOS isotonic probability for played rows.
4. `prob_live_oos_proxy` — live proxy from OOS-labeled history.
5. `prob_live_safe_pre_clip` — pre-clip live-safe probability after market-gap guards/shrink.
6. `prob_base` — EV-driving base probability.
7. `prob_used` — final probability for EV and shortlist filters.

#### Safety rules

Safety layer applies:

- clipping to bounded range,
- market-implied probability computation,
- underdog gap guard,
- shrink logic for large model-market gaps,
- hard blocking for strict market-gap violations.

Strict guard rows must be marked:

- `blocked_by = MODEL_MARKET_GAP`

Outputs:

- `combined_nba_predictions_iso_<DATE>.csv`
- `bet_shortlist_<DATE>.csv`
- `strategy_params.txt`
- `strategy_params.json`
- `metrics_snapshot.json`
- `local_matched_games_<DATE>.csv`
- `local_matched_games_latest.csv`

---

## 5. Live Probability Modules

### 5.1 `live_probability_pipeline.py`

Role:

- orchestrates probability-chain transformation,
- loads active strategy parameters from Script-5 outputs,
- coordinates OOS isotonic calibration, live proxy generation, and safety filtering.

Functions:

- load active strategy params,
- build chain configuration,
- prepare live probability columns.

Guarantee:

- output dataframe contains all downstream probability/safety fields, even when some components are unavailable.

### 5.2 `live_oos_proxy.py`

Role:

- estimate live probability proxy from historical OOS-labeled data,
- bin probabilities into quantile buckets,
- use Wilson lower bounds or smoothed rates,
- optionally dedupe repeated games by date/home/away,
- optionally blend proxy output with base probability.

Required behavior:

- mark proxy not ready when training data is insufficient,
- record fallback when preferred source column is unavailable,
- still emit metadata columns for upcoming rows when proxy is unavailable.

Important emitted fields:

- `ready`
- `train_rows`
- `global_win_rate`
- `bin_edges`
- `bin_n`
- `bin_win_rate`
- `source_col_used`
- `fallback_used`
- `fallback_reason`
- `recent_window_used`

### 5.3 `live_safety.py`

Role:

- compute market-implied probabilities,
- compare model vs market probabilities,
- apply underdog and gap guards,
- blend model and market probabilities,
- produce final `prob_used`,
- decide whether a row is blocked.

Important constants include:

- clip bounds,
- underdog odds threshold,
- underdog probability threshold,
- gap threshold,
- underdog cap,
- gap blend temperature.

Behavior:

- when `live_oos_proxy_ready` is true, base probability can come from proxy,
- large positive model-market gaps can trigger block or cap,
- `prob_used` is final probability for betting filters and EV.

---

## 6. Output and Dashboard Specification

### 6.1 Output tree

Main data products:

- `2026/output/Gathering_Data`
- `2026/output/LightGBM`
- `2026/output/LightGBM/Kelly`
- `2026/bet_log`
- `web/public/data`

### 6.2 Dashboard assets

Dashboard builder copies or synthesizes:

- `combined_latest.csv`
- `local_matched_games_latest.csv`
- `bet_log_flat_live.csv`
- `metrics_snapshot.json`
- `strategy_params.json`
- `dashboard_state.json`

Dashboard UI reads only from `web/public/data`.

### 6.3 Dashboard state

Dashboard state includes:

- as-of date,
- window size,
- window start/end,
- active filters text,
- source file references,
- bet-log freshness metadata,
- strategy match count in current window.

---

## 7. Validation and Testing

### 7.1 Output schema verification
**Script:** `2026/scripts/verify_outputs.py`

Checks:

- latest ACC and ISO files exist,
- ACC and ISO filenames match by date,
- required columns exist,
- headers are single-line and canonical,
- shortlist exists with expected schema,
- ledger exists and is not older than shortlist when shortlist has rows.

### 7.2 Pipeline consistency verification
**Script:** `2026/scripts/verify_pipeline_consistency.py`

Checks:

- probability-chain parity between Script 5 and ledger views,
- required strategy parameters are discoverable,
- missing required strategy parameters fail loudly,
- fallback behavior without params is predictable.

### 7.3 Live safety tests
**File:** `2026/tests/test_live_safety.py`

Coverage:

- market-gap blocking,
- underdog cap behavior,
- OOS proxy readiness,
- proxy-fill behavior for upcoming rows.

### 7.4 Top-level pipeline regression test
**File:** `tests/test_pipeline_outputs.py`

Coverage:

- ACC/ISO existence,
- shortlist schema,
- probability columns on played rows,
- ledger recency.

---

## 8. Automation Specification

### 8.1 GitHub Actions

Workflow chain:

1. collect previous-day results,
2. collect next-game data,
3. build predictions and betting lines,
4. calculate betting statistics and shortlist,
5. publish dashboard assets.

CI also validates schemas and mirrors current data into dashboard-readable assets.

### 8.2 Local wrappers

Main wrapper: `master_run.sh`

Purpose:

- set `SOURCE_ROOT`,
- set `LGBM_DIR`,
- execute Scripts 1–5,
- run pipeline output checks.

Secondary wrapper: `run_pipeline.sh`

Purpose:

- validate required outputs exist,
- derive date anchor from `metrics_snapshot.json`,
- validate local matched-game exports,
- build dashboard assets.

---

## 9. Configuration Rules

### 9.1 Environment variables

Important variables:

- `SOURCE_ROOT`
- `LGBM_DIR`
- `N_WINDOW`
- `STRATEGY_VARIANT`
- `ODDS_API_KEY`
- `STRATEGY_PARAMS_PATH`
- `SEASON_YEAR`
- `TARGET_DATE`

### 9.2 Default strategy behavior

- local default variant: `acc`,
- CI may default to `iso` depending on workflow path,
- probability clipping defaults to a conservative band,
- historical local window defaults to 200 games.

### 9.3 Strategy parameter sources

Active strategy parameters can come from:

1. `metrics_snapshot.json`
2. `strategy_params.txt`
3. `STRATEGY_PARAMS_PATH` override

Missing required parameters should fail fast.

---

## 10. Canonical Invariants

- canonical CSV headers are snake_case,
- probability columns are always present in combined outputs,
- shortlist files always exist (header-only is valid),
- ledger rows deduplicate by game identity,
- live proxy and live safety metadata persist,
- dashboard assets always reflect latest eligible outputs,
- pipeline does not mix simulated strategy metrics with settled real bets.

---

## 11. Operational Guarantees

The system is designed for:

- reproducible daily outputs,
- auditable probability transformations,
- strict market-aware blocking of risky rows,
- separation of historical model performance from real ledger outcomes,
- dashboard/CI consistency with generated artifacts.

---

## 12. Practical Notes

- historical artifacts and generated outputs are operating state, not sample-only,
- live proxy and safety layers are intentionally conservative and traceable,
- missing/malformed upstream inputs should fail loudly,
- some outputs (especially shortlist) may be empty by design when no bets pass filters.
