# 2026 NBA Pipeline Data Schema

This document defines the expected CSV schemas and file-creation behavior for Script 4/5 and the live bet ledger.

## 1) `combined_nba_predictions_acc_<DATE>.csv`

Location: `2026/output/LightGBM/`

Creation behavior:
- Produced by Script 4 for each run date.
- Refreshed by Script 5 with the calibrated probability chain to keep ACC/ISO aligned.
- Header is always written as a **single line** with canonical snake-case names.

Required leading columns:
- `home_team`
- `away_team`
- `home_team_prob`
- `odds_1`
- `odds_2`
- `result`
- `date`
- `accuracy`

Required probability-chain columns:
- `prob_iso`
- `prob_iso_insample`
- `prob_iso_oos_time`
- `prob_live_oos_proxy`
- `prob_live_safe_pre_clip`
- `prob_base`
- `prob_live_safe`
- `prob_used`

Notes:
- Probability columns are numeric when computable.
- If a specific probability cannot be computed, the field is emitted as `NaN` (not blank/missing column).

## 2) `combined_nba_predictions_iso_<DATE>.csv`

Location: `2026/output/LightGBM/Kelly/`

Creation behavior:
- Always created by Script 5 on each run date.
- Uses the same canonical schema as the ACC output, including the full probability chain.
- If calibration is limited (for example, not enough past rows), file is still emitted with fallback probabilities and `NaN` where needed.

## 3) `bet_shortlist_<DATE>.csv`

Location: `2026/output/LightGBM/Kelly/`

Creation behavior:
- Always created by Script 5 on each run date.
- Header is always a single line.
- If there are no qualifying bets, the file contains header-only output.

Expected header:
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

## 4) `bet_log_flat_live.csv`

Locations:
- `2026/bet_log/bet_log_flat_live.csv`
- `2026/output/LightGBM/bet_log_flat_live.csv` (export mirror)

Creation behavior:
- Updated by `2026/scripts/maintain_bet_log_flat_live.py` every run.
- Appends new bets and settles pending bets when outcomes are available.
- Deduplicated by `game_key` (`date_home_away_pick`).

Core required columns:
- `date`, `home_team`, `away_team`
- `stake`, `odds`, `pick`
- `status`, `won`, `pnl`
- `prob_used`, `prob_base`, `prob_live_oos_proxy`, `prob_live_safe_pre_clip`
- `ev_per_100`
- `created_at_utc`, `settled_at_utc`, `source`

## 5) CI Regression Verification

The pipeline runs two checks after Scripts 4 and 5:
- `2026/scripts/verify_outputs.py`
- `tests/test_pipeline_outputs.py`

These checks validate:
- ACC header shape and probability-column presence.
- ISO file existence for the same latest date.
- Shortlist existence + canonical single-line header (even when empty).
- Ledger recency (at least one entry after January 2026).
