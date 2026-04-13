# SPEC Gap Audit (2026 Pipeline)

Date: 2026-04-13  
Scope: `SPEC.md` vs implemented behavior in scripts/modules/tests.

## Method

- Reviewed Script 1–5 implementations in `2026/src`.
- Reviewed live modules: `live_probability_pipeline.py`, `live_oos_proxy.py`, `live_safety.py`.
- Reviewed validation/test coverage in `2026/scripts`, `2026/tests`, and `tests`.
- Ran repo verifiers and targeted tests.

## Commands Run

- `python 2026/scripts/verify_outputs.py`
- `python 2026/scripts/verify_pipeline_consistency.py`
- `pytest -q 2026/tests/test_live_safety.py tests/test_pipeline_outputs.py`

## Script-by-Script Audit

## 1) `1_get_data_previous_game_day_2026.py`
**Status:** ✅ Mostly aligned

- Implements schedule-page fetch, previous completed game-day collection, box-score retrieval, local-cache reuse, HTML validity checks, requests-first + Selenium fallback, and parsing logic resilient to Basketball-Reference quirks/comments.
- Writes/updates daily historical snapshot and refreshed source HTML artifacts.

## 2) `2_get_data_next_game_day_2026.py`
**Status:** ✅ Aligned (updated)

- Finds next game day on/after anchor date, checks subsequent months when needed, normalizes names to internal codes, writes expected CSV schema.
- Empty-matchup day path writes header-only CSV for downstream stability.
- **Updated in this patch:** exit pause is now a true no-op so Script 2 never blocks for interactive input.

## 3) `3_predict_games_hybrid_2026.py`
**Status:** ✅ Mostly aligned

- Fails fast when `ODDS_API_KEY` is missing.
- Supports fallback to latest `games_df_*.csv` when today's file is absent.
- Includes team normalization/mapping (including PHX/CHA related mappings) prior to odds merge.
- Produces daily prediction CSV with probability-chain placeholders expected downstream.

## 4) `4_calculate_betting_statistics_2026.py`
**Status:** ✅ Aligned

- Rebuilds cumulative ACC snapshots and upserts by canonical game identity.
- Dedup prefers row completeness and recency via scoring/sort.
- Preserves normalized schema and writes dated cumulative artifact.

## 5) `5_Isotonic_based_betting_strategy_2026.py`
**Status:** ✅ Mostly aligned

- Builds isotonic calibration + live/OOS probability chain.
- Persists shortlist, strategy params, metrics snapshot, and local-matched artifacts (including latest aliases).
- Applies safety guard behavior and keeps blocking metadata (`blocked_by`, market-gap flags) in outputs.

## Live modules

## `live_probability_pipeline.py`
**Status:** ✅ Aligned

- Loads active strategy params from snapshot/txt/override paths.
- Enforces required params for strict callers.
- Ensures downstream probability/safety columns exist.

## `live_oos_proxy.py`
**Status:** ✅ Aligned

- Builds proxy from OOS-labeled data with fallback source handling.
- Emits readiness + metadata fields for downstream/traceability.

## `live_safety.py`
**Status:** ✅ Aligned

- Computes market-implied probabilities, applies guard logic, caps/shrink behavior, and final `prob_used`.
- Emits `blocked_by` markers for market-gap guard outcomes.

## Validation/Test Coverage Audit

**Status:** ✅ Improved

- Output/schema and consistency checks are implemented and passing.
- Live safety unit coverage exists and passes.
- Top-level pipeline output regression checks pass.
- Added targeted deterministic tests for:
  - blocked/garbled box-score HTML detection,
  - next-game-day month rollover schedule parsing,
  - partial/degenerate odds-bookmaker payload handling.

## Summary of Gaps

1. **Closed in this patch:** Script 2 interactive pause behavior tightened to hard non-blocking no-op.
2. **Closed in this patch:** added deterministic tests for the scrape/API edge cases previously identified.
3. **Remaining low-risk gap:** broaden integration-style fixtures for additional anti-bot/schedule/odds variants to further raise confidence.

## Recommended Next Steps

- Expand fixture corpus for additional anti-bot HTML signatures and malformed table structures.
- Add more bookmaker payload permutations (multi-book fallback priority, missing team name mapping).
- Keep `verify_outputs.py` and `verify_pipeline_consistency.py` in CI gates (already aligned with spec contracts).
