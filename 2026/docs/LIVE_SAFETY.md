# Live Safety

The live probability path for upcoming games uses this precedence:

1. `prob_live_oos_proxy`
2. `home_team_prob`

`prob_iso_oos_time` remains the historical evaluation column. `prob_iso` / `iso_proba_home_win` remains the in-sample diagnostic column.

The live guard sequence is:

1. Build `prob_live_safe_pre_clip` from the proxy (or raw fallback).
2. Compute `market_implied_p_raw` and `market_implied_p_devig`.
3. Compute `model_market_gap`.
4. If `odds_1 >= 2.00`, `prob_live_safe_pre_clip >= 0.60`, and `model_market_gap >= 0.12`, set `blocked_by=MODEL_MARKET_GAP` and cap at `UNDERDOG_CAP=0.55`.
5. Blend with market probability using `TAU_GAP=0.08`.
6. Apply continuous shrink above `0.60` using base `alpha=0.85`, reduced to `0.70` for market-gap rows.
7. Clip final `prob_used` into `[0.35, 0.80]`.

Proxy tuning:

- `min_train_rows`: minimum OOS-eligible played rows before the proxy is active.
- `n_bins`: proxy quantile bins; defaults to 25.
- `min_bin_n`: minimum per-bin support before fallback to global rate.
- `recent_n`: optional recent rolling subset for regime adaptation.
- `TAU_GAP`: smaller values force stronger market blending when model and market diverge.
