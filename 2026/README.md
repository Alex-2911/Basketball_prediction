# 2026 Live Probability Columns + Market Safety

## Column precedence for live probabilities

For upcoming rows, `prob_base` is selected as:

1. `prob_live_oos_proxy` when `live_oos_proxy_ready=True`
2. fallback: `home_team_prob`

`prob_live_safe_pre_clip` is then clipped to `[0.35, 0.80]` before market-aware guards.

## Market-aware guard thresholds

- `UNDERDOG_ODDS_GUARD_MIN = 2.00`
- `UNDERDOG_PROB_GUARD_MIN = 0.60`
- `GAP_GUARD_MIN = 0.12`
- `UNDERDOG_CAP = 0.55`
- `TAU_GAP = 0.08`

Policy is **A (strict)**: if hard guard triggers, `blocked_by="MODEL_MARKET_GAP"` and rows are excluded from live shortlist / bet-log additions.

## Tuning

### Live OOS proxy

- `n_bins` controls quantile bin granularity (default 25).
- `min_train_rows` controls readiness (default 300).
- `min_bin_n` controls small-bin fallback to global rate (default 25).

### Market blend

- `TAU_GAP` controls how quickly model probability is blended toward market as model-market gap increases.
  - smaller tau => stronger market pull for the same gap.
