from __future__ import annotations

import numpy as np
import pandas as pd

from odds_utils import compute_market_probs


UNDERDOG_ODDS_GUARD_MIN = 2.00
UNDERDOG_PROB_GUARD_MIN = 0.60
GAP_GUARD_MIN = 0.12
UNDERDOG_CAP = 0.55
TAU_GAP = 0.08
USE_BLEND_ALWAYS = True
BLEND_PROB_START = 0.60
BASE_SHRINK = 0.85
FLAGGED_SHRINK = 0.70
P_MIN = 0.35
P_MAX = 0.80
STRICT_BLOCK_POLICY = True


def apply_live_safety(df: pd.DataFrame, *, live_oos_proxy_ready: bool) -> pd.DataFrame:
    out = df.copy()

    raw_prob = pd.to_numeric(out.get("home_team_prob"), errors="coerce")
    proxy_prob = pd.to_numeric(out.get("prob_live_oos_proxy"), errors="coerce")
    out["live_oos_proxy_used"] = bool(live_oos_proxy_ready) & proxy_prob.notna()

    prob_live_base = raw_prob.copy()
    if live_oos_proxy_ready:
        prob_live_base = proxy_prob.combine_first(prob_live_base)

    out["prob_live_safe_pre_clip"] = prob_live_base

    odds_1 = pd.to_numeric(out.get("odds_1"), errors="coerce")
    odds_2 = pd.to_numeric(out.get("odds_2"), errors="coerce")
    market_raw, market_devig = compute_market_probs(odds_1, odds_2)
    out["market_implied_p_raw"] = market_raw
    out["market_implied_p_devig"] = market_devig

    p_market = market_devig.combine_first(market_raw)
    out["model_market_gap"] = out["prob_live_safe_pre_clip"] - p_market

    gap_flag = (
        odds_1.ge(float(UNDERDOG_ODDS_GUARD_MIN))
        & out["prob_live_safe_pre_clip"].ge(float(UNDERDOG_PROB_GUARD_MIN))
        & out["model_market_gap"].ge(float(GAP_GUARD_MIN))
    )
    gap_flag = gap_flag.fillna(False)
    out["model_market_gap_flag"] = gap_flag
    out["live_underdog_upscale_guard_triggered"] = gap_flag

    prob_guarded = pd.to_numeric(out["prob_live_safe_pre_clip"], errors="coerce").copy()
    prob_guarded = prob_guarded.where(~gap_flag, np.minimum(prob_guarded, float(UNDERDOG_CAP)))

    blend_weight = np.exp(-np.abs(pd.to_numeric(out["model_market_gap"], errors="coerce")) / float(TAU_GAP))
    prob_blended = prob_guarded.copy()
    if USE_BLEND_ALWAYS:
        valid_market = p_market.notna() & prob_guarded.notna()
        prob_blended.loc[valid_market] = (
            blend_weight.loc[valid_market] * prob_guarded.loc[valid_market]
            + (1.0 - blend_weight.loc[valid_market]) * p_market.loc[valid_market]
        )

    alpha = np.where(pd.to_numeric(prob_blended, errors="coerce") > float(BLEND_PROB_START), float(BASE_SHRINK), 1.0)
    alpha = np.where(gap_flag, np.minimum(alpha, float(FLAGGED_SHRINK)), alpha)
    prob_used_raw = 0.5 + alpha * (pd.to_numeric(prob_blended, errors="coerce") - 0.5)

    out["prob_base"] = pd.to_numeric(prob_guarded, errors="coerce").clip(lower=float(P_MIN), upper=float(P_MAX))
    out["prob_used"] = pd.to_numeric(prob_used_raw, errors="coerce").clip(lower=float(P_MIN), upper=float(P_MAX))
    out["live_shrink_triggered"] = (out["prob_used"] + 1e-12) < out["prob_base"]
    out["blocked_by"] = np.where(gap_flag & STRICT_BLOCK_POLICY, "MODEL_MARKET_GAP", "PASS")

    return out
