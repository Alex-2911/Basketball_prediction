from __future__ import annotations

import numpy as np
import pandas as pd

from odds_utils import compute_market_probs

P_MIN = 0.35
P_MAX = 0.80
UNDERDOG_ODDS_GUARD_MIN = 2.00
UNDERDOG_PROB_GUARD_MIN = 0.60
GAP_GUARD_MIN = 0.12
UNDERDOG_CAP = 0.55
TAU_GAP = 0.08
USE_BLEND_ALWAYS = True


def apply_live_safety(df: pd.DataFrame, *, live_oos_proxy_ready: bool) -> pd.DataFrame:
    out = df.copy()

    out["home_team_prob"] = pd.to_numeric(out.get("home_team_prob"), errors="coerce")
    out["prob_iso"] = pd.to_numeric(out.get("prob_iso", out.get("iso_proba_home_win")), errors="coerce")
    out["prob_iso_oos_time"] = pd.to_numeric(out.get("prob_iso_oos_time"), errors="coerce")
    out["prob_live_oos_proxy"] = pd.to_numeric(out.get("prob_live_oos_proxy"), errors="coerce")
    odds_1 = pd.to_numeric(out.get("odds_1"), errors="coerce")
    odds_2 = pd.to_numeric(out.get("odds_2"), errors="coerce")

    p_raw, p_devig = compute_market_probs(odds_1, odds_2)
    out["market_implied_p_raw"] = p_raw
    out["market_implied_p_devig"] = p_devig

    p_market = pd.Series(p_devig, index=out.index, dtype=float)
    p_market = p_market.where(p_market.notna(), pd.Series(p_raw, index=out.index, dtype=float))

    base = out["home_team_prob"].copy()
    if live_oos_proxy_ready:
        base = out["prob_live_oos_proxy"].combine_first(base)

    out["prob_base"] = base
    out["prob_live_safe_pre_clip"] = pd.to_numeric(base, errors="coerce").clip(P_MIN, P_MAX)

    out["model_market_gap"] = out["prob_live_safe_pre_clip"] - p_market
    underdog_guard = (
        (odds_1 >= UNDERDOG_ODDS_GUARD_MIN)
        & (out["prob_live_safe_pre_clip"] >= UNDERDOG_PROB_GUARD_MIN)
        & (out["model_market_gap"] >= GAP_GUARD_MIN)
    ).fillna(False)

    out["model_market_gap_flag"] = (out["model_market_gap"] >= GAP_GUARD_MIN).fillna(False)
    out["live_underdog_upscale_guard_triggered"] = underdog_guard

    prob_guarded = out["prob_live_safe_pre_clip"].where(~underdog_guard, np.minimum(out["prob_live_safe_pre_clip"], UNDERDOG_CAP))

    if USE_BLEND_ALWAYS:
        blend_weight = np.exp(-np.abs(out["model_market_gap"]) / TAU_GAP)
    else:
        blend_weight = np.ones(len(out), dtype=float)
    out["prob_blended"] = blend_weight * prob_guarded + (1.0 - blend_weight) * p_market

    alpha = np.where(out["prob_blended"] > 0.60, 0.85, 1.0)
    alpha = np.where(out["model_market_gap_flag"].astype(bool), np.minimum(alpha, 0.70), alpha)

    out["live_shrink_triggered"] = (alpha < 1.0)
    out["prob_used"] = (0.5 + alpha * (out["prob_blended"] - 0.5)).clip(P_MIN, P_MAX)

    out["blocked_by"] = np.where(underdog_guard, "MODEL_MARKET_GAP", "PASS")
    out["implied_prob_1"] = out["market_implied_p_raw"]

    return out
