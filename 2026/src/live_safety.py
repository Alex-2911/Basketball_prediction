from __future__ import annotations

import numpy as np
import pandas as pd


UNDERDOG_ODDS = 2.30
HIGH_PROB_FOR_UNDERDOG = 0.60
GAP_ABS_THRESHOLD = 0.20
GAP_CAP = 0.15
BASE_SHRINK = 0.85
STRONGER_SHRINK = 0.75
P_MIN = 0.35
P_MAX = 0.80


def apply_live_safety(df: pd.DataFrame, *, live_oos_proxy_ready: bool) -> pd.DataFrame:
    out = df.copy()

    odds = pd.to_numeric(out.get("odds_1"), errors="coerce")
    implied = np.where(odds > 0, 1.0 / odds, np.nan)
    out["implied_prob_1"] = implied

    base = pd.to_numeric(out.get("home_team_prob"), errors="coerce")
    if "prob_live_oos_proxy" in out.columns and live_oos_proxy_ready:
        base = pd.to_numeric(out["prob_live_oos_proxy"], errors="coerce").combine_first(base)
    if "prob_iso_oos_time" in out.columns:
        base = pd.to_numeric(out["prob_iso_oos_time"], errors="coerce").combine_first(base)

    out["prob_live_base"] = base

    gap = base - out["implied_prob_1"]
    out["model_market_gap"] = gap

    underdog_trigger = (odds >= UNDERDOG_ODDS) & (base >= HIGH_PROB_FOR_UNDERDOG)
    gap_trigger = gap.abs() >= GAP_ABS_THRESHOLD
    flag = (underdog_trigger | gap_trigger).fillna(False)
    out["model_market_gap_flag"] = flag
    out["live_underdog_upscale_guard_triggered"] = underdog_trigger.fillna(False)

    cap_target = out["implied_prob_1"] + GAP_CAP
    guarded_base = np.where(flag, np.minimum(base, cap_target), base)
    out["prob_live_base"] = guarded_base

    shrink = np.where(flag, STRONGER_SHRINK, BASE_SHRINK)
    out["live_shrink_triggered"] = flag

    out["prob_live_safe_pre_clip"] = 0.5 + shrink * (pd.to_numeric(out["prob_live_base"], errors="coerce") - 0.5)
    out["prob_base"] = out["prob_live_safe_pre_clip"]
    out["prob_used"] = out["prob_base"].clip(lower=P_MIN, upper=P_MAX)

    return out
