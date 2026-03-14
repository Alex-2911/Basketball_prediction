from __future__ import annotations

import numpy as np
import pandas as pd


def implied_prob_decimal(odds):
    odds_num = pd.to_numeric(odds, errors="coerce")
    return np.where(odds_num > 0, 1.0 / odds_num, np.nan)


def devig_two_way(p1_raw, p2_raw):
    p1 = pd.to_numeric(p1_raw, errors="coerce")
    p2 = pd.to_numeric(p2_raw, errors="coerce")
    denom = p1 + p2
    return np.where(denom > 0, p1 / denom, np.nan)


def compute_market_probs(odds_1, odds_2=None):
    p_raw = implied_prob_decimal(odds_1)
    if odds_2 is None:
        return p_raw, np.full_like(np.asarray(p_raw, dtype=float), np.nan, dtype=float)
    p2_raw = implied_prob_decimal(odds_2)
    p_devig = devig_two_way(p_raw, p2_raw)
    return p_raw, p_devig
