from __future__ import annotations

import numpy as np
import pandas as pd


def implied_prob_decimal(odds: pd.Series | np.ndarray | float) -> pd.Series:
    values = pd.to_numeric(pd.Series(odds), errors="coerce")
    return pd.Series(np.where(values > 0, 1.0 / values, np.nan), index=values.index, dtype=float)


def devig_two_way(
    p1_raw: pd.Series | np.ndarray | float,
    p2_raw: pd.Series | np.ndarray | float,
) -> pd.Series:
    p1 = pd.to_numeric(pd.Series(p1_raw), errors="coerce")
    p2 = pd.to_numeric(pd.Series(p2_raw), errors="coerce")
    denom = p1 + p2
    return pd.Series(np.where(denom > 0, p1 / denom, np.nan), index=p1.index, dtype=float)


def compute_market_probs(
    odds_1: pd.Series | np.ndarray | float,
    odds_2: pd.Series | np.ndarray | float | None = None,
) -> tuple[pd.Series, pd.Series]:
    p_raw = implied_prob_decimal(odds_1)
    if odds_2 is None:
        return p_raw, pd.Series(np.nan, index=p_raw.index, dtype=float)

    p2_raw = implied_prob_decimal(odds_2)
    p_devig = devig_two_way(p_raw, p2_raw)
    return p_raw, p_devig
