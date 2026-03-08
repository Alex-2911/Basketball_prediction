from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class LiveOOSProxyResult:
    proxy: pd.Series
    ready: bool
    train_rows: int
    win_rate: float


def compute_time_oos_isotonic(
    df_played: pd.DataFrame,
    *,
    prob_col: str,
    target_col: str,
    date_col: str,
    min_train: int = 50,
    min_step: int = 10,
) -> pd.Series:
    out = pd.Series(np.nan, index=df_played.index, dtype=float)
    valid = df_played[prob_col].notna() & df_played[target_col].notna() & df_played[date_col].notna()
    work = df_played.loc[valid, [prob_col, target_col, date_col]].sort_values(date_col)
    if len(work) < min_train:
        return out

    indices = work.index.to_list()
    for start in range(min_train, len(indices), min_step):
        train_idx = indices[:start]
        val_idx = indices[start : start + min_step]
        if not val_idx:
            break

        train_y = df_played.loc[train_idx, target_col].astype(int)
        if train_y.nunique() < 2:
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(
            df_played.loc[train_idx, prob_col].astype(float),
            train_y,
        )
        out.loc[val_idx] = iso.transform(df_played.loc[val_idx, prob_col].astype(float))

    return out


def compute_live_oos_proxy(
    df_all: pd.DataFrame,
    *,
    played_mask: pd.Series,
    upcoming_mask: pd.Series,
    prob_col: str,
    target_col: str,
    oos_col: str,
    min_train_oos: int = 200,
) -> LiveOOSProxyResult:
    proxy = pd.Series(np.nan, index=df_all.index, dtype=float)

    train_mask = played_mask & df_all[oos_col].notna() & df_all[prob_col].notna() & df_all[target_col].notna()
    train_rows = int(train_mask.sum())
    ready = bool(train_rows >= min_train_oos and df_all.loc[train_mask, target_col].nunique() >= 2)
    win_rate = float(df_all.loc[train_mask, target_col].mean()) if train_rows else float("nan")

    if not ready:
        return LiveOOSProxyResult(proxy=proxy, ready=False, train_rows=train_rows, win_rate=win_rate)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(
        df_all.loc[train_mask, prob_col].astype(float),
        df_all.loc[train_mask, target_col].astype(int),
    )

    pred_mask = upcoming_mask & df_all[prob_col].notna()
    if pred_mask.any():
        proxy.loc[pred_mask] = iso.transform(df_all.loc[pred_mask, prob_col].astype(float))

    return LiveOOSProxyResult(proxy=proxy, ready=True, train_rows=train_rows, win_rate=win_rate)
