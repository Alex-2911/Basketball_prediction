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
    work = df_played.loc[valid, [prob_col, target_col, date_col]].copy()
    if work.empty:
        return out

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values([date_col, prob_col])
    if len(work) < min_train:
        return out

    date_to_indices = (
        work.reset_index()
        .groupby(date_col)["index"]
        .apply(list)
        .to_dict()
    )
    unique_dates = sorted(date_to_indices.keys())

    pending_dates: list[pd.Timestamp] = []
    pending_idx: list[int] = []

    def _fit_predict(val_dates: list[pd.Timestamp], val_idx: list[int]) -> None:
        if not val_idx:
            return
        first_val_date = min(val_dates)
        train_dates = [d for d in unique_dates if d < first_val_date]
        if not train_dates:
            return
        train_idx = [idx for d in train_dates for idx in date_to_indices[d]]
        if len(train_idx) < min_train:
            return
        train_y = df_played.loc[train_idx, target_col].astype(int)
        if train_y.nunique() < 2:
            return

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(
            df_played.loc[train_idx, prob_col].astype(float),
            train_y,
        )
        out.loc[val_idx] = iso.transform(df_played.loc[val_idx, prob_col].astype(float))

    for dt in unique_dates:
        day_idx = date_to_indices[dt]
        pending_dates.append(dt)
        pending_idx.extend(day_idx)

        if len(pending_idx) >= min_step:
            _fit_predict(pending_dates, pending_idx)
            pending_dates = []
            pending_idx = []

    _fit_predict(pending_dates, pending_idx)
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
