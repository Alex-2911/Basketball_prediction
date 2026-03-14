from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _dedupe_games(df: pd.DataFrame, *, date_col: str, home_col: str, away_col: str) -> pd.DataFrame:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).copy()
    work["_game_key"] = (
        work[date_col].dt.strftime("%Y-%m-%d")
        + "_"
        + work[home_col].astype(str)
        + "_"
        + work[away_col].astype(str)
    )
    work = work.sort_values(date_col).drop_duplicates(subset="_game_key", keep="last").copy()
    return work.drop(columns="_game_key")


def _wilson_lower_bound(wins: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    phat = wins / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    radius = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return float(max(0.0, center - radius))


@dataclass
class LiveOOSProxyResult:
    proxy: pd.Series
    ready: bool
    train_rows: int
    win_rate: float
    bin_n: pd.Series
    bin_win_rate: pd.Series
    source_col_used: str
    recent_window_used: int | None


@dataclass
class LiveOOSProxyModel:
    ready: bool
    train_rows: int
    global_win_rate: float
    bin_edges: np.ndarray
    bin_n: np.ndarray
    bin_win_rate: np.ndarray
    source_col_used: str
    min_bin_n: int
    recent_window_used: int | None
    predictor: callable | None


def compute_time_oos_isotonic(
    df_played: pd.DataFrame,
    *,
    prob_col: str,
    target_col: str,
    date_col: str,
    min_train: int = 50,
    min_step: int = 10,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.Series:
    out = pd.Series(np.nan, index=df_played.index, dtype=float)
    required = [prob_col, target_col, date_col, home_col, away_col]
    valid = df_played[required].notna().all(axis=1)
    work = _dedupe_games(
        df_played.loc[valid, required].copy(),
        date_col=date_col,
        home_col=home_col,
        away_col=away_col,
    )
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


def build_live_oos_proxy(
    df_played: pd.DataFrame,
    *,
    prob_source_cols: list[str] | None = None,
    target_col: str = "win",
    n_bins: int = 25,
    min_train_rows: int = 300,
    min_bin_n: int = 25,
    use_wilson_lb: bool = True,
    smoothing_alpha: float = 1.0,
    date_col: str = "game_date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    recent_n: int | None = None,
) -> LiveOOSProxyModel:
    prob_source_cols = prob_source_cols or ["prob_iso_oos_time", "home_team_prob"]
    required = [target_col, date_col, home_col, away_col]
    if not all(col in df_played.columns for col in required):
        return LiveOOSProxyModel(
            ready=False,
            train_rows=0,
            global_win_rate=float("nan"),
            bin_edges=np.array([]),
            bin_n=np.array([]),
            bin_win_rate=np.array([]),
            source_col_used="",
            min_bin_n=min_bin_n,
            recent_window_used=recent_n,
            predictor=None,
        )

    chosen_col = ""
    work = pd.DataFrame()
    for col in prob_source_cols:
        if col not in df_played.columns:
            continue
        candidate = df_played.loc[
            df_played[col].notna() & df_played[target_col].notna(),
            [date_col, home_col, away_col, target_col, col],
        ].copy()
        if candidate.empty:
            continue
        candidate = _dedupe_games(
            candidate,
            date_col=date_col,
            home_col=home_col,
            away_col=away_col,
        )
        if recent_n is not None:
            candidate = candidate.tail(int(recent_n)).copy()
        if len(candidate) >= int(min_train_rows) and candidate[target_col].astype(int).nunique() >= 2:
            chosen_col = col
            work = candidate
            break

    if work.empty or not chosen_col:
        return LiveOOSProxyModel(
            ready=False,
            train_rows=len(work),
            global_win_rate=float(pd.to_numeric(work.get(target_col), errors="coerce").mean()) if not work.empty else float("nan"),
            bin_edges=np.array([]),
            bin_n=np.array([]),
            bin_win_rate=np.array([]),
            source_col_used=chosen_col,
            min_bin_n=min_bin_n,
            recent_window_used=recent_n,
            predictor=None,
        )

    x = pd.to_numeric(work[chosen_col], errors="coerce")
    y = pd.to_numeric(work[target_col], errors="coerce").astype(int)
    valid = x.notna() & y.notna()
    work = work.loc[valid].copy()
    x = x.loc[valid]
    y = y.loc[valid]
    train_rows = int(len(work))
    global_win_rate = float(y.mean()) if train_rows else float("nan")
    if train_rows < int(min_train_rows) or y.nunique() < 2:
        return LiveOOSProxyModel(
            ready=False,
            train_rows=train_rows,
            global_win_rate=global_win_rate,
            bin_edges=np.array([]),
            bin_n=np.array([]),
            bin_win_rate=np.array([]),
            source_col_used=chosen_col,
            min_bin_n=min_bin_n,
            recent_window_used=recent_n,
            predictor=None,
        )

    try:
        cats, edges = pd.qcut(x, q=min(int(n_bins), train_rows), retbins=True, duplicates="drop")
    except ValueError:
        edges = np.linspace(float(x.min()), float(x.max()), num=min(int(n_bins), 10) + 1)
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([float(x.min()), float(x.max()) + 1e-9])
        cats = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")

    work["_bin"] = cats
    grouped = work.groupby("_bin", observed=True)
    bin_n = grouped[target_col].size().astype(int)
    bin_wins = grouped[target_col].sum().astype(float)
    if smoothing_alpha > 0:
        bin_rate = (bin_wins + smoothing_alpha * global_win_rate) / (bin_n + smoothing_alpha)
    else:
        bin_rate = bin_wins / bin_n
    if use_wilson_lb:
        bin_rate = pd.Series(
            [_wilson_lower_bound(float(w), int(n)) for w, n in zip(bin_wins, bin_n)],
            index=bin_n.index,
            dtype=float,
        )

    bin_rate = bin_rate.astype(float).fillna(global_win_rate)
    edges = np.asarray(edges, dtype=float)
    if len(bin_n) != max(0, len(edges) - 1):
        aligned_n = np.zeros(max(0, len(edges) - 1), dtype=int)
        aligned_r = np.full(max(0, len(edges) - 1), global_win_rate, dtype=float)
        for idx, interval in enumerate(bin_n.index):
            pos = idx
            aligned_n[pos] = int(bin_n.iloc[idx])
            aligned_r[pos] = float(bin_rate.iloc[idx])
        bin_n_arr = aligned_n
        bin_rate_arr = aligned_r
    else:
        bin_n_arr = bin_n.to_numpy(dtype=int)
        bin_rate_arr = bin_rate.to_numpy(dtype=float)

    def predictor(p_values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = pd.to_numeric(pd.Series(p_values), errors="coerce").to_numpy(dtype=float)
        out_prob = np.full(values.shape, np.nan, dtype=float)
        out_n = np.zeros(values.shape, dtype=int)
        out_rate = np.full(values.shape, np.nan, dtype=float)
        if len(edges) < 2:
            return out_prob, out_n, out_rate
        idxs = np.searchsorted(edges, values, side="right") - 1
        idxs = np.clip(idxs, 0, len(bin_rate_arr) - 1)
        valid_values = np.isfinite(values)
        out_prob[valid_values] = bin_rate_arr[idxs[valid_values]]
        out_n[valid_values] = bin_n_arr[idxs[valid_values]]
        out_rate[valid_values] = bin_rate_arr[idxs[valid_values]]
        weak = valid_values & (out_n < int(min_bin_n))
        out_prob[weak] = global_win_rate
        out_rate[weak] = global_win_rate
        return out_prob, out_n, out_rate

    return LiveOOSProxyModel(
        ready=True,
        train_rows=train_rows,
        global_win_rate=global_win_rate,
        bin_edges=edges,
        bin_n=bin_n_arr,
        bin_win_rate=bin_rate_arr,
        source_col_used=chosen_col,
        min_bin_n=min_bin_n,
        recent_window_used=recent_n,
        predictor=predictor,
    )


def apply_live_oos_proxy(
    df_upcoming: pd.DataFrame,
    proxy_obj: LiveOOSProxyModel,
    *,
    in_col: str = "home_team_prob",
) -> pd.DataFrame:
    out = df_upcoming.copy()
    out["prob_live_oos_proxy"] = np.nan
    out["live_oos_proxy_ready"] = bool(proxy_obj.ready)
    out["live_oos_proxy_used"] = False
    out["live_oos_proxy_train_rows"] = int(proxy_obj.train_rows)
    out["live_oos_proxy_bin_n"] = 0
    out["live_oos_proxy_bin_winrate"] = np.nan
    out["live_oos_proxy_recent_window_used"] = proxy_obj.recent_window_used
    if not proxy_obj.ready or proxy_obj.predictor is None or in_col not in out.columns:
        return out

    pred_prob, pred_n, pred_rate = proxy_obj.predictor(out[in_col])
    out["prob_live_oos_proxy"] = pred_prob
    out["live_oos_proxy_bin_n"] = pred_n
    out["live_oos_proxy_bin_winrate"] = pred_rate
    out["live_oos_proxy_used"] = pd.to_numeric(out["prob_live_oos_proxy"], errors="coerce").notna()
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
    date_col: str = "game_date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    n_bins: int = 25,
    min_bin_n: int = 25,
    recent_n: int | None = None,
) -> LiveOOSProxyResult:
    proxy = pd.Series(np.nan, index=df_all.index, dtype=float)
    proxy_bin_n = pd.Series(0, index=df_all.index, dtype=int)
    proxy_bin_rate = pd.Series(np.nan, index=df_all.index, dtype=float)

    proxy_model = build_live_oos_proxy(
        df_all.loc[played_mask].copy(),
        prob_source_cols=[oos_col, prob_col],
        target_col=target_col,
        n_bins=n_bins,
        min_train_rows=min_train_oos,
        min_bin_n=min_bin_n,
        date_col=date_col,
        home_col=home_col,
        away_col=away_col,
        recent_n=recent_n,
    )

    if not proxy_model.ready:
        return LiveOOSProxyResult(
            proxy=proxy,
            ready=False,
            train_rows=proxy_model.train_rows,
            win_rate=proxy_model.global_win_rate,
            bin_n=proxy_bin_n,
            bin_win_rate=proxy_bin_rate,
            source_col_used=proxy_model.source_col_used,
            recent_window_used=proxy_model.recent_window_used,
        )

    upcoming = apply_live_oos_proxy(df_all.loc[upcoming_mask].copy(), proxy_model, in_col=prob_col)
    proxy.loc[upcoming.index] = upcoming["prob_live_oos_proxy"]
    proxy_bin_n.loc[upcoming.index] = upcoming["live_oos_proxy_bin_n"].astype(int)
    proxy_bin_rate.loc[upcoming.index] = upcoming["live_oos_proxy_bin_winrate"].astype(float)

    return LiveOOSProxyResult(
        proxy=proxy,
        ready=True,
        train_rows=proxy_model.train_rows,
        win_rate=proxy_model.global_win_rate,
        bin_n=proxy_bin_n,
        bin_win_rate=proxy_bin_rate,
        source_col_used=proxy_model.source_col_used,
        recent_window_used=proxy_model.recent_window_used,
    )
