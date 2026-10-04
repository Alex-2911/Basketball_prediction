from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
class LiveOOSProxy:
    ready: bool
    train_rows: int
    global_win_rate: float
    bin_edges: np.ndarray
    bin_n: np.ndarray
    bin_win_rate: np.ndarray
    source_col_used: str
    min_bin_n: int
    recent_window_used: int | None
    fallback_used: bool = False
    fallback_reason: str = ""

    def predict_proxy(self, p_in: pd.Series | np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = pd.to_numeric(pd.Series(p_in), errors="coerce").to_numpy(dtype=float)
        out_prob = np.full(values.shape, np.nan, dtype=float)
        out_n = np.zeros(values.shape, dtype=int)
        out_rate = np.full(values.shape, np.nan, dtype=float)
        out_bin = np.full(values.shape, -1, dtype=int)

        if not self.ready or len(self.bin_edges) < 2 or len(self.bin_win_rate) == 0:
            return out_prob, out_n, out_rate, out_bin

        idxs = np.searchsorted(self.bin_edges, values, side="right") - 1
        idxs = np.clip(idxs, 0, len(self.bin_win_rate) - 1)
        valid = np.isfinite(values)
        out_prob[valid] = self.bin_win_rate[idxs[valid]]
        out_n[valid] = self.bin_n[idxs[valid]]
        out_rate[valid] = self.bin_win_rate[idxs[valid]]
        out_bin[valid] = idxs[valid]

        weak = valid & (out_n < int(self.min_bin_n))
        out_prob[weak] = self.global_win_rate
        out_rate[weak] = self.global_win_rate
        return out_prob, out_n, out_rate, out_bin


def build_live_oos_proxy(
    df_played: pd.DataFrame,
    prob_source_cols: list[str] | None = None,
    target_col: str = "win",
    n_bins: int = 25,
    min_train_rows: int = 300,
    min_bin_n: int = 25,
    use_wilson_lb: bool = True,
    smoothing_alpha: float = 1.0,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    recent_n: int | None = None,
) -> LiveOOSProxy:
    prob_source_cols = prob_source_cols or ["prob_iso_oos_time", "home_team_prob"]
    required = [target_col, date_col, home_col, away_col]
    if not all(col in df_played.columns for col in required):
        return LiveOOSProxy(False, 0, float("nan"), np.array([]), np.array([]), np.array([]), "", min_bin_n, recent_n, False, "missing_required_columns")

    chosen_col = ""
    work = pd.DataFrame()
    fallback_used = False
    fallback_reason = ""
    for col in prob_source_cols:
        if col not in df_played.columns:
            continue
        candidate = df_played.loc[
            df_played[col].notna() & df_played[target_col].notna(),
            [date_col, home_col, away_col, target_col, col],
        ].copy()
        if candidate.empty:
            continue
        candidate = _dedupe_games(candidate, date_col=date_col, home_col=home_col, away_col=away_col)
        if recent_n is not None:
            candidate = candidate.tail(int(recent_n)).copy()
        if len(candidate) >= int(min_train_rows) and pd.to_numeric(candidate[target_col], errors="coerce").astype(int).nunique() >= 2:
            chosen_col = col
            work = candidate
            if col != prob_source_cols[0]:
                fallback_used = True
                fallback_reason = f"fallback_to_{col}"
            break

    if work.empty or not chosen_col:
        return LiveOOSProxy(False, len(work), float("nan"), np.array([]), np.array([]), np.array([]), chosen_col, min_bin_n, recent_n, True, "no_valid_source")

    x = pd.to_numeric(work[chosen_col], errors="coerce")
    y = pd.to_numeric(work[target_col], errors="coerce").astype(int)
    valid = x.notna() & y.notna()
    work = work.loc[valid].copy()
    x = x.loc[valid]
    y = y.loc[valid]
    train_rows = int(len(work))
    global_win_rate = float(y.mean()) if train_rows else float("nan")
    if train_rows < int(min_train_rows) or y.nunique() < 2:
        return LiveOOSProxy(False, train_rows, global_win_rate, np.array([]), np.array([]), np.array([]), chosen_col, min_bin_n, recent_n, fallback_used, "insufficient_rows_or_classes")

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

    if len(bin_n) == max(0, len(edges) - 1):
        bin_n_arr = bin_n.to_numpy(dtype=int)
        bin_rate_arr = bin_rate.to_numpy(dtype=float)
    else:
        bin_n_arr = np.zeros(max(0, len(edges) - 1), dtype=int)
        bin_rate_arr = np.full(max(0, len(edges) - 1), global_win_rate, dtype=float)
        for idx in range(min(len(bin_n_arr), len(bin_n))):
            bin_n_arr[idx] = int(bin_n.iloc[idx])
            bin_rate_arr[idx] = float(bin_rate.iloc[idx])

    return LiveOOSProxy(
        ready=True,
        train_rows=train_rows,
        global_win_rate=global_win_rate,
        bin_edges=edges,
        bin_n=bin_n_arr,
        bin_win_rate=bin_rate_arr,
        source_col_used=chosen_col,
        min_bin_n=min_bin_n,
        recent_window_used=recent_n,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def apply_live_oos_proxy(
    df_upcoming: pd.DataFrame,
    proxy_obj: LiveOOSProxy,
    in_col: str = "home_team_prob",
    blend_col: str | None = None,
    blend_alpha: float = 0.80,
    small_bin_alpha: float = 0.95,
    min_bin_n_for_full_proxy: int = 200,
) -> pd.DataFrame:
    out = df_upcoming.copy()
    out["prob_live_oos_proxy"] = np.nan
    out["prob_live_oos_proxy_raw"] = np.nan
    out["live_oos_proxy_ready"] = bool(proxy_obj.ready)
    out["live_oos_proxy_used"] = False
    out["live_oos_proxy_train_rows"] = int(proxy_obj.train_rows)
    out["live_oos_proxy_bin_n"] = 0
    out["live_oos_proxy_bin_id"] = -1
    out["live_oos_proxy_bin_winrate"] = np.nan
    out["live_oos_proxy_source_col_used"] = proxy_obj.source_col_used
    out["live_oos_proxy_fallback_used"] = bool(proxy_obj.fallback_used)
    out["live_oos_proxy_fallback_reason"] = proxy_obj.fallback_reason
    out["live_oos_proxy_blend_alpha"] = np.nan
    out["live_oos_proxy_blend_input"] = np.nan
    out["live_oos_proxy_reason"] = ""
    if not proxy_obj.ready or in_col not in out.columns:
        if in_col in out.columns:
            out["live_oos_proxy_reason"] = "proxy_unavailable"
        return out

    p_proxy, bin_n, bin_rate, bin_id = proxy_obj.predict_proxy(out[in_col])
    out["prob_live_oos_proxy_raw"] = p_proxy
    out["live_oos_proxy_bin_n"] = bin_n
    out["live_oos_proxy_bin_id"] = bin_id
    out["live_oos_proxy_bin_winrate"] = bin_rate
    base_col = blend_col if blend_col is not None and blend_col in out.columns else in_col
    base_values = pd.to_numeric(out[base_col], errors="coerce")
    alpha = pd.Series(float(blend_alpha), index=out.index, dtype=float)
    alpha.loc[pd.to_numeric(out["live_oos_proxy_bin_n"], errors="coerce") < int(min_bin_n_for_full_proxy)] = float(small_bin_alpha)
    proxy_values = pd.to_numeric(out["prob_live_oos_proxy_raw"], errors="coerce")
    blended = (alpha * base_values) + ((1.0 - alpha) * proxy_values)
    out["live_oos_proxy_blend_alpha"] = alpha
    out["live_oos_proxy_blend_input"] = base_values
    out["prob_live_oos_proxy"] = np.where(proxy_values.notna(), blended, np.nan)
    out["live_oos_proxy_used"] = pd.to_numeric(out["prob_live_oos_proxy"], errors="coerce").notna()
    reasons = np.where(
        alpha >= 0.95,
        "proxy_blend_small_bin",
        "proxy_blend",
    )
    reasons = np.where(out["live_oos_proxy_fallback_used"], out["live_oos_proxy_fallback_reason"].astype(str) + "|" + reasons, reasons)
    out["live_oos_proxy_reason"] = reasons
    return out
