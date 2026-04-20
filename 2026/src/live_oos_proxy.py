from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROB_SOURCE_COLS = ["prob_iso_oos_time", "home_team_prob"]




def _proxy_get(proxy_obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(proxy_obj, dict):
        return proxy_obj.get(key, default)
    if is_dataclass(proxy_obj):
        return asdict(proxy_obj).get(key, default)
    if hasattr(proxy_obj, key):
        return getattr(proxy_obj, key)
    getter = getattr(proxy_obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return default


def _proxy_has(proxy_obj: Any, key: str) -> bool:
    sentinel = object()
    return _proxy_get(proxy_obj, key, sentinel) is not sentinel

def _wilson_lower_bound(wins: float, n: float, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    p_hat = wins / n
    denom = 1.0 + (z**2) / n
    center = p_hat + (z**2) / (2.0 * n)
    margin = z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2)))
    return float((center - margin) / denom)


def _select_source_col(
    df_played: pd.DataFrame,
    prob_source_cols: list[str],
    target_col: str,
    min_train_rows: int,
) -> tuple[str, pd.Series]:
    for col in prob_source_cols:
        if col not in df_played.columns:
            continue
        mask = df_played[col].notna() & df_played[target_col].notna()
        if int(mask.sum()) < min_train_rows:
            continue
        if df_played.loc[mask, target_col].nunique() < 2:
            continue
        return col, mask

    fallback = "home_team_prob" if "home_team_prob" in df_played.columns else prob_source_cols[0]
    mask = df_played.get(fallback, pd.Series(index=df_played.index, dtype=float)).notna() & df_played[target_col].notna()
    return fallback, mask


def _make_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    q_edges = np.unique(np.quantile(values, q=np.linspace(0, 1, n_bins + 1)))
    if q_edges.size >= 3:
        q_edges[0] = min(q_edges[0], 0.0)
        q_edges[-1] = max(q_edges[-1], 1.0)
        return q_edges

    return np.linspace(0.0, 1.0, n_bins + 1, dtype=float)


def build_live_oos_proxy(
    df_played: pd.DataFrame,
    prob_source_cols: list[str] = DEFAULT_PROB_SOURCE_COLS,
    target_col: str = "win",
    n_bins: int = 25,
    min_train_rows: int = 300,
    min_bin_n: int = 25,
    use_wilson_lb: bool = True,
    smoothing_alpha: float = 1.0,
) -> dict[str, Any]:
    if target_col not in df_played.columns:
        raise KeyError(f"Missing target_col={target_col}")

    source_col, train_mask = _select_source_col(df_played, prob_source_cols, target_col, min_train_rows)

    work = df_played.loc[train_mask, [source_col, target_col]].copy()
    work[source_col] = pd.to_numeric(work[source_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[source_col, target_col])

    train_rows = int(len(work))
    global_wr = float(work[target_col].mean()) if train_rows else float("nan")
    ready = bool(train_rows >= min_train_rows and work[target_col].nunique() >= 2)

    edges = _make_edges(work[source_col], n_bins)
    bin_n = np.zeros(len(edges) - 1, dtype=int)
    bin_wr = np.full(len(edges) - 1, np.nan, dtype=float)

    if ready:
        clipped = np.clip(work[source_col].to_numpy(dtype=float), edges[0], edges[-1])
        bin_idx = np.clip(np.digitize(clipped, bins=edges[1:-1], right=False), 0, len(edges) - 2)
        y = work[target_col].to_numpy(dtype=float)

        for i in range(len(edges) - 1):
            m = bin_idx == i
            n_i = int(m.sum())
            bin_n[i] = n_i
            if n_i == 0:
                continue
            wins = float(y[m].sum())
            if use_wilson_lb:
                score = _wilson_lower_bound(wins, n_i)
            else:
                score = (wins + smoothing_alpha) / (n_i + 2.0 * smoothing_alpha)
            bin_wr[i] = float(score)

        low_n = bin_n < int(min_bin_n)
        if np.any(low_n):
            bin_wr[low_n] = global_wr

        bin_wr = np.where(np.isnan(bin_wr), global_wr, bin_wr)

    def predict_proxy(p: float) -> tuple[float, int, float]:
        if not ready or p is None or np.isnan(p):
            return float("nan"), 0, float("nan")
        p_clipped = float(np.clip(p, edges[0], edges[-1]))
        idx = int(np.clip(np.digitize([p_clipped], bins=edges[1:-1], right=False)[0], 0, len(edges) - 2))
        return float(bin_wr[idx]), int(bin_n[idx]), float(bin_wr[idx])

    return {
        "ready": ready,
        "train_rows": train_rows,
        "global_win_rate": global_wr,
        "bin_edges": edges,
        "bin_n": bin_n,
        "bin_win_rate": bin_wr,
        "source_col_used": source_col,
        "predict_proxy": predict_proxy,
    }


def apply_live_oos_proxy(
    df_upcoming: pd.DataFrame,
    proxy_obj: Any,
    in_col: str = "home_team_prob",
) -> pd.DataFrame:
    out = df_upcoming.copy()
    ready = bool(_proxy_get(proxy_obj, "ready", False))
    train_rows = int(_proxy_get(proxy_obj, "train_rows", 0) or 0)

    out["prob_live_oos_proxy"] = np.nan
    out["live_oos_proxy_ready"] = ready
    out["live_oos_proxy_train_rows"] = train_rows
    out["live_oos_proxy_bin_n"] = 0
    out["live_oos_proxy_bin_winrate"] = np.nan

    predict_fn = _proxy_get(proxy_obj, "predict_proxy")
    if not ready or not callable(predict_fn) or not _proxy_has(proxy_obj, "predict_proxy"):
        return out

    p_in = pd.to_numeric(out.get(in_col), errors="coerce")
    preds = p_in.apply(predict_fn)
    out["prob_live_oos_proxy"] = preds.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    out["live_oos_proxy_bin_n"] = preds.apply(lambda x: x[1] if isinstance(x, tuple) else 0).astype(int)
    out["live_oos_proxy_bin_winrate"] = preds.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
    return out
