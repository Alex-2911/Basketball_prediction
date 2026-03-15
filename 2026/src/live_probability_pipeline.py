from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from live_calibration import compute_time_oos_isotonic
from live_oos_proxy import apply_live_oos_proxy, build_live_oos_proxy
from live_safety import apply_live_safety

DEFAULT_CONFIG: dict[str, Any] = {
    "date_col": "game_date",
    "result_col": "home_team_won",
    "result_raw_col": "result_raw",
    "pred_proba_col": "pred_home_win_proba",
    "prob_iso_oos_time_col": "prob_iso_oos_time",
    "prob_live_oos_proxy_col": "prob_live_oos_proxy",
    "min_train_oos_time": 50,
    "min_step_oos_time": 10,
    "min_train_oos_proxy": 300,
    "proxy_n_bins": 25,
    "proxy_min_bin_n": 25,
    "today_date": None,
    "tomorrow_date": None,
    "compute_oos_chain": True,
}


def _read_strategy_params_txt(path: Path) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if not path.exists():
        return params

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        raw = v.strip()
        try:
            val: Any = float(raw)
        except ValueError:
            val = raw
        params[key] = val
    return params


def _extract_strategy_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params_used") if isinstance(payload, dict) else None
    if not isinstance(params, dict):
        return {}

    out = {
        "home_win_rate_threshold": params.get("home_win_rate_threshold"),
        "odds_min": params.get("odds_min"),
        "odds_max": params.get("odds_max"),
        "prob_threshold": params.get("prob_threshold"),
    }
    if "min_ev" in params:
        out["min_ev"] = params.get("min_ev")
    return {k: v for k, v in out.items() if v is not None}


def load_active_strategy_params(repo_root: Path) -> dict[str, Any]:
    env_override = os.environ.get("STRATEGY_PARAMS_PATH", "").strip()
    if env_override:
        override_path = Path(env_override)
        if override_path.exists() and override_path.suffix.lower() == ".json":
            try:
                payload = json.loads(override_path.read_text(encoding="utf-8"))
                params = _extract_strategy_params(payload)
                if params:
                    return params
            except json.JSONDecodeError:
                pass
        if override_path.exists():
            params = _read_strategy_params_txt(override_path)
            if params:
                return params

    metrics_candidates = [
        repo_root / "2026" / "output" / "LightGBM" / "metrics_snapshot.json",
        repo_root / "2026" / "output" / "LightGBM" / "Kelly" / "metrics_snapshot.json",
    ]
    for path in metrics_candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        params = _extract_strategy_params(payload)
        if params:
            if "min_ev" not in params:
                local = payload.get("local_window_200", {}) if isinstance(payload, dict) else {}
                if isinstance(local, dict) and "min_EV_applied" in local:
                    params["min_ev"] = local.get("min_EV_applied")
            return params

    txt_candidates = [
        repo_root / "2026" / "output" / "LightGBM" / "strategy_params.txt",
        repo_root / "2026" / "output" / "LightGBM" / "Kelly" / "strategy_params.txt",
    ]
    for path in txt_candidates:
        params = _read_strategy_params_txt(path)
        if not params:
            continue
        mapped = {
            "home_win_rate_threshold": params.get("home_win_rate_threshold"),
            "odds_min": params.get("odds_min"),
            "odds_max": params.get("odds_max"),
            "prob_threshold": params.get("prob_threshold"),
        }
        if "min_ev" in params:
            mapped["min_ev"] = params.get("min_ev")
        return {k: v for k, v in mapped.items() if v is not None}

    return {}


def prepare_live_probability_columns(
    df: pd.DataFrame,
    *,
    clip_lo: float,
    clip_hi: float,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    out = df.copy()

    date_col = cfg["date_col"]
    result_col = cfg["result_col"]
    result_raw_col = cfg["result_raw_col"]
    pred_proba_col = cfg["pred_proba_col"]
    prob_iso_oos_time_col = cfg["prob_iso_oos_time_col"]

    out["home_team_prob"] = pd.to_numeric(out.get("home_team_prob", out.get(pred_proba_col)), errors="coerce")
    out["prob_iso"] = pd.to_numeric(out.get("prob_iso", out.get("iso_proba_home_win")), errors="coerce")

    if "odds_1" not in out.columns and "closing_home_odds" in out.columns:
        out["odds_1"] = pd.to_numeric(out["closing_home_odds"], errors="coerce")
    if "odds_2" not in out.columns and "closing_away_odds" in out.columns:
        out["odds_2"] = pd.to_numeric(out["closing_away_odds"], errors="coerce")

    played_mask = out[result_raw_col].notna() & (out[result_raw_col].astype(str) != "0") if result_raw_col in out.columns else out[result_col].notna()

    if cfg["compute_oos_chain"] and date_col in out.columns and result_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out[prob_iso_oos_time_col] = compute_time_oos_isotonic(
            out.loc[played_mask].copy(),
            prob_col=pred_proba_col,
            target_col=result_col,
            date_col=date_col,
            min_train=int(cfg["min_train_oos_time"]),
            min_step=int(cfg["min_step_oos_time"]),
        )

        played_df = out.loc[played_mask].copy()
        played_df["home_team_prob"] = pd.to_numeric(played_df["home_team_prob"], errors="coerce")
        played_df["win"] = pd.to_numeric(played_df[result_col], errors="coerce")
        proxy_obj = build_live_oos_proxy(
            played_df,
            prob_source_cols=[prob_iso_oos_time_col, "home_team_prob"],
            target_col="win",
            n_bins=int(cfg["proxy_n_bins"]),
            min_train_rows=int(cfg["min_train_oos_proxy"]),
            min_bin_n=int(cfg["proxy_min_bin_n"]),
            use_wilson_lb=True,
        )

        out["live_oos_proxy_ready"] = bool(proxy_obj["ready"])
        out["live_oos_proxy_train_rows"] = int(proxy_obj["train_rows"])
        out["live_oos_proxy_bin_n"] = 0
        out["live_oos_proxy_bin_winrate"] = np.nan

        upcoming_mask = (~played_mask)
        today_date = cfg.get("today_date")
        tomorrow_date = cfg.get("tomorrow_date")
        if today_date is not None and tomorrow_date is not None and date_col in out.columns:
            upcoming_mask = upcoming_mask & out[date_col].dt.date.isin([today_date, tomorrow_date])

        upcoming_df = out.loc[upcoming_mask].copy()
        upcoming_df["home_team_prob"] = pd.to_numeric(upcoming_df["home_team_prob"], errors="coerce")
        upcoming_with_proxy = apply_live_oos_proxy(upcoming_df, proxy_obj, in_col="home_team_prob")
        for c in [
            "prob_live_oos_proxy",
            "live_oos_proxy_ready",
            "live_oos_proxy_train_rows",
            "live_oos_proxy_bin_n",
            "live_oos_proxy_bin_winrate",
        ]:
            out.loc[upcoming_with_proxy.index, c] = upcoming_with_proxy[c]
    else:
        out[prob_iso_oos_time_col] = pd.to_numeric(out.get(prob_iso_oos_time_col, np.nan), errors="coerce")
        out["prob_live_oos_proxy"] = pd.to_numeric(out.get("prob_live_oos_proxy", np.nan), errors="coerce")

    proxy_ready_col = out.get("live_oos_proxy_ready", False)
    if isinstance(proxy_ready_col, pd.Series):
        live_oos_proxy_ready = bool(proxy_ready_col.fillna(False).astype(bool).any())
    else:
        live_oos_proxy_ready = bool(proxy_ready_col)

    out = apply_live_safety(out, live_oos_proxy_ready=live_oos_proxy_ready)
    out["prob_live_safe"] = out["prob_base"]
    out["prob_used"] = pd.to_numeric(out["prob_used"], errors="coerce").clip(clip_lo, clip_hi)

    for c in [
        "prob_base",
        "prob_live_safe_pre_clip",
        "prob_live_oos_proxy",
        "market_implied_p_raw",
        "market_implied_p_devig",
        "model_market_gap",
        "model_market_gap_flag",
        "live_underdog_upscale_guard_triggered",
        "live_shrink_triggered",
        "live_oos_proxy_ready",
        "live_oos_proxy_train_rows",
        "live_oos_proxy_bin_n",
        "live_oos_proxy_bin_winrate",
        "blocked_by",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    return out
