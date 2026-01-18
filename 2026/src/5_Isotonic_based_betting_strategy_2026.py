#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_Isotonic_based_betting_strategy_2026.py  (DROP-IN, FIXED)

Key fixes vs your current version:
1) metrics_snapshot "as_of_date" is derived from the latest *played* game in df_all
   (not from df_past which could be artificially capped).
2) "window/bankroll window dates" are based on last N played games (df_past sorted),
   not on matched subset ordering.
3) resolve_params_source checks BOTH:
      public/data/strategy_params.json
      web/public/data/strategy_params.json

Outputs (same intent as before):
- pred_dir/home_win_rates_sorted_YYYY-MM-DD.csv
- pred_dir/Kelly/nba_grid_search_results_YYYY-MM-DD.csv
- pred_dir/Kelly/combined_nba_predictions_iso_YYYY-MM-DD.csv
- pred_dir/bet_shortlist_YYYY-MM-DD.csv
- out_dir/local_matched_games_YYYY-MM-DD.csv
- out_dir/metrics_snapshot.json
- out_dir/strategy_params.txt
- out_dir/summary.json
- (repo_root)/public/data/last_run.json  (for trace)

Additional output (ONLY ADDITION, minimal change):
- (repo_root)/web/public/data/local_matched_games_latest.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from nba_utils_2026 import (
    get_current_date,
    get_directory_paths,
    normalize_team_code,
)

# -----------------------------
# CONSTANTS / COLUMN NAMES
# -----------------------------

DATE_COL = "game_date"
RESULT_COL = "home_team_won"
RESULT_RAW_COL = "result_raw"
PRED_PROBA_COL = "pred_home_win_proba"
HOME_ODDS_COL = "closing_home_odds"
AWAY_ODDS_COL = "closing_away_odds"
HOMEWR_COL = "home_win_rate"
ISO_COL = "iso_proba_home_win"

# Grid search (historical / documentation)
FLAT_STAKE = 100.0
ODDS_MIN_GRID = [1.10, 1.25, 1.40, 1.60]
ODDS_MAX_GRID = [2.00, 2.10, 2.50, 3.00]
PROB_MIN_GRID = [0.55, 0.60, 0.65, 0.70]
HOMEWR_MIN_GRID = [0.50, 0.55, 0.60, 0.65]

# Dashboard / shortlist logic
MIN_EV_DEFAULT = -5.0
PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80
LOCAL_SEARCH_N = 150
FAIR_COMPARE_N = 200
START_BANKROLL = 1000.0

STRATEGY_VARIANTS = {"acc", "iso"}


@dataclass
class StrategyParams:
    min_home_win_rate: float
    min_odds: float
    max_odds: float
    min_iso_proba: float


# -----------------------------
# LOGGING
# -----------------------------

def setup_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s] INFO: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


# -----------------------------
# HELPERS
# -----------------------------

def to_float_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace("[^0-9.]", "", regex=True)
         .replace("", np.nan)
         .astype(float)
    )


def _extract_date_from_filename(filename: str, prefix: str) -> Optional[str]:
    if not filename.startswith(prefix) or not filename.endswith(".csv"):
        return None
    date_str = filename[len(prefix):-4]
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    return date_str


def resolve_dated_file(
    pred_dirs: list[str],
    prefix: str,
    target_ymd: str,
    *,
    latest_on_or_before: Optional[datetime] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    # exact date first
    for d in pred_dirs:
        p = Path(d) / f"{prefix}{target_ymd}.csv"
        if p.exists():
            return p, target_ymd

    # else latest within constraint
    candidates: list[tuple[datetime, Path, str]] = []
    for d in pred_dirs:
        for p in Path(d).glob(f"{prefix}*.csv"):
            ds = _extract_date_from_filename(p.name, prefix)
            if not ds:
                continue
            dt = datetime.strptime(ds, "%Y-%m-%d")
            if latest_on_or_before and dt > latest_on_or_before:
                continue
            candidates.append((dt, p, ds))

    if not candidates:
        return None, None

    dt, p, ds = max(candidates, key=lambda x: x[0])
    logging.info("Falling back to latest %s (%s).", prefix.rstrip("_"), dt.date())
    return p, ds


def resolve_strategy_variant() -> str:
    raw = os.environ.get("STRATEGY_VARIANT", "").strip().lower()
    if not raw:
        raw = "iso" if os.environ.get("CI") else "acc"
    if raw not in STRATEGY_VARIANTS:
        logging.warning("Unknown STRATEGY_VARIANT=%s; defaulting to acc.", raw)
        return "acc"
    return raw


def resolve_strategy_variant_label(variant: str) -> str:
    return "ISO/KELLY" if variant == "iso" else "ACC"


def resolve_params_source(output_dir: Path, variant: str) -> Path:
    # explicit override
    env_path = os.environ.get("STRATEGY_PARAMS_PATH", "").strip()
    if env_path:
        return Path(env_path)

    # dashboard json candidates (both layouts)
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "public" / "data" / "strategy_params.json",
        repo_root / "web" / "public" / "data" / "strategy_params.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback to txt in output_dir
    return output_dir / "strategy_params.txt"


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _validate_params(params: dict, required=None, name="params"):
    required = required or ["home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold"]
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"{name} missing keys: {missing}. Got: {list(params.keys())}")


def _params_to_dict(params: StrategyParams) -> dict:
    return {
        "home_win_rate_threshold": float(params.min_home_win_rate),
        "odds_min": float(params.min_odds),
        "odds_max": float(params.max_odds),
        "prob_threshold": float(params.min_iso_proba),
    }


def _compute_prob_used(df: pd.DataFrame, lo: float, hi: float, src=ISO_COL, dst="prob_used") -> pd.DataFrame:
    out = df.copy()
    out[dst] = pd.to_numeric(out[src], errors="coerce").clip(lower=lo, upper=hi)
    return out


def _compute_ev_per_100(df: pd.DataFrame, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100") -> pd.DataFrame:
    out = df.copy()
    out[dst] = (out[prob_col] * (out[odds_col] - 1.0) - (1.0 - out[prob_col])) * 100.0
    return out


def _make_game_key(df: pd.DataFrame, date_col: str, home_col="home_team", away_col="away_team", dst="game_key") -> pd.DataFrame:
    out = _ensure_datetime(df, date_col)
    out[dst] = (
        out[date_col].dt.strftime("%Y-%m-%d") + "_" +
        out[home_col].astype(str) + "_" +
        out[away_col].astype(str)
    )
    return out


def resolve_output_dir(base_dir: str, prediction_dir: str) -> Path:
    # preferred explicit env
    lgbm_dir = os.environ.get("LGBM_DIR", "").strip()
    if lgbm_dir:
        p = Path(lgbm_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # in CI you sometimes pass SOURCE_ROOT=/.../2026
    source_root = os.environ.get("SOURCE_ROOT", "").strip()
    if source_root:
        sp = Path(source_root)
        if sp.exists():
            out = sp / "output" / "LightGBM"
            out.mkdir(parents=True, exist_ok=True)
            return out

    base = Path(base_dir)
    candidates = [
        base / "2026" / "output" / "LightGBM",
        base / "2026" / "LightGBM",
        Path(prediction_dir),
    ]
    for c in candidates:
        if c.exists():
            c.mkdir(parents=True, exist_ok=True)
            return c

    # last resort
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]


# -----------------------------
# IO / LOADING
# -----------------------------

def load_combined_df(pred_dir: str, ymd: str) -> pd.DataFrame:
    path = Path(pred_dir) / f"combined_nba_predictions_acc_{ymd}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Combined predictions file not found: {path}")

    logging.info("Loading combined predictions: %s", path)

    try:
        df = pd.read_csv(path, encoding="utf-7")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # keep raw result
    if "result" in df.columns:
        df[RESULT_RAW_COL] = df["result"]
    else:
        df[RESULT_RAW_COL] = np.nan

    # date
    if DATE_COL not in df.columns:
        if "date" in df.columns:
            df[DATE_COL] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df[DATE_COL] = pd.NaT
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # proba
    if PRED_PROBA_COL not in df.columns:
        if "home_team_prob" in df.columns:
            df[PRED_PROBA_COL] = to_float_series(df["home_team_prob"])
        else:
            df[PRED_PROBA_COL] = np.nan

    # odds
    if HOME_ODDS_COL not in df.columns:
        if "odds_1" in df.columns:
            df[HOME_ODDS_COL] = to_float_series(df["odds_1"])
        else:
            df[HOME_ODDS_COL] = np.nan

    if AWAY_ODDS_COL not in df.columns:
        if "odds_2" in df.columns:
            df[AWAY_ODDS_COL] = to_float_series(df["odds_2"])
        else:
            df[AWAY_ODDS_COL] = np.nan

    # result boolean (ONLY for played games)
    if RESULT_COL not in df.columns:
        df[RESULT_COL] = np.nan
        if "home_team" in df.columns and "result" in df.columns:
            mask = df["result"].notna() & (df["result"].astype(str) != "0")
            df.loc[mask, RESULT_COL] = (
                df.loc[mask, "result"].astype(str) == df.loc[mask, "home_team"].astype(str)
            ).astype(int)

    return df


def merge_today_predictions(df_all: pd.DataFrame, today_pred_path: Optional[Path], today_date) -> pd.DataFrame:
    if not today_pred_path or not today_pred_path.exists():
        logging.info("No nba_games_predict file found – skipping merge of upcoming games.")
        return df_all

    logging.info("Merging upcoming games from %s", today_pred_path)

    try:
        tmp = pd.read_csv(today_pred_path, encoding="utf-7")
    except Exception:
        tmp = pd.read_csv(today_pred_path, encoding="utf-8")

    tmp.columns = (
        tmp.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # numeric cleanup
    if "home_team_prob" in tmp.columns:
        tmp["home_team_prob"] = to_float_series(tmp["home_team_prob"])
    if "odds_1" in tmp.columns:
        tmp["odds_1"] = to_float_series(tmp["odds_1"])
    if "odds_2" in tmp.columns:
        tmp["odds_2"] = to_float_series(tmp["odds_2"])

    # date cleanup
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    else:
        tmp["date"] = pd.NaT
    tmp.loc[tmp["date"].isna(), "date"] = pd.Timestamp(today_date)

    if "result" not in tmp.columns:
        tmp["result"] = np.nan

    tmp[DATE_COL] = pd.to_datetime(tmp["date"], errors="coerce")
    if PRED_PROBA_COL not in tmp.columns and "home_team_prob" in tmp.columns:
        tmp[PRED_PROBA_COL] = tmp["home_team_prob"]
    if HOME_ODDS_COL not in tmp.columns and "odds_1" in tmp.columns:
        tmp[HOME_ODDS_COL] = tmp["odds_1"]
    if AWAY_ODDS_COL not in tmp.columns and "odds_2" in tmp.columns:
        tmp[AWAY_ODDS_COL] = tmp["odds_2"]

    for col in ["home_team", "away_team"]:
        if col not in df_all.columns:
            df_all[col] = np.nan
        if col not in tmp.columns:
            tmp[col] = np.nan

    key_cols = [DATE_COL, "home_team", "away_team"]
    existing = df_all[key_cols].drop_duplicates()
    tmp_merge = tmp.merge(existing, on=key_cols, how="left", indicator=True)
    new_rows = tmp_merge[tmp_merge["_merge"] == "left_only"].drop(columns=["_merge"])

    if new_rows.empty:
        logging.info("No new upcoming games to add from nba_games_predict.")
        return df_all

    # align schemas
    needed_cols = set(df_all.columns) | set(new_rows.columns)
    for c in needed_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan
        if c not in new_rows.columns:
            new_rows[c] = np.nan

    new_rows[RESULT_COL] = np.nan
    new_rows[RESULT_RAW_COL] = new_rows.get("result", np.nan)

    df_all = pd.concat([df_all, new_rows[df_all.columns]], ignore_index=True)
    return df_all


def compute_home_win_rates(df_all: pd.DataFrame, ymd: str, pred_dir: str) -> str:
    df = df_all.copy()
    df = _ensure_datetime(df, DATE_COL)
    df["date"] = df[DATE_COL]

    team_results = {}
    for team in df["home_team"].dropna().unique():
        team_games = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date", ascending=False).head(20)
        home_games = team_games[team_games["home_team"] == team]
        total_home = len(home_games)
        home_wins = len(home_games[home_games.get("result", pd.Series(index=home_games.index)).astype(str) == str(team)])
        hwr = round(home_wins / total_home, 2) if total_home > 0 else 0.0
        team_results[team] = {
            "Total Last 20 Games": len(team_games),
            "Total Home Games": total_home,
            "Home Wins": home_wins,
            "Home Win Rate": hwr,
        }

    hwr_df = pd.DataFrame.from_dict(team_results, orient="index").sort_values("Home Win Rate", ascending=False)
    out_path = os.path.join(pred_dir, f"home_win_rates_sorted_{ymd}.csv")
    hwr_df.to_csv(out_path, index=True, encoding="utf-8")
    logging.info("Saved home win rates to %s", out_path)
    return out_path


def attach_home_win_rate(df: pd.DataFrame, hwr_path: str) -> pd.DataFrame:
    if not os.path.exists(hwr_path):
        logging.warning("Home win rate file missing: %s", hwr_path)
        df[HOMEWR_COL] = 0.0
        return df

    hwr = pd.read_csv(hwr_path, encoding="utf-8")
    cols = list(hwr.columns)

    # handle both "index as team" and explicit team column
    if "Home Win Rate" in cols and len(cols) >= 1:
        # team is index column if file saved with index=True
        if cols[0] != "Home Win Rate":
            team_col = cols[0]
            win_col = "Home Win Rate"
        else:
            # fallback
            team_col = cols[0]
            win_col = cols[-1]
    else:
        # generic attempt
        lower = {c.lower(): c for c in cols}
        team_col = lower.get("team", cols[0])
        win_col = next((c for c in cols if "win rate" in c.lower()), cols[-1])

    hwr["_team_norm"] = hwr[team_col].astype(str).str.strip().map(normalize_team_code)
    df["_home_team_norm"] = df["home_team"].astype(str).str.strip().map(normalize_team_code)

    hwr_m = hwr[["_team_norm", win_col]].drop_duplicates("_team_norm")
    out = df.merge(hwr_m, left_on="_home_team_norm", right_on="_team_norm", how="left")

    out.rename(columns={win_col: HOMEWR_COL}, inplace=True)
    out[HOMEWR_COL] = pd.to_numeric(out[HOMEWR_COL], errors="coerce").fillna(0.0)

    out.drop(columns=["_team_norm", "_home_team_norm"], inplace=True, errors="ignore")
    return out


def split_past_future(df_all: pd.DataFrame, today_date, tomorrow_date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_all.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["game_day"] = df[DATE_COL].dt.date

    if RESULT_RAW_COL not in df.columns:
        df[RESULT_RAW_COL] = np.nan

    played = df[RESULT_RAW_COL].notna() & (df[RESULT_RAW_COL].astype(str) != "0")
    df_past = df[played].copy()
    df_future = df[~played & df["game_day"].isin([today_date, tomorrow_date])].copy()

    logging.info("Split -> past=%d future=%d", len(df_past), len(df_future))
    return df_past, df_future


# -----------------------------
# ISOTONIC / GRID SEARCH
# -----------------------------

def fit_isotonic(df_past: pd.DataFrame) -> IsotonicRegression:
    m = df_past[RESULT_COL].notna() & df_past[PRED_PROBA_COL].notna()
    if m.sum() == 0:
        raise RuntimeError("No valid rows for isotonic fit.")
    y = df_past.loc[m, RESULT_COL].astype(int).values
    p = df_past.loc[m, PRED_PROBA_COL].astype(float).values
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def compute_calibration_metrics(df_past: pd.DataFrame) -> Tuple[float, float, float, float]:
    m = df_past[RESULT_COL].notna() & df_past[PRED_PROBA_COL].notna() & df_past[ISO_COL].notna()
    if m.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    y = df_past.loc[m, RESULT_COL].astype(int).values
    p_raw = df_past.loc[m, PRED_PROBA_COL].astype(float).values
    p_iso = df_past.loc[m, ISO_COL].astype(float).values
    b0 = brier_score_loss(y, p_raw)
    b1 = brier_score_loss(y, p_iso)
    ll0 = log_loss(y, p_raw, eps=1e-15)
    ll1 = log_loss(y, p_iso, eps=1e-15)
    return b0, b1, ll0, ll1


def evaluate_strategy(df: pd.DataFrame, params: StrategyParams) -> dict:
    if df.empty:
        return {"n_bets": 0, "total_profit": 0.0, "roi_per_bet": 0.0}

    conds = [
        df[HOMEWR_COL] >= params.min_home_win_rate,
        df[HOME_ODDS_COL].between(params.min_odds, params.max_odds),
        df[ISO_COL] >= params.min_iso_proba,
        df[HOME_ODDS_COL].notna(),
        df[ISO_COL].notna(),
        df[RESULT_COL].notna(),
    ]
    mask = np.logical_and.reduce(conds)
    sel = df[mask].copy()
    if sel.empty:
        return {"n_bets": 0, "total_profit": 0.0, "roi_per_bet": 0.0}

    stake = FLAT_STAKE
    sel["profit"] = np.where(
        sel[RESULT_COL].astype(int) == 1,
        (sel[HOME_ODDS_COL] - 1.0) * stake,
        -stake,
    )
    tot = float(sel["profit"].sum())
    n = int(len(sel))
    return {"n_bets": n, "total_profit": tot, "roi_per_bet": tot / (n * stake)}


def grid_search(df_past: pd.DataFrame) -> Tuple[StrategyParams, pd.DataFrame]:
    rows = []
    for hwr in HOMEWR_MIN_GRID:
        for o_min in ODDS_MIN_GRID:
            for o_max in ODDS_MAX_GRID:
                if o_max <= o_min:
                    continue
                for p_min in PROB_MIN_GRID:
                    sp = StrategyParams(hwr, o_min, o_max, p_min)
                    m = evaluate_strategy(df_past, sp)
                    m.update(
                        min_home_win_rate=hwr,
                        min_odds=o_min,
                        max_odds=o_max,
                        min_iso_proba=p_min,
                    )
                    rows.append(m)

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("Grid search produced no results.")
    res = res.sort_values(by=["roi_per_bet", "n_bets"], ascending=[False, False]).reset_index(drop=True)

    best = res.iloc[0]
    best_params = StrategyParams(
        float(best["min_home_win_rate"]),
        float(best["min_odds"]),
        float(best["max_odds"]),
        float(best["min_iso_proba"]),
    )
    return best_params, res


# -----------------------------
# LOCAL SEARCH + FAIR CHOICE
# -----------------------------

def evaluate_params_on_hist_window(
    hist_window: pd.DataFrame,
    params: dict,
    *,
    min_ev: float,
    flat_stake_backtest: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
) -> Tuple[dict, pd.DataFrame]:
    _validate_params(params, name="params_to_eval")

    df = hist_window.copy()
    df = _ensure_datetime(df, DATE_COL)

    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100")
    df = df.dropna(subset=[HOMEWR_COL, "prob_used", HOME_ODDS_COL, RESULT_COL, "EV_€_per_100"])

    prob_thr_eff = max(float(params["prob_threshold"]), float(prob_clip_lo))
    mask = (
        (df[HOMEWR_COL] >= float(params["home_win_rate_threshold"])) &
        (df[HOME_ODDS_COL] >= float(params["odds_min"])) &
        (df[HOME_ODDS_COL] <= float(params["odds_max"])) &
        (df["prob_used"] >= prob_thr_eff) &
        (df["EV_€_per_100"] > float(min_ev))
    )
    subset = df.loc[mask].copy()
    if subset.empty:
        return {
            "n_trades": 0,
            "win_rate_%": 0.0,
            "avg_EV_€_per_100": 0.0,
            "profit_€": 0.0,
            "roi_%": 0.0,
            "prob_thr_eff": round(prob_thr_eff, 3),
        }, subset

    subset["pnl"] = np.where(
        subset[RESULT_COL].astype(int) == 1,
        float(flat_stake_backtest) * (subset[HOME_ODDS_COL] - 1.0),
        -float(flat_stake_backtest),
    )

    profit = float(subset["pnl"].sum())
    n = int(len(subset))
    stake_sum = n * float(flat_stake_backtest)
    roi = (profit / stake_sum * 100.0) if stake_sum else 0.0

    metrics = {
        "n_trades": n,
        "win_rate_%": round(float(subset[RESULT_COL].mean() * 100.0), 2),
        "avg_EV_€_per_100": round(float(subset["EV_€_per_100"].mean()), 2),
        "profit_€": round(profit, 2),
        "roi_%": round(roi, 2),
        "prob_thr_eff": round(prob_thr_eff, 3),
    }
    return metrics, subset


def find_best_local_params_lastN(
    hist_df: pd.DataFrame,
    *,
    window_n: int,
    flat_stake_backtest: float,
    min_ev: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
    min_trades_local: int = 10,
) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    if hist_df is None or hist_df.empty:
        return None, None

    df = _ensure_datetime(hist_df, DATE_COL).sort_values(DATE_COL).tail(int(window_n)).copy()
    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100")
    df = df.dropna(subset=[HOMEWR_COL, "prob_used", HOME_ODDS_COL, RESULT_COL, "EV_€_per_100"])

    if len(df) < 20:
        return None, None

    prob_grid_eff = [p for p in PROB_MIN_GRID if p >= prob_clip_lo] or [prob_clip_lo]

    best_profit = float("-inf")
    best_params = None
    best_subset = None

    for hwr in HOMEWR_MIN_GRID:
        for o_min in ODDS_MIN_GRID:
            for o_max in ODDS_MAX_GRID:
                if o_max <= o_min:
                    continue
                for pmin in prob_grid_eff:
                    mask = (
                        (df[HOMEWR_COL] >= hwr) &
                        (df[HOME_ODDS_COL] >= o_min) &
                        (df[HOME_ODDS_COL] <= o_max) &
                        (df["prob_used"] >= pmin) &
                        (df["EV_€_per_100"] > float(min_ev))
                    )
                    sub = df.loc[mask].copy()
                    if sub.empty or len(sub) < min_trades_local:
                        continue
                    sub["pnl"] = np.where(
                        sub[RESULT_COL].astype(int) == 1,
                        float(flat_stake_backtest) * (sub[HOME_ODDS_COL] - 1.0),
                        -float(flat_stake_backtest),
                    )
                    profit = float(sub["pnl"].sum())
                    if profit > best_profit:
                        best_profit = profit
                        best_subset = sub
                        best_params = {
                            "home_win_rate_threshold": round(float(hwr), 2),
                            "odds_min": round(float(o_min), 2),
                            "odds_max": round(float(o_max), 2),
                            "prob_threshold": round(float(pmin), 2),
                            "n_trades": int(len(sub)),
                            "profit_€": round(profit, 2),
                            "roi_%": round(profit / (len(sub) * flat_stake_backtest) * 100.0, 2),
                        }

    return best_params, best_subset


def choose_params_fair_lastN(
    global_params: dict,
    local_params: Optional[dict],
    *,
    hist_window: pd.DataFrame,
    min_ev: float,
    flat_stake_backtest: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
    min_trades: int = 10,
    roi_edge_min_pp: float = 0.0,
) -> Tuple[bool, dict, dict, Optional[dict], pd.DataFrame, Optional[pd.DataFrame]]:
    metrics_g, subset_g = evaluate_params_on_hist_window(
        hist_window,
        global_params,
        min_ev=min_ev,
        flat_stake_backtest=flat_stake_backtest,
        prob_clip_lo=prob_clip_lo,
        prob_clip_hi=prob_clip_hi,
    )

    if not local_params:
        return False, global_params.copy(), metrics_g, None, subset_g, None

    metrics_l, subset_l = evaluate_params_on_hist_window(
        hist_window,
        local_params,
        min_ev=min_ev,
        flat_stake_backtest=flat_stake_backtest,
        prob_clip_lo=prob_clip_lo,
        prob_clip_hi=prob_clip_hi,
    )

    use_local = (
        metrics_l["n_trades"] >= int(min_trades) and
        metrics_l["profit_€"] > 0 and
        metrics_l["roi_%"] >= (metrics_g["roi_%"] + float(roi_edge_min_pp))
    )

    return use_local, (local_params.copy() if use_local else global_params.copy()), metrics_g, metrics_l, subset_g, subset_l


# -----------------------------
# EXPORTS / SNAPSHOT
# -----------------------------

def prepare_local_matched_export(matched_subset: pd.DataFrame, stake: float) -> pd.DataFrame:
    df = matched_subset.copy()
    if "date" not in df.columns:
        df["date"] = df[DATE_COL]
    df = _ensure_datetime(df, "date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # normalize expected columns
    if "home_win_rate" not in df.columns:
        df["home_win_rate"] = df[HOMEWR_COL]
    if "prob_iso" not in df.columns:
        df["prob_iso"] = df[ISO_COL]
    if "prob_used" not in df.columns:
        df["prob_used"] = df["prob_iso"]
    if "odds_1" not in df.columns:
        df["odds_1"] = df[HOME_ODDS_COL]
    if "win" not in df.columns:
        df["win"] = df[RESULT_COL]

    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col="odds_1", dst="EV_€_per_100")

    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    df["odds_1"] = pd.to_numeric(df["odds_1"], errors="coerce")
    df["pnl"] = np.where(df["win"] == 1, stake * (df["odds_1"] - 1.0), -stake)

    df = df.dropna(subset=["win", "odds_1"]).copy()
    df["win"] = df["win"].clip(0, 1).astype(int)
    df["stake"] = float(stake)

    cols = [
        "date", "home_team", "away_team",
        "home_win_rate", "prob_iso", "prob_used",
        "odds_1", "EV_€_per_100", "win", "pnl", "stake"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].sort_values("date").reset_index(drop=True)


def export_local_matched_games_settled(export_df: pd.DataFrame, *, output_dir: Path, as_of_date: str) -> Optional[Path]:
    if export_df is None or export_df.empty:
        logging.info("No settled local matched games to export.")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"local_matched_games_{as_of_date}.csv"
    export_df.to_csv(p, index=False, encoding="utf-8")
    logging.info("Exported local matched games to %s (%d rows).", p, len(export_df))
    return p


def build_metrics_snapshot(
    export_df: pd.DataFrame,
    *,
    params_used: dict,
    params_used_type: str,
    min_ev: float,
    as_of_date: str,
    stake: float,
    strategy_variant: str,
    strategy_variant_label: str,
    params_source: str,
    combined_file_path: str,
    local_matched_games_source: Optional[str],
    bankroll_last_200_eur: Optional[float],
    bankroll_2026_ytd_eur: Optional[float],
    profit_2026_ytd_eur: Optional[float],
    settled_summary: Optional[dict],
) -> dict:
    realized_count = int(len(export_df)) if export_df is not None else 0
    profit_sum = float(export_df["pnl"].sum()) if realized_count else 0.0
    roi = profit_sum / (realized_count * float(stake)) if realized_count else 0.0
    win_rate = float(export_df["win"].mean()) if realized_count else 0.0
    ev_mean = float(export_df["EV_€_per_100"].mean()) if realized_count else 0.0
    if np.isnan(ev_mean):
        ev_mean = 0.0

    sharpe_style = 0.0
    if realized_count > 1:
        sd = float(export_df["pnl"].std(ddof=1))
        mu = float(export_df["pnl"].mean())
        sharpe_style = (mu / sd) if sd > 0 else 0.0

    snap = {
        "meta": {
            "eval_base_date_max": as_of_date,
            "strategy_variant": strategy_variant,
            "strategy_variant_label": strategy_variant_label,
            "params_source": params_source,
            "combined_file_path": combined_file_path,
            "local_matched_games_source": local_matched_games_source,
        },
        "params_used_type": params_used_type,
        "params_used": params_used,
        "realized": {
            "count": realized_count,
            "profit_sum": round(profit_sum, 2),
            "roi": round(roi, 4),
            "win_rate": round(win_rate, 4),
            "sharpe_style": round(float(sharpe_style), 4),
        },
        "ev_stats": {"mean": round(ev_mean, 2)},
        "filter_params": {
            "home_win_rate_threshold": float(params_used["home_win_rate_threshold"]),
            "odds_min": float(params_used["odds_min"]),
            "odds_max": float(params_used["odds_max"]),
            "prob_threshold": float(params_used["prob_threshold"]),
            "min_EV": float(min_ev),
        },
        "bankroll": {
            "deposit_eur": round(float(START_BANKROLL), 2),
            "bankroll_last_200_eur": round(float(bankroll_last_200_eur), 2) if bankroll_last_200_eur is not None else None,
            "bankroll_2026_ytd_eur": round(float(bankroll_2026_ytd_eur), 2) if bankroll_2026_ytd_eur is not None else None,
            "profit_2026_ytd_eur": round(float(profit_2026_ytd_eur), 2) if profit_2026_ytd_eur is not None else None,
            "flat_stake_eur": round(float(stake), 2),
        },
    }
    if settled_summary is not None:
        snap["settled_bets_2026"] = settled_summary
    return snap


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", path)


def write_strategy_params(params_used: dict, *, min_ev: float, as_of_date: str, stake: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "strategy_params.txt"
    lines = [f"as_of_date={as_of_date}", f"min_ev={float(min_ev)}", f"stake={float(stake)}"]
    for k in sorted(params_used.keys()):
        lines.append(f"{k}={params_used[k]}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved %s", out)


# -----------------------------
# NEW: write latest CSV for dashboard
# -----------------------------

def write_latest_local_matched_csv(export_df: Optional[pd.DataFrame]) -> None:
    """
    Minimal add-on: writes web/public/data/local_matched_games_latest.csv
    using the already computed matched_export (no filter logic change).
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]
        out_path = repo_root / "web" / "public" / "data" / "local_matched_games_latest.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cols = [
            "date", "home_team", "away_team",
            "home_win_rate", "prob_iso", "prob_used",
            "odds_1", "EV_€_per_100", "win", "pnl", "stake"
        ]

        if export_df is None or export_df.empty:
            pd.DataFrame(columns=cols).to_csv(out_path, index=False, encoding="utf-8")
            logging.info("Wrote EMPTY local_matched_games_latest.csv -> %s", out_path)
            return

        df = export_df.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df[cols].to_csv(out_path, index=False, encoding="utf-8")
        logging.info("Wrote local_matched_games_latest.csv -> %s (%d rows)", out_path, len(df))

    except Exception as e:
        logging.warning("Could not write local_matched_games_latest.csv: %s", e)


# -----------------------------
# REAL BETS (optional)
# -----------------------------

def find_bet_log_path(output_dir: Path) -> Optional[Path]:
    for c in [output_dir / "bet_log_flat_live.csv", output_dir / "bet_log_live.csv"]:
        if c.exists():
            return c
    return None


def _norm_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def build_settled_bets(bet_log_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    if bet_log_df is None or bet_log_df.empty:
        return pd.DataFrame()

    df = bet_log_df.copy()

    # minimal schema resolver
    date_col = "date" if "date" in df.columns else ("game_date" if "game_date" in df.columns else None)
    home_col = "home_team" if "home_team" in df.columns else ("home" if "home" in df.columns else None)
    away_col = "away_team" if "away_team" in df.columns else ("away" if "away" in df.columns else None)
    odds_col = "odds" if "odds" in df.columns else ("odds_1" if "odds_1" in df.columns else None)
    stake_col = "stake" if "stake" in df.columns else ("stake_eur" if "stake_eur" in df.columns else None)
    status_col = "status" if "status" in df.columns else None
    won_col = "win" if "win" in df.columns else ("won" if "won" in df.columns else None)

    if not date_col or not home_col or not away_col or not odds_col or not stake_col:
        return pd.DataFrame()

    df["date"] = _norm_date(df[date_col])
    df["home_team"] = _norm_text(df[home_col])
    df["away_team"] = _norm_text(df[away_col])
    df["odds_1"] = pd.to_numeric(df[odds_col], errors="coerce")
    df["stake"] = pd.to_numeric(df[stake_col], errors="coerce")

    if status_col:
        df["status"] = _norm_text(df[status_col]).str.upper()
        df = df[df["status"] == "SETTLED"].copy()

    if df.empty:
        return pd.DataFrame()

    if won_col and won_col in df.columns:
        df["win"] = pd.to_numeric(df[won_col], errors="coerce")
    else:
        df["win"] = np.nan

    # if win missing -> derive from results_df using result==home_team
    if df["win"].isna().any() and results_df is not None and not results_df.empty:
        res = results_df.copy()
        if "date" not in res.columns:
            res["date"] = res[DATE_COL]
        res = _ensure_datetime(res, "date")
        res["date"] = res["date"].dt.strftime("%Y-%m-%d")
        res["home_team_n"] = _norm_text(res["home_team"])
        res["away_team_n"] = _norm_text(res["away_team"])

        # result_raw might be teamcode winner
        if "result" in res.columns:
            res["result_n"] = _norm_text(res["result"])
        elif RESULT_RAW_COL in res.columns:
            res["result_n"] = _norm_text(res[RESULT_RAW_COL])
        else:
            res["result_n"] = ""

        res["home_won"] = (res["result_n"] == res["home_team_n"]).astype(float)

        df = df.merge(
            res[["date", "home_team_n", "away_team_n", "home_won"]],
            left_on=["date", "home_team", "away_team"],
            right_on=["date", "home_team_n", "away_team_n"],
            how="left",
        )
        df["win"] = df["win"].fillna(df["home_won"])
        df.drop(columns=["home_team_n", "away_team_n", "home_won"], inplace=True, errors="ignore")

    df = df.dropna(subset=["stake", "odds_1", "win"]).copy()
    df["win"] = df["win"].clip(0, 1).astype(int)
    df["pnl"] = np.where(df["win"] == 1, df["stake"] * (df["odds_1"] - 1.0), -df["stake"])
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"]).sort_values("date").reset_index(drop=True)
    return df


# -----------------------------
# UPCOMING SHORTLIST
# -----------------------------

def build_flat_shortlist_today(
    upcoming_df: pd.DataFrame,
    params_used: dict,
    *,
    min_ev: float,
    flat_stake_live: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
) -> pd.DataFrame:
    if upcoming_df is None or upcoming_df.empty:
        return pd.DataFrame()

    _validate_params(params_used, name="params_used")

    df = upcoming_df.copy()
    if "date" not in df.columns:
        df["date"] = df[DATE_COL]
    df = _ensure_datetime(df, "date")

    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100")
    df = _make_game_key(df, date_col="date", dst="game_key")

    df = df.sort_values("EV_€_per_100", ascending=False).drop_duplicates("game_key").reset_index(drop=True)

    prob_thr_eff = max(float(params_used["prob_threshold"]), float(prob_clip_lo))

    mask = (
        (df[HOMEWR_COL] >= float(params_used["home_win_rate_threshold"])) &
        (df[HOME_ODDS_COL] >= float(params_used["odds_min"])) &
        (df[HOME_ODDS_COL] <= float(params_used["odds_max"])) &
        (df["prob_used"] >= prob_thr_eff) &
        (df["EV_€_per_100"] > float(min_ev))
    )

    picks = df.loc[mask].copy()
    if picks.empty:
        return picks

    picks["stake_flat"] = float(flat_stake_live)
    picks["EV_€"] = ((picks["prob_used"] * (picks[HOME_ODDS_COL] - 1.0) - (1.0 - picks["prob_used"])) * picks["stake_flat"]).round(2)
    picks["potential_profit_if_win"] = (picks["stake_flat"] * (picks[HOME_ODDS_COL] - 1.0)).round(2)
    picks["fair_odds"] = (1.0 / picks["prob_used"]).round(3)
    picks["edge_pct"] = ((picks[HOME_ODDS_COL] / picks["fair_odds"] - 1.0) * 100.0).round(2)

    picks["home_win_rate"] = picks[HOMEWR_COL]
    picks["prob_iso"] = picks[ISO_COL]
    picks["odds_1"] = picks[HOME_ODDS_COL]

    return picks.sort_values("date").reset_index(drop=True)


# -----------------------------
# AS-OF DATE FIX (IMPORTANT)
# -----------------------------

def resolve_as_of_date_from_df_all(df_all: pd.DataFrame, fallback: str) -> str:
    """
    FIX: derive as_of_date from latest played game in df_all (result_raw present and not '0').
    This avoids stale as_of_date when df_past split is unexpectedly capped.
    """
    if df_all is None or df_all.empty:
        return fallback

    df = df_all.copy()
    if RESULT_RAW_COL not in df.columns:
        return fallback

    played = df[RESULT_RAW_COL].notna() & (df[RESULT_RAW_COL].astype(str) != "0")
    if not played.any():
        return fallback

    dt = pd.to_datetime(df.loc[played, DATE_COL], errors="coerce")
    if dt.notna().any():
        return dt.max().strftime("%Y-%m-%d")
    return fallback


# -----------------------------
# MAIN
# -----------------------------

def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: today).")
    args = parser.parse_args()

    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d")
        target_ymd = args.date
    else:
        now_dt, _, ymd_str = get_current_date()
        target_dt = now_dt
        target_ymd = ymd_str

    today_date = target_dt.date()
    tomorrow_date = (target_dt + timedelta(days=1)).date()

    paths = get_directory_paths()
    pred_dirs = paths["PREDICTION_DIRS"]

    strategy_variant = resolve_strategy_variant()
    strategy_variant_label = resolve_strategy_variant_label(strategy_variant)
    logging.info("Strategy variant: %s", strategy_variant_label)

    combined_path, combined_date = resolve_dated_file(
        pred_dirs,
        "combined_nba_predictions_acc_",
        target_ymd,
        latest_on_or_before=target_dt,
    )
    if not combined_path or not combined_date:
        raise FileNotFoundError("No combined_nba_predictions_acc_*.csv files found.")

    # allow fallback to latest combined
    if combined_date != target_ymd:
        logging.info("Using latest combined date %s instead of %s", combined_date, target_ymd)
        target_ymd = combined_date
        target_dt = datetime.strptime(target_ymd, "%Y-%m-%d")
        today_date = target_dt.date()
        tomorrow_date = (target_dt + timedelta(days=1)).date()

    pred_dir = str(combined_path.parent)
    out_dir = resolve_output_dir(paths["BASE_DIR"], pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kelly_dir = Path(pred_dir) / "Kelly"
    kelly_dir.mkdir(parents=True, exist_ok=True)

    params_source_path = resolve_params_source(out_dir, strategy_variant)
    params_source = str(params_source_path)

    # 1) LOAD COMBINED
    df_all = load_combined_df(pred_dir, target_ymd)
    df_all = _ensure_datetime(df_all, DATE_COL)

    # 1b) HOME WIN RATES (based on df_all played history)
    hwr_path = compute_home_win_rates(df_all, target_ymd, pred_dir)

    # 2) MERGE TODAY PREDICTIONS (optional)
    today_pred_path, today_pred_date = resolve_dated_file(
        pred_dirs,
        "nba_games_predict_",
        target_ymd,
        latest_on_or_before=target_dt,
    )
    pred_date = datetime.strptime(today_pred_date, "%Y-%m-%d").date() if today_pred_date else today_date
    df_all = merge_today_predictions(df_all, today_pred_path, pred_date)

    # 3) ATTACH HOME WIN RATE
    df_all = attach_home_win_rate(df_all, hwr_path)

    # 4) SPLIT PAST / FUTURE
    df_past, df_future = split_past_future(df_all, today_date, tomorrow_date)
    if df_past.empty:
        logging.warning("No past games available – cannot fit isotonic.")
        return

    # 5) FIT ISOTONIC + APPLY
    iso = fit_isotonic(df_past)
    df_all[ISO_COL] = np.nan
    m_iso = df_all[PRED_PROBA_COL].notna()
    df_all.loc[m_iso, ISO_COL] = iso.transform(df_all.loc[m_iso, PRED_PROBA_COL].astype(float).values)

    # refresh past/future views with ISO column
    df_past, df_future = split_past_future(df_all, today_date, tomorrow_date)

    b0, b1, ll0, ll1 = compute_calibration_metrics(df_past)
    logging.info("Brier before=%.6f after=%.6f | LogLoss before=%.6f after=%.6f", b0, b1, ll0, ll1)

    # 6) GRID SEARCH (doc)
    best_params, df_grid = grid_search(df_past)
    best_params_dict = _params_to_dict(best_params)

    grid_path = kelly_dir / f"nba_grid_search_results_{target_ymd}.csv"
    df_grid.to_csv(grid_path, index=False, encoding="utf-8")
    logging.info("Saved grid search -> %s", grid_path)

    iso_path = kelly_dir / f"combined_nba_predictions_iso_{target_ymd}.csv"
    df_all.to_csv(iso_path, index=False, encoding="utf-8")
    logging.info("Saved ISO combined -> %s", iso_path)

    combined_source_path = iso_path if strategy_variant == "iso" else combined_path

    # 7) LOCAL SEARCH on last N games
    min_EV = MIN_EV_DEFAULT
    local_params, _local_subset = find_best_local_params_lastN(
        df_past,
        window_n=LOCAL_SEARCH_N,
        flat_stake_backtest=FLAT_STAKE,
        min_ev=min_EV,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
        min_trades_local=10,
    )

    # 8) FAIR COMPARE on last 200 (GLOBAL vs LOCAL)
    df_past_sorted = _ensure_datetime(df_past, DATE_COL).sort_values(DATE_COL)
    hist_window_fair = df_past_sorted.tail(int(FAIR_COMPARE_N)).copy()

    use_local, params_used, metrics_g, metrics_l, subset_g, subset_l = choose_params_fair_lastN(
        best_params_dict,
        local_params,
        hist_window=hist_window_fair,
        min_ev=min_EV,
        flat_stake_backtest=FLAT_STAKE,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
        min_trades=10,
        roi_edge_min_pp=0.0,
    )

    matched_window_df = subset_l if use_local else subset_g
    matched_export = prepare_local_matched_export(matched_window_df, stake=FLAT_STAKE) if matched_window_df is not None else pd.DataFrame()

    # -------------------------
    # FIXED: as_of_date from df_all played max (NOT df_past)
    # -------------------------
    as_of_date = resolve_as_of_date_from_df_all(df_all, fallback=target_ymd)

    export_path = export_local_matched_games_settled(
        matched_export,
        output_dir=out_dir,
        as_of_date=as_of_date,
    )

    # -------------------------
    # FIXED: bankroll_last_200 uses last 200 played games (sorted), not matched subset
    # (this is "model EV bankroll", same style as your previous loop)
    # -------------------------
    last_200_games = df_past_sorted.tail(200).copy()
    last_200_games = _compute_prob_used(last_200_games, lo=PROB_CLIP_LO, hi=PROB_CLIP_HI, src=ISO_COL, dst="prob_used")
    last_200_games = last_200_games.dropna(subset=["prob_used", HOME_ODDS_COL])

    bankroll_last_200 = float(START_BANKROLL)
    for _, r in last_200_games.iterrows():
        p = float(r["prob_used"])
        o = float(r[HOME_ODDS_COL])
        bankroll_last_200 += FLAT_STAKE * (p * (o - 1.0) - (1.0 - p))

    # real bets (optional)
    bet_log_path = find_bet_log_path(out_dir)
    settled_summary = None
    bankroll_2026 = None
    profit_2026 = None

    if bet_log_path and bet_log_path.exists():
        bet_log_df = pd.read_csv(bet_log_path)
        settled_bets_df = build_settled_bets(bet_log_df, df_past_sorted)
        if not settled_bets_df.empty:
            settled_bets_df["date_dt"] = pd.to_datetime(settled_bets_df["date"], errors="coerce")
            y2026 = settled_bets_df[settled_bets_df["date_dt"].dt.year == 2026].copy()
        else:
            y2026 = pd.DataFrame()

        profit_2026 = float(y2026["pnl"].sum()) if not y2026.empty else 0.0
        bankroll_2026 = float(START_BANKROLL) + profit_2026
        stake_sum = float(y2026["stake"].sum()) if not y2026.empty else 0.0
        settled_summary = {
            "count": int(len(y2026)),
            "wins": int(y2026["win"].sum()) if not y2026.empty else 0,
            "profit_sum": round(profit_2026, 2),
            "roi": round((profit_2026 / stake_sum), 4) if stake_sum else 0.0,
            "avg_odds": round(float(y2026["odds_1"].mean()), 2) if not y2026.empty else 0.0,
        }

    snapshot = build_metrics_snapshot(
        matched_export,
        params_used=params_used,
        params_used_type="LOCAL" if use_local else "GLOBAL",
        min_ev=min_EV,
        as_of_date=as_of_date,
        stake=FLAT_STAKE,
        strategy_variant=strategy_variant,
        strategy_variant_label=strategy_variant_label,
        params_source=params_source,
        combined_file_path=str(combined_source_path),
        local_matched_games_source=str(export_path) if export_path else None,
        bankroll_last_200_eur=bankroll_last_200,
        bankroll_2026_ytd_eur=bankroll_2026,
        profit_2026_ytd_eur=profit_2026,
        settled_summary=settled_summary,
    )

    # write outputs
    write_json(out_dir / "metrics_snapshot.json", snapshot)
    write_strategy_params(params_used, min_ev=min_EV, as_of_date=as_of_date, stake=FLAT_STAKE, output_dir=out_dir)
    write_json(out_dir / "summary.json", {
        "as_of_date": as_of_date,
        "strategy_variant": strategy_variant,
        "strategy_variant_label": strategy_variant_label,
        "params_source": params_source,
        "combined_file_path": str(combined_source_path),
        "local_matched_games_source": str(export_path) if export_path else None,
    })

    # shortlist for today/tomorrow
    flat_today = build_flat_shortlist_today(
        upcoming_df=df_future,
        params_used=params_used,
        min_ev=min_EV,
        flat_stake_live=FLAT_STAKE,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
    )
    if not flat_today.empty:
        shortlist_path = Path(pred_dir) / f"bet_shortlist_{target_ymd}.csv"
        flat_today.to_csv(shortlist_path, index=False, encoding="utf-8")
        logging.info("Saved shortlist -> %s (%d rows)", shortlist_path, len(flat_today))
    else:
        logging.info("No shortlist bets for %s", target_ymd)

    # ---------------------------------------------------------------------
    # ONLY ADDITION: also publish "latest" file for the hoops-insight dashboard
    # ---------------------------------------------------------------------
    write_latest_local_matched_csv(matched_export)

    logging.info("DONE. metrics_snapshot as_of=%s | out_dir=%s", as_of_date, out_dir)


if __name__ == "__main__":
    main()
