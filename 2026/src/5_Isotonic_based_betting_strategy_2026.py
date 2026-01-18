#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_Isotonic_based_betting_strategy_2026.py  (DROP-IN)

Core requirement implemented:
- Script 5 ALWAYS generates strategy params each run (no reading from strategy_params.json).
- web/public/data/local_matched_games_latest.csv contains the ACTUAL FILTERED subset
  from the LAST 200 PLAYED games window, ending at the latest played game date (typically yesterday).

Everything else is kept as close as possible to your existing version.
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

# Grid search
FLAT_STAKE = 100.0
ODDS_MIN_GRID = [1.10, 1.25, 1.40, 1.60, 2.00, 2.30]
ODDS_MAX_GRID = [2.00, 2.10, 2.50, 3.00, 3.20]
PROB_MIN_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HOMEWR_MIN_GRID = [0.50, 0.55, 0.60, 0.65]

# Dashboard / shortlist logic
MIN_EV_DEFAULT = -5.0
PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80

# IMPORTANT: dashboard window is last 200 played games
LOCAL_SEARCH_N = 200
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

def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent and not (p / "web" / "public").exists():
        p = p.parent
    return p


def get_yesterday_date(target_dt: datetime) -> datetime.date:
    # Always treat "yesterday" as the last completed day relative to target_dt
    return (target_dt - timedelta(days=1)).date()


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
    env_path = os.environ.get("STRATEGY_PARAMS_PATH", "").strip()
    if env_path:
        return Path(env_path)

    # find repo root by walking up until we see "web/public"
    repo_root = Path(__file__).resolve()
    while repo_root != repo_root.parent and not (repo_root / "web" / "public").exists():
        repo_root = repo_root.parent

    candidates = [
        repo_root / "public" / "data" / "strategy_params.json",
        repo_root / "web" / "public" / "data" / "strategy_params.json",
    ]
    for c in candidates:
        if c.exists():
            return c

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


def resolve_output_dir(base_dir: str, prediction_dir: str) -> Path:
    lgbm_dir = os.environ.get("LGBM_DIR", "").strip()
    if lgbm_dir:
        p = Path(lgbm_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

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

    if "result" in df.columns:
        df[RESULT_RAW_COL] = df["result"]
    else:
        df[RESULT_RAW_COL] = np.nan

    if DATE_COL not in df.columns:
        if "date" in df.columns:
            df[DATE_COL] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df[DATE_COL] = pd.NaT
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if PRED_PROBA_COL not in df.columns:
        if "home_team_prob" in df.columns:
            df[PRED_PROBA_COL] = to_float_series(df["home_team_prob"])
        else:
            df[PRED_PROBA_COL] = np.nan

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

    if "home_team_prob" in tmp.columns:
        tmp["home_team_prob"] = to_float_series(tmp["home_team_prob"])
    if "odds_1" in tmp.columns:
        tmp["odds_1"] = to_float_series(tmp["odds_1"])
    if "odds_2" in tmp.columns:
        tmp["odds_2"] = to_float_series(tmp["odds_2"])

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

    if "Home Win Rate" in cols and len(cols) >= 1:
        if cols[0] != "Home Win Rate":
            team_col = cols[0]
            win_col = "Home Win Rate"
        else:
            team_col = cols[0]
            win_col = cols[-1]
    else:
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
# ISOTONIC / METRICS
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


# -----------------------------
# STRATEGY SEARCH (LOCAL on last 200)
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


# -----------------------------
# EXPORT HELPERS
# -----------------------------

def prepare_local_matched_export(matched_subset: pd.DataFrame, stake: float) -> pd.DataFrame:
    df = matched_subset.copy()
    if "date" not in df.columns:
        df["date"] = df[DATE_COL]
    df = _ensure_datetime(df, "date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

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

    out = df[cols].copy()

    # IMPORTANT: prevent duplicates (your current latest CSV has duplicates)
    out = out.sort_values(["date", "home_team", "away_team", "EV_€_per_100"], ascending=[True, True, True, False])
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

    return out.sort_values("date").reset_index(drop=True)


def write_latest_local_matched_csv(
    df_past_sorted: pd.DataFrame,
    *,
    params_used: dict,
    target_dt: datetime,
    window_n: int,
    min_ev: float,
    stake: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
) -> None:
    """
    Writes web/public/data/local_matched_games_latest.csv as:
    - played games only
    - limited to last completed day (yesterday relative to target_dt)
    - last N games inside that cutoff
    - filtered by params_used + EV > min_ev
    - exported with win/pnl/stake columns

    This is the ONLY place that defines "latest" for the dashboard.
    """
    try:
        _validate_params(params_used, name="params_used")

        repo_root = find_repo_root()
        out_path = repo_root / "web" / "public" / "data" / "local_matched_games_latest.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cols = [
            "date", "home_team", "away_team",
            "home_win_rate", "prob_iso", "prob_used",
            "odds_1", "EV_€_per_100", "win", "pnl", "stake"
        ]

        if df_past_sorted is None or df_past_sorted.empty:
            pd.DataFrame(columns=cols).to_csv(out_path, index=False, encoding="utf-8")
            logging.info("Wrote EMPTY local_matched_games_latest.csv -> %s", out_path)
            return

        # --- cutoff to yesterday ---
        df = df_past_sorted.copy()
        df = _ensure_datetime(df, DATE_COL)
        
        max_played_ts = pd.to_datetime(df[DATE_COL], errors="coerce").max()
        if pd.isna(max_played_ts):
            pd.DataFrame(columns=cols).to_csv(out_path, index=False, encoding="utf-8")
            logging.info("Wrote EMPTY local_matched_games_latest.csv -> %s (no played dates)", out_path)
            return
        
        cutoff_date = max_played_ts.date()
        df = df[df[DATE_COL].dt.date <= cutoff_date].copy()

        # last N played games inside cutoff
        df = df.sort_values(DATE_COL).tail(int(window_n)).copy()

        # compute prob_used + EV
        df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
        df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100")

        # apply filters (same as evaluate_params_on_hist_window)
        prob_thr_eff = max(float(params_used["prob_threshold"]), float(prob_clip_lo))
        mask = (
            (df[HOMEWR_COL] >= float(params_used["home_win_rate_threshold"])) &
            (df[HOME_ODDS_COL] >= float(params_used["odds_min"])) &
            (df[HOME_ODDS_COL] <= float(params_used["odds_max"])) &
            (df["prob_used"] >= prob_thr_eff) &
            (df["EV_€_per_100"] > float(min_ev))
        )
        subset = df.loc[mask].copy()

        if subset.empty:
            pd.DataFrame(columns=cols).to_csv(out_path, index=False, encoding="utf-8")
            logging.info("Wrote EMPTY (no matches) local_matched_games_latest.csv -> %s", out_path)
            return

        export_df = prepare_local_matched_export(subset, stake=float(stake))
        for c in cols:
            if c not in export_df.columns:
                export_df[c] = np.nan

        export_df[cols].to_csv(out_path, index=False, encoding="utf-8")
        logging.info(
            "Wrote local_matched_games_latest.csv -> %s (%d rows) [window_n=%d cutoff<=%s]",
            out_path,
            len(export_df),
            int(window_n),
            str(cutoff_date),
        )

  
        (out_path.parent / "local_matched_games_latest__written_by_script5.txt").write_text(
            f"written_at_utc={datetime.utcnow().isoformat()}Z\n"
            f"rows={len(export_df)}\n"
            f"first_date={export_df['date'].iloc[0] if len(export_df) else 'NA'}\n"
            f"last_date={export_df['date'].iloc[-1] if len(export_df) else 'NA'}\n",
            encoding="utf-8",
        )

    except Exception as e:
        logging.warning("Could not write local_matched_games_latest.csv: %s", e)



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", path)


def write_strategy_params(params_used: dict, *, min_ev: float, as_of_date: str, stake: float, output_dir: Path) -> None:
    """
    Keep as txt for assets builder, but write the GENERATED params each run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "strategy_params.txt"
    lines = [f"as_of_date={as_of_date}", f"min_ev={float(min_ev)}", f"stake={float(stake)}"]
    for k in sorted(params_used.keys()):
        lines.append(f"{k}={params_used[k]}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved %s", out)


# -----------------------------
# AS-OF DATE (latest played)
# -----------------------------

def resolve_as_of_date_from_df_past(df_past_sorted: pd.DataFrame, fallback: str) -> str:
    if df_past_sorted is None or df_past_sorted.empty:
        return fallback
    dt = pd.to_datetime(df_past_sorted[DATE_COL], errors="coerce")
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

    # 1b) HOME WIN RATES
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

    # refresh past/future views with ISO
    df_past, df_future = split_past_future(df_all, today_date, tomorrow_date)

    # sort past chronologically
    df_past_sorted = _ensure_datetime(df_past, DATE_COL).sort_values(DATE_COL).copy()

    # AS OF DATE = max played date (yesterday in practice)
    as_of_date = resolve_as_of_date_from_df_past(df_past_sorted, fallback=target_ymd)

    # 6) CALIBRATION METRICS (window overall)
    b0, b1, ll0, ll1 = compute_calibration_metrics(df_past)
    logging.info("Brier before=%.6f after=%.6f | LogLoss before=%.6f after=%.6f", b0, b1, ll0, ll1)

    # 7) SAVE ISO COMBINED (unchanged)
    iso_path = kelly_dir / f"combined_nba_predictions_iso_{target_ymd}.csv"
    df_all.to_csv(iso_path, index=False, encoding="utf-8")
    logging.info("Saved ISO combined -> %s", iso_path)

    combined_source_path = iso_path if strategy_variant == "iso" else combined_path

    # ------------------------------------------------------------------
    # CORE: generate LOCAL params from LAST 200 played games (dashboard)
    # ------------------------------------------------------------------
    min_EV = MIN_EV_DEFAULT
    hist_window_200 = df_past_sorted.tail(int(FAIR_COMPARE_N)).copy()  # last 200 played games

    local_params, _ = find_best_local_params_lastN(
        df_past_sorted,
        window_n=LOCAL_SEARCH_N,          # =200
        flat_stake_backtest=FLAT_STAKE,
        min_ev=min_EV,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
        min_trades_local=10,
    )

    # fallback if search found nothing
    if not local_params:
        logging.warning("LOCAL param search returned None; using safe fallback.")
        local_params = {
            "home_win_rate_threshold": 0.50,
            "odds_min": 2.30,
            "odds_max": 3.20,
            "prob_threshold": 0.45,
            "n_trades": 0,
            "profit_€": 0.0,
            "roi_%": 0.0,
        }

    # Build matched subset STRICTLY on the last-200 window using these LOCAL params
    metrics_local, subset_local = evaluate_params_on_hist_window(
        hist_window_200,
        local_params,
        min_ev=min_EV,
        flat_stake_backtest=FLAT_STAKE,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
    )

    matched_export_latest = prepare_local_matched_export(subset_local, stake=FLAT_STAKE) if subset_local is not None else pd.DataFrame()

    # Bankroll over last-200 window (model EV, same style)
    last_200_games = hist_window_200.copy()
    last_200_games = _compute_prob_used(last_200_games, lo=PROB_CLIP_LO, hi=PROB_CLIP_HI, src=ISO_COL, dst="prob_used")
    last_200_games = last_200_games.dropna(subset=["prob_used", HOME_ODDS_COL])

    bankroll_last_200 = float(START_BANKROLL)
    for _, r in last_200_games.iterrows():
        p = float(r["prob_used"])
        o = float(r[HOME_ODDS_COL])
        bankroll_last_200 += FLAT_STAKE * (p * (o - 1.0) - (1.0 - p))

    # -------------------------
    # Write the ONLY file you want to be correct on dashboard:
    # web/public/data/local_matched_games_latest.csv
    # -------------------------
    write_latest_local_matched_csv(
      df_past_sorted,
      params_used=local_params,
      target_dt=target_dt,
      window_n=FAIR_COMPARE_N,   # 200
      min_ev=min_EV,
      stake=FLAT_STAKE,
      prob_clip_lo=PROB_CLIP_LO,
      prob_clip_hi=PROB_CLIP_HI,
    )


    # Also write strategy params TXT (generated each run)
    write_strategy_params(local_params, min_ev=min_EV, as_of_date=as_of_date, stake=FLAT_STAKE, output_dir=out_dir)

    # Minimal snapshot for trace (keep structure, but based on LOCAL params + last-200)
    snapshot = {
        "meta": {
            "eval_base_date_max": as_of_date,
            "strategy_variant": strategy_variant,
            "strategy_variant_label": strategy_variant_label,
            "params_source": params_source,
            "combined_file_path": str(combined_source_path),
            "local_matched_games_source": None,
        },
        "params_used_type": "LOCAL",
        "params_used": local_params,
        "local_window_200": {
            "min_EV_applied": float(min_EV),
            "metrics": metrics_local,
            "window_start": str(pd.to_datetime(hist_window_200[DATE_COL], errors="coerce").min().date()) if not hist_window_200.empty else None,
            "window_end": str(pd.to_datetime(hist_window_200[DATE_COL], errors="coerce").max().date()) if not hist_window_200.empty else None,
            "window_games": int(len(hist_window_200)),
            "matched_games": int(len(matched_export_latest)),
        },
        "bankroll": {
            "deposit_eur": round(float(START_BANKROLL), 2),
            "bankroll_last_200_eur": round(float(bankroll_last_200), 2),
            "flat_stake_eur": round(float(FLAT_STAKE), 2),
        },
    }

    write_json(out_dir / "metrics_snapshot.json", snapshot)
    write_json(out_dir / "summary.json", {
        "as_of_date": as_of_date,
        "strategy_variant": strategy_variant,
        "strategy_variant_label": strategy_variant_label,
        "params_source": params_source,
        "combined_file_path": str(combined_source_path),
        "local_matched_games_latest_written": True,
        "local_window_games": int(len(hist_window_200)),
        "local_matched_games": int(len(matched_export_latest)),
    })

    logging.info("DONE. local_matched_games_latest.csv updated using LOCAL params on last-200 window ending %s.", as_of_date)

    p = find_repo_root() / "web" / "public" / "data" / "local_matched_games_latest.csv"

    logging.info("AFTER WRITE: latest size=%d bytes mtime=%s", p.stat().st_size, datetime.fromtimestamp(p.stat().st_mtime))
    logging.info("AFTER WRITE HEAD:\n%s", "\n".join(p.read_text(encoding="utf-8").splitlines()[:5]))



if __name__ == "__main__":
    main()
