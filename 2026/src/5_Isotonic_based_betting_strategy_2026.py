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

from live_probability_pipeline import build_probability_chain_config, prepare_live_probability_columns

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
PROB_ISO_OOS_TIME_COL = "prob_iso_oos_time"
PROB_LIVE_OOS_PROXY_COL = "prob_live_oos_proxy"
PROB_LIVE_SAFE_COL = "prob_live_safe"
PROB_COL_HIST = PROB_ISO_OOS_TIME_COL
PROB_COL_LIVE = PROB_LIVE_SAFE_COL
MIN_TRAIN_OOS_TIME = 50
MIN_STEP_OOS_TIME = 10
MIN_TRAIN_OOS_PROXY = 300

# Grid search
FLAT_STAKE = 100.0
ODDS_MIN_GRID = [1.10, 1.25, 1.40, 1.60, 2.00, 2.30]
ODDS_MAX_GRID = [2.00, 2.10, 2.50, 3.00, 3.20]
PROB_MIN_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HOMEWR_MIN_GRID = [0.50, 0.55, 0.60, 0.65]

# Dashboard / shortlist logic
MIN_EV_DEFAULT = 0.0
PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80

# IMPORTANT: dashboard window is last 200 played games
LOCAL_SEARCH_N = 200
FAIR_COMPARE_N = 200
MIN_HIST_ROWS_FOR_LOCAL = 100

START_BANKROLL = 1000.0

STRATEGY_VARIANTS = {"acc", "iso"}

OUTPUT_BASE_COLUMNS = [
    "home_team",
    "away_team",
    "home_team_prob",
    "odds_1",
    "odds_2",
    "result",
    "date",
    "accuracy",
]

DATE_SOURCE_CANDIDATES = [
    "date",
    DATE_COL,
    "game_date",
    "date_x",
    "Date",
    "DATE",
    "datetime",
    "timestamp",
]

OUTPUT_PROBABILITY_COLUMNS = [
    "prob_iso",
    "prob_iso_insample",
    "prob_iso_oos_time",
    "prob_live_oos_proxy",
    "prob_live_safe_pre_clip",
    "prob_base",
    "prob_live_safe",
    "prob_used",
]

LOCAL_MATCHED_EXPORT_COLUMNS = [
    "date", "home_team", "away_team",
    "home_win_rate", "prob_iso", "prob_used",
    "odds_1", "EV_€_per_100", "win", "pnl", "stake"
]

SHORTLIST_COLUMNS = [
    DATE_COL, "home_team", "away_team", "home_team_prob", "prob_iso", PROB_ISO_OOS_TIME_COL,
    PROB_LIVE_OOS_PROXY_COL, "prob_live_safe_pre_clip", "prob_base", "prob_used",
    "odds_1", "market_implied_p_raw", "market_implied_p_devig", "model_market_gap", "model_market_gap_flag",
    "live_underdog_upscale_guard_triggered", "live_shrink_triggered",
    "live_oos_proxy_ready", "live_oos_proxy_train_rows", "live_oos_proxy_bin_n",
    "live_oos_proxy_bin_winrate", "blocked_by", HOMEWR_COL, "EV_€_per_100",
]


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




def _normalize_output_column_name(col: str) -> str:
    normalized = str(col).strip().lower().replace("\n", " ")
    normalized = "_".join(normalized.split())
    aliases = {
        "odds1": "odds_1",
        "odds2": "odds_2",
    }
    return aliases.get(normalized, normalized)


def canonicalize_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_normalize_output_column_name(c) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="last")]

    for col in OUTPUT_BASE_COLUMNS + OUTPUT_PROBABILITY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    ordered = OUTPUT_BASE_COLUMNS + OUTPUT_PROBABILITY_COLUMNS
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def ensure_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee probability chain columns exist as numeric values (NaN when unavailable)."""
    out = df.copy()
    fallback_map = {
        "prob_iso": [ISO_COL, "home_team_prob", PRED_PROBA_COL],
        "prob_iso_insample": ["prob_iso", ISO_COL],
        "prob_iso_oos_time": ["prob_iso"],
        "prob_live_oos_proxy": ["prob_iso_oos_time", "prob_iso"],
        "prob_live_safe_pre_clip": ["prob_base", "prob_live_oos_proxy", "prob_iso_oos_time", "prob_iso"],
        "prob_base": ["prob_live_safe_pre_clip", "prob_live_oos_proxy", "prob_iso_oos_time", "prob_iso"],
        "prob_live_safe": ["prob_base", "prob_live_safe_pre_clip", "prob_iso"],
        "prob_used": ["prob_live_safe", "prob_base", "prob_iso"],
    }

    for col, fallbacks in fallback_map.items():
        if col not in out.columns:
            out[col] = np.nan
        series = pd.to_numeric(out[col], errors="coerce")
        for source_col in fallbacks:
            if source_col in out.columns:
                series = series.fillna(pd.to_numeric(out[source_col], errors="coerce"))
        out[col] = series

    return out


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
    out[col] = parse_mixed_datetime(out[col])
    return out


def parse_mixed_datetime(values):
    """Robust parser for mixed date-only and timestamp strings."""
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def _canonical_game_key(df: pd.DataFrame) -> pd.Series:
    date_source = "date" if "date" in df.columns else DATE_COL
    date_part = parse_mixed_datetime(df.get(date_source)).dt.strftime("%Y-%m-%d").fillna("")
    home_part = df.get("home_team", pd.Series(index=df.index, dtype=object)).astype(str).str.strip().str.upper()
    away_part = df.get("away_team", pd.Series(index=df.index, dtype=object)).astype(str).str.strip().str.upper()
    return date_part + "|" + home_part + "|" + away_part


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

def _read_combined_snapshot(path: Path) -> pd.DataFrame:
    logging.info("Loading combined predictions snapshot: %s", path)

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

    snap_date = _extract_date_from_filename(path.name, "combined_nba_predictions_acc_")
    df["source_snapshot_date"] = snap_date
    return df


def _normalize_combined_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "result" in out.columns:
        out[RESULT_RAW_COL] = out["result"]
    else:
        out[RESULT_RAW_COL] = np.nan

    if DATE_COL not in out.columns:
        if "date" in out.columns:
            out[DATE_COL] = parse_mixed_datetime(out["date"])
        else:
            out[DATE_COL] = pd.NaT
    out[DATE_COL] = parse_mixed_datetime(out[DATE_COL])

    if PRED_PROBA_COL not in out.columns:
        if "home_team_prob" in out.columns:
            out[PRED_PROBA_COL] = to_float_series(out["home_team_prob"])
        else:
            out[PRED_PROBA_COL] = np.nan

    if HOME_ODDS_COL not in out.columns:
        if "odds_1" in out.columns:
            out[HOME_ODDS_COL] = to_float_series(out["odds_1"])
        else:
            out[HOME_ODDS_COL] = np.nan

    if AWAY_ODDS_COL not in out.columns:
        if "odds_2" in out.columns:
            out[AWAY_ODDS_COL] = to_float_series(out["odds_2"])
        else:
            out[AWAY_ODDS_COL] = np.nan

    if RESULT_COL not in out.columns:
        out[RESULT_COL] = np.nan
    if "home_team" in out.columns and "result" in out.columns:
        mask = (
            out["result"].notna()
            & (out["result"].astype(str) != "0")
            & out[RESULT_COL].isna()
        )
        out.loc[mask, RESULT_COL] = (
            out.loc[mask, "result"].astype(str) == out.loc[mask, "home_team"].astype(str)
        ).astype(int)

    return out


def load_combined_df(pred_dir: str, ymd: str) -> pd.DataFrame:
    pattern = "combined_nba_predictions_acc_*.csv"
    snapshots: list[tuple[datetime, Path]] = []
    for path in Path(pred_dir).glob(pattern):
        snap_date = _extract_date_from_filename(path.name, "combined_nba_predictions_acc_")
        if not snap_date:
            continue
        dt = datetime.strptime(snap_date, "%Y-%m-%d")
        if dt <= datetime.strptime(ymd, "%Y-%m-%d"):
            snapshots.append((dt, path))

    if not snapshots:
        raise FileNotFoundError(f"No combined predictions file found at or before {ymd} in {pred_dir}")

    snapshots.sort(key=lambda x: x[0])
    frames = [_read_combined_snapshot(path) for _, path in snapshots]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = _normalize_combined_schema(df)

    if {"home_team", "away_team", DATE_COL, "source_snapshot_date"}.issubset(df.columns):
        df["_snap_dt"] = parse_mixed_datetime(df["source_snapshot_date"])
        df["_game_key_norm"] = _canonical_game_key(df)
        df = df.sort_values(["_snap_dt", DATE_COL]).drop_duplicates(subset=["_game_key_norm"], keep="last")
        df = df.drop(columns=["_snap_dt", "_game_key_norm"])
        logging.info(
            "Rebuilt cumulative combined file from %d snapshots up to %s -> %d unique games.",
            len(snapshots),
            ymd,
            int(len(df)),
        )

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
        tmp["date"] = parse_mixed_datetime(tmp["date"])
    else:
        tmp["date"] = pd.NaT
    tmp.loc[tmp["date"].isna(), "date"] = pd.Timestamp(today_date)

    if "result" not in tmp.columns:
        tmp["result"] = np.nan

    tmp[DATE_COL] = parse_mixed_datetime(tmp["date"])
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
    if {"home_team", "away_team"}.issubset(df_all.columns):
        df_all["_game_key_norm"] = _canonical_game_key(df_all)
        if DATE_COL in df_all.columns:
            df_all = df_all.sort_values(DATE_COL)
        df_all = df_all.drop_duplicates(subset=["_game_key_norm"], keep="last")
        df_all = df_all.drop(columns=["_game_key_norm"], errors="ignore")
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

    base_df = df.drop(columns=[HOMEWR_COL], errors="ignore").copy()
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
    base_df["_home_team_norm"] = base_df["home_team"].astype(str).str.strip().map(normalize_team_code)

    hwr_m = hwr[["_team_norm", win_col]].drop_duplicates("_team_norm")
    out = base_df.merge(hwr_m, left_on="_home_team_norm", right_on="_team_norm", how="left")

    out.rename(columns={win_col: HOMEWR_COL}, inplace=True)
    out[HOMEWR_COL] = pd.to_numeric(out[HOMEWR_COL], errors="coerce").fillna(0.0)

    out.drop(columns=["_team_norm", "_home_team_norm"], inplace=True, errors="ignore")
    return out


def split_past_future(df_all: pd.DataFrame, today_date, tomorrow_date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_all.copy()
    df[DATE_COL] = parse_mixed_datetime(df[DATE_COL])
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

def fit_isotonic(df_past: pd.DataFrame) -> Optional[IsotonicRegression]:
    m = df_past[RESULT_COL].notna() & df_past[PRED_PROBA_COL].notna()
    if m.sum() == 0:
        logging.warning("No valid rows for isotonic fit; using base probabilities.")
        return None
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



def build_live_probability_columns(df_all: pd.DataFrame, today_date, tomorrow_date) -> tuple[pd.DataFrame, dict]:
    df = prepare_live_probability_columns(
        df_all,
        clip_lo=PROB_CLIP_LO,
        clip_hi=PROB_CLIP_HI,
        config=build_probability_chain_config(
            date_col=DATE_COL,
            result_col=RESULT_COL,
            result_raw_col=RESULT_RAW_COL,
            pred_proba_col=PRED_PROBA_COL,
            prob_iso_oos_time_col=PROB_ISO_OOS_TIME_COL,
            min_train_oos_time=MIN_TRAIN_OOS_TIME,
            min_step_oos_time=MIN_STEP_OOS_TIME,
            min_train_oos_proxy=MIN_TRAIN_OOS_PROXY,
            today_date=today_date,
            tomorrow_date=tomorrow_date,
            compute_oos_chain=True,
        ),
    )
    meta = {
        "live_oos_proxy_ready": bool(pd.Series(df.get("live_oos_proxy_ready", False)).fillna(False).astype(bool).any()),
        "live_oos_proxy_train_rows": int(pd.to_numeric(df.get("live_oos_proxy_train_rows", 0), errors="coerce").fillna(0).max() if isinstance(df.get("live_oos_proxy_train_rows", 0), pd.Series) else df.get("live_oos_proxy_train_rows", 0)),
    }
    return df, meta
def run_self_test(df_all: pd.DataFrame, live_meta: dict | None = None) -> None:
    required_cols = [
        "home_team_prob", "prob_iso", PROB_ISO_OOS_TIME_COL, PROB_LIVE_OOS_PROXY_COL,
        "prob_live_safe_pre_clip", "prob_base", "prob_used", "market_implied_p_raw",
        "market_implied_p_devig", "model_market_gap", "model_market_gap_flag",
        "live_underdog_upscale_guard_triggered", "live_shrink_triggered", "live_oos_proxy_ready",
        "live_oos_proxy_train_rows", "live_oos_proxy_bin_n", "live_oos_proxy_bin_winrate", "blocked_by",
    ]
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")

    played_mask = df_all[RESULT_RAW_COL].notna() & (df_all[RESULT_RAW_COL].astype(str) != "0")
    played_sorted = df_all.loc[played_mask].sort_values(DATE_COL)
    if len(played_sorted) > MIN_TRAIN_OOS_TIME + MIN_STEP_OOS_TIME:
        recent = played_sorted.tail(min(120, len(played_sorted)))
        ratio = recent[PROB_ISO_OOS_TIME_COL].notna().mean()
        ratio_prob_used = recent["prob_used"].notna().mean() if "prob_used" in recent.columns else 0.0
        if ratio < 0.5:
            logging.warning(
                "probability coverage sparse on recent played rows (prob_iso_oos_time=%.3f, prob_used=%.3f).",
                ratio,
                ratio_prob_used,
            )

    if live_meta is not None:
        train_rows = int(live_meta.get("live_oos_proxy_train_rows", 0))
        ready = bool(live_meta.get("live_oos_proxy_ready", False))
        if train_rows >= MIN_TRAIN_OOS_PROXY and not ready:
            raise AssertionError("Expected live_oos_proxy_ready=True when train_rows is sufficient")

    suspicious = df_all[(pd.to_numeric(df_all.get("odds_1"), errors="coerce") >= 2.30) & (pd.to_numeric(df_all.get("prob_live_safe_pre_clip"), errors="coerce") >= 0.60)]
    if not suspicious.empty:
        if not suspicious["model_market_gap_flag"].fillna(False).any():
            raise AssertionError("Expected underdog/high-prob rows to trigger model_market_gap_flag")
        reduced = suspicious["prob_used"] <= 0.55
        if not reduced.any():
            raise AssertionError("Expected underdog/high-prob rows to cap/shrink prob_used")


def classify_local_search_history(hist_rows: int, min_required: int = MIN_HIST_ROWS_FOR_LOCAL) -> str:
    return "insufficient_history" if int(hist_rows) < int(min_required) else "ok"

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

    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=PROB_COL_HIST, dst="prob_used")
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
    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=PROB_COL_HIST, dst="prob_used")
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


def _iter_param_grid(prob_clip_lo: float):
    prob_grid_eff = [p for p in PROB_MIN_GRID if p >= prob_clip_lo] or [prob_clip_lo]
    for hwr in HOMEWR_MIN_GRID:
        for o_min in ODDS_MIN_GRID:
            for o_max in ODDS_MAX_GRID:
                if o_max <= o_min:
                    continue
                for pmin in prob_grid_eff:
                    yield {
                        "home_win_rate_threshold": round(float(hwr), 2),
                        "odds_min": round(float(o_min), 2),
                        "odds_max": round(float(o_max), 2),
                        "prob_threshold": round(float(pmin), 2),
                    }


def _evaluate_params_on_subset(
    df: pd.DataFrame,
    params: dict,
    *,
    min_ev: float,
    stake: float,
    prob_clip_lo: float,
):
    if df.empty:
        return 0, 0.0, 0.0

    prob_thr_eff = max(float(params["prob_threshold"]), float(prob_clip_lo))
    mask = (
        (df[HOMEWR_COL] >= float(params["home_win_rate_threshold"])) &
        (df[HOME_ODDS_COL] >= float(params["odds_min"])) &
        (df[HOME_ODDS_COL] <= float(params["odds_max"])) &
        (df["prob_used"] >= prob_thr_eff) &
        (df["EV_€_per_100"] > float(min_ev))
    )
    sub = df.loc[mask]
    if sub.empty:
        return 0, 0.0, 0.0
    pnl = np.where(
        sub[RESULT_COL].astype(int) == 1,
        float(stake) * (sub[HOME_ODDS_COL] - 1.0),
        -float(stake),
    )
    profit = float(np.sum(pnl))
    trades = int(len(sub))
    roi = (profit / (trades * float(stake)) * 100.0) if trades else 0.0
    return trades, profit, roi


def find_best_local_params_walk_forward(
    hist_df: pd.DataFrame,
    *,
    tail_n: int,
    min_ev: float,
    stake: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
    n_splits: int = 4,
    min_trades_test_split: int = 10,
):
    if hist_df is None or hist_df.empty:
        return None

    df = _ensure_datetime(hist_df, DATE_COL).sort_values(DATE_COL).tail(int(tail_n)).copy()
    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=PROB_COL_HIST, dst="prob_used")
    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100")
    df = df.dropna(subset=[HOMEWR_COL, "prob_used", HOME_ODDS_COL, RESULT_COL, "EV_€_per_100"]).copy()
    if len(df) < max(80, n_splits * 20):
        return None

    fold_size = len(df) // n_splits
    if fold_size <= 0:
        return None

    best = None
    for params in _iter_param_grid(prob_clip_lo):
        test_rois = []
        test_trades = []
        wf_test_profit_total = 0.0

        for split_idx in range(n_splits):
            start = split_idx * fold_size
            end = len(df) if split_idx == n_splits - 1 else (split_idx + 1) * fold_size
            test_df = df.iloc[start:end]
            trades, profit, roi = _evaluate_params_on_subset(
                test_df,
                params,
                min_ev=min_ev,
                stake=stake,
                prob_clip_lo=prob_clip_lo,
            )
            test_rois.append(float(roi))
            test_trades.append(int(trades))
            wf_test_profit_total += float(profit)

        active_splits = int(sum(t >= min_trades_test_split for t in test_trades))
        total_test_trades = int(sum(test_trades))
        q20_trades = float(np.quantile(test_trades, 0.2)) if test_trades else 0.0

        full_trades, full_profit, full_roi = _evaluate_params_on_subset(
            df,
            params,
            min_ev=min_ev,
            stake=stake,
            prob_clip_lo=prob_clip_lo,
        )

        mean_roi = float(np.mean(test_rois)) if test_rois else 0.0
        std_roi = float(np.std(test_rois)) if test_rois else 0.0
        score = mean_roi - 0.5 * std_roi

        candidate = {
            "score_mode": "lcb_roi",
            "score": float(score),
            "params": params,
            "tail_n": int(tail_n),
            "splits_used": int(n_splits),
            "test_trades_total": int(total_test_trades),
            "active_splits": int(active_splits),
            "q20_trades": float(q20_trades),
            "wf_test_profit_total": float(wf_test_profit_total),
            "full_profit": float(full_profit),
            "full_roi": float(full_roi),
            "full_trades": int(full_trades),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


# -----------------------------
# EXPORT HELPERS
# -----------------------------

def prepare_local_matched_export(matched_subset: pd.DataFrame, stake: float) -> pd.DataFrame:
    df = matched_subset.copy()
    if df.empty:
        logging.info("local_matched export is empty before normalization; writing schema-only export.")
        return pd.DataFrame(columns=LOCAL_MATCHED_EXPORT_COLUMNS)

    row_count_before = int(len(df))
    date_source_col = next((c for c in DATE_SOURCE_CANDIDATES if c in df.columns), None)
    if date_source_col is None:
        raise RuntimeError(
            "local_matched_games dated export missing required date column "
            f"(checked candidates={DATE_SOURCE_CANDIDATES})"
        )
    parsed_dates = pd.to_datetime(df[date_source_col], errors="coerce")
    valid_date_mask = parsed_dates.notna()
    invalid_count = int((~valid_date_mask).sum())
    if invalid_count > 0:
        logging.warning(
            "Dropping %d local_matched rows with invalid %s values during date normalization.",
            invalid_count,
            date_source_col,
        )
    df = df.loc[valid_date_mask].copy()
    if df.empty:
        raise RuntimeError(
            "local_matched_games dated export has rows but 0 valid date rows after normalization "
            f"(source={date_source_col})"
        )
    df["date"] = parsed_dates.loc[valid_date_mask].dt.strftime("%Y-%m-%d")

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

    for c in LOCAL_MATCHED_EXPORT_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    out = df[LOCAL_MATCHED_EXPORT_COLUMNS].copy()

    # IMPORTANT: prevent duplicates (your current latest CSV has duplicates)
    out = out.sort_values(["date", "home_team", "away_team", "EV_€_per_100"], ascending=[True, True, True, False])
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

    out = out.sort_values("date").reset_index(drop=True)
    logging.info(
        "local_matched export prepared: before_normalization=%d after_normalization=%d date_source=%s",
        row_count_before,
        int(len(out)),
        date_source_col,
    )
    return out


def evaluate_strategy_stability(
    hist_df: pd.DataFrame,
    params: dict,
    *,
    windows: list[int],
    min_ev: float,
    stake: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
    min_trades_per_window: int,
) -> dict:
    if hist_df is None or hist_df.empty:
        return {"hits": 0, "rows_eval": 0, "details": []}

    df_base = _ensure_datetime(hist_df, DATE_COL).sort_values(DATE_COL).copy()
    details = []
    hits = 0
    for w in windows:
        window_df = df_base.tail(int(w)).copy()
        window_df = _compute_prob_used(window_df, lo=prob_clip_lo, hi=prob_clip_hi, src=PROB_COL_HIST, dst="prob_used")
        window_df = _compute_ev_per_100(window_df, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100")
        window_df = window_df.dropna(subset=[HOMEWR_COL, "prob_used", HOME_ODDS_COL, RESULT_COL, "EV_€_per_100"])
        trades, profit, roi = _evaluate_params_on_subset(
            window_df,
            params,
            min_ev=min_ev,
            stake=stake,
            prob_clip_lo=prob_clip_lo,
        )
        passed = bool(trades >= int(min_trades_per_window) and profit > 0.0)
        hits += int(passed)
        details.append(
            {
                "window": int(w),
                "rows": int(len(window_df)),
                "trades": int(trades),
                "profit": float(profit),
                "roi": float(roi),
                "pass": passed,
            }
        )
    return {"hits": int(hits), "rows_eval": int(len(df_base)), "details": details}


def log_local_matched_write_intent(*, dest_path: Path, source_name: str, df: pd.DataFrame) -> None:
    row_count = int(len(df))
    if row_count <= 0:
        logging.warning(
            "[LOCAL MATCHED EXPORT] writing header-only file -> %s | source=%s rows=0",
            dest_path,
            source_name,
        )
        return

    parsed_dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
    if parsed_dates.notna().any():
        min_date = parsed_dates.min().strftime("%Y-%m-%d")
        max_date = parsed_dates.max().strftime("%Y-%m-%d")
    else:
        min_date = "NA"
        max_date = "NA"
    logging.info(
        "[LOCAL MATCHED EXPORT] writing file -> %s | source=%s rows=%d min_date=%s max_date=%s",
        dest_path,
        source_name,
        row_count,
        min_date,
        max_date,
    )


def summarize_local_matched_df(df: pd.DataFrame) -> dict:
    row_count = int(len(df))
    unique_games = 0
    if row_count > 0 and {"date", "home_team", "away_team"}.issubset(df.columns):
        unique_games = int(df[["date", "home_team", "away_team"]].drop_duplicates().shape[0])
    parsed_dates = pd.to_datetime(df["date"], errors="coerce") if ("date" in df.columns and row_count > 0) else pd.Series(dtype="datetime64[ns]")
    if parsed_dates.notna().any():
        min_date = parsed_dates.min().strftime("%Y-%m-%d")
        max_date = parsed_dates.max().strftime("%Y-%m-%d")
    else:
        min_date = None
        max_date = None
    return {
        "rows": row_count,
        "unique_games": unique_games,
        "min_date": min_date,
        "max_date": max_date,
    }


def write_csv_with_audit(df: pd.DataFrame, dest_path: Path, *, source_name: str, allow_header_only: bool) -> tuple[bool, bool, str]:
    """
    Returns (written, unchanged_content, action) where action is one of
    {'rewritten', 'unchanged'}.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and not allow_header_only:
        raise RuntimeError(f"Refusing to write header-only local_matched file without explicit allow_header_only=True: {dest_path}")

    stats = summarize_local_matched_df(df)
    logging.info(
        "[LOCAL MATCHED EXPORT] write audit | source=%s rows=%d unique_games=%d min_date=%s max_date=%s path=%s",
        source_name,
        stats["rows"],
        stats["unique_games"],
        stats["min_date"] or "NA",
        stats["max_date"] or "NA",
        dest_path,
    )
    if df.empty:
        logging.warning("[LOCAL MATCHED EXPORT] header-only file is about to be written -> %s", dest_path)

    existing_content = dest_path.read_text(encoding="utf-8") if dest_path.exists() else None
    new_content = df.to_csv(index=False, encoding="utf-8")
    unchanged = existing_content is not None and existing_content == new_content
    if unchanged:
        return True, True, "unchanged"
    dest_path.write_text(new_content, encoding="utf-8")
    return True, False, "rewritten"


def write_local_matched_export_report(
    *,
    output_dir: Path,
    as_of_date: str,
    params_used: dict,
    selected_source: str,
    source_rows: int,
    rows_exported: int,
    date_min: str,
    date_max: str,
    max_date_used_for_filename: str,
    dated_path: Path,
    latest_path: Path,
    dated_status: str,
    latest_status: str,
    reason_if_nothing_written: str,
) -> Path:
    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = output_dir / f"local_matched_games_export_report_{run_date}.txt"
    lines = [
        f"timestamp_utc={datetime.utcnow().isoformat()}Z",
        f"as_of_date={as_of_date}",
        f"params_used={json.dumps(params_used, sort_keys=True)}",
        f"selected_source={selected_source}",
        f"source_rows={source_rows}",
        f"rows_exported={rows_exported}",
        f"date_range={date_min}..{date_max}",
        f"max_date_used_for_filename={max_date_used_for_filename}",
        f"dated_output={dated_path}",
        f"latest_output={latest_path}",
        f"dated_status={dated_status}",
        f"latest_status={latest_status}",
        f"reason_if_nothing_written={reason_if_nothing_written}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("[LOCAL MATCHED EXPORT] report=%s", report_path)
    return report_path


def rebuild_historical_subset_from_hist_df(
    hist_df: pd.DataFrame,
    *,
    params_used: dict,
    window_n: int,
    min_ev: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
) -> pd.DataFrame:
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()
    hist_window = _ensure_datetime(hist_df.copy(), DATE_COL).sort_values(DATE_COL).tail(int(window_n)).copy()
    _, rebuilt_subset = evaluate_params_on_hist_window(
        hist_window,
        params_used,
        min_ev=min_ev,
        flat_stake_backtest=FLAT_STAKE,
        prob_clip_lo=prob_clip_lo,
        prob_clip_hi=prob_clip_hi,
    )
    return rebuilt_subset.copy() if rebuilt_subset is not None else pd.DataFrame()


def resolve_historical_subset_for_local_matched_export(
    *,
    subset_local: Optional[pd.DataFrame],
    hist_df: pd.DataFrame,
    hist_window_200: pd.DataFrame,
    params_used: dict,
    min_ev: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
) -> tuple[pd.DataFrame, str, bool]:
    """
    Resolve one canonical dataframe for local_matched export (historical strategy subset only).
    Returns (subset_df, source_name, fallback_used).
    """
    source_name = "historical_strategy_subset"
    fallback_used = False
    resolved_subset = subset_local.copy() if subset_local is not None else pd.DataFrame()

    if resolved_subset.empty:
        fallback_subset = rebuild_historical_subset_from_hist_df(
            hist_df,
            params_used=params_used,
            window_n=FAIR_COMPARE_N,
            min_ev=min_ev,
            prob_clip_lo=prob_clip_lo,
            prob_clip_hi=prob_clip_hi,
        )
        if not fallback_subset.empty:
            resolved_subset = fallback_subset.copy()
            source_name = "hist_df+params_used"
            fallback_used = True
            logging.info("[LOCAL_MATCHED] fallback reconstruction used from hist_df + params_used")

    if resolved_subset.empty and not hist_window_200.empty:
        _, fallback_subset_window = evaluate_params_on_hist_window(
            hist_window_200,
            params_used,
            min_ev=min_ev,
            flat_stake_backtest=FLAT_STAKE,
            prob_clip_lo=prob_clip_lo,
            prob_clip_hi=prob_clip_hi,
        )
        if fallback_subset_window is not None and not fallback_subset_window.empty:
            resolved_subset = fallback_subset_window.copy()
            source_name = "hist_window_200+params_used"
            fallback_used = True
            logging.info("[LOCAL_MATCHED] fallback reconstruction used from hist_window_200 + params_used")

    summary = summarize_local_matched_df(prepare_local_matched_export(resolved_subset, stake=FLAT_STAKE) if not resolved_subset.empty else pd.DataFrame(columns=LOCAL_MATCHED_EXPORT_COLUMNS))
    if summary["rows"] > 0:
        logging.info(
            "[LOCAL_MATCHED] source=%s rows=%d unique_games=%d date_range=%s..%s fallback_used=%s",
            source_name,
            summary["rows"],
            summary["unique_games"],
            summary["min_date"],
            summary["max_date"],
            str(fallback_used).lower(),
        )
    else:
        logging.warning(
            "[LOCAL_MATCHED][WARN] export is genuinely empty after fallback | source=%s fallback_used=%s",
            source_name,
            str(fallback_used).lower(),
        )
    return resolved_subset, source_name, fallback_used


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
    historical_subset: Optional[pd.DataFrame] = None,
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

        cols = LOCAL_MATCHED_EXPORT_COLUMNS

        source_name = "historical_subset"
        if historical_subset is not None:
            subset = historical_subset.copy()
            cutoff_date = None
            logging.info(
                "[LOCAL MATCHED EXPORT] web latest source=historical_subset rows=%d",
                int(len(subset)),
            )
            if subset.empty:
                subset = rebuild_historical_subset_from_hist_df(
                    df_past_sorted,
                    params_used=params_used,
                    window_n=window_n,
                    min_ev=min_ev,
                    prob_clip_lo=prob_clip_lo,
                    prob_clip_hi=prob_clip_hi,
                )
                source_name = "hist_df+params_used"
                logging.info(
                    "[LOCAL MATCHED EXPORT] web latest fallback source=hist_df+params_used rows=%d",
                    int(len(subset)),
                )
        else:
            if df_past_sorted is None or df_past_sorted.empty:
                empty_df = pd.DataFrame(columns=cols)
                written, unchanged, _ = write_csv_with_audit(
                    empty_df,
                    out_path,
                    source_name="df_past_sorted(empty)",
                    allow_header_only=True,
                )
                logging.warning("local_matched_games export is genuinely empty for current strategy/window.")
                logging.info(
                    "[LOCAL MATCHED EXPORT] latest alias written=%s unchanged_content=%s path=%s",
                    str(written).lower(),
                    str(unchanged).lower(),
                    out_path,
                )
                return

            # --- cutoff to yesterday ---
            df = df_past_sorted.copy()
            df = _ensure_datetime(df, DATE_COL)
            
            max_played_ts = parse_mixed_datetime(df[DATE_COL]).max()
            if pd.isna(max_played_ts):
                empty_df = pd.DataFrame(columns=cols)
                written, unchanged, _ = write_csv_with_audit(
                    empty_df,
                    out_path,
                    source_name="df_past_sorted(no_played_dates)",
                    allow_header_only=True,
                )
                logging.warning("local_matched_games export is genuinely empty for current strategy/window.")
                logging.info(
                    "[LOCAL MATCHED EXPORT] latest alias written=%s unchanged_content=%s path=%s",
                    str(written).lower(),
                    str(unchanged).lower(),
                    out_path,
                )
                return
            
            cutoff_date = max_played_ts.date()
            df = df[df[DATE_COL].dt.date <= cutoff_date].copy()

            # last N played games inside cutoff
            df = df.sort_values(DATE_COL).tail(int(window_n)).copy()

            # compute prob_used + EV
            df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=PROB_COL_HIST, dst="prob_used")
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
            source_name = "df_past_sorted(last_window_filtered)"

        if subset.empty:
            empty_df = pd.DataFrame(columns=cols)
            written, unchanged, _ = write_csv_with_audit(
                empty_df,
                out_path,
                source_name=source_name,
                allow_header_only=True,
            )
            logging.warning("local_matched_games export is genuinely empty for current strategy/window.")
            logging.info(
                "[LOCAL MATCHED EXPORT] latest alias written=%s unchanged_content=%s path=%s",
                str(written).lower(),
                str(unchanged).lower(),
                out_path,
            )
            return

        export_df = prepare_local_matched_export(subset, stake=float(stake))
        for c in cols:
            if c not in export_df.columns:
                export_df[c] = np.nan

        written, unchanged, _ = write_csv_with_audit(
            export_df[cols],
            out_path,
            source_name=source_name,
            allow_header_only=True,
        )
        logging.info(
            "Wrote local_matched_games_latest.csv -> %s (%d rows) [window_n=%d cutoff<=%s]",
            out_path,
            len(export_df),
            int(window_n),
            str(cutoff_date) if cutoff_date is not None else "historical_subset",
        )
        logging.info(
            "[LOCAL MATCHED EXPORT] latest alias written=%s unchanged_content=%s path=%s",
            str(written).lower(),
            str(unchanged).lower(),
            out_path,
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


def validate_structured_csv(
    path: Path,
    required_cols: list[str],
    *,
    min_data_rows: int = 1,
    unique_key_cols: Optional[list[str]] = None,
) -> None:
    if not path.exists():
        raise RuntimeError(f"CSV validation failed: missing file {path}")

    line_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for _ in fh:
            line_count += 1
            if line_count >= (min_data_rows + 1):
                break
    if line_count < (min_data_rows + 1):
        raise RuntimeError(
            f"CSV validation failed: {path} has insufficient lines "
            f"(required >= {min_data_rows + 1}, found {line_count})"
        )

    sample = pd.read_csv(path, nrows=5)
    missing = [c for c in required_cols if c not in sample.columns]
    if missing:
        raise RuntimeError(f"CSV validation failed: {path} missing required columns {missing}")

    if unique_key_cols:
        key_missing = [c for c in unique_key_cols if c not in sample.columns]
        if key_missing:
            raise RuntimeError(
                f"CSV validation failed: {path} missing uniqueness key columns {key_missing}"
            )
        key_df = pd.read_csv(path, usecols=unique_key_cols)
        dup_count = int(key_df.duplicated(subset=unique_key_cols, keep=False).sum())
        if dup_count > 0:
            raise RuntimeError(
                f"CSV validation failed: {path} has duplicated key rows on "
                f"{unique_key_cols} (duplicate_rows={dup_count})"
            )


def write_strategy_params(params_used: dict, *, min_ev: float, as_of_date: str, stake: float, output_dir: Path) -> None:
    """
    Keep txt/json aliases for local tooling and persist dated strategy params
    for historical snapshot resolution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "strategy_params.txt"
    out_dated = output_dir / f"strategy_params_{as_of_date}.txt"
    out_json = output_dir / "strategy_params.json"
    out_json_dated = output_dir / f"strategy_params_{as_of_date}.json"
    lines = [f"as_of_date={as_of_date}", f"min_ev={float(min_ev)}", f"stake={float(stake)}"]
    for k in sorted(params_used.keys()):
        lines.append(f"{k}={params_used[k]}")
    txt_payload = "\n".join(lines) + "\n"
    out.write_text(txt_payload, encoding="utf-8")
    out_dated.write_text(txt_payload, encoding="utf-8")

    json_payload = {
        "as_of_date": as_of_date,
        "min_ev": float(min_ev),
        "stake": float(stake),
        "params_used": {k: params_used[k] for k in sorted(params_used.keys())},
        "source": "script5_local_last200",
    }
    json_text = json.dumps(json_payload, indent=2)
    out_json.write_text(json_text, encoding="utf-8")
    out_json_dated.write_text(json_text, encoding="utf-8")
    logging.info("Saved %s", out)
    logging.info("Saved %s", out_dated)
    logging.info("Saved %s", out_json)
    logging.info("Saved %s", out_json_dated)


def write_local_matched_artifacts(
    export_df: pd.DataFrame,
    *,
    as_of_date: str,
    output_dir: Path,
    params_used: dict,
    source_name: str,
    allow_header_only: bool,
    intentional_empty: bool,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_df = export_df.copy()
    for col in LOCAL_MATCHED_EXPORT_COLUMNS:
        if col not in normalized_df.columns:
            normalized_df[col] = np.nan
    normalized_df = normalized_df[LOCAL_MATCHED_EXPORT_COLUMNS].copy()

    row_count_before = int(len(normalized_df))
    date_source_col = next((c for c in DATE_SOURCE_CANDIDATES if c in normalized_df.columns), None)
    if row_count_before == 0:
        date_source_col = "date"
        if not intentional_empty:
            raise RuntimeError(
                "Refusing to write header-only local_matched artifacts without an explicit genuine-empty confirmation."
            )
        logging.warning(
            "[LOCAL_MATCHED][WARN] genuine empty export after fallback; writing intentional header-only dated artifact."
        )
    else:
        if date_source_col is None:
            raise RuntimeError("local_matched_games dated export missing required date column")
        parsed_dates = parse_mixed_datetime(normalized_df[date_source_col])
        valid_mask = parsed_dates.notna()
        dropped = int((~valid_mask).sum())
        if dropped:
            logging.warning(
                "Dropping %d invalid local_matched rows before writing artifacts (date source=%s).",
                dropped,
                date_source_col,
            )
        normalized_df = normalized_df.loc[valid_mask].copy()
        if normalized_df.empty:
            raise RuntimeError(
                "local_matched_games dated export has rows but 0 valid date rows after normalization "
                f"(source={date_source_col})"
            )
        normalized_df["date"] = parsed_dates.loc[valid_mask].dt.strftime("%Y-%m-%d")
        if normalized_df["date"].isna().any():
            raise RuntimeError("local_matched_games dated export has null date values after normalization")

    resolved_local_matched_date = (
        str(parse_mixed_datetime(normalized_df["date"]).max().date())
        if not normalized_df.empty
        else as_of_date
    )
    dated_path = output_dir / f"local_matched_games_{resolved_local_matched_date}.csv"
    latest_path = output_dir / "local_matched_games_latest.csv"
    summary = summarize_local_matched_df(normalized_df)

    logging.info("local_matched export source dataframe: %s", source_name)
    logging.info("local_matched export output paths: dated=%s latest=%s", dated_path, latest_path)
    logging.info("local_matched export rows before normalization: %d", row_count_before)
    logging.info("local_matched export rows after normalization: %d", int(len(normalized_df)))
    logging.info("local_matched export writing empty valid export: %s", "yes" if normalized_df.empty else "no")
    logging.info("local_matched export date source column: %s", date_source_col)
    logging.info("local_matched export naming max date: %s", resolved_local_matched_date)
    logging.info(
        "local_matched export date range: %s..%s",
        summary["min_date"] or "NA",
        summary["max_date"] or "NA",
    )
    logging.info("local_matched export columns: %s", list(normalized_df.columns))
    logging.info("local_matched export head(3):\n%s", normalized_df.head(3).to_string(index=False))
    logging.info(
        "local_matched export non-null date=%d valid parsed=%d rows=%d",
        int(normalized_df["date"].notna().sum()),
        int(parse_mixed_datetime(normalized_df["date"]).notna().sum()),
        int(len(normalized_df)),
    )

    dated_written, dated_unchanged, dated_action = write_csv_with_audit(
        normalized_df,
        dated_path,
        source_name=source_name,
        allow_header_only=allow_header_only,
    )
    latest_written, latest_unchanged, latest_action = write_csv_with_audit(
        normalized_df,
        latest_path,
        source_name=f"{source_name} (latest_alias)",
        allow_header_only=allow_header_only,
    )
    if normalized_df.empty:
        dated_action = "skipped_empty_source"
        latest_action = "skipped_empty_source"
    logging.info(
        "[LOCAL MATCHED EXPORT] dated written=%s unchanged_content=%s action=%s path=%s",
        str(dated_written).lower(),
        str(dated_unchanged).lower(),
        dated_action,
        dated_path,
    )
    logging.info(
        "[LOCAL MATCHED EXPORT] latest alias written=%s unchanged_content=%s action=%s path=%s",
        str(latest_written).lower(),
        str(latest_unchanged).lower(),
        latest_action,
        latest_path,
    )
    if latest_unchanged:
        logging.info("[LOCAL_MATCHED] latest alias unchanged by content")
        logging.info("[LOCAL MATCHED EXPORT] export evaluated successfully, no content change")
    else:
        logging.info("[LOCAL_MATCHED] latest alias updated with new content")

    reason_if_nothing_written = "" if not normalized_df.empty else "skipped due to empty source dataframe"
    write_local_matched_export_report(
        output_dir=output_dir,
        as_of_date=as_of_date,
        params_used=params_used,
        selected_source=source_name,
        source_rows=row_count_before,
        rows_exported=int(len(normalized_df)),
        date_min=summary["min_date"] or "NA",
        date_max=summary["max_date"] or "NA",
        max_date_used_for_filename=resolved_local_matched_date,
        dated_path=dated_path,
        latest_path=latest_path,
        dated_status=dated_action,
        latest_status=latest_action,
        reason_if_nothing_written=reason_if_nothing_written,
    )

    logging.info("Saved %s (%d rows)", dated_path, len(normalized_df))
    logging.info("Updated %s -> mirror content of %s", latest_path, dated_path.name)
    return resolved_local_matched_date


def validate_dated_dashboard_artifacts(*, output_dir: Path, as_of_date: str, local_matched_date: str) -> None:
    matched_dated = output_dir / f"local_matched_games_{local_matched_date}.csv"
    matched_latest = output_dir / "local_matched_games_latest.csv"
    strategy_json_dated = output_dir / f"strategy_params_{as_of_date}.json"
    strategy_txt_dated = output_dir / f"strategy_params_{as_of_date}.txt"
    strategy_json_alias = output_dir / "strategy_params.json"
    strategy_txt_alias = output_dir / "strategy_params.txt"

    required = [matched_dated, matched_latest, strategy_json_dated, strategy_txt_dated, strategy_json_alias, strategy_txt_alias]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Validation failed. Missing dashboard artifacts: {missing}")

    strategy_payload = json.loads(strategy_json_dated.read_text(encoding="utf-8"))
    if str(strategy_payload.get("as_of_date")) != as_of_date:
        raise RuntimeError(
            "Validation failed. strategy_params dated JSON has mismatched as_of_date: "
            f"expected={as_of_date} got={strategy_payload.get('as_of_date')}"
        )
    if f"as_of_date={as_of_date}" not in strategy_txt_dated.read_text(encoding="utf-8"):
        raise RuntimeError("Validation failed. strategy_params dated TXT is missing matching as_of_date.")

    if matched_dated.read_text(encoding="utf-8") != matched_latest.read_text(encoding="utf-8"):
        raise RuntimeError("Validation failed. local_matched_games_latest.csv does not mirror dated local_matched file.")

    matched_df = pd.read_csv(matched_dated)
    if "date" not in matched_df.columns:
        raise RuntimeError("local_matched_games dated export missing required date column")
    if matched_df.empty:
        logging.info("Validated empty local_matched dated export with schema-only header.")
        logging.info("Validated dated dashboard artifacts for as_of_date=%s", as_of_date)
        logging.info("- %s", matched_dated)
        logging.info("- %s", strategy_json_dated)
        logging.info("- %s", strategy_txt_dated)
        return
    valid_dates = parse_mixed_datetime(matched_df["date"]).notna()
    valid_date_count = int(valid_dates.sum())
    if valid_date_count <= 0:
        raise RuntimeError("local_matched_games dated export has 0 valid date rows after normalization")
    if not valid_dates.all():
        raise RuntimeError("local_matched_games dated export includes invalid date rows after normalization")

    logging.info("Validated dated dashboard artifacts for as_of_date=%s", as_of_date)
    logging.info("- %s", matched_dated)
    logging.info("- %s", strategy_json_dated)
    logging.info("- %s", strategy_txt_dated)


# -----------------------------
# AS-OF DATE (latest played)
# -----------------------------



def build_bet_shortlist(df_all: pd.DataFrame, params: dict, min_ev: float) -> pd.DataFrame:
    out = df_all.copy()
    played = out[RESULT_RAW_COL].notna() & (out[RESULT_RAW_COL].astype(str) != "0")
    out = out[~played].copy()
    out = _compute_ev_per_100(out, prob_col="prob_used", odds_col=HOME_ODDS_COL, dst="EV_€_per_100")

    # Keep shortlist filtering aligned with historical evaluation logic:
    # effective probability threshold cannot be below the configured clipping floor.
    prob_thr_eff = max(float(params["prob_threshold"]), float(PROB_CLIP_LO))
    mask = (
        (out[HOMEWR_COL] >= float(params["home_win_rate_threshold"])) &
        (out[HOME_ODDS_COL] >= float(params["odds_min"])) &
        (out[HOME_ODDS_COL] <= float(params["odds_max"])) &
        (out["prob_used"] >= prob_thr_eff) &
        (out["EV_€_per_100"] > float(min_ev))
    )
    shortlist = out.loc[mask].copy()
    if "blocked_by" in shortlist.columns:
        shortlist = shortlist[shortlist["blocked_by"].fillna("PASS").eq("PASS")].copy()
    for c in SHORTLIST_COLUMNS:
        if c not in shortlist.columns:
            shortlist[c] = np.nan
    return shortlist[SHORTLIST_COLUMNS]
def resolve_as_of_date_from_df_past(df_past_sorted: pd.DataFrame, fallback: str) -> str:
    if df_past_sorted is None or df_past_sorted.empty:
        return fallback
    dt = parse_mixed_datetime(df_past_sorted[DATE_COL])
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

    requested_dt = target_dt
    requested_ymd = target_ymd
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

    source_ymd = combined_date
    if combined_date != requested_ymd:
        logging.info(
            "Using latest combined source date %s for requested run date %s",
            combined_date,
            requested_ymd,
        )

    pred_dir = str(combined_path.parent)
    out_dir = resolve_output_dir(paths["BASE_DIR"], pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kelly_dir = Path(pred_dir) / "Kelly"
    kelly_dir.mkdir(parents=True, exist_ok=True)

    params_source_path = resolve_params_source(out_dir, strategy_variant)
    params_source = str(params_source_path)

    # 1) LOAD COMBINED
    df_all = load_combined_df(pred_dir, source_ymd)
    df_all = _ensure_datetime(df_all, DATE_COL)

    # 1b) HOME WIN RATES
    hwr_path = compute_home_win_rates(df_all, requested_ymd, pred_dir)

    # 2) MERGE TODAY PREDICTIONS (optional)
    today_pred_path, today_pred_date = resolve_dated_file(
        pred_dirs,
        "nba_games_predict_",
        requested_ymd,
        latest_on_or_before=requested_dt,
    )
    pred_date = datetime.strptime(today_pred_date, "%Y-%m-%d").date() if today_pred_date else requested_dt.date()
    df_all = merge_today_predictions(df_all, today_pred_path, pred_date)

    # 3) ATTACH HOME WIN RATE
    df_all = attach_home_win_rate(df_all, hwr_path)

    # 4) SPLIT PAST / FUTURE
    df_past, df_future = split_past_future(df_all, today_date, tomorrow_date)
    if df_past.empty:
        logging.warning("No past games available – isotonic fit skipped; falling back to base probabilities.")

    # 5) FIT ISOTONIC + APPLY (in-sample reference only)
    iso = fit_isotonic(df_past) if not df_past.empty else None
    df_all[ISO_COL] = np.nan
    m_iso = df_all[PRED_PROBA_COL].notna()
    if iso is not None:
        df_all.loc[m_iso, ISO_COL] = iso.transform(df_all.loc[m_iso, PRED_PROBA_COL].astype(float).values)
    else:
        df_all.loc[m_iso, ISO_COL] = pd.to_numeric(df_all.loc[m_iso, PRED_PROBA_COL], errors="coerce")
    df_all["prob_iso"] = df_all[ISO_COL]
    df_all["home_team_prob"] = pd.to_numeric(df_all[PRED_PROBA_COL], errors="coerce")
    if "odds_1" not in df_all.columns and HOME_ODDS_COL in df_all.columns:
        df_all["odds_1"] = pd.to_numeric(df_all[HOME_ODDS_COL], errors="coerce")

    # 5b) Build time-OOS calibration + live OOS proxy + live safety
    df_all, live_meta = build_live_probability_columns(df_all, today_date, tomorrow_date)
    df_all = ensure_probability_columns(df_all)
    logging.info("Probability columns: PROB_COL_HIST=%s PROB_COL_LIVE=%s", PROB_COL_HIST, PROB_COL_LIVE)

    # refresh past/future views with calibrated columns
    df_past, df_future = split_past_future(df_all, today_date, tomorrow_date)

    # sort past chronologically
    df_past_sorted = _ensure_datetime(df_past, DATE_COL).sort_values(DATE_COL).copy()

    # AS OF DATE = max played date (yesterday in practice)
    as_of_date = resolve_as_of_date_from_df_past(df_past_sorted, fallback=target_ymd)

    # 6) CALIBRATION METRICS (window overall)
    b0, b1, ll0, ll1 = compute_calibration_metrics(df_past)
    logging.info("Brier before=%.6f after=%.6f | LogLoss before=%.6f after=%.6f", b0, b1, ll0, ll1)

    # 7) SAVE ISO COMBINED (unchanged)
    iso_path = kelly_dir / f"combined_nba_predictions_iso_{requested_ymd}.csv"
    df_all = canonicalize_output_dataframe(df_all)
    df_all = ensure_probability_columns(df_all)
    df_all.to_csv(iso_path, index=False, encoding="utf-8", lineterminator="\n")
    logging.info("Saved ISO combined -> %s", iso_path)
    validate_structured_csv(
        iso_path,
        required_cols=["home_team", "away_team", "date", "prob_used"],
        min_data_rows=1,
        unique_key_cols=["date", "home_team", "away_team"],
    )

    # Keep ACC file schema aligned with enriched probabilities for downstream consumers.
    acc_path = Path(pred_dir) / f"combined_nba_predictions_acc_{requested_ymd}.csv"
    df_all.to_csv(acc_path, index=False, encoding="utf-8", lineterminator="\n")
    logging.info("Refreshed ACC combined with calibrated probabilities -> %s", acc_path)
    validate_structured_csv(
        acc_path,
        required_cols=["home_team", "away_team", "date", "prob_used"],
        min_data_rows=1,
        unique_key_cols=["date", "home_team", "away_team"],
    )

    combined_source_path = iso_path if strategy_variant == "iso" else combined_path
    snapshot_combined_source_path = combined_source_path
    if as_of_date != requested_ymd:
        snapshot_path = (
            kelly_dir / f"combined_nba_predictions_iso_{as_of_date}.csv"
            if strategy_variant == "iso"
            else Path(pred_dir) / f"combined_nba_predictions_acc_{as_of_date}.csv"
        )
        if not snapshot_path.exists():
            df_all.to_csv(snapshot_path, index=False, encoding="utf-8", lineterminator="\n")
            logging.info(
                "Wrote as-of aligned combined snapshot for metrics date %s -> %s",
                as_of_date,
                snapshot_path,
            )
        snapshot_combined_source_path = snapshot_path

    # ------------------------------------------------------------------
    # CORE: robust LOCAL params selection (walk-forward + coverage/stability)
    # ------------------------------------------------------------------
    min_EV = MIN_EV_DEFAULT
    logging.info("Min EV applied = %s", int(min_EV) if float(min_EV).is_integer() else min_EV)
    logging.info("Historical prob column: %s | Live prob column: %s", PROB_COL_HIST, PROB_COL_LIVE)
    logging.info("[LOCAL] Using df_all played base for ladder (rows=%d)", int(len(df_past_sorted)))
    hist_window_200 = df_past_sorted.tail(int(FAIR_COMPARE_N)).copy()
    hist_rows = int(len(df_past_sorted))
    history_status = classify_local_search_history(hist_rows, MIN_HIST_ROWS_FOR_LOCAL)
    insufficient_history = history_status == "insufficient_history"

    if insufficient_history:
        logging.warning(
            "LOCAL search skipped_insufficient_history: hist_rows=%d min_required=%d",
            hist_rows,
            int(MIN_HIST_ROWS_FOR_LOCAL),
        )
        logging.warning(
            "Preserving previous local_matched/strategy_params artifacts because historical input is insufficient."
        )

    local_params = None
    local_tail_used = None
    local_ladder_attempts = []
    global_params = None
    if not insufficient_history:
        global_params, _ = find_best_local_params_lastN(
            df_past_sorted,
            window_n=max(int(LOCAL_SEARCH_N), int(len(df_past_sorted))),
            flat_stake_backtest=FLAT_STAKE,
            min_ev=min_EV,
            prob_clip_lo=PROB_CLIP_LO,
            prob_clip_hi=PROB_CLIP_HI,
            min_trades_local=10,
        )

        logging.info("=== [ROBUST++++] LOCAL PARAMS (WALK-FORWARD on FULL hist_df, coverage gate) ===")
        for tail_n in (300, 400, 500):
            if len(df_past_sorted) < tail_n:
                local_ladder_attempts.append(
                    f"  tail={tail_n} | rows={len(df_past_sorted)} | gate_pass=False | reason=insufficient_rows"
                )
                continue
            wf = find_best_local_params_walk_forward(
                df_past_sorted,
                tail_n=tail_n,
                min_ev=min_EV,
                stake=FLAT_STAKE,
                prob_clip_lo=PROB_CLIP_LO,
                prob_clip_hi=PROB_CLIP_HI,
                n_splits=4,
                min_trades_test_split=10,
            )
            if wf is None:
                local_ladder_attempts.append(
                    f"  tail={tail_n} | rows={tail_n} | gate_pass=False | reason=walk_forward_unavailable"
                )
                continue
            gate_pass = bool(wf["active_splits"] >= 2 and wf["q20_trades"] >= 10.0 and wf["test_trades_total"] >= 25)
            if gate_pass and local_params is None:
                local_params = dict(wf["params"])
                local_tail_used = int(tail_n)
                logging.info("score_mode             : %s", wf["score_mode"])
                logging.info("LCB(mean_test_ROI - 0.5*std): %.2f", float(wf["score"]))
                logging.info("params: %s", wf["params"])
                logging.info("LOCAL tail used        : %d", int(tail_n))
                logging.info(
                    "splits_used: %d | test_trades_total: %d | active_splits: %d | q20_trades: %.2f",
                    int(wf["splits_used"]),
                    int(wf["test_trades_total"]),
                    int(wf["active_splits"]),
                    float(wf["q20_trades"]),
                )
                logging.info(
                    "wf_test_profit_total: %.2f | FULL profit: %.2f | FULL ROI: %.2f%% | FULL trades: %d",
                    float(wf["wf_test_profit_total"]),
                    float(wf["full_profit"]),
                    float(wf["full_roi"]),
                    int(wf["full_trades"]),
                )
            local_ladder_attempts.append(
                "  tail={tail} | rows={rows} | gate_pass={gate} | score={score:.2f} | "
                "best_total={total} | best_active={active} | best_q={q:.1f}".format(
                    tail=tail_n,
                    rows=tail_n,
                    gate=str(gate_pass),
                    score=float(wf["score"]),
                    total=int(wf["test_trades_total"]),
                    active=int(wf["active_splits"]),
                    q=float(wf["q20_trades"]),
                )
            )

        logging.info("LOCAL ladder attempts summary:")
        for line in local_ladder_attempts:
            logging.info(line)

        if local_params and global_params:
            windows = [300, 400, 500]
            min_trades_per_window = 25
            hits_needed = 2
            local_eval = evaluate_strategy_stability(
                df_past_sorted,
                local_params,
                windows=windows,
                min_ev=min_EV,
                stake=FLAT_STAKE,
                prob_clip_lo=PROB_CLIP_LO,
                prob_clip_hi=PROB_CLIP_HI,
                min_trades_per_window=min_trades_per_window,
            )
            global_eval = evaluate_strategy_stability(
                df_past_sorted,
                global_params,
                windows=windows,
                min_ev=min_EV,
                stake=FLAT_STAKE,
                prob_clip_lo=PROB_CLIP_LO,
                prob_clip_hi=PROB_CLIP_HI,
                min_trades_per_window=min_trades_per_window,
            )
            compare_n = int(local_tail_used or 400)
            global_compare, _ = evaluate_params_on_hist_window(
                df_past_sorted.tail(compare_n),
                global_params,
                min_ev=min_EV,
                flat_stake_backtest=FLAT_STAKE,
                prob_clip_lo=PROB_CLIP_LO,
                prob_clip_hi=PROB_CLIP_HI,
            )
            local_compare, _ = evaluate_params_on_hist_window(
                df_past_sorted.tail(compare_n),
                local_params,
                min_ev=min_EV,
                flat_stake_backtest=FLAT_STAKE,
                prob_clip_lo=PROB_CLIP_LO,
                prob_clip_hi=PROB_CLIP_HI,
            )
            logging.info("=== [ROBUST++++] PROFIT/STABILITY EVAL (GLOBAL vs LOCAL) ===")
            logging.info(
                "Windows: %s | stability hits needed: %d | min trades per window: %d",
                windows,
                hits_needed,
                min_trades_per_window,
            )
            logging.info("GLOBAL eval rows      : %d", int(global_eval["rows_eval"]))
            logging.info("LOCAL eval rows       : %d", int(local_eval["rows_eval"]))
            logging.info("LOCAL tail found      : %s", str(local_tail_used))
            logging.info("LOCAL tail evaluated  : %d", compare_n)
            if local_eval["hits"] >= hits_needed and float(local_compare["profit_€"]) >= float(global_compare["profit_€"]):
                logging.info("✅ FOUND PROFITABLE CONFIG")
                logging.info("%s", {"chosen": "LOCAL", "compareN": compare_n})
                logging.info("params: %s", local_params)
            else:
                local_params = dict(global_params)
                logging.info("✅ FOUND PROFITABLE CONFIG")
                logging.info("%s", {"chosen": "GLOBAL", "compareN": compare_n})
                logging.info("params: %s", local_params)

    used_global_fallback = False
    used_safe_fallback = False
    # fallback if search found nothing
    if not local_params:
        if global_params:
            used_global_fallback = True
            local_params = dict(global_params)
            logging.warning("LOCAL param search returned None; using GLOBAL params fallback.")
        else:
            used_safe_fallback = True
            logging.warning("LOCAL+GLOBAL param search returned None; using safe fallback.")
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

    historical_subset_for_export, local_matched_export_source, local_matched_fallback_used = resolve_historical_subset_for_local_matched_export(
        subset_local=subset_local,
        hist_df=df_past_sorted,
        hist_window_200=hist_window_200,
        params_used=local_params,
        min_ev=min_EV,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
    )

    matched_export_latest = prepare_local_matched_export(
        historical_subset_for_export,
        stake=FLAT_STAKE,
    )
    if matched_export_latest.empty and not historical_subset_for_export.empty:
        logging.warning(
            "[LOCAL_MATCHED] normalized export became empty; forcing fallback reconstruction from hist_df + params_used"
        )
        rebuilt_subset = rebuild_historical_subset_from_hist_df(
            df_past_sorted,
            params_used=local_params,
            window_n=FAIR_COMPARE_N,
            min_ev=min_EV,
            prob_clip_lo=PROB_CLIP_LO,
            prob_clip_hi=PROB_CLIP_HI,
        )
        if not rebuilt_subset.empty:
            historical_subset_for_export = rebuilt_subset.copy()
            local_matched_export_source = "hist_df+params_used(rebuild_after_empty_normalization)"
            local_matched_fallback_used = True
            matched_export_latest = prepare_local_matched_export(
                historical_subset_for_export,
                stake=FLAT_STAKE,
            )
    logging.info("[LOCAL MATCHED EXPORT] source=%s rows=%d fallback_used=%s", local_matched_export_source, int(len(matched_export_latest)), str(local_matched_fallback_used).lower())
    if not matched_export_latest.empty:
        logging.info(
            "[LOCAL MATCHED EXPORT] date range=%s -> %s",
            str(matched_export_latest["date"].iloc[0]),
            str(matched_export_latest["date"].iloc[-1]),
        )
    else:
        logging.warning(
            "[LOCAL MATCHED EXPORT] WARNING: historical subset empty; writing header-only file"
        )

    shortlist = build_bet_shortlist(df_all, local_params, min_EV)
    shortlist_path = kelly_dir / f"bet_shortlist_{target_ymd}.csv"
    shortlist = shortlist.reindex(columns=SHORTLIST_COLUMNS)
    shortlist.to_csv(shortlist_path, index=False, encoding="utf-8")
    logging.info("Saved bet shortlist -> %s (%d rows)", shortlist_path, len(shortlist))
    logging.info(
        "[LOCAL_MATCHED] verification live_shortlist_rows=%d resolved_historical_rows=%d export_rows=%d",
        int(len(shortlist)),
        int(len(historical_subset_for_export)),
        int(len(matched_export_latest)),
    )

    # Bankroll over last-200 window (model EV, same style)
    last_200_games = hist_window_200.copy()
    last_200_games = _compute_prob_used(last_200_games, lo=PROB_CLIP_LO, hi=PROB_CLIP_HI, src=PROB_COL_HIST, dst="prob_used")
    last_200_games = last_200_games.dropna(subset=["prob_used", HOME_ODDS_COL])

    bankroll_last_200 = float(START_BANKROLL)
    for _, r in last_200_games.iterrows():
        p = float(r["prob_used"])
        o = float(r[HOME_ODDS_COL])
        bankroll_last_200 += FLAT_STAKE * (p * (o - 1.0) - (1.0 - p))

    resolved_local_matched_date = as_of_date
    if not insufficient_history:
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
          historical_subset=historical_subset_for_export,
        )

        resolved_local_matched_date = write_local_matched_artifacts(
            matched_export_latest,
            as_of_date=as_of_date,
            output_dir=out_dir,
            params_used=local_params,
            source_name=local_matched_export_source,
            allow_header_only=bool(matched_export_latest.empty),
            intentional_empty=bool(matched_export_latest.empty),
        )

        # Also write strategy params TXT (generated each run)
        write_strategy_params(local_params, min_ev=min_EV, as_of_date=as_of_date, stake=FLAT_STAKE, output_dir=out_dir)
        validate_dated_dashboard_artifacts(
            output_dir=out_dir,
            as_of_date=as_of_date,
            local_matched_date=resolved_local_matched_date,
        )

    # Minimal snapshot for trace (keep structure, but based on LOCAL params + last-200)
    fallback_used = bool(insufficient_history or used_global_fallback or used_safe_fallback)
    fallback_reason = (
        "skipped_insufficient_history"
        if insufficient_history
        else ("global_fallback" if used_global_fallback else ("safe_fallback" if used_safe_fallback else None))
    )
    if insufficient_history:
        params_used_type = "fallback"
    elif used_global_fallback:
        params_used_type = "GLOBAL"
    elif used_safe_fallback:
        params_used_type = "safe_fallback"
    else:
        params_used_type = "LOCAL"
    local_search_status = "skipped_insufficient_history" if insufficient_history else "ran"

    snapshot = {
        "meta": {
            "as_of_date": as_of_date,
            "eval_base_date_max": as_of_date,
            "strategy_variant": strategy_variant,
            "strategy_variant_label": strategy_variant_label,
            "params_source": params_source,
            "combined_file_path": str(snapshot_combined_source_path),
            "local_matched_games_source": local_matched_export_source,
            "local_search_status": local_search_status,
        },
        "as_of_date": as_of_date,
        "params_used_type": params_used_type,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "local_search_status": local_search_status,
        "params_source": params_source,
        "params_source_type": params_used_type,
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
        "live_oos_proxy": {
            "ready": bool(live_meta.get("live_oos_proxy_ready", False)),
            "train_rows": int(live_meta.get("live_oos_proxy_train_rows", 0)),
        },
    }

    metrics_snapshot_path = out_dir / "metrics_snapshot.json"
    metrics_snapshot_dated_path = out_dir / f"metrics_snapshot_{as_of_date}.json"
    write_json(metrics_snapshot_path, snapshot)
    write_json(metrics_snapshot_dated_path, snapshot)
    write_json(out_dir / "summary.json", {
        "as_of_date": as_of_date,
        "strategy_variant": strategy_variant,
        "strategy_variant_label": strategy_variant_label,
        "params_source": params_source,
        "combined_file_path": str(snapshot_combined_source_path),
        "local_matched_games_latest_written": (not insufficient_history),
        "params_used_type": params_used_type,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "local_search_status": "skipped_insufficient_history" if insufficient_history else "ran",
        "local_window_games": int(len(hist_window_200)),
        "local_matched_games": int(len(matched_export_latest)),
        "prob_col_hist": PROB_COL_HIST,
        "prob_col_live": PROB_COL_LIVE,
    })

    run_self_test(df_all, live_meta=live_meta)
    if insufficient_history:
        logging.info("DONE. LOCAL search skipped due to insufficient history; prior local artifacts preserved.")
    else:
        logging.info("DONE. local_matched_games_latest.csv updated using LOCAL params on last-200 window ending %s.", as_of_date)
        p = find_repo_root() / "web" / "public" / "data" / "local_matched_games_latest.csv"
        logging.info("AFTER WRITE: latest size=%d bytes mtime=%s", p.stat().st_size, datetime.fromtimestamp(p.stat().st_mtime))
        logging.info("AFTER WRITE HEAD:\n%s", "\n".join(p.read_text(encoding="utf-8").splitlines()[:5]))

       
if __name__ == "__main__":
    main()
