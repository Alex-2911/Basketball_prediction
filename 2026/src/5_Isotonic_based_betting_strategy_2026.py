#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
5_Isotonic_based_betting_strategy_2026.py

Step 5 of the 2026 pipeline (GitHub version, aligned with local notebook):

1) Load combined historical predictions:
       combined_nba_predictions_acc_YYYY-MM-DD.csv
   from 2026/LightGBM

2) Make sure the following columns exist (creating them if necessary):
   - game_date           (from 'date')
   - home_team_won       (from 'result' == 'home_team', only for played games)
   - pred_home_win_proba (from 'home_team_prob')
   - closing_home_odds   (from 'odds_1')
   - closing_away_odds   (from 'odds_2')

3) Merge in:
   - today's predictions from nba_games_predict_YYYY-MM-DD.csv
   - home win rates from home_win_rates_sorted_YYYY-MM-DD.csv

4) Split into:
   - df_past   (played games, for calibration + grid search)
   - df_future (upcoming games today/tomorrow, for shortlist)

5) Fit an Isotonic Regression on df_past and compute:
   - iso_proba_home_win

6) Run a grid search over a small parameter space:
   StrategyParams(min_home_win_rate, min_odds, max_odds, min_iso_proba)

7) Save:
   - Kelly/nba_grid_search_results_YYYY-MM-DD.csv
   - Kelly/combined_nba_predictions_iso_YYYY-MM-DD.csv

8) Build TODAY'S FLAT-STAKE SHORTLIST (global/local params):
   - local search uses LOCAL_SEARCH_N only
   - fair GLOBAL vs LOCAL uses FAIR_COMPARE_N only
   - prob_used clipped to [0.35, 0.80]
   - EV per 100 uses prob_used; min_EV default -5
   - print shortlist + explainer (all games + why_not)
   - save shortlist to bet_shortlist_YYYY-MM-DD.csv

If there are no upcoming games OR no suitable bets, script still succeeds.
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

# -------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -------------------------------------------------------------------------

DATE_COL = "game_date"
RESULT_COL = "home_team_won"
RESULT_RAW_COL = "result_raw"
PRED_PROBA_COL = "pred_home_win_proba"
HOME_ODDS_COL = "closing_home_odds"
AWAY_ODDS_COL = "closing_away_odds"
HOMEWR_COL = "home_win_rate"
ISO_COL = "iso_proba_home_win"

# Historical grid search (flat stake)
FLAT_STAKE = 100.0

ODDS_MIN_GRID = [1.10, 1.25, 1.40, 1.60]
ODDS_MAX_GRID = [2.00, 2.10, 2.50, 3.00]
PROB_MIN_GRID = [0.55, 0.60, 0.65, 0.70]
HOMEWR_MIN_GRID = [0.50, 0.55, 0.60, 0.65]

# LOCAL SHORTLIST FILTERS (match local notebook behavior)
LOCAL_MAX_KELLY_FRACTION = 0.10  # 10 % cap
LOCAL_PROB_CAP = 0.75            # cap prob_used at 0.75

# Flat-stake shortlist (two-window logic)
MIN_EV_DEFAULT = -5.0
PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80
LOCAL_SEARCH_N = 150
FAIR_COMPARE_N = 200

START_BANKROLL = 1000.0


@dataclass
class StrategyParams:
    min_home_win_rate: float
    min_odds: float
    max_odds: float
    min_iso_proba: float


# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s] INFO: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def to_float_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric cleanup (also ok if already floats).
    """
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace("[^0-9.]", "", regex=True)
         .replace("", np.nan)
         .astype(float)
    )


def load_combined_df(pred_dir: str, ymd_str: str) -> pd.DataFrame:
    """
    Load combined_nba_predictions_acc_YYYY-MM-DD.csv and guarantee that
    basic columns exist and are in a normalized format.
    """
    path = os.path.join(pred_dir, f"combined_nba_predictions_acc_{ymd_str}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Combined predictions file not found: {path}")

    logging.info("Loading predictions from %s", path)

    # Try utf-7 first (as you had), fall back to utf-8 if needed
    try:
        df = pd.read_csv(path, encoding="utf-7")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")

    # Normalize column names
    df.columns = (
        df.columns
          .astype(str)
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )

    # Keep the raw 'result' for played/upcoming detection
    if "result" in df.columns:
        df[RESULT_RAW_COL] = df["result"]
    else:
        df[RESULT_RAW_COL] = np.nan

    # DATE_COL
    if DATE_COL not in df.columns:
        if "date" in df.columns:
            logging.info("DATE_COL 'game_date' not in dataframe – creating it from 'date' column.")
            df[DATE_COL] = pd.to_datetime(df["date"], errors="coerce")
        else:
            logging.warning("No 'date' column in dataframe; DATE_COL will be NaT.")
            df[DATE_COL] = pd.NaT
    else:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # PRED_PROBA_COL
    if PRED_PROBA_COL not in df.columns:
        src = "home_team_prob" if "home_team_prob" in df.columns else None
        if src is not None:
            logging.info(
                "PRED_PROBA_COL 'pred_home_win_proba' not in dataframe – creating it from '%s'.",
                src,
            )
            df[PRED_PROBA_COL] = to_float_series(df[src])
        else:
            logging.warning("No suitable probability column found. Setting pred_home_win_proba to NaN.")
            df[PRED_PROBA_COL] = np.nan

    # HOME / AWAY odds
    if HOME_ODDS_COL not in df.columns:
        if "odds_1" in df.columns:
            logging.info(
                "HOME_ODDS_COL 'closing_home_odds' not in dataframe – "
                "creating it from 'odds_1' (home odds from Odds API)."
            )
            df[HOME_ODDS_COL] = to_float_series(df["odds_1"])
        else:
            logging.warning("No 'odds_1' column – setting closing_home_odds to NaN.")
            df[HOME_ODDS_COL] = np.nan

    if AWAY_ODDS_COL not in df.columns:
        if "odds_2" in df.columns:
            logging.info(
                "AWAY_ODDS_COL 'closing_away_odds' not in dataframe – "
                "creating it from 'odds_2' (away odds from Odds API)."
            )
            df[AWAY_ODDS_COL] = to_float_series(df["odds_2"])
        else:
            logging.warning("No 'odds_2' column – setting closing_away_odds to NaN.")
            df[AWAY_ODDS_COL] = np.nan

    # RESULT_COL (do NOT mark future games as 0-loss)
    if RESULT_COL not in df.columns:
        logging.info(
            "RESULT_COL 'home_team_won' not in dataframe – "
            "creating it as 1 if result==home_team, NaN otherwise."
        )
        if "home_team" in df.columns and "result" in df.columns:
            mask_valid = df["result"].notna() & (df["result"].astype(str) != "0")
            df.loc[mask_valid, RESULT_COL] = (
                df.loc[mask_valid, "result"].astype(str)
                == df.loc[mask_valid, "home_team"].astype(str)
            ).astype(int)
            df.loc[~mask_valid, RESULT_COL] = np.nan
        else:
            df[RESULT_COL] = np.nan

    return df


def merge_today_predictions(
    df_all: pd.DataFrame,
    pred_dir: str,
    ymd_str: str,
    today_date,
) -> pd.DataFrame:
    """
    Merge in upcoming games from nba_games_predict_YYYY-MM-DD.csv
    if they are not already present in df_all.
    """
    today_pred_path = os.path.join(pred_dir, f"nba_games_predict_{ymd_str}.csv")
    if not os.path.exists(today_pred_path):
        logging.info("No TODAY_PRED file found (%s) – skipping merge of upcoming games.", today_pred_path)
        return df_all

    logging.info("Merging upcoming games from %s", today_pred_path)

    try:
        tmp = pd.read_csv(today_pred_path, encoding="utf-7", sep=",", quotechar='"', decimal=".")
    except Exception:
        tmp = pd.read_csv(today_pred_path, encoding="utf-8", sep=",", quotechar='"', decimal=".")

    # Normalize columns
    tmp.columns = (
        tmp.columns
           .astype(str)
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

    # if still NaT, assume "today"
    tmp.loc[tmp["date"].isna(), "date"] = pd.Timestamp(today_date)

    # ensure result column exists
    if "result" not in tmp.columns:
        tmp["result"] = np.nan

    tmp[DATE_COL] = pd.to_datetime(tmp["date"], errors="coerce")

    # --- ensure ALL MODEL COLUMNS ALSO EXIST HERE ---
    if PRED_PROBA_COL not in tmp.columns and "home_team_prob" in tmp.columns:
        tmp[PRED_PROBA_COL] = tmp["home_team_prob"]

    if HOME_ODDS_COL not in tmp.columns and "odds_1" in tmp.columns:
        tmp[HOME_ODDS_COL] = tmp["odds_1"]
    if AWAY_ODDS_COL not in tmp.columns and "odds_2" in tmp.columns:
        tmp[AWAY_ODDS_COL] = tmp["odds_2"]

    # Make sure main df has away_team for key matching
    if "away_team" not in df_all.columns:
        df_all["away_team"] = np.nan

    key_cols = [DATE_COL, "home_team", "away_team"]
    for col in key_cols:
        if col not in df_all.columns:
            df_all[col] = np.nan

    existing_keys = df_all[key_cols].drop_duplicates()

    tmp_merge = tmp.merge(
        existing_keys,
        on=key_cols,
        how="left",
        indicator=True,
    )
    new_rows = tmp_merge[tmp_merge["_merge"] == "left_only"].drop(columns=["_merge"])

    if new_rows.empty:
        logging.info("No new upcoming games to add from TODAY_PRED.")
        return df_all

    # Align columns between df_all and new_rows
    needed_cols = set(df_all.columns) | set(new_rows.columns)
    for col in needed_cols:
        if col not in df_all.columns:
            df_all[col] = np.nan
        if col not in new_rows.columns:
            new_rows[col] = np.nan

    # 'home_team_won' is unknown for future games
    new_rows[RESULT_COL] = np.nan
    new_rows[RESULT_RAW_COL] = new_rows["result"]

    df_all = pd.concat(
        [df_all, new_rows[df_all.columns]],
        ignore_index=True,
    )

    return df_all


def attach_home_win_rate(df: pd.DataFrame, hwr_path: str) -> pd.DataFrame:
    """
    Attach home win rate (HOMEWR_COL) to df based on the
    home_win_rates_sorted_YYYY-MM-DD.csv file.
    """
    if not os.path.exists(hwr_path):
        logging.warning("Home win rate file not found at %s; skipping merge.", hwr_path)
        return df

    try:
        hwr = pd.read_csv(hwr_path, encoding="utf-7", sep=",", decimal=",")
    except Exception:
        try:
            hwr = pd.read_csv(hwr_path, encoding="utf-8", sep=",", decimal=",")
        except Exception as e:
            logging.warning("Failed to read home win rate file %s: %s", hwr_path, e)
            return df

    if hwr.empty:
        logging.warning("Home win rate file %s is empty; skipping merge.", hwr_path)
        return df

    cols = list(hwr.columns)
    cols_lower = [c.lower().strip() for c in cols]

    team_col = None
    winrate_col = None

    # SPECIAL CASE: 4-column format (first col is team, last col is win rate)
    if len(cols) == 4 and "home win rate" in cols_lower:
        team_col = cols[0]
        winrate_col = cols[cols_lower.index("home win rate")]
        logging.info(
            "Detected home-win-rate file format with 4 columns; using '%s' as team column and '%s' as win-rate column.",
            team_col, winrate_col
        )
    else:
        lower_to_orig = {c.lower().strip(): c for c in cols}

        for lc, orig in lower_to_orig.items():
            if lc in {"team", "home_team", "team_code"}:
                team_col = orig
                break
            if "team" in lc and ("abbr" in lc or "code" in lc or "home" in lc):
                team_col = orig
                break

        for lc, orig in lower_to_orig.items():
            if "home_win_rate" in lc or "home win rate" in lc or "win_rate" in lc:
                winrate_col = orig
                break

        if winrate_col is None:
            for c in cols:
                try:
                    vals = pd.to_numeric(hwr[c], errors="coerce")
                    if vals.notna().sum() == 0:
                        continue
                    frac_between = (((vals >= 0.0) & (vals <= 1.0)).sum() / vals.notna().sum())
                    if frac_between > 0.9:
                        winrate_col = c
                        break
                except Exception:
                    continue

        if team_col is None:
            for c in cols:
                sample = hwr[c].dropna().astype(str).str.strip().head(20).tolist()
                if not sample:
                    continue
                if all(len(x) <= 4 for x in sample) and all(x.upper() == x for x in sample):
                    team_col = c
                    break

        if team_col is None or winrate_col is None:
            logging.warning("Could not identify team and/or win-rate columns in %s; cols=%s – skipping merge.", hwr_path, cols)
            return df

        logging.info("Using '%s' as team column and '%s' as win-rate column.", team_col, winrate_col)

    hwr["_team_norm"] = (
        hwr[team_col]
        .astype(str)
        .str.strip()
        .map(normalize_team_code)
    )

    df["_home_team_norm"] = (
        df["home_team"]
        .astype(str)
        .str.strip()
        .map(normalize_team_code)
    )

    hwr_for_merge = hwr[["_team_norm", winrate_col]].drop_duplicates("_team_norm")

    df = df.merge(
        hwr_for_merge,
        left_on="_home_team_norm",
        right_on="_team_norm",
        how="left",
    )

    # standardize to HOMEWR_COL
    if HOMEWR_COL in df.columns and HOMEWR_COL != winrate_col:
        df[HOMEWR_COL] = df[HOMEWR_COL].fillna(df[winrate_col])
        df.drop(columns=[winrate_col], inplace=True)
    else:
        df.rename(columns={winrate_col: HOMEWR_COL}, inplace=True)

    df.drop(columns=["_team_norm", "_home_team_norm"], inplace=True, errors="ignore")
    df[HOMEWR_COL] = pd.to_numeric(df[HOMEWR_COL], errors="coerce").fillna(0.0)

    logging.info("Merged home win rates into dataframe; %d rows with non-null values.", df[HOMEWR_COL].notna().sum())
    return df


def split_past_future(df_all: pd.DataFrame, today_date, tomorrow_date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Past = games where RESULT_RAW_COL is non-null and != '0'
    Future = games where RESULT_RAW_COL is null or '0' and game_day in {today, tomorrow}
    """
    df_all[DATE_COL] = pd.to_datetime(df_all[DATE_COL], errors="coerce")
    df_all["game_day"] = df_all[DATE_COL].dt.date

    if RESULT_RAW_COL not in df_all.columns:
        df_all[RESULT_RAW_COL] = np.nan

    played_mask = df_all[RESULT_RAW_COL].notna() & (df_all[RESULT_RAW_COL].astype(str) != "0")

    df_past = df_all[played_mask].copy()
    df_future = df_all[~played_mask & df_all["game_day"].isin([today_date, tomorrow_date])].copy()

    logging.info("Split into %d past games and %d future games.", len(df_past), len(df_future))
    return df_past, df_future


def compute_home_win_rates(df_all: pd.DataFrame, target_ymd: str, pred_dir: str) -> str:
    """
    Compute home win rates for all teams based on the last 20 games (home or away),
    but only counting *home* games for the win rate.
    """
    df = df_all.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif DATE_COL in df.columns:
        df["date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    else:
        logging.warning("No 'date' or '%s' column found – cannot compute home win rates.", DATE_COL)
        out_path = os.path.join(pred_dir, f"home_win_rates_sorted_{target_ymd}.csv")
        pd.DataFrame().to_csv(out_path, index=True)
        return out_path

    def get_last_20_games_all_teams(df_local: pd.DataFrame) -> pd.DataFrame:
        team_results = {}

        for team in df_local["home_team"].dropna().unique():
            team_games = df_local[
                (df_local["home_team"] == team) | (df_local["away_team"] == team)
            ].sort_values(by="date", ascending=False).head(20)

            home_games = team_games[team_games["home_team"] == team]

            total_home_games = len(home_games)
            home_wins = len(home_games[home_games["result"] == team])
            home_win_rate = round(home_wins / total_home_games, 2) if total_home_games > 0 else 0.0

            team_results[team] = {
                "Total Last 20 Games": len(team_games),
                "Total Home Games": total_home_games,
                "Home Wins": home_wins,
                "Home Win Rate": home_win_rate,
            }

        hwr_df = pd.DataFrame.from_dict(team_results, orient="index")
        hwr_df.sort_values(by="Home Win Rate", ascending=False, inplace=True)
        return hwr_df

    home_win_rates_all_teams_sorted = get_last_20_games_all_teams(df)

    logging.info("\n🏀 Home Win Rates (Sorted) for All Teams:")
    logging.info("\n%s", home_win_rates_all_teams_sorted.to_string())

    out_path = os.path.join(pred_dir, f"home_win_rates_sorted_{target_ymd}.csv")
    home_win_rates_all_teams_sorted.to_csv(out_path, index=True, encoding="utf-8")
    logging.info("📁 Sorted home win rates saved to: %s", out_path)

    return out_path


def fit_isotonic(df_past: pd.DataFrame) -> IsotonicRegression:
    mask = df_past[RESULT_COL].notna() & df_past[PRED_PROBA_COL].notna()
    if mask.sum() == 0:
        raise RuntimeError("No valid rows to fit isotonic regression (missing y_true or probabilities).")

    y_true = df_past.loc[mask, RESULT_COL].astype(int).values
    p_raw = df_past.loc[mask, PRED_PROBA_COL].astype(float).values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_true)

    logging.info("Isotonic fitted on %d games.", mask.sum())
    return iso


def compute_calibration_metrics(df_past: pd.DataFrame) -> Tuple[float, float, float, float]:
    mask = df_past[RESULT_COL].notna() & df_past[PRED_PROBA_COL].notna() & df_past[ISO_COL].notna()
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    y_true = df_past.loc[mask, RESULT_COL].astype(int).values
    p_raw = df_past.loc[mask, PRED_PROBA_COL].astype(float).values
    p_iso = df_past.loc[mask, ISO_COL].astype(float).values

    brier_before = brier_score_loss(y_true, p_raw)
    brier_after = brier_score_loss(y_true, p_iso)

    logloss_before = log_loss(y_true, p_raw, eps=1e-15)
    logloss_after = log_loss(y_true, p_iso, eps=1e-15)

    return brier_before, brier_after, logloss_before, logloss_after


def evaluate_strategy(df: pd.DataFrame, params: StrategyParams) -> dict:
    if df.empty:
        return {"n_bets": 0, "total_profit": 0.0, "roi_per_bet": 0.0}

    conds = []

    if HOMEWR_COL in df.columns and pd.api.types.is_numeric_dtype(df[HOMEWR_COL]):
        conds.append(df[HOMEWR_COL] >= params.min_home_win_rate)

    conds.append(df[HOME_ODDS_COL].between(params.min_odds, params.max_odds))
    conds.append(df[ISO_COL] >= params.min_iso_proba)

    conds.append(df[HOME_ODDS_COL].notna())
    conds.append(df[ISO_COL].notna())
    conds.append(df[RESULT_COL].notna())

    mask = np.logical_and.reduce(conds)
    df_sel = df[mask].copy()

    n_bets = len(df_sel)
    if n_bets == 0:
        return {"n_bets": 0, "total_profit": 0.0, "roi_per_bet": 0.0}

    stake = FLAT_STAKE
    df_sel["profit"] = np.where(
        df_sel[RESULT_COL].astype(int) == 1,
        (df_sel[HOME_ODDS_COL] - 1.0) * stake,
        -stake,
    )

    total_profit = float(df_sel["profit"].sum())
    roi_per_bet = total_profit / (n_bets * stake)

    return {"n_bets": n_bets, "total_profit": total_profit, "roi_per_bet": roi_per_bet}


def grid_search(df_past: pd.DataFrame) -> Tuple[StrategyParams, pd.DataFrame]:
    results = []

    for min_hwr in HOMEWR_MIN_GRID:
        for min_odds in ODDS_MIN_GRID:
            for max_odds in ODDS_MAX_GRID:
                if max_odds <= min_odds:
                    continue
                for min_prob in PROB_MIN_GRID:
                    params = StrategyParams(
                        min_home_win_rate=min_hwr,
                        min_odds=min_odds,
                        max_odds=max_odds,
                        min_iso_proba=min_prob,
                    )
                    metrics = evaluate_strategy(df_past, params)
                    metrics.update(
                        min_home_win_rate=min_hwr,
                        min_odds=min_odds,
                        max_odds=max_odds,
                        min_iso_proba=min_prob,
                    )
                    results.append(metrics)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        raise RuntimeError("Grid search produced no results (df_res is empty).")

    df_res = df_res.sort_values(by=["roi_per_bet", "n_bets"], ascending=[False, False]).reset_index(drop=True)

    best_row = df_res.iloc[0]
    best_params = StrategyParams(
        min_home_win_rate=float(best_row["min_home_win_rate"]),
        min_odds=float(best_row["min_odds"]),
        max_odds=float(best_row["max_odds"]),
        min_iso_proba=float(best_row["min_iso_proba"]),
    )

    return best_params, df_res


def _validate_params(params: dict, required=None, name="params"):
    required = required or ["home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold"]
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"{name} missing keys: {missing}. Got: {list(params.keys())}")


def _ensure_datetime(df: pd.DataFrame, col=DATE_COL) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns and not np.issubdtype(out[col].dtype, np.datetime64):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _compute_prob_used(df: pd.DataFrame, lo: float, hi: float, src=ISO_COL, dst="prob_used") -> pd.DataFrame:
    out = df.copy()
    if src not in out.columns:
        raise KeyError(f"Missing column '{src}' needed to compute '{dst}'.")
    out[dst] = out[src].clip(lower=lo, upper=hi)
    return out


def _compute_ev_per_100(df: pd.DataFrame, prob_col="prob_used", odds_col=HOME_ODDS_COL, stake_for_ev=100.0, dst="EV_€_per_100"):
    out = df.copy()
    out[dst] = (out[prob_col] * (out[odds_col] - 1.0) - (1.0 - out[prob_col])) * stake_for_ev
    return out


def _make_game_key(df: pd.DataFrame, date_col: str, home_col="home_team", away_col="away_team", dst="game_key"):
    out = _ensure_datetime(df, date_col)
    out[dst] = (
        out[date_col].dt.strftime("%Y-%m-%d") + "_" +
        out[home_col].astype(str) + "_" +
        out[away_col].astype(str)
    )
    return out


def _params_to_dict(params: StrategyParams) -> dict:
    return {
        "home_win_rate_threshold": float(params.min_home_win_rate),
        "odds_min": float(params.min_odds),
        "odds_max": float(params.max_odds),
        "prob_threshold": float(params.min_iso_proba),
    }


def evaluate_params_on_hist_window(
    hist_window: pd.DataFrame,
    params: dict,
    *,
    min_ev: float,
    flat_stake_backtest: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
):
    _validate_params(params, name="params_to_eval")

    df = hist_window.copy()
    df = _ensure_datetime(df, DATE_COL)
    if "date" not in df.columns and DATE_COL in df.columns:
        df["date"] = df[DATE_COL]
    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, stake_for_ev=100.0, dst="EV_€_per_100")
    df = df.dropna(subset=[HOMEWR_COL, "prob_used", HOME_ODDS_COL, RESULT_COL, "EV_€_per_100"])

    prob_thr_eff = max(float(params["prob_threshold"]), float(prob_clip_lo))

    mask = (
        (df[HOMEWR_COL]   >= float(params["home_win_rate_threshold"])) &
        (df[HOME_ODDS_COL] >= float(params["odds_min"])) &
        (df[HOME_ODDS_COL] <= float(params["odds_max"])) &
        (df["prob_used"]   >= prob_thr_eff) &
        (df["EV_€_per_100"] >  float(min_ev))
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
        -float(flat_stake_backtest)
    )

    profit = float(subset["pnl"].sum())
    n_trades = int(len(subset))
    total_stake = n_trades * float(flat_stake_backtest)
    roi = (profit / total_stake * 100.0) if total_stake > 0 else 0.0

    metrics = {
        "n_trades": n_trades,
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
    homewr_grid,
    odds_min_grid,
    odds_max_grid,
    prob_min_grid,
    flat_stake_backtest: float,
    min_ev: float,
    min_trades_local: int = 10,
    prob_clip_lo: float,
    prob_clip_hi: float,
    window_n: int,
):
    if hist_df is None or hist_df.empty:
        return None, None, None, None, None

    hist_recent = _ensure_datetime(hist_df, DATE_COL).sort_values(DATE_COL).copy()
    hist_window = hist_recent.tail(int(window_n)).copy()

    needed = [DATE_COL, "home_team", "away_team", HOMEWR_COL, ISO_COL, HOME_ODDS_COL, RESULT_COL]
    missing_cols = [c for c in needed if c not in hist_window.columns]
    if missing_cols:
        raise KeyError(f"hist_df missing columns: {missing_cols}")

    hist_window = _compute_prob_used(hist_window, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
    hist_window = _compute_ev_per_100(hist_window, prob_col="prob_used", odds_col=HOME_ODDS_COL, stake_for_ev=100.0, dst="EV_€_per_100")
    hist_window = hist_window.dropna(subset=[HOMEWR_COL, "prob_used", HOME_ODDS_COL, RESULT_COL, "EV_€_per_100"])

    if len(hist_window) < 20:
        return None, None, None, None, hist_window

    prob_min_grid_eff = [p for p in prob_min_grid if p >= prob_clip_lo]
    if not prob_min_grid_eff:
        prob_min_grid_eff = [prob_clip_lo]

    best_profit = float("-inf")
    best_subset = None
    best_params_local = None
    best_n_trades = 0

    for hw_cut in homewr_grid:
        for o_min in odds_min_grid:
            for o_max in odds_max_grid:
                if o_max <= o_min:
                    continue
                for p_min in prob_min_grid_eff:
                    mask = (
                        (hist_window[HOMEWR_COL]   >= hw_cut) &
                        (hist_window[HOME_ODDS_COL] >= o_min) &
                        (hist_window[HOME_ODDS_COL] <= o_max) &
                        (hist_window["prob_used"]   >= p_min) &
                        (hist_window["EV_€_per_100"] >  float(min_ev))
                    )

                    subset = hist_window.loc[mask].copy()
                    if subset.empty:
                        continue

                    subset["pnl"] = np.where(
                        subset[RESULT_COL].astype(int) == 1,
                        float(flat_stake_backtest) * (subset[HOME_ODDS_COL] - 1.0),
                        -float(flat_stake_backtest)
                    )

                    total_profit = float(subset["pnl"].sum())
                    n_trades = int(len(subset))

                    if n_trades < min_trades_local:
                        continue

                    if total_profit > best_profit:
                        best_profit = total_profit
                        best_subset = subset
                        best_n_trades = n_trades
                        best_params_local = {
                            "home_win_rate_threshold": round(float(hw_cut), 2),
                            "odds_min": round(float(o_min), 2),
                            "odds_max": round(float(o_max), 2),
                            "prob_threshold": round(float(p_min), 2),
                            "n_trades": n_trades,
                            "win_rate_%": round(float(subset[RESULT_COL].mean() * 100.0), 2),
                            "avg_EV_€_per_100": round(float(subset["EV_€_per_100"].mean()), 2),
                        }

    if best_params_local is None:
        return None, None, None, None, hist_window

    total_stake = best_n_trades * float(flat_stake_backtest)
    roi_local = (best_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    return best_params_local, best_subset, float(roi_local), float(best_profit), hist_window


def print_local_search_results(best_params_local, roi_local_search, window_n: int):
    if best_params_local is None:
        print(f"\nNo robust local params found on last {window_n} games (min 10 trades).")
        return

    print(f"\n=== LOCAL PARAMS (FOUND BY SEARCH, LAST {window_n} GAMES) ===")
    print(f"home_win_rate_threshold : {best_params_local['home_win_rate_threshold']}")
    print(f"odds_min                : {best_params_local['odds_min']}")
    print(f"odds_max                : {best_params_local['odds_max']}")
    print(f"prob_threshold (USED)   : {best_params_local['prob_threshold']}")
    print(f"n_trades (last {window_n})     : {best_params_local['n_trades']}")
    print(f"win_rate_%              : {best_params_local['win_rate_%']}")
    print(f"avg EV €/100            : {best_params_local['avg_EV_€_per_100']}")
    print(f"ROI % (search subset)   : {roi_local_search:.2f}%")


def print_local_matched_games(best_subset_window, window_n: int):
    print(f"\n=== LOCAL MATCHED GAMES (LAST {window_n} WINDOW) ===")
    if best_subset_window is None or best_subset_window.empty:
        print(f"n_trades (last {window_n} window) : 0")
        return

    df = best_subset_window.copy()
    if "date" not in df.columns and DATE_COL in df.columns:
        df["date"] = df[DATE_COL]
    df["home_win_rate"] = df[HOMEWR_COL]
    df["prob_iso"] = df[ISO_COL]
    df["odds_1"] = df[HOME_ODDS_COL]
    df["win"] = df[RESULT_COL].astype(int)
    if "EV_€_per_100" not in df.columns:
        if "ev_per_100" in df.columns:
            df["EV_€_per_100"] = df["ev_per_100"]
        elif "EV_per_100" in df.columns:
            df["EV_€_per_100"] = df["EV_per_100"]

    cols = [
        "date", "home_team", "away_team", "home_win_rate", "prob_iso",
        "prob_used", "odds_1", "EV_€_per_100", "win", "pnl",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    print(f"n_trades (last {window_n} window) : {len(df)}")
    print(
        df[cols]
        .sort_values("date")
        .round({
            "home_win_rate": 3,
            "prob_iso": 3,
            "prob_used": 3,
            "odds_1": 3,
            "EV_€_per_100": 2,
            "pnl": 1,
        })
        .to_string(index=False)
    )


def export_local_matched_games(
    best_subset_window,
    window_n: int,
    output_dir: str,
    as_of_date: str,
) -> None:
    if best_subset_window is None or best_subset_window.empty:
        logging.info("No local matched games to export for last %s window.", window_n)
        return

    df_export = prepare_local_matched_export(best_subset_window, stake=FLAT_STAKE)
    if df_export.empty:
        logging.info("No settled local matched games to export for last %s window.", window_n)
        return

    export_path = os.path.join(output_dir, f"local_matched_games_{as_of_date}.csv")
    df_export.to_csv(export_path, index=False, encoding="utf-8")
    logging.info("Exported local matched games to %s (%d rows).", export_path, len(df_export))


def resolve_output_dir(base_dir: str, prediction_dir: str) -> Path:
    lgbm_dir = os.environ.get("LGBM_DIR")
    if lgbm_dir:
        out_dir = Path(lgbm_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    source_root = os.environ.get("SOURCE_ROOT")
    if source_root:
        source_path = Path(source_root)
        if source_path.exists():
            out_dir = source_path / "LightGBM"
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

    base_path = Path(base_dir)
    out_dir = base_path / "2026" / "LightGBM"
    if out_dir.exists() or base_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    out_dir = Path(prediction_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _as_of_date_from_df(df: pd.DataFrame, fallback: str) -> str:
    if df is None or df.empty:
        return fallback
    date_col = "date" if "date" in df.columns else DATE_COL
    if date_col not in df.columns:
        return fallback
    date_vals = pd.to_datetime(df[date_col], errors="coerce")
    result_col = None
    for candidate in (RESULT_COL, "win", "home_team_won", "result"):
        if candidate in df.columns:
            result_col = candidate
            break
    if result_col:
        date_vals = date_vals[df[result_col].notna()]
    if date_vals.notna().any():
        return date_vals.max().strftime("%Y-%m-%d")
    return fallback


def _resolve_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _normalize_date(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")


def find_bet_log_path(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "bet_log_flat_live.csv",
        output_dir / "bet_log_live.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    flat_candidates = sorted(output_dir.glob("bet_log_flat_live_*.csv"))
    if flat_candidates:
        return flat_candidates[-1]
    live_candidates = sorted(output_dir.glob("bet_log_live_*.csv"))
    if live_candidates:
        return live_candidates[-1]
    return None


def build_settled_bets(
    bet_log_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    if bet_log_df is None or bet_log_df.empty or results_df is None or results_df.empty:
        return pd.DataFrame()

    bet_df = bet_log_df.copy()
    results = results_df.copy()

    bet_date_col = _resolve_first_col(bet_df, ["date", "game_date"])
    bet_home_col = _resolve_first_col(bet_df, ["home_team", "home", "team_home"])
    bet_away_col = _resolve_first_col(bet_df, ["away_team", "away", "team_away"])
    odds_col = _resolve_first_col(bet_df, ["odds_1", "odds", "home_odds", "closing_home_odds"])
    stake_col = _resolve_first_col(bet_df, ["stake", "stake_eur", "stake_flat"])

    if not bet_date_col or not bet_home_col or not bet_away_col:
        return pd.DataFrame()

    bet_df["date"] = _normalize_date(bet_df, bet_date_col)
    bet_df["home_team"] = _normalize_text(bet_df[bet_home_col])
    bet_df["away_team"] = _normalize_text(bet_df[bet_away_col])

    if odds_col:
        bet_df["odds_1"] = pd.to_numeric(bet_df[odds_col], errors="coerce")
    else:
        bet_df["odds_1"] = np.nan

    if stake_col:
        bet_df["stake"] = pd.to_numeric(bet_df[stake_col], errors="coerce")
    else:
        bet_df["stake"] = np.nan

    results_date_col = _resolve_first_col(results, ["date", DATE_COL, "game_date"])
    results_home_col = _resolve_first_col(results, ["home_team", "home"])
    results_away_col = _resolve_first_col(results, ["away_team", "away"])
    results_win_col = _resolve_first_col(results, ["home_team_won", "win", "result"])

    if not results_date_col or not results_home_col or not results_away_col or not results_win_col:
        return pd.DataFrame()

    results["date"] = _normalize_date(results, results_date_col)
    results["home_team"] = _normalize_text(results[results_home_col])
    results["away_team"] = _normalize_text(results[results_away_col])
    results["win"] = pd.to_numeric(results[results_win_col], errors="coerce")

    merged = bet_df.merge(
        results[["date", "home_team", "away_team", "win"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    merged = merged.dropna(subset=["stake", "odds_1"]).copy()

    pnl_col = _resolve_first_col(merged, ["pnl", "profit_eur", "profit"])
    if pnl_col:
        merged["pnl"] = pd.to_numeric(merged[pnl_col], errors="coerce")
    else:
        merged["pnl"] = np.nan

    needs_pnl = merged["pnl"].isna() & merged["win"].notna()
    merged.loc[needs_pnl, "pnl"] = np.where(
        merged.loc[needs_pnl, "win"] == 1,
        merged.loc[needs_pnl, "stake"] * (merged.loc[needs_pnl, "odds_1"] - 1.0),
        -merged.loc[needs_pnl, "stake"],
    )

    merged = merged.dropna(subset=["win", "pnl"]).copy()
    if merged.empty:
        return merged

    merged["win"] = merged["win"].clip(lower=0, upper=1).astype(int)
    merged = merged.drop_duplicates(subset=["date", "home_team", "away_team"]).sort_values("date")

    return merged.reset_index(drop=True)


def prepare_local_matched_export(best_subset_window: pd.DataFrame, stake: float) -> pd.DataFrame:
    df = best_subset_window.copy()
    if "date" not in df.columns and DATE_COL in df.columns:
        df["date"] = df[DATE_COL]
    df = _ensure_datetime(df, "date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    if "home_win_rate" not in df.columns and HOMEWR_COL in df.columns:
        df["home_win_rate"] = df[HOMEWR_COL]
    if "prob_iso" not in df.columns and ISO_COL in df.columns:
        df["prob_iso"] = df[ISO_COL]
    if "prob_used" not in df.columns and "prob_iso" in df.columns:
        df["prob_used"] = df["prob_iso"]
    if "odds_1" not in df.columns and HOME_ODDS_COL in df.columns:
        df["odds_1"] = df[HOME_ODDS_COL]
    if "win" not in df.columns and RESULT_COL in df.columns:
        df["win"] = df[RESULT_COL]
    if "pnl" not in df.columns and "pnl_flat" in df.columns:
        df["pnl"] = df["pnl_flat"]

    if "EV_€_per_100" not in df.columns:
        if "ev_per_100" in df.columns:
            df["EV_€_per_100"] = df["ev_per_100"]
        elif "EV_per_100" in df.columns:
            df["EV_€_per_100"] = df["EV_per_100"]
        elif "prob_used" in df.columns and HOME_ODDS_COL in df.columns:
            df = _compute_ev_per_100(
                df,
                prob_col="prob_used",
                odds_col=HOME_ODDS_COL,
                stake_for_ev=100.0,
                dst="EV_€_per_100",
            )

    df["home_win_rate"] = pd.to_numeric(df["home_win_rate"], errors="coerce")
    df["prob_iso"] = pd.to_numeric(df["prob_iso"], errors="coerce")
    df["prob_used"] = pd.to_numeric(df["prob_used"], errors="coerce")
    df["odds_1"] = pd.to_numeric(df["odds_1"], errors="coerce")
    df["EV_€_per_100"] = pd.to_numeric(df["EV_€_per_100"], errors="coerce")
    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    if "pnl" not in df.columns and df["win"].notna().any() and df["odds_1"].notna().any():
        df["pnl"] = np.where(
            df["win"] == 1,
            float(stake) * (df["odds_1"] - 1.0),
            -float(stake),
        )

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")

    df = df.dropna(subset=["win", "pnl"]).copy()
    if df.empty:
        return df

    df["win"] = df["win"].clip(lower=0, upper=1).astype(int)
    df["stake"] = float(stake)

    cols = [
        "date",
        "home_team",
        "away_team",
        "home_win_rate",
        "prob_iso",
        "prob_used",
        "odds_1",
        "EV_€_per_100",
        "win",
        "pnl",
        "stake",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].sort_values("date").reset_index(drop=True)
    return df


def build_metrics_snapshot(
    export_df: pd.DataFrame,
    *,
    params_used: dict,
    params_used_type: str | None,
    min_ev: float,
    as_of_date: str,
    stake: float,
    bankroll_window: float | None = None,
    bankroll_2026: float | None = None,
    profit_2026: float | None = None,
    settled_summary: dict | None = None,
) -> dict:
    realized_count = int(len(export_df))
    profit_sum = float(export_df["pnl"].sum()) if realized_count > 0 else 0.0
    roi = profit_sum / (realized_count * float(stake)) if realized_count > 0 else 0.0
    win_rate = float(export_df["win"].mean()) if realized_count > 0 else 0.0
    ev_col = "EV_€_per_100" if "EV_€_per_100" in export_df.columns else "ev_per_100"
    ev_mean = float(export_df[ev_col].mean()) if realized_count > 0 else 0.0
    if np.isnan(ev_mean):
        ev_mean = 0.0

    sharpe_style = 0.0
    if realized_count > 1:
        pnl_std = float(export_df["pnl"].std(ddof=1))
        pnl_mean = float(export_df["pnl"].mean())
        if pnl_std > 0:
            sharpe_style = pnl_mean / pnl_std

    snapshot = {
        "meta": {
            "eval_base_date_max": as_of_date,
            "strategy_results_label": f"Simulated (last {FAIR_COMPARE_N} games window)",
            "live_bets_label": "Live bets (2026 YTD, unfiltered)",
            "data_scopes": {
                "simulated_window_games": FAIR_COMPARE_N,
                "live_bets_window": "2026 YTD",
            },
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
        "ev_stats": {
            "mean": round(ev_mean, 2),
        },
        "filter_params": {
            "home_win_rate_threshold": float(params_used["home_win_rate_threshold"]),
            "odds_min": float(params_used["odds_min"]),
            "odds_max": float(params_used["odds_max"]),
            "prob_threshold": float(params_used["prob_threshold"]),
            "min_EV": float(min_ev),
        },
    }
    if bankroll_window is not None or bankroll_2026 is not None or profit_2026 is not None:
        snapshot["bankroll"] = {
            "deposit_eur": round(float(START_BANKROLL), 2),
            "bankroll_last_200_eur": round(float(bankroll_window), 2)
            if bankroll_window is not None
            else None,
            "bankroll_2026_ytd_eur": round(float(bankroll_2026), 2) if bankroll_2026 is not None else None,
            "profit_2026_ytd_eur": round(float(profit_2026), 2) if profit_2026 is not None else None,
            "flat_stake_eur": round(float(stake), 2),
        }
    if settled_summary:
        snapshot["settled_bets_2026"] = settled_summary
    return snapshot


def write_metrics_snapshot(snapshot: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics_snapshot.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    logging.info("Saved metrics snapshot to %s", out_path)
    return out_path


def write_strategy_params(
    params_used: dict,
    *,
    min_ev: float,
    as_of_date: str,
    stake: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "strategy_params.txt"
    lines = [
        f"as_of_date={as_of_date}",
        f"min_ev={float(min_ev)}",
        f"stake={float(stake)}",
    ]
    for key in sorted(params_used.keys()):
        lines.append(f"{key}={params_used[key]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved strategy params to %s", out_path)
    return out_path


def export_local_matched_games_settled(
    export_df: pd.DataFrame,
    *,
    output_dir: Path,
    as_of_date: str,
) -> Path | None:
    if export_df is None or export_df.empty:
        logging.info("No settled local matched games to export.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / f"local_matched_games_{as_of_date}.csv"
    export_df.to_csv(export_path, index=False, encoding="utf-8")

    logging.info("Exported settled local matched games to %s (%d rows).", export_path, len(export_df))
    return export_path


def check_metrics_snapshot_consistency(export_df: pd.DataFrame, output_dir: Path) -> None:
    snapshot_path = output_dir / "metrics_snapshot.json"
    if not snapshot_path.exists():
        return

    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("WARNING: metrics_snapshot.json could not be parsed for consistency checks.")
        return

    realized = snapshot.get("realized", {})
    expected_count = realized.get("count")
    expected_profit = realized.get("profit_sum")

    if expected_count is not None and len(export_df) != int(expected_count):
        logging.warning(
            "local_matched_games row count mismatch (expected %s, got %s).",
            expected_count,
            len(export_df),
        )
    if expected_profit is not None:
        pnl_sum = float(export_df["pnl"].sum())
        if abs(pnl_sum - float(expected_profit)) > 0.01:
            logging.warning(
                "local_matched_games pnl sum mismatch (expected %.2f, got %.2f).",
                float(expected_profit),
                pnl_sum,
            )


def update_last_run_trace(
    export_df: pd.DataFrame,
    export_path: Path | None,
    snapshot: dict,
    settled_bets_df: pd.DataFrame | None,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    trace_path = repo_root / "public" / "data" / "last_run.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    trace_data: dict = {}
    if trace_path.exists():
        try:
            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Unable to parse existing last_run.json; overwriting.")

    resolved_path = str(export_path.resolve()) if export_path is not None else ""
    expected_rows = snapshot.get("realized", {}).get("count")
    settled_rows = int(len(export_df)) if export_df is not None else 0
    profit_sum = float(export_df["pnl"].sum()) if export_df is not None and not export_df.empty else 0.0

    settled_2026 = settled_bets_df.copy() if settled_bets_df is not None else pd.DataFrame()
    if not settled_2026.empty and "date" in settled_2026.columns:
        settled_2026["date"] = pd.to_datetime(settled_2026["date"], errors="coerce")
        settled_2026 = settled_2026[settled_2026["date"].dt.year == 2026]

    net_pl = float(settled_2026["pnl"].sum()) if not settled_2026.empty else 0.0
    stake_sum = float(settled_2026["stake"].sum()) if not settled_2026.empty else 0.0
    win_count = int(settled_2026["win"].sum()) if not settled_2026.empty else 0
    settled_count = int(len(settled_2026))
    avg_odds = float(settled_2026["odds_1"].mean()) if not settled_2026.empty else 0.0

    bankroll_start = float(START_BANKROLL)
    bankroll_end = bankroll_start + net_pl

    trace_data.update(
        {
            "local_matched_games_source": resolved_path,
            "local_matched_games_rows_expected": int(expected_rows)
            if expected_rows is not None
            else settled_rows,
            "local_matched_games_rows_settled": settled_rows,
            "local_matched_games_profit_sum_table": round(profit_sum, 2),
            "bankroll_2026_start": round(bankroll_start, 2),
            "bankroll_2026_net_pl": round(net_pl, 2),
            "bankroll_2026_end": round(bankroll_end, 2),
            "settled_bets_2026_count": settled_count,
            "settled_bets_2026_wins": win_count,
            "settled_bets_2026_profit_sum": round(net_pl, 2),
            "settled_bets_2026_roi": round(net_pl / stake_sum, 4) if stake_sum else 0.0,
            "settled_bets_2026_avg_odds": round(avg_odds, 2) if settled_count else 0.0,
        }
    )

    trace_path.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")
    logging.info("Updated trace info at %s", trace_path)


def choose_params_fair_lastN(
    global_params: dict,
    local_params: dict | None,
    *,
    hist_window: pd.DataFrame,
    min_ev: float,
    flat_stake_backtest: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
    min_trades: int = 10,
    roi_edge_min_pp: float = 0.0,
):
    _validate_params(global_params, name="best_params (GLOBAL)")

    metrics_g, subset_g = evaluate_params_on_hist_window(
        hist_window, global_params,
        min_ev=min_ev,
        flat_stake_backtest=flat_stake_backtest,
        prob_clip_lo=prob_clip_lo,
        prob_clip_hi=prob_clip_hi,
    )

    if local_params is None:
        return False, global_params.copy(), metrics_g, None, subset_g, None

    _validate_params(local_params, name="best_params_local (LOCAL)")

    metrics_l, subset_l = evaluate_params_on_hist_window(
        hist_window, local_params,
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


def upcoming_filter_eval_and_reasons(
    upcoming_df: pd.DataFrame,
    params_used: dict,
    *,
    min_ev: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
):
    if upcoming_df is None or upcoming_df.empty:
        print("\n=== UPCOMING FILTER EVAL (UNIQUE GAMES) ===")
        print("No upcoming games to evaluate.")
        print("\n=== ALL UPCOMING GAMES & FILTER REASONS ===")
        print("No upcoming games to evaluate.")
        return

    _validate_params(params_used, name="params_used")

    df = upcoming_df.copy()
    date_col = "date" if "date" in df.columns else DATE_COL
    if "date" not in df.columns and DATE_COL in df.columns:
        df["date"] = df[DATE_COL]

    df = _ensure_datetime(df, date_col)
    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, stake_for_ev=100.0, dst="EV_€_per_100")
    df = _make_game_key(df, date_col=date_col, home_col="home_team", away_col="away_team", dst="game_key")

    df = (
        df.sort_values("EV_€_per_100", ascending=False)
          .drop_duplicates(subset="game_key", keep="first")
          .reset_index(drop=True)
    )

    prob_thr_eff = max(float(params_used["prob_threshold"]), float(prob_clip_lo))

    m_hwr = (df[HOMEWR_COL] >= float(params_used["home_win_rate_threshold"]))
    m_odmin = m_hwr & (df[HOME_ODDS_COL] >= float(params_used["odds_min"]))
    m_odmax = m_odmin & (df[HOME_ODDS_COL] <= float(params_used["odds_max"]))
    m_prob = m_odmax & (df["prob_used"] >= prob_thr_eff)
    m_ev = m_prob & (df["EV_€_per_100"] > float(min_ev))

    print("\n=== UPCOMING FILTER EVAL (UNIQUE GAMES) ===")
    print(f"All upcoming unique games                         : {len(df)}")
    print(f"+ home_win_rate >= {params_used['home_win_rate_threshold']}                 : {int(m_hwr.sum())}")
    print(f"+ odds_1 >= {params_used['odds_min']}                           : {int(m_odmin.sum())}")
    print(f"+ odds_1 <= {params_used['odds_max']}                           : {int(m_odmax.sum())}")
    print(f"+ prob_used >= {prob_thr_eff} (eff threshold)           : {int(m_prob.sum())}")
    print(f"+ EV_€_per_100 > {min_ev}                             : {int(m_ev.sum())}")

    why = []
    for _, r in df.iterrows():
        if r[HOMEWR_COL] < float(params_used["home_win_rate_threshold"]):
            why.append(f"home_win_rate {r[HOMEWR_COL]:.2f} < threshold")
        elif r[HOME_ODDS_COL] < float(params_used["odds_min"]):
            why.append(f"odds {r[HOME_ODDS_COL]:.2f} < min {params_used['odds_min']}")
        elif r[HOME_ODDS_COL] > float(params_used["odds_max"]):
            why.append(f"odds {r[HOME_ODDS_COL]:.2f} > max {params_used['odds_max']}")
        elif r["prob_used"] < float(prob_thr_eff):
            why.append(f"prob_used {r['prob_used']:.2f} < prob_thr_eff")
        elif r["EV_€_per_100"] <= float(min_ev):
            why.append(f"EV {r['EV_€_per_100']:.2f} <= min_EV {min_ev}")
        else:
            why.append("QUALIFIES")

    df["why_not"] = why

    print("\n=== ALL UPCOMING GAMES & FILTER REASONS ===")
    print(f"n_trades (upcoming window) : {len(df)}")
    print(
        df[["date", "home_team", "away_team", HOMEWR_COL, ISO_COL, HOME_ODDS_COL, "why_not"]]
        .rename(columns={HOMEWR_COL: "home_win_rate", ISO_COL: "prob_iso", HOME_ODDS_COL: "odds_1"})
        .sort_values(["date", "home_team"])
        .round({"home_win_rate": 3, "prob_iso": 3, "odds_1": 3})
        .to_string(index=False)
    )


def build_flat_shortlist_today(
    upcoming_df: pd.DataFrame,
    params_used: dict,
    *,
    min_ev: float,
    flat_stake_live: float,
    prob_clip_lo: float,
    prob_clip_hi: float,
):
    if upcoming_df is None or upcoming_df.empty:
        return pd.DataFrame()

    _validate_params(params_used, name="params_used")

    df = upcoming_df.copy()
    date_col = "date" if "date" in df.columns else DATE_COL
    if "date" not in df.columns and DATE_COL in df.columns:
        df["date"] = df[DATE_COL]

    df = _ensure_datetime(df, date_col)
    df = _compute_prob_used(df, lo=prob_clip_lo, hi=prob_clip_hi, src=ISO_COL, dst="prob_used")
    df = _compute_ev_per_100(df, prob_col="prob_used", odds_col=HOME_ODDS_COL, stake_for_ev=100.0, dst="EV_€_per_100")
    df = _make_game_key(df, date_col=date_col, home_col="home_team", away_col="away_team", dst="game_key")

    df = (
        df.sort_values("EV_€_per_100", ascending=False)
          .drop_duplicates(subset="game_key", keep="first")
          .reset_index(drop=True)
    )

    prob_thr_eff = max(float(params_used["prob_threshold"]), float(prob_clip_lo))

    mask = (
        (df[HOMEWR_COL]    >= float(params_used["home_win_rate_threshold"])) &
        (df[HOME_ODDS_COL] >= float(params_used["odds_min"])) &
        (df[HOME_ODDS_COL] <= float(params_used["odds_max"])) &
        (df["prob_used"]   >= prob_thr_eff) &
        (df["EV_€_per_100"] >  float(min_ev))
    )

    picks = df.loc[mask].copy()
    if picks.empty:
        return pd.DataFrame()

    picks["stake_flat"] = float(flat_stake_live)
    picks["potential_profit_if_win"] = (picks["stake_flat"] * (picks[HOME_ODDS_COL] - 1.0)).round(2)
    picks["fair_odds"] = (1.0 / picks["prob_used"]).round(3)
    picks["edge_pct"] = ((picks[HOME_ODDS_COL] / picks["fair_odds"] - 1.0) * 100.0).round(2)
    picks["EV_€"] = ((picks["prob_used"] * (picks[HOME_ODDS_COL] - 1.0) - (1.0 - picks["prob_used"])) * picks["stake_flat"]).round(2)

    picks["home_win_rate"] = picks[HOMEWR_COL]
    picks["prob_iso"] = picks[ISO_COL]
    picks["odds_1"] = picks[HOME_ODDS_COL]

    return picks.sort_values("date").reset_index(drop=True)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD (default: today from nba_utils_2026).",
    )
    args = parser.parse_args()

    if args.date:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d")
        target_ymd = args.date
        logging.info("Using explicit --date: %s", target_ymd)
    else:
        now_dt, _, ymd_str = get_current_date()
        target_dt = now_dt
        target_ymd = ymd_str
        logging.info("No --date passed, using today's date from nba_utils_2026: %s", target_ymd)

    today_date = target_dt.date()
    tomorrow_date = (target_dt + timedelta(days=1)).date()

    paths = get_directory_paths()
    pred_dir = paths["PREDICTION_DIR"]
    out_dir = resolve_output_dir(paths["BASE_DIR"], pred_dir)
    kelly_dir = os.path.join(pred_dir, "Kelly")
    os.makedirs(kelly_dir, exist_ok=True)

    # 1) LOAD COMBINED
    df_all = load_combined_df(pred_dir, target_ymd)

    DATE_COL = "game_date"
    if DATE_COL not in df_all.columns:
        if "date" in df_all.columns:
            df_all[DATE_COL] = df_all["date"]
        else:
            raise KeyError(f"Neither '{DATE_COL}' nor 'date' exists. Columns: {list(df_all.columns)}")

    df_all = _ensure_datetime(df_all, DATE_COL)

    # 1b) COMPUTE HOME WIN RATES (before we merge future games)
    hwr_path = compute_home_win_rates(df_all, target_ymd, pred_dir)

    # 2) MERGE TODAY'S PREDICTIONS (if any) TO GET FUTURE GAMES INTO df_all
    df_all = merge_today_predictions(df_all, pred_dir, target_ymd, today_date)

    # 3) ATTACH HOME WIN RATE (file exists for sure)
    df_all = attach_home_win_rate(df_all, hwr_path)

    # 4) SPLIT PAST / FUTURE
    df_past, df_future = split_past_future(df_all, today_date, tomorrow_date)

    # 5) FIT ISOTONIC ON PAST, APPLY TO ALL (PAST + FUTURE)
    if df_past.empty:
        logging.warning("No past games available – cannot fit isotonic. Exiting.")
        print("=== Script 5 finished (no past games) ===")
        return

    iso = fit_isotonic(df_past)

    df_all[ISO_COL] = np.nan
    mask_iso = df_all[PRED_PROBA_COL].notna()
    df_all.loc[mask_iso, ISO_COL] = iso.transform(df_all.loc[mask_iso, PRED_PROBA_COL].astype(float).values)

    # Update df_past / df_future with ISO values
    df_past = df_all.loc[df_all.index.isin(df_past.index)].copy()
    df_future = df_all.loc[df_all.index.isin(df_future.index)].copy()

    brier_before, brier_after, logloss_before, logloss_after = compute_calibration_metrics(df_past)
    logging.info("Brier before: %.6f | after: %.6f", brier_before, brier_after)
    logging.info("Log-loss before: %.6f | after: %.6f", logloss_before, logloss_after)

    # 6) GRID SEARCH ON PAST (for documentation only)
    logging.info("Starting grid search...")
    best_params, df_grid = grid_search(df_past)
    best_metrics = evaluate_strategy(df_past, best_params)
    logging.info(
        "Best strategy (historical, ISO-based): %s | %d bets | flat profit %.2f | ROI per bet %.4f",
        best_params,
        best_metrics["n_bets"],
        best_metrics["total_profit"],
        best_metrics["roi_per_bet"],
    )

    # 7) SAVE GRID SEARCH + ISO DF
    grid_path = os.path.join(kelly_dir, f"nba_grid_search_results_{target_ymd}.csv")
    df_grid.to_csv(grid_path, index=False, encoding="utf-8")
    logging.info("Saved grid search results to %s", grid_path)

    iso_path = os.path.join(kelly_dir, f"combined_nba_predictions_iso_{target_ymd}.csv")
    df_all.to_csv(iso_path, index=False, encoding="utf-8")
    logging.info("Saved full dataframe with %s to %s", ISO_COL, iso_path)

    min_EV = MIN_EV_DEFAULT
    print("\nMin EV applied =", int(min_EV) if min_EV == int(min_EV) else min_EV)

    best_params_dict = _params_to_dict(best_params)

    best_params_local, _, roi_local_search, _, _ = find_best_local_params_lastN(
        df_past,
        homewr_grid=HOMEWR_MIN_GRID,
        odds_min_grid=ODDS_MIN_GRID,
        odds_max_grid=ODDS_MAX_GRID,
        prob_min_grid=PROB_MIN_GRID,
        flat_stake_backtest=FLAT_STAKE,
        min_ev=min_EV,
        min_trades_local=10,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
        window_n=LOCAL_SEARCH_N,
    )

    print_local_search_results(best_params_local, roi_local_search, window_n=LOCAL_SEARCH_N)

    hist_window_fair = _ensure_datetime(df_past, DATE_COL).sort_values(DATE_COL).tail(int(FAIR_COMPARE_N)).copy()
    if hist_window_fair is None or hist_window_fair.empty:
        USE_LOCAL = False
        params_used = best_params_dict.copy()
        metrics_global_N = None
        metrics_local_N = None
        subset_global_N = None
        subset_local_N = None
        print("\nNo hist window available for fair GLOBAL vs LOCAL comparison; using GLOBAL.")
    else:
        USE_LOCAL, params_used, metrics_global_N, metrics_local_N, subset_global_N, subset_local_N = choose_params_fair_lastN(
            best_params_dict,
            best_params_local,
            hist_window=hist_window_fair,
            min_ev=min_EV,
            flat_stake_backtest=FLAT_STAKE,
            prob_clip_lo=PROB_CLIP_LO,
            prob_clip_hi=PROB_CLIP_HI,
            min_trades=10,
            roi_edge_min_pp=0.0,
        )

        print(f"\n=== FAIR COMPARISON ON LAST {FAIR_COMPARE_N} (GLOBAL vs LOCAL) ===")
        print("GLOBAL lastN:", metrics_global_N)
        if metrics_local_N is not None:
            print("LOCAL  lastN:", metrics_local_N)
        else:
            print("LOCAL  lastN: None (no local params)")

    # params_used becomes the single source of truth for active filters + dashboard stats.
    matched_window_df = subset_local_N if USE_LOCAL else subset_global_N
    print_local_matched_games(matched_window_df, window_n=FAIR_COMPARE_N)

    matched_df = matched_window_df.copy() if matched_window_df is not None else pd.DataFrame()

    # --- COLUMN RESOLVER (robust for matched_df/local_matched_df schemas) ---
    BR_DATE_COL = "date" if "date" in matched_df.columns else ("game_date" if "game_date" in matched_df.columns else None)
    BR_PROB_COL = (
        "prob_iso"
        if "prob_iso" in matched_df.columns
        else ("iso_proba_home_win" if "iso_proba_home_win" in matched_df.columns else ("prob_used" if "prob_used" in matched_df.columns else None))
    )
    BR_ODDS_COL = "odds_1" if "odds_1" in matched_df.columns else ("closing_home_odds" if "closing_home_odds" in matched_df.columns else None)

    missing = [name for name, col in [("BR_DATE_COL", BR_DATE_COL), ("BR_PROB_COL", BR_PROB_COL), ("BR_ODDS_COL", BR_ODDS_COL)] if col is None]
    if missing:
        raise KeyError(
            "Missing required columns for bankroll calc. "
            f"Could not resolve: {missing}. Available columns: {list(matched_df.columns)}"
        )

    result_candidates = ["win", "home_team_won", "result", "pnl"]
    resolved_result_col = next((col for col in result_candidates if col in matched_df.columns), None)
    if resolved_result_col:
        historical_df = matched_df[matched_df[resolved_result_col].notna()].copy()
    else:
        historical_df = matched_df.copy()

    # --- BANKROLL SECTION 1: Last 200 games ---
    last_200_df = historical_df.tail(200).copy()
    bankroll_window = 1000  # 1000€ Deposit
    flat_stake = 100        # 100€ flat stake

    for _, row in last_200_df.iterrows():
        prob = row[BR_PROB_COL]
        odds = row[BR_ODDS_COL]
        bankroll_window += flat_stake * (prob * (odds - 1) - (1 - prob))

    bet_log_path = find_bet_log_path(out_dir)
    settled_bets_df = pd.DataFrame()
    if bet_log_path and bet_log_path.exists():
        bet_log_df = pd.read_csv(bet_log_path)
        settled_bets_df = build_settled_bets(bet_log_df, df_past)

    if not settled_bets_df.empty:
        settled_bets_df["date"] = pd.to_datetime(settled_bets_df["date"], errors="coerce")
        settled_2026_df = settled_bets_df[settled_bets_df["date"].dt.year == 2026].copy()
    else:
        settled_2026_df = pd.DataFrame()

    profit_2026 = float(settled_2026_df["pnl"].sum()) if not settled_2026_df.empty else 0.0
    bankroll_2026 = float(START_BANKROLL) + profit_2026
    stake_sum_2026 = float(settled_2026_df["stake"].sum()) if not settled_2026_df.empty else 0.0
    settled_summary = {
        "count": int(len(settled_2026_df)),
        "wins": int(settled_2026_df["win"].sum()) if not settled_2026_df.empty else 0,
        "profit_sum": round(profit_2026, 2),
        "roi": round(profit_2026 / stake_sum_2026, 4) if stake_sum_2026 else 0.0,
        "avg_odds": round(float(settled_2026_df["odds_1"].mean()), 2) if not settled_2026_df.empty else 0.0,
    }

    print("\n=== APPLIED FILTER VALUES ===")
    print("Window size used for bankroll calculation        : 200")
    print("Initial deposit 2026                            : 1000 €")
    print("Flat stake per game (2026)                      : 100 €")
    print(f"Bankroll result for 2026 YTD                    : {profit_2026:.2f} €")

    if matched_window_df is None or matched_window_df.empty:
        local_matched_df = pd.DataFrame()
    else:
        local_matched_df = prepare_local_matched_export(matched_window_df, stake=FLAT_STAKE)
    export_dir = out_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    local_export_path = export_dir / f"local_matched_df_export_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    local_matched_df.to_csv(local_export_path, index=False)
    print(f"\nCSV Export saved: {local_export_path}")
    as_of_date = _as_of_date_from_df(df_past, fallback=target_ymd)
    snapshot = build_metrics_snapshot(
        local_matched_df,
        params_used=params_used,
        params_used_type="LOCAL" if USE_LOCAL else "GLOBAL",
        min_ev=min_EV,
        as_of_date=as_of_date,
        stake=FLAT_STAKE,
        bankroll_window=bankroll_window,
        bankroll_2026=bankroll_2026,
        profit_2026=profit_2026,
        settled_summary=settled_summary,
    )
    write_metrics_snapshot(snapshot, out_dir)
    write_strategy_params(
        params_used,
        min_ev=min_EV,
        as_of_date=as_of_date,
        stake=FLAT_STAKE,
        output_dir=out_dir,
    )
    export_path = export_local_matched_games_settled(
        local_matched_df,
        output_dir=out_dir,
        as_of_date=as_of_date,
    )
    if local_matched_df is not None and not local_matched_df.empty:
        check_metrics_snapshot_consistency(local_matched_df, out_dir)
    update_last_run_trace(local_matched_df, export_path, snapshot, settled_2026_df)

    upcoming_filter_eval_and_reasons(
        upcoming_df=df_future,
        params_used=params_used,
        min_ev=min_EV,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
    )

    print("\n=== TODAY'S FLAT-STAKE SHORTLIST (GLOBAL/LOCAL PARAMS + EV>min_EV) ===")

    flat_today = build_flat_shortlist_today(
        upcoming_df=df_future,
        params_used=params_used,
        min_ev=min_EV,
        flat_stake_live=FLAT_STAKE,
        prob_clip_lo=PROB_CLIP_LO,
        prob_clip_hi=PROB_CLIP_HI,
    )

    if flat_today.empty:
        print("No games meet the combined strategy + EV>min_EV for flat staking today.")
    else:
        cols_out = [
            "date", "home_team", "away_team", "home_win_rate",
            "prob_iso", "prob_used", "odds_1", "EV_€_per_100",
            "stake_flat", "EV_€", "potential_profit_if_win", "fair_odds", "edge_pct",
        ]
        for c in cols_out:
            if c not in flat_today.columns:
                flat_today[c] = np.nan

        print(
            flat_today[cols_out]
            .sort_values("date")
            .round({
                "home_win_rate": 3,
                "prob_iso": 3,
                "prob_used": 3,
                "odds_1": 3,
                "EV_€_per_100": 2,
                "stake_flat": 2,
                "EV_€": 2,
                "potential_profit_if_win": 2,
                "fair_odds": 3,
                "edge_pct": 1,
            })
            .to_string(index=False)
        )

        shortlist_path = os.path.join(pred_dir, f"bet_shortlist_{target_ymd}.csv")
        flat_today.to_csv(shortlist_path, index=False, encoding="utf-8")
        logging.info("Saved bet shortlist (%d rows) to %s", len(flat_today), shortlist_path)

        n_games = len(flat_today)
        tot_stake = n_games * float(FLAT_STAKE)

        print(f"\nGames selected      : {n_games}")
        print(f"Flat stake / game   : {float(FLAT_STAKE):.2f} €")
        print(f"Total stake today   : {tot_stake:.2f} €")
        print(f"Total EV today (€)  : {flat_today['EV_€'].sum():.2f} €")

    print("=== Script 5 finished ===")
    print("\n=== BANKROLL SUMMARY (BOTTOM SECTION) ===")
    print(f"Bankroll (Last 200 games window)   : {bankroll_window:.2f} €")
    print(f"Bankroll (2026 YTD ONLY)           : {bankroll_2026:.2f} €")


if __name__ == "__main__":
    main()
