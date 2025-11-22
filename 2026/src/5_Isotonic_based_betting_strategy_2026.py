#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
5_Isotonic_based_betting_strategy_2026.py

Step 5 of the 2026 pipeline:

1) Load combined historical predictions:
       combined_nba_predictions_acc_YYYY-MM-DD.csv
   from 2026/output/LightGBM

2) Make sure the following columns exist (creating them if necessary):
   - game_date           (from 'date')
   - home_team_won       (from 'result' == 'home_team', only for played games)
   - pred_home_win_proba (from 'home_team_prob')
   - closing_home_odds   (from 'odds_1')
   - closing_away_odds   (from 'odds_2')

3) Optionally merge in:
   - tonight's predictions from nba_games_predict_YYYY-MM-DD.csv
   - home win rates from home_win_rates_sorted_YYYY-MM-DD.csv

4) Split into:
   - df_past   (played games, for calibration + grid search)
   - df_future (upcoming games today/tomorrow, for shortlist)

5) Fit an Isotonic Regression on df_past and compute:
   - iso_proba_home_win

6) Run a grid search over a small parameter space:
   StrategyParams(min_home_win_rate, min_odds, max_odds, min_iso_proba)

   Backtest on df_past with flat stakes, and pick the best combo
   (by ROI per bet, tie-breaking by number of bets).

7) Save:
   - Kelly/nba_grid_search_results_YYYY-MM-DD.csv
   - Kelly/combined_nba_predictions_iso_YYYY-MM-DD.csv

8) Apply *LOCAL-STYLE* SHORTLIST FILTERS (fixed, not from grid search):
   - home_win_rate    >= 0.50
   - iso_proba_home_win >= 0.45
   - closing_home_odds between 1.10 and 3.40

   For all qualifying games:
   - compute prob_iso, prob_used (capped at 0.75), EV, Kelly, stake from bankroll
   - save to bet_shortlist_YYYY-MM-DD.csv
   - print a clear shortlist summary to stdout (GitHub Actions log)

If there are no upcoming games OR no suitable bets, script still succeeds.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

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

# Grid search (historical backtest) — same as vorher
FLAT_STAKE = 100.0
ODDS_MIN_GRID = [1.10, 1.25, 1.40, 1.60]
ODDS_MAX_GRID = [2.00, 2.10, 2.50, 3.00]
PROB_MIN_GRID = [0.55, 0.60, 0.65, 0.70]
HOMEWR_MIN_GRID = [0.50, 0.55, 0.60, 0.65]

# SHORTLIST FILTERS — "genau wie lokal"
MIN_HOME_WIN_RATE_SHORTLIST = 0.50
MIN_ISO_PROBA_SHORTLIST = 0.45
MIN_ODDS_SHORTLIST = 1.10
MAX_ODDS_SHORTLIST = 3.40

# Kelly / Bankroll
START_BANKROLL = 1000.0
MAX_KELLY_FRACTION = 0.10  # cap at 10 % vom Bankroll
MAX_PROB_USED = 0.75       # cap für prob_used = min(iso_proba, 0.75)


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
        format="[%(asctime)s] %(levelname)s: %(message)s",
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

    df = pd.read_csv(path, encoding="utf-7")

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
        src = None
        if "home_team_prob" in df.columns:
            src = "home_team_prob"

        if src is not None:
            logging.info(
                "PRED_PROBA_COL 'pred_home_win_proba' not in dataframe – creating it from '%s'.",
                src,
            )
            df[PRED_PROBA_COL] = to_float_series(df[src])
        else:
            logging.warning(
                "No suitable probability column found. Setting pred_home_win_proba to NaN."
            )
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

    # RESULT_COL (but we must NOT mark future games as 0-loss)
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
        logging.info(
            "No TODAY_PRED file found (%s) – skipping merge of upcoming games.",
            today_pred_path,
        )
        return df_all

    logging.info("Merging upcoming games from %s", today_pred_path)

    tmp = pd.read_csv(
        today_pred_path,
        encoding="utf-7",
        sep=",",
        quotechar='"',
        decimal=",",
    )

    # If schema is weird, fallback to manual header
    expected = {"home_team", "away_team", "home_team_prob"}
    norm_cols = {c.lower().strip() for c in tmp.columns}
    if not expected.issubset(norm_cols):
        tmp = pd.read_csv(
            today_pred_path,
            encoding="utf-7",
            sep=",",
            quotechar='"',
            decimal=",",
            header=None,
            names=[
                "home_team",
                "away_team",
                "home_team_prob",
                "odds_1",
                "odds_2",
                "result",
                "date",
            ],
        )

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

    # ensure these exist
    if "result" not in tmp.columns:
        tmp["result"] = np.nan

    tmp[DATE_COL] = pd.to_datetime(tmp["date"], errors="coerce")

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


def attach_home_win_rate(
    df: pd.DataFrame,
    hwr_path: str,
) -> pd.DataFrame:
    """
    Attach home win rate (HOMEWR_COL) to df based on the
    home_win_rates_sorted_YYYY-MM-DD.csv file.
    """
    if not os.path.exists(hwr_path):
        logging.warning(
            "Home win rate file not found at %s; skipping merge.",
            hwr_path,
        )
        return df

    try:
        hwr = pd.read_csv(
            hwr_path,
            encoding="utf-7",
            sep=",",
            decimal=",",
        )
    except Exception as e:
        logging.warning(
            "Failed to read home win rate file %s: %s",
            hwr_path,
            e,
        )
        return df

    if hwr.empty:
        logging.warning(
            "Home win rate file %s is empty; skipping merge.",
            hwr_path,
        )
        return df

    cols = list(hwr.columns)
    cols_lower = [c.lower().strip() for c in cols]

    team_col = None
    winrate_col = None

    # --- SPECIAL CASE: current 4-column format ---
    if len(cols) == 4 and "home win rate" in cols_lower:
        team_col = cols[0]
        winrate_col = cols[cols_lower.index("home win rate")]
        logging.info(
            "Detected home-win-rate file format with 4 columns; "
            "using '%s' as team column and '%s' as win-rate column.",
            team_col,
            winrate_col,
        )
    else:
        # Generic detection (future proof)
        lower_to_orig = {c.lower().strip(): c for c in cols}

        # team column by name
        for lc, orig in lower_to_orig.items():
            if lc in {"team", "home_team", "team_code"}:
                team_col = orig
                break
            if "team" in lc and ("abbr" in lc or "code" in lc or "home" in lc):
                team_col = orig
                break

        # winrate column by name
        for lc, orig in lower_to_orig.items():
            if "home_win_rate" in lc or "home win rate" in lc or "win_rate" in lc:
                winrate_col = orig
                break

        # fallback for win-rate: numeric column in [0,1]
        if winrate_col is None:
            for c in cols:
                try:
                    vals = pd.to_numeric(hwr[c], errors="coerce")
                    if vals.notna().sum() == 0:
                        continue
                    frac_between = (
                        ((vals >= 0.0) & (vals <= 1.0)).sum()
                        / vals.notna().sum()
                    )
                    if frac_between > 0.9:
                        winrate_col = c
                        break
                except Exception:
                    continue

        # last fallback for team column
        if team_col is None:
            for c in cols:
                sample = (
                    hwr[c].dropna().astype(str).str.strip().head(20).tolist()
                )
                if not sample:
                    continue
                if all(len(x) <= 4 for x in sample) and all(x.upper() == x for x in sample):
                    team_col = c
                    break

        if team_col is None or winrate_col is None:
            logging.warning(
                "Could not identify team and/or win-rate columns in %s; "
                "cols=%s – skipping merge.",
                hwr_path,
                cols,
            )
            return df

        logging.info(
            "Using '%s' as team column and '%s' as win-rate column.",
            team_col,
            winrate_col,
        )

    # --- normalize team codes and merge ---
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

    # cleanup
    df.drop(columns=["_team_norm", "_home_team_norm"], inplace=True, errors="ignore")

    df[HOMEWR_COL] = pd.to_numeric(df[HOMEWR_COL], errors="coerce").fillna(0.0)

    logging.info(
        "Merged home win rates into dataframe; %d rows with non-null values.",
        df[HOMEWR_COL].notna().sum(),
    )

    return df


def split_past_future(
    df_all: pd.DataFrame,
    today_date,
    tomorrow_date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Past = games where RESULT_RAW_COL is non-null and != '0'
    Future = games where RESULT_RAW_COL is null or '0' and
             game_day in {today, tomorrow}
    """
    df_all[DATE_COL] = pd.to_datetime(df_all[DATE_COL], errors="coerce")
    df_all["game_day"] = df_all[DATE_COL].dt.date

    if RESULT_RAW_COL not in df_all.columns:
        df_all[RESULT_RAW_COL] = np.nan

    played_mask = df_all[RESULT_RAW_COL].notna() & (
        df_all[RESULT_RAW_COL].astype(str) != "0"
    )

    df_past = df_all[played_mask].copy()
    df_future = df_all[
        ~played_mask & df_all["game_day"].isin([today_date, tomorrow_date])
    ].copy()

    logging.info(
        "Split into %d past games and %d future games.",
        len(df_past),
        len(df_future),
    )
    return df_past, df_future


def fit_isotonic(df_past: pd.DataFrame) -> IsotonicRegression:
    """
    Fit isotonic regression on past games.
    """
    mask = df_past[RESULT_COL].notna() & df_past[PRED_PROBA_COL].notna()
    if mask.sum() == 0:
        raise RuntimeError(
            "No valid rows to fit isotonic regression "
            "(missing y_true or probabilities)."
        )

    y_true = df_past.loc[mask, RESULT_COL].astype(int).values
    p_raw = df_past.loc[mask, PRED_PROBA_COL].astype(float).values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_true)

    logging.info("Isotonic fitted on %d games.", mask.sum())
    return iso


def compute_calibration_metrics(df_past: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Brier score + log loss before/after isotonic.
    """
    mask = (
        df_past[RESULT_COL].notna()
        & df_past[PRED_PROBA_COL].notna()
        & df_past[ISO_COL].notna()
    )
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
    """
    Evaluate one parameter combo on a backtest dataframe (df_past).
    Flat stake 100€ per bet.
    """
    if df.empty:
        return {
            "n_bets": 0,
            "total_profit": 0.0,
            "roi_per_bet": 0.0,
        }

    conds = []

    # home win rate filter
    if HOMEWR_COL in df.columns and pd.api.types.is_numeric_dtype(df[HOMEWR_COL]):
        conds.append(df[HOMEWR_COL] >= params.min_home_win_rate)

    # odds range
    conds.append(df[HOME_ODDS_COL].between(params.min_odds, params.max_odds))

    # probability threshold (ISO based)
    conds.append(df[ISO_COL] >= params.min_iso_proba)

    # valid rows: need closing_home_odds + iso prob + result
    conds.append(df[HOME_ODDS_COL].notna())
    conds.append(df[ISO_COL].notna())
    conds.append(df[RESULT_COL].notna())

    mask = np.logical_and.reduce(conds)

    df_sel = df[mask].copy()
    n_bets = len(df_sel)
    if n_bets == 0:
        return {
            "n_bets": 0,
            "total_profit": 0.0,
            "roi_per_bet": 0.0,
        }

    stake = FLAT_STAKE
    df_sel["profit"] = np.where(
        df_sel[RESULT_COL].astype(int) == 1,
        (df_sel[HOME_ODDS_COL] - 1.0) * stake,
        -stake,
    )

    total_profit = float(df_sel["profit"].sum())
    roi_per_bet = total_profit / (n_bets * stake)

    return {
        "n_bets": n_bets,
        "total_profit": total_profit,
        "roi_per_bet": roi_per_bet,
    }


def grid_search(
    df_past: pd.DataFrame,
) -> Tuple[StrategyParams, pd.DataFrame]:
    """
    Try a small grid of StrategyParams and pick the best by ROI per bet.
    Tie-breaker: more bets.
    """
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

    # pick best by ROI; tie-break by number of bets
    df_res = df_res.sort_values(
        by=["roi_per_bet", "n_bets"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_row = df_res.iloc[0]
    best_params = StrategyParams(
        min_home_win_rate=float(best_row["min_home_win_rate"]),
        min_odds=float(best_row["min_odds"]),
        max_odds=float(best_row["max_odds"]),
        min_iso_proba=float(best_row["min_iso_proba"]),
    )

    return best_params, df_res

def build_shortlist_local_style(df_future: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """
    Baut Shortlist EXACT wie im lokalen Notebook:
    - prob_iso (Isotonic-Proba)
    - prob_used = max(prob_iso, raw_pred)
    - EV pro 100€
    - Kelly full + Kelly fraction (max 0.1)
    - stake in €
    - expected profit €
    - fair odds + edge %
    """

    if df_future.empty:
        return pd.DataFrame()

    df = df_future.copy()

    # RAW prob + ISO prob
    df["prob_iso"] = df[ISO_COL]
    df["prob_raw"] = df[PRED_PROBA_COL]

    # prob_used = max(prob_iso, prob_raw)
    df["prob_used"] = df[["prob_iso", "prob_raw"]].max(axis=1)

    # EV pro Einheit (1€)
    df["ev_per_unit"] = (
        df["prob_used"] * (df[HOME_ODDS_COL] - 1.0)
        - (1.0 - df["prob_used"])
    )

    # EV pro 100€
    df["EV_€_per_100"] = df["ev_per_unit"] * 100.0

    # Kelly full
    df["kelly_full"] = (
        (df[HOME_ODDS_COL] * df["prob_used"] - (1 - df["prob_used"]))
        / (df[HOME_ODDS_COL] - 1)
    )

    # Kelly fraction (max 10% of stake)
    df["kelly_fraction_used"] = df["kelly_full"].clip(lower=0, upper=0.10)

    # Stake
    df["stake_eur"] = bankroll * df["kelly_fraction_used"]

    # Expected profit
    df["exp_profit_eur"] = df["stake_eur"] * df["ev_per_unit"]

    # Fair odds
    df["fair_odds"] = 1.0 / df["prob_used"].clip(lower=1e-9)

    # Edge %
    df["edge_pct"] = (df["fair_odds"] / df[HOME_ODDS_COL] - 1.0) * 100.0

    # Sort wie lokal
    df = df.sort_values("exp_profit_eur", ascending=False)

    # Lokale Filter:
    mask = (
        (df[HOMEWR_COL] >= MIN_HOME_WIN_RATE_SHORTLIST) &
        (df["prob_iso"] >= MIN_ISO_PROBA_SHORTLIST) &
        (df[HOME_ODDS_COL].between(MIN_ODDS_SHORTLIST, MAX_ODDS_SHORTLIST))
    )

    shortlist = df[mask].copy()

    return shortlist

# -------------------------------------------------------------------------
# BANKROLL / KELLY
# -------------------------------------------------------------------------

def load_current_bankroll(pred_dir: str) -> float:
    """
    Try to load current bankroll from bet_log_live.csv in pred_dir.
    Fallback: START_BANKROLL if file/columns are missing.

    Heuristics:
      - if 'bankroll_after' or 'cum_bankroll' column exists -> last non-null
      - elif 'profit' column exists -> START_BANKROLL + sum(profit)
      - else -> START_BANKROLL
    """
    path = os.path.join(pred_dir, "bet_log_live.csv")
    bankroll = START_BANKROLL

    if not os.path.exists(path):
        logging.info(
            "No bet_log_live.csv found at %s – using start bankroll %.2f €.",
            path,
            bankroll,
        )
        return bankroll

    try:
        df_log = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        logging.warning(
            "Failed to read %s: %s – using start bankroll %.2f €.",
            path,
            e,
            bankroll,
        )
        return bankroll

    if df_log.empty:
        logging.info(
            "bet_log_live.csv is empty – using start bankroll %.2f €.",
            bankroll,
        )
        return bankroll

    for col in ["bankroll_after", "cum_bankroll"]:
        if col in df_log.columns:
            vals = pd.to_numeric(df_log[col], errors="coerce")
            if vals.notna().any():
                bankroll = float(vals.dropna().iloc[-1])
                logging.info(
                    "Using bankroll from column '%s' in bet_log_live.csv: %.2f €",
                    col,
                    bankroll,
                )
                return bankroll

    if "profit" in df_log.columns:
        prof = pd.to_numeric(df_log["profit"], errors="coerce").fillna(0.0)
        bankroll = float(START_BANKROLL + prof.sum())
        logging.info(
            "Using bankroll = START(%.2f) + sum(profit)=%.2f -> %.2f €",
            START_BANKROLL,
            prof.sum(),
            bankroll,
        )
        return bankroll

    logging.info(
        "No bankroll/profit columns found in bet_log_live.csv – using start bankroll %.2f €.",
        bankroll,
    )
    return bankroll


def apply_kelly_and_ev(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """
    Add prob_iso, prob_used, EV, Kelly, stake, expected profit, fair odds, edge.
    Uses ISO probability capped at MAX_PROB_USED as prob_used (wie lokal).
    """
    if df.empty:
        return df

    df = df.copy()

    # prob_iso = ISO_COL
    df["prob_iso"] = df[ISO_COL].astype(float)

    # prob_used = min(prob_iso, MAX_PROB_USED)
    df["prob_used"] = df["prob_iso"].clip(upper=MAX_PROB_USED)

    b = df[HOME_ODDS_COL].astype(float) - 1.0  # "b" in Kelly-Formel
    p = df["prob_used"]
    q = 1.0 - p

    # EV per 1€ Einsatz
    df["ev_per_unit"] = p * b - q
    df["EV_€_per_100"] = df["ev_per_unit"] * 100.0

    # Kelly full fraction
    # k = ev_per_unit / b, aber b kann 0 sein (odds=1) -> dann 0
    df["kelly_full"] = np.where(
        b > 0,
        df["ev_per_unit"] / b,
        0.0,
    )

    # only positive Kelly, capped at MAX_KELLY_FRACTION
    df["kelly_fraction_used"] = np.where(
        df["kelly_full"] > 0,
        np.minimum(df["kelly_full"], MAX_KELLY_FRACTION),
        0.0,
    )

    df["stake_eur"] = df["kelly_fraction_used"] * float(bankroll)
    df["exp_profit_eur"] = df["ev_per_unit"] * df["stake_eur"]

    # Fair odds + edge in %
    df["fair_odds"] = np.where(
        df["prob_used"] > 0,
        1.0 / df["prob_used"],
        np.nan,
    )
    df["edge_pct"] = (df[HOME_ODDS_COL] / df["fair_odds"] - 1.0) * 100.0

    return df


# -------------------------------------------------------------------------
# SHORTLIST (LOCAL-STYLE FILTERS)
# -------------------------------------------------------------------------

def log_future_games_with_reasons(df_future: pd.DataFrame) -> None:
    """
    Loggt alle zukünftigen Spiele mit den Gründen, warum sie
    die LOCAL-Shortlist-Filter nicht schaffen (oder 'QUALIFIES').
    Ausgabe ähnlich dem lokalen Script:

    === ALL UPCOMING GAMES & FILTER REASONS ===
          date home_team away_team  home_win_rate  prob_iso  odds_1                           why_not
    2025-11-22       CHI       WAS           0.71     0.364    1.14              prob_iso 0.36 < 0.45
    ...
    """
    if df_future.empty:
        return

    rows = []
    for _, r in df_future.iterrows():
        date_val = r.get(DATE_COL, pd.NaT)
        home = r.get("home_team", "")
        away = r.get("away_team", "")
        hwr = r.get(HOMEWR_COL, np.nan)
        iso = r.get(ISO_COL, np.nan)
        odds = r.get(HOME_ODDS_COL, np.nan)

        reasons = []

        # Home-win-rate-Filter
        if pd.notna(hwr) and hwr < MIN_HOME_WIN_RATE_SHORTLIST:
            reasons.append(f"home_win_rate {hwr:.2f} < {MIN_HOME_WIN_RATE_SHORTLIST}")

        # Odds-Bereich
        if pd.notna(odds):
            if odds < MIN_ODDS_SHORTLIST or odds > MAX_ODDS_SHORTLIST:
                reasons.append(f"odds {odds:.2f} not in [{MIN_ODDS_SHORTLIST}, {MAX_ODDS_SHORTLIST}]")

        # ISO-Filter
        if pd.notna(iso) and iso < MIN_ISO_PROBA_SHORTLIST:
            reasons.append(f"prob_iso {iso:.2f} < {MIN_ISO_PROBA_SHORTLIST}")

        why_not = "QUALIFIES" if not reasons else "; ".join(reasons)

        rows.append(
            {
                "date": date_val.date() if isinstance(date_val, pd.Timestamp) else date_val,
                "home_team": home,
                "away_team": away,
                HOMEWR_COL: hwr,
                "prob_iso": iso,
                HOME_ODDS_COL: odds,
                "why_not": why_not,
            }
        )

    df_reasons = pd.DataFrame(rows)

    print("=== ALL UPCOMING GAMES & FILTER REASONS ===")
    with pd.option_context(
        "display.width", 160,
        "display.max_columns", None,
        "display.max_rows", None,
        "display.float_format", lambda x: f"{x:0.3f}",
    ):
        print(df_reasons)
        print()



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
        now_dt, friendly, ymd_str = get_current_date()
        target_dt = now_dt
        target_ymd = ymd_str
        logging.info(
            "No --date passed, using today's date from nba_utils_2026: %s",
            target_ymd,
        )

    today_date = target_dt.date()
    tomorrow_date = (target_dt + timedelta(days=1)).date()

    # PATHS
    paths = get_directory_paths()
    pred_dir = paths["PREDICTION_DIR"]
    kelly_dir = os.path.join(pred_dir, "Kelly")
    os.makedirs(kelly_dir, exist_ok=True)

    # 1) LOAD COMBINED
    df_all = load_combined_df(pred_dir, target_ymd)

    # 2) MERGE TODAY'S PREDICTIONS (if any) TO GET FUTURE GAMES INTO df_all
    df_all = merge_today_predictions(df_all, pred_dir, target_ymd, today_date)

    # 3) ATTACH HOME WIN RATE IF AVAILABLE
    hwr_path = os.path.join(
        pred_dir,
        f"home_win_rates_sorted_{target_ymd}.csv",
    )
    df_all = attach_home_win_rate(df_all, hwr_path)

    # 4) SPLIT PAST / FUTURE
    df_past, df_future = split_past_future(df_all, today_date, tomorrow_date)

    # 5) FIT ISOTONIC ON PAST, APPLY TO ALL (PAST + FUTURE)
    if df_past.empty:
        logging.warning("No past games available – cannot fit isotonic. Exiting.")
        print("=== Script 5 finished (no past games for isotonic) ===")
        return

    iso = fit_isotonic(df_past)

    df_all[ISO_COL] = np.nan
    mask_iso = df_all[PRED_PROBA_COL].notna()
    df_all.loc[mask_iso, ISO_COL] = iso.transform(
        df_all.loc[mask_iso, PRED_PROBA_COL].astype(float).values
    )

    # For metrics we need ISO on df_past too
    df_past = df_all.loc[df_all.index.isin(df_past.index)].copy()

    brier_before, brier_after, logloss_before, logloss_after = compute_calibration_metrics(df_past)
    logging.info("Brier before: %.6f | after: %.6f", brier_before, brier_after)
    logging.info("Log-loss before: %.6f | after: %.6f", logloss_before, logloss_after)

    # 6) GRID SEARCH ON PAST (historical evaluation, not for shortlist filters)
    logging.info("Starting grid search...")
    best_params, df_grid = grid_search(df_past)
    logging.info(
        "Best strategy (historical, ISO-based): %s | %d bets | flat profit %.2f | ROI per bet %.4f",
        best_params,
        evaluate_strategy(df_past, best_params)["n_bets"],
        evaluate_strategy(df_past, best_params)["total_profit"],
        evaluate_strategy(df_past, best_params)["roi_per_bet"],
    )

    # 7) SAVE GRID SEARCH + ISO DF (full)
    grid_path = os.path.join(kelly_dir, f"nba_grid_search_results_{target_ymd}.csv")
    df_grid.to_csv(grid_path, index=False, encoding="utf-8")
    logging.info("Saved grid search results to %s", grid_path)

    # add prob_iso column to full dataframe for consistency
    df_all["prob_iso"] = df_all[ISO_COL]

    iso_path = os.path.join(kelly_dir, f"combined_nba_predictions_iso_{target_ymd}.csv")
    df_all.to_csv(iso_path, index=False, encoding="utf-8")
    logging.info("Saved full dataframe with %s to %s", ISO_COL, iso_path)

    # 8) BUILD SHORTLIST FOR FUTURE GAMES (TODAY / TOMORROW) – LOCAL STYLE
    if df_future.empty:
        logging.info("No future games found in file – nothing to bet on today.")
        print("=== Script 5 finished (no future games) ===")
        return

    # Re-sync future rows to include ISO + HWR etc.
    # Re-sync future rows to include ISO + HWR etc.
    df_future = df_all.loc[df_all.index.isin(df_future.index)].copy()

    # Alle zukünftigen Spiele + Gründe loggen (wie lokal)
    log_future_games_with_reasons(df_future)

    # Load bankroll from live bet log
    bankroll = load_current_bankroll(pred_dir)
    print(f"\n💰 Current bankroll for sizing bets: {bankroll:.2f} €\n")

    # Apply local shortlist filters + Kelly sizing
    shortlist = build_shortlist_local_style(df_future, bankroll)


    if shortlist.empty:
        logging.info(
            "Future games found but no bets passed LOCAL filters – empty shortlist for today."
        )
        print("=== TONIGHT'S SHORTLIST (ISOTONIC + KELLY) ===")
        print("No games passed the local filters today.\n")
        print("=== Script 5 finished ===")
        return

    # Save shortlist to standard LightGBM output folder
    shortlist_path = os.path.join(pred_dir, f"bet_shortlist_{target_ymd}.csv")
    shortlist.to_csv(shortlist_path, index=False, encoding="utf-8")
    logging.info("Saved bet shortlist (%d rows) to %s", len(shortlist), shortlist_path)

    # Pretty print for GitHub Actions log (ähnlich lokal)
    display_cols = [
        "game_date",
        "home_team",
        "away_team",
        HOMEWR_COL,
        "prob_iso",
        "prob_used",
        HOME_ODDS_COL,
        "EV_€_per_100",
        "kelly_full",
        "kelly_fraction_used",
        "stake_eur",
        "exp_profit_eur",
        "fair_odds",
        "edge_pct",
    ]
    for c in display_cols:
        if c not in shortlist.columns:
            # skip missing columns (zur Sicherheit)
            display_cols.remove(c)

    print("=== TONIGHT'S SHORTLIST (ISOTONIC + KELLY) ===")
    # Round some numeric columns for nice log output
    shortprint = shortlist.copy()
    for col in [
        HOMEWR_COL,
        "prob_iso",
        "prob_used",
        HOME_ODDS_COL,
        "EV_€_per_100",
        "kelly_full",
        "kelly_fraction_used",
        "stake_eur",
        "exp_profit_eur",
        "fair_odds",
        "edge_pct",
    ]:
        if col in shortprint.columns:
            shortprint[col] = pd.to_numeric(shortprint[col], errors="coerce")

    with pd.option_context(
        "display.width", 160,
        "display.max_columns", None,
        "display.max_rows", None,
        "display.float_format", lambda x: f"{x:0.3f}",
    ):
        print(shortprint[display_cols])
        print()

    print("=== TONIGHT'S SHORTLIST (SAVE SNAPSHOT) ===")
    print(f"💾 Saved shortlist to {shortlist_path}\n")
    print("=== Script 5 finished ===")


if __name__ == "__main__":
    main()
