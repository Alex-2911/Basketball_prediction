#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
5_kelly_isotonic_and_shortlist_2026.py

Step 5 of the 2026 pipeline:
- Load combined_nba_predictions_acc_{DATE}.csv from PREDICTION_DIR
- Fit Isotonic Regression calibration on past games
- Run a grid search over filter params (home win rate, odds, iso prob)
- Evaluate flat-stake + Kelly P&L historically
- Pick best strategy
- Save:
    - nba_grid_search_results_{DATE}.csv
    - combined_nba_predictions_iso_{DATE}.csv  (adds iso_proba_home_win)
    - nba_bets_shortlist_{DATE}.csv           (recommended bets for future games)
    - nba_bets_why_not_{DATE}.csv             (diagnostics: why each game is NOT a bet)
All paths are taken from nba_utils_2026.get_directory_paths().
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

# Import shared utils
from nba_utils_2026 import get_directory_paths, get_current_date

# -----------------------------------------------------------------------------
# Column configuration – ADJUST HERE IF YOUR FILE CHANGES
# -----------------------------------------------------------------------------

GAME_ID_COL = "game_id"
DATE_COL = "game_date"
HOME_TEAM_COL = "home_team"
AWAY_TEAM_COL = "away_team"
PRED_PROBA_COL = "pred_home_win_proba"      # model prob for home win
RESULT_COL = "home_team_won"                # 1/0, NaN for future
HOME_ODDS_COL = "closing_home_odds"         # decimal odds for home
AWAY_ODDS_COL = "closing_away_odds"         # optional
HOME_WIN_RATE_COL: Optional[str] = "home_team_win_rate_last_20"  # set to None to disable
PRED_PROBA_COL = "pred_home_win_proba"


# -----------------------------------------------------------------------------
# Kelly & grid search configuration
# -----------------------------------------------------------------------------

INITIAL_BANKROLL = 1000.0
KELLY_CAP = 0.10  # max % of bankroll per bet

MIN_HOME_WIN_RATE_LIST = [0.50, 0.55, 0.60, 0.65]
MIN_ODDS_LIST = [1.20, 1.25, 1.30, 1.35]
MAX_ODDS_LIST = [1.70, 1.80, 1.90, 2.10]
MIN_ISO_PROBA_LIST = [0.55, 0.60, 0.65, 0.70]


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class StrategyParams:
    min_home_win_rate: float
    min_odds: float
    max_odds: float
    min_iso_proba: float


@dataclass
class StrategyResult:
    params: StrategyParams
    n_bets: int
    n_wins: int
    total_profit_flat: float
    roi_flat: float
    total_profit_kelly: float
    roi_kelly: float
    brier_before: float
    brier_after: float
    logloss_before: float
    logloss_after: float


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5/5 – Isotonic calibration + Kelly optimization + shortlist export (2026)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Prediction date in YYYY-MM-DD (default: today via nba_utils_2026.get_current_date)",
    )
    parser.add_argument(
        "--input-filename-template",
        type=str,
        default="combined_nba_predictions_acc_{date}.csv",
        help="Template for input file name inside PREDICTION_DIR (must contain '{date}')",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=INITIAL_BANKROLL,
        help=f"Initial bankroll for Kelly simulation (default: {INITIAL_BANKROLL})",
    )
    return parser.parse_args()


def load_predictions(
    prediction_dir: Path,
    date_str: str,
    filename_template: str,
) -> pd.DataFrame:
    input_path = prediction_dir / filename_template.format(date=date_str)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    logging.info("Loading predictions from %s", input_path)
    df = pd.read_csv(input_path)

    # ------------------------------------------------------------------
    # Normalize column names from acc-file to what the script expects
    # ------------------------------------------------------------------
    RESULT_COL = "home_team_won"
    DATE_COL = "game_date"
    
    # 1) Fix the date column: acc file uses 'date'
    if DATE_COL not in df.columns:
        if "date" in df.columns:
            logging.info(
                f"DATE_COL '{DATE_COL}' not in dataframe – creating it from 'date' column."
            )
            df[DATE_COL] = df["date"]
        else:
            logging.warning(
                f"Neither '{DATE_COL}' nor 'date' found – "
                f"using run date for all rows: {run_date_ymd}"
            )
            df[DATE_COL] = run_date_ymd
    
    # 2) Fix the result column: build home_team_won from 'home_team' vs 'result'
    if RESULT_COL not in df.columns:
        if {"home_team", "result"}.issubset(df.columns):
            logging.info(
                f"RESULT_COL '{RESULT_COL}' not in dataframe – "
                f"creating it as 1 if result==home_team else 0."
            )
            df[RESULT_COL] = (df["result"] == df["home_team"]).astype(int)
        else:
            logging.warning(
                f"Cannot construct '{RESULT_COL}' – "
                f"missing either 'home_team' or 'result' in dataframe."
            )

        # Ensure we have the prediction probability column in the expected name
    if PRED_PROBA_COL not in df.columns:
        if "home_team_prob" in df.columns:
            logging.info(
                f"PRED_PROBA_COL '{PRED_PROBA_COL}' not in dataframe – "
                "creating it from 'home_team_prob'."
            )
            df[PRED_PROBA_COL] = df["home_team_prob"].astype(float)
        else:
            raise KeyError(
                f"Neither '{PRED_PROBA_COL}' nor 'home_team_prob' found in predictions file; "
                "cannot run isotonic calibration."
            )



    
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    else:
        logging.warning(
            "DATE_COL '%s' not in dataframe – using --date for all rows (%s).",
            DATE_COL,
            date_str,
        )
        df[DATE_COL] = pd.to_datetime(date_str)

    return df


def split_past_future(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if RESULT_COL not in df.columns:
        logging.info(
            "RESULT_COL '%s' not found – assuming all games are past (results known).",
            RESULT_COL,
        )
        return df.copy(), df.iloc[0:0].copy()

    mask_past = df[RESULT_COL].notna()
    df_past = df.loc[mask_past].copy()
    df_future = df.loc[~mask_past].copy()
    logging.info("Split into %d past games and %d future games.", len(df_past), len(df_future))
    return df_past, df_future


def fit_isotonic(df_past: pd.DataFrame) -> Tuple[IsotonicRegression, Tuple[float, float, float, float]]:
    """
    Fit Isotonic Regression on past games and compute Brier/log-loss
    before & after calibration.
    """
    y_true = df_past[RESULT_COL].astype(int).values
    p_raw = df_past[PRED_PROBA_COL].astype(float).values

    eps = 1e-6
    p_raw_clipped = np.clip(p_raw, eps, 1 - eps)

    brier_before = brier_score_loss(y_true, p_raw_clipped)
    logloss_before = log_loss(y_true, p_raw_clipped)

    iso = IsotonicRegression(out_of_bounds="clip")
    p_iso = iso.fit_transform(p_raw, y_true)

    p_iso_clipped = np.clip(p_iso, eps, 1 - eps)
    brier_after = brier_score_loss(y_true, p_iso_clipped)
    logloss_after = log_loss(y_true, p_iso_clipped)

    logging.info("Isotonic fitted on %d games.", len(df_past))
    logging.info("Brier before: %.6f | after: %.6f", brier_before, brier_after)
    logging.info("Log-loss before: %.6f | after: %.6f", logloss_before, logloss_after)

    return iso, (brier_before, brier_after, logloss_before, logloss_after)


def apply_isotonic(df: pd.DataFrame, iso: IsotonicRegression) -> pd.DataFrame:
    df = df.copy()
    df["iso_proba_home_win"] = iso.transform(df[PRED_PROBA_COL].astype(float).values)
    return df


def kelly_fraction(prob: float, odds: float) -> float:
    """
    Standard Kelly fraction for a single-outcome bet (home win).
    Returns fraction of bankroll to stake; clipped later with KELLY_CAP.
    """
    if odds <= 1.0:
        return 0.0
    edge = prob * (odds - 1) - (1 - prob)
    if edge <= 0:
        return 0.0
    return edge / (odds - 1)


def simulate_kelly(df_bets: pd.DataFrame, initial_bankroll: float) -> Tuple[float, int, int]:
    bankroll = initial_bankroll
    n_wins = 0
    n_bets = 0

    for _, row in df_bets.iterrows():
        prob = float(row["iso_proba_home_win"])
        odds = float(row[HOME_ODDS_COL])
        result = int(row[RESULT_COL]) if (RESULT_COL in row and not pd.isna(row[RESULT_COL])) else None

        f = kelly_fraction(prob, odds)
        f = max(0.0, min(KELLY_CAP, f))
        stake = bankroll * f

        if stake <= 0:
            continue

        n_bets += 1

        if result is None:
            # Should not happen for historical backtest, but just in case:
            continue

        if result == 1:
            bankroll += stake * (odds - 1)
            n_wins += 1
        else:
            bankroll -= stake

    total_profit = bankroll - initial_bankroll
    return total_profit, n_bets, n_wins


def evaluate_strategy(
    df_past_iso: pd.DataFrame,
    params: StrategyParams,
    metrics_before_after: Tuple[float, float, float, float],
    initial_bankroll: float,
) -> StrategyResult:
    df = df_past_iso.copy()

    conds = []
    conds.append(df[HOME_ODDS_COL].between(params.min_odds, params.max_odds))
    conds.append(df["iso_proba_home_win"] >= params.min_iso_proba)
    if HOME_WIN_RATE_COL is not None and HOME_WIN_RATE_COL in df.columns:
        conds.append(df[HOME_WIN_RATE_COL] >= params.min_home_win_rate)

    mask = np.logical_and.reduce(conds)
    df_bets = df.loc[mask].copy()

    if df_bets.empty:
        return StrategyResult(
            params=params,
            n_bets=0,
            n_wins=0,
            total_profit_flat=0.0,
            roi_flat=0.0,
            total_profit_kelly=0.0,
            roi_kelly=0.0,
            brier_before=metrics_before_after[0],
            brier_after=metrics_before_after[1],
            logloss_before=metrics_before_after[2],
            logloss_after=metrics_before_after[3],
        )

    df_bets["result_int"] = df_bets[RESULT_COL].astype(int)
    df_bets["profit_flat"] = np.where(
        df_bets["result_int"] == 1,
        df_bets[HOME_ODDS_COL] - 1.0,
        -1.0,
    )
    total_profit_flat = float(df_bets["profit_flat"].sum())
    roi_flat = total_profit_flat / len(df_bets)

    total_profit_kelly, _, _ = simulate_kelly(df_bets, initial_bankroll)
    roi_kelly = total_profit_kelly / initial_bankroll

    return StrategyResult(
        params=params,
        n_bets=len(df_bets),
        n_wins=int(df_bets["result_int"].sum()),
        total_profit_flat=total_profit_flat,
        roi_flat=float(roi_flat),
        total_profit_kelly=float(total_profit_kelly),
        roi_kelly=float(roi_kelly),
        brier_before=metrics_before_after[0],
        brier_after=metrics_before_after[1],
        logloss_before=metrics_before_after[2],
        logloss_after=metrics_before_after[3],
    )


def grid_search(
    df_past_iso: pd.DataFrame,
    metrics_before_after: Tuple[float, float, float, float],
    initial_bankroll: float,
) -> List[StrategyResult]:
    logging.info("Starting grid search...")
    results: List[StrategyResult] = []

    for min_home_win_rate in MIN_HOME_WIN_RATE_LIST:
        for min_odds in MIN_ODDS_LIST:
            for max_odds in MAX_ODDS_LIST:
                if max_odds <= min_odds:
                    continue
                for min_iso_proba in MIN_ISO_PROBA_LIST:
                    params = StrategyParams(
                        min_home_win_rate=min_home_win_rate,
                        min_odds=min_odds,
                        max_odds=max_odds,
                        min_iso_proba=min_iso_proba,
                    )
                    res = evaluate_strategy(
                        df_past_iso,
                        params,
                        metrics_before_after,
                        initial_bankroll=initial_bankroll,
                    )
                    results.append(res)

    logging.info("Grid search finished with %d combinations.", len(results))
    return results


def select_best_strategy(results: List[StrategyResult]) -> StrategyResult:
    # sort by (flat profit, number of bets, ROI per bet)
    results_sorted = sorted(
        results,
        key=lambda r: (r.total_profit_flat, r.n_bets, r.roi_flat),
        reverse=True,
    )
    best = results_sorted[0]
    logging.info(
        "Best strategy: %s | %d bets | flat profit %.2f | ROI per bet %.4f",
        best.params,
        best.n_bets,
        best.total_profit_flat,
        best.roi_flat,
    )
    return best


def results_to_dataframe(results: List[StrategyResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "min_home_win_rate": r.params.min_home_win_rate,
                "min_odds": r.params.min_odds,
                "max_odds": r.params.max_odds,
                "min_iso_proba": r.params.min_iso_proba,
                "n_bets": r.n_bets,
                "n_wins": r.n_wins,
                "win_rate": r.n_wins / r.n_bets if r.n_bets > 0 else 0.0,
                "total_profit_flat": r.total_profit_flat,
                "roi_flat_per_bet": r.roi_flat,
                "total_profit_kelly": r.total_profit_kelly,
                "roi_kelly": r.roi_kelly,
                "brier_before": r.brier_before,
                "brier_after": r.brier_after,
                "logloss_before": r.logloss_before,
                "logloss_after": r.logloss_after,
            }
        )
    return pd.DataFrame(rows)


def build_shortlist(
    df_future_iso: pd.DataFrame,
    best: StrategyResult,
    bankroll: float,
) -> pd.DataFrame:
    df = df_future_iso.copy()
    p = best.params

    conds = []
    conds.append(df[HOME_ODDS_COL].between(p.min_odds, p.max_odds))
    conds.append(df["iso_proba_home_win"] >= p.min_iso_proba)
    if HOME_WIN_RATE_COL is not None and HOME_WIN_RATE_COL in df.columns:
        conds.append(df[HOME_WIN_RATE_COL] >= p.min_home_win_rate)

    mask = np.logical_and.reduce(conds)
    df_bets = df.loc[mask].copy()

    if df_bets.empty:
        logging.info("No bets for today with best strategy parameters.")
        return df_bets

    fractions = []
    stakes = []
    edges = []

    for _, row in df_bets.iterrows():
        prob = float(row["iso_proba_home_win"])
        odds = float(row[HOME_ODDS_COL])
        f = kelly_fraction(prob, odds)
        f = max(0.0, min(KELLY_CAP, f))
        stake = bankroll * f
        edge = prob * odds - 1.0

        fractions.append(f)
        stakes.append(stake)
        edges.append(edge)

    df_bets["kelly_fraction"] = fractions
    df_bets["stake_recommended"] = stakes
    df_bets["edge"] = edges

    df_bets = df_bets.sort_values("edge", ascending=False)

    cols_order = [
        DATE_COL,
        HOME_TEAM_COL,
        AWAY_TEAM_COL,
        HOME_ODDS_COL,
        "iso_proba_home_win",
    ]
    if HOME_WIN_RATE_COL is not None and HOME_WIN_RATE_COL in df_bets.columns:
        cols_order.append(HOME_WIN_RATE_COL)
    cols_order += ["edge", "kelly_fraction", "stake_recommended"]

    existing_cols = [c for c in cols_order if c in df_bets.columns]
    df_bets = df_bets[existing_cols].copy()

    return df_bets


def attach_why_not(
    df_future_iso: pd.DataFrame,
    best: StrategyResult,
) -> pd.DataFrame:
    df = df_future_iso.copy()
    p = best.params

    reasons = []
    for _, row in df.iterrows():
        r: List[str] = []

        odds = float(row.get(HOME_ODDS_COL, np.nan))
        iso_p = float(row.get("iso_proba_home_win", np.nan))
        if HOME_WIN_RATE_COL is not None and HOME_WIN_RATE_COL in df.columns:
            win_rate = float(row.get(HOME_WIN_RATE_COL, np.nan))
        else:
            win_rate = np.nan

        if not np.isfinite(odds):
            r.append("missing_home_odds")
        else:
            if odds < p.min_odds:
                r.append(f"odds<{p.min_odds}")
            if odds > p.max_odds:
                r.append(f"odds>{p.max_odds}")

        if not np.isfinite(iso_p) or iso_p < p.min_iso_proba:
            r.append(f"iso_proba<{p.min_iso_proba}")

        if HOME_WIN_RATE_COL is not None and HOME_WIN_RATE_COL in df.columns:
            if not np.isfinite(win_rate) or win_rate < p.min_home_win_rate:
                r.append(f"{HOME_WIN_RATE_COL}<{p.min_home_win_rate}")

        reasons.append(";".join(r) if r else "passes_all_filters")

    df["why_not_bet"] = reasons
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    # Resolve paths once via nba_utils_2026
    paths = get_directory_paths()
    prediction_dir = Path(paths["PREDICTION_DIR"])
    kelly_output_dir = prediction_dir / "Kelly"
    kelly_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve date
    if args.date is None:
        _, _, ymd = get_current_date(days_offset=0)
        date_str = ymd
        logging.info("No --date passed, using today's date from nba_utils_2026: %s", date_str)
    else:
        # quick sanity check
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError as e:
            raise SystemExit(f"Invalid --date format (expected YYYY-MM-DD): {args.date}") from e
        date_str = args.date

    df = load_predictions(prediction_dir, date_str, args.input_filename_template)
    df_past, df_future = split_past_future(df)

    if df_past.empty:
        logging.warning(
            "No past games with known results found. "
            "Isotonic calibration / grid search cannot run – exiting."
        )
        return

    # 1) Isotonic calibration
    iso, metrics_before_after = fit_isotonic(df_past)
    df_iso = apply_isotonic(df, iso)
    df_past_iso, df_future_iso = split_past_future(df_iso)

    # 2) Grid search
    results = grid_search(
        df_past_iso,
        metrics_before_after,
        initial_bankroll=args.initial_bankroll,
    )
    best = select_best_strategy(results)

    # 3) Save grid-search overview
    df_results = results_to_dataframe(results)
    grid_out = kelly_output_dir / f"nba_grid_search_results_{date_str}.csv"
    df_results.to_csv(grid_out, index=False)
    logging.info("Saved grid search results to %s", grid_out)

    # 4) Save full dataframe with iso_proba_home_win
    full_out = kelly_output_dir / f"combined_nba_predictions_iso_{date_str}.csv"
    df_iso.to_csv(full_out, index=False)
    logging.info("Saved full dataframe with iso_proba_home_win to %s", full_out)

    # 5) Shortlist + why-not for future games
    if not df_future_iso.empty:
        shortlist = build_shortlist(
            df_future_iso,
            best,
            bankroll=args.initial_bankroll,
        )
        shortlist_out = kelly_output_dir / f"nba_bets_shortlist_{date_str}.csv"
        shortlist.to_csv(shortlist_out, index=False)
        logging.info(
            "Saved shortlist for %d future games to %s",
            len(shortlist),
            shortlist_out,
        )

        why_not_df = attach_why_not(df_future_iso, best)
        why_not_out = kelly_output_dir / f"nba_bets_why_not_{date_str}.csv"
        why_not_df.to_csv(why_not_out, index=False)
        logging.info("Saved why-not diagnostics to %s", why_not_out)
    else:
        logging.info("No future games found in file – nothing to bet on today.")

    logging.info("Step 5 finished.")


if __name__ == "__main__":
    main()
