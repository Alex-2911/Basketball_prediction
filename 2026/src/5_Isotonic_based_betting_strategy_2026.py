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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


# Columns in combined_nba_predictions_acc_YYYY-MM-DD.csv
# Adjust these if your file uses different names.
GAME_ID_COL = "game_id"
DATE_COL = "game_date"                  # parsed as datetime
HOME_TEAM_COL = "home_team"
AWAY_TEAM_COL = "away_team"
PRED_PROBA_COL = "pred_home_win_proba"  # model probability for home win (uncalibrated)
RESULT_COL = "home_team_won"            # 1 if home team actually won, 0 if lost, NaN for future games
HOME_ODDS_COL = "closing_home_odds"     # decimal odds for home side
AWAY_ODDS_COL = "closing_away_odds"     # optional; not required
HOME_WIN_RATE_COL: Optional[str] = "home_team_win_rate_last_20"  # set to None to disable


# Default paths – change BASE_DIR to your project root or override with CLI.
BASE_DIR = Path(".")

# Kelly configuration
INITIAL_BANKROLL = 1000.0
KELLY_CAP = 0.10  # never bet more than 10% of bankroll on a single game


# Grid search configuration
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
        description="Script 5/5 – Isotonic calibration + Kelly optimization + shortlist export"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Prediction date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(BASE_DIR),
        help="Base project directory (default: current directory)",
    )
    parser.add_argument(
        "--input-filename-template",
        type=str,
        default="combined_nba_predictions_acc_{date}.csv",
        help="Template for input file name, must contain '{date}' placeholder",
    )
    return parser.parse_args()


def load_predictions(base_dir: Path, date_str: str, filename_template: str) -> pd.DataFrame:
    input_path = base_dir / "data" / "predictions" / filename_template.format(date=date_str)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    logging.info("Loading predictions from %s", input_path)
    df = pd.read_csv(input_path)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    else:
        logging.warning("DATE_COL '%s' not in dataframe. Using date argument for all rows.", DATE_COL)
        df[DATE_COL] = pd.to_datetime(date_str)

    return df


def split_past_future(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if RESULT_COL not in df.columns:
        logging.info(
            "RESULT_COL '%s' not found – assuming all games are past (results known). "
            "Isotonic calibration and grid search will still run.",
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
    Fit isotonic regression on past games and compute Brier/log loss before & after.
    Returns the fitted model and the metrics.
    """
    y_true = df_past[RESULT_COL].astype(int).values
    p_raw = df_past[PRED_PROBA_COL].astype(float).values

    # For log_loss, clip probabilities away from 0/1
    eps = 1e-6
    p_raw_clipped = np.clip(p_raw, eps, 1 - eps)

    brier_before = brier_score_loss(y_true, p_raw_clipped)
    logloss_before = log_loss(y_true, p_raw_clipped)

    iso = IsotonicRegression(out_of_bounds="clip")
    p_iso = iso.fit_transform(p_raw, y_true)

    p_iso_clipped = np.clip(p_iso, eps, 1 - eps)
    brier_after = brier_score_loss(y_true, p_iso_clipped)
    logloss_after = log_loss(y_true, p_iso_clipped)

    logging.info("Isotonic calibration fitted on %d games.", len(df_past))
    logging.info("Brier before: %.6f | after: %.6f", brier_before, brier_after)
    logging.info("Log-loss before: %.6f | after: %.6f", logloss_before, logloss_after)

    return iso, (brier_before, brier_after, logloss_before, logloss_after)


def apply_isotonic(df: pd.DataFrame, iso: IsotonicRegression) -> pd.DataFrame:
    df = df.copy()
    df["iso_proba_home_win"] = iso.transform(df[PRED_PROBA_COL].astype(float).values)
    return df


def kelly_fraction(prob: float, odds: float) -> float:
    """
    Standard Kelly fraction for a single-outcome bet.
    Returns fraction of bankroll to stake (can be negative if edge < 0).
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
            # For simulation on historical data we always have results,
            # but guard anyway.
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
) -> StrategyResult:
    """
    Filter past games according to params and evaluate flat-stake + Kelly P&L.
    """
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

    # Flat stake: 1 unit per bet
    df_bets["result_int"] = df_bets[RESULT_COL].astype(int)
    df_bets["profit_flat"] = np.where(
        df_bets["result_int"] == 1,
        df_bets[HOME_ODDS_COL] - 1.0,
        -1.0,
    )
    total_profit_flat = df_bets["profit_flat"].sum()
    roi_flat = total_profit_flat / len(df_bets)

    # Kelly simulation
    total_profit_kelly, _, _ = simulate_kelly(df_bets, INITIAL_BANKROLL)
    roi_kelly = total_profit_kelly / INITIAL_BANKROLL

    return StrategyResult(
        params=params,
        n_bets=len(df_bets),
        n_wins=int(df_bets["result_int"].sum()),
        total_profit_flat=float(total_profit_flat),
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
                    res = evaluate_strategy(df_past_iso, params, metrics_before_after)
                    results.append(res)

    logging.info("Grid search finished with %d combinations.", len(results))
    return results


def select_best_strategy(results: List[StrategyResult]) -> StrategyResult:
    # Sort primarily by total_profit_flat, then by n_bets (more bets preferred), then by roi_flat.
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
        row = {
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
        rows.append(row)
    return pd.DataFrame(rows)


def build_shortlist(
    df_future_iso: pd.DataFrame,
    best: StrategyResult,
    bankroll: float,
) -> pd.DataFrame:
    df = df_future_iso.copy()
    params = best.params

    conds = []
    conds.append(df[HOME_ODDS_COL].between(params.min_odds, params.max_odds))
    conds.append(df["iso_proba_home_win"] >= params.min_iso_proba)
    if HOME_WIN_RATE_COL is not None and HOME_WIN_RATE_COL in df.columns:
        conds.append(df[HOME_WIN_RATE_COL] >= params.min_home_win_rate)

    mask = np.logical_and.reduce(conds)
    df_bets = df.loc[mask].copy()

    if df_bets.empty:
        logging.info("No bets for today with the best strategy parameters.")
        return df_bets

    # Compute Kelly fraction and recommended stake
    fractions = []
    stakes = []
    for _, row in df_bets.iterrows():
        prob = float(row["iso_proba_home_win"])
        odds = float(row[HOME_ODDS_COL])
        f = kelly_fraction(prob, odds)
        f = max(0.0, min(KELLY_CAP, f))
        stake = bankroll * f

        fractions.append(f)
        stakes.append(stake)

    df_bets["kelly_fraction"] = fractions
    df_bets["stake_recommended"] = stakes

    # Sort by edge (prob * odds - 1)
    df_bets["edge"] = df_bets["iso_proba_home_win"] * df_bets[HOME_ODDS_COL] - 1.0
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
    """
    For each future game, show which filter(s) failed so that it didn't become a bet.
    """
    df = df_future_iso.copy()
    p = best.params

    reasons = []
    for _, row in df.iterrows():
        r: List[str] = []
        odds = float(row.get(HOME_ODDS_COL, np.nan))
        iso_p = float(row.get("iso_proba_home_win", np.nan))
        win_rate = (
            float(row.get(HOME_WIN_RATE_COL, np.nan))
            if HOME_WIN_RATE_COL and HOME_WIN_RATE_COL in df.columns
            else np.nan
        )

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

        if not r:
            reasons.append("passes_all_filters")
        else:
            reasons.append(";".join(r))

    df["why_not_bet"] = reasons
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    args = parse_args()

    base_dir = Path(args.base_dir)
    date_str = args.date

    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(base_dir, date_str, args.input_filename_template)
    df_past, df_future = split_past_future(df)

    if df_past.empty:
        logging.warning(
            "No past games (with known results) found. "
            "Isotonic calibration / grid search can't run – exiting."
        )
        return

    # 1) Isotonic calibration
    iso, metrics_before_after = fit_isotonic(df_past)
    df_iso = apply_isotonic(df, iso)
    df_past_iso, df_future_iso = split_past_future(df_iso)

    # 2) Grid search on past games
    results = grid_search(df_past_iso, metrics_before_after)
    best = select_best_strategy(results)

    # 3) Save grid-search overview
    df_results = results_to_dataframe(results)
    grid_out = output_dir / f"nba_grid_search_results_{date_str}.csv"
    df_results.to_csv(grid_out, index=False)
    logging.info("Saved grid search results to %s", grid_out)

    # 4) Save full dataframe with isotonic probability
    full_out = output_dir / f"combined_nba_predictions_iso_{date_str}.csv"
    df_iso.to_csv(full_out, index=False)
    logging.info("Saved full dataframe with iso_proba_home_win to %s", full_out)

    # 5) Shortlist for today's bets (if future games exist)
    if not df_future_iso.empty:
        shortlist = build_shortlist(df_future_iso, best, INITIAL_BANKROLL)
        shortlist_out = output_dir / f"nba_bets_shortlist_{date_str}.csv"
        shortlist.to_csv(shortlist_out, index=False)
        logging.info(
            "Saved shortlist for %d future games to %s",
            len(shortlist),
            shortlist_out,
        )

        # Why-not diagnostics
        why_not_df = attach_why_not(df_future_iso, best)
        why_not_out = output_dir / f"nba_bets_why_not_{date_str}.csv"
        why_not_df.to_csv(why_not_out, index=False)
        logging.info("Saved why-not diagnostics to %s", why_not_out)
    else:
        logging.info("No future games found in file – nothing to bet on today.")

    logging.info("Done.")


if __name__ == "__main__":
    main()
