#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 4 of 5 (2025‑26 season): Calculate Betting Statistics

This script merges actual outcomes with predicted results to evaluate
betting performance for the 2025‑26 NBA season.  It calculates overall
and subset accuracies (e.g., home‑favored vs. away‑favored), and updates
a combined CSV with the results.

Prior steps:
    1. Run ``1_get_data_previous_game_day.py`` to generate statistics.
    2. Run ``2_get_data_next_game_day_2026.py`` to create the games_df CSV.
    3. Run ``3_predict_next_game_day_2026.py`` to produce predictions for
       the next game day.

Then execute this script to compute betting statistics.
"""

import pandas as pd
import os
import numpy as np
import logging
from datetime import timedelta
from pathlib import Path

# Import shared utilities from the 2026 version
from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    find_file_in_date_range,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Maximum days to look back for files
MAX_DAYS_BACK = 120  # Configurable range for searching files


LIVE_PROB_COLUMNS = [
    "prob_iso",
    "prob_iso_insample",
    "prob_iso_oos_time",
    "prob_live_oos_proxy",
    "prob_live_safe_pre_clip",
    "prob_base",
    "prob_live_safe",
    "prob_used",
]

CANONICAL_BASE_COLUMNS = [
    "home_team",
    "away_team",
    "home_team_prob",
    "odds_1",
    "odds_2",
    "result",
    "date",
    "accuracy",
]

ACC_PREFIX = "combined_nba_predictions_acc_"


def normalize_column_name(col: str) -> str:
    """Normalize headers to canonical snake_case names."""
    normalized = str(col).strip().lower().replace("\n", " ")
    normalized = "_".join(normalized.split())
    alias_map = {
        "odds1": "odds_1",
        "odds2": "odds_2",
        "odds_1.0": "odds_1",
        "odds_2.0": "odds_2",
    }
    return alias_map.get(normalized, normalized)


def normalize_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with stable, canonical column names and order."""
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="last")]

    for col in CANONICAL_BASE_COLUMNS + LIVE_PROB_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    if "pred_home_win_proba" in out.columns:
        out["home_team_prob"] = out["home_team_prob"].fillna(
            pd.to_numeric(out["pred_home_win_proba"], errors="coerce")
        )
    if "iso_proba_home_win" in out.columns:
        out["home_team_prob"] = out["home_team_prob"].fillna(
            pd.to_numeric(out["iso_proba_home_win"], errors="coerce")
        )
    if "closing_home_odds" in out.columns:
        out["odds_1"] = out["odds_1"].fillna(pd.to_numeric(out["closing_home_odds"], errors="coerce"))
    if "closing_away_odds" in out.columns:
        out["odds_2"] = out["odds_2"].fillna(pd.to_numeric(out["closing_away_odds"], errors="coerce"))

    if out["prob_iso"].isna().all() and "prob_iso_insample" in out.columns:
        out["prob_iso"] = out["prob_iso_insample"]

    ordered = CANONICAL_BASE_COLUMNS + LIVE_PROB_COLUMNS
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def build_game_key(df: pd.DataFrame) -> pd.Series:
    """Build stable game key from date + home_team + away_team."""
    work = df.copy()
    date_source = "date" if "date" in work.columns else "game_date"
    work["_game_date"] = pd.to_datetime(work.get(date_source), errors="coerce").dt.strftime("%Y-%m-%d")
    return (
        work["_game_date"].fillna("")
        + "|"
        + work.get("home_team", pd.Series(index=work.index, dtype=str)).astype(str).str.strip().str.upper()
        + "|"
        + work.get("away_team", pd.Series(index=work.index, dtype=str)).astype(str).str.strip().str.upper()
    )


def row_completeness_score(df: pd.DataFrame) -> pd.Series:
    """Prefer more complete resolved rows during dedupe/upsert."""
    score = pd.Series(0, index=df.index, dtype=float)
    result_col = df.get("result", pd.Series(index=df.index, dtype=object))
    score += result_col.notna() & ~result_col.astype(str).str.strip().isin(["", "0", "nan", "None"])
    score += pd.to_numeric(df.get("accuracy"), errors="coerce").notna()
    score += pd.to_numeric(df.get("home_team_prob"), errors="coerce").notna()
    score += pd.to_numeric(df.get("odds_1"), errors="coerce").notna()
    score += pd.to_numeric(df.get("odds_2"), errors="coerce").notna()
    return score


def upsert_by_game_key(base_df: pd.DataFrame, updates_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([base_df, updates_df], ignore_index=True)
    merged = normalize_prediction_dataframe(merged)
    merged["_game_key"] = build_game_key(merged)
    merged["_score"] = row_completeness_score(merged)
    merged["_date_sort"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.sort_values(by=["_score", "_date_sort"], ascending=[False, False])
    merged = merged.drop_duplicates(subset=["_game_key"], keep="first")
    merged = merged.drop(columns=["_game_key", "_score", "_date_sort"], errors="ignore")
    return normalize_prediction_dataframe(merged)


def load_latest_previous_acc(prediction_dir: str, before_date: str) -> pd.DataFrame:
    """Load latest previous cumulative ACC file strictly before `before_date`."""
    cutoff = pd.to_datetime(before_date, errors="coerce")
    candidates = []
    for p in Path(prediction_dir).glob(f"{ACC_PREFIX}*.csv"):
        date_part = p.stem.replace(ACC_PREFIX, "")
        dt = pd.to_datetime(date_part, errors="coerce")
        if pd.isna(dt) or dt >= cutoff:
            continue
        candidates.append((dt, p))

    if not candidates:
        logging.info("No previous ACC file found before %s in %s", before_date, prediction_dir)
        return normalize_prediction_dataframe(pd.DataFrame())

    _, latest_path = max(candidates, key=lambda x: x[0])
    logging.info("Using previous cumulative ACC file: %s", latest_path)
    return normalize_prediction_dataframe(pd.read_csv(latest_path))

# Get current date information
today, today_str, today_str_format = get_current_date()
yesterday, yesterday_str, yesterday_str_format = get_current_date(days_offset=1)

print(f"Today's date: {today_str_format}")
print(f"Looking for data from: {yesterday_str_format}")

# Get directory paths
paths = get_directory_paths()
BASE_DIR = paths['BASE_DIR']
DATA_DIR = paths['DATA_DIR']
STAT_DIR = paths['STAT_DIR']
target_folder = paths['NEXT_GAME_DIR']
directory_path = paths['PREDICTION_DIR']
prediction_dirs = paths['PREDICTION_DIRS']


def find_most_recent_prediction_file():
    """Find the most recent prediction file within the specified days range."""
    days_back = 0
    file_found = False

    while not file_found and days_back <= MAX_DAYS_BACK:
        # Recalculate the date string on every loop iteration
        date_to_check = yesterday - timedelta(days=days_back)
        date_str = date_to_check.strftime("%Y-%m-%d")

        logging.info(f"Checking prediction file for date: {date_str}")
        filename = f"nba_games_predict_{date_str}.csv"
        for prediction_dir in prediction_dirs:
            file_path = os.path.join(prediction_dir, filename)
            if os.path.isfile(file_path):
                file_found = True
                logging.info(f"Found prediction file for {date_str}: {file_path}")
                return file_path, date_str, prediction_dir
        days_back += 1

    logging.warning("No prediction file found in the last %d days.", MAX_DAYS_BACK)
    return None, None, None


def find_most_recent_statistics_file():
    """Find the most recent statistics file within the specified days range."""
    file_path, date_str = find_file_in_date_range(
        STAT_DIR,
        f"nba_games_{{}}.csv",
        MAX_DAYS_BACK,
    )
    if file_path:
        logging.info(f"Found statistics file for {date_str}: {file_path}")
        return date_str
    else:
        logging.warning("No statistics file found within the specified range.")
        return None


def process_prediction_file(predict_file, last_prediction, prediction_dir):
    """
    Process the prediction file and update the combined predictions.

    Args:
        predict_file (list): List containing the path to the prediction file.
        last_prediction (str): Date string of the last prediction.

    Returns:
        DataFrame: Combined predictions DataFrame or None if no file found.
    """
    if not predict_file:
        logging.warning("No prediction file found to process.")
        return None

    # Read prediction file; use default encoding and convert decimal comma to period
    predict_df = pd.read_csv(predict_file)
    # Normalize decimal columns in odds
    for col in ['odds 1', 'odds 2', 'odds_1', 'odds_2']:
        if col in predict_df.columns:
            predict_df[col] = predict_df[col].astype(str).str.replace(',', '.').astype(float)
    predict_df = normalize_prediction_dataframe(predict_df)

    # Start from latest previous cumulative ACC file, then upsert new prediction day rows.
    combined_df = load_latest_previous_acc(prediction_dir, today_str_format)

    # Upsert and sort
    predict_df['accuracy'] = np.nan  # add placeholder column
    combined_df = upsert_by_game_key(combined_df, predict_df)
    combined_df = combined_df.sort_values(by='date', ascending=False)
    # Display top rows for user
    logging.info("Combined predictions (latest 10 rows):\n%s", combined_df.head(10).to_string(index=False))
    logging.info("Combined predictions updated")
    return combined_df


def update_betting_statistics(combined_df, most_recent_date, prediction_dir):
    """
    Update betting statistics with actual game results.

    Args:
        combined_df (DataFrame): DataFrame with combined predictions.
        most_recent_date (str): Date string of the most recent statistics file.

    Returns:
        DataFrame: Updated statistics DataFrame or None if update failed.
    """
    # Copy of predictions to update
    season_df = combined_df.copy()
    logging.info(f"Updating statistics using games from {most_recent_date}")

    # Read the most recent games data
    daily_games_df = pd.read_csv(os.path.join(STAT_DIR, f"nba_games_{most_recent_date}.csv"))
    # Filter to current season only
    daily_games_df = daily_games_df[daily_games_df['season'] == CURRENT_SEASON].copy()

    # Convert date columns to datetime
    season_df['date'] = pd.to_datetime(season_df['date'], errors='coerce')
    daily_games_df['date'] = pd.to_datetime(daily_games_df['date'], errors='coerce')

    # Normalize placeholder results to NaN
    season_df['result'] = (
        season_df['result']
        .astype(str)
        .str.strip()
        .replace(["", "0", "1", "nan", "None"], np.nan)
    )

    # Update result column based on actual winners
    for _, row in daily_games_df.iterrows():
        date = row['date']
        winning_team = row['team'] if row['won'] == 1 else None
        if not winning_team:
            continue
        mask = (season_df['date'] == date) & (
            (season_df['home_team'] == winning_team) | (season_df['away_team'] == winning_team)
        )
        season_df.loc[mask, 'result'] = winning_team

    # Ensure probabilities are numeric and keep both played + upcoming rows
    season_df['home_team_prob'] = pd.to_numeric(season_df['home_team_prob'], errors='coerce')

    played_mask = (
        (season_df['result'] == season_df['home_team'])
        | (season_df['result'] == season_df['away_team'])
    )
    home_win = (season_df['home_team_prob'] >= 0.5) & (season_df['result'] == season_df['home_team'])
    away_win = (season_df['home_team_prob'] < 0.5) & (season_df['result'] == season_df['away_team'])
    season_df['accuracy'] = np.where(played_mask, (home_win | away_win).astype(int), np.nan)

    # Overall accuracy on played rows only
    overall = season_df.loc[played_mask, 'accuracy'].mean()
    logging.info(f"Overall accuracy: {overall:.2%}")

    # Subset accuracy (home_team_prob > 0.60 and < 0.40)
    high_conf_home = season_df[season_df['home_team_prob'] > 0.60]['accuracy'].mean()
    low_conf_home = season_df[season_df['home_team_prob'] <= 0.40]['accuracy'].mean()
    logging.info(f"Accuracy for home_team_prob > 0.60: {high_conf_home:.2%}")
    logging.info(f"Accuracy for home_team_prob <= 0.40: {low_conf_home:.2%}")

    # Save updated DataFrame with today's date
    save_path = os.path.join(prediction_dir, f'combined_nba_predictions_acc_{today_str_format}.csv')
    # Drop unnamed columns and keep upcoming rows for downstream live pipeline
    season_df.drop(columns=['Unnamed: 8'], errors='ignore', inplace=True)
    season_df.dropna(
        subset=["date", "home_team", "away_team", "home_team_prob"],
        inplace=True,
    )
    previous_rows = int(len(combined_df))
    updated_rows = season_df[pd.to_datetime(season_df["date"], errors="coerce") == pd.to_datetime(most_recent_date)].copy()
    updated_rows = updated_rows[
        (updated_rows["result"] == updated_rows["home_team"])
        | (updated_rows["result"] == updated_rows["away_team"])
    ]
    season_df = upsert_by_game_key(normalize_prediction_dataframe(pd.DataFrame()), season_df)
    season_df = season_df.sort_values('date', ascending=False)
    season_df = normalize_prediction_dataframe(season_df)
    season_df.to_csv(save_path, index=False)
    played_rows = season_df[
        (season_df["result"] == season_df["home_team"])
        | (season_df["result"] == season_df["away_team"])
    ].copy()
    date_series = pd.to_datetime(season_df["date"], errors="coerce").dropna()
    min_date = date_series.min().strftime("%Y-%m-%d") if not date_series.empty else "NA"
    max_date = date_series.max().strftime("%Y-%m-%d") if not date_series.empty else "NA"
    logging.info("Previous ACC rows: %d", previous_rows)
    logging.info("Updated rows from latest resolved day: %d", int(len(updated_rows)))
    logging.info("ACC rows after upsert: %d", int(len(season_df)))
    logging.info("ACC played rows: %d", int(len(played_rows)))
    logging.info("ACC date range: %s -> %s", min_date, max_date)
    logging.info(f"Updated betting statistics saved to {save_path}")
    return season_df


def main():
    """Main execution function for updating betting statistics."""
    # Find most recent prediction file
    predict_file, last_prediction, prediction_dir = find_most_recent_prediction_file()
    if not predict_file:
        print("No recent prediction file found.")
        return

    # Process prediction file
    combined_df = process_prediction_file(predict_file, last_prediction, prediction_dir)
    if combined_df is None:
        print("Failed to process prediction file.")
        return

    # Find most recent statistics file
    most_recent_date = find_most_recent_statistics_file()
    if not most_recent_date:
        print("No recent statistics file found.")
        return

    # Update betting statistics
    updated_df = update_betting_statistics(combined_df, most_recent_date, prediction_dir)
    if updated_df is not None:
        print("Betting statistics updated successfully.")
    else:
        print("Failed to update betting statistics.")


if __name__ == "__main__":
    main()

    # Don't prompt in GitHub Actions / CI
    in_ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    if not in_ci:
        try:
            input("Press Enter to close this window...")
        except EOFError:
            pass
