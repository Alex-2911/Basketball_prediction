#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 4 of 5 (2025‑26 season): Calculate Betting Statistics

Robustified:
 - encoding fallback when reading CSVs
 - column name normalization (lowercase + underscore)
 - canonical odds/date column naming (odds_1 / odds_2 / date)
 - validation of STAT file schema ('team' and 'won') before winner assignment
"""

import pandas as pd
import os
import numpy as np
import logging
from datetime import timedelta

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

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    """ Read CSV with an encoding fallback and normalize decimal commas. """
    if path is None:
        raise FileNotFoundError("No path provided to read_csv_with_fallback()")
    last_exc = None
    for enc in ("utf-7", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            # Normalize decimal comma in numeric-ish columns heuristically
            for col in df.columns:
                if df[col].dtype == object:
                    # replace comma decimal if appears in many rows
                    sample = df[col].astype(str).head(20).str.contains(",").sum()
                    if sample:
                        try:
                            df[col] = df[col].astype(str).str.replace(",", ".").replace("", np.nan)
                            # attempt numeric convert
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except Exception:
                            pass
            return df
        except Exception as e:
            last_exc = e
            logging.debug("read_csv_with_fallback try encoding=%s failed: %s", enc, e)
            continue
    raise last_exc

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ Normalize columns to lower-case and underscores. """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def canonicalize_odds_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """ Ensure odds_1, odds_2, date columns exist using common variants. """
    df = df.copy()
    col_map = {}
    # odds_1 variants
    if "odds_1" not in df.columns:
        for c in ("odds 1", "odds1", "odds1_", "odds.1"):
            if c in df.columns:
                col_map[c] = "odds_1"
                break
    # odds_2 variants
    if "odds_2" not in df.columns:
        for c in ("odds 2", "odds2", "odds_2_", "odds.2"):
            if c in df.columns:
                col_map[c] = "odds_2"
                break
    # date variants
    if "date" not in df.columns:
        for c in ("game_date", "game_date_", "date_utc"):
            if c in df.columns:
                col_map[c] = "date"
                break

    if col_map:
        df = df.rename(columns=col_map)

    # ensure numeric odds
    for o in ("odds_1", "odds_2"):
        if o in df.columns:
            df[o] = pd.to_numeric(df[o].astype(str).str.replace(",", "."), errors="coerce")

    # normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def find_most_recent_prediction_file():
    """ Find the most recent prediction file within the specified days range. """
    days_back = 0
    file_found = False

    while not file_found and days_back <= MAX_DAYS_BACK:
        # Recalculate the date string on every loop iteration
        date_to_check = yesterday - timedelta(days=days_back)
        date_str = date_to_check.strftime("%Y-%m-%d")

        logging.info("Checking prediction file for date: %s", date_str)
        filename = f"nba_games_predict_{date_str}.csv"
        for prediction_dir in prediction_dirs:
            file_path = os.path.join(prediction_dir, filename)
            if os.path.isfile(file_path):
                file_found = True
                logging.info("Found prediction file for %s: %s", date_str, file_path)
                return file_path, date_str, prediction_dir
        days_back += 1

    logging.warning("No prediction file found in the last %d days.", MAX_DAYS_BACK)
    return None, None, None

def find_most_recent_statistics_file():
    """ Find the most recent statistics file within the specified days range. """
    file_path, date_str = find_file_in_date_range(
        STAT_DIR,
        f"nba_games_{{}}.csv",
        MAX_DAYS_BACK,
    )
    if file_path:
        logging.info("Found statistics file for %s: %s", date_str, file_path)
        return date_str
    else:
        logging.warning("No statistics file found within the specified range.")
        return None

def process_prediction_file(predict_file, last_prediction, prediction_dir):
    """ Process the prediction file and update the combined predictions. """
    if not predict_file:
        logging.warning("No prediction file found to process.")
        return None

    # Read prediction file with fallback
    predict_df = read_csv_with_fallback(predict_file)

    # Normalize columns for robust downstream handling
    predict_df = normalize_columns(predict_df)
    predict_df = canonicalize_odds_and_date(predict_df)

    # Path for combined data (one file per prediction date)
    combined_file_path = os.path.join(prediction_dir, f'combined_nba_predictions_acc_{last_prediction}.csv')
    # Load existing combined file or create new
    try:
        combined_df = read_csv_with_fallback(combined_file_path)
        combined_df = normalize_columns(combined_df)
    except FileNotFoundError:
        combined_df = pd.DataFrame()

    # Append and sort by date descending (use 'date' column if present)
    predict_df['accuracy'] = np.nan  # add placeholder column
    combined_df = pd.concat([combined_df, predict_df], ignore_index=True, sort=False)
    if 'date' in combined_df.columns:
        combined_df = combined_df.sort_values(by='date', ascending=False)
    # Display top rows for user
    logging.info("Combined predictions (latest 10 rows):\n%s", combined_df.head(10).to_string(index=False))
    logging.info("Combined predictions updated")
    return combined_df

def update_betting_statistics(combined_df, most_recent_date, prediction_dir):
    """ Update betting statistics with actual game results. """
    if combined_df is None or combined_df.empty:
        logging.warning("Combined dataframe is empty, nothing to update.")
        return None

    season_df = combined_df.copy()
    logging.info("Updating statistics using games from %s", most_recent_date)

    # Read the most recent games data with fallback
    stats_path = os.path.join(STAT_DIR, f"nba_games_{most_recent_date}.csv")
    if not os.path.exists(stats_path):
        logging.error("Expected stats file missing: %s", stats_path)
        return None

    daily_games_df = read_csv_with_fallback(stats_path)
    daily_games_df = normalize_columns(daily_games_df)

    # Validate required columns
    required = {"team", "won"}
    if not required.issubset(set(daily_games_df.columns)):
        logging.error("Daily statistics file %s missing required columns. Found columns: %s", stats_path, list(daily_games_df.columns))
        raise SystemExit(f"Missing required columns in stats file: {required - set(daily_games_df.columns)}")

    # Filter to current season only (if column exists)
    if 'season' in daily_games_df.columns:
        daily_games_df = daily_games_df[daily_games_df['season'] == CURRENT_SEASON].copy()

    # Convert date columns to datetime where possible
    if 'date' in season_df.columns:
        season_df['date'] = pd.to_datetime(season_df['date'], errors='coerce')
    daily_games_df['date'] = pd.to_datetime(daily_games_df.get('date', pd.Series(dtype='datetime64[ns]')), errors='coerce')

    # Normalize placeholder results to NaN in season_df
    if 'result' in season_df.columns:
        season_df['result'] = (
            season_df['result']
            .astype(str)
            .str.strip()
            .replace(["", "0", "nan", "None"], np.nan)
        )

    # Update result column based on actual winners
    for _, row in daily_games_df.iterrows():
        date = row.get('date')
        winning_team = row.get('team') if row.get('won') == 1 else None
        if not winning_team or pd.isna(date):
            continue
        mask = (season_df['date'] == pd.to_datetime(date)) & (
            (season_df.get('home_team') == winning_team) | (season_df.get('away_team') == winning_team)
        )
        season_df.loc[mask, 'result'] = winning_team

    # Keep only rows with valid outcomes
    if 'home_team' in season_df.columns and 'away_team' in season_df.columns:
        season_df = season_df[
            (season_df['result'] == season_df['home_team'])
            | (season_df['result'] == season_df['away_team'])
        ]
    else:
        logging.warning("home_team/away_team columns missing from combined data; skipping row filtering by teams.")

    # Ensure probabilities are numeric (handle multiple possible column names)
    if 'home_team_prob' in season_df.columns:
        season_df['home_team_prob'] = pd.to_numeric(season_df['home_team_prob'], errors='coerce')
    elif 'pred_home_win_proba' in season_df.columns:
        season_df['home_team_prob'] = pd.to_numeric(season_df['pred_home_win_proba'], errors='coerce')

    # Compute accuracy
    if 'home_team_prob' in season_df.columns and 'result' in season_df.columns:
        home_win = (season_df['home_team_prob'] >= 0.5) & (season_df['result'] == season_df['home_team'])
        away_win = (season_df['home_team_prob'] < 0.5) & (season_df['result'] == season_df['away_team'])
        season_df['accuracy'] = (home_win | away_win).astype(int)
    else:
        season_df['accuracy'] = np.nan

    # Overall accuracy
    overall = season_df['accuracy'].mean() if 'accuracy' in season_df.columns else np.nan
    logging.info("Overall accuracy: %s", f"{overall:.2%}" if not pd.isna(overall) else "N/A")

    # Subset accuracy (home_team_prob > 0.60 and <= 0.40)
    if 'home_team_prob' in season_df.columns:
        high_conf_home = season_df[season_df['home_team_prob'] > 0.60]['accuracy'].mean()
        low_conf_home = season_df[season_df['home_team_prob'] <= 0.40]['accuracy'].mean()
        logging.info("Accuracy for home_team_prob > 0.60: %s", f"{high_conf_home:.2%}" if not pd.isna(high_conf_home) else "N/A")
        logging.info("Accuracy for home_team_prob <= 0.40: %s", f"{low_conf_home:.2%}" if not pd.isna(low_conf_home) else "N/A")

    # Save updated DataFrame with today's date (use today_str_format)
    save_path = os.path.join(prediction_dir, f'combined_nba_predictions_acc_{today_str_format}.csv')
    # Drop unnamed columns generically
    unnamed = [c for c in season_df.columns if c.startswith("unnamed")]
    if unnamed:
        season_df.drop(columns=unnamed, errors='ignore', inplace=True)

    season_df.dropna(
        subset=["date", "home_team", "away_team", "home_team_prob", "result", "accuracy"],
        inplace=True,
    )
    season_df.to_csv(save_path, index=False)
    logging.info("Updated betting statistics saved to %s", save_path)
    return season_df

def main():
    """ Main execution function for updating betting statistics. """
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
