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
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            file_found = True
            logging.info(f"Found prediction file for {date_str}: {file_path}")
            return [file_path], date_str
        else:
            days_back += 1

    logging.warning("No prediction file found in the last %d days.", MAX_DAYS_BACK)
    return None, None


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


def process_prediction_file(predict_file, last_prediction):
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
    predict_df = pd.read_csv(predict_file[0])
    # Normalize decimal columns in odds
    for col in ['odds 1', 'odds 2']:
        if col in predict_df.columns:
            predict_df[col] = predict_df[col].astype(str).str.replace(',', '.').astype(float)

    # Path for combined data (one file per prediction date)
    combined_file_path = os.path.join(directory_path, f'combined_nba_predictions_acc_{last_prediction}.csv')
    # Load existing combined file or create new
    try:
        combined_df = pd.read_csv(combined_file_path)
    except FileNotFoundError:
        combined_df = pd.DataFrame()

    # Append and sort by date descending
    predict_df['accuracy'] = np.nan  # add placeholder column
    combined_df = pd.concat([combined_df, predict_df], ignore_index=True)
    combined_df = combined_df.sort_values(by='date', ascending=False)
    # Display top rows for user
    logging.info("Combined predictions (latest 10 rows):\n%s", combined_df.head(10).to_string(index=False))
    logging.info("Combined predictions updated")
    return combined_df


def update_betting_statistics(combined_df, most_recent_date):
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

    # Ensure probabilities are numeric
    season_df['home_team_prob'] = pd.to_numeric(season_df['home_team_prob'], errors='coerce')

    # Compute accuracy
    home_win = (season_df['home_team_prob'] >= 0.5) & (season_df['result'] == season_df['home_team'])
    away_win = (season_df['home_team_prob'] < 0.5) & (season_df['result'] == season_df['away_team'])
    season_df['accuracy'] = (home_win | away_win).astype(int)

    # Overall accuracy
    overall = season_df['accuracy'].mean()
    logging.info(f"Overall accuracy: {overall:.2%}")

    # Subset accuracy (home_team_prob > 0.60 and < 0.40)
    high_conf_home = season_df[season_df['home_team_prob'] > 0.60]['accuracy'].mean()
    low_conf_home = season_df[season_df['home_team_prob'] <= 0.40]['accuracy'].mean()
    logging.info(f"Accuracy for home_team_prob > 0.60: {high_conf_home:.2%}")
    logging.info(f"Accuracy for home_team_prob <= 0.40: {low_conf_home:.2%}")

    # Save updated DataFrame with today's date
    save_path = os.path.join(directory_path, f'combined_nba_predictions_acc_{today_str_format}.csv')
    # Drop unnamed columns and NaNs
    season_df.drop(columns=['Unnamed: 8'], errors='ignore', inplace=True)
    season_df.dropna(inplace=True)
    season_df.to_csv(save_path, index=False)
    logging.info(f"Updated betting statistics saved to {save_path}")
    return season_df


def main():
    """Main execution function for updating betting statistics."""
    # Find most recent prediction file
    predict_file, last_prediction = find_most_recent_prediction_file()
    if not predict_file:
        print("No recent prediction file found.")
        return

    # Process prediction file
    combined_df = process_prediction_file(predict_file, last_prediction)
    if combined_df is None:
        print("Failed to process prediction file.")
        return

    # Find most recent statistics file
    most_recent_date = find_most_recent_statistics_file()
    if not most_recent_date:
        print("No recent statistics file found.")
        return

    # Update betting statistics
    updated_df = update_betting_statistics(combined_df, most_recent_date)
    if updated_df is not None:
        print("Betting statistics updated successfully.")
    else:
        print("Failed to update betting statistics.")


if __name__ == "__main__":
    main()
    # Keep the terminal window open after script completion so users can review
    # the output and logs. In non-interactive environments (e.g., GitHub
    # Actions), input() will immediately raise EOFError, so we catch and
    # ignore it to avoid breaking automated runs.
    try:
        input("Press Enter to close this window...")
    except EOFError:
        pass