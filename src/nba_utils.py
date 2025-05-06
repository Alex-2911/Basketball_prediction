#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Prediction Utilities Library

This module provides common utility functions and configurations used
across the NBA prediction scripts.
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
import calendar
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
try:
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("webdriver_manager not installed. Some functions may not work.")
    ChromeDriverManager = None

# ============================================================================
# GLOBAL CONFIGURATIONS
# ============================================================================

# Season Configuration
CURRENT_SEASON = 2025
ROLLING_WINDOW_SIZE = 8

# Date Utilities
def get_current_date(days_offset=1):
    """
    Returns the current date with optional offset of days.

    Args:
        days_offset (int): Number of days to offset (negative means past)

    Returns:
        tuple: (datetime_obj, formatted_str, formatted_str_ymd)
    """
    date = datetime.now() - timedelta(days=days_offset)
    date_str = date.strftime("%a, %b ") + str(int(date.strftime("%d"))) + date.strftime(", %Y")
    date_ymd = date.strftime("%Y-%m-%d")

    return date, date_str, date_ymd

# Directory Structure
def get_directory_paths():
    """
    Returns a dictionary of standard directory paths used in the application.

    Returns:
        dict: Dictionary of path strings
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "output", "Gathering_Data")

    paths = {
        'BASE_DIR': base_dir,
        'DATA_DIR': data_dir,
        'STAT_DIR': os.path.join(data_dir, "Whole_Statistic"),
        'STANDINGS_DIR': os.path.join(data_dir, "data", f"{CURRENT_SEASON}_standings"),
        'SCORES_DIR': os.path.join(data_dir, "data", f"{CURRENT_SEASON}_scores"),
        'NEXT_GAME_DIR': os.path.join(data_dir, "Next_Game"),
        'PREDICTION_DIR': os.path.join(base_dir, "output", "LightGBM", f"1_{CURRENT_SEASON}_Prediction")
    }

    # Create directories if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths

# Team Code Mapping
def get_team_codes():
    """
    Returns a dictionary mapping team names to their abbreviated codes.

    Returns:
        dict: Dictionary mapping team names to codes
    """
    return {
        'Atlanta Hawks': 'ATL',
        'Boston Celtics': 'BOS',
        'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHA',
        'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW',
        'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHO',
        'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS'
    }

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def get_latest_file(folder, prefix, ext):
    """
    Find the most recent file in a folder with a given prefix and extension.

    Args:
        folder (str): Directory path to search
        prefix (str): File name prefix
        ext (str): File extension

    Returns:
        str: Path to the latest file, or None if not found
    """
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    return max(files, key=os.path.getctime) if files else None

def find_file_in_date_range(directory, filename_pattern, max_days_back=120):
    """
    Find a file matching the pattern within a specified number of days back.

    Args:
        directory (str): Directory to search
        filename_pattern (str): Pattern with {} placeholder for date
        max_days_back (int): Maximum number of days to look back

    Returns:
        tuple: (file_path, date_str) or (None, None) if not found
    """
    for days_back in range(max_days_back + 1):
        date_to_check = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        file_path = os.path.join(directory, filename_pattern.format(date_to_check))

        if os.path.exists(file_path):
            return file_path, date_to_check

    return None, None

def copy_missing_files(src_dir, dst_dir):
    """
    Copy files that exist in source directory but not in destination directory.

    Args:
        src_dir (str): Source directory
        dst_dir (str): Destination directory
    """
    import shutil

    src_files = set(os.listdir(src_dir))
    dst_files = set(os.listdir(dst_dir))
    diff = src_files - dst_files

    for file_name in diff:
        if not file_name.startswith('.') and not file_name.endswith('.ipynb'):
            shutil.copy2(os.path.join(src_dir, file_name), dst_dir)
            logging.info(f'File {file_name} copied successfully')

# ============================================================================
# WEB SCRAPING
# ============================================================================

def get_html(url, selector, sleep=5, retries=3, headless=True):
    """
    Retrieves HTML content from a webpage using Selenium WebDriver.

    Args:
        url (str): URL to scrape
        selector (str): CSS selector to find the element
        sleep (int): Base sleep time in seconds
        retries (int): Number of retry attempts
        headless (bool): Whether to use headless browser

    Returns:
        str: HTML content or None if failed
    """
    html = None
    driver = None

    try:
        # WebDriver options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")

        # Initialize WebDriver with webdriver-manager
        logging.getLogger('webdriver_manager').setLevel(logging.ERROR)
        if ChromeDriverManager:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            # Fallback to default Chrome service
            driver = webdriver.Chrome(options=chrome_options)

        for attempt in range(retries):
            try:
                driver.get(url)
                import time
                time.sleep(sleep * (2 ** attempt))  # Exponential backoff
                element = driver.find_element(By.CSS_SELECTOR, selector)
                html = element.get_attribute("innerHTML")
                break
            except TimeoutException:
                logging.warning(f"Attempt {attempt + 1}: Timeout error on {url}. Retrying...")
            except WebDriverException as e:
                logging.error(f"Webdriver error: {e}")
                break
    finally:
        if driver is not None:
            driver.quit()

    if html is None:
        logging.error(f"Failed to retrieve HTML content from {url} after {retries} attempts.")

    return html

def parse_html(html_content_or_file):
    """
    Parse HTML content either from a string or file.

    Args:
        html_content_or_file (str): HTML content or path to HTML file

    Returns:
        BeautifulSoup: Parsed soup object or None if failed
    """
    try:
        if os.path.isfile(html_content_or_file):
            with open(html_content_or_file, encoding='utf-8') as f:
                html = f.read()
        else:
            html = html_content_or_file

        soup = BeautifulSoup(html, 'html.parser')
        [s.decompose() for s in soup.select("tr.over_header, tr.thead")]
        return soup
    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        return None

# ============================================================================
# DATA PROCESSING
# ============================================================================

def rename_duplicated_columns(df):
    """
    Rename duplicated columns by appending a suffix.

    Args:
        df (DataFrame): Input DataFrame

    Returns:
        DataFrame: DataFrame with renamed columns
    """
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def preprocess_nba_data(stats_path):
    """
    Preprocess NBA data by adding target column and handling missing values.

    Args:
        stats_path (str): Path to the statistics CSV file

    Returns:
        DataFrame: Preprocessed DataFrame
    """
    # Load the data
    df = pd.read_csv(stats_path, index_col=0)

    # Sort by date
    df = df.sort_values("date")

    # Define function to add target column
    def add_target(group):
        """Adds a target column to the DataFrame group based on the 'won' column."""
        group = group.copy()  # Create a copy to avoid SettingWithCopyWarning
        group['target'] = group['won'].shift(-1)
        return group

    # Apply the preprocessing function to each team group
    df = df.groupby('team', as_index=False).apply(add_target)

    # Handle missing values - avoid inplace with chained assignment
    df = df.copy()
    df['target'] = df['target'].fillna(2)
    df['target'] = df['target'].astype(int)

    # Identify and remove columns with null values
    nulls = pd.isnull(df).sum()
    nulls = nulls[nulls > 0]
    valid_columns = df.columns[~df.columns.isin(nulls.index)]
    df = df[valid_columns].copy()

    return df

def calculate_rolling_averages(df, window_size=ROLLING_WINDOW_SIZE):
    """
    Calculate rolling averages for numeric columns grouped by team and season.

    Args:
        df (DataFrame): Input DataFrame
        window_size (int): Size of the rolling window

    Returns:
        DataFrame: DataFrame with rolling averages
    """
    # Create a copy to avoid warnings
    df_copy = df.copy()

    # Ensure season is converted to a string to avoid dimensionality issues
    df_copy['season'] = df_copy['season'].astype(str)

    # Define function to calculate rolling averages for a team
    def find_team_averages(team_df):
        # Select only numeric columns to avoid errors with non-numeric data
        numeric_cols = team_df.select_dtypes(include=[np.number]).columns
        team_numeric = team_df[numeric_cols]

        # Calculate rolling averages
        rolling = team_numeric.rolling(window_size, min_periods=1).mean()

        # Add back any non-numeric columns that should be preserved
        for col in team_df.columns:
            if col not in numeric_cols:
                rolling[col] = team_df[col]

        return rolling

    # Apply rolling average by team and season
    result = pd.DataFrame()

    # Group by team first
    for team, team_data in df_copy.groupby('team'):
        # Then by season within each team
        for season, season_data in team_data.groupby('season'):
            # Calculate rolling averages
            rolling_data = find_team_averages(season_data)
            # Append to result
            result = pd.concat([result, rolling_data], ignore_index=True)

    return result

def add_next_game_columns(df):
    """
    Add columns for the next game information using a simple and direct approach.

    Args:
        df (DataFrame): Input DataFrame

    Returns:
        DataFrame: DataFrame with added columns
    """
    # Create a copy of the DataFrame
    result_df = df.copy()

    # Initialize new columns
    result_df['home_next'] = None
    result_df['team_opp_next'] = None
    result_df['date_next'] = None

    # Manual approach without using advanced pandas functions
    # Convert DataFrame to a list of dictionaries for simpler processing
    rows = df.to_dict('records')

    # Group rows by team
    team_groups = {}
    for i, row in enumerate(rows):
        team = str(row.get('team', ''))
        if team not in team_groups:
            team_groups[team] = []
        # Store original index with row data
        team_groups[team].append((i, row))

    # Process each team
    for team, team_rows in team_groups.items():
        # Sort the team's rows by date
        sorted_team_rows = sorted(team_rows, key=lambda x: x[1].get('date', ''))

        # For each row except the last one, set next game information
        for i in range(len(sorted_team_rows) - 1):
            current_idx = sorted_team_rows[i][0]  # Original index of current row
            next_row = sorted_team_rows[i + 1][1]  # Next row data

            # Set next game info in the result DataFrame
            if 'home' in next_row:
                result_df.at[current_idx, 'home_next'] = next_row.get('home')
            if 'team_opp' in next_row:
                result_df.at[current_idx, 'team_opp_next'] = next_row.get('team_opp')
            if 'date' in next_row:
                result_df.at[current_idx, 'date_next'] = next_row.get('date')

    return result_df

# ============================================================================
# BETTING UTILITIES
# ============================================================================

def kelly_frac(p, o, f=1.0):
    """
    Calculate the Kelly criterion fraction.

    Args:
        p (float): Probability of winning
        o (float): Odds (decimal)
        f (float): Fraction of Kelly to use

    Returns:
        float: Kelly fraction (between 0 and 1)
    """
    b = o - 1
    return max(((b*p-(1-p))/b)*f, 0) if b>0 else 0

def impute_prob(ml):
    """
    Convert moneyline odds to implied probability.

    Args:
        ml (int): Moneyline odds

    Returns:
        float: Implied probability or None if invalid
    """
    if ml is None:
        return None
    ml = int(ml)
    return abs(ml)/(abs(ml)+100) if ml<0 else 100/(ml+100)

def am_to_dec(ml):
    """
    Convert American odds to decimal odds.

    Args:
        ml (int or None): American odds

    Returns:
        float: Decimal odds or None if invalid
    """
    if pd.isna(ml):
        return None
    ml = int(ml)
    return (ml/100 + 1) if ml > 0 else (100/abs(ml) + 1)

# ============================================================================
# BETTING STATISTICS
# ============================================================================

def get_home_win_rates(df):
    """
    Calculate home win rates for all teams based on recent games.

    Args:
        df (DataFrame): DataFrame with game results

    Returns:
        DataFrame: DataFrame with home win rates for each team
    """
    team_results = {}

    for team in df['home_team'].unique():
        # Get last 20 games for the team (home or away)
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)]
        team_games = team_games.sort_values(by='date', ascending=False).head(20)

        # Filter only home games from those 20
        home_games = team_games[team_games['home_team'] == team]

        # Calculate home win rate
        total_home_games = len(home_games)
        home_wins = len(home_games[home_games['result'] == team])
        home_win_rate = round(home_wins / total_home_games, 2) if total_home_games > 0 else 0

        # Store results in dictionary
        team_results[team] = {
            "Total Last 20 Games": len(team_games),
            "Total Home Games": total_home_games,
            "Home Wins": home_wins,
            "Home Win Rate": home_win_rate
        }

    # Convert to DataFrame
    home_win_rates_df = pd.DataFrame.from_dict(team_results, orient='index')

    # Sort by Home Win Rate in descending order
    home_win_rates_df.sort_values(by="Home Win Rate", ascending=False, inplace=True)

    return home_win_rates_df