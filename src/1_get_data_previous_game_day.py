#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 1 of 5: Get Data for Previous Game Day

This script collects and processes data for the previous NBA game day.
It scrapes data from basketball-reference.com and updates the statistics database.

Ensure this script is run before executing "2_get_data_next_game_day.py".
"""

import os
import pandas as pd
import shutil
from io import StringIO
import numpy as np
import re
import logging
import time
import calendar
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# Define the current season
current_season = 2025

# Configure today's date
today = datetime.now()
today_str = today.strftime("%a, %b ") + str(int(today.strftime("%d"))) + today.strftime(", %Y")
today_date = datetime.strptime(today_str, "%a, %b %d, %Y")
today_str_format = today_date.strftime("%Y-%m-%d")
yesterday = datetime.now() - timedelta(days=1)

print(f"Today's date: {today_str}")

# Directories
DATA_DIR = os.path.join("D:\\", "1. Python", "1. NBA Script", "2025", "Gathering_Data")
STAT_DIR = os.path.join(DATA_DIR, "Whole_Statistic")
STANDINGS_DIR = os.path.join(DATA_DIR, "data", f"{current_season}_standings")
SCORES_DIR = os.path.join(DATA_DIR, "data", f"{current_season}_scores")
DST_DIR = os.path.join("D:\\", "_Laufwerk C", "11. Sorare", "NBA", "2025", "Gathering_Data", "Whole_Statistic")

# Path to chromedriver
chromedriver_path = r"D:\1. Python\3. chromedriver-win64\chromedriver-win64\chromedriver.exe"
chrome_options = Options()
chrome_options.add_argument("--headless")

# Configure month information
if today.day == 1:
    current_month = today.month - 1
    
    if current_month == 0:
        current_month = 12
        current_year = today.year - 1
        last_month = 12
        last_month_name = calendar.month_name[last_month].lower()
    else:
        current_year = today.year
        last_month = current_month
        last_month_name = calendar.month_name[last_month].lower()
else:
    current_month = today.month
    current_year = today.year
    last_month = None

# Configure logging
chrome_options.add_argument("--disable-ipv6")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_current_year_and_month():
    """Returns the current year and month."""
    return today.year, today.month

def get_html(url, selector, sleep=5, retries=3, headless=True):
    """Retrieves HTML content from a webpage using Selenium WebDriver."""
    html = None
    driver = None

    try:
        # WebDriver options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")

        # Initialize WebDriver with Service
        service = Service(chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        for attempt in range(retries):
            try:
                driver.get(url)
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

def scrape_season_for_month(season, month, month_name, standings_dir, get_html_function):
    """
    Scrapes NBA games data for a specific month and season from basketball-reference.com.

    Args:
        season (int): The NBA season year.
        month (int): The month number (1-12).
        month_name (str): The name of the month.
        standings_dir (str): Directory path to save the scraped data.
        get_html_function (function): Function to get HTML content from a URL.
    """
    if season < current_year or (season == current_year and month < current_month):
        logging.warning("Invalid year or month already passed.")
        return

    logging.info(f"Scraping games for: {season}, {month_name.title()}")
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    selector = "#content .filter"
    html_content = get_html_function(url, selector)

    if not html_content:
        logging.error(f"Failed to retrieve data from {url}.")
        return

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all("a", href=re.compile(f"/leagues/NBA_[0-9]{{4}}_games-[a-z]+\\.html"))
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]

    for url in standings_pages:
        save_path = os.path.join(standings_dir, url.split("/")[-1])
        if os.path.exists(save_path):
            logging.info(f"Path already exists: {save_path}")
            continue

        if month_name in save_path.lower():
            html = get_html_function(url, "#all_schedule")
            if html:
                try:
                    with open(save_path, "w+", encoding='utf-8') as f:
                        f.write(html)
                    logging.info(f"Data for {month_name.title()} saved.")
                except Exception as e:
                    logging.error(f"Failed to save data for {month_name.title()}: {e}")
            else:
                logging.error(f"Failed to retrieve data for {month_name.title()} from {url}.")

def scrape_game(standings_file, scores_dir, get_html_function):
    """
    Scrapes box scores for NBA games from the provided standings file.

    Args:
        standings_file (str): Path to the file containing the standings data.
        scores_dir (str): Directory path to save the scraped box scores.
        get_html_function (function): Function to get HTML content from a URL.
    """
    # Calculate the date of yesterday
    yesterday_date = yesterday.strftime("%Y%m%d")  # Format: YYYYMMDD
    
    with open(standings_file, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]

    filtered_urls = [url for url in box_scores if yesterday_date in url]

    for url in filtered_urls:
        save_path = os.path.join(scores_dir, url.split("/")[-1])
        
        if os.path.exists(save_path):
            continue

        html = get_html_function(url, "#content")
        
        if not html:
            continue

        try:
            with open(save_path, "wb+") as f:
                f.write(html.encode("utf-8"))
            print(f"Box score saved: {save_path}")
        except Exception as e:
            print(f"Failed to save box score: {e}")

def process_standings_files(standings_dir, current_season):
    """
    Process standings files for a specific season.

    Args:
        standings_dir (str): Directory containing standings files.
        current_season (int): The current NBA season year.
    """
    standings_files = os.listdir(standings_dir)

    # Filter files for the current season
    files = [s for s in standings_files if str(current_season) in s]

    for f in files:
        filepath = os.path.join(standings_dir, f)
        scrape_game(filepath, SCORES_DIR, get_html)

def get_first_game_date(standings_file):
    """
    Extract the date of the first game day from the standings file.
    
    Args:
        standings_file (str): Path to the standings HTML file.
    
    Returns:
        str: The date of the first game, formatted as 'Day, Month Date, Year'.
    """
    with open(standings_file, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the first game date in the standings table
    table = soup.find("table", {"id": "schedule"})
    
    if not table:
        print(f"No schedule table found in {standings_file}.")
        return None

    # Look for the first non-header row (actual game data)
    first_game_row = table.find_all("tr")[1]  # Skip the header row and take the first game row
    if first_game_row:
        game_date_tag = first_game_row.find("th", {"data-stat": "date_game"})
        if game_date_tag:
            return game_date_tag.text.strip()  # Returns the date of the first game
    
    return None

def find_most_recent_file(max_days=150):
    """
    Find the most recent statistics file within the specified number of days.
    
    Args:
        max_days (int): Maximum number of days to look back.
        
    Returns:
        str: Date string of the most recent file in YYYY-MM-DD format.
    """
    for days_back in range(1, max_days + 1):
        date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        if file_exists(date_str):
            return date_str
    return None

def file_exists(date_str):
    """
    Check if a file exists for the given date.
    
    Args:
        date_str (str): Date string in YYYY-MM-DD format.
        
    Returns:
        bool: True if the file exists, False otherwise.
    """
    filename = f"nba_games_{date_str}.csv"
    return os.path.isfile(os.path.join(STAT_DIR, filename))

def parse_html(box_score):
    """Parse HTML content from a box score file."""
    try:
        with open(box_score, encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        [s.decompose() for s in soup.select("tr.over_header, tr.thead")]
        return soup
    except Exception as e:
        logging.error(f"Error parsing HTML for {box_score}: {e}")
        return None

def read_line_score(soup):
    """Read line score from the soup object."""
    line_score = pd.read_html(StringIO(str(soup)), attrs={'id': 'line_score'})[0]
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    line_score = line_score[["team", "total"]]
    return line_score

def read_stats(soup, team, stat):
    """Read team statistics from the soup object."""
    df = pd.read_html(StringIO(str(soup)), attrs={'id': f'box-{team}-game-{stat}'}, index_col=0)[0]
    return df.apply(pd.to_numeric, errors="coerce")

def read_season_info(soup):
    """Extract season information from the soup object."""
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all('a')]
    return os.path.basename(hrefs[1]).split("_")[0]

def rename_duplicated_columns(df):
    """Rename duplicated columns by appending a suffix."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def copy_missing_files(src_dir, dst_dir):
    """Copy missing files from source to destination directory."""
    src_files = set(os.listdir(src_dir))
    dst_files = set(os.listdir(dst_dir))
    diff = src_files - dst_files

    for file_name in diff:
        if not file_name.startswith('.') and not file_name.endswith('.ipynb'):
            shutil.copy2(os.path.join(src_dir, file_name), dst_dir)
            logging.info(f'File {file_name} copied successfully')

def process_nba_data():
    """Main function to process NBA data."""
    # Use the most recent date for the file
    last_file_date = find_most_recent_file()
    print(f"Processing file for: {last_file_date}")

    filename = f"nba_games_{last_file_date}.csv"

    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load existing statistics
        existing_statistics = pd.read_csv(os.path.join(STAT_DIR, filename))
    except FileNotFoundError:
        logging.error(f"File {filename} not found in {STAT_DIR}.")
        return

    base_cols = None
    games = []

    # List of all box score HTML files
    box_scores = [os.path.join(SCORES_DIR, f) for f in os.listdir(SCORES_DIR) if f.endswith(".html")]

    if not box_scores:
        logging.warning("No box score files found in the SCORES_DIR.")
        return

    logging.info(f"Number of box score files found: {len(box_scores)}")

    # Process each box score
    for box_score in box_scores:
        try:
            date = pd.Timestamp(os.path.basename(box_score)[:8]).date()
            if date < pd.Timestamp(yesterday).date():
                continue

            logging.debug(f"Processing box score: {box_score}, Date: {date}")

            soup = parse_html(box_score)
            if soup is None:
                continue

            line_score = read_line_score(soup)
            teams = list(line_score["team"])
            summaries = []

            for team in teams:
                basic = read_stats(soup, team, "basic")
                advanced = read_stats(soup, team, "advanced")

                totals = pd.concat([basic.iloc[-1], advanced.iloc[-1]])
                totals.index = totals.index.str.lower()

                maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()])
                maxes.index = maxes.index.str.lower() + "_max"

                summary = pd.concat([totals, maxes])
                if base_cols is None:
                    base_cols = [b for b in summary.index.drop_duplicates(keep="first") if "bpm" not in b]
                summary = summary[base_cols]
                summaries.append(summary)

            summary = pd.concat(summaries, axis=1).T
            game = pd.concat([summary, line_score], axis=1)
            game["home"] = [0, 1]

            game_opp = game.iloc[::-1].reset_index()
            game_opp.columns += "_opp"

            full_game = pd.concat([game, game_opp], axis=1)

            full_game["season"] = read_season_info(soup)
            full_game["date"] = pd.Timestamp(os.path.basename(box_score)[:8])
            full_game["won"] = full_game["total"] > full_game["total_opp"]
            games.append(full_game)

            if len(games) % 100 == 0:
                logging.info(f"{len(games)} / {len(box_scores)} processed.")

        except Exception as e:
            logging.error(f"Error processing {box_score}: {e}")

    # Warning if no games were processed
    if not games:
        logging.warning("No games were played yesterday or no valid box scores were found.")
        return
    else:
        logging.info(f"{len(games)} games were processed.")

    # If games were processed, create a DataFrame
    games_df = pd.concat(games, ignore_index=True)
    print(f"Sample processed data:\n{games_df.head(1).to_string(index=False)}")

    # Rename duplicated columns in games_df (if any)
    games_df = rename_duplicated_columns(games_df)

    # Align columns to existing statistics
    games_df = games_df.reindex(columns=existing_statistics.columns)

    # Combine new data with existing statistics
    combined_statistics = pd.concat([existing_statistics, games_df], ignore_index=True)

    # Save the combined statistics
    file_name = f"nba_games_{today_date}.csv"
    combined_statistics.to_csv(os.path.join(STAT_DIR, file_name), index=False)
    logging.info(f"Combined statistics saved to: {os.path.join(STAT_DIR, file_name)}")

    # Copy any missing files
    copy_missing_files(STAT_DIR, DST_DIR)

def main():
    """Main execution function."""
    # Create directories if they don't exist
    os.makedirs(STANDINGS_DIR, exist_ok=True)
    os.makedirs(SCORES_DIR, exist_ok=True)
    os.makedirs(STAT_DIR, exist_ok=True)
    
    next_month = current_month + 1 if current_month < 12 else 1
    next_year = current_year if next_month != 1 else current_year + 1
    current_month_name = calendar.month_name[current_month].lower()
    next_month_name = calendar.month_name[next_month].lower()

    # File removal logic for current month
    file_to_remove = f"NBA_{current_season}_games-{current_month_name}.html"
    file_path = os.path.join(STANDINGS_DIR, file_to_remove)

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"File {file_to_remove} has been removed.")
        else:
            logging.info(f"File {file_to_remove} does not exist.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    # Scrape games for the current and next month
    if today.day == 1 and last_month:
        scrape_season_for_month(current_season, last_month, last_month_name, STANDINGS_DIR, get_html)
        scrape_season_for_month(current_season, next_month, next_month_name, STANDINGS_DIR, get_html)
    else:
        scrape_season_for_month(current_season, current_month, current_month_name, STANDINGS_DIR, get_html)

    # Process standings files
    process_standings_files(STANDINGS_DIR, current_season)
    
    # Get first game date to determine season start
    standings_file = os.path.join(STANDINGS_DIR, f'NBA_{current_season}_games-october.html')
    first_game_date_str = get_first_game_date(standings_file)
    
    if first_game_date_str:
        print(f"The first game is scheduled on: {first_game_date_str}")
        first_game_date = datetime.strptime(first_game_date_str, "%a, %b %d, %Y").date()
        
        # Check if season has started
        if today.date() >= first_game_date:
            # Process NBA data
            process_nba_data()
        else:
            print(f"Season has not started yet. Today: {today.date()}, First game: {first_game_date}")
    else:
        print("Could not determine first game date.")

if __name__ == "__main__":
    main()
