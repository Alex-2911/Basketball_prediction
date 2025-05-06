#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 2 of 5: Get Data for Next Game Day

This script collects data for the next NBA game day.
It scrapes data from basketball-reference.com and saves upcoming games information.

Ensure "1_get_data_previous_game_day.py" is executed before running this script.
"""

import os
import requests
import pandas as pd
import calendar
import shutil
from bs4 import BeautifulSoup
from datetime import datetime

# Import shared utilities
from nba_utils import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    get_team_codes
)

# Get current date information
today, today_str, today_str_format = get_current_date()
today_date = datetime.strptime(today_str, "%a, %b %d, %Y")

print(f"Today's date: {today_str}")

# Set up month information
month_for_coming_games = today.month
month_name_for_coming_games = calendar.month_name[month_for_coming_games].lower()
print(month_name_for_coming_games)

# Get directory paths
paths = get_directory_paths()
DATA_DIR = paths['DATA_DIR']
STANDINGS_DIR = paths['STANDINGS_DIR']
target_folder = paths['NEXT_GAME_DIR']
dst_dir = paths['NEXT_GAME_DIR']

# Define file paths
file_name = f"NBA_{CURRENT_SEASON}_games-{month_name_for_coming_games}.html"
file_path = os.path.join(STANDINGS_DIR, file_name)

print(f"Checking file: {file_path}")

def scrape_season_for_month(season, month, month_name, standings_dir):
    """
    Scrapes NBA games data for a specific month and season from basketball-reference.com.

    Args:
        season (int): The NBA season year.
        month (int): The month number (1-12).
        month_name (str): The name of the month.
        standings_dir (str): Directory path to save the scraped data.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    print(url)

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all("a", href=True)
        month_link = None
        for link in links:
            if f"NBA_{season}_games-{month_name}" in link['href']:
                month_link = "https://www.basketball-reference.com" + link['href']
                break
        if month_link:
            month_response = requests.get(month_link)
            month_response.raise_for_status()
            with open(os.path.join(standings_dir, f"NBA_{season}_games-{month_name}.html"), "w", encoding='utf-8') as f:
                f.write(month_response.text)
            print(f"Data for {month_name.title()} saved.")
        else:
            print(f"No link found for {month_name.title()} {season}.")
    except requests.RequestException as e:
        print(f"An error occurred while fetching data: {e}")

def find_games_for_next_day(today_date, file_paths):
    """
    Find games scheduled for the next day after today_date.

    Args:
        today_date (datetime): The current date to find games after.
        file_paths (list): List of paths to files containing the game schedules.

    Returns:
        list: List of dictionaries containing game information.
    """
    next_game_info = []
    next_game_date = None  # Placeholder to track the date of the first game found

    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, "r", encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')

            table = soup.find("table", {"id": "schedule"})

            if table:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip the header row
                    game_date_tag = row.find("th", {"data-stat": "date_game"})
                    if game_date_tag:
                        game_date_str = game_date_tag.text.strip()
                        try:
                            game_date = datetime.strptime(game_date_str, "%a, %b %d, %Y")
                        except ValueError:
                            continue

                        # If it's the first game on or after today_date, set next_game_date
                        if game_date >= today_date and next_game_date is None:
                            next_game_date = game_date

                        # Collect games for the determined next game date
                        if game_date == next_game_date:
                            cols = row.find_all("td")
                            if len(cols) >= 4:
                                visitor_team = cols[1].text.strip()
                                home_team = cols[3].text.strip()
                                next_game_info.append({
                                    'date': game_date,
                                    'home_team': home_team,
                                    'visitor_team': visitor_team
                                })
                        # Stop once we've collected all games for the next available game day
                        elif next_game_date is not None and game_date > next_game_date:
                            return next_game_info

        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

    return next_game_info

def main():
    """Main execution function."""
    # Directory paths are already created in get_directory_paths()

    # Ensure standings file exists
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        scrape_season_for_month(CURRENT_SEASON, month_for_coming_games, month_name_for_coming_games, STANDINGS_DIR)

    # Find next game information
    next_game_info = find_games_for_next_day(today_date, [file_path])

    # Look ahead if no games found for the current month
    if not next_game_info:
        next_month = month_for_coming_games % 12 + 1  # Handle month transition properly
        months_checked = 0
        max_months_to_check = 12  # Check up to 12 months ahead

        while not next_game_info and months_checked < max_months_to_check:
            month_name = calendar.month_name[next_month].lower()
            next_file_path = os.path.join(STANDINGS_DIR, f"NBA_{CURRENT_SEASON}_games-{month_name}.html")
            print(next_file_path, 'AAAA')

            if not os.path.exists(next_file_path):
                scrape_season_for_month(CURRENT_SEASON, next_month, month_name, STANDINGS_DIR)

            next_game_info = find_games_for_next_day(today_date, [next_file_path])
            next_month = (next_month % 12) + 1  # Move to the next month
            months_checked += 1

    # Get team code mapping
    team_codes = get_team_codes()

    # Process game information if available
    games = []
    if next_game_info:
        for game in next_game_info:
            home_team_name = team_codes.get(game['home_team'], game['home_team'])
            visitor_team_name = team_codes.get(game['visitor_team'], game['visitor_team'])
            games.append((home_team_name, visitor_team_name, game['date'].strftime("%Y-%m-%d")))

            print(f"The game is on {game['date'].strftime('%a, %b %d, %Y')}.")
            print(f"Teams: {game['visitor_team']} vs {game['home_team']}")
    else:
        print("No games found for today.")

    # Create DataFrame
    games_df = pd.DataFrame(games, columns=['home_team', 'away_team', 'game_date'])
    print(games_df)

    # Save the DataFrame to a CSV file
    output_file_path = os.path.join(target_folder, f'games_df_{today_str_format}.csv')
    games_df.to_csv(output_file_path)
    print(f"games_df has been saved in the folder: {target_folder}")

    # Copy files to destination directory
    if os.path.exists(target_folder) and os.path.isdir(target_folder):
        # Create the destination folder if it does not exist
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Get a list of all files in the source directory
        files_to_copy = [file_name for file_name in os.listdir(target_folder) if not file_name.startswith('.ipynb_checkpoints')]

        # Flag to check if any new files were copied
        new_files_copied = False

        # Copy each file from source to destination if it does not already exist
        for file_name in files_to_copy:
            target_folder_file_path = os.path.join(target_folder, file_name)
            dst_file_path = os.path.join(dst_dir, file_name)

            # Copy the file if it does not exist in the destination folder
            if not os.path.exists(dst_file_path):
                shutil.copy2(target_folder_file_path, dst_file_path)
                print(f'File "{file_name}" copied successfully.')
                new_files_copied = True

        # If no new files were copied, display a message
        if not new_files_copied:
            print("No new files to copy.")
    else:
        print(f'Source folder "{target_folder}" does not exist or is not a directory.')

if __name__ == "__main__":
    main()