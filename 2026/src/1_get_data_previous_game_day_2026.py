#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 1 of 5 (2026): Get Data for Previous Game Day

Adapted for the 2025-26 season folder structure and enhanced with:
- CLI args: --date (anchor day) or --collect-date (exact game date to pull)
- Robust month scraping & single-day box-score filtering
- Deterministic output filename: nba_games_<COLLECT_DATE>.csv
- Console pause at end so the window stays open

Usage examples
--------------
# Collect games for the day before the given anchor (collect 2025-10-21)
python 1_get_data_previous_game_day_2026.py --date 2025-10-22

# Collect games for an exact day
python 1_get_data_previous_game_day_2026.py --collect-date 2025-10-21

If no args are supplied, it behaves like the legacy script (collects "yesterday").
"""

import os
import re
import sys
import argparse
import logging
import calendar
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup

# Import shared utilities for 2026
from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    get_html,
    parse_html,
    rename_duplicated_columns,
    copy_missing_files
)

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def month_name_lower(d: date) -> str:
    return calendar.month_name[d.month].lower()

def read_line_score(soup):
    line_score = pd.read_html(StringIO(str(soup)), attrs={'id': 'line_score'})[0]
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    line_score = line_score[["team", "total"]]
    return line_score

def read_stats(soup, team, stat):
    df = pd.read_html(StringIO(str(soup)), attrs={'id': f'box-{team}-game-{stat}'}, index_col=0)[0]
    return df.apply(pd.to_numeric, errors="coerce")

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all('a')]
    return os.path.basename(hrefs[1]).split("_")[0]

def scrape_season_for_month(season, month_name, standings_dir):
    """
    Download (if missing) the monthly schedule page for the given season/month.
    Uses Selenium-backed get_html from utilities.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    selector = "#content .filter"
    html_content = get_html(url, selector)
    if not html_content:
        logging.error(f"Failed to retrieve {url}")
        return

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all("a", href=re.compile(f"/leagues/NBA_[0-9]{{4}}_games-[a-z]+\\.html"))
    monthly = [f"https://www.basketball-reference.com{l['href']}" for l in links]

    for murl in monthly:
        if f"NBA_{season}_games-{month_name}" not in murl:
            continue
        save_path = os.path.join(standings_dir, murl.split("/")[-1])
        if os.path.exists(save_path):
            logging.info(f"Already have {save_path}")
            return
        html = get_html(murl, "#all_schedule")
        if not html:
            logging.warning(f"Could not fetch monthly page: {murl}")
            return
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"Saved {save_path}")
        return

def scrape_game_day_boxscores(standings_file, scores_dir, target_games_date: date):
    """
    From a monthly standings page, find boxscore links for the target_games_date and save them.
    """
    with open(standings_file, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    hrefs = [a.get('href') for a in soup.find_all("a")]
    # basketball-reference box score URL contains YYYYMMDD in path before .html
    wanted_tag = target_games_date.strftime("%Y%m%d")
    box_scores = [
        f"https://www.basketball-reference.com{h}"
        for h in hrefs if h and "boxscores" in h and h.endswith(".html") and wanted_tag in h
    ]

    saved = 0
    for url in box_scores:
        save_path = os.path.join(scores_dir, os.path.basename(url))
        if os.path.exists(save_path):
            continue
        page_html = get_html(url, "#content")
        if not page_html:
            logging.warning(f"Failed to fetch box score: {url}")
            continue
        with open(save_path, "wb") as f:
            f.write(page_html.encode("utf-8"))
        saved += 1
        logging.info(f"Saved box score → {save_path}")
    return saved

def process_saved_boxscores(scores_dir, existing_statistics, target_games_date: date):
    """
    Parse saved boxscore HTML files for exactly target_games_date and return a DataFrame.
    """
    box_files = [os.path.join(scores_dir, f) for f in os.listdir(scores_dir) if f.endswith(".html")]
    if not box_files:
        logging.warning("No box score files found.")
        return pd.DataFrame()

    games = []
    base_cols = None
    for p in box_files:
        try:
            fdate = pd.Timestamp(os.path.basename(p)[:8]).date()
            if fdate != target_games_date:
                continue
            soup = parse_html(p)
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
            full_game["date"] = pd.Timestamp(os.path.basename(p)[:8])
            full_game["won"] = full_game["total"] > full_game["total_opp"]
            games.append(full_game)

        except Exception as e:
            logging.error(f"Error processing {p}: {e}")

    if not games:
        return pd.DataFrame()

    games_df = pd.concat(games, ignore_index=True)
    games_df = rename_duplicated_columns(games_df)

    # Align columns to existing statistics if provided
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    return games_df

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    # Parse CLI
    parser = argparse.ArgumentParser(description="Collect previous NBA game day data (2026 season).")
    parser.add_argument("--date", type=str, help="Anchor date in YYYY-MM-DD; script collects games from the day before.")
    parser.add_argument("--collect-date", type=str, help="Exact game date to collect in YYYY-MM-DD (overrides --date).")
    args = parser.parse_args()

    # Establish anchor/today and target (games) date
    # Default behavior (legacy): collect 'yesterday' relative to get_current_date()
    today_dt, today_str, today_str_format = get_current_date()
    if args.collect_date:
        target_games_date = parse_ymd(args.collect_date)
        logging.info(f"Collecting games for exact date: {target_games_date}")
    elif args.date:
        anchor = parse_ymd(args.date)
        target_games_date = anchor - timedelta(days=1)
        logging.info(f"Anchor date: {anchor} → collecting prior day: {target_games_date}")
    else:
        # Legacy: utils' get_current_date returns a shifted date; we still interpret "yesterday"
        _, _, _ = get_current_date(days_offset=1)  # keep side-effects identical if any
        target_games_date = (datetime.now() - timedelta(days=1)).date()
        logging.info(f"No args provided → collecting yesterday: {target_games_date}")

    # Paths
    paths = get_directory_paths()
    DATA_DIR = paths['DATA_DIR']
    STAT_DIR = paths['STAT_DIR']
    STANDINGS_DIR = paths['STANDINGS_DIR']
    SCORES_DIR = paths['SCORES_DIR']
    DST_DIR = STAT_DIR

    # Ensure monthly standings page exists for target month
    month_name = month_name_lower(target_games_date)
    monthly_file = os.path.join(STANDINGS_DIR, f"NBA_{CURRENT_SEASON}_games-{month_name}.html")
    if not os.path.exists(monthly_file):
        logging.info(f"Monthly standings missing → scraping {month_name} {CURRENT_SEASON}")
        scrape_season_for_month(CURRENT_SEASON, month_name, STANDINGS_DIR)
    if not os.path.exists(monthly_file):
        logging.warning(f"Monthly standings file still missing: {monthly_file}")

    # Try to load an existing stats file to align columns (latest available)
    existing_statistics = None
    try:
        # Search backwards up to 150 days for an existing daily file to get column layout
        for back in range(0, 151):
            cand = (target_games_date - timedelta(days=back)).strftime("%Y-%m-%d")
            f = os.path.join(STAT_DIR, f"nba_games_{cand}.csv")
            if os.path.exists(f):
                existing_statistics = pd.read_csv(f)
                logging.info(f"Using existing statistics layout from: {f}")
                break
    except Exception as e:
        logging.warning(f"Could not load existing stats layout: {e}")

    # Scrape and save box scores for the exact target date
    saved = 0
    if os.path.exists(monthly_file):
        saved = scrape_game_day_boxscores(monthly_file, SCORES_DIR, target_games_date)
        logging.info(f"Saved {saved} box scores for {target_games_date}")
    else:
        logging.warning("Proceeding without monthly file; expecting box scores possibly already present.")

    # Process saved boxscores into a dataframe (exact date only)
    games_df = process_saved_boxscores(SCORES_DIR, existing_statistics, target_games_date)

    if games_df is None or games_df.empty:
        logging.warning(f"No games parsed for {target_games_date}. Nothing to append.")
        # Still keep the window open at the end
        try:
            input("No games parsed. Press Enter to close this window...")
        except EOFError:
            pass
        return

    # If we have an existing layout, ensure alignment again (safety)
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    # Load previous file for target date if exists, else build from closest prior
    out_daily = os.path.join(STAT_DIR, f"nba_games_{target_games_date}.csv")
    if os.path.exists(out_daily):
        prev = pd.read_csv(out_daily)
        combined = pd.concat([prev, games_df], ignore_index=True)
    elif existing_statistics is not None:
        combined = pd.concat([existing_statistics, games_df], ignore_index=True).drop_duplicates()
    else:
        combined = games_df

    # Save
    combined.to_csv(out_daily, index=False)
    logging.info(f"Combined statistics saved → {out_daily}")

    # Copy any missing files (kept for parity with legacy behavior)
    copy_missing_files(STAT_DIR, DST_DIR)

    # Keep console open
    try:
        input("Done. Press Enter to close this window...")
    except EOFError:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
        try:
            input("\nAn error occurred. Press Enter to close this window...")
        except EOFError:
            pass
