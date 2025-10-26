#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 1 of 5 (2026): Get Data for Previous Game Day

- scrapes the most recent box scores from basketball-reference
- refreshes the current month's schedule HTML so we always get new links
- parses only yesterday's games
- appends them to our running dataset
- saves a NEW daily snapshot file with TODAY'S date in the filename
  (so 2025-10-24 games → nba_games_2025-10-25.csv)

Double-click runnable: no CLI args needed.
You can still call with --date or --collect-date if you want.
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

from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    get_html,
    parse_html,
    rename_duplicated_columns,
    copy_missing_files,
)

# ──────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ──────────────────────────────────────────────────────────────
# helpers
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
    download (always fresh) the monthly schedule html for given month.
    we *delete* the stale file first so selenium is forced to refetch.
    """
    os.makedirs(standings_dir, exist_ok=True)
    monthly_filename = f"NBA_{season}_games-{month_name}.html"
    monthly_path = os.path.join(standings_dir, monthly_filename)

    if os.path.exists(monthly_path):
        try:
            os.remove(monthly_path)
            logging.info(f"Deleted outdated monthly file: {monthly_path}")
        except Exception as e:
            logging.error(f"Could not delete {monthly_path}: {e}")

    # hit the season landing page to discover month URLs
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    selector = "#content .filter"
    html_content = get_html(url, selector)
    if not html_content:
        logging.error(f"Failed to retrieve {url}")
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all("a", href=re.compile(r"/leagues/NBA_[0-9]{4}_games-[a-z]+\.html"))

    # find the entry that matches the month_name we want
    wanted_url = None
    for l in links:
        href = l.get("href", "")
        if f"NBA_{season}_games-{month_name}" in href:
            wanted_url = "https://www.basketball-reference.com" + href
            break

    if not wanted_url:
        logging.warning(f"No monthly url found for month '{month_name}' in season {season}")
        return None

    logging.info(f"Fetching fresh month page: {wanted_url}")
    month_html = get_html(wanted_url, "#all_schedule")
    if not month_html:
        logging.warning(f"Could not fetch monthly page: {wanted_url}")
        return None

    try:
        with open(monthly_path, "w", encoding="utf-8") as f:
            f.write(month_html)
        logging.info(f"Saved fresh monthly file → {monthly_path}")
    except Exception as e:
        logging.error(f"Error saving {monthly_path}: {e}")
        return None

    return monthly_path

def scrape_game_day_boxscores(standings_file, scores_dir, target_games_date: date):
    """
    from that month page:
    - find only boxscores for target_games_date (yyyyMMdd in URL)
    - download any missing boxscore htmls into scores_dir
    """
    os.makedirs(scores_dir, exist_ok=True)

    with open(standings_file, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    hrefs = [a.get('href') for a in soup.find_all("a")]

    wanted_tag = target_games_date.strftime("%Y%m%d")
    box_scores = [
        "https://www.basketball-reference.com" + h
        for h in hrefs
        if h and "boxscores" in h and h.endswith(".html") and wanted_tag in h
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
    read all downloaded boxscore htmls for *exactly* target_games_date
    build per-game rows aligned with existing_statistics columns
    """
    box_files = [
        os.path.join(scores_dir, f)
        for f in os.listdir(scores_dir)
        if f.endswith(".html")
    ]
    if not box_files:
        logging.warning("No box score files found.")
        return pd.DataFrame()

    games = []
    base_cols = None

    for p in box_files:
        try:
            fdate = pd.Timestamp(os.path.basename(p)[:8]).date()
            if fdate != target_games_date:
                # skip old html from other days
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
                    base_cols = [
                        b for b in summary.index.drop_duplicates(keep="first")
                        if "bpm" not in b
                    ]
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

    # align columns to historical layout if we have it
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    return games_df


# ──────────────────────────────────────────────────────────────
# main()
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Collect previous NBA game day data (2026 season)."
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Anchor date in YYYY-MM-DD; script collects games from the day before."
    )
    parser.add_argument(
        "--collect-date",
        type=str,
        help="Exact game date to collect in YYYY-MM-DD (overrides --date)."
    )
    args = parser.parse_args()

    # figure out which day to scrape (= yesterday by default)
    # and what filename date we want to save under (= today by default)
    now_dt, _, today_str_ymd = get_current_date(days_offset=0)
    today_date = now_dt.date()

    if args.collect_date:
        target_games_date = parse_ymd(args.collect_date)         # scrape this date's games
        save_as_date = today_date                                # still save as "today"
        logging.info(f"Collecting games for exact date: {target_games_date}")
    elif args.date:
        anchor = parse_ymd(args.date)
        target_games_date = anchor - timedelta(days=1)
        save_as_date = anchor                                    # save as anchor date
        logging.info(f"Anchor date: {anchor} → collecting prior day: {target_games_date}")
    else:
        # double-click / default:
        # we run on "today", scrape "yesterday", and save file as "today"
        target_games_date = today_date - timedelta(days=1)
        save_as_date = today_date
        logging.info(f"No args → scraping yesterday {target_games_date} and saving snapshot as {save_as_date}")

    # paths
    paths = get_directory_paths()
    STAT_DIR = paths["STAT_DIR"]
    STANDINGS_DIR = paths["STANDINGS_DIR"]
    SCORES_DIR = paths["SCORES_DIR"]
    DST_DIR = STAT_DIR

    # refresh the month page for the target_games_date month
    month_name = month_name_lower(target_games_date)
    fresh_monthly_file = scrape_season_for_month(
        CURRENT_SEASON,
        month_name,
        STANDINGS_DIR
    )

    # if scrape failed somehow, fall back to whatever file is already there
    if fresh_monthly_file is None:
        fresh_monthly_file = os.path.join(
            STANDINGS_DIR,
            f"NBA_{CURRENT_SEASON}_games-{month_name}.html"
        )
        if not os.path.exists(fresh_monthly_file):
            logging.warning(f"No monthly file available for {month_name}, cannot continue scraping box scores.")
            _pause_and_exit_ok()
            return

    # load an existing stats file to get column layout
    existing_statistics = None
    try:
        # walk backwards from target_games_date to find the most recent .csv
        for back in range(0, 151):
            cand = (target_games_date - timedelta(days=back)).strftime("%Y-%m-%d")
            f = os.path.join(STAT_DIR, f"nba_games_{cand}.csv")
            if os.path.exists(f):
                existing_statistics = pd.read_csv(f)
                logging.info(f"Using existing statistics layout from: {f}")
                break
    except Exception as e:
        logging.warning(f"Could not load existing stats layout: {e}")

    # download (if missing) yesterday's individual box scores into SCORES_DIR
    saved = scrape_game_day_boxscores(
        fresh_monthly_file,
        SCORES_DIR,
        target_games_date
    )
    logging.info(f"Saved {saved} new box score file(s) for {target_games_date}")

    # parse those box scores -> dataframe
    games_df = process_saved_boxscores(
        SCORES_DIR,
        existing_statistics,
        target_games_date
    )

    if games_df is None or games_df.empty:
        logging.warning(f"No games parsed for {target_games_date}. Nothing to append.")
        _pause_and_exit_ok()
        return

    # if we have an existing layout, make sure order is identical again
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    # build combined dataset for saving
    out_daily = os.path.join(STAT_DIR, f"nba_games_{save_as_date}.csv")

    if os.path.exists(out_daily):
        # we already have today's snapshot → append new rows from yesterday's games
        prev = pd.read_csv(out_daily)
        combined = pd.concat([prev, games_df], ignore_index=True)
    elif existing_statistics is not None:
        # first run of the day: seed with most recent historical data + new games
        combined = pd.concat([existing_statistics, games_df], ignore_index=True).drop_duplicates()
    else:
        # very first day ever
        combined = games_df

    combined.to_csv(out_daily, index=False)
    logging.info(f"Combined statistics saved → {out_daily}")

    # mirror to DST_DIR (legacy behavior)
    copy_missing_files(STAT_DIR, DST_DIR)

    _pause_and_exit_ok()


def _pause_and_exit_ok():
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
