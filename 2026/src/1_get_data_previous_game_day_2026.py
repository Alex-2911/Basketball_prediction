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

In GitHub Actions:
  - Selenium is headless via nba_utils_2026.get_html()
  - We added safe fetching with retries + timeouts so the job won't hang
  - No interactive input() blocking at the end
"""

import os
import re
import sys
import time
import argparse
import logging
import calendar
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup

from selenium.common.exceptions import TimeoutException, WebDriverException

from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    get_html,
    parse_html,
    rename_duplicated_columns,
    copy_missing_files,
)

# -----------------------------------------------------------------------------
# logging config
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------------------------------
# local helpers
# -----------------------------------------------------------------------------

def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def month_name_lower(d: date) -> str:
    return calendar.month_name[d.month].lower()

def read_line_score(soup: BeautifulSoup) -> pd.DataFrame:
    line_score = pd.read_html(
        StringIO(str(soup)),
        attrs={'id': 'line_score'}
    )[0]

    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    line_score = line_score[["team", "total"]]
    return line_score

def read_stats(soup: BeautifulSoup, team: str, stat: str) -> pd.DataFrame:
    df = pd.read_html(
        StringIO(str(soup)),
        attrs={'id': f'box-{team}-game-{stat}'},
        index_col=0
    )[0]
    return df.apply(pd.to_numeric, errors="coerce")

def read_season_info(soup: BeautifulSoup) -> str:
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all('a')]
    # Example href like '/leagues/NBA_2026_games.html'
    return os.path.basename(hrefs[1]).split("_")[0]

def fetch_boxscore_html_safe(
    url: str,
    css_selector: str = "#content",
    timeout_seconds: int = 40,
    retry: int = 3
) -> str | None:
    """
    Robust fetch of a single box score page.

    - Tries up to `retry` times
    - Gives each attempt a hard wall-clock budget
    - Catches Selenium timeouts / crashes instead of hanging the whole pipeline

    Returns:
        HTML string on success
        None on total failure
    """
    for attempt in range(1, retry + 1):
        start = time.time()
        try:
            # NOTE:
            # If your get_html() does NOT accept timeout_seconds yet,
            # change the next line to: html = get_html(url, css_selector)
            html = get_html(url, css_selector)

            if html:
                return html
            else:
                logging.warning(
                    f"[WARN] Empty HTML from {url} (attempt {attempt}/{retry})"
                )

        except TimeoutException:
            logging.warning(
                f"[WARN] Timeout loading {url} (attempt {attempt}/{retry})"
            )
        except WebDriverException as e:
            logging.warning(
                f"[WARN] WebDriverException on {url}: {e} (attempt {attempt}/{retry})"
            )
        except Exception as e:
            logging.warning(
                f"[WARN] Unexpected error on {url}: {e} (attempt {attempt}/{retry})"
            )
        finally:
            elapsed = time.time() - start
            if elapsed > timeout_seconds + 5:
                logging.warning(
                    f"[WARN] Hard-stop: {url} exceeded ~{timeout_seconds}s wall clock"
                )

        # short pause so we don't hammer BRef + let Chrome reset
        time.sleep(2)

    logging.error(f"[ERROR] Failed to fetch {url} after {retry} tries. Skipping.")
    return None

def scrape_season_for_month(season: str, month_name: str, standings_dir: str) -> str | None:
    """
    Always download a *fresh* monthly schedule HTML for the given month.
    We delete any stale local file first so we always pull new data.

    Returns:
        path to the saved monthly file on success
        None on total failure
    """
    os.makedirs(standings_dir, exist_ok=True)
    monthly_filename = f"NBA_{season}_games-{month_name}.html"
    monthly_path = os.path.join(standings_dir, monthly_filename)

    # remove old copy so we don't reuse stale HTML
    if os.path.exists(monthly_path):
        try:
            os.remove(monthly_path)
            logging.info(f"Deleted outdated monthly file: {monthly_path}")
        except Exception as e:
            logging.error(f"Could not delete {monthly_path}: {e}")

    # hit the main season page to discover month-specific URLs
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    selector = "#content .filter"
    html_content = get_html(url, selector)
    if not html_content:
        logging.error(f"Failed to retrieve {url}")
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all(
        "a",
        href=re.compile(r"/leagues/NBA_[0-9]{4}_games-[a-z]+\.html")
    )

    wanted_url = None
    for l in links:
        href = l.get("href", "")
        if f"NBA_{season}_games-{month_name}" in href:
            wanted_url = "https://www.basketball-reference.com" + href
            break

    if not wanted_url:
        logging.warning(
            f"No monthly url found for month '{month_name}' in season {season}"
        )
        return None

    logging.info(f"Fetching fresh month page: {wanted_url}")
    # Pull the month page itself
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

def scrape_game_day_boxscores(
    standings_file: str,
    scores_dir: str,
    target_games_date: date
) -> int:
    """
    From the month schedule page:
    - find boxscore links for target_games_date
    - download any missing boxscore htmls into scores_dir (robustly)

    Returns:
        number of new box score files saved
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
        if h
        and "boxscores" in h
        and h.endswith(".html")
        and wanted_tag in h
    ]

    saved = 0
    for url in box_scores:
        save_path = os.path.join(scores_dir, os.path.basename(url))

        # skip if we already have this game locally
        if os.path.exists(save_path):
            continue

        # robust fetch with retries / timeout
        page_html = fetch_boxscore_html_safe(
            url,
            css_selector="#content",
            timeout_seconds=40,
            retry=3
        )

        if page_html is None:
            logging.warning(f"[SKIP] Could not fetch {url} after retries.")
            continue

        try:
            with open(save_path, "wb") as f:
                f.write(page_html.encode("utf-8"))
            saved += 1
            logging.info(f"Saved box score → {save_path}")
        except Exception as e:
            logging.error(f"Error saving {save_path}: {e}")

    return saved

def process_saved_boxscores(
    scores_dir: str,
    existing_statistics: pd.DataFrame | None,
    target_games_date: date
) -> pd.DataFrame:
    """
    Read all downloaded boxscore htmls for *exactly* target_games_date
    and build per-game rows aligned with existing_statistics columns.
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
                        if "bpm" not in b  # drop BPM columns
                    ]
                summary = summary[base_cols]
                summaries.append(summary)

            summary = pd.concat(summaries, axis=1).T

            game = pd.concat([summary, line_score], axis=1)
            game["home"] = [0, 1]  # first row = away, second row = home

            # build opponent columns by reversing
            game_opp = game.iloc[::-1].reset_index()
            game_opp.columns += "_opp"

            full_game = pd.concat([game, game_opp], axis=1)

            # season/year + metadata
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

    # align to existing_statistics columns if we have them
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    return games_df

def _pause_and_exit_ok():
    """
    Local run: keep console window open.
    GitHub Actions: exit immediately (GITHUB_ACTIONS is set to "true").
    """
    in_ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    if in_ci:
        return
    try:
        input("Done. Press Enter to close this window...")
    except EOFError:
        pass

# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------

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

    # figure out which day we're collecting
    now_dt, _, today_str_ymd = get_current_date(days_offset=0)
    today_date = now_dt.date()

    if args.collect_date:
        target_games_date = parse_ymd(args.collect_date)
        save_as_date = today_date
        logging.info(
            f"Collecting games for exact date: {target_games_date}"
        )
    elif args.date:
        anchor = parse_ymd(args.date)
        target_games_date = anchor - timedelta(days=1)
        save_as_date = anchor
        logging.info(
            f"Anchor date {anchor} → collecting prior day {target_games_date}"
        )
    else:
        target_games_date = today_date - timedelta(days=1)
        save_as_date = today_date
        logging.info(
            f"No args → scraping yesterday {target_games_date} "
            f"and saving snapshot as {save_as_date}"
        )

    # get the directory layout (STAT_DIR, etc.)
    paths = get_directory_paths()
    STAT_DIR = paths["STAT_DIR"]
    STANDINGS_DIR = paths["STANDINGS_DIR"]
    SCORES_DIR = paths["SCORES_DIR"]
    DST_DIR = STAT_DIR  # you mirror to same place at the end

    # always refresh the monthly file for that game's month
    month_name = month_name_lower(target_games_date)
    fresh_monthly_file = scrape_season_for_month(
        CURRENT_SEASON,
        month_name,
        STANDINGS_DIR
    )

    if fresh_monthly_file is None:
        # fallback: maybe we already have a saved file from a previous run
        fresh_monthly_file = os.path.join(
            STANDINGS_DIR,
            f"NBA_{CURRENT_SEASON}_games-{month_name}.html"
        )
        if not os.path.exists(fresh_monthly_file):
            logging.warning(
                f"No monthly file available for {month_name}, cannot continue."
            )
            _pause_and_exit_ok()
            return

    # try to load an existing statistics CSV to match the schema
    existing_statistics = None
    try:
        for back in range(0, 151):
            cand = (target_games_date - timedelta(days=back)).strftime("%Y-%m-%d")
            f = os.path.join(STAT_DIR, f"nba_games_{cand}.csv")
            if os.path.exists(f):
                existing_statistics = pd.read_csv(f)
                logging.info(
                    f"Using existing statistics layout from: {f}"
                )
                break
    except Exception as e:
        logging.warning(
            f"Could not load existing stats layout: {e}"
        )

    # download box scores for the target date (robust)
    saved = scrape_game_day_boxscores(
        fresh_monthly_file,
        SCORES_DIR,
        target_games_date
    )
    logging.info(
        f"Saved {saved} new box score file(s) for {target_games_date}"
    )

    # parse box scores into a tidy game-level DataFrame
    games_df = process_saved_boxscores(
        SCORES_DIR,
        existing_statistics,
        target_games_date
    )

    if games_df is None or games_df.empty:
        logging.warning(
            f"No games parsed for {target_games_date}. Nothing to append."
        )
        _pause_and_exit_ok()
        return

    # align columns once more (safety)
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    out_daily = os.path.join(
        STAT_DIR,
        f"nba_games_{save_as_date}.csv"
    )

    # combine:
    # - if we already have today's snapshot, append fresh rows
    # - else, glue new games_df onto the historical layout
    if os.path.exists(out_daily):
        prev = pd.read_csv(out_daily)
        combined = pd.concat([prev, games_df], ignore_index=True)
    elif existing_statistics is not None:
        combined = pd.concat(
            [existing_statistics, games_df],
            ignore_index=True
        ).drop_duplicates()
    else:
        combined = games_df

    combined.to_csv(out_daily, index=False)
    logging.info(
        f"Combined statistics saved → {out_daily}"
    )

    # optional sync / copy to DST_DIR
    copy_missing_files(STAT_DIR, DST_DIR)

    _pause_and_exit_ok()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error in script 1.")
        _pause_and_exit_ok()
