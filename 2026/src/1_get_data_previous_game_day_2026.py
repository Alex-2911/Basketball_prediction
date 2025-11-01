#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 1 of 5 (2026): Get Data for Previous Game Day

What it does:
- downloads the current month's NBA schedule HTML from basketball-reference.com
- finds yesterday's (or specified) games
- for each game:
    - makes sure we have a CLEAN box score HTML (not anti-bot junk)
    - if local file is stale/garbage, re-fetches it (first via requests,
      then with a headless Chrome fallback)
- parses each valid box score into team-level stats (home/away rows)
- appends those stats to our rolling dataset
- writes/updates a daily snapshot CSV like nba_games_2025-11-01.csv

Usage:
- No args  -> scrape yesterday's games, save under today's date.
- --date X -> treat X as "today", scrape X-1.
- --collect-date Y -> scrape games from Y exactly, save under real today.
"""

import os
import re
import time
import argparse
import logging
import calendar
from typing import Tuple, Optional
import pandas as pd

from io import StringIO
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup

from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    fetch_html_requests,
    fetch_boxscore_via_selenium,
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
# helper fns (date utils)
# -----------------------------------------------------------------------------

def parse_ymd(s: str) -> date:
    """'2025-10-31' -> datetime.date(2025,10,31)"""
    return datetime.strptime(s, "%Y-%m-%d").date()


def month_name_lower(d: date) -> str:
    """datetime.date -> 'october', 'november', ..."""
    return calendar.month_name[d.month].lower()


# -----------------------------------------------------------------------------
# HTML validation + table readers
# -----------------------------------------------------------------------------

def file_is_valid_html_boxscore(txt: Optional[str]) -> bool:
    """
    Heuristics to decide if this HTML is a REAL Basketball Reference box score
    and not an anti-bot / "enable JavaScript" placeholder.

    We want to be strict, because saving garbage breaks parsing later.
    """
    if txt is None:
        return False
    txt_strip = txt.strip()
    if txt_strip == "":
        return False

    # obvious bot-wall / block content
    bad_signals = [
        "enable JavaScript",
        "unusual activity",
        "are you a robot",
        "temporarily blocked",
        "please verify you are a human",
        "captcha",
    ]
    low = txt_strip.lower()
    for bad in bad_signals:
        if bad.lower() in low:
            return False

    # must have at least one of the structures we rely on every single time
    must_have = [
        'table id="line_score"',
        "boxscore",  # appears in real box score markup
    ]
    for sig in must_have:
        if sig in txt_strip:
            return True

    return False


def read_team_tables_from_html(raw_html: str, team_abbr: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (basic_df, advanced_df) for the given team from the box score HTML.
    Both are numeric DataFrames with player rows + 'Team Totals' last row.

    Example table IDs on Basketball Reference:
    - id='box-CHI-game-basic'
    - id='box-CHI-game-advanced'
    """
    basic = pd.read_html(
        StringIO(raw_html),
        attrs={'id': f'box-{team_abbr}-game-basic'},
        index_col=0
    )[0]
    basic = basic.apply(pd.to_numeric, errors="coerce")

    advanced = pd.read_html(
        StringIO(raw_html),
        attrs={'id': f'box-{team_abbr}-game-advanced'},
        index_col=0
    )[0]
    advanced = advanced.apply(pd.to_numeric, errors="coerce")

    return basic, advanced


# -----------------------------------------------------------------------------
# scraping helpers
# -----------------------------------------------------------------------------

def scrape_season_for_month(
    season: str,
    month_name: str,
    standings_dir: str
) -> str | None:
    """
    Download the fresh monthly schedule HTML for `month_name`
    (e.g. 'october') for the given season.
    Saves as .../NBA_2026_games-october.html

    Returns:
        path to saved file on success
        None on total failure
    """
    os.makedirs(standings_dir, exist_ok=True)
    monthly_filename = f"NBA_{season}_games-{month_name}.html"
    monthly_path = os.path.join(standings_dir, monthly_filename)

    # delete stale copy so we always refetch and don't trust old HTML
    if os.path.exists(monthly_path):
        try:
            os.remove(monthly_path)
            logging.info(f"Deleted outdated monthly file: {monthly_path}")
        except Exception as e:
            logging.error(f"Could not delete {monthly_path}: {e}")

    # 1. fetch season overview page: /leagues/NBA_2026_games.html
    season_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html_season = fetch_html_requests(season_url)
    if not html_season:
        logging.error(f"Failed to retrieve {season_url}")
        return None

    soup_season = BeautifulSoup(html_season, 'html.parser')

    # find monthly page link that matches this month, eg /leagues/NBA_2026_games-october.html
    links = soup_season.find_all(
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

    # 2. fetch that month's page (schedule table for that month)
    logging.info(f"Fetching fresh month page: {wanted_url}")
    month_html = fetch_html_requests(wanted_url)
    if not month_html:
        logging.warning(f"Could not fetch monthly page: {wanted_url}")
        return None

    # 3. save it
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
    For each game on target_games_date:
    - make sure we end up with a *valid* local HTML that actually has box score tables
    - if local file is stale/anti-bot, try to refresh:
        1) fetch_html_requests()
        2) fallback fetch_boxscore_via_selenium()
    Returns number of new/updated box score files successfully saved.
    """
    os.makedirs(scores_dir, exist_ok=True)

    # Parse the monthly standings page to get all boxscore links for that day
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
        filename = os.path.basename(url)
        save_path = os.path.join(scores_dir, filename)

        def local_copy_is_valid(path: str) -> bool:
            """Check if an existing local file is good enough to parse."""
            if not os.path.exists(path):
                return False
            try:
                with open(path, "r", encoding="utf-8") as f2:
                    txt = f2.read()
                return file_is_valid_html_boxscore(txt)
            except Exception:
                return False

        # 1) If local is already valid, keep it
        if local_copy_is_valid(save_path):
            logging.info(f"Local box score OK, reusing → {save_path}")
            continue

        logging.info(
            f"Local box score STALE/bad → {save_path} (needs refetch)"
        )

        # 2) Try direct requests first
        page_html = fetch_html_requests(url)
        if not file_is_valid_html_boxscore(page_html):
            # 3) requests wasn't good, try headless browser fallback
            logging.info("requests HTML not valid, trying Selenium fallback...")
            page_html = fetch_boxscore_via_selenium(url)

        # After both attempts, if still invalid, skip this game
        if not file_is_valid_html_boxscore(page_html):
            logging.warning(
                f"[SKIP] Could not obtain a clean box score for {url}"
            )
            continue

        # 4) Valid → write/overwrite our local copy
        try:
            with open(save_path, "w", encoding="utf-8") as f3:
                f3.write(page_html)
            saved += 1
            logging.info(f"Saved/updated valid box score → {save_path}")
        except Exception as e:
            logging.error(f"Error saving {save_path}: {e}")

    return saved


def process_saved_boxscores(
    scores_dir: str,
    existing_statistics: pd.DataFrame | None,
    target_games_date: date
) -> pd.DataFrame:
    """
    Load each downloaded boxscore .html from scores_dir for exactly target_games_date,
    build per-team rows, align with existing_statistics columns.

    Output rows (2 per game: away row then home row) include:
      - numeric stats from 'basic' + 'advanced' per team
      - *_max columns (max single-player line for each stat)
      - 'total' points, 'home' flag
      - mirrored opponent columns with suffix _opp
      - won (True/False)
      - date, season
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
            # filenames start with YYYYMMDD...
            fdate = pd.Timestamp(os.path.basename(p)[:8]).date()
            if fdate != target_games_date:
                continue

            with open(p, "r", encoding="utf-8") as fh:
                raw_txt = fh.read()

            # final validity check before parsing with pandas
            if not file_is_valid_html_boxscore(raw_txt):
                logging.warning(f"[SKIP PARSE] {p} failed final validity check.")
                continue

            # --- line score table (final scores, away/home) ---
            line_score = pd.read_html(
                StringIO(raw_txt),
                attrs={'id': 'line_score'}
            )[0]

            cols = list(line_score.columns)
            cols[0] = "team"
            cols[-1] = "total"
            line_score.columns = cols
            line_score = line_score[["team", "total"]]

            teams = list(line_score["team"])  # typically [AWAY_ABBR, HOME_ABBR]

            # --- per-team stats ---
            summaries = []

            for team_abbr in teams:
                basic, advanced = read_team_tables_from_html(raw_txt, team_abbr)

                # last row in each table is "Team Totals"
                totals = pd.concat([basic.iloc[-1], advanced.iloc[-1]])
                totals.index = totals.index.str.lower()

                # max across players (exclude last row "Team Totals")
                maxes = pd.concat([
                    basic.iloc[:-1].max(),
                    advanced.iloc[:-1].max()
                ])
                maxes.index = maxes.index.str.lower() + "_max"

                summary = pd.concat([totals, maxes])

                # lock a shared base_cols ordering for the whole run
                if base_cols is None:
                    base_cols = [
                        b for b in summary.index.drop_duplicates(keep="first")
                        if "bpm" not in b
                    ]

                summary = summary[base_cols]
                summaries.append(summary)

            # shape: 2 rows (away first, home second)
            summary = pd.concat(summaries, axis=1).T

            # attach final team points
            game = pd.concat([summary, line_score], axis=1)

            # mark home flag: first row away(0), second row home(1)
            game["home"] = [0, 1]

            # mirror opponent stats by flipping the two rows
            game_opp = game.iloc[::-1].reset_index()
            game_opp.columns += "_opp"

            full_game = pd.concat([game, game_opp], axis=1)

            # add extra metadata
            full_game["season"] = CURRENT_SEASON
            full_game["date"] = pd.Timestamp(os.path.basename(p)[:8])
            full_game["won"] = full_game["total"] > full_game["total_opp"]

            games.append(full_game)

        except Exception as e:
            logging.error(f"Error parsing/processing {p}: {e}")

    if not games:
        return pd.DataFrame()

    games_df = pd.concat(games, ignore_index=True)
    games_df = rename_duplicated_columns(games_df)

    # align final columns to previous snapshot schema if available
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    return games_df


def _pause_and_exit_ok():
    """
    Local run: keep console window open.
    In GitHub Actions: exit immediately (GITHUB_ACTIONS is 'true').
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

    # figure out which day to collect
    now_dt, _, _ = get_current_date(days_offset=0)
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

    # resolve directory layout
    paths = get_directory_paths()
    STAT_DIR = paths["STAT_DIR"]
    STANDINGS_DIR = paths["STANDINGS_DIR"]
    SCORES_DIR = paths["SCORES_DIR"]
    DST_DIR = STAT_DIR  # right now same, but keep hook

    # 1. refresh the monthly schedule HTML for the target_games_date month
    month_name = month_name_lower(target_games_date)
    fresh_monthly_file = scrape_season_for_month(
        CURRENT_SEASON,
        month_name,
        STANDINGS_DIR
    )

    if fresh_monthly_file is None:
        # fallback: use an existing file if any (older cached month file)
        fallback_path = os.path.join(
            STANDINGS_DIR,
            f"NBA_{CURRENT_SEASON}_games-{month_name}.html"
        )
        if not os.path.exists(fallback_path):
            logging.warning(
                f"No monthly file available for {month_name}, cannot continue."
            )
            _pause_and_exit_ok()
            return
        fresh_monthly_file = fallback_path

    # 2. pick an existing statistics layout so we preserve column order
    existing_statistics = None
    try:
        for back in range(0, 151):
            cand = (target_games_date - timedelta(days=back)).strftime("%Y-%m-%d")
            fpath = os.path.join(STAT_DIR, f"nba_games_{cand}.csv")
            if os.path.exists(fpath):
                existing_statistics = pd.read_csv(fpath)
                logging.info(
                    f"Using existing statistics layout from: {fpath}"
                )
                break
    except Exception as e:
        logging.warning(
            f"Could not load existing stats layout: {e}"
        )

    # 3. download/refresh box score HTMLs for the target day (with validation)
    saved = scrape_game_day_boxscores(
        fresh_monthly_file,
        SCORES_DIR,
        target_games_date
    )
    logging.info(
        f"Saved {saved} new/updated box score file(s) for {target_games_date}"
    )

    # 4. parse those box scores into a tidy per-team/per-game frame
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

    # safety align columns again
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    out_daily = os.path.join(
        STAT_DIR,
        f"nba_games_{save_as_date}.csv"
    )

    # 5. merge new data into the rolling snapshot for save_as_date
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

    # 6. mirror to DST_DIR (right now DST_DIR == STAT_DIR)
    copy_missing_files(STAT_DIR, DST_DIR)

    _pause_and_exit_ok()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error in script 1.")
        _pause_and_exit_ok()
