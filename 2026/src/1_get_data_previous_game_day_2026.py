#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 1 of 5 (2026): Get Data for Previous Game Day

Pipeline:
1. Decide target_games_date (usually yesterday) and save_as_date (usually today).
2. Download fresh monthly schedule page from Basketball Reference for that date.
3. From that schedule HTML, collect all box score URLs for that day.
4. For each box score URL:
   - Try to fetch HTML via requests (fast, cheap).
   - If it's missing proper tables / 'line_score', try Selenium once.
   - Save only if "clean".
5. Parse all saved box scores for that day into per-team rows.
6. Merge into rolling snapshot CSV nba_games_<save_as_date>.csv.

This feeds the feature set for predicting tonight's games.

Key ideas:
- We reuse ONE Selenium driver across all fallback fetches (lazy init).
- We bail early in CI if we can't parse any games (still considered "success",
  because maybe BRef is not fully updated yet).

Usage:
python 1_get_data_previous_game_day_2026.py
python 1_get_data_previous_game_day_2026.py --date 2025-11-01
python 1_get_data_previous_game_day_2026.py --collect-date 2025-10-31
"""

import os
import re
import time
import argparse
import logging
import calendar
from typing import Optional, Tuple, List

import pandas as pd
from io import StringIO
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup

from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    fetch_html_requests,
    build_driver,
    fetch_boxscore_via_driver,
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
# small local helpers
# -----------------------------------------------------------------------------

def parse_ymd(s: str) -> date:
    """'2025-10-31' -> datetime.date(2025,10,31)"""
    return datetime.strptime(s, "%Y-%m-%d").date()


def month_name_lower(d: date) -> str:
    """datetime.date -> 'october', 'november', ..."""
    return calendar.month_name[d.month].lower()


def _has_real_line_score(html_text: str) -> bool:
    """
    Heuristic to test if HTML looks like a real BRef box score.
    We demand that there's a <table id="line_score"> or at least that substring.
    """
    if not isinstance(html_text, str):
        return False
    return 'id="line_score"' in html_text


def read_line_score_df(raw_html: str) -> pd.DataFrame:
    """
    Parse the line_score table out of a box score page into:
    columns ['team','total'].
    Raises if table is not found / can't be parsed.
    """
    line_score = pd.read_html(
        StringIO(raw_html),
        attrs={'id': 'line_score'}
    )[0]

    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    line_score = line_score[["team", "total"]]
    return line_score


def read_team_tables_from_html(raw_html: str, team_abbr: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract basic + advanced tables for a given team (e.g. 'CHI', 'PHX', ...).
    Returns (basic_df, advanced_df), both numeric.
    Raises if tables not found.
    """
    basic = pd.read_html(
        StringIO(raw_html),
        attrs={'id': f'box-{team_abbr}-game-basic'},
        index_col=0
    )[0]
    basic = basic.apply(pd.to_numeric, errors="coerce")

    adv = pd.read_html(
        StringIO(raw_html),
        attrs={'id': f'box-{team_abbr}-game-advanced'},
        index_col=0
    )[0]
    adv = adv.apply(pd.to_numeric, errors="coerce")

    return basic, adv


def summarize_team_stats(basic_df: pd.DataFrame, adv_df: pd.DataFrame) -> pd.Series:
    """
    Build the per-team stat summary row:
      - totals row (last row of each table)
      - max per-column (best single-player values across the lineup)
    We also drop BPM-ish stuff by filtering later.
    """
    totals = pd.concat([basic_df.iloc[-1], adv_df.iloc[-1]])
    totals.index = totals.index.str.lower()

    maxes = pd.concat([
        basic_df.iloc[:-1].max(),
        adv_df.iloc[:-1].max()
    ])
    maxes.index = maxes.index.str.lower() + "_max"

    out = pd.concat([totals, maxes])
    return out


def scrape_season_for_month(
    season: int,
    month_name: str,
    standings_dir: str
) -> Optional[str]:
    """
    Download a fresh monthly schedule HTML for `month_name`
    (e.g. 'october') for the given season.
    Saves as .../NBA_2026_games-october.html

    Returns path to saved file on success, else None.
    """
    os.makedirs(standings_dir, exist_ok=True)
    monthly_filename = f"NBA_{season}_games-{month_name}.html"
    monthly_path = os.path.join(standings_dir, monthly_filename)

    # delete any stale copy first so we are guaranteed a fresh pull
    if os.path.exists(monthly_path):
        try:
            os.remove(monthly_path)
            logging.info(f"Deleted outdated monthly file: {monthly_path}")
        except Exception as e:
            logging.error(f"Could not delete {monthly_path}: {e}")

    # 1. fetch season overview page (has the monthly links)
    season_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html_season = fetch_html_requests(season_url)
    if not html_season:
        logging.error(f"Failed to retrieve {season_url}")
        return None

    soup_season = BeautifulSoup(html_season, "html.parser")

    # find the monthly link that matches this month
    links = soup_season.find_all(
        "a",
        href=re.compile(r"/leagues/NBA_[0-9]{4}_games-[a-z]+\.html")
    )

    wanted_url = None
    for a in links:
        href = a.get("href", "")
        if f"NBA_{season}_games-{month_name}" in href:
            wanted_url = "https://www.basketball-reference.com" + href
            break

    if not wanted_url:
        logging.warning(
            f"No monthly url found for month '{month_name}' in season {season}"
        )
        return None

    # 2. fetch the month page
    logging.info(f"Fetching fresh month page: {wanted_url}")
    month_html = fetch_html_requests(wanted_url)
    if not month_html:
        logging.warning(f"Could not fetch monthly page: {wanted_url}")
        return None

    # 3. save locally
    try:
        with open(monthly_path, "w", encoding="utf-8") as f:
            f.write(month_html)
        logging.info(f"Saved fresh monthly file → {monthly_path}")
    except Exception as e:
        logging.error(f"Error saving {monthly_path}: {e}")
        return None

    return monthly_path


def extract_boxscore_urls_for_date(
    standings_file: str,
    target_games_date: date
) -> List[str]:
    """
    From a saved monthly schedule HTML file, extract all box score links
    that match target_games_date (YYYYMMDD).
    Returns absolute URLs.
    """
    with open(standings_file, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    hrefs = [a.get("href") for a in soup.find_all("a")]

    tag = target_games_date.strftime("%Y%m%d")
    out_urls = [
        "https://www.basketball-reference.com" + h
        for h in hrefs
        if h
        and "boxscores" in h
        and h.endswith(".html")
        and tag in h
    ]
    return sorted(set(out_urls))


# -----------------------------------------------------------------------------
# box score download with fallback to Selenium
# -----------------------------------------------------------------------------

def download_and_cache_boxscores(
    urls: List[str],
    scores_dir: str
) -> Tuple[int, dict]:
    """
    For each box score URL:
    - Check if we already have a clean file on disk:
        - file exists
        - file contains a 'line_score' table
    - Else:
        - Try requests
        - If still not clean, try Selenium (build driver lazily)

    We save only good HTML.
    We reuse ONE Selenium driver for all URLs that need fallback.

    Returns:
        (saved_new_count, local_html_by_path)

        local_html_by_path is a dict:
           { save_path : raw_html_string }
        for *all* box scores we ended up with (either existing or newly fetched).
        We'll parse from memory later to avoid re-reading from disk again.
    """
    os.makedirs(scores_dir, exist_ok=True)

    saved_count = 0
    local_html_map: dict[str, str] = {}

    driver = None  # lazy-init only if we actually need Selenium

    def ensure_driver():
        nonlocal driver
        if driver is None:
            logging.info("Initializing Selenium fallback driver (first use)...")
            driver = build_driver()

    for url in urls:
        filename = os.path.basename(url)
        save_path = os.path.join(scores_dir, filename)

        def path_is_clean(p: str) -> bool:
            if not os.path.exists(p):
                return False
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    txt = fh.read()
                return _has_real_line_score(txt)
            except Exception:
                return False

        # 1. maybe we already have a valid file
        if path_is_clean(save_path):
            with open(save_path, "r", encoding="utf-8") as fh:
                local_html_map[save_path] = fh.read()
            logging.info(f"Local box score OK, reusing → {save_path}")
            continue

        # 2. try requests
        html_txt = fetch_html_requests(url)

        if not _has_real_line_score(html_txt or ""):
            # 3. slow fallback via Selenium
            logging.info(
                f"Local box score STALE/bad → {save_path} "
                f"(no real line_score table)"
            )
            ensure_driver()
            html_txt = fetch_boxscore_via_driver(driver, url)

        # after both attempts, if still no data => skip
        if not _has_real_line_score(html_txt or ""):
            logging.warning(
                f"[SKIP] Could not obtain a clean box score for {url}"
            )
            continue

        # looks good → save/update file on disk
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_txt)
            saved_count += 1
            local_html_map[save_path] = html_txt
            logging.info(f"Saved/updated valid box score → {save_path}")
        except Exception as e:
            logging.error(f"Error saving {save_path}: {e}")

    # cleanup driver
    if driver is not None:
        try:
            driver.quit()
        except Exception:
            pass

    return saved_count, local_html_map


# -----------------------------------------------------------------------------
# parse box scores into tidy rows
# -----------------------------------------------------------------------------

def build_game_frame_from_boxscore_html(
    raw_html: str,
    game_date: date
) -> pd.DataFrame:
    """
    Given a *single* box score HTML:
    - get teams + scores from line_score
    - build per-team stat rows (totals + maxes)
    - attach home flag, opponent mirror, metadata (date, season, won)

    Returns 2-row DataFrame [away_row, home_row] for that single game.
    """
    # line score
    line_score = read_line_score_df(raw_html)
    teams = list(line_score["team"])  # [away, home]

    # collect per-team summaries in the same order
    summaries = []
    base_cols = None

    for t_abbr in teams:
        basic_df, adv_df = read_team_tables_from_html(raw_html, t_abbr)
        row_summary = summarize_team_stats(basic_df, adv_df)

        if base_cols is None:
            # lock column order on first team and drop BPM-ish noise
            base_cols = [
                c for c in row_summary.index.drop_duplicates(keep="first")
                if "bpm" not in c
            ]

        row_summary = row_summary[base_cols]
        summaries.append(row_summary)

    # shape: 2 rows, stats columns
    summary_df = pd.concat(summaries, axis=1).T  # away row 0, home row 1

    # attach score info
    game_df = pd.concat([summary_df, line_score.reset_index(drop=True)], axis=1)

    # first row = away, second row = home
    game_df["home"] = [0, 1]

    # create opponent columns by reversing
    opp_df = game_df.iloc[::-1].reset_index(drop=True)
    opp_df.columns = [col + "_opp" for col in opp_df.columns]

    merged = pd.concat([game_df.reset_index(drop=True), opp_df], axis=1)

    # metadata
    merged["season"] = CURRENT_SEASON
    merged["date"] = pd.Timestamp(game_date)
    merged["won"] = merged["total"] > merged["total_opp"]

    return merged


def process_all_boxscores_for_day(
    local_html_map: dict,
    target_games_date: date,
    existing_statistics: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Turn every valid cached box score html from that calendar day
    into tidy per-team rows and union everything.
    Align columns to match existing_statistics if provided.

    Returns DataFrame, can be empty.
    """
    frames = []

    for save_path, raw_html in local_html_map.items():
        try:
            # filename starts with YYYYMMDD... we use that to confirm match
            basename = os.path.basename(save_path)
            fdate = pd.Timestamp(basename[:8]).date()
            if fdate != target_games_date:
                continue

            if not _has_real_line_score(raw_html):
                # should not happen but let's be safe
                continue

            game_frame = build_game_frame_from_boxscore_html(
                raw_html,
                target_games_date
            )
            frames.append(game_frame)

        except Exception as e:
            logging.error(f"Error parsing/processing {save_path}: {e}")

    if not frames:
        return pd.DataFrame()

    games_df = pd.concat(frames, ignore_index=True)
    games_df = rename_duplicated_columns(games_df)

    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    return games_df


# -----------------------------------------------------------------------------
# interactive-pause helper for local runs
# -----------------------------------------------------------------------------

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

    # figure out which calendar day we're actually scraping
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

    # directory layout
    paths = get_directory_paths()
    STAT_DIR = paths["STAT_DIR"]
    STANDINGS_DIR = paths["STANDINGS_DIR"]
    SCORES_DIR = paths["SCORES_DIR"]
    DST_DIR = STAT_DIR  # mirror at end (currently same)

    # 1. refresh the monthly schedule HTML for the target_games_date month
    month_name = month_name_lower(target_games_date)
    fresh_monthly_file = scrape_season_for_month(
        CURRENT_SEASON,
        month_name,
        STANDINGS_DIR
    )

    if fresh_monthly_file is None:
        # fallback: try local cached file
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

    # 2. build "reference schema" from most recent historical CSV
    existing_statistics = None
    try:
        # look back up to ~5 months just to be safe
        for back in range(0, 151):
            cand_date = (target_games_date - timedelta(days=back)).strftime("%Y-%m-%d")
            fpath = os.path.join(STAT_DIR, f"nba_games_{cand_date}.csv")
            if os.path.exists(fpath):
                existing_statistics = pd.read_csv(fpath)
                logging.info(f"Using existing statistics layout from: {fpath}")
                break
    except Exception as e:
        logging.warning(
            f"Could not load existing stats layout: {e}"
        )

    # 3. get all box score URLs for that target day
    urls = extract_boxscore_urls_for_date(
        fresh_monthly_file,
        target_games_date
    )

    # 4. download box scores (requests first, then Selenium fallback),
    #    cache locally, return all usable HTML in memory
    saved_count, local_html_map = download_and_cache_boxscores(
        urls,
        SCORES_DIR
    )
    logging.info(
        f"Saved {saved_count} new/updated box score file(s) for {target_games_date}"
    )

    # 5. turn all these box scores into final per-team stats rows
    games_df = process_all_boxscores_for_day(
        local_html_map,
        target_games_date,
        existing_statistics
    )

    if games_df is None or games_df.empty:
        logging.warning(
            f"No games parsed for {target_games_date}. Nothing to append."
        )
        _pause_and_exit_ok()
        return

    # just to be safe, align again with previous schema if present
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    # 6. write/update snapshot for save_as_date
    out_daily = os.path.join(
        STAT_DIR,
        f"nba_games_{save_as_date}.csv"
    )

    if os.path.exists(out_daily):
        prev = pd.read_csv(out_daily)
        combined = pd.concat([prev, games_df], ignore_index=True)
    elif existing_statistics is not None and not existing_statistics.empty:
        combined = pd.concat(
            [existing_statistics, games_df],
            ignore_index=True
        ).drop_duplicates()
    else:
        combined = games_df

    combined.to_csv(out_daily, index=False)
    logging.info(f"Combined statistics saved → {out_daily}")

    # 7. optional mirror hook
    copy_missing_files(STAT_DIR, DST_DIR)

    _pause_and_exit_ok()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error in script 1.")
        _pause_and_exit_ok()
