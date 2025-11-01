#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 1 of 5 (2026): Get Data for Previous Game Day

Pipeline:
1. Figure out "target_games_date":
   - default: yesterday
   - --date X: scrape X-1
   - --collect-date Y: scrape Y exactly
   Also decide which date we save the snapshot under ("save_as_date").

2. Download fresh monthly schedule HTML from basketball-reference.com
   for that date's month (NBA_2026_games-october.html etc.).

3. From that HTML, collect all box score links for target_games_date.

4. For each box score link:
   - Check if we already have a VALID local copy in scores_dir
     (valid = contains real <table id="line_score">).
   - If not, first try fetch_html_requests().
   - If that still isn't valid, try fetch_boxscore_via_selenium().
   - Only save if valid.

5. Parse all valid local box scores from that date into per-team rows
   (away+home), attach metadata, align columns with our historical CSV,
   and write/update nba_games_<save_as_date>.csv in Whole_Statistic.
"""

import os
import re
import time
import argparse
import logging
import calendar
import pandas as pd

from io import StringIO
from datetime import datetime, timedelta, date
from typing import Tuple, Optional
from bs4 import BeautifulSoup

from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    fetch_html_requests,
    fetch_boxscore_via_selenium,
    boxscore_html_is_valid,
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
# helpers: dates
# -----------------------------------------------------------------------------

def parse_ymd(s: str) -> date:
    """'2025-10-31' -> datetime.date(2025, 10, 31)"""
    return datetime.strptime(s, "%Y-%m-%d").date()


def month_name_lower(d: date) -> str:
    """datetime.date(2025,10,31) -> 'october'."""
    return calendar.month_name[d.month].lower()


# -----------------------------------------------------------------------------
# helpers: table readers
# -----------------------------------------------------------------------------

def read_line_score_from_html(raw_html: str) -> pd.DataFrame:
    """
    Pull the final score table (#line_score) from a box score HTML string.
    Return DataFrame with columns ['team','total'].
    Raise if the table isn't there → caller will handle.
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
    Get that team's 'basic' and 'advanced' tables from a box score HTML string.

    Returns (basic_df, advanced_df) where both are numeric frames.
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


# -----------------------------------------------------------------------------
# step 1: fetch monthly page and save locally
# -----------------------------------------------------------------------------

def scrape_season_for_month(
    season: int,
    month_name: str,
    standings_dir: str
) -> Optional[str]:
    """
    Download fresh monthly schedule HTML for `month_name`
    (e.g. 'october') for the given season (e.g. 2026),
    and save it to STANDINGS_DIR/NBA_2026_games-october.html.

    Return path to saved file OR None if we couldn't get anything.
    """
    os.makedirs(standings_dir, exist_ok=True)

    monthly_filename = f"NBA_{season}_games-{month_name}.html"
    monthly_path = os.path.join(standings_dir, monthly_filename)

    # delete stale copy first → forces fresh pull
    if os.path.exists(monthly_path):
        try:
            os.remove(monthly_path)
            logging.info(f"Deleted outdated monthly file: {monthly_path}")
        except Exception as e:
            logging.error(f"Could not delete {monthly_path}: {e}")

    # 1. fetch the season overview page: /leagues/NBA_2026_games.html
    season_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html_season = fetch_html_requests(season_url)
    if not html_season:
        logging.error(f"Failed to retrieve {season_url}")
        return None

    soup_season = BeautifulSoup(html_season, "html.parser")

    # 2. find link for this specific month
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

    # 3. get that month subpage
    logging.info(f"Fetching fresh month page: {wanted_url}")
    month_html = fetch_html_requests(wanted_url)
    if not month_html:
        logging.warning(f"Could not fetch monthly page: {wanted_url}")
        return None

    # 4. save
    try:
        with open(monthly_path, "w", encoding="utf-8") as f:
            f.write(month_html)
        logging.info(f"Saved fresh monthly file → {monthly_path}")
    except Exception as e:
        logging.error(f"Error saving {monthly_path}: {e}")
        return None

    return monthly_path


# -----------------------------------------------------------------------------
# step 2: download / refresh boxscore HTMLs
# -----------------------------------------------------------------------------

def scrape_game_day_boxscores(
    standings_file: str,
    scores_dir: str,
    target_games_date: date
) -> int:
    """
    For each game played on target_games_date:
    - Build the box score URL from schedule page.
    - Check if local cached file exists AND is valid (has real line_score).
      * If valid: keep.
      * If missing or invalid:
          1. Try fetch_html_requests(url)
          2. If still invalid, try fetch_boxscore_via_selenium(url)
      * Only save HTML if valid.

    Return the number of box score files we actually wrote/overwrote.
    """
    os.makedirs(scores_dir, exist_ok=True)

    # read the monthly standings HTML we just saved
    with open(standings_file, "r", encoding="utf-8") as f:
        monthly_html = f.read()

    soup = BeautifulSoup(monthly_html, "html.parser")
    hrefs = [a.get("href") for a in soup.find_all("a")]

    wanted_tag = target_games_date.strftime("%Y%m%d")
    box_urls = [
        "https://www.basketball-reference.com" + h
        for h in hrefs
        if h
        and "boxscores" in h
        and h.endswith(".html")
        and wanted_tag in h
    ]

    saved = 0

    for url in box_urls:
        filename = os.path.basename(url)
        save_path = os.path.join(scores_dir, filename)

        # helper: load local HTML if present
        def load_local_html_if_any(path: str) -> Optional[str]:
            if not os.path.exists(path):
                return None
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read()
            except Exception:
                return None

        local_html = load_local_html_if_any(save_path)
        local_valid = boxscore_html_is_valid(local_html) if local_html else False

        if local_valid:
            logging.info(f"Local box score OK, reusing → {save_path}")
            continue
        else:
            if local_html:
                logging.info(
                    f"Local box score STALE/bad → {save_path} "
                    "(no real line_score table)"
                )

        # Try requests first
        html_req = fetch_html_requests(url)
        if html_req and boxscore_html_is_valid(html_req):
            try:
                with open(save_path, "w", encoding="utf-8") as fh:
                    fh.write(html_req)
                saved += 1
                logging.info(f"Saved/updated valid box score → {save_path}")
            except Exception as e:
                logging.error(f"Error saving {save_path}: {e}")
            continue

        # Still not valid → try selenium fallback
        html_sel = fetch_boxscore_via_selenium(url)
        if html_sel and boxscore_html_is_valid(html_sel):
            try:
                with open(save_path, "w", encoding="utf-8") as fh:
                    fh.write(html_sel)
                saved += 1
                logging.info(
                    f"Saved/updated valid box score (selenium) → {save_path}"
                )
            except Exception as e:
                logging.error(f"Error saving {save_path}: {e}")
        else:
            logging.warning(
                f"[SKIP] Could not obtain a clean box score for {url}"
            )

    return saved


# -----------------------------------------------------------------------------
# step 3: parse (now-valid) local boxscores into rows
# -----------------------------------------------------------------------------

def process_saved_boxscores(
    scores_dir: str,
    existing_statistics: pd.DataFrame | None,
    target_games_date: date
) -> pd.DataFrame:
    """
    Look at every .html in scores_dir, keep only those:
    - with filename prefix == target_games_date (YYYYMMDD),
    - that actually contain the <table id="line_score">.

    For each valid game:
      - read final score (line_score)
      - read team stats (basic + advanced)
      - build one row per team (away/home)
      - attach opponent mirror columns
      - attach metadata: season, date, won

    Return concatenated DataFrame of all games that date,
    optionally reindexed to match existing_statistics columns.
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
            # 1. match file date to target_games_date
            fdate = pd.Timestamp(os.path.basename(p)[:8]).date()
            if fdate != target_games_date:
                continue

            with open(p, "r", encoding="utf-8") as fh:
                raw_html = fh.read()

            # 2. validate HTML is real
            if not boxscore_html_is_valid(raw_html):
                logging.warning(
                    f"[SKIP PARSE] {p} is missing a valid line_score table "
                    "(likely anti-bot HTML)"
                )
                continue

            # 3. line_score = final scores
            line_score = read_line_score_from_html(raw_html)
            teams = list(line_score["team"])  # [away_team, home_team]

            # 4. build per-team summary for both teams
            summaries = []
            for team_abbr in teams:
                basic_df, adv_df = read_team_tables_from_html(raw_html, team_abbr)

                # last row is "Team Totals"
                totals = pd.concat([basic_df.iloc[-1], adv_df.iloc[-1]])
                totals.index = totals.index.str.lower()

                # max per-player stats (exclude last row)
                maxes = pd.concat([basic_df.iloc[:-1].max(), adv_df.iloc[:-1].max()])
                maxes.index = maxes.index.str.lower() + "_max"

                summary = pd.concat([totals, maxes])

                # lock column ordering (base_cols) on the first team we parse
                if base_cols is None:
                    base_cols = [
                        c for c in summary.index.drop_duplicates(keep="first")
                        if "bpm" not in c.lower()
                    ]

                summary = summary[base_cols]
                summaries.append(summary)

            # shape now: 2 rows, same columns (away first, home second)
            summary_df = pd.concat(summaries, axis=1).T

            # add final team scores
            game_df = pd.concat([summary_df, line_score], axis=1)

            # mark home flag: first row = away(0), second row = home(1)
            game_df["home"] = [0, 1]

            # create mirrored opponent columns by reversing rows
            opp_df = game_df.iloc[::-1].reset_index(drop=True)
            opp_df.columns = [f"{c}_opp" for c in opp_df.columns]

            full_game = pd.concat([game_df.reset_index(drop=True), opp_df], axis=1)

            # attach metadata
            full_game["season"] = CURRENT_SEASON
            full_game["date"] = pd.Timestamp(os.path.basename(p)[:8])
            full_game["won"] = full_game["total"] > full_game["total_opp"]

            games.append(full_game)

        except Exception as e:
            logging.error(f"Error processing {p}: {e}")

    if not games:
        return pd.DataFrame()

    games_df = pd.concat(games, ignore_index=True)
    games_df = rename_duplicated_columns(games_df)

    # Align with existing stats columns from historical CSV
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    return games_df


# -----------------------------------------------------------------------------
# interactive pause helper (for local runs)
# -----------------------------------------------------------------------------

def _pause_and_exit_ok():
    """
    Local run: wait for Enter so the console window doesn't close.
    In GitHub Actions: just return.
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
    DST_DIR = STAT_DIR  # right now same, but keep hook

    # STEP 1: refresh monthly schedule HTML for that date's month
    month_name = month_name_lower(target_games_date)
    fresh_monthly_file = scrape_season_for_month(
        CURRENT_SEASON,
        month_name,
        STANDINGS_DIR
    )

    if fresh_monthly_file is None:
        # fallback: reuse previous monthly file if available
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

    # STEP 2: load existing snapshot schema so we can align columns
    existing_statistics = None
    try:
        for back in range(0, 151):
            cand_date = (target_games_date - timedelta(days=back)).strftime("%Y-%m-%d")
            candidate_csv = os.path.join(STAT_DIR, f"nba_games_{cand_date}.csv")
            if os.path.exists(candidate_csv):
                existing_statistics = pd.read_csv(candidate_csv)
                logging.info(
                    f"Using existing statistics layout from: {candidate_csv}"
                )
                break
    except Exception as e:
        logging.warning(
            f"Could not load existing stats layout: {e}"
        )

    # STEP 3: (re)download box scores for the target date (requests → selenium fallback)
    saved_ct = scrape_game_day_boxscores(
        fresh_monthly_file,
        SCORES_DIR,
        target_games_date
    )
    logging.info(
        f"Saved {saved_ct} new box score file(s) for {target_games_date}"
    )

    # STEP 4: parse all valid box scores for that date into rows
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

    # final column alignment safety
    if existing_statistics is not None and not existing_statistics.empty:
        games_df = games_df.reindex(columns=existing_statistics.columns)

    # STEP 5: write/update today's snapshot
    out_daily = os.path.join(
        STAT_DIR,
        f"nba_games_{save_as_date}.csv"
    )

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

    # STEP 6: mirror/sync (no-op for now but we leave the call so future dirs stay in sync)
    copy_missing_files(STAT_DIR, DST_DIR)

    _pause_and_exit_ok()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error in script 1.")
        _pause_and_exit_ok()
