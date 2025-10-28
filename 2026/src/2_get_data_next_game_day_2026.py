# -*- coding: utf-8 -*-
"""
Script 2 of 5 (2026): Get Data for NEXT Game Day

This script:
  - looks at the NBA schedule on Basketball-Reference
  - figures out the NEXT game day (all games on the next date with games)
  - writes those matchups as games_df_<today>.csv into NEXT_GAME_DIR

Inputs:
  - local monthly schedule HTML(s) in STANDINGS_DIR (downloaded if missing)
  - team name mapping from nba_utils_2026.get_team_codes()

Outputs:
  - 2026/data/next_game_day/games_df_<DD-MM-YYYY>.csv
    (path depends on get_directory_paths())

Assumptions:
  - Season naming matches Basketball-Reference (e.g. 2025-26 season → 2026)
  - Script 1 (previous_game_day) already created directory structure

CI behavior:
  - uses SEASON_YEAR env var if set
  - uses TARGET_DATE env var or --date override if provided
  - NO interactive input() blocking in GitHub Actions
"""

import os
import shutil
import argparse
import calendar
import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
    get_team_codes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────
# helpers
# ─────────────────────────────

def parse_target_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD -> datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def determine_season_year(target_date: datetime, fallback_season: int) -> int:
    """
    Basketball-Reference labels seasons by the year they END.
    2025-26 season is called 2026.

    We mostly just trust CURRENT_SEASON unless SEASON_YEAR env var is set.
    """
    env_season = os.getenv("SEASON_YEAR")
    if env_season is not None:
        try:
            return int(env_season)
        except ValueError:
            logging.warning(
                f"Invalid SEASON_YEAR '{env_season}', falling back to {fallback_season}."
            )
    return fallback_season


def scrape_season_for_month(season: int, month_num: int, month_name: str, standings_dir: str) -> Optional[str]:
    """
    Download the <month> schedule page for a given Basketball-Reference season,
    e.g. https://www.basketball-reference.com/leagues/NBA_2026_games-october.html

    Saves it as NBA_<season>_games-<month_name>.html in standings_dir.
    Returns the local path, or None on failure.
    """
    os.makedirs(standings_dir, exist_ok=True)

    # Step 1: get the main season page to discover monthly links
    season_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    try:
        resp = requests.get(season_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"[SEASON PAGE] Failed {season_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all("a", href=True)

    month_link = None
    for link in links:
        href = link["href"]
        if f"NBA_{season}_games-{month_name}" in href:
            month_link = "https://www.basketball-reference.com" + href
            break

    if not month_link:
        logging.warning(
            f"[SEASON PAGE] No monthly link for {month_name} {season}"
        )
        return None

    # Step 2: download the monthly schedule
    try:
        mresp = requests.get(month_link, timeout=15)
        mresp.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"[MONTH PAGE] Failed {month_link}: {e}")
        return None

    out_path = os.path.join(
        standings_dir, f"NBA_{season}_games-{month_name}.html"
    )

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(mresp.text)
        logging.info(
            f"[OK] Saved month schedule → {out_path}"
        )
    except Exception as e:
        logging.error(f"[WRITE FAIL] {out_path}: {e}")
        return None

    return out_path


def find_games_for_next_day(target_date: datetime, file_paths: List[str]) -> List[Dict[str, object]]:
    """
    Given a list of local monthly HTML schedule files, find:
      - the first game_date >= target_date
      - all games on that date

    Returns list of dict rows with:
      { "date": datetime, "home_team": <full name>, "visitor_team": <full name> }
    """
    next_game_date: Optional[datetime] = None
    collected: List[Dict[str, object]] = []

    for path in file_paths:
        if not os.path.exists(path):
            logging.warning(f"[MISS] schedule file not found: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
        except Exception as e:
            logging.error(f"[READ FAIL] {path}: {e}")
            continue

        table = soup.find("table", {"id": "schedule"})
        if not table:
            continue

        rows = table.find_all("tr")
        # we'll manually skip header rows (they have no <td> usually)
        for row in rows:
            # date is in the <th data-stat="date_game">
            th_date = row.find("th", {"data-stat": "date_game"})
            if not th_date:
                continue

            raw_date = th_date.get_text(strip=True)
            try:
                game_date = datetime.strptime(raw_date, "%a, %b %d, %Y")
            except ValueError:
                continue

            # first future date we encounter (>= target_date)
            if game_date >= target_date and next_game_date is None:
                next_game_date = game_date

            # collect only rows on that same next_game_date
            if next_game_date is not None and game_date == next_game_date:
                tds = row.find_all("td")
                # typical schedule table columns:
                #  - visitor_team_name at data-stat="visitor_team_name" (index 1 in raw table structure)
                #  - home_team_name    at data-stat="home_team_name"    (index 3)
                if len(tds) >= 4:
                    visitor = tds[1].get_text(strip=True)
                    home    = tds[3].get_text(strip=True)
                    collected.append({
                        "date": game_date,
                        "home_team": home,
                        "visitor_team": visitor,
                    })

            # once we're past that date, we can stop
            if next_game_date is not None and game_date > next_game_date:
                return collected

    return collected


def _pause_and_exit_ok():
    """
    Keep window open if running locally.
    In GitHub Actions, GITHUB_ACTIONS="true" so we just return.
    """
    in_ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    if in_ci:
        return
    try:
        input("Done. Press Enter to close this window...")
    except EOFError:
        pass


# ─────────────────────────────
# main
# ─────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the NEXT NBA game day and save matchups as CSV."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=os.getenv("TARGET_DATE"),
        help="Anchor date (YYYY-MM-DD). We'll search for the first game day on/after this date."
    )
    args = parser.parse_args()

    # figure out what 'today' is for scheduling
    if args.date:
        try:
            anchor_dt = parse_target_date(args.date)
        except ValueError:
            logging.error(f"Invalid --date '{args.date}'. Use YYYY-MM-DD.")
            return

        today_date = anchor_dt
        today_str_human = anchor_dt.strftime("%a, %b %d, %Y")
        today_str_for_filename = anchor_dt.strftime("%d-%m-%Y")
    else:
        # fallback to util
        now_dt, today_str_human, today_str_for_filename = get_current_date()
        # get_current_date returns now_dt as datetime already in script1 style
        # but today_str_human is "%a, %b %d, %Y"
        # so parse that back into datetime for consistent downstream logic
        today_date = datetime.strptime(today_str_human, "%a, %b %d, %Y")

    logging.info(f"Using anchor date: {today_str_human}")

    # resolve directories
    paths = get_directory_paths()
    standings_dir   = paths["STANDINGS_DIR"]
    next_game_dir   = paths["NEXT_GAME_DIR"]

    os.makedirs(next_game_dir, exist_ok=True)

    # pick the Basketball-Reference "season year"
    season_year = determine_season_year(today_date, CURRENT_SEASON)

    # figure out which month file(s) we need
    month_num   = today_date.month
    month_name  = calendar.month_name[month_num].lower()

    monthly_html = os.path.join(
        standings_dir,
        f"NBA_{season_year}_games-{month_name}.html"
    )

    # ensure we have this month's schedule locally (download if missing)
    if not os.path.exists(monthly_html):
        logging.info(f"Schedule file missing locally → {monthly_html}")
        pulled = scrape_season_for_month(
            season_year, month_num, month_name, standings_dir
        )
        if pulled is not None:
            monthly_html = pulled

    # gather from this month first
    games_info = find_games_for_next_day(today_date, [monthly_html])

    # fallback for known opening night if site access was blocked / partial
    # (You can edit these pairs if opening night changes for a new season)
    if not games_info and today_date.strftime("%Y-%m-%d") == "2025-10-21":
        games_info = [
            {
                "date": today_date,
                "home_team": "Oklahoma City Thunder",
                "visitor_team": "Houston Rockets",
            },
            {
                "date": today_date,
                "home_team": "Los Angeles Lakers",
                "visitor_team": "Golden State Warriors",
            },
        ]

    # try subsequent months if still empty
    if not games_info:
        next_month = (month_num % 12) + 1
        checked = 0
        while not games_info and checked < 12:
            nm_name = calendar.month_name[next_month].lower()
            nm_html = os.path.join(
                standings_dir,
                f"NBA_{season_year}_games-{nm_name}.html"
            )

            if not os.path.exists(nm_html):
                scrape_season_for_month(
                    season_year, next_month, nm_name, standings_dir
                )

            games_info = find_games_for_next_day(today_date, [nm_html])

            next_month = (next_month % 12) + 1
            checked += 1

    # map full team names -> 3-letter codes
    team_codes = get_team_codes()

    rows_for_csv: List[tuple] = []
    if games_info:
        for g in games_info:
            home_code = team_codes.get(g["home_team"], g["home_team"])
            away_code = team_codes.get(g["visitor_team"], g["visitor_team"])

            rows_for_csv.append(
                (
                    home_code,
                    away_code,
                    g["date"].strftime("%Y-%m-%d")
                )
            )

            logging.info(
                f"NEXT GAME: {g['visitor_team']} at {g['home_team']} "
                f"on {g['date'].strftime('%a, %b %d, %Y')}"
            )
    else:
        logging.warning("No upcoming games found.")
        # we'll still save an empty CSV for downstream scripts so they don't crash

    # build final DataFrame
    next_games_df = pd.DataFrame(
        rows_for_csv,
        columns=["home_team", "away_team", "game_date"]
    )

    csv_name   = f"games_df_{today_str_for_filename}.csv"
    output_csv = os.path.join(next_game_dir, csv_name)

    next_games_df.to_csv(output_csv, index=False)
    logging.info(f"Saved upcoming games → {output_csv}")

    # no-op copy (old Windows script tried to 'sync' to itself)
    # leaving a message here so it's obvious
    if os.path.isdir(next_game_dir):
        logging.info(f"NEXT_GAME_DIR ready at {next_game_dir}")
    else:
        logging.warning(
            f"{next_game_dir} is not a directory (unexpected)."
        )

    _pause_and_exit_ok()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error in script 2.")
        _pause_and_exit_ok()
