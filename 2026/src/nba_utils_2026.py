#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nba_utils_2026.py

Shared helpers for the 2026 pipeline.

This module MUST provide (used by script 1 & script 2):
- CURRENT_SEASON
- get_current_date()
- get_directory_paths()
- fetch_html_requests()
- fetch_boxscore_via_selenium()
- parse_html()
- rename_duplicated_columns()
- copy_missing_files()
- get_team_codes()
- normalize_team_code(), normalize_team_codes_inplace()

Keep this file VERY STABLE, because all pipeline steps import from here.
"""

import os
import glob
import time
import shutil
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup

# selenium imports (used in fallback only, for deep refresh of box scores)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# -----------------------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------------------

CURRENT_SEASON = 2026  # scraping NBA_2026_*


# -----------------------------------------------------------------------------
# TEAM NAMES / CODES
# -----------------------------------------------------------------------------

def get_team_codes() -> dict[str, str]:
    """
    Map Basketball Reference full team names to our internal 3-letter codes.
    We use these codes consistently in features, rolling stats, predictions, etc.
    """
    return {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",
        "Charlotte Hornets": "CHO",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "LA Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",  # normalize PHO -> PHX
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }

# Abbreviation cleanup: sportsbook / NBA / BR all disagree on some codes.
TEAM_ALIASES: dict[str, str] = {
    "PHO": "PHX",
    "PHX": "PHX",
    "BKN": "BRK",
    "BRK": "BRK",
    "CHA": "CHO",
    "CHO": "CHO",
    "WSH": "WAS",
    "WAS": "WAS",
    "GS":  "GSW",
    "GSW": "GSW",
    "NO":  "NOP",
    "NOP": "NOP",
    "NY":  "NYK",
    "NYK": "NYK",
    "SA":  "SAS",
    "SAS": "SAS",
    "UTAH": "UTA",
    "UTA": "UTA",
    "OKL": "OKC",
    "OKC": "OKC",
}

def normalize_team_code(code: str | None) -> str | None:
    """
    Normalize a single team abbreviation to the code we use in training data.
    Example: 'PHO' -> 'PHX', 'BKN' -> 'BRK', 'CHA' -> 'CHO'.
    """
    if code is None:
        return None
    code_up = str(code).strip().upper()
    if code_up == "":
        return code_up
    return TEAM_ALIASES.get(code_up, code_up)

def normalize_team_codes_inplace(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    In-place: normalizes abbrev columns (like 'home_team', 'away_team').
    Returns df for chaining.
    """
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(normalize_team_code)
    return df


# -----------------------------------------------------------------------------
# DATE / PATH HELPERS
# -----------------------------------------------------------------------------

def get_current_date(days_offset: int = 0) -> Tuple[datetime, str, str]:
    """
    Returns:
      now_dt (datetime),
      friendly_str like 'Sat, Nov 1, 2025',
      ymd_str like '2025-11-01'
    """
    d = datetime.now() - timedelta(days=days_offset)
    friendly = d.strftime("%a, %b ") + str(int(d.strftime("%d"))) + d.strftime(", %Y")
    ymd = d.strftime("%Y-%m-%d")
    return d, friendly, ymd


def get_directory_paths() -> Dict[str, str]:
    """
    Standardized folder layout for 2026 branch.
    Ensures dirs exist.
    """
    base_dir = os.getcwd()  # repo root during Actions
    data_dir = os.path.join(base_dir, "2026", "output", "Gathering_Data")

    paths = {
        "BASE_DIR": base_dir,
        "DATA_DIR": data_dir,
        "STAT_DIR": os.path.join(data_dir, "Whole_Statistic"),
        "STANDINGS_DIR": os.path.join(
            data_dir, "data", f"{CURRENT_SEASON}_standings"
        ),
        "SCORES_DIR": os.path.join(
            data_dir, "data", f"{CURRENT_SEASON}_scores"
        ),
        "NEXT_GAME_DIR": os.path.join(data_dir, "Next_Game"),
        "PREDICTION_DIR": os.path.join(
            base_dir, "2026", "output", "LightGBM"
        ),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return paths


# -----------------------------------------------------------------------------
# BASIC FILE OPS
# -----------------------------------------------------------------------------

def copy_missing_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy any files from src_dir to dst_dir that dst_dir doesn't have.
    Skip hidden files and notebooks.
    """
    src_files = set(os.listdir(src_dir))
    dst_files = set(os.listdir(dst_dir))
    for name in (src_files - dst_files):
        if not name.startswith(".") and not name.endswith(".ipynb"):
            shutil.copy2(os.path.join(src_dir, name), dst_dir)
            logging.info(f"File {name} copied successfully")


def rename_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basketball Reference sometimes repeats column names ('mp', etc.).
    This gives duplicates suffixes _1, _2... so pandas doesn't break.
    """
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        cols[idxs] = [
            dup if i == 0 else f"{dup}_{i}" for i in range(len(idxs))
        ]
    df.columns = cols
    return df
# -----------------------------------------------------------------------------
# BASIC FILE OPS
# -----------------------------------------------------------------------------

def copy_missing_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy any files from src_dir to dst_dir that dst_dir doesn't have.
    Skip hidden files and notebooks.
    """
    src_files = set(os.listdir(src_dir))
    dst_files = set(os.listdir(dst_dir))
    for name in (src_files - dst_files):
        if not name.startswith(".") and not name.endswith(".ipynb"):
            shutil.copy2(os.path.join(src_dir, name), dst_dir)
            logging.info(f"File {name} copied successfully")


def rename_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basketball Reference sometimes repeats column names ('mp', etc.).
    This gives duplicates suffixes _1, _2... so pandas doesn't break.
    """
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        cols[idxs] = [
            dup if i == 0 else f"{dup}_{i}" for i in range(len(idxs))
        ]
    df.columns = cols
    return df


def get_latest_file(folder: str, prefix: str, ext: str) -> Optional[str]:
    """
    Return the most recently modified file in `folder`
    that matches <prefix>*<ext>, or None if nothing matches.

    Example:
        get_latest_file(
            folder=STAT_DIR,
            prefix="nba_games_",
            ext=".csv"
        )
    """
    pattern = os.path.join(folder, f"{prefix}*{ext}")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)


def find_file_in_date_range(
    directory: str,
    filename_pattern: str,
    max_days_back: int = 120,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find a dated file walking backward in time.

    filename_pattern MUST have exactly one '{}' placeholder where
    the date string (YYYY-MM-DD) goes.

    We'll try:
      today (0 days back),
      yesterday (1 day back),
      ...
      today-max_days_back.

    We return:
        (full_path, matched_date_string)
    or
        (None, None) if nothing was found.

    Example call:
        file_path, date_used = find_file_in_date_range(
            STAT_DIR,
            "nba_games_{}.csv",
            max_days_back=150
        )
    """
    for days_back in range(max_days_back + 1):
        date_to_check = (
            datetime.now() - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        candidate_name = filename_pattern.format(date_to_check)
        full_path = os.path.join(directory, candidate_name)

        if os.path.exists(full_path):
            return full_path, date_to_check

    return None, None

# -----------------------------------------------------------------------------
# HTML CLEANING / PARSING
# -----------------------------------------------------------------------------

def parse_html(html_or_path: str) -> Optional[BeautifulSoup]:
    """
    Accept raw HTML string OR a path to an .html file.
    Return BeautifulSoup with Basketball Reference's repeated header rows removed.

    Important: BRef puts <tr class="over_header"> and <tr class="thead"> rows
    inside <tbody>. If we don't drop them, pandas.read_html() often returns
    MultiIndex columns. Then code like .index.str.lower() explodes with
    "Can only use .str accessor with Index, not MultiIndex".
    """
    try:
        if os.path.isfile(html_or_path):
            with open(html_or_path, encoding="utf-8") as f:
                html = f.read()
        else:
            html = html_or_path

        soup = BeautifulSoup(html, "html.parser")

        # remove duplicate header/overheader rows to avoid MultiIndex columns
        for bad in soup.select("tr.over_header, tr.thead"):
            bad.decompose()

        return soup
    except Exception as e:
        logging.error(f"parse_html() failed: {e}")
        return None


# -----------------------------------------------------------------------------
# HTTP SCRAPE (REQUESTS)
# -----------------------------------------------------------------------------

def fetch_html_requests(url: str, timeout: int = 20) -> Optional[str]:
    """
    Plain requests GET with a desktop-ish User-Agent.
    Used first (cheap, fast). Returns HTML text or None.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        logging.warning(
            f"[fetch_html_requests] {url} -> status {resp.status_code}"
        )
        return None
    except Exception as e:
        logging.warning(f"[fetch_html_requests] error for {url}: {e}")
        return None


# -----------------------------------------------------------------------------
# SELENIUM FALLBACK FOR BOX SCORES
# -----------------------------------------------------------------------------

def _build_headless_driver() -> webdriver.Chrome:
    """
    Spin up a hardened headless Chrome that works in GitHub Actions:
    - no-sandbox
    - disable-dev-shm-usage
    - fixed window size
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-dev-tools")
    chrome_options.add_argument("--remote-debugging-port=9222")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(20)
    return driver


def fetch_boxscore_via_selenium(url: str, timeout_s: int = 20) -> Optional[str]:
    """
    SECOND TRY if requests() HTML looked like bot-blocked junk.

    We:
    - start headless Chrome
    - open the box score url
    - grab #content innerHTML
    - wrap it in a <div id='content'>...</div> so downstream parsing sees a container

    Returns HTML string or None on failure.
    """
    driver = None
    try:
        driver = _build_headless_driver()
        driver.get(url)

        # tiny pause so dynamic content is injected
        time.sleep(2)

        el = driver.find_element(By.CSS_SELECTOR, "#content")
        inner = el.get_attribute("innerHTML")
        if not inner or inner.strip() == "":
            logging.warning(f"[selenium] empty #content for {url}")
            return None

        html_out = "<div id='content'>" + inner + "</div>"
        return html_out

    except TimeoutException:
        logging.warning(f"[selenium] timeout {url}")
        return None
    except WebDriverException as e:
        logging.warning(f"[selenium] WebDriverException on {url}: {e}")
        return None
    except Exception as e:
        logging.warning(f"[selenium] unexpected {url}: {e}")
        return None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
