#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nba_utils_2026.py

Shared utilities for the 2025-26 / 2026 season pipeline:
- Date + path management
- File ops
- HTML fetching (requests first, Selenium fallback if needed)
- Basketball Reference parsing helpers
- Feature engineering (rolling windows, next-game columns)
- Team code normalization across data sources
- Betting math helpers

IMPORTANT RUNTIME NOTES
-----------------------
1. GitHub Actions daily scraper (Script 1) should:
   - Use fetch_html_requests() for season/month schedule pages.
   - Use fetch_boxscore_via_selenium() ONLY when the plain cached
     HTML for a box score is invalid or missing.

2. Basketball Reference sometimes returns anti-bot HTML that LOOKS
   like a page but has no <table id="line_score">.
   We treat those as invalid and skip/keep trying.

3. We never keep "bad" HTML in cache. We only save if valid.
"""

import os
import glob
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# Selenium imports kept for fallback mode / local runs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

CURRENT_SEASON = 2026          # used when labeling parsed box score rows
ROLLING_WINDOW_SIZE = 9        # rolling feature window (games)


# ============================================================================
# DATE UTILITIES
# ============================================================================

def get_current_date(days_offset: int = 0) -> Tuple[datetime, str, str]:
    """
    Return "current" date with optional offset (default: 0 days).

    Returns:
        (datetime_obj, friendly_str, ymd_str)

    Example:
        d, "Thu, Oct 23, 2025", "2025-10-23"
    """
    d = datetime.now() - timedelta(days=days_offset)
    friendly = d.strftime("%a, %b ") + str(int(d.strftime("%d"))) + d.strftime(", %Y")
    ymd = d.strftime("%Y-%m-%d")
    return d, friendly, ymd


# ============================================================================
# DIRECTORY LAYOUT
# ============================================================================

def get_directory_paths() -> Dict[str, str]:
    """
    Resolve standard directory paths for the 2026 project structure.

    Returns:
        {
          "BASE_DIR":        /.../Basketball_prediction
          "DATA_DIR":        .../2026/output/Gathering_Data
          "STAT_DIR":        .../Whole_Statistic
          "STANDINGS_DIR":   .../data/2026_standings
          "SCORES_DIR":      .../data/2026_scores
          "NEXT_GAME_DIR":   .../Next_Game
          "PREDICTION_DIR":  .../2026/output/LightGBM
        }
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "2026", "output", "Gathering_Data")

    paths = {
        "BASE_DIR": base_dir,
        "DATA_DIR": data_dir,
        "STAT_DIR": os.path.join(data_dir, "Whole_Statistic"),
        "STANDINGS_DIR": os.path.join(data_dir, "data", f"{CURRENT_SEASON}_standings"),
        "SCORES_DIR": os.path.join(data_dir, "data", f"{CURRENT_SEASON}_scores"),
        "NEXT_GAME_DIR": os.path.join(data_dir, "Next_Game"),
        "PREDICTION_DIR": os.path.join(base_dir, "2026", "output", "LightGBM"),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return paths


# ============================================================================
# TEAM NAMES / CODES
# ============================================================================

def get_team_codes() -> Dict[str, str]:
    """
    Map official Basketball Reference / NBA team names
    to the short codes we use in csv/model features.
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
        "Phoenix Suns": "PHX",  # internal canonical is PHX
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def get_latest_file(folder: str, prefix: str, ext: str) -> Optional[str]:
    """
    Return the most recently modified file in `folder`
    matching <prefix>*<ext>, or None.
    """
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    return max(files, key=os.path.getctime) if files else None


def find_file_in_date_range(
    directory: str,
    filename_pattern: str,
    max_days_back: int = 120
) -> Tuple[Optional[str], Optional[str]]:
    """
    filename_pattern must contain {} where the date (YYYY-MM-DD) goes.

    Tries today, yesterday, ... up to max_days_back, and returns:
        (file_path_found, "YYYY-MM-DD_when_found")
    If nothing found: (None, None)
    """
    for days_back in range(max_days_back + 1):
        date_to_check = (
            datetime.now() - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")
        candidate = os.path.join(
            directory,
            filename_pattern.format(date_to_check)
        )
        if os.path.exists(candidate):
            return candidate, date_to_check
    return None, None


def copy_missing_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy any CSV/etc. that exists in src_dir but not in dst_dir.
    Skips hidden files and notebooks.

    Used at the end of Script 1 to make sure output dirs are synced.
    """
    import shutil
    src_files = set(os.listdir(src_dir))
    dst_files = set(os.listdir(dst_dir))

    for name in (src_files - dst_files):
        if not name.startswith(".") and not name.endswith(".ipynb"):
            shutil.copy2(os.path.join(src_dir, name), dst_dir)
            logging.info(f"File {name} copied successfully")


# ============================================================================
# RAW HTML FETCHING
# ============================================================================

def fetch_html_requests(url: str, timeout: int = 20) -> Optional[str]:
    """
    Fetch raw HTML using 'requests' with a semi-realistic User-Agent
    so basketball-reference doesn't instantly reject us.

    Returns:
        page.text (str) on 200
        None otherwise
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
        logging.warning(
            f"[fetch_html_requests] error for {url}: {e}"
        )
        return None


def build_driver() -> webdriver.Chrome:
    """
    Spin up a hardened headless Chrome driver.

    NOTE:
    - Script 1 in CI should *not* depend on this directly in a tight loop,
      because that's what sometimes hung the job.
    - We keep it for fallback/manual use and for future scripts that
      really need JS rendering.
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


def boxscore_html_is_valid(html_text: str) -> bool:
    """
    Check if a supposed 'box score page' from Basketball Reference
    is actually a real box score (and not anti-bot garbage).

    We consider it valid if it contains a <table id="line_score">.
    """
    if not html_text or "line_score" not in html_text:
        return False

    soup_check = BeautifulSoup(html_text, "html.parser")
    line_tbl = soup_check.find("table", id="line_score")
    return line_tbl is not None


def fetch_boxscore_via_selenium(url: str, timeout_s: int = 20) -> Optional[str]:
    """
    Launch a fresh headless Chrome, open the box score URL, pull #content,
    wrap it so pandas can read the tables, and return that HTML.

    Returns:
        wrapped_html (str) if valid
        None if it times out or delivers anti-bot junk
    """
    driver = None
    try:
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
        driver.set_page_load_timeout(timeout_s)

        driver.get(url)

        # wait up to ~20s for #content to appear
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#content"))
            )
        except TimeoutException:
            logging.warning(f"[fetch_boxscore_via_selenium] timeout waiting for #content: {url}")
            return None

        el = driver.find_element(By.CSS_SELECTOR, "#content")
        inner = el.get_attribute("innerHTML")
        wrapped_html = f"<div id='content'>{inner}</div>"

        # validate before returning
        if not boxscore_html_is_valid(wrapped_html):
            logging.warning(
                f"[fetch_boxscore_via_selenium] got HTML but no valid line_score table for {url}"
            )
            return None

        return wrapped_html

    except (TimeoutException, WebDriverException) as e:
        logging.warning(f"[fetch_boxscore_via_selenium] Selenium error for {url}: {e}")
        return None
    except Exception as e:
        logging.warning(f"[fetch_boxscore_via_selenium] Unexpected error for {url}: {e}")
        return None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def get_html_with_driver(
    driver: webdriver.Chrome,
    url: str,
    selector: str,
    sleep: int = 5,
    retries: int = 3,
) -> Optional[str]:
    """
    Use an EXISTING Selenium driver to fetch `url`, wait a little,
    and return innerHTML of the first element matching `selector`.

    This is for local/manual scripts. Script 1 in CI should not rely on this.
    """
    html = None

    for attempt in range(retries):
        try:
            driver.get(url)

            # polite backoff: 5s, 10s, 20s
            time.sleep(sleep * (2 ** attempt))

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            el = driver.find_element(By.CSS_SELECTOR, selector)
            html = el.get_attribute("innerHTML")
            break

        except (TimeoutException, NoSuchElementException):
            logging.warning(
                f"[get_html_with_driver] Timeout / missing {selector} on {url} "
                f"(attempt {attempt+1}/{retries})"
            )
        except WebDriverException as e:
            logging.error(
                f"[get_html_with_driver] WebDriver error for {url}: {e} "
                f"(attempt {attempt+1}/{retries})"
            )
            break

    if html is None:
        logging.error(f"[get_html_with_driver] Failed to retrieve HTML from {url}")
    return html


def get_html(
    url: str,
    selector: str,
    sleep: int = 5,
    retries: int = 3,
) -> Optional[str]:
    """
    Legacy convenience wrapper:
    - build a driver
    - get HTML from `selector`
    - quit the driver

    Keep for backwards compatibility. Prefer fetch_html_requests() or
    fetch_boxscore_via_selenium() in new code.
    """
    driver = None
    try:
        driver = build_driver()
        return get_html_with_driver(
            driver,
            url,
            selector,
            sleep=sleep,
            retries=retries,
        )
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


# ============================================================================
# HTML PARSING HELPERS
# ============================================================================

def parse_html(html_or_path: str) -> Optional[BeautifulSoup]:
    """
    Accept raw HTML string OR a file path.
    Return a BeautifulSoup with Basketball Reference's repeated
    header rows removed (tr.over_header, tr.thead).
    """
    try:
        if os.path.isfile(html_or_path):
            with open(html_or_path, encoding="utf-8") as f:
                html = f.read()
        else:
            html = html_or_path

        soup = BeautifulSoup(html, "html.parser")

        # Remove repeated header junk that pandas.read_html hates
        for s in soup.select("tr.over_header, tr.thead"):
            s.decompose()

        return soup

    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        return None


# ============================================================================
# DATA PROCESSING / FEATURE ENGINEERING
# ============================================================================

def rename_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basketball Reference sometimes repeats column names.
    We rename duplicates with suffixes _1, _2, ... so pandas stops yelling.
    """
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.values.tolist()
        cols[idxs] = [dup if i == 0 else f"{dup}_{i}" for i in range(len(idxs))]
    df.columns = cols
    return df


def preprocess_nba_data(stats_path: str) -> pd.DataFrame:
    """
    Load 'nba_games_YYYY-MM-DD.csv' (whole dataset snapshot),
    sort by date, create 'target' = next game's result for that team,
    and drop columns that are 100% NaN except for core identifiers.

    Output columns include:
      team, team_opp, home, won, season, date, target, stats...
    """
    df = pd.read_csv(stats_path, index_col=0)
    df = df.sort_values("date")

    def _add_target(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["target"] = g["won"].shift(-1)
        return g

    df = df.groupby("team", as_index=False).apply(_add_target)
    df = df.copy()
    df["target"] = df["target"].fillna(2).astype(int)

    # drop columns that are entirely NaN, EXCEPT core id columns
    nulls = pd.isnull(df).sum()
    drop_full_na = nulls[nulls > 0].index.tolist()

    core_keep = {"team", "team_opp", "home", "won", "season", "date", "target"}
    truly_drop = [c for c in drop_full_na if c not in core_keep and df[c].isna().all()]

    if truly_drop:
        df = df.drop(columns=truly_drop)

    return df


def calculate_rolling_averages(
    df: pd.DataFrame,
    window_size: int = ROLLING_WINDOW_SIZE
) -> pd.DataFrame:
    """
    Per team *and* season, compute rolling means over the last `window_size`
    games for numeric stats.

    We keep non-numeric columns (team, season, etc.) side by side.
    """
    X = df.copy()
    X["season"] = X["season"].astype(str)

    def _roll(team_df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = team_df.select_dtypes(include=[np.number]).columns
        rolled = team_df[numeric_cols].rolling(
            window_size,
            min_periods=1
        ).mean()

        # copy non-numeric columns unchanged
        for c in team_df.columns:
            if c not in numeric_cols:
                rolled[c] = team_df[c]

        return rolled

    out = []
    for team, g_team in X.groupby("team"):
        for season, g_season in g_team.groupby("season"):
            out.append(_roll(g_season))

    return pd.concat(out, ignore_index=True)


def add_next_game_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row (each team's game), attach info about that team's NEXT game:
    - home_next: will they be home (1) or away (0) next time?
    - team_opp_next: who do they play next?
    - date_next: when?

    This is used as features for predicting future games.
    """
    result = df.copy()
    result["home_next"] = None
    result["team_opp_next"] = None
    result["date_next"] = None

    rows = df.to_dict("records")
    groups: Dict[str, List[Tuple[int, dict]]] = {}
    for i, r in enumerate(rows):
        t = str(r.get("team", ""))
        groups.setdefault(t, []).append((i, r))

    for t, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda x: x[1].get("date", ""))
        for i in range(len(lst_sorted) - 1):
            cur_idx = lst_sorted[i][0]
            nxt = lst_sorted[i + 1][1]
            result.at[cur_idx, "home_next"] = nxt.get("home")
            result.at[cur_idx, "team_opp_next"] = nxt.get("team_opp")
            result.at[cur_idx, "date_next"] = nxt.get("date")

    return result


# ============================================================================
# TEAM CODE NORMALIZATION
# ============================================================================

TEAM_ALIASES: Dict[str, str] = {
    # Sportsbooks / BR / NBA.com / random abbreviations we see
    "PHO": "PHX",
    "PHX": "PHX",
    "BKN": "BRK",
    "BRK": "BRK",
    "CHO": "CHO",
    "CHA": "CHO",
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


def normalize_team_code(code: Optional[str]) -> Optional[str]:
    """
    Normalize a single team abbreviation.
    Examples:
      'PHO' -> 'PHX'
      'BKN'/'BRK' -> 'BRK'
      'CHA' -> 'CHO'
    """
    if not isinstance(code, str) or code.strip() == "":
        return code
    return TEAM_ALIASES.get(code.strip().upper(), code.strip().upper())


def normalize_team_codes_inplace(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Normalize team codes for a set of columns in-place.
    Returns the same df for chaining.
    """
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(normalize_team_code)
    return df


# ============================================================================
# BETTING UTILS
# ============================================================================

def kelly_frac(p: float, o: float, f: float = 1.0) -> float:
    """
    Kelly fraction for decimal odds.

    p: model win probability (0..1)
    o: decimal odds (>1)
    f: fraction of Kelly to use (0..1), e.g. 0.25 = quarter Kelly

    Returns:
        fraction of bankroll to stake (0..1). Never negative.
    """
    try:
        b = float(o) - 1.0
        if b <= 0 or p is None or np.isnan(p):
            return 0.0
        return max(((b * p - (1 - p)) / b) * float(f), 0.0)
    except Exception:
        return 0.0


def impute_prob(ml) -> Optional[float]:
    """
    Convert American odds to implied win probability.
    Handles strings ('1,85', 'nan') and NaN.

    Returns:
        p in [0,1] or None if invalid.
    """
    if ml is None:
        return None

    try:
        if isinstance(ml, str):
            s = ml.strip().lower().replace(",", ".")
            if s == "" or s == "nan":
                return None
            ml = float(s)

        if isinstance(ml, float):
            if pd.isna(ml):
                return None
            ml = int(round(ml))
        else:
            ml = int(ml)

    except (ValueError, TypeError):
        return None

    # favorite (negative moneyline)
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)

    # underdog (positive moneyline)
    return 100 / (ml + 100)


def am_to_dec(ml) -> Optional[float]:
    """
    American odds -> decimal odds.

    +150  -> 2.50
    -200  -> 1.50
    """
    if ml is None:
        return None

    try:
        if isinstance(ml, str):
            s = ml.strip().lower().replace(",", ".")
            if s == "" or s == "nan":
                return None
            ml = float(s)

        if isinstance(ml, float):
            if pd.isna(ml):
                return None
            ml = int(round(ml))
        else:
            ml = int(ml)

    except (ValueError, TypeError):
        return None

    # positive moneyline
    if ml > 0:
        return ml / 100 + 1.0

    # negative moneyline
    return 100.0 / abs(ml) + 1.0


def get_home_win_rates(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute (for each team) how strong they are at home recently.

    We:
    - take that team's last 20 total games
    - look at just the ones where they were home
    - compute "home win rate"

    Expects columns:
      ['home_team','away_team','result','date']
    where 'result' is the winner's team code for finished games.

    Returns:
        DataFrame sorted by Home Win Rate desc.
    """
    df_local = pred_df.copy()

    # make sure date is datetime
    if "date" in df_local.columns and not np.issubdtype(
        df_local["date"].dtype,
        np.datetime64
    ):
        df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce")

    teams = df_local["home_team"].dropna().unique().tolist()
    out = {}

    for t in teams:
        tg = df_local[
            (df_local["home_team"] == t) | (df_local["away_team"] == t)
        ].copy()

        tg = tg.sort_values("date", ascending=False).head(20)

        home_rows = tg[tg["home_team"] == t]
        total_home = len(home_rows)
        home_wins = int((home_rows["result"] == t).sum())
        rate = round(home_wins / total_home, 2) if total_home > 0 else 0.0

        out[t] = {
            "Total Last 20 Games": len(tg),
            "Total Home Games": total_home,
            "Home Wins": home_wins,
            "Home Win Rate": rate,
        }

    df_rate = pd.DataFrame.from_dict(out, orient="index")
    return df_rate.sort_values("Home Win Rate", ascending=False)


# ============================================================================
# MISC
# ============================================================================

def safe_to_numeric_comma(x) -> Optional[float]:
    """
    Convert string with comma decimal ('1,85') to float (1.85).
    Returns None if invalid.
    """
    try:
        if isinstance(x, str):
            x = x.replace(",", ".")
        v = float(x)
        return v
    except Exception:
        return None
