#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Prediction Utilities Library (2025-26 / 2026 season)

Common utility functions and configuration shared across all scripts:
- Date/paths
- File ops
- Web scraping (Selenium)
- Data wrangling (preprocess, rolling averages, next-game columns)
- Betting helpers (odds conversion, Kelly)
- Team code normalization (PHO→PHX, BKN↔BRK, etc.)
"""

import os
import glob
import logging
import calendar
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# Selenium / webdriver-manager (optional)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("webdriver_manager not installed. Some functions may not work.")
    ChromeDriverManager = None


# ============================================================================
# GLOBAL CONFIGURATIONS
# ============================================================================

CURRENT_SEASON = 2026
ROLLING_WINDOW_SIZE = 9

# ----------------------------------------------------------------------------
# Date utilities
# ----------------------------------------------------------------------------
def get_current_date(days_offset: int = 0) -> Tuple[datetime, str, str]:
    """
    Return "current" date with an optional offset (default=1 day back).

    Returns:
        (datetime_obj, friendly_str, ymd_str)
        friendly_str like: "Thu, Oct 23, 2025"
        ymd_str like: "2025-10-23"
    """
    d = datetime.now() - timedelta(days=days_offset)
    friendly = d.strftime("%a, %b ") + str(int(d.strftime("%d"))) + d.strftime(", %Y")
    ymd = d.strftime("%Y-%m-%d")
    return d, friendly, ymd


# ----------------------------------------------------------------------------
# Directory structure
# ----------------------------------------------------------------------------
def get_directory_paths() -> Dict[str, str]:
    """
    Resolve standard directory paths for the 2026 project structure.

    Returns:
        dict with BASE_DIR, DATA_DIR, STAT_DIR, STANDINGS_DIR, SCORES_DIR,
        NEXT_GAME_DIR, PREDICTION_DIR
    """
    # Pin the base folder (your repo root for 2026 season)
    base_dir = r"D:\1. Python\6. GitHub\Basketball_prediction\2026"
    data_dir = os.path.join(base_dir, "output", "Gathering_Data")

    paths = {
        "BASE_DIR": base_dir,
        "DATA_DIR": data_dir,
        "STAT_DIR": os.path.join(data_dir, "Whole_Statistic"),
        "STANDINGS_DIR": os.path.join(data_dir, "data", f"{CURRENT_SEASON}_standings"),
        "SCORES_DIR": os.path.join(data_dir, "data", f"{CURRENT_SEASON}_scores"),
        "NEXT_GAME_DIR": os.path.join(data_dir, "Next_Game"),
        "PREDICTION_DIR": os.path.join(base_dir, "output", "LightGBM"),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return paths


# ----------------------------------------------------------------------------
# Team name/code maps (full → abbrev used in your stats)
# ----------------------------------------------------------------------------
def get_team_codes() -> Dict[str, str]:
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
        "Phoenix Suns": "PHX",  # prefer PHX internally
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
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    return max(files, key=os.path.getctime) if files else None


def find_file_in_date_range(directory: str, filename_pattern: str, max_days_back: int = 120) -> Tuple[Optional[str], Optional[str]]:
    """
    filename_pattern must contain {} where the date (YYYY-MM-DD) goes.
    """
    for days_back in range(max_days_back + 1):
        date_to_check = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        file_path = os.path.join(directory, filename_pattern.format(date_to_check))
        if os.path.exists(file_path):
            return file_path, date_to_check
    return None, None


def copy_missing_files(src_dir: str, dst_dir: str) -> None:
    import shutil
    src_files = set(os.listdir(src_dir))
    dst_files = set(os.listdir(dst_dir))
    for name in (src_files - dst_files):
        if not name.startswith(".") and not name.endswith(".ipynb"):
            shutil.copy2(os.path.join(src_dir, name), dst_dir)
            logging.info(f"File {name} copied successfully")


# ============================================================================
# WEB SCRAPING
# ============================================================================

def get_html(url: str, selector: str, sleep: int = 5, retries: int = 3, headless: bool = True) -> Optional[str]:
    """
    Fetch element.innerHTML via Selenium. Quiet webdriver-manager, headless by default.
    """
    html = None
    driver = None
    try:
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

        logging.getLogger("webdriver_manager").setLevel(logging.ERROR)
        if ChromeDriverManager:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        for attempt in range(retries):
            try:
                driver.get(url)
                import time
                time.sleep(sleep * (2 ** attempt))  # exponential backoff
                el = driver.find_element(By.CSS_SELECTOR, selector)
                html = el.get_attribute("innerHTML")
                break
            except TimeoutException:
                logging.warning(f"Timeout while loading {url} (attempt {attempt+1}/{retries})")
            except WebDriverException as e:
                logging.error(f"WebDriver error for {url}: {e}")
                break
    finally:
        if driver:
            driver.quit()

    if html is None:
        logging.error(f"Failed to retrieve HTML from {url}")
    return html


def parse_html(html_or_path: str) -> Optional[BeautifulSoup]:
    """
    Accept raw HTML or a file path, return soup stripped of header rows.
    """
    try:
        if os.path.isfile(html_or_path):
            with open(html_or_path, encoding="utf-8") as f:
                html = f.read()
        else:
            html = html_or_path
        soup = BeautifulSoup(html, "html.parser")
        [s.decompose() for s in soup.select("tr.over_header, tr.thead")]
        return soup
    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

def rename_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.values.tolist()
        cols[idxs] = [dup if i == 0 else f"{dup}_{i}" for i in range(len(idxs))]
    df.columns = cols
    return df


def preprocess_nba_data(stats_path: str) -> pd.DataFrame:
    """
    Load + sort by date, add target (won shifted -1 within team),
    drop columns that contain nulls to keep modeling simple.
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

    nulls = pd.isnull(df).sum()
    keep = df.columns[~df.columns.isin(nulls[nulls > 0].index)]
    df = df[keep].copy()
    return df


def calculate_rolling_averages(df: pd.DataFrame, window_size: int = ROLLING_WINDOW_SIZE) -> pd.DataFrame:
    """
    Rolling means per team-season for numeric columns.
    """
    X = df.copy()
    X["season"] = X["season"].astype(str)

    def _roll(team_df: pd.DataFrame) -> pd.DataFrame:
        numeric = team_df.select_dtypes(include=[np.number]).columns
        r = team_df[numeric].rolling(window_size, min_periods=1).mean()
        for c in team_df.columns:
            if c not in numeric:
                r[c] = team_df[c]
        return r

    out = []
    for team, g1 in X.groupby("team"):
        for season, g2 in g1.groupby("season"):
            out.append(_roll(g2))
    return pd.concat(out, ignore_index=True)


def add_next_game_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 'home_next', 'team_opp_next', 'date_next' from the next row for each team.
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
# TEAM CODE NORMALIZATION (aliases across data sources)
# ============================================================================

TEAM_ALIASES: Dict[str, str] = {
    # Common cross-site differences
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
    Normalize team abbreviation differences between sources.
    Example: 'PHO' → 'PHX', 'BKN' → 'BRK'.
    Returns uppercase normalized code; returns input if None/empty.
    """
    if not isinstance(code, str) or code.strip() == "":
        return code
    return TEAM_ALIASES.get(code.strip().upper(), code.strip().upper())


def normalize_team_codes_inplace(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Normalize all given columns in a DataFrame using TEAM_ALIASES.
    """
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(normalize_team_code)
    return df


# ============================================================================
# BETTING UTILITIES
# ============================================================================

def kelly_frac(p: float, o: float, f: float = 1.0) -> float:
    """
    Kelly fraction for decimal odds.
    p: win probability (0..1)
    o: decimal odds (>1)
    f: fraction of Kelly to use (0..1)
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
    American odds → implied probability.
    Robust to None/NaN/strings like "nan" / "1,85".
    Returns None if odds missing/invalid.
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

    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def am_to_dec(ml) -> Optional[float]:
    """
    American → Decimal odds. Robust to None/NaN/strings.
    Returns None if invalid.
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

    return (ml / 100 + 1.0) if ml > 0 else (100.0 / abs(ml) + 1.0)


# ============================================================================
# BETTING STATISTICS
# ============================================================================

def get_home_win_rates(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute home win rates using last 20 games per team (home/away),
    then filter to home games within that window.

    Expects columns: ['home_team','away_team','result','date'] (date parseable)
    """
    # Ensure date is datetime
    if "date" in pred_df.columns and not np.issubdtype(pred_df["date"].dtype, np.datetime64):
        pred_df = pred_df.copy()
        pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")

    teams = pred_df["home_team"].dropna().unique().tolist()
    out = {}
    for t in teams:
        tg = pred_df[(pred_df["home_team"] == t) | (pred_df["away_team"] == t)].copy()
        tg = tg.sort_values("date", ascending=False).head(20)

        home = tg[tg["home_team"] == t]
        total_home = len(home)
        home_wins = int((home["result"] == t).sum())
        rate = round(home_wins / total_home, 2) if total_home > 0 else 0.0

        out[t] = {
            "Total Last 20 Games": len(tg),
            "Total Home Games": total_home,
            "Home Wins": home_wins,
            "Home Win Rate": rate,
        }

    df = pd.DataFrame.from_dict(out, orient="index")
    return df.sort_values("Home Win Rate", ascending=False)


# ============================================================================
# SMALL HELPERS
# ============================================================================

def safe_to_numeric_comma(x) -> Optional[float]:
    """
    Convert string with comma decimal to float; return None if invalid.
    """
    try:
        if isinstance(x, str):
            x = x.replace(",", ".")
        v = float(x)
        return v
    except Exception:
        return None
