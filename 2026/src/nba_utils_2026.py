#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nba_utils_2026.py

Shared helpers for the 2026 pipeline.

This module MUST provide:
- CURRENT_SEASON
- get_current_date()
- get_directory_paths()
- fetch_html_requests()
- fetch_boxscore_via_selenium()     <-- required by script 1
- rename_duplicated_columns()
- copy_missing_files()

Plus: supporting imports that function 1 expects.
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

# selenium imports (used in fallback only)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# -----------------------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------------------

CURRENT_SEASON = 2026  # <- we are scraping NBA_2026_*
# you can tweak ROLLING_WINDOW_SIZE etc. in other scripts if needed


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

        # very light throttle — some pages (especially right after final buzzer)
        # still do a bit of dynamic insert, so give them a moment:
        time.sleep(2)

        el = driver.find_element(By.CSS_SELECTOR, "#content")
        inner = el.get_attribute("innerHTML")
        if not inner or inner.strip() == "":
            logging.warning(f"[selenium] empty #content for {url}")
            return None

        # make sure we ship something that looks like full doc-ish html to pandas
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
