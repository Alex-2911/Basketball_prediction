#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_Isotonic_based_betting_strategy_2026.py  (DROP-IN)

Core requirement implemented:
- Script 5 ALWAYS generates strategy params each run (no reading from strategy_params.json).
- web/public/data/local_matched_games_latest.csv contains the ACTUAL FILTERED subset
  from the LAST 200 PLAYED games window, ending at the latest played game date (typically yesterday).

Everything else is kept as close as possible to your existing version.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from live_probability_pipeline import build_probability_chain_config, prepare_live_probability_columns

from nba_utils_2026 import (
    get_current_date,
    get_directory_paths,
    normalize_team_code,
)

# Allow importing helper from 2026/scripts (2026 is not a valid package name)
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from persist_script11_watchlist_history import persist_script11_watchlist_history
except Exception:
    persist_script11_watchlist_history = None

# -----------------------------
# CONSTANTS / COLUMN NAMES
# -----------------------------

DATE_COL = "game_date"
RESULT_COL = "home_team_won"
RESULT_RAW_COL = "result_raw"
PRED_PROBA_COL = "pred_home_win_proba"
HOME_ODDS_COL = "closing_home_odds"
AWAY_ODDS_COL = "closing_away_odds"
HOMEWR_COL = "home_win_rate"
ISO_COL = "iso_proba_home_win"
PROB_ISO_OOS_TIME_COL = "prob_iso_oos_time"
PROB_LIVE_OOS_PROXY_COL = "prob_live_oos_proxy"
PROB_LIVE_SAFE_COL = "prob_live_safe"
PROB_COL_HIST = PROB_ISO_OOS_TIME_COL
PROB_COL_LIVE = PROB_LIVE_SAFE_COL
MIN_TRAIN_OOS_TIME = 50
MIN_STEP_OOS_TIME = 10
MIN_TRAIN_OOS_PROXY = 300

# Grid search
FLAT_STAKE = 100.0
ODDS_MIN_GRID = [1.10, 1.25, 1.40, 1.60, 2.00, 2.30]
ODDS_MAX_GRID = [2.00, 2.10, 2.50, 3.00, 3.20]
PROB_MIN_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HOMEWR_MIN_GRID = [0.50, 0.55, 0.60, 0.65]

# Dashboard / shortlist logic
MIN_EV_DEFAULT = 0.0
PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80

# IMPORTANT: dashboard window is last 200 played games
LOCAL_SEARCH_N = 200
FAIR_COMPARE_N = 200
MIN_HIST_ROWS_FOR_LOCAL = 100
LOCAL_TAIL_LADDER = [300, 400, 500]
WALK_TRAIN_MIN_DAYS = 21
WALK_TEST_DAYS = 21
WALK_STEP_DAYS = 7
MIN_WALK_SPLITS = 4
MIN_TEST_TRADES_TOTAL = 50
MIN_ACTIVE_SPLITS = 3
MIN_TRADES_PER_ACTIVE_SPLIT = 5
MIN_TRADES_TRAIN = 20
MIN_TRADES_PER_WINDOW = 25
N_WINDOWS = [300, 400, 500]
STABILITY_HITS_NEEDED = 2
USE_SOFT_GATE = True
SOFT_Q = 0.20
MIN_Q_TRADES = 4
SCORE_MODE = "lcb_roi"
LCB_K = 0.5

START_BANKROLL = 1000.0

STRATEGY_VARIANTS = {"acc", "iso"}

OUTPUT_BASE_COLUMNS = [
    "home_team",
    "away_team",
    "home_team_prob",
    "odds_1",
    "odds_2",
    "result",
    "date",
    "accuracy",
]

DATE_SOURCE_CANDIDATES = [
    "date",
    DATE_COL,
    "game_date",
    "date_x",
    "Date",
    "DATE",
    "datetime",
    "timestamp",
]

OUTPUT_PROBABILITY_COLUMNS = [
    "prob_iso",
    "prob_iso_insample",
    "prob_iso_oos_time",
    "prob_live_oos_proxy",
    "prob_live_safe_pre_clip",
    "prob_base",
    "prob_live_safe",
    "prob_used",
]

LOCAL_MATCHED_EXPORT_COLUMNS = [
    "date", "home_team", "away_team",
    "home_win_rate", "prob_iso", "prob_used",
    "odds_1", "EV_€_per_100", "win", "pnl", "stake"
]

SHORTLIST_COLUMNS = [
    DATE_COL, "home_team", "away_team", "home_team_prob", "prob_iso", PROB_ISO_OOS_TIME_COL,
    PROB_LIVE_OOS_PROXY_COL, "prob_live_safe_pre_clip", "prob_base", "prob_used",
    "odds_1", "market_implied_p_raw", "market_implied_p_devig", "model_market_gap", "model_market_gap_flag",
    "live_underdog_upscale_guard_triggered", "live_shrink_triggered",
    "live_oos_proxy_ready", "live_oos_proxy_train_rows", "live_oos_proxy_bin_n",
    "live_oos_proxy_bin_winrate", "blocked_by", HOMEWR_COL, "EV_€_per_100",
]


@dataclass
class StrategyParams:
    min_home_win_rate: float
    min_odds: float
    max_odds: float
    min_iso_proba: float

# (rest of file unchanged...)
