"""
Core constants and configuration for NBA Prediction System

This module contains shared constants used across the prediction system.
"""

# Season configuration
ROLLING_WINDOW_SIZE = 9

# Team name mappings and aliases
TEAM_ALIAS_MAP = {
    # Odds API uses different codes than Basketball Reference
    "PHO": "PHX",  # Phoenix Suns
    "PHX": "PHO",
    "BKN": "BRK",  # Brooklyn Nets
    "BRK": "BKN",
    "CHA": "CHO",  # Charlotte Hornets
    "CHO": "CHA",
}

FULL_NAME_TO_ABBREV = {
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
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# LightGBM model parameters (from generate_predictions.py)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 10,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 10,
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": 42,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "max_depth": 7,
    "min_child_weight": 5,
}

# Kelly Criterion defaults
KELLY_DEFAULTS = {
    "bet_fraction": 0.5,  # Half Kelly
    "cap_fraction": 0.30,  # Max 30% of bankroll per bet
    "absolute_cap": 300.0,  # Absolute maximum stake
}

# Strategy thresholds for betting
STRATEGY_THRESHOLDS = {
    "odds_min": 1.18,
    "odds_max": 3.00,
    "prob_min": 0.40,
    "home_win_rate_min": 0.50,
}

# Preferred sportsbooks for odds (in order of preference)
PREFERRED_SPORTSBOOKS = ["draftkings", "fanduel"]

# File search configuration
MAX_DAYS_BACK = 120  # Maximum days to look back for prediction files
