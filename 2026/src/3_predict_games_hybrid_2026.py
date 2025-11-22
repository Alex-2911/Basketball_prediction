#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 3 of 5 (2025-26 season): Predict Next Game Day with LightGBM + Odds

- Builds rolling features on team game logs
- Trains LightGBM on historical data
- Predicts win probabilities for the next game day
- Fetches bookmaker odds from The Odds API (hard-coded key)
- Outputs `nba_games_predict_YYYY-MM-DD.csv` with:
    home_team, away_team, home_team_prob, result, odds 1, odds 2, date

Run order:
    1. 1_get_data_previous_game_day.py
    2. 2_get_data_next_game_day_2026.py  -> games_df_YYYY-MM-DD.csv
    3. 3_predict_next_game_day_2026.py   -> nba_games_predict_YYYY-MM-DD.csv
    4. 4_calculate_betting_statistics_2026.py
    5. 5_kelly_betting_2026.py
"""

import os
import glob
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Shared utilities
from nba_utils_2026 import (
    CURRENT_SEASON,
    get_current_date,
    get_directory_paths,
)

# ----------------------------------------------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------------------------------------------

ROLLING_WINDOW_SIZE = 9  # your notebook uses 9 but keeps suffix "_7" – we keep that for compatibility
MAX_DAYS_BACK = 120

# Hard-coded Odds API key (as you requested)
ODDS_API_KEY = "8e9d506f8573b01023028cef1bf645b5"

# NBA bookmakers use PHX & CHA; Basketball-Reference uses PHO & CHO.
# We map only for API calls, and map back for display.
FETCH_ABBR_MAP = {"PHO": "PHX", "CHO": "CHA"}
REVERSE_MAP = {v: k for k, v in FETCH_ABBR_MAP.items()}

# Full-name → 3-letter codes for The Odds API
FULL_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------------------------------------------------------
# HELPERS: FILES & DATES
# ----------------------------------------------------------------------------------------------------------------------

def find_latest_stats_file(stat_dir: str) -> str:
    """
    Find the most recent nba_games_YYYY-MM-DD.csv file in STAT_DIR.
    """
    pattern = os.path.join(stat_dir, "nba_games_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No nba_games_*.csv files found in {stat_dir}")
    latest = max(files, key=os.path.getctime)
    logging.info("Using stats file: %s", latest)
    return latest


def load_games_df(next_game_dir: str):
    """
    Load games_df_YYYY-MM-DD.csv from NEXT_GAME_DIR.
    If today's file is missing, fallback to the latest available.

    Returns:
        games_df (DataFrame), game_day (date)
    """
    today, today_str, _ = get_current_date()
    file_path = os.path.join(next_game_dir, f"games_df_{today_str}.csv")

    if not os.path.exists(file_path):
        files = sorted(glob.glob(os.path.join(next_game_dir, "games_df_*.csv")))
        if not files:
            raise FileNotFoundError(f"No games_df_*.csv found in {next_game_dir}")
        file_path = files[-1]
        logging.info("Using latest games_df file instead of today: %s", file_path)
    else:
        logging.info("Using games_df file: %s", file_path)

    games_df = pd.read_csv(file_path, index_col=0)
    if games_df.empty:
        raise SystemExit("No upcoming games – season might be over.")

    games_df = games_df.reset_index(drop=True).copy()
    games_df["home_team"] = games_df["home_team"].astype(str).str.strip()
    games_df["away_team"] = games_df["away_team"].astype(str).str.strip()
    games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.date

    unique_days = games_df["game_date"].dropna().unique()
    if len(unique_days) == 1:
        game_day = unique_days[0]
    else:
        game_day = sorted(unique_days)[0]

    logging.info("Upcoming games on %s:\n%s",
                 game_day,
                 games_df.to_string(index=False))

    return games_df, game_day


# ----------------------------------------------------------------------------------------------------------------------
# HELPERS: DATA PREP
# ----------------------------------------------------------------------------------------------------------------------

def add_target(group: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'target' column = next game's 'won' for that team.
    """
    group["target"] = group["won"].shift(-1)
    return group


def preprocess_nba_data(df_path: str) -> pd.DataFrame:
    """
    Load and preprocess nba_games_*.csv with scaling and rolling features
    like in your notebook.
    """
    df = pd.read_csv(df_path, index_col=0)
    df = df.sort_values("date")

    # add target per team
    df = df.groupby("team", group_keys=False).apply(add_target)
    df["target"].fillna(2, inplace=True)  # 2 means "unknown" (future)
    df["target"] = df["target"].astype(int)

    # remove columns with ANY nulls
    nulls = pd.isnull(df).sum()
    null_cols = nulls[nulls > 0].index
    df = df.drop(columns=list(null_cols))

    # columns to NOT scale
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

    # scale the rest
    selected_columns = df.columns[~df.columns.isin(removed_columns)]
    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    # rolling features
    df_rolling = df[list(selected_columns) + ["won", "team", "season"]]

    def find_team_averages(team_df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = team_df.select_dtypes(include=[np.number])
        return numeric_cols.rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean()

    df_rolling = df_rolling.reset_index(drop=True)
    df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

    # rename rolling columns with _7 suffix (to match your existing scripts)
    rolling_cols = {
        col: f"{col}_7" for col in df_rolling.columns
        if col not in ["team", "season"]
    }
    df_rolling.rename(columns=rolling_cols, inplace=True)

    # concat original + rolling
    df = df.reset_index(drop=True)
    df = pd.concat([df, df_rolling], axis=1)

    # drop any remaining NaNs
    df = df.dropna()

    return df, rolling_cols, removed_columns


def shift_col(team: pd.DataFrame, col_name: str) -> pd.Series:
    return team[col_name].shift(-1)


def add_next_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'home_next', 'team_opp_next', 'date_next' per team (shifted).
    """
    df = df.reset_index(drop=True)

    def add_col(col_name: str) -> pd.Series:
        if "team" not in df.columns:
            raise KeyError("The 'team' column is missing in df")
        return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

    df["home_next"] = add_col("home")
    df["team_opp_next"] = add_col("team_opp")
    df["date_next"] = add_col("date")

    return df


def inject_upcoming_games(df: pd.DataFrame, games_df: pd.DataFrame, game_day) -> pd.DataFrame:
    """
    Override 'team_opp_next', 'home_next', 'date_next' for the teams
    that are actually playing on game_day (your mapping loop from the notebook).
    """
    games_df_fixed = games_df.reset_index(drop=True).copy()
    games_df_fixed["home_team"] = games_df_fixed["home_team"].astype(str).str.strip()
    games_df_fixed["away_team"] = games_df_fixed["away_team"].astype(str).str.strip()
    games_df_fixed["game_date"] = pd.to_datetime(games_df_fixed["game_date"]).dt.date

    for _, game in games_df_fixed.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]
        gd = game["game_date"]

        try:
            last_home_idx = df.loc[df["team"] == home_team].iloc[::-1].index[0]
            last_away_idx = df.loc[df["team"] == away_team].iloc[::-1].index[0]
        except IndexError:
            logging.warning("No history for %s or %s in df – skipping this matchup.",
                            home_team, away_team)
            continue

        # home team row
        df.loc[last_home_idx, "team_opp_next"] = away_team
        df.loc[last_home_idx, "home_next"] = 1
        df.loc[last_home_idx, "date_next"] = gd

        # away team row
        df.loc[last_away_idx, "team_opp_next"] = home_team
        df.loc[last_away_idx, "home_next"] = 0
        df.loc[last_away_idx, "date_next"] = gd

        logging.info("Mapped %s (home) vs %s on %s", home_team, away_team, gd)

    return df


# ----------------------------------------------------------------------------------------------------------------------
# HELPERS: ODDS API
# ----------------------------------------------------------------------------------------------------------------------

def translate_to_api(df: pd.DataFrame) -> pd.DataFrame:
    """Convert team codes for API queries (PHO->PHX, CHO->CHA)."""
    return df.replace(FETCH_ABBR_MAP)


def translate_back(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bookmaker abbreviations back (PHX->PHO, CHA->CHO)."""
    return df.replace(REVERSE_MAP)


def get_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def fetch_odds(games_df: pd.DataFrame,
               api_key: str,
               preferred: list | None = None) -> pd.DataFrame:
    """
    Fetch moneyline odds for the given (home_team, away_team) pairs from The Odds API.
    Returns a DataFrame with 'home_team', 'away_team', 'odds 1', 'odds 2' in American format.
    """
    sess = get_session()
    resp = sess.get(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    # Build lookup: (home_abbr, away_abbr) -> (moneyline_home, moneyline_away)
    lookup = {}
    for ev in data:
        home_abbr = FULL_TO_ABBREV.get(ev["home_team"])
        away_abbr = FULL_TO_ABBREV.get(ev["away_team"])
        if not home_abbr or not away_abbr or not ev.get("bookmakers"):
            continue

        bookmakers = ev["bookmakers"]
        bm = None
        if preferred:
            for key in preferred:
                bm = next((b for b in bookmakers if b["key"] == key), None)
                if bm:
                    break
        if bm is None:
            bm = bookmakers[0]

        market = next((m for m in bm["markets"] if m["key"] == "h2h"), None)
        if not market:
            continue

        prices = {}
        for out in market["outcomes"]:
            abbr = FULL_TO_ABBREV.get(out["name"])
            if abbr:
                prices[abbr] = out["price"]

        lookup[(home_abbr, away_abbr)] = (prices.get(home_abbr), prices.get(away_abbr))

    # Build odds DataFrame aligned to games_df
    rows = []
    for _, gm in games_df.iterrows():
        h, a = gm.home_team, gm.away_team
        o1, o2 = lookup.get((h, a), (None, None))
        if o1 is None or o2 is None:
            logging.warning("No odds found for %s vs %s", h, a)
        rows.append({"home_team": h, "away_team": a, "odds 1": o1, "odds 2": o2})

    return pd.DataFrame(rows)


def american_to_decimal(ml):
    """
    Convert American odds to decimal odds.
    """
    if pd.isna(ml):
        return np.nan
    ml = int(ml)
    if ml > 0:
        return ml / 100.0 + 1.0
    else:
        return 100.0 / abs(ml) + 1.0


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Dates and dirs
    today, today_str, today_str_format = get_current_date()
    paths = get_directory_paths()
    stat_dir = paths["STAT_DIR"]
    next_game_dir = paths["NEXT_GAME_DIR"]
    prediction_dir = paths["PREDICTION_DIR"]

    logging.info("Today: %s", today_str_format)
    logging.info("STAT_DIR: %s", stat_dir)
    logging.info("NEXT_GAME_DIR: %s", next_game_dir)
    logging.info("PREDICTION_DIR: %s", prediction_dir)

    # 1) Load historical stats
    stats_file = find_latest_stats_file(stat_dir)
    df, rolling_cols, removed_columns = preprocess_nba_data(stats_file)

    # 2) Add shifted "next" columns
    df = add_next_columns(df)

    # 3) Load upcoming games & inject next-game info into df
    games_df, game_day = load_games_df(next_game_dir)
    df = inject_upcoming_games(df, games_df, game_day)

    # 4) Build "full" matrix merging team vs opponent (your notebook logic)
    #    Use original col names from rolling_cols as keys for the opponent side.
    full = df.merge(
        df[list(rolling_cols.keys()) + ["team_opp_next", "date_next", "team"]],
        left_on=["team", "date_next"],
        right_on=["team_opp_next", "date_next"],
        suffixes=("_x", "_y"),
    )

    # 5) Features & train/pred split
    # add all object columns to removed_columns
    removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
    selected_columns = full.columns[~full.columns.isin(removed_columns)]
    selected_features = selected_columns.unique()

    full_train = full[full["target"] != 2]
    full_pred = full[full["target"] == 2]

    X = full_train[selected_features].values
    y = full_train["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6) LightGBM model (same params as notebook)
    params = {
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

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info("LightGBM accuracy on hold-out: %.2f%%", acc * 100)

    # 7) Predict probabilities for future rows (target == 2)
    if full_pred.empty:
        logging.warning("No future rows (target == 2). Probably no upcoming games in stats yet.")
        return

    full_pred = full_pred.copy()
    full_pred["proba"] = model.predict_proba(full_pred[selected_features])[:, 1]

    # 8) Build team → win_prob lookup and attach to tonight's games
    team_winprob = (
        full_pred[["team_x", "proba"]]
        .drop_duplicates(subset="team_x")
        .set_index("team_x")["proba"]
        .to_dict()
    )

    logging.info("Team win probability lookup (sample): %s",
                 list(team_winprob.items())[:5])

    games_today = games_df.copy()
    games_today["home_team_prob"] = games_today["home_team"].map(team_winprob)
    games_today["away_team_prob"] = games_today["away_team"].map(team_winprob)

    logging.info("Games with model probs:\n%s",
                 games_today[["home_team", "away_team", "home_team_prob", "away_team_prob"]]
                 .to_string(index=False))

    # 9) Fetch odds via The Odds API (hard-coded key)
    api_key = ODDS_API_KEY

    query_df = translate_to_api(games_today[["home_team", "away_team"]].copy())
    odds_df = fetch_odds(query_df, api_key, preferred=["draftkings", "fanduel"])
    odds_df = translate_back(odds_df)

    # 10) Build final predictions DataFrame (same structure as your notebook output)
    home_team_preds = (
        games_today[["home_team", "away_team", "home_team_prob"]]
        .rename(columns={"home_team_prob": "home_team_prob"})
        .assign(result=0, date=game_day)
    )

    home_team_preds = home_team_preds.merge(
        odds_df[["home_team", "away_team", "odds 1", "odds 2"]],
        on=["home_team", "away_team"],
        how="left",
    )

    # Convert American odds to decimal and round
    home_team_preds["odds 1"] = home_team_preds["odds 1"].apply(american_to_decimal)
    home_team_preds["odds 2"] = home_team_preds["odds 2"].apply(american_to_decimal)
    home_team_preds["odds 1"] = home_team_preds["odds 1"].round(2)
    home_team_preds["odds 2"] = home_team_preds["odds 2"].round(2)

    # Print to console for today
    cols = ["home_team", "away_team", "home_team_prob", "odds 1", "odds 2", "result", "date"]
    logging.info("Final predictions for %s:\n%s",
                 game_day,
                 home_team_preds[cols].to_string(index=False))

    # 11) Save CSV in PREDICTION_DIR
    os.makedirs(prediction_dir, exist_ok=True)
    out_name = f"nba_games_predict_{game_day}.csv"
    out_path = os.path.join(prediction_dir, out_name)

    if os.path.exists(out_path):
        logging.info("File already exists: %s (not overwriting)", out_path)
    else:
        home_team_preds[cols].to_csv(out_path, index=False)
        logging.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
