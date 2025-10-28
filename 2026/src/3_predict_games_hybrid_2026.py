# -*- coding: utf-8 -*-
"""
3_predict_games_hybrid_2026.py

This script predicts win probabilities for today's NBA matchups,
merges them with betting odds, estimates value edges, and writes out
a CSV in the standard betting format.

It is a fusion of:
- your working local notebook logic (rolling window, target=won.shift(-1),
  manual override of `team_opp_next` / `home_next` / `date_next` using games_df,
  self-merge to create matchups, LightGBM training, predict_proba, odds fetch)
AND
- the "2026" repo style (paths, odds helpers, Kelly downstream compatibility).

Main outputs:
1. LightGBM model accuracy / feature importances (for sanity check)
2. CSV: nba_games_predict_<YYYY-MM-DD>.csv
   columns:
     home_team, away_team, home_team_prob, result, odds 1, odds 2, date
"""

import os
import glob
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

TEAM_ALIAS_FOR_ODDS = {
    "PHO": "PHX",
    "PHX": "PHX",
    "BKN": "BRK",
    "BRK": "BRK",
    "CHA": "CHO",   # sportsbook "CHA" -> our "CHO"
    "CHO": "CHO",
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
    # rest already match (BOS, LAL, etc.)
}

def normalize_code_for_odds(abbr: str) -> str:
    """Return our canonical team code (PHO->PHX, CHA->CHO, BKN->BRK, etc.)."""
    if not isinstance(abbr, str):
        return abbr
    return TEAM_ALIAS_FOR_ODDS.get(abbr.upper(), abbr.upper())


# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

ROLLING_WINDOW_SIZE = 9
CURRENT_SEASON = 2025   # mostly for reference/logging
API_KEY = "8e9d506f8573b01023028cef1bf645b5"

# you already have these folders in the 2026 repo structure;
# adjust if your paths differ
def get_directory_paths() -> Dict[str, str]:
    """
    Return a dictionary of important directories.
    You should keep these paths consistent with your repo.
    Works on both Windows and Linux (GitHub Actions).
    """
    base_repo = os.getcwd()
    return {
        "STAT_DIR": os.path.join(base_repo, "2026", "output", "Gathering_Data", "Whole_Statistic"),
        "NEXT_GAME_DIR": os.path.join(base_repo, "2026", "output", "Gathering_Data", "Next_Game"),
        "PREDICTION_DIR": os.path.join(base_repo, "2026", "output", "LightGBM"),
    }

def get_current_date(offset_days: int = 0) -> Tuple[datetime, str, str]:
    """
    Returns today's date in different representations:
    - dt (datetime)
    - pretty "YYYY-MM-DD"
    - same "YYYY-MM-DD" (kept for interface compatibility)
    """
    dt = datetime.now() - timedelta(days=offset_days)
    ds = dt.strftime("%Y-%m-%d")
    return dt, ds, ds

def get_latest_file(directory: str, prefix: str, ext: str) -> str:
    """
    Find latest file in directory that matches prefix + *.ext
    """
    pattern = os.path.join(directory, f"{prefix}*{ext}")
    files = glob.glob(pattern)
    if not files:
        return ""
    return max(files, key=os.path.getctime)

# Mapping between sportsbook names and our abbreviations
TEAM_ALIAS_FOR_ODDS = {
    "PHO": "PHX", "PHX": "PHX",
    "BKN": "BRK", "BRK": "BRK",
    "CHA": "CHO", "CHO": "CHO",  # internal style often "CHO"
    "GS":  "GSW", "GSW": "GSW",
    "NO":  "NOP", "NOP": "NOP",
    "NY":  "NYK", "NYK": "NYK",
    "SA":  "SAS", "SAS": "SAS",
    "UTAH":"UTA", "UTA": "UTA",
    "OKL": "OKC", "OKC": "OKC",
}

def normalize_code_for_odds(abbr: str) -> str:
    if not isinstance(abbr, str):
        return abbr
    return TEAM_ALIAS_FOR_ODDS.get(abbr.upper(), abbr.upper())

# full-name → our abbrev (used by odds fetch)
FULL_TO_ABBREV = {
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
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────────────────
# STEP 1. LOAD TODAY'S DATA
# ─────────────────────────────────────────────────────────

def load_games_df(paths: Dict[str, str], today_str_format: str) -> pd.DataFrame:
    """
    Load today's games_df_<date>.csv.
    If not found, fall back to the most recent games_df_*.csv.
    Must contain columns: home_team, away_team, game_date
    """
    next_game_dir = paths["NEXT_GAME_DIR"]
    direct_path = os.path.join(next_game_dir, f"games_df_{today_str_format}.csv")

    if os.path.exists(direct_path):
        file_path = direct_path
    else:
        # fallback to most recent
        file_path = get_latest_file(next_game_dir, prefix="games_df_", ext=".csv")
        if not file_path:
            raise FileNotFoundError(
                f"No games_df_*.csv found in {next_game_dir}"
            )
        logging.info(
            f"games_df for {today_str_format} not found. Falling back to {file_path}"
        )

    games_df = pd.read_csv(file_path)
    # handle idx col if present
    if "Unnamed: 0" in games_df.columns:
        games_df = games_df.drop(columns=["Unnamed: 0"])

    if games_df.empty:
        logging.warning("games_df is empty (season might be over).")

    logging.info(
        f"Loaded game schedule from {file_path} with {len(games_df)} games"
    )
    return games_df


def load_stats_df(paths: Dict[str, str], today_str_format: str) -> pd.DataFrame:
    """
    Load nba_games_<date>.csv from STAT_DIR.
    If today's file is missing, use the most recent nba_games_*.csv.
    """
    stat_dir = paths["STAT_DIR"]
    direct_path = os.path.join(stat_dir, f"nba_games_{today_str_format}.csv")

    if os.path.exists(direct_path):
        df_path = direct_path
    else:
        logging.info(
            f"Stats file for {today_str_format} not found. Searching latest in {stat_dir}..."
        )
        df_path = get_latest_file(stat_dir, prefix="nba_games_", ext=".csv")
        if not df_path:
            raise FileNotFoundError(
                f"No nba_games_*.csv files found in {stat_dir}"
            )
        logging.info(f"Using latest stats file: {df_path}")

    df = pd.read_csv(df_path)
    # Sometimes index col sneaks in
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    logging.info(f"Loaded stats with {len(df)} rows and {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────
# STEP 2. PREPROCESS (TARGET, SCALING, ROLLING)
# ─────────────────────────────────────────────────────────

def add_target_per_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, target = 'won' of NEXT row
    Then fill last row (future) as 2.
    """
    def add_target(group):
        group = group.sort_values("date")
        group["target"] = group["won"].shift(-1)
        return group

    df = df.sort_values("date")
    df = df.groupby("team", group_keys=False).apply(add_target)
    df["target"] = df["target"].fillna(2).astype(int)
    return df


def scale_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    MinMax-scale all numeric stat columns except a few meta columns.
    Returns the scaled df and the list of scaled columns.
    """
    removed_cols_for_scaling = ["season", "date", "won", "target", "team", "team_opp"]
    to_scale = df.columns[~df.columns.isin(removed_cols_for_scaling)]

    scaler = MinMaxScaler()
    df[to_scale] = scaler.fit_transform(df[to_scale])
    return df, list(to_scale)



def rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages for numeric columns grouped by team+season,
    BUT EXCLUDE 'target' from rolling features so we don't create target_7 leaks.
    """
    def team_roll(g: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = g.select_dtypes(include=[np.number]).copy()
        # do not roll future label info
        numeric_cols = numeric_cols.drop(columns=["target"], errors="ignore")
        rolled = numeric_cols.rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean()
        return rolled

    df_numeric = df.groupby(["team", "season"], group_keys=False).apply(team_roll)

    rename_map = {col: f"{col}_7" for col in df_numeric.columns}
    df_numeric = df_numeric.rename(columns=rename_map)

    df = pd.concat(
        [df.reset_index(drop=True), df_numeric.reset_index(drop=True)],
        axis=1
    )
    return df




# ─────────────────────────────────────────────────────────
# STEP 3. ADD NEXT-GAME INFO
# ─────────────────────────────────────────────────────────

def add_next_game_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, attach info about that team's NEXT game using shift(-1):
        home_next, team_opp_next, date_next
    """
    def shift_col(team_df: pd.DataFrame, col_name: str) -> pd.Series:
        return team_df[col_name].shift(-1)

    def add_cols(team_df: pd.DataFrame) -> pd.DataFrame:
        team_df = team_df.copy()
        team_df["home_next"] = shift_col(team_df, "home")
        team_df["team_opp_next"] = shift_col(team_df, "team_opp")
        team_df["date_next"] = shift_col(team_df, "date")
        return team_df

    df2 = (
        df
        .groupby("team", group_keys=False)
        .apply(add_cols)
    )
    return df2


def override_next_game_with_schedule(df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook logic:
    For each (home_team, away_team, game_date) in games_df,
    find the LATEST row of each team and overwrite that row's
    team_opp_next, home_next, date_next
    with today's matchup info.

    This forces the "next game" for that last observation to be TODAY's game,
    so our merge will later align all upcoming games.
    """
    if games_df.empty:
        logging.warning("No upcoming games in schedule; skip override_next_game_with_schedule.")
        return df

    df = df.copy()

    # assure required columns
    needed_cols = ["team_opp_next", "home_next", "date_next"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    for idx, game in games_df.iterrows():
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        game_day  = game.get("game_date")

        if pd.isna(home_team) or pd.isna(away_team) or pd.isna(game_day):
            logging.warning(f"Skipping row {idx} in games_df due to missing values.")
            continue

        # last row for home team
        home_mask = df["team"] == home_team
        if home_mask.any():
            last_home_idx = df[home_mask].index.max()
            df.loc[last_home_idx, "team_opp_next"] = away_team
            df.loc[last_home_idx, "home_next"] = 1
            df.loc[last_home_idx, "date_next"] = game_day

        else:
            logging.warning(f"Could not find recent row for home team {home_team}")

        # last row for away team
        away_mask = df["team"] == away_team
        if away_mask.any():
            last_away_idx = df[away_mask].index.max()
            df.loc[last_away_idx, "team_opp_next"] = home_team
            df.loc[last_away_idx, "home_next"] = 0
            df.loc[last_away_idx, "date_next"] = game_day
        else:
            logging.warning(f"Could not find recent row for away team {away_team}")

    return df


# ─────────────────────────────────────────────────────────
# STEP 4. BUILD MATCHUPS ("full")
# ─────────────────────────────────────────────────────────

def build_matchup_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-create the notebook's self-merge, but safely.

    Concept:
    - left side: each team's last known stats row, with columns including:
        team, team_opp_next, date_next, home_next, target, etc.
    - right side: opponent snapshot.
      We take a subset of columns (rolling features etc.) and prefix them
      as "_right", so we don't clash.

    We merge on:
        left.team == right.team_opp_next
        left.date_next == right.date_next

    After merge we rename:
        left.team          -> team_x (this is the focal team)
        right.team         -> team_y (opponent)
        left.home_next     -> home_next
        left.target        -> target
        left.date_next     -> date_next

    Then:
      full_train = target != 2
      full_pred  = target == 2
    """
    df_left = df.copy()

    # build df_right with suffix _right for numeric/stat columns
    banned_right_cols = {
        "team", "team_opp", "team_opp_next", "home", "home_next",
        "date", "date_next", "target", "won", "season"
    }

    keep_for_right = [c for c in df.columns if c not in banned_right_cols]
    df_right = df[keep_for_right + ["team_opp_next", "date_next", "team"]].copy()

    # rename df_right's feature columns with _right suffix,
    # but keep merge keys + team plain to map later
    rename_map = {}
    for c in keep_for_right:
        rename_map[c] = f"{c}_right"

    df_right = df_right.rename(columns=rename_map)

    full = df_left.merge(
        df_right,
        left_on=["team", "date_next"],
        right_on=["team_opp_next", "date_next"],
        how="inner"
    )

    # rename columns to stable names
    full = full.rename(columns={
        "team_x": "team",  # pandas may have created team_x / team_y; handle carefully below
    }) if "team_x" in full.columns else full

    # We want focal team as team_x and opp as team_y
    # after merge, df_left columns kept their original names
    # df_right has "team" (the opponent's team) but not suffixed
    if "team_x" in full.columns and "team_y" in full.columns:
        # if pandas auto-created team_x/team_y because of collision:
        team_x_col = "team_x"
        team_y_col = "team_y"
    else:
        # if pandas DIDN'T create team_x/team_y because we renamed df_right cols:
        team_x_col = "team"
        team_y_col = "team_y" if "team_y" in full.columns else "team_right"
        if team_y_col not in full.columns and "team" in df_right.columns:
            # after merge, df_right "team" might appear as just "team_y"
            # but to be safe, detect it
            candidates = [c for c in full.columns if c.endswith("_y")]
            if len(candidates) == 1:
                team_y_col = candidates[0]

    # final standard names
    if team_x_col not in full.columns:
        raise RuntimeError("Could not identify 'team_x' in merged data.")
    if team_y_col not in full.columns:
        raise RuntimeError("Could not identify 'team_y' (opponent) in merged data.")

    full = full.rename(columns={
        team_x_col: "team_x",
        team_y_col: "team_y",
    })

    return full


def split_train_pred(full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split full into training rows (target != 2) and prediction rows (target == 2).
    """
    if "target" not in full.columns:
        raise RuntimeError("Expected 'target' column in full after merge.")
    full_train = full[full["target"] != 2].copy()
    full_pred  = full[full["target"] == 2].copy()
    return full_train, full_pred


# ─────────────────────────────────────────────────────────
# STEP 5. TRAIN MODEL
# ─────────────────────────────────────────────────────────

def build_feature_list(full: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Select model features, making sure we do NOT leak outcome info.
    We drop:
      - any column explicitly known to be metadata
      - any column containing 'target' or 'won'
      - any non-numeric / object columns
    """
    banned_explicit = {
        "team_x",
        "team_y",
        "target",
        "home_next",
        "date_next",
        "season",
        "won",
        "team",
        "team_opp",
        "team_opp_next",
        "date",
        "home",
    }

    feature_cols: List[str] = []
    for col in full.columns:
        if col in banned_explicit:
            continue

        # remove anything derived from target/won to kill leakage
        if "target" in col.lower():
            continue
        if "won" in col.lower():
            continue

        # must be numeric
        if full[col].dtype == object:
            continue
        if not pd.api.types.is_numeric_dtype(full[col]):
            continue

        feature_cols.append(col)

    return feature_cols, sorted(list(banned_explicit))


    feature_cols: List[str] = []
    for col in full.columns:
        if col in banned_cols:
            continue
        # skip object/datetime
        if full[col].dtype == object:
            continue
        # skip obviously non-numeric
        if not pd.api.types.is_numeric_dtype(full[col]):
            continue
        feature_cols.append(col)

    return feature_cols, banned_cols


def train_lightgbm(full_train: pd.DataFrame,
                   feature_cols: List[str]) -> Tuple[lgb.LGBMClassifier, float]:
    """
    Train LightGBM with fixed params from notebook.
    Return model and accuracy on holdout.
    """
    if len(full_train) < 10:
        raise ValueError("Not enough training samples (<10).")

    X = full_train[feature_cols].values
    y = full_train["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

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

    logging.info(f"LightGBM model trained. Accuracy: {acc:.2%}")

    # feature importances log
    importances = model.feature_importances_
    pairs = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    logging.info("Top feature importances (first 10):")
    for i, (name, score) in enumerate(pairs, start=1):
        logging.info(f"  {i}. {name}: {score}")

    return model, acc


# ─────────────────────────────────────────────────────────
# STEP 6. PREDICT NEXT GAMES
# ─────────────────────────────────────────────────────────

def predict_upcoming(full_pred: pd.DataFrame,
                     model: lgb.LGBMClassifier,
                     feature_cols: List[str],
                     games_df: pd.DataFrame) -> pd.DataFrame:
    """
    From full_pred (target==2 rows), get model P(home team wins).

    We interpret team_x as "the team whose 'home_next' tells us if it's home".
    We only keep rows where home_next == 1, to ensure team_x is the home team.

    Then we align them with today's schedule order.
    Result columns:
        home_team, away_team, home_team_prob, result, date
    """
    if full_pred.empty:
        logging.warning("No rows for prediction (full_pred is empty).")
        return pd.DataFrame()

    X_pred = full_pred[feature_cols].values
    probs = model.predict_proba(X_pred)[:, 1]

    pred_df = full_pred.copy()
    pred_df["proba"] = probs

    # keep only rows where team_x is home in its NEXT game
    pred_df = pred_df[pred_df["home_next"] == 1].copy()

    # build nice output
    out_rows = []
    for _, row in pred_df.iterrows():
        out_rows.append({
            "home_team": row["team_x"],
            "away_team": row["team_y"],
            "home_team_prob": float(row["proba"]),
            "result": 0,
            "date": row["date_next"],
        })

    preds = pd.DataFrame(out_rows)

    if preds.empty:
        logging.warning("No aligned rows where team_x is home in its next game.")

    # optional: restrict to only teams present in games_df (safety)
    if not games_df.empty and "home_team" in games_df.columns and "away_team" in games_df.columns:
        pairs = set(zip(games_df["home_team"], games_df["away_team"]))
        preds = preds[preds.apply(
            lambda r: (r["home_team"], r["away_team"]) in pairs,
            axis=1
        )].copy()

    if preds.empty:
        logging.warning("After schedule alignment, no predictions remain for today's games.")

    return preds


# ─────────────────────────────────────────────────────────
# STEP 7. ODDS FETCH + MERGE
# ─────────────────────────────────────────────────────────

def get_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def fetch_odds(games_df: pd.DataFrame, api_key: str, preferred: List[str] = None) -> pd.DataFrame:
    """
    Fetch H2H odds and return a DataFrame with columns:
    home_team | away_team | odds 1 | odds 2
    where team codes are normalized to our internal canon
    (PHX, BRK, CHO, etc.).
    """

    # sportsbook full name -> "raw" abbrev first
    full_to_abbrev = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",          # force BRK not BKN
        "Charlotte Hornets": "CHO",      # force CHO not CHA
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
        "Phoenix Suns": "PHX",            # force PHX not PHO
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }

    session = get_session()
    response = session.get(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    lookup = {}  # (HOME, AWAY) -> (ml_home, ml_away) using canonical codes

    for event in data:
        home_full = event.get("home_team")
        away_full = event.get("away_team")

        raw_home = full_to_abbrev.get(home_full)
        raw_away = full_to_abbrev.get(away_full)
        if not raw_home or not raw_away:
            continue

        # normalize to canon (PHO->PHX etc.)
        home_code = normalize_code_for_odds(raw_home)
        away_code = normalize_code_for_odds(raw_away)

        bookmakers = event.get("bookmakers", [])
        chosen = None
        if preferred:
            for bkey in preferred:
                chosen = next((b for b in bookmakers if b.get("key") == bkey), None)
                if chosen:
                    break
        if not chosen and bookmakers:
            chosen = bookmakers[0]
        if not chosen:
            continue

        market = next(
            (m for m in chosen.get("markets", []) if m.get("key") == "h2h"),
            None
        )
        if not market:
            continue

        prices_by_code = {}
        for outcome in market.get("outcomes", []):
            full_name = outcome.get("name")               # e.g. "Phoenix Suns"
            raw_abbr  = full_to_abbrev.get(full_name)     # -> "PHX"
            if raw_abbr:
                canon_abbr = normalize_code_for_odds(raw_abbr)
                prices_by_code[canon_abbr] = outcome.get("price")

        ml_home = prices_by_code.get(home_code)
        ml_away = prices_by_code.get(away_code)

        lookup[(home_code, away_code)] = (ml_home, ml_away)

    # now align to our schedule games_df
    odds_rows = []
    for _, gm in games_df.iterrows():
        # normalize schedule teams to canon too
        h = normalize_code_for_odds(gm["home_team"])
        a = normalize_code_for_odds(gm["away_team"])

        o1, o2 = lookup.get((h, a), (None, None))
        if o1 is None or o2 is None:
            logging.warning(f"No odds found for {h} vs {a}")
        odds_rows.append(
            {"home_team": h, "away_team": a, "odds 1": o1, "odds 2": o2}
        )

    return pd.DataFrame(odds_rows)


def american_to_decimal(ml: float) -> float:
    """
    Convert American moneyline to decimal odds.
    """
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        return round(ml / 100.0 + 1.0, 2)
    else:
        return round(100.0 / abs(ml) + 1.0, 2)


def implied_prob(ml: float) -> float:
    """
    Convert American moneyline to implied probability.
    """
    if ml is None or (isinstance(ml, float) and np.isnan(ml)):
        return np.nan
    ml = float(ml)
    if ml < 0:
        # favorite
        return abs(ml) / (abs(ml) + 100.0)
    else:
        # underdog
        return 100.0 / (ml + 100.0)
        
def impute_prob(moneyline):
    """
    Convert American moneyline (e.g. -150, +200) to implied win probability in [0,1].
    If moneyline is NaN/None -> returns None.
    """
    if moneyline is None or (isinstance(moneyline, float) and np.isnan(moneyline)):
        return None
    ml = int(moneyline)
    # For negative odds (favorite): prob = |ml| / (|ml| + 100)
    # For positive odds (underdog): prob = 100 / (ml + 100)
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def merge_predictions_with_odds(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Join model preds with odds and compute implied probs + model edge.
    Both DataFrames are normalized to canonical team codes first.
    """
    preds = preds.copy()
    odds = odds.copy()

    # normalize again just to be paranoid
    preds["home_team"] = preds["home_team"].apply(normalize_code_for_odds)
    preds["away_team"] = preds["away_team"].apply(normalize_code_for_odds)
    odds["home_team"]  = odds["home_team"].apply(normalize_code_for_odds)
    odds["away_team"]  = odds["away_team"].apply(normalize_code_for_odds)

    df = preds.merge(odds, on=["home_team", "away_team"], how="left")

    for col in ["odds 1", "odds 2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["imp_prob_home"] = df["odds 1"].apply(impute_prob)
    df["imp_prob_away"] = df["odds 2"].apply(impute_prob)

    df["value_home"] = np.where(
        df["imp_prob_home"].notna(),
        df["home_team_prob"] - df["imp_prob_home"],
        np.nan
    )
    df["value_away"] = np.where(
        df["imp_prob_away"].notna(),
        (1.0 - df["home_team_prob"]) - df["imp_prob_away"],
        np.nan
    )

    return df


def build_home_team_preds_csv(preds: pd.DataFrame,
                              odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final table to save:
      home_team, away_team, home_team_prob, result, odds 1, odds 2, date
    Odds converted to DECIMAL for bankroll workflow.
    """
    if preds.empty:
        logging.warning("build_home_team_preds_csv: preds empty.")
        return pd.DataFrame(columns=[
            "home_team", "away_team",
            "home_team_prob", "result",
            "odds 1", "odds 2",
            "date"
        ])

    out = preds.merge(
        odds_df[["home_team", "away_team", "odds 1", "odds 2"]],
        on=["home_team", "away_team"],
        how="left"
    ).copy()

    out["odds 1"] = out["odds 1"].apply(american_to_decimal)
    out["odds 2"] = out["odds 2"].apply(american_to_decimal)

    out["result"] = 0  # unknown yet
    # 'date' is already present in preds

    # ensure column order
    out = out[[
        "home_team",
        "away_team",
        "home_team_prob",
        "result",
        "odds 1",
        "odds 2",
        "date",
    ]]

    return out

FINAL_EXPORT_CODES = {
    # what model/odds use  -> what you want to SEE in the CSV
    "PHX": "PHO",
    "CHO": "CHO",  # stays same
    "BRK": "BRK",  # stays same (но можешь тут поменять на BKN если захочешь bkn)
    # если надо, можешь добавить другие маппинги
}

def prettify_team_codes_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert canonical internal codes (PHX, CHO, BRK, ...) into the
    display codes you actually want in the saved CSV (PHO, CHO, ...).
    Only touches the final exported table.
    """
    out = df.copy()

    for col in ["home_team", "away_team"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: FINAL_EXPORT_CODES.get(str(x), str(x))
            )

    return out

# ─────────────────────────────────────────────────────────
# STEP 8. SAVE
# ─────────────────────────────────────────────────────────

def save_predictions_csv(df_to_save: pd.DataFrame,
                          paths: Dict[str, str],
                          today_str_format: str) -> str:
    """
    Save the final predictions in the fixed betting format.
    """
    pred_dir = paths["PREDICTION_DIR"]
    os.makedirs(pred_dir, exist_ok=True)

    filename = f"nba_games_predict_{today_str_format}.csv"
    filepath = os.path.join(pred_dir, filename)

    if os.path.exists(filepath):
        logging.info(f"Prediction file already exists: {filepath}")
    else:
        df_to_save.to_csv(filepath, index=False)
        logging.info(f"Saved predictions to {filepath}")

    logging.info(f"Prediction CSV saved to {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> str:
    # 1. Paths + date
    paths = get_directory_paths()
    today_dt, today_pretty, today_str_format = get_current_date(0)

    # 2. Load inputs
    games_df = load_games_df(paths, today_str_format)
    stats_df = load_stats_df(paths, today_str_format)

    # 3. Preprocess base stats
    df = stats_df.copy()
    df = add_target_per_team(df)

    # remove columns that are completely null to match notebook
    nulls = df.isnull().sum()
    drop_cols = nulls[nulls > 0].index.tolist()
    # BUT we don't want to drop columns we actually need (team,date,...).
    # In notebook you dropped all null columns before shift etc,
    # but by now df already has all needed cols.
    # We'll be conservative: drop columns that are entirely NaN AND not key cols:
    key_keep = {"team", "date", "won", "home", "team_opp", "season"}
    truly_all_nan = [c for c in drop_cols if c not in key_keep and df[c].isna().all()]
    if truly_all_nan:
        df = df.drop(columns=truly_all_nan)

    df, scaled_cols = scale_numeric(df)
    df = rolling_averages(df)
    df = add_next_game_columns(df)
    
    df["team"] = df["team"].apply(normalize_code_for_odds)
    df["team_opp"] = df["team_opp"].apply(normalize_code_for_odds)
    games_df["home_team"] = games_df["home_team"].apply(normalize_code_for_odds)
    games_df["away_team"] = games_df["away_team"].apply(normalize_code_for_odds)

    # 4. Inject today's schedule into df (override last rows)
    #    so that each team's NEXT game == today's game
    if not games_df.empty:
        df = override_next_game_with_schedule(df, games_df)
    else:
        logging.warning("No games_df rows; season may be over.")

    # 5. Build matchup table
    full = build_matchup_full(df)
    
        # 6. Split train vs predict
    full_train, full_pred = split_train_pred(full)

    logging.info(
        f"Training data contains {len(full_train)} rows and {len(full_train.columns)} columns"
    )
    if full_pred.empty:
        logging.warning("No prediction rows with target==2 found in 'full'.")

    # 7. Build feature list + train LightGBM
    feature_cols, banned_cols = build_feature_list(full_train)
    model, acc = train_lightgbm(full_train, feature_cols)

    # 8. Predict probabilities for upcoming games
    preds_raw = predict_upcoming(full_pred, model, feature_cols, games_df)

    if preds_raw.empty:
        logging.warning("No aligned predictions for today. Will still try odds/save in empty mode.")

    # 9. Fetch odds
    odds_df = fetch_odds(games_df, API_KEY, preferred=["draftkings", "fanduel"])

    # 10. Merge predictions with odds and compute value edges (for logging / diagnostics)
    if preds_raw.empty:
        merged_value = pd.DataFrame()
    else:
        merged_value = merge_predictions_with_odds(preds_raw, odds_df)
        # (не сохраняем merged_value, это просто для отладки/анализа)

    # 11. Build final CSV frame (decimal odds etc.)
    csv_frame = build_home_team_preds_csv(preds_raw, odds_df)

    # 11.1 apply pretty code mapping (PHX -> PHO, etc.) ONLY for output
    csv_frame = prettify_team_codes_for_output(csv_frame)

    # 12. Save (even if empty, this keeps your daily file pipeline consistent)
    output_path = save_predictions_csv(csv_frame, paths, today_str_format)

    return output_path



    return output_path


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("An unexpected error occurred during prediction.")
    finally:
        try:
            input("Prediction complete. Press Enter to close this window...")
        except EOFError:
            pass
