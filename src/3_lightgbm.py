#!/usr/bin/env python
# coding: utf-8

#########################################################################################################################
# CALCULATE PREDICTION FOR NEXT GAME DAY
#
# Script 3 of 5
# This script calculates game predictions for the next NBA game day using historical data,
# rolling averages, and LightGBM, then outputs results with probabilities and odds.
# Ensure `src/2_get_data_next_game_day.py` has been run to produce the games_df CSV.
#########################################################################################################################

import os
import glob
import pandas as pd
import numpy as np
import requests
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Import shared utilities
from nba_utils import (
    CURRENT_SEASON,
    ROLLING_WINDOW_SIZE,
    get_current_date,
    get_directory_paths,
    get_latest_file,
    preprocess_nba_data,
    calculate_rolling_averages,
    add_next_game_columns,
    impute_prob,
    am_to_dec
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get current date and directory paths
today, today_str, today_str_format = get_current_date()
paths = get_directory_paths()

# Paths
BASE_DIR      = paths['BASE_DIR']
DATA_DIR      = paths['DATA_DIR']
STAT_DIR      = paths['STAT_DIR']
target_folder = paths['NEXT_GAME_DIR']
directory_path = paths['PREDICTION_DIR']

df_path_stat = os.path.join(STAT_DIR, f"nba_games_{today_str_format}.csv")

# Get latest games file
games_df_path = get_latest_file(target_folder, prefix="games_df_", ext=".csv")
if not games_df_path:
    raise FileNotFoundError(f"No games_df_*.csv found in {target_folder}")

# ──────────────────────────────────────────────────────────────────────────
# LOAD NEXT-GAME LINEUP
# ──────────────────────────────────────────────────────────────────────────
# Define directory and date format
# Check if file exists
file_path_today_games = f"{target_folder}/games_df_{today_str_format}.csv"
if not os.path.exists(file_path_today_games):
    # List files and pick the latest one
    files = sorted(glob.glob(f"{target_folder}/games_df_*.csv"))
    if files:
        file_path_today_games = files[-1]  # Use the latest available file
        print(f"Using the latest file: {file_path_today_games}")
    else:
        print("No files found in the directory.")
        exit()

# Proceed to read the file
games_df = pd.read_csv(file_path_today_games, index_col=0)
print(games_df.head(60).to_string(index=False))

# Proceed with loading the data
df = pd.read_csv(df_path_stat, index_col=0)
print(df)  # Display a portion of the data

# Preprocess the NBA data (uses the function from nba_utils)
df = preprocess_nba_data(df_path_stat)

# Columns to be excluded from scaling
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

# Selecting columns that are not in the 'removed_columns' list
selected_columns = df.columns[~df.columns.isin(removed_columns)]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the selected columns and update the DataFrame
df[selected_columns] = scaler.fit_transform(df[selected_columns])

# Display home win percentage
print(df.groupby(["home"]).apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0]))

####################################################################################################
# CALCULATE THE AVERAGE FOR THE PREVIOUS SEASONS WITH THE ROLLING WINDOW FOR LEARNING THE MODEL #
####################################################################################################

# Filter out the games from the current season
df_rolling = df[list(selected_columns) + ["won", "team", "season"]]

# Calculate rolling averages using the function from nba_utils
df_rolling = calculate_rolling_averages(df_rolling, ROLLING_WINDOW_SIZE)

# Rename the rolling columns with _7 suffix
rolling_cols = {col: f"{col}_7" for col in df_rolling.columns if col not in ['team', 'season']}

# Rename the columns
df_rolling.rename(columns=rolling_cols, inplace=True)

# Combine original data with rolling averages
df = df.reset_index(drop=True)
df_rolling = df_rolling.reset_index(drop=True)
df = pd.concat([df, df_rolling], axis=1)

# Drop any rows with NaN values
df = df.dropna()

print(df)

# Display rows where target is 2
target_2_rows = df[df['target'] == 2]['target']
print(target_2_rows)

# Add next game columns using the function from nba_utils
df = add_next_game_columns(df)

# Display rows where target is 2
target_2_rows = df[df['target'] == 2]['target']
print(target_2_rows)

# Update next game information from games_df
for _, game in games_df.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']
    game_day = game['game_date']

    print(home_team)
    print(away_team)
    print(game_day)

    last_home_team_index = df.loc[df['team'] == home_team].iloc[::-1].index[0]
    df.loc[last_home_team_index, 'team_opp_next'] = away_team
    df.loc[last_home_team_index, 'home_next'] = 1
    df.loc[last_home_team_index, 'date_next'] = game_day

    last_away_team_index = df.loc[df['team'] == away_team].iloc[::-1].index[0]
    df.loc[last_away_team_index, 'team_opp_next'] = home_team
    df.loc[last_away_team_index, 'home_next'] = 0
    df.loc[last_away_team_index, 'date_next'] = game_day

# Merging DataFrames
# Convert rolling_cols dictionary keys to a list and add other columns
full = df.merge(df[list(rolling_cols.keys()) + ["team_opp_next", "date_next", "team"]],
                left_on=["team", "date_next"],
                right_on=["team_opp_next", "date_next"])

# Display basic info and first few rows of the merged DataFrame
print("Full DataFrame Info:")
print(full.info())
print("\nFirst few rows of the merged DataFrame:")
print(full.head())

# Print number of rows in the merged DataFrame
num_rows = full.shape[0]
print(f"Number of rows in 'full' DataFrame: {num_rows}")

# Extract and print rows with target == 2
target_2_rows = full[full['target'] == 2]['target']
print("\nRows where 'target' == 2:")
print(target_2_rows)

# Filter for entries on game day
mask = full['date_next'] == game_day
filtered_df = full.loc[mask, ['team_x', 'team_opp_next_x', 'team_y', 'team_opp_next_y', 'date_next', 'home_next']]
print(filtered_df)

# Remove object columns
removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns

# Select features
selected_columns = full.columns[~full.columns.isin(removed_columns)]
selected_features = selected_columns.unique()
print(selected_features)

# Prepare training and prediction data
full_train = full[full["target"] != 2]
full_pred = full[full["target"] == 2]
print(full_pred)

X = full_train[selected_features].values
y = full_train["target"].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'num_leaves': [10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 5, 10]
}

# Create a LightGBM classifier
base_estimator = lgb.LGBMClassifier(objective='binary',
                                     metric='auc',
                                     boosting_type='gbdt',
                                     verbosity=-1,
                                     random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=base_estimator,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

# Note: Grid search commented out to save time, using predetermined best parameters
# grid_search.fit(X_train, y_train)
# print("Best parameters found:", grid_search.best_params_)

# Best parameters found: {'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 5, 'num_leaves': 10}
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 10,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 10,
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'max_depth': 7,
    'min_child_weight': 5
}

model = lgb.LGBMClassifier(**params)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Calculate feature importances
importances = model.feature_importances_
feat_importances = dict(zip(selected_columns, importances))
sorted_feat_importances = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)

# Print the sorted feature importances
for feature, importance in sorted_feat_importances[:30]:
    print("{}: {}".format(feature, importance))

# Predict on new data
full_pred["proba"] = model.predict_proba(full_pred[selected_features])[:,1]
print(full_pred["proba"])

# Filter for home teams
home_teams_prob = list(games_df['home_team'])
away_teams_prob = list(games_df['away_team'])

# Filter the rows where team_x is a home team
full_pred_prob = full_pred['team_x'].isin(home_teams_prob)
print(full_pred[full_pred_prob]['proba'])

# Get team information
team_x = full_pred.loc[full_pred_prob, 'team_x']
team_y = full_pred.loc[full_pred_prob, 'team_y']
team_pairs = pd.concat([team_x, team_y], axis=1)
print(team_pairs)

# Helper function for API requests with retries
def get_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# NBA team name mappings
full_to_abbrev = {
    "Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BRK",
    "Charlotte Hornets":"CHA","Chicago Bulls":"CHI","Cleveland Cavaliers":"CLE",
    "Dallas Mavericks":"DAL","Denver Nuggets":"DEN","Detroit Pistons":"DET",
    "Golden State Warriors":"GSW","Houston Rockets":"HOU","Indiana Pacers":"IND",
    "LA Clippers":"LAC","Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL",
    "Memphis Grizzlies":"MEM","Miami Heat":"MIA","Milwaukee Bucks":"MIL",
    "Minnesota Timberwolves":"MIN","New Orleans Pelicans":"NOP",
    "New York Knicks":"NYK","Oklahoma City Thunder":"OKC","Orlando Magic":"ORL",
    "Philadelphia 76ers":"PHI","Phoenix Suns":"PHX","Portland Trail Blazers":"POR",
    "Sacramento Kings":"SAC","San Antonio Spurs":"SAS","Toronto Raptors":"TOR",
    "Utah Jazz":"UTA","Washington Wizards":"WAS"
}

def fetch_odds(games_df: pd.DataFrame, api_key: str,
               preferred: list=None) -> pd.DataFrame:
    """
    Fetch odds data from the-odds-api.com for NBA games.

    Args:
        games_df (DataFrame): DataFrame with game information
        api_key (str): API key for the-odds-api.com
        preferred (list): List of preferred bookmakers

    Returns:
        DataFrame: DataFrame with odds information
    """
    sess = get_session()
    r = sess.get(
      "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
      params={"apiKey":api_key,"regions":"us","markets":"h2h","oddsFormat":"american"},
      timeout=10
    )
    r.raise_for_status()
    data = r.json()

    # build lookup: (home,away) → (priceH,priceA)
    lookup = {}
    for ev in data:
        h_abbr = full_to_abbrev.get(ev["home_team"])
        a_abbr = full_to_abbrev.get(ev["away_team"])
        if not h_abbr or not a_abbr or not ev.get("bookmakers"):
            continue

        # pick bookmaker
        bms = ev["bookmakers"]
        bm = None
        if preferred:
            for key in preferred:
                bm = next((b for b in bms if b["key"]==key), None)
                if bm: break
        if bm is None: bm = bms[0]

        mkt = next((m for m in bm["markets"] if m["key"]=="h2h"), None)
        if not mkt: continue

        prices = {}
        for out in mkt["outcomes"]:
            abbr = full_to_abbrev.get(out["name"])
            if abbr:
                prices[abbr] = out["price"]

        lookup[(h_abbr, a_abbr)] = (prices.get(h_abbr), prices.get(a_abbr))

    rows = []
    for _, gm in games_df.iterrows():
        h,a = gm.home_team, gm.away_team
        o1,o2 = lookup.get((h,a),(None,None))
        if o1 is None or o2 is None:
            logging.warning(f"No odds found for {h} vs {a}")
        rows.append({"home_team":h,"away_team":a,"odds 1":o1,"odds 2":o2})
    return pd.DataFrame(rows)

def merge_with_odds(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Merge prediction data with odds data.

    Args:
        preds (DataFrame): DataFrame with predictions
        odds (DataFrame): DataFrame with odds

    Returns:
        DataFrame: Merged DataFrame
    """
    df = preds.merge(odds, on=["home_team","away_team"], how="left")
    tmp = preds.drop(columns=[c for c in ["odds 1","odds 2"] if c in preds], errors="ignore")
    df = tmp.merge(odds, on=["home_team","away_team"], how="left")
    df["imp_prob_home"] = df["odds 1"].apply(impute_prob)
    df["imp_prob_away"] = df["odds 2"].apply(impute_prob)
    return df

# Create predictions DataFrame
home_team_preds_ml = (
    full_pred
      .loc[full_pred_prob, ['team_x','team_y','proba']]
      .rename(columns={
          'team_x': 'home_team',
          'team_y': 'away_team',
          'proba':  'home_team_prob'
      })
      .assign(result=0, date=game_day)
)

# API key for odds data
API_KEY = "8e9d506f8573b01023028cef1bf645b5"
odds_df = fetch_odds(home_team_preds_ml, API_KEY, preferred=["draftkings","fanduel"])

# Merge predictions with odds
final_df = merge_with_odds(home_team_preds_ml, odds_df)

# Calculate value based on probabilities
final_df["value_home"] = final_df["home_team_prob"] - final_df["imp_prob_home"]
final_df["value_away"] = (1 - final_df["home_team_prob"]) - final_df["imp_prob_away"]
print(final_df.sort_values("value_home", ascending=False).head())

# Build final predictions DataFrame
home_team_preds = (
    full_pred
    .loc[full_pred_prob, ['team_x', 'team_y', 'proba']]
    .rename(columns={
        'team_x': 'home_team',
        'team_y': 'away_team',
        'proba':  'home_team_prob'
    })
    .assign(result=0, date=game_day)
)

# Merge in the actual American odds
home_team_preds = home_team_preds.merge(
    odds_df[['home_team', 'away_team', 'odds 1', 'odds 2']],
    on=['home_team', 'away_team'],
    how='left'
)

# Convert American odds to decimal odds
home_team_preds['odds 1'] = home_team_preds['odds 1'].apply(am_to_dec)
home_team_preds['odds 2'] = home_team_preds['odds 2'].apply(am_to_dec)

# Round to two decimal places
home_team_preds['odds 1'] = home_team_preds['odds 1'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
home_team_preds['odds 2'] = home_team_preds['odds 2'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

# Display predictions
cols = ['home_team', 'away_team', 'home_team_prob', 'odds 1', 'odds 2', 'result', 'date']
print(home_team_preds[cols].to_string(index=False))

# Save predictions
cols = [
    "home_team",
    "away_team",
    "home_team_prob",
    "result",
    "odds 1",
    "odds 2",
    "date"
]

# Subset/reorder columns
to_save = home_team_preds[cols]

# Save to file
file_name = f"nba_games_predict_{today_str_format}.csv"
full_path = os.path.join(directory_path, file_name)
os.makedirs(directory_path, exist_ok=True)

if os.path.exists(full_path):
    print(f"✅ File already exists: {full_path}")
else:
    to_save.to_csv(full_path, index=False)
    print(f"💾 Saved predictions to {full_path}")