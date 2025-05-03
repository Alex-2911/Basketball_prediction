#!/usr/bin/env python
# coding: utf-8

# In[1]:


#########################################################################################################################
# CALCUATE PREDICTION FOR NEXT GAME DAY #

# Script 3 of 4
# This script Calculates game predictions for the next NBA game day using historical data, rolling averages, and machine learning models,
# and outputs results with probabilities.

# Ensure `_2. 03012025_get_data_next_game_day.ipynb` is executed before running this script.
#########################################################################################################################


# In[2]:


ROLLING_WINDOW_SIZE = 8
current_season = 2025


# In[3]:


import pandas as pd
import datetime
import numpy as np
import lightgbm as lgb
import os

import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import glob
import datetime
from datetime import datetime, timedelta

import subprocess
import shutil

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


# In[4]:


today = (datetime.now()- timedelta(days=0)).strftime("%Y-%m-%d")


# In[5]:


# Constants
target_folder = "D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Next_Game\\"
STAT_DIR = "D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Whole_Statistic\\"

df_path = os.path.join(STAT_DIR, f"nba_games_{today}.csv")

directory_path = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
dst_dir = r'C:\_Laufwerk C\11. Sorare\NBA\2025\LightGBM'

open_office_path = "C:\Program Files (x86)\OpenOffice 4\program/scalc"


# In[6]:


# Define directory and date format
# Check if file exists
file_path = f"{target_folder}games_df_{today}.csv"
if not os.path.exists(file_path):
    # List files and pick the latest one
    files = sorted(glob.glob(f"{target_folder}games_df_*.csv"))
    if files:
        file_path = files[-1]  # Use the latest available file
        print(f"Using the latest file: {file_path}")
    else:
        print("No files found in the directory.")
        exit()

# Proceed to read the file
games_df = pd.read_csv(file_path, index_col=0)
print(games_df.head(60).to_string(index=False))


# In[7]:


# Function to find the most recent file in the directory if the desired one is not available
def get_latest_available_file(target_folder, prefix="nba_games_", extension=".csv"):
    """Returns the latest available CSV file matching the pattern."""
    available_files = [f for f in os.listdir(target_folder) if f.startswith(prefix) and f.endswith(extension)]
    if available_files:
        latest_file = max(available_files, key=lambda f: os.path.getctime(os.path.join(target_folder, f)))
        return os.path.join(target_folder, latest_file)
    return None

# Check if the specific file for today exists; if not, fallback to the most recent available file
if not os.path.exists(df_path):
    print(f"File for {today} not found. Searching for the latest available file...")
    df_path = get_latest_available_file(DST_DIR)
    if df_path:
        print(f"Using the latest available file: {df_path}")
    else:
        raise FileNotFoundError(f"No suitable file found in the directory: {DST_DIR}")

# Proceed with loading the data
df = pd.read_csv(df_path, index_col=0)
print(df)#.tail())  # Display a portion of the data

# Function to add a target column
def add_target(group):
    """Adds a target column to the DataFrame group based on the 'won' column."""
    group['target'] = group['won'].shift(-1)
    return group

def preprocess_nba_data():
    # Load the data
    df = pd.read_csv(df_path, index_col=0)

    # Sort by date
    df = df.sort_values("date")

    # Apply the preprocessing function to each team group
    df = df.groupby('team').apply(add_target)

    # Handle missing values
    df['target'].fillna(2, inplace=True)
    df['target'] = df['target'].astype(int)

    # Identify and remove columns with null values
    nulls = pd.isnull(df).sum()
    nulls = nulls[nulls > 0]
    valid_columns = df.columns[~df.columns.isin(nulls.index)]
    df = df[valid_columns].copy()

    return df

if __name__ == "__main__":
    df = preprocess_nba_data()

    # Columns to be excluded from scaling
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

    # Selecting columns that are not in the 'removed_columns' list
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the selected columns and update the DataFrame
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    #df.to_csv("D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Whole_Statistic\\df_orig.csv", index=False)



# In[8]:


df.groupby(["home"]).apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])

#print(df_rolling.head(60).to_string(index=False))


# In[9]:


####################################################################################################
# CALCULATE THE AVERAGE FOR THE PREVIOUS SEASONS WITH THE ROLLING WINDOW OF 7 FOR LEARNING THE MODEL #
####################################################################################################

# Filter out the games from the current season
df_rolling = df[list(selected_columns) + ["won", "team", "season"]]
#df_rolling = df_rolling[df_rolling['season'] != current_season].copy()

#print(df_rolling.columns)
def find_team_averages(team):
    numeric_columns = team.select_dtypes(include=[np.number])  # Select only numeric columns
    rolling = numeric_columns.rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean()  # Calculate rolling mean
    #rolling[['team', 'season']] = team[['team', 'season']]  # Retain 'team' and 'season' columns in the result
    return rolling

# Apply rolling average
df_rolling.reset_index(drop=True, inplace=True)
df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)


# Renaming columns with _7 suffix for numeric columns only
rolling_cols = {col: f"{col}_7" for col in df_rolling.columns if col not in ['team', 'season']} #, 'season','season_rolling','season_original','target']}


# Rename the columns
df_rolling.rename(columns=rolling_cols, inplace=True)


# In[10]:


df = df.reset_index(drop=True)
df_rolling = df_rolling.reset_index(drop=True)

df = pd.concat([df, df_rolling], axis=1)

#df.to_csv("D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Whole_Statistic\\df_pd.concat.csv", index=False)


df = df.dropna()

print(df)

target_2_rows = df[df['target'] == 2]['target']
print(target_2_rows)


# In[11]:


def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    # Ensure the 'team' column is not part of the index and is correctly formatted
    if 'team' in df.columns:
        return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))
    else:
        raise KeyError("The 'team' column is missing or not properly formatted in the DataFrame.")

# Ensure the 'team' column exists and is not part of the index
if 'team' not in df.columns:
    print("The 'team' column is missing. Ensure the column is present in your DataFrame.")

# Reset the index to avoid potential issues with multi-indexing
df = df.reset_index(drop=True)

# Add shifted columns for "home", "team_opp", and "date"
df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

# Drop rows where any of the next columns contain NaN values (optional)
#df = df.dropna(subset=["home_next", "team_opp_next", "date_next"])

# Optionally, save the DataFrame to a CSV file
#df.to_csv("D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Whole_Statistic\\df_dropna_target_2.csv", index=False)
#df.to_csv("D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Whole_Statistic\\df.csv", index=False)


# Display the first few rows to check the output
#print(df.head())

target_2_rows = df[df['target'] == 2]['target']
print(target_2_rows)


# In[12]:


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


# In[13]:


# Merging DataFrames
# Convert rolling_cols dictionary keys to a list and add other columns
full = df.merge(df[list(rolling_cols.keys()) + ["team_opp_next", "date_next", "team"]], 
                left_on=["team", "date_next"], 
                right_on=["team_opp_next", "date_next"])


# Save the merged DataFrame
output_path = "D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Whole_Statistic\\full_new.csv"
full.to_csv(output_path, index=False)
#print(f"Merged data saved to: {output_path}")

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


# In[14]:


mask = full['date_next'] == game_day
filtered_df = full.loc[mask, ['team_x', 'team_opp_next_x', 'team_y', 'team_opp_next_y', 'date_next', 'home_next']]

print(filtered_df)


# In[15]:


removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns


# In[16]:


selected_columns = full.columns[~full.columns.isin(removed_columns)]
selected_features = selected_columns.unique()

selected_features


# In[17]:


full_train = full[full["target"] != 2]
full_pred = full[full["target"] == 2]

print(full_pred)

X = full_train[selected_features].values
y = full_train["target"].values


# In[18]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


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

# Perform grid search
#grid_search.fit(X_train, y_train)

# Print the best parameters
#print("Best parameters found:", grid_search.best_params_)


# In[20]:


#Best parameters found: {'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 5, 'num_leaves': 10}

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



# In[21]:


# Train the model using X_train and y_train
model.fit(X_train, y_train)

# Predict the target values for the test set X_test
y_pred = model.predict(X_test)

# Check the accuracy of the model using the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[22]:


importances = model.feature_importances_

# create a dictionary to store feature importances with column names
feat_importances = dict(zip(selected_columns, importances))

# sort the dictionary by importance score in descending order
sorted_feat_importances = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)


# Print the sorted feature importances
for feature, importance in sorted_feat_importances[:30]:
    print("{}: {}".format(feature, importance))


# In[23]:


# predict on new data
full_pred["proba"] = model.predict_proba(full_pred[selected_features])[:,1]
full_pred["proba"]


# In[24]:


home_teams_prob = list(games_df['home_team'])
away_teams_prob = list(games_df['away_team'])

#print(home_teams_prob)
#print(away_teams_prob)

# Filter the rows where team_x is a home team
full_pred_prob = full_pred['team_x'].isin(home_teams_prob)
#print(full_pred_prob)

#full_pred_prob = full_pred['team_x'].isin(home_teams_prob)
full_pred[full_pred_prob]['proba']


# In[25]:


# Filter rows where full_pred_prob is True

team_x = full_pred.loc[full_pred_prob, 'team_x']
team_y = full_pred.loc[full_pred_prob, 'team_y']
#print(team_x)
#print(team_y)

team_pairs = pd.concat([team_x, team_y], axis=1)


print(team_pairs)


# In[27]:


import requests, pandas as pd, logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# full-name → 3-letter abbr
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

def get_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def fetch_odds(games_df: pd.DataFrame, api_key: str,
               preferred: list=None) -> pd.DataFrame:
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

def impute_prob(ml):
    if ml is None: return None
    ml = int(ml)
    return abs(ml)/(abs(ml)+100) if ml<0 else 100/(ml+100)

def merge_with_odds(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    df = preds.merge(odds, on=["home_team","away_team"], how="left")
    tmp = preds.drop(columns=[c for c in ["odds 1","odds 2"] if c in preds], errors="ignore")
    df = tmp.merge(odds, on=["home_team","away_team"], how="left")
    df["imp_prob_home"] = df["odds 1"].apply(impute_prob)
    df["imp_prob_away"] = df["odds 2"].apply(impute_prob)
    return df


# In[28]:


# assume you already have `home_team_preds` DataFrame in this notebook



# 1) Re-create your home_team_preds DataFrame
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

API_KEY   = "8e9d506f8573b01023028cef1bf645b5"
odds_df    = fetch_odds(home_team_preds_ml, API_KEY, preferred=["draftkings","fanduel"])

# inspect to confirm you have 'odds 1' & 'odds 2' columns:
# print(odds_df.head())
# print("Columns:", odds_df.columns.tolist())

final_df = merge_with_odds(home_team_preds_ml, odds_df)
#print(final_df.head())

# save out:
# final_df.to_csv(
#     "D:/1. Python/1. NBA Script/2025/LightGBM/1. 2025_Prediction/predictions_with_odds.csv",
#     index=False
# )

final_df["value_home"] = final_df["home_team_prob"] - final_df["imp_prob_home"]
final_df["value_away"] = (1 - final_df["home_team_prob"]) - final_df["imp_prob_away"]
print(final_df.sort_values("value_home", ascending=False).head())

# import matplotlib.pyplot as plt

# plt.scatter(final_df["imp_prob_home"], final_df["home_team_prob"])
# plt.plot([0,1],[0,1], linestyle="--")
# plt.xlabel("Market Implied Probability")
# plt.ylabel("Model Predicted Probability")
# plt.title("Model vs Market Comparison")
# plt.show()



# In[30]:


# 1) Build your predictions DataFrame (no placeholder zeros)
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

# 2) Merge in the actual American odds (from odds_df)
home_team_preds = home_team_preds.merge(
    odds_df[['home_team', 'away_team', 'odds 1', 'odds 2']],
    on=['home_team', 'away_team'],
    how='left'
)

# 3) Convert American odds to decimal odds
def am_to_dec(ml):
    if pd.isna(ml):
        return None
    ml = int(ml)
    return (ml/100 + 1) if ml > 0 else (100/abs(ml) + 1)

home_team_preds['odds 1'] = home_team_preds['odds 1'].apply(am_to_dec)
home_team_preds['odds 2'] = home_team_preds['odds 2'].apply(am_to_dec)

# Round to two decimal places
home_team_preds['odds 1'] = home_team_preds['odds 1'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
home_team_preds['odds 2'] = home_team_preds['odds 2'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

# 4) Display in the exact format requested
cols = ['home_team', 'away_team', 'home_team_prob', 'odds 1', 'odds 2', 'result', 'date']
print(home_team_preds[cols].to_string(index=False))


# In[32]:


# pick the exact column order
cols = [
    "home_team",
    "away_team",
    "home_team_prob",
    "result",    
    "odds 1",
    "odds 2",
    "date"
]

# subset/reorder
to_save = home_team_preds[cols]

# now save it
file_name = f"nba_games_predict_{today}.csv"
full_path = os.path.join(directory_path, file_name)
os.makedirs(directory_path, exist_ok=True)

if os.path.exists(full_path):
    print(f"✅ File already exists: {full_path}")
else:
    to_save.to_csv(full_path, index=False)
    print(f"💾 Saved predictions to {full_path}")


# In[34]:


# Open folder using subprocess on Windows
if os.name == 'nt':
    subprocess.Popen(f'explorer {directory_path}')
print(directory_path)
file_path = directory_path + "/" + file_name

src_files = set(os.listdir(directory_path))
dst_files = set(os.listdir(dst_dir))

diff = src_files - dst_files
diff_ = {file for file in diff if not file.startswith('.')}

print('Files in source but not in destination:')


# In[35]:


if diff_:
    for file_name in diff_:
        file_to_copy = os.path.join(directory_path, file_name)
        shutil.copy2(file_to_copy, dst_dir)
        print(f"Copied {file_name} to {dst_dir}")
else:
    print('No files to copy')





# In[ ]:




