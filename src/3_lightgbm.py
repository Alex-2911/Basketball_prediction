#!/usr/bin/env python
# coding: utf-8

#########################################################################################################################
# CALCULATE PREDICTION FOR NEXT GAME DAY
#
# Script 3 of 4
# This script calculates game predictions for the next NBA game day using historical data,
# rolling averages, and LightGBM, then outputs results with probabilities and odds.
# Ensure `src/2_get_data_next_game_day.py` has been run to produce the games_df CSV.
#########################################################################################################################

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
ROLLING_WINDOW_SIZE = 8
CURRENT_SEASON = 2025

today = datetime.now().strftime("%Y-%m-%d")

# Paths
BASE_DIR        = os.getcwd()
DATA_DIR        = os.path.join(BASE_DIR, "output", "Gathering_Data")
STAT_DIR        = os.path.join(DATA_DIR, "Whole_Statistic")
NEXT_DIR        = os.path.join(DATA_DIR, "Next_Game")
PRED_DIR        = os.path.join(BASE_DIR, "output", "LightGBM", "1_2025_Prediction")

# Historical stats file (contains full-season stats with 'team', 'date', 'won', etc.)
stats_df_path   = os.path.join(STAT_DIR, f"nba_games_{today}.csv")
# Next-game file (contains only home_team, away_team, game_date)
def get_latest_file(folder, prefix, ext):
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    return max(files, key=os.path.getctime) if files else None

games_df_path  = get_latest_file(NEXT_DIR, prefix="games_df_", ext=".csv")
if not games_df_path:
    raise FileNotFoundError(f"No games_df_*.csv found in {NEXT_DIR}")

# ──────────────────────────────────────────────────────────────────────────
# LOAD NEXT-GAME LINEUP
# ──────────────────────────────────────────────────────────────────────────
games_df = pd.read_csv(games_df_path, index_col=0)
print(f"Loaded games_df ({len(games_df)} rows) from {games_df_path}")

# ──────────────────────────────────────────────────────────────────────────
# PREPROCESS HISTORICAL STATS
# ──────────────────────────────────────────────────────────────────────────
def add_target(group):
    group['target'] = group['won'].shift(-1)
    return group


def preprocess_nba_data():
    df = pd.read_csv(stats_df_path, index_col=0)
    # ensure 'date'
    if 'game_date' in df.columns:
        df = df.rename(columns={'game_date':'date'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    # add target per team
    df = df.groupby('team', group_keys=False).apply(add_target)
    df['target'].fillna(2, inplace=True)
    df['target'] = df['target'].astype(int)
    # drop any columns with nulls
    nulls = df.isna().sum()
    df = df.loc[:, nulls == 0]
    return df

if __name__ == "__main__":
    hist_df = preprocess_nba_data()
    # drop non-feature columns
    drop_cols = ['season','date','won','target','team','team_opp']
    features = hist_df.columns.difference(drop_cols)
    # scale
    scaler = MinMaxScaler()
    hist_df[features] = scaler.fit_transform(hist_df[features])

    # rolling features
    roll = hist_df[features].rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean()
    roll.columns = [f"{c}_w{ROLLING_WINDOW_SIZE}" for c in roll.columns]
    hist_df = pd.concat([hist_df.reset_index(drop=True), roll], axis=1).dropna()

    # split train/test
    X = hist_df.drop(columns=['target']).loc[hist_df['target']!=2, features.union(roll.columns)]
    y = hist_df.loc[hist_df['target']!=2, 'target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train LightGBM
    params = {
        'objective':'binary', 'metric':'auc', 'boosting_type':'gbdt',
        'num_leaves':10,'learning_rate':0.1,'max_depth':7,
        'min_child_weight':5,'feature_fraction':0.9,
        'bagging_fraction':0.9,'bagging_freq':10,
        'verbosity':-1,'random_state':42
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

    # prepare full_pred
    full_pred = hist_df[hist_df['target']==2].copy()
    pred_feats = list(features) + list(roll.columns)
    full_pred['proba'] = model.predict_proba(full_pred[pred_feats])[:,1]

    # filter to next-game teams
    home_list = games_df['home_team'].tolist()
    full_pred = full_pred[full_pred['team'].isin(home_list)]
    full_pred = full_pred.rename(columns={'team':'home_team','team_opp':'away_team'})

    # fetch odds
    import requests, logging
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    full_to_abbrev = { ... }  # same mapping
    def fetch_odds(...): ...  # as before
    odds_df = fetch_odds(full_pred[['home_team','away_team']], API_KEY, preferred=[...])

    # build final dataframe
    final = full_pred[['home_team','away_team','proba']].rename(columns={'proba':'home_team_prob'})
    final = final.merge(odds_df, on=['home_team','away_team'], how='left')
    # convert & round odds
    def am_to_dec(x):
        if pd.isna(x): return None
        p = int(x)
        return round((p/100+1 if p>0 else 100/abs(p)+1),2)
    final['odds 1'] = final['odds_american_home'].apply(am_to_dec)
    final['odds 2'] = final['odds_american_away'].apply(am_to_dec)
    final['result']=0; final['date']=today

    # order & save
    out = final[['home_team','away_team','home_team_prob','odds 1','odds 2','result','date']]
    os.makedirs(PRED_DIR,exist_ok=True)
    out_path = os.path.join(PRED_DIR, f"nba_games_predict_{today}.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")
