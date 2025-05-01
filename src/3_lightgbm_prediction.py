#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 3 of 5: Calculate Predictions for Next Game Day

This script calculates game predictions for the next NBA game day using historical data,
rolling averages, and machine learning models, and outputs results with probabilities.

Ensure "2_get_data_next_game_day.py" is executed before running this script.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import glob
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shutil
import subprocess

# Constants
ROLLING_WINDOW_SIZE = 7
current_season = 2025

# Get current date
today = datetime.now().strftime("%Y-%m-%d")

# Directories and paths
target_folder = "D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Next_Game\\"
STAT_DIR = "D:\\1. Python\\1. NBA Script\\2025\\Gathering_Data\\Whole_Statistic\\"
df_path = os.path.join(STAT_DIR, f"nba_games_{today}.csv")
directory_path = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
dst_dir = r'D:\_Laufwerk C\11. Sorare\NBA\2025\LightGBM'

def get_latest_available_file(target_folder, prefix="nba_games_", extension=".csv"):
    """Returns the latest available CSV file matching the pattern."""
    available_files = [f for f in os.listdir(target_folder) if f.startswith(prefix) and f.endswith(extension)]
    if available_files:
        latest_file = max(available_files, key=lambda f: os.path.getctime(os.path.join(target_folder, f)))
        return os.path.join(target_folder, latest_file)
    return None

def add_target(group):
    """Adds a target column to the DataFrame group based on the 'won' column."""
    group['target'] = group['won'].shift(-1)
    return group

def preprocess_nba_data(df_path):
    """
    Preprocesses NBA data for prediction model.
    
    Args:
        df_path (str): Path to the NBA games data CSV file.
        
    Returns:
        DataFrame: Preprocessed data ready for the model.
    """
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

def find_team_averages(team):
    """Calculate rolling averages for numeric columns."""
    numeric_columns = team.select_dtypes(include=[np.number])  # Select only numeric columns
    rolling = numeric_columns.rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean()  # Calculate rolling mean
    return rolling

def shift_col(team, col_name):
    """Shift column values for predictions."""
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    """Add shifted column grouped by team."""
    # Ensure the 'team' column is not part of the index and is correctly formatted
    if 'team' in df.columns:
        return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))
    else:
        raise KeyError("The 'team' column is missing or not properly formatted in the DataFrame.")

def main():
    """Main execution function."""
    # Create destination directory if needed
    os.makedirs(directory_path, exist_ok=True)
    
    # Find games data file for today
    file_path = f"{target_folder}games_df_{today}.csv"
    if not os.path.exists(file_path):
        # List files and pick the latest one
        files = sorted(glob.glob(f"{target_folder}games_df_*.csv"))
        if files:
            file_path = files[-1]  # Use the latest available file
            print(f"Using the latest file: {file_path}")
        else:
            print("No game files found in the directory.")
            return

    # Load the games data
    games_df = pd.read_csv(file_path, index_col=0)
    print(games_df.head(60).to_string(index=False))

    # Check if the statistics file exists; if not, use the latest available
    if not os.path.exists(df_path):
        print(f"File for {today} not found. Searching for the latest available file...")
        latest_file = get_latest_available_file(STAT_DIR)
        if latest_file:
            print(f"Using the latest available file: {latest_file}")
            df_path = latest_file
        else:
            raise FileNotFoundError(f"No suitable file found in the directory: {STAT_DIR}")

    # Load and preprocess the NBA data
    df = preprocess_nba_data(df_path)
    
    # Calculate team home advantage statistics
    home_advantage = df.groupby(["home"]).apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])
    print(home_advantage)

    # Columns to be excluded from scaling
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    
    # Selecting columns that are not in the 'removed_columns' list
    selected_columns = df.columns[~df.columns.isin(removed_columns)]
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Scale the selected columns and update the DataFrame
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    # Calculate rolling averages
    df_rolling = df[selected_columns]
    df_rolling = df_rolling.reset_index(drop=True)
    df_rolling = df.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

    # Add _7 suffix to rolling average columns
    rolling_cols = {col: f"{col}_7" for col in df_rolling.columns if col not in ['team', 'season']}
    df_rolling.rename(columns=rolling_cols, inplace=True)

    # Reset indices and merge dataframes
    df = df.reset_index(drop=True)
    df_rolling = df_rolling.reset_index(drop=True)
    df = pd.concat([df, df_rolling], axis=1)
    df = df.dropna()

    # Add information about next games
    df["home_next"] = add_col(df, "home")
    df["team_opp_next"] = add_col(df, "team_opp")
    df["date_next"] = add_col(df, "date")

    # Update next game information for prediction
    for _, game in games_df.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        game_day = game['game_date']
        
        print(home_team)
        print(away_team)
        print(game_day)

        # Update home team's next game info
        last_home_team_index = df.loc[df['team'] == home_team].iloc[::-1].index[0]
        df.loc[last_home_team_index, 'team_opp_next'] = away_team
        df.loc[last_home_team_index, 'home_next'] = 1
        df.loc[last_home_team_index, 'date_next'] = game_day
        
        # Update away team's next game info
        last_away_team_index = df.loc[df['team'] == away_team].iloc[::-1].index[0]
        df.loc[last_away_team_index, 'team_opp_next'] = home_team
        df.loc[last_away_team_index, 'home_next'] = 0
        df.loc[last_away_team_index, 'date_next'] = game_day

    # Create merged feature set
    full = df.merge(df[list(rolling_cols.keys()) + ["team_opp_next", "date_next", "team"]], 
                    left_on=["team", "date_next"], 
                    right_on=["team_opp_next", "date_next"])

    # Save merged data
    output_path = os.path.join(STAT_DIR, "full_new.csv")
    full.to_csv(output_path, index=False)
    
    # Display basic info
    print("Full DataFrame Info:")
    print(full.info())
    print("\nFirst few rows of the merged DataFrame:")
    print(full.head())
    print(f"Number of rows in 'full' DataFrame: {full.shape[0]}")
    
    # Extract rows with target == 2 (for prediction)
    target_2_rows = full[full['target'] == 2]['target']
    print("\nRows where 'target' == 2:")
    print(target_2_rows)
    
    # Filter the DataFrame for the prediction date
    mask = full['date_next'] == game_day
    filtered_df = full.loc[mask, ['team_x', 'team_opp_next_x', 'team_y', 'team_opp_next_y', 'date_next', 'home_next']]
    print(filtered_df)
    
    # Update removed columns and select features
    removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
    selected_columns = full.columns[~full.columns.isin(removed_columns)]
    selected_features = selected_columns.unique()
    
    # Prepare training and prediction data
    full_train = full[full["target"] != 2]
    full_pred = full[full["target"] == 2]
    print(full_pred)
    
    X = full_train[selected_features].values
    y = full_train["target"].values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define LightGBM parameters
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
    
    # Initialize and train the model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Get feature importances
    importances = model.feature_importances_
    feat_importances = dict(zip(selected_features, importances))
    sorted_feat_importances = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Print top feature importances
    print("\nTop 30 Feature Importances:")
    for feature, importance in sorted_feat_importances[:30]:
        print(f"{feature}: {importance}")
    
    # Generate predictions
    full_pred["proba"] = model.predict_proba(full_pred[selected_features])[:,1]
    print(full_pred["proba"])
    
    # Filter the rows where team_x is a home team
    home_teams_prob = list(games_df['home_team'])
    full_pred_prob = full_pred['team_x'].isin(home_teams_prob)
    print(full_pred[full_pred_prob]['proba'])
    
    # Create a DataFrame with predictions
    home_team_preds = full_pred.loc[full_pred_prob, ['team_x', 'team_y', 'proba']]
    home_team_preds.columns = ['home_team', 'away_team', 'home_team_prob']
    home_team_preds['odds 1'] = 0
    home_team_preds['odds 2'] = 0
    home_team_preds['result'] = 0
    home_team_preds['date'] = game_day
    
    print(home_team_preds.to_string(index=False))
    
    # Save the predictions
    file_name = f"nba_games_predict_{today}.csv"
    full_path = os.path.join(directory_path, file_name)
    
    if not os.path.isfile(full_path):
        home_team_preds.to_csv(full_path, index=False)
        print(f"Predictions saved to: {full_path}")
    else:
        print(f"A file with the name '{file_name}' already exists.")
    
    # Copy the file to the destination directory
    if os.name == 'nt':
        subprocess.Popen(f'explorer {directory_path}')
    
    src_files = set(os.listdir(directory_path))
    dst_files = set(os.listdir(dst_dir))
    diff = src_files - dst_files
    
    if diff:
        file_to_copy = os.path.join(directory_path, diff.pop())
        shutil.copy2(file_to_copy, dst_dir)
        print('File copied successfully')
        print(dst_dir)
    else:
        print('No files to copy')

if __name__ == "__main__":
    main()
