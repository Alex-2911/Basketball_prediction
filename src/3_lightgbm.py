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
from datetime import datetime
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


def setup_paths_and_files():
    """
    Setup paths and files needed for the script.

    Returns:
        tuple: (paths, today_info, df_path_stat, games_df_path, games_df)
    """
    # Get current date and directory paths
    today, today_str, today_str_format = get_current_date(0)
    paths = get_directory_paths()

    # Define paths
    base_dir = paths['BASE_DIR']
    data_dir = paths['DATA_DIR']
    stat_dir = paths['STAT_DIR']
    target_folder = paths['NEXT_GAME_DIR']
    directory_path = paths['PREDICTION_DIR']

    # Get statistics file path
    df_path_stat = os.path.join(stat_dir, f"nba_games_{today_str_format}.csv")

    # Get latest games file
    games_df_path = get_latest_file(target_folder, prefix="games_df_", ext=".csv")
    if not games_df_path:
        raise FileNotFoundError(f"No games_df_*.csv found in {target_folder}")

    # Check if today's file exists
    file_path_today_games = f"{target_folder}/games_df_{today_str_format}.csv"
    if not os.path.exists(file_path_today_games):
        # List files and pick the latest one
        files = sorted(glob.glob(f"{target_folder}/games_df_*.csv"))
        if files:
            file_path_today_games = files[-1]  # Use the latest available file
            logging.info(f"Using the latest file: {file_path_today_games}")
        else:
            logging.error("No files found in the directory.")
            exit()

    # Read the games file
    games_df = pd.read_csv(file_path_today_games, index_col=0)
    logging.info(f"Games schedule:\n{games_df.head(60).to_string(index=False)}")

    return paths, (today, today_str, today_str_format), df_path_stat, games_df_path, games_df


def load_and_preprocess_data(df_path_stat, games_df):
    """
    Load and preprocess NBA data for model training.

    Args:
        df_path_stat (str): Path to statistics CSV file
        games_df (DataFrame): DataFrame with upcoming games

    Returns:
        tuple: (df, selected_columns, rolling_cols)
    """
    # Load and preprocess the data
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
    logging.info(f"Home win percentage:\n{df.groupby(['home']).apply(lambda x: x[x['won'] == 1].shape[0] / x.shape[0])}")

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

    # Add next game columns using the function from nba_utils
    df = add_next_game_columns(df)

    return df, selected_columns, rolling_cols


def prepare_training_and_prediction_data(df, rolling_cols, games_df, game_day):
    """
    Prepare both training data (historical) and prediction data (upcoming games).
    This function handles both sets of data separately to avoid empty dataset issues.

    Args:
        df (DataFrame): Historical team data
        rolling_cols (dict): Dictionary of rolling column names
        games_df (DataFrame): DataFrame with upcoming games
        game_day (str): Date of the games

    Returns:
        tuple: (train_df, pred_df, removed_columns)
    """
    logging.info("Preparing both training and prediction data separately")

    # ----- STEP 1: Prepare historical data for training -----

    # Filter historical data (exclude target=2 which represents future games)
    historical_data = []

    # Convert df to records for easier processing
    records = df.to_dict('records')

    # Get unique teams
    unique_teams = set()
    for record in records:
        if 'team' in record and record['team'] is not None:
            unique_teams.add(str(record['team']))

    logging.info(f"Found {len(unique_teams)} unique teams in historical data")

    # Process historical games (where we know the result)
    for record in records:
        # Skip records without valid target (future games)
        if 'target' not in record or record['target'] == 2:
            continue

        # Skip records without team or opponent info
        if 'team' not in record or 'team_opp' not in record:
            continue

        team = str(record['team'])
        opponent = str(record['team_opp'])

        # Find opponent data
        opponent_record = None
        for opp_record in records:
            if 'team' in opp_record and str(opp_record['team']) == opponent:
                opponent_record = opp_record
                break

        if opponent_record is None:
            continue

        # Create feature dictionary for this historical matchup
        feature_dict = {
            'team_x': team,
            'team_y': opponent,
            'target': record.get('target', 0),  # Result of the game
            'home': record.get('home', 0)       # Whether team was home
        }

        # Add team features
        for col in df.columns:
            if col in record and col not in ['team', 'team_opp', 'target', 'home']:
                feature_dict[f"{col}_x"] = record[col]

        # Add opponent features
        for col in df.columns:
            if col in opponent_record and col not in ['team', 'team_opp', 'target', 'home']:
                feature_dict[f"{col}_y"] = opponent_record[col]

        historical_data.append(feature_dict)

    # Create training DataFrame
    train_df = pd.DataFrame(historical_data) if historical_data else pd.DataFrame()
    logging.info(f"Created training dataset with {len(train_df)} records")

    # ----- STEP 2: Prepare upcoming games for prediction -----

    # Create a team lookup dictionary with the latest record for each team
    team_latest = {}
    for record in records:
        if 'team' not in record or 'date' not in record:
            continue

        team = str(record['team'])
        date = record.get('date')

        # Convert date to datetime if needed
        if isinstance(date, str):
            try:
                date = pd.to_datetime(date)
            except:
                continue

        # Update if this is the latest record
        if team not in team_latest or date > team_latest[team][1]:
            team_latest[team] = (record, date)

    # Build features for upcoming games
    prediction_data = []

    for _, game in games_df.iterrows():
        home_team = str(game['home_team'])
        away_team = str(game['away_team'])

        # Skip if we don't have data for either team
        if home_team not in team_latest or away_team not in team_latest:
            logging.warning(f"Missing data for game: {home_team} vs {away_team}")
            continue

        home_record = team_latest[home_team][0]
        away_record = team_latest[away_team][0]

        # Create prediction feature dictionary
        pred_dict = {
            'team_x': home_team,
            'team_y': away_team,
            'target': 2,  # Placeholder for upcoming games
            'home': 1,    # Home team is always 1
            'date_next': game_day
        }

        # Add home team features
        for col in df.columns:
            if col in home_record and col not in ['team', 'team_opp', 'target', 'home']:
                pred_dict[f"{col}_x"] = home_record[col]

        # Add away team features
        for col in df.columns:
            if col in away_record and col not in ['team', 'team_opp', 'target', 'home']:
                pred_dict[f"{col}_y"] = away_record[col]

        prediction_data.append(pred_dict)

    # Create prediction DataFrame
    pred_df = pd.DataFrame(prediction_data) if prediction_data else pd.DataFrame()
    logging.info(f"Created prediction dataset with {len(pred_df)} records")

    # Define columns to exclude from model features
    removed_columns = ["team_x", "team_y", "target", "home", "date", "date_next",
                      "season", "won", "team", "team_opp"]

    # Return both datasets and removed columns list
    return train_df, pred_df, removed_columns


def train_model_with_historical_data(train_df, removed_columns):
    """
    Train a LightGBM model using historical data.
    This version properly handles date columns and ensures all features are numeric.

    Args:
        train_df (DataFrame): Training data with historical matchups
        removed_columns (list): Columns to exclude from training

    Returns:
        tuple: (model, selected_features)
    """
    # Display info about training data
    logging.info(f"Training data shape: {train_df.shape}")
    logging.info(f"Training data columns: {train_df.columns.tolist()}")

    # Check if we have enough data
    if len(train_df) < 10:
        logging.error("Not enough training data available")
        raise ValueError("Insufficient training data")

    # Expand removed_columns to include any date-like columns or non-numeric columns
    additional_cols_to_remove = []
    for col in train_df.columns:
        # Skip columns already in removed_columns
        if col in removed_columns:
            continue

        # Check if column contains date-like strings
        if train_df[col].dtype == 'object':
            try:
                # Try to parse as date
                pd.to_datetime(train_df[col].iloc[0])
                additional_cols_to_remove.append(col)
                logging.info(f"Removing date column from features: {col}")
            except (ValueError, TypeError):
                # If column has string values that aren't dates, also remove
                if isinstance(train_df[col].iloc[0], str):
                    additional_cols_to_remove.append(col)
                    logging.info(f"Removing string column from features: {col}")

        # Check if column is not numeric
        elif not pd.api.types.is_numeric_dtype(train_df[col]):
            additional_cols_to_remove.append(col)
            logging.info(f"Removing non-numeric column from features: {col}")

    # Add the additional columns to remove
    all_removed_columns = removed_columns + additional_cols_to_remove

    # Select features
    selected_columns = [col for col in train_df.columns if col not in all_removed_columns]

    # Further check for any remaining non-numeric columns
    numeric_cols = []
    for col in selected_columns:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        else:
            logging.warning(f"Skipping non-numeric column: {col}")

    selected_features = numeric_cols

    logging.info(f"Selected {len(selected_features)} features for training")

    # Make sure we have enough features
    if len(selected_features) < 1:
        logging.error("No valid numeric features found for training")
        raise ValueError("No valid features for training")

    # Prepare X and y
    X = train_df[selected_features].values
    y = train_df["target"].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters
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

    # Create and train model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy*100:.2f}%")

    # Show feature importance
    importances = model.feature_importances_
    feat_importances = dict(zip(selected_features, importances))
    sorted_feat_importances = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)

    # Display top feature importances
    logging.info("Top 10 features by importance:")
    for i, (feature, importance) in enumerate(sorted_feat_importances[:min(10, len(sorted_feat_importances))]):
        logging.info(f"{i+1}. {feature}: {importance:.4f}")

    return model, selected_features


def generate_predictions(model, pred_df, selected_features, games_df, game_day):
    """
    Generate predictions for upcoming games.

    Args:
        model: Trained model
        pred_df: DataFrame with prediction features
        selected_features: List of features to use
        games_df: DataFrame with upcoming games
        game_day: Date of the games

    Returns:
        DataFrame: Predictions DataFrame
    """
    # Make sure we have data to predict on
    if len(pred_df) == 0:
        logging.warning("No prediction data available")
        return pd.DataFrame()

    # Use only the features the model was trained on
    X_pred = pred_df[selected_features].values

    # Generate probabilities
    probabilities = model.predict_proba(X_pred)[:, 1]

    # Create prediction DataFrame
    predictions = []

    for i, (_, row) in enumerate(pred_df.iterrows()):
        home_team = row['team_x']
        away_team = row['team_y']

        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_team_prob': probabilities[i],
            'result': 0,
            'date': game_day
        })

    return pd.DataFrame(predictions)


def get_session():
    """
    Create a requests session with retry capability.

    Returns:
        Session: Requests session with retry configuration
    """
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def fetch_odds(games_df, api_key, preferred=None):
    """
    Fetch odds data from the-odds-api.com for NBA games.

    Args:
        games_df (DataFrame): DataFrame with game information
        api_key (str): API key for the-odds-api.com
        preferred (list): List of preferred bookmakers

    Returns:
        DataFrame: DataFrame with odds information
    """
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


def merge_with_odds(preds, odds):
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


def prepare_final_predictions(home_team_preds_ml, odds_df, game_day):
    """
    Prepare the final predictions DataFrame with odds.

    Args:
        home_team_preds_ml (DataFrame): DataFrame with team predictions
        odds_df (DataFrame): DataFrame with odds information
        game_day (str): Date of the upcoming games

    Returns:
        tuple: (final_df, home_team_preds)
    """
    # Merge predictions with odds
    final_df = merge_with_odds(home_team_preds_ml, odds_df)

    # Calculate value based on probabilities
    final_df["value_home"] = final_df["home_team_prob"] - final_df["imp_prob_home"]
    final_df["value_away"] = (1 - final_df["home_team_prob"]) - final_df["imp_prob_away"]
    logging.info(f"Final predictions with value:\n{final_df.sort_values('value_home', ascending=False).head()}")

    # Build final predictions DataFrame
    home_team_preds = (
        home_team_preds_ml
        .rename(columns={
            'home_team_prob': 'home_team_prob'
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
    logging.info(f"Final predictions:\n{home_team_preds[cols].to_string(index=False)}")

    return final_df, home_team_preds


def save_predictions(home_team_preds, directory_path, today_str_format):
    """
    Save predictions to a CSV file.

    Args:
        home_team_preds (DataFrame): DataFrame with predictions
        directory_path (str): Directory to save the file
        today_str_format (str): Date string in YYYY-MM-DD format

    Returns:
        str: Path to the saved file
    """
    # Columns to include in the output
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
        logging.info(f"✅ File already exists: {full_path}")
    else:
        to_save.to_csv(full_path, index=False)
        logging.info(f"💾 Saved predictions to {full_path}")

    return full_path


def main():
    """Main execution function."""
    # Setup paths and files
    paths, today_info, df_path_stat, games_df_path, games_df = setup_paths_and_files()
    today, today_str, today_str_format = today_info
    directory_path = paths['PREDICTION_DIR']

    # Load and preprocess data
    df, selected_columns, rolling_cols = load_and_preprocess_data(df_path_stat, games_df)

    # Prepare training and prediction data
    train_df, pred_df, removed_columns = prepare_training_and_prediction_data(df, rolling_cols, games_df, today_str_format)

    # Train model with historical data only
    model, selected_features = train_model_with_historical_data(train_df, removed_columns)

    # Generate predictions
    home_team_preds_ml = generate_predictions(model, pred_df, selected_features, games_df, today_str_format)

    # Get odds data
    API_KEY = "8e9d506f8573b01023028cef1bf645b5"
    odds_df = fetch_odds(home_team_preds_ml, API_KEY, preferred=["draftkings","fanduel"])

    # Prepare final predictions
    final_df, home_team_preds = prepare_final_predictions(home_team_preds_ml, odds_df, today_str_format)

    # Save predictions
    output_path = save_predictions(home_team_preds, directory_path, today_str_format)

    return output_path


if __name__ == "__main__":
    main()