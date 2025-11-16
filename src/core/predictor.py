# -*- coding: utf-8 -*-
"""
NBA Game Prediction Module

This module provides classes for preprocessing game data, building matchups,
and training/predicting with LightGBM models for NBA game outcomes.

Classes:
    GameDataPreprocessor: Handles data loading and preprocessing
    MatchupBuilder: Builds team vs opponent matchup pairs
    LightGBMPredictor: Trains and runs LightGBM prediction model

Example:
    >>> preprocessor = GameDataPreprocessor(paths)
    >>> games_df = preprocessor.load_upcoming_games(date_str)
    >>> stats_df = preprocessor.load_historical_stats(date_str)
    >>> df = preprocessor.preprocess_pipeline(stats_df, games_df)
    >>>
    >>> builder = MatchupBuilder()
    >>> full_df = builder.build_matchup_pairs(df)
    >>> train_df, pred_df = builder.split_train_prediction(full_df)
    >>> features = builder.select_features(train_df)
    >>>
    >>> predictor = LightGBMPredictor()
    >>> predictor.train(train_df, features)
    >>> predictions = predictor.predict(pred_df, features)
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.core.constants import LGBM_PARAMS, ROLLING_WINDOW_SIZE
from src.utils.error_handlers import (
    DataValidationError,
    ErrorContext,
    log_dataframe_info,
    validate_dataframe,
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# TEAM NAME NORMALIZATION
# ─────────────────────────────────────────────────────────

TEAM_ALIAS_FOR_ODDS = {
    "PHO": "PHX",
    "PHX": "PHX",
    "BKN": "BRK",
    "BRK": "BRK",
    "CHA": "CHO",
    "CHO": "CHO",
    "GS": "GSW",
    "GSW": "GSW",
    "NO": "NOP",
    "NOP": "NOP",
    "NY": "NYK",
    "NYK": "NYK",
    "SA": "SAS",
    "SAS": "SAS",
    "UTAH": "UTA",
    "UTA": "UTA",
    "OKL": "OKC",
    "OKC": "OKC",
}


def normalize_team_code(abbr: str) -> str:
    """
    Normalize team abbreviation to canonical form.

    Args:
        abbr: Team abbreviation (e.g., "PHO", "BKN", "CHA")

    Returns:
        Canonical team code (e.g., "PHX", "BRK", "CHO")

    Example:
        >>> normalize_team_code("PHO")
        "PHX"
        >>> normalize_team_code("BKN")
        "BRK"
    """
    if not isinstance(abbr, str):
        return abbr
    return TEAM_ALIAS_FOR_ODDS.get(abbr.upper(), abbr.upper())


# ─────────────────────────────────────────────────────────
# PATH UTILITIES
# ─────────────────────────────────────────────────────────


def get_directory_paths() -> Dict[str, str]:
    """
    Get dictionary of important data directories.

    Returns:
        Dictionary with keys: STAT_DIR, NEXT_GAME_DIR, PREDICTION_DIR
        All paths are absolute and cross-platform compatible.

    Example:
        >>> paths = get_directory_paths()
        >>> stats_dir = paths["STAT_DIR"]
    """
    script_dir = Path(__file__).parent
    base_repo = script_dir.parent.parent

    return {
        "STAT_DIR": str(base_repo / "output" / "Gathering_Data" / "Whole_Statistic"),
        "NEXT_GAME_DIR": str(base_repo / "output" / "Gathering_Data" / "Next_Game"),
        "PREDICTION_DIR": str(base_repo / "output" / "LightGBM"),
    }


def get_latest_file(directory: str, prefix: str, ext: str) -> str:
    """
    Find the most recent file matching pattern in directory.

    Args:
        directory: Directory path to search
        prefix: File name prefix to match
        ext: File extension (e.g., ".csv")

    Returns:
        Path to the latest matching file, or empty string if none found

    Example:
        >>> latest = get_latest_file("/data", "nba_games_", ".csv")
        "/data/nba_games_2025-11-15.csv"
    """
    pattern = os.path.join(directory, f"{prefix}*{ext}")
    files = glob.glob(pattern)
    if not files:
        return ""
    return max(files, key=os.path.getctime)


# ─────────────────────────────────────────────────────────
# GAME DATA PREPROCESSOR
# ─────────────────────────────────────────────────────────


class GameDataPreprocessor:
    """
    Handles loading and preprocessing of NBA game data.

    This class provides methods to load upcoming games, historical statistics,
    and preprocess the data with target variables, feature scaling, rolling
    averages, and next-game metadata.

    Attributes:
        paths: Dictionary of data directory paths
        rolling_window: Size of rolling window for feature averaging

    Example:
        >>> paths = get_directory_paths()
        >>> preprocessor = GameDataPreprocessor(paths, rolling_window=9)
        >>> games_df = preprocessor.load_upcoming_games("2025-11-16")
        >>> stats_df = preprocessor.load_historical_stats("2025-11-16")
    """

    def __init__(self, paths: Optional[Dict[str, str]] = None, rolling_window: int = ROLLING_WINDOW_SIZE):
        """
        Initialize the preprocessor.

        Args:
            paths: Dictionary with STAT_DIR, NEXT_GAME_DIR, PREDICTION_DIR.
                   If None, uses default paths from get_directory_paths()
            rolling_window: Number of games for rolling average (default from constants)
        """
        self.paths = paths or get_directory_paths()
        self.rolling_window = rolling_window
        self.scaler = MinMaxScaler()
        self.scaled_columns: List[str] = []

    def load_upcoming_games(self, date_str: str) -> pd.DataFrame:
        """
        Load schedule of upcoming games for specified date.

        Attempts to load games_df_{date}.csv. If not found, falls back to
        most recent games_df_*.csv file.

        Args:
            date_str: Date in format "YYYY-MM-DD"

        Returns:
            DataFrame with columns: home_team, away_team, game_date

        Raises:
            FileNotFoundError: If no games_df files exist
            DataValidationError: If required columns are missing

        Example:
            >>> games_df = preprocessor.load_upcoming_games("2025-11-16")
            >>> print(games_df.columns)
            Index(['home_team', 'away_team', 'game_date'])
        """
        with ErrorContext("Loading game schedule", logger=logger):
            next_game_dir = self.paths["NEXT_GAME_DIR"]
            direct_path = os.path.join(next_game_dir, f"games_df_{date_str}.csv")

            if os.path.exists(direct_path):
                file_path = direct_path
            else:
                file_path = get_latest_file(next_game_dir, prefix="games_df_", ext=".csv")
                if not file_path:
                    raise FileNotFoundError(f"No games_df_*.csv found in {next_game_dir}")
                logger.info(f"games_df for {date_str} not found. Falling back to {file_path}")

            games_df = pd.read_csv(file_path)

            # Remove index column if present
            if "Unnamed: 0" in games_df.columns:
                games_df = games_df.drop(columns=["Unnamed: 0"])

            if games_df.empty:
                logger.warning("games_df is empty (season might be over).")

            # Validate required columns
            validate_dataframe(
                games_df, required_columns=["home_team", "away_team", "game_date"], allow_empty=True
            )

            logger.info(f"Loaded game schedule from {file_path} with {len(games_df)} games")
            return games_df

    def load_historical_stats(self, date_str: str) -> pd.DataFrame:
        """
        Load historical game statistics for teams.

        Attempts to load nba_games_{date}.csv. If not found, falls back to
        most recent nba_games_*.csv file.

        Args:
            date_str: Date in format "YYYY-MM-DD"

        Returns:
            DataFrame with team game statistics and features

        Raises:
            FileNotFoundError: If no stats files exist

        Example:
            >>> stats_df = preprocessor.load_historical_stats("2025-11-16")
            >>> print(len(stats_df))
            5000
        """
        with ErrorContext("Loading game statistics", logger=logger):
            stat_dir = self.paths["STAT_DIR"]
            direct_path = os.path.join(stat_dir, f"nba_games_{date_str}.csv")

            if os.path.exists(direct_path):
                df_path = direct_path
            else:
                logger.info(f"Stats file for {date_str} not found. Searching latest in {stat_dir}...")
                df_path = get_latest_file(stat_dir, prefix="nba_games_", ext=".csv")
                if not df_path:
                    raise FileNotFoundError(f"No nba_games_*.csv files found in {stat_dir}")
                logger.info(f"Using latest stats file: {df_path}")

            df = pd.read_csv(df_path)

            # Remove index column if present
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            log_dataframe_info(df, name="Game statistics", logger=logger)
            return df

    def add_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target variable for each team indicating next game outcome.

        For each team, creates 'target' column where:
        - target = won value of NEXT game (shift -1)
        - target = 2 for future games (unknown outcome)

        Args:
            df: DataFrame with columns: team, date, won

        Returns:
            DataFrame with added 'target' column (int: 0, 1, or 2)

        Example:
            >>> df = preprocessor.add_target_variable(df)
            >>> df[df['target'] == 2]  # Future games
        """

        def add_target(group):
            group = group.sort_values("date")
            group["target"] = group["won"].shift(-1)
            return group

        df = df.sort_values("date")
        df = df.groupby("team", group_keys=False).apply(add_target)
        df["target"] = df["target"].fillna(2).astype(int)

        logger.info(f"Added target variable: {len(df[df['target'] == 2])} future games")
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MinMax scaling to numeric features.

        Scales all numeric columns except metadata columns (season, date, won,
        target, team, team_opp) to range [0, 1].

        Args:
            df: DataFrame with numeric features

        Returns:
            DataFrame with scaled numeric features

        Note:
            Stores scaled column names in self.scaled_columns
            Stores fitted scaler in self.scaler

        Example:
            >>> df = preprocessor.scale_features(df)
            >>> print(preprocessor.scaled_columns[:5])
            ['fg_pct', 'fg3_pct', 'ft_pct', 'orb', 'drb']
        """
        removed_cols_for_scaling = ["season", "date", "won", "target", "team", "team_opp"]
        to_scale = df.columns[~df.columns.isin(removed_cols_for_scaling)]

        self.scaler = MinMaxScaler()
        df[to_scale] = self.scaler.fit_transform(df[to_scale])
        self.scaled_columns = list(to_scale)

        logger.info(f"Scaled {len(to_scale)} numeric columns")
        return df

    def compute_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling averages for numeric features by team and season.

        Creates new columns with '_7' suffix containing rolling averages.
        Excludes 'target' column to prevent data leakage.

        Args:
            df: DataFrame with team, season, and numeric features

        Returns:
            DataFrame with original columns plus rolling average columns

        Example:
            >>> df = preprocessor.compute_rolling_averages(df)
            >>> print([c for c in df.columns if c.endswith('_7')][:3])
            ['fg_pct_7', 'fg3_pct_7', 'ft_pct_7']
        """

        def team_roll(g: pd.DataFrame) -> pd.DataFrame:
            numeric_cols = g.select_dtypes(include=[np.number]).copy()
            # Don't roll target to prevent leakage
            numeric_cols = numeric_cols.drop(columns=["target"], errors="ignore")
            rolled = numeric_cols.rolling(self.rolling_window, min_periods=1).mean()
            return rolled

        df_numeric = df.groupby(["team", "season"], group_keys=False).apply(team_roll)

        # Rename with _7 suffix
        rename_map = {col: f"{col}_7" for col in df_numeric.columns}
        df_numeric = df_numeric.rename(columns=rename_map)

        # Concatenate with original
        df = pd.concat([df.reset_index(drop=True), df_numeric.reset_index(drop=True)], axis=1)

        logger.info(
            f"Computed rolling averages with window={self.rolling_window}, "
            f"created {len(df_numeric.columns)} rolling features"
        )
        return df

    def add_next_game_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata about each team's next game.

        For each team, adds columns indicating:
        - home_next: Whether next game is home (1) or away (0)
        - team_opp_next: Opponent in next game
        - date_next: Date of next game

        Args:
            df: DataFrame with team, home, team_opp, date columns

        Returns:
            DataFrame with added next-game metadata columns

        Example:
            >>> df = preprocessor.add_next_game_metadata(df)
            >>> print(df[['team', 'home_next', 'team_opp_next', 'date_next']].tail())
        """

        def shift_col(team_df: pd.DataFrame, col_name: str) -> pd.Series:
            return team_df[col_name].shift(-1)

        def add_cols(team_df: pd.DataFrame) -> pd.DataFrame:
            team_df = team_df.copy()
            team_df["home_next"] = shift_col(team_df, "home")
            team_df["team_opp_next"] = shift_col(team_df, "team_opp")
            team_df["date_next"] = shift_col(team_df, "date")
            return team_df

        df = df.groupby("team", group_keys=False).apply(add_cols)

        logger.info("Added next game metadata columns")
        return df

    def override_with_schedule(self, df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Override next-game metadata with today's actual schedule.

        For each game in games_df, updates the last row of each team to point
        to today's matchup. This ensures predictions align with actual schedule.

        Args:
            df: DataFrame with team statistics and next-game columns
            games_df: DataFrame with today's schedule (home_team, away_team, game_date)

        Returns:
            DataFrame with updated next-game metadata for teams playing today

        Example:
            >>> df = preprocessor.override_with_schedule(df, games_df)
            >>> # Last row for each team now points to today's game
        """
        if games_df.empty:
            logger.warning("No upcoming games in schedule; skip override_with_schedule.")
            return df

        df = df.copy()

        # Ensure required columns exist
        needed_cols = ["team_opp_next", "home_next", "date_next"]
        for col in needed_cols:
            if col not in df.columns:
                df[col] = np.nan

        for idx, game in games_df.iterrows():
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            game_day = game.get("game_date")

            if pd.isna(home_team) or pd.isna(away_team) or pd.isna(game_day):
                logger.warning(f"Skipping row {idx} in games_df due to missing values.")
                continue

            # Update last row for home team
            home_mask = df["team"] == home_team
            if home_mask.any():
                last_home_idx = df[home_mask].index.max()
                df.loc[last_home_idx, "team_opp_next"] = away_team
                df.loc[last_home_idx, "home_next"] = 1
                df.loc[last_home_idx, "date_next"] = game_day
            else:
                logger.warning(f"Could not find recent row for home team {home_team}")

            # Update last row for away team
            away_mask = df["team"] == away_team
            if away_mask.any():
                last_away_idx = df[away_mask].index.max()
                df.loc[last_away_idx, "team_opp_next"] = home_team
                df.loc[last_away_idx, "home_next"] = 0
                df.loc[last_away_idx, "date_next"] = game_day
            else:
                logger.warning(f"Could not find recent row for away team {away_team}")

        logger.info(f"Overrode next-game metadata for {len(games_df)} scheduled games")
        return df

    def preprocess_pipeline(
        self, stats_df: pd.DataFrame, games_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run full preprocessing pipeline on stats data.

        Executes all preprocessing steps in sequence:
        1. Add target variable
        2. Remove all-null columns
        3. Scale numeric features
        4. Compute rolling averages
        5. Add next-game metadata
        6. Normalize team codes
        7. Override with schedule (if provided)

        Args:
            stats_df: Historical game statistics
            games_df: Optional schedule of upcoming games

        Returns:
            Fully preprocessed DataFrame ready for matchup building

        Example:
            >>> df = preprocessor.preprocess_pipeline(stats_df, games_df)
            >>> # df is now ready for MatchupBuilder
        """
        with ErrorContext("Preprocessing pipeline", logger=logger):
            df = stats_df.copy()

            # Step 1: Add target
            df = self.add_target_variable(df)

            # Step 2: Remove all-null columns (except key columns)
            key_keep = {"team", "date", "won", "home", "team_opp", "season"}
            nulls = df.isnull().sum()
            drop_cols = nulls[nulls > 0].index.tolist()
            truly_all_nan = [c for c in drop_cols if c not in key_keep and df[c].isna().all()]
            if truly_all_nan:
                df = df.drop(columns=truly_all_nan)
                logger.info(f"Dropped {len(truly_all_nan)} all-null columns")

            # Step 3: Scale features
            df = self.scale_features(df)

            # Step 4: Rolling averages
            df = self.compute_rolling_averages(df)

            # Step 5: Next-game metadata
            df = self.add_next_game_metadata(df)

            # Step 6: Normalize team codes
            df["team"] = df["team"].apply(normalize_team_code)
            df["team_opp"] = df["team_opp"].apply(normalize_team_code)

            # Step 7: Override with schedule if provided
            if games_df is not None and not games_df.empty:
                # Normalize schedule team codes
                games_df = games_df.copy()
                games_df["home_team"] = games_df["home_team"].apply(normalize_team_code)
                games_df["away_team"] = games_df["away_team"].apply(normalize_team_code)
                df = self.override_with_schedule(df, games_df)

            logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
            return df


# ─────────────────────────────────────────────────────────
# MATCHUP BUILDER
# ─────────────────────────────────────────────────────────


class MatchupBuilder:
    """
    Builds team vs opponent matchup pairs for model training and prediction.

    Creates matchup rows by merging each team's stats with their opponent's stats,
    resulting in a single row per game with both teams' features.

    Example:
        >>> builder = MatchupBuilder()
        >>> matchups = builder.build_matchup_pairs(preprocessed_df)
        >>> train_df, pred_df = builder.split_train_prediction(matchups)
        >>> features = builder.select_features(train_df)
    """

    def __init__(self):
        """Initialize the matchup builder."""
        pass

    def build_matchup_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build matchup pairs by merging team stats with opponent stats.

        Creates one row per matchup with:
        - team_x: Focal team (with original features)
        - team_y: Opponent (with features suffixed '_right')
        - home_next: Whether team_x is home in next game
        - target: Outcome for team_x (0=loss, 1=win, 2=unknown)

        Args:
            df: Preprocessed DataFrame with team stats and next-game metadata

        Returns:
            DataFrame of matchups ready for training/prediction

        Raises:
            RuntimeError: If merge fails to identify team columns correctly

        Example:
            >>> matchups = builder.build_matchup_pairs(df)
            >>> print(matchups[['team_x', 'team_y', 'home_next', 'target']].head())
        """
        df_left = df.copy()

        # Build right side with opponent features
        banned_right_cols = {
            "team",
            "team_opp",
            "team_opp_next",
            "home",
            "home_next",
            "date",
            "date_next",
            "target",
            "won",
            "season",
        }

        keep_for_right = [c for c in df.columns if c not in banned_right_cols]
        df_right = df[keep_for_right + ["team_opp_next", "date_next", "team"]].copy()

        # Rename right-side features with _right suffix
        rename_map = {c: f"{c}_right" for c in keep_for_right}
        df_right = df_right.rename(columns=rename_map)

        # Merge: team (left) plays against team_opp_next (right) on date_next
        full = df_left.merge(
            df_right,
            left_on=["team", "date_next"],
            right_on=["team_opp_next", "date_next"],
            how="inner",
        )

        # Identify team columns after merge
        if "team_x" in full.columns and "team_y" in full.columns:
            team_x_col = "team_x"
            team_y_col = "team_y"
        else:
            team_x_col = "team"
            team_y_col = "team_y" if "team_y" in full.columns else "team_right"
            if team_y_col not in full.columns:
                candidates = [c for c in full.columns if c.endswith("_y")]
                if len(candidates) == 1:
                    team_y_col = candidates[0]

        # Validate team columns exist
        if team_x_col not in full.columns:
            raise RuntimeError("Could not identify 'team_x' in merged data.")
        if team_y_col not in full.columns:
            raise RuntimeError("Could not identify 'team_y' (opponent) in merged data.")

        # Standardize column names
        full = full.rename(columns={team_x_col: "team_x", team_y_col: "team_y"})

        logger.info(
            f"Built {len(full)} matchup pairs with {len(full.columns)} total features "
            f"(including opponent features)"
        )
        return full

    def split_train_prediction(self, full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split matchups into training and prediction sets.

        Splits based on target column:
        - Training: target in {0, 1} (known outcomes)
        - Prediction: target == 2 (unknown outcomes, future games)

        Args:
            full: DataFrame of matchup pairs with 'target' column

        Returns:
            Tuple of (training_df, prediction_df)

        Raises:
            RuntimeError: If 'target' column is missing

        Example:
            >>> train_df, pred_df = builder.split_train_prediction(matchups)
            >>> print(f"Train: {len(train_df)}, Predict: {len(pred_df)}")
            Train: 5000, Predict: 10
        """
        if "target" not in full.columns:
            raise RuntimeError("Expected 'target' column in full matchup DataFrame.")

        full_train = full[full["target"] != 2].copy()
        full_pred = full[full["target"] == 2].copy()

        logger.info(f"Split into {len(full_train)} training rows and {len(full_pred)} prediction rows")
        return full_train, full_pred

    def select_features(self, full: pd.DataFrame) -> List[str]:
        """
        Select feature columns for model training, excluding metadata and leakage.

        Excludes:
        - Team identifiers (team_x, team_y)
        - Date/time columns (date, date_next)
        - Target-related columns (target, won)
        - Season identifier
        - Non-numeric columns

        Args:
            full: DataFrame of matchup pairs

        Returns:
            List of feature column names safe for training

        Example:
            >>> features = builder.select_features(train_df)
            >>> print(f"Selected {len(features)} features")
            Selected 250 features
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
            # Skip banned columns
            if col in banned_explicit:
                continue

            # Skip columns with 'target' or 'won' to prevent leakage
            if "target" in col.lower() or "won" in col.lower():
                continue

            # Must be numeric
            if full[col].dtype == object:
                continue
            if not pd.api.types.is_numeric_dtype(full[col]):
                continue

            feature_cols.append(col)

        logger.info(f"Selected {len(feature_cols)} features for modeling")
        return feature_cols


# ─────────────────────────────────────────────────────────
# LIGHTGBM PREDICTOR
# ─────────────────────────────────────────────────────────


class LightGBMPredictor:
    """
    LightGBM classifier for predicting NBA game outcomes.

    Trains a gradient boosting model to predict win probability and generates
    predictions with calibrated probabilities.

    Attributes:
        model: Trained LightGBM classifier
        params: Model hyperparameters
        feature_names: Names of features used in training
        accuracy: Holdout accuracy from last training

    Example:
        >>> predictor = LightGBMPredictor()
        >>> predictor.train(train_df, features)
        >>> predictions = predictor.predict(pred_df, features)
        >>> importances = predictor.get_feature_importances(top_n=10)
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the predictor.

        Args:
            params: LightGBM hyperparameters. If None, uses LGBM_PARAMS from constants.
        """
        self.params = params or LGBM_PARAMS.copy()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: List[str] = []
        self.accuracy: Optional[float] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def train(self, train_df: pd.DataFrame, feature_cols: List[str], test_size: float = 0.2) -> float:
        """
        Train LightGBM model on training data.

        Splits data into train/test, trains model, and evaluates accuracy
        on holdout set.

        Args:
            train_df: Training DataFrame with features and 'target' column
            feature_cols: List of feature column names to use
            test_size: Fraction of data to use for validation (default 0.2)

        Returns:
            Accuracy score on holdout test set

        Raises:
            ValueError: If insufficient training samples (<10 rows)
            DataValidationError: If target column missing

        Example:
            >>> accuracy = predictor.train(train_df, features)
            >>> print(f"Model accuracy: {accuracy:.2%}")
            Model accuracy: 68.5%
        """
        with ErrorContext("Training LightGBM model", logger=logger):
            if len(train_df) < 10:
                raise ValueError(f"Not enough training samples: {len(train_df)} < 10")

            if "target" not in train_df.columns:
                raise DataValidationError("Training data missing 'target' column")

            # Prepare data
            X = train_df[feature_cols].values
            y = train_df["target"].values

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Store for later evaluation
            self.X_test = X_test
            self.y_test = y_test
            self.feature_names = feature_cols

            # Train model
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"LightGBM model trained on {len(X_train)} samples")
            logger.info(f"Holdout accuracy: {self.accuracy:.2%}")

            # Log top feature importances
            importances = self.get_feature_importances(top_n=10)
            logger.info("Top 10 feature importances:")
            for i, (name, score) in enumerate(importances, start=1):
                logger.info(f"  {i}. {name}: {score:.0f}")

            return self.accuracy

    def predict(
        self,
        pred_df: pd.DataFrame,
        feature_cols: List[str],
        games_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate win probability predictions for upcoming games.

        Predicts probability that team_x (home team) wins their next game.

        Args:
            pred_df: Prediction DataFrame with features and next-game metadata
            feature_cols: List of feature column names (must match training)
            games_df: Optional schedule to filter predictions (recommended)

        Returns:
            DataFrame with columns:
                - home_team: Team playing at home
                - away_team: Team playing away
                - home_team_prob: Probability home team wins
                - result: 0 (placeholder for unknown outcome)
                - date: Game date

        Example:
            >>> predictions = predictor.predict(pred_df, features, games_df)
            >>> print(predictions[['home_team', 'away_team', 'home_team_prob']])
        """
        with ErrorContext("Generating predictions", logger=logger):
            if self.model is None:
                raise RuntimeError("Model not trained. Call train() first.")

            if pred_df.empty:
                logger.warning("No prediction data provided (pred_df is empty).")
                return pd.DataFrame()

            # Validate features match
            if feature_cols != self.feature_names:
                logger.warning("Feature columns don't match training features")

            # Predict probabilities
            X_pred = pred_df[feature_cols].values
            probs = self.model.predict_proba(X_pred)[:, 1]

            pred_df = pred_df.copy()
            pred_df["proba"] = probs

            # Keep only rows where team_x is home (home_next == 1)
            pred_df = pred_df[pred_df["home_next"] == 1].copy()

            # Build output DataFrame
            out_rows = []
            for _, row in pred_df.iterrows():
                out_rows.append(
                    {
                        "home_team": row["team_x"],
                        "away_team": row["team_y"],
                        "home_team_prob": float(row["proba"]),
                        "result": 0,
                        "date": row["date_next"],
                    }
                )

            preds = pd.DataFrame(out_rows)

            if preds.empty:
                logger.warning("No predictions where team_x is home team.")
                return preds

            # Filter to only games in schedule (if provided)
            if games_df is not None and not games_df.empty:
                if "home_team" in games_df.columns and "away_team" in games_df.columns:
                    pairs = set(zip(games_df["home_team"], games_df["away_team"]))
                    preds = preds[
                        preds.apply(lambda r: (r["home_team"], r["away_team"]) in pairs, axis=1)
                    ].copy()

            if preds.empty:
                logger.warning("After schedule alignment, no predictions remain.")

            logger.info(f"Generated {len(preds)} predictions")
            return preds

    def get_feature_importances(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get feature importances from trained model.

        Args:
            top_n: Number of top features to return (None = all features)

        Returns:
            List of (feature_name, importance_score) tuples, sorted by importance

        Raises:
            RuntimeError: If model not trained

        Example:
            >>> importances = predictor.get_feature_importances(top_n=20)
            >>> for name, score in importances:
            ...     print(f"{name}: {score:.0f}")
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        importances = self.model.feature_importances_
        pairs = sorted(
            zip(self.feature_names, importances), key=lambda x: x[1], reverse=True
        )

        if top_n is not None:
            pairs = pairs[:top_n]

        return pairs

    def evaluate_accuracy(self) -> Optional[float]:
        """
        Get accuracy score from last training run.

        Returns:
            Accuracy score (0-1), or None if not trained

        Example:
            >>> acc = predictor.evaluate_accuracy()
            >>> if acc is not None:
            ...     print(f"Current model accuracy: {acc:.2%}")
        """
        return self.accuracy


# ─────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example usage of the prediction module.
    """
    # Setup
    paths = get_directory_paths()
    date_str = "2025-11-16"

    # Initialize components
    preprocessor = GameDataPreprocessor(paths)
    builder = MatchupBuilder()
    predictor = LightGBMPredictor()

    try:
        # Load data
        logger.info("Loading data...")
        games_df = preprocessor.load_upcoming_games(date_str)
        stats_df = preprocessor.load_historical_stats(date_str)

        # Preprocess
        logger.info("Preprocessing...")
        df = preprocessor.preprocess_pipeline(stats_df, games_df)

        # Build matchups
        logger.info("Building matchups...")
        full = builder.build_matchup_pairs(df)
        train_df, pred_df = builder.split_train_prediction(full)

        # Select features
        features = builder.select_features(train_df)

        # Train model
        logger.info("Training model...")
        accuracy = predictor.train(train_df, features)

        # Generate predictions
        logger.info("Generating predictions...")
        predictions = predictor.predict(pred_df, features, games_df)

        logger.info(f"\n{'=' * 60}")
        logger.info("PREDICTION PIPELINE COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Model Accuracy: {accuracy:.2%}")
        logger.info(f"Predictions Generated: {len(predictions)}")
        logger.info(f"\n{predictions.to_string()}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise
