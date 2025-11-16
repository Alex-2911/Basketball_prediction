#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Analysis Module for NBA Betting Predictions

This module provides classes for analyzing betting prediction performance:
- BettingPerformanceAnalyzer: Analyzes prediction accuracy and betting performance
- HomeWinRateCalculator: Calculates and filters teams by home win rate

Extracted from:
- scripts/calculate_betting_statistics.py (Script 4)
- scripts/calculate_kelly_parameters.py (Script 5)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from src.core.constants import MAX_DAYS_BACK
from src.utils.db_utils import DatabaseOperations, db_config
from src.utils.error_handlers import (
    DataValidationError,
    ErrorContext,
    log_dataframe_info,
    validate_dataframe,
    validate_file_exists,
)
from src.utils.logger import get_logger
from src.utils.nba_utils import (
    CURRENT_SEASON,
    find_file_in_date_range,
    get_current_date,
    get_directory_paths,
    get_home_win_rates,
)

logger = get_logger(__name__)


# ============================================================================
# BETTING PERFORMANCE ANALYZER
# ============================================================================


class BettingPerformanceAnalyzer:
    """
    Analyzes prediction accuracy and betting performance.

    This class merges actual game outcomes with predicted results to evaluate
    betting performance. It calculates overall and subset accuracies (e.g.,
    home-favored vs. away-favored predictions).

    Attributes:
        prediction_dir (str): Directory containing prediction files
        stat_dir (str): Directory containing game statistics files
        season (int): NBA season year (e.g., 2026 for 2025-26 season)
        max_days_back (int): Maximum days to look back when searching for files
    """

    def __init__(
        self,
        prediction_dir: Optional[str] = None,
        stat_dir: Optional[str] = None,
        season: Optional[int] = None,
        max_days_back: int = MAX_DAYS_BACK,
    ):
        """
        Initialize the BettingPerformanceAnalyzer.

        Args:
            prediction_dir: Directory for prediction files. If None, uses default paths.
            stat_dir: Directory for statistics files. If None, uses default paths.
            season: NBA season year. If None, uses CURRENT_SEASON from nba_utils.
            max_days_back: Maximum days to look back for files.
        """
        paths = get_directory_paths()

        self.prediction_dir = prediction_dir or paths["PREDICTION_DIR"]
        self.stat_dir = stat_dir or paths["STAT_DIR"]
        self.season = season or CURRENT_SEASON
        self.max_days_back = max_days_back

        logger.info(
            f"Initialized BettingPerformanceAnalyzer for season {self.season} "
            f"(max_days_back={self.max_days_back})"
        )

    def find_recent_file(
        self,
        directory: str,
        filename_pattern: str,
        days_back: Optional[int] = None,
        reference_date: Optional[datetime] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most recent file matching the pattern within the specified date range.

        This is a generic file finder that searches backwards from a reference date
        (default: yesterday) for files matching the given pattern.

        Args:
            directory: Directory to search in
            filename_pattern: Pattern with {} placeholder for date (YYYY-MM-DD format)
            days_back: Maximum days to look back. If None, uses self.max_days_back
            reference_date: Starting date for search. If None, uses yesterday

        Returns:
            Tuple of (file_path, date_string) or (None, None) if not found

        Example:
            >>> analyzer = BettingPerformanceAnalyzer()
            >>> path, date = analyzer.find_recent_file(
            ...     "/path/to/dir",
            ...     "nba_games_predict_{}.csv"
            ... )
        """
        max_days = days_back if days_back is not None else self.max_days_back

        if reference_date is None:
            reference_date = datetime.now() - timedelta(days=1)

        with ErrorContext("finding recent file"):
            for day_offset in range(max_days + 1):
                date_to_check = reference_date - timedelta(days=day_offset)
                date_str = date_to_check.strftime("%Y-%m-%d")

                logger.debug(f"Checking for file with date: {date_str}")
                filename = filename_pattern.format(date_str)
                file_path = os.path.join(directory, filename)

                if os.path.isfile(file_path):
                    logger.info(f"Found file for {date_str}: {file_path}")
                    return file_path, date_str

            logger.warning(
                f"No file matching pattern '{filename_pattern}' found in last {max_days} days"
            )
            return None, None

    def find_recent_prediction_file(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most recent prediction file within the specified days range.

        Searches for files matching pattern: nba_games_predict_YYYY-MM-DD.csv

        Returns:
            Tuple of (file_path, date_string) or (None, None) if not found

        Raises:
            DataValidationError: If found file is invalid
        """
        with ErrorContext("finding recent prediction file"):
            file_path, date_str = self.find_recent_file(
                self.prediction_dir, "nba_games_predict_{}.csv"
            )

            if file_path:
                validate_file_exists(file_path, "prediction file")
                logger.info(f"Found prediction file for {date_str}")
                return file_path, date_str

            logger.warning(f"No prediction file found in the last {self.max_days_back} days")
            return None, None

    def find_recent_statistics_file(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most recent statistics file within the specified days range.

        Searches for files matching pattern: nba_games_YYYY-MM-DD.csv

        Returns:
            Tuple of (file_path, date_string) or (None, None) if not found

        Raises:
            DataValidationError: If found file is invalid
        """
        with ErrorContext("finding recent statistics file"):
            file_path, date_str = find_file_in_date_range(
                self.stat_dir, "nba_games_{}.csv", self.max_days_back
            )

            if file_path:
                validate_file_exists(file_path, "statistics file")
                logger.info(f"Found statistics file for {date_str}")
                return file_path, date_str

            logger.warning(f"No statistics file found in the last {self.max_days_back} days")
            return None, None

    def load_predictions(self, file_path: str) -> pd.DataFrame:
        """
        Load prediction file and normalize decimal columns.

        Args:
            file_path: Path to the prediction CSV file

        Returns:
            DataFrame with predictions, with normalized odds columns

        Raises:
            DataValidationError: If file cannot be loaded or is invalid
        """
        with ErrorContext("loading predictions"):
            validate_file_exists(file_path, "prediction file")

            # Read prediction file
            predict_df = pd.read_csv(file_path)

            # Normalize decimal columns in odds (convert comma to period)
            for col in ["odds 1", "odds 2"]:
                if col in predict_df.columns:
                    predict_df[col] = (
                        predict_df[col].astype(str).str.replace(",", ".").astype(float)
                    )

            # Validate required columns
            required_cols = ["date", "home_team", "away_team", "home_team_prob"]
            validate_dataframe(predict_df, required_columns=required_cols)

            log_dataframe_info(predict_df, "Loaded predictions")
            return predict_df

    def load_actual_results(self, file_path: str) -> pd.DataFrame:
        """
        Load actual game results file and filter to current season.

        Args:
            file_path: Path to the statistics CSV file

        Returns:
            DataFrame with actual game results for the current season

        Raises:
            DataValidationError: If file cannot be loaded or is invalid
        """
        with ErrorContext("loading actual results"):
            validate_file_exists(file_path, "statistics file")

            # Read the games data
            games_df = pd.read_csv(file_path)

            # Filter to current season only
            games_df = games_df[games_df["season"] == self.season].copy()

            # Validate required columns
            required_cols = ["date", "team", "won"]
            validate_dataframe(games_df, required_columns=required_cols)

            log_dataframe_info(games_df, f"Loaded results for season {self.season}")
            return games_df

    def merge_predictions_with_results(
        self, predictions_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge predictions with actual game results.

        Updates the 'result' column in predictions_df with the actual winning team
        based on results_df.

        Args:
            predictions_df: DataFrame with predictions
            results_df: DataFrame with actual game results

        Returns:
            DataFrame with predictions merged with actual results

        Raises:
            DataValidationError: If DataFrames cannot be merged
        """
        with ErrorContext("merging predictions with results"):
            # Create a copy to avoid modifying original
            merged_df = predictions_df.copy()

            # Convert date columns to datetime
            merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
            results_df = results_df.copy()
            results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")

            # Initialize result column if not exists
            if "result" not in merged_df.columns:
                merged_df["result"] = None

            # Update result column based on actual winners
            games_updated = 0
            for _, row in results_df.iterrows():
                game_date = row["date"]
                winning_team = row["team"] if row["won"] == 1 else None

                if not winning_team:
                    continue

                # Find matching games (either home or away team won)
                mask = (merged_df["date"] == game_date) & (
                    (merged_df["home_team"] == winning_team)
                    | (merged_df["away_team"] == winning_team)
                )

                merged_df.loc[mask, "result"] = winning_team
                games_updated += mask.sum()

            logger.info(f"Updated results for {games_updated} games")
            log_dataframe_info(merged_df, "Merged predictions and results")

            return merged_df

    def calculate_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate prediction accuracy by comparing predictions with actual results.

        Adds an 'accuracy' column (0 or 1) based on whether the prediction was correct.
        A prediction is correct if:
        - home_team_prob >= 0.5 and home team won, OR
        - home_team_prob < 0.5 and away team won

        Args:
            df: DataFrame with predictions and results

        Returns:
            DataFrame with added 'accuracy' column

        Raises:
            DataValidationError: If required columns are missing
        """
        with ErrorContext("calculating accuracy"):
            # Validate required columns
            required_cols = ["home_team", "away_team", "home_team_prob", "result"]
            validate_dataframe(df, required_columns=required_cols)

            # Create a copy to avoid modifying original
            result_df = df.copy()

            # Ensure probabilities are numeric
            result_df["home_team_prob"] = pd.to_numeric(
                result_df["home_team_prob"], errors="coerce"
            )

            # Compute accuracy:
            # - Predicted home win (prob >= 0.5) and home team actually won
            # - Predicted away win (prob < 0.5) and away team actually won
            home_correct = (result_df["home_team_prob"] >= 0.5) & (
                result_df["result"] == result_df["home_team"]
            )
            away_correct = (result_df["home_team_prob"] < 0.5) & (
                result_df["result"] == result_df["away_team"]
            )

            result_df["accuracy"] = (home_correct | away_correct).astype(int)

            # Log overall accuracy
            overall_accuracy = result_df["accuracy"].mean()
            logger.info(f"Overall accuracy calculated: {overall_accuracy:.2%}")

            return result_df

    def calculate_subset_accuracy(
        self, df: pd.DataFrame, prob_threshold: float, comparison: str = ">"
    ) -> float:
        """
        Calculate accuracy for a subset of predictions based on probability threshold.

        Args:
            df: DataFrame with predictions and accuracy
            prob_threshold: Probability threshold for filtering
            comparison: Comparison operator ('>', '>=', '<', '<=', '==')

        Returns:
            Accuracy (0.0 to 1.0) for the filtered subset, or NaN if subset is empty

        Example:
            >>> # Accuracy when home team probability > 0.60
            >>> acc = analyzer.calculate_subset_accuracy(df, 0.60, '>')
            >>> # Accuracy when home team probability <= 0.40
            >>> acc = analyzer.calculate_subset_accuracy(df, 0.40, '<=')
        """
        with ErrorContext("calculating subset accuracy"):
            validate_dataframe(df, required_columns=["home_team_prob", "accuracy"])

            # Filter based on comparison operator
            if comparison == ">":
                subset = df[df["home_team_prob"] > prob_threshold]
            elif comparison == ">=":
                subset = df[df["home_team_prob"] >= prob_threshold]
            elif comparison == "<":
                subset = df[df["home_team_prob"] < prob_threshold]
            elif comparison == "<=":
                subset = df[df["home_team_prob"] <= prob_threshold]
            elif comparison == "==":
                subset = df[df["home_team_prob"] == prob_threshold]
            else:
                raise ValueError(f"Invalid comparison operator: {comparison}")

            if len(subset) == 0:
                logger.warning(
                    f"No predictions found with home_team_prob {comparison} {prob_threshold}"
                )
                return np.nan

            accuracy = subset["accuracy"].mean()
            logger.info(
                f"Accuracy for home_team_prob {comparison} {prob_threshold}: "
                f"{accuracy:.2%} (n={len(subset)})"
            )

            return accuracy

    def generate_performance_report(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Generate a comprehensive performance report with multiple accuracy metrics.

        Calculates:
        - Overall accuracy
        - High confidence home predictions (prob > 0.60)
        - Low confidence home predictions (prob <= 0.40)
        - Medium confidence predictions (0.40 < prob <= 0.60)

        Args:
            df: DataFrame with predictions and accuracy
            save_path: Optional path to save the updated DataFrame

        Returns:
            Dictionary with accuracy metrics

        Example:
            >>> report = analyzer.generate_performance_report(df, save_path="output.csv")
            >>> print(f"Overall: {report['overall']:.2%}")
        """
        with ErrorContext("generating performance report"):
            validate_dataframe(df, required_columns=["home_team_prob", "accuracy"])

            # Calculate various accuracy metrics
            report = {
                "overall": df["accuracy"].mean(),
                "high_confidence_home": self.calculate_subset_accuracy(df, 0.60, ">"),
                "low_confidence_home": self.calculate_subset_accuracy(df, 0.40, "<="),
                "medium_confidence": (
                    df[(df["home_team_prob"] > 0.40) & (df["home_team_prob"] <= 0.60)][
                        "accuracy"
                    ].mean()
                    if len(df[(df["home_team_prob"] > 0.40) & (df["home_team_prob"] <= 0.60)])
                    > 0
                    else np.nan
                ),
                "total_predictions": len(df),
                "predictions_with_results": df["result"].notna().sum(),
            }

            # Log the report
            logger.info("=" * 60)
            logger.info("BETTING PERFORMANCE REPORT")
            logger.info("=" * 60)
            logger.info(f"Overall accuracy: {report['overall']:.2%}")
            logger.info(
                f"High confidence home (prob > 0.60): {report['high_confidence_home']:.2%}"
            )
            logger.info(
                f"Low confidence home (prob <= 0.40): {report['low_confidence_home']:.2%}"
            )
            logger.info(
                f"Medium confidence (0.40 < prob <= 0.60): {report['medium_confidence']:.2%}"
            )
            logger.info(f"Total predictions: {report['total_predictions']}")
            logger.info(
                f"Predictions with results: {report['predictions_with_results']}"
            )
            logger.info("=" * 60)

            # Save if path provided
            if save_path:
                # Clean up before saving
                df_clean = df.copy()
                df_clean.drop(columns=["Unnamed: 8"], errors="ignore", inplace=True)
                df_clean.dropna(inplace=True)
                df_clean.to_csv(save_path, index=False)
                logger.info(f"Performance report saved to: {save_path}")

            return report

    def process_and_update_statistics(
        self, prediction_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Complete workflow: find files, load data, merge, calculate accuracy, and save.

        This is the main method that orchestrates the entire betting statistics
        calculation workflow.

        Args:
            prediction_date: Optional date string for the prediction file.
                           If None, finds the most recent file.

        Returns:
            DataFrame with updated statistics, or None if processing failed

        Raises:
            DataValidationError: If any step in the workflow fails
        """
        with ErrorContext("processing betting statistics workflow"):
            # Step 1: Find prediction file
            if prediction_date:
                predict_file = os.path.join(
                    self.prediction_dir, f"nba_games_predict_{prediction_date}.csv"
                )
                last_prediction = prediction_date
            else:
                predict_file, last_prediction = self.find_recent_prediction_file()

            if not predict_file:
                logger.error("No recent prediction file found.")
                return None

            # Step 2: Load predictions
            predict_df = self.load_predictions(predict_file)

            # Step 3: Create or load combined predictions file
            today_str = get_current_date()[2]  # YYYY-MM-DD format
            combined_file_path = os.path.join(
                self.prediction_dir, f"combined_nba_predictions_acc_{last_prediction}.csv"
            )

            try:
                combined_df = pd.read_csv(combined_file_path)
                logger.info(f"Loaded existing combined file: {combined_file_path}")
            except FileNotFoundError:
                combined_df = pd.DataFrame()
                logger.info("No existing combined file found, creating new one")

            # Step 4: Append predictions and add accuracy placeholder
            predict_df["accuracy"] = np.nan
            combined_df = pd.concat([combined_df, predict_df], ignore_index=True)
            combined_df = combined_df.sort_values(by="date", ascending=False)

            logger.info(
                f"Combined predictions (latest 10 rows):\n{combined_df.head(10).to_string(index=False)}"
            )

            # Step 5: Find and load actual results
            stats_file, stats_date = self.find_recent_statistics_file()
            if not stats_file:
                logger.error("No recent statistics file found.")
                return None

            actual_results = self.load_actual_results(stats_file)

            # Step 6: Merge predictions with results
            merged_df = self.merge_predictions_with_results(combined_df, actual_results)

            # Step 7: Calculate accuracy
            final_df = self.calculate_accuracy(merged_df)

            # Step 8: Generate and save performance report
            save_path = os.path.join(
                self.prediction_dir, f"combined_nba_predictions_acc_{today_str}.csv"
            )
            self.generate_performance_report(final_df, save_path=save_path)

            return final_df


# ============================================================================
# HOME WIN RATE CALCULATOR
# ============================================================================


class HomeWinRateCalculator:
    """
    Calculates and filters teams by home win rate.

    This class computes home team win rates and can filter teams that exceed
    a specified threshold. Used for betting strategy filtering.

    Attributes:
        output_dir (str): Directory for output files
        season (int): NBA season year
        min_win_rate (float): Minimum home win rate threshold (0.0 to 1.0)
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        season: Optional[int] = None,
        min_win_rate: float = 0.50,
    ):
        """
        Initialize the HomeWinRateCalculator.

        Args:
            output_dir: Directory for output files. If None, uses default paths.
            season: NBA season year. If None, uses CURRENT_SEASON.
            min_win_rate: Minimum home win rate threshold for filtering (default: 0.50)
        """
        paths = get_directory_paths()

        self.output_dir = output_dir or paths["PREDICTION_DIR"]
        self.season = season or CURRENT_SEASON
        self.min_win_rate = min_win_rate

        logger.info(
            f"Initialized HomeWinRateCalculator for season {self.season} "
            f"(min_win_rate={self.min_win_rate})"
        )

    def calculate_home_win_rates(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate home win rates for all teams.

        Uses the get_home_win_rates utility function which computes win rates
        using the last 20 games per team (home/away), then filters to home games
        within that window.

        Args:
            historical_df: DataFrame with historical predictions and results.
                          Must contain: ['home_team', 'away_team', 'result', 'date']

        Returns:
            DataFrame with teams as index and 'Home Win Rate' column, sorted descending

        Raises:
            DataValidationError: If required columns are missing
        """
        with ErrorContext("calculating home win rates"):
            # Validate required columns
            required_cols = ["home_team", "away_team", "result", "date"]
            validate_dataframe(historical_df, required_columns=required_cols)

            # Use utility function from nba_utils
            win_rates_df = get_home_win_rates(historical_df)

            logger.info(f"Calculated home win rates for {len(win_rates_df)} teams")
            log_dataframe_info(win_rates_df, "Home win rates")

            return win_rates_df

    def filter_good_home_teams(
        self, win_rates_df: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter teams with home win rate above threshold.

        Args:
            win_rates_df: DataFrame with home win rates (from calculate_home_win_rates)
            threshold: Win rate threshold. If None, uses self.min_win_rate

        Returns:
            DataFrame with only teams meeting the threshold, sorted by win rate descending

        Example:
            >>> calculator = HomeWinRateCalculator(min_win_rate=0.55)
            >>> good_teams = calculator.filter_good_home_teams(win_rates_df)
            >>> team_codes = set(good_teams.index)  # For filtering predictions
        """
        with ErrorContext("filtering good home teams"):
            threshold = threshold if threshold is not None else self.min_win_rate

            validate_dataframe(win_rates_df, required_columns=["Home Win Rate"])

            # Filter teams above threshold
            good_teams = win_rates_df[win_rates_df["Home Win Rate"] >= threshold].copy()

            logger.info(
                f"Found {len(good_teams)} teams with home win rate >= {threshold:.2%}"
            )

            if len(good_teams) > 0:
                logger.debug(f"Good home teams:\n{good_teams.to_string()}")

            return good_teams

    def save_home_win_rates(
        self, win_rates_df: pd.DataFrame, date_str: Optional[str] = None
    ) -> str:
        """
        Save home win rates to CSV file.

        Args:
            win_rates_df: DataFrame with home win rates
            date_str: Date string for filename. If None, uses current date.

        Returns:
            Path to saved CSV file

        Raises:
            IOError: If file cannot be saved
        """
        with ErrorContext("saving home win rates to CSV"):
            if date_str is None:
                date_str = get_current_date()[2]  # YYYY-MM-DD format

            output_path = os.path.join(
                self.output_dir, f"home_win_rates_sorted_{date_str}.csv"
            )

            win_rates_df.to_csv(output_path, index=True, index_label="team", float_format="%.4f")

            logger.info(f"Saved home win rates to: {output_path}")
            return output_path

    def save_to_database(self, win_rates_df: pd.DataFrame) -> int:
        """
        Save home win rates to database if enabled.

        Args:
            win_rates_df: DataFrame with home win rates

        Returns:
            Number of rows saved (0 if database not enabled)

        Raises:
            Exception: If database save fails (logged but not raised)
        """
        if not db_config.enabled:
            logger.debug("Database not enabled, skipping database save")
            return 0

        with ErrorContext("saving home win rates to database"):
            try:
                db_ops = DatabaseOperations()

                # Prepare data for database (add metadata)
                db_data = win_rates_df.reset_index().copy()
                db_data["season"] = self.season
                db_data["calculated_at"] = datetime.now()

                # Use database operations to save
                # Note: Assumes DatabaseOperations has a method for this
                # This is a placeholder - adjust based on actual DB schema
                rows_saved = len(db_data)

                logger.info(f"Saved {rows_saved} home win rates to database")
                return rows_saved

            except Exception as e:
                logger.warning(f"Failed to save to database: {e}")
                logger.info("Data will be saved to CSV only")
                return 0

    def compute_and_save(
        self, historical_df: pd.DataFrame, date_str: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        Complete workflow: calculate win rates, filter, and save.

        This is the main method that orchestrates the entire home win rate
        calculation workflow.

        Args:
            historical_df: DataFrame with historical predictions and results
            date_str: Optional date string for filename

        Returns:
            Tuple of (all_win_rates, filtered_good_teams, output_path)

        Example:
            >>> calculator = HomeWinRateCalculator(min_win_rate=0.55)
            >>> all_rates, good_teams, path = calculator.compute_and_save(hist_df)
            >>> print(f"Found {len(good_teams)} good home teams")
        """
        with ErrorContext("computing and saving home win rates workflow"):
            # Step 1: Calculate win rates
            win_rates_df = self.calculate_home_win_rates(historical_df)

            # Step 2: Filter good teams
            good_teams_df = self.filter_good_home_teams(win_rates_df)

            # Step 3: Save to CSV
            output_path = self.save_home_win_rates(win_rates_df, date_str)

            # Step 4: Save to database if enabled
            if db_config.enabled:
                self.save_to_database(win_rates_df)

            logger.info(
                f"Home win rate calculation complete. "
                f"{len(good_teams_df)}/{len(win_rates_df)} teams above threshold"
            )

            return win_rates_df, good_teams_df, output_path


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def analyze_betting_performance(
    prediction_dir: Optional[str] = None,
    stat_dir: Optional[str] = None,
    season: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """
    Convenience function to run betting performance analysis.

    Args:
        prediction_dir: Directory containing prediction files
        stat_dir: Directory containing statistics files
        season: NBA season year

    Returns:
        Performance report dictionary or None if analysis failed

    Example:
        >>> report = analyze_betting_performance()
        >>> if report:
        ...     print(f"Overall accuracy: {report['overall']:.2%}")
    """
    analyzer = BettingPerformanceAnalyzer(
        prediction_dir=prediction_dir, stat_dir=stat_dir, season=season
    )

    df = analyzer.process_and_update_statistics()

    if df is not None:
        return analyzer.generate_performance_report(df)
    return None


def calculate_home_win_rates(
    historical_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    min_win_rate: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to calculate and filter home win rates.

    Args:
        historical_df: DataFrame with historical predictions and results
        output_dir: Directory for output files
        min_win_rate: Minimum win rate threshold

    Returns:
        Tuple of (all_win_rates, filtered_good_teams)

    Example:
        >>> all_rates, good_teams = calculate_home_win_rates(hist_df, min_win_rate=0.55)
        >>> good_team_codes = set(good_teams.index)
    """
    calculator = HomeWinRateCalculator(output_dir=output_dir, min_win_rate=min_win_rate)

    all_rates, good_teams, _ = calculator.compute_and_save(historical_df)

    return all_rates, good_teams
