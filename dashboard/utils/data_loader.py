"""
Data loading utilities for the dashboard.

Loads prediction data, statistics, and historical results from CSV files
or database (if configured).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import glob
import os

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "2026" / "data"
OUTPUT_DIR = ROOT_DIR / "2026" / "output"


def get_latest_file(pattern: str, directory: Path) -> Optional[Path]:
    """
    Find the most recent file matching a pattern.

    Args:
        pattern: Glob pattern (e.g., "nba_games_*.csv")
        directory: Directory to search

    Returns:
        Path to latest file or None if not found
    """
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_latest_predictions() -> Optional[pd.DataFrame]:
    """
    Load the most recent prediction file.

    Returns:
        DataFrame with predictions or None if not found
    """
    pred_file = get_latest_file("nba_games_predict_*.csv", OUTPUT_DIR)
    if pred_file is None:
        return None

    df = pd.read_csv(pred_file)
    return df


def load_enriched_predictions() -> Optional[pd.DataFrame]:
    """
    Load the most recent enriched predictions with betting suggestions.

    Returns:
        DataFrame with enriched predictions or None if not found
    """
    enrich_file = get_latest_file("combined_nba_predictions_enrich_*.csv", OUTPUT_DIR)
    if enrich_file is None:
        return None

    df = pd.read_csv(enrich_file)
    return df


def load_betting_statistics() -> Optional[pd.DataFrame]:
    """
    Load betting statistics and accuracy metrics.

    Returns:
        DataFrame with betting stats or None if not found
    """
    stats_file = get_latest_file("combined_nba_predictions_acc_*.csv", OUTPUT_DIR)
    if stats_file is None:
        return None

    df = pd.read_csv(stats_file)
    return df


def load_historical_games(days_back: int = 30) -> Optional[pd.DataFrame]:
    """
    Load historical game data from the past N days.

    Args:
        days_back: Number of days of history to load

    Returns:
        DataFrame with historical games or None if not found
    """
    all_games = []

    # Load all game files from the past N days
    for i in range(days_back):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        game_file = DATA_DIR / f"nba_games_{date_str}.csv"
        if game_file.exists():
            df = pd.read_csv(game_file)
            df['file_date'] = date_str
            all_games.append(df)

    if not all_games:
        return None

    return pd.concat(all_games, ignore_index=True)


def load_all_predictions(limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Load multiple prediction files for historical analysis.

    Args:
        limit: Maximum number of files to load

    Returns:
        DataFrame with all predictions or None if not found
    """
    pred_files = sorted(
        OUTPUT_DIR.glob("nba_games_predict_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]

    if not pred_files:
        return None

    all_preds = []
    for file in pred_files:
        df = pd.read_csv(file)
        # Extract date from filename: nba_games_predict_2025-10-23.csv
        date_str = file.stem.replace("nba_games_predict_", "")
        df['prediction_date'] = date_str
        all_preds.append(df)

    return pd.concat(all_preds, ignore_index=True)


def get_team_stats(team: str, df: pd.DataFrame) -> dict:
    """
    Calculate statistics for a specific team.

    Args:
        team: Team abbreviation (e.g., "LAL")
        df: DataFrame with game data

    Returns:
        Dictionary with team statistics
    """
    team_games = df[df['team'] == team]

    if len(team_games) == 0:
        return {}

    stats = {
        'games_played': len(team_games),
        'wins': team_games['won'].sum() if 'won' in team_games.columns else 0,
        'losses': len(team_games) - (team_games['won'].sum() if 'won' in team_games.columns else 0),
        'avg_points': team_games['pts'].mean() if 'pts' in team_games.columns else 0,
        'avg_opp_points': team_games['opp_pts'].mean() if 'opp_pts' in team_games.columns else 0,
    }

    if 'won' in team_games.columns and stats['games_played'] > 0:
        stats['win_pct'] = stats['wins'] / stats['games_played']
    else:
        stats['win_pct'] = 0.0

    return stats


def calculate_model_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate model performance metrics.

    Args:
        df: DataFrame with predictions and actual results

    Returns:
        Dictionary with performance metrics
    """
    metrics = {}

    if 'won' in df.columns and 'predicted_prob' in df.columns:
        # Accuracy
        df['predicted'] = (df['predicted_prob'] > 0.5).astype(int)
        metrics['accuracy'] = (df['predicted'] == df['won']).mean()

        # Brier score (lower is better)
        metrics['brier_score'] = ((df['predicted_prob'] - df['won']) ** 2).mean()

        # Log loss
        import numpy as np
        epsilon = 1e-15
        probs = df['predicted_prob'].clip(epsilon, 1 - epsilon)
        metrics['log_loss'] = -(df['won'] * np.log(probs) + (1 - df['won']) * np.log(1 - probs)).mean()

    if 'recommended_stake' in df.columns:
        # Calculate ROI
        df_with_stakes = df[df['recommended_stake'] > 0].copy()
        if len(df_with_stakes) > 0:
            # Simplified ROI calculation (would need actual bet outcomes)
            metrics['total_bets'] = len(df_with_stakes)
            metrics['avg_stake'] = df_with_stakes['recommended_stake'].mean()

    return metrics


def get_available_data_summary() -> dict:
    """
    Get summary of available data files.

    Returns:
        Dictionary with file counts and date ranges
    """
    summary = {
        'prediction_files': len(list(OUTPUT_DIR.glob("nba_games_predict_*.csv"))),
        'enriched_files': len(list(OUTPUT_DIR.glob("combined_nba_predictions_enrich_*.csv"))),
        'stats_files': len(list(OUTPUT_DIR.glob("combined_nba_predictions_acc_*.csv"))),
        'game_files': len(list(DATA_DIR.glob("nba_games_*.csv"))),
    }

    # Get date range of predictions
    pred_files = list(OUTPUT_DIR.glob("nba_games_predict_*.csv"))
    if pred_files:
        dates = []
        for f in pred_files:
            try:
                date_str = f.stem.replace("nba_games_predict_", "")
                dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            except:
                continue

        if dates:
            summary['earliest_prediction'] = min(dates).strftime("%Y-%m-%d")
            summary['latest_prediction'] = max(dates).strftime("%Y-%m-%d")

    return summary
