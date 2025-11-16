#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Migrate Predictions from CSV to PostgreSQL

This script migrates prediction data from CSV files to the PostgreSQL database.
It processes predictions_*.csv files and uploads them to the predictions table.

Usage:
    python migrate_predictions.py [--dry-run] [--batch-size 500]

Options:
    --dry-run       Show what would be migrated without actually inserting data
    --batch-size    Number of rows to insert in each batch (default: 500)
    --latest-only   Migrate only the most recent predictions file
"""

import os
import sys
import argparse
import glob
from datetime import datetime
from typing import List

import pandas as pd

# Add parent directory to path for imports

from src.utils.db_utils import DatabaseOperations, db_pool, db_config
from src.utils.logger import get_logger
from src.utils.error_handlers import ErrorContext, validate_dataframe, log_dataframe_info
from src.utils.nba_utils import get_directory_paths

logger = get_logger(__name__)


def find_prediction_files(pred_dir: str, latest_only: bool = False) -> List[str]:
    """Find all predictions_*.csv files.

    Args:
        pred_dir: Directory containing prediction CSV files
        latest_only: If True, return only the most recent file

    Returns:
        List of CSV file paths, sorted by date
    """
    pattern = os.path.join(pred_dir, "predictions_*.csv")
    files = glob.glob(pattern)

    if not files:
        logger.warning(f"No prediction files found matching: {pattern}")
        return []

    files.sort()
    logger.info(f"Found {len(files)} prediction files")

    if latest_only and files:
        logger.info(f"Latest only mode: using {os.path.basename(files[-1])}")
        return [files[-1]]

    return files


def prepare_predictions_for_db(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Prepare predictions DataFrame for database insertion.

    Args:
        df: Raw DataFrame from CSV
        source_file: Source filename (used to extract prediction_date)

    Returns:
        Cleaned DataFrame ready for database
    """
    df_clean = df.copy()

    # Convert date column
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date
    elif 'game_date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['game_date']).dt.date
        df_clean.drop('game_date', axis=1, inplace=True)

    # Extract prediction date from filename (e.g., predictions_2025-10-22.csv)
    try:
        basename = os.path.basename(source_file)
        date_str = basename.replace('predictions_', '').replace('.csv', '')
        prediction_date = pd.to_datetime(date_str).date()
        df_clean['prediction_date'] = prediction_date
    except Exception as e:
        logger.warning(f"Could not extract date from filename {source_file}: {e}")
        df_clean['prediction_date'] = datetime.now().date()

    # Ensure probability is in [0, 1] range
    if 'home_team_prob' in df_clean.columns:
        df_clean['home_team_prob'] = df_clean['home_team_prob'].clip(0, 1)

    # Handle odds columns
    for col in ['odds_1', 'odds_2']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)

    # Handle result column
    if 'result' not in df_clean.columns:
        df_clean['result'] = None

    # Handle model_version
    if 'model_version' not in df_clean.columns:
        df_clean['model_version'] = 'legacy'

    return df_clean


def migrate_prediction_file(
    file_path: str,
    db_ops: DatabaseOperations,
    dry_run: bool = False,
    batch_size: int = 500
) -> int:
    """Migrate a single prediction CSV file to database.

    Args:
        file_path: Path to prediction CSV
        db_ops: DatabaseOperations instance
        dry_run: If True, don't insert data
        batch_size: Rows per batch

    Returns:
        Number of rows processed
    """
    filename = os.path.basename(file_path)

    with ErrorContext(f"Migrating {filename}", logger=logger):
        logger.info(f"Reading {file_path}")
        df = pd.read_csv(file_path)

        # Validate structure
        validate_dataframe(
            df,
            required_columns=['home_team', 'away_team'],
            min_rows=1
        )

        log_dataframe_info(df, name=filename, logger=logger)

        # Prepare for database
        df_clean = prepare_predictions_for_db(df, file_path)

        if dry_run:
            logger.info(f"[DRY RUN] Would insert {len(df_clean)} predictions from {filename}")
            logger.info(f"[DRY RUN] Sample:\n{df_clean.head(3)}")
            return len(df_clean)

        # Insert in batches
        total_rows = len(df_clean)
        rows_inserted = 0

        for i in range(0, total_rows, batch_size):
            batch = df_clean.iloc[i:i+batch_size]
            try:
                count = db_ops.save_predictions(batch)
                rows_inserted += count
                logger.info(f"Progress: {rows_inserted}/{total_rows} rows")
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                raise

        logger.info(f"Successfully migrated {rows_inserted} predictions from {filename}")
        return rows_inserted


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate predictions from CSV to PostgreSQL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without inserting data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per batch (default: 500)"
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Migrate only the most recent file"
    )
    args = parser.parse_args()

    # Check database enabled
    if not db_config.enabled:
        logger.error("Database not enabled. Set USE_DATABASE=true in .env")
        logger.error("See docs/DATABASE_SETUP.md for setup instructions")
        return 1

    # Get paths
    paths = get_directory_paths()
    pred_dir = paths['PRED_DIR']

    logger.info("=" * 60)
    logger.info("Predictions Migration to PostgreSQL")
    logger.info("=" * 60)
    logger.info(f"Source directory: {pred_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Find files
    csv_files = find_prediction_files(pred_dir, args.latest_only)

    if not csv_files:
        logger.error("No prediction files found")
        return 1

    # Initialize database
    try:
        db_pool.initialize()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    db_ops = DatabaseOperations()

    # Migrate files
    total_rows = 0
    successful = 0
    failed = 0

    for csv_file in csv_files:
        try:
            rows = migrate_prediction_file(csv_file, db_ops, args.dry_run, args.batch_size)
            total_rows += rows
            successful += 1
        except Exception as e:
            logger.error(f"Failed to migrate {os.path.basename(csv_file)}: {e}")
            failed += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Files processed: {len(csv_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total predictions: {total_rows}")

    if args.dry_run:
        logger.info("[DRY RUN] No data was inserted")

    logger.info("=" * 60)

    try:
        db_pool.close()
    except Exception as e:
        logger.warning(f"Error closing pool: {e}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Migration interrupted")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
