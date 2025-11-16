#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Migrate Game Statistics from CSV to PostgreSQL

This script migrates historical game statistics data from CSV files to the
PostgreSQL database. It processes all nba_games_*.csv files in the STAT_DIR
and uploads them to the game_statistics table.

Usage:
    python migrate_game_statistics.py [--dry-run] [--batch-size 1000]

Options:
    --dry-run       Show what would be migrated without actually inserting data
    --batch-size    Number of rows to insert in each batch (default: 1000)
    --latest-only   Migrate only the most recent CSV file
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../2026/src'))

from db_utils import DatabaseOperations, db_pool, db_config
from logger import get_logger
from error_handlers import ErrorContext, validate_dataframe, log_dataframe_info
from nba_utils_2026 import get_directory_paths

logger = get_logger(__name__)


def find_csv_files(stat_dir: str, latest_only: bool = False) -> List[str]:
    """Find all nba_games_*.csv files in the STAT_DIR.

    Args:
        stat_dir: Directory containing CSV files
        latest_only: If True, return only the most recent file

    Returns:
        List of CSV file paths, sorted by date
    """
    pattern = os.path.join(stat_dir, "nba_games_*.csv")
    files = glob.glob(pattern)

    if not files:
        logger.warning(f"No CSV files found matching pattern: {pattern}")
        return []

    # Sort by filename (which contains date in YYYY-MM-DD format)
    files.sort()

    logger.info(f"Found {len(files)} CSV files")

    if latest_only and files:
        logger.info(f"Latest only mode: using {os.path.basename(files[-1])}")
        return [files[-1]]

    return files


def prepare_dataframe_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for database insertion.

    Args:
        df: Raw DataFrame from CSV

    Returns:
        Cleaned DataFrame ready for database insertion
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # Convert date column to proper datetime
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date

    # Convert boolean columns
    if 'won' in df_clean.columns:
        df_clean['won'] = df_clean['won'].astype(bool)

    if 'home' in df_clean.columns:
        df_clean['home'] = df_clean['home'].astype(int)

    # Convert numeric columns, replacing NaN with None for database
    numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if col not in ['home', 'won']:  # Skip already converted columns
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)

    # Ensure season is string
    if 'season' in df_clean.columns:
        df_clean['season'] = df_clean['season'].astype(str)

    return df_clean


def migrate_csv_file(
    file_path: str,
    db_ops: DatabaseOperations,
    dry_run: bool = False,
    batch_size: int = 1000
) -> int:
    """Migrate a single CSV file to the database.

    Args:
        file_path: Path to CSV file
        db_ops: DatabaseOperations instance
        dry_run: If True, don't actually insert data
        batch_size: Number of rows per batch

    Returns:
        Number of rows processed
    """
    filename = os.path.basename(file_path)

    with ErrorContext(f"Migrating {filename}", logger=logger):
        # Read CSV file
        logger.info(f"Reading {file_path}")
        df = pd.read_csv(file_path)

        # Validate basic structure
        validate_dataframe(
            df,
            required_columns=['season', 'date', 'team', 'team_opp', 'home', 'won'],
            min_rows=1
        )

        log_dataframe_info(df, name=filename, logger=logger)

        # Prepare for database
        df_clean = prepare_dataframe_for_db(df)

        if dry_run:
            logger.info(f"[DRY RUN] Would insert {len(df_clean)} rows from {filename}")
            logger.info(f"[DRY RUN] Sample data:\n{df_clean.head(3)}")
            return len(df_clean)

        # Insert in batches
        total_rows = len(df_clean)
        rows_inserted = 0

        for i in range(0, total_rows, batch_size):
            batch = df_clean.iloc[i:i+batch_size]
            try:
                count = db_ops.save_game_statistics(batch)
                rows_inserted += count
                logger.info(
                    f"Progress: {rows_inserted}/{total_rows} rows "
                    f"({100*rows_inserted//total_rows}%)"
                )
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                raise

        logger.info(f"Successfully migrated {rows_inserted} rows from {filename}")
        return rows_inserted


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate game statistics from CSV to PostgreSQL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without inserting data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows to insert per batch (default: 1000)"
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Migrate only the most recent CSV file"
    )
    args = parser.parse_args()

    # Check if database is enabled
    if not db_config.enabled:
        logger.error("Database is not enabled. Set USE_DATABASE=true in .env")
        logger.error("See docs/DATABASE_SETUP.md for configuration instructions")
        return 1

    # Get directory paths
    paths = get_directory_paths()
    stat_dir = paths['STAT_DIR']

    logger.info("=" * 60)
    logger.info("Game Statistics Migration to PostgreSQL")
    logger.info("=" * 60)
    logger.info(f"Source directory: {stat_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Latest only: {args.latest_only}")
    logger.info("=" * 60)

    # Find CSV files
    csv_files = find_csv_files(stat_dir, latest_only=args.latest_only)

    if not csv_files:
        logger.error("No CSV files found to migrate")
        return 1

    # Initialize database
    try:
        db_pool.initialize()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    # Create database operations instance
    db_ops = DatabaseOperations()

    # Migrate each file
    total_rows = 0
    successful_files = 0
    failed_files = 0

    for csv_file in csv_files:
        try:
            rows = migrate_csv_file(csv_file, db_ops, args.dry_run, args.batch_size)
            total_rows += rows
            successful_files += 1
        except Exception as e:
            logger.error(f"Failed to migrate {os.path.basename(csv_file)}: {e}")
            failed_files += 1
            continue

    # Summary
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(csv_files)}")
    logger.info(f"Successful: {successful_files}")
    logger.info(f"Failed: {failed_files}")
    logger.info(f"Total rows: {total_rows}")

    if args.dry_run:
        logger.info("[DRY RUN] No data was actually inserted")

    logger.info("=" * 60)

    # Cleanup
    try:
        db_pool.close()
    except Exception as e:
        logger.warning(f"Error closing database pool: {e}")

    return 0 if failed_files == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error during migration: {e}")
        sys.exit(1)
