#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Migrate Enriched Predictions (Kelly Stakes & PnL) to PostgreSQL

This script migrates enriched prediction data (with Kelly criterion stakes and
profit/loss calculations) from CSV files to the PostgreSQL database.

Usage:
    python migrate_enriched_predictions.py [--dry-run] [--batch-size 500]

Options:
    --dry-run       Show what would be migrated without inserting data
    --batch-size    Number of rows to insert in each batch (default: 500)
    --latest-only   Migrate only the most recent enriched predictions file
"""

import os
import sys
import argparse
import glob
from datetime import datetime
from typing import List, Optional

import pandas as pd

# Add parent directory to path for imports

from src.utils.db_utils import DatabaseOperations, db_pool, db_config
from src.utils.logger import get_logger
from src.utils.error_handlers import ErrorContext, validate_dataframe, log_dataframe_info
from src.utils.nba_utils import get_directory_paths

logger = get_logger(__name__)


def find_enriched_files(enrich_dir: str, latest_only: bool = False) -> List[str]:
    """Find all enriched_predictions_*.csv files.

    Args:
        enrich_dir: Directory containing enriched prediction files
        latest_only: If True, return only the most recent file

    Returns:
        List of CSV file paths, sorted by date
    """
    pattern = os.path.join(enrich_dir, "enriched_predictions_*.csv")
    files = glob.glob(pattern)

    if not files:
        logger.warning(f"No enriched prediction files found: {pattern}")
        return []

    files.sort()
    logger.info(f"Found {len(files)} enriched prediction files")

    if latest_only and files:
        logger.info(f"Latest only: using {os.path.basename(files[-1])}")
        return [files[-1]]

    return files


def prepare_enriched_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare enriched predictions DataFrame for database insertion.

    Args:
        df: Raw DataFrame from CSV

    Returns:
        Cleaned DataFrame ready for database
    """
    df_clean = df.copy()

    # Convert date columns
    for date_col in ['date', 'game_date', 'prediction_date']:
        if date_col in df_clean.columns:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col]).dt.date

    # Kelly stake columns
    stake_cols = ['stake_raw', 'stake_platt', 'stake_iso']
    for col in stake_cols:
        if col in df_clean.columns:
            # Replace NaN with 0 for stakes
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = 0

    # PnL columns
    pnl_cols = ['pnl_raw', 'pnl_platt', 'pnl_iso']
    for col in pnl_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = 0

    # Probability columns
    prob_cols = ['prob_platt', 'prob_iso']
    for col in prob_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)

    # Numeric columns that can be NULL
    numeric_nullable = ['odds_1', 'odds_2', 'home_team_prob']
    for col in numeric_nullable:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)

    # Edge columns (expected value)
    edge_cols = ['edge_raw', 'edge_platt', 'edge_iso']
    for col in edge_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)

    # Boolean columns
    if 'result' in df_clean.columns:
        # Convert to string representation for database
        df_clean['result'] = df_clean['result'].astype(str).replace({'nan': None, 'None': None})

    return df_clean


def match_predictions_to_ids(
    enriched_df: pd.DataFrame,
    db_ops: DatabaseOperations
) -> pd.DataFrame:
    """Match enriched predictions to existing prediction IDs in database.

    Args:
        enriched_df: DataFrame with enriched predictions
        db_ops: DatabaseOperations instance

    Returns:
        DataFrame with prediction_id column added
    """
    logger.info("Matching enriched predictions to existing prediction IDs...")

    # Get all predictions from database
    with ErrorContext("Fetching existing predictions", logger=logger):
        existing_preds = db_ops.get_all_predictions_for_matching()

    if existing_preds.empty:
        logger.warning("No existing predictions in database - enriched data needs predictions first")
        raise ValueError(
            "Database contains no predictions. "
            "Run migrate_predictions.py before migrating enriched predictions."
        )

    # Merge on game identifiers
    merge_cols = ['home_team', 'away_team', 'date']

    # Ensure date columns are same type
    enriched_df['date'] = pd.to_datetime(enriched_df['date']).dt.date
    existing_preds['date'] = pd.to_datetime(existing_preds['date']).dt.date

    # Merge to get prediction IDs
    matched = enriched_df.merge(
        existing_preds[['id', 'home_team', 'away_team', 'date']],
        on=merge_cols,
        how='left'
    )

    # Check for unmatched rows
    unmatched = matched['id'].isna().sum()
    if unmatched > 0:
        logger.warning(f"{unmatched} enriched predictions could not be matched to existing predictions")
        logger.warning("These rows will be skipped")

    # Rename id column to prediction_id
    matched.rename(columns={'id': 'prediction_id'}, inplace=True)

    # Drop rows without prediction_id
    matched = matched[matched['prediction_id'].notna()]

    logger.info(f"Successfully matched {len(matched)} enriched predictions")

    return matched


def migrate_enriched_file(
    file_path: str,
    db_ops: DatabaseOperations,
    dry_run: bool = False,
    batch_size: int = 500
) -> int:
    """Migrate a single enriched predictions CSV file.

    Args:
        file_path: Path to enriched CSV
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
        df_clean = prepare_enriched_for_db(df)

        # Match to existing prediction IDs
        if not dry_run:
            df_clean = match_predictions_to_ids(df_clean, db_ops)

            if df_clean.empty:
                logger.warning("No enriched predictions could be matched - skipping file")
                return 0

        if dry_run:
            logger.info(f"[DRY RUN] Would insert {len(df_clean)} enriched predictions")
            logger.info(f"[DRY RUN] Sample:\n{df_clean.head(3)}")
            return len(df_clean)

        # Insert in batches
        total_rows = len(df_clean)
        rows_inserted = 0

        for i in range(0, total_rows, batch_size):
            batch = df_clean.iloc[i:i+batch_size]
            try:
                count = db_ops.save_enriched_predictions(batch)
                rows_inserted += count
                logger.info(f"Progress: {rows_inserted}/{total_rows} rows")
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                raise

        logger.info(f"Successfully migrated {rows_inserted} enriched predictions")
        return rows_inserted


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate enriched predictions (Kelly stakes) to PostgreSQL"
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
        logger.error("See docs/DATABASE_SETUP.md for setup")
        return 1

    # Get paths
    paths = get_directory_paths()
    enrich_dir = paths['ENRICHED_DIR']

    logger.info("=" * 60)
    logger.info("Enriched Predictions Migration to PostgreSQL")
    logger.info("=" * 60)
    logger.info(f"Source directory: {enrich_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Find files
    csv_files = find_enriched_files(enrich_dir, args.latest_only)

    if not csv_files:
        logger.error("No enriched prediction files found")
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
            rows = migrate_enriched_file(csv_file, db_ops, args.dry_run, args.batch_size)
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
    logger.info(f"Total enriched predictions: {total_rows}")

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
