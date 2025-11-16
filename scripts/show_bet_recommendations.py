import glob
import os
from pathlib import Path

import pandas as pd

# Import database utilities
from src.utils.db_utils import DatabaseOperations, db_config
from src.utils.error_handlers import (
    DataValidationError,
    ErrorContext,
    log_dataframe_info,
    validate_dataframe,
)

# Import error handling and logging infrastructure
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Path to your latest enriched file (cross-platform compatible) ---
with ErrorContext("Finding enriched predictions file", logger=logger):
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Navigate to the 2026 directory (parent of src)
    base_repo = script_dir.parent
    directory_path = base_repo / "output" / "LightGBM"

    # Find the most recent enriched file
    enriched_files = list(directory_path.glob("combined_nba_predictions_enriched_*.csv"))
    if not enriched_files:
        raise FileNotFoundError(
            f"No enriched predictions file found in {directory_path}. "
            "Please run script 5 first to generate the enriched data."
        )
    # Sort by modification time and get the most recent
    enriched_path = max(enriched_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using enriched predictions file: {enriched_path}")


def main():
    """Main execution function for displaying proposed bets."""
    # --- Load and filter ---
    with ErrorContext("Loading and filtering betting data", logger=logger):
        # Try database first if enabled
        df = None
        if db_config.enabled:
            try:
                db_ops = DatabaseOperations()
                # Get latest enriched predictions from database
                # Note: This requires a view or query that joins predictions with enriched_predictions
                logger.info("Attempting to load enriched predictions from database...")
                # For now, fall back to CSV as we don't have a specific method for this yet
                # TODO: Add get_latest_enriched_predictions() method to DatabaseOperations
            except Exception as e:
                logger.warning(f"Failed to load from database: {e}")

        # Fall back to CSV
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.info(f"Loading enriched predictions from CSV: {enriched_path}")
            df = pd.read_csv(enriched_path)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        log_dataframe_info(df, name="Enriched predictions", logger=logger)

        bets = df[(df["stake_raw"] > 0) | (df["stake_platt"] > 0) | (df["stake_iso"] > 0)].copy()

        # --- Display summary ---
        cols = [
            "date",
            "home_team",
            "away_team",
            "odds_1",
            "home_team_prob",
            "prob_platt",
            "prob_iso",
            "win",
            "stake_raw",
            "pnl_raw",
            "stake_platt",
            "pnl_platt",
            "stake_iso",
            "pnl_iso",
        ]

        logger.info("\n=== Bets Placed (Raw / Platt / Iso Kelly) ===")
        if not bets.empty:
            logger.info(f"\n{bets[cols].sort_values('date').to_string(index=False)}")
            logger.info(f"Total bets found: {len(bets)}")
        else:
            logger.warning("No bets found in this enriched dataset.")


if __name__ == "__main__":
    try:
        main()
        logger.info("=" * 60)
        logger.info("Script 6 completed successfully")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR in Script 6")
        logger.error("=" * 60)
        logger.exception(f"Unexpected error: {e}")
        raise
    finally:
        # Keep the console window open so the user can read the logs.  In a non-interactive
        # environment (e.g. GitHub Actions), input() will raise EOFError, which we catch and ignore.
        in_ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
        if not in_ci:
            try:
                input("\nPress Enter to close this window...")
            except (EOFError, KeyboardInterrupt):
                pass
