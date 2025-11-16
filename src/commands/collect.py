"""
Data collection commands

Provides CLI interface for data collection operations using core modules.
"""

from datetime import datetime, date, timedelta
from typing import Optional

from src.core.collector import HistoricalGameCollector, UpcomingGameCollector
from src.utils.logger import get_logger
from src.utils.nba_utils import CURRENT_SEASON

logger = get_logger(__name__)


def run_historical_collection(
    date_str: Optional[str] = None,
    collect_date: Optional[str] = None
) -> bool:
    """
    Collect historical game data using HistoricalGameCollector

    Args:
        date_str: Anchor date in YYYY-MM-DD; collects games from the day before
        collect_date: Exact game date to collect in YYYY-MM-DD (overrides date_str)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Starting historical data collection...")

        # Determine the collection date
        if collect_date:
            collection_date = datetime.strptime(collect_date, "%Y-%m-%d").date()
        elif date_str:
            anchor = datetime.strptime(date_str, "%Y-%m-%d").date()
            collection_date = anchor - timedelta(days=1)
        else:
            # Default: collect yesterday's games
            collection_date = date.today() - timedelta(days=1)

        logger.info(f"Collecting games for {collection_date}")

        # Use the core module
        collector = HistoricalGameCollector(season=CURRENT_SEASON)
        games_df = collector.collect_games_for_date(collection_date)

        if games_df is not None and not games_df.empty:
            # Save to database if configured
            collector.save_to_database(games_df)
            logger.info(f"✅ Successfully collected {len(games_df)} games")
            return True
        else:
            logger.warning("No games found for the specified date")
            return False

    except Exception as e:
        logger.error(f"Error during historical collection: {e}", exc_info=True)
        return False


def run_upcoming_collection(date_str: Optional[str] = None) -> bool:
    """
    Collect upcoming game schedule using UpcomingGameCollector

    Args:
        date_str: Date for which to collect upcoming games (YYYY-MM-DD)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Starting upcoming games collection...")

        # Determine the target date
        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            # Default: today
            target_date = datetime.now()

        logger.info(f"Finding games on or after {target_date.date()}")

        # Use the core module
        collector = UpcomingGameCollector(season=CURRENT_SEASON)
        games_df = collector.find_next_games(target_date)

        if games_df is not None and not games_df.empty:
            # Save to database if configured
            collector.save_to_database(games_df)
            logger.info(f"✅ Successfully found {len(games_df)} upcoming games")
            return True
        else:
            logger.warning("No upcoming games found")
            return False

    except Exception as e:
        logger.error(f"Error during upcoming games collection: {e}", exc_info=True)
        return False
