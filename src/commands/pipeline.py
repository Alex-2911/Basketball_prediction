"""
Pipeline orchestration

Coordinates the complete NBA prediction workflow.
"""

from src.utils.logger import get_logger
from .analyze import run_all_analysis
from .collect import run_historical_collection, run_upcoming_collection
from .predict import run_prediction

logger = get_logger(__name__)


def run_full_pipeline(
    skip_collection: bool = False,
    skip_analysis: bool = False
) -> bool:
    """
    Run the complete NBA prediction pipeline

    Pipeline steps:
    1. Collect historical game data
    2. Collect upcoming game schedule
    3. Generate predictions
    4. Calculate statistics
    5. Calculate Kelly parameters
    6. Generate recommendations

    Args:
        skip_collection: Skip data collection steps (use existing data)
        skip_analysis: Skip analysis steps (only collect and predict)

    Returns:
        True if all steps succeeded, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Starting NBA Prediction Pipeline")
    logger.info("=" * 60)

    # Step 1 & 2: Data Collection
    if not skip_collection:
        logger.info("\n📊 STEP 1/6: Collecting historical game data")
        logger.info("-" * 60)
        if not run_historical_collection():
            logger.error("Historical data collection failed")
            return False

        logger.info("\n📅 STEP 2/6: Collecting upcoming game schedule")
        logger.info("-" * 60)
        if not run_upcoming_collection():
            logger.error("Upcoming games collection failed")
            return False
    else:
        logger.info("\n⏭️  STEP 1-2/6: Skipping data collection (using existing data)")

    # Step 3: Predictions
    logger.info("\n🔮 STEP 3/6: Generating predictions")
    logger.info("-" * 60)
    if not run_prediction():
        logger.error("Prediction generation failed")
        return False

    # Step 4-6: Analysis
    if not skip_analysis:
        logger.info("\n📈 STEP 4/6: Calculating betting statistics")
        logger.info("-" * 60)

        logger.info("\n💰 STEP 5/6: Calculating Kelly Criterion parameters")
        logger.info("-" * 60)

        logger.info("\n🎯 STEP 6/6: Generating betting recommendations")
        logger.info("-" * 60)

        if not run_all_analysis():
            logger.error("Betting analysis failed")
            return False
    else:
        logger.info("\n⏭️  STEP 4-6/6: Skipping analysis")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)

    return True
