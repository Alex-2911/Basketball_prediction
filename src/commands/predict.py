"""
Prediction command

Generates predictions for upcoming games using core predictor and betting modules.
"""

import os
from datetime import date
from typing import Optional

from src.core.betting import OddsManager, fetch_and_merge_odds
from src.core.predictor import (
    GameDataPreprocessor,
    LightGBMPredictor,
    MatchupBuilder,
    get_directory_paths,
)
from src.utils.logger import get_logger
from src.utils.nba_utils import CURRENT_SEASON

logger = get_logger(__name__)


def run_prediction(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> bool:
    """
    Generate predictions for upcoming games

    Args:
        model_path: Path to saved LightGBM model (optional, not yet implemented)
        output_dir: Directory for prediction output files (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("🚀 Starting prediction generation...")

        # Get today's date
        today = date.today()
        today_str = today.strftime("%Y-%m-%d")

        # Initialize components
        preprocessor = GameDataPreprocessor(season=CURRENT_SEASON)
        builder = MatchupBuilder()
        predictor = LightGBMPredictor()

        # Step 1: Load data
        logger.info("📊 Loading data...")
        try:
            games_df = preprocessor.load_upcoming_games(today_str)
        except FileNotFoundError:
            logger.error("No upcoming games file found. Run 'nba-predict collect upcoming' first")
            return False

        try:
            stats_df = preprocessor.load_historical_stats(today_str)
        except FileNotFoundError:
            logger.error("No historical stats file found. Run 'nba-predict collect historical' first")
            return False

        if games_df.empty:
            logger.warning("No upcoming games to predict")
            return True  # Not an error, just no games

        # Step 2: Preprocess data
        logger.info("⚙️  Preprocessing data...")
        df = preprocessor.preprocess_pipeline(stats_df, games_df)

        # Step 3: Build matchups
        logger.info("🏀 Building matchups...")
        matchups = builder.build_matchup_pairs(df)
        train_df, pred_df = builder.split_train_prediction(matchups)
        features = builder.select_features(train_df)

        # Step 4: Train model
        logger.info("🤖 Training LightGBM model...")
        accuracy = predictor.train(train_df, features)
        logger.info(f"Model accuracy: {accuracy:.2%}")

        # Step 5: Generate predictions
        logger.info("🔮 Generating predictions...")
        predictions_df = predictor.predict(pred_df, features, games_df)

        # Step 6: Fetch odds and merge
        api_key = os.getenv("ODDS_API_KEY")
        if api_key:
            logger.info("💰 Fetching odds from The Odds API...")
            try:
                final_df = fetch_and_merge_odds(
                    predictions_df=predictions_df,
                    games_df=games_df,
                    api_key=api_key,
                )
                logger.info("✅ Odds fetched and merged successfully")
            except Exception as e:
                logger.warning(f"Could not fetch odds: {e}. Continuing without odds.")
                final_df = predictions_df
        else:
            logger.warning("No ODDS_API_KEY found. Skipping odds fetching.")
            final_df = predictions_df

        # Step 7: Save predictions
        dirs = get_directory_paths()
        prediction_dir = dirs["prediction_dir"]
        os.makedirs(prediction_dir, exist_ok=True)

        output_file = prediction_dir / f"nba_games_predict_{today_str}.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"✅ Predictions saved to: {output_file}")

        # Display top predictions
        logger.info("\n🎯 Today's Top Predictions:")
        for _, row in final_df.nlargest(3, 'home_team_prob').iterrows():
            logger.info(f"  {row['home_team']} vs {row['away_team']}: "
                       f"{row['home_team_prob']:.1%} home win probability")

        logger.info("✅ Prediction generation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during prediction generation: {e}", exc_info=True)
        return False
