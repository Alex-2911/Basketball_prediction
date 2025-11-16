"""
Analysis commands

Provides CLI interface for betting analysis and Kelly Criterion calculations.
"""

import os
from datetime import date
from typing import Optional

from src.core.analyzer import BettingPerformanceAnalyzer, HomeWinRateCalculator
from src.core.betting import (
    BankrollSimulator,
    BetRecommendationDisplay,
    KellyCriterionCalculator,
    ProbabilityCalibrator,
    calibrate_and_simulate,
)
from src.utils.logger import get_logger
from src.utils.nba_utils import CURRENT_SEASON

logger = get_logger(__name__)


def run_statistics() -> bool:
    """
    Calculate betting statistics and accuracy metrics

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("📊 Calculating betting statistics...")

        # Initialize analyzer
        analyzer = BettingPerformanceAnalyzer(season=CURRENT_SEASON)

        # Process and update statistics
        df = analyzer.process_and_update_statistics()

        if df is None or df.empty:
            logger.warning("No predictions found to analyze")
            return False

        # Generate performance report
        report = analyzer.generate_performance_report(df)

        # Display results
        logger.info("\n📈 Betting Performance Report:")
        logger.info(f"  Overall Accuracy: {report['overall']:.2%}")
        logger.info(f"  High Confidence (>60%): {report['high_confidence']:.2%}")
        logger.info(f"  Low Confidence (≤40%): {report['low_confidence']:.2%}")

        logger.info("✅ Statistics calculation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}", exc_info=True)
        return False


def run_kelly() -> bool:
    """
    Calculate Kelly Criterion betting parameters

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("💰 Calculating Kelly Criterion parameters...")

        # Initialize components
        calculator = HomeWinRateCalculator()
        calibrator = ProbabilityCalibrator()
        kelly_calc = KellyCriterionCalculator()
        simulator = BankrollSimulator(starting_bankroll=1000.0)

        # Step 1: Load latest predictions
        analyzer = BettingPerformanceAnalyzer(season=CURRENT_SEASON)

        try:
            # Try to load combined statistics file first
            stats_file = analyzer.find_recent_statistics_file()
            if stats_file:
                combined_df = analyzer.load_actual_results(stats_file)
                logger.info(f"Loaded combined statistics from {stats_file}")
            else:
                # Fallback to prediction file
                pred_file = analyzer.find_recent_prediction_file()
                if not pred_file:
                    logger.error("No prediction or statistics files found")
                    return False
                combined_df = analyzer.load_predictions(pred_file)
                logger.info(f"Loaded predictions from {pred_file}")
        except Exception as e:
            logger.error(f"Could not load data: {e}")
            return False

        # Step 2: Calculate home win rates
        logger.info("🏠 Calculating home win rates...")
        win_rates_df, good_teams_df, rates_path = calculator.compute_and_save(combined_df)
        logger.info(f"Home win rates saved to {rates_path}")

        # Step 3: Calibrate and simulate (if we have historical data)
        if 'result' in combined_df.columns:
            logger.info("📈 Calibrating probabilities and simulating season...")
            results = calibrate_and_simulate(
                combined_df=combined_df,
                good_home_teams=set(good_teams_df.index),
                starting_bankroll=1000.0,
            )

            # Display simulation results
            if results:
                metrics = simulator.calculate_metrics(results)
                logger.info("\n📊 Season Simulation Results:")
                logger.info(f"  Final Bankroll: €{metrics['final_bankroll']:.2f}")
                logger.info(f"  ROI: {metrics['roi']:.1f}%")
                logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
                logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
                if metrics['sharpe_ratio'] is not None:
                    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        else:
            logger.info("No historical results available for calibration/simulation")

        # Step 4: Apply Kelly to latest predictions
        logger.info("🎯 Calculating Kelly stakes for today's games...")
        today = date.today()
        today_str = today.strftime("%Y-%m-%d")

        # Filter for today's games only (if date column exists)
        if 'date' in combined_df.columns:
            todays_games = combined_df[combined_df['date'] == today_str].copy()
        else:
            # Assume all predictions are for today
            todays_games = combined_df.copy()

        if not todays_games.empty:
            # Apply Kelly calculations
            todays_games_with_kelly = kelly_calc.apply_kelly_to_dataframe(
                df=todays_games,
                bankroll=1000.0,
            )

            # Save enriched predictions
            from src.core.predictor import get_directory_paths
            dirs = get_directory_paths()
            prediction_dir = dirs["prediction_dir"]
            os.makedirs(prediction_dir, exist_ok=True)

            enriched_path = prediction_dir / f"combined_nba_predictions_enriched_{today_str}.csv"
            todays_games_with_kelly.to_csv(enriched_path, index=False)
            logger.info(f"Enriched predictions saved to {enriched_path}")

            # Filter and save recommended bets
            value_bets = kelly_calc.filter_value_bets(
                df=todays_games_with_kelly,
                odds_range=(1.18, 3.00),
                prob_threshold=0.40,
            )

            if not value_bets.empty:
                stakes_path = prediction_dir / f"kelly_stakes_{today_str}.csv"
                value_bets.to_csv(stakes_path, index=False)
                logger.info(f"✅ {len(value_bets)} recommended bets saved to {stakes_path}")
            else:
                logger.info("No value bets found for today")
        else:
            logger.info("No games to analyze for today")

        logger.info("✅ Kelly Criterion calculation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error calculating Kelly parameters: {e}", exc_info=True)
        return False


def run_recommendations() -> bool:
    """
    Display today's betting recommendations

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("🎯 Loading betting recommendations...")

        # Initialize display
        display = BetRecommendationDisplay()

        # Display recommendations
        display.display_recommendations()

        return True

    except Exception as e:
        logger.error(f"Error displaying recommendations: {e}", exc_info=True)
        return False


def run_all_analysis() -> bool:
    """
    Run all analysis steps in sequence

    Returns:
        True if all steps succeeded, False otherwise
    """
    logger.info("Starting complete betting analysis...")

    # Run statistics
    if not run_statistics():
        logger.error("Statistics calculation failed")
        return False

    # Run Kelly Criterion
    if not run_kelly():
        logger.error("Kelly Criterion calculation failed")
        return False

    # Generate recommendations
    if not run_recommendations():
        logger.error("Recommendation generation failed")
        return False

    logger.info("Complete analysis finished successfully")
    return True
