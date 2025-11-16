"""Core business logic for NBA Prediction System

This package contains the core functionality:
- collector: Web scraping NBA data (HistoricalGameCollector, UpcomingGameCollector)
- predictor: Machine learning predictions (GameDataPreprocessor, MatchupBuilder, LightGBMPredictor)
- analyzer: Performance analysis (BettingPerformanceAnalyzer, HomeWinRateCalculator)
- betting: Odds fetching and Kelly Criterion (OddsManager, ProbabilityCalibrator, KellyCriterionCalculator, BankrollSimulator, BetRecommendationDisplay)
- constants: Shared constants and configuration
"""

# Collector classes
from .collector import HistoricalGameCollector, UpcomingGameCollector

# Predictor classes and functions
from .predictor import (
    GameDataPreprocessor,
    LightGBMPredictor,
    MatchupBuilder,
    get_directory_paths,
    normalize_team_code,
)

# Analyzer classes
from .analyzer import (
    BettingPerformanceAnalyzer,
    HomeWinRateCalculator,
    analyze_betting_performance,
    calculate_home_win_rates,
)

# Betting classes and functions
from .betting import (
    BankrollSimulator,
    BetRecommendationDisplay,
    KellyCriterionCalculator,
    OddsManager,
    ProbabilityCalibrator,
    calibrate_and_simulate,
    fetch_and_merge_odds,
)

# Constants
from . import constants

__all__ = [
    # Collector
    "HistoricalGameCollector",
    "UpcomingGameCollector",
    # Predictor
    "GameDataPreprocessor",
    "MatchupBuilder",
    "LightGBMPredictor",
    "normalize_team_code",
    "get_directory_paths",
    # Analyzer
    "BettingPerformanceAnalyzer",
    "HomeWinRateCalculator",
    "analyze_betting_performance",
    "calculate_home_win_rates",
    # Betting
    "OddsManager",
    "ProbabilityCalibrator",
    "KellyCriterionCalculator",
    "BankrollSimulator",
    "BetRecommendationDisplay",
    "calibrate_and_simulate",
    "fetch_and_merge_odds",
    # Constants
    "constants",
]
