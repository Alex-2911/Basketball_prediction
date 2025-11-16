"""
Betting System Module for NBA Predictions

This module provides a comprehensive betting system with odds management,
probability calibration, Kelly Criterion calculations, bankroll simulation,
and bet recommendation display.

Classes:
    OddsManager: Handles odds fetching, format conversion, implied probabilities
    ProbabilityCalibrator: Calibrates raw probabilities using Platt scaling or isotonic regression
    KellyCriterionCalculator: Calculates Kelly fractions and bet stakes
    BankrollSimulator: Simulates betting performance over a season
    BetRecommendationDisplay: Displays and formats bet recommendations
"""

import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from urllib3.util.retry import Retry

from src.core.constants import (
    KELLY_DEFAULTS,
    PREFERRED_SPORTSBOOKS,
    STRATEGY_THRESHOLDS,
)
from src.utils.error_handlers import (
    DataValidationError,
    ErrorContext,
    NetworkError,
    log_dataframe_info,
    validate_dataframe,
)
from src.utils.logger import get_logger
from src.utils.nba_utils import kelly_frac

logger = get_logger(__name__)


# Team name mappings for odds API
FULL_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

TEAM_ALIAS_FOR_ODDS = {
    "PHO": "PHX",
    "PHX": "PHX",
    "BKN": "BRK",
    "BRK": "BRK",
    "CHA": "CHO",
    "CHO": "CHO",
    "GS": "GSW",
    "GSW": "GSW",
    "NO": "NOP",
    "NOP": "NOP",
    "NY": "NYK",
    "NYK": "NYK",
    "SA": "SAS",
    "SAS": "SAS",
    "UTAH": "UTA",
    "UTA": "UTA",
    "OKL": "OKC",
    "OKC": "OKC",
}


class OddsManager:
    """
    Manages odds fetching, format conversion, and probability calculations.

    This class handles all odds-related operations including:
    - Fetching H2H odds from The Odds API
    - Converting between American and decimal odds formats
    - Calculating implied probabilities from odds
    - Normalizing team codes for API compatibility
    - Merging odds with predictions
    """

    def __init__(self, api_key: Optional[str] = None, preferred_books: Optional[List[str]] = None):
        """
        Initialize the OddsManager.

        Args:
            api_key: API key for The Odds API (if None, must be provided in fetch_odds)
            preferred_books: List of preferred sportsbook keys in order of preference
                           Defaults to PREFERRED_SPORTSBOOKS from constants
        """
        self.api_key = api_key
        self.preferred_books = preferred_books or PREFERRED_SPORTSBOOKS
        self.session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        """
        Create a requests session with retry logic.

        Returns:
            Configured requests.Session with retry adapter
        """
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    @staticmethod
    def normalize_team_code(team_code: str) -> str:
        """
        Normalize team code to canonical format for odds API.

        Converts variations like PHO->PHX, BKN->BRK, CHA->CHO, etc.

        Args:
            team_code: Team abbreviation (e.g., "PHO", "BKN")

        Returns:
            Normalized team code (e.g., "PHX", "BRK")

        Example:
            >>> OddsManager.normalize_team_code("PHO")
            'PHX'
            >>> OddsManager.normalize_team_code("BKN")
            'BRK'
        """
        if not isinstance(team_code, str):
            return team_code
        return TEAM_ALIAS_FOR_ODDS.get(team_code.upper(), team_code.upper())

    @staticmethod
    def american_to_decimal(odds: Union[int, float]) -> float:
        """
        Convert American moneyline odds to decimal odds.

        Args:
            odds: American odds (e.g., -150, +200)

        Returns:
            Decimal odds (e.g., 1.67, 3.00)
            Returns np.nan if input is None or NaN

        Example:
            >>> OddsManager.american_to_decimal(-150)
            1.67
            >>> OddsManager.american_to_decimal(200)
            3.0
        """
        if pd.isna(odds):
            return np.nan
        odds = float(odds)
        if odds > 0:
            return round(odds / 100.0 + 1.0, 2)
        else:
            return round(100.0 / abs(odds) + 1.0, 2)

    @staticmethod
    def decimal_to_american(odds: Union[float, int]) -> int:
        """
        Convert decimal odds to American moneyline odds.

        Args:
            odds: Decimal odds (e.g., 1.67, 3.00)

        Returns:
            American odds (e.g., -150, +200)
            Returns None if input is invalid

        Example:
            >>> OddsManager.decimal_to_american(1.67)
            -150
            >>> OddsManager.decimal_to_american(3.0)
            200
        """
        if pd.isna(odds) or odds <= 1.0:
            return None
        odds = float(odds)
        if odds >= 2.0:
            return int(round((odds - 1.0) * 100))
        else:
            return int(round(-100.0 / (odds - 1.0)))

    @staticmethod
    def implied_probability(odds: Union[int, float], format: str = "american") -> float:
        """
        Calculate implied probability from odds.

        Args:
            odds: Odds value (American or decimal depending on format)
            format: "american" or "decimal"

        Returns:
            Implied probability as a float between 0 and 1
            Returns np.nan if odds are invalid

        Example:
            >>> OddsManager.implied_probability(-150, "american")
            0.6
            >>> OddsManager.implied_probability(1.67, "decimal")
            0.599
        """
        if odds is None or (isinstance(odds, float) and np.isnan(odds)):
            return np.nan

        try:
            if format == "american":
                odds = float(odds)
                if odds < 0:
                    # Favorite
                    return abs(odds) / (abs(odds) + 100.0)
                else:
                    # Underdog
                    return 100.0 / (odds + 100.0)
            elif format == "decimal":
                odds = float(odds)
                if odds <= 1.0:
                    return np.nan
                return 1.0 / odds
            else:
                raise ValueError(f"Invalid format: {format}. Use 'american' or 'decimal'")
        except (ValueError, TypeError):
            return np.nan

    def fetch_odds(
        self,
        games_df: pd.DataFrame,
        api_key: Optional[str] = None,
        preferred_books: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch H2H odds from The Odds API and align with game schedule.

        Args:
            games_df: DataFrame with columns ['home_team', 'away_team']
            api_key: API key for The Odds API (overrides instance api_key)
            preferred_books: List of preferred sportsbook keys (overrides instance preference)

        Returns:
            DataFrame with columns: ['home_team', 'away_team', 'odds 1', 'odds 2']
            where 'odds 1' is home team odds, 'odds 2' is away team odds (American format)

        Raises:
            NetworkError: If API request fails
            DataValidationError: If games_df is invalid

        Example:
            >>> odds_mgr = OddsManager(api_key="your_key")
            >>> games = pd.DataFrame({'home_team': ['LAL'], 'away_team': ['BOS']})
            >>> odds_df = odds_mgr.fetch_odds(games)
        """
        with ErrorContext("Fetching odds from API", logger=logger):
            # Validate input
            validate_dataframe(
                games_df,
                required_columns=["home_team", "away_team"],
                allow_empty=True,
            )

            # Use provided api_key or instance api_key
            key = api_key or self.api_key
            if not key:
                raise ValueError("API key must be provided either in __init__ or fetch_odds()")

            # Use provided books or instance preference
            books = preferred_books or self.preferred_books

            try:
                response = self.session.get(
                    "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
                    params={
                        "apiKey": key,
                        "regions": "us",
                        "markets": "h2h",
                        "oddsFormat": "american",
                    },
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Failed to fetch odds from API: {e}")

            # Build lookup dictionary: (HOME, AWAY) -> (ml_home, ml_away)
            lookup = {}

            for event in data:
                home_full = event.get("home_team")
                away_full = event.get("away_team")

                # Convert full names to abbreviations
                raw_home = FULL_NAME_TO_ABBREV.get(home_full)
                raw_away = FULL_NAME_TO_ABBREV.get(away_full)
                if not raw_home or not raw_away:
                    continue

                # Normalize to canonical codes
                home_code = self.normalize_team_code(raw_home)
                away_code = self.normalize_team_code(raw_away)

                # Find preferred bookmaker
                bookmakers = event.get("bookmakers", [])
                chosen = None
                if books:
                    for book_key in books:
                        chosen = next((b for b in bookmakers if b.get("key") == book_key), None)
                        if chosen:
                            break
                if not chosen and bookmakers:
                    chosen = bookmakers[0]
                if not chosen:
                    continue

                # Extract h2h market
                market = next(
                    (m for m in chosen.get("markets", []) if m.get("key") == "h2h"), None
                )
                if not market:
                    continue

                # Build prices by team code
                prices_by_code = {}
                for outcome in market.get("outcomes", []):
                    full_name = outcome.get("name")
                    raw_abbr = FULL_NAME_TO_ABBREV.get(full_name)
                    if raw_abbr:
                        canon_abbr = self.normalize_team_code(raw_abbr)
                        prices_by_code[canon_abbr] = outcome.get("price")

                ml_home = prices_by_code.get(home_code)
                ml_away = prices_by_code.get(away_code)

                lookup[(home_code, away_code)] = (ml_home, ml_away)

            # Align with game schedule
            odds_rows = []
            for _, game in games_df.iterrows():
                h = self.normalize_team_code(game["home_team"])
                a = self.normalize_team_code(game["away_team"])

                o1, o2 = lookup.get((h, a), (None, None))
                if o1 is None or o2 is None:
                    logger.warning(f"No odds found for {h} vs {a}")
                odds_rows.append({"home_team": h, "away_team": a, "odds 1": o1, "odds 2": o2})

            logger.info(f"Fetched odds for {len(odds_rows)} games")
            return pd.DataFrame(odds_rows)

    def merge_with_predictions(
        self, predictions_df: pd.DataFrame, odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge predictions with odds and calculate value edges.

        Args:
            predictions_df: DataFrame with columns ['home_team', 'away_team', 'home_team_prob']
            odds_df: DataFrame with columns ['home_team', 'away_team', 'odds 1', 'odds 2']

        Returns:
            Merged DataFrame with additional columns:
            - imp_prob_home: Implied probability from home odds
            - imp_prob_away: Implied probability from away odds
            - value_home: Model edge on home team (prob - implied_prob)
            - value_away: Model edge on away team

        Example:
            >>> preds = pd.DataFrame({
            ...     'home_team': ['LAL'], 'away_team': ['BOS'], 'home_team_prob': [0.65]
            ... })
            >>> odds = pd.DataFrame({
            ...     'home_team': ['LAL'], 'away_team': ['BOS'], 'odds 1': [-150], 'odds 2': [130]
            ... })
            >>> merged = odds_mgr.merge_with_predictions(preds, odds)
        """
        with ErrorContext("Merging predictions with odds", logger=logger):
            # Normalize team codes in both DataFrames
            preds = predictions_df.copy()
            odds = odds_df.copy()

            preds["home_team"] = preds["home_team"].apply(self.normalize_team_code)
            preds["away_team"] = preds["away_team"].apply(self.normalize_team_code)
            odds["home_team"] = odds["home_team"].apply(self.normalize_team_code)
            odds["away_team"] = odds["away_team"].apply(self.normalize_team_code)

            # Merge
            df = preds.merge(odds, on=["home_team", "away_team"], how="left")

            # Ensure odds are numeric
            for col in ["odds 1", "odds 2"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Calculate implied probabilities
            df["imp_prob_home"] = df["odds 1"].apply(
                lambda x: self.implied_probability(x, "american")
            )
            df["imp_prob_away"] = df["odds 2"].apply(
                lambda x: self.implied_probability(x, "american")
            )

            # Calculate value edges
            df["value_home"] = np.where(
                df["imp_prob_home"].notna(),
                df["home_team_prob"] - df["imp_prob_home"],
                np.nan,
            )
            df["value_away"] = np.where(
                df["imp_prob_away"].notna(),
                (1.0 - df["home_team_prob"]) - df["imp_prob_away"],
                np.nan,
            )

            logger.info(f"Merged {len(df)} predictions with odds")
            return df


class ProbabilityCalibrator:
    """
    Calibrates raw model probabilities using various methods.

    Supports:
    - Platt scaling (logistic regression)
    - Isotonic regression

    These methods adjust raw model probabilities to be better aligned
    with observed frequencies, improving betting decisions.
    """

    def __init__(self):
        """Initialize the ProbabilityCalibrator."""
        self.platt_model = None
        self.isotonic_model = None
        self._is_platt_fitted = False
        self._is_isotonic_fitted = False

    def fit_platt_scaling(
        self, probabilities: np.ndarray, outcomes: np.ndarray, test_size: float = 0.2
    ) -> LogisticRegression:
        """
        Fit Platt scaling calibration (logistic regression on probabilities).

        Args:
            probabilities: Array of raw model probabilities (shape: [n_samples,] or [n_samples, 1])
            outcomes: Array of binary outcomes (0 or 1)
            test_size: Fraction of data to use for fitting (uses calibration set)

        Returns:
            Fitted LogisticRegression model

        Example:
            >>> calibrator = ProbabilityCalibrator()
            >>> probs = np.array([0.3, 0.7, 0.5, 0.9])
            >>> outcomes = np.array([0, 1, 0, 1])
            >>> model = calibrator.fit_platt_scaling(probs, outcomes)
        """
        with ErrorContext("Fitting Platt scaling calibration", logger=logger):
            # Reshape if needed
            if probabilities.ndim == 1:
                X = probabilities.reshape(-1, 1)
            else:
                X = probabilities

            # Split for calibration
            _, X_cal, _, y_cal = train_test_split(
                X, outcomes, test_size=test_size, random_state=42
            )

            # Fit logistic regression
            self.platt_model = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.platt_model.fit(X_cal, y_cal)
            self._is_platt_fitted = True

            logger.info(f"Fitted Platt scaling on {len(X_cal)} calibration samples")
            return self.platt_model

    def fit_isotonic_regression(
        self, probabilities: np.ndarray, outcomes: np.ndarray
    ) -> IsotonicRegression:
        """
        Fit isotonic regression calibration.

        Isotonic regression enforces monotonicity while fitting
        observed frequencies to predicted probabilities.

        Args:
            probabilities: Array of raw model probabilities
            outcomes: Array of binary outcomes (0 or 1)

        Returns:
            Fitted IsotonicRegression model

        Example:
            >>> calibrator = ProbabilityCalibrator()
            >>> probs = np.array([0.3, 0.7, 0.5, 0.9])
            >>> outcomes = np.array([0, 1, 0, 1])
            >>> model = calibrator.fit_isotonic_regression(probs, outcomes)
        """
        with ErrorContext("Fitting isotonic regression calibration", logger=logger):
            # Flatten if needed
            X = probabilities.flatten() if probabilities.ndim > 1 else probabilities

            # Fit isotonic regression
            self.isotonic_model = IsotonicRegression(out_of_bounds="clip")
            self.isotonic_model.fit(X, outcomes)
            self._is_isotonic_fitted = True

            logger.info(f"Fitted isotonic regression on {len(X)} samples")
            return self.isotonic_model

    def calibrate(
        self, probabilities: np.ndarray, method: str = "platt"
    ) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            probabilities: Array of raw probabilities to calibrate
            method: Calibration method - "platt" or "isotonic"

        Returns:
            Calibrated probabilities

        Raises:
            ValueError: If method is invalid or model not fitted

        Example:
            >>> # After fitting
            >>> calibrator.fit_platt_scaling(train_probs, train_outcomes)
            >>> calibrated = calibrator.calibrate(test_probs, method="platt")
        """
        if method == "platt":
            if not self._is_platt_fitted:
                raise ValueError("Platt scaling model not fitted. Call fit_platt_scaling() first.")
            X = probabilities.reshape(-1, 1) if probabilities.ndim == 1 else probabilities
            return self.platt_model.predict_proba(X)[:, 1]

        elif method == "isotonic":
            if not self._is_isotonic_fitted:
                raise ValueError(
                    "Isotonic regression model not fitted. Call fit_isotonic_regression() first."
                )
            X = probabilities.flatten() if probabilities.ndim > 1 else probabilities
            return self.isotonic_model.transform(X)

        else:
            raise ValueError(f"Invalid method: {method}. Use 'platt' or 'isotonic'")

    def calibrate_predictions(
        self, df: pd.DataFrame, prob_col: str = "home_team_prob", methods: List[str] = None
    ) -> pd.DataFrame:
        """
        Calibrate all predictions in a DataFrame.

        Args:
            df: DataFrame with probability column
            prob_col: Name of column containing raw probabilities
            methods: List of methods to apply (default: ["platt", "isotonic"])

        Returns:
            DataFrame with additional columns:
            - prob_platt: Platt-calibrated probabilities
            - prob_iso: Isotonic-calibrated probabilities

        Example:
            >>> # After fitting both models
            >>> df_calibrated = calibrator.calibrate_predictions(predictions_df)
        """
        if methods is None:
            methods = ["platt", "isotonic"]

        result = df.copy()
        probabilities = df[prob_col].values

        for method in methods:
            if method == "platt" and self._is_platt_fitted:
                result["prob_platt"] = self.calibrate(probabilities, method="platt")
            elif method == "isotonic" and self._is_isotonic_fitted:
                result["prob_iso"] = self.calibrate(probabilities, method="isotonic")

        return result


class KellyCriterionCalculator:
    """
    Calculates Kelly Criterion bet sizes and manages bet recommendations.

    The Kelly Criterion optimizes bet sizing based on:
    - Win probability
    - Odds offered
    - Bankroll size

    This implementation supports fractional Kelly (e.g., half-Kelly)
    and applies safety caps to prevent excessive betting.
    """

    def __init__(
        self,
        bet_fraction: float = None,
        cap_fraction: float = None,
        absolute_cap: float = None,
    ):
        """
        Initialize the KellyCriterionCalculator.

        Args:
            bet_fraction: Fraction of Kelly to use (default from KELLY_DEFAULTS)
            cap_fraction: Maximum fraction of bankroll per bet (default from KELLY_DEFAULTS)
            absolute_cap: Absolute maximum stake in currency units (default from KELLY_DEFAULTS)
        """
        self.bet_fraction = bet_fraction or KELLY_DEFAULTS["bet_fraction"]
        self.cap_fraction = cap_fraction or KELLY_DEFAULTS["cap_fraction"]
        self.absolute_cap = absolute_cap or KELLY_DEFAULTS["absolute_cap"]

    def calculate_kelly_fraction(
        self,
        probability: float,
        odds: float,
        fraction: Optional[float] = None,
    ) -> float:
        """
        Calculate Kelly fraction for a bet.

        Args:
            probability: Win probability (0 to 1)
            odds: Decimal odds (>1)
            fraction: Fraction of Kelly to use (overrides instance setting)

        Returns:
            Kelly fraction (0 to 1) representing portion of bankroll to bet

        Example:
            >>> calc = KellyCriterionCalculator()
            >>> kelly = calc.calculate_kelly_fraction(0.55, 2.0)  # 55% win prob, 2.0 decimal odds
            >>> print(f"Bet {kelly:.2%} of bankroll")
        """
        frac = fraction if fraction is not None else self.bet_fraction
        return kelly_frac(probability, odds, frac)

    def calculate_stake(
        self,
        kelly_fraction: float,
        bankroll: float,
        cap_pct: Optional[float] = None,
        abs_cap: Optional[float] = None,
    ) -> float:
        """
        Calculate actual stake amount with safety caps.

        Args:
            kelly_fraction: Kelly fraction from calculate_kelly_fraction()
            bankroll: Current bankroll size
            cap_pct: Maximum percentage of bankroll (overrides instance setting)
            abs_cap: Absolute maximum stake (overrides instance setting)

        Returns:
            Stake amount to bet

        Example:
            >>> calc = KellyCriterionCalculator()
            >>> kelly = 0.10  # 10% Kelly
            >>> stake = calc.calculate_stake(kelly, bankroll=1000)
            >>> print(f"Bet ${stake:.2f}")
        """
        cap_pct = cap_pct if cap_pct is not None else self.cap_fraction
        abs_cap = abs_cap if abs_cap is not None else self.absolute_cap

        # Apply Kelly fraction
        stake = kelly_fraction * bankroll

        # Apply percentage cap
        stake = min(stake, cap_pct * bankroll)

        # Apply absolute cap
        stake = min(stake, abs_cap)

        return stake

    def generate_bet_recommendation(
        self,
        home_team: str,
        away_team: str,
        probability: float,
        odds: float,
        bankroll: float,
        side: str = "home",
    ) -> Optional[Tuple[str, float]]:
        """
        Generate a formatted bet recommendation if positive edge exists.

        Args:
            home_team: Home team code
            away_team: Away team code
            probability: Win probability for the side being bet
            odds: Decimal odds for the side being bet
            bankroll: Current bankroll
            side: "home" or "away"

        Returns:
            Tuple of (recommendation_text, stake) if positive edge, else (None, 0.0)

        Example:
            >>> calc = KellyCriterionCalculator()
            >>> rec, stake = calc.generate_bet_recommendation(
            ...     "LAL", "BOS", 0.60, 2.0, 1000, side="home"
            ... )
            >>> if rec:
            ...     print(rec)
        """
        kelly = self.calculate_kelly_fraction(probability, odds)

        if kelly <= 0:
            return None, 0.0

        stake = self.calculate_stake(kelly, bankroll)

        recommendation = (
            f"✅ {home_team}–{away_team} ({side}): "
            f"p̂={probability:.4f}, odds={odds:.2f} → "
            f"Kelly={kelly:.4f}, stake=${stake:.2f}"
        )

        return recommendation, stake

    def filter_value_bets(
        self,
        df: pd.DataFrame,
        odds_range: Optional[Tuple[float, float]] = None,
        prob_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Filter DataFrame for value bets based on strategy thresholds.

        Args:
            df: DataFrame with odds and probability columns
            odds_range: Tuple of (min_odds, max_odds) in decimal format
            prob_threshold: Minimum probability to consider

        Returns:
            Filtered DataFrame containing only value bets

        Example:
            >>> calc = KellyCriterionCalculator()
            >>> value_bets = calc.filter_value_bets(
            ...     predictions_df,
            ...     odds_range=(1.18, 3.00),
            ...     prob_threshold=0.40
            ... )
        """
        result = df.copy()

        # Apply odds range filter
        if odds_range is not None:
            min_odds, max_odds = odds_range
            if "odds_1" in result.columns:
                result = result[result["odds_1"].between(min_odds, max_odds)]

        # Apply probability threshold
        if prob_threshold is not None:
            if "home_team_prob" in result.columns:
                result = result[result["home_team_prob"] >= prob_threshold]
            elif "raw_prob" in result.columns:
                result = result[result["raw_prob"] >= prob_threshold]

        return result

    def apply_kelly_to_dataframe(
        self,
        df: pd.DataFrame,
        bankroll: float,
        prob_col: str = "home_team_prob",
        odds_col: str = "odds_1",
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Apply Kelly calculations to all rows in a DataFrame.

        Args:
            df: DataFrame with probability and odds columns
            bankroll: Starting bankroll (can be updated row by row for simulation)
            prob_col: Name of probability column
            odds_col: Name of odds column
            prefix: Prefix for output columns (e.g., "raw_", "platt_")

        Returns:
            DataFrame with additional columns:
            - {prefix}kelly_frac: Kelly fraction for each bet
            - {prefix}stake: Stake amount for each bet

        Example:
            >>> calc = KellyCriterionCalculator()
            >>> df_with_kelly = calc.apply_kelly_to_dataframe(
            ...     predictions_df, bankroll=1000
            ... )
        """
        result = df.copy()

        # Calculate Kelly fractions
        result[f"{prefix}kelly_frac"] = result.apply(
            lambda row: self.calculate_kelly_fraction(row[prob_col], row[odds_col])
            if pd.notna(row[prob_col]) and pd.notna(row[odds_col])
            else 0.0,
            axis=1,
        )

        # Calculate stakes
        result[f"{prefix}stake"] = result[f"{prefix}kelly_frac"].apply(
            lambda kf: self.calculate_stake(kf, bankroll)
        )

        return result


class BankrollSimulator:
    """
    Simulates betting performance over a season.

    Tracks:
    - Bankroll evolution over time
    - Profit/Loss per bet
    - Expected value vs actual returns
    - Performance metrics (ROI, Sharpe ratio, max drawdown)
    """

    def __init__(self, starting_bankroll: float = 1000.0):
        """
        Initialize the BankrollSimulator.

        Args:
            starting_bankroll: Initial bankroll amount
        """
        self.starting_bankroll = starting_bankroll

    def simulate_season(
        self,
        df: pd.DataFrame,
        strategy_config: Dict[str, any],
        methods: List[str] = None,
    ) -> pd.DataFrame:
        """
        Simulate betting over a full season with multiple calibration methods.

        Args:
            df: DataFrame with columns:
                - home_team, away_team, date
                - home_team_prob (raw probabilities)
                - prob_platt, prob_iso (calibrated probabilities)
                - odds_1 (home odds in decimal)
                - result or win (actual outcome)
            strategy_config: Dictionary with keys:
                - odds_min, odds_max: Odds range filter
                - prob_min: Minimum probability threshold
                - good_home_teams: Set of teams to bet on (home side only)
                - bet_fraction: Fraction of Kelly
                - cap_fraction: Max % of bankroll per bet
                - absolute_cap: Absolute max stake
            methods: List of probability methods to simulate (default: ["raw", "platt", "iso"])

        Returns:
            DataFrame with simulation results including:
            - {method}_kelly_frac, {method}_stake, {method}_pnl, {method}_ev, {method}_bank
            for each method

        Example:
            >>> simulator = BankrollSimulator(starting_bankroll=1000)
            >>> config = {
            ...     'odds_min': 1.18, 'odds_max': 3.00, 'prob_min': 0.40,
            ...     'good_home_teams': {'LAL', 'BOS', 'GSW'},
            ...     'bet_fraction': 0.5, 'cap_fraction': 0.30, 'absolute_cap': 300
            ... }
            >>> results = simulator.simulate_season(historical_df, config)
        """
        with ErrorContext("Simulating season bankroll", logger=logger):
            if methods is None:
                methods = ["raw", "platt", "iso"]

            result = df.copy()
            result = result.sort_values("date").reset_index(drop=True)

            # Determine win column
            win_col = "win" if "win" in result.columns else None
            if win_col is None and "result" in result.columns and "home_team" in result.columns:
                result["win"] = (result["result"] == result["home_team"]).astype(int)
                win_col = "win"

            # Initialize columns for each method
            for method in methods:
                result[f"kelly_frac_{method}"] = 0.0
                result[f"stake_{method}"] = 0.0
                result[f"pnl_{method}"] = 0.0
                result[f"ev_{method}"] = 0.0
                result[f"bank_{method}"] = np.nan

            # Initialize bankrolls
            bank = {method: self.starting_bankroll for method in methods}

            # Extract config
            odds_min = strategy_config.get("odds_min", STRATEGY_THRESHOLDS["odds_min"])
            odds_max = strategy_config.get("odds_max", STRATEGY_THRESHOLDS["odds_max"])
            prob_min = strategy_config.get("prob_min", STRATEGY_THRESHOLDS["prob_min"])
            good_homes = strategy_config.get("good_home_teams", set())
            bet_frac = strategy_config.get("bet_fraction", KELLY_DEFAULTS["bet_fraction"])
            cap_frac = strategy_config.get("cap_fraction", KELLY_DEFAULTS["cap_fraction"])
            abs_cap = strategy_config.get("absolute_cap", KELLY_DEFAULTS["absolute_cap"])

            # Initialize calculator
            calc = KellyCriterionCalculator(
                bet_fraction=bet_frac,
                cap_fraction=cap_frac,
                absolute_cap=abs_cap,
            )

            # Simulate each bet
            for i, row in result.iterrows():
                odds = row.get("odds_1", np.nan)
                is_home = row.get("home_team") in good_homes if good_homes else True

                for method in methods:
                    # Determine probability column
                    if method == "raw":
                        prob_col = "home_team_prob"
                    elif method == "platt":
                        prob_col = "prob_platt"
                    elif method == "iso":
                        prob_col = "prob_iso"
                    else:
                        continue

                    prob = row.get(prob_col, np.nan)

                    # Check betting criteria
                    should_bet = (
                        is_home
                        and pd.notna(odds)
                        and pd.notna(prob)
                        and (odds >= odds_min)
                        and (odds <= odds_max)
                        and (prob >= prob_min)
                    )

                    if should_bet:
                        # Calculate Kelly and stake
                        kf = calc.calculate_kelly_fraction(prob, odds)
                        stake = calc.calculate_stake(kf, bank[method])

                        # Calculate P&L
                        if win_col and pd.notna(row[win_col]):
                            won = bool(row[win_col])
                            pnl = stake * (odds - 1.0) if won else -stake
                        else:
                            pnl = 0.0

                        # Calculate EV
                        ev = (prob * (odds - 1.0) - (1 - prob)) * stake

                        # Update results
                        result.at[i, f"kelly_frac_{method}"] = kf
                        result.at[i, f"stake_{method}"] = stake
                        result.at[i, f"pnl_{method}"] = pnl
                        result.at[i, f"ev_{method}"] = ev

                        # Update bankroll
                        bank[method] += pnl

                    # Record bankroll
                    result.at[i, f"bank_{method}"] = bank[method]

            # Log final results
            for method in methods:
                final_bank = bank[method]
                roi = (final_bank - self.starting_bankroll) / self.starting_bankroll * 100
                logger.info(
                    f"{method.capitalize()} method: "
                    f"Starting=${self.starting_bankroll:.2f}, "
                    f"Final=${final_bank:.2f}, "
                    f"ROI={roi:.2f}%"
                )

            return result

    def plot_bankroll_paths(
        self,
        results: pd.DataFrame,
        methods: List[str] = None,
        title: str = "Bankroll Evolution",
    ) -> None:
        """
        Plot bankroll evolution over time for different methods.

        Args:
            results: DataFrame from simulate_season() with bank_{method} columns
            methods: List of methods to plot (default: ["raw", "platt", "iso"])
            title: Plot title

        Example:
            >>> simulator = BankrollSimulator()
            >>> results = simulator.simulate_season(df, config)
            >>> simulator.plot_bankroll_paths(results)
        """
        if methods is None:
            methods = ["raw", "platt", "iso"]

        # Filter to only rows where at least one method has a stake
        mask = False
        for method in methods:
            stake_col = f"stake_{method}"
            if stake_col in results.columns:
                mask = mask | (results[stake_col] > 0)

        df_plot = results[mask].copy()

        if df_plot.empty:
            logger.warning("No bets to plot")
            return

        plt.figure(figsize=(12, 7))

        colors = {"raw": "C0", "platt": "C1", "iso": "C2"}

        for method in methods:
            bank_col = f"bank_{method}"
            if bank_col in df_plot.columns:
                plt.plot(
                    df_plot["date"],
                    df_plot[bank_col],
                    label=f"{method.capitalize()}-Kelly",
                    color=colors.get(method, None),
                    linewidth=2,
                )

        plt.axhline(
            y=self.starting_bankroll,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Starting bankroll",
        )

        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Bankroll ($)", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)

    def calculate_roi(self, final_bankroll: float) -> float:
        """
        Calculate Return on Investment.

        Args:
            final_bankroll: Final bankroll amount

        Returns:
            ROI as percentage

        Example:
            >>> simulator = BankrollSimulator(starting_bankroll=1000)
            >>> roi = simulator.calculate_roi(1250)
            >>> print(f"ROI: {roi:.2f}%")  # 25.00%
        """
        return (final_bankroll - self.starting_bankroll) / self.starting_bankroll * 100

    def calculate_metrics(
        self, results: pd.DataFrame, method: str = "raw"
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a betting strategy.

        Args:
            results: DataFrame from simulate_season()
            method: Method to analyze ("raw", "platt", or "iso")

        Returns:
            Dictionary with metrics:
            - roi: Return on Investment (%)
            - total_pnl: Total profit/loss
            - num_bets: Number of bets placed
            - win_rate: Percentage of winning bets
            - avg_stake: Average stake per bet
            - max_drawdown: Maximum drawdown from peak (%)
            - sharpe_ratio: Sharpe ratio of returns

        Example:
            >>> simulator = BankrollSimulator()
            >>> results = simulator.simulate_season(df, config)
            >>> metrics = simulator.calculate_metrics(results, method="platt")
            >>> print(f"Win rate: {metrics['win_rate']:.2f}%")
        """
        stake_col = f"stake_{method}"
        pnl_col = f"pnl_{method}"
        bank_col = f"bank_{method}"

        # Filter to actual bets
        bets = results[results[stake_col] > 0].copy()

        if bets.empty:
            return {
                "roi": 0.0,
                "total_pnl": 0.0,
                "num_bets": 0,
                "win_rate": 0.0,
                "avg_stake": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        # Basic metrics
        total_pnl = bets[pnl_col].sum()
        num_bets = len(bets)
        avg_stake = bets[stake_col].mean()
        win_rate = (bets[pnl_col] > 0).sum() / num_bets * 100 if num_bets > 0 else 0.0

        # ROI
        final_bank = results[bank_col].iloc[-1] if bank_col in results.columns else self.starting_bankroll
        roi = self.calculate_roi(final_bank)

        # Max drawdown
        bankroll_series = results[bank_col].dropna()
        if not bankroll_series.empty:
            running_max = bankroll_series.expanding().max()
            drawdown = (bankroll_series - running_max) / running_max * 100
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0.0

        # Sharpe ratio (annualized, assuming ~82 games per season)
        returns = bets[pnl_col] / bets[stake_col]
        if len(returns) > 1:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(82) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "roi": roi,
            "total_pnl": total_pnl,
            "num_bets": num_bets,
            "win_rate": win_rate,
            "avg_stake": avg_stake,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }


class BetRecommendationDisplay:
    """
    Displays and formats bet recommendations.

    Handles:
    - Finding latest enriched prediction files
    - Loading bet recommendations
    - Formatting bets as tables
    - Displaying recommended bets
    """

    def __init__(self, prediction_dir: Optional[str] = None):
        """
        Initialize the BetRecommendationDisplay.

        Args:
            prediction_dir: Directory containing prediction files
                          (defaults to output/LightGBM from project root)
        """
        if prediction_dir is None:
            # Auto-detect prediction directory
            script_dir = Path(__file__).parent.parent.parent
            self.prediction_dir = script_dir / "output" / "LightGBM"
        else:
            self.prediction_dir = Path(prediction_dir)

    def find_latest_enriched_file(self, pattern: str = "combined_nba_predictions_enriched_*.csv") -> Optional[Path]:
        """
        Find the most recent enriched predictions file.

        Args:
            pattern: Glob pattern to match files

        Returns:
            Path to most recent file, or None if not found

        Example:
            >>> display = BetRecommendationDisplay()
            >>> latest_file = display.find_latest_enriched_file()
            >>> if latest_file:
            ...     print(f"Found: {latest_file}")
        """
        with ErrorContext("Finding enriched predictions file", logger=logger):
            enriched_files = list(self.prediction_dir.glob(pattern))

            if not enriched_files:
                logger.warning(
                    f"No enriched predictions file found in {self.prediction_dir}. "
                    "Please run the Kelly Criterion script first."
                )
                return None

            # Get most recent by modification time
            latest_file = max(enriched_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using enriched predictions file: {latest_file}")

            return latest_file

    def load_recommendations(
        self,
        file_path: Optional[Union[str, Path]] = None,
        min_stake: float = 0.0,
    ) -> pd.DataFrame:
        """
        Load bet recommendations from enriched predictions file.

        Args:
            file_path: Path to enriched predictions CSV (auto-detects if None)
            min_stake: Minimum stake to include (filters out zero-stake bets)

        Returns:
            DataFrame with bet recommendations

        Raises:
            FileNotFoundError: If file not found and auto-detection fails

        Example:
            >>> display = BetRecommendationDisplay()
            >>> bets = display.load_recommendations(min_stake=1.0)
        """
        with ErrorContext("Loading bet recommendations", logger=logger):
            # Auto-detect file if not provided
            if file_path is None:
                file_path = self.find_latest_enriched_file()
                if file_path is None:
                    raise FileNotFoundError(
                        f"No enriched predictions file found in {self.prediction_dir}"
                    )

            # Load data
            df = pd.read_csv(file_path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            log_dataframe_info(df, name="Enriched predictions", logger=logger)

            # Filter for bets with non-zero stakes
            mask = (
                (df.get("stake_raw", 0) > min_stake)
                | (df.get("stake_platt", 0) > min_stake)
                | (df.get("stake_iso", 0) > min_stake)
            )
            bets = df[mask].copy()

            logger.info(f"Loaded {len(bets)} bet recommendations with stake > ${min_stake}")

            return bets

    def format_bet_table(
        self,
        bets_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> str:
        """
        Format bets as a readable table string.

        Args:
            bets_df: DataFrame with bet data
            columns: Columns to include in table (auto-selects if None)

        Returns:
            Formatted table string

        Example:
            >>> display = BetRecommendationDisplay()
            >>> bets = display.load_recommendations()
            >>> table = display.format_bet_table(bets)
            >>> print(table)
        """
        if bets_df.empty:
            return "No bets to display"

        # Default columns
        if columns is None:
            columns = [
                "date",
                "home_team",
                "away_team",
                "odds_1",
                "home_team_prob",
                "prob_platt",
                "prob_iso",
                "stake_raw",
                "pnl_raw",
                "stake_platt",
                "pnl_platt",
                "stake_iso",
                "pnl_iso",
            ]

        # Filter to existing columns
        available_cols = [c for c in columns if c in bets_df.columns]

        # Add win column if available
        if "win" in bets_df.columns and "win" not in available_cols:
            available_cols.insert(-6, "win")  # Insert before stakes

        # Sort by date
        df_display = bets_df[available_cols].sort_values("date")

        return df_display.to_string(index=False)

    def display_recommendations(
        self,
        file_path: Optional[Union[str, Path]] = None,
        min_stake: float = 0.0,
        show_summary: bool = True,
    ) -> None:
        """
        Load and display bet recommendations.

        Args:
            file_path: Path to enriched predictions CSV (auto-detects if None)
            min_stake: Minimum stake to include
            show_summary: Whether to show summary statistics

        Example:
            >>> display = BetRecommendationDisplay()
            >>> display.display_recommendations(min_stake=1.0)
        """
        try:
            # Load bets
            bets = self.load_recommendations(file_path=file_path, min_stake=min_stake)

            # Display table
            logger.info("\n" + "=" * 80)
            logger.info("BET RECOMMENDATIONS (Raw / Platt / Iso Kelly)")
            logger.info("=" * 80 + "\n")

            if not bets.empty:
                table = self.format_bet_table(bets)
                logger.info(f"\n{table}\n")

                # Summary statistics
                if show_summary:
                    logger.info("=" * 80)
                    logger.info("SUMMARY")
                    logger.info("=" * 80)
                    logger.info(f"Total bets: {len(bets)}")

                    for method in ["raw", "platt", "iso"]:
                        stake_col = f"stake_{method}"
                        pnl_col = f"pnl_{method}"

                        if stake_col in bets.columns and pnl_col in bets.columns:
                            method_bets = bets[bets[stake_col] > 0]
                            if not method_bets.empty:
                                total_stake = method_bets[stake_col].sum()
                                total_pnl = method_bets[pnl_col].sum()
                                num_bets = len(method_bets)
                                logger.info(
                                    f"{method.capitalize()}: {num_bets} bets, "
                                    f"Total stake=${total_stake:.2f}, "
                                    f"Total P&L=${total_pnl:.2f}"
                                )
                    logger.info("=" * 80)
            else:
                logger.warning("No bets found in this enriched dataset.")

        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error displaying recommendations: {e}")


# Convenience functions for backward compatibility and ease of use


def fetch_and_merge_odds(
    predictions_df: pd.DataFrame,
    games_df: pd.DataFrame,
    api_key: str,
    preferred_books: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to fetch odds and merge with predictions.

    Args:
        predictions_df: Predictions DataFrame
        games_df: Games schedule DataFrame
        api_key: The Odds API key
        preferred_books: Preferred sportsbooks

    Returns:
        Merged DataFrame with odds and value calculations

    Example:
        >>> merged = fetch_and_merge_odds(preds, games, api_key="your_key")
    """
    odds_mgr = OddsManager(api_key=api_key, preferred_books=preferred_books)
    odds_df = odds_mgr.fetch_odds(games_df)
    return odds_mgr.merge_with_predictions(predictions_df, odds_df)


def calibrate_and_simulate(
    historical_df: pd.DataFrame,
    today_predictions_df: pd.DataFrame,
    strategy_config: Dict[str, any],
    starting_bankroll: float = 1000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to calibrate probabilities and simulate season.

    Args:
        historical_df: Historical predictions with outcomes
        today_predictions_df: Today's predictions to calibrate
        strategy_config: Strategy configuration dict
        starting_bankroll: Starting bankroll amount

    Returns:
        Tuple of (calibrated_today_df, simulated_historical_df)

    Example:
        >>> today_cal, season_sim = calibrate_and_simulate(
        ...     historical, today, config, starting_bankroll=1000
        ... )
    """
    # Calibrate
    calibrator = ProbabilityCalibrator()

    X_hist = historical_df[["home_team_prob"]].values
    y_hist = historical_df["accuracy"].astype(int).values

    calibrator.fit_platt_scaling(X_hist, y_hist)
    calibrator.fit_isotonic_regression(X_hist, y_hist)

    today_calibrated = calibrator.calibrate_predictions(today_predictions_df)

    # Simulate
    simulator = BankrollSimulator(starting_bankroll=starting_bankroll)
    season_results = simulator.simulate_season(historical_df, strategy_config)

    return today_calibrated, season_results
