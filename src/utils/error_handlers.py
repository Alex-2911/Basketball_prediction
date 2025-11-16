#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error handling infrastructure for Basketball prediction pipeline.

Provides:
- Custom exception classes for domain-specific errors
- Retry decorators for network operations
- Graceful error recovery utilities
- Data validation helpers
"""

import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────


class BasketballPredictionError(Exception):
    """Base exception for all basketball prediction errors."""

    pass


class DataValidationError(BasketballPredictionError):
    """Raised when data validation fails."""

    pass


class NetworkError(BasketballPredictionError):
    """Raised when network operations fail after retries."""

    pass


class ScrapingError(BasketballPredictionError):
    """Raised when web scraping fails."""

    pass


class ModelTrainingError(BasketballPredictionError):
    """Raised when ML model training fails."""

    pass


class ConfigurationError(BasketballPredictionError):
    """Raised when configuration is invalid or missing."""

    pass


class FileNotFoundError(BasketballPredictionError):
    """Raised when required files are not found."""

    pass


# ─────────────────────────────────────────────────────────
# RETRY DECORATORS
# ─────────────────────────────────────────────────────────


def retry_on_network_error(
    max_retries: int = 4,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
    ),
):
    """
    Decorator to retry network operations with exponential backoff.

    Default retry schedule: 2s, 4s, 8s, 16s

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff (2.0 = double each time)
        exceptions: Tuple of exception types to catch and retry

    Example:
        @retry_on_network_error(max_retries=3)
        def fetch_data(url):
            return requests.get(url)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise NetworkError(
                            f"Network operation failed after {max_retries} retries"
                        ) from e

                    wait_time = backoff_factor**attempt
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)

            # Should never reach here
            raise NetworkError(f"Retry logic error in {func.__name__}")

        return wrapper

    return decorator


def get_requests_session_with_retries(
    max_retries: int = 4,
    backoff_factor: float = 2.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Create a requests Session with automatic retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        status_forcelist: HTTP status codes that trigger a retry

    Returns:
        Configured requests.Session object

    Example:
        session = get_requests_session_with_retries()
        response = session.get("https://api.example.com/data")
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],  # formerly method_whitelist
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# ─────────────────────────────────────────────────────────
# DATA VALIDATION
# ─────────────────────────────────────────────────────────


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 1,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """
    Validate DataFrame meets basic requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        min_rows: Minimum number of rows required
        allow_empty: If True, empty DataFrames are valid

    Returns:
        The validated DataFrame (unchanged)

    Raises:
        DataValidationError: If validation fails

    Example:
        df = validate_dataframe(
            df,
            required_columns=["team", "date", "points"],
            min_rows=10
        )
    """
    if df is None:
        raise DataValidationError("DataFrame is None")

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got {type(df)}")

    if df.empty and not allow_empty:
        raise DataValidationError("DataFrame is empty")

    if len(df) < min_rows:
        raise DataValidationError(f"DataFrame has {len(df)} rows, expected at least {min_rows}")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    return df


def validate_file_exists(file_path: str, description: str = "File") -> Path:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file
        description: Human-readable description for error messages

    Returns:
        Path object for the validated file

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        csv_path = validate_file_exists(
            "data/games.csv",
            description="Game data CSV"
        )
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")

    if not path.is_file():
        raise DataValidationError(f"{description} is not a file: {file_path}")

    return path


def validate_api_key(api_key: Optional[str], key_name: str = "API_KEY") -> str:
    """
    Validate that API key exists and is non-empty.

    Args:
        api_key: The API key to validate
        key_name: Name of the key for error messages

    Returns:
        The validated API key

    Raises:
        ConfigurationError: If API key is invalid

    Example:
        api_key = validate_api_key(
            os.getenv("ODDS_API_KEY"),
            key_name="ODDS_API_KEY"
        )
    """
    if not api_key:
        raise ConfigurationError(
            f"{key_name} not found in environment variables. "
            f"Please create a .env file based on .env.example and add your key."
        )

    if not isinstance(api_key, str):
        raise ConfigurationError(f"{key_name} must be a string, got {type(api_key)}")

    if len(api_key.strip()) == 0:
        raise ConfigurationError(f"{key_name} is empty")

    return api_key.strip()


# ─────────────────────────────────────────────────────────
# GRACEFUL ERROR RECOVERY
# ─────────────────────────────────────────────────────────


def safe_file_operation(
    operation: Callable, fallback_value: Any = None, log_error: bool = True
) -> Any:
    """
    Execute a file operation with graceful error handling.

    Args:
        operation: Callable that performs the file operation
        fallback_value: Value to return if operation fails
        log_error: Whether to log errors

    Returns:
        Result of operation, or fallback_value if it fails

    Example:
        df = safe_file_operation(
            lambda: pd.read_csv("data.csv"),
            fallback_value=pd.DataFrame()
        )
    """
    try:
        return operation()
    except Exception as e:
        if log_error:
            logger = logging.getLogger(__name__)
            logger.error(f"File operation failed: {e}")
        return fallback_value


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform division with zero-division protection.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division by zero

    Returns:
        Result of division, or default if denominator is zero

    Example:
        win_rate = safe_division(wins, total_games, default=0.0)
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def handle_missing_data(
    df: pd.DataFrame, strategy: str = "drop", columns: Optional[list] = None, fill_value: Any = None
) -> pd.DataFrame:
    """
    Handle missing data in DataFrame with specified strategy.

    Args:
        df: DataFrame to process
        strategy: "drop", "fill", or "forward_fill"
        columns: Specific columns to process (None = all columns)
        fill_value: Value to use when strategy="fill"

    Returns:
        DataFrame with missing data handled

    Example:
        df = handle_missing_data(df, strategy="fill", fill_value=0)
    """
    df = df.copy()

    if strategy == "drop":
        return df.dropna(subset=columns) if columns else df.dropna()

    elif strategy == "fill":
        if fill_value is None:
            raise ValueError("fill_value required when strategy='fill'")
        if columns:
            df[columns] = df[columns].fillna(fill_value)
        else:
            df = df.fillna(fill_value)
        return df

    elif strategy == "forward_fill":
        if columns:
            df[columns] = df[columns].fillna(method="ffill")
        else:
            df = df.fillna(method="ffill")
        return df

    else:
        raise ValueError(f"Unknown strategy: {strategy}. " f"Use 'drop', 'fill', or 'forward_fill'")


# ─────────────────────────────────────────────────────────
# LOGGING UTILITIES
# ─────────────────────────────────────────────────────────


def log_dataframe_info(
    df: pd.DataFrame, name: str = "DataFrame", logger: Optional[logging.Logger] = None
) -> None:
    """
    Log useful information about a DataFrame.

    Args:
        df: DataFrame to log information about
        name: Name to use in log messages
        logger: Logger instance (uses root logger if None)

    Example:
        log_dataframe_info(games_df, name="Games Data")
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"{name}: {len(df)} rows, {len(df.columns)} columns")

    if df.empty:
        logger.warning(f"{name} is empty")
        return

    # Log missing data
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(f"{name} missing data:")
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            logger.info(f"  - {col}: {count} ({pct:.1f}%)")


def log_function_call(func_name: str, args: dict, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a function call with its arguments.

    Args:
        func_name: Name of the function being called
        args: Dictionary of argument names and values
        logger: Logger instance (uses root logger if None)

    Example:
        log_function_call("scrape_data", {"url": url, "date": date})
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    logger.debug(f"Calling {func_name}({args_str})")


# ─────────────────────────────────────────────────────────
# CONTEXT MANAGERS
# ─────────────────────────────────────────────────────────


class ErrorContext:
    """
    Context manager for comprehensive error handling.

    Example:
        with ErrorContext("Loading game data", logger=my_logger):
            df = pd.read_csv("games.csv")
            # ... more operations
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        raise_on_error: bool = True,
    ):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.raise_on_error = raise_on_error

    def __enter__(self):
        self.logger.info(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation_name}")
            return True

        self.logger.error(
            f"Error during {self.operation_name}: {exc_val}", exc_info=(exc_type, exc_val, exc_tb)
        )

        return not self.raise_on_error  # Suppress exception if raise_on_error=False


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Example 1: Retry decorator
    @retry_on_network_error(max_retries=3)
    def fetch_odds(url):
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    # Example 2: Data validation
    try:
        df = pd.DataFrame({"team": ["LAL", "BOS"], "points": [110, 105]})
        validate_dataframe(df, required_columns=["team", "points"], min_rows=2)
        logger.info("✓ Data validation passed")
    except DataValidationError as e:
        logger.error(f"✗ Data validation failed: {e}")

    # Example 3: Error context
    with ErrorContext("Example operation", logger=logger, raise_on_error=False):
        log_dataframe_info(df, name="Example DataFrame", logger=logger)

    print("\nError handling module loaded successfully!")
