#!/usr/bin/env python
"""
Logging Infrastructure for Basketball Prediction System

Provides centralized logging configuration with:
- Console and file logging
- Rotating file handlers
- Script-specific log files
- Configurable log levels

Usage:
    from logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("An error occurred", exc_info=True)
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class LoggerSetup:
    """
    Centralized logging configuration.

    Creates logger instances with console and file handlers
    configured according to configuration settings.
    """

    _initialized = False
    _log_dir: Optional[Path] = None

    @classmethod
    def initialize(
        cls,
        log_dir: str = "logs",
        log_level: str = "INFO",
        console_enabled: bool = True,
        file_enabled: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        """
        Initialize logging system (call once at startup).

        Args:
            log_dir: Directory for log files
            log_level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_enabled: Enable console logging
            file_enabled: Enable file logging
            max_bytes: Max size per log file before rotation
            backup_count: Number of backup files to keep
        """
        if cls._initialized:
            return

        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)

        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        # Remove any existing handlers
        root_logger.handlers.clear()

        # Console handler
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler (main log)
        if file_enabled:
            main_log_file = cls._log_dir / "basketball_prediction.log"
            file_handler = RotatingFileHandler(
                main_log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        cls._initialized = True

        # Log initialization
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log directory: {cls._log_dir}")

    @classmethod
    def get_script_logger(cls, script_name: str, separate_file: bool = True) -> logging.Logger:
        """
        Get logger for a specific script.

        Args:
            script_name: Name of the script (e.g., '1_get_data')
            separate_file: Create separate log file for this script

        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.initialize()

        logger = logging.getLogger(script_name)

        # Add script-specific file handler if requested
        if separate_file and cls._log_dir:
            # Create script-specific log file
            script_log_file = cls._log_dir / f"{script_name}.log"

            # Check if handler already exists
            handler_exists = any(
                isinstance(h, RotatingFileHandler) and h.baseFilename == str(script_log_file)
                for h in logger.handlers
            )

            if not handler_exists:
                file_handler = RotatingFileHandler(
                    script_log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5,
                    encoding="utf-8",
                )
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - "
                    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        return logger


def get_logger(name: str, separate_file: bool = False) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name (typically __name__)
        separate_file: Create separate log file for this logger

    Returns:
        Logger instance

    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if not LoggerSetup._initialized:
        LoggerSetup.initialize()

    if separate_file:
        # Extract script name from module name
        script_name = name.split(".")[-1]
        return LoggerSetup.get_script_logger(script_name, separate_file=True)

    return logging.getLogger(name)


def initialize_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_enabled: bool = True,
    file_enabled: bool = True,
):
    """
    Initialize logging system (convenience function).

    Call this once at the start of your script.

    Args:
        log_dir: Directory for log files
        log_level: Default log level
        console_enabled: Enable console logging
        file_enabled: Enable file logging

    Example:
        >>> from logger import initialize_logging, get_logger
        >>> initialize_logging(log_level="DEBUG")
        >>> logger = get_logger(__name__)
    """
    LoggerSetup.initialize(
        log_dir=log_dir,
        log_level=log_level,
        console_enabled=console_enabled,
        file_enabled=file_enabled,
    )


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls.

    Args:
        logger: Logger instance

    Example:
        >>> from logger import get_logger, log_function_call
        >>> logger = get_logger(__name__)
        >>>
        >>> @log_function_call(logger)
        >>> def my_function(arg1, arg2):
        >>>     return arg1 + arg2
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}(args={args}, kwargs={kwargs})")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance

    Example:
        >>> from logger import get_logger, log_execution_time
        >>> logger = get_logger(__name__)
        >>>
        >>> @log_execution_time(logger)
        >>> def slow_function():
        >>>     time.sleep(2)
    """
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


def create_run_log(script_name: str) -> Path:
    """
    Create a timestamped log file for this run.

    Args:
        script_name: Name of the script

    Returns:
        Path to the run log file

    Example:
        >>> log_file = create_run_log("1_get_data_previous_game_day_2026")
        >>> # Creates: logs/runs/1_get_data_previous_game_day_2026_2025-11-15_120000.log
    """
    if not LoggerSetup._log_dir:
        LoggerSetup.initialize()

    runs_dir = LoggerSetup._log_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = runs_dir / f"{script_name}_{timestamp}.log"

    return log_file


if __name__ == "__main__":
    # Test logging setup
    initialize_logging(log_level="DEBUG")

    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    try:
        raise ValueError("Test exception")
    except Exception:
        logger.exception("Exception caught:")

    print("\n✅ Logging test complete. Check logs/ directory for output.")

    # Test script-specific logger
    script_logger = get_logger("test_script", separate_file=True)
    script_logger.info("This goes to a separate file")

    print(f"✅ Script-specific log created: logs/test_script.log")
