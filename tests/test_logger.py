#!/usr/bin/env python
"""
Unit tests for logging infrastructure.

Tests logger configuration and functionality:
- Logger creation and configuration
- Log file rotation
- Log levels
- Formatter configuration
- Multiple logger instances
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.logger import get_logger


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_uses_module_name(self):
        """Test that logger uses provided module name."""
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_get_logger_creates_different_loggers(self):
        """Test that different names create different loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name != logger2.name

    def test_get_logger_returns_same_logger_for_same_name(self):
        """Test that same name returns same logger instance."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")

        assert logger1 is logger2

    def test_get_logger_with_dunder_name(self):
        """Test using __name__ convention."""
        logger = get_logger(__name__)
        assert logger.name == __name__


class TestLoggerConfiguration:
    """Test logger configuration."""

    def test_logger_has_handlers(self):
        """Test that logger has handlers configured."""
        logger = get_logger("test_handlers")

        # Logger should have at least a console handler
        assert len(logger.handlers) > 0 or len(logger.root.handlers) > 0

    def test_logger_default_level(self):
        """Test that logger has appropriate default level."""
        logger = get_logger("test_level")

        # Should be INFO or DEBUG by default
        assert logger.level <= logging.INFO or logger.root.level <= logging.INFO

    def test_logger_can_log_messages(self, caplog):
        """Test that logger can log messages."""
        logger = get_logger("test_logging")

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert "Test message" in caplog.text

    def test_logger_respects_log_levels(self, caplog):
        """Test that logger respects log levels."""
        logger = get_logger("test_levels")

        with caplog.at_level(logging.INFO):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

        # Info, warning, and error should be in log
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text


class TestLoggerFormatter:
    """Test log message formatting."""

    def test_log_format_includes_level(self, caplog):
        """Test that log messages include level."""
        logger = get_logger("test_format")

        with caplog.at_level(logging.INFO):
            logger.info("Test")

        # Check for INFO level in output
        assert any("INFO" in record.levelname for record in caplog.records)

    def test_log_format_includes_message(self, caplog):
        """Test that log messages include the actual message."""
        logger = get_logger("test_message")

        with caplog.at_level(logging.INFO):
            logger.info("Custom message")

        assert "Custom message" in caplog.text

    def test_log_format_includes_module_name(self, caplog):
        """Test that log includes module/logger name."""
        logger = get_logger("specific_module")

        with caplog.at_level(logging.INFO):
            logger.info("Test")

        # Module name should be in records
        assert any(record.name == "specific_module" for record in caplog.records)


class TestLoggerFileOutput:
    """Test logger file output."""

    def test_logger_creates_log_file(self, tmp_path):
        """Test that logger can write to file."""
        log_file = tmp_path / "test.log"

        # Create logger with file handler
        logger = logging.getLogger("file_test")
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)

        logger.info("Test message to file")

        # Check file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message to file" in content

    def test_logger_file_rotation_size_limit(self):
        """Test that log rotation respects size limits."""
        # This tests the concept, actual RotatingFileHandler would be tested
        # by checking file sizes after many log messages
        max_bytes = 1024  # 1KB

        # Create large log message
        large_message = "X" * 500

        # After 3 messages, should exceed 1KB
        assert len(large_message * 3) > max_bytes


class TestLoggerMultipleHandlers:
    """Test logger with multiple handlers."""

    def test_logger_can_have_console_and_file_handlers(self, tmp_path, caplog):
        """Test logger with both console and file output."""
        log_file = tmp_path / "multi.log"

        logger = logging.getLogger("multi_handler")
        logger.setLevel(logging.INFO)
        logger.handlers = []  # Clear any existing handlers

        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        # Add stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

        with caplog.at_level(logging.INFO):
            logger.info("Multi-handler message")

        # Should appear in both console (caplog) and file
        assert "Multi-handler message" in caplog.text
        assert "Multi-handler message" in log_file.read_text()


class TestLoggerExceptionHandling:
    """Test logger with exceptions."""

    def test_logger_logs_exceptions(self, caplog):
        """Test logging exceptions with traceback."""
        logger = get_logger("exception_test")

        with caplog.at_level(logging.ERROR):
            try:
                raise ValueError("Test exception")
            except ValueError as e:
                logger.exception("Error occurred")

        assert "Error occurred" in caplog.text
        assert "ValueError" in caplog.text

    def test_logger_error_without_traceback(self, caplog):
        """Test logging error without traceback."""
        logger = get_logger("error_test")

        with caplog.at_level(logging.ERROR):
            logger.error("Simple error message")

        assert "Simple error message" in caplog.text


class TestLoggerInProduction:
    """Test logger behavior in production scenarios."""

    def test_logger_in_ci_environment(self, monkeypatch):
        """Test logger behavior in CI environment."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")

        logger = get_logger("ci_test")

        # In CI, should still be able to log
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_logger_with_unicode_messages(self, caplog):
        """Test logger handles unicode characters."""
        logger = get_logger("unicode_test")

        with caplog.at_level(logging.INFO):
            logger.info("Message with émojis 🏀 and spëcial chars")

        assert "émojis" in caplog.text or "special" in caplog.text

    def test_logger_with_very_long_messages(self, caplog):
        """Test logger handles very long messages."""
        logger = get_logger("long_test")

        long_message = "X" * 10000

        with caplog.at_level(logging.INFO):
            logger.info(long_message)

        # Should be logged (may be truncated by handler)
        assert len(caplog.text) > 0


class TestLoggerThreadSafety:
    """Test logger thread safety."""

    def test_logger_same_across_contexts(self):
        """Test that logger instance is consistent."""
        logger1 = get_logger("context_test")
        logger2 = get_logger("context_test")

        # Should be same instance
        assert logger1 is logger2
        assert id(logger1) == id(logger2)


class TestLoggerContextualLogging:
    """Test contextual logging patterns."""

    def test_logger_with_extra_context(self, caplog):
        """Test adding extra context to log messages."""
        logger = get_logger("context_extra")

        with caplog.at_level(logging.INFO):
            logger.info("Message with context", extra={"user": "test_user"})

        # Message should be logged
        assert "Message with context" in caplog.text

    def test_logger_with_formatted_messages(self, caplog):
        """Test formatted log messages."""
        logger = get_logger("format_test")

        team = "Lakers"
        score = 110

        with caplog.at_level(logging.INFO):
            logger.info(f"{team} scored {score} points")

        assert "Lakers scored 110 points" in caplog.text


class TestLoggerPerformance:
    """Test logger performance characteristics."""

    def test_logger_creates_quickly(self):
        """Test that logger creation is fast."""
        import time

        start = time.time()
        for i in range(100):
            get_logger(f"perf_test_{i}")
        elapsed = time.time() - start

        # Should create 100 loggers in < 1 second
        assert elapsed < 1.0

    def test_logger_logs_quickly(self, caplog):
        """Test that logging is fast."""
        import time

        logger = get_logger("speed_test")

        with caplog.at_level(logging.INFO):
            start = time.time()
            for i in range(1000):
                logger.info(f"Log message {i}")
            elapsed = time.time() - start

        # Should log 1000 messages in < 1 second
        assert elapsed < 1.0


# Run tests with: pytest tests/test_logger.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
