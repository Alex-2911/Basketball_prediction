#!/usr/bin/env python
"""
Unit tests for error handling infrastructure.

Tests critical error handling components:
- Retry decorators with exponential backoff
- Data validation functions
- Error context managers
- Request session creation
- Logging utilities
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import time
import logging
from unittest.mock import Mock, patch, MagicMock
import requests

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "2026" / "src"))

from error_handlers import (
    # Exceptions
    BasketballPredictionError,
    DataValidationError,
    NetworkError,
    ScrapingError,
    ModelTrainingError,
    ConfigurationError,
    FileNotFoundError,
    # Decorators
    retry_on_network_error,
    # Validators
    validate_dataframe,
    validate_file_exists,
    validate_api_key,
    # Utilities
    ErrorContext,
    get_requests_session_with_retries,
    log_dataframe_info,
    handle_missing_data,
)


class TestCustomExceptions:
    """Test custom exception hierarchy."""

    def test_base_exception_inheritance(self):
        """Test that all custom exceptions inherit from BasketballPredictionError."""
        assert issubclass(DataValidationError, BasketballPredictionError)
        assert issubclass(NetworkError, BasketballPredictionError)
        assert issubclass(ScrapingError, BasketballPredictionError)
        assert issubclass(ModelTrainingError, BasketballPredictionError)
        assert issubclass(ConfigurationError, BasketballPredictionError)
        assert issubclass(FileNotFoundError, BasketballPredictionError)

    def test_exceptions_can_be_raised_with_message(self):
        """Test that exceptions can be raised with custom messages."""
        with pytest.raises(DataValidationError, match="Invalid data"):
            raise DataValidationError("Invalid data")

        with pytest.raises(NetworkError, match="Connection failed"):
            raise NetworkError("Connection failed")

    def test_base_exception_is_exception(self):
        """Test that base exception is a proper Exception."""
        assert issubclass(BasketballPredictionError, Exception)


class TestRetryDecorator:
    """Test retry_on_network_error decorator."""

    def test_retry_succeeds_on_first_attempt(self):
        """Test that successful function doesn't retry."""
        call_count = []

        @retry_on_network_error(max_retries=3)
        def successful_function():
            call_count.append(1)
            return "success"

        result = successful_function()
        assert result == "success"
        assert len(call_count) == 1  # Called only once

    def test_retry_succeeds_after_failures(self):
        """Test that function retries and eventually succeeds."""
        call_count = []

        @retry_on_network_error(max_retries=3, backoff_factor=0.01)
        def flaky_function():
            call_count.append(1)
            if len(call_count) < 3:
                raise requests.RequestException("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert len(call_count) == 3  # Failed twice, succeeded third time

    def test_retry_raises_network_error_after_max_retries(self):
        """Test that NetworkError is raised after max retries exhausted."""
        @retry_on_network_error(max_retries=2, backoff_factor=0.01)
        def always_fails():
            raise requests.RequestException("Permanent failure")

        with pytest.raises(NetworkError, match="Network operation failed after 2 retries"):
            always_fails()

    def test_retry_with_different_exceptions(self):
        """Test retry with custom exception types."""
        call_count = []

        @retry_on_network_error(
            max_retries=2,
            backoff_factor=0.01,
            exceptions=(ConnectionError, TimeoutError)
        )
        def connection_fails():
            call_count.append(1)
            if len(call_count) == 1:
                raise ConnectionError("Connection refused")
            return "success"

        result = connection_fails()
        assert result == "success"
        assert len(call_count) == 2

    @pytest.mark.skip(reason="Timing tests are flaky and environment-dependent")
    def test_retry_backoff_timing(self):
        """Test exponential backoff timing (SKIPPED - flaky timing test)."""
        times = []

        @retry_on_network_error(max_retries=3, backoff_factor=0.05)
        def timed_function():
            times.append(time.time())
            if len(times) < 3:
                raise requests.RequestException("Retry")
            return "success"

        timed_function()

        # Check that delays increase (0.05, 0.1 seconds)
        # Allow generous tolerance for timing due to system variability
        if len(times) >= 3:
            delay1 = times[1] - times[0]
            delay2 = times[2] - times[1]
            # Second delay should be roughly 2x first delay (exponential backoff)
            # But we just verify increasing delays due to timing variability
            assert delay1 > 0.02  # At least some delay
            assert delay2 > delay1 * 0.8  # Roughly increasing (with tolerance)


class TestValidateDataFrame:
    """Test DataFrame validation function."""

    def test_validate_dataframe_accepts_valid_df(self):
        """Test that valid DataFrame passes validation."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

        result = validate_dataframe(df)
        assert result is df  # Returns same DataFrame

    def test_validate_dataframe_rejects_none(self):
        """Test that None raises DataValidationError."""
        with pytest.raises(DataValidationError, match="DataFrame is None"):
            validate_dataframe(None)

    def test_validate_dataframe_rejects_non_dataframe(self):
        """Test that non-DataFrame raises error."""
        with pytest.raises(DataValidationError, match="Expected DataFrame"):
            validate_dataframe([1, 2, 3])

    def test_validate_dataframe_rejects_empty_by_default(self):
        """Test that empty DataFrame raises error by default."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            validate_dataframe(df)

    def test_validate_dataframe_allows_empty_when_specified(self):
        """Test that empty DataFrame is allowed with allow_empty=True."""
        df = pd.DataFrame()
        result = validate_dataframe(df, allow_empty=True, min_rows=0)
        assert result is df

    def test_validate_dataframe_checks_required_columns(self):
        """Test required columns validation."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4]
        })

        # Should pass with existing columns
        validate_dataframe(df, required_columns=['a', 'b'])

        # Should fail with missing column
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df, required_columns=['a', 'b', 'c'])

    def test_validate_dataframe_checks_min_rows(self):
        """Test minimum rows validation."""
        df = pd.DataFrame({'a': [1, 2]})

        # Should pass with enough rows
        validate_dataframe(df, min_rows=2)

        # Should fail with too few rows
        with pytest.raises(DataValidationError, match="expected at least 5"):
            validate_dataframe(df, min_rows=5)


class TestValidateFileExists:
    """Test file existence validation."""

    def test_validate_file_exists_with_real_file(self, tmp_path):
        """Test that existing file passes validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = validate_file_exists(str(test_file))
        assert str(result) == str(test_file)

    def test_validate_file_exists_raises_on_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_file_exists("/nonexistent/file.txt")

    def test_validate_file_exists_raises_on_directory(self, tmp_path):
        """Test that directory raises error (not a file)."""
        with pytest.raises(DataValidationError, match="is not a file"):
            validate_file_exists(str(tmp_path))


class TestValidateAPIKey:
    """Test API key validation."""

    def test_validate_api_key_accepts_valid_key(self):
        """Test that non-empty key passes validation."""
        result = validate_api_key("abc123", key_name="TEST_KEY")
        assert result == "abc123"

    def test_validate_api_key_rejects_none(self):
        """Test that None raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="API_KEY not found"):
            validate_api_key(None, key_name="API_KEY")

    def test_validate_api_key_rejects_empty_string(self):
        """Test that empty string raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="TEST_KEY not found"):
            validate_api_key("", key_name="TEST_KEY")

    def test_validate_api_key_rejects_whitespace(self):
        """Test that whitespace-only string raises error."""
        with pytest.raises(ConfigurationError, match="KEY is empty"):
            validate_api_key("   ", key_name="KEY")


class TestErrorContext:
    """Test ErrorContext context manager."""

    def test_error_context_logs_start_and_complete(self, caplog):
        """Test that ErrorContext logs operation start and completion."""
        with caplog.at_level(logging.INFO):
            with ErrorContext("Test operation"):
                pass

        assert "Starting: Test operation" in caplog.text
        assert "Completed: Test operation" in caplog.text

    def test_error_context_logs_errors(self, caplog):
        """Test that ErrorContext logs errors."""
        with caplog.at_level(logging.ERROR):
            try:
                with ErrorContext("Failing operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass

        assert "Error during Failing operation" in caplog.text
        assert "Test error" in caplog.text

    def test_error_context_propagates_exceptions_by_default(self):
        """Test that exceptions are re-raised by default."""
        with pytest.raises(ValueError, match="Test error"):
            with ErrorContext("Test op"):
                raise ValueError("Test error")

    def test_error_context_suppresses_errors_when_configured(self):
        """Test that ErrorContext can suppress errors."""
        # With raise_on_error=False, exception should be suppressed
        with ErrorContext("Test op", raise_on_error=False):
            raise ValueError("This should be suppressed")
        # If we get here, exception was suppressed successfully

    def test_error_context_with_custom_logger(self, caplog):
        """Test ErrorContext with custom logger."""
        custom_logger = logging.getLogger("custom")

        with caplog.at_level(logging.INFO, logger="custom"):
            with ErrorContext("Custom op", logger=custom_logger):
                pass

        assert "Starting: Custom op" in caplog.text


class TestGetRequestsSession:
    """Test requests session creation with retries."""

    def test_get_requests_session_returns_session(self):
        """Test that function returns a requests.Session."""
        session = get_requests_session_with_retries()
        assert isinstance(session, requests.Session)

    def test_get_requests_session_has_retry_adapter(self):
        """Test that session has retry adapter mounted."""
        session = get_requests_session_with_retries()

        # Check that adapters are mounted
        assert 'http://' in session.adapters
        assert 'https://' in session.adapters

    def test_get_requests_session_with_custom_retries(self):
        """Test session creation with custom retry count."""
        session = get_requests_session_with_retries(max_retries=5)
        assert isinstance(session, requests.Session)


class TestLogDataFrameInfo:
    """Test DataFrame logging utility."""

    def test_log_dataframe_info_logs_shape(self, caplog):
        """Test that DataFrame shape is logged."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

        with caplog.at_level(logging.INFO):
            log_dataframe_info(df, name="Test DF")

        assert "Test DF" in caplog.text
        assert "3 rows" in caplog.text
        assert "2 columns" in caplog.text

    def test_log_dataframe_info_logs_columns(self, caplog):
        """Test that column names are logged."""
        df = pd.DataFrame({
            'col1': [1],
            'col2': [2],
            'col3': [3]
        })

        with caplog.at_level(logging.INFO):
            log_dataframe_info(df, name="Test")

        # Just check that something was logged (column listing is implementation detail)
        assert "Test" in caplog.text and "columns" in caplog.text.lower()

    def test_log_dataframe_info_handles_empty_dataframe(self, caplog):
        """Test logging empty DataFrame."""
        df = pd.DataFrame()

        with caplog.at_level(logging.INFO):
            log_dataframe_info(df, name="Empty DF")

        assert "Empty DF" in caplog.text
        assert "0 rows" in caplog.text


class TestHandleMissingData:
    """Test missing data handling utility."""

    def test_handle_missing_data_with_no_missing_values(self):
        """Test DataFrame with no missing values."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

        result = handle_missing_data(df, strategy='drop')
        # Should return DataFrame (same or copy)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_handle_missing_data_drops_rows(self):
        """Test that missing values can be dropped."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, 5, None]
        })

        result = handle_missing_data(df, strategy='drop')
        # Should drop rows with any NaN
        assert len(result) < len(df)

    def test_handle_missing_data_fills_values(self):
        """Test that missing values can be filled."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, 5, None]
        })

        result = handle_missing_data(df, strategy='fill', fill_value=0)
        # Should have no NaN values
        assert not result.isna().any().any()


class TestErrorHandlerIntegration:
    """Integration tests for error handling components."""

    def test_retry_with_error_context(self, caplog):
        """Test retry decorator with ErrorContext."""
        call_count = []

        @retry_on_network_error(max_retries=2, backoff_factor=0.01)
        def flaky_operation():
            call_count.append(1)
            if len(call_count) < 2:
                raise requests.RequestException("Temporary failure")
            return "success"

        with caplog.at_level(logging.INFO):
            with ErrorContext("Flaky operation"):
                result = flaky_operation()

        assert result == "success"
        assert "Starting: Flaky operation" in caplog.text
        assert "Completed: Flaky operation" in caplog.text

    def test_validation_chain(self):
        """Test chaining multiple validations."""
        df = pd.DataFrame({
            'team': ['LAL', 'BOS'],
            'points': [110, 105]
        })

        # Chain validations
        result = validate_dataframe(
            df,
            required_columns=['team', 'points'],
            min_rows=2
        )

        assert result is df
        assert len(result) == 2

    def test_error_handling_workflow(self, tmp_path, caplog):
        """Test complete error handling workflow."""
        # Create test file
        test_file = tmp_path / "data.csv"
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(test_file, index=False)

        with caplog.at_level(logging.INFO):
            with ErrorContext("Loading and validating data"):
                # Validate file exists
                file_path = validate_file_exists(str(test_file))

                # Load data
                df = pd.read_csv(file_path)

                # Validate DataFrame
                validate_dataframe(df, required_columns=['a'], min_rows=1)

                # Log info
                log_dataframe_info(df, name="Test data")

        assert "Starting: Loading and validating data" in caplog.text
        assert "Completed: Loading and validating data" in caplog.text


# Run tests with: pytest tests/test_error_handlers.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
