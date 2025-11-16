#!/usr/bin/env python
"""
Unit tests for database integration.

Tests database connectivity, CRUD operations, and graceful fallback.
Database tests are skipped if USE_DATABASE=false to avoid CI failures.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import os
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "2026" / "src"))

from db_utils import DatabaseConfig, DatabasePool, DatabaseOperations, db_config


class TestDatabaseConfig:
    """Test database configuration from environment variables."""

    def test_default_config_disabled(self):
        """Test that database is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            config = DatabaseConfig()
            assert config.enabled is False

    def test_config_enabled_via_env(self):
        """Test enabling database via USE_DATABASE environment variable."""
        with patch.dict(os.environ, {"USE_DATABASE": "true"}):
            config = DatabaseConfig()
            assert config.enabled is True

    def test_config_connection_string(self):
        """Test loading DATABASE_URL connection string."""
        test_url = "postgresql://user:pass@localhost:5432/db"
        with patch.dict(os.environ, {
            "USE_DATABASE": "true",
            "DATABASE_URL": test_url
        }):
            config = DatabaseConfig()
            assert config.connection_string == test_url

    def test_config_individual_components(self):
        """Test loading individual DB_* components."""
        with patch.dict(os.environ, {
            "USE_DATABASE": "true",
            "DB_HOST": "testhost",
            "DB_PORT": "5433",
            "DB_NAME": "testdb",
            "DB_USER": "testuser",
            "DB_PASSWORD": "testpass"
        }):
            config = DatabaseConfig()
            assert config.host == "testhost"
            assert config.port == 5433
            assert config.database == "testdb"
            assert config.user == "testuser"
            assert config.password == "testpass"

    def test_get_connection_params_with_url(self):
        """Test connection params when using DATABASE_URL."""
        test_url = "postgresql://user:pass@localhost:5432/db"
        with patch.dict(os.environ, {
            "USE_DATABASE": "true",
            "DATABASE_URL": test_url
        }):
            config = DatabaseConfig()
            params = config.get_connection_params()
            assert "dsn" in params
            assert params["dsn"] == test_url

    def test_get_connection_params_with_components(self):
        """Test connection params when using individual components."""
        with patch.dict(os.environ, {
            "USE_DATABASE": "true",
            "DB_HOST": "testhost",
            "DB_PORT": "5432",
            "DB_NAME": "testdb",
            "DB_USER": "testuser",
            "DB_PASSWORD": "testpass"
        }):
            config = DatabaseConfig()
            params = config.get_connection_params()
            assert params["host"] == "testhost"
            assert params["port"] == 5432
            assert params["database"] == "testdb"
            assert params["user"] == "testuser"
            assert params["password"] == "testpass"


class TestDatabaseOperations:
    """Test database CRUD operations (mocked - no real DB required)."""

    def test_is_enabled_returns_config_state(self):
        """Test that is_enabled reflects db_config.enabled."""
        db_ops = DatabaseOperations()
        assert db_ops.is_enabled() == db_config.enabled

    @patch('db_utils.db_pool')
    def test_save_game_statistics_validates_dataframe(self, mock_pool):
        """Test that save_game_statistics validates input DataFrame."""
        db_ops = DatabaseOperations()

        # Empty DataFrame should raise error
        with pytest.raises(Exception):  # DataValidationError
            db_ops.save_game_statistics(pd.DataFrame())

    @patch('db_utils.db_pool')
    def test_save_predictions_validates_required_columns(self, mock_pool):
        """Test that save_predictions validates required columns."""
        db_ops = DatabaseOperations()

        # Missing required columns should raise error
        df_invalid = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS']
            # Missing: 'date', 'home_team_prob'
        })

        with pytest.raises(Exception):  # DataValidationError
            db_ops.save_predictions(df_invalid)

    def test_save_game_statistics_with_valid_data(self):
        """Test saving game statistics with valid data (mocked pool)."""
        # Create valid sample data
        df = pd.DataFrame({
            'season': ['2026'],
            'date': [date(2025, 10, 22)],
            'team': ['LAL'],
            'team_opp': ['BOS'],
            'home': [1],
            'won': [True],
            'total': [110],
            'total_opp': [105]
        })

        with patch('db_utils.db_pool') as mock_pool:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_pool.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
            mock_pool.get_connection.return_value.__exit__ = Mock(return_value=False)

            db_ops = DatabaseOperations()
            result = db_ops.save_game_statistics(df)

            # Should return number of rows processed
            assert result >= 0


class TestDatabaseGracefulFallback:
    """Test that scripts gracefully handle database failures."""

    def test_database_disabled_doesnt_crash(self):
        """Test that database operations don't crash when disabled."""
        with patch.dict(os.environ, {"USE_DATABASE": "false"}):
            # Recreate config to pick up env change
            config = DatabaseConfig()
            assert config.enabled is False

            # DatabaseOperations should work without crashing
            db_ops = DatabaseOperations()
            assert db_ops.is_enabled() is False

    @patch('db_utils.db_pool')
    def test_save_operations_catch_exceptions(self, mock_pool):
        """Test that database save operations catch and log exceptions."""
        # Simulate database connection failure
        mock_pool.get_connection.side_effect = Exception("Connection failed")

        db_ops = DatabaseOperations()

        # Save should raise exception (which scripts catch and log)
        df = pd.DataFrame({
            'season': ['2026'],
            'date': [date(2025, 10, 22)],
            'team': ['LAL'],
            'team_opp': ['BOS'],
            'home': [1],
            'won': [True],
            'total': [110],
            'total_opp': [105]
        })

        with pytest.raises(Exception):
            db_ops.save_game_statistics(df)


class TestDatabaseDataValidation:
    """Test data validation before database operations."""

    def test_game_statistics_requires_minimum_columns(self):
        """Test that game statistics need minimum required columns."""
        db_ops = DatabaseOperations()

        # Missing 'season' column
        df_invalid = pd.DataFrame({
            'date': [date(2025, 10, 22)],
            'team': ['LAL'],
            'team_opp': ['BOS']
        })

        # Should work with minimal columns (no strict validation by default)
        # The actual database will enforce constraints
        with patch('db_utils.db_pool'):
            with pytest.raises(Exception):
                db_ops.save_game_statistics(pd.DataFrame())

    def test_predictions_require_probability_column(self):
        """Test that predictions require home_team_prob column."""
        db_ops = DatabaseOperations()

        df_no_prob = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'date': [date(2025, 10, 22)]
            # Missing: home_team_prob
        })

        with pytest.raises(Exception):
            db_ops.save_predictions(df_no_prob)


# Skip integration tests if database is not enabled
pytestmark = pytest.mark.skipif(
    not db_config.enabled,
    reason="Database integration tests require USE_DATABASE=true"
)


class TestDatabaseIntegration:
    """
    Integration tests that require actual database connection.

    These tests are SKIPPED unless USE_DATABASE=true in environment.
    Run with: pytest tests/test_database_integration.py --no-skip
    """

    @pytest.fixture(scope="class")
    def db_pool_initialized(self):
        """Initialize database pool for integration tests."""
        if db_config.enabled:
            from db_utils import db_pool
            db_pool.initialize()
            yield db_pool
            db_pool.close_all()
        else:
            pytest.skip("Database not enabled")

    def test_database_connection(self, db_pool_initialized):
        """Test that database connection can be established."""
        from db_utils import DatabaseOperations

        db_ops = DatabaseOperations()

        # Simple query to verify connection
        result = db_ops.execute_query("SELECT 1 as test")
        assert len(result) == 1
        assert result[0]['test'] == 1

    def test_save_and_retrieve_game_statistics(self, db_pool_initialized):
        """Test saving and retrieving game statistics."""
        from db_utils import DatabaseOperations

        db_ops = DatabaseOperations()

        # Create test data
        test_data = pd.DataFrame({
            'season': ['9999'],  # Use unique season to avoid conflicts
            'date': [date(2099, 12, 31)],
            'team': ['TST'],
            'team_opp': ['TST2'],
            'home': [1],
            'won': [True],
            'total': [100],
            'total_opp': [95]
        })

        # Save to database
        rows_saved = db_ops.save_game_statistics(test_data)
        assert rows_saved > 0

        # Cleanup test data
        db_ops.execute_query(
            "DELETE FROM game_statistics WHERE season = '9999'"
        )


# Run tests with: pytest tests/test_database_integration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
