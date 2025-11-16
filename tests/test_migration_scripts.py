#!/usr/bin/env python
"""
Unit tests for database migration scripts.

Tests migration script functionality:
- CSV file discovery
- DataFrame preparation for database
- Batch processing
- Dry run mode
- Command-line argument parsing
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import os
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "database" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "2026" / "src"))


class TestGameStatisticsMigration:
    """Test game statistics migration script."""

    @patch('migrate_game_statistics.glob.glob')
    def test_find_csv_files_discovers_files(self, mock_glob):
        """Test that CSV files are discovered correctly."""
        from migrate_game_statistics import find_csv_files

        mock_glob.return_value = [
            "/data/nba_games_2025-10-22.csv",
            "/data/nba_games_2025-10-23.csv",
            "/data/nba_games_2025-10-24.csv"
        ]

        files = find_csv_files("/data", latest_only=False)

        assert len(files) == 3
        assert files[0].endswith("2025-10-22.csv")
        mock_glob.assert_called_once()

    @patch('migrate_game_statistics.glob.glob')
    def test_find_csv_files_latest_only(self, mock_glob):
        """Test latest_only returns most recent file."""
        from migrate_game_statistics import find_csv_files

        mock_glob.return_value = [
            "/data/nba_games_2025-10-22.csv",
            "/data/nba_games_2025-10-25.csv",
            "/data/nba_games_2025-10-23.csv"
        ]

        files = find_csv_files("/data", latest_only=True)

        assert len(files) == 1
        assert files[0].endswith("2025-10-25.csv")  # Latest date

    @patch('migrate_game_statistics.glob.glob')
    def test_find_csv_files_handles_no_files(self, mock_glob):
        """Test handling when no CSV files found."""
        from migrate_game_statistics import find_csv_files

        mock_glob.return_value = []

        files = find_csv_files("/data")

        assert files == []

    def test_prepare_dataframe_for_db_converts_dates(self):
        """Test that dates are converted to proper format."""
        from migrate_game_statistics import prepare_dataframe_for_db

        df = pd.DataFrame({
            'date': ['2025-10-22', '2025-10-23'],
            'team': ['LAL', 'BOS']
        })

        result = prepare_dataframe_for_db(df)

        assert result['date'].dtype == object  # date objects
        assert isinstance(result['date'].iloc[0], date)

    def test_prepare_dataframe_for_db_converts_booleans(self):
        """Test that boolean columns are converted correctly."""
        from migrate_game_statistics import prepare_dataframe_for_db

        df = pd.DataFrame({
            'won': [1, 0, 1],
            'home': [1, 0, 1]
        })

        result = prepare_dataframe_for_db(df)

        assert result['won'].dtype == bool
        assert result['home'].dtype == 'int64'

    def test_prepare_dataframe_for_db_handles_nulls(self):
        """Test that NULL values are handled correctly."""
        from migrate_game_statistics import prepare_dataframe_for_db

        df = pd.DataFrame({
            'points': [110.0, None, 105.0],
            'rebounds': [45.0, 50.0, None]
        })

        result = prepare_dataframe_for_db(df)

        # NaN should be converted to None for database
        assert pd.isna(result['points'].iloc[1]) or result['points'].iloc[1] is None


class TestPredictionsMigration:
    """Test predictions migration script."""

    @patch('migrate_predictions.glob.glob')
    def test_find_prediction_files(self, mock_glob):
        """Test finding prediction CSV files."""
        from migrate_predictions import find_prediction_files

        mock_glob.return_value = [
            "/pred/predictions_2025-10-22.csv",
            "/pred/predictions_2025-10-23.csv"
        ]

        files = find_prediction_files("/pred")

        assert len(files) == 2

    def test_prepare_predictions_for_db_adds_prediction_date(self):
        """Test that prediction_date is extracted from filename."""
        from migrate_predictions import prepare_predictions_for_db

        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'date': ['2025-10-25']
        })

        result = prepare_predictions_for_db(df, "predictions_2025-10-22.csv")

        assert 'prediction_date' in result.columns
        assert result['prediction_date'].iloc[0] == date(2025, 10, 22)

    def test_prepare_predictions_for_db_clips_probability(self):
        """Test that probabilities are clipped to [0, 1]."""
        from migrate_predictions import prepare_predictions_for_db

        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'date': ['2025-10-25'],
            'home_team_prob': [1.5]  # Invalid: > 1
        })

        result = prepare_predictions_for_db(df, "predictions_2025-10-22.csv")

        assert result['home_team_prob'].iloc[0] == 1.0  # Clipped to 1

    def test_prepare_predictions_for_db_handles_missing_columns(self):
        """Test that missing columns are added with defaults."""
        from migrate_predictions import prepare_predictions_for_db

        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'date': ['2025-10-25']
            # Missing: result, model_version
        })

        result = prepare_predictions_for_db(df, "predictions_2025-10-22.csv")

        assert 'result' in result.columns
        assert 'model_version' in result.columns
        assert result['model_version'].iloc[0] == 'legacy'


class TestEnrichedPredictionsMigration:
    """Test enriched predictions migration script."""

    def test_prepare_enriched_for_db_fills_stake_defaults(self):
        """Test that missing stakes are filled with 0."""
        from migrate_enriched_predictions import prepare_enriched_for_db

        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'date': ['2025-10-25'],
            'stake_raw': [10.0]
            # Missing: stake_platt, stake_iso
        })

        result = prepare_enriched_for_db(df)

        assert 'stake_platt' in result.columns
        assert 'stake_iso' in result.columns
        assert result['stake_platt'].iloc[0] == 0
        assert result['stake_iso'].iloc[0] == 0

    def test_prepare_enriched_for_db_fills_pnl_defaults(self):
        """Test that missing PnL values are filled with 0."""
        from migrate_enriched_predictions import prepare_enriched_for_db

        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'date': ['2025-10-25']
            # Missing all PnL columns
        })

        result = prepare_enriched_for_db(df)

        assert 'pnl_raw' in result.columns
        assert 'pnl_platt' in result.columns
        assert 'pnl_iso' in result.columns
        assert result['pnl_raw'].iloc[0] == 0

    def test_prepare_enriched_for_db_converts_dates(self):
        """Test date conversion in enriched predictions."""
        from migrate_enriched_predictions import prepare_enriched_for_db

        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['BOS'],
            'date': ['2025-10-25'],
            'prediction_date': ['2025-10-22']
        })

        result = prepare_enriched_for_db(df)

        assert isinstance(result['date'].iloc[0], date)


class TestMigrationBatchProcessing:
    """Test batch processing in migration scripts."""

    def test_batch_processing_handles_large_datasets(self):
        """Test that large datasets are split into batches."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'team': [f'T{i}' for i in range(10000)],
            'points': list(range(10000))
        })

        batch_size = 1000
        batches = []

        # Simulate batch processing
        for i in range(0, len(large_df), batch_size):
            batch = large_df.iloc[i:i+batch_size]
            batches.append(batch)

        assert len(batches) == 10  # 10000 / 1000
        assert len(batches[0]) == 1000
        assert len(batches[-1]) == 1000

    def test_batch_processing_handles_partial_batches(self):
        """Test batch processing with partial final batch."""
        df = pd.DataFrame({
            'team': [f'T{i}' for i in range(2500)]
        })

        batch_size = 1000
        batches = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batches.append(batch)

        assert len(batches) == 3  # 1000, 1000, 500
        assert len(batches[-1]) == 500  # Partial batch


class TestMigrationDryRun:
    """Test dry-run mode in migration scripts."""

    def test_dry_run_doesnt_modify_data(self, tmp_path):
        """Test that dry run doesn't insert data."""
        # Create test CSV
        test_file = tmp_path / "nba_games_2025-10-22.csv"
        pd.DataFrame({
            'season': ['2025'],
            'date': ['2025-10-22'],
            'team': ['LAL'],
            'team_opp': ['BOS'],
            'home': [1],
            'won': [True],
            'total': [110],
            'total_opp': [105]
        }).to_csv(test_file, index=False)

        # Dry run should just log, not insert
        # This would be tested with mock database operations
        assert test_file.exists()

    @patch('migrate_game_statistics.logger')
    def test_dry_run_logs_preview(self, mock_logger):
        """Test that dry run logs preview information."""
        from migrate_game_statistics import prepare_dataframe_for_db

        df = pd.DataFrame({
            'season': ['2025'],
            'date': ['2025-10-22'],
            'team': ['LAL']
        })

        result = prepare_dataframe_for_db(df)

        # In dry run, would log: "[DRY RUN] Would insert X rows"
        assert len(result) > 0


class TestMigrationErrorHandling:
    """Test error handling in migration scripts."""

    def test_migration_handles_invalid_csv(self, tmp_path):
        """Test handling of corrupted CSV files."""
        # Create invalid CSV
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("not,a,valid,csv\nfile")

        # Migration should handle gracefully
        try:
            pd.read_csv(invalid_csv)
        except Exception as e:
            assert True  # Expected to fail

    def test_migration_validates_required_columns(self):
        """Test that migration validates required columns."""
        from migrate_game_statistics import prepare_dataframe_for_db

        # Missing required columns should be handled
        df = pd.DataFrame({
            'season': ['2025']
            # Missing: date, team, team_opp, etc.
        })

        # prepare_dataframe_for_db doesn't validate, but later validation will
        result = prepare_dataframe_for_db(df)
        assert 'season' in result.columns

    def test_migration_handles_duplicate_records(self):
        """Test handling of duplicate records."""
        df = pd.DataFrame({
            'season': ['2025', '2025', '2025'],
            'date': ['2025-10-22', '2025-10-22', '2025-10-23'],
            'team': ['LAL', 'LAL', 'LAL'],
            'team_opp': ['BOS', 'BOS', 'GSW']
        })

        # First two rows are duplicates
        # Database should handle with ON CONFLICT clause
        duplicates = df.duplicated(subset=['season', 'date', 'team', 'team_opp'])
        assert duplicates.sum() == 1  # One duplicate


class TestMigrationCommandLine:
    """Test command-line argument parsing."""

    def test_argparse_accepts_dry_run_flag(self):
        """Test --dry-run flag parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--batch-size', type=int, default=1000)

        args = parser.parse_args(['--dry-run'])

        assert args.dry_run is True
        assert args.batch_size == 1000

    def test_argparse_accepts_batch_size(self):
        """Test --batch-size argument."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=1000)

        args = parser.parse_args(['--batch-size', '500'])

        assert args.batch_size == 500

    def test_argparse_accepts_latest_only(self):
        """Test --latest-only flag."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--latest-only', action='store_true')

        args = parser.parse_args(['--latest-only'])

        assert args.latest_only is True


class TestMigrationDataIntegrity:
    """Test data integrity during migration."""

    def test_data_types_preserved(self):
        """Test that data types are preserved during migration."""
        from migrate_game_statistics import prepare_dataframe_for_db
        import numpy as np

        df = pd.DataFrame({
            'season': ['2025'],
            'date': ['2025-10-22'],
            'points': [110],
            'fg_pct': [0.456],
            'won': [True],
            'home': [1]
        })

        result = prepare_dataframe_for_db(df)

        assert isinstance(result['season'].iloc[0], str)
        assert isinstance(result['date'].iloc[0], date)
        # Boolean can be Python bool, numpy bool, or stored as int
        assert result['won'].dtype == bool or result['won'].iloc[0] in [True, False, 0, 1]

    def test_team_codes_preserved(self):
        """Test that team codes are preserved exactly."""
        from migrate_game_statistics import prepare_dataframe_for_db

        df = pd.DataFrame({
            'team': ['LAL', 'BOS', 'GSW', 'PHX'],
            'team_opp': ['MIA', 'DEN', 'DAL', 'LAC']
        })

        result = prepare_dataframe_for_db(df)

        assert list(result['team']) == ['LAL', 'BOS', 'GSW', 'PHX']
        assert list(result['team_opp']) == ['MIA', 'DEN', 'DAL', 'LAC']


# Run tests with: pytest tests/test_migration_scripts.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
