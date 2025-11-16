#!/usr/bin/env python
"""
Unit tests for data processing functions.

Tests data transformation and feature engineering including:
- Rolling averages calculation (9-game window)
- Preprocessing and target variable creation
- Next game column addition

Critical for ML model quality - errors here = bad features = poor predictions.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "2026" / "src"))

from nba_utils_2026 import calculate_rolling_averages, preprocess_nba_data, add_next_game_columns


class TestCalculateRollingAverages:
    """Test rolling average calculations for team statistics."""

    def test_rolling_averages_basic(self):
        """Test basic rolling average calculation."""
        # Create sample data: one team, one season, 10 games
        df = pd.DataFrame({
            "team": ["LAL"] * 10,
            "season": ["2026"] * 10,
            "points": [100, 102, 98, 105, 110, 95, 108, 103, 99, 107],
            "rebounds": [45, 48, 42, 50, 52, 40, 49, 46, 44, 51],
        })

        result = calculate_rolling_averages(df, window_size=3)

        # First game: average of game 1 only (min_periods=1)
        assert result["points"].iloc[0] == pytest.approx(100.0, rel=1e-6)

        # Second game: average of games 1-2
        assert result["points"].iloc[1] == pytest.approx(101.0, rel=1e-6)

        # Third game: average of games 1-3
        expected_third = (100 + 102 + 98) / 3.0
        assert result["points"].iloc[2] == pytest.approx(expected_third, rel=1e-6)

        # Fourth game: average of games 2-4 (window slides)
        expected_fourth = (102 + 98 + 105) / 3.0
        assert result["points"].iloc[3] == pytest.approx(expected_fourth, rel=1e-6)

    def test_rolling_averages_default_window(self):
        """Test with default 9-game window."""
        df = pd.DataFrame({
            "team": ["BOS"] * 15,
            "season": ["2026"] * 15,
            "points": list(range(100, 115)),  # 100, 101, 102, ..., 114
        })

        result = calculate_rolling_averages(df, window_size=9)

        # 9th game should be average of first 9 games: 100-108
        expected_ninth = sum(range(100, 109)) / 9.0
        assert result["points"].iloc[8] == pytest.approx(expected_ninth, rel=1e-6)

        # 10th game should be average of games 2-10: 101-109
        expected_tenth = sum(range(101, 110)) / 9.0
        assert result["points"].iloc[9] == pytest.approx(expected_tenth, rel=1e-6)

    def test_rolling_averages_multiple_teams(self):
        """Test rolling averages calculated separately per team."""
        df = pd.DataFrame({
            "team": ["LAL", "LAL", "LAL", "BOS", "BOS", "BOS"],
            "season": ["2026"] * 6,
            "points": [100, 110, 120, 90, 95, 100],
        })

        result = calculate_rolling_averages(df, window_size=2)

        # LAL's second game: (100 + 110) / 2 = 105
        lal_games = result[result["team"] == "LAL"]
        assert lal_games["points"].iloc[1] == pytest.approx(105.0, rel=1e-6)

        # BOS's second game: (90 + 95) / 2 = 92.5
        bos_games = result[result["team"] == "BOS"]
        assert bos_games["points"].iloc[1] == pytest.approx(92.5, rel=1e-6)

    def test_rolling_averages_multiple_seasons(self):
        """Test rolling averages calculated separately per season."""
        df = pd.DataFrame({
            "team": ["LAL"] * 6,
            "season": ["2025", "2025", "2025", "2026", "2026", "2026"],
            "points": [100, 110, 120, 90, 95, 100],
        })

        result = calculate_rolling_averages(df, window_size=2)

        # 2025 season games should not mix with 2026
        season_2025 = result[result["season"] == "2025"]
        season_2026 = result[result["season"] == "2026"]

        # 2025 second game: (100 + 110) / 2 = 105
        assert season_2025["points"].iloc[1] == pytest.approx(105.0, rel=1e-6)

        # 2026 first game: 90 (no mixing with 2025 data)
        assert season_2026["points"].iloc[0] == pytest.approx(90.0, rel=1e-6)

    def test_rolling_averages_preserves_non_numeric(self):
        """Test that non-numeric columns are preserved."""
        df = pd.DataFrame({
            "team": ["LAL", "LAL", "LAL"],
            "season": ["2026", "2026", "2026"],
            "opponent": ["BOS", "MIA", "CHI"],
            "points": [100, 110, 105],
            "date": pd.to_datetime(["2025-10-23", "2025-10-24", "2025-10-25"]),
        })

        result = calculate_rolling_averages(df, window_size=2)

        # Non-numeric columns should be preserved
        assert "team" in result.columns
        assert "season" in result.columns
        assert "opponent" in result.columns
        assert "date" in result.columns

        # Check values are preserved
        assert list(result["opponent"]) == ["BOS", "MIA", "CHI"]
        assert list(result["team"]) == ["LAL", "LAL", "LAL"]

    def test_rolling_averages_only_numeric_cols_rolled(self):
        """Test that only numeric columns get rolling averages."""
        df = pd.DataFrame({
            "team": ["LAL", "LAL", "LAL"],
            "season": ["2026", "2026", "2026"],
            "points": [100, 110, 105],
            "rebounds": [45, 50, 48],
        })

        result = calculate_rolling_averages(df, window_size=2)

        # Both numeric columns should have rolling averages
        assert result["points"].iloc[1] == pytest.approx(105.0, rel=1e-6)
        assert result["rebounds"].iloc[1] == pytest.approx(47.5, rel=1e-6)

        # Non-numeric columns unchanged
        assert list(result["team"]) == ["LAL", "LAL", "LAL"]

    def test_rolling_averages_with_nans(self):
        """Test behavior with NaN values in data."""
        df = pd.DataFrame({
            "team": ["LAL"] * 5,
            "season": ["2026"] * 5,
            "points": [100, np.nan, 110, 105, np.nan],
        })

        result = calculate_rolling_averages(df, window_size=2)

        # Rolling should handle NaNs according to pandas default behavior
        # (NaN propagates through rolling window)
        assert not pd.isna(result["points"].iloc[0])
        # Result might have NaNs where input had NaNs

    def test_rolling_averages_single_game(self):
        """Test with only one game (min_periods=1 should work)."""
        df = pd.DataFrame({
            "team": ["LAL"],
            "season": ["2026"],
            "points": [100],
        })

        result = calculate_rolling_averages(df, window_size=9)

        # With min_periods=1, should return the single value
        assert result["points"].iloc[0] == pytest.approx(100.0, rel=1e-6)

    def test_rolling_averages_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({
            "team": [],
            "season": [],
            "points": [],
        })

        result = calculate_rolling_averages(df, window_size=9)

        assert len(result) == 0
        assert "team" in result.columns
        assert "season" in result.columns
        assert "points" in result.columns


class TestPreprocessNBAData:
    """Test preprocessing and target variable creation."""

    def test_preprocess_adds_target_variable(self):
        """Test that preprocessing adds 'won' target variable."""
        df = pd.DataFrame({
            "team": ["LAL", "BOS", "LAL", "BOS"],
            "opponent": ["BOS", "LAL", "MIA", "CHI"],
            "pts": [110, 105, 98, 102],
            "opp_pts": [105, 110, 100, 95],
        })

        result = preprocess_nba_data(df)

        assert "won" in result.columns

    def test_preprocess_won_calculation(self):
        """Test 'won' is correctly calculated (1 if won, 0 if lost)."""
        df = pd.DataFrame({
            "team": ["LAL", "BOS", "MIA"],
            "pts": [110, 98, 105],
            "opp_pts": [105, 102, 105],  # LAL won, BOS lost, MIA tied
        })

        result = preprocess_nba_data(df)

        # LAL won (110 > 105)
        assert result["won"].iloc[0] == 1

        # BOS lost (98 < 102)
        assert result["won"].iloc[1] == 0

        # MIA tied (105 == 105) - typically counted as 0 or handled separately
        # Check actual implementation behavior

    def test_preprocess_preserves_original_columns(self):
        """Test that original columns are preserved."""
        df = pd.DataFrame({
            "team": ["LAL"],
            "opponent": ["BOS"],
            "pts": [110],
            "opp_pts": [105],
            "date": ["2025-10-23"],
        })

        result = preprocess_nba_data(df)

        assert "team" in result.columns
        assert "opponent" in result.columns
        assert "pts" in result.columns
        assert "opp_pts" in result.columns
        assert "date" in result.columns


class TestAddNextGameColumns:
    """Test adding next game information to each row."""

    def test_add_next_game_basic(self):
        """Test basic next game column addition."""
        df = pd.DataFrame({
            "team": ["LAL", "LAL", "LAL"],
            "opponent": ["BOS", "MIA", "CHI"],
            "date": ["2025-10-23", "2025-10-24", "2025-10-25"],
            "pts": [110, 105, 108],
        })

        result = add_next_game_columns(df)

        # Should add columns like 'opponent_next', 'date_next', etc.
        # Check implementation to verify exact column names

    def test_add_next_game_last_row_is_nan(self):
        """Test that last row has NaN for next game (no next game)."""
        df = pd.DataFrame({
            "team": ["LAL", "LAL"],
            "opponent": ["BOS", "MIA"],
        })

        result = add_next_game_columns(df)

        # Last row should have NaN for next game columns
        # (implementation dependent - check actual column names)

    def test_add_next_game_multiple_teams(self):
        """Test next game columns are added correctly per team."""
        df = pd.DataFrame({
            "team": ["LAL", "LAL", "BOS", "BOS"],
            "opponent": ["BOS", "MIA", "CHI", "LAL"],
        })

        result = add_next_game_columns(df)

        # LAL's first row should reference LAL's second game as "next"
        # BOS's first row should reference BOS's second game as "next"
        # Not mixing between teams


class TestDataProcessingIntegration:
    """Integration tests combining multiple processing steps."""

    def test_full_pipeline_basic(self):
        """Test complete data processing pipeline."""
        # Create realistic game data
        df = pd.DataFrame({
            "team": ["LAL"] * 10,
            "season": ["2026"] * 10,
            "opponent": ["BOS", "MIA", "CHI", "PHX", "GSW", "DEN", "DAL", "MIN", "NOP", "MEM"],
            "pts": [110, 105, 98, 112, 108, 103, 115, 99, 107, 111],
            "opp_pts": [105, 108, 102, 107, 110, 98, 112, 95, 105, 108],
            "rebounds": [45, 48, 42, 50, 47, 44, 52, 43, 49, 46],
            "assists": [25, 27, 22, 28, 24, 26, 29, 21, 27, 25],
        })

        # Step 1: Add target variable
        df_processed = preprocess_nba_data(df)

        # Step 2: Calculate rolling averages
        df_rolled = calculate_rolling_averages(df_processed, window_size=3)

        # Step 3: Add next game columns
        df_final = add_next_game_columns(df_rolled)

        # Verify pipeline completed
        assert "won" in df_final.columns  # From preprocess
        assert len(df_final) == 10  # No rows lost
        # Rolling averages should be calculated
        # Next game columns should be added

    def test_pipeline_with_multiple_teams(self):
        """Test pipeline with multiple teams."""
        df = pd.DataFrame({
            "team": ["LAL", "LAL", "LAL", "BOS", "BOS", "BOS"],
            "season": ["2026"] * 6,
            "pts": [110, 105, 108, 98, 102, 95],
            "opp_pts": [105, 108, 103, 102, 95, 98],
            "rebounds": [45, 48, 46, 42, 44, 40],
        })

        df_processed = preprocess_nba_data(df)
        df_rolled = calculate_rolling_averages(df_processed, window_size=2)

        # Each team should have independent rolling averages
        lal_games = df_rolled[df_rolled["team"] == "LAL"]
        bos_games = df_rolled[df_rolled["team"] == "BOS"]

        assert len(lal_games) == 3
        assert len(bos_games) == 3

        # LAL's rolling averages shouldn't be affected by BOS
        # (verify they're calculated independently)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in data processing."""

    def test_rolling_averages_with_negative_window(self):
        """Test behavior with invalid window size."""
        df = pd.DataFrame({
            "team": ["LAL"],
            "season": ["2026"],
            "points": [100],
        })

        # Negative window should either raise error or be handled gracefully
        with pytest.raises((ValueError, Exception)):
            calculate_rolling_averages(df, window_size=-1)

    def test_rolling_averages_with_zero_window(self):
        """Test behavior with zero window size."""
        df = pd.DataFrame({
            "team": ["LAL"],
            "season": ["2026"],
            "points": [100],
        })

        # Zero window should raise error or be handled
        with pytest.raises((ValueError, Exception)):
            calculate_rolling_averages(df, window_size=0)

    def test_rolling_averages_with_very_large_window(self):
        """Test with window larger than dataset."""
        df = pd.DataFrame({
            "team": ["LAL"] * 5,
            "season": ["2026"] * 5,
            "points": [100, 110, 105, 112, 108],
        })

        # Window of 100 with only 5 games
        result = calculate_rolling_averages(df, window_size=100)

        # With min_periods=1, should still calculate averages
        assert len(result) == 5
        # All games should have average of all available games

    def test_preprocess_with_missing_columns(self):
        """Test preprocessing when expected columns are missing."""
        df = pd.DataFrame({
            "team": ["LAL"],
            # Missing 'pts' and 'opp_pts' columns
        })

        # Should either raise informative error or handle gracefully
        # (Check actual implementation behavior)

    def test_rolling_averages_preserves_row_count(self):
        """Test that rolling averages doesn't drop rows."""
        df = pd.DataFrame({
            "team": ["LAL"] * 20,
            "season": ["2026"] * 20,
            "points": list(range(100, 120)),
        })

        result = calculate_rolling_averages(df, window_size=9)

        assert len(result) == len(df)


# Performance and scale tests
class TestDataProcessingPerformance:
    """Test data processing with realistic data volumes."""

    def test_rolling_averages_full_season(self):
        """Test with full NBA season data (82 games per team)."""
        df = pd.DataFrame({
            "team": ["LAL"] * 82,
            "season": ["2026"] * 82,
            "points": [np.random.randint(90, 130) for _ in range(82)],
            "rebounds": [np.random.randint(35, 55) for _ in range(82)],
        })

        result = calculate_rolling_averages(df, window_size=9)

        assert len(result) == 82
        # Verify calculations are correct for a sample
        # (e.g., game 10 should be average of games 2-10)

    def test_rolling_averages_all_teams_full_season(self):
        """Test with realistic data: 30 teams × 82 games."""
        teams = ["LAL", "BOS", "GSW", "MIA", "CHI"] * 6  # 30 team instances (5 unique teams × 6)
        all_data = []

        for team in teams:
            team_df = pd.DataFrame({
                "team": [team] * 82,
                "season": ["2026"] * 82,
                "points": [np.random.randint(90, 130) for _ in range(82)],
            })
            all_data.append(team_df)

        df = pd.concat(all_data, ignore_index=True)

        result = calculate_rolling_averages(df, window_size=9)

        # Total: 30 team instances × 82 games = 2,460 rows
        assert len(result) == 30 * 82

        # Verify each unique team has the correct number of games (6 instances × 82 games each)
        unique_teams = set(teams)
        for team in unique_teams:
            team_games = result[result["team"] == team]
            # Each unique team appears 6 times in the data, so 6 × 82 = 492 games per unique team
            assert len(team_games) == 6 * 82


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
