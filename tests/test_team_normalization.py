#!/usr/bin/env python
"""
Unit tests for team code normalization.

Tests team abbreviation mapping to handle inconsistencies across:
- Basketball-Reference.com
- Sportsbooks (DraftKings, FanDuel, etc.)
- The Odds API
- NBA.com

Critical for data integrity - mismatched team codes = wrong predictions.
"""

import pandas as pd
import pytest

from src.utils.nba_utils import normalize_team_code, normalize_team_codes_inplace


class TestNormalizeTeamCode:
    """Test single team code normalization."""

    # Phoenix Suns: PHO -> PHX
    def test_normalize_pho_to_phx(self):
        """Test PHO normalizes to PHX (Phoenix Suns)."""
        assert normalize_team_code("PHO") == "PHX"

    def test_normalize_phx_stays_phx(self):
        """Test PHX stays as PHX."""
        assert normalize_team_code("PHX") == "PHX"

    # Brooklyn Nets: BKN -> BRK
    def test_normalize_bkn_to_brk(self):
        """Test BKN normalizes to BRK (Brooklyn Nets)."""
        assert normalize_team_code("BKN") == "BRK"

    def test_normalize_brk_stays_brk(self):
        """Test BRK stays as BRK."""
        assert normalize_team_code("BRK") == "BRK"

    # Charlotte Hornets: CHA -> CHO
    def test_normalize_cha_to_cho(self):
        """Test CHA normalizes to CHO (Charlotte Hornets)."""
        assert normalize_team_code("CHA") == "CHO"

    def test_normalize_cho_stays_cho(self):
        """Test CHO stays as CHO."""
        assert normalize_team_code("CHO") == "CHO"

    # Washington Wizards: WSH -> WAS
    def test_normalize_wsh_to_was(self):
        """Test WSH normalizes to WAS (Washington Wizards)."""
        assert normalize_team_code("WSH") == "WAS"

    def test_normalize_was_stays_was(self):
        """Test WAS stays as WAS."""
        assert normalize_team_code("WAS") == "WAS"

    # Golden State Warriors: GS -> GSW
    def test_normalize_gs_to_gsw(self):
        """Test GS normalizes to GSW (Golden State Warriors)."""
        assert normalize_team_code("GS") == "GSW"

    def test_normalize_gsw_stays_gsw(self):
        """Test GSW stays as GSW."""
        assert normalize_team_code("GSW") == "GSW"

    # New Orleans Pelicans: NO -> NOP
    def test_normalize_no_to_nop(self):
        """Test NO normalizes to NOP (New Orleans Pelicans)."""
        assert normalize_team_code("NO") == "NOP"

    def test_normalize_nop_stays_nop(self):
        """Test NOP stays as NOP."""
        assert normalize_team_code("NOP") == "NOP"

    # New York Knicks: NY -> NYK
    def test_normalize_ny_to_nyk(self):
        """Test NY normalizes to NYK (New York Knicks)."""
        assert normalize_team_code("NY") == "NYK"

    def test_normalize_nyk_stays_nyk(self):
        """Test NYK stays as NYK."""
        assert normalize_team_code("NYK") == "NYK"

    # San Antonio Spurs: SA -> SAS
    def test_normalize_sa_to_sas(self):
        """Test SA normalizes to SAS (San Antonio Spurs)."""
        assert normalize_team_code("SA") == "SAS"

    def test_normalize_sas_stays_sas(self):
        """Test SAS stays as SAS."""
        assert normalize_team_code("SAS") == "SAS"

    # Utah Jazz: UTAH -> UTA
    def test_normalize_utah_to_uta(self):
        """Test UTAH normalizes to UTA (Utah Jazz)."""
        assert normalize_team_code("UTAH") == "UTA"

    def test_normalize_uta_stays_uta(self):
        """Test UTA stays as UTA."""
        assert normalize_team_code("UTA") == "UTA"

    # Oklahoma City Thunder: OKL -> OKC
    def test_normalize_okl_to_okc(self):
        """Test OKL normalizes to OKC (Oklahoma City Thunder)."""
        assert normalize_team_code("OKL") == "OKC"

    def test_normalize_okc_stays_okc(self):
        """Test OKC stays as OKC."""
        assert normalize_team_code("OKC") == "OKC"


class TestNormalizeTeamCodeEdgeCases:
    """Test edge cases and error handling for team normalization."""

    def test_normalize_lowercase_input(self):
        """Test lowercase input is converted to uppercase."""
        assert normalize_team_code("pho") == "PHX"
        assert normalize_team_code("bkn") == "BRK"
        assert normalize_team_code("cha") == "CHO"

    def test_normalize_mixed_case_input(self):
        """Test mixed case input."""
        assert normalize_team_code("PhO") == "PHX"
        assert normalize_team_code("BkN") == "BRK"

    def test_normalize_with_whitespace(self):
        """Test input with leading/trailing whitespace."""
        assert normalize_team_code(" PHO ") == "PHX"
        assert normalize_team_code("  BKN  ") == "BRK"

    def test_normalize_none_input(self):
        """Test None input returns None."""
        assert normalize_team_code(None) is None

    def test_normalize_empty_string(self):
        """Test empty string returns empty string."""
        assert normalize_team_code("") == ""

    def test_normalize_whitespace_only(self):
        """Test whitespace-only string."""
        result = normalize_team_code("   ")
        assert result == "   " or result == ""

    def test_normalize_unknown_code_unchanged(self):
        """Test unknown team code returns as-is (uppercase)."""
        assert normalize_team_code("XYZ") == "XYZ"
        assert normalize_team_code("FOO") == "FOO"

    def test_normalize_standard_codes_unchanged(self):
        """Test standard codes that don't need normalization."""
        standard_codes = ["LAL", "BOS", "MIA", "CHI", "DEN", "MIL", "DAL", "TOR"]
        for code in standard_codes:
            assert normalize_team_code(code) == code

    def test_normalize_non_string_input(self):
        """Test non-string input (e.g., int, float)."""
        # Should return the input unchanged if not a string
        assert normalize_team_code(123) == 123
        assert normalize_team_code(45.6) == 45.6


class TestNormalizeTeamCodesInplace:
    """Test DataFrame column normalization."""

    def test_normalize_single_column(self):
        """Test normalizing a single column in DataFrame."""
        df = pd.DataFrame({"team": ["PHO", "BKN", "CHA", "LAL"], "points": [110, 105, 98, 112]})

        result = normalize_team_codes_inplace(df, cols=["team"])

        assert list(result["team"]) == ["PHX", "BRK", "CHO", "LAL"]
        assert list(result["points"]) == [110, 105, 98, 112]

    def test_normalize_multiple_columns(self):
        """Test normalizing multiple columns in DataFrame."""
        df = pd.DataFrame(
            {"home_team": ["PHO", "BKN"], "away_team": ["CHA", "GS"], "score": [110, 105]}
        )

        result = normalize_team_codes_inplace(df, cols=["home_team", "away_team"])

        assert list(result["home_team"]) == ["PHX", "BRK"]
        assert list(result["away_team"]) == ["CHO", "GSW"]

    def test_normalize_mixed_codes(self):
        """Test DataFrame with mix of codes needing and not needing normalization."""
        df = pd.DataFrame({"team": ["PHO", "LAL", "BKN", "BOS", "CHA", "MIA"]})

        result = normalize_team_codes_inplace(df, cols=["team"])

        assert list(result["team"]) == ["PHX", "LAL", "BRK", "BOS", "CHO", "MIA"]

    def test_normalize_with_none_values(self):
        """Test DataFrame with None values."""
        df = pd.DataFrame({"team": ["PHO", None, "BKN", "LAL"]})

        result = normalize_team_codes_inplace(df, cols=["team"])

        assert result["team"].iloc[0] == "PHX"
        assert pd.isna(result["team"].iloc[1]) or result["team"].iloc[1] is None
        assert result["team"].iloc[2] == "BRK"
        assert result["team"].iloc[3] == "LAL"

    def test_normalize_returns_dataframe(self):
        """Test that function returns the DataFrame (for chaining)."""
        df = pd.DataFrame({"team": ["PHO", "BKN"]})

        result = normalize_team_codes_inplace(df, cols=["team"])

        assert isinstance(result, pd.DataFrame)
        assert result is df  # Should be same object (in-place modification)

    def test_normalize_nonexistent_column(self):
        """Test behavior with column that doesn't exist."""
        df = pd.DataFrame({"team": ["PHO", "BKN"], "points": [110, 105]})

        # Should not raise error, just skip the missing column
        result = normalize_team_codes_inplace(df, cols=["team", "nonexistent_col"])

        assert list(result["team"]) == ["PHX", "BRK"]
        assert "nonexistent_col" not in result.columns

    def test_normalize_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"team": []})

        result = normalize_team_codes_inplace(df, cols=["team"])

        assert len(result) == 0
        assert "team" in result.columns


class TestNormalizationConsistency:
    """Test consistency of normalization across different contexts."""

    def test_all_phoenix_codes_map_to_phx(self):
        """Test all Phoenix variants map to PHX."""
        phoenix_variants = ["PHO", "PHX", "pho", "phx", " PHO ", "PhO"]
        expected = "PHX"

        for variant in phoenix_variants:
            result = normalize_team_code(variant)
            assert result == expected, f"Failed for variant: {variant}"

    def test_all_brooklyn_codes_map_to_brk(self):
        """Test all Brooklyn variants map to BRK."""
        brooklyn_variants = ["BKN", "BRK", "bkn", "brk", " BKN "]
        expected = "BRK"

        for variant in brooklyn_variants:
            result = normalize_team_code(variant)
            assert result == expected, f"Failed for variant: {variant}"

    def test_all_charlotte_codes_map_to_cho(self):
        """Test all Charlotte variants map to CHO."""
        charlotte_variants = ["CHA", "CHO", "cha", "cho", " CHA "]
        expected = "CHO"

        for variant in charlotte_variants:
            result = normalize_team_code(variant)
            assert result == expected, f"Failed for variant: {variant}"

    def test_idempotency(self):
        """Test that normalizing twice gives same result."""
        codes = ["PHO", "BKN", "CHA", "LAL", "BOS"]

        for code in codes:
            first = normalize_team_code(code)
            second = normalize_team_code(first)
            assert first == second, f"Not idempotent for: {code}"


class TestRealWorldScenarios:
    """Test real-world scenarios from actual data sources."""

    def test_sportsbook_api_response(self):
        """Test normalizing team codes from sportsbook API."""
        # Simulate API response with various team code formats
        api_teams = ["PHO", "GS", "NO", "NY", "SA", "BKN"]
        expected = ["PHX", "GSW", "NOP", "NYK", "SAS", "BRK"]

        results = [normalize_team_code(team) for team in api_teams]
        assert results == expected

    def test_basketball_reference_codes(self):
        """Test codes commonly from Basketball-Reference.com."""
        br_codes = ["PHX", "BRK", "CHO", "GSW", "NOP"]
        # These should stay as-is (already in canonical form)
        results = [normalize_team_code(code) for code in br_codes]
        assert results == br_codes

    def test_matchup_dataframe(self):
        """Test realistic matchup DataFrame (home vs away)."""
        df = pd.DataFrame(
            {
                "date": ["2025-10-23", "2025-10-23", "2025-10-23"],
                "home_team": ["PHO", "BKN", "GS"],
                "away_team": ["LAL", "CHA", "NO"],
                "home_odds": [-110, 150, -200],
                "away_odds": [-110, -170, 180],
            }
        )

        result = normalize_team_codes_inplace(df, cols=["home_team", "away_team"])

        assert list(result["home_team"]) == ["PHX", "BRK", "GSW"]
        assert list(result["away_team"]) == ["LAL", "CHO", "NOP"]
        # Other columns should be unchanged
        assert list(result["home_odds"]) == [-110, 150, -200]


# Parameterized tests for all mappings
class TestAllTeamMappings:
    """Comprehensive parameterized tests for all team code mappings."""

    @pytest.mark.parametrize(
        "input_code,expected_output",
        [
            ("PHO", "PHX"),
            ("PHX", "PHX"),
            ("BKN", "BRK"),
            ("BRK", "BRK"),
            ("CHA", "CHO"),
            ("CHO", "CHO"),
            ("WSH", "WAS"),
            ("WAS", "WAS"),
            ("GS", "GSW"),
            ("GSW", "GSW"),
            ("NO", "NOP"),
            ("NOP", "NOP"),
            ("NY", "NYK"),
            ("NYK", "NYK"),
            ("SA", "SAS"),
            ("SAS", "SAS"),
            ("UTAH", "UTA"),
            ("UTA", "UTA"),
            ("OKL", "OKC"),
            ("OKC", "OKC"),
        ],
    )
    def test_all_mappings(self, input_code, expected_output):
        """Test all team code mappings comprehensively."""
        assert normalize_team_code(input_code) == expected_output

    @pytest.mark.parametrize(
        "standard_code",
        [
            "ATL",
            "BOS",
            "CHI",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "HOU",
            "IND",
            "LAC",
            "LAL",
            "MEM",
            "MIA",
            "MIL",
            "MIN",
            "ORL",
            "PHI",
            "POR",
            "SAC",
            "TOR",
        ],
    )
    def test_standard_codes_unchanged(self, standard_code):
        """Test that standard NBA team codes remain unchanged."""
        assert normalize_team_code(standard_code) == standard_code


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
