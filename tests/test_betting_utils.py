#!/usr/bin/env python
"""
Unit tests for betting utility functions.

Tests critical financial calculations including:
- Kelly Criterion stake sizing
- American to Decimal odds conversion
- Odds to implied probability conversion

These functions handle real money, so comprehensive testing is essential.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "2026" / "src"))

from nba_utils_2026 import kelly_frac, am_to_dec, impute_prob


class TestKellyCriterion:
    """
    Test Kelly Criterion calculation for stake sizing.

    Formula: f = ((b * p - (1 - p)) / b) * fraction
    where b = decimal_odds - 1, p = win probability, fraction = Kelly fraction

    Critical for bankroll management - errors here = wrong bet sizes.
    """

    def test_kelly_basic_positive_edge(self):
        """Test basic Kelly calculation with positive edge."""
        # p=0.6, decimal_odds=2.0 (American +100), full Kelly (f=1.0)
        # Expected: (1 * 0.6 - 0.4) / 1 = 0.2 (20% of bankroll)
        stake = kelly_frac(p=0.6, o=2.0, f=1.0)
        assert stake == pytest.approx(0.2, rel=1e-9)

    def test_kelly_half_fraction(self):
        """Test with half-Kelly (conservative approach)."""
        # Same as above but f=0.5
        # Expected: 0.2 * 0.5 = 0.1 (10% of bankroll)
        stake = kelly_frac(p=0.6, o=2.0, f=0.5)
        assert stake == pytest.approx(0.1, rel=1e-9)

    def test_kelly_no_edge(self):
        """Test when probability equals implied odds (no edge)."""
        # p=0.5, decimal_odds=2.0 (50% prob) -> no edge, stake should be 0
        stake = kelly_frac(p=0.5, o=2.0, f=1.0)
        assert stake == pytest.approx(0.0, rel=1e-9)

    def test_kelly_negative_edge(self):
        """Test with negative edge (should return 0, never negative)."""
        # p=0.4, decimal_odds=2.0 -> negative edge
        # Kelly would be negative, but function should return 0
        stake = kelly_frac(p=0.4, o=2.0, f=1.0)
        assert stake == 0.0

    def test_kelly_underdog_bet(self):
        """Test with underdog odds (higher decimal odds)."""
        # p=0.4, decimal_odds=3.0 (American +200), full Kelly
        # b = 2.0, kelly = (2 * 0.4 - 0.6) / 2 = 0.2 / 2 = 0.1
        stake = kelly_frac(p=0.4, o=3.0, f=1.0)
        assert stake == pytest.approx(0.1, rel=1e-9)

    def test_kelly_favorite_bet(self):
        """Test with favorite odds (lower decimal odds)."""
        # p=0.7, decimal_odds=1.5 (American -200)
        # b = 0.5, kelly = (0.5 * 0.7 - 0.3) / 0.5 = 0.05 / 0.5 = 0.1
        stake = kelly_frac(p=0.7, o=1.5, f=1.0)
        assert stake == pytest.approx(0.1, rel=1e-9)

    def test_kelly_edge_case_probability_zero(self):
        """Test edge case: probability = 0."""
        stake = kelly_frac(p=0.0, o=2.0, f=1.0)
        assert stake == 0.0

    def test_kelly_edge_case_probability_one(self):
        """Test edge case: probability = 1 (sure thing)."""
        # p=1.0, decimal_odds=2.0, b=1.0
        # kelly = (1 * 1 - 0) / 1 = 1.0 (100% of bankroll)
        stake = kelly_frac(p=1.0, o=2.0, f=1.0)
        assert stake == pytest.approx(1.0, rel=1e-9)

    def test_kelly_edge_case_odds_too_low(self):
        """Test edge case: decimal odds <= 1.0 (impossible in real betting)."""
        # b = 1.0 - 1.0 = 0.0 -> should return 0
        stake = kelly_frac(p=0.6, o=1.0, f=1.0)
        assert stake == 0.0

        stake = kelly_frac(p=0.6, o=0.5, f=1.0)
        assert stake == 0.0

    def test_kelly_edge_case_none_probability(self):
        """Test error handling: None probability."""
        stake = kelly_frac(p=None, o=2.0, f=1.0)
        assert stake == 0.0

    def test_kelly_edge_case_nan_probability(self):
        """Test error handling: NaN probability."""
        stake = kelly_frac(p=np.nan, o=2.0, f=1.0)
        assert stake == 0.0

    def test_kelly_realistic_scenario_1(self):
        """Real-world scenario: Good home team, half Kelly."""
        # Team has 60% win probability, sportsbook offers +120 (2.20 decimal)
        # Expected with f=0.5: ~0.133
        stake = kelly_frac(p=0.6, o=2.20, f=0.5)
        assert 0.13 < stake < 0.14

    def test_kelly_realistic_scenario_2(self):
        """Real-world scenario: Strong favorite, quarter Kelly."""
        # Team has 75% win probability, sportsbook offers 1.50 decimal (-200)
        # Using conservative f=0.25
        stake = kelly_frac(p=0.75, o=1.50, f=0.25)
        assert stake > 0.0
        assert stake < 0.3  # Should be reasonable


class TestAmericanToDecimalOdds:
    """
    Test American to Decimal odds conversion.

    American odds:
    - Positive (+150): How much you win on $100 bet
    - Negative (-200): How much you need to bet to win $100

    Decimal odds: Total return per $1 bet (includes stake)
    """

    def test_am_to_dec_even_money_positive(self):
        """Test +100 (even money) -> 2.0."""
        assert am_to_dec(100) == pytest.approx(2.0, rel=1e-9)

    def test_am_to_dec_even_money_negative(self):
        """Test -100 (even money) -> 2.0."""
        assert am_to_dec(-100) == pytest.approx(2.0, rel=1e-9)

    def test_am_to_dec_underdog_plus_150(self):
        """Test +150 -> 2.5."""
        # Win $150 on $100 bet = $250 total return = 2.5x
        assert am_to_dec(150) == pytest.approx(2.5, rel=1e-9)

    def test_am_to_dec_underdog_plus_200(self):
        """Test +200 -> 3.0."""
        assert am_to_dec(200) == pytest.approx(3.0, rel=1e-9)

    def test_am_to_dec_underdog_plus_300(self):
        """Test +300 -> 4.0."""
        assert am_to_dec(300) == pytest.approx(4.0, rel=1e-9)

    def test_am_to_dec_favorite_minus_150(self):
        """Test -150 -> 1.67 (approximately)."""
        # Bet $150 to win $100, total return $250 on $150 = 1.6667
        result = am_to_dec(-150)
        assert result == pytest.approx(1.6667, rel=1e-3)

    def test_am_to_dec_favorite_minus_200(self):
        """Test -200 -> 1.5."""
        # Bet $200 to win $100, total return $300 on $200 = 1.5
        assert am_to_dec(-200) == pytest.approx(1.5, rel=1e-9)

    def test_am_to_dec_favorite_minus_300(self):
        """Test -300 -> 1.33 (approximately)."""
        result = am_to_dec(-300)
        assert result == pytest.approx(1.3333, rel=1e-3)

    def test_am_to_dec_heavy_favorite_minus_500(self):
        """Test -500 -> 1.2."""
        assert am_to_dec(-500) == pytest.approx(1.2, rel=1e-9)

    def test_am_to_dec_long_shot_plus_1000(self):
        """Test +1000 -> 11.0."""
        assert am_to_dec(1000) == pytest.approx(11.0, rel=1e-9)

    def test_am_to_dec_none_input(self):
        """Test None input -> None."""
        assert am_to_dec(None) is None

    def test_am_to_dec_nan_string(self):
        """Test 'nan' string -> None."""
        assert am_to_dec("nan") is None
        assert am_to_dec("NaN") is None

    def test_am_to_dec_empty_string(self):
        """Test empty string -> None."""
        assert am_to_dec("") is None
        assert am_to_dec("  ") is None

    def test_am_to_dec_string_with_comma(self):
        """Test European format '150' -> 2.5."""
        # String "150" should be parsed as +150
        assert am_to_dec("150") == pytest.approx(2.5, rel=1e-9)

    def test_am_to_dec_float_input(self):
        """Test float input (rounded to int)."""
        # 150.7 rounds to 151 -> decimal = 2.51
        assert am_to_dec(150.7) == pytest.approx(2.51, rel=1e-3)
        # -200.3 rounds to -200 -> decimal = 1.5
        assert am_to_dec(-200.3) == pytest.approx(1.5, rel=1e-9)

    def test_am_to_dec_invalid_string(self):
        """Test invalid string -> None."""
        assert am_to_dec("abc") is None
        assert am_to_dec("--200") is None


class TestImpliedProbability:
    """
    Test American odds to implied probability conversion.

    Implied probability = how often you need to win to break even.
    - Favorite (-200): Need to win 2/3 times (66.67%)
    - Underdog (+200): Need to win 1/3 times (33.33%)
    """

    def test_impute_prob_even_money_positive(self):
        """Test +100 -> 50% probability."""
        prob = impute_prob(100)
        assert prob == pytest.approx(0.5, rel=1e-9)

    def test_impute_prob_even_money_negative(self):
        """Test -100 -> 50% probability."""
        prob = impute_prob(-100)
        assert prob == pytest.approx(0.5, rel=1e-9)

    def test_impute_prob_underdog_plus_150(self):
        """Test +150 -> 40% probability."""
        # 100 / (150 + 100) = 100 / 250 = 0.4
        prob = impute_prob(150)
        assert prob == pytest.approx(0.4, rel=1e-9)

    def test_impute_prob_underdog_plus_200(self):
        """Test +200 -> 33.33% probability."""
        # 100 / (200 + 100) = 100 / 300 = 0.3333
        prob = impute_prob(200)
        assert prob == pytest.approx(0.3333, rel=1e-3)

    def test_impute_prob_underdog_plus_300(self):
        """Test +300 -> 25% probability."""
        prob = impute_prob(300)
        assert prob == pytest.approx(0.25, rel=1e-9)

    def test_impute_prob_favorite_minus_150(self):
        """Test -150 -> 60% probability."""
        # 150 / (150 + 100) = 150 / 250 = 0.6
        prob = impute_prob(-150)
        assert prob == pytest.approx(0.6, rel=1e-9)

    def test_impute_prob_favorite_minus_200(self):
        """Test -200 -> 66.67% probability."""
        # 200 / (200 + 100) = 200 / 300 = 0.6667
        prob = impute_prob(-200)
        assert prob == pytest.approx(0.6667, rel=1e-3)

    def test_impute_prob_favorite_minus_300(self):
        """Test -300 -> 75% probability."""
        # 300 / (300 + 100) = 300 / 400 = 0.75
        prob = impute_prob(-300)
        assert prob == pytest.approx(0.75, rel=1e-9)

    def test_impute_prob_heavy_favorite_minus_500(self):
        """Test -500 -> 83.33% probability."""
        prob = impute_prob(-500)
        assert prob == pytest.approx(0.8333, rel=1e-3)

    def test_impute_prob_long_shot_plus_1000(self):
        """Test +1000 -> ~9.09% probability."""
        # 100 / (1000 + 100) = 100 / 1100 = 0.0909
        prob = impute_prob(1000)
        assert prob == pytest.approx(0.0909, rel=1e-3)

    def test_impute_prob_none_input(self):
        """Test None input -> None."""
        assert impute_prob(None) is None

    def test_impute_prob_nan_string(self):
        """Test 'nan' string -> None."""
        assert impute_prob("nan") is None
        assert impute_prob("NaN") is None

    def test_impute_prob_empty_string(self):
        """Test empty string -> None."""
        assert impute_prob("") is None

    def test_impute_prob_float_input(self):
        """Test float input (rounded to int)."""
        # 150.7 rounds to 151 -> probability ~= 0.398
        prob = impute_prob(150.7)
        assert prob == pytest.approx(0.398, rel=1e-2)

    def test_impute_prob_string_with_comma(self):
        """Test European decimal format '1,50' (treated as string)."""
        # This should try to convert "1.50" after comma replacement
        # But since impute_prob expects American odds (integers), this is edge case
        # Current implementation would round 1.5 -> 1 or 2
        result = impute_prob("1,50")
        # After conversion: "1.50" -> 1.5 -> round to 2 -> +2 odds
        assert result is not None  # Should handle gracefully

    def test_impute_prob_invalid_string(self):
        """Test invalid string -> None."""
        assert impute_prob("abc") is None

    def test_impute_prob_consistency_with_am_to_dec(self):
        """Test that impute_prob and am_to_dec are consistent."""
        # For any American odds, implied_prob should equal 1 / decimal_odds
        for american_odds in [100, -100, 150, -150, 200, -200]:
            decimal = am_to_dec(american_odds)
            implied = impute_prob(american_odds)

            if decimal is not None and implied is not None:
                expected_prob = 1.0 / decimal
                assert implied == pytest.approx(expected_prob, rel=1e-2)


class TestBettingUtilsIntegration:
    """Integration tests combining multiple betting functions."""

    def test_kelly_with_converted_odds(self):
        """Test Kelly calculation using am_to_dec conversion."""
        # Model says 60% win probability
        # Sportsbook offers +120 (American)
        p = 0.6
        american_odds = 120
        decimal_odds = am_to_dec(american_odds)

        # Calculate stake with half Kelly
        stake = kelly_frac(p=p, o=decimal_odds, f=0.5)

        # Should recommend positive stake (there's value)
        assert stake > 0.0
        assert stake < 0.3  # But reasonable (< 30% of bankroll)

    def test_no_value_bet_scenario(self):
        """Test scenario where there's no betting value."""
        # Model says 50% probability
        # Sportsbook offers -110 (implied 52.4% probability)
        p = 0.5
        american_odds = -110
        decimal_odds = am_to_dec(american_odds)
        implied_prob = impute_prob(american_odds)

        # Our probability (50%) is lower than implied (52.4%)
        assert p < implied_prob

        # Kelly should recommend 0 stake
        stake = kelly_frac(p=p, o=decimal_odds, f=0.5)
        assert stake == 0.0

    def test_value_bet_scenario(self):
        """Test scenario where there IS betting value."""
        # Model says 60% probability
        # Sportsbook offers +100 (implied 50% probability)
        p = 0.6
        american_odds = 100
        decimal_odds = am_to_dec(american_odds)
        implied_prob = impute_prob(american_odds)

        # Our probability (60%) is higher than implied (50%)
        assert p > implied_prob

        # Kelly should recommend positive stake
        stake = kelly_frac(p=p, o=decimal_odds, f=0.5)
        assert stake > 0.0


# Parameterized tests for comprehensive coverage
class TestOddsConversionParameterized:
    """Parameterized tests for comprehensive odds conversion coverage."""

    @pytest.mark.parametrize("american,expected_decimal", [
        (100, 2.0),
        (-100, 2.0),
        (150, 2.5),
        (200, 3.0),
        (-150, 1.6667),
        (-200, 1.5),
        (300, 4.0),
        (-300, 1.3333),
        (500, 6.0),
        (-500, 1.2),
    ])
    def test_american_to_decimal_various(self, american, expected_decimal):
        """Test various American to Decimal conversions."""
        result = am_to_dec(american)
        assert result == pytest.approx(expected_decimal, rel=1e-3)

    @pytest.mark.parametrize("american,expected_prob", [
        (100, 0.5),
        (-100, 0.5),
        (200, 0.3333),
        (-200, 0.6667),
        (300, 0.25),
        (-300, 0.75),
        (400, 0.2),
        (-400, 0.8),
    ])
    def test_american_to_probability_various(self, american, expected_prob):
        """Test various American to Probability conversions."""
        result = impute_prob(american)
        assert result == pytest.approx(expected_prob, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
