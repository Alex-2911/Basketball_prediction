"""
Example Usage of src.core.betting Module

This script demonstrates how to use the betting module for:
1. Fetching and merging odds
2. Calibrating probabilities
3. Calculating Kelly stakes
4. Simulating bankroll performance
5. Displaying bet recommendations
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.betting import (
    OddsManager,
    ProbabilityCalibrator,
    KellyCriterionCalculator,
    BankrollSimulator,
    BetRecommendationDisplay,
    fetch_and_merge_odds,
    calibrate_and_simulate,
)
from src.core.constants import KELLY_DEFAULTS, STRATEGY_THRESHOLDS


# ============================================================================
# EXAMPLE 1: Fetching and Merging Odds
# ============================================================================

def example_odds_fetching():
    """Example: Fetch odds from API and merge with predictions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Odds Fetching and Merging")
    print("=" * 80 + "\n")

    # Sample game schedule
    games_df = pd.DataFrame({
        'home_team': ['LAL', 'GSW', 'BOS'],
        'away_team': ['MIA', 'PHO', 'CHI'],
        'game_date': ['2025-11-16', '2025-11-16', '2025-11-16']
    })

    # Sample predictions
    predictions_df = pd.DataFrame({
        'home_team': ['LAL', 'GSW', 'BOS'],
        'away_team': ['MIA', 'PHO', 'CHI'],
        'home_team_prob': [0.65, 0.58, 0.72],
        'date': ['2025-11-16', '2025-11-16', '2025-11-16']
    })

    # Initialize OddsManager
    api_key = os.getenv("ODDS_API_KEY")
    if api_key:
        odds_mgr = OddsManager(api_key=api_key, preferred_books=["draftkings", "fanduel"])

        # Fetch odds
        odds_df = odds_mgr.fetch_odds(games_df)
        print("Odds fetched:")
        print(odds_df)

        # Merge with predictions
        merged_df = odds_mgr.merge_with_predictions(predictions_df, odds_df)
        print("\nMerged predictions with odds:")
        print(merged_df[['home_team', 'away_team', 'home_team_prob', 'odds 1', 'imp_prob_home', 'value_home']])
    else:
        print("⚠️ ODDS_API_KEY not found. Skipping API call.")

    # Demonstrate odds conversion
    print("\n--- Odds Conversion Examples ---")
    print(f"American -150 → Decimal: {OddsManager.american_to_decimal(-150)}")
    print(f"American +200 → Decimal: {OddsManager.american_to_decimal(200)}")
    print(f"Decimal 1.67 → American: {OddsManager.decimal_to_american(1.67)}")
    print(f"Decimal 3.00 → American: {OddsManager.decimal_to_american(3.00)}")
    print(f"Implied probability from -150: {OddsManager.implied_probability(-150, 'american'):.4f}")


# ============================================================================
# EXAMPLE 2: Probability Calibration
# ============================================================================

def example_calibration():
    """Example: Calibrate probabilities using Platt scaling and isotonic regression."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Probability Calibration")
    print("=" * 80 + "\n")

    # Generate synthetic historical data
    np.random.seed(42)
    n_samples = 1000

    # Raw probabilities (slightly overconfident)
    raw_probs = np.random.beta(2, 2, n_samples)

    # True outcomes (probabilities closer to 0.5 than model predicts)
    outcomes = (np.random.random(n_samples) < (raw_probs * 0.8 + 0.1)).astype(int)

    historical_df = pd.DataFrame({
        'home_team_prob': raw_probs,
        'accuracy': outcomes
    })

    # Initialize calibrator
    calibrator = ProbabilityCalibrator()

    # Fit Platt scaling
    platt_model = calibrator.fit_platt_scaling(raw_probs, outcomes)
    print("✓ Platt scaling fitted")

    # Fit isotonic regression
    iso_model = calibrator.fit_isotonic_regression(raw_probs, outcomes)
    print("✓ Isotonic regression fitted")

    # Calibrate today's predictions
    today_predictions = pd.DataFrame({
        'home_team': ['LAL', 'GSW', 'BOS'],
        'away_team': ['MIA', 'PHO', 'CHI'],
        'home_team_prob': [0.65, 0.58, 0.72],
    })

    calibrated = calibrator.calibrate_predictions(today_predictions)
    print("\nCalibrated predictions:")
    print(calibrated[['home_team', 'away_team', 'home_team_prob', 'prob_platt', 'prob_iso']])


# ============================================================================
# EXAMPLE 3: Kelly Criterion Calculations
# ============================================================================

def example_kelly_calculations():
    """Example: Calculate Kelly fractions and stakes."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Kelly Criterion Calculations")
    print("=" * 80 + "\n")

    # Initialize calculator with default settings
    kelly_calc = KellyCriterionCalculator(
        bet_fraction=0.5,  # Half Kelly
        cap_fraction=0.30,  # Max 30% of bankroll
        absolute_cap=300.0  # Max $300 per bet
    )

    # Example bet scenarios
    scenarios = [
        {'prob': 0.60, 'odds': 2.00, 'description': 'Strong edge (60% @ 2.00)'},
        {'prob': 0.55, 'odds': 1.90, 'description': 'Moderate edge (55% @ 1.90)'},
        {'prob': 0.45, 'odds': 2.20, 'description': 'No edge (45% @ 2.20)'},
        {'prob': 0.70, 'odds': 1.50, 'description': 'High prob, low odds (70% @ 1.50)'},
    ]

    bankroll = 1000.0

    print(f"Bankroll: ${bankroll:.2f}")
    print(f"Settings: {kelly_calc.bet_fraction:.0%} Kelly, "
          f"Max {kelly_calc.cap_fraction:.0%} of bankroll, "
          f"Absolute cap ${kelly_calc.absolute_cap:.2f}")
    print("\n" + "-" * 80)

    for scenario in scenarios:
        prob = scenario['prob']
        odds = scenario['odds']
        desc = scenario['description']

        kelly_frac = kelly_calc.calculate_kelly_fraction(prob, odds)
        stake = kelly_calc.calculate_stake(kelly_frac, bankroll)

        print(f"\n{desc}")
        print(f"  Kelly fraction: {kelly_frac:.4f} ({kelly_frac*100:.2f}% of bankroll)")
        print(f"  Stake: ${stake:.2f}")

        if stake > 0:
            expected_value = (prob * (odds - 1.0) - (1 - prob)) * stake
            print(f"  Expected value: ${expected_value:.2f}")

    # Apply Kelly to a DataFrame
    print("\n" + "-" * 80)
    print("Applying Kelly to DataFrame:")

    bets_df = pd.DataFrame({
        'home_team': ['LAL', 'GSW', 'BOS', 'MIA'],
        'home_team_prob': [0.60, 0.55, 0.45, 0.70],
        'odds_1': [2.00, 1.90, 2.20, 1.50],
    })

    df_with_kelly = kelly_calc.apply_kelly_to_dataframe(bets_df, bankroll=bankroll)
    print(df_with_kelly[['home_team', 'home_team_prob', 'odds_1', 'kelly_frac', 'stake']])


# ============================================================================
# EXAMPLE 4: Bankroll Simulation
# ============================================================================

def example_bankroll_simulation():
    """Example: Simulate betting over a season."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Bankroll Simulation")
    print("=" * 80 + "\n")

    # Generate synthetic season data
    np.random.seed(42)
    n_games = 100

    dates = pd.date_range('2024-10-01', periods=n_games, freq='D')

    # Create synthetic betting history
    home_teams = np.random.choice(['LAL', 'GSW', 'BOS', 'MIA', 'CHI'], n_games)
    away_teams = np.random.choice(['LAL', 'GSW', 'BOS', 'MIA', 'CHI'], n_games)

    # Ensure home != away
    mask = home_teams == away_teams
    away_teams[mask] = np.roll(away_teams[mask], 1)

    # Generate probabilities and outcomes
    raw_probs = np.random.beta(3, 2, n_games)  # Slightly favoring home
    platt_probs = raw_probs * 0.9 + 0.05  # Calibrated (less confident)
    iso_probs = raw_probs * 0.85 + 0.075  # More conservative

    # Actual outcomes (home team wins)
    home_wins = (np.random.random(n_games) < raw_probs * 0.85).astype(int)

    # Odds (slightly unfavorable on average)
    odds = 1.0 / (raw_probs * 0.9 + 0.05) * np.random.uniform(0.95, 1.05, n_games)

    season_df = pd.DataFrame({
        'date': dates,
        'home_team': home_teams,
        'away_team': away_teams,
        'home_team_prob': raw_probs,
        'prob_platt': platt_probs,
        'prob_iso': iso_probs,
        'odds_1': odds,
        'win': home_wins,
        'accuracy': home_wins,  # For calibration
    })

    # Strategy configuration
    strategy_config = {
        'odds_min': 1.18,
        'odds_max': 3.00,
        'prob_min': 0.50,
        'good_home_teams': set(),  # Bet on all home teams
        'bet_fraction': 0.5,
        'cap_fraction': 0.30,
        'absolute_cap': 300.0,
    }

    # Run simulation
    simulator = BankrollSimulator(starting_bankroll=1000.0)
    results = simulator.simulate_season(season_df, strategy_config)

    print("Simulation complete!")

    # Calculate metrics for each method
    for method in ['raw', 'platt', 'iso']:
        metrics = simulator.calculate_metrics(results, method=method)
        print(f"\n{method.upper()} Method Performance:")
        print(f"  ROI: {metrics['roi']:.2f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"  Number of bets: {metrics['num_bets']}")
        print(f"  Win rate: {metrics['win_rate']:.2f}%")
        print(f"  Average stake: ${metrics['avg_stake']:.2f}")
        print(f"  Max drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

    # Note: Plotting is commented out to avoid blocking in automated runs
    # Uncomment to visualize in interactive mode
    # simulator.plot_bankroll_paths(results, title="Season Bankroll Evolution")


# ============================================================================
# EXAMPLE 5: Bet Recommendation Display
# ============================================================================

def example_bet_display():
    """Example: Display bet recommendations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Bet Recommendation Display")
    print("=" * 80 + "\n")

    # This example assumes enriched predictions file exists
    # In practice, this would be generated by the Kelly calculation script

    display = BetRecommendationDisplay()

    # Try to find latest file (may not exist in fresh setup)
    latest_file = display.find_latest_enriched_file()

    if latest_file:
        print(f"Found enriched file: {latest_file}\n")

        # Load recommendations
        try:
            bets = display.load_recommendations(min_stake=1.0)
            print(f"Loaded {len(bets)} bets with stake > $1.00\n")

            if not bets.empty:
                # Format and display
                table = display.format_bet_table(bets)
                print("Bet Recommendations:")
                print(table)
        except Exception as e:
            print(f"Could not load recommendations: {e}")
    else:
        print("⚠️ No enriched predictions file found.")
        print("Run the Kelly Criterion calculation script first to generate bet recommendations.")


# ============================================================================
# EXAMPLE 6: Convenience Functions
# ============================================================================

def example_convenience_functions():
    """Example: Using convenience functions for common workflows."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Convenience Functions")
    print("=" * 80 + "\n")

    # Generate sample data
    np.random.seed(42)

    # Historical data
    historical_df = pd.DataFrame({
        'home_team': ['LAL'] * 50,
        'away_team': ['BOS'] * 50,
        'home_team_prob': np.random.beta(2, 2, 50),
        'accuracy': np.random.randint(0, 2, 50),
        'date': pd.date_range('2024-10-01', periods=50),
        'odds_1': np.random.uniform(1.5, 2.5, 50),
        'win': np.random.randint(0, 2, 50),
    })

    # Today's predictions
    today_df = pd.DataFrame({
        'home_team': ['LAL', 'GSW'],
        'away_team': ['MIA', 'BOS'],
        'home_team_prob': [0.65, 0.58],
        'date': ['2025-11-16', '2025-11-16'],
    })

    # Strategy config
    strategy_config = {
        'odds_min': 1.18,
        'odds_max': 3.00,
        'prob_min': 0.40,
        'good_home_teams': {'LAL', 'GSW', 'BOS'},
        'bet_fraction': 0.5,
        'cap_fraction': 0.30,
        'absolute_cap': 300.0,
    }

    # Use convenience function
    print("Using calibrate_and_simulate() convenience function...")
    today_calibrated, season_results = calibrate_and_simulate(
        historical_df,
        today_df,
        strategy_config,
        starting_bankroll=1000.0
    )

    print("\nToday's calibrated predictions:")
    print(today_calibrated[['home_team', 'away_team', 'home_team_prob', 'prob_platt', 'prob_iso']])

    print("\n✓ Season simulation complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("NBA BETTING MODULE - USAGE EXAMPLES")
    print("=" * 80)

    try:
        # Run all examples
        example_odds_fetching()
        example_calibration()
        example_kelly_calculations()
        example_bankroll_simulation()
        example_bet_display()
        example_convenience_functions()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
