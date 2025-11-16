# Betting Module Documentation

## Overview

The `src.core.betting` module provides a comprehensive, production-quality betting system for NBA predictions. It consolidates betting logic from three scripts into five well-organized classes:

1. **OddsManager** - Odds fetching, format conversion, implied probabilities
2. **ProbabilityCalibrator** - Probability calibration (Platt scaling, isotonic regression)
3. **KellyCriterionCalculator** - Kelly Criterion bet sizing and recommendations
4. **BankrollSimulator** - Season-long bankroll simulation and performance metrics
5. **BetRecommendationDisplay** - Bet recommendation display and formatting

## Installation & Setup

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Set up environment variables
export ODDS_API_KEY="your_odds_api_key_here"
```

## Quick Start

```python
from src.core.betting import (
    OddsManager,
    ProbabilityCalibrator,
    KellyCriterionCalculator,
    BankrollSimulator,
    BetRecommendationDisplay,
)

# 1. Fetch odds
odds_mgr = OddsManager(api_key="your_key")
odds_df = odds_mgr.fetch_odds(games_df)

# 2. Calibrate probabilities
calibrator = ProbabilityCalibrator()
calibrator.fit_platt_scaling(historical_probs, outcomes)
calibrated_df = calibrator.calibrate_predictions(predictions_df)

# 3. Calculate Kelly stakes
kelly_calc = KellyCriterionCalculator()
df_with_stakes = kelly_calc.apply_kelly_to_dataframe(predictions_df, bankroll=1000)

# 4. Simulate season
simulator = BankrollSimulator(starting_bankroll=1000)
results = simulator.simulate_season(historical_df, strategy_config)

# 5. Display recommendations
display = BetRecommendationDisplay()
display.display_recommendations()
```

---

## Class Reference

### 1. OddsManager

Handles all odds-related operations including API fetching, format conversion, and merging with predictions.

#### Initialization

```python
odds_mgr = OddsManager(
    api_key="your_odds_api_key",  # Optional if provided in fetch_odds()
    preferred_books=["draftkings", "fanduel"]  # Sportsbook preference order
)
```

#### Key Methods

##### `fetch_odds(games_df, api_key=None, preferred_books=None)`

Fetches H2H odds from The Odds API for scheduled games.

**Parameters:**
- `games_df` (DataFrame): Schedule with columns `['home_team', 'away_team']`
- `api_key` (str, optional): API key (overrides instance key)
- `preferred_books` (list, optional): Preferred sportsbooks

**Returns:**
- DataFrame with columns: `['home_team', 'away_team', 'odds 1', 'odds 2']`

**Example:**
```python
games = pd.DataFrame({
    'home_team': ['LAL', 'GSW'],
    'away_team': ['BOS', 'MIA']
})
odds_df = odds_mgr.fetch_odds(games)
```

##### `american_to_decimal(odds)`

Convert American moneyline to decimal odds.

**Example:**
```python
decimal = OddsManager.american_to_decimal(-150)  # Returns 1.67
```

##### `decimal_to_american(odds)`

Convert decimal odds to American moneyline.

**Example:**
```python
american = OddsManager.decimal_to_american(2.5)  # Returns 150
```

##### `implied_probability(odds, format='american')`

Calculate implied probability from odds.

**Example:**
```python
prob = OddsManager.implied_probability(-150, 'american')  # Returns 0.60
```

##### `merge_with_predictions(predictions_df, odds_df)`

Merge predictions with odds and calculate value edges.

**Parameters:**
- `predictions_df` (DataFrame): Predictions with `['home_team', 'away_team', 'home_team_prob']`
- `odds_df` (DataFrame): Odds from `fetch_odds()`

**Returns:**
- Merged DataFrame with additional columns:
  - `imp_prob_home`: Implied probability from home odds
  - `imp_prob_away`: Implied probability from away odds
  - `value_home`: Model edge on home team
  - `value_away`: Model edge on away team

**Example:**
```python
merged = odds_mgr.merge_with_predictions(predictions_df, odds_df)
print(merged[['home_team', 'home_team_prob', 'imp_prob_home', 'value_home']])
```

##### `normalize_team_code(team_code)`

Normalize team codes to canonical format (PHO→PHX, BKN→BRK, etc.).

---

### 2. ProbabilityCalibrator

Calibrates raw model probabilities to better align with observed frequencies.

#### Initialization

```python
calibrator = ProbabilityCalibrator()
```

#### Key Methods

##### `fit_platt_scaling(probabilities, outcomes, test_size=0.2)`

Fit Platt scaling calibration (logistic regression on probabilities).

**Parameters:**
- `probabilities` (array): Raw model probabilities
- `outcomes` (array): Binary outcomes (0 or 1)
- `test_size` (float): Fraction for calibration set

**Returns:**
- Fitted LogisticRegression model

**Example:**
```python
model = calibrator.fit_platt_scaling(
    historical_df['home_team_prob'].values,
    historical_df['accuracy'].values
)
```

##### `fit_isotonic_regression(probabilities, outcomes)`

Fit isotonic regression calibration.

**Example:**
```python
model = calibrator.fit_isotonic_regression(
    historical_df['home_team_prob'].values,
    historical_df['accuracy'].values
)
```

##### `calibrate(probabilities, method='platt')`

Apply calibration to new probabilities.

**Parameters:**
- `probabilities` (array): Probabilities to calibrate
- `method` (str): 'platt' or 'isotonic'

**Returns:**
- Calibrated probabilities

##### `calibrate_predictions(df, prob_col='home_team_prob', methods=None)`

Calibrate all predictions in a DataFrame.

**Parameters:**
- `df` (DataFrame): Predictions to calibrate
- `prob_col` (str): Column with raw probabilities
- `methods` (list): Methods to apply (default: ['platt', 'isotonic'])

**Returns:**
- DataFrame with added columns: `prob_platt`, `prob_iso`

**Example:**
```python
# After fitting both models
calibrated = calibrator.calibrate_predictions(today_predictions_df)
```

---

### 3. KellyCriterionCalculator

Calculates optimal bet sizes using the Kelly Criterion with safety caps.

#### Initialization

```python
kelly_calc = KellyCriterionCalculator(
    bet_fraction=0.5,      # Half-Kelly (conservative)
    cap_fraction=0.30,     # Max 30% of bankroll per bet
    absolute_cap=300.0     # Absolute max stake
)
```

#### Key Methods

##### `calculate_kelly_fraction(probability, odds, fraction=None)`

Calculate Kelly fraction for a bet.

**Parameters:**
- `probability` (float): Win probability (0 to 1)
- `odds` (float): Decimal odds (>1)
- `fraction` (float, optional): Kelly fraction override

**Returns:**
- Kelly fraction (0 to 1)

**Example:**
```python
kelly = kelly_calc.calculate_kelly_fraction(0.60, 2.0)  # 10% of bankroll
```

##### `calculate_stake(kelly_fraction, bankroll, cap_pct=None, abs_cap=None)`

Calculate actual stake with safety caps.

**Example:**
```python
stake = kelly_calc.calculate_stake(0.10, bankroll=1000)  # Returns 100.0
```

##### `generate_bet_recommendation(home_team, away_team, probability, odds, bankroll, side='home')`

Generate formatted bet recommendation if positive edge exists.

**Returns:**
- Tuple of (recommendation_text, stake) or (None, 0.0)

**Example:**
```python
rec, stake = kelly_calc.generate_bet_recommendation(
    'LAL', 'BOS', 0.60, 2.0, 1000, side='home'
)
if rec:
    print(rec)  # "✅ LAL–BOS (home): p̂=0.6000, odds=2.00 → Kelly=0.1000, stake=$100.00"
```

##### `filter_value_bets(df, odds_range=None, prob_threshold=None)`

Filter DataFrame for value bets.

**Example:**
```python
value_bets = kelly_calc.filter_value_bets(
    predictions_df,
    odds_range=(1.18, 3.00),
    prob_threshold=0.40
)
```

##### `apply_kelly_to_dataframe(df, bankroll, prob_col='home_team_prob', odds_col='odds_1', prefix='')`

Apply Kelly calculations to all rows.

**Returns:**
- DataFrame with added columns: `{prefix}kelly_frac`, `{prefix}stake`

**Example:**
```python
df_with_kelly = kelly_calc.apply_kelly_to_dataframe(
    predictions_df,
    bankroll=1000,
    prefix='raw_'
)
```

---

### 4. BankrollSimulator

Simulates betting performance over a season with multiple calibration methods.

#### Initialization

```python
simulator = BankrollSimulator(starting_bankroll=1000.0)
```

#### Key Methods

##### `simulate_season(df, strategy_config, methods=['raw', 'platt', 'iso'])`

Simulate betting over a full season.

**Parameters:**
- `df` (DataFrame): Historical data with columns:
  - `home_team`, `away_team`, `date`
  - `home_team_prob` (raw), `prob_platt`, `prob_iso`
  - `odds_1` (decimal)
  - `result` or `win` (actual outcome)
- `strategy_config` (dict): Configuration with keys:
  - `odds_min`, `odds_max`: Odds range filter
  - `prob_min`: Minimum probability threshold
  - `good_home_teams`: Set of teams to bet on
  - `bet_fraction`: Fraction of Kelly
  - `cap_fraction`: Max % of bankroll per bet
  - `absolute_cap`: Absolute max stake
- `methods` (list): Probability methods to simulate

**Returns:**
- DataFrame with simulation results including stake, P&L, EV, and bankroll columns for each method

**Example:**
```python
config = {
    'odds_min': 1.18,
    'odds_max': 3.00,
    'prob_min': 0.40,
    'good_home_teams': {'LAL', 'GSW', 'BOS'},
    'bet_fraction': 0.5,
    'cap_fraction': 0.30,
    'absolute_cap': 300.0,
}
results = simulator.simulate_season(historical_df, config)
```

##### `plot_bankroll_paths(results, methods=['raw', 'platt', 'iso'], title='Bankroll Evolution')`

Plot bankroll evolution over time.

**Example:**
```python
simulator.plot_bankroll_paths(results, title="Season Performance")
```

##### `calculate_roi(final_bankroll)`

Calculate Return on Investment percentage.

##### `calculate_metrics(results, method='raw')`

Calculate comprehensive performance metrics.

**Returns:**
- Dictionary with:
  - `roi`: Return on Investment (%)
  - `total_pnl`: Total profit/loss
  - `num_bets`: Number of bets placed
  - `win_rate`: Winning percentage
  - `avg_stake`: Average stake per bet
  - `max_drawdown`: Maximum drawdown (%)
  - `sharpe_ratio`: Sharpe ratio

**Example:**
```python
metrics = simulator.calculate_metrics(results, method='platt')
print(f"ROI: {metrics['roi']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

---

### 5. BetRecommendationDisplay

Displays and formats bet recommendations from enriched prediction files.

#### Initialization

```python
display = BetRecommendationDisplay(
    prediction_dir="/path/to/predictions"  # Optional, auto-detects if None
)
```

#### Key Methods

##### `find_latest_enriched_file(pattern='combined_nba_predictions_enriched_*.csv')`

Find most recent enriched predictions file.

**Returns:**
- Path to latest file or None

##### `load_recommendations(file_path=None, min_stake=0.0)`

Load bet recommendations from file.

**Parameters:**
- `file_path` (Path/str, optional): Path to file (auto-detects if None)
- `min_stake` (float): Minimum stake filter

**Returns:**
- DataFrame with bet recommendations

**Example:**
```python
bets = display.load_recommendations(min_stake=1.0)
```

##### `format_bet_table(bets_df, columns=None)`

Format bets as readable table string.

**Returns:**
- Formatted table string

##### `display_recommendations(file_path=None, min_stake=0.0, show_summary=True)`

Load and display bet recommendations with summary.

**Example:**
```python
display.display_recommendations(min_stake=1.0, show_summary=True)
```

---

## Convenience Functions

### `fetch_and_merge_odds(predictions_df, games_df, api_key, preferred_books=None)`

One-step function to fetch odds and merge with predictions.

**Example:**
```python
from src.core.betting import fetch_and_merge_odds

merged = fetch_and_merge_odds(
    predictions_df,
    games_df,
    api_key="your_key",
    preferred_books=["draftkings", "fanduel"]
)
```

### `calibrate_and_simulate(historical_df, today_predictions_df, strategy_config, starting_bankroll=1000.0)`

One-step function to calibrate and simulate.

**Returns:**
- Tuple of (calibrated_today_df, simulated_historical_df)

**Example:**
```python
from src.core.betting import calibrate_and_simulate

config = {
    'odds_min': 1.18,
    'odds_max': 3.00,
    'prob_min': 0.40,
    'good_home_teams': {'LAL', 'GSW'},
    'bet_fraction': 0.5,
    'cap_fraction': 0.30,
    'absolute_cap': 300.0,
}

today_calibrated, season_results = calibrate_and_simulate(
    historical_df,
    today_predictions_df,
    config,
    starting_bankroll=1000
)
```

---

## Complete Workflow Example

Here's a complete workflow from fetching odds to displaying recommendations:

```python
import os
import pandas as pd
from src.core.betting import (
    OddsManager,
    ProbabilityCalibrator,
    KellyCriterionCalculator,
    BankrollSimulator,
    BetRecommendationDisplay,
)
from src.core.constants import STRATEGY_THRESHOLDS, KELLY_DEFAULTS

# ===== 1. Load data =====
games_df = pd.read_csv('output/Gathering_Data/Next_Game/games_df_2025-11-16.csv')
predictions_df = pd.read_csv('output/LightGBM/nba_games_predict_2025-11-16.csv')
historical_df = pd.read_csv('output/LightGBM/combined_nba_predictions_acc_2025-11-16.csv')

# ===== 2. Fetch and merge odds =====
odds_mgr = OddsManager(api_key=os.getenv('ODDS_API_KEY'))
odds_df = odds_mgr.fetch_odds(games_df)
predictions_with_odds = odds_mgr.merge_with_predictions(predictions_df, odds_df)

# ===== 3. Calibrate probabilities =====
calibrator = ProbabilityCalibrator()

# Fit on historical data
X_hist = historical_df[['home_team_prob']].values
y_hist = historical_df['accuracy'].astype(int).values

calibrator.fit_platt_scaling(X_hist, y_hist)
calibrator.fit_isotonic_regression(X_hist, y_hist)

# Calibrate today's predictions
today_calibrated = calibrator.calibrate_predictions(predictions_with_odds)

# ===== 4. Calculate Kelly stakes for today =====
kelly_calc = KellyCriterionCalculator(
    bet_fraction=KELLY_DEFAULTS['bet_fraction'],
    cap_fraction=KELLY_DEFAULTS['cap_fraction'],
    absolute_cap=KELLY_DEFAULTS['absolute_cap']
)

bankroll = 1000.0
today_with_stakes = kelly_calc.apply_kelly_to_dataframe(
    today_calibrated,
    bankroll=bankroll,
    prefix='today_'
)

# Filter for value bets
value_bets = kelly_calc.filter_value_bets(
    today_with_stakes,
    odds_range=(STRATEGY_THRESHOLDS['odds_min'], STRATEGY_THRESHOLDS['odds_max']),
    prob_threshold=STRATEGY_THRESHOLDS['prob_min']
)

print("Today's Value Bets:")
print(value_bets[['home_team', 'away_team', 'home_team_prob', 'odds 1', 'today_stake']])

# ===== 5. Simulate season performance =====
simulator = BankrollSimulator(starting_bankroll=1000.0)

strategy_config = {
    'odds_min': STRATEGY_THRESHOLDS['odds_min'],
    'odds_max': STRATEGY_THRESHOLDS['odds_max'],
    'prob_min': STRATEGY_THRESHOLDS['prob_min'],
    'good_home_teams': {'LAL', 'GSW', 'BOS', 'MIA'},
    'bet_fraction': KELLY_DEFAULTS['bet_fraction'],
    'cap_fraction': KELLY_DEFAULTS['cap_fraction'],
    'absolute_cap': KELLY_DEFAULTS['absolute_cap'],
}

# Add calibrated columns to historical data
historical_calibrated = calibrator.calibrate_predictions(historical_df)

# Run simulation
season_results = simulator.simulate_season(historical_calibrated, strategy_config)

# Calculate metrics
for method in ['raw', 'platt', 'iso']:
    metrics = simulator.calculate_metrics(season_results, method=method)
    print(f"\n{method.upper()} Method:")
    print(f"  ROI: {metrics['roi']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Plot results
simulator.plot_bankroll_paths(season_results)

# ===== 6. Display recommendations =====
display = BetRecommendationDisplay()
display.display_recommendations(min_stake=1.0, show_summary=True)
```

---

## Configuration

Default configuration is available in `src.core.constants`:

```python
from src.core.constants import (
    KELLY_DEFAULTS,
    STRATEGY_THRESHOLDS,
    PREFERRED_SPORTSBOOKS
)

# Kelly Criterion defaults
KELLY_DEFAULTS = {
    'bet_fraction': 0.5,      # Half Kelly
    'cap_fraction': 0.30,     # Max 30% of bankroll
    'absolute_cap': 300.0,    # Max $300 per bet
}

# Strategy thresholds
STRATEGY_THRESHOLDS = {
    'odds_min': 1.18,
    'odds_max': 3.00,
    'prob_min': 0.40,
    'home_win_rate_min': 0.50,
}

# Preferred sportsbooks
PREFERRED_SPORTSBOOKS = ['draftkings', 'fanduel']
```

---

## Error Handling

All classes use the `ErrorContext` manager for comprehensive error handling:

```python
from src.utils.error_handlers import ErrorContext

with ErrorContext("Fetching odds", logger=logger):
    odds_df = odds_mgr.fetch_odds(games_df)
```

Errors are logged and raised with context information.

---

## Testing

Run the example usage script to verify installation:

```bash
PYTHONPATH=/home/user/Basketball_prediction python examples/betting_module_usage.py
```

All examples should complete successfully.

---

## Migration from Old Scripts

### From `generate_predictions.py`:
```python
# Old
odds_df = fetch_odds(games_df, API_KEY, preferred=['draftkings'])
decimal_odds = american_to_decimal(-150)
prob = implied_prob(-150)

# New
odds_mgr = OddsManager(api_key=API_KEY)
odds_df = odds_mgr.fetch_odds(games_df, preferred_books=['draftkings'])
decimal_odds = OddsManager.american_to_decimal(-150)
prob = OddsManager.implied_probability(-150, 'american')
```

### From `calculate_kelly_parameters.py`:
```python
# Old
platt = LogisticRegression().fit(X_cal, y_cal)
df['prob_platt'] = platt.predict_proba(X)[:, 1]

# New
calibrator = ProbabilityCalibrator()
calibrator.fit_platt_scaling(X, y)
df = calibrator.calibrate_predictions(df)
```

### From `show_bet_recommendations.py`:
```python
# Old
enriched_path = max(glob.glob('*.csv'), key=os.path.getmtime)
df = pd.read_csv(enriched_path)

# New
display = BetRecommendationDisplay()
df = display.load_recommendations()
```

---

## Best Practices

1. **Always calibrate probabilities** before betting - raw model probabilities are often overconfident
2. **Use fractional Kelly** (e.g., half-Kelly) to reduce variance
3. **Apply multiple safety caps** - both percentage and absolute
4. **Backtest strategies** using the simulator before live betting
5. **Monitor max drawdown** - if it exceeds 30%, reduce bet sizing
6. **Track Sharpe ratio** - higher is better (>1.0 is good for betting)
7. **Compare methods** - raw vs Platt vs isotonic - and choose the most conservative

---

## Performance Considerations

- **Odds fetching**: Rate-limited by The Odds API (500 requests/month free tier)
- **Calibration**: Fast (<1s for 1000 samples)
- **Simulation**: Fast (<1s for 100 games)
- **Memory**: Lightweight, works with large DataFrames (10k+ rows)

---

## Troubleshooting

### "No enriched predictions file found"
Run the Kelly Criterion script (Script 5) first to generate enriched predictions.

### "API key not found"
Set the environment variable:
```bash
export ODDS_API_KEY="your_key_here"
```

### "No odds found for team"
Check team code normalization. The module handles PHO/PHX, BKN/BRK, etc. automatically.

### "ModuleNotFoundError: No module named 'src'"
Run scripts from project root with PYTHONPATH:
```bash
PYTHONPATH=/home/user/Basketball_prediction python your_script.py
```

---

## Future Enhancements

Potential additions to the module:

1. **Multi-market support** - spreads, totals, player props
2. **Live odds tracking** - monitor line movements
3. **Arbitrage detection** - find risk-free bets across books
4. **Advanced calibration** - beta calibration, splines
5. **Portfolio optimization** - Kelly for multiple simultaneous bets
6. **Database integration** - save/load from PostgreSQL
7. **Web interface** - Flask dashboard for recommendations

---

## License

This module is part of the Basketball Prediction project.

---

## Support

For issues or questions:
1. Check this documentation
2. Review example usage in `examples/betting_module_usage.py`
3. Examine source code docstrings
4. Raise an issue in the project repository
