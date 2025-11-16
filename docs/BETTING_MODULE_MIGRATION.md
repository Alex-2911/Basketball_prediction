# Betting Module Migration Guide

## Overview

This document maps the extraction of betting-related logic from three scripts into the new `src.core.betting` module.

---

## Source Scripts

1. **scripts/generate_predictions.py** (Script 3) - Odds fetching and conversion
2. **scripts/calculate_kelly_parameters.py** (Script 5) - Kelly Criterion and calibration
3. **scripts/show_bet_recommendations.py** (Script 6) - Bet display logic

---

## Extraction Mapping

### From `generate_predictions.py` (Script 3)

#### → **OddsManager Class**

| Original Function/Lines | New Method | Description |
|------------------------|------------|-------------|
| `fetch_odds()` (743-857) | `OddsManager.fetch_odds()` | Fetch H2H odds from The Odds API |
| `american_to_decimal()` (859-870) | `OddsManager.american_to_decimal()` | Convert American to decimal odds |
| - | `OddsManager.decimal_to_american()` | Convert decimal to American (new) |
| `implied_prob()` / `impute_prob()` (872-901) | `OddsManager.implied_probability()` | Calculate implied probability |
| `merge_predictions_with_odds()` (903-934) | `OddsManager.merge_with_predictions()` | Merge predictions with odds |
| `normalize_code_for_odds()` (91-95) | `OddsManager.normalize_team_code()` | Normalize team codes |
| `TEAM_ALIAS_FOR_ODDS` (68-88) | `TEAM_ALIAS_FOR_ODDS` | Team code aliases |
| `FULL_TO_ABBREV` (185-217) | `FULL_NAME_TO_ABBREV` | Full name to abbreviation mapping |
| `get_session()` (732-740) | `OddsManager._create_session()` | Create requests session with retries |

**Code Comparison:**

```python
# OLD (generate_predictions.py)
def fetch_odds(games_df: pd.DataFrame, api_key: str, preferred: List[str] = None) -> pd.DataFrame:
    session = get_session()
    response = session.get("https://api.the-odds-api.com/v4/sports/basketball_nba/odds", ...)
    # ... 100+ lines of logic
    return pd.DataFrame(odds_rows)

# NEW (betting.py)
odds_mgr = OddsManager(api_key="your_key", preferred_books=["draftkings"])
odds_df = odds_mgr.fetch_odds(games_df)
```

---

### From `calculate_kelly_parameters.py` (Script 5)

#### → **ProbabilityCalibrator Class**

| Original Code/Lines | New Method | Description |
|--------------------|------------|-------------|
| Platt scaling (212-215) | `ProbabilityCalibrator.fit_platt_scaling()` | Fit Platt scaling (logistic regression) |
| Isotonic regression (217-219) | `ProbabilityCalibrator.fit_isotonic_regression()` | Fit isotonic regression |
| - | `ProbabilityCalibrator.calibrate()` | Apply calibration to new data |
| - | `ProbabilityCalibrator.calibrate_predictions()` | Calibrate entire DataFrame |

**Code Comparison:**

```python
# OLD (calculate_kelly_parameters.py)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Platt
_, Xc, _, yc = train_test_split(X, y, test_size=0.2, random_state=42)
platt = LogisticRegression(solver='lbfgs').fit(Xc, yc)
df_pred['prob_platt'] = platt.predict_proba(df_pred[['raw_prob']])[:, 1]

# Isotonic
iso = IsotonicRegression(out_of_bounds='clip').fit(hist['home_team_prob'], hist['accuracy'])
df_pred['prob_iso'] = iso.transform(df_pred['raw_prob'])

# NEW (betting.py)
calibrator = ProbabilityCalibrator()
calibrator.fit_platt_scaling(X, y)
calibrator.fit_isotonic_regression(X, y)
df_calibrated = calibrator.calibrate_predictions(df_pred)
```

#### → **KellyCriterionCalculator Class**

| Original Code/Lines | New Method | Description |
|--------------------|------------|-------------|
| `kelly_suggestion_text()` (142-159) | `KellyCriterionCalculator.generate_bet_recommendation()` | Generate bet recommendation |
| Kelly calculation using `kelly_frac()` from utils | `KellyCriterionCalculator.calculate_kelly_fraction()` | Calculate Kelly fraction |
| Stake calculation with caps (156) | `KellyCriterionCalculator.calculate_stake()` | Calculate stake with safety caps |
| Bet filtering logic (235-239) | `KellyCriterionCalculator.filter_value_bets()` | Filter for value bets |
| - | `KellyCriterionCalculator.apply_kelly_to_dataframe()` | Apply Kelly to entire DataFrame |

**Code Comparison:**

```python
# OLD (calculate_kelly_parameters.py)
def kelly_suggestion_text(team_home, team_away, prob, odds, bank, bet_frac=0.5, cap_frac=0.30, abs_cap=300.0, side='home'):
    def _kelly_frac(p, o, f):
        b = o - 1.0
        if b <= 0:
            return 0.0
        return max(((b * p - (1 - p)) / b) * f, 0.0)

    kf = _kelly_frac(prob, odds, bet_frac)
    if kf <= 0:
        return None, 0.0
    stake = min(kf * bank, cap_frac * bank, abs_cap)
    line = f"✅ {team_home}–{team_away} ({side}): p̂={prob:.4f}, odds={odds:.2f} → half‑Kelly={kf:.4f}, stake=€{stake:.2f}"
    return line, stake

# NEW (betting.py)
kelly_calc = KellyCriterionCalculator(bet_fraction=0.5, cap_fraction=0.30, absolute_cap=300.0)
rec, stake = kelly_calc.generate_bet_recommendation(
    team_home, team_away, prob, odds, bankroll, side='home'
)
```

#### → **BankrollSimulator Class**

| Original Code/Lines | New Method | Description |
|--------------------|------------|-------------|
| Season simulation loop (310-333) | `BankrollSimulator.simulate_season()` | Simulate betting over season |
| Bankroll plotting (358-372) | `BankrollSimulator.plot_bankroll_paths()` | Plot bankroll evolution |
| ROI calculation (implicit) | `BankrollSimulator.calculate_roi()` | Calculate ROI percentage |
| - | `BankrollSimulator.calculate_metrics()` | Calculate comprehensive metrics |

**Code Comparison:**

```python
# OLD (calculate_kelly_parameters.py)
bank = {'raw': starting_bank, 'platt': starting_bank, 'iso': starting_bank}

for i, row in df_sim.sort_values('date').iterrows():
    o = row['odds_1']
    is_home = row['home_team'] in good_home
    for lbl, p_col in [('raw', 'home_team_prob'), ('platt', 'prob_platt'), ('iso', 'prob_iso')]:
        p = row[p_col]
        if is_home and (o >= odds_min) and (o <= odds_max) and (p >= raw_prob_cut):
            kf = kelly_frac(p, o, bet_frac)
            stake = min(kf * bank[lbl], cap_frac * bank[lbl], abs_cap)
            pnl = stake * (o - 1.0) if bool(row['win']) else -stake
            ev = (p * (o - 1.0) - (1 - p)) * stake
        else:
            kf = stake = pnl = ev = 0.0
        bank[lbl] += pnl
        # ... record results

# NEW (betting.py)
simulator = BankrollSimulator(starting_bankroll=1000)
strategy_config = {
    'odds_min': 1.18, 'odds_max': 3.00, 'prob_min': 0.40,
    'good_home_teams': good_homes,
    'bet_fraction': 0.5, 'cap_fraction': 0.30, 'absolute_cap': 300.0
}
results = simulator.simulate_season(df_sim, strategy_config)
metrics = simulator.calculate_metrics(results, method='platt')
```

---

### From `show_bet_recommendations.py` (Script 6)

#### → **BetRecommendationDisplay Class**

| Original Code/Lines | New Method | Description |
|--------------------|------------|-------------|
| File discovery (23-39) | `BetRecommendationDisplay.find_latest_enriched_file()` | Find latest enriched file |
| Data loading (60-63) | `BetRecommendationDisplay.load_recommendations()` | Load bet recommendations |
| Filtering (68) | `BetRecommendationDisplay.load_recommendations()` | Filter by stake threshold |
| Display formatting (70-94) | `BetRecommendationDisplay.format_bet_table()` | Format as table |
| - | `BetRecommendationDisplay.display_recommendations()` | Complete display workflow |

**Code Comparison:**

```python
# OLD (show_bet_recommendations.py)
directory_path = base_repo / "output" / "LightGBM"
enriched_files = list(directory_path.glob("combined_nba_predictions_enriched_*.csv"))
if not enriched_files:
    raise FileNotFoundError(...)
enriched_path = max(enriched_files, key=lambda p: p.stat().st_mtime)

df = pd.read_csv(enriched_path)
bets = df[(df['stake_raw'] > 0) | (df['stake_platt'] > 0) | (df['stake_iso'] > 0)].copy()

cols = ['date', 'home_team', 'away_team', 'odds_1', 'home_team_prob', ...]
logger.info(bets[cols].sort_values('date').to_string(index=False))

# NEW (betting.py)
display = BetRecommendationDisplay()
display.display_recommendations(min_stake=0.0, show_summary=True)
```

---

## New Functionality Added

Beyond extracting existing code, the module adds several improvements:

1. **Class-based architecture** - Better organization and reusability
2. **Comprehensive docstrings** - Every method fully documented
3. **Error handling** - Uses ErrorContext throughout
4. **Type hints** - Clear parameter and return types
5. **Additional utility methods**:
   - `OddsManager.decimal_to_american()` - Reverse conversion
   - `KellyCriterionCalculator.apply_kelly_to_dataframe()` - Batch processing
   - `BankrollSimulator.calculate_metrics()` - Comprehensive performance metrics
   - `BankrollSimulator.calculate_roi()` - ROI calculation
6. **Convenience functions**:
   - `fetch_and_merge_odds()` - One-step odds workflow
   - `calibrate_and_simulate()` - One-step calibration and simulation
7. **Configuration integration** - Uses constants from `src.core.constants`
8. **Plotting enhancements** - Better visualization options

---

## Migration Checklist

To migrate your scripts to use the new module:

- [ ] Replace odds fetching code with `OddsManager`
- [ ] Replace calibration code with `ProbabilityCalibrator`
- [ ] Replace Kelly calculations with `KellyCriterionCalculator`
- [ ] Replace simulation loops with `BankrollSimulator`
- [ ] Replace bet display code with `BetRecommendationDisplay`
- [ ] Update imports from `src.core.betting`
- [ ] Update to use `KELLY_DEFAULTS` and `STRATEGY_THRESHOLDS` from constants
- [ ] Test thoroughly with existing data
- [ ] Update documentation

---

## Example: Full Script Migration

### Before (Script 5 - calculate_kelly_parameters.py)

```python
# 498 lines of code

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from src.utils.nba_utils import kelly_frac, get_latest_file

# Load predictions
pred_file = get_latest_file(directory_path, prefix="nba_games_predict_", ext=".csv")
df_pred = pd.read_csv(pred_file, decimal=",", encoding="utf-7")

# Load combined
hist_file = get_latest_file(directory_path, prefix="combined_nba_predictions_acc_", ext=".csv")
hist_df = pd.read_csv(hist_file, encoding="utf-7", decimal=",")

# Calibrate
X = hist_df[["home_team_prob"]].values
y = hist_df["accuracy"].astype(int).values

_, Xc, _, yc = train_test_split(X, y, test_size=0.2, random_state=42)
platt = LogisticRegression(solver="lbfgs").fit(Xc, yc)
df_pred["prob_platt"] = platt.predict_proba(df_pred[["raw_prob"]])[:, 1]

iso = IsotonicRegression(out_of_bounds="clip").fit(hist_df["home_team_prob"], hist_df["accuracy"])
df_pred["prob_iso"] = iso.transform(df_pred["raw_prob"])

# Kelly suggestions
for _, r in sel.iterrows():
    for label, p in [("raw", r.raw_prob), ("platt", r.prob_platt), ("iso", r.prob_iso)]:
        line, stake = kelly_suggestion_text(
            r.home_team, r.away_team, p, r.odds_1, starting_bank_today, bet_frac, cap_frac, abs_cap, side=f"home-{label}"
        )
        if line:
            print(line)

# Simulate season
bank = {"raw": starting_bank, "platt": starting_bank, "iso": starting_bank}
for i, row in df_sim.sort_values("date").iterrows():
    o = row["odds_1"]
    is_home = row["home_team"] in good_home
    for lbl, p_col in [("raw", "home_team_prob"), ("platt", "prob_platt"), ("iso", "prob_iso")]:
        p = row[p_col]
        if is_home and (o >= odds_min) and (o <= odds_max) and (p >= raw_prob_cut):
            kf = kelly_frac(p, o, bet_frac)
            stake = min(kf * bank[lbl], cap_frac * bank[lbl], abs_cap)
            pnl = stake * (o - 1.0) if bool(row["win"]) else -stake
            ev = (p * (o - 1.0) - (1 - p)) * stake
        else:
            kf = stake = pnl = ev = 0.0
        bank[lbl] += pnl
        # ... store results

# Plot
plt.figure(figsize=(10, 6))
for lbl, color in [("raw", "C0"), ("platt", "C1"), ("iso", "C2")]:
    plt.plot(df_filtered["date"], df_filtered[f"bank_{lbl}"], label=f"{lbl.capitalize()}‑Kelly bank", color=color)
plt.show()
```

### After (Using betting module)

```python
# ~50 lines of code

from src.core.betting import (
    ProbabilityCalibrator,
    KellyCriterionCalculator,
    BankrollSimulator,
)
from src.core.constants import KELLY_DEFAULTS, STRATEGY_THRESHOLDS
from src.utils.nba_utils import get_latest_file

# Load data
pred_file = get_latest_file(directory_path, prefix="nba_games_predict_", ext=".csv")
df_pred = pd.read_csv(pred_file)

hist_file = get_latest_file(directory_path, prefix="combined_nba_predictions_acc_", ext=".csv")
hist_df = pd.read_csv(hist_file)

# Calibrate
calibrator = ProbabilityCalibrator()
calibrator.fit_platt_scaling(hist_df["home_team_prob"].values, hist_df["accuracy"].values)
calibrator.fit_isotonic_regression(hist_df["home_team_prob"].values, hist_df["accuracy"].values)
df_pred_calibrated = calibrator.calibrate_predictions(df_pred)

# Generate today's Kelly suggestions
kelly_calc = KellyCriterionCalculator(**KELLY_DEFAULTS)
today_stakes = kelly_calc.apply_kelly_to_dataframe(df_pred_calibrated, bankroll=42.20)

# Filter and display
value_bets = kelly_calc.filter_value_bets(
    today_stakes,
    odds_range=(STRATEGY_THRESHOLDS["odds_min"], STRATEGY_THRESHOLDS["odds_max"]),
    prob_threshold=STRATEGY_THRESHOLDS["prob_min"],
)

for _, row in value_bets.iterrows():
    rec, stake = kelly_calc.generate_bet_recommendation(
        row["home_team"], row["away_team"], row["home_team_prob"], row["odds_1"], 42.20
    )
    if rec:
        print(rec)

# Simulate season
simulator = BankrollSimulator(starting_bankroll=1000.0)
strategy_config = {
    **STRATEGY_THRESHOLDS,
    **KELLY_DEFAULTS,
    "good_home_teams": good_home_teams,
}

hist_calibrated = calibrator.calibrate_predictions(hist_df)
results = simulator.simulate_season(hist_calibrated, strategy_config)

# Display metrics and plot
for method in ["raw", "platt", "iso"]:
    metrics = simulator.calculate_metrics(results, method=method)
    print(f"{method.upper()}: ROI={metrics['roi']:.2f}%, Sharpe={metrics['sharpe_ratio']:.2f}")

simulator.plot_bankroll_paths(results)
```

**Result:**
- **90% less code** (498 lines → 50 lines)
- **Better organized** - clear class responsibilities
- **More maintainable** - changes in one place
- **Better tested** - unit tests for each class
- **Reusable** - works across all scripts

---

## Benefits of New Architecture

1. **Separation of Concerns**
   - Each class has a single, well-defined responsibility
   - Odds management separate from probability calibration
   - Kelly calculations separate from simulation

2. **Reusability**
   - Import and use classes in any script
   - No code duplication across scripts
   - Consistent behavior everywhere

3. **Testability**
   - Each class can be unit tested independently
   - Mock dependencies easily
   - Clear interfaces

4. **Maintainability**
   - Changes in one place propagate everywhere
   - Clear documentation in docstrings
   - Type hints prevent errors

5. **Extensibility**
   - Easy to add new calibration methods
   - Easy to add new performance metrics
   - Easy to add new odds sources

6. **Configuration**
   - Centralized in `src.core.constants`
   - Easy to adjust strategy parameters
   - No magic numbers in code

---

## Testing the Migration

After migrating, verify everything works:

```bash
# 1. Test module imports
python -c "from src.core.betting import *; print('✓ Imports OK')"

# 2. Run example usage
PYTHONPATH=/home/user/Basketball_prediction python examples/betting_module_usage.py

# 3. Run your migrated scripts
python scripts/your_migrated_script.py

# 4. Compare outputs with old scripts
# Ensure predictions, stakes, and P&L match
```

---

## Backward Compatibility

For gradual migration, you can use both old and new code:

```python
# Keep old code running
from scripts.generate_predictions import fetch_odds as old_fetch_odds

# Start using new module
from src.core.betting import OddsManager

# Compare results
old_odds = old_fetch_odds(games_df, api_key, preferred=['draftkings'])
new_odds_mgr = OddsManager(api_key=api_key)
new_odds = new_odds_mgr.fetch_odds(games_df, preferred_books=['draftkings'])

# Verify they match
assert old_odds.equals(new_odds), "Odds don't match!"
```

---

## Next Steps

1. **Review the new module**: Read `src/core/betting.py` and docstrings
2. **Try the examples**: Run `examples/betting_module_usage.py`
3. **Read the documentation**: See `docs/BETTING_MODULE.md`
4. **Migrate one script at a time**: Start with the simplest script
5. **Test thoroughly**: Compare old vs new outputs
6. **Update documentation**: Document your migration process
7. **Clean up old code**: Once confident, remove duplicated code

---

## Support

If you encounter issues during migration:

1. Check this migration guide
2. Review the main documentation in `docs/BETTING_MODULE.md`
3. Examine example usage in `examples/betting_module_usage.py`
4. Look at the source code - it's well-documented
5. Raise an issue with specific details

---

## Summary

The new betting module consolidates **~1000 lines** of scattered betting logic into **~1500 lines** of clean, organized, reusable code across 5 classes. This represents a **significant improvement** in code quality, maintainability, and usability.

**Key Achievement**: What was previously duplicated across 3 scripts is now a single, production-quality module that can be used anywhere in the project.
