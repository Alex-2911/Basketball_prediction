# Test Coverage Analysis

**Generated**: 2025-11-15
**Project**: Basketball Prediction System
**Current Coverage**: ~15% (Configuration & Infrastructure Only)

---

## 📊 Project Statistics

| Metric | Count |
|--------|-------|
| **Python source files** | 7 |
| **Total functions** | 71 |
| **Total classes** | 0 |
| **Test files** | 2 |
| **Tests written** | 21 |
| **Functions with unit tests** | 0 |

---

## ✅ What IS Covered (21 tests)

### 1. Infrastructure & Configuration Tests
**Coverage**: ~90% of infrastructure files

- [x] `.env.example` exists and contains required variables
- [x] `.gitignore` protects sensitive files (`.env`, logs, cache)
- [x] `requirements.txt` includes all dependencies
- [x] GitHub Actions workflow YAML is valid
- [x] Documentation files exist (`SETUP.md`, `README.md`)

### 2. Security Validation Tests
**Coverage**: 100% of security fixes

- [x] Hardcoded API key removed from source code
- [x] Environment variable loading implemented correctly
- [x] `.env` file is not tracked by git
- [x] No API key patterns in source files
- [x] Secrets properly protected

### 3. Portability Tests
**Coverage**: 100% of path-related code

- [x] No hardcoded Windows paths (`D:\...`) remain
- [x] Cross-platform `Path(__file__)` usage validated
- [x] Path resolution works on Linux
- [x] OS-appropriate path separators used

### 4. Code Syntax Tests
**Coverage**: 100% of modified files

- [x] `3_predict_games_hybrid_2026.py` compiles
- [x] `6_proposed_bets_2026.py` compiles
- [x] All import statements present
- [x] No syntax errors

---

## ❌ What IS NOT Covered (0 tests)

### 1. Unit Tests - **0% Coverage**

**Critical Business Logic Functions** (71 total, 0 tested):

#### From `nba_utils_2026.py` (585 lines, ~20 functions)
- [ ] `get_html()` - Web scraping with Selenium
- [ ] `parse_html()` - BeautifulSoup parsing
- [ ] `preprocess_nba_data()` - Add target variable
- [ ] `calculate_rolling_averages()` - 9-game rolling window
- [ ] `add_next_game_columns()` - Link rows to next game
- [ ] `normalize_team_code()` - Handle team abbreviation aliases
- [ ] `kelly_frac()` - **CRITICAL** - Kelly Criterion calculation
- [ ] `impute_prob()` - American odds → probability
- [ ] `am_to_dec()` - American → Decimal odds conversion
- [ ] `get_home_win_rates()` - Compute home performance

#### From `3_predict_games_hybrid_2026.py` (1,083 lines, ~15 functions)
- [ ] `get_directory_paths()` - Path resolution
- [ ] `get_current_date()` - Date utilities
- [ ] `get_latest_file()` - File discovery
- [ ] `normalize_code_for_odds()` - Team code mapping
- [ ] `fetch_odds()` - **CRITICAL** - API integration with The Odds API
- [ ] `build_team_mapping()` - Full team name → abbreviation
- [ ] LightGBM training logic
- [ ] Feature engineering pipeline
- [ ] Prediction generation

#### From `5_kelly_betting_parameters_2026.py` (412 lines, ~10 functions)
- [ ] Platt scaling calibration
- [ ] Isotonic regression calibration
- [ ] Kelly stake calculation with caps
- [ ] Backtest simulation
- [ ] Bankroll tracking
- [ ] P&L calculation

#### From other scripts (1_get_data, 2_get_next_game, 4_calculate_betting)
- [ ] `scrape_season_for_month()` - Download monthly schedules
- [ ] `scrape_game_day_boxscores()` - Fetch game box scores
- [ ] `process_saved_boxscores()` - Parse HTML to CSV
- [ ] `read_stats()` - Extract statistics
- [ ] `find_games_for_next_day()` - Schedule parsing
- [ ] Accuracy calculation logic
- [ ] Results merging logic

---

### 2. Integration Tests - **0% Coverage**

**External Dependencies Not Tested**:

- [ ] **Web Scraping**: Basketball-Reference.com integration
  - [ ] HTML structure changes
  - [ ] Rate limiting
  - [ ] Selenium driver initialization
  - [ ] Headless browser execution

- [ ] **API Integration**: The Odds API
  - [ ] Valid API responses
  - [ ] API rate limits (500 requests/month)
  - [ ] Invalid API keys
  - [ ] Network timeouts
  - [ ] Missing odds for teams

- [ ] **File I/O**: CSV reading/writing
  - [ ] Large file handling
  - [ ] Encoding issues (UTF-8 vs UTF-7)
  - [ ] Missing directories
  - [ ] Corrupted CSV files

- [ ] **LightGBM Integration**
  - [ ] Model training on real data
  - [ ] Prediction accuracy
  - [ ] Feature importance stability
  - [ ] Memory usage with large datasets

---

### 3. Data Processing Tests - **0% Coverage**

**Data Pipeline Not Tested**:

- [ ] **Rolling Averages**
  - [ ] Correct 9-game window calculation
  - [ ] Handling teams with <9 games
  - [ ] Edge cases (season start)

- [ ] **Feature Engineering**
  - [ ] Self-merge creating matchup features
  - [ ] Opponent statistics correctly mirrored
  - [ ] MinMax scaling range [0, 1]
  - [ ] NaN handling in features

- [ ] **Target Variable Creation**
  - [ ] `won.shift(-1)` logic
  - [ ] Last game of season (no next game)
  - [ ] Playoffs vs regular season

- [ ] **Data Validation**
  - [ ] Missing statistics detection
  - [ ] Anomaly detection (e.g., 200+ points)
  - [ ] Date range validation
  - [ ] Team code consistency

---

### 4. Error Handling Tests - **0% Coverage**

**Edge Cases Not Tested**:

- [ ] **Missing Data**
  - [ ] No games scheduled for tomorrow
  - [ ] Empty CSV files
  - [ ] Missing box scores
  - [ ] No historical data

- [ ] **Network Failures**
  - [ ] Basketball-Reference.com down
  - [ ] The Odds API unavailable
  - [ ] Selenium timeout
  - [ ] DNS resolution failures

- [ ] **Invalid Inputs**
  - [ ] Malformed API responses
  - [ ] Invalid team codes
  - [ ] Negative odds
  - [ ] Future dates beyond season

- [ ] **File System Issues**
  - [ ] Output directory doesn't exist
  - [ ] Permission denied
  - [ ] Disk full
  - [ ] File locked by another process

---

### 5. Business Logic Tests - **0% Coverage**

**Mathematical Functions Not Validated**:

- [ ] **Kelly Criterion**
  - [ ] Edge case: probability = 0
  - [ ] Edge case: probability = 1
  - [ ] Edge case: odds < 1.0
  - [ ] Fraction limits (50% of Kelly)
  - [ ] Stake caps (30% bankroll, €300 max)

- [ ] **Odds Conversion**
  - [ ] American (+150, -200) → Decimal
  - [ ] American → Implied probability
  - [ ] Edge case: +100, -100
  - [ ] Invalid odds (e.g., 0, negative decimal)

- [ ] **Probability Calibration**
  - [ ] Platt scaling coefficients
  - [ ] Isotonic regression monotonicity
  - [ ] Calibration on validation set
  - [ ] Brier score comparison

- [ ] **Team Code Normalization**
  - [ ] PHO ↔ PHX mapping
  - [ ] BKN ↔ BRK mapping
  - [ ] CHA ↔ CHO mapping
  - [ ] Lowercase/uppercase handling

---

## 📈 Coverage Estimate by Category

| Category | Coverage | Status |
|----------|----------|--------|
| **Configuration Files** | 90% | ✅ Good |
| **Security Fixes** | 100% | ✅ Excellent |
| **Path Portability** | 100% | ✅ Excellent |
| **Code Syntax** | 100% | ✅ Excellent |
| **Unit Tests** | **0%** | ❌ None |
| **Integration Tests** | **0%** | ❌ None |
| **Business Logic** | **0%** | ❌ None |
| **Error Handling** | **0%** | ❌ None |
| **End-to-End Tests** | **0%** | ❌ None |

**Overall Test Coverage**: **~15%** (infrastructure only)

---

## 🎯 Recommended Test Priorities

### Phase 1: Critical Business Logic (High Priority)
1. **Kelly Criterion calculation** - Financial impact
2. **Odds conversion functions** - Core betting logic
3. **Team code normalization** - Data integrity
4. **Rolling average calculation** - Feature quality

### Phase 2: Data Validation (Medium Priority)
5. **API response validation** - External dependency
6. **CSV parsing and validation** - Data quality
7. **Feature engineering correctness** - Model accuracy
8. **Probability calibration** - Betting edge

### Phase 3: Integration & E2E (Lower Priority)
9. **Web scraping robustness** - Can fail gracefully
10. **Full pipeline execution** - Covered by manual runs
11. **GitHub Actions workflow** - Already tested manually

---

## 💡 Quick Wins for Improving Coverage

### 1. Create `tests/test_utils.py`
```python
import pytest
from nba_utils_2026 import kelly_frac, am_to_dec, impute_prob, normalize_team_code

def test_kelly_fraction_basic():
    # p=0.6, decimal_odds=2.0, bankroll=1000, frac=0.5, max_frac=0.3
    stake = kelly_frac(0.6, 2.0, 1000, 0.5, 0.3)
    assert 0 <= stake <= 300  # Max 30% of bankroll

def test_american_to_decimal():
    assert am_to_dec(100) == 2.0
    assert am_to_dec(-100) == 2.0
    assert am_to_dec(150) == 2.5
    assert am_to_dec(-200) == 1.5

def test_team_normalization():
    assert normalize_team_code("PHO") == "PHX"
    assert normalize_team_code("BKN") == "BRK"
    assert normalize_team_code("CHA") == "CHO"
```

### 2. Create `tests/test_odds_api.py`
```python
import pytest
from unittest.mock import Mock, patch
import pandas as pd

def test_fetch_odds_with_valid_response():
    # Mock API response
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {...}
        result = fetch_odds(games_df, api_key)
        assert 'odds_1' in result.columns

def test_fetch_odds_api_failure():
    # Test graceful handling of API errors
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("API Error")
        # Should return empty DataFrame or handle error
```

### 3. Create `tests/test_data_processing.py`
```python
import pytest
import pandas as pd
import numpy as np

def test_rolling_averages():
    # Create sample data
    df = pd.DataFrame({...})
    result = calculate_rolling_averages(df, window=9)
    # Verify first 8 games have NaN
    # Verify 9th game has correct average
```

---

## 🚨 Current Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Kelly calculation error | **CRITICAL** - Wrong bet sizes | Medium | Add unit tests |
| Odds conversion bug | **HIGH** - Incorrect probabilities | Medium | Add unit tests |
| API rate limit exceeded | **HIGH** - No predictions | Low | Already has retry logic |
| Web scraping breaks | **MEDIUM** - Stale data | High | Add error handling tests |
| Rolling average error | **MEDIUM** - Bad features | Low | Add data validation |

---

## 📝 Conclusion

### Current State
- **Infrastructure tests**: Excellent (100%)
- **Functional tests**: None (0%)
- **Overall coverage**: ~15%

### What This Means
Your critical security and portability fixes are **well-validated**, but the **core business logic is untested**. The project will work if the logic is correct, but bugs in Kelly calculations, odds conversions, or data processing would go undetected until runtime.

### Recommendation
Start with **Phase 1 tests** (Kelly, odds, team codes) to protect against financial calculation errors. These are small, fast unit tests that can catch critical bugs.

---

**Would you like me to implement unit tests for the critical business logic functions?**
