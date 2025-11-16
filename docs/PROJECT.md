# Basketball Prediction System

**End-to-End NBA Betting Prediction & Analytics Platform**

Version: 1.0
Season: 2025-26 (2026)
License: MIT

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Data Flow](#data-flow)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Betting Strategy](#betting-strategy)
7. [Technology Stack](#technology-stack)
8. [Quick Start](#quick-start)
9. [Project Structure](#project-structure)
10. [Contributing](#contributing)

---

## Overview

The Basketball Prediction System is a sophisticated, automated platform for predicting NBA game outcomes and identifying value betting opportunities. It combines:

- **Web scraping** for real-time NBA statistics
- **Machine learning** (LightGBM) for win probability predictions
- **Probability calibration** using Platt scaling and Isotonic regression
- **Kelly Criterion** for optimal bet sizing
- **Automated workflows** via GitHub Actions

### What It Does

1. **Scrapes** NBA game data from Basketball-Reference.com daily
2. **Calculates** rolling statistical averages (9-game windows)
3. **Trains** a LightGBM model to predict win probabilities
4. **Fetches** live betting odds from multiple sportsbooks
5. **Identifies** value bets where model probability > implied probability
6. **Calculates** optimal stake sizes using Kelly Criterion
7. **Tracks** prediction accuracy and betting performance over time

### Key Metrics

- **~24,773** historical games per season
- **155+** features per game (basic + advanced stats)
- **9-game** rolling window for averages
- **3 calibration methods** (Raw, Platt, Isotonic)
- **Half-Kelly** conservative bankroll management
- **30% max** stake cap per bet
- **€300** absolute max bet size

---

## Key Features

### 🤖 Automated Data Collection
- Daily scraping of completed games (06:00 UTC)
- Selenium-based web automation
- Retry logic for network failures
- HTML parsing with BeautifulSoup
- Team code normalization for data consistency

### 📊 Advanced Statistics
- **Basic stats**: Points, rebounds, assists, FG%, 3P%, etc.
- **Advanced stats**: TS%, eFG%, ORtg, DRtg, Pace, etc.
- **Rolling averages**: 9-game moving windows per team
- **Opponent stats**: Mirrored features with `_opp` suffix

### 🧠 Machine Learning
- **Algorithm**: LightGBM Gradient Boosting
- **Features**: 300+ (team stats + rolling averages + matchups)
- **Target**: Binary win/loss for next game
- **Train/Test**: 80/20 split
- **Validation**: AUC, accuracy, calibration metrics

### 🎯 Probability Calibration
- **Raw probabilities**: Direct model output
- **Platt scaling**: Logistic regression recalibration
- **Isotonic regression**: Non-parametric monotonic calibration
- **Purpose**: Improve probability estimates for betting

### 💰 Bankroll Management
- **Kelly Criterion**: Mathematically optimal stake sizing
- **Half-Kelly**: Conservative 50% of full Kelly (reduces variance)
- **Filters**:
  - Good home teams only (win rate ≥ 50%)
  - Odds between 1.18 - 3.00
  - Probability ≥ 0.40
- **Caps**: 30% of bankroll, max €300 per bet

### 📈 Performance Tracking
- Prediction accuracy (overall + by confidence)
- Bankroll growth over time
- P&L per bet (raw, Platt, Isotonic)
- Visualizations (matplotlib charts)

### 🔄 CI/CD Automation
- GitHub Actions workflows
- Daily execution schedule
- Automatic git commits
- Artifact storage
- Test suite validation

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
├─────────────────────────────────────────────────────────────┤
│  Basketball-Reference.com  │  The Odds API                  │
│  (Game Statistics)         │  (Live Betting Odds)           │
└──────────────┬──────────────┴────────────┬──────────────────┘
               │                           │
               ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA COLLECTION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  Script 1: Scrape Previous Games (Selenium + BeautifulSoup)│
│  Script 2: Get Next Game Schedule                          │
│  Script 3: Fetch Live Odds (API Integration)               │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PROCESSING LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  • Team code normalization (PHO→PHX, BKN→BRK, etc.)        │
│  • Rolling average calculation (9-game windows)            │
│  • Feature engineering (self-merge for matchups)           │
│  • MinMax scaling [0, 1]                                   │
│  • Target variable creation (won.shift(-1))                │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                 MACHINE LEARNING LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  • LightGBM training (80/20 split)                         │
│  • Hyperparameter tuning                                   │
│  • Win probability prediction                              │
│  • Probability calibration (Platt + Isotonic)              │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   BETTING STRATEGY LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  • Value edge calculation (model prob - implied prob)      │
│  • Kelly Criterion stake sizing                            │
│  • Bankroll management (half-Kelly + caps)                │
│  • Filter application (home teams, odds range, prob min)   │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   ANALYTICS & OUTPUT LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  • Prediction accuracy tracking                            │
│  • Bankroll simulation & backtesting                       │
│  • Performance visualization                               │
│  • CSV export + optional PostgreSQL/Supabase storage       │
└─────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         SCRIPTS (2026/src/)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1_get_data_previous_game_day_2026.py                         │
│  ├─ Web Scraping (Selenium)                                   │
│  ├─ HTML Parsing (BeautifulSoup)                              │
│  └─ CSV Export                                                │
│                                                                │
│  2_get_data_next_game_day_2026.py                             │
│  ├─ Schedule Extraction                                       │
│  └─ Next Games CSV                                            │
│                                                                │
│  3_predict_games_hybrid_2026.py                               │
│  ├─ Feature Engineering                                       │
│  ├─ LightGBM Training                                         │
│  ├─ Odds Fetching (API)                                       │
│  └─ Prediction CSV                                            │
│                                                                │
│  4_calculate_betting_statistics_2026.py                       │
│  ├─ Accuracy Calculation                                      │
│  └─ Stats CSV                                                 │
│                                                                │
│  5_kelly_betting_parameters_2026.py                           │
│  ├─ Probability Calibration                                   │
│  ├─ Kelly Criterion                                           │
│  ├─ Backtest Simulation                                       │
│  └─ Enriched CSV + Charts                                     │
│                                                                │
│  6_proposed_bets_2026.py                                      │
│  └─ Display Today's Bets                                      │
│                                                                │
│  nba_utils_2026.py                                            │
│  ├─ Common Utilities                                          │
│  ├─ Team Normalization                                        │
│  ├─ Betting Functions                                         │
│  └─ Data Processing                                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                       WORKFLOWS (.github/)                     │
├────────────────────────────────────────────────────────────────┤
│  daily_prediction_pipeline.yml                                │
│  ├─ Install dependencies                                      │
│  ├─ Run tests (pytest)                                        │
│  ├─ Execute all 6 scripts                                     │
│  ├─ Commit & push results                                     │
│  └─ Upload artifacts                                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                         TESTS (tests/)                         │
├────────────────────────────────────────────────────────────────┤
│  test_betting_utils.py (67 tests)                             │
│  test_team_normalization.py (83 tests)                        │
│  test_data_processing.py (24 tests)                           │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Daily Execution Pipeline

```
Time: 06:00 UTC
Trigger: GitHub Actions Schedule

┌─────────────────────────────────────────────────────────────┐
│ STEP 0: Pre-execution                                       │
├─────────────────────────────────────────────────────────────┤
│ ✓ Checkout repository                                      │
│ ✓ Install Python 3.12                                      │
│ ✓ Install Google Chrome (headless)                         │
│ ✓ Install dependencies (pip)                               │
│ ✓ Create .env file from secrets                            │
│ ✓ RUN TESTS (pytest) ← Fails if tests fail                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Scrape Yesterday's Games                           │
├─────────────────────────────────────────────────────────────┤
│ Input:  Yesterday's date                                    │
│ Source: Basketball-Reference.com                            │
│ Method: Selenium (headless Chrome)                          │
│ Output: nba_games_YYYY-MM-DD.csv                           │
│         (~24,773 rows × 155+ columns)                       │
│                                                             │
│ Contains:                                                   │
│ • Team, opponent, date, season                             │
│ • Basic stats (pts, reb, ast, stl, blk, etc.)             │
│ • Advanced stats (TS%, eFG%, ORtg, DRtg, Pace)            │
│ • Shooting percentages (FG%, 3P%, FT%)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Get Today's/Tomorrow's Schedule                    │
├─────────────────────────────────────────────────────────────┤
│ Input:  Current date                                        │
│ Source: Basketball-Reference.com                            │
│ Output: games_df_YYYY-MM-DD.csv                            │
│         (home_team, away_team, game_date)                   │
│                                                             │
│ Fallback logic:                                             │
│ • If no games found, check next day                        │
│ • Handle season opening games                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Generate Predictions                               │
├─────────────────────────────────────────────────────────────┤
│ Inputs:                                                     │
│ • Historical games (nba_games_*.csv)                       │
│ • Today's schedule (games_df_*.csv)                        │
│ • API key from environment                                 │
│                                                             │
│ Process:                                                    │
│ 1. Load historical data (last 350 games)                   │
│ 2. Preprocess: add target (won.shift(-1))                  │
│ 3. Calculate rolling 9-game averages                       │
│ 4. Normalize team codes (PHO→PHX, etc.)                    │
│ 5. Feature engineering: self-merge for matchups            │
│ 6. MinMax scaling [0, 1]                                   │
│ 7. Train/test split (80/20)                                │
│ 8. Train LightGBM classifier                               │
│ 9. Predict probabilities for today's games                 │
│ 10. Fetch live odds from The Odds API                      │
│ 11. Calculate value edges                                  │
│                                                             │
│ Output: nba_games_predict_YYYY-MM-DD.csv                   │
│         (home_team, away_team, home_team_prob, odds, date) │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Calculate Accuracy Statistics                      │
├─────────────────────────────────────────────────────────────┤
│ Inputs:                                                     │
│ • Predictions (nba_games_predict_*.csv)                    │
│ • Actual results (nba_games_*.csv)                         │
│                                                             │
│ Process:                                                    │
│ 1. Merge predictions with actual outcomes                  │
│ 2. Calculate overall accuracy                              │
│ 3. Calculate confidence-based accuracy                     │
│    • High confidence (prob > 0.60)                         │
│    • Low confidence (prob < 0.40)                          │
│                                                             │
│ Output: combined_nba_predictions_acc_YYYY-MM-DD.csv        │
│         (predictions + actual results + accuracy)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Calculate Kelly Betting Parameters                 │
├─────────────────────────────────────────────────────────────┤
│ Inputs:                                                     │
│ • Combined predictions (from step 4)                       │
│ • Bankroll: €1,000 (configurable)                         │
│                                                             │
│ Process:                                                    │
│ 1. Probability calibration:                                │
│    • Platt scaling (logistic regression)                   │
│    • Isotonic regression (non-parametric)                  │
│ 2. Calculate home win rates (last 20 games)                │
│ 3. Apply Kelly Criterion:                                  │
│    f* = (bp - q) / b × fraction                            │
│    where b = decimal_odds - 1                              │
│          p = win probability                               │
│          q = 1 - p                                         │
│          fraction = 0.5 (half-Kelly)                       │
│ 4. Apply filters:                                          │
│    • Home team win rate ≥ 50%                              │
│    • Odds: 1.18 ≤ o ≤ 3.00                                 │
│    • Probability ≥ 0.40                                    │
│ 5. Apply caps:                                             │
│    • Max 30% of bankroll                                   │
│    • Max €300 absolute                                     │
│ 6. Backtest simulation (season-long P&L)                   │
│ 7. Generate performance charts                             │
│                                                             │
│ Outputs:                                                    │
│ • home_win_rates_sorted_YYYY-MM-DD.csv                     │
│ • kelly_stakes_YYYY-MM-DD.csv                              │
│ • combined_nba_predictions_enriched_YYYY-MM-DD.csv         │
│ • Charts: bankroll growth, P&L by method                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Display Today's Recommended Bets                   │
├─────────────────────────────────────────────────────────────┤
│ Input:  Enriched predictions (from step 5)                 │
│                                                             │
│ Process:                                                    │
│ Filter to bets with positive stake (any method)            │
│                                                             │
│ Output: Console display of recommended bets                │
│         (team, odds, probability, stake, expected value)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ POST-EXECUTION                                              │
├─────────────────────────────────────────────────────────────┤
│ ✓ Git add output files                                     │
│ ✓ Git commit with timestamp                                │
│ ✓ Git push to remote                                       │
│ ✓ Upload artifacts (CSVs, charts)                          │
└─────────────────────────────────────────────────────────────┘
```

### Data Schema Evolution

```
RAW DATA (Basketball-Reference.com)
↓
PREPROCESSED (add target variable 'won')
↓
ROLLING AVERAGES (9-game windows per team)
↓
NORMALIZED (team codes: PHO→PHX, etc.)
↓
FEATURED (self-merge: home vs away matchups)
↓
SCALED (MinMax [0, 1])
↓
PREDICTED (LightGBM probabilities)
↓
CALIBRATED (Platt + Isotonic)
↓
VALUED (compare model prob vs implied prob)
↓
SIZED (Kelly Criterion stakes)
↓
FILTERED (bankroll management rules)
↓
EXECUTED (recommended bets)
```

---

## Machine Learning Pipeline

### Model Architecture

```python
Algorithm: LightGBM Gradient Boosting
Task: Binary Classification
Target: won (1 if team wins next game, 0 otherwise)

Hyperparameters:
{
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 10,
    "learning_rate": 0.1,
    "max_depth": 7,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "lambda_l1": 0.5,        # L1 regularization
    "lambda_l2": 0.5,        # L2 regularization
}
```

### Feature Engineering

**1. Raw Features** (155+ columns)
- Basic box score stats
- Advanced analytics
- Shooting percentages
- Possession-based metrics

**2. Rolling Averages** (9-game window)
- Applied to all numeric features
- Calculated per team-season
- Uses `min_periods=1` for early season

**3. Opponent Features** (mirrored)
- All features duplicated with `_opp` suffix
- Created via self-merge on game matchups

**4. Scaled Features** (MinMax [0, 1])
- Ensures all features in same range
- Improves convergence speed
- Required for distance-based algorithms

**Total Features**: ~300-400 after engineering

### Training Process

```python
1. Load historical data (last 350 games ≈ 4.3 games/team)
2. Preprocess: add target variable
3. Calculate rolling averages
4. Self-merge to create matchups
5. Scale features
6. Split: 80% train, 20% test
7. Train LightGBM
8. Evaluate: AUC, accuracy, log-loss
9. Feature importance analysis
10. Predict on upcoming games
```

### Probability Calibration

**Why Calibrate?**
- Raw ML probabilities often poorly calibrated
- Calibration improves betting decision quality
- Better estimates of true win probability

**Methods**:

1. **Raw** (baseline)
   - Direct LightGBM output
   - No calibration

2. **Platt Scaling**
   - Logistic regression on validation set
   - Parametric approach
   - Formula: `P(y=1|f) = 1 / (1 + exp(A·f + B))`

3. **Isotonic Regression**
   - Non-parametric approach
   - Monotonic step function
   - More flexible than Platt

**Comparison**: Brier score / log-loss on validation set

---

## Betting Strategy

### Kelly Criterion

**Formula**:
```
f* = (bp - q) / b

where:
  f* = fraction of bankroll to bet
  b  = decimal_odds - 1 (profit per unit)
  p  = probability of winning
  q  = 1 - p (probability of losing)
```

**Example**:
```
Team: Lakers
Model Probability: 60% (0.6)
Sportsbook Odds: +120 (2.20 decimal)
Implied Probability: 45.45%

b = 2.20 - 1 = 1.20
p = 0.60
q = 0.40

f* = (1.20 × 0.60 - 0.40) / 1.20
   = (0.72 - 0.40) / 1.20
   = 0.32 / 1.20
   = 0.2667 (26.67% of bankroll)

Half-Kelly: 0.2667 × 0.5 = 13.33%
On €1,000 bankroll: €133.30 stake
```

### Bankroll Management

**Conservative Approach**:
- **Half-Kelly**: Reduces variance, slower growth
- **Stake cap**: Max 30% of bankroll
- **Absolute cap**: Max €300 per bet
- **Filters**: Only good home teams, reasonable odds

**Filter Rules**:
```python
1. Home team win rate ≥ 50% (last 20 games)
2. Odds range: 1.18 ≤ decimal_odds ≤ 3.00
3. Model probability ≥ 0.40
4. Positive expected value (model prob > implied prob)
```

### Value Betting

**Edge Calculation**:
```
Value Edge = Model Probability - Implied Probability

Example:
Model: 60%
Implied: 45.45%
Edge: +14.55% (positive value)
```

**Expected Value**:
```
EV = (Probability × Win Amount) - ((1 - Probability) × Loss Amount)

Example:
Stake: €100
Odds: 2.20 (win €120 profit)
Probability: 60%

EV = (0.60 × 120) - (0.40 × 100)
   = 72 - 40
   = €32 per €100 bet
```

---

## Technology Stack

### Core Technologies

**Programming Language**:
- Python 3.12

**Web Scraping**:
- Selenium 4.7.0 (browser automation)
- BeautifulSoup 4.11.2 (HTML parsing)
- webdriver-manager 4.0.2 (Chrome driver)

**Data Processing**:
- pandas 2.2.3 (DataFrames)
- numpy 1.24.4 (numerical computing)

**Machine Learning**:
- LightGBM 3.3.5 (gradient boosting)
- scikit-learn 1.2.2 (preprocessing, calibration)

**Visualization**:
- matplotlib 3.8.2 (charts)

**Testing**:
- pytest 7.4.3 (unit tests)
- pytest-cov 4.1.0 (coverage)

**Other**:
- python-dotenv 1.0.0 (environment variables)
- xlsxwriter 3.1.2 (Excel export)
- lxml 5.4.0 (XML/HTML processing)
- requests 2.31.0 (HTTP client with retries)

**Database** (Optional):
- psycopg2-binary 2.9.9 (PostgreSQL driver)
- Supabase/PostgreSQL (production data storage)

### Data Storage

The system supports **dual-mode operation**:

**CSV Mode** (Default):
- Local file-based storage
- Simple, portable, version-controllable
- Suitable for development and small-scale use
- No external dependencies

**Database Mode** (Optional):
- PostgreSQL/Supabase integration
- Production-grade relational database
- Advanced querying and analytics
- Concurrent access support
- Automatic audit trails
- Foreign key constraints and data integrity

Enable database mode with `USE_DATABASE=true` in `.env`. See [DATABASE_SETUP.md](DATABASE_SETUP.md) for setup instructions.

### External Services

**Data Sources**:
- Basketball-Reference.com (game statistics)
- The Odds API (live betting odds)

**Infrastructure**:
- GitHub Actions (CI/CD)
- Git (version control)

---

## Quick Start

### Prerequisites

- Python 3.12
- Google Chrome browser
- Git
- API key from The Odds API

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Basketball_prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ODDS_API_KEY

# 5. Run tests
pytest tests/ -v

# 6. Run pipeline manually (optional)
cd 2026/src
python 1_get_data_previous_game_day_2026.py
python 2_get_data_next_game_day_2026.py
python 3_predict_games_hybrid_2026.py
python 4_calculate_betting_statistics_2026.py
python 5_kelly_betting_parameters_2026.py
python 6_proposed_bets_2026.py
```

### GitHub Actions Setup

```bash
# 1. Add API key as secret
# Go to: Settings → Secrets → Actions
# Add: ODDS_API_KEY = <your-key>

# 2. Enable workflows
# Go to: Actions tab
# Enable workflows

# 3. Manual trigger (optional)
# Go to: Actions → Daily Prediction Pipeline
# Click: Run workflow
```

---

## Project Structure

```
Basketball_prediction/
├── 2025/                           # 2024-25 season (archived)
│   ├── src/                        # Scripts
│   └── output/                     # Generated data
│
├── 2026/                           # 2025-26 season (ACTIVE)
│   ├── src/                        # Source code
│   │   ├── 1_get_data_previous_game_day_2026.py
│   │   ├── 2_get_data_next_game_day_2026.py
│   │   ├── 3_predict_games_hybrid_2026.py
│   │   ├── 4_calculate_betting_statistics_2026.py
│   │   ├── 5_kelly_betting_parameters_2026.py
│   │   ├── 6_proposed_bets_2026.py
│   │   └── nba_utils_2026.py       # Shared utilities
│   └── output/                     # Generated data
│       ├── Gathering_Data/
│       │   ├── Next_Game/          # Upcoming schedules
│       │   ├── Whole_Statistic/    # Historical games
│       │   └── data/
│       │       ├── 2026_standings/ # Monthly schedules (HTML)
│       │       └── 2026_scores/    # Box scores (HTML)
│       └── LightGBM/               # Predictions & analytics
│
├── .github/
│   └── workflows/
│       ├── daily_prediction_pipeline.yml
│       └── 1_get_data_previous_game_day 2026.yml
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_betting_utils.py       # 67 tests
│   ├── test_team_normalization.py  # 83 tests
│   └── test_data_processing.py     # 24 tests
│
├── docs/                           # Documentation
│   └── PROJECT.md                  # This file
│
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
├── SETUP.md                        # Setup instructions
├── LICENSE                         # MIT License
├── PRODUCTION_READY_TESTS.md       # Test documentation
└── TEST_COVERAGE_ANALYSIS.md       # Coverage analysis
```

---

## Contributing

### Development Workflow

1. **Create branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Edit code
3. **Write tests**: Add tests for new functionality
4. **Run tests**: `pytest tests/ -v`
5. **Commit**: `git commit -m "feat: your feature"`
6. **Push**: `git push origin feature/your-feature`
7. **Pull request**: Create PR on GitHub

### Coding Standards

- **Style**: Follow PEP 8
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type annotations
- **Tests**: Maintain >80% coverage for new code
- **Commits**: Use conventional commits (feat, fix, docs, test, etc.)

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_betting_utils.py -v

# Run with coverage
pytest tests/ --cov=2026/src --cov-report=html

# Run only critical tests (fast)
pytest tests/test_betting_utils.py tests/test_team_normalization.py -v
```

---

## License

MIT License - See [LICENSE](../LICENSE) file for details.

---

## Disclaimer

⚠️ **EDUCATIONAL PURPOSES ONLY**

This tool is for educational and research purposes. Sports betting involves financial risk. Past performance does not guarantee future results. Always bet responsibly and within your means.

- **No guarantees**: The model's predictions are not guaranteed to be accurate
- **Financial risk**: You can lose money betting on sports
- **Responsible gambling**: Only bet what you can afford to lose
- **Legal compliance**: Ensure sports betting is legal in your jurisdiction
- **No liability**: The authors are not responsible for any financial losses

---

## Support

- **Documentation**: See [docs/](../) folder
- **Issues**: Report bugs on GitHub Issues
- **Tests**: See [PRODUCTION_READY_TESTS.md](../PRODUCTION_READY_TESTS.md)
- **Setup**: See [SETUP.md](../SETUP.md)

---

**Last Updated**: 2025-11-15
**Version**: 1.0
**Status**: Production Ready (Core Functionality)
