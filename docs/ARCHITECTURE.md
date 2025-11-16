# System Architecture

**Basketball Prediction System - Technical Architecture**

Version: 1.0
Last Updated: 2025-11-15

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Patterns](#architectural-patterns)
3. [Component Details](#component-details)
4. [Data Models](#data-models)
5. [API Integration](#api-integration)
6. [Security Architecture](#security-architecture)
7. [Deployment Architecture](#deployment-architecture)
8. [Performance Considerations](#performance-considerations)
9. [Error Handling & Resilience](#error-handling--resilience)
10. [Future Architecture](#future-architecture)

---

## Overview

### Design Principles

1. **Modularity**: Each script has a single, well-defined responsibility
2. **Automation**: Fully automated via GitHub Actions
3. **Idempotency**: Scripts can be re-run safely
4. **Data-Driven**: All decisions based on historical data
5. **Test-First**: Critical functions have 100% test coverage
6. **Cross-Platform**: Works on Windows, Linux, macOS

### Architecture Style

**Pipeline Architecture** (Sequential Data Processing)

```
Input → Transform → Enrich → Model → Decide → Output
```

Each stage produces output consumed by the next stage, with CSV files as the data interchange format.

---

## Architectural Patterns

### 1. Pipeline Pattern

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Script  │ → │ Script  │ → │ Script  │ → │ Script  │
│    1    │    │    2    │    │    3    │    │    4    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     ↓              ↓              ↓              ↓
  CSV File      CSV File      CSV File      CSV File
```

**Benefits**:
- Simple to understand and debug
- Each stage independently testable
- Easy to add new stages
- Clear data lineage

**Tradeoffs**:
- File I/O overhead
- Not real-time
- Sequential (not parallel)

### 2. Shared Utilities Pattern

```
┌─────────────────────────────────────┐
│         nba_utils_2026.py          │
│  (Shared Functions & Constants)    │
└─────────────────────────────────────┘
       ↑           ↑           ↑
       │           │           │
  ┌────┴───┐  ┌───┴────┐ ┌────┴───┐
  │Script 1│  │Script 3│ │Script 5│
  └────────┘  └────────┘ └────────┘
```

**Shared Utilities**:
- Team code normalization
- Kelly Criterion calculations
- Odds conversions
- Rolling averages
- Web scraping helpers

### 3. Adapter Pattern (External Services)

```
┌──────────────────────┐
│   Script 3           │
│ (Our Application)    │
└──────┬───────────────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌─────────────┐  ┌──────────────┐
│ Basketball  │  │  The Odds    │
│ Reference   │  │     API      │
│  (Scrape)   │  │ (REST API)   │
└─────────────┘  └──────────────┘
```

**Adapters Handle**:
- Different data formats (HTML vs JSON)
- Network failures & retries
- Rate limiting
- Data normalization

---

## Component Details

### Script 1: Data Collection (Scraping)

**Purpose**: Scrape completed NBA games from Basketball-Reference.com

**Architecture**:
```python
┌─────────────────────────────────────────┐
│  1_get_data_previous_game_day_2026.py  │
├─────────────────────────────────────────┤
│                                         │
│  main()                                 │
│  ├─ get_directory_paths()              │
│  ├─ scrape_season_for_month()          │
│  │  └─ get_html() [Selenium]           │
│  ├─ scrape_game_day_boxscores()        │
│  │  └─ get_html() [Selenium]           │
│  └─ process_saved_boxscores()          │
│     ├─ parse_html() [BeautifulSoup]    │
│     ├─ read_stats()                     │
│     └─ pd.concat() → CSV               │
│                                         │
└─────────────────────────────────────────┘

External Dependencies:
- Selenium (browser automation)
- ChromeDriver (headless Chrome)
- BeautifulSoup (HTML parsing)
- Basketball-Reference.com (data source)
```

**Key Design Decisions**:

1. **Why Selenium?**
   - Basketball-Reference uses JavaScript rendering
   - Static HTML parsing insufficient
   - Need full browser execution

2. **Why Save HTML Files?**
   - Debugging: Can inspect raw HTML
   - Resilience: Can re-parse without re-scraping
   - Rate limiting: Avoid hammering the server

3. **Retry Logic**:
   ```python
   retries = Retry(
       total=3,
       backoff_factor=1,
       status_forcelist=[500, 502, 503, 504]
   )
   ```

**Data Flow**:
```
Basketball-Reference.com
    ↓ [Selenium GET]
HTML Files (2026/output/.../data/2026_scores/)
    ↓ [BeautifulSoup Parse]
DataFrame (155+ columns)
    ↓ [Pandas Concat]
nba_games_YYYY-MM-DD.csv
```

---

### Script 2: Schedule Collection

**Purpose**: Get upcoming game schedule

**Architecture**:
```python
┌─────────────────────────────────────────┐
│  2_get_data_next_game_day_2026.py      │
├─────────────────────────────────────────┤
│                                         │
│  main()                                 │
│  ├─ find_games_for_next_day()          │
│  │  ├─ get_html()                      │
│  │  ├─ parse_html()                    │
│  │  └─ extract schedule table          │
│  └─ pd.DataFrame() → CSV               │
│                                         │
└─────────────────────────────────────────┘
```

**Fallback Logic**:
```python
if no_games_found(today):
    check_tomorrow()
if still_no_games:
    return_empty_schedule()
```

**Output Schema**:
```python
{
    "home_team": str,    # e.g., "LAL"
    "away_team": str,    # e.g., "BOS"
    "game_date": str,    # e.g., "2025-10-23"
}
```

---

### Script 3: Prediction & Odds Integration

**Purpose**: Train ML model, predict games, fetch odds

**Architecture**:
```python
┌──────────────────────────────────────────┐
│   3_predict_games_hybrid_2026.py        │
├──────────────────────────────────────────┤
│                                          │
│  main()                                  │
│  ├─ Load Data                            │
│  │  ├─ get_latest_file()                │
│  │  └─ pd.read_csv()                    │
│  │                                       │
│  ├─ Data Processing                      │
│  │  ├─ preprocess_nba_data()            │
│  │  ├─ calculate_rolling_averages()     │
│  │  ├─ normalize_team_codes()           │
│  │  └─ add_next_game_columns()          │
│  │                                       │
│  ├─ Feature Engineering                  │
│  │  ├─ self_merge() [home vs away]     │
│  │  └─ MinMaxScaler()                   │
│  │                                       │
│  ├─ Model Training                       │
│  │  ├─ train_test_split(80/20)         │
│  │  ├─ LGBMClassifier.fit()             │
│  │  ├─ accuracy_score()                 │
│  │  └─ feature_importances_            │
│  │                                       │
│  ├─ Prediction                           │
│  │  └─ model.predict_proba()            │
│  │                                       │
│  ├─ Odds Fetching                        │
│  │  ├─ fetch_odds() [API call]         │
│  │  ├─ normalize_team_code()            │
│  │  └─ merge with predictions           │
│  │                                       │
│  └─ Output                               │
│     └─ nba_games_predict_*.csv          │
│                                          │
└──────────────────────────────────────────┘
```

**Machine Learning Pipeline**:

```
Raw Data (155+ features)
    ↓
Preprocessing
    ├─ Add target (won.shift(-1))
    ├─ Filter valid rows
    └─ Type conversions
    ↓
Rolling Averages
    ├─ Group by (team, season)
    ├─ 9-game window
    └─ min_periods=1
    ↓
Normalization
    ├─ Team codes (PHO→PHX)
    └─ Consistent abbreviations
    ↓
Feature Engineering
    ├─ Self-merge (home vs away)
    ├─ Create opponent features (_opp)
    └─ ~300 total features
    ↓
Scaling
    ├─ MinMaxScaler [0, 1]
    └─ Fit on training data only
    ↓
Model Training
    ├─ LightGBM binary classification
    ├─ Hyperparameters (fixed)
    └─ 80/20 split
    ↓
Prediction
    ├─ predict_proba()
    └─ [prob_loss, prob_win]
```

**API Integration**:

```python
def fetch_odds(games_df, api_key):
    """
    Fetch live odds from The Odds API

    Endpoint: GET /v4/sports/basketball_nba/odds

    Parameters:
        regions: us
        markets: h2h (head-to-head / moneyline)
        oddsFormat: american
        apiKey: from environment

    Returns:
        DataFrame with:
        - home_team
        - away_team
        - home_odds (American)
        - away_odds (American)
        - bookmaker
    """
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

    response = requests.get(url, params={
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "apiKey": api_key
    })

    # Parse JSON response
    # Normalize team codes
    # Merge with predictions

    return enriched_df
```

**Error Handling**:
```python
try:
    odds_df = fetch_odds(...)
except requests.exceptions.RequestException as e:
    logger.error(f"API request failed: {e}")
    # Continue with predictions only (no odds)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Fail gracefully
```

---

### Script 4: Accuracy Tracking

**Purpose**: Evaluate prediction accuracy

**Architecture**:
```python
┌──────────────────────────────────────────┐
│  4_calculate_betting_statistics_2026.py │
├──────────────────────────────────────────┤
│                                          │
│  main()                                  │
│  ├─ Load Predictions                     │
│  ├─ Load Actual Results                  │
│  ├─ Merge on (home_team, date)          │
│  ├─ Calculate Metrics                    │
│  │  ├─ Overall accuracy                 │
│  │  ├─ High confidence (prob > 0.60)    │
│  │  └─ Low confidence (prob < 0.40)     │
│  └─ Output CSV                           │
│                                          │
└──────────────────────────────────────────┘
```

**Accuracy Metrics**:
```python
# Overall
accuracy = correct_predictions / total_games

# High Confidence
high_conf = df[df['prob'] > 0.60]
high_accuracy = high_conf['correct'].mean()

# Low Confidence
low_conf = df[df['prob'] < 0.40]
low_accuracy = low_conf['correct'].mean()
```

---

### Script 5: Kelly Criterion & Backtesting

**Purpose**: Calculate optimal bet sizes and simulate performance

**Architecture**:
```python
┌──────────────────────────────────────────┐
│  5_kelly_betting_parameters_2026.py     │
├──────────────────────────────────────────┤
│                                          │
│  main()                                  │
│  ├─ Load Combined Data                   │
│  │                                       │
│  ├─ Probability Calibration              │
│  │  ├─ Platt Scaling                    │
│  │  │  ├─ LogisticRegression()          │
│  │  │  └─ fit(raw_prob, actual)         │
│  │  └─ Isotonic Regression               │
│  │     ├─ IsotonicRegression()          │
│  │     └─ fit(raw_prob, actual)         │
│  │                                       │
│  ├─ Home Win Rate Calculation            │
│  │  ├─ Last 20 games per team           │
│  │  └─ Filter: win_rate >= 50%          │
│  │                                       │
│  ├─ Kelly Criterion                      │
│  │  ├─ For each game:                   │
│  │  │  ├─ Calculate f* = (bp-q)/b       │
│  │  │  ├─ Apply fraction (0.5)          │
│  │  │  ├─ Apply caps (30%, €300)        │
│  │  │  └─ Filter (odds, prob range)     │
│  │  └─ Three methods:                   │
│  │     ├─ Raw probabilities             │
│  │     ├─ Platt probabilities           │
│  │     └─ Isotonic probabilities        │
│  │                                       │
│  ├─ Backtest Simulation                  │
│  │  ├─ Start bankroll = €1,000          │
│  │  ├─ For each bet:                    │
│  │  │  ├─ Calculate stake               │
│  │  │  ├─ Resolve outcome               │
│  │  │  ├─ Update bankroll               │
│  │  │  └─ Track P&L                     │
│  │  └─ Generate performance charts      │
│  │                                       │
│  └─ Output                               │
│     ├─ home_win_rates_*.csv             │
│     ├─ kelly_stakes_*.csv               │
│     ├─ enriched_*.csv                   │
│     └─ Charts (PNG)                     │
│                                          │
└──────────────────────────────────────────┘
```

**Calibration Architecture**:

```python
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Platt Scaling
platt = LogisticRegression()
platt.fit(raw_probs.reshape(-1, 1), actual_outcomes)
calibrated_probs_platt = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

# Isotonic Regression
isotonic = IsotonicRegression(out_of_bounds='clip')
isotonic.fit(raw_probs, actual_outcomes)
calibrated_probs_iso = isotonic.predict(raw_probs)
```

**Kelly Criterion Implementation**:

```python
def kelly_frac(p: float, o: float, f: float = 1.0) -> float:
    """
    Kelly fraction for decimal odds

    Args:
        p: win probability (0..1)
        o: decimal odds (>1)
        f: fraction of Kelly to use (0..1)

    Returns:
        stake fraction (0..1 of bankroll)
    """
    b = float(o) - 1.0
    if b <= 0 or p is None or np.isnan(p):
        return 0.0

    kelly = (b * p - (1 - p)) / b
    return max(kelly * float(f), 0.0)

# Apply with caps
def calculate_stake(prob, odds, bankroll, fraction=0.5):
    kelly_pct = kelly_frac(prob, odds, fraction)
    stake = bankroll * kelly_pct

    # Apply caps
    stake = min(stake, bankroll * 0.30)  # Max 30%
    stake = min(stake, 300.0)            # Max €300

    return stake
```

---

### Script 6: Bet Display

**Purpose**: Show today's recommended bets

**Architecture**:
```python
┌──────────────────────────────────────────┐
│  6_proposed_bets_2026.py                │
├──────────────────────────────────────────┤
│                                          │
│  main()                                  │
│  ├─ Load Enriched Data                   │
│  ├─ Filter: stake > 0 (any method)      │
│  ├─ Sort by date                         │
│  └─ Display                              │
│     ├─ Team matchup                     │
│     ├─ Odds                              │
│     ├─ Probabilities (raw/platt/iso)    │
│     ├─ Stakes (raw/platt/iso)           │
│     └─ Expected P&L                      │
│                                          │
└──────────────────────────────────────────┘
```

**Display Format**:
```
=== Bets Placed (Raw / Platt / Iso Kelly) ===

Date       | Home  | Away  | Odds | Prob  | Stake | Expected P&L
-----------|-------|-------|------|-------|-------|-------------
2025-10-23 | LAL   | BOS   | -110 | 60%   | €133  | +€12
2025-10-23 | GSW   | PHX   | +120 | 55%   | €89   | +€8
```

---

## Data Models

### Game Statistics Schema

**File**: `nba_games_YYYY-MM-DD.csv`

```python
{
    # Identifiers
    "team": str,              # e.g., "LAL"
    "opponent": str,          # e.g., "BOS"
    "date": str,              # e.g., "2025-10-23"
    "season": int,            # e.g., 2026
    "home": bool,             # True if home game

    # Basic Stats
    "pts": int,               # Points scored
    "opp_pts": int,           # Opponent points
    "fg": int,                # Field goals made
    "fga": int,               # Field goals attempted
    "fg_pct": float,          # FG%
    "fg3": int,               # 3-pointers made
    "fg3a": int,              # 3-pointers attempted
    "fg3_pct": float,         # 3P%
    "ft": int,                # Free throws made
    "fta": int,               # Free throws attempted
    "ft_pct": float,          # FT%
    "orb": int,               # Offensive rebounds
    "drb": int,               # Defensive rebounds
    "trb": int,               # Total rebounds
    "ast": int,               # Assists
    "stl": int,               # Steals
    "blk": int,               # Blocks
    "tov": int,               # Turnovers
    "pf": int,                # Personal fouls

    # Advanced Stats
    "ts_pct": float,          # True shooting %
    "efg_pct": float,         # Effective FG%
    "fg3a_per_fga_pct": float,# 3PA / FGA
    "fta_per_fga_pct": float, # FTA / FGA
    "orb_pct": float,         # Offensive rebound %
    "drb_pct": float,         # Defensive rebound %
    "trb_pct": float,         # Total rebound %
    "ast_pct": float,         # Assist %
    "stl_pct": float,         # Steal %
    "blk_pct": float,         # Block %
    "tov_pct": float,         # Turnover %
    "usg_pct": float,         # Usage %
    "off_rtg": float,         # Offensive rating
    "def_rtg": float,         # Defensive rating
    "pace": float,            # Pace (possessions/48min)

    # Target
    "won": int,               # 1 if won, 0 if lost

    # ... 155+ total columns
}
```

### Prediction Schema

**File**: `nba_games_predict_YYYY-MM-DD.csv`

```python
{
    "home_team": str,         # e.g., "LAL"
    "away_team": str,         # e.g., "BOS"
    "date": str,              # e.g., "2025-10-23"
    "home_team_prob": float,  # e.g., 0.62 (62% win prob)
    "result": str,            # Actual outcome (if known)
    "odds_1": float,          # Home team odds (American)
    "odds_2": float,          # Away team odds (American)
}
```

### Enriched Schema

**File**: `combined_nba_predictions_enriched_YYYY-MM-DD.csv`

```python
{
    # From Predictions
    "home_team": str,
    "away_team": str,
    "date": str,
    "home_team_prob": float,   # Raw probability

    # Calibrated Probabilities
    "prob_platt": float,       # Platt scaled
    "prob_iso": float,         # Isotonic

    # Odds
    "home_best_odds": float,   # Best available (American)
    "home_decimal_odds": float,# Decimal format
    "implied_prob": float,     # From odds

    # Home Win Rate
    "home_win_rate": float,    # Last 20 games

    # Kelly Stakes
    "stake_raw": float,        # Using raw prob
    "stake_platt": float,      # Using Platt prob
    "stake_iso": float,        # Using Isotonic prob

    # Actual Outcomes (if known)
    "win": int,                # 1/0/NaN
    "pnl_raw": float,          # Profit/loss (raw)
    "pnl_platt": float,        # Profit/loss (Platt)
    "pnl_iso": float,          # Profit/loss (Isotonic)
}
```

---

## API Integration

### The Odds API

**Base URL**: `https://api.the-odds-api.com/v4`

**Authentication**: API key in query parameter

**Endpoint**: `/sports/basketball_nba/odds`

**Request**:
```http
GET /v4/sports/basketball_nba/odds?apiKey=XXX&regions=us&markets=h2h&oddsFormat=american HTTP/1.1
Host: api.the-odds-api.com
```

**Response** (simplified):
```json
[
  {
    "id": "abc123",
    "sport_key": "basketball_nba",
    "commence_time": "2025-10-23T19:00:00Z",
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "bookmakers": [
      {
        "key": "draftkings",
        "title": "DraftKings",
        "markets": [
          {
            "key": "h2h",
            "outcomes": [
              {"name": "Los Angeles Lakers", "price": -110},
              {"name": "Boston Celtics", "price": -110}
            ]
          }
        ]
      }
    ]
  }
]
```

**Rate Limits**:
- Free tier: 500 requests/month
- 1 request per second recommended
- Tracked via `requests-remaining` header

**Error Handling**:
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

response = session.get(url, timeout=10)
response.raise_for_status()
```

---

## Security Architecture

### Secrets Management

**Environment Variables**:
```
ODDS_API_KEY=xxx  # The Odds API key
```

**Storage**:
- Local: `.env` file (gitignored)
- GitHub: Repository Secrets
- Never in code or version control

**Access**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ODDS_API_KEY")

if not api_key:
    raise ValueError("ODDS_API_KEY not found")
```

### Data Security

**Sensitive Data**:
- API keys
- Personal betting history
- Financial data (bankroll, stakes)

**Protection**:
- All sensitive files in `.gitignore`
- No PII (personally identifiable information) collected
- All data processing local

### Network Security

**HTTPS Only**:
- All API calls use HTTPS
- Certificate verification enabled
- No plain HTTP

**Input Validation**:
```python
def validate_team_code(code: str) -> bool:
    """Validate team code before use"""
    valid_codes = get_team_codes().values()
    return code in valid_codes

def validate_odds(odds: float) -> bool:
    """Validate odds are in reasonable range"""
    return -10000 < odds < 10000

def validate_probability(prob: float) -> bool:
    """Validate probability is between 0 and 1"""
    return 0.0 <= prob <= 1.0
```

---

## Deployment Architecture

### GitHub Actions

**Workflow File**: `.github/workflows/daily_prediction_pipeline.yml`

**Execution Environment**:
```yaml
runs-on: ubuntu-latest

python-version: "3.11"

chrome: google-chrome-stable (headless)
```

**Execution Flow**:
```
1. Checkout code (actions/checkout@v4)
2. Setup Python (actions/setup-python@v5)
3. Install Chrome (apt-get)
4. Install dependencies (pip)
5. Create .env (from secrets)
6. RUN TESTS (pytest) ← Fails if tests fail
7. Run Scripts 1-6
8. Commit & Push results
9. Upload artifacts
```

**Schedule**:
```yaml
schedule:
  - cron: "0 6 * * *"  # Daily at 06:00 UTC
```

**Manual Trigger**:
```yaml
workflow_dispatch:  # Allows manual runs
```

### Local Development

**Environment Setup**:
```bash
# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env with your API key
```

**Running Locally**:
```bash
cd 2026/src

# Individual scripts
python 1_get_data_previous_game_day_2026.py
python 2_get_data_next_game_day_2026.py
# ... etc

# Or in sequence
for i in {1..6}; do
    python ${i}_*.py
done
```

---

## Performance Considerations

### Data Volume

**Historical Data**:
- ~24,773 games per season
- ~155 columns per game
- File size: ~50MB CSV

**Processing Time**:
- Script 1 (Scraping): ~5-10 minutes
- Script 2 (Schedule): ~30 seconds
- Script 3 (Prediction): ~1-2 minutes
- Script 4 (Stats): ~10 seconds
- Script 5 (Kelly): ~30 seconds
- Script 6 (Display): <1 second

**Total Pipeline**: ~10-15 minutes

### Optimization Opportunities

**Current**:
- Sequential execution
- Full dataset reloading each script
- CSV file I/O

**Potential Improvements**:
1. **Database**: SQLite/PostgreSQL instead of CSV
2. **Caching**: Cache rolling averages, features
3. **Incremental**: Only process new games
4. **Parallel**: Run independent steps concurrently
5. **Vectorization**: More numpy, less Python loops

### Memory Usage

**Peak Memory**:
- Loading full dataset: ~500MB
- LightGBM training: ~1GB
- Total: ~2GB max

**GitHub Actions Limits**:
- Free tier: 7GB RAM available
- Well within limits ✅

---

## Error Handling & Resilience

### Retry Logic

**Web Scraping**:
```python
from selenium.common.exceptions import TimeoutException

max_retries = 3
for attempt in range(max_retries):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "content"))
        )
        break
    except TimeoutException:
        if attempt == max_retries - 1:
            raise
        time.sleep(2 ** attempt)  # Exponential backoff
```

**API Calls**:
```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
```

### Graceful Degradation

**Missing Odds**:
```python
try:
    odds_df = fetch_odds(api_key)
except Exception as e:
    logger.warning(f"Failed to fetch odds: {e}")
    # Continue with predictions only
    odds_df = pd.DataFrame()  # Empty
```

**Missing Games**:
```python
if games_df.empty:
    logger.info("No games scheduled for today")
    # Write empty output
    games_df.to_csv(output_path)
    sys.exit(0)  # Success, just no games
```

### Logging

**Levels**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.debug("Detailed info for debugging")
logger.info("General information")
logger.warning("Something unexpected but not critical")
logger.error("An error occurred")
logger.critical("Critical failure")
```

**Output**:
- Console: GitHub Actions logs
- Future: File-based logging

---

## Future Architecture

### Planned Improvements

**1. Database Migration**
```
CSV Files → SQLite/PostgreSQL
Benefits:
- Faster queries
- Better data integrity
- Concurrent access
- Simpler schema evolution
```

**2. Configuration Management**
```
Hardcoded Values → config.yaml
Benefits:
- Centralized configuration
- Environment-specific settings
- Easier maintenance
```

**3. Logging Infrastructure**
```
Console Only → File + Rotation
Benefits:
- Persistent logs
- Better debugging
- Audit trail
```

**4. Model Versioning**
```
Single Model → MLflow Tracking
Benefits:
- A/B testing
- Performance comparison
- Experiment tracking
```

**5. Real-time Pipeline**
```
Batch (Daily) → Streaming (Live)
Benefits:
- Live game updates
- In-play betting
- Real-time odds monitoring
```

**6. Web Dashboard**
```
CSV Files → Web UI (Flask/Streamlit)
Benefits:
- Interactive visualizations
- User-friendly interface
- Mobile access
```

### Scalability Roadmap

**Phase 1: Current** (✅ Complete)
- CSV-based pipeline
- Daily batch processing
- Manual review

**Phase 2: Enhanced** (In Progress)
- Add logging
- Add configuration
- Add data validation

**Phase 3: Production** (Future)
- Database migration
- Model versioning
- API endpoints

**Phase 4: Enterprise** (Future)
- Real-time streaming
- Web dashboard
- Multi-user support

---

## Conclusion

The Basketball Prediction System uses a **pipeline architecture** for simplicity and reliability. While not the most performant approach, it's easy to understand, maintain, and extend.

**Strengths**:
- Simple, clear data flow
- Easy to debug
- Independent stages
- Well-tested critical functions

**Areas for Improvement**:
- Centralized configuration
- Database storage
- Enhanced logging
- Real-time capabilities

The current architecture is **production-ready for batch processing** but has clear paths for future enhancement as requirements evolve.

---

**Last Updated**: 2025-11-15
**Version**: 1.0
**Status**: Production Ready (Batch Pipeline)
