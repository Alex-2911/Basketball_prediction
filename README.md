# Basketball Prediction System

[![Tests](https://github.com/razi90/Basketball_prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/razi90/Basketball_prediction/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/razi90/Basketball_prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/razi90/Basketball_prediction)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 262 passing](https://img.shields.io/badge/tests-262%20passing-brightgreen.svg)](https://github.com/razi90/Basketball_prediction)

**End-to-End NBA Betting Prediction & Analytics Platform (2025-26 Season)**

Automates **NBA data scraping, machine learning predictions, and betting analytics** with production-grade infrastructure:
- **Web Scraping**: Selenium + BeautifulSoup for real-time game data
- **Machine Learning**: LightGBM with probability calibration (Platt + Isotonic)
- **Betting Strategy**: Kelly Criterion optimal stake sizing
- **Error Handling**: Comprehensive logging, retries, and graceful fallbacks
- **Storage**: Dual-mode (CSV + optional PostgreSQL/Supabase)
- **Testing**: 262 unit tests with 100% pass rate (pytest)
- **CI/CD**: Automated testing & quality checks via GitHub Actions

---

## Execution Workflow

### 1. Script 1: Get Previous Game Day Data
**File**: `2026/src/1_get_data_previous_game_day_2026.py`
- **Purpose**: Scrapes completed NBA games from Basketball-Reference.com
- **Output**:
  - CSV: `nba_games_<date>.csv` (game statistics)
  - Database: Automatic save to PostgreSQL (if enabled)
- **Features**: Retry logic, error handling, data validation

### 2. Script 2: Get Next Game Day Schedule
**File**: `2026/src/2_get_data_next_game_day_2026.py`
- **Purpose**: Scrapes upcoming game schedule
- **Output**:
  - CSV: `games_df_<date>.csv` (upcoming matchups)
  - Database: Game schedule saved to DB (if enabled)
- **Features**: Month-by-month scraping, automatic season detection

### 3. Script 3: Predict Games (Hybrid Model)
**File**: `2026/src/3_predict_games_hybrid_2026.py`
- **Purpose**: Generates win probability predictions using LightGBM
- **Output**:
  - CSV: `nba_games_predict_<date>.csv` (predictions + odds)
  - Database: Predictions with model version tracking (if enabled)
- **Features**:
  - Rolling 9-game averages
  - Live odds integration (The Odds API)
  - Probability calibration (Platt + Isotonic)

### 4. Script 4: Calculate Betting Statistics
**File**: `2026/src/4_calculate_betting_statistics_2026.py`
- **Purpose**: Tracks prediction accuracy and betting performance
- **Output**: CSV: `combined_nba_predictions_acc_<date>.csv`
- **Features**: Accuracy by team, home/away splits, confidence levels

### 5. Script 5: Kelly Betting Parameters
**File**: `2026/src/5_kelly_betting_parameters_2026.py`
- **Purpose**: Calculates optimal bet sizes using Kelly Criterion
- **Output**:
  - CSV: `combined_nba_predictions_enriched_<date>.csv`
  - Database: Enriched predictions with stakes and PnL (if enabled)
  - Visualization: Bankroll simulation charts
- **Features**:
  - Half-Kelly conservative sizing
  - Stake caps (30% max, €300 absolute)
  - Three strategies (Raw, Platt, Isotonic)

### 6. Script 6: Display Proposed Bets
**File**: `2026/src/6_proposed_bets_2026.py`
- **Purpose**: Shows today's recommended bets
- **Output**: Console display of filtered betting opportunities
- **Features**: Database-first loading with CSV fallback

---

## Quick Start

### Prerequisites

- Python 3.12
- Google Chrome (for Selenium web scraping)
- The Odds API key (free tier: https://the-odds-api.com/)
- Optional: Supabase account (for database storage)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Basketball_prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your ODDS_API_KEY
```

4. **Install the CLI** (makes `nba-predict` command available)
```bash
pip install -e .
```

5. **Optional: Set up database** (see [DATABASE_SETUP.md](docs/DATABASE_SETUP.md))
```bash
# In .env, add:
USE_DATABASE=true
DATABASE_URL=postgresql://postgres:[password]@db.yourproject.supabase.co:5432/postgres
```

### ⚡ Using the CLI (Recommended)

The modern CLI provides a clean interface to all functionality:

**Run the complete pipeline:**
```bash
nba-predict pipeline
```

**Or run individual steps:**
```bash
# Collect data
nba-predict collect historical
nba-predict collect upcoming

# Generate predictions
nba-predict predict

# Run analysis
nba-predict analyze stats
nba-predict analyze kelly
nba-predict analyze recommend

# Or run all analysis at once
nba-predict analyze all

# Launch dashboard
nba-predict dashboard
```

**Advanced usage:**
```bash
# Skip data collection (use existing data)
nba-predict pipeline --skip-collection

# Skip analysis (only predict)
nba-predict pipeline --skip-analysis

# Collect specific date
nba-predict collect historical --date 2025-11-15

# Get help
nba-predict --help
nba-predict collect --help
```

### 🐳 Docker Setup (Alternative)

![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED.svg?logo=docker)

**Run everything in containers with one command:**

```bash
# Quick start - Dashboard only
docker-compose up dashboard

# Full stack - Dashboard + Database
docker-compose up -d

# Run prediction scripts
docker-compose run --rm app nba-predict predict
```

**Benefits:**
- ✅ **No dependency conflicts** - Isolated environment
- ✅ **Works everywhere** - Linux, Mac, Windows
- ✅ **Includes Chrome/Selenium** - Pre-configured for web scraping
- ✅ **Database included** - Optional PostgreSQL container
- ✅ **One-command setup** - No manual installs

See **[DOCKER.md](DOCKER.md)** for complete Docker setup guide.

### Alternative: Running Scripts Directly

Scripts can also be run directly:

```bash
# Collect historical game data
python 2026/src/collect_historical_games.py

# Collect upcoming game schedule
python 2026/src/collect_upcoming_games.py

# Generate predictions
python 2026/src/generate_predictions.py

# Calculate betting statistics
python 2026/src/calculate_betting_statistics.py

# Calculate Kelly Criterion parameters
python 2026/src/calculate_kelly_parameters.py

# View betting recommendations
python 2026/src/show_bet_recommendations.py
```

**Note:** The CLI interface (`nba-predict`) is recommended for better UX and error handling.

---

## Key Features

### 🗄️ Dual-Mode Storage

The system supports **both CSV and database storage** simultaneously:

**CSV Mode** (Default):
- Works out of the box with no setup
- Portable, version-controllable
- Perfect for development

**Database Mode** (Optional):
- Production-grade PostgreSQL/Supabase
- Advanced querying and analytics
- Automatic audit trails
- Foreign key constraints
- Concurrent access support

Enable database: `USE_DATABASE=true` in `.env`

See **[DATABASE_SETUP.md](docs/DATABASE_SETUP.md)** for complete setup guide.

### 📊 Interactive Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.17.0-3F4F75.svg)

Launch the **interactive web dashboard** to visualize predictions and analytics:

```bash
streamlit run dashboard/app.py
```

**Dashboard Features**:
- **📊 Today's Games**: Live predictions with win probabilities and betting recommendations
- **📈 Performance Analytics**: Model accuracy, calibration plots, ROI trends
- **🏀 Team Analytics**: Team-specific performance, home/away splits, matchup analysis
- **💰 Betting History**: Complete betting history with profit/loss tracking

**Key Visualizations**:
- Win probability gauges and confidence indicators
- Calibration plots (predicted vs actual outcomes)
- Cumulative profit curves over time
- Performance breakdown by confidence level
- Interactive filtering and data export

See **[dashboard/README.md](dashboard/README.md)** for complete dashboard documentation.

### 🧪 Testing

![Tests Passing](https://img.shields.io/badge/tests-262%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-check%20badge%20above-blue.svg)

Run the test suite:
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=2026/src --cov-report=html

# Database tests (requires USE_DATABASE=true)
pytest tests/test_database_integration.py -v

# Run specific test modules
pytest tests/test_error_handlers.py -v
pytest tests/test_migration_scripts.py -v
```

**Test Coverage** (262 tests, 100% pass rate):
- ✅ **Betting utilities** (38 tests) - Kelly Criterion, odds conversion, stake sizing
- ✅ **Data processing** (24 tests) - Rolling averages, preprocessing, target creation
- ✅ **Team normalization** (60+ tests) - All team code mappings
- ✅ **Database integration** (25 tests) - CRUD operations, graceful fallback
- ✅ **Error handling** (38 tests) - Retry logic, validation, context managers
- ✅ **Migration scripts** (25 tests) - CSV to PostgreSQL migration
- ✅ **Logger infrastructure** (25 tests) - Logging configuration and output

**CI/CD**: Automated testing on every push via GitHub Actions

### 📚 Documentation

- **[PROJECT.md](docs/PROJECT.md)** - Complete system overview and architecture
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed technical documentation
- **[DATABASE_SETUP.md](docs/DATABASE_SETUP.md)** - Supabase/PostgreSQL setup guide
- **[ERROR_HANDLING.md](docs/ERROR_HANDLING.md)** - Error handling infrastructure
- **[database/README.md](database/README.md)** - Database migration quick reference

### 🤖 Automation

GitHub Actions workflows run daily at 06:00 UTC:
- Scrape previous day's games
- Generate predictions for today
- Calculate betting statistics
- Upload results as artifacts

### 🔒 Error Handling

Comprehensive error handling throughout:
- Automatic retries with exponential backoff
- Graceful fallbacks (database → CSV)
- Detailed logging with context
- Input validation and data checks
- Network failure recovery

---

## Database Migration (Optional)

To migrate existing CSV data to PostgreSQL:

```bash
# 1. Set up Supabase and apply schema
# See docs/DATABASE_SETUP.md for detailed instructions

# 2. Test migration (dry run)
python database/scripts/migrate_game_statistics.py --dry-run

# 3. Run migrations
python database/scripts/migrate_game_statistics.py
python database/scripts/migrate_predictions.py
python database/scripts/migrate_enriched_predictions.py
```

**Migration Features**:
- `--dry-run` preview without inserting data
- `--batch-size` control memory usage
- `--latest-only` migrate only recent files
- Automatic deduplication
- Comprehensive logging

---

## Project Structure

```
Basketball_prediction/
├── 2026/
│   ├── src/                      # Main Python scripts (1-6)
│   ├── data/                     # CSV outputs
│   └── output/                   # Predictions and statistics
├── database/
│   ├── schemas/                  # PostgreSQL schema files
│   └── scripts/                  # Migration scripts
├── docs/                         # Documentation
├── tests/                        # Unit tests (pytest)
├── .github/workflows/            # CI/CD automation
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Technology Stack

- **Python 3.12** - Core language
- **LightGBM** - Machine learning
- **Selenium + BeautifulSoup** - Web scraping
- **pandas + numpy** - Data processing
- **PostgreSQL/Supabase** - Optional database (production)
- **pytest** - Testing framework
- **GitHub Actions** - Automation
- **The Odds API** - Live betting odds

---

## Contributing

See [PROJECT.md](docs/PROJECT.md) for detailed contribution guidelines and architecture documentation.

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Documentation**: See `/docs` directory
- **Issues**: GitHub Issues
- **Database Setup**: [DATABASE_SETUP.md](docs/DATABASE_SETUP.md)
- **API Key**: https://the-odds-api.com/
