# Basketball Prediction Setup Guide

This guide will help you set up the Basketball prediction project for both local development and automated GitHub Actions workflows.

## 🔐 Security Notice

**IMPORTANT**: The previous API key that was committed to the repository has been removed. If you were using this project before, you need to:

1. **Rotate your API key** at [The Odds API](https://the-odds-api.com/)
2. **Never commit the `.env` file** - it's now in `.gitignore`

## 📋 Prerequisites

- Python 3.11 or higher
- Git
- Google Chrome (for Selenium web scraping)
- API key from [The Odds API](https://the-odds-api.com/)

## 🚀 Local Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Basketball_prediction
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
ODDS_API_KEY=your_actual_api_key_here
```

**Get your free API key:**
1. Visit [https://the-odds-api.com/](https://the-odds-api.com/)
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Paste it into the `.env` file

### 5. Run the Scripts

Execute the scripts in order:

```bash
# Step 1: Collect previous game data
python 2026/src/1_get_data_previous_game_day_2026.py

# Step 2: Get next game schedule
python 2026/src/2_get_data_next_game_day_2026.py

# Step 3: Generate predictions
python 2026/src/3_predict_games_hybrid_2026.py

# Step 4: Calculate statistics (optional - requires completed games)
python 2026/src/4_calculate_betting_statistics_2026.py

# Step 5: Calculate Kelly parameters
python 2026/src/5_kelly_betting_parameters_2026.py

# Step 6: View recommended bets
python 2026/src/6_proposed_bets_2026.py
```

## 🤖 GitHub Actions Setup (Automated Daily Runs)

### 1. Add API Key as GitHub Secret

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `ODDS_API_KEY`
5. Value: Your API key from The Odds API
6. Click **Add secret**

### 2. Enable GitHub Actions

The workflows are already configured in `.github/workflows/`:

- **`daily_prediction_pipeline.yml`** - Complete pipeline (all 6 scripts)
  - Runs daily at 06:00 UTC
  - Can be triggered manually

- **`1_get_data_previous_game_day 2026.yml`** - Legacy workflow (script 1 only)
  - Still available but superseded by complete pipeline

### 3. Manual Trigger

To run the pipeline manually:

1. Go to **Actions** tab in GitHub
2. Select "🏀 Complete Daily NBA Prediction Pipeline"
3. Click **Run workflow**
4. Select branch and click **Run workflow**

### 4. View Results

After the workflow completes:

- **Artifacts**: Download CSV files from the workflow run page
- **Committed Data**: Check the `2026/output/` directory in the repository
- **Logs**: View step-by-step execution logs in Actions tab

## 📁 Project Structure

```
Basketball_prediction/
├── 2026/
│   ├── src/                          # Python scripts
│   │   ├── 1_get_data_previous_game_day_2026.py
│   │   ├── 2_get_data_next_game_day_2026.py
│   │   ├── 3_predict_games_hybrid_2026.py
│   │   ├── 4_calculate_betting_statistics_2026.py
│   │   ├── 5_kelly_betting_parameters_2026.py
│   │   ├── 6_proposed_bets_2026.py
│   │   └── nba_utils_2026.py
│   └── output/                       # Generated data
│       ├── Gathering_Data/
│       │   ├── Next_Game/
│       │   └── Whole_Statistic/
│       └── LightGBM/
├── .github/workflows/                # GitHub Actions
├── .env.example                      # Template for environment variables
├── .env                              # Your secrets (DO NOT COMMIT)
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── SETUP.md                          # This file
└── README.md                         # Project documentation
```

## 🔧 Troubleshooting

### Script 3 fails with "ODDS_API_KEY not found"

**Solution**: Make sure you created the `.env` file with your API key.

```bash
# Check if .env exists
ls -la .env

# Verify contents (should show ODDS_API_KEY=...)
cat .env
```

### GitHub Actions fails with API key error

**Solution**: Verify the secret is properly set:

1. Go to Settings → Secrets → Actions
2. Confirm `ODDS_API_KEY` is listed
3. Update the secret if needed

### Chrome/Selenium errors

**Local**: Install Google Chrome browser
**GitHub Actions**: Chrome is automatically installed in the workflow

### Path errors on Windows

The project now uses cross-platform paths (fixed from previous hardcoded Windows paths). If you still see path issues, ensure you're using forward slashes `/` or `pathlib.Path`.

## ⚠️ Disclaimer

This tool is for educational purposes only. Sports betting involves financial risk. Past performance does not guarantee future results. Always bet responsibly and within your means.

## 📞 Support

For issues or questions:
- Check the [Issues](../../issues) page
- Review the main [README.md](README.md)
- Consult the script comments and docstrings
