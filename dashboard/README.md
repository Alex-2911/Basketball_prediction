# 📊 NBA Prediction Dashboard

Interactive Streamlit dashboard for visualizing NBA predictions, model performance, and betting analytics.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **Streamlit** - Interactive web dashboard framework
- **Plotly** - Interactive charts and visualizations

### 2. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📱 Features

### 🏠 Home Page
- System status overview
- Quick metrics (tests, model info, strategy)
- Quick links to documentation

### 📊 Today's Games
- View upcoming games with predictions
- Win probability for each matchup
- Betting recommendations with Kelly Criterion stakes
- Market odds comparison (if available)
- Filter by confidence level
- Download predictions as CSV

### 📈 Performance Analytics
- Overall model accuracy metrics
- Brier score and log loss
- Calibration plot (predicted vs actual probabilities)
- Performance by confidence level
- Accuracy trends over time
- 7-day rolling average accuracy

### 🏀 Team Analytics
- Team-specific performance analysis
- Home vs away splits
- Performance trends over time
- Best and worst matchups
- Model confidence calibration by team
- Recent games history

### 💰 Betting History
- Complete betting history with results
- Cumulative profit/loss tracking
- ROI analysis
- Performance by stake size
- Best and worst bets
- Kelly Criterion explanation

## 📁 Dashboard Structure

```
dashboard/
├── app.py                      # Main application (home page)
├── pages/
│   ├── 1_📊_Today's_Games.py   # Today's predictions
│   ├── 2_📈_Performance.py     # Model performance
│   ├── 3_🏀_Team_Analytics.py  # Team-specific analysis
│   └── 4_💰_Betting_History.py # Betting history & ROI
├── components/
│   └── charts.py               # Reusable chart components
├── utils/
│   └── data_loader.py          # Data loading utilities
└── README.md                   # This file
```

## 📊 Data Requirements

The dashboard loads data from:

### Prediction Files
Located in `2026/output/`:
- `nba_games_predict_*.csv` - Base predictions with win probabilities
- `combined_nba_predictions_enrich_*.csv` - Enriched with betting recommendations
- `combined_nba_predictions_acc_*.csv` - Accuracy statistics

### Game Data
Located in `2026/data/`:
- `nba_games_*.csv` - Historical game results

### Generating Data

Run the prediction scripts to generate data:

```bash
cd 2026/src

# 1. Get previous game data
python 1_get_data_previous_game_day_2026.py

# 2. Get upcoming games
python 2_get_data_next_game_day_2026.py

# 3. Generate predictions
python 3_predict_games_hybrid_2026.py

# 4. Calculate statistics
python 4_calculate_betting_statistics_2026.py

# 5. Enrich with betting recommendations
python 5_enrich_with_betting_recs_2026.py

# 6. Display suggestions (console output)
python 6_display_betting_suggestions_2026.py
```

## 🎨 Customization

### Changing Theme

Edit `.streamlit/config.toml` to customize colors and theme:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Adding New Pages

Create a new file in `dashboard/pages/`:

```python
# pages/5_🆕_New_Page.py

import streamlit as st

st.set_page_config(page_title="New Page", page_icon="🆕", layout="wide")

st.title("🆕 My New Page")
# Your content here
```

Streamlit automatically adds it to the sidebar!

### Creating Custom Charts

Add new chart functions to `components/charts.py`:

```python
def create_my_chart(df: pd.DataFrame) -> go.Figure:
    """Create a custom visualization."""
    fig = go.Figure()
    # Your Plotly code here
    return fig
```

## 🔧 Troubleshooting

### Dashboard Won't Start
```bash
# Check if Streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install --upgrade streamlit plotly
```

### No Data Showing
```bash
# Verify prediction files exist
ls -la 2026/output/nba_games_predict_*.csv

# If missing, run prediction scripts
cd 2026/src && python 3_predict_games_hybrid_2026.py
```

### Port Already in Use
```bash
# Run on a different port
streamlit run dashboard/app.py --server.port 8502
```

### Permission Errors
```bash
# Make sure you have read permissions
chmod +r 2026/output/*.csv
chmod +r 2026/data/*.csv
```

## 📖 Tips & Tricks

### Keyboard Shortcuts
- **`R`** - Rerun the app (refresh data)
- **`C`** - Clear cache
- **`?`** - Show keyboard shortcuts

### Performance
- Dashboard auto-refreshes when files change
- Use caching for large datasets (already implemented)
- Filter data to reduce load times

### Sharing
- Run with `--server.address 0.0.0.0` to allow network access
- Deploy to Streamlit Cloud for public access (free!)
- Export charts as PNG using Plotly's built-in download

## 🌐 Deployment (Optional)

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Deploy!

### Docker
```bash
# Build image
docker build -t nba-dashboard .

# Run container
docker run -p 8501:8501 nba-dashboard
```

### Heroku
```bash
# Create Procfile
echo "web: streamlit run dashboard/app.py --server.port=$PORT" > Procfile

# Deploy
heroku create
git push heroku main
```

## 🤝 Contributing

To add new features:
1. Create a new page in `pages/`
2. Add new charts to `components/charts.py`
3. Add new data loaders to `utils/data_loader.py`
4. Test locally before committing

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python/
- **Project Docs**: See `/docs` directory

## 🐛 Known Issues

- Large datasets (>100 days) may slow down performance
- Real-time updates require manual refresh
- Mobile view is optimized but desktop recommended

## 📝 License

Same license as the main project (MIT)
