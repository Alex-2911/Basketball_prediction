# Basketball Prediction

Automates **basketball data scraping, parsing, predictions, and betting statistics** from [basketball-reference.com](https://www.basketball-reference.com).  
Integrates **Selenium/BeautifulSoup** for web scraping, calculates rolling averages, applies **LightGBM** for outcome forecasts, and merges predictions with actual results to evaluate accuracy. Features **robust error handling, dynamic seasons, efficient updates,** and **scalable analytics**.

---

## Execution Workflow

### 1. Script 1: `_1. 03010225_get_data_previous_game_day_and_parse.ipynb`
- **Purpose**: Gathers and updates the dataset with data from the previous game day.  
- **Output**: Saves an updated dataset in the specified directory.

### 2. Script 2: `_2. 03010225_get_data_next_game_day.ipynb`
- **Purpose**: Collects game data for upcoming game day(s).  
- **Output**: Saves the data to the “next game” folder and updates the dataset.

### 3. Script 3: `_3. 03010225_lightgbm.ipynb`
- **Purpose**: Computes predictions for the upcoming game day using historical data, rolling averages, and LightGBM.  
- **Output**: Generates predictions with probabilities, saved to the “prediction” directory.

### 4. Script 4: `_4. 03012025_calculate_betting_statistics.ipynb`
- **Purpose**: Merges actual outcomes with predicted results to evaluate betting performance.  
- **Output**: Calculates overall and subset accuracies (e.g., home-favored vs. away-favored), updating a combined CSV.

### 5. Script 5: `_5. 03012025_grid_search_for_best_betting_parameters.ipynb`
- **Purpose**: Performs a grid search to find optimal betting parameters (home win rates, odds thresholds, etc.).  
- **Output**: Displays and saves a summary of the best parameters and filters today’s games to highlight top home teams.

---

## Important

1. Run **Script 1** first to ensure all previous game data is processed.  
2. Then **Script 2** to gather data for upcoming game days.  
3. Execute **Script 3** to calculate predictions using the updated data.  
4. Next, **Script 4** to generate betting statistics and assess accuracy.  
5. Finally, **Script 5** to optimize betting parameters and create a final summary.

---

## Installation

Install all required libraries using:
```bash
pip install -r requirements.txt
