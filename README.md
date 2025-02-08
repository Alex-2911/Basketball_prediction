# Basketball_prediction
Automates basketball data scraping, parsing, predictions, betting stats from basketball-reference.com. Integrates Selenium/BeautifulSoup for scraping, calculates rolling averages, uses LightGBM for outcome forecasts, merges predictions with actual results to evaluate accuracy. Offers robust error handling, dynamic seasons, efficient updates, and scalable analytics.

## Execution Workflow

1. **Script 1**: `_1. 03012025_get_data_previous_game_day and parse to statistics.ipynb`  
   - Gathers and updates the dataset with data from the previous game day.  
   - Saves the updated dataset to the specified directory.

2. **Script 2**: `_2. 03012025_get_data_next_game_day.ipynb`  
   - Collects game data for the upcoming game day(s).  
   - Saves the data to the next game folder and updates the dataset.

3. **Script 3**: `_3. 03012025_lightgbm.ipynb`  
   - Computes predictions for the upcoming game day using historical data, rolling averages, and machine learning models.  
   - Outputs predictions with probabilities and saves them to the prediction directory.

4. **Script 4**: `_4_03012025_calculate_betting_statistics.ipynb`  
   - Merges actual outcomes with predicted results to evaluate betting performance.  
   - Calculates overall and subset accuracy (e.g., home-team-favored vs. away-team-favored) and updates a combined CSV.

### Important:
- Run **Script 1** first to ensure all previous game data is processed.  
- Then, run **Script 2** to collect data for the upcoming game day.  
- Execute **Script 3** to generate predictions using the updated data.  
- Finally, run **Script 4** to evaluate prediction accuracy and betting statistics.
