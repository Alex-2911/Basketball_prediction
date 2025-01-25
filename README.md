# Basketball_prediction
Automates basketball game data scraping, parsing, and predictions using basketball-reference.com. Integrates Selenium/BeautifulSoup for web scraping, calculates rolling averages, and applies LightGBM for outcome forecasts. Features robust error handling, dynamic season logic, efficient dataset updates, and scalable design for advanced sports analytics.

## Execution Workflow

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

Important:
Execute Script 1 first to ensure all previous game data is processed.
Then, run Script 2 to collect data for the upcoming game day.
Finally, execute Script 3 to calculate predictions based on the processed data.

