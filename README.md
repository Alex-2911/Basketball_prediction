Basketball_prediction
Automates basketball data scraping, parsing, predictions, and betting stats from basketball-reference.com. Integrates Selenium/BeautifulSoup for scraping, calculates rolling averages, uses LightGBM for outcome forecasts, and merges predictions with actual results to evaluate accuracy. Offers robust error handling, dynamic seasons, efficient updates, and scalable analytics.

Execution Workflow
Script 1: _1. 03010225_get_data_previous_game_day_and_parse.ipynb

Gathers and updates the dataset with data from the previous game day.
Saves the updated dataset to the specified directory.
Script 2: _2. 03010225_get_data_next_game_day.ipynb

Collects game data for the upcoming game day(s).
Saves the data to the next game folder and updates the dataset.
Script 3: _3. 03010225_lightgbm.ipynb

Computes predictions for the upcoming game day using historical data, rolling averages, and machine learning models.
Outputs predictions with probabilities and saves them to the prediction directory.
Script 4: _4. 03012025_calculate_betting_statistics.ipynb

Merges actual outcomes with predicted results to evaluate betting performance.
Calculates overall and subset accuracy (e.g., home-team-favored vs. away-team-favored) and updates a combined CSV.
Script 5: _5. 03012025_grid_search_for_best_betting_parameters.ipynb

Performs a grid search to find optimal betting parameters (home win rates, odds thresholds, etc.).
Displays and saves a summary of the best parameters and filters today’s games to highlight top home teams.
Important
Run Script 1 first to ensure all previous game data is processed.
Then run Script 2 to collect data for the upcoming game day.
Execute Script 3 to generate predictions using the updated data.
Run Script 4 to evaluate prediction accuracy and betting statistics.
Finally, run Script 5 to fine-tune betting parameters, generate a summary, and highlight key matchups.
