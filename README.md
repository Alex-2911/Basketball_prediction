# Basketball_prediction
Automates basketball game data scraping, parsing, and processing from basketball-reference.com. Updates datasets with new stats, handles dynamic season logic, and manages files efficiently. Features include Selenium/BeautifulSoup integration, advanced stats aggregation, error logging, and scalability for sports data analysis.

## Execution Workflow

This project consists of two scripts that must be executed sequentially:

1. **Script 1**: `_1. 03012025_get_data_previous_game_day and parse to statistics.ipynb`
   - Gathers and updates the dataset with data from the previous game day.
   - Saves the updated dataset to the specified directory.

2. **Script 2**: `_2. 03012025_get_data_next_game_day.ipynb`
   - Collects game data for the upcoming game day(s).
   - Saves the data to the next game folder and updates the dataset.

**Important**: Run _1. 03012025_get_data_previous_game_day and parse to statistics.ipynb` first to ensure all previous game data is processed before running `02_collec_2. 03012025_get_data_next_game_day.ipynb`.

