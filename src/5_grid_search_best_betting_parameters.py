#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 5 of 5: Grid Search for Best Betting Parameters

This script performs a grid search to find optimal betting parameters, 
displays and saves the results (including a summary in Excel),
and then filters today's games to highlight top home teams.

Ensure "4_calculate_betting_statistics.py" is executed before running this script.
"""

import pandas as pd
import os
import glob
import numpy as np
import logging
from datetime import datetime, timedelta
from itertools import product
import xlsxwriter

# Configure date and paths
today = datetime.now() - timedelta(days=0)
today_str = today.strftime("%Y-%m-%d")
print(today_str)

directory_path = r'D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction'
print(directory_path)

target_folder = r'D:\1. Python\1. NBA Script\2025\Gathering_Data\Next_Game'
print(target_folder)

# Create output directory if it doesn't exist
os.makedirs(directory_path, exist_ok=True)
summary_output_file = os.path.join(directory_path, f"grid_search_summary_{today_str}.csv")

games_file = f"games_df_{today_str}.csv"  # Today's file name

def get_last_20_games_all_teams(df):
    """
    Calculate home win rates for all teams based on their last 20 games.
    
    Args:
        df (DataFrame): DataFrame containing game results
        
    Returns:
        DataFrame: DataFrame with team home win rates, sorted in descending order
    """
    team_results = {}

    for team in df['home_team'].unique():
        # Get last 20 games for the team (home or away)
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)]
        team_games = team_games.sort_values(by='date', ascending=False).head(20)

        # Filter only home games from those 20
        home_games = team_games[team_games['home_team'] == team]

        # Calculate home win rate
        total_home_games = len(home_games)
        home_wins = len(home_games[home_games['result'] == team])
        home_win_rate = round(home_wins / total_home_games, 2) if total_home_games > 0 else 0

        # Store results in dictionary
        team_results[team] = {
            "Total Last 20 Games": len(team_games),
            "Total Home Games": total_home_games,
            "Home Wins": home_wins,
            "Home Win Rate": home_win_rate
        }

    # Convert to DataFrame
    home_win_rates_df = pd.DataFrame.from_dict(team_results, orient='index')

    # Sort by Home Win Rate in descending order
    home_win_rates_df.sort_values(by="Home Win Rate", ascending=False, inplace=True)

    return home_win_rates_df

def perform_grid_search(df):
    """
    Perform grid search to find optimal betting parameters.
    
    Args:
        df (DataFrame): DataFrame with betting data
        
    Returns:
        tuple: Best parameters, best result, best games, and iteration information
    """
    # Setup grid search parameters
    bet_amount = 100.0
    home_win_rate_thresholds = np.arange(0.55, 0.95, 0.05)
    odds_1_min_values = np.arange(1.05, 1.8, 0.1)
    odds_1_max_values = np.arange(1.5, 2.5, 0.1)
    prob_threshold_values = np.arange(0.45, 0.85, 0.05)
    
    # Calculate win column based on home team matching result
    df['win'] = (df['result'] == df['home_team']).astype(int)
    
    # Initialize variables for tracking best results
    best_result = float('-inf')
    best_params = {}
    best_iteration = 0
    games_best = None
    iteration_count = 0
    
    # Get home win rates for all teams
    home_win_rates_all_teams_sorted = get_last_20_games_all_teams(df)
    
    # Grid Search
    for threshold in home_win_rate_thresholds:
        # Get teams that meet the home win rate threshold
        top_teams = home_win_rates_all_teams_sorted[
            home_win_rates_all_teams_sorted['Home Win Rate'] >= threshold
        ].index.tolist()
        
        # Loop through all parameter combinations
        for odds1_min, odds1_max, prob_threshold in product(odds_1_min_values, odds_1_max_values, prob_threshold_values):
            iteration_count += 1
            
            # Create conditions for filtering games
            cond = (df['odds 1'].between(odds1_min, odds1_max)) & \
                   (df['home_team_prob'] >= prob_threshold) & \
                   (df['home_team'].isin(top_teams))
            
            # Filter games that meet the criteria
            sub_df = df.loc[cond].copy()
            
            # Skip empty sets
            if sub_df.empty:
                continue
                
            # Calculate betting results
            sub_df['bet_result'] = -bet_amount  # Initial loss
            # Add winnings for correct predictions
            sub_df.loc[sub_df['win'] == 1, 'bet_result'] += sub_df.loc[sub_df['win'] == 1, 'odds 1'] * bet_amount
            
            # Calculate total profit
            total = sub_df['bet_result'].sum()
            
            # Check if this is the best result so far
            if total > best_result:
                best_result = total
                best_params = {
                    'home_win_rate_threshold': threshold,
                    'odds_1_min_value': odds1_min,
                    'odds_1_max_value': odds1_max,
                    'prob_threshold': prob_threshold
                }
                best_iteration = iteration_count
                games_best = sub_df.copy()
    
    return best_params, best_result, games_best, best_iteration, home_win_rates_all_teams_sorted

def check_best_home_teams_playing_today(top_team_names, today_str):
    """
    Check if any of the best home teams are playing today.
    
    Args:
        top_team_names (list): List of top team names
        today_str (str): Today's date string
        
    Returns:
        DataFrame: DataFrame with best home games for today, or None if none found
    """
    # Check if the file exists
    file_path = os.path.join(target_folder, games_file)
    
    if os.path.exists(file_path):
        # Load today's games
        today_games_df = pd.read_csv(file_path)
        
        # Convert 'home_team' to string to avoid filter issues
        today_games_df['home_team'] = today_games_df['home_team'].astype(str)
        
        # Filter for games where the home team is in the best teams list
        best_home_games_today = today_games_df[today_games_df['home_team'].isin(top_team_names)]
        
        if not best_home_games_today.empty:
            print("\n✅ Best Home Teams Are Playing Today!\n")
            print(best_home_games_today)
            
            # Save filtered data
            output_file = f"best_home_games_2024_{today_str}.csv"
            best_home_games_today.to_csv(os.path.join(directory_path, output_file), index=False)
            print(f"\n✅ Data saved to: {os.path.join(directory_path, output_file)}")
            
            return best_home_games_today
        else:
            print("\n⚠ No Best Home Teams are playing today.")
            return pd.DataFrame()  # Return empty DataFrame
    else:
        print(f"\n⚠ No game data found for {today_str}. Check if the file exists: {file_path}")
        return pd.DataFrame()  # Return empty DataFrame

def create_excel_summary(df_summary, best_home_games_today, today_str):
    """
    Create an Excel summary file with multiple tables.
    
    Args:
        df_summary (DataFrame): DataFrame with summary metrics
        best_home_games_today (DataFrame): DataFrame with best home games
        today_str (str): Today's date string
    """
    # Build the Excel file path for today's date
    excel_file = os.path.join(directory_path, f"betting_summary_{today_str}.xlsx")
    
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # 1) Write df_summary to "Summary" sheet
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        
        workbook = writer.book
        summary_worksheet = writer.sheets["Summary"]
        
        # Column widths for the summary table
        summary_worksheet.set_column("A:A", 40)  # Metric
        summary_worksheet.set_column("B:B", 20)  # Value
        
        # Header format for the first table
        header_format_1 = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "fg_color": "#D7E4BC",
            "border": 1
        })
        
        # Apply it to df_summary headers (row=0)
        for col_num, col_name in enumerate(df_summary.columns):
            summary_worksheet.write(0, col_num, col_name, header_format_1)
        
        # If best home games data is available, add it below summary
        if not best_home_games_today.empty:
            # 2) Append best_home_games_today below df_summary (+1 blank row)
            blank_rows = 1
            startrow = df_summary.shape[0] + blank_rows + 1
            
            best_home_games_today.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
                startrow=startrow
            )
            
            # Format columns for the second table
            summary_worksheet.set_column("A:C", 15)
            
            # Second header format
            header_format_2 = workbook.add_format({
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "fg_color": "#FFE699",
                "border": 1
            })
            
            # Apply header format to best_home_games_today columns
            for col_num, col_name in enumerate(best_home_games_today.columns):
                summary_worksheet.write(startrow, col_num, col_name, header_format_2)
    
    print(f"✅ Formatted summary Excel file saved to: {excel_file}")

def main():
    """Main execution function."""
    try:
        # Load the dataset from the combined predictions file
        read_file_path = os.path.join(directory_path, f'combined_nba_predictions_acc_{today_str}.csv')
        df = pd.read_csv(read_file_path)
        
        # Ensure the date column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Get last 20 games and home win rates for all teams
        home_win_rates_all_teams_sorted = get_last_20_games_all_teams(df)
        
        # Display sorted results
        print("\n🏀 Home Win Rates (Sorted) for All Teams:")
        print(home_win_rates_all_teams_sorted)
        
        # Save to CSV (Optional)
        output_file = os.path.join(directory_path, f'home_win_rates_sorted_{today_str}.csv')
        home_win_rates_all_teams_sorted.to_csv(output_file, index=True)
        print(f"\n📁 Sorted home win rates saved to: {output_file}")
        
        # Perform grid search to find optimal parameters
        best_params, best_result, games_best, best_iteration, _ = perform_grid_search(df)
        
        # --- Display Results ---
        print("\n=== Best Grid Search Parameters ===")
        for key, value in best_params.items():
            print(f"{key.replace('_',' ').title()}: {round(value, 2)}")
        print(f"Achieved at Iteration: {best_iteration}")
        print("-" * 50)
        
        # Calculate betting stats
        bet_amount = 100.0
        num_games = len(games_best)
        total_bet = bet_amount * num_games
        profit_ratio = (best_result / total_bet * 100) if total_bet else 0
        
        print("\n=== Betting Overview ===")
        print(f"Number of games fulfilling criteria: {num_games}")
        print(f"Total Bet Amount: {total_bet:.2f} euros")
        print(f"Max Profit: {round(best_result, 2)} euros")
        print(f"Profitability Ratio: {profit_ratio:.2f}%")
        print("-" * 50)
        
        win_count = games_best['win'].sum()
        loss_count = num_games - win_count
        win_ratio = (win_count / num_games * 100) if num_games else 0
        
        print("\n=== Performance Summary ===")
        print(f"Total Wins: {int(win_count)}")
        print(f"Total Losses: {int(loss_count)}")
        print(f"Win Ratio: {win_ratio:.2f}%")
        print("-" * 50)
        
        # Convert home win rates to numeric and display teams fulfilling the threshold
        home_win_rates_all_teams_sorted['Home Win Rate'] = pd.to_numeric(
            home_win_rates_all_teams_sorted['Home Win Rate'], errors='coerce'
        ).round(2)
        
        threshold_val = round(best_params['home_win_rate_threshold'], 2)
        teams_criteria = home_win_rates_all_teams_sorted[
            home_win_rates_all_teams_sorted['Home Win Rate'] >= threshold_val
        ]
        
        print("\n=== Teams Fulfilling the Home Win Rate Threshold ===")
        print(teams_criteria[['Home Win Rate']].to_string())
        print("-" * 50)
        
        # Display games fulfilling the best parameters
        games_best['date'] = pd.to_datetime(games_best['date'], errors='coerce')
        games_output = games_best[['home_team', 'away_team', 'home_team_prob', 'odds 1', 'odds 2', 'result', 'bet_result', 'date']].sort_values(by='date', ascending=False)
        
        print("\n=== List of Games Fulfilling the Best Parameters ===")
        print(f"{'Home Team':<10} {'Away Team':<10} {'Home Prob':<12} {'Odds 1':<10} {'Odds 2':<10} {'Result':<8} {'Bet Result':<12} {'Date':<15}")
        print("-" * 85)
        
        for _, row in games_output.iterrows():
            date_str = row['date'].strftime("%Y-%m-%d") if pd.notnull(row['date']) else 'N/A'
            print(f"{row['home_team']:<10} {row['away_team']:<10} {row['home_team_prob']:<12.2f} {row['odds 1']:<10.2f} {row['odds 2']:<10.2f} {row['result']:<8} {row['bet_result']:<12.2f} {date_str:<15}")
        
        print("-" * 85)
        
        # Create summary data for reporting
        summary_data = {
            "Best Home Win Rate Threshold": best_params['home_win_rate_threshold'],
            "Best Odds 1 Min Value": best_params['odds_1_min_value'],
            "Best Odds 1 Max Value": best_params['odds_1_max_value'],
            "Best Home Team Probability Threshold": best_params['prob_threshold'],
            "Achieved at Iteration": best_iteration,
            "Number of Games Fulfilling Criteria": num_games,
            "Total Bet Amount (euros)": total_bet,
            "Max Profit (euros)": best_result,
            "Profitability Ratio (%)": profit_ratio,
            "Total Wins": win_count,
            "Total Losses": loss_count,
            "Win Ratio (%)": win_ratio
        }
        
        # Define formatters for nice display
        formatters = {
            "Best Home Win Rate Threshold": lambda x: f"{x:.2f}",
            "Best Odds 1 Min Value": lambda x: f"{x:.2f}",
            "Best Odds 1 Max Value": lambda x: f"{x:.2f}",
            "Best Home Team Probability Threshold": lambda x: f"{x:.2f}",
            "Achieved at Iteration": lambda x: f"{int(x)}",
            "Number of Games Fulfilling Criteria": lambda x: f"{int(x)}",
            "Total Bet Amount (euros)": lambda x: f"{x:.2f}",
            "Max Profit (euros)": lambda x: f"{x:.2f}",
            "Profitability Ratio (%)": lambda x: f"{x:.2f}",
            "Total Wins": lambda x: f"{int(x)}",
            "Total Losses": lambda x: f"{int(x)}",
            "Win Ratio (%)": lambda x: f"{x:.2f}"
        }
        
        # Apply formatting to each value
        formatted_summary = {metric: formatters[metric](value) for metric, value in summary_data.items()}
        
        # Create a two-column DataFrame ("Metric" and "Value")
        df_summary = pd.DataFrame(formatted_summary.items(), columns=['Metric', 'Value'])
        
        # Get the list of teams meeting criteria
        top_team_names = teams_criteria.index.tolist()
        
        # Check if any top teams are playing today
        best_home_games_today = check_best_home_teams_playing_today(top_team_names, today_str)
        
        # Create Excel summary report
        create_excel_summary(df_summary, best_home_games_today, today_str)
        
    except FileNotFoundError:
        print(f"Error: File not found at {read_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
