#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# KELLY CRITERION FOR BEST BETTING PARAMETERS


# In[ ]:


import pandas as pd
import os
import glob
import numpy as np
import logging
from datetime import datetime, timedelta
from itertools import product
import shutil  # Make sure to import shutil

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


# In[ ]:


days_back = 0

today = datetime.now() - timedelta(days=days_back)
today_str = (today).strftime("%Y-%m-%d")
print(today_str)

# yesterday = today - timedelta(days_back+1)
# yesterday_str = (yesterday).strftime("%Y-%m-%d")

yesterday_str = (datetime.now() - timedelta(days=days_back+1)).strftime("%Y-%m-%d")

print(yesterday_str)


# In[ ]:

# Directories
DATA_DIR_PRED = os.path.join("output", "LightGBM")
target_folder = os.path.join(DATA_DIR_PRED,"1. 2025_Prediction")

print(directory_path)
print(target_folder)



# In[ ]:


# Set directory path
read_file_path = os.path.join(directory_path, f'combined_nba_predictions_acc_{today_str}.csv')

# Load the dataset
df = pd.read_csv(read_file_path,encoding="utf-7")

# Ensure the date column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Function to get last 20 games and home win rates for all teams
def get_last_20_games_all_teams(df):
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

# Get last 20 games and home win rates for all teams
home_win_rates_all_teams_sorted = get_last_20_games_all_teams(df)

# Display sorted results
print("\n🏀 Home Win Rates (Sorted) for All Teams:")
print(home_win_rates_all_teams_sorted)

# Save to CSV (Optional)
output_file = os.path.join(directory_path, f'home_win_rates_sorted_{today_str}.csv')
#home_win_rates_all_teams_sorted.to_csv(output_file, index=True)
#print(f"\n📁 Sorted home win rates saved to: {output_file}")


# In[ ]:


# import os
# import pandas as pd
# from datetime import datetime
# from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import IsotonicRegression
# from sklearn.model_selection import train_test_split

# # ───────────────────────────────────────────────────────────
# # CONFIG
# # ───────────────────────────────────────────────────────────
# BASE_DIR      = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
# TODAY_STR     = datetime.now().strftime("%Y-%m-%d")

# INPUT_FILE    = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{TODAY_STR}.csv")
# HWR_FILE      = os.path.join(BASE_DIR, f"home_win_rates_sorted_{TODAY_STR}.csv")
# home_win_cut  = 0.55  # ← 🧱 apply this filter

# # ───────────────────────────────────────────────────────────
# # LOAD & CLEAN
# # ───────────────────────────────────────────────────────────
# df = pd.read_csv(INPUT_FILE)
# df.columns = ['home_team', 'away_team', 'home_team_prob', 'odds_1', 'odds_2', 'result', 'date', 'result_flag']

# df['date'] = pd.to_datetime(df['date'], errors='coerce')
# df = df[df['date'].dt.month == 4]  # Filter for April only

# df['home_team_prob'] = pd.to_numeric(df['home_team_prob'], errors='coerce')
# df['odds_1'] = pd.to_numeric(df['odds_1'], errors='coerce')
# df['result_flag'] = pd.to_numeric(df['result_flag'], errors='coerce')
# df['implied_prob'] = 1 / df['odds_1']

# # ───────────────────────────────────────────────────────────
# # LOAD HOME WIN RATES AND FILTER
# # ───────────────────────────────────────────────────────────
# hwr_df = pd.read_csv(HWR_FILE, index_col=0)
# good_home_teams = hwr_df[hwr_df['Home Win Rate'] >= home_win_cut].index.tolist()
# df = df[df['home_team'].isin(good_home_teams)]

# # ───────────────────────────────────────────────────────────
# # CALIBRATION
# # ───────────────────────────────────────────────────────────
# X = df[['home_team_prob']].values
# y = df['result_flag'].values
# _, Xc, _, yc = train_test_split(X, y, test_size=0.2, random_state=42)

# platt = LogisticRegression().fit(Xc, yc)
# df['prob_platt'] = platt.predict_proba(X)[:, 1]

# iso = IsotonicRegression(out_of_bounds='clip').fit(df['home_team_prob'], df['result_flag'])
# df['prob_iso'] = iso.transform(df['home_team_prob'])

# # ───────────────────────────────────────────────────────────
# # KELLY LOGIC
# # ───────────────────────────────────────────────────────────
# def kelly(p, o, f=0.5):
#     b = o - 1
#     return max((b * p - (1 - p)) / b * f, 0) if b > 0 else 0

# for label in ['raw', 'platt', 'iso']:
#     prob_col = 'home_team_prob' if label == 'raw' else f'prob_{label}'
#     df[f'edge_{label}'] = df[prob_col] - df['implied_prob']
#     df[f'kelly_{label}'] = df.apply(lambda r: kelly(r[prob_col], r['odds_1']), axis=1)

# # ───────────────────────────────────────────────────────────
# # FILTER TO BET-WORTHY GAMES
# # ───────────────────────────────────────────────────────────
# bet_df = df[df['kelly_raw'] > 0].copy()
# bet_df['stake'] = bet_df['kelly_raw'] * 100
# bet_df['profit'] = bet_df.apply(
#     lambda r: r['stake'] * (r['odds_1'] - 1) if r['result_flag'] == 1 else -r['stake'],
#     axis=1
# )

# # ───────────────────────────────────────────────────────────
# # SUMMARY
# # ───────────────────────────────────────────────────────────
# total_games = len(df)
# bet_games = len(bet_df)
# wins = bet_df['result_flag'].sum()
# losses = bet_games - wins
# roi = bet_df['profit'].sum() / bet_df['stake'].sum() if bet_games > 0 else 0

# # ───────────────────────────────────────────────────────────
# # OUTPUT
# # ───────────────────────────────────────────────────────────
# pd.set_option('display.float_format', '{:.4f}'.format)
# print("\n📊 Filtered Bettable Games (home win rate ≥ 0.55):\n")
# print(bet_df[['date', 'home_team', 'away_team', 'home_team_prob', 'odds_1', 'implied_prob',
#               'edge_raw', 'kelly_raw', 'stake', 'profit', 'result_flag']].sort_values('date').to_string(index=False))

# print("\n📈 FILTERED APRIL SUMMARY (home win rate ≥ 0.55):")
# print(f"Total games considered:  {total_games}")
# print(f"Bet-worthy games:        {bet_games}")
# print(f"→ Wins: {wins} | Losses: {losses}")
# print(f"Total profit:           €{bet_df['profit'].sum():.2f}")
# print(f"Total stake:            €{bet_df['stake'].sum():.2f}")
# print(f"ROI:                     {roi:.2%}")


# In[ ]:


# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# # Parameters
# odds_min = 1.2
# odds_max = 2.8
# prob_thresh = 0.40
# home_win_cut = 0.55
# starting_bank = 1000.0
# bet_frac = 0.5
# cap_frac = 0.30
# abs_cap = 300.0

# # Paths (define these before running)


# # Load data
# df = pd.read_csv(INPUT_FILE)
# hwr_df = pd.read_csv(HWR_FILE, index_col=0)
# home_win_rate = hwr_df['Home Win Rate'].to_dict()

# # Date filter and mapping
# df['date'] = pd.to_datetime(df['date'], errors='coerce')
# df = df[df['date'].dt.month.isin([4, 5])]


# df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
# df['date'] = pd.to_datetime(df['date'], errors='coerce')
# # April and May only
# df = df[df['date'].dt.month.isin([4, 5])]

# df['home_win_rate'] = df['home_team'].map(home_win_rate)
# df['result_flag'] = (df['result'] == df['home_team']).astype(int)

# # Platt calibration
# X = df[['home_team_prob']].values
# y = df['result_flag'].values
# _, Xc, _, yc = train_test_split(X, y, test_size=0.2, random_state=42)
# platt = LogisticRegression(solver='lbfgs').fit(Xc, yc)
# df['prob_platt'] = platt.predict_proba(X)[:, 1]

# # Isotonic calibration
# iso = IsotonicRegression(out_of_bounds='clip').fit(df['home_team_prob'], df['result_flag'])
# df['prob_iso'] = iso.transform(df['home_team_prob'])

# # Kelly function
# def kelly(p, o, f):
#     b = o - 1
#     return max(((b * p - (1 - p)) / b) * f, 0) if b > 0 else 0

# # Strategy simulation function
# def simulate(df, prob_col):
#     df_filtered = df[
#         (df['home_win_rate'] >= home_win_cut) &
#         (df['odds_1'] >= odds_min) &
#         (df['odds_1'] <= odds_max) &
#         (df[prob_col] >= prob_thresh)
#     ].copy()

#     bankroll = starting_bank
#     stakes, profits = [], []
#     for _, r in df_filtered.iterrows():
#         p, o = r[prob_col], r['odds_1']
#         kf = kelly(p, o, bet_frac)
#         stake = min(kf * bankroll, cap_frac * bankroll, abs_cap)
#         profit = stake * (o - 1) if r['result_flag'] == 1 else -stake
#         bankroll += profit
#         stakes.append(stake)
#         profits.append(profit)

#     df_filtered['stake'] = stakes
#     df_filtered['profit'] = profits
#     summary = {
#         'total_games': len(df),
#         'bet_games': len(df_filtered),
#         'wins': (df_filtered['profit'] > 0).sum(),
#         'losses': (df_filtered['profit'] < 0).sum(),
#         'total_profit': df_filtered['profit'].sum(),
#         'final_bankroll': bankroll,
#         'roi': df_filtered['profit'].sum() / df_filtered['stake'].sum() if df_filtered['stake'].sum() > 0 else 0
#     }
#     return df_filtered, summary

# # Run simulations
# platt_df, platt_summary = simulate(df, 'prob_platt')
# iso_df, iso_summary   = simulate(df, 'prob_iso')

# # Display Platt results
# display(platt_df[['date','home_team','away_team','prob_platt','odds_1','stake','profit']])
# print("Platt Summary:", platt_summary)

# # Display Iso results
# display(iso_df[['date','home_team','away_team','prob_iso','odds_1','stake','profit']])
# print("Iso Summary:", iso_summary)



# In[ ]:


# import pandas as pd
# import os

# # Parameters
# odds_min = 1.2
# odds_max = 2.8
# prob_thresh = 0.40
# home_win_cut = 0.55
# starting_bank = 1000.0
# bet_frac = 0.5
# cap_frac = 0.30
# abs_cap = 300.0

# # Paths (define these before running)
# # csv_path = ...
# # HWR_FILE = ...

# # Load data
# df = pd.read_csv(csv_path)
# hwr_df = pd.read_csv(HWR_FILE, index_col=0)
# home_win_rate = hwr_df['Home Win Rate'].to_dict()

# # Convert date column to datetime
# df['date'] = pd.to_datetime(df['date'], errors='coerce')

# # Filter for April and May games only
# df = df[df['date'].dt.month.isin([4, 5])]


# # Clean and map
# df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
# df['home_win_rate'] = df['home_team'].map(home_win_rate)

# # Filter games
# df_filtered = df[
#     (df['home_win_rate'] >= home_win_cut) &
#     (df['odds_1'] >= odds_min) &
#     (df['odds_1'] <= odds_max) &
#     (df['home_team_prob'] >= prob_thresh)
# ].copy()

# # Calculate stakes and profits
# bankroll = starting_bank
# stakes = []
# profits = []

# for _, row in df_filtered.iterrows():
#     p = row['home_team_prob']
#     o = row['odds_1']
#     if o <= 1:
#         stakes.append(0.0)
#         profits.append(0.0)
#         continue
#     kf = max(((o - 1) * p - (1 - p)) / (o - 1), 0) * bet_frac
#     stake = min(kf * bankroll, cap_frac * bankroll, abs_cap)
#     profit = stake * (o - 1) if row['result'] == row['home_team'] else -stake
#     bankroll += profit
#     stakes.append(stake)
#     profits.append(profit)

# df_filtered['stake'] = stakes
# df_filtered['profit'] = profits

# # Summary
# total_games = len(df)
# bet_games = len(df_filtered)
# wins = (df_filtered['profit'] > 0).sum()
# losses = (df_filtered['profit'] < 0).sum()
# total_profit = sum(profits)
# roi = total_profit / sum(stakes) if stakes else 0

# # Output
# print("\n📋 Filtered Bets:\n")
# print(df_filtered[['date', 'home_team', 'away_team', 'home_team_prob', 'odds_1',
#                    'home_win_rate', 'stake', 'profit']].to_string(index=False))

# print("\n📈 PERFORMANCE SUMMARY:")
# print(f"Total games considered: {total_games}")
# print(f"Bet-worthy games:       {bet_games}")
# print(f"Wins: {wins} | Losses: {losses}")
# print(f"Total profit:          €{total_profit:.2f}")
# print(f"Final bankroll:        €{bankroll:.2f}")
# print(f"ROI:                    {roi:.2%}")


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR     = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
TODAY_STR    = today_str
INPUT_FILE   = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{TODAY_STR}.csv")
HWR_FILE     = os.path.join(BASE_DIR, f"home_win_rates_sorted_{TODAY_STR}.csv")
OUTPUT_FILE  = os.path.join(BASE_DIR, f"combined_nba_predictions_enriched_{TODAY_STR}.csv")
OUTPUT_FILE_filtered = os.path.join(BASE_DIR, f"combined_nba_predictions_enriched_filtered_{TODAY_STR}.csv")

odds_min     = 1.2
odds_max     = 2.8
prob_thresh  = 0.40
home_win_cut = 0.55

starting_bank = 1000.0
bet_frac      = 0.5
cap_frac      = 0.30
abs_cap       = 300.0

# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────
def kelly_frac(p, o, frac=1.0):
    b = o - 1
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    return max(((b * p - (1 - p)) / b) * frac, 0.0)

# ──────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, encoding="utf-7", decimal=",")
df.columns = (df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"\s+","_",regex=True)
)
df['date']           = pd.to_datetime(df['date'], errors='coerce')
df['odds_1']         = pd.to_numeric(df['odds_1'].astype(str).str.replace(',','.'), errors='coerce')
df['home_team_prob'] = pd.to_numeric(df['home_team_prob'].astype(str).str.replace(',','.'), errors='coerce')
df['win']            = (df['result'] == df['home_team']).astype(int)

# load home‐win‐rates
hwr = pd.read_csv(HWR_FILE, index_col=0)
good_home = set(hwr.loc[hwr['Home Win Rate'] >= home_win_cut].index)

# ──────────────────────────────────────────────────────────────
# 2) FIT PLATT & ISOTONIC ON HISTORICAL ACCURACY
# ──────────────────────────────────────────────────────────────
X = df[['home_team_prob']].values
y = df['accuracy'].astype(int).values

# Platt
platt = LogisticRegression(solver='lbfgs')
platt.fit(X,y)
df['prob_platt'] = platt.predict_proba(X)[:,1]

# Isotonic
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(X.flatten(), y)
df['prob_iso'] = iso.transform(X.flatten())

# ──────────────────────────────────────────────────────────────
# 3) PREPARE COLUMNS
# ──────────────────────────────────────────────────────────────
for lbl in ('raw','platt','iso'):
    df[f'kelly_frac_{lbl}'] = 0.0
    df[f'stake_{lbl}']      = 0.0
    df[f'pnl_{lbl}']        = 0.0
    df[f'ev_{lbl}']         = 0.0
    df[f'bank_{lbl}']       = np.nan

# ──────────────────────────────────────────────────────────────
# 4) SIMULATE ALL THREE BANKROLLS
# ──────────────────────────────────────────────────────────────
bank = {'raw':starting_bank, 'platt':starting_bank, 'iso':starting_bank}

for i,row in df.sort_values('date').iterrows():
    o      = row['odds_1']
    is_home= row['home_team'] in good_home

    for lbl, p_col in [('raw','home_team_prob'),
                       ('platt','prob_platt'),
                       ('iso','prob_iso')]:
        p = row[p_col]
        if is_home and odds_min<=o<=odds_max and p>=prob_thresh:
            kf    = kelly_frac(p, o, bet_frac)
            stake = min(kf*bank[lbl], cap_frac*bank[lbl], abs_cap)
            won   = bool(row['win'])
            pnl   = stake*(o-1) if won else -stake
            ev    = (p*(o-1) - (1-p))*stake
        else:
            kf=stake=pnl=ev=0.0

        bank[lbl] += pnl
        df.at[i, f'kelly_frac_{lbl}'] = kf
        df.at[i, f'stake_{lbl}']      = stake
        df.at[i, f'pnl_{lbl}']        = pnl
        df.at[i, f'ev_{lbl}']         = ev
        df.at[i, f'bank_{lbl}']       = bank[lbl]

# ──────────────────────────────────────────────────────────────
# 5) SAVE unfiltered
# ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
print("✅ Wrote enriched file →", OUTPUT_FILE)

# ──────────────────────────────────────────────────────────────
# 5) FILTER OUT “NO‐BET” ROWS
#    Keep any row where at least one strategy actually staked > 0
# ──────────────────────────────────────────────────────────────
mask = (
    (df["stake_raw"]   > 0) |
    (df["stake_platt"] > 0) |
    (df.get("stake_iso", 0) > 0)
)
df = df.loc[mask].reset_index(drop=True)
# ──────────────────────────────────────────────────────────────
# 6) SAVE filtered
# ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE_filtered, index=False, float_format="%.4f")
print("✅ Wrote enriched file →", OUTPUT_FILE_filtered)

# ──────────────────────────────────────────────────────────────
# 7) PLOT ALL THREE CURVES
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
for lbl, color in [('raw','C0'),('platt','C1'),('iso','C2')]:
    plt.plot(df['date'], df[f'bank_{lbl}'],
             label=f"{lbl.capitalize()}-Kelly bank", color=color)

plt.xlabel("Date")
plt.ylabel("Bankroll (€)")
plt.title("Raw vs Platt vs Iso-Kelly Bankroll Paths")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR      = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
TODAY_STR     = pd.Timestamp.now().strftime("%Y-%m-%d")

# today's raw predictions
PRED_FILE     = os.path.join(BASE_DIR, f"nba_games_predict_{TODAY_STR}.csv")
# home-win rates lookup
HWR_FILE      = os.path.join(BASE_DIR, f"home_win_rates_sorted_{TODAY_STR}.csv")
# historical combined preds (for Platt + iso)
HIST_FILE     = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{TODAY_STR}.csv")

# strategy thresholds
odds_min      = 1.2
odds_max      = 2.8
raw_prob_cut  = 0.40
home_win_cut  = 0.55

starting_bank = 42.20
bet_frac      = 0.5
cap_frac      = 0.30
abs_cap       = 300.0

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────
def kelly_frac(p,o,f): 
    b = o-1
    return max(((b*p-(1-p))/b)*f,0) if b>0 else 0

# ──────────────────────────────────────────────────────────────
# 1) LOAD & PREP
# ──────────────────────────────────────────────────────────────
# raw today
df = pd.read_csv(PRED_FILE, decimal=",", encoding="utf-7")
df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
df['date']           = pd.to_datetime(df['date']).dt.date
df['odds_1']         = pd.to_numeric(df['odds_1'].astype(str).str.replace(",","."), errors='coerce')
df['raw_prob']       = pd.to_numeric(df['home_team_prob'].astype(str).str.replace(",","."), errors='coerce')

# home-win rates
hwr = pd.read_csv(HWR_FILE, index_col=0)
good_homes = hwr[hwr["Home Win Rate"]>=home_win_cut].index.tolist()

# historical for calibration
hist = pd.read_csv(HIST_FILE, decimal=",", encoding="utf-7")
hist.columns = hist.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
X = hist[['home_team_prob']].values
y = hist['accuracy'].astype(int).values

# Platt
_, Xc, _, yc = train_test_split(X,y,test_size=0.2,random_state=42)
platt = LogisticRegression(solver='lbfgs').fit(Xc,yc)
df['prob_platt'] = platt.predict_proba(df[['raw_prob']])[:,1]

# Isotonic
iso = IsotonicRegression(out_of_bounds='clip').fit(hist['home_team_prob'],hist['accuracy'])
df['prob_iso']   = iso.transform(df['raw_prob'])

# win flags
df['win'] = (df['result']==df['home_team']).astype(int)

# ──────────────────────────────────────────────────────────────
# 2) FILTER & PRINT
# ──────────────────────────────────────────────────────────────
print(f"Good home teams by win-rate ≥ {home_win_cut:.2f}: " +
      ", ".join(good_homes) + "\n")

print("Today's schedule:")
print(df[['home_team','away_team','raw_prob','odds_1','odds_2','date']]
      .assign(raw_prob=lambda d: d['raw_prob'].map('{:.4f}'.format),
              odds_1=lambda d: d['odds_1'].map('{:.2f}'.format),
              odds_2=lambda d: d['odds_2'].map('{:.2f}'.format))
      .to_string(index=False))
print()

# games passing all filters
sel = df[
    (df.home_team.isin(good_homes)) &
    (df.odds_1.between(odds_min,odds_max)) &
    (df.raw_prob>=raw_prob_cut)
].copy()

print("Games with good home teams today: " +
      ", ".join(sel['home_team']+" vs "+sel['away_team']) or "None")
print(f"Games with odds_1 between {odds_min:.1f} and {odds_max:.1f}: " +
      ", ".join(sel['home_team']+" vs "+sel['away_team']) or "None")
print(f"Games with raw_prob ≥ {raw_prob_cut:.2f}: " +
      ", ".join(sel['home_team']+" vs "+sel['away_team']) or "None")
print()

# 3) KELLY CALCS
for idx, r in sel.iterrows():
    for label, p in [('raw',r.raw_prob),('platt',r.prob_platt),('iso',r.prob_iso)]:
        if p >= raw_prob_cut:
            kf    = kelly_frac(p, r.odds_1, bet_frac)
            stake = min(kf*starting_bank, cap_frac*starting_bank, abs_cap)
            print(f"✅ {r.home_team}–{r.away_team} ({label}):")
            print(f"  P̂(home)={p:.4f}, odds₁={r.odds_1:.2f}")
            print(f"   → Kelly frac: {kf:.4f}, Stake on €{starting_bank:.2f}: €{stake:.2f}")
    print()

# done


# In[ ]:


import os, re
import pandas as pd

# # # ────────────────────────────────────────────────────────────
# # # SETTINGS: thresholds, Kelly params, file paths
# # # ────────────────────────────────────────────────────────────
# odds_min           = 1.2
# odds_max           = 2.8
# prob_thresh        = 0.40
# home_win_rate_cut  = 0.6

# bet_fraction       = 0.1      # full-Kelly
# max_cap_frac       = 0.10     # max 50% of current bank per bet
# abs_bet_cap        = 300.0    # hard €300 cap

#────────────────────────────────────────────────────────────
#SETTINGS: thresholds, Kelly params, file paths
#────────────────────────────────────────────────────────────
odds_min           = 1.2      # widen your odds window at the low end
odds_max           = 2.8      # and at the high end
prob_thresh        = 0.40     # drop your model cutoff from 0.40 → 0.35
home_win_rate_cut  = 0.55     # relax home‐court cut from 0.55 → 0.50

bet_fraction       = 0.5      # use ½‐Kelly instead of full Kelly
max_cap_frac       = 0.30     # cap each bet at 30% of your current bank
abs_bet_cap        = 300.0    # hard cap €300 instead of €500


directory_path     = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
#today_str          = pd.Timestamp.now().strftime("%Y-%m-%d")
today_file         = os.path.join(directory_path, f"nba_games_predict_{today_str}.csv")
hwr_file           = os.path.join(directory_path, f"home_win_rates_sorted_{today_str}.csv")
out_file           = os.path.join(directory_path, f"kelly_bets_modul-7_{today_str}.csv")

# expected schema when saving
expected_cols = [
    "home_team","away_team","home_team_prob","odds_1","odds_2",
    "date","kelly_fraction_fixed","bet_amount_fixed","bet_result",
    "running_bank","win","result"
]

def kelly_frac(p, o, f=1.0):
    b = o - 1
    if b <= 0:
        return 0.0
    return max(((b * p - (1 - p)) / b) * f, 0.0)

def save_todays_bets():
    # 1) load home-win rates → strong homes
    hwr = pd.read_csv(hwr_file, index_col=0)
    good_home = hwr.loc[hwr["Home Win Rate"] >= home_win_rate_cut].index.tolist()
    print(f"Good home teams by win-rate ≥ {home_win_rate_cut}: {', '.join(good_home) or 'None'}")

    # 2) load today’s games & basic clean
    if not os.path.exists(today_file):
        print(f"⚠ No data for today at {today_file}")
        return

    g = pd.read_csv(today_file, encoding="utf-7", decimal=",")
    g.columns = g.columns.str.strip()
    g['date'] = pd.to_datetime(g['date'], errors='coerce')
    g['odds_1'] = pd.to_numeric(g['odds 1'].astype(str).str.replace(',', '.'), errors='coerce')
    g['odds_2'] = pd.to_numeric(g['odds 2'].astype(str).str.replace(',', '.'), errors='coerce')
    g['home_team_prob'] = pd.to_numeric(
        g['home_team_prob'].astype(str).str.replace(',', '.'), errors='coerce'
    )
    g['win'] = (g['result'] == g['home_team']).astype(int)

    # Print full schedule for today
    print("\nToday's schedule:")
    if not g.empty:
        print(
            g[['home_team','away_team','home_team_prob','odds_1','odds_2','date']]
             .assign(
                 date=lambda df: df['date'].dt.strftime('%Y-%m-%d'),
                 home_team_prob=lambda df: df['home_team_prob'].map('{:.4f}'.format),
                 odds_1=lambda df: df['odds_1'].map('{:.2f}'.format),
                 odds_2=lambda df: df['odds_2'].map('{:.2f}'.format),
             )
             .to_string(index=False)
        )
    else:
        print("No games scheduled today.")

    # explain home-team filter
    home_filtered = g[g.home_team.isin(good_home)].copy()
    print(f"\nGames with good home teams today: {', '.join(home_filtered['home_team'] + ' vs ' + home_filtered['away_team']) if not home_filtered.empty else 'None'}")

    # 3a) odds filter explanation
    odds_filtered = home_filtered[(home_filtered.odds_1 >= odds_min) & (home_filtered.odds_1 <= odds_max)]
    print(f"Games with odds_1 between {odds_min} and {odds_max}: {', '.join(odds_filtered['home_team'] + ' vs ' + odds_filtered['away_team']) if not odds_filtered.empty else 'None'}")

    # 3b) probability filter explanation
    prob_filtered = odds_filtered[odds_filtered.home_team_prob >= prob_thresh]
    print(f"Games with home_team_prob ≥ {prob_thresh}: {', '.join(prob_filtered['home_team'] + ' vs ' + prob_filtered['away_team']) if not prob_filtered.empty else 'None'}")

    # final selection
    sel = prob_filtered.sort_values('date').copy()
    if sel.empty:
        print("\n⚠ No games passed the in-sample filters today.")
        return

    # 4) Compute Kelly fraction for every filtered game
    sel['kelly_frac'] = sel.apply(
        lambda r: kelly_frac(r['home_team_prob'], r['odds_1'], bet_fraction),
        axis=1
    )
    sel['precap_stake'] = sel['kelly_frac'] * total_bankroll

    # detailed Kelly breakdown
    for _, r in sel.iterrows():
        b = r['odds_1'] - 1
        num = b * r['home_team_prob'] - (1 - r['home_team_prob'])
        den = b
        kf = num / den if den > 0 else 0.0
        print(f"\n✅ {r['home_team']}–{r['away_team']} Kelly Calculation:")
        print(f"P̂(home): {r['home_team_prob']:.4f}, Odds₁: {r['odds_1']:.2f}")
        print(f" Kelly frac: {kf:.4f}, Stake on €{total_bankroll:.2f}: €{kf * total_bankroll:.2f}")

    # 5) dynamic Kelly loop & saving (unchanged)
    bankroll = total_bankroll
    fracs, stakes, pnls, run_bank = [], [], [], []
    for _, r in sel.iterrows():
        kf     = r['kelly_frac']
        stake  = min(kf * bankroll, max_cap_frac * bankroll, abs_bet_cap)
        profit = stake * (r.odds_1 if r.win else -1)
        bankroll += profit
        fracs.append(kf); stakes.append(stake); pnls.append(profit); run_bank.append(bankroll)

    sel['kelly_fraction_fixed'] = fracs
    sel['bet_amount_fixed']     = stakes
    sel['bet_result']           = pnls
    sel['running_bank']         = run_bank

    if os.path.exists(out_file):
        old = pd.read_csv(out_file, encoding='utf-8')
    else:
        old = pd.DataFrame(columns=expected_cols)

    new = sel[expected_cols]
    combined = pd.concat([old, new], ignore_index=True)
    combined.drop_duplicates(['date','home_team','away_team'], keep='last', inplace=True)
    combined.to_csv(out_file, index=False, float_format='%.2f', encoding='utf-8')

    print(f"\n✅ Saved {len(new)} new bets → {out_file}")

# run
total_bankroll = 42.2  # your current bank
save_todays_bets()

#print(combined)


# In[ ]:


#!/usr/bin/env python3
#PLATT

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR        = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
TODAY_STR       = today_str
RAW_PRED_FILE   = os.path.join(BASE_DIR, f"nba_games_predict_{TODAY_STR}.csv")
HWR_FILE        = os.path.join(BASE_DIR, f"home_win_rates_sorted_{TODAY_STR}.csv")
JOURNAL_FILE    = os.path.join(BASE_DIR, "bet_journal.csv")

# historical combined file (for Platt scaling)
HIST_PRED_FILE  = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{TODAY_STR}.csv")

# strategy thresholds
ODDS_MIN     = 1.2    # min decimal odds
ODDS_MAX     = 2.8    # max decimal odds
PROB_THRESH  = 0.40   # Platt-scaled prob ≥ 0.40
HOME_CUT     = 0.55   # home-win-rate ≥ 0.55

# staking parameters
START_BANK   = 42.20  # initial bankroll for sizing
BET_FRAC     = 0.5     # use half-Kelly
CAP_FRAC     = 0.30    # max 30% of bankroll
ABS_CAP      = 300.0   # hard cap €300

# ──────────────────────────────────────────────────────────────
# 1) LOAD & CALIBRATE
# ──────────────────────────────────────────────────────────────
# a) load historical combined preds (must include ['home_team_prob','accuracy'])
hist = pd.read_csv(HIST_PRED_FILE, encoding="utf-7", decimal=",")
X_hist = hist["home_team_prob"].values.reshape(-1,1)
y_hist = hist["accuracy"].astype(int).values

# split off a calibration set
_, X_calib, _, y_calib = train_test_split(
    X_hist, y_hist, test_size=0.2, random_state=42
)

# fit Platt scaler
platt = LogisticRegression(solver="lbfgs")
platt.fit(X_calib, y_calib)

# b) load today’s raw preds
df = pd.read_csv(RAW_PRED_FILE, encoding="utf-7", decimal=",")
df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

# parse columns
df["date"]      = pd.to_datetime(df["date"], errors="coerce").dt.date
df["odds_1"]    = pd.to_numeric(df["odds_1"].astype(str).str.replace(",", "."), errors="coerce")
df["raw_prob"]  = pd.to_numeric(df["home_team_prob"].astype(str).str.replace(",", "."), errors="coerce")
df["prob_platt"] = platt.predict_proba(df[["raw_prob"]])[:,1]
df["win"]       = (df["result"] == df["home_team"]).astype(int)

# ──────────────────────────────────────────────────────────────
# 2) FILTER & SIZE
# ──────────────────────────────────────────────────────────────
# load home-win rates
hwr = pd.read_csv(HWR_FILE, index_col=0)
good_home = hwr[hwr["Home Win Rate"] >= HOME_CUT].index.tolist()

bets = (
    df[df.home_team.isin(good_home)]
      .query("@ODDS_MIN <= odds_1 <= @ODDS_MAX and prob_platt >= @PROB_THRESH")
      .sort_values("date")
      .copy()
)

#print(bets)

if bets.empty:
    print("No signals today; nothing to journal.")
    exit(0)

def kelly_frac(p, o, frac=1.0):
    b = o - 1
    return max(((b*p - (1-p)) / b) * frac, 0.0) if b > 0 else 0.0

stakes, pnls, evs = [], [], []
bank = START_BANK

for _, r in bets.iterrows():
    kf    = kelly_frac(r["prob_platt"], r["odds_1"], BET_FRAC)
    stake = min(kf*bank, CAP_FRAC*bank, ABS_CAP)
    pnl   = stake * (r["odds_1"] if r["win"] else -1)
    ev    = (r["prob_platt"]*(r["odds_1"]-1) - (1-r["prob_platt"])) * stake
    bank += pnl
    stakes.append(stake)
    pnls.append(pnl)
    evs.append(ev)

bets["stake"] = stakes
bets["pnl"]   = pnls
bets["ev"]    = evs

print(bets)

# ──────────────────────────────────────────────────────────────
# 3) JOURNAL APPEND (no duplicates)
# ──────────────────────────────────────────────────────────────
new_bets = bets[bets["stake"]>0][["date","home_team","away_team","stake","pnl","ev"]].copy()

print(new_bets)

# ensure date types match
new_bets["date"] = pd.to_datetime(new_bets["date"])
if os.path.exists(JOURNAL_FILE):
    journal = pd.read_csv(JOURNAL_FILE, parse_dates=["date"])
else:
    journal = pd.DataFrame(columns=new_bets.columns)

# only keep truly new rows
to_add = (
    new_bets
      .merge(journal, on=["date","home_team","away_team","stake","pnl","ev"], 
             how="left", indicator=True)
      .query("_merge=='left_only'")
      .loc[:, new_bets.columns]
)

if not to_add.empty:
    updated = pd.concat([journal, to_add], ignore_index=True)
    updated.to_csv(JOURNAL_FILE, index=False)
    print(f"Appended {len(to_add)} new bet(s) to journal.")
else:
    print("No new bets to append.")



# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR     = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
TODAY_STR    = today_str

HIST_FILE    = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{TODAY_STR}.csv")
PRED_FILE    = os.path.join(BASE_DIR, f"nba_games_predict_{TODAY_STR}.csv")
HWR_FILE     = os.path.join(BASE_DIR, f"home_win_rates_sorted_{TODAY_STR}.csv")

# Strategy thresholds
ODDS_MIN     = 1.2
ODDS_MAX     = 2.8
PROB_THRESH  = 0.40
HOME_CUT     = 0.55

START_BANK   = 1000.0
BET_FRAC     = 0.5
CAP_FRAC     = 0.30
ABS_CAP      = 300.0

# ──────────────────────────────────────────────────────────────
# 1) LOAD & NORMALIZE HISTORICAL
# ──────────────────────────────────────────────────────────────
hist = pd.read_csv(HIST_FILE, encoding="utf-7", decimal=",")
# unify names
hist.columns = (
    hist.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
)

# coerce numeric
for col in ["odds_1","odds_2","home_team_prob"]:
    hist[col] = pd.to_numeric(hist[col].astype(str).str.replace(",", "."), errors="coerce")

# parse date & accuracy
hist["date"]     = pd.to_datetime(hist["date"], errors="coerce").dt.date
hist["accuracy"] = hist["accuracy"].astype(int)

# ──────────────────────────────────────────────────────────────
# 2) FIT PLATT ON HISTORICAL
# ──────────────────────────────────────────────────────────────
X = hist[["home_team_prob"]].values
y = hist["accuracy"].values

# hold out calibration split
_, X_calib, _, y_calib = train_test_split(X, y, test_size=0.2, random_state=42)

platt = LogisticRegression(solver="lbfgs")
platt.fit(X_calib, y_calib)

# inject calibrated probs back into hist
hist["prob_platt"] = platt.predict_proba(hist[["home_team_prob"]])[:,1]

# ──────────────────────────────────────────────────────────────
# 3) BACKTEST ON HISTORICAL (PLATT‐CALIBRATED)
# ──────────────────────────────────────────────────────────────
# load home-win rates
hwr = pd.read_csv(HWR_FILE, index_col=0)
good_home = hwr.loc[hwr["Home Win Rate"] >= HOME_CUT].index.tolist()

# filter hist to “would‐have‐bet” rows
bet_hist = hist[
    hist["home_team"].isin(good_home) &
    hist["odds_1"].between(ODDS_MIN, ODDS_MAX) &
    (hist["prob_platt"] >= PROB_THRESH)
].copy()

if bet_hist.empty:
    print("❌ No historical bets match your criteria.")
    exit(0)

# Kelly function
def kelly_frac(p,o,frac=1.0):
    b = o - 1
    return max(((b*p - (1-p))/b)*frac, 0.0) if b>0 else 0.0

# stake, pnl, ev
bet_hist["kelly"] = bet_hist.apply(lambda r: kelly_frac(r.prob_platt, r.odds_1, BET_FRAC), axis=1)
bet_hist["stake"] = bet_hist["kelly"] * START_BANK
bet_hist["accuracy"]   = bet_hist["accuracy"].astype(int)
bet_hist["pnl"]   = bet_hist["stake"] * (bet_hist["odds_1"] * bet_hist["accuracy"] - 1)
bet_hist["ev"]    = (bet_hist["prob_platt"]*(bet_hist["odds_1"]-1)
                     - (1-bet_hist["prob_platt"])) * bet_hist["stake"]

# ──────────────────────────────────────────────────────────────
# 4) AGGREGATE & PLOT
# ──────────────────────────────────────────────────────────────
daily = (
    bet_hist.groupby("date")[["pnl","ev"]]
            .sum()
            .assign(
               cum_pnl=lambda df: df.pnl.cumsum(),
               cum_ev =lambda df: df.ev.cumsum()
            )
            .reset_index()
)

plt.figure(figsize=(10,5))
plt.plot(daily["date"], daily["cum_pnl"], label="Cumulative PnL")
plt.plot(daily["date"], daily["cum_ev"],  "--", label="Cumulative EV")
plt.xlabel("Date"); plt.ylabel("€")
plt.title("Season-to-Date Backtest (Platt-Calibrated)")
plt.legend(); plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# — you already have:
#   hist["prob_platt"], hist["accuracy"]  and  df["prob_platt"]

# 1) calibration curve on historical platt‐calibrated probs
prob_true, prob_pred = calibration_curve(
    hist["accuracy"],            # 0/1 actual outcomes
    hist["prob_platt"],     # platt-calibrated model p’s
    n_bins=10
)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label="Calibration")
plt.plot([0,1],[0,1], "--", label="Perfectly calibrated")

# 2) overlay today's games
today_probs = df["prob_platt"].values
for p in today_probs:
    plt.axvline(p, color="C1", linestyle="--", alpha=0.7)
plt.scatter(today_probs, [0]*len(today_probs), color="C1", s=100, label="Today’s p_platt", zorder=5)

plt.xlabel("Predicted probability (platt)")
plt.ylabel("Observed frequency")
plt.title("Calibration + Today’s Games")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[ ]:


############################################# GRID SEARCH #################################################################


# In[ ]:


import os
import pandas as pd
import numpy as np
from itertools import product

# ──────────────────────────────────────────────────────────────
# 0) Paths & data load (same as before)
# ──────────────────────────────────────────────────────────────
directory      = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
today_str      = today_str
pred_file      = os.path.join(directory, f"combined_nba_predictions_acc_{today_str}.csv")
hwr_file       = os.path.join(directory, f"home_win_rates_sorted_{today_str}.csv")

df = pd.read_csv(pred_file, encoding="utf-7", decimal=",")
df.columns = df.columns.str.strip()
df["date"]           = pd.to_datetime(df["date"], errors="coerce")
df["odds_1"]         = pd.to_numeric(df["odds 1"].astype(str).str.replace(",", "."), errors="coerce")
df["home_team_prob"] = pd.to_numeric(df["home_team_prob"].astype(str).str.replace(",", "."), errors="coerce")
df["win"]            = (df["result"] == df["home_team"]).astype(int)

hwr = pd.read_csv(hwr_file, index_col=0)

# ──────────────────────────────────────────────────────────────
# 1) Fixed Kelly & stake calc
# ──────────────────────────────────────────────────────────────
def kelly_frac(p, o, frac=1.0):
    b = o - 1
    if b <= 0: return 0.0
    return max(((b * p - (1-p)) / b) * frac, 0.0)

def backtest(odds_min, odds_max, prob_thresh, home_win_cut,
             starting_bank=1000.0, bet_frac=1.0, cap_frac=0.5, abs_cap=500.0):
    # pick good homes
    good_home = hwr[hwr["Home Win Rate"] >= home_win_cut].index

    # filter signals
    bets = (
        df[df.home_team.isin(good_home)]
        .query("@odds_min <= odds_1 <= @odds_max and home_team_prob >= @prob_thresh")
        .sort_values("date")
        .copy()
    )
    if bets.empty:
        return None

    # dynamic‐Kelly stake + PnL
    #bank = starting_bank
    bank = 100.0
    #print(bank)

    stakes, pnls = [], []
    for _, r in bets.iterrows():
        kf    = kelly_frac(r.home_team_prob, r.odds_1, bet_frac)
        stake = min(kf * bank, cap_frac * bank, abs_cap)
        pnl   = stake * (r.odds_1 if r.win else -1)
        bank += pnl
        stakes.append(stake)
        pnls.append(pnl)

    bets["stake"] = stakes
    bets["pnl"]   = pnls

    total_stake  = bets["stake"].sum()
    total_profit = bets["pnl"].sum()
    roi          = total_profit / total_stake * 100 if total_stake else np.nan
    win_rate     = bets["win"].mean() * 100 if not bets.empty else np.nan

    return {
        "odds_min": odds_min,
        "odds_max": odds_max,
        "prob_thresh": prob_thresh,
        "home_win_cut": home_win_cut,
        "n_trades": len(bets),
        "profit": total_profit,
        "ROI_%": round(roi,2),
        "win_rate_%": round(win_rate,2),
    }

# ──────────────────────────────────────────────────────────────
# 2) Define your search grid
# ──────────────────────────────────────────────────────────────
odds_mins    = np.arange(1.1, 2.6, 0.1)    # e.g. 1.5 → 2.5
odds_maxs    = np.arange(1.7, 3.1, 0.1)    # e.g. 1.7 → 3.0
prob_ths     = np.arange(0.40, 0.71, 0.05) # 0.40 → 0.70
home_cuts    = np.arange(0.45, 0.81, 0.05) # 0.50 → 0.80

results = []
for om, oM, pt, hw in product(odds_mins, odds_maxs, prob_ths, home_cuts):
    if oM <= om: 
        continue
    out = backtest(om, oM, pt, hw)
    if out:
        results.append(out)

# ──────────────────────────────────────────────────────────────
# 3) Summarize top performers
# ──────────────────────────────────────────────────────────────


res_df = pd.DataFrame(results)

# Top 10 by profit
print("=== Top 10 by Profit ===")
print(res_df.sort_values("profit", ascending=False).head(10).to_string(index=False))

# Top 10 by ROI
print("\n=== Top 10 by ROI ===")
print(res_df.sort_values("ROI_%", ascending=False).head(10).to_string(index=False))


# In[ ]:


# after you’ve built your `res_df` (the DataFrame of grid‐search outputs):

import matplotlib.pyplot as plt

# pivot for max‐profit instead of max‐ROI
profit_pivot = res_df.pivot_table(
    index='odds_min',
    columns='odds_max',
    values='profit',
    aggfunc='max'
)

plt.figure(figsize=(10, 8))
plt.imshow(profit_pivot.values, aspect='auto', origin='lower')  # default colormap
plt.colorbar(label='Max Profit')
plt.xticks(
    range(len(profit_pivot.columns)),
    [f"{x:.1f}" for x in profit_pivot.columns],
    rotation=45
)
plt.yticks(
    range(len(profit_pivot.index)),
    [f"{y:.1f}" for y in profit_pivot.index]
)
plt.xlabel('odds_max')
plt.ylabel('odds_min')
plt.title('Max Profit by odds_min / odds_max')
plt.tight_layout()
plt.show()


# In[ ]:


# assume `res_df` is your DataFrame of grid–search results

# 1) Best combination for each odds_min
best_by_min = (
    res_df
    .loc[res_df.groupby('odds_min')['ROI_%'].idxmax()]
    .sort_values('odds_min')
    [['odds_min','odds_max','prob_thresh','home_win_cut','ROI_%','n_trades']]
)
print("▶ Best ROI at each odds_min:\n", best_by_min.to_string(index=False))

# 2) Best combination for each odds_max
best_by_max = (
    res_df
    .loc[res_df.groupby('odds_max')['ROI_%'].idxmax()]
    .sort_values('odds_max')
    [['odds_min','odds_max','prob_thresh','home_win_cut','ROI_%','n_trades']]
)
print("\n▶ Best ROI at each odds_max:\n", best_by_max.to_string(index=False))

# 3) Pivot‐table heatmap of ROI by odds_min vs odds_max
import matplotlib.pyplot as plt

pivot = res_df.pivot_table(
    index='odds_min',
    columns='odds_max',
    values='ROI_%',
    aggfunc='max'
)

plt.figure(figsize=(8,6))
plt.imshow(pivot, aspect='auto', origin='lower')
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title("Max ROI_% by odds_min / odds_max")
plt.xlabel("odds_max")
plt.ylabel("odds_min")
plt.colorbar(label="ROI_%")
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(res_df["n_trades"], res_df["profit"], c=res_df["ROI_%"], cmap="viridis", alpha=0.7)
plt.colorbar(label="ROI %")
plt.xlabel("Number of Bets")
plt.ylabel("Total Profit (units)")
plt.title("Grid Search Results — Size vs Profit (colored by ROI)")
plt.show()


# In[ ]:


# assuming `res_df` holds your full grid results
frontier = res_df[res_df.n_trades >= 50]
top_frontier = (
    frontier
    .sort_values(['ROI_%','profit'], ascending=[False, False])
    .head(10)
)
print(top_frontier.to_string(index=False))


# In[ ]:


import numpy as np

# 1) compute the two baselines
mean_profit = res_df["profit"].mean()
mean_roi    = res_df["ROI_%"].mean()

print(f"Average profit = {mean_profit:.0f}  |  average ROI = {mean_roi:.1f}%")

# 2) filter for sets that exceed *both*
candidates = res_df[
    (res_df["profit"] > mean_profit) &
    (res_df["ROI_%"]   > mean_roi)
]

# 3) sort by a combined score (e.g. profit*ROI) or just by profit
candidates = candidates.assign(
    score = candidates["profit"] * candidates["ROI_%"]  
).sort_values("score", ascending=False)

# 4) inspect the top handful
print(candidates[[
    "odds_min","odds_max","prob_thresh","home_win_cut",
    "profit","ROI_%"
]].head(10).to_string(index=False))


# In[ ]:


#######################################################################################################################################################


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# CONFIGURATION: adjust paths and strategy parameters below
# ──────────────────────────────────────────────────────────────
base_dir     = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
#today_str    = pd.Timestamp.now().strftime("%Y-%m-%d")
pred_file    = os.path.join(base_dir, f"combined_nba_predictions_acc_{today_str}.csv")
hwr_file     = os.path.join(base_dir, f"home_win_rates_sorted_{today_str}.csv")
journal_file = os.path.join(base_dir, "bet_journal.csv")

# Strategy filters

# odds_min      = 1.20    # avoid the very shortest chalk 
# odds_max      = 2.20    # cap before the long‐shot collapse
# prob_thresh   = 0.50    # model shows > 50 % yields ~60 % actual wins 
# home_win_cut  = 0.55    # stick to genuinely strong home teams


odds_min      = 1.2
odds_max      = 2.8
prob_thresh   = 0.4
home_win_cut  = 0.55

starting_bank = 1000.0
bet_frac      = 0.5
cap_frac      = 0.3
abs_cap       = 300.0

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────
def kelly_frac(p, o, frac=1.0):
    b = o - 1
    if b <= 0:
        return 0.0
    return max(((b * p - (1 - p)) / b) * frac, 0.0)

# ──────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN DATA
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(pred_file, encoding="utf-7", decimal=",")
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"\s+", "_", regex=True)
)
# keep date as Timestamp
df['date']           = pd.to_datetime(df['date'], errors='coerce')
df['odds_1']         = pd.to_numeric(df['odds_1'].astype(str).str.replace(',', '.'), errors='coerce')
df['home_team_prob'] = pd.to_numeric(df['home_team_prob'].astype(str).str.replace(',', '.'), errors='coerce')
df['win']            = (df['result'] == df['home_team']).astype(int)

hwr = pd.read_csv(hwr_file, index_col=0)

# ──────────────────────────────────────────────────────────────
# 2) SELECT BETS BASED ON STRATEGY
# ──────────────────────────────────────────────────────────────
good_home = hwr[hwr['Home Win Rate'] >= home_win_cut].index.tolist()

bets = (
    df[df.home_team.isin(good_home)]
      .query('@odds_min <= odds_1 <= @odds_max and home_team_prob >= @prob_thresh')
      .sort_values('date')
      .copy()
)

if bets.empty:
    print("No signals today; no bets to journal.")
    exit(0)

# ──────────────────────────────────────────────────────────────
# 3) CALCULATE STAKE, PNL, AND EV FOR EACH BET
# ──────────────────────────────────────────────────────────────
stakes, pnls, evs = [], [], []
bank = starting_bank

for _, r in bets.iterrows():
    kf    = kelly_frac(r.home_team_prob, r.odds_1, bet_frac)
    stake = min(kf * bank, cap_frac * bank, abs_cap)
    # option B: profit-only
    pnl = stake * (r.odds_1 - 1) if r.win else -stake
    ev    = (r.home_team_prob * (r.odds_1 - 1) - (1 - r.home_team_prob)) * stake
    bank += pnl
    stakes.append(stake)
    pnls.append(pnl)
    evs.append(ev)

bets['stake'] = stakes
bets['pnl']   = pnls
bets['ev']    = evs

# ──────────────────────────────────────────────────────────────
# 4) APPEND ONLY NEW REAL BETS TO JOURNAL
# ──────────────────────────────────────────────────────────────
new_bets = bets[bets['stake'] > 0].loc[:, ['date','home_team','away_team','stake','pnl','ev']]
if new_bets.empty:
    print("No real bets today; nothing to append.")
    exit(0)

# load or init journal
if os.path.exists(journal_file):
    journal = pd.read_csv(journal_file, parse_dates=['date'])
else:
    journal = pd.DataFrame(columns=new_bets.columns)

# ensure both are datetime64
journal['date']  = pd.to_datetime(journal['date'])
new_bets['date'] = pd.to_datetime(new_bets['date'])

# concat & drop duplicates on key triplet
updated = pd.concat([journal, new_bets], ignore_index=True)
updated.drop_duplicates(subset=['date','home_team','away_team'], keep='last', inplace=True)

# save back
updated.to_csv(journal_file, index=False, float_format='%.2f')
print(f"Appended {len(updated) - len(journal)} new bet(s); journal now has {len(updated)} rows.")

# ──────────────────────────────────────────────────────────────
# 5) AGGREGATE & PLOT PERFORMANCE OVER TIME
# ──────────────────────────────────────────────────────────────
daily = (
    updated
      .groupby('date', as_index=False)
      .agg(stake=('stake','sum'),
           pnl  =('pnl','sum'),
           ev   =('ev','sum'))
      .assign(cum_pnl=lambda d: d['pnl'].cumsum(),
              cum_ev = lambda d: d['ev'].cumsum())
)

plt.figure(figsize=(10,6))
plt.plot(daily['date'], daily['cum_pnl'], label='Cumulative PnL')
plt.plot(daily['date'], daily['cum_ev'],  label='Cumulative EV', linestyle='--')
plt.xlabel('Date')
plt.ylabel('€')
plt.title('Realized PnL vs. Expected Value Over Time')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import os
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 0) Paths & filenames (adjust to your own dirs)
# ──────────────────────────────────────────────────────────────
base_dir     = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
today_str    = pd.Timestamp.now().strftime("%Y-%m-%d")
pred_file    = os.path.join(base_dir, f"combined_nba_predictions_acc_{today_str}.csv")

# ──────────────────────────────────────────────────────────────
# 1) Load & tidy your predictions
# ──────────────────────────────────────────────────────────────
pred = pd.read_csv(
    pred_file,
    encoding="utf-8",
    decimal=",",     # decimals use comma
    #thousands=".",   # thousands sep uses dot
)

# rename any weird column names
pred = pred.rename(columns={
    'odds 1':         'odds_1',
    'odds 2':         'odds_2',
    # if you have stray +AC0- or +AF8- prefixes:
})
# strip whitespace and stray encoding artifacts
pred.columns = (
    pred.columns
        .str.strip()
        .str.replace(r'\+AC0-','', regex=True)
        .str.replace(r'\+AF8-','_',  regex=True)
)

# coerce your key columns to numeric
for col in ["odds_1", "home_team_prob"]:
    pred[col] = (
        pred[col]
          .astype(str)
          .str.replace(r"[^0-9,\.]", "", regex=True)  # keep digits, comma, dot
          .str.replace(",", ".", regex=False)         # comma→dot for decimals
          .str.replace(r"\.(?=\d{3}\b)", "", regex=True)  # drop thousands-sep dots
    )
    pred[col] = pd.to_numeric(pred[col], errors="coerce")

# ──────────────────────────────────────────────────────────────
# 2) Build your true‐outcome flag
# ──────────────────────────────────────────────────────────────
# assumes you have columns "result" (winner name) and "home_team"
pred["win"] = (pred["result"] == pred["home_team"]).astype(int)


# ──────────────────────────────────────────────────────────────
# 3) Diagnostics
# ──────────────────────────────────────────────────────────────

# 3a) Mean odds & model‐prob by actual outcome
print("\n► Mean odds_1 and home_team_prob by actual outcome (0=loss,1=win):")
print(pred
    .groupby("win")[["odds_1","home_team_prob"]]
    .mean()
    .round(3)
)

# 3b) Pearson‐R of each feature vs. win
corr = pred[["win","odds_1","home_team_prob"]].corr().round(3)
print("\n► Pearson-R of features vs. win:")
print(corr.loc["win", ["odds_1","home_team_prob"]])

# 3c) Empirical win‐rate by quartile
for col in ["home_team_prob","odds_1"]:
    print(f"\n► Win-rate by {col!r} quartile:")
    quart = (
        pred
        .assign(quartile=pd.qcut(pred[col], 4))
        .groupby("quartile", observed=True)["win"]
        .agg(win_rate="mean", n_games="size")
        .round(3)
    )
    print(quart)


# In[ ]:


from sklearn.metrics import brier_score_loss, roc_auc_score

brier = brier_score_loss(pred.win, pred.home_team_prob)
auc   = roc_auc_score(pred.win, pred.home_team_prob)
print(f"Brier score: {brier:.3f}")   # lower is better (0 is perfect)
print(f"ROC AUC:      {auc:.3f}")     # 0.5 is random, 1.0 is perfect


# In[ ]:


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(pred.win, pred.home_team_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration curve")
plt.show()


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

# ──────────────────────────────────────────────────────────────
# 0) CONFIGURATION (make sure these are set already)
# ──────────────────────────────────────────────────────────────
# BASE_DIR, TODAY_STR, HWR_FILE, HIST_FILE, ODDS_MIN, ODDS_MAX,
# HOME_CUT, PROB_THRESH, START_BANK, BET_FRAC, CAP_FRAC, ABS_CAP
#
# HIST_FILE should be your "combined_nba_predictions_acc_{TODAY_STR}.csv"
# which contains columns like 'home_team_prob', 'result', 'accuracy', 'date', 'odds 1', 'odds 2', etc.

# ──────────────────────────────────────────────────────────────
# 1) LOAD & NORMALIZE COLUMN NAMES
# ──────────────────────────────────────────────────────────────
hist = pd.read_csv(HIST_FILE, encoding="utf-7", decimal=",")
# unify column names: strip whitespace, lowercase, replace spaces with underscores
hist.columns = (
    hist.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
)

# ──────────────────────────────────────────────────────────────
# 2) FIT ISOTONIC CALIBRATOR ON HISTORICAL P(home) → accuracy
# ──────────────────────────────────────────────────────────────
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(hist["home_team_prob"].values, hist["accuracy"].values)

# add calibrated prob to hist
hist["prob_iso"] = iso.transform(hist["home_team_prob"].values)

# ──────────────────────────────────────────────────────────────
# 3) CLEAN & PARSE TYPES
# ──────────────────────────────────────────────────────────────
# parse dates
hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date

# odds_1, odds_2 → numeric
hist["odds_1"] = pd.to_numeric(
    hist["odds_1"].astype(str).str.replace(",", "."), errors="coerce"
)
hist["odds_2"] = pd.to_numeric(
    hist["odds_2"].astype(str).str.replace(",", "."), errors="coerce"
)

# win indicator
hist["win"] = (hist["result"] == hist["home_team"]).astype(int)

# ──────────────────────────────────────────────────────────────
# 4) FILTER “WERE-WE-TO-BET” ROWS UNDER ISO-KELLY
# ──────────────────────────────────────────────────────────────
# load home-win rates
hwr = pd.read_csv(HWR_FILE, index_col=0)

good_homes = hwr.loc[hwr["Home Win Rate"] >= HOME_CUT].index.tolist()

# apply all filters
bets_iso = hist.loc[
    hist["home_team"].isin(good_homes) &
    (hist["odds_1"] >= ODDS_MIN) &
    (hist["odds_1"] <= ODDS_MAX) &
    (hist["prob_iso"] >= PROB_THRESH)
].sort_values("date").copy()

# if empty, skip
if bets_iso.empty:
    print("No isotonic-Kelly signals this season.")
else:
    # ──────────────────────────────────────────────────────────────
    # 5) SIZE WITH KELLY, COMPUTE PnL & EV
    # ──────────────────────────────────────────────────────────────
    def kelly_frac(p,o,f=1.0):
        b = o - 1
        return max(((b*p - (1-p))/b) * f, 0.0) if b>0 else 0.0

    bank = START_BANK
    stakes, pnls, evs = [], [], []
    for _, row in bets_iso.iterrows():
        kf    = kelly_frac(row["prob_iso"], row["odds_1"], BET_FRAC)
        stake = min(kf*bank, CAP_FRAC*bank, ABS_CAP)
        pnl   = stake * (row["odds_1"] if row["win"] else -1)
        ev    = (row["prob_iso"]*(row["odds_1"]-1) - (1-row["prob_iso"])) * stake
        bank += pnl
        stakes.append(stake)
        pnls.append(pnl)
        evs.append(ev)

    bets_iso["stake"] = stakes
    bets_iso["pnl"]   = pnls
    bets_iso["ev"]    = evs

    # ──────────────────────────────────────────────────────────────
    # 6) BUILD & PLOT EQUITY CURVE
    # ──────────────────────────────────────────────────────────────
    daily = (
        bets_iso
        .groupby("date")[["pnl","ev"]]
        .sum()
        .assign(
            cum_pnl=lambda df: df["pnl"].cumsum(),
            cum_ev =lambda df: df["ev"].cumsum()
        )
        .reset_index()
    )

    plt.figure(figsize=(10,5))
    plt.plot(daily["date"], daily["cum_pnl"], label="Iso-Kelly PnL")
    plt.plot(daily["date"], daily["cum_ev"],  linestyle="--", label="Iso-Kelly EV")
    plt.xlabel("Date"); plt.ylabel("€"); plt.title("Season-to-Date Backtest (Isotonic-Kelly)")
    plt.legend(); plt.tight_layout()
    plt.show()


# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# 0) CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
BANK     = 18.11   # your real bankroll

# adjust these to the *actual* filenames you have in BASE_DIR:
JOURNALS = {
    "Raw-Kelly":   "bet_journal_raw.csv",
    "Platt-Kelly": "bet_journal_calibrated.csv",
    "Iso-Kelly":   "bet_journal_iso.csv",
}

# ──────────────────────────────────────────────────────────────
# 1) LOAD & BUILD CURVES
# ──────────────────────────────────────────────────────────────
curves = {}
for label, fn in JOURNALS.items():
    path = os.path.join(BASE_DIR, fn)
    if not os.path.exists(path):
        print(f" – Skipping {label}: file not found ({fn})")
        continue

    df = pd.read_csv(path, parse_dates=["date"])
    cols = set(df.columns)

    # find amount column
    if   "stake"           in cols: amt_col = "stake"
    elif "bet_amount_fixed" in cols: amt_col = "bet_amount_fixed"
    else:
        raise KeyError(f"{fn!r} has no 'stake' or 'bet_amount_fixed'; columns are {cols}")

    # find PnL column
    if   "pnl"           in cols: pnl_col = "pnl"
    elif "bet_result"    in cols: pnl_col = "bet_result"
    else:
        raise KeyError(f"{fn!r} has no 'pnl' or 'bet_result'; columns are {cols}")

    # filter out the zero‐bets
    df = df[df[amt_col] > 0].copy()

    # daily sum & cumulative
    daily = df.groupby("date")[pnl_col].sum().cumsum()

    # scale from a 1000€ demo bank to your real BANK
    scaled = daily * (BANK / 1000)
    curves[label] = scaled

# ──────────────────────────────────────────────────────────────
# 2) PLOT THEM TOGETHER
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
for label, series in curves.items():
    plt.plot(series.index, series.values, label=f"{label} PnL")

plt.title("Raw vs Platt vs Iso-Kelly (scaled to €18.11 bank)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (€)")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

# ──────────────────────────────────────────────────────────────
# 0) CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
HIST_FILE   = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{pd.Timestamp.now():%Y-%m-%d}.csv")
PRED_FILE   = os.path.join(BASE_DIR, f"nba_games_predict_{pd.Timestamp.now():%Y-%m-%d}.csv")
HWR_FILE    = os.path.join(BASE_DIR, f"home_win_rates_sorted_{pd.Timestamp.now():%Y-%m-%d}.csv")

# strategy thresholds
ODDS_MIN    = 1.2
ODDS_MAX    = 2.8
PROB_THRESH = 0.40
HOME_CUT    = 0.55

START_BANK  = 1000.0
BET_FRAC    = 0.5
CAP_FRAC    = 0.30
ABS_CAP     = 300.0

# ──────────────────────────────────────────────────────────────
# 1) LOAD & CALIBRATE (ISOTONIC)
# ──────────────────────────────────────────────────────────────
# a) fit iso on your full season history
hist      = pd.read_csv(HIST_FILE, encoding="utf-7", decimal=",")
iso       = IsotonicRegression(out_of_bounds="clip")
iso.fit(hist["home_team_prob"], hist["accuracy"])

# b) load today’s preds
pred      = pd.read_csv(PRED_FILE, encoding="utf-7", decimal=",")
pred.columns = pred.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
pred["date"]            = pd.to_datetime(pred["date"],errors="coerce").dt.date
pred["odds_1"]          = pd.to_numeric(pred["odds_1"].astype(str).str.replace(",", "."),errors="coerce")
pred["raw_prob"]        = pd.to_numeric(pred["home_team_prob"].astype(str).str.replace(",", "."),errors="coerce")
pred["prob_iso"]        = iso.transform(pred["raw_prob"])
pred["accuracy"]             = (pred["result"] == pred["home_team"]).astype(int)

# load home-win rates
hwr       = pd.read_csv(HWR_FILE, index_col=0)
good_home = hwr[hwr["Home Win Rate"]>=HOME_CUT].index

# ──────────────────────────────────────────────────────────────
# 2) FILTER & BACKTEST
# ──────────────────────────────────────────────────────────────
def backtest(df, prob_col):
    bank, rows = START_BANK, []
    for _, r in df.iterrows():
        p   = r[prob_col]
        o   = r["odds_1"]
        w   = r["accuracy"]
        b   = o - 1
        kf  = max(((b*p - (1-p)) / b) * BET_FRAC, 0) if b>0 else 0
        stake = min(kf * bank, CAP_FRAC * bank, ABS_CAP)
        pnl   = stake * (o if w else -1)
        ev    = (p*(o-1) - (1-p)) * stake
        bank += pnl
        rows.append((r["date"], pnl, ev))
    return pd.DataFrame(rows, columns=["date","pnl","ev"])

# apply your filters
mask = (
    pred["home_team"].isin(good_home) &
    pred["odds_1"].between(ODDS_MIN,ODDS_MAX) &
    (pred["raw_prob"]>=PROB_THRESH)
)
bets_raw = pred[mask]

mask_iso = (
    pred["home_team"].isin(good_home) &
    pred["odds_1"].between(ODDS_MIN,ODDS_MAX) &
    (pred["prob_iso"]>=PROB_THRESH)
)
bets_iso = pred[mask_iso]

# backtest
df_raw = backtest(bets_raw, "raw_prob")
df_iso = backtest(bets_iso, "prob_iso")

# aggregate daily & cumulative
def agg(df):
    d = (df.groupby("date")[["pnl","ev"]]
           .sum()
           .assign(cum_pnl=lambda x: x["pnl"].cumsum(),
                   cum_ev =lambda x: x["ev"].cumsum()))
    return d

agg_raw = agg(df_raw)
agg_iso = agg(df_iso)

# ──────────────────────────────────────────────────────────────
# 3) PLOT
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10,5))
plt.plot(agg_raw.index, agg_raw["cum_pnl"], label="Raw-Kelly PnL",   linewidth=2)
plt.plot(agg_iso.index, agg_iso["cum_pnl"], label="Iso-Kelly PnL", linewidth=2, linestyle="--")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (€)")
plt.title("Season-to-Date Backtest: Raw vs. Isotonic Kelly")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np

# after you’ve read `df` and coerced odds & probs:
# and after loading home_win_rates as above...

results = []
for hw_cut in np.arange(0.50, 0.90, 0.05):   # try 60%, 65%, …, 80%
    good = home_win_rates_all_teams_sorted[home_win_rates_all_teams_sorted["Home Win Rate"] >= hw_cut].index
    mask = (
        df["home_team"].isin(good) &
        df["odds_1"].between(1.9, 2.3) &
        (df["home_team_prob"] >= 0.5)
    )
    sub = df[mask].copy()
    if sub.empty:
        continue
    sub["win"] = (sub["result"] == sub["home_team"]).astype(int)
    sub["pnl"] = -100.0
    won = sub["win"] == 1
    sub.loc[won, "pnl"] += sub.loc[won, "odds_1"] * 100

    n = len(sub)
    profit = sub["pnl"].sum()
    roi    = profit / (100 * n) * 100
    wr     = sub["win"].mean() * 100

    results.append({
        "home_win_cutoff": hw_cut,
        "n_bets": n,
        "profit": profit,
        "ROI (%)": round(roi,1),
        "win_rate (%)": round(wr,1)
    })

print(pd.DataFrame(results).set_index("home_win_cutoff"))


# In[ ]:





# In[ ]:


import os
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 0) Paths & filenames
# ──────────────────────────────────────────────────────────────
base_dir      = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
#today_str     = pd.Timestamp.now().strftime("%Y-%m-%d")

pred_file     = os.path.join(base_dir, f"combined_nba_predictions_acc_{today_str}.csv")
journal_file  = os.path.join(base_dir, "bet_journal.csv")
out_file      = os.path.join(base_dir, "data_journal.csv")

# ──────────────────────────────────────────────────────────────
# 1) Load your model’s predictions (keep their column names intact)
# ──────────────────────────────────────────────────────────────
pred = pd.read_csv(pred_file, encoding="utf-7", decimal=",")

# ensure the key columns have the right dtypes
pred["date"]            = pd.to_datetime(pred["date"], errors="coerce").dt.date
pred["odds 1"]          = pd.to_numeric(pred["odds 1"].astype(str).str.replace(",", "."), errors="coerce")
pred["home_team_prob"]  = pd.to_numeric(pred["home_team_prob"].astype(str).str.replace(",", "."), errors="coerce")
pred["win"]             = (pred["result"] == pred["home_team"]).astype(int)

# ──────────────────────────────────────────────────────────────
# 2) Load your running bet journal (also unchanged)
# ──────────────────────────────────────────────────────────────
jnl = pd.read_csv(journal_file, parse_dates=["date"])
jnl["date"] = jnl["date"].dt.date

# ──────────────────────────────────────────────────────────────
# 3) Merge on date, home_team, away_team
# ──────────────────────────────────────────────────────────────
df = (
    pred
    .merge(
        jnl[["date","home_team","away_team","stake","pnl"]],
        on=["date","home_team","away_team"],
        how="inner"
    )
    .sort_values("date")
    .reset_index(drop=True)
)

# ──────────────────────────────────────────────────────────────
# 4) Append to your “data_journal.csv” with the very same columns
# ──────────────────────────────────────────────────────────────
# write header only if the file doesn’t exist yet
write_header = not os.path.exists(out_file)

# round all floats to 3 decimal places (or adjust per-column via a dict)
df = df.round(3)

# df.to_csv(
#     out_file,
#     mode='a',
#     header=write_header,
#     index=False,
#     sep=';',                  # use semicolon if your locale needs it
#     float_format='%.3f'       # prints each float with 3 decimals
# )
df = df.drop_duplicates().reset_index(drop=True)

df.to_csv(out_file)
print(f"Appended {len(df)} rows to {out_file}")


# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss

# ──────────────────────────────────────────────────────────────
# 0) Paths & load
# ──────────────────────────────────────────────────────────────
base_dir     = r"D:\1. Python\1. NBA Script\2025\LightGBM\1. 2025_Prediction"
#today_str    = pd.Timestamp.now().strftime("%Y-%m-%d")

# your model’s daily predictions (must include date, home_team, away_team, odds 1, home_team_prob, result)
pred_file    = os.path.join(base_dir, f"combined_nba_predictions_acc_{today_str}.csv")
# your running bet journal (must include date, home_team, away_team, stake, pnl, win)
journal_file = os.path.join(base_dir, "bet_journal.csv")

# load predictions
pred = pd.read_csv(pred_file, encoding="utf-7", decimal=",")
pred.columns = pred.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
pred["date"]           = pd.to_datetime(pred["date"], errors="coerce").dt.date
pred["odds_1"]         = pd.to_numeric(pred["odds_1"].astype(str).str.replace(",", "."), errors="coerce")
pred["home_team_prob"] = pd.to_numeric(pred["home_team_prob"].astype(str).str.replace(",", "."), errors="coerce")
pred["win"]            = (pred["result"] == pred["home_team"]).astype(int)

# load journal
jnl = pd.read_csv(journal_file, parse_dates=["date"])
# ensure same dtypes & names
jnl["date"] = jnl["date"].dt.date
jnl.columns = jnl.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)

# ──────────────────────────────────────────────────────────────
# 1) merge predictions ↔ journal
# ──────────────────────────────────────────────────────────────
df = (
    pred
    .merge(
        jnl[["date","home_team","away_team","stake","pnl"]],
        on=["date","home_team","away_team"],
        how="inner",
    )
    .sort_values("date")
    .reset_index(drop=True)
)

journal_data = os.path.join(base_dir, "data_journal.csv")
df.to_csv(journal_data,
           mode='a',
           header=not os.path.exists(journal_file),
           index=False)


# ──────────────────────────────────────────────────────────────
# 2) recompute EV, implied & cumulative metrics
# ──────────────────────────────────────────────────────────────
# EV = stake * (p*(o-1) - (1-p))
df["ev"]       = df["stake"] * (df["home_team_prob"]*(df["odds_1"]-1) - (1-df["home_team_prob"]))
df["cum_pnl"]  = df["pnl"].cumsum()
df["cum_ev"]   = df["ev"].cumsum()

# variance of each bet for Z‐test
df["var_pnl"] = (
    df["stake"]**2 *
    (
      df["home_team_prob"] * (df["odds_1"]-1)**2 +
      (1-df["home_team_prob"])*1**2 -
      ( (df["home_team_prob"]*(df["odds_1"]-1) - (1-df["home_team_prob"]))**2 )
    )
)

# ──────────────────────────────────────────────────────────────
# 3) statistical tests & metrics
# ──────────────────────────────────────────────────────────────
# Z‐test: realized vs. expected
z_stat = (df["pnl"].sum() - df["ev"].sum()) / np.sqrt(df["var_pnl"].sum())
print(f"Z‐statistic: {z_stat:.2f} (±1.96 for 95% conf)")

# bootstrap CI on total edge
n_boot=5000
diffs = df["pnl"] - df["ev"]
boot = [ diffs.sample(frac=1, replace=True).sum() for _ in range(n_boot) ]
ci_lo, ci_hi = np.percentile(boot, [2.5,97.5])
print(f"Bootstrapped 95% CI for total PnL−EV: [{ci_lo:.0f}, {ci_hi:.0f}]")

# daily PnL, drawdown & Sharpe
daily = df.groupby("date")["pnl"].sum()
cum   = daily.cumsum()
dd    = cum - cum.cummax()
max_dd = dd.min()
sharpe = daily.mean()/daily.std()*np.sqrt(252)
print(f"Max Drawdown: {max_dd:.0f}")
print(f"Sharpe Ratio (annualized): {sharpe:.2f}")

# calibration
y_true = df["win"]
y_prob = df["home_team_prob"]
print(f"Brier Score: {brier_score_loss(y_true,y_prob):.4f}")
print(f"Log Loss:    {log_loss(y_true,y_prob):.4f}")

# reliability diagram bins
bins = np.linspace(0,1,11)
df["prob_bin"] = pd.cut(df["home_team_prob"], bins)
rel = (df
  .groupby("prob_bin")
  .agg(mean_pred=("home_team_prob","mean"),
       actual_win=("win","mean"),
       n=("win","size"))
  .dropna()
)

# segmentation
df["odds_bin"]   = pd.cut(df["odds_1"],   bins=np.arange(1.5,3.1,0.25))
df["edge_bin"]   = pd.cut(df["ev"]/df["stake"], bins=np.linspace(-0.3,0.3,9))
df["hwrate_bin"] = pd.cut(df["home_team_prob"], bins=np.linspace(0.5,1.0,11))

seg = (
    df
    .groupby(["odds_bin","edge_bin","hwrate_bin"])
    .agg(n_bets  =("pnl","size"),
         total_ev=("ev","sum"),
         profit   =("pnl","sum"),
         win_rate =("win","mean"))
    .reset_index()
)
seg.to_csv("segmentation_results.csv", index=False)
print("Segmentation saved to segmentation_results.csv")

#df.to_csv("data.csv", index=False)


# ──────────────────────────────────────────────────────────────
# 4) plotting
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
plt.plot(df["date"], df["cum_pnl"], label="Cumulative PnL")
plt.plot(df["date"], df["cum_ev"],  label="Cumulative EV", linestyle="--")
plt.title("Realized PnL vs. EV Over Time")
plt.xlabel("Date"); plt.ylabel("Value")
plt.legend(); plt.tight_layout()

plt.figure(figsize=(8,4))
plt.plot(dd.index, dd.values, label="Drawdown")
plt.title("Drawdown Over Time")
plt.xlabel("Date"); plt.ylabel("Drawdown")
plt.tight_layout()

plt.figure(figsize=(6,6))
plt.scatter(rel["mean_pred"], rel["actual_win"], s=rel["n"]*10, alpha=0.7)
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.title("Calibration (Reliability Diagram)")
plt.xlabel("Mean Predicted Prob"); plt.ylabel("Actual Win Rate")
plt.tight_layout()

plt.show()

#print(df).to_string(index=false)
#print(df)

