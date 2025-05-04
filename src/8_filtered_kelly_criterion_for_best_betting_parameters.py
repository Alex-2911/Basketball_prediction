#!/usr/bin/env python
# coding: utf-8

# In[1]:


#########################################################################################################################
# KELLY CRITERION FOR BEST BETTING PARAMETERS

# Script 5 of 5
#This notebook performs a grid search to find optimal betting parameters, displays and saves the results (including a summary in Excel),
#and then filters today’s games to highlight top home teams.

# Ensure `4_calculate_betting_statistics` is executed before running this script.
#########################################################################################################################


# In[2]:


import pandas as pd
import os
import glob
import numpy as np
import logging
from datetime import datetime, timedelta
from itertools import product
import shutil  # Make sure to import shutil


# In[3]:


days_back = 0

today = datetime.now() - timedelta(days=days_back)
today_str = (today).strftime("%Y-%m-%d")
print(today_str)

yesterday_str = (datetime.now() - timedelta(days=days_back+1)).strftime("%Y-%m-%d")

print(yesterday_str)


# In[4]:

BASE_DIR        = os.getcwd()
DATA_DIR        = os.path.join(BASE_DIR, "output", "Gathering_Data")
target_folder   = os.path.join(DATA_DIR, "Next_Game")

directory_path  = os.path.join(BASE_DIR, "output", "LightGBM", "1_2025_Prediction")

# In[5]:


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


# In[6]:


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
home_win_cut  = 0.50

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

df['odds_2'] = pd.to_numeric(df['odds_2'], errors='coerce')


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


# In[16]:


# Prepare a list to accumulate rows
rows = []

for _, r in sel.iterrows():
    for label, p in [('raw', r.raw_prob),
                     ('platt', r.prob_platt),
                     ('iso',  r.prob_iso)]:
        if p >= raw_prob_cut:
            kf    = kelly_frac(p, r.odds_1, bet_frac)
            stake = min(kf*starting_bank, cap_frac*starting_bank, abs_cap)
            rows.append({
                'home_team':    r.home_team,
                'away_team':    r.away_team,
                'date':         r.date,
                'method':       label,
                'prob':         p,
                'odds':         r.odds_1,
                'kelly_frac':   kf,
                'stake':        stake,
            })

# Create the DataFrame
out_df = pd.DataFrame(rows)

# Optionally pivot for wide format:
wide_df = out_df.pivot_table(
    index=['home_team','away_team','date'],
    columns='method',
    values=['prob','odds','kelly_frac','stake']
)
# Flatten the column MultiIndex
wide_df.columns = [f'{metric}_{method}' for metric,method in wide_df.columns]

# Choose whichever you prefer (long vs wide)
final_df = wide_df.reset_index()

# Ensure output folder exists
out_folder = os.path.join(BASE_DIR)
os.makedirs(out_folder, exist_ok=True)

# Save to CSV
out_path = os.path.join(out_folder, f"kelly_stakes_{TODAY_STR}.csv")
final_df.to_csv(out_path, index=False)

print(f"✅ Kelly stakes summary saved to {out_path}")
##############################################################################################################################



# In[7]:


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

