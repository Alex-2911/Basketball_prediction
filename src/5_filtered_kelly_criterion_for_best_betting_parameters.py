#!/usr/bin/env python
# coding: utf-8

#########################################################################################################################
# KELLY CRITERION FOR BEST BETTING PARAMETERS

# Script 5 of 5
# This notebook performs a grid search to find optimal betting parameters, displays and saves the results,
# and then filters today's games to highlight top home teams.

# Ensure `4_calculate_betting_statistics` is executed before running this script.
#########################################################################################################################

import pandas as pd
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

# Import shared utilities
from nba_utils import (
    get_current_date,
    get_directory_paths,
    kelly_frac,
    get_home_win_rates
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get current date information
today, today_str, today_str_format = get_current_date()
yesterday, yesterday_str, yesterday_str_format = get_current_date(days_offset=1)

print(f"Today's date: {today_str_format}")
print(f"Yesterday's date: {yesterday_str_format}")

# ──────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────
# Get directory paths
paths = get_directory_paths()
BASE_DIR = paths['BASE_DIR']
DATA_DIR = paths['DATA_DIR']
target_folder = paths['NEXT_GAME_DIR']
directory_path = paths['PREDICTION_DIR']

# Input files
PRED_FILE = os.path.join(directory_path, f"nba_games_predict_{today_str_format}.csv")
HWR_FILE = os.path.join(directory_path, f"home_win_rates_sorted_{today_str_format}.csv")
HIST_FILE = os.path.join(directory_path, f"combined_nba_predictions_acc_{today_str_format}.csv")

# Output files
OUTPUT_FILE = os.path.join(directory_path, f"combined_nba_predictions_enriched_{today_str_format}.csv")
OUTPUT_FILE_filtered = os.path.join(directory_path, f"combined_nba_predictions_enriched_filtered_{today_str_format}.csv")
output_file_home = os.path.join(directory_path, f'home_win_rates_sorted_{today_str_format}.csv')
out_path_kelly = os.path.join(directory_path, f"kelly_stakes_{today_str_format}.csv")

# Strategy thresholds
odds_min = 1.19
odds_max = 2.8
raw_prob_cut = 0.40
home_win_cut = 0.50

# Betting parameters
starting_bank = 42.20
bet_frac = 0.5
cap_frac = 0.30
abs_cap = 300.0

# Load the dataset
df = pd.read_csv(HIST_FILE, encoding="utf-7")

# Ensure the date column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Calculate home win rates using the utility function
home_win_rates_all_teams_sorted = get_home_win_rates(df)

# Display sorted results
print("\n🏀 Home Win Rates (Sorted) for All Teams:")
print(home_win_rates_all_teams_sorted)

# Save to CSV
home_win_rates_all_teams_sorted.to_csv(
    output_file_home,
    index=True,  # preserve the team names in the CSV's first column
    index_label="team",
    float_format="%.4f"
)

# ──────────────────────────────────────────────────────────────
# 1) LOAD & PREP
# ──────────────────────────────────────────────────────────────
# Load today's predictions
df = pd.read_csv(PRED_FILE, decimal=",", encoding="utf-7")
df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
df['date'] = pd.to_datetime(df['date']).dt.date
df['odds_1'] = pd.to_numeric(df['odds_1'].astype(str).str.replace(",", "."), errors='coerce')
df['raw_prob'] = pd.to_numeric(df['home_team_prob'].astype(str).str.replace(",", "."), errors='coerce')

# Load home-win rates
hwr = pd.read_csv(HWR_FILE, index_col=0)
good_homes = hwr[hwr["Home Win Rate"] >= home_win_cut].index.tolist()

# Load historical data for calibration
hist = pd.read_csv(HIST_FILE, decimal=",", encoding="utf-7")
hist.columns = hist.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
X = hist[['home_team_prob']].values
y = hist['accuracy'].astype(int).values

# Platt calibration
_, Xc, _, yc = train_test_split(X, y, test_size=0.2, random_state=42)
platt = LogisticRegression(solver='lbfgs').fit(Xc, yc)
df['prob_platt'] = platt.predict_proba(df[['raw_prob']])[:, 1]

# Isotonic calibration
iso = IsotonicRegression(out_of_bounds='clip').fit(hist['home_team_prob'], hist['accuracy'])
df['prob_iso'] = iso.transform(df['raw_prob'])

# Win flags
df['win'] = (df['result'] == df['home_team']).astype(int)

# ──────────────────────────────────────────────────────────────
# 2) FILTER & PRINT
# ──────────────────────────────────────────────────────────────
print(f"Good home teams by win-rate ≥ {home_win_cut:.2f}: " +
      ", ".join(good_homes) + "\n")

df['odds_2'] = pd.to_numeric(df['odds_2'], errors='coerce')

print("Today's schedule:")
print(df[['home_team', 'away_team', 'raw_prob', 'odds_1', 'odds_2', 'date']]
      .assign(raw_prob=lambda d: d['raw_prob'].map('{:.4f}'.format),
              odds_1=lambda d: d['odds_1'].map('{:.2f}'.format),
              odds_2=lambda d: d['odds_2'].map('{:.2f}'.format))
      .to_string(index=False))
print()

# Games passing all filters
sel = df[
    (df.home_team.isin(good_homes)) &
    (df.odds_1.between(odds_min, odds_max)) &
    (df.raw_prob >= raw_prob_cut)
].copy()

print("Games with good home teams today: " +
      ", ".join(sel['home_team'] + " vs " + sel['away_team']) or "None")
print(f"Games with odds_1 between {odds_min:.1f} and {odds_max:.1f}: " +
      ", ".join(sel['home_team'] + " vs " + sel['away_team']) or "None")
print(f"Games with raw_prob ≥ {raw_prob_cut:.2f}: " +
      ", ".join(sel['home_team'] + " vs " + sel['away_team']) or "None")
print()

# ──────────────────────────────────────────────────────────────
# 3) KELLY CALCULATIONS
# ──────────────────────────────────────────────────────────────
for idx, r in sel.iterrows():
    for label, p in [('raw', r.raw_prob), ('platt', r.prob_platt), ('iso', r.prob_iso)]:
        if p >= raw_prob_cut:
            kf = kelly_frac(p, r.odds_1, bet_frac)
            stake = min(kf * starting_bank, cap_frac * starting_bank, abs_cap)
            print(f"✅ {r.home_team}–{r.away_team} ({label}):")
            print(f"  P̂(home)={p:.4f}, odds₁={r.odds_1:.2f}")
            print(f"   → Kelly frac: {kf:.4f}, Stake on €{starting_bank:.2f}: €{stake:.2f}")
    print()

# Prepare a list to accumulate rows
rows = []

for _, r in sel.iterrows():
    for label, p in [('raw', r.raw_prob),
                     ('platt', r.prob_platt),
                     ('iso', r.prob_iso)]:
        if p >= raw_prob_cut:
            kf = kelly_frac(p, r.odds_1, bet_frac)
            stake = min(kf * starting_bank, cap_frac * starting_bank, abs_cap)
            rows.append({
                'home_team': r.home_team,
                'away_team': r.away_team,
                'date': r.date,
                'method': label,
                'prob': p,
                'odds': r.odds_1,
                'kelly_frac': kf,
                'stake': stake,
            })

# Create the DataFrame
out_df = pd.DataFrame(rows)

# Optionally pivot for wide format:
wide_df = out_df.pivot_table(
    index=['home_team', 'away_team', 'date'],
    columns='method',
    values=['prob', 'odds', 'kelly_frac', 'stake']
)
# Flatten the column MultiIndex
wide_df.columns = [f'{metric}_{method}' for metric, method in wide_df.columns]

# Choose whichever you prefer (long vs wide)
final_df = wide_df.reset_index()

# Save to CSV
final_df.to_csv(out_path_kelly, index=False, float_format='%.2f')

print(f"✅ Kelly stakes summary saved to {out_path_kelly}")

# ──────────────────────────────────────────────────────────────
# CALCULATION OF SEASON STATISTICS
# ──────────────────────────────────────────────────────────────
# Updated betting parameters for season simulation
starting_bank = 1000.0
bet_frac = 0.5
cap_frac = 0.30
abs_cap = 300.0

# ──────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(HIST_FILE, encoding="utf-7", decimal=",")
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(r"\s+", "_", regex=True)
              )
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['odds_1'] = pd.to_numeric(df['odds_1'].astype(str).str.replace(',', '.'), errors='coerce')
df['home_team_prob'] = pd.to_numeric(df['home_team_prob'].astype(str).str.replace(',', '.'), errors='coerce')
df['win'] = (df['result'] == df['home_team']).astype(int)

# Load home-win-rates
hwr = pd.read_csv(HWR_FILE, index_col=0)
good_home = set(hwr.loc[hwr['Home Win Rate'] >= home_win_cut].index)

# ──────────────────────────────────────────────────────────────
# 2) FIT PLATT & ISOTONIC ON HISTORICAL ACCURACY
# ──────────────────────────────────────────────────────────────
X = df[['home_team_prob']].values
y = df['accuracy'].astype(int).values

# Platt
platt = LogisticRegression(solver='lbfgs')
platt.fit(X, y)
df['prob_platt'] = platt.predict_proba(X)[:, 1]

# Isotonic
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(X.flatten(), y)
df['prob_iso'] = iso.transform(X.flatten())

# ──────────────────────────────────────────────────────────────
# 3) PREPARE COLUMNS
# ──────────────────────────────────────────────────────────────
for lbl in ('raw', 'platt', 'iso'):
    df[f'kelly_frac_{lbl}'] = 0.0
    df[f'stake_{lbl}'] = 0.0
    df[f'pnl_{lbl}'] = 0.0
    df[f'ev_{lbl}'] = 0.0
    df[f'bank_{lbl}'] = np.nan

# ──────────────────────────────────────────────────────────────
# 4) SIMULATE ALL THREE BANKROLLS
# ──────────────────────────────────────────────────────────────
bank = {'raw': starting_bank, 'platt': starting_bank, 'iso': starting_bank}

for i, row in df.sort_values('date').iterrows():
    o = row['odds_1']
    is_home = row['home_team'] in good_home

    for lbl, p_col in [('raw', 'home_team_prob'),
                       ('platt', 'prob_platt'),
                       ('iso', 'prob_iso')]:
        p = row[p_col]
        if is_home and odds_min <= o <= odds_max and p >= raw_prob_cut:
            kf = kelly_frac(p, o, bet_frac)
            stake = min(kf * bank[lbl], cap_frac * bank[lbl], abs_cap)
            won = bool(row['win'])
            pnl = stake * (o - 1) if won else -stake
            ev = (p * (o - 1) - (1 - p)) * stake
        else:
            kf = stake = pnl = ev = 0.0

        bank[lbl] += pnl
        df.at[i, f'kelly_frac_{lbl}'] = kf
        df.at[i, f'stake_{lbl}'] = stake
        df.at[i, f'pnl_{lbl}'] = pnl
        df.at[i, f'ev_{lbl}'] = ev
        df.at[i, f'bank_{lbl}'] = bank[lbl]

# ──────────────────────────────────────────────────────────────
# 5) SAVE UNFILTERED
# ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
print("✅ Wrote enriched file →", OUTPUT_FILE)

# ──────────────────────────────────────────────────────────────
# 6) FILTER OUT "NO-BET" ROWS
#    Keep any row where at least one strategy actually staked > 0
# ──────────────────────────────────────────────────────────────
mask = (
    (df["stake_raw"] > 0) |
    (df["stake_platt"] > 0) |
    (df.get("stake_iso", 0) > 0)
)
df = df.loc[mask].reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# 7) SAVE FILTERED
# ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE_filtered, index=False, float_format="%.4f")
print("✅ Wrote enriched file →", OUTPUT_FILE_filtered)

# ──────────────────────────────────────────────────────────────
# 8) PLOT ALL THREE CURVES
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for lbl, color in [('raw', 'C0'), ('platt', 'C1'), ('iso', 'C2')]:
    plt.plot(df['date'], df[f'bank_{lbl}'],
             label=f"{lbl.capitalize()}-Kelly bank", color=color)

plt.xlabel("Date")
plt.ylabel("Bankroll (€)")
plt.title("Raw vs Platt vs Iso-Kelly Bankroll Paths")
plt.legend()
plt.tight_layout()
plt.show()
