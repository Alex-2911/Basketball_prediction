import pandas as pd
import os

# --- Path to your latest enriched file ---
directory_path = r"D:\1. Python\6. GitHub\Basketball_prediction\2026\output\LightGBM"
enriched_path = os.path.join(directory_path, "combined_nba_predictions_enriched_2025-10-26.csv")

# --- Load and filter ---
df = pd.read_csv(enriched_path)
df['date'] = pd.to_datetime(df['date'], errors='coerce')

bets = df[
    (df['stake_raw']   > 0) |
    (df['stake_platt'] > 0) |
    (df['stake_iso']   > 0)
].copy()

# --- Display summary ---
cols = [
    'date', 'home_team', 'away_team', 'odds_1',
    'home_team_prob', 'prob_platt', 'prob_iso',
    'win', 'stake_raw', 'pnl_raw', 'stake_platt', 'pnl_platt', 'stake_iso', 'pnl_iso'
]

print("\n=== Bets Placed (Raw / Platt / Iso Kelly) ===")
if not bets.empty:
    print(bets[cols].sort_values('date').to_string(index=False))
else:
    print("No bets found in this enriched dataset.")

# --- Keep console open (for Windows & fallback-friendly) ---
try:
    input("\nPress Enter to close this window...")
except EOFError:
    pass
