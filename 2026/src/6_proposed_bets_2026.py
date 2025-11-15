import pandas as pd
import os
from pathlib import Path
import glob

# --- Path to your latest enriched file (cross-platform compatible) ---
# Get the directory where this script is located
script_dir = Path(__file__).parent
# Navigate to the 2026 directory (parent of src)
base_repo = script_dir.parent
directory_path = base_repo / "output" / "LightGBM"

# Find the most recent enriched file
enriched_files = list(directory_path.glob("combined_nba_predictions_enriched_*.csv"))
if not enriched_files:
    raise FileNotFoundError(
        f"No enriched predictions file found in {directory_path}. "
        "Please run script 5 first to generate the enriched data."
    )
# Sort by modification time and get the most recent
enriched_path = max(enriched_files, key=lambda p: p.stat().st_mtime)

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
