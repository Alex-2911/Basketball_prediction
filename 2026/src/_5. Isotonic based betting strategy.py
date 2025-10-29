#!/usr/bin/env python
# coding: utf-8

# In[1]:


####################################################################################################
# SCRIPT 5 — ISOTONIC CALIBRATED BETTING ENGINE (DAILY DRIVER)
####################################################################################################

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression

# ========= PATHS / DATES =========
BASE_DIR = r"D:\1. Python\1. NBA Script\2026\LightGBM"

today = datetime.now()
today_date = today.date()
tomorrow_date = (today + timedelta(days=1)).date()

today_str = today.strftime("%Y-%m-%d")
yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")

COMBINED_FILE = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{today_str}.csv")
HWR_FILE      = os.path.join(BASE_DIR, f"home_win_rates_sorted_{today_str}.csv")
TODAY_PRED    = os.path.join(BASE_DIR, f"nba_games_predict_{today_str}.csv")

print("Using data for:", today_str)
print("Combined file:", COMBINED_FILE)
print("Home win rates file:", HWR_FILE)
print("Today's prediction file:", TODAY_PRED)

# ========= BET CONFIG / SEARCH SPACE =========
FLAT_STAKE      = 100.0   # stake per bet in backtest
LOOKAHEAD_HRS   = 36      # still used later for shortlist window, but we'll also do day-based logic
ODDS_MIN_GRID   = np.arange(1.1, 3.1, 0.1)
ODDS_MAX_GRID   = np.arange(1.2, 3.6, 0.1)
PROB_MIN_GRID   = np.arange(0.40, 0.90, 0.05)
HOMEWR_MIN_GRID = np.arange(0.50, 0.90, 0.05)

# ========= helper: numeric cleaner =========
def to_float_series(s):
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace("[^0-9.]", "", regex=True)
         .replace("", np.nan)
         .astype(float)
    )

# ========= 1A. LOAD HISTORICAL/COMBINED FILE =========
if not os.path.exists(COMBINED_FILE):
    raise FileNotFoundError(f"Missing combined file: {COMBINED_FILE}")

df_all = pd.read_csv(COMBINED_FILE, encoding="utf-7", decimal=",")
df_all.columns = (
    df_all.columns
         .str.strip()
         .str.lower()
         .str.replace(r"\s+","_", regex=True)
)

# coerce types
df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
if "odds_1" in df_all.columns:
    df_all["odds_1"] = to_float_series(df_all["odds_1"])
if "odds_2" in df_all.columns:
    df_all["odds_2"] = to_float_series(df_all["odds_2"])
df_all["home_team_prob"] = to_float_series(df_all["home_team_prob"])

# result-based win flag
df_all["win"] = (df_all["result"] == df_all["home_team"]).astype(int)

# mark rows we consider already decided
df_all["is_played"] = (
    df_all["result"].notna()
    & (df_all["result"].astype(str) != "0")
)

# make sure away_team exists (combined sometimes might)
if "away_team" not in df_all.columns:
    df_all["away_team"] = np.nan

# ========= 1B. LOAD TONIGHT'S PREDICTIONS AND MERGE =========
new_rows = pd.DataFrame()
if os.path.exists(TODAY_PRED):
    tmp = pd.read_csv(
        TODAY_PRED,
        encoding="utf-7",
        sep=",",
        quotechar='"',
        decimal=","
    )

    expected = {"home_team","away_team","home_team_prob"}
    if not expected.issubset({c.lower().strip() for c in tmp.columns}):
        # fallback schema
        tmp = pd.read_csv(
            TODAY_PRED,
            encoding="utf-7",
            sep=",",
            quotechar='"',
            decimal=",",
            header=None,
            names=["home_team","away_team","home_team_prob","odds_1","odds_2","result","date"]
        )

    # normalize
    tmp.columns = (
        tmp.columns
           .str.strip()
           .str.lower()
           .str.replace(r"\s+","_", regex=True)
    )

    # numeric cleanup
    tmp["home_team_prob"] = to_float_series(tmp.get("home_team_prob", np.nan))
    tmp["odds_1"]         = to_float_series(tmp.get("odds_1", np.nan))
    tmp["odds_2"]         = to_float_series(tmp.get("odds_2", np.nan))

    # date cleanup
    tmp["date"] = pd.to_datetime(tmp.get("date", pd.NaT), errors="coerce")

    # if still NaT, assume "today"
    tmp.loc[tmp["date"].isna(), "date"] = pd.Timestamp(today_date)

    # mark state
    tmp["win"]       = np.nan
    tmp["is_played"] = False

    # ensure these cols exist for key-matching
    if "away_team" not in df_all.columns:
        df_all["away_team"] = np.nan

    # anti-duplicate by (date, home_team, away_team)
    key_cols = ["date","home_team","away_team"]
    merged_keys = df_all[key_cols].drop_duplicates()

    tmp_merge = tmp.merge(
        merged_keys,
        on=key_cols,
        how="left",
        indicator=True
    )
    new_rows = tmp_merge[tmp_merge["_merge"] == "left_only"].drop(columns=["_merge"])

    # align columns before concat
    needed_cols = set(df_all.columns) | set(new_rows.columns)
    for col in needed_cols:
        if col not in df_all.columns:
            df_all[col] = np.nan
        if col not in new_rows.columns:
            new_rows[col] = np.nan

    # force upcoming rows to `is_played = False`
    new_rows["is_played"] = False

    # append to df_all
    if not new_rows.empty:
        df_all = pd.concat(
            [df_all, new_rows[df_all.columns]],
            ignore_index=True
        )

# ========= 1C. UPCOMING COUNT (DAY-BASED, NOT TIME-OF-DAY) =========
df_all["game_day"] = df_all["date"].dt.date

upcoming_mask = (
    (~df_all["is_played"])
    &
    (
        (df_all["game_day"] == today_date) |
        (df_all["game_day"] == tomorrow_date)
    )
)

n_upcoming = int(upcoming_mask.sum())

print("Rows total      :", len(df_all))
print("Completed games :", int(df_all["is_played"].sum()))
print("Upcoming games  :", n_upcoming)

# optional sanity print of upcoming slate
if n_upcoming > 0:
    preview_cols = [
        "date","home_team","away_team",
        "home_team_prob","odds_1","odds_2","is_played"
    ]
    print("\nUpcoming (today/tomorrow):")
    print(
        df_all.loc[upcoming_mask, preview_cols]
              .sort_values("date")
              .round({"home_team_prob":3,"odds_1":3,"odds_2":3})
              .to_string(index=False)
    )
else:
    print("\nNo upcoming (today/tomorrow) games detected after merge.")


# In[2]:


####################################################################################################
# 2. ISOTONIC CALIBRATION
#    Fit on completed games only, then apply to all rows.
####################################################################################################

hist_mask = df_all["is_played"] & df_all["home_team_prob"].notna()
hist_calib = df_all.loc[hist_mask, ["home_team_prob", "win"]].copy()

if hist_calib.empty:
    print("Not enough completed games to calibrate, using raw model probs.")
    df_all["prob_iso"] = df_all["home_team_prob"]
    iso = None
else:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(
        hist_calib["home_team_prob"].values,
        hist_calib["win"].values
    )
    df_all["prob_iso"] = iso.transform(df_all["home_team_prob"].values)

# quick diagnostic: calibration reliability buckets
if hist_calib.shape[0] >= 10:
    calib_bins = (
        df_all.loc[hist_mask]
              .assign(prob_bin=pd.cut(
                  df_all.loc[hist_mask, "prob_iso"],
                  bins=[0,0.4,0.5,0.6,0.7,0.8,1.0]
              ))
              .groupby("prob_bin", observed=True)
              .agg(
                  avg_calib_prob=("prob_iso","mean"),
                  actual_home_win_rate=("win","mean"),
                  n_games=("win","size")
              )
              .round(3)
    )
    print("\nCalibration bins:")
    print(calib_bins)
else:
    print("\nSkipping calibration bins (not enough historical samples).")


# In[3]:


# Set directory path
read_file_path = os.path.join(BASE_DIR, f'combined_nba_predictions_acc_{today_str}.csv')

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
output_file = os.path.join(BASE_DIR, f'home_win_rates_sorted_{today_str}.csv')
home_win_rates_all_teams_sorted.to_csv(output_file, index=True)
print(f"\n📁 Sorted home win rates saved to: {output_file}")


# In[4]:


####################################################################################################
# 3. HOME STRENGTH LOOKUP
#    Attach rolling home win rate for each home_team.
####################################################################################################

if not os.path.exists(HWR_FILE):
    raise FileNotFoundError(f"Missing home win rate file: {HWR_FILE}")

hwr = pd.read_csv(HWR_FILE, index_col=0)
hwr.columns = [c.strip() for c in hwr.columns]  # should include "Home Win Rate"

df_all["home_win_rate"] = df_all["home_team"].map(hwr["Home Win Rate"])

print("Example home_win_rate mapping:")
print(df_all[["home_team","home_win_rate"]].head())


# In[5]:


####################################################################################################
# 4. GRID SEARCH FOR BEST PARAM COMBO (HISTORICAL ONLY)
#
# We'll search:
#   - minimum home_win_rate
#   - odds_1 in [odds_min, odds_max]
#   - minimum calibrated prob_iso
#
# Scoring:
#   stake = FLAT_STAKE each bet
#   pnl = +stake*(odds_1-1) if home wins, else -stake
#
# We pick the combo with the highest total profit.
####################################################################################################

hist_df = df_all[df_all["is_played"]].copy()

best_profit = float("-inf")
best_params = None
best_subset = None

for hw_cut in HOMEWR_MIN_GRID:
    strong_homes = hwr[hwr["Home Win Rate"] >= hw_cut].index.tolist()

    for o_min in ODDS_MIN_GRID:
        for o_max in ODDS_MAX_GRID:
            if o_max <= o_min:
                continue

            for p_min in PROB_MIN_GRID:

                mask_subset = (
                    (hist_df["home_team"].isin(strong_homes)) &
                    (hist_df["odds_1"]      >= o_min) &
                    (hist_df["odds_1"]      <= o_max) &
                    (hist_df["prob_iso"]    >= p_min)
                )
                subset = hist_df.loc[mask_subset].copy()
                if subset.empty:
                    continue

                # flat-stake pnl
                subset["pnl"] = np.where(
                    subset["win"] == 1,
                    FLAT_STAKE * (subset["odds_1"] - 1.0),
                    -FLAT_STAKE
                )

                total_profit = subset["pnl"].sum()
                n_trades     = len(subset)

                if total_profit > best_profit and n_trades > 0:
                    best_profit = total_profit
                    best_params = {
                        "home_win_rate_threshold": round(hw_cut, 2),
                        "odds_min": round(o_min, 2),
                        "odds_max": round(o_max, 2),
                        "prob_threshold": round(p_min, 2),
                        "n_trades": n_trades,
                        "win_rate_%": round(subset["win"].mean() * 100, 2),
                    }
                    best_subset = subset.copy()

if best_params is None:
    print("No profitable parameter combo found. Check data / ranges.")
else:
    total_stake = best_params["n_trades"] * FLAT_STAKE
    roi_pct = (best_profit / total_stake * 100.0) if total_stake else 0.0

    print("\n=== BEST PARAMS (ISOTONIC, HISTORICAL) ===")
    print(f"home_win_rate_threshold : {best_params['home_win_rate_threshold']}")
    print(f"odds_min                : {best_params['odds_min']}")
    print(f"odds_max                : {best_params['odds_max']}")
    print(f"prob_threshold (iso)    : {best_params['prob_threshold']}")
    print(f"n_trades                : {best_params['n_trades']}")
    print(f"win_rate_%              : {best_params['win_rate_%']}")
    print(f"total_profit €          : {best_profit:.2f}")
    print(f"ROI %                   : {roi_pct:.2f}%")

    print("\nSample of historical bets that match best params:")
    cols_preview = [
        "date","home_team","away_team",
        "home_win_rate","prob_iso","odds_1","result","win","pnl"
    ]
    print(
        best_subset[cols_preview]
        .sort_values("date")
        .head(15)
        .round(3)
        .to_string(index=False)
    )


####################################################################################################
# 6. SEASON PERFORMANCE DASHBOARD (ISOTONIC STRATEGY)
#    Visualize cumulative profit, rolling win rate, and calibration reliability.
#    NEW: Save historical qualified bets to CSV.
####################################################################################################

import matplotlib.pyplot as plt

# --- Filter for historical (played) games ---
hist = df_all[df_all["is_played"]].copy()

# Only consider those that would have qualified under tuned params
mask_hist = (
    (hist["home_win_rate"] >= best_params["home_win_rate_threshold"]) &
    (hist["odds_1"]      >= best_params["odds_min"]) &
    (hist["odds_1"]      <= best_params["odds_max"]) &
    (hist["prob_iso"]    >= best_params["prob_threshold"])
)
hist_sel = hist.loc[mask_hist].copy()

if hist_sel.empty:
    print("No historical bets matching current isotonic strategy — not enough data for chart.")
else:
    # Calculate flat-stake PnL per game
    hist_sel["pnl"] = np.where(
        hist_sel["win"] == 1,
        FLAT_STAKE * (hist_sel["odds_1"] - 1.0),
        -FLAT_STAKE
    )

    # Explicit: which side are we 'betting'
    hist_sel["bet_side"] = hist_sel["home_team"]

    # Sort chronologically
    hist_sel = hist_sel.sort_values("date").reset_index(drop=True)

    # Cumulative profit
    hist_sel["cum_profit"] = hist_sel["pnl"].cumsum()

    # Rolling win rate (last 10 bets)
    hist_sel["rolling_win_rate"] = hist_sel["win"].rolling(10, min_periods=3).mean() * 100

    # Calibration reliability: predicted vs actual
    calib_plot = (
        hist_sel
        .assign(prob_bin=pd.cut(hist_sel["prob_iso"], bins=np.arange(0,1.05,0.1)))
        .groupby("prob_bin", observed=True)
        .agg(predicted_prob=("prob_iso", "mean"),
             actual_win=("win", "mean"))
        .dropna()
    )

    # ============================
    # NEW: SAVE HISTORICAL SIGNAL
    # ============================
    HIST_EXPORT_PATH = os.path.join(BASE_DIR, "bet_history_strategy.csv")

    hist_export_cols = [
        "date",
        "home_team",
        "away_team",
        "bet_side",
        "home_win_rate",
        "prob_iso",
        "odds_1",
        "win",          # 1 if home actually won
        "pnl",          # profit (FLAT_STAKE basis) on that game
        "cum_profit"    # running bankroll using FLAT_STAKE bets
    ]

    # round some numeric columns for readability in the CSV
    hist_to_save = (
        hist_sel[hist_export_cols]
        .copy()
        .sort_values("date")
        .round({
            "home_win_rate":3,
            "prob_iso":3,
            "odds_1":3,
            "pnl":2,
            "cum_profit":2
        })
    )

    hist_to_save.to_csv(HIST_EXPORT_PATH, index=False)
    print(f"💾 Saved historical qualified bets to {HIST_EXPORT_PATH}")
    print(f"   ({len(hist_to_save)} rows total so far)")

    # --- Plot 1: Cumulative profit ---
    plt.figure(figsize=(8,4))
    plt.plot(hist_sel["date"], hist_sel["cum_profit"], marker="o")
    plt.title("Cumulative Profit (Isotonic Strategy)")
    plt.xlabel("Date")
    plt.ylabel("Total Profit (€)")
    plt.grid(True)
    plt.show()

    # --- Plot 2: Rolling win rate ---
    plt.figure(figsize=(8,4))
    plt.plot(hist_sel["date"], hist_sel["rolling_win_rate"], marker="o")
    plt.title("Rolling 10-Game Win Rate (%)")
    plt.axhline(50, color="gray", linestyle="--", label="Break-even 50%")
    plt.ylabel("Win Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot 3: Calibration curve ---
    plt.figure(figsize=(5,5))
    plt.plot(calib_plot["predicted_prob"], calib_plot["actual_win"], marker="o")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.title("Calibration Curve (Isotonic Model)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Rate")
    plt.grid(True)
    plt.show()

    print(f"\nSeason summary (so far):")
    print(f"  Bets placed        : {len(hist_sel)}")
    print(f"  Win rate           : {hist_sel['win'].mean() * 100:.2f}%")
    print(f"  Total profit (€)   : {hist_sel['cum_profit'].iloc[-1]:.2f}")
    print(f"  Avg EV per bet (€) : {hist_sel['pnl'].mean():.2f}")


# In[6]:


####################################################################################################
# 5. APPLY BEST PARAMS TO UPCOMING GAMES (NEXT 36H) + REASONS
####################################################################################################

CARD_OUT_PATH = os.path.join(BASE_DIR, f"bet_shortlist_{today_str}.csv")


def load_today_predictions_safe(path_csv, iso_model, hwr_df):
    """
    Fallback loader if df_all has no future rows.
    Returns columns:
      date, home_team, away_team, home_team_prob, prob_iso,
      home_win_rate, odds_1, odds_2, is_played
    or empty DF.
    """
    if not os.path.exists(path_csv):
        return pd.DataFrame()

    # First try read with header
    tmp = pd.read_csv(
        path_csv,
        encoding="utf-7",
        sep=",",
        quotechar='"',
        decimal=","
    )

    # Check if header looked good; if not, force schema
    expected = {"home_team","away_team","home_team_prob"}
    if not expected.issubset({c.lower().strip() for c in tmp.columns}):
        tmp = pd.read_csv(
            path_csv,
            encoding="utf-7",
            sep=",",
            quotechar='"',
            decimal=",",
            header=None,
            names=["home_team","away_team","home_team_prob","odds_1","odds_2","result","date"]
        )

    # normalize cols
    tmp.columns = (
        tmp.columns
           .str.strip()
           .str.lower()
           .str.replace(r"\s+","_", regex=True)
    )

    # numeric cleanup
    def to_float_series(s):
        return (
            s.astype(str)
             .str.replace(",", ".", regex=False)
             .str.replace("[^0-9.]", "", regex=True)
             .replace("", np.nan)
             .astype(float)
        )

    if "home_team_prob" in tmp.columns:
        tmp["home_team_prob"] = to_float_series(tmp["home_team_prob"])
    else:
        tmp["home_team_prob"] = np.nan

    if "odds_1" in tmp.columns:
        tmp["odds_1"] = to_float_series(tmp["odds_1"])
    else:
        tmp["odds_1"] = np.nan

    if "odds_2" in tmp.columns:
        tmp["odds_2"] = to_float_series(tmp["odds_2"])
    else:
        tmp["odds_2"] = np.nan

    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    else:
        tmp["date"] = pd.NaT

    # attach home win rate
    tmp["home_win_rate"] = tmp["home_team"].map(hwr_df["Home Win Rate"])

    # calibrated prob
    if iso_model is not None:
        tmp["prob_iso"] = iso_model.transform(tmp["home_team_prob"].values)
    else:
        tmp["prob_iso"] = tmp["home_team_prob"]

    tmp["is_played"] = False

    keep_cols = [
        "date","home_team","away_team",
        "home_team_prob","prob_iso",
        "home_win_rate","odds_1","odds_2",
        "is_played"
    ]
    for k in keep_cols:
        if k not in tmp.columns:
            tmp[k] = np.nan

    return tmp[keep_cols].copy()

if best_params is None:
    print("\n(No best params found; skipping shortlist.)")
else:
    now_norm = pd.Timestamp.now().normalize()
    cutoff   = now_norm + pd.Timedelta(hours=LOOKAHEAD_HRS)

    # take future games from df_all
    upcoming = df_all[
        (~df_all["is_played"]) &
        (df_all["date"] >= now_norm) &
        (df_all["date"] <= cutoff)
    ].copy()

    # if we don't have any future rows in df_all (common early season),
    # fall back to today's prediction csv
    if upcoming.empty:
        hwr_df_local = pd.read_csv(HWR_FILE, index_col=0)
        hwr_df_local.columns = [c.strip() for c in hwr_df_local.columns]
        iso_model = iso if "iso" in globals() else None
        upcoming = load_today_predictions_safe(TODAY_PRED, iso_model, hwr_df_local)

    # still nothing? then no slate
    if upcoming.empty:
        print("\n=== TONIGHT'S SHORTLIST (ISOTONIC-CALIBRATED STRATEGY) ===")
        print("No upcoming games found in df_all or today's prediction file.")
    else:
        # shortlist filter
        mask_card = (
            (upcoming["home_win_rate"] >= best_params["home_win_rate_threshold"]) &
            (upcoming["odds_1"]      >= best_params["odds_min"]) &
            (upcoming["odds_1"]      <= best_params["odds_max"]) &
            (upcoming["prob_iso"]    >= best_params["prob_threshold"])
        )
        card = upcoming.loc[mask_card].copy()

        print("\n=== TONIGHT'S SHORTLIST (ISOTONIC-CALIBRATED STRATEGY) ===")
        if card.empty:
            print("No games match the tuned iso strategy in the next slate.")
        else:
            # EV for flat €100 stake using calibrated probability
            card["EV_€_per_100"] = (
                card["prob_iso"] * (card["odds_1"] - 1.0)
                - (1.0 - card["prob_iso"])
            ) * FLAT_STAKE

            cols_card = [
                "date",
                "home_team","away_team",
                "home_win_rate",
                "prob_iso",
                "odds_1",
                "EV_€_per_100"
            ]
            print(
                card[cols_card]
                .sort_values("date")
                .round({
                    "home_win_rate":3,
                    "prob_iso":3,
                    "odds_1":3,
                    "EV_€_per_100":2
                })
                .to_string(index=False)
            )

        # diagnostic: show ALL upcoming with reason why they failed or "QUALIFIES"
        reasons = []
        for _, r in upcoming.iterrows():
            fail_reasons = []
            if pd.notna(r["home_win_rate"]) and r["home_win_rate"] < best_params["home_win_rate_threshold"]:
                fail_reasons.append(
                    f"home_win_rate {r['home_win_rate']:.2f} < {best_params['home_win_rate_threshold']}"
                )
            if pd.notna(r["odds_1"]) and r["odds_1"] < best_params["odds_min"]:
                fail_reasons.append(
                    f"odds {r['odds_1']:.2f} < min {best_params['odds_min']}"
                )
            if pd.notna(r["odds_1"]) and r["odds_1"] > best_params["odds_max"]:
                fail_reasons.append(
                    f"odds {r['odds_1']:.2f} > max {best_params['odds_max']}"
                )
            if pd.notna(r["prob_iso"]) and r["prob_iso"] < best_params["prob_threshold"]:
                fail_reasons.append(
                    f"prob_iso {r['prob_iso']:.2f} < {best_params['prob_threshold']}"
                )

            if not fail_reasons:
                fail_reasons.append("QUALIFIES")

            reasons.append("; ".join(fail_reasons))

        diag = upcoming.copy()
        diag["why_not"] = reasons

        diag_cols = [
            "date",
            "home_team","away_team",
            "home_win_rate",
            "prob_iso",
            "odds_1",
            "why_not"
        ]
        print("\n=== ALL UPCOMING GAMES & FILTER REASONS ===")
        print(
            diag[diag_cols]
            .sort_values("date")
            .round({
                "home_win_rate":3,
                "prob_iso":3,
                "odds_1":3
            })
            .to_string(index=False)
        )

print("\n=== TONIGHT'S SHORTLIST (ISOTONIC-CALIBRATED STRATEGY) ===")
if card.empty:
    print("No games match the tuned iso strategy in the next slate.")
else:
    # EV for flat €100 stake using calibrated probability
    card["EV_€_per_100"] = (
        card["prob_iso"] * (card["odds_1"] - 1.0)
        - (1.0 - card["prob_iso"])
    ) * FLAT_STAKE

    cols_card = [
        "date",
        "home_team","away_team",
        "home_win_rate",
        "prob_iso",
        "odds_1",
        "EV_€_per_100"
    ]

    # Pretty print to console
    print(
        card[cols_card]
        .sort_values("date")
        .round({
            "home_win_rate":3,
            "prob_iso":3,
            "odds_1":3,
            "EV_€_per_100":2
        })
        .to_string(index=False)
    )

    # === NEW: save today's qualified bets to CSV snapshot ===
    # We'll only save the rows that actually QUALIFY (i.e. `card`, not all upcoming)
    export_cols = [
        "date",
        "home_team",
        "away_team",
        "home_win_rate",
        "prob_iso",
        "odds_1",
        "EV_€_per_100"
    ]

    # Safety: make sure CARD_OUT_PATH exists in scope
    if "CARD_OUT_PATH" in globals() and CARD_OUT_PATH:
        # Save a clean version (rounded for readability)
        card_to_save = (
            card[export_cols]
            .sort_values("date")
            .round({
                "home_win_rate":3,
                "prob_iso":3,
                "odds_1":3,
                "EV_€_per_100":2
            })
        )
        card_to_save.to_csv(CARD_OUT_PATH, index=False)
        print(f"\n💾 Saved shortlist to {CARD_OUT_PATH}")
    else:
        print("\n[WARN] CARD_OUT_PATH not defined, shortlist not saved.")


# In[7]:


####################################################################################################
# 6. BET LOGGING / TRACKING
#
# Goal:
#   - Append tonight's bets (the shortlist "card") into a persistent bet_log.csv
#   - Later, after games end, you can update results and compute actual profit
#
# Usage tonight:
#   1. Set YOUR_STAKE_EUR to what you are actually betting per game (e.g. 20.0)
#   2. Run this cell AFTER the shortlist "card" is created in Section 5.
#
# Usage tomorrow morning:
#   1. Open bet_log.csv manually in Excel / etc, fill in:
#         actual_winner  (e.g. "MIL")
#      Save.
#   2. Re-run ONLY the "RECALC AFTER RESULTS" block at the bottom of this cell
#      to compute won / net_profit and print summary.
####################################################################################################

import pandas as pd
import os
import numpy as np

BET_LOG_PATH = os.path.join(BASE_DIR, "bet_log.csv")

# --- 1) nightly append of planned bets ---------------------------------------
YOUR_STAKE_EUR = 20.0  # <-- change this to the real € you plan to stake per pick

if 'card' in globals() and not card.empty:
    # we take just the bets the model said QUALIFIES
    to_log = card.copy().reset_index(drop=True)

    # normalize columns we care about
    to_log = to_log[[
        "date",
        "home_team",
        "away_team",
        "home_win_rate",
        "prob_iso",
        "odds_1"
    ]].copy()

    # add stake and placeholders for result we don't know yet
    to_log["stake_eur"]       = YOUR_STAKE_EUR
    to_log["actual_winner"]   = ""          # fill tomorrow: e.g. "MIL"
    to_log["won"]             = np.nan      # will become 1/0 after update
    to_log["payout_eur"]      = np.nan      # gross return if it wins
    to_log["net_profit_eur"]  = np.nan      # +€ if win, -stake if lose

    # load old log if exists, else create new
    if os.path.exists(BET_LOG_PATH):
        old = pd.read_csv(BET_LOG_PATH)
        # make sure date column stays nice round trip
        if "date" in old.columns:
            old["date"] = pd.to_datetime(old["date"], errors="coerce")
    else:
        old = pd.DataFrame()

    # concat and drop perfect duplicates so we don't log twice
    combined_log = pd.concat([old, to_log], ignore_index=True)
    combined_log.drop_duplicates(
        subset=["date", "home_team", "away_team", "odds_1"],
        keep="last",
        inplace=True
    )

    # sort by date for readability
    combined_log = combined_log.sort_values("date").reset_index(drop=True)

    # write back
    combined_log.to_csv(BET_LOG_PATH, index=False)
    print(f"✅ Logged {len(to_log)} planned bets to {BET_LOG_PATH}")

    # preview
    print(combined_log.tail(10).to_string(index=False))

else:
    print("ℹ No 'card' shortlist found or it's empty, so nothing was logged.")


# --- 2) RECALC AFTER RESULTS (run tomorrow morning) --------------------------
# This block:
#   - re-loads bet_log.csv
#   - for any row where actual_winner is filled (not empty string),
#     computes win/loss and profit.
#
# You can re-run this block any time to refresh PnL stats.

if os.path.exists(BET_LOG_PATH):
    bet_hist = pd.read_csv(BET_LOG_PATH)

    # ensure numeric types are sane after reload
    bet_hist["odds_1"]      = pd.to_numeric(bet_hist["odds_1"], errors="coerce")
    bet_hist["stake_eur"]   = pd.to_numeric(bet_hist["stake_eur"], errors="coerce")

    # mark won (1) or lost (0) if we know actual_winner
    bet_hist["won"] = np.where(
        (bet_hist["actual_winner"].astype(str) != "") &
        (bet_hist["actual_winner"].astype(str) == bet_hist["home_team"].astype(str)),
        1,
        np.where(
            (bet_hist["actual_winner"].astype(str) != "") &
            (bet_hist["actual_winner"].astype(str) != bet_hist["home_team"].astype(str)),
            0,
            np.nan  # still unknown / game not played
        )
    )

    # payout:
    #   if win: stake * odds_1  (gross cash returned)
    #   if loss: 0
    bet_hist["payout_eur"] = np.where(
        bet_hist["won"] == 1,
        bet_hist["stake_eur"] * bet_hist["odds_1"],
        np.where(bet_hist["won"] == 0, 0.0, np.nan)
    )

    # net profit:
    #   win: (stake * odds_1) - stake
    #   lose: -stake
    bet_hist["net_profit_eur"] = np.where(
        bet_hist["won"] == 1,
        bet_hist["stake_eur"] * (bet_hist["odds_1"] - 1.0),
        np.where(bet_hist["won"] == 0, -bet_hist["stake_eur"], np.nan)
    )

    # re-save the enriched log
    bet_hist.to_csv(BET_LOG_PATH, index=False)

    # running totals so far
    known_bets = bet_hist[bet_hist["won"].notna()].copy()
    total_profit = known_bets["net_profit_eur"].sum() if not known_bets.empty else 0.0
    total_staked = known_bets["stake_eur"].sum() if not known_bets.empty else 0.0
    winrate = known_bets["won"].mean()*100 if not known_bets.empty else np.nan

    print("\n📊 Updated bankroll snapshot (for all settled bets):")
    print(f"Bets settled        : {len(known_bets)}")
    print(f"Win rate (%)        : {winrate:.2f}" if not np.isnan(winrate) else "Win rate (%)        : n/a")
    print(f"Total staked (€)    : {total_staked:.2f}")
    print(f"Total net profit (€): {total_profit:.2f}")

    # last few rows for sanity
    print("\nLast logged rows:")
    print(bet_hist.tail(10).to_string(index=False))
else:
    print("ℹ bet_log.csv does not exist yet (no bets logged).")


# In[8]:


####################################################################################################
# 6. SEASON PERFORMANCE DASHBOARD (ISOTONIC STRATEGY)
#    Visualize cumulative profit, rolling win rate, and calibration reliability.
####################################################################################################

import matplotlib.pyplot as plt

# --- Filter for historical (played) games ---
hist = df_all[df_all["is_played"]].copy()

# Only consider those that would have qualified under tuned params
mask_hist = (
    (hist["home_win_rate"] >= best_params["home_win_rate_threshold"]) &
    (hist["odds_1"]      >= best_params["odds_min"]) &
    (hist["odds_1"]      <= best_params["odds_max"]) &
    (hist["prob_iso"]    >= best_params["prob_threshold"])
)
hist_sel = hist.loc[mask_hist].copy()

if hist_sel.empty:
    print("No historical bets matching current isotonic strategy — not enough data for chart.")
else:
    # Calculate flat-stake PnL per game
    hist_sel["pnl"] = np.where(
        hist_sel["win"] == 1,
        FLAT_STAKE * (hist_sel["odds_1"] - 1.0),
        -FLAT_STAKE
    )

    # Sort chronologically
    hist_sel = hist_sel.sort_values("date").reset_index(drop=True)
    hist_sel["cum_profit"] = hist_sel["pnl"].cumsum()

    # Rolling win rate (last 10 bets)
    hist_sel["rolling_win_rate"] = hist_sel["win"].rolling(10, min_periods=3).mean() * 100

    # Calibration reliability: predicted vs actual
    calib_plot = (
        hist_sel
        .assign(prob_bin=pd.cut(hist_sel["prob_iso"], bins=np.arange(0,1.05,0.1)))
        .groupby("prob_bin", observed=True)
        .agg(predicted_prob=("prob_iso", "mean"), actual_win=("win", "mean"))
        .dropna()
    )

    # --- Plot 1: Cumulative profit ---
    plt.figure(figsize=(8,4))
    plt.plot(hist_sel["date"], hist_sel["cum_profit"], marker="o")
    plt.title("Cumulative Profit (Isotonic Strategy)")
    plt.xlabel("Date")
    plt.ylabel("Total Profit (€)")
    plt.grid(True)
    plt.show()

    # --- Plot 2: Rolling win rate ---
    plt.figure(figsize=(8,4))
    plt.plot(hist_sel["date"], hist_sel["rolling_win_rate"], color="orange", marker="o")
    plt.title("Rolling 10-Game Win Rate (%)")
    plt.axhline(50, color="gray", linestyle="--", label="Break-even 50%")
    plt.ylabel("Win Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot 3: Calibration curve ---
    plt.figure(figsize=(5,5))
    plt.plot(calib_plot["predicted_prob"], calib_plot["actual_win"], marker="o")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.title("Calibration Curve (Isotonic Model)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Rate")
    plt.grid(True)
    plt.show()

    print(f"\nSeason summary (so far):")
    print(f"  Bets placed       : {len(hist_sel)}")
    print(f"  Win rate           : {hist_sel['win'].mean() * 100:.2f}%")
    print(f"  Total profit (€)   : {hist_sel['cum_profit'].iloc[-1]:.2f}")
    print(f"  Avg EV per bet (€) : {hist_sel['pnl'].mean():.2f}")


# In[9]:


####################################################################################################
# 6. SETTLE LAST NIGHT'S BETS AUTOMATICALLY
#
# - Reads bet_log.csv
# - Uses combined_nba_predictions_acc_<today>.csv to fill in winners
# - Recomputes won / payout / net PnL
# - Saves bet_log.csv back to disk
####################################################################################################

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

BET_LOG_PATH = os.path.join(BASE_DIR, "bet_log.csv")

def settle_bets_and_update_log(
    bet_log_path: str,
    combined_path: str,
    stake_col: str = "stake_eur"
):
    # 1. If there's no bet log yet, nothing to settle
    if not os.path.exists(bet_log_path):
        print(f"⚠ No bet log yet at {bet_log_path}, skipping settlement.")
        return None

    # 2. Load bet log
    bets = pd.read_csv(bet_log_path)

    # Normalize dtypes
    bets["date"] = pd.to_datetime(bets["date"], errors="coerce").dt.date
    if "actual_winner" not in bets.columns:
        bets["actual_winner"] = np.nan
    if "won" not in bets.columns:
        bets["won"] = np.nan
    if "payout_eur" not in bets.columns:
        bets["payout_eur"] = np.nan
    if "net_profit_eur" not in bets.columns:
        bets["net_profit_eur"] = np.nan

    # 3. Load combined predictions/results for today (already includes results from yesterday’s slate)
    if not os.path.exists(combined_path):
        print(f"⚠ Combined file {combined_path} not found. Can't auto-settle.")
        return bets

    combined = pd.read_csv(combined_path, encoding="utf-7", decimal=",")
    combined.columns = (
        combined.columns
                .str.strip()
                .str.lower()
                .str.replace(r"\s+","_", regex=True)
    )
    # clean types
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
    combined["result"] = combined["result"].astype(str)
    combined["home_team"] = combined["home_team"].astype(str)

    # 4. Merge each bet with actual winner from combined
    merged = bets.merge(
        combined[["date", "home_team", "result"]],
        on=["date", "home_team"],
        how="left",
        suffixes=("", "_actual")
    )

    # 5. Fill in actual_winner if we have a result
    merged["actual_winner"] = np.where(
        merged["result"].notna() & (merged["result"] != "0"),
        merged["result"],
        merged["actual_winner"]  # keep previous if already filled
    )

    # 6. Compute won? payout? pnl?
    # if home_team == actual_winner -> win
    merged["won"] = np.where(
        merged["actual_winner"].notna() &
        (merged["actual_winner"] == merged["home_team"]),
        1.0,
        np.where(
            merged["actual_winner"].notna(),
            0.0,
            np.nan  # still not graded
        )
    )

    # payout if decided
    merged["payout_eur"] = np.where(
        merged["won"] == 1.0,
        merged[stake_col] * merged["odds_1"],
        np.where(
            merged["won"] == 0.0,
            0.0,
            np.nan
        )
    )

    # net profit relative to stake
    merged["net_profit_eur"] = np.where(
        merged["won"] == 1.0,
        merged["payout_eur"] - merged[stake_col],
        np.where(
            merged["won"] == 0.0,
            -merged[stake_col],
            np.nan
        )
    )

    # 7. Save back to disk
    merged.to_csv(bet_log_path, index=False)
    print(f"✅ Bet log settled & saved to {bet_log_path}")

    # 8. Print bankroll snapshot
    decided = merged[merged["won"].notna()].copy()
    total_staked = decided[stake_col].sum()
    total_profit = decided["net_profit_eur"].sum()
    win_rate = decided["won"].mean() * 100 if len(decided) else 0.0

    print("\n📊 Bankroll status (all settled bets so far):")
    print(f"Bets settled        : {len(decided)}")
    print(f"Win rate (%)        : {win_rate:.2f}")
    print(f"Total staked (€)    : {total_staked:.2f}")
    print(f"Total net profit (€): {total_profit:.2f}")

    # and show the last few rows for sanity
    print("\nLatest bet log rows:")
    print(
        merged.sort_values(["date","home_team"])
              .tail(10)
              .to_string(index=False)
    )

    return merged


# ---------- run it ----------
settled_df = settle_bets_and_update_log(
    bet_log_path = BET_LOG_PATH,
    combined_path = COMBINED_FILE
)


# In[10]:


####################################################################################################
# CROSSCHECK: APPLY LAST SEASON BEST PARAMS TO THIS SEASON SO FAR
#
# Assumptions:
# - You're running this in the *current season* notebook,
#   AFTER you've built df_all, added columns:
#       date, is_played, win, home_win_rate, odds_1, prob_iso
# - df_all["win"] == 1 if home_team actually won (for completed games)
# - df_all["is_played"] True only for settled games
####################################################################################################

import numpy as np
import pandas as pd
import os

# last season's tuned parameters (fixed from grid search on 2024-25 season)
best_params_last_season = {
    "home_win_rate_threshold": 0.70,
    "odds_min": 1.50,
    "odds_max": 2.00,
    "prob_threshold": 0.55,
}

SIM_STAKE = 100.0  # same stake for comparability

# restrict to completed games in THIS season so far
hist_curr = df_all[df_all["is_played"]].copy()

mask_curr = (
    (hist_curr["home_win_rate"] >= best_params_last_season["home_win_rate_threshold"]) &
    (hist_curr["odds_1"]      >= best_params_last_season["odds_min"]) &
    (hist_curr["odds_1"]      <= best_params_last_season["odds_max"]) &
    (hist_curr["prob_iso"]    >= best_params_last_season["prob_threshold"])
)

curr_hits = hist_curr.loc[mask_curr].copy()

if curr_hits.empty:
    print("No current-season games match last season's strategy (yet).")
else:
    curr_hits["bet_side"] = curr_hits["home_team"]
    curr_hits["pnl"] = np.where(
        curr_hits["win"] == 1,
        SIM_STAKE * (curr_hits["odds_1"] - 1.0),
        -SIM_STAKE
    )
    curr_hits = curr_hits.sort_values("date").reset_index(drop=True)
    curr_hits["cum_profit"] = curr_hits["pnl"].cumsum()

    total_bets   = len(curr_hits)
    total_wins   = int(curr_hits["win"].sum())
    win_rate_pct = curr_hits["win"].mean() * 100.0
    total_profit = curr_hits["cum_profit"].iloc[-1]
    avg_profit   = curr_hits["pnl"].mean()
    roi_pct      = (total_profit / (SIM_STAKE * total_bets) * 100.0) if total_bets else 0.0

    print("\n=== THIS SEASON SO FAR (USING LAST SEASON PARAMS) ===")
    print(f"Total qualifying bets      : {total_bets}")
    print(f"Wins                       : {total_wins}")
    print(f"Win rate (%)               : {win_rate_pct:.2f}")
    print(f"Total profit (€)           : {total_profit:.2f}")
    print(f"Avg profit per bet (€)     : {avg_profit:.2f}")
    print(f"ROI on stake (%)           : {roi_pct:.2f}")
    print(f"Flat stake simulated (€)   : {SIM_STAKE:.2f} per bet")

    show_cols = [
        "date",
        "home_team",
        "away_team",
        "home_win_rate",
        "prob_iso",
        "odds_1",
        "win",
        "pnl",
        "cum_profit"
    ]
    print("\nCurrent season qualifying bets under last season params:")
    print(curr_hits[show_cols].round({
        "home_win_rate":3,
        "prob_iso":3,
        "odds_1":3,
        "pnl":2,
        "cum_profit":2
    }).to_string(index=False))


# In[ ]:




