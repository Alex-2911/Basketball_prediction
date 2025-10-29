#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script 5 — Isotonic-Calibrated Betting Engine (CI Version)

This version is designed to run in GitHub Actions.
It:
 - loads season history (combined_nba_predictions_acc_<today>.csv)
 - calibrates isotonic
 - builds rolling home win rates
 - finds optimal betting params from history
 - builds tonight's shortlist using those params
 - appends to bet_log.csv and settles results
 - exports:
    * home_win_rates_sorted_<today>.csv
    * bet_history_strategy.csv
    * bet_shortlist_<today>.csv
    * bet_log.csv (updated)
All under 2026/output/LightGBM.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression

########################################
# 0. CONSTANTS / PATHS / DATES
########################################

# canonical folder for CI artifacts
BASE_DIR = os.path.join("2026", "output", "LightGBM")

# stake assumptions
FLAT_STAKE    = 100.0   # for backtest/gridsearch
YOUR_STAKE_EUR = 20.0   # what you actually bet live (used for bet_log.csv)
LOOKAHEAD_HRS = 36      # upcoming window for shortlist

# grid search space
ODDS_MIN_GRID   = np.arange(1.1, 3.1, 0.1)
ODDS_MAX_GRID   = np.arange(1.2, 3.6, 0.1)
PROB_MIN_GRID   = np.arange(0.40, 0.90, 0.05)
HOMEWR_MIN_GRID = np.arange(0.50, 0.90, 0.05)

# date handling (runner is UTC, we'll still just use "today" UTC as key)
now_dt = datetime.utcnow()
today_date = now_dt.date()
tomorrow_date = (now_dt + timedelta(days=1)).date()
today_str = now_dt.strftime("%Y-%m-%d")

# inputs
COMBINED_FILE = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{today_str}.csv")
PRED_TODAY    = os.path.join(BASE_DIR, f"nba_games_predict_{today_str}.csv")

# dynamic outputs we will create
HWR_FILE      = os.path.join(BASE_DIR, f"home_win_rates_sorted_{today_str}.csv")
CARD_OUT_PATH = os.path.join(BASE_DIR, f"bet_shortlist_{today_str}.csv")
BET_LOG_PATH  = os.path.join(BASE_DIR, "bet_log.csv")
HIST_EXPORT_PATH = os.path.join(BASE_DIR, "bet_history_strategy.csv")

print(f"[INFO] Using base dir: {BASE_DIR}")
print(f"[INFO] Today's date (UTC): {today_str}")
print(f"[INFO] Combined file expected: {COMBINED_FILE}")
print(f"[INFO] Today's prediction expected: {PRED_TODAY}")

########################################
# helper: numeric cleaner
########################################
def to_float_series(s):
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace("[^0-9.]", "", regex=True)
         .replace("", np.nan)
         .astype(float)
    )

########################################
# 1. LOAD COMBINED DATA
########################################

if not os.path.exists(COMBINED_FILE):
    raise FileNotFoundError(
        f"Missing combined file for {today_str}: {COMBINED_FILE}\n"
        "Make sure workflow 4 ran today and pushed combined_nba_predictions_acc_<today>.csv."
    )

df_all = pd.read_csv(COMBINED_FILE)
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
else:
    df_all["odds_1"] = np.nan

if "odds_2" in df_all.columns:
    df_all["odds_2"] = to_float_series(df_all["odds_2"])
else:
    df_all["odds_2"] = np.nan

if "home_team_prob" in df_all.columns:
    df_all["home_team_prob"] = to_float_series(df_all["home_team_prob"])
else:
    df_all["home_team_prob"] = np.nan

# mark winners
df_all["win"] = (df_all["result"] == df_all["home_team"]).astype(int)

# decide what is already finished
df_all["is_played"] = (
    df_all["result"].notna()
    & (df_all["result"].astype(str) != "0")
)

if "away_team" not in df_all.columns:
    df_all["away_team"] = np.nan

# upcoming mask helper
df_all["game_day"] = df_all["date"].dt.date

print("[INFO] df_all loaded. Rows total:", len(df_all))
print("[INFO] Completed games:", int(df_all["is_played"].sum()))

########################################
# 2. IF WE HAVE TODAY'S PRED FILE, MERGE IT IN
########################################

new_rows = pd.DataFrame()
if os.path.exists(PRED_TODAY):
    tmp = pd.read_csv(PRED_TODAY)

    # try first-pass headers, else fallback schema
    expected = {"home_team","away_team","home_team_prob"}
    if not expected.issubset({c.lower().strip() for c in tmp.columns}):
        tmp = pd.read_csv(
            PRED_TODAY,
            header=None,
            names=[
                "home_team","away_team","home_team_prob",
                "odds_1","odds_2","result","date"
            ]
        )

    # normalize cols
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
    tmp.loc[tmp["date"].isna(), "date"] = pd.Timestamp(today_date)

    # mark as future
    tmp["win"]       = np.nan
    tmp["is_played"] = False

    # anti-duplicate merge on (date, home_team, away_team)
    key_cols = ["date","home_team","away_team"]
    if "away_team" not in df_all.columns:
        df_all["away_team"] = np.nan

    merged_keys = df_all[key_cols].drop_duplicates()
    tmp_merge = tmp.merge(
        merged_keys,
        on=key_cols,
        how="left",
        indicator=True
    )
    new_rows = tmp_merge[tmp_merge["_merge"] == "left_only"].drop(columns=["_merge"])

    # align columns for concat
    needed_cols = set(df_all.columns) | set(new_rows.columns)
    for col in needed_cols:
        if col not in df_all.columns:
            df_all[col] = np.nan
        if col not in new_rows.columns:
            new_rows[col] = np.nan

    if not new_rows.empty:
        df_all = pd.concat([df_all, new_rows[df_all.columns]], ignore_index=True)
        print(f"[INFO] Added {len(new_rows)} upcoming rows from today's predictions.")

# recompute helper cols
df_all["game_day"] = df_all["date"].dt.date

upcoming_mask = (
    (~df_all["is_played"]) &
    (
        (df_all["game_day"] == today_date) |
        (df_all["game_day"] == tomorrow_date)
    )
)

print("[INFO] Upcoming games (today/tomorrow):", int(upcoming_mask.sum()))

########################################
# 3. FIT ISOTONIC CALIBRATION ON PLAYED GAMES
########################################

hist_mask = df_all["is_played"] & df_all["home_team_prob"].notna()
hist_calib = df_all.loc[hist_mask, ["home_team_prob","win"]].copy()

if hist_calib.empty or hist_calib["win"].nunique() < 2:
    print("[WARN] Not enough completed games to calibrate isotonic. Using raw probs.")
    df_all["prob_iso"] = df_all["home_team_prob"]
    iso = None
else:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(hist_calib["home_team_prob"].values, hist_calib["win"].values)
    df_all["prob_iso"] = iso.transform(df_all["home_team_prob"].values)

########################################
# 4. BUILD HOME WIN RATES (last 20 games logic)
########################################

# We compute rolling-ish home strength and save it daily for transparency,
# exactly like you were doing manually.

df_for_hwr = df_all[["date","home_team","away_team","result"]].copy()

def get_last_20_games_all_teams(df_games):
    team_results = {}
    # we assume df_games has full season (today inclusive),
    # so we just slice last 20 team games by date each call.
    for team in df_games["home_team"].dropna().unique():
        team_games = df_games[
            (df_games["home_team"] == team) |
            (df_games["away_team"] == team)
        ].sort_values("date", ascending=False).head(20)

        home_games = team_games[team_games["home_team"] == team]

        total_home_games = len(home_games)
        home_wins = (home_games["result"] == team).sum()
        home_win_rate = round(home_wins / total_home_games, 2) if total_home_games > 0 else 0.0

        team_results[team] = {
            "Total Last 20 Games": len(team_games),
            "Total Home Games": total_home_games,
            "Home Wins": home_wins,
            "Home Win Rate": home_win_rate
        }

    out_df = pd.DataFrame.from_dict(team_results, orient="index")
    out_df.sort_values(by="Home Win Rate", ascending=False, inplace=True)
    return out_df

hwr_df = get_last_20_games_all_teams(df_all)
os.makedirs(BASE_DIR, exist_ok=True)
hwr_df.to_csv(HWR_FILE, index=True, encoding="utf-8")
print(f"[INFO] Saved home win rate table -> {HWR_FILE}")

# attach map to df_all
df_all["home_win_rate"] = df_all["home_team"].map(hwr_df["Home Win Rate"])

########################################
# 5. GRID SEARCH BEST PARAMS ON HISTORY
########################################

hist_df = df_all[df_all["is_played"]].copy()

best_profit = float("-inf")
best_params = None
best_subset = None

for hw_cut in HOMEWR_MIN_GRID:
    strong_homes = hwr_df[hwr_df["Home Win Rate"] >= hw_cut].index.tolist()

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
    print("[WARN] No profitable parameter combo found in history.")
else:
    total_stake = best_params["n_trades"] * FLAT_STAKE
    roi_pct = (best_profit / total_stake * 100.0) if total_stake else 0.0

    print("=== BEST PARAMS (ISOTONIC, HISTORICAL) ===")
    print(best_params)
    print(f"total_profit €: {best_profit:.2f}, ROI%: {roi_pct:.2f}")

########################################
# 6. SAVE FULL HISTORICAL QUALIFIED BETS (BANKROLL TRACK)
########################################

hist_sel = pd.DataFrame()
if best_params is not None:
    mask_hist = (
        (hist_df["home_win_rate"] >= best_params["home_win_rate_threshold"]) &
        (hist_df["odds_1"]      >= best_params["odds_min"]) &
        (hist_df["odds_1"]      <= best_params["odds_max"]) &
        (hist_df["prob_iso"]    >= best_params["prob_threshold"])
    )
    hist_sel = hist_df.loc[mask_hist].copy()

if hist_sel.empty:
    print("[INFO] No historical bets matched best_params; bet_history_strategy.csv will still be created but may be empty.")
    # still create empty file with headers
    empty_cols = [
        "date","home_team","away_team","bet_side",
        "home_win_rate","prob_iso","odds_1",
        "win","pnl","cum_profit"
    ]
    pd.DataFrame(columns=empty_cols).to_csv(HIST_EXPORT_PATH, index=False, encoding="utf-8")
else:
    hist_sel["pnl"] = np.where(
        hist_sel["win"] == 1,
        FLAT_STAKE * (hist_sel["odds_1"] - 1.0),
        -FLAT_STAKE
    )
    hist_sel["bet_side"] = hist_sel["home_team"]

    hist_sel = hist_sel.sort_values("date").reset_index(drop=True)
    hist_sel["cum_profit"] = hist_sel["pnl"].cumsum()

    out_cols = [
        "date","home_team","away_team","bet_side",
        "home_win_rate","prob_iso","odds_1",
        "win","pnl","cum_profit"
    ]
    hist_to_save = (
        hist_sel[out_cols]
        .round({
            "home_win_rate":3,
            "prob_iso":3,
            "odds_1":3,
            "pnl":2,
            "cum_profit":2
        })
    )
    hist_to_save.to_csv(HIST_EXPORT_PATH, index=False, encoding="utf-8")
    print(f"[INFO] Saved betting history -> {HIST_EXPORT_PATH} ({len(hist_to_save)} rows).")

########################################
# 7. BUILD TONIGHT'S SHORTLIST + SAVE CARD
########################################

card = pd.DataFrame()

if best_params is not None:
    # upcoming within next 36h based on timestamp
    now_norm = pd.Timestamp.utcnow().normalize()
    cutoff   = now_norm + pd.Timedelta(hours=LOOKAHEAD_HRS)

    upcoming = df_all[
        (~df_all["is_played"]) &
        (df_all["date"] >= now_norm) &
        (df_all["date"] <= cutoff)
    ].copy()

    # if no direct future rows (early season edge case),
    # fall back purely to today's predictions file (PRED_TODAY)
    if upcoming.empty and os.path.exists(PRED_TODAY):
        tmp_pred = pd.read_csv(PRED_TODAY)
        tmp_pred.columns = (
            tmp_pred.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r"\s+","_", regex=True)
        )

        tmp_pred["home_team_prob"] = to_float_series(tmp_pred.get("home_team_prob", np.nan))
        tmp_pred["odds_1"]         = to_float_series(tmp_pred.get("odds_1", np.nan))
        tmp_pred["odds_2"]         = to_float_series(tmp_pred.get("odds_2", np.nan))

        tmp_pred["date"] = pd.to_datetime(tmp_pred.get("date", pd.NaT), errors="coerce")
        tmp_pred.loc[tmp_pred["date"].isna(), "date"] = pd.Timestamp(today_date)

        # attach home_win_rate from hwr_df
        tmp_pred["home_win_rate"] = tmp_pred["home_team"].map(hwr_df["Home Win Rate"])

        # calibrated
        if "prob_iso" not in tmp_pred.columns:
            if "home_team_prob" in tmp_pred.columns and iso is not None:
                tmp_pred["prob_iso"] = iso.transform(tmp_pred["home_team_prob"].values)
            else:
                tmp_pred["prob_iso"] = tmp_pred.get("home_team_prob", np.nan)

        tmp_pred["is_played"] = False

        needed_cols = [
            "date","home_team","away_team",
            "home_team_prob","prob_iso",
            "home_win_rate","odds_1","odds_2",
            "is_played"
        ]
        for c in needed_cols:
            if c not in tmp_pred.columns:
                tmp_pred[c] = np.nan

        upcoming = tmp_pred[needed_cols].copy()

    if not upcoming.empty:
        mask_card = (
            (upcoming["home_win_rate"] >= best_params["home_win_rate_threshold"]) &
            (upcoming["odds_1"]      >= best_params["odds_min"]) &
            (upcoming["odds_1"]      <= best_params["odds_max"]) &
            (upcoming["prob_iso"]    >= best_params["prob_threshold"])
        )
        card = upcoming.loc[mask_card].copy()

        if not card.empty:
            card["EV_€_per_100"] = (
                card["prob_iso"] * (card["odds_1"] - 1.0)
                - (1.0 - card["prob_iso"])
            ) * FLAT_STAKE

            export_cols = [
                "date",
                "home_team",
                "away_team",
                "home_win_rate",
                "prob_iso",
                "odds_1",
                "EV_€_per_100"
            ]

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

            card_to_save.to_csv(CARD_OUT_PATH, index=False, encoding="utf-8")
            print(f"[INFO] Saved tonight's shortlist -> {CARD_OUT_PATH} ({len(card_to_save)} rows).")
        else:
            # create empty shortlist file so workflow still commits something predictable
            pd.DataFrame(columns=[
                "date","home_team","away_team",
                "home_win_rate","prob_iso","odds_1","EV_€_per_100"
            ]).to_csv(CARD_OUT_PATH, index=False, encoding="utf-8")
            print("[INFO] No qualifying games for tonight; wrote empty shortlist.")
    else:
        # also emit an empty file for stability
        pd.DataFrame(columns=[
            "date","home_team","away_team",
            "home_win_rate","prob_iso","odds_1","EV_€_per_100"
        ]).to_csv(CARD_OUT_PATH, index=False, encoding="utf-8")
        print("[INFO] No upcoming games at all; wrote empty shortlist.")
else:
    # if best_params is None we still want predictable output
    pd.DataFrame(columns=[
        "date","home_team","away_team",
        "home_win_rate","prob_iso","odds_1","EV_€_per_100"
    ]).to_csv(CARD_OUT_PATH, index=False, encoding="utf-8")
    print("[WARN] No best_params -> empty shortlist written anyway.")

########################################
# 8. UPDATE / APPEND bet_log.csv
########################################

# We'll:
# - load existing BET_LOG_PATH if present
# - append today's card (qualified bets)
# - dedupe
# - save back
# - then run auto-settlement using df_all results

if os.path.exists(BET_LOG_PATH):
    bet_log_old = pd.read_csv(BET_LOG_PATH)
    # make sure date is parsed when we need it later
    if "date" in bet_log_old.columns:
        bet_log_old["date"] = pd.to_datetime(bet_log_old["date"], errors="coerce")
else:
    bet_log_old = pd.DataFrame()

if not card.empty:
    to_log = card.copy().reset_index(drop=True)

    # columns we want to keep in bet log
    to_log = to_log[[
        "date",
        "home_team",
        "away_team",
        "home_win_rate",
        "prob_iso",
        "odds_1"
    ]].copy()

    to_log["stake_eur"]      = YOUR_STAKE_EUR
    to_log["actual_winner"]  = ""
    to_log["won"]            = np.nan
    to_log["payout_eur"]     = np.nan
    to_log["net_profit_eur"] = np.nan

    bet_log_combined = pd.concat([bet_log_old, to_log], ignore_index=True)

    bet_log_combined.drop_duplicates(
        subset=["date","home_team","away_team","odds_1"],
        keep="last",
        inplace=True
    )
else:
    bet_log_combined = bet_log_old.copy()

# settlement pass:
# join bet_log_combined with df_all (which has results+odds)
df_for_settle = df_all.copy()
df_for_settle["date_only"] = df_for_settle["date"].dt.date

if "date" in bet_log_combined.columns:
    bet_log_combined["date"] = pd.to_datetime(bet_log_combined["date"], errors="coerce")
bet_log_combined["date_only"] = bet_log_combined["date"].dt.date

merged = bet_log_combined.merge(
    df_for_settle[["date_only","home_team","result","odds_1"]].rename(
        columns={"odds_1":"odds_latest"}
    ),
    on=["date_only","home_team"],
    how="left"
)

# fill in actual_winner if result is known
merged["actual_winner"] = np.where(
    merged["result"].notna() & (merged["result"] != "0"),
    merged["result"],
    merged.get("actual_winner", "")
)

# compute won / payout / net_profit
merged["won"] = np.where(
    merged["actual_winner"].notna() &
    (merged["actual_winner"] == merged["home_team"]),
    1.0,
    np.where(
        merged["actual_winner"].notna(),
        0.0,
        np.nan
    )
)

# use stake_eur * odds_1 from the log row, fallback to odds_latest from df_all if odds_1 missing
merged["odds_effective"] = np.where(
    merged["odds_1"].notna(),
    merged["odds_1"],
    merged["odds_latest"]
)

merged["payout_eur"] = np.where(
    merged["won"] == 1.0,
    merged["stake_eur"] * merged["odds_effective"],
    np.where(
        merged["won"] == 0.0,
        0.0,
        np.nan
    )
)

merged["net_profit_eur"] = np.where(
    merged["won"] == 1.0,
    merged["stake_eur"] * (merged["odds_effective"] - 1.0),
    np.where(
        merged["won"] == 0.0,
        -merged["stake_eur"],
        np.nan
    )
)

# drop helper cols and save
final_cols = [
    "date",
    "home_team",
    "away_team",
    "home_win_rate",
    "prob_iso",
    "odds_1",
    "stake_eur",
    "actual_winner",
    "won",
    "payout_eur",
    "net_profit_eur"
]
merged = merged.sort_values("date").reset_index(drop=True)
merged[final_cols].to_csv(BET_LOG_PATH, index=False, encoding="utf-8")

print(f"[INFO] Updated bet log -> {BET_LOG_PATH}")
print("[INFO] Tail of bet_log after update/settle:")
print(merged[final_cols].tail(10).to_string(index=False))

########################################
# 9. DONE
########################################

print("[INFO] Script 5 completed successfully.")
