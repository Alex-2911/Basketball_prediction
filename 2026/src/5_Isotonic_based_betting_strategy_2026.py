#!/usr/bin/env python
# coding: utf-8

####################################################################################################
# SCRIPT 5 — ISOTONIC CALIBRATED BETTING ENGINE (DAILY DRIVER)
#
# What this script does end-to-end:
#
# 0.  Sync bet_log_live.csv with yesterday's shortlist:
#       - Append yesterday's picks with default stake_eur (50€)
#       - Deduplicate (date, home_team, away_team)
#       - Backup bet_log_live_<today>.csv
#
# 1.  Settle results:
#       - Look up actual winners from the latest combined_nba_predictions_acc_*.csv
#       - Mark won = 1/0 in bet_log_live.csv
#
# 2.  Performance stats:
#       - Compute bankroll evolution and ROI
#       - Save detailed XLSX snapshot
#
# 3+. Betting model prep for tonight:
#       - Load historical + today's games
#       - Isotonic calibration
#       - Home win rate calc
#       - Grid search best_params
#       - Kelly sizing on upcoming slate (next 36h)
#       - Save bet_shortlist_<today>.csv with stake_eur and param context
#
####################################################################################################

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

########################
# PATH HANDLING
########################

# Try local Windows path first. If not found (e.g. on GitHub Actions),
# fall back to repo-relative path.
LOCAL_BASE = r"D:\1. Python\1. NBA Script\2026\LightGBM"
REPO_BASE  = os.path.join(os.getcwd(), "2026", "output", "LightGBM")

if os.path.exists(LOCAL_BASE):
    BASE_DIR = LOCAL_BASE
else:
    BASE_DIR = REPO_BASE

os.makedirs(BASE_DIR, exist_ok=True)

print(f"[INFO] Using BASE_DIR = {BASE_DIR}")

########################
# COMMON DATES
########################

now_dt        = datetime.now()
today_date    = now_dt.date()
tomorrow_date = (now_dt + timedelta(days=1)).date()

today_str     = now_dt.strftime("%Y-%m-%d")
yesterday_str = (now_dt - timedelta(days=1)).strftime("%Y-%m-%d")

########################
# FILE PATHS
########################

LIVE_LOG_PATH      = os.path.join(BASE_DIR, "bet_log_live.csv")
BACKUP_LOG_PATH    = os.path.join(BASE_DIR, f"bet_log_live_{today_str}.csv")
SHORTLIST_PATH     = os.path.join(BASE_DIR, f"bet_shortlist_{yesterday_str}.csv")
CARD_OUT_PATH      = os.path.join(BASE_DIR, f"bet_shortlist_{today_str}.csv")
SUMMARY_XLSX_PATH  = os.path.join(BASE_DIR, f"betting_summary_{today_str}.xlsx")

COMBINED_FILE      = os.path.join(BASE_DIR, f"combined_nba_predictions_acc_{today_str}.csv")
HWR_FILE           = os.path.join(BASE_DIR, f"home_win_rates_sorted_{today_str}.csv")
TODAY_PRED         = os.path.join(BASE_DIR, f"nba_games_predict_{today_str}.csv")

########################
# GLOBAL CONFIG
########################

DEFAULT_STAKE_EUR       = 50.0           # default log stake when adding yesterday's bets
STARTING_BANKROLL       = 1000.0         # baseline bankroll for summary calc
FLAT_STAKE_BACKTEST     = 100.0          # virtual flat stake in grid search backtest
LOOKAHEAD_HRS           = 36             # consider next 36h for tonight's slate
KELLY_FRACTION          = 0.5            # 0.5 Kelly by default
MAX_RISK_PCT_PER_BET    = 0.05           # cap each bet at 5% bankroll
MIN_STAKE_ABS           = 10.0           # round down micro bets under 10€
TARGET_COLS = [
    "date",
    "home_team",
    "away_team",
    "home_win_rate",
    "prob_iso",
    "odds_1",
    "stake_eur",
    "won",
]

ODDS_MIN_GRID   = np.arange(1.1, 3.1, 0.1)
ODDS_MAX_GRID   = np.arange(1.2, 3.6, 0.1)
PROB_MIN_GRID   = np.arange(0.40, 0.90, 0.05)
HOMEWR_MIN_GRID = np.arange(0.50, 0.90, 0.05)

########################
# HELPERS
########################

def _clean_numeric(col):
    """ Convert things like '2,30' -> 2.30, keep NaN if empty. """
    return (
        col.astype(str)
           .str.replace(",", ".", regex=False)
           .str.replace("[^0-9.]", "", regex=True)
           .replace("", np.nan)
           .astype(float)
    )

def _coerce_prob_iso(series: pd.Series) -> pd.Series:
    """
    Fix crazy values like 875 meaning 0.875:
    - if 1 < x <= 1000, assume divide by 1000
    """
    def fix(v):
        try:
            if pd.isna(v):
                return np.nan
            v = float(v)
        except Exception:
            return np.nan
        if v > 1.0 and v <= 1000.0:
            return v / 1000.0
        return v
    return series.apply(fix)

def _normalize_existing_live(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean existing bet_log_live.csv """
    if df is None or df.empty:
        return pd.DataFrame(columns=TARGET_COLS)

    out = df.copy()
    out.columns = (
        out.columns
           .str.strip()
           .str.lower()
           .str.replace(r"\s+", "_", regex=True)
    )

    if "stake" in out.columns and "stake_eur" not in out.columns:
        out = out.rename(columns={"stake": "stake_eur"})
    if "win" in out.columns and "won" not in out.columns:
        out = out.rename(columns={"win": "won"})

    keep_cols = [c for c in TARGET_COLS if c in out.columns]
    out = out[keep_cols].copy()
    for col in TARGET_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["date"]          = pd.to_datetime(out["date"], errors="coerce")
    out["home_win_rate"] = _clean_numeric(out["home_win_rate"])
    out["prob_iso"]      = _coerce_prob_iso(_clean_numeric(out["prob_iso"]))
    out["odds_1"]        = _clean_numeric(out["odds_1"])
    out["stake_eur"]     = _clean_numeric(out["stake_eur"])
    out["won"]           = pd.to_numeric(out["won"], errors="coerce")

    out = out[TARGET_COLS].sort_values("date").reset_index(drop=True)
    return out

def _normalize_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert yesterday's bet_shortlist into rows we append to the log. """
    if df is None or df.empty:
        return pd.DataFrame(columns=TARGET_COLS)

    raw = df.copy()
    raw.columns = (
        raw.columns
           .str.strip()
           .str.lower()
           .str.replace(r"\s+", "_", regex=True)
    )

    needed = ["date","home_team","away_team","home_win_rate","prob_iso","odds_1"]
    for col in needed:
        if col not in raw.columns:
            raw[col] = np.nan

    out = pd.DataFrame({
        "date":          pd.to_datetime(raw["date"], errors="coerce"),
        "home_team":     raw["home_team"].astype(str),
        "away_team":     raw["away_team"].astype(str),
        "home_win_rate": _clean_numeric(raw["home_win_rate"]),
        "prob_iso":      _coerce_prob_iso(_clean_numeric(raw["prob_iso"])),
        "odds_1":        _clean_numeric(raw["odds_1"]),
        "stake_eur":     float(DEFAULT_STAKE_EUR),  # <- default 50€
        "won":           np.nan                    # not settled yet
    })

    out = out.sort_values("date").reset_index(drop=True)
    return out

def load_live_log():
    if os.path.exists(LIVE_LOG_PATH):
        raw = pd.read_csv(LIVE_LOG_PATH)
    else:
        raw = pd.DataFrame()
    return _normalize_existing_live(raw)

def backup_live_log(df: pd.DataFrame):
    df.to_csv(BACKUP_LOG_PATH, index=False)

def append_yesterday_shortlist(live_df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(SHORTLIST_PATH):
        sh_raw  = pd.read_csv(SHORTLIST_PATH)
        sh_norm = _normalize_shortlist(sh_raw)
    else:
        sh_norm = pd.DataFrame(columns=TARGET_COLS)

    combined = pd.concat([live_df, sh_norm], ignore_index=True)
    combined.drop_duplicates(
        subset=["date","home_team","away_team"],
        keep="last",
        inplace=True
    )
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined

def settle_results_in_log(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    Update 'won' using the most recent combined_nba_predictions_acc_*.csv.
    """
    pattern = os.path.join(BASE_DIR, "combined_nba_predictions_acc_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("[WARN] No combined_nba_predictions_acc_*.csv found. Skipping settlement.")
        return df_log
    latest_results_path = files[-1]
    print(f"[INFO] Settling results using {os.path.basename(latest_results_path)}")

    df_res = pd.read_csv(latest_results_path, encoding="utf-7")

    # normalize cols in both
    for d in [df_log, df_res]:
        d.columns = (
            d.columns
             .str.strip()
             .str.lower()
             .str.replace(r"\s+", "_", regex=True)
        )

    df_log["date"] = pd.to_datetime(df_log["date"], errors="coerce").dt.date
    df_res["date"] = pd.to_datetime(df_res["date"], errors="coerce").dt.date

    if "result" not in df_res.columns:
        print("[WARN] 'result' column missing in combined file. Skipping settlement.")
        return df_log

    df_res_small = df_res[["date","home_team","away_team","result"]].dropna()

    merged = pd.merge(
        df_log,
        df_res_small,
        how="left",
        on=["date","home_team","away_team"]
    )

    def decide_won(r):
        if pd.isna(r.get("result")):
            return r.get("won", np.nan)
        if r["result"] == r["home_team"]:
            return 1
        if r["result"] == r["away_team"]:
            return 0
        return np.nan

    merged["won"] = merged.apply(decide_won, axis=1)
    merged = merged.drop(columns=["result"])

    # normalize back to TARGET_COLS/ types
    merged = _normalize_existing_live(merged)
    return merged

def compute_and_print_stats(df_log: pd.DataFrame):
    """
    KPI summary + bankroll evolution.
    Saves XLSX snapshot for record keeping.
    """
    df = df_log.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["odds_1"] = pd.to_numeric(df["odds_1"], errors="coerce")
    df["stake_eur"] = pd.to_numeric(df["stake_eur"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce")

    # profit per bet
    df["profit_eur"] = np.where(
        df["won"] == 1,
        df["stake_eur"] * (df["odds_1"] - 1.0),
        np.where(df["won"] == 0, -df["stake_eur"], np.nan)
    )

    df = df.sort_values("date").reset_index(drop=True)
    df["cum_profit"] = df["profit_eur"].fillna(0).cumsum()
    df["bankroll"]   = STARTING_BANKROLL + df["cum_profit"]

    # summary
    settled = df[df["won"].isin([0,1])].copy()
    total_bets = len(settled)
    wins = settled["won"].sum()
    losses = total_bets - wins
    win_rate = (wins / total_bets * 100.0) if total_bets else 0.0
    total_staked = settled["stake_eur"].sum() if total_bets else 0.0
    total_profit = settled["profit_eur"].sum() if total_bets else 0.0
    roi = (total_profit / total_staked * 100.0) if total_staked else 0.0
    final_bankroll = df["bankroll"].iloc[-1] if len(df) else STARTING_BANKROLL

    print("\n📊 BETTING PERFORMANCE SUMMARY")
    print("──────────────────────────────")
    print(f"Total bets made     : {total_bets}")
    print(f"Wins / Losses       : {int(wins)} / {int(losses)}")
    print(f"Win rate (%)        : {win_rate:.2f}")
    print(f"Total staked (€)    : {total_staked:.2f}")
    print(f"Total profit (€)    : {total_profit:.2f}")
    print(f"ROI (%)             : {roi:.2f}")
    print(f"Final bankroll (€)  : {final_bankroll:.2f}")

    # bankroll chart (still fine in local, Actions will render as artifact logs only)
    plt.figure(figsize=(8,4))
    plt.plot(df["date"], df["bankroll"], marker="o", linewidth=2)
    plt.title("🏀 Bankroll Evolution (Actual Bets)")
    plt.xlabel("Date")
    plt.ylabel("Bankroll (€)")
    plt.grid(True)
    plt.tight_layout()

    # save xlsx snapshot
    df.to_excel(SUMMARY_XLSX_PATH, index=False)
    print(f"\n💾 Detailed results saved to: {SUMMARY_XLSX_PATH}")

    return final_bankroll  # we can reuse as current bankroll for Kelly

def to_float_series(s):
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace("[^0-9.]", "", regex=True)
         .replace("", np.nan)
         .astype(float)
    )

def kelly_fraction_row(p, odds_decimal):
    """
    Kelly fraction for single outcome bet.
    f* = (p*odds - 1) / (odds - 1)
    """
    if pd.isna(p) or pd.isna(odds_decimal):
        return np.nan
    b = odds_decimal - 1.0
    if b <= 0:
        return np.nan
    numer = p * odds_decimal - 1.0
    denom = b
    return numer / denom

def load_today_predictions_safe(path_csv, iso_model, hwr_df):
    """Fallback if df_all has no upcoming games."""
    if not os.path.exists(path_csv):
        return pd.DataFrame()

    tmp = pd.read_csv(
        path_csv,
        encoding="utf-7",
        sep=",",
        quotechar='"',
        decimal=","
    )

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

    tmp.columns = (
        tmp.columns
           .str.strip()
           .str.lower()
           .str.replace(r"\s+","_", regex=True)
    )

    def _to_float_series(s):
        return (
            s.astype(str)
             .str.replace(",", ".", regex=False)
             .str.replace("[^0-9.]", "", regex=True)
             .replace("", np.nan)
             .astype(float)
        )

    if "home_team_prob" in tmp.columns:
        tmp["home_team_prob"] = _to_float_series(tmp["home_team_prob"])
    else:
        tmp["home_team_prob"] = np.nan

    if "odds_1" in tmp.columns:
        tmp["odds_1"] = _to_float_series(tmp["odds_1"])
    else:
        tmp["odds_1"] = np.nan

    if "odds_2" in tmp.columns:
        tmp["odds_2"] = _to_float_series(tmp["odds_2"])
    else:
        tmp["odds_2"] = np.nan

    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    else:
        tmp["date"] = pd.NaT

    tmp["home_win_rate"] = tmp["home_team"].map(hwr_df["Home Win Rate"])

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

########################################
# MAIN FLOW
########################################

def main():

    # --------------------------------------------------------------------------------
    # 0. SYNC LOG
    # --------------------------------------------------------------------------------
    live_df = load_live_log()
    backup_live_log(live_df)

    live_df = append_yesterday_shortlist(live_df)

    live_df.to_csv(LIVE_LOG_PATH, index=False)
    print("✅ bet_log_live.csv synchronized with yesterday's shortlist.")
    print(f"   LIVE LOG PATH    : {LIVE_LOG_PATH}")
    print(f"   DAILY BACKUP PATH: {BACKUP_LOG_PATH}")
    print(f"   Yesterday picks source (if existed): {SHORTLIST_PATH}")
    print(f"Total rows now: {len(live_df)}")
    print("Columns      :", list(live_df.columns))
    print(live_df.tail(min(20, len(live_df))).to_string(index=False))

    # --------------------------------------------------------------------------------
    # 1. SETTLE RESULTS
    # --------------------------------------------------------------------------------
    live_df = settle_results_in_log(live_df)
    live_df.to_csv(LIVE_LOG_PATH, index=False)
    print("\n✅ Updated bet_log_live.csv with actual game outcomes.")
    print(live_df.tail(min(10, len(live_df))).to_string(index=False))

    # --------------------------------------------------------------------------------
    # 2. KPIs / BANKROLL
    # --------------------------------------------------------------------------------
    bankroll_after = compute_and_print_stats(live_df)

    # --------------------------------------------------------------------------------
    # 3. MODEL PREP FOR TONIGHT (Sections 4-9 in your notebook)
    # --------------------------------------------------------------------------------

    # 3.1 Load combined historical/today file
    if not os.path.exists(COMBINED_FILE):
        raise FileNotFoundError(f"Missing combined file: {COMBINED_FILE}")

    df_all = pd.read_csv(COMBINED_FILE, encoding="utf-7", decimal=",")
    df_all.columns = (
        df_all.columns
             .str.strip()
             .str.lower()
             .str.replace(r"\s+","_", regex=True)
    )

    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    if "odds_1" in df_all.columns:
        df_all["odds_1"] = to_float_series(df_all["odds_1"])
    if "odds_2" in df_all.columns:
        df_all["odds_2"] = to_float_series(df_all["odds_2"])
    df_all["home_team_prob"] = to_float_series(df_all["home_team_prob"])

    df_all["win"] = (df_all["result"] == df_all["home_team"]).astype(int)

    df_all["is_played"] = (
        df_all["result"].notna() &
        (df_all["result"].astype(str) != "0")
    )

    if "away_team" not in df_all.columns:
        df_all["away_team"] = np.nan

    # merge in today's preds df (anti-duplicate)
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
            tmp = pd.read_csv(
                TODAY_PRED,
                encoding="utf-7",
                sep=",",
                quotechar='"',
                decimal=",",
                header=None,
                names=["home_team","away_team","home_team_prob","odds_1","odds_2","result","date"]
            )

        tmp.columns = (
            tmp.columns
               .str.strip()
               .str.lower()
               .str.replace(r"\s+","_", regex=True)
        )

        tmp["home_team_prob"] = to_float_series(tmp.get("home_team_prob", np.nan))
        tmp["odds_1"]         = to_float_series(tmp.get("odds_1", np.nan))
        tmp["odds_2"]         = to_float_series(tmp.get("odds_2", np.nan))
        tmp["date"]           = pd.to_datetime(tmp.get("date", pd.NaT), errors="coerce")
        tmp.loc[tmp["date"].isna(), "date"] = pd.Timestamp(today_date)

        tmp["win"]       = np.nan
        tmp["is_played"] = False

        if "away_team" not in df_all.columns:
            df_all["away_team"] = np.nan

        key_cols = ["date","home_team","away_team"]
        merged_keys = df_all[key_cols].drop_duplicates()

        tmp_merge = tmp.merge(
            merged_keys,
            on=key_cols,
            how="left",
            indicator=True
        )
        new_rows = tmp_merge[tmp_merge["_merge"] == "left_only"].drop(columns=["_merge"])

        needed_cols = set(df_all.columns) | set(new_rows.columns)
        for col in needed_cols:
            if col not in df_all.columns:
                df_all[col] = np.nan
            if col not in new_rows.columns:
                new_rows[col] = np.nan

        new_rows["is_played"] = False

        if not new_rows.empty:
            df_all = pd.concat([df_all, new_rows[df_all.columns]], ignore_index=True)

    # 3.2 upcoming count
    df_all["game_day"] = df_all["date"].dt.date
    upcoming_mask = (
        (~df_all["is_played"]) &
        (
            (df_all["game_day"] == today_date) |
            (df_all["game_day"] == tomorrow_date)
        )
    )
    n_upcoming = int(upcoming_mask.sum())
    print("\n[INFO] Rows total:", len(df_all))
    print("[INFO] Completed games:", int(df_all["is_played"].sum()))
    print("[INFO] Upcoming games (today/tomorrow):", n_upcoming)
    if n_upcoming > 0:
        preview_cols = [
            "date","home_team","away_team",
            "home_team_prob","odds_1","odds_2","is_played"
        ]
        print(
            df_all.loc[upcoming_mask, preview_cols]
                  .sort_values("date")
                  .round({"home_team_prob":3,"odds_1":3,"odds_2":3})
                  .to_string(index=False)
        )

    # 3.3 ISOTONIC CALIBRATION
    hist_mask = df_all["is_played"] & df_all["home_team_prob"].notna()
    hist_calib = df_all.loc[hist_mask, ["home_team_prob","win"]].copy()

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

    # 3.4 HOME WIN RATE rolling (last 20 games home form)
    df_tmp = pd.read_csv(COMBINED_FILE, encoding="utf-7")
    df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce")

    def get_last_20_games_all_teams(dfsrc):
        team_results = {}
        for team in dfsrc["home_team"].unique():
            team_games = dfsrc[(dfsrc["home_team"] == team) | (dfsrc["away_team"] == team)]
            team_games = team_games.sort_values("date", ascending=False).head(20)

            home_games = team_games[team_games["home_team"] == team]

            total_home_games = len(home_games)
            home_wins = len(home_games[home_games["result"] == team])
            home_win_rate = round(home_wins / total_home_games, 2) if total_home_games > 0 else 0

            team_results[team] = {
                "Total Last 20 Games": len(team_games),
                "Total Home Games": total_home_games,
                "Home Wins": home_wins,
                "Home Win Rate": home_win_rate
            }
        out_df = pd.DataFrame.from_dict(team_results, orient="index")
        out_df.sort_values(by="Home Win Rate", ascending=False, inplace=True)
        return out_df

    home_win_rates_all_teams_sorted = get_last_20_games_all_teams(df_tmp)
    home_win_rates_all_teams_sorted.to_csv(HWR_FILE, index=True)
    print(f"\n📁 Sorted home win rates saved to: {HWR_FILE}")

    # attach home_win_rate to df_all
    hwr = pd.read_csv(HWR_FILE, index_col=0)
    hwr.columns = [c.strip() for c in hwr.columns]  # expect "Home Win Rate"
    df_all["home_win_rate"] = df_all["home_team"].map(hwr["Home Win Rate"])

    # 3.5 GRID SEARCH BEST PARAMS
    hist_df = df_all[df_all["is_played"]].copy()
    best_profit = float("-inf")
    best_params = None

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

                    subset["pnl"] = np.where(
                        subset["win"] == 1,
                        FLAT_STAKE_BACKTEST * (subset["odds_1"] - 1.0),
                        -FLAT_STAKE_BACKTEST
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
                        }

    print("\n=== BEST PARAMS (ISOTONIC, HISTORICAL) ===")
    if best_params is None:
        print("None found")
        return
    else:
        print(f"home_win_rate_threshold : {best_params['home_win_rate_threshold']}")
        print(f"odds_min                : {best_params['odds_min']}")
        print(f"odds_max                : {best_params['odds_max']}")
        print(f"prob_threshold (iso)    : {best_params['prob_threshold']}")

    # 3.6 SHORTLIST NEXT 36H + KELLY
    now_norm = pd.Timestamp.now().normalize()
    cutoff   = now_norm + pd.Timedelta(hours=LOOKAHEAD_HRS)

    upcoming = df_all[
        (~df_all["is_played"]) &
        (df_all["date"] >= now_norm) &
        (df_all["date"] <= cutoff)
    ].copy()

    if upcoming.empty:
        # fallback: TODAY_PRED
        hwr_df_local = pd.read_csv(HWR_FILE, index_col=0)
        hwr_df_local.columns = [c.strip() for c in hwr_df_local.columns]
        iso_model = None if 'iso' not in locals() else iso
        upcoming = load_today_predictions_safe(TODAY_PRED, iso_model, hwr_df_local)

    if upcoming.empty:
        print("\n=== TONIGHT'S SHORTLIST (ISOTONIC + KELLY) ===")
        print("No upcoming games found in df_all or today's prediction file.")
        return

    mask_card = (
        (upcoming["home_win_rate"] >= best_params["home_win_rate_threshold"]) &
        (upcoming["odds_1"]      >= best_params["odds_min"]) &
        (upcoming["odds_1"]      <= best_params["odds_max"]) &
        (upcoming["prob_iso"]    >= best_params["prob_threshold"])
    )
    card = upcoming.loc[mask_card].copy()

    # EV per 100 stake (still useful)
    card["EV_€_per_100"] = (
        card["prob_iso"] * (card["odds_1"] - 1.0)
        - (1.0 - card["prob_iso"])
    ) * FLAT_STAKE_BACKTEST

    # Kelly sizing
    current_bankroll = bankroll_after
    print(f"\n💰 Current bankroll for sizing bets: {current_bankroll:.2f} €")

    card["kelly_full"] = card.apply(
        lambda r: kelly_fraction_row(r["prob_iso"], r["odds_1"]),
        axis=1
    )

    card["kelly_fraction_used"] = (card["kelly_full"] * KELLY_FRACTION).clip(lower=0)
    card["kelly_fraction_used"] = card["kelly_fraction_used"].clip(upper=MAX_RISK_PCT_PER_BET)

    card["stake_eur"] = (card["kelly_fraction_used"] * current_bankroll).round(2)
    card.loc[card["stake_eur"] < MIN_STAKE_ABS, "stake_eur"] = 0.0

    # print shortlist
    print("\n=== TONIGHT'S SHORTLIST (ISOTONIC + KELLY) ===")
    if card.empty:
        print("No games match the tuned iso strategy in the next slate.")
    else:
        cols_card_print = [
            "date",
            "home_team","away_team",
            "home_win_rate",
            "prob_iso",
            "odds_1",
            "EV_€_per_100",
            "kelly_full",
            "kelly_fraction_used",
            "stake_eur"
        ]
        print(
            card[cols_card_print]
            .sort_values("date")
            .round({
                "home_win_rate":3,
                "prob_iso":3,
                "odds_1":3,
                "EV_€_per_100":2,
                "kelly_full":3,
                "kelly_fraction_used":3,
                "stake_eur":2
            })
            .to_string(index=False)
        )

    # reasons diag
    reasons = []
    for _, r in upcoming.iterrows():
        r_reasons = []
        if pd.notna(r["home_win_rate"]) and r["home_win_rate"] < best_params["home_win_rate_threshold"]:
            r_reasons.append(f"home_win_rate {r['home_win_rate']:.2f} < {best_params['home_win_rate_threshold']}")
        if pd.notna(r["odds_1"]) and r["odds_1"] < best_params["odds_min"]:
            r_reasons.append(f"odds {r['odds_1']:.2f} < min {best_params['odds_min']}")
        if pd.notna(r["odds_1"]) and r["odds_1"] > best_params["odds_max"]:
            r_reasons.append(f"odds {r['odds_1']:.2f} > max {best_params['odds_max']}")
        if pd.notna(r["prob_iso"]) and r["prob_iso"] < best_params["prob_threshold"]:
            r_reasons.append(f"prob_iso {r['prob_iso']:.2f} < {best_params['prob_threshold']}")
        if not r_reasons:
            r_reasons.append("QUALIFIES")
        reasons.append("; ".join(r_reasons))

    diag = upcoming.copy()
    diag["why_not"] = reasons
    print("\n=== ALL UPCOMING GAMES & FILTER REASONS ===")
    print(
        diag[["date","home_team","away_team","home_win_rate","prob_iso","odds_1","why_not"]]
        .sort_values("date")
        .round({"home_win_rate":3,"prob_iso":3,"odds_1":3})
        .to_string(index=False)
    )

    # save shortlist snapshot for tomorrow logging
    print("\n=== TONIGHT'S SHORTLIST (SAVE SNAPSHOT) ===")
    if card.empty:
        print("No games match the tuned iso strategy in the next slate.")
    else:
        card_to_save = card.copy()
        card_to_save["param_home_win_rate_threshold"] = best_params["home_win_rate_threshold"]
        card_to_save["param_odds_min"]               = best_params["odds_min"]
        card_to_save["param_odds_max"]               = best_params["odds_max"]
        card_to_save["param_prob_threshold"]         = best_params["prob_threshold"]
        card_to_save["bankroll_at_bettime"]          = current_bankroll

        export_cols = [
            "date",
            "home_team",
            "away_team",
            "home_win_rate",
            "prob_iso",
            "odds_1",
            "EV_€_per_100",
            "kelly_full",
            "kelly_fraction_used",
            "stake_eur",
            "param_home_win_rate_threshold",
            "param_odds_min",
            "param_odds_max",
            "param_prob_threshold",
            "bankroll_at_bettime"
        ]

        card_to_save = (
            card_to_save[export_cols]
            .sort_values("date")
            .round({
                "home_win_rate":3,
                "prob_iso":3,
                "odds_1":3,
                "EV_€_per_100":2,
                "kelly_full":3,
                "kelly_fraction_used":3,
                "stake_eur":2,
                "bankroll_at_bettime":2
            })
        )

        card_to_save.to_csv(CARD_OUT_PATH, index=False)
        print(f"💾 Saved shortlist to {CARD_OUT_PATH}")


if __name__ == "__main__":
    main()
