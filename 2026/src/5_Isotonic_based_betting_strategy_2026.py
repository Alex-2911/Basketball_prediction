#!/usr/bin/env python
# coding: utf-8

####################################################################################################
# SCRIPT 5 — ISOTONIC CALIBRATED BETTING ENGINE (DAILY DRIVER)
####################################################################################################

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

########################
# PATH HANDLING (GitHub version only)
########################

# GitHub-safe path (no Windows local path)
REPO_BASE = os.path.join(os.getcwd(), "2026", "output", "LightGBM")
os.makedirs(REPO_BASE, exist_ok=True)

print(f"[INFO] Using BASE_DIR = {REPO_BASE}")

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

LIVE_LOG_PATH      = os.path.join(REPO_BASE, "bet_log_live.csv")
BACKUP_LOG_PATH    = os.path.join(REPO_BASE, f"bet_log_live_{today_str}.csv")
SHORTLIST_PATH     = os.path.join(REPO_BASE, f"bet_shortlist_{yesterday_str}.csv")
CARD_OUT_PATH      = os.path.join(REPO_BASE, f"bet_shortlist_{today_str}.csv")
SUMMARY_XLSX_PATH  = os.path.join(REPO_BASE, f"betting_summary_{today_str}.xlsx")

COMBINED_FILE      = os.path.join(REPO_BASE, f"combined_nba_predictions_acc_{today_str}.csv")
HWR_FILE           = os.path.join(REPO_BASE, f"home_win_rates_sorted_{today_str}.csv")
TODAY_PRED         = os.path.join(REPO_BASE, f"nba_games_predict_{today_str}.csv")

########################
# GLOBAL CONFIG
########################

DEFAULT_STAKE_EUR       = 50.0
STARTING_BANKROLL       = 1000.0
FLAT_STAKE_BACKTEST     = 100.0
LOOKAHEAD_HRS           = 36
KELLY_FRACTION          = 0.5
MAX_RISK_PCT_PER_BET    = 0.05
MIN_STAKE_ABS           = 10.0

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
    return (
        col.astype(str)
           .str.replace(",", ".", regex=False)
           .str.replace("[^0-9.]", "", regex=True)
           .replace("", np.nan)
           .astype(float)
    )

def _coerce_prob_iso(series: pd.Series) -> pd.Series:
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
        "stake_eur":     float(DEFAULT_STAKE_EUR),
        "won":           np.nan
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
    pattern = os.path.join(REPO_BASE, "combined_nba_predictions_acc_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("[WARN] No combined_nba_predictions_acc_*.csv found. Skipping settlement.")
        return df_log
    latest_results_path = files[-1]
    print(f"[INFO] Settling results using {os.path.basename(latest_results_path)}")
    df_res = pd.read_csv(latest_results_path, encoding="utf-7")
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
    merged = pd.merge(df_log, df_res_small, how="left", on=["date","home_team","away_team"])
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
    merged = _normalize_existing_live(merged)
    return merged

def compute_and_print_stats(df_log: pd.DataFrame):
    df = df_log.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["odds_1"] = pd.to_numeric(df["odds_1"], errors="coerce")
    df["stake_eur"] = pd.to_numeric(df["stake_eur"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce")
    df["profit_eur"] = np.where(
        df["won"] == 1,
        df["stake_eur"] * (df["odds_1"] - 1.0),
        np.where(df["won"] == 0, -df["stake_eur"], np.nan)
    )
    df = df.sort_values("date").reset_index(drop=True)
    df["cum_profit"] = df["profit_eur"].fillna(0).cumsum()
    df["bankroll"]   = STARTING_BANKROLL + df["cum_profit"]
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
    plt.figure(figsize=(8,4))
    plt.plot(df["date"], df["bankroll"], marker="o", linewidth=2)
    plt.title("🏀 Bankroll Evolution (Actual Bets)")
    plt.xlabel("Date")
    plt.ylabel("Bankroll (€)")
    plt.grid(True)
    plt.tight_layout()
    df.to_excel(SUMMARY_XLSX_PATH, index=False)
    print(f"\n💾 Detailed results saved to: {SUMMARY_XLSX_PATH}")
    return final_bankroll

def kelly_fraction_row(p, odds_decimal):
    if pd.isna(p) or pd.isna(odds_decimal):
        return np.nan
    b = odds_decimal - 1.0
    if b <= 0:
        return np.nan
    numer = p * odds_decimal - 1.0
    denom = b
    return numer / denom

########################################
# MAIN FLOW (unchanged structure)
########################################

def main():
    live_df = load_live_log()
    backup_live_log(live_df)
    live_df = append_yesterday_shortlist(live_df)
    live_df.to_csv(LIVE_LOG_PATH, index=False)
    live_df = settle_results_in_log(live_df)
    live_df.to_csv(LIVE_LOG_PATH, index=False)
    bankroll_after = compute_and_print_stats(live_df)

    if not os.path.exists(COMBINED_FILE):
        raise FileNotFoundError(f"Missing combined file: {COMBINED_FILE}")
    df_all = pd.read_csv(COMBINED_FILE, encoding="utf-7", decimal=",")
    df_all.columns = df_all.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all["odds_1"] = _clean_numeric(df_all.get("odds_1", np.nan))
    df_all["odds_2"] = _clean_numeric(df_all.get("odds_2", np.nan))
    df_all["home_team_prob"] = _clean_numeric(df_all.get("home_team_prob", np.nan))
    df_all["win"] = (df_all["result"] == df_all["home_team"]).astype(int)
    df_all["is_played"] = df_all["result"].notna() & (df_all["result"].astype(str) != "0")

    hist_mask = df_all["is_played"] & df_all["home_team_prob"].notna()
    hist_calib = df_all.loc[hist_mask, ["home_team_prob","win"]].copy()
    if hist_calib.empty:
        df_all["prob_iso"] = df_all["home_team_prob"]
        iso = None
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(hist_calib["home_team_prob"].values, hist_calib["win"].values)
        df_all["prob_iso"] = iso.transform(df_all["home_team_prob"].values)

    now_norm = pd.Timestamp.now().normalize()
    cutoff   = now_norm + pd.Timedelta(hours=LOOKAHEAD_HRS)
    upcoming = df_all[(~df_all["is_played"]) & (df_all["date"] >= now_norm) & (df_all["date"] <= cutoff)].copy()

    if upcoming.empty:
        print("\nNo upcoming games found.")
        return

    upcoming["prob_used"] = np.clip(upcoming["prob_iso"], 0.55, 0.75)
    upcoming["EV_€_per_100"] = (
        upcoming["prob_used"] * (upcoming["odds_1"] - 1.0)
        - (1.0 - upcoming["prob_used"])
    ) * FLAT_STAKE_BACKTEST
    upcoming["exp_profit_eur"] = upcoming["EV_€_per_100"] / 100 * upcoming["stake_eur"].fillna(FLAT_STAKE_BACKTEST)

    current_bankroll = bankroll_after
    upcoming["kelly_full"] = upcoming.apply(
        lambda r: kelly_fraction_row(r["prob_used"], r["odds_1"]), axis=1
    )
    upcoming["kelly_fraction_used"] = (upcoming["kelly_full"] * KELLY_FRACTION).clip(lower=0)
    upcoming["kelly_fraction_used"] = upcoming["kelly_fraction_used"].clip(upper=MAX_RISK_PCT_PER_BET)
    upcoming["stake_eur"] = (upcoming["kelly_fraction_used"] * current_bankroll).round(2)
    upcoming.loc[upcoming["stake_eur"] < MIN_STAKE_ABS, "stake_eur"] = 0.0

    cols_card_print = [
        "date","home_team","away_team","home_win_rate",
        "prob_iso","odds_1","EV_€_per_100","kelly_full",
        "kelly_fraction_used","stake_eur"
    ]
    print("\n=== TONIGHT'S SHORTLIST (ISOTONIC + KELLY) ===")
    print(upcoming[cols_card_print].sort_values("date").round(3).to_string(index=False))

    upcoming.to_csv(CARD_OUT_PATH, index=False)
    print(f"\n💾 Saved shortlist to {CARD_OUT_PATH}")

if __name__ == "__main__":
    main()
