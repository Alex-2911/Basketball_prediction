#!/usr/bin/env python
# coding: utf-8

####################################################################################################
# SCRIPT 5 — ISOTONIC CALIBRATED BETTING ENGINE (GITHUB VERSION, MATCHING LOCAL LOGIC)
####################################################################################################

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

########################
# PATH HANDLING (GitHub version)
########################

# GitHub-safe base: repo_root/2026/output/LightGBM
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

# yesterday's shortlist (actual bets you planned) to append into log
SHORTLIST_PATH     = os.path.join(REPO_BASE, f"bet_shortlist_{yesterday_str}.csv")

# tonight's shortlist (output)
CARD_OUT_PATH      = os.path.join(REPO_BASE, f"bet_shortlist_{today_str}.csv")

SUMMARY_XLSX_PATH  = os.path.join(REPO_BASE, f"betting_summary_{today_str}.xlsx")

COMBINED_FILE      = os.path.join(REPO_BASE, f"combined_nba_predictions_acc_{today_str}.csv")
HWR_FILE           = os.path.join(REPO_BASE, f"home_win_rates_sorted_{today_str}.csv")
TODAY_PRED         = os.path.join(REPO_BASE, f"nba_games_predict_{today_str}.csv")

########################
# GLOBAL CONFIG
########################

DEFAULT_STAKE_EUR       = 50.0       # default "paper" stake for shortlist if needed
STARTING_BANKROLL       = 1000.0     # bankroll baseline if no history
FLAT_STAKE_BACKTEST     = 100.0      # stake used during grid-search backtest
LOOKAHEAD_HRS           = 36         # look ahead for upcoming games

KELLY_FRACTION          = 0.5        # half Kelly
MAX_RISK_PCT_PER_BET    = 0.10       # cap: 10% of bankroll per bet
MIN_STAKE_ABS           = 10.0       # don't bet less than 10 €

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

# Strategy grid (same spirit as local version)
ODDS_MIN_GRID   = np.arange(1.1, 3.1, 0.1)
ODDS_MAX_GRID   = np.arange(1.2, 3.6, 0.1)
PROB_MIN_GRID   = np.arange(0.45, 0.90, 0.05)
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
        # in case probabilities were stored as 0–1000 or similar
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

    # Map old names
    if "stake" in out.columns and "stake_eur" not in out.columns:
        out = out.rename(columns={"stake": "stake_eur"})
    if "win" in out.columns and "won" not in out.columns:
        out = out.rename(columns={"win": "won"})

    keep_cols = [c for c in TARGET_COLS if c in out.columns]
    out = out[keep_cols].copy()

    # ensure all target cols exist
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
    """
    Normalize yesterday's shortlist into the TARGET_COLS scheme.
    """
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
    """
    Append yesterday's shortlist into the live log (for "planned" bets).
    """
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
    Settles bets in the log using the latest combined_nba_predictions_acc_*.csv that
    has 'result' and team names.
    """
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
    df["date"]      = pd.to_datetime(df["date"], errors="coerce")
    df["odds_1"]    = pd.to_numeric(df["odds_1"], errors="coerce")
    df["stake_eur"] = pd.to_numeric(df["stake_eur"], errors="coerce")
    df["won"]       = pd.to_numeric(df["won"], errors="coerce")

    df["profit_eur"] = np.where(
        df["won"] == 1,
        df["stake_eur"] * (df["odds_1"] - 1.0),
        np.where(df["won"] == 0, -df["stake_eur"], np.nan)
    )

    df = df.sort_values("date").reset_index(drop=True)
    df["cum_profit"] = df["profit_eur"].fillna(0).cumsum()
    df["bankroll"]   = STARTING_BANKROLL + df["cum_profit"]

    settled = df[df["won"].isin([0,1])].copy()

    total_bets    = len(settled)
    wins          = settled["won"].sum()
    losses        = total_bets - wins
    win_rate      = (wins / total_bets * 100.0) if total_bets else 0.0
    total_staked  = settled["stake_eur"].sum() if total_bets else 0.0
    total_profit  = settled["profit_eur"].sum() if total_bets else 0.0
    roi           = (total_profit / total_staked * 100.0) if total_staked else 0.0
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

    # bankroll evolution plot
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
    """
    Kelly fraction for a single-outcome bet (home team ML).
    p = probability used for staking (after cap)
    odds_decimal = decimal odds on the home team (odds_1)
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
    """
    Fallback loader if df_all has no future rows.
    Returns columns:
      date, home_team, away_team, home_team_prob, prob_iso,
      home_win_rate, odds_1, odds_2, is_played
    or empty DF.
    """
    if not os.path.exists(path_csv):
        return pd.DataFrame()

    tmp = pd.read_csv(
        path_csv,
        encoding="utf-7",
        sep=",",
        quotechar='"',
        decimal=","
    )

    # check if header plausible; if not, enforce schema
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

########################
# STRATEGY GRID SEARCH
########################

def grid_search_best_params(df_all: pd.DataFrame):
    """
    Simple backtest grid search over historical played games.

    Uses:
      - df_all["prob_iso"]
      - df_all["home_win_rate"]
      - df_all["odds_1"]
      - df_all["win"] (1/0)
    """

    hist = df_all[
        df_all["is_played"]
        & df_all["prob_iso"].notna()
        & df_all["odds_1"].notna()
    ].copy()

    if hist.empty:
        print("[GRID] No historical rows for grid search.")
        return None

    best_profit = -1e9
    best_params = None

    print(f"[GRID] Running grid search on {len(hist)} settled games...")

    for odds_min in ODDS_MIN_GRID:
        for odds_max in ODDS_MAX_GRID:
            if odds_max <= odds_min:
                continue
            for prob_min in PROB_MIN_GRID:
                for hwr_min in HOMEWR_MIN_GRID:
                    mask = (
                        (hist["home_win_rate"].fillna(0) >= hwr_min) &
                        (hist["odds_1"] >= odds_min) &
                        (hist["odds_1"] <= odds_max) &
                        (hist["prob_iso"] >= prob_min)
                    )

                    sub = hist[mask]
                    n_bets = len(sub)
                    if n_bets < 10:   # skip tiny strategies
                        continue

                    # realized profit with flat stake
                    stake = FLAT_STAKE_BACKTEST
                    profit = np.where(
                        sub["win"] == 1,
                        stake * (sub["odds_1"] - 1.0),
                        -stake
                    ).sum()

                    if profit > best_profit:
                        best_profit = profit
                        best_params = {
                            "home_win_rate_threshold": float(hwr_min),
                            "odds_min": float(odds_min),
                            "odds_max": float(odds_max),
                            "prob_threshold": float(prob_min),
                            "n_bets": int(n_bets),
                            "profit": float(profit),
                        }

    if best_params is None:
        print("[GRID] No viable combination found (maybe data too small).")
    else:
        print("\n[GRID] Best strategy parameters (flat-stake backtest):")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    return best_params

########################################
# MAIN FLOW
########################################

def main():
    # 1) Load & backup existing live log
    live_df = load_live_log()
    backup_live_log(live_df)

    # 2) Append yesterday's shortlist
    live_df = append_yesterday_shortlist(live_df)
    live_df.to_csv(LIVE_LOG_PATH, index=False)

    # 3) Settle results
    live_df = settle_results_in_log(live_df)
    live_df.to_csv(LIVE_LOG_PATH, index=False)

    # 4) Compute stats & bankroll
    bankroll_after = compute_and_print_stats(live_df)

    # 5) Load today's combined file
    if not os.path.exists(COMBINED_FILE):
        raise FileNotFoundError(f"Missing combined file: {COMBINED_FILE}")

    df_all = pd.read_csv(COMBINED_FILE, encoding="utf-7", decimal=",")
    df_all.columns = (
        df_all.columns
             .str.strip()
             .str.lower()
             .str.replace(r"\s+","_", regex=True)
    )

    df_all["date"]           = pd.to_datetime(df_all["date"], errors="coerce")
    df_all["odds_1"]         = _clean_numeric(df_all.get("odds_1", np.nan))
    df_all["odds_2"]         = _clean_numeric(df_all.get("odds_2", np.nan))
    df_all["home_team_prob"] = _clean_numeric(df_all.get("home_team_prob", np.nan))

    # attach home_win_rate if not present, from HWR file
    if "home_win_rate" not in df_all.columns and os.path.exists(HWR_FILE):
        hwr_df = pd.read_csv(HWR_FILE, index_col=0)
        hwr_df.columns = [c.strip() for c in hwr_df.columns]
        if "Home Win Rate" in hwr_df.columns:
            df_all["home_win_rate"] = df_all["home_team"].map(hwr_df["Home Win Rate"])
    if "home_win_rate" not in df_all.columns:
        df_all["home_win_rate"] = np.nan

    # define win / is_played
    if "result" in df_all.columns:
        df_all["win"] = (df_all["result"] == df_all["home_team"]).astype(int)
        df_all["is_played"] = df_all["result"].notna() & (df_all["result"].astype(str) != "0")
    else:
        df_all["win"] = 0
        df_all["is_played"] = False

    ################################################################################################
    # 5. ISOTONIC CALIBRATION (IN-SAMPLE)
    ################################################################################################

    hist_mask  = df_all["is_played"] & df_all["home_team_prob"].notna()
    hist_calib = df_all.loc[hist_mask, ["home_team_prob", "win"]].copy()

    if hist_calib.empty:
        print("[ISO] Not enough completed games to calibrate, using raw model probs.")
        df_all["prob_iso"] = df_all["home_team_prob"]
        iso = None
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(hist_calib["home_team_prob"].values, hist_calib["win"].values)
        df_all["prob_iso"] = iso.transform(df_all["home_team_prob"].values)

        # quick in-sample calibration bins
        if hist_calib.shape[0] >= 10:
            calib_bins = (
                df_all.loc[hist_mask]
                      .assign(prob_bin=pd.cut(
                          df_all.loc[hist_mask, "prob_iso"],
                          bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
                      ))
                      .groupby("prob_bin", observed=True)
                      .agg(
                          avg_calib_prob=("prob_iso", "mean"),
                          actual_home_win_rate=("win", "mean"),
                          n_games=("win", "size")
                      )
                      .round(3)
            )
            print("\n[ISO] In-sample calibration bins:")
            print(calib_bins)
        else:
            print("\n[ISO] Skipping in-sample bins (not enough samples).")

    ################################################################################################
    # 5b. OUT-OF-SAMPLE (CV) ISOTONIC CALIBRATION DIAGNOSTICS
    ################################################################################################

    if hist_calib.shape[0] < 30 or hist_calib["win"].nunique() < 2:
        print("\n[CV ISO] Not enough completed games / class variety for OOS calibration diagnostics.")
    else:
        approx_folds = hist_calib.shape[0] // 20
        n_splits = max(3, min(5, max(2, approx_folds)))

        X = hist_calib["home_team_prob"].values
        y = hist_calib["win"].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oos_iso = pd.Series(index=hist_calib.index, dtype=float)

        for train_idx, val_idx in skf.split(X, y):
            iso_cv = IsotonicRegression(out_of_bounds="clip")
            iso_cv.fit(X[train_idx], y[train_idx])
            oos_iso.iloc[val_idx] = iso_cv.transform(X[val_idx])

        df_all.loc[hist_mask, "prob_iso_oos"] = oos_iso

        calib_bins_oos = (
            df_all.loc[hist_mask]
                  .assign(prob_bin=pd.cut(
                      df_all.loc[hist_mask, "prob_iso_oos"],
                      bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
                  ))
                  .groupby("prob_bin", observed=True)
                  .agg(
                      avg_calib_prob=("prob_iso_oos", "mean"),
                      actual_home_win_rate=("win", "mean"),
                      n_games=("win", "size")
                  )
                  .round(3)
        )

        print(f"\n[CV ISO] Using {n_splits}-fold stratified CV for OOS calibration.")
        print(calib_bins_oos)

    ################################################################################################
    # 6. GRID SEARCH FOR BEST STRATEGY PARAMS
    ################################################################################################

    best_params = grid_search_best_params(df_all)

    ################################################################################################
    # 7–9. APPLY BEST PARAMS TO UPCOMING GAMES + KELLY SIZING + REASONS
    ################################################################################################

    card = pd.DataFrame()  # shortlist

    if best_params is None:
        print("\n(No best params found; skipping shortlist.)")

    else:
        now_norm = pd.Timestamp.now().normalize()
        cutoff   = now_norm + pd.Timedelta(hours=LOOKAHEAD_HRS)

        # upcoming from df_all
        upcoming = df_all[
            (~df_all["is_played"]) &
            (df_all["date"] >= now_norm) &
            (df_all["date"] <= cutoff)
        ].copy()

        # if no future rows, fall back to TODAY_PRED
        if upcoming.empty and os.path.exists(HWR_FILE) and os.path.exists(TODAY_PRED):
            hwr_df_local = pd.read_csv(HWR_FILE, index_col=0)
            hwr_df_local.columns = [c.strip() for c in hwr_df_local.columns]
            iso_model = iso if "iso" in globals() else None
            upcoming = load_today_predictions_safe(TODAY_PRED, iso_model, hwr_df_local)

        if upcoming.empty:
            print("\n=== TONIGHT'S SHORTLIST (ISOTONIC + KELLY) ===")
            print("No upcoming games found in df_all or today's prediction file.")
        else:
            # shortlist filter based on best_params
            mask_card = (
                (upcoming["home_win_rate"] >= best_params["home_win_rate_threshold"]) &
                (upcoming["odds_1"]      >= best_params["odds_min"]) &
                (upcoming["odds_1"]      <= best_params["odds_max"]) &
                (upcoming["prob_iso"]    >= best_params["prob_threshold"])
            )
            card = upcoming.loc[mask_card].copy()

            current_bankroll = bankroll_after
            print(f"\n💰 Current bankroll for sizing bets: {current_bankroll:.2f} €")

            # capped prob for staking
            card["prob_used"] = card["prob_iso"].clip(lower=0.50, upper=0.75)

            # EV per 100€ using prob_used
            FLAT_STAKE = 100.0
            card["EV_€_per_100"] = (
                card["prob_used"] * (card["odds_1"] - 1.0) - (1.0 - card["prob_used"])
            ) * FLAT_STAKE

            # Kelly using prob_used
            card["kelly_full"] = card.apply(
                lambda r: kelly_fraction_row(r["prob_used"], r["odds_1"]),
                axis=1
            )

            card["kelly_fraction_used"] = (card["kelly_full"] * KELLY_FRACTION).clip(lower=0)
            card["kelly_fraction_used"] = card["kelly_fraction_used"].clip(upper=MAX_RISK_PCT_PER_BET)

            card["stake_eur"] = (card["kelly_fraction_used"] * current_bankroll).round(2)

            card["exp_profit_eur"] = (
                card["stake_eur"] * (card["prob_used"] * card["odds_1"] - 1.0)
            ).round(2)

            # kill garbage / unsafe stakes
            card.loc[card["odds_1"].isna(), "stake_eur"] = 0.0
            card.loc[card["stake_eur"] < MIN_STAKE_ABS, "stake_eur"] = 0.0
            card.loc[card["exp_profit_eur"] <= 0,       "stake_eur"] = 0.0
            card.loc[card["EV_€_per_100"] < 0,          "stake_eur"] = 0.0
            card.loc[card["stake_eur"] == 0.0, "kelly_fraction_used"] = 0.0

            # derived values for inspection
            card["fair_odds"] = (1.0 / card["prob_used"]).round(3)
            card["edge_pct"]  = ((card["odds_1"] / card["fair_odds"] - 1.0) * 100).round(2)

            print("\n=== TONIGHT'S SHORTLIST (ISOTONIC + KELLY) ===")
            if card.empty:
                print("No games match the tuned iso strategy in the next slate.")
            else:
                cols_card = [
                    "date",
                    "home_team","away_team",
                    "home_win_rate",
                    "prob_iso",
                    "prob_used",
                    "odds_1",
                    "EV_€_per_100",
                    "kelly_full",
                    "kelly_fraction_used",
                    "stake_eur",
                    "exp_profit_eur",
                    "fair_odds",
                    "edge_pct"
                ]
                print(
                    card[cols_card]
                    .sort_values("date")
                    .round({
                        "home_win_rate":3,
                        "prob_iso":3,
                        "prob_used":3,
                        "odds_1":3,
                        "EV_€_per_100":2,
                        "kelly_full":3,
                        "kelly_fraction_used":3,
                        "stake_eur":2,
                        "exp_profit_eur":2,
                        "fair_odds":3,
                        "edge_pct":2
                    })
                    .to_string(index=False)
                )

            # diagnostics: all upcoming & reasons
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

    # ---- final save of today's shortlist snapshot ----
    print("\n=== TONIGHT'S SHORTLIST (SAVE SNAPSHOT) ===")
    if card.empty:
        print("No games match the tuned iso strategy in the next slate.")
    else:
        card_to_save = card.copy()

        if best_params is not None:
            card_to_save["param_home_win_rate_threshold"] = best_params["home_win_rate_threshold"]
            card_to_save["param_odds_min"]               = best_params["odds_min"]
            card_to_save["param_odds_max"]               = best_params["odds_max"]
            card_to_save["param_prob_threshold"]         = best_params["prob_threshold"]
        else:
            card_to_save["param_home_win_rate_threshold"] = np.nan
            card_to_save["param_odds_min"]               = np.nan
            card_to_save["param_odds_max"]               = np.nan
            card_to_save["param_prob_threshold"]         = np.nan

        card_to_save["bankroll_at_bettime"] = bankroll_after

        export_cols = [
            "date",
            "home_team",
            "away_team",
            "home_win_rate",
            "prob_iso",
            "prob_used",
            "odds_1",
            "EV_€_per_100",
            "kelly_full",
            "kelly_fraction_used",
            "stake_eur",
            "exp_profit_eur",
            "param_home_win_rate_threshold",
            "param_odds_min",
            "param_odds_max",
            "param_prob_threshold",
            "bankroll_at_bettime",
            "fair_odds",
            "edge_pct"
        ]

        for col in export_cols:
            if col not in card_to_save.columns:
                card_to_save[col] = np.nan

        card_to_save = (
            card_to_save[export_cols]
            .sort_values("date")
            .round({
                "home_win_rate":3,
                "prob_iso":3,
                "prob_used":3,
                "odds_1":3,
                "EV_€_per_100":2,
                "kelly_full":3,
                "kelly_fraction_used":3,
                "stake_eur":2,
                "exp_profit_eur":2,
                "bankroll_at_bettime":2,
                "fair_odds":3,
                "edge_pct":2
            })
        )

        card_to_save.to_csv(CARD_OUT_PATH, index=False)
        print(f"💾 Saved shortlist to {CARD_OUT_PATH}")


if __name__ == "__main__":
    main()
