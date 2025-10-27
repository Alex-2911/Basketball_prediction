#!/usr/bin/env python
# coding: utf-8

# Kelly Criterion Betting Strategy (2025-26 Season) — Resilient Version
#
# Script 5 of 5 for the 2025‑26 NBA season. This version is robust to missing
# 'combined' accuracy files. If the combined file is missing, it runs in a
# fallback mode that:
#   • uses only today's prediction file,
#   • skips calibration & home‑win filters,
#   • still prints Kelly suggestions (home and away) based on raw probabilities.
#
# Normal mode (when the combined file exists) keeps your full workflow:
#   • computes home‑win rates,
#   • fits Platt & Isotonic calibrations,
#   • filters by good home teams + odds + probability,
#   • simulates season‑long bankrolls,
#   • saves enriched CSVs and shows bankroll plots.
#
# The console stays open at the end (or on errors) so you can read the output.

import os
import sys
import traceback
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

# ---- 2026 utilities ----
from nba_utils_2026 import (
    get_current_date,
    get_directory_paths,
    get_latest_file,
    kelly_frac,
    get_home_win_rates,
)

# ─────────────────────────────────────────────────────────
# BACKFILL OVERRIDE
# set this to the historical date you want to generate
# e.g. "2025-10-23", "2025-10-24", "2025-10-25", "2025-10-26"
# BACKFILL_DATE = "2025-10-25"   # <-- change this per run

#if BACKFILL_DATE:
#    import datetime as _dt

#    def get_current_date(days_offset: int = 0):
#        """
#        Override utils.get_current_date() for backfill runs.
#        Returns the same tuple shape:
#        (datetime_obj, friendly_str, ymd_str)
#        """
#        d = _dt.datetime.strptime(BACKFILL_DATE, "%Y-%m-%d")
#        # apply days_offset if caller passes it (script 5 doesn't, but keep it correct)
#        d = d - _dt.timedelta(days=days_offset)
#
#        friendly = (
#            d.strftime("%a, %b ")
#            + str(int(d.strftime("%d")))
#            + d.strftime(", %Y")
#        )
#        ymd = d.strftime("%Y-%m-%d")
#        return d, friendly, ymd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_predictions(directory_path: str, force_date: str = None):
    """
    Load predictions for a specific date if force_date is given (YYYY-MM-DD),
    otherwise fall back to the most recent file.
    """
    if force_date is not None:
        forced_path = os.path.join(directory_path, f"nba_games_predict_{force_date}.csv")
        if not os.path.exists(forced_path):
            raise FileNotFoundError(f"Could not find {forced_path} for forced date mode.")
        pred_file = forced_path
    else:
        pred_file = get_latest_file(directory_path, prefix="nba_games_predict_", ext=".csv")
        if not pred_file:
            raise FileNotFoundError(
                f"No nba_games_predict_*.csv found in {directory_path}. "
                "Run script 3 to generate predictions."
            )

    df = pd.read_csv(pred_file, decimal=",", encoding="utf-7")

    # normalize exactly like before
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    df["date"]    = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["odds_1"]  = pd.to_numeric(df["odds_1"].astype(str).str.replace(",", "."), errors="coerce")
    df["odds_2"]  = pd.to_numeric(df["odds_2"].astype(str).str.replace(",", "."), errors="coerce")
    df["raw_prob"] = pd.to_numeric(df["home_team_prob"].astype(str).str.replace(",", "."), errors="coerce")

    logging.info(f"Loaded predictions file: {pred_file} with {len(df)} rows")
    return pred_file, df



def try_load_combined(directory_path: str):
    """Try to load the latest combined accuracy CSV. Return (path, df) or (None, None)."""
    hist_file = get_latest_file(directory_path, prefix="combined_nba_predictions_acc_", ext=".csv")
    if not hist_file:
        logging.warning("No combined_nba_predictions_acc_*.csv found. Fallback mode will be used.")
        return None, None
    df = pd.read_csv(hist_file, encoding="utf-7", decimal=",")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logging.info(f"Loaded combined file: {hist_file} with {len(df)} rows")
    return hist_file, df


def compute_home_win_rates_save(hist_df: pd.DataFrame, output_file_home: str):
    """Compute and save home win rates (sorted)."""
    hwr_sorted = get_home_win_rates(hist_df)
    hwr_sorted.to_csv(output_file_home, index=True, index_label="team", float_format="%.4f")
    logging.info(f"Saved home win rates to: {output_file_home}")
    return hwr_sorted


def kelly_suggestion_text(team_home, team_away, prob, odds, bank, bet_frac=0.5, cap_frac=0.30, abs_cap=300.0, side="home"):
    """Return a printable suggestion line for Kelly staking if positive; else None."""
    def _kelly_frac(p, o, f):
        b = o - 1.0
        if b <= 0:
            return 0.0
        return max(((b*p - (1-p)) / b) * f, 0.0)

    kf = _kelly_frac(prob, odds, bet_frac)
    if kf <= 0:
        return None, 0.0
    stake = min(kf * bank, cap_frac * bank, abs_cap)
    line = f"✅ {team_home}–{team_away} ({side}): p̂={prob:.4f}, odds={odds:.2f} → half‑Kelly={kf:.4f}, stake=€{stake:.2f}"
    return line, stake


def main():
    # ---- config & paths ----
    today, today_str, today_str_format = get_current_date()  # may be offset per your utils
    paths = get_directory_paths()
    directory_path = paths["PREDICTION_DIR"]
    output_file_home = os.path.join(directory_path, f"home_win_rates_sorted_{today_str_format}.csv")
    OUTPUT_FILE = os.path.join(directory_path, f"combined_nba_predictions_enriched_{today_str_format}.csv")
    OUTPUT_FILE_filtered = os.path.join(directory_path, f"combined_nba_predictions_enriched_filtered_{today_str_format}.csv")
    out_path_kelly = os.path.join(directory_path, f"kelly_stakes_{today_str_format}.csv")

    print(f"Today's date (utils): {today_str_format}")

    # Strategy thresholds & bankroll parameters
    odds_min = 1.18
    odds_max = 3.00
    raw_prob_cut = 0.40
    home_win_cut = 0.50

    # Small “today” bank for suggestions
    starting_bank_today = 42.20
    bet_frac = 0.5
    cap_frac = 0.30
    abs_cap = 300.0

    # ---- load predictions ----
    pred_path, df_pred = load_predictions(directory_path, force_date=today_str_format)
    print(f"Using prediction file: {pred_path}")

    # ---- try to load combined accuracy (normal mode) ----
    hist_path, hist_df = try_load_combined(directory_path)
    fallback = hist_df is None

    if not fallback:
        # NORMAL MODE
        print("\nNormal mode: combined accuracy file found.\n")

        # Home win rates
        hwr_sorted = compute_home_win_rates_save(hist_df, output_file_home)
        good_homes = set(hwr_sorted.loc[hwr_sorted["Home Win Rate"] >= home_win_cut].index)

        # Calibration (Platt & Isotonic) using historical combined
        hist = hist_df.copy()
        hist.columns = hist.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        X = hist[["home_team_prob"]].values
        y = hist["accuracy"].astype(int).values

        # Platt
        _, Xc, _, yc = train_test_split(X, y, test_size=0.2, random_state=42)
        platt = LogisticRegression(solver="lbfgs").fit(Xc, yc)
        df_pred["prob_platt"] = platt.predict_proba(df_pred[["raw_prob"]])[:, 1]

        # Isotonic
        iso = IsotonicRegression(out_of_bounds="clip").fit(hist["home_team_prob"], hist["accuracy"])
        df_pred["prob_iso"] = iso.transform(df_pred["raw_prob"])

        # Display schedule (clean formatting)
        print("Today's schedule:")
        print(
            df_pred[["home_team", "away_team", "raw_prob", "odds_1", "odds_2", "date"]]
            .assign(
                raw_prob=lambda d: d["raw_prob"].map("{:.4f}".format),
                odds_1=lambda d: d["odds_1"].map("{:.2f}".format),
                odds_2=lambda d: d["odds_2"].map("{:.2f}".format),
            )
            .to_string(index=False)
        )
        print()

        # Selection filters (home‑side strategy as in your workflow)
        sel = df_pred[
            (df_pred.home_team.isin(good_homes)) &
            (df_pred.odds_1.between(odds_min, odds_max)) &
            (df_pred.raw_prob >= raw_prob_cut)
        ].copy()

        # Kelly suggestions (home)
        rows = []
        print("Kelly suggestions (home side):")
        for _, r in sel.iterrows():
            for label, p in [("raw", r.raw_prob), ("platt", r.prob_platt), ("iso", r.prob_iso)]:
                line, stake = kelly_suggestion_text(
                    r.home_team, r.away_team, p, r.odds_1,
                    starting_bank_today, bet_frac, cap_frac, abs_cap, side=f"home-{label}"
                )
                if line:
                    print(line)
                    rows.append({
                        "home_team": r.home_team,
                        "away_team": r.away_team,
                        "date": r.date,
                        "side": f"home-{label}",
                        "prob": p,
                        "odds": r.odds_1,
                        "stake": stake,
                    })
        if not rows:
            print("No positive‑edge home bets under current filters.")
        print()

        # Season‑long simulation (uses hist_df)
        # Prepare df_sim
        df_sim = hist_df.copy()
        df_sim.columns = (
            df_sim.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
        df_sim["date"] = pd.to_datetime(df_sim["date"], errors="coerce")
        df_sim["odds_1"] = pd.to_numeric(df_sim["odds_1"].astype(str).str.replace(",", "."), errors="coerce")
        df_sim["home_team_prob"] = pd.to_numeric(df_sim["home_team_prob"].astype(str).str.replace(",", "."), errors="coerce")
        df_sim["win"] = (df_sim["result"] == df_sim["home_team"]).astype(int)

        # Fit calibrations on full hist (already defined 'platt' and 'iso' above for df_pred; re‑fit for df_sim)
        X_all = df_sim[["home_team_prob"]].values
        y_all = df_sim["accuracy"].astype(int).values
        platt_all = LogisticRegression(solver="lbfgs").fit(X_all, y_all)
        df_sim["prob_platt"] = platt_all.predict_proba(X_all)[:, 1]
        iso_all = IsotonicRegression(out_of_bounds="clip").fit(X_all.flatten(), y_all)
        df_sim["prob_iso"] = iso_all.transform(X_all.flatten())

        # Kelly sim parameters
        starting_bank = 1000.0
        bet_frac = 0.5
        cap_frac = 0.30
        abs_cap = 300.0

        for lbl in ("raw", "platt", "iso"):
            df_sim[f"kelly_frac_{lbl}"] = 0.0
            df_sim[f"stake_{lbl}"] = 0.0
            df_sim[f"pnl_{lbl}"] = 0.0
            df_sim[f"ev_{lbl}"] = 0.0
            df_sim[f"bank_{lbl}"] = np.nan

        bank = {"raw": starting_bank, "platt": starting_bank, "iso": starting_bank}
        good_home = set(hwr_sorted.loc[hwr_sorted["Home Win Rate"] >= home_win_cut].index)

        for i, row in df_sim.sort_values("date").iterrows():
            o = row["odds_1"]
            is_home = row["home_team"] in good_home
            for lbl, p_col in [("raw", "home_team_prob"), ("platt", "prob_platt"), ("iso", "prob_iso")]:
                p = row[p_col]
                if is_home and (o >= odds_min) and (o <= odds_max) and (p >= raw_prob_cut):
                    # Use provided Kelly function for sim
                    kf = kelly_frac(p, o, bet_frac)
                    stake = min(kf * bank[lbl], cap_frac * bank[lbl], abs_cap)
                    pnl = stake * (o - 1.0) if bool(row["win"]) else -stake
                    ev = (p * (o - 1.0) - (1 - p)) * stake
                else:
                    kf = stake = pnl = ev = 0.0
                bank[lbl] += pnl
                df_sim.at[i, f"kelly_frac_{lbl}"] = kf
                df_sim.at[i, f"stake_{lbl}"] = stake
                df_sim.at[i, f"pnl_{lbl}"] = pnl
                df_sim.at[i, f"ev_{lbl}"] = ev
                df_sim.at[i, f"bank_{lbl}"] = bank[lbl]

        # Save outputs
        df_sim.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
        logging.info(f"✅ Wrote enriched file → {OUTPUT_FILE}")

        mask = (
            (df_sim["stake_raw"] > 0) |
            (df_sim["stake_platt"] > 0) |
            (df_sim.get("stake_iso", 0) > 0)
        )
        df_filtered = df_sim.loc[mask].reset_index(drop=True)
        df_filtered.to_csv(OUTPUT_FILE_filtered, index=False, float_format="%.4f")
        logging.info(f"✅ Wrote enriched file → {OUTPUT_FILE_filtered}")

        # Plot bankroll paths
        plt.figure(figsize=(10, 6))
        for lbl, color in [("raw", "C0"), ("platt", "C1"), ("iso", "C2")]:
            plt.plot(df_filtered["date"], df_filtered[f"bank_{lbl}"], label=f"{lbl.capitalize()}‑Kelly bank", color=color)
        plt.xlabel("Date")
        plt.ylabel("Bankroll (€)")
        plt.title("Raw vs Platt vs Iso‑Kelly Bankroll Paths")
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

        # Optionally also save Kelly suggestions summary for today
        # Always save the Kelly stakes file, even if empty
        pd.DataFrame(rows or []).to_csv(out_path_kelly, index=False, float_format="%.4f")
        if rows:
            logging.info(f"✅ Wrote Kelly stakes (today) → {out_path_kelly}")
        else:
            logging.warning(f"⚠️ No Kelly suggestions — empty file saved at {out_path_kelly}")


    else:
        # FALLBACK MODE
        print("\n⚠️  Fallback mode: combined accuracy file missing.")
        print("    → Skipping calibration & home‑win filters; proposing Kelly bets from raw probabilities only.\n")

        # Display schedule
        print("Today's schedule:")
        print(
            df_pred[["home_team", "away_team", "raw_prob", "odds_1", "odds_2", "date"]]
            .assign(
                raw_prob=lambda d: d["raw_prob"].map("{:.4f}".format),
                odds_1=lambda d: d["odds_1"].map("{:.2f}".format),
                odds_2=lambda d: d["odds_2"].map("{:.2f}".format),
            )
            .to_string(index=False)
        )
        print()

        # In fallback, consider both home and away Kelly stakes.
        rows = []
        for _, r in df_pred.iterrows():
            # Home side
            line_h, stake_h = kelly_suggestion_text(
                r.home_team, r.away_team, r.raw_prob, r.odds_1,
                starting_bank_today, bet_frac, cap_frac, abs_cap, side="home-raw"
            )
            if line_h:
                print(line_h)
                rows.append({
                    "home_team": r.home_team, "away_team": r.away_team, "date": r.date,
                    "side": "home-raw", "prob": r.raw_prob, "odds": r.odds_1, "stake": stake_h
                })

            # Away side uses (1 - p) and odds_2
            away_prob = 1.0 - (r.raw_prob if pd.notnull(r.raw_prob) else 0.0)
            line_a, stake_a = kelly_suggestion_text(
                r.home_team, r.away_team, away_prob, r.odds_2,
                starting_bank_today, bet_frac, cap_frac, abs_cap, side="away-raw"
            )
            if line_a:
                print(line_a)
                rows.append({
                    "home_team": r.home_team, "away_team": r.away_team, "date": r.date,
                    "side": "away-raw", "prob": away_prob, "odds": r.odds_2, "stake": stake_a
                })

        if not rows:
            print("No positive‑edge Kelly bets found from raw probabilities.")

        # Save simple Kelly summary (today)
        # Always save the Kelly stakes file, even if empty
        pd.DataFrame(rows or []).to_csv(out_path_kelly, index=False, float_format="%.4f")
        if rows:
            logging.info(f"✅ Wrote Kelly stakes (today) → {out_path_kelly}")
        else:
            logging.warning(f"⚠️ No Kelly suggestions — empty file saved at {out_path_kelly}")


        # Keep a tiny plot open (optional) to ensure the window remains until user closes
        plt.figure(figsize=(4, 2))
        plt.title("Fallback mode — close when done reading")
        plt.axis("off")
        plt.show(block=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass
        sys.exit(1)
    else:
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass
