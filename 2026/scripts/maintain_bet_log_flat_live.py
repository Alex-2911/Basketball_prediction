from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80
MIN_EV = -5
STAKE = 100
ODDS_MIN = 2.10
ODDS_MAX = 3.10
HOME_WIN_RATE_MIN = 0.55
PROB_MIN = 0.40


LEDGER_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_win_rate",
    "prob_iso",
    "prob_used",
    "odds_1",
    "EV_€_per_100",
    "stake",
    "potential_profit_if_win",
    "status",
    "win",
    "pnl",
    "settled_at",
    "game_key",
]


@dataclass
class CombinedColumns:
    date: str
    home: str
    away: str
    prob: str
    odds: str
    home_win_rate: str | None
    result: str | None


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed").dt.strftime(
        "%Y-%m-%d"
    )


def _normalize_team(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)


def _resolve_first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _resolve_combined_columns(df: pd.DataFrame) -> CombinedColumns:
    date_col = _resolve_first(df.columns, ["game_date", "date", "game_day"])
    home_col = _resolve_first(df.columns, ["home_team", "home"])
    away_col = _resolve_first(df.columns, ["away_team", "away"])
    prob_col = _resolve_first(
        df.columns,
        ["prob_iso", "iso_proba_home_win", "pred_home_win_proba", "home_team_prob"],
    )
    odds_col = _resolve_first(df.columns, ["odds_1", "closing_home_odds"])
    home_win_rate_col = _resolve_first(df.columns, ["home_win_rate", "home_winrate"])
    result_col = _resolve_first(df.columns, ["home_team_won", "result", "win"])

    missing = [
        label
        for label, col in [
            ("date", date_col),
            ("home_team", home_col),
            ("away_team", away_col),
            ("prob", prob_col),
            ("odds", odds_col),
        ]
        if col is None
    ]
    if missing:
        raise KeyError(
            "Combined predictions file missing required columns: "
            f"{missing}. Available: {list(df.columns)}"
        )

    return CombinedColumns(
        date=date_col,
        home=home_col,
        away=away_col,
        prob=prob_col,
        odds=odds_col,
        home_win_rate=home_win_rate_col,
        result=result_col,
    )


def _coerce_win(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    normalized = series.astype(str).str.strip().str.lower()
    win_map = {
        "1": 1,
        "true": 1,
        "w": 1,
        "win": 1,
        "home": 1,
        "0": 0,
        "false": 0,
        "l": 0,
        "loss": 0,
        "away": 0,
    }
    mapped = normalized.map(win_map)
    if numeric.notna().any():
        numeric_converted = numeric.where(numeric.isna(), (numeric >= 1).astype(int))
        return numeric_converted.fillna(mapped)
    return mapped


def _latest_file(path: Path, pattern: str) -> Path | None:
    candidates = sorted(path.glob(pattern))
    return candidates[-1] if candidates else None


def _load_combined_data(repo_root: Path) -> pd.DataFrame:
    iso_dir = repo_root / "2026" / "output" / "LightGBM" / "Kelly"
    acc_dir = repo_root / "2026" / "output" / "LightGBM"

    iso_path = _latest_file(iso_dir, "combined_nba_predictions_iso_*.csv")
    if iso_path and iso_path.exists():
        return pd.read_csv(iso_path)

    acc_path = _latest_file(acc_dir, "combined_nba_predictions_acc_*.csv")
    if acc_path and acc_path.exists():
        return pd.read_csv(acc_path)

    raise FileNotFoundError(
        "No combined_nba_predictions_iso_*.csv or combined_nba_predictions_acc_*.csv files found."
    )


def _prepare_combined(df: pd.DataFrame) -> pd.DataFrame:
    columns = _resolve_combined_columns(df)

    combined = df.copy()
    combined["date"] = _normalize_date(combined[columns.date])
    combined["home_team"] = _normalize_team(combined[columns.home])
    combined["away_team"] = _normalize_team(combined[columns.away])
    combined["prob_iso"] = pd.to_numeric(combined[columns.prob], errors="coerce")
    combined["odds_1"] = pd.to_numeric(combined[columns.odds], errors="coerce")

    if columns.home_win_rate:
        combined["home_win_rate"] = pd.to_numeric(
            combined[columns.home_win_rate], errors="coerce"
        )
    else:
        combined["home_win_rate"] = np.nan

    combined["prob_used"] = combined["prob_iso"].clip(PROB_CLIP_LO, PROB_CLIP_HI)
    combined["EV_€_per_100"] = (
        combined["prob_used"] * (combined["odds_1"] - 1)
        - (1 - combined["prob_used"])
    ) * 100

    if columns.result:
        combined["win"] = _coerce_win(combined[columns.result])
    else:
        combined["win"] = np.nan

    combined["game_key"] = (
        combined["date"].fillna("")
        + "_"
        + combined["home_team"].fillna("")
        + "_"
        + combined["away_team"].fillna("")
    )
    return combined


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    ledger = pd.read_csv(path)

    if "win" not in ledger.columns and "won" in ledger.columns:
        ledger["win"] = ledger["won"]

    if "pnl" not in ledger.columns and "pnl_flat" in ledger.columns:
        ledger["pnl"] = ledger["pnl_flat"]

    if "stake" not in ledger.columns:
        if "stake_flat" in ledger.columns:
            ledger["stake"] = ledger["stake_flat"]
        else:
            ledger["stake"] = STAKE

    for column in LEDGER_COLUMNS:
        if column not in ledger.columns:
            ledger[column] = np.nan

    ledger["date"] = _normalize_date(ledger["date"])
    ledger["home_team"] = _normalize_team(ledger["home_team"])
    ledger["away_team"] = _normalize_team(ledger["away_team"])
    ledger["game_key"] = (
        ledger["date"].fillna("")
        + "_"
        + ledger["home_team"].fillna("")
        + "_"
        + ledger["away_team"].fillna("")
    )

    if "status" in ledger.columns:
        status_filled = ledger["status"].fillna("")
    else:
        status_filled = ""

    needs_status = status_filled.astype(str).str.strip().eq("")
    if needs_status.any():
        ledger.loc[needs_status, "status"] = np.where(
            ledger.loc[needs_status, "win"].notna(), "SETTLED", "PENDING"
        )

    settled_mask = ledger["status"].astype(str).str.upper() == "SETTLED"
    missing_settled_at = settled_mask & ledger["settled_at"].isna()
    if missing_settled_at.any():
        ledger.loc[missing_settled_at, "settled_at"] = ledger.loc[
            missing_settled_at, "date"
        ]

    ledger["settled_at"] = ledger["settled_at"].astype("string")

    return ledger[LEDGER_COLUMNS]


def _append_new_bets(ledger: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    existing_keys = set(ledger["game_key"].dropna())

    upcoming = combined[combined["win"].isna()].copy()
    qualifiers = upcoming[
        (upcoming["home_win_rate"] >= HOME_WIN_RATE_MIN)
        & (upcoming["odds_1"] >= ODDS_MIN)
        & (upcoming["odds_1"] <= ODDS_MAX)
        & (upcoming["prob_used"] >= PROB_MIN)
        & (upcoming["EV_€_per_100"] > MIN_EV)
    ].copy()

    new_rows = qualifiers[~qualifiers["game_key"].isin(existing_keys)].copy()
    if new_rows.empty:
        return ledger

    new_rows["stake"] = STAKE
    new_rows["potential_profit_if_win"] = new_rows["stake"] * (
        new_rows["odds_1"] - 1
    )
    new_rows["status"] = "PENDING"
    new_rows["win"] = np.nan
    new_rows["pnl"] = np.nan
    new_rows["settled_at"] = np.nan

    append_df = new_rows[LEDGER_COLUMNS]
    return pd.concat([ledger, append_df], ignore_index=True)


def _settle_pending_bets(ledger: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    results = combined[combined["win"].notna()].set_index("game_key")
    pending_mask = ledger["status"].astype(str).str.upper() == "PENDING"
    if not pending_mask.any():
        return ledger

    for idx in ledger[pending_mask].index:
        game_key = ledger.at[idx, "game_key"]
        if game_key not in results.index:
            continue

        win = int(results.at[game_key, "win"])
        stake = float(ledger.at[idx, "stake"]) if pd.notna(ledger.at[idx, "stake"]) else STAKE
        odds = float(ledger.at[idx, "odds_1"]) if pd.notna(ledger.at[idx, "odds_1"]) else np.nan
        if pd.isna(odds):
            continue

        ledger.at[idx, "status"] = "SETTLED"
        ledger.at[idx, "win"] = win
        ledger.at[idx, "pnl"] = stake * (odds - 1) if win == 1 else -stake
        ledger.at[idx, "settled_at"] = results.at[game_key, "date"]

    return ledger


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ledger_path = repo_root / "2026" / "bet_log" / "bet_log_flat_live.csv"
    export_path = repo_root / "2026" / "output" / "LightGBM" / "bet_log_flat_live.csv"

    combined_raw = _load_combined_data(repo_root)
    combined = _prepare_combined(combined_raw)

    ledger = _load_ledger(ledger_path)
    ledger = _settle_pending_bets(ledger, combined)
    ledger = _append_new_bets(ledger, combined)

    ledger = ledger.sort_values(["date", "home_team", "away_team"], na_position="last")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    ledger.to_csv(ledger_path, index=False, encoding="utf-8")
    ledger.to_csv(export_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
