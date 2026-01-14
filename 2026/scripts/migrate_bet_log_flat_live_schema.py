from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80
STAKE_DEFAULT = 100

LEDGER_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "stake",
    "odds",
    "pick",
    "status",
    "won",
    "pnl",
    "prob_used",
    "ev_per_100",
    "created_at_utc",
    "settled_at_utc",
    "source",
]


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


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in LEDGER_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df


def _normalize_status(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _dedupe_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger

    status_rank = _normalize_status(ledger["status"]).map({"PENDING": 0, "SETTLED": 1}).fillna(0)
    settled_ts = pd.to_datetime(ledger["settled_at_utc"], errors="coerce")
    created_ts = pd.to_datetime(ledger["created_at_utc"], errors="coerce")

    ledger = ledger.assign(
        _status_rank=status_rank,
        _settled_ts=settled_ts,
        _created_ts=created_ts,
    ).sort_values([
        "_status_rank",
        "_settled_ts",
        "_created_ts",
    ])

    ledger = ledger.drop_duplicates(subset=["game_key"], keep="last")
    return ledger.drop(columns=["_status_rank", "_settled_ts", "_created_ts"])


def migrate(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)

    rename_map = {
        "odds_1": "odds",
        "stake_flat": "stake",
        "EV_€_per_100": "ev_per_100",
        "win": "won",
        "settled_at": "settled_at_utc",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "odds" not in df.columns and "closing_home_odds" in df.columns:
        df["odds"] = df["closing_home_odds"]

    if "prob_used" not in df.columns and "prob_iso" in df.columns:
        df["prob_used"] = pd.to_numeric(df["prob_iso"], errors="coerce").clip(
            PROB_CLIP_LO, PROB_CLIP_HI
        )

    df = _ensure_columns(df)

    df["date"] = _normalize_date(df["date"])
    df["home_team"] = _normalize_team(df["home_team"])
    df["away_team"] = _normalize_team(df["away_team"])

    if df["stake"].isna().any():
        df["stake"] = df["stake"].fillna(STAKE_DEFAULT)

    if df["pick"].isna().any():
        df["pick"] = df["pick"].fillna("HOME")

    df["won"] = _coerce_win(df["won"]).astype("float")

    status = _normalize_status(df["status"].fillna(""))
    needs_status = status.eq("")
    status = status.mask(needs_status, np.where(df["won"].notna(), "SETTLED", "PENDING"))
    df["status"] = status

    prob_iso_col = _resolve_first(df.columns, ["prob_iso", "probability", "prob"])
    if prob_iso_col:
        prob_iso = pd.to_numeric(df[prob_iso_col], errors="coerce")
        bad_prob = prob_iso > 1
    else:
        bad_prob = pd.Series(False, index=df.index)

    prob_used = pd.to_numeric(df["prob_used"], errors="coerce")
    bad_prob_used = prob_used > 1

    df = df[~(bad_prob | bad_prob_used)].copy()

    needs_pnl = df["pnl"].isna() & df["won"].notna()
    if needs_pnl.any():
        odds = pd.to_numeric(df["odds"], errors="coerce")
        stake = pd.to_numeric(df["stake"], errors="coerce").fillna(STAKE_DEFAULT)
        df.loc[needs_pnl, "pnl"] = np.where(
            df.loc[needs_pnl, "won"] == 1,
            stake.loc[needs_pnl] * (odds.loc[needs_pnl] - 1),
            -stake.loc[needs_pnl],
        )

    now_utc = datetime.now(timezone.utc).isoformat()
    df["created_at_utc"] = df["created_at_utc"].fillna(now_utc)

    df["game_key"] = (
        df["date"].fillna("")
        + "_"
        + df["home_team"].fillna("")
        + "_"
        + df["away_team"].fillna("")
    )

    df = _dedupe_ledger(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["date", "home_team", "away_team"], na_position="last")
    df[LEDGER_COLUMNS].to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "2026" / "output" / "LightGBM" / "bet_log_flat_live.csv"
    default_fallback = repo_root / "2026" / "bet_log" / "bet_log_flat_live.csv"
    default_output = repo_root / "2026" / "bet_log" / "bet_log_flat_live.csv"

    parser = argparse.ArgumentParser(
        description="Migrate bet_log_flat_live.csv to the canonical schema."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input if default_input.exists() else default_fallback,
        help="Path to the existing bet_log_flat_live.csv to migrate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination path for the migrated ledger.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input bet log not found: {args.input}")

    migrate(args.input, args.output)


if __name__ == "__main__":
    main()
