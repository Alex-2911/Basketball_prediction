from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_frame(rows_df: pd.DataFrame, run_date: str, source: str | None) -> pd.DataFrame:
    df = rows_df.copy()
    rename = {"date": "game_date", "odds 1": "odds_1", "odds 2": "odds_2"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ["game_date", "home_team", "away_team", "blocked_by"]:
        if col not in df.columns:
            df[col] = ""
    df["blocked_by"] = df["blocked_by"].fillna("").astype(str)
    df["run_date"] = run_date
    df["created_utc"] = _utc_now_iso()
    if source is not None:
        df["engine_state"] = source

    df["game_key"] = (
        df["game_date"].astype(str) + "__" + df["home_team"].astype(str) + "__" + df["away_team"].astype(str)
    )
    return df


def classify_script11_row(row: pd.Series) -> str:
    blocked = str(row.get("blocked_by", ""))

    if "DATA_INCOMPLETE" in blocked:
        return "DATA_INCOMPLETE"

    if row.get("rules_passed", 0) >= 4 and row.get("EV_€_per_100", -999) > 0:
        return "CANONICAL_MODEL_SIGNAL"

    if (
        row.get("home_win_rate", 0) >= 0.50
        and row.get("odds_1", 0) >= 1.30
        and row.get("odds_1", 0) <= 1.70
        and row.get("prob_used", 0) >= 0.55
        and row.get("EV_€_per_100", 999) <= 0
        and "MODEL_MARKET_GAP" not in blocked
    ):
        return "LOW_PRICE_NEGATIVE_EV"

    if (
        row.get("home_win_rate", 0) >= 0.60
        and row.get("odds_1", 0) >= 2.00
        and row.get("odds_1", 0) <= 2.80
        and row.get("prob_base", 0) >= 0.60
        and "MODEL_MARKET_GAP" in blocked
        and row.get("prob_used", 1) < 0.55
    ):
        return "RAW_MODEL_MARKET_GAP_HOME_DOG"

    if "MODEL_MARKET_GAP" in blocked:
        return "LIVE_WATCH_ONLY"

    if row.get("EV_€_per_100", 999) <= 0:
        return "NO_VALUE_SKIP"

    return "LIVE_WATCH_ONLY"


def _reconcile_outcomes(history: pd.DataFrame, combined_predictions_path: str | Path | None) -> pd.DataFrame:
    if combined_predictions_path is None:
        return history
    path = Path(combined_predictions_path)
    if not path.exists():
        return history
    combined = pd.read_csv(path)
    if "date" in combined.columns and "game_date" not in combined.columns:
        combined = combined.rename(columns={"date": "game_date"})
    need = {"game_date", "home_team", "away_team"}
    if not need.issubset(combined.columns):
        return history

    cols = [c for c in ["game_date", "home_team", "away_team", "result", "result_raw", "home_team_won"] if c in combined.columns]
    outcomes = combined[cols].drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")
    merged = history.merge(outcomes, on=["game_date", "home_team", "away_team"], how="left", suffixes=("", "_c"))

    for col in ["result", "result_raw", "home_team_won"]:
        cc = f"{col}_c"
        if cc in merged.columns:
            merged[col] = merged.get(col).where(merged.get(col).notna(), merged[cc])
            merged = merged.drop(columns=[cc])

    result = merged.get("result", "").fillna("").astype(str)
    home = merged["home_team"].astype(str)
    away = merged["away_team"].astype(str)
    settled = result.isin(home) | result.isin(away)
    merged["settled"] = settled

    if "home_team_won" not in merged.columns:
        merged["home_team_won"] = pd.NA
    merged.loc[settled & (result == home), "home_team_won"] = 1
    merged.loc[settled & (result == away), "home_team_won"] = 0
    merged["home_team_won"] = pd.to_numeric(merged["home_team_won"], errors="coerce")

    odds = pd.to_numeric(merged.get("odds_1"), errors="coerce")
    merged["pnl_ml_100"] = pd.NA
    merged.loc[merged["home_team_won"] == 1, "pnl_ml_100"] = (odds - 1.0) * 100.0
    merged.loc[merged["home_team_won"] == 0, "pnl_ml_100"] = -100.0
    return merged


def persist_script11_watchlist_history(rows_df: pd.DataFrame, output_dir: str | Path, run_date: str, params_used: dict[str, Any] | None = None, chosen: Any = None, compareN: Any = None, combined_predictions_path: str | Path | None = None, source: str | None = None) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_frame(rows_df, run_date=run_date, source=source)

    normalized["stage2_candidate_type"] = normalized.apply(classify_script11_row, axis=1)
    normalized["params_chosen"] = chosen
    normalized["params_compareN"] = compareN
    normalized["params_used"] = "" if params_used is None else str(params_used)

    history_path = out_dir / "script11_watchlist_history.csv"
    if history_path.exists():
        prior = pd.read_csv(history_path)
        merged = pd.concat([prior, normalized], ignore_index=True, sort=False)
    else:
        merged = normalized.copy()

    merged = merged.drop_duplicates(subset=["game_key", "run_date"], keep="last")
    merged = _reconcile_outcomes(merged, combined_predictions_path)
    merged = merged.sort_values(["game_date", "home_team", "away_team", "created_utc"], kind="stable")

    dated_path = out_dir / f"script11_watchlist_history_{run_date}.csv"
    latest_path = out_dir / "script11_watchlist_history_latest.csv"

    merged.to_csv(history_path, index=False)
    merged.loc[merged["run_date"].astype(str) == str(run_date)].to_csv(dated_path, index=False)
    merged.loc[merged["run_date"].astype(str) == str(run_date)].to_csv(latest_path, index=False)
    return merged
