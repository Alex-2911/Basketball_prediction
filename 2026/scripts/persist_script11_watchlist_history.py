from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_frame(rows_df: pd.DataFrame, run_date: str, source: str | None) -> pd.DataFrame:
    df = rows_df.copy()

    # Avoid duplicate-label failures when frames contain both alias and canonical columns.
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")].copy()

    rename = {"date": "game_date", "odds 1": "odds_1", "odds 2": "odds_2"}
    for source_col, target_col in rename.items():
        if source_col in df.columns and target_col in df.columns:
            df = df.drop(columns=[source_col])

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")].copy()

    for col in ["game_date", "home_team", "away_team", "blocked_by"]:
        if col not in df.columns:
            df[col] = ""
    df["blocked_by"] = df["blocked_by"].fillna("").astype(str)

    # Normalize game_date to a stable string key before game_key construction and merges.
    if "game_date" in df.columns:
        parsed_game_date = pd.to_datetime(df["game_date"], errors="coerce")
        df["game_date"] = parsed_game_date.dt.strftime("%Y-%m-%d").where(
            parsed_game_date.notna(),
            df["game_date"].fillna("").astype(str),
        )

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

    history = history.copy()
    combined = combined.copy()

    # Ensure merge keys have identical dtypes. GitHub runs can produce datetime64 in
    # history and object/string in combined, which breaks pandas merge.
    for frame in (history, combined):
        parsed_game_date = pd.to_datetime(frame["game_date"], errors="coerce")
        frame["game_date"] = parsed_game_date.dt.strftime("%Y-%m-%d").where(
            parsed_game_date.notna(),
            frame["game_date"].fillna("").astype(str),
        )
        frame["home_team"] = frame["home_team"].fillna("").astype(str)
        frame["away_team"] = frame["away_team"].fillna("").astype(str)

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
        # Daily Script 11 output is a snapshot. Replace the current run_date slice
        # so stale rows from earlier broken runs cannot remain under today's run_date.
        if "run_date" in prior.columns:
            prior = prior.loc[prior["run_date"].astype(str) != str(run_date)].copy()
        merged = pd.concat([prior, normalized], ignore_index=True, sort=False)
    else:
        merged = normalized.copy()

    merged = merged.drop_duplicates(subset=["game_key", "run_date"], keep="last")
    merged = _reconcile_outcomes(merged, combined_predictions_path)
    merged = merged.sort_values(["game_date", "home_team", "away_team", "created_utc"], kind="stable")

    dated_path = out_dir / f"script11_watchlist_history_{run_date}.csv"
    latest_path = out_dir / "script11_watchlist_history_latest.csv"

    latest_slice = merged.loc[merged["run_date"].astype(str) == str(run_date)].copy()

    merged.to_csv(history_path, index=False)
    latest_slice.to_csv(dated_path, index=False)
    latest_slice.to_csv(latest_path, index=False)

    def _counts(col: str) -> dict[str, int]:
        if col not in latest_slice.columns:
            return {}
        return {
            str(k): int(v)
            for k, v in latest_slice[col].fillna("").astype(str).value_counts(dropna=False).items()
        }

    summary = {
        "run_date": str(run_date),
        "rows": int(len(latest_slice)),
        "sources": _counts("engine_state"),
        "stage2_candidate_type": _counts("stage2_candidate_type"),
        "created_utc": _utc_now_iso(),
    }

    summary_dated_path = out_dir / f"script11_watchlist_history_summary_{run_date}.json"
    summary_latest_path = out_dir / "script11_watchlist_history_summary_latest.json"
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_dated_path.write_text(summary_text, encoding="utf-8")
    summary_latest_path.write_text(summary_text, encoding="utf-8")

    return merged
