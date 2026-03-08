from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
UNDERDOG_ODDS_THRESHOLD = 2.30
MODEL_MARKET_GAP_PROB_THRESHOLD = 0.60
HARD_VETO_MODEL_MARKET_GAP = True


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
        ["prob_live_safe", "prob_used", "prob_live_oos_proxy", "prob_iso_oos_time", "prob_iso_insample", "prob_iso", "iso_proba_home_win", "pred_home_win_proba", "home_team_prob"],
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


def _derive_home_won(df: pd.DataFrame) -> pd.Series:
    for col in ("home_team_won", "home_win", "home_result", "win"):
        if col in df.columns:
            return _coerce_win(df[col])

    if "home_score" in df.columns and "away_score" in df.columns:
        home_score = pd.to_numeric(df["home_score"], errors="coerce")
        away_score = pd.to_numeric(df["away_score"], errors="coerce")
        return (home_score > away_score).astype(float)

    if "home_points" in df.columns and "away_points" in df.columns:
        home_score = pd.to_numeric(df["home_points"], errors="coerce")
        away_score = pd.to_numeric(df["away_points"], errors="coerce")
        return (home_score > away_score).astype(float)

    if "result" in df.columns:
        home = _normalize_team(df["home_team"])
        away = _normalize_team(df["away_team"])
        result_str = _normalize_team(df["result"])
        match_mask = result_str.eq(home) | result_str.eq(away)
        home_won = pd.Series(np.nan, index=df.index, dtype="float")
        if match_mask.any():
            home_won.loc[match_mask] = (result_str[match_mask] == home[match_mask]).astype(float)
        result_numeric = pd.to_numeric(df["result"], errors="coerce")
        if result_numeric.notna().any():
            result_numeric = result_numeric.where(
                result_numeric.isna(), (result_numeric >= 1).astype(float)
            )
            home_won = home_won.fillna(result_numeric)
        return home_won

    return pd.Series(np.nan, index=df.index, dtype="float")


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
    combined["prob_selected"] = pd.to_numeric(combined[columns.prob], errors="coerce")
    combined["prob_iso"] = pd.to_numeric(combined.get("prob_iso_insample", combined.get("prob_iso", np.nan)), errors="coerce")
    combined["prob_iso_oos_time"] = pd.to_numeric(combined.get("prob_iso_oos_time", np.nan), errors="coerce")
    combined["prob_live_oos_proxy"] = pd.to_numeric(combined.get("prob_live_oos_proxy", np.nan), errors="coerce")
    combined["prob_live_safe"] = pd.to_numeric(combined.get("prob_live_safe", combined["prob_selected"]), errors="coerce")
    combined["odds_1"] = pd.to_numeric(combined[columns.odds], errors="coerce")

    if columns.home_win_rate:
        combined["home_win_rate"] = pd.to_numeric(
            combined[columns.home_win_rate], errors="coerce"
        )
    else:
        combined["home_win_rate"] = np.nan

    combined["prob_live_safe_pre_clip"] = combined["prob_live_safe"]
    combined["prob_base"] = combined["prob_live_safe_pre_clip"].clip(PROB_CLIP_LO, PROB_CLIP_HI)
    combined["prob_used"] = combined["prob_base"]
    combined["model_market_gap_flag"] = (
        (combined["odds_1"] > UNDERDOG_ODDS_THRESHOLD)
        & (combined["prob_used"] > MODEL_MARKET_GAP_PROB_THRESHOLD)
    )
    combined["ev_per_100"] = (
        combined["prob_used"] * (combined["odds_1"] - 1)
        - (1 - combined["prob_used"])
    ) * 100

    combined["win"] = _derive_home_won(combined)

    bad_prob = combined["prob_iso"] > 1
    bad_prob_used = combined["prob_used"] > 1
    combined = combined[~(bad_prob | bad_prob_used)].copy()

    combined["pick"] = "HOME"
    combined["game_key"] = (
        combined["date"].fillna("")
        + "_"
        + combined["home_team"].fillna("")
        + "_"
        + combined["away_team"].fillna("")
        + "_"
        + combined["pick"].fillna("")
    )
    return combined


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in LEDGER_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df


def _normalize_status(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    ledger = pd.read_csv(path)

    rename_map = {
        "odds_1": "odds",
        "stake_flat": "stake",
        "EV_€_per_100": "ev_per_100",
        "win": "won",
        "settled_at": "settled_at_utc",
    }
    ledger = ledger.rename(columns={k: v for k, v in rename_map.items() if k in ledger.columns})

    if "odds" not in ledger.columns and "closing_home_odds" in ledger.columns:
        ledger["odds"] = ledger["closing_home_odds"]

    ledger = _ensure_columns(ledger)

    ledger["date"] = _normalize_date(ledger["date"])
    ledger["home_team"] = _normalize_team(ledger["home_team"])
    ledger["away_team"] = _normalize_team(ledger["away_team"])

    if ledger["stake"].isna().any():
        ledger["stake"] = ledger["stake"].fillna(STAKE)

    if ledger["pick"].isna().any():
        ledger["pick"] = ledger["pick"].fillna("HOME")

    ledger["pick"] = ledger["pick"].astype(str).str.strip().str.upper()

    if ledger["prob_used"].isna().any() and "prob_iso" in ledger.columns:
        prob_iso = pd.to_numeric(ledger["prob_iso"], errors="coerce")
        ledger["prob_used"] = prob_iso.clip(PROB_CLIP_LO, PROB_CLIP_HI)

    prob_iso = pd.to_numeric(ledger["prob_iso"], errors="coerce") if "prob_iso" in ledger.columns else None
    prob_used = pd.to_numeric(ledger["prob_used"], errors="coerce")
    bad_prob = prob_iso > 1 if prob_iso is not None else pd.Series(False, index=ledger.index)
    bad_prob_used = prob_used > 1
    ledger = ledger[~(bad_prob | bad_prob_used)].copy()

    ledger["won"] = _coerce_win(ledger["won"]).astype("float")

    status = _normalize_status(ledger["status"].fillna(""))
    needs_status = status.eq("")
    status = status.mask(needs_status, np.where(ledger["won"].notna(), "SETTLED", "PENDING"))
    ledger["status"] = status

    needs_pnl = ledger["pnl"].isna() & ledger["won"].notna()
    if needs_pnl.any():
        odds = pd.to_numeric(ledger["odds"], errors="coerce")
        stake = pd.to_numeric(ledger["stake"], errors="coerce").fillna(STAKE)
        ledger.loc[needs_pnl, "pnl"] = np.where(
            ledger.loc[needs_pnl, "won"] == 1,
            stake.loc[needs_pnl] * (odds.loc[needs_pnl] - 1),
            -stake.loc[needs_pnl],
        )

    ledger["settled_at_utc"] = ledger["settled_at_utc"].astype("string")
    ledger["created_at_utc"] = ledger["created_at_utc"].astype("string")

    ledger["game_key"] = (
        ledger["date"].fillna("")
        + "_"
        + ledger["home_team"].fillna("")
        + "_"
        + ledger["away_team"].fillna("")
        + "_"
        + ledger["pick"].fillna("")
    )

    return ledger


def _dedupe_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger

    return ledger.drop_duplicates(subset=["game_key"], keep="first")


def _append_new_bets(ledger: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    existing_keys = set(ledger["game_key"].dropna())

    upcoming = combined[combined["win"].isna()].copy()
    qualifiers = upcoming[
        (upcoming["home_win_rate"] >= HOME_WIN_RATE_MIN)
        & (upcoming["odds_1"] >= ODDS_MIN)
        & (upcoming["odds_1"] <= ODDS_MAX)
        & (upcoming["prob_used"] >= PROB_MIN)
        & (upcoming["ev_per_100"] > MIN_EV)
    ].copy()

    if HARD_VETO_MODEL_MARKET_GAP and "model_market_gap_flag" in qualifiers.columns:
        qualifiers = qualifiers[~qualifiers["model_market_gap_flag"].fillna(False)].copy()

    new_rows = qualifiers[~qualifiers["game_key"].isin(existing_keys)].copy()
    if new_rows.empty:
        return ledger

    now_utc = datetime.now(timezone.utc).isoformat()

    new_rows["stake"] = STAKE
    new_rows["odds"] = new_rows["odds_1"]
    new_rows["pick"] = "HOME"
    new_rows["status"] = "PENDING"
    new_rows["won"] = np.nan
    new_rows["pnl"] = np.nan
    new_rows["created_at_utc"] = now_utc
    new_rows["settled_at_utc"] = np.nan
    new_rows["source"] = "auto"

    append_df = new_rows[
        [
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
            "game_key",
        ]
    ]

    combined_ledger = pd.concat([ledger, append_df], ignore_index=True)
    return combined_ledger


def _settle_pending_bets(ledger: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    results = combined[combined["win"].notna()].set_index("game_key")
    pending_mask = _normalize_status(ledger["status"]) == "PENDING"
    if not pending_mask.any():
        return ledger

    now_utc = datetime.now(timezone.utc).isoformat()

    for idx in ledger[pending_mask].index:
        game_key = ledger.at[idx, "game_key"]
        if game_key not in results.index:
            continue

        win = int(results.at[game_key, "win"])
        stake = float(ledger.at[idx, "stake"]) if pd.notna(ledger.at[idx, "stake"]) else STAKE
        odds = float(ledger.at[idx, "odds"]) if pd.notna(ledger.at[idx, "odds"]) else np.nan
        if pd.isna(odds):
            continue

        ledger.at[idx, "status"] = "SETTLED"
        ledger.at[idx, "won"] = win
        ledger.at[idx, "pnl"] = stake * (odds - 1) if win == 1 else -stake
        ledger.at[idx, "settled_at_utc"] = now_utc

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
    ledger = _dedupe_ledger(ledger)

    ledger = ledger.sort_values(["date", "home_team", "away_team"], na_position="last")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    ledger_output = _ensure_columns(ledger)[LEDGER_COLUMNS]
    ledger_output.to_csv(ledger_path, index=False, encoding="utf-8")
    ledger_output.to_csv(export_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
