from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_probability_pipeline import build_probability_chain_config, load_required_strategy_params, prepare_live_probability_columns


PROB_CLIP_LO = 0.35
PROB_CLIP_HI = 0.80
STAKE = 100
UNDERDOG_ODDS_GUARD_MIN = 2.00
UNDERDOG_PROB_GUARD_MIN = 0.60
GAP_GUARD_MIN = 0.12
UNDERDOG_CAP = 0.55
TAU_GAP = 0.08
USE_BLEND_ALWAYS = True
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
    "prob_base",
    "prob_live_oos_proxy",
    "prob_live_safe_pre_clip",
    "market_implied_p_raw",
    "market_implied_p_devig",
    "model_market_gap",
    "model_market_gap_flag",
    "live_underdog_upscale_guard_triggered",
    "live_shrink_triggered",
    "live_oos_proxy_ready",
    "live_oos_proxy_train_rows",
    "live_oos_proxy_bin_n",
    "live_oos_proxy_bin_winrate",
    "blocked_by",
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
        [
            "prob_live_safe",
            "prob_used",
            "prob_live_oos_proxy",
            "prob_iso_oos_time",
            "prob_iso_insample",
            "prob_iso",
            "iso_proba_home_win",
            "pred_home_win_proba",
            "home_team_prob",
        ],
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


def _to_bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
        "nan": np.nan,
        "none": np.nan,
        "": np.nan,
    })
    return mapped

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




def _load_shortlist_data(repo_root: Path) -> pd.DataFrame:
    shortlist_dir = repo_root / "2026" / "output" / "LightGBM" / "Kelly"
    shortlist_files = sorted(shortlist_dir.glob("bet_shortlist_*.csv"))
    if not shortlist_files:
        return pd.DataFrame()

    frames = []
    for shortlist_path in shortlist_files:
        shortlist = pd.read_csv(shortlist_path)
        shortlist.columns = (
            shortlist.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
        if "game_date" not in shortlist.columns and "date" in shortlist.columns:
            shortlist["game_date"] = shortlist["date"]
        frames.append(shortlist)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _prepare_combined(df: pd.DataFrame) -> pd.DataFrame:
    columns = _resolve_combined_columns(df)

    combined = df.copy()
    combined["date"] = _normalize_date(combined[columns.date])
    combined["home_team"] = _normalize_team(combined[columns.home])
    combined["away_team"] = _normalize_team(combined[columns.away])
    combined["game_date"] = pd.to_datetime(combined[columns.date], errors="coerce", format="mixed")

    combined["odds_1"] = pd.to_numeric(combined.get("odds_1", combined[columns.odds]), errors="coerce")
    combined["odds_2"] = pd.to_numeric(combined.get("odds_2", np.nan), errors="coerce")
    combined["pred_home_win_proba"] = pd.to_numeric(combined.get("pred_home_win_proba", combined[columns.prob]), errors="coerce")
    combined["home_team_prob"] = pd.to_numeric(combined.get("home_team_prob", combined["pred_home_win_proba"]), errors="coerce")
    combined["prob_iso"] = pd.to_numeric(combined.get("prob_iso", combined.get("prob_iso_insample", np.nan)), errors="coerce")
    combined["home_team_won"] = _derive_home_won(combined)
    combined["result_raw"] = np.where(combined["home_team_won"].notna(), combined["home_team_won"], 0)

    combined = prepare_live_probability_columns(
        combined,
        clip_lo=PROB_CLIP_LO,
        clip_hi=PROB_CLIP_HI,
        config=build_probability_chain_config(
            date_col="game_date",
            result_col="home_team_won",
            result_raw_col="result_raw",
            pred_proba_col="pred_home_win_proba",
            prob_iso_oos_time_col="prob_iso_oos_time",
            compute_oos_chain=False,
        ),
    )

    if columns.home_win_rate:
        combined["home_win_rate"] = pd.to_numeric(combined[columns.home_win_rate], errors="coerce")
    else:
        combined["home_win_rate"] = np.nan

    combined["ev_per_100"] = (combined["prob_used"] * (combined["odds_1"] - 1) - (1 - combined["prob_used"])) * 100

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


def _append_new_bets(
    ledger: pd.DataFrame,
    combined: pd.DataFrame,
    params: dict[str, float],
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    existing_keys = set(ledger["game_key"].dropna())

    home_win_rate_min = float(params["home_win_rate_threshold"])
    odds_min = float(params["odds_min"])
    odds_max = float(params["odds_max"])
    prob_min = float(params["prob_threshold"])
    min_ev = float(params["min_ev"])

    candidates = combined.copy()

    shortlist_keys: set[str] = set()
    if not shortlist.empty:
        required = {"home_team", "away_team"}
        date_col = "game_date" if "game_date" in shortlist.columns else "date"
        if date_col in shortlist.columns and required.issubset(shortlist.columns):
            shortlist_keys = set(
                (
                    _normalize_date(shortlist[date_col]).fillna("")
                    + "_"
                    + _normalize_team(shortlist["home_team"]).fillna("")
                    + "_"
                    + _normalize_team(shortlist["away_team"]).fillna("")
                    + "_HOME"
                ).tolist()
            )

    if shortlist_keys:
        qualifiers = candidates[candidates["game_key"].isin(shortlist_keys)].copy()
    else:
        qualifiers = candidates[
            (candidates["home_win_rate"] >= home_win_rate_min)
            & (candidates["odds_1"] >= odds_min)
            & (candidates["odds_1"] <= odds_max)
            & (candidates["prob_used"] >= prob_min)
            & (candidates["ev_per_100"] > min_ev)
        ].copy()

    if "blocked_by" in qualifiers.columns:
        qualifiers = qualifiers[qualifiers["blocked_by"].fillna("PASS").eq("PASS")].copy()

    if HARD_VETO_MODEL_MARKET_GAP and "model_market_gap_flag" in qualifiers.columns:
        gap_flag_raw = _to_bool_series(qualifiers["model_market_gap_flag"])
        gap_flag = np.where(gap_flag_raw.notna(), gap_flag_raw.astype(bool), False)
        qualifiers = qualifiers[~pd.Series(gap_flag, index=qualifiers.index)].copy()

    new_rows = qualifiers[~qualifiers["game_key"].isin(existing_keys)].copy()
    if new_rows.empty:
        return ledger

    now_utc = datetime.now(timezone.utc).isoformat()

    new_rows["stake"] = STAKE
    new_rows["odds"] = new_rows["odds_1"]
    new_rows["pick"] = "HOME"
    new_rows["status"] = np.where(new_rows["win"].notna(), "SETTLED", "PENDING")
    new_rows["won"] = new_rows["win"]
    new_rows["pnl"] = np.where(
        new_rows["win"].notna(),
        np.where(new_rows["win"] == 1, STAKE * (new_rows["odds_1"] - 1), -STAKE),
        np.nan,
    )
    new_rows["created_at_utc"] = now_utc
    new_rows["settled_at_utc"] = np.where(new_rows["win"].notna(), now_utc, np.nan)
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
            "prob_base",
            "prob_live_oos_proxy",
            "prob_live_safe_pre_clip",
            "market_implied_p_raw",
            "market_implied_p_devig",
            "model_market_gap",
            "model_market_gap_flag",
            "live_underdog_upscale_guard_triggered",
            "live_shrink_triggered",
            "live_oos_proxy_ready",
            "live_oos_proxy_train_rows",
            "live_oos_proxy_bin_n",
            "live_oos_proxy_bin_winrate",
            "blocked_by",
            "ev_per_100",
            "created_at_utc",
            "settled_at_utc",
            "source",
            "game_key",
        ]
    ]

    if ledger.empty:
        return append_df.copy()

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

    shortlist = _load_shortlist_data(repo_root)

    ledger = _load_ledger(ledger_path)
    ledger = _settle_pending_bets(ledger, combined)
    active_params = load_required_strategy_params(repo_root)
    ledger = _append_new_bets(ledger, combined, params=active_params, shortlist=shortlist)
    ledger = _dedupe_ledger(ledger)

    ledger = ledger.sort_values(["date", "home_team", "away_team"], na_position="last")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    ledger_output = _ensure_columns(ledger)[LEDGER_COLUMNS]
    ledger_output.to_csv(ledger_path, index=False, encoding="utf-8")
    ledger_output.to_csv(export_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
