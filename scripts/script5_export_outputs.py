#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_win_rate",
    "prob_iso",
    "prob_used",
    "odds_1",
    "ev_per_100",
    "win",
    "pnl",
    "stake",
]


def resolve_source_root(base_dir: Path) -> Path:
    source_root = os.environ.get("SOURCE_ROOT")
    if source_root:
        path = Path(source_root)
        if path.exists():
            return path
    fallback = base_dir / "2026"
    if fallback.exists():
        return fallback
    return base_dir


def find_bet_log(lightgbm_dir: Path) -> Path | None:
    primary = lightgbm_dir / "bet_log_live.csv"
    if primary.exists():
        return primary
    candidates = sorted(lightgbm_dir.glob("bet_log_live_*.csv"))
    return candidates[-1] if candidates else None


def compute_ev_per_100(prob_used: pd.Series, odds_1: pd.Series) -> pd.Series:
    return (prob_used * odds_1 - 1.0) * 100.0


def prepare_export_df(raw_df: pd.DataFrame, default_stake: float) -> pd.DataFrame:
    df = raw_df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "prob_used" not in df.columns and "prob_iso" in df.columns:
        df["prob_used"] = df["prob_iso"]

    if "win" not in df.columns:
        if "won" in df.columns:
            df["win"] = df["won"]
        elif "home_team_won" in df.columns:
            df["win"] = df["home_team_won"]

    if "pnl" not in df.columns:
        if "profit_eur" in df.columns:
            df["pnl"] = df["profit_eur"]
        elif "profit" in df.columns:
            df["pnl"] = df["profit"]

    if "stake" not in df.columns:
        if "stake_eur" in df.columns:
            df["stake"] = df["stake_eur"]
        else:
            df["stake"] = float(default_stake)

    df["home_win_rate"] = pd.to_numeric(df.get("home_win_rate"), errors="coerce")
    df["prob_iso"] = pd.to_numeric(df.get("prob_iso"), errors="coerce")
    df["prob_used"] = pd.to_numeric(df.get("prob_used"), errors="coerce")
    df["odds_1"] = pd.to_numeric(df.get("odds_1"), errors="coerce")
    df["win"] = pd.to_numeric(df.get("win"), errors="coerce")
    df["pnl"] = pd.to_numeric(df.get("pnl"), errors="coerce")
    df["stake"] = pd.to_numeric(df.get("stake"), errors="coerce")

    if "ev_per_100" not in df.columns:
        df["ev_per_100"] = compute_ev_per_100(df["prob_used"], df["odds_1"])
    df["ev_per_100"] = pd.to_numeric(df["ev_per_100"], errors="coerce")

    df = df.dropna(subset=["win", "pnl"]).copy()
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df["win"] = df["win"].clip(lower=0, upper=1).astype(int)
    df["stake"] = df["stake"].fillna(float(default_stake))

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[REQUIRED_COLUMNS].sort_values("date").reset_index(drop=True)


def build_metrics_snapshot(
    export_df: pd.DataFrame,
    *,
    as_of_date: str,
    stake: float,
    params: dict,
    min_ev: float,
) -> dict:
    realized_count = int(len(export_df))
    profit_sum = float(export_df["pnl"].sum()) if realized_count > 0 else 0.0
    roi = profit_sum / (realized_count * float(stake)) if realized_count > 0 else 0.0
    win_rate = float(export_df["win"].mean()) if realized_count > 0 else 0.0
    ev_mean = float(export_df["ev_per_100"].mean()) if realized_count > 0 else 0.0

    sharpe_style = 0.0
    if realized_count > 1:
        pnl_std = float(export_df["pnl"].std(ddof=1))
        pnl_mean = float(export_df["pnl"].mean())
        if pnl_std > 0:
            sharpe_style = pnl_mean / pnl_std

    return {
        "meta": {"eval_base_date_max": as_of_date},
        "realized": {
            "count": realized_count,
            "profit_sum": round(profit_sum, 2),
            "roi": round(roi, 4),
            "win_rate": round(win_rate, 4),
            "sharpe_style": round(float(sharpe_style), 4),
        },
        "ev_stats": {"mean": round(ev_mean, 2)},
        "filter_params": {
            "home_win_rate_threshold": float(params.get("home_win_rate_threshold", 0.0)),
            "odds_min": float(params.get("odds_min", 0.0)),
            "odds_max": float(params.get("odds_max", 0.0)),
            "prob_threshold": float(params.get("prob_threshold", 0.0)),
            "min_EV": float(min_ev),
        },
    }


def write_strategy_params(output_dir: Path, *, as_of_date: str, stake: float, min_ev: float, params: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "strategy_params.txt"
    lines = [
        f"as_of_date: {as_of_date}",
        f"min_ev: {float(min_ev)}",
        f"stake: {float(stake)}",
    ]
    n_window = os.environ.get("N_WINDOW")
    if n_window:
        lines.append(f"n_window: {n_window}")
    for key in sorted(params.keys()):
        lines.append(f"{key}: {params[key]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1]
    source_root = resolve_source_root(base_dir)
    output_dir = source_root / "output" / "LightGBM"
    output_dir.mkdir(parents=True, exist_ok=True)

    bet_log_path = find_bet_log(output_dir)
    if bet_log_path is None:
        raw_df = pd.DataFrame()
    else:
        raw_df = pd.read_csv(bet_log_path)

    default_stake = float(os.environ.get("STAKE", 100.0))
    export_df = prepare_export_df(raw_df, default_stake)

    env_as_of = os.environ.get("AS_OF_DATE")
    if env_as_of:
        as_of_date = env_as_of
    elif not export_df.empty and export_df["date"].notna().any():
        as_of_date = str(export_df["date"].max())
    else:
        as_of_date = date.today().strftime("%Y-%m-%d")

    params = {
        "home_win_rate_threshold": os.environ.get("HOME_WIN_RATE_THRESHOLD", 0.0),
        "odds_min": os.environ.get("ODDS_MIN", 0.0),
        "odds_max": os.environ.get("ODDS_MAX", 0.0),
        "prob_threshold": os.environ.get("PROB_THRESHOLD", 0.0),
    }
    min_ev = float(os.environ.get("MIN_EV", 0.0))

    snapshot = build_metrics_snapshot(
        export_df,
        as_of_date=as_of_date,
        stake=default_stake,
        params=params,
        min_ev=min_ev,
    )

    metrics_path = output_dir / "metrics_snapshot.json"
    metrics_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    export_path = output_dir / f"local_matched_games_{as_of_date}.csv"
    export_df.to_csv(export_path, index=False, encoding="utf-8")

    write_strategy_params(
        output_dir,
        as_of_date=as_of_date,
        stake=default_stake,
        min_ev=min_ev,
        params=params,
    )

    realized = snapshot["realized"]
    if len(export_df) != realized["count"]:
        print(
            "WARN: local_matched_games rows mismatch "
            f"(expected {realized['count']}, got {len(export_df)})."
        )
    pnl_sum = float(export_df["pnl"].sum()) if not export_df.empty else 0.0
    if abs(pnl_sum - float(realized["profit_sum"])) > 0.01:
        print(
            "WARN: local_matched_games pnl sum mismatch "
            f"(expected {realized['profit_sum']:.2f}, got {pnl_sum:.2f})."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
