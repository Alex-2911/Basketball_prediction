from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE_THRESHOLDS = {
    "home_team_prob_min": 0.60,
    "odds_1_min": 1.60,
    "home_win_rate_min": 0.50,
}

WINDOWS = {
    "home_team_prob": 0.05,
    "odds_1": 0.35,
    "home_win_rate": 0.10,
}

OUT_DIR = Path("2026/output/LightGBM")


def _extract_date_from_name(path: Path) -> str:
    stem = path.stem
    prefix = "combined_nba_predictions_acc_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return ""


def _latest_input_file() -> Path:
    files = list(OUT_DIR.glob("combined_nba_predictions_acc_*.csv"))
    if not files:
        raise FileNotFoundError("No combined_nba_predictions_acc_*.csv files found.")
    dated = [(f, _extract_date_from_name(f)) for f in files]
    dated = [x for x in dated if x[1]]
    if not dated:
        return max(files, key=lambda p: p.stat().st_mtime)
    return max(dated, key=lambda t: t[1])[0]


def _derive_home_win(row: pd.Series) -> float | None:
    htw = row.get("home_team_won")
    if pd.notna(htw):
        try:
            v = float(htw)
            if v in (0.0, 1.0):
                return v
        except Exception:
            pass

    result = str(row.get("result", "")).strip()
    home = str(row.get("home_team", "")).strip()
    away = str(row.get("away_team", "")).strip()
    if not result:
        return None
    if result in {"0", "1", "0.0", "1.0"}:
        return None
    if result == home:
        return 1.0
    if result == away:
        return 0.0
    return None


def _base_filter(df: pd.DataFrame) -> pd.Series:
    return (
        (df["home_team_prob"] > BASE_THRESHOLDS["home_team_prob_min"])
        & (df["odds_1"] > BASE_THRESHOLDS["odds_1_min"])
        & (df["home_win_rate"] > BASE_THRESHOLDS["home_win_rate_min"])
    )


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    n = int(len(df))
    wins = int(df["home_win"].sum()) if n else 0
    losses = n - wins
    avg_odds = float(df["odds_1"].mean()) if n else 0.0
    profit = float(df["home_ml_pnl_100"].sum()) if n else 0.0
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate_%": round((wins / n) * 100, 2) if n else 0.0,
        "avg_odds": round(avg_odds, 4),
        "profit_100_flat": round(profit, 2),
        "roi_%": round((profit / (n * 100)) * 100, 2) if n else 0.0,
        "avg_home_team_prob": round(float(df["home_team_prob"].mean()), 4) if n else 0.0,
        "avg_home_win_rate": round(float(df["home_win_rate"].mean()), 4) if n else 0.0,
    }


def main() -> None:
    run_date = datetime.now(timezone.utc).date().isoformat()
    created_utc = datetime.now(timezone.utc).isoformat()

    input_file = _latest_input_file()
    df = pd.read_csv(input_file)

    numeric_cols = ["home_team_prob", "odds_1", "odds_2", "home_win_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["home_win"] = df.apply(_derive_home_win, axis=1)
    settled = df[df["home_win"].isin([0.0, 1.0])].copy()
    settled["home_ml_pnl_100"] = settled.apply(
        lambda r: (r["odds_1"] - 1.0) * 100.0 if r["home_win"] == 1.0 else -100.0,
        axis=1,
    )

    candidates = df[df["home_win"].isna() & _base_filter(df)].copy()

    base_hist = settled[_base_filter(settled)].copy()

    rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []

    general = _summary(base_hist)
    rows.append(
        {
            "run_date": run_date,
            "setup": "general_base_setup",
            "today_home_team": "",
            "today_away_team": "",
            "today_home_team_prob": "",
            "today_odds_1": "",
            "today_odds_2": "",
            "today_home_win_rate": "",
            **general,
        }
    )

    per_game_summaries: list[dict[str, Any]] = []

    for _, cand in candidates.iterrows():
        hprob = float(cand["home_team_prob"])
        odds_1 = float(cand["odds_1"])
        hwr = float(cand["home_win_rate"])

        bucket = base_hist[
            base_hist["home_team_prob"].between(hprob - WINDOWS["home_team_prob"], hprob + WINDOWS["home_team_prob"])
            & base_hist["odds_1"].between(odds_1 - WINDOWS["odds_1"], odds_1 + WINDOWS["odds_1"])
            & base_hist["home_win_rate"].between(hwr - WINDOWS["home_win_rate"], hwr + WINDOWS["home_win_rate"])
        ].copy()

        bucket_summary = _summary(bucket)

        row = {
            "run_date": run_date,
            "setup": "per_game_similar_bucket",
            "today_home_team": cand.get("home_team", ""),
            "today_away_team": cand.get("away_team", ""),
            "today_home_team_prob": hprob,
            "today_odds_1": odds_1,
            "today_odds_2": float(cand.get("odds_2", float("nan"))),
            "today_home_win_rate": hwr,
            **bucket_summary,
        }
        rows.append(row)
        per_game_summaries.append(row)

        if not bucket.empty:
            sample = bucket[
                [
                    "game_date", "home_team", "away_team", "home_team_prob", "home_win_rate",
                    "odds_1", "odds_2", "home_win", "home_ml_pnl_100",
                ]
            ].copy()
            sample.insert(0, "today_away_team", cand.get("away_team", ""))
            sample.insert(0, "today_home_team", cand.get("home_team", ""))
            sample.insert(0, "setup", "per_game_similar_bucket")
            sample.insert(0, "run_date", run_date)
            match_rows.extend(sample.to_dict(orient="records"))

    scan_df = pd.DataFrame(rows)
    matches_df = pd.DataFrame(match_rows, columns=[
        "run_date", "setup", "today_home_team", "today_away_team", "game_date", "home_team", "away_team",
        "home_team_prob", "home_win_rate", "odds_1", "odds_2", "home_win", "home_ml_pnl_100",
    ])

    summary = {
        "run_date": run_date,
        "created_utc": created_utc,
        "input_file": str(input_file),
        "thresholds": {"base": BASE_THRESHOLDS, "windows": WINDOWS},
        "candidate_count": int(len(candidates)),
        "general_setup_summary": general,
        "per_game_summaries": per_game_summaries,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_latest = OUT_DIR / "setup_profitability_scan_latest.csv"
    scan_dated = OUT_DIR / f"setup_profitability_scan_{run_date}.csv"
    matches_latest = OUT_DIR / "setup_profitability_scan_matches_latest.csv"
    matches_dated = OUT_DIR / f"setup_profitability_scan_matches_{run_date}.csv"
    summary_latest = OUT_DIR / "setup_profitability_scan_summary_latest.json"
    summary_dated = OUT_DIR / f"setup_profitability_scan_summary_{run_date}.json"

    scan_df.to_csv(scan_latest, index=False)
    scan_df.to_csv(scan_dated, index=False)
    matches_df.to_csv(matches_latest, index=False)
    matches_df.to_csv(matches_dated, index=False)
    summary_latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_dated.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input file: {input_file}")
    print(f"Candidates found: {len(candidates)}")
    print(f"Wrote: {scan_latest}")


if __name__ == "__main__":
    main()
