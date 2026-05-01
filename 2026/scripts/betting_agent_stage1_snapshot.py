from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_ROOT = Path(os.getenv("SOURCE_ROOT", Path.cwd())).resolve()
LGBM_DIR = Path(os.getenv("LGBM_DIR", SOURCE_ROOT / "2026" / "output" / "LightGBM")).resolve()

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_latest_date(input_dir: Path, allow_combined_fallback: bool = False) -> str:
    dates: list[str] = []
    patterns = ["nba_games_predict_*.csv"]
    if allow_combined_fallback:
        patterns.append("combined_nba_predictions_acc_*.csv")
    for pattern in patterns:
        for path in input_dir.glob(pattern):
            m = DATE_PATTERN.search(path.name)
            if m:
                dates.append(m.group(1))
    if not dates:
        raise FileNotFoundError("Could not infer target date: no dated prediction files were found.")
    return sorted(set(dates))[-1]


def _coalesce(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for target, candidates in mapping.items():
        if target in out.columns:
            continue
        for col in candidates:
            if col in out.columns:
                out[target] = out[col]
                break
    return out


def _parse_local_report(report_path: Path | None) -> tuple[str | None, dict[str, str]]:
    if report_path is None or not report_path.exists():
        return None, {}
    fields: dict[str, str] = {}
    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    if (
        fields.get("source_rows") == "0"
        and fields.get("rows_exported") == "0"
        and "empty source dataframe" in fields.get("reason_if_nothing_written", "").lower()
    ):
        return "empty_source_dataframe", fields
    return None, fields


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_snapshot(input_dir: Path, output_dir: Path, target_date: str, allow_combined_fallback: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily_path = input_dir / f"nba_games_predict_{target_date}.csv"
    combined_path = input_dir / f"combined_nba_predictions_acc_{target_date}.csv"
    home_rates_path = input_dir / f"home_win_rates_sorted_{target_date}.csv"
    strategy_path = input_dir / "strategy_params.json"
    local_report_path = input_dir / f"local_matched_games_export_report_{target_date}.txt"
    kelly_path = input_dir / "Kelly" / f"bet_shortlist_{target_date}.csv"

    files_found = {
        "daily_predictions": daily_path.exists(),
        "combined_predictions": combined_path.exists(),
        "home_win_rates": home_rates_path.exists(),
        "strategy_params": strategy_path.exists(),
        "local_matched_export_report": local_report_path.exists(),
        "kelly_shortlist": kelly_path.exists(),
    }

    df_daily = _load_csv_if_exists(daily_path)
    df_combined = _load_csv_if_exists(combined_path)
    df_home = _load_csv_if_exists(home_rates_path)
    df_kelly = _load_csv_if_exists(kelly_path)

    canonical_export_status, _ = _parse_local_report(local_report_path if local_report_path.exists() else None)

    if df_daily is None and (df_combined is None or not allow_combined_fallback):
        raise FileNotFoundError("No daily prediction input found for target date")

    base = df_daily.copy() if df_daily is not None else df_combined.copy()
    base = _coalesce(
        base,
        {
            "game_date": ["date"],
            "home_odds": ["odds_1", "odds 1", "closing_home_odds"],
            "away_odds": ["odds_2", "odds 2", "closing_away_odds"],
            "home_prob_raw": ["home_team_prob", "pred_home_win_proba"],
            "prob_live_safe": ["prob_live_oos_proxy"],
        },
    )

    if df_combined is not None:
        c = _coalesce(
            df_combined,
            {"game_date": ["date"], "home_prob_raw": ["home_team_prob", "pred_home_win_proba"]},
        )
        merge_cols = [
            "game_date",
            "home_team",
            "away_team",
            "prob_used",
            "prob_live_safe",
            "prob_iso_oos_time",
            "market_implied_p_devig",
            "model_market_gap",
            "model_market_gap_flag",
            "blocked_by",
        ]
        available = [x for x in merge_cols if x in c.columns]
        if {"game_date", "home_team", "away_team"}.issubset(set(available)):
            c = c[available].drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")
            base = base.merge(c, on=["game_date", "home_team", "away_team"], how="left", suffixes=("", "_c"))

    if df_home is not None and {"home_team", "home_win_rate"}.issubset(df_home.columns):
        base = base.merge(df_home[["home_team", "home_win_rate"]].drop_duplicates("home_team"), on="home_team", how="left")

    base["home_prob_raw"] = pd.to_numeric(base.get("home_prob_raw"), errors="coerce")
    base["away_prob_raw"] = 1.0 - base["home_prob_raw"]
    base["home_odds"] = pd.to_numeric(base.get("home_odds"), errors="coerce")
    base["away_odds"] = pd.to_numeric(base.get("away_odds"), errors="coerce")

    base["home_implied_raw"] = 1.0 / base["home_odds"]
    base["away_implied_raw"] = 1.0 / base["away_odds"]
    denom = base["home_implied_raw"] + base["away_implied_raw"]
    base["home_implied_devig"] = base["home_implied_raw"] / denom
    base["away_implied_devig"] = base["away_implied_raw"] / denom

    base["home_ev_raw_per_100"] = (base["home_prob_raw"] * base["home_odds"] - 1.0) * 100.0
    base["away_ev_raw_per_100"] = (base["away_prob_raw"] * base["away_odds"] - 1.0) * 100.0

    if "prob_used" in base.columns:
        base["prob_used"] = pd.to_numeric(base["prob_used"], errors="coerce")
        base["home_ev_prob_used_per_100"] = (base["prob_used"] * base["home_odds"] - 1.0) * 100.0
    else:
        base["prob_used"] = pd.NA
        base["home_ev_prob_used_per_100"] = pd.NA

    has_kelly_rows = df_kelly is not None and not df_kelly.empty
    kelly_keys: set[tuple[str, str, str]] = set()
    if has_kelly_rows and {"home_team", "away_team"}.issubset(df_kelly.columns):
        if "game_date" not in df_kelly.columns and "date" in df_kelly.columns:
            df_kelly = df_kelly.rename(columns={"date": "game_date"})
        if "game_date" in df_kelly.columns:
            for _, r in df_kelly.iterrows():
                kelly_keys.add((str(r.get("game_date")), str(r.get("home_team")), str(r.get("away_team"))))

    def has_canonical(row: pd.Series) -> bool:
        return (str(row.get("game_date")), str(row.get("home_team")), str(row.get("away_team"))) in kelly_keys

    base["canonical_signal"] = base.apply(has_canonical, axis=1)
    base["canonical_reason"] = base["canonical_signal"].map(lambda x: "official_shortlist_match" if x else "")
    if not has_kelly_rows:
        base.loc[:, "canonical_reason"] = "official_shortlist_empty"

    base["canonical_export_status"] = canonical_export_status or ""

    if "model_market_gap_flag" in base.columns:
        base["model_market_gap_flag"] = base["model_market_gap_flag"].fillna(False).astype(bool)
    else:
        base["model_market_gap_flag"] = False
    if "blocked_by" in base.columns:
        base["blocked_by"] = base["blocked_by"].fillna("")
    else:
        base["blocked_by"] = ""

    def classify(row: pd.Series) -> tuple[str, str, str]:
        if pd.isna(row.get("game_date")) or pd.isna(row.get("home_odds")) or pd.isna(row.get("away_odds")):
            return "DATA_INCOMPLETE", "missing slate fields", "none"
        if bool(row.get("canonical_signal")):
            return "CANONICAL_MODEL_SIGNAL", "official shortlist or local matched signal", "canonical_review"
        if bool(row.get("model_market_gap_flag")) or "MODEL_MARKET_GAP" in str(row.get("blocked_by", "")):
            return "MODEL_MARKET_GAP_REVIEW", "market gap guard triggered", "market_gap_review"
        if row.get("away_ev_raw_per_100", -999) > 0:
            return "RAW_UNDERDOG_TEMPTATION", "away raw EV positive without canonical signal", "near_miss_review"
        if row.get("home_prob_raw", 0) >= 0.60 and row.get("home_ev_raw_per_100", 0) <= 0 and row.get("home_odds", 99) <= 1.60:
            return "LOW_PRICE_FAVORITE_NO_VALUE", "short-priced favorite with non-positive raw EV", "none"
        if row.get("home_ev_raw_per_100", -999) > 0 or row.get("away_ev_raw_per_100", -999) > 0:
            return "LIVE_WATCH_ONLY", "interesting raw edge needs game-flow confirmation", "live_watch_review"
        return "NO_VALUE_SKIP", "no meaningful raw edge and no canonical signal", "none"

    classified = base.apply(classify, axis=1, result_type="expand")
    base[["stage1_bucket", "stage1_reason", "allowed_review_type"]] = classified

    base["target_date"] = target_date
    base["source_daily_file"] = daily_path.name if daily_path.exists() else ""
    base["source_combined_file"] = combined_path.name if combined_path.exists() else ""

    out_cols = [
        "target_date","game_date","home_team","away_team","home_prob_raw","away_prob_raw","home_odds","away_odds",
        "home_implied_raw","away_implied_raw","home_implied_devig","away_implied_devig","home_ev_raw_per_100","away_ev_raw_per_100",
        "prob_used","home_ev_prob_used_per_100","prob_live_safe","prob_iso_oos_time","market_implied_p_devig","model_market_gap",
        "model_market_gap_flag","blocked_by","home_win_rate","canonical_signal","canonical_reason","canonical_export_status",
        "stage1_bucket","stage1_reason","allowed_review_type","source_daily_file","source_combined_file"
    ]
    for col in out_cols:
        if col not in base.columns:
            base[col] = pd.NA
    snapshot = base[out_cols].copy()

    canonical_rows = int(snapshot["canonical_signal"].fillna(False).astype(bool).sum())
    manifest = {
        "target_date": target_date,
        "created_utc": _utc_now_iso(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "files_found": files_found,
        "files_missing": [k for k, v in files_found.items() if not v],
        "canonical_rows_found": canonical_rows,
        "daily_games_found": int(len(snapshot)),
        "status": "ok" if canonical_rows > 0 else "ok_no_canonical_bets",
    }
    return snapshot, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(LGBM_DIR))
    parser.add_argument("--output-dir", default=str(LGBM_DIR / "betting_agent_stage1"))
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--allow-combined-fallback", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_date = args.target_date or _infer_latest_date(input_dir, allow_combined_fallback=args.allow_combined_fallback)

    daily_required_path = input_dir / f"nba_games_predict_{target_date}.csv"
    if not daily_required_path.exists() and (args.target_date is not None or not args.allow_combined_fallback):
        empty = pd.DataFrame(columns=["target_date","game_date","home_team","away_team"])
        manifest = {"target_date": target_date, "created_utc": _utc_now_iso(), "input_dir": str(input_dir.resolve()), "output_dir": str(output_dir.resolve()), "files_found": {"daily_predictions": False}, "files_missing": ["daily_predictions"], "canonical_rows_found": 0, "daily_games_found": 0, "status": "data_incomplete_daily_predictions_missing"}
        snapshot = empty
    else:
        snapshot, manifest = build_snapshot(input_dir=input_dir, output_dir=output_dir, target_date=target_date, allow_combined_fallback=args.allow_combined_fallback)

    csv_path = output_dir / f"stage1_daily_snapshot_{target_date}.csv"
    json_path = output_dir / f"stage1_daily_snapshot_{target_date}.json"
    manifest_path = output_dir / f"stage1_manifest_{target_date}.json"

    snapshot.to_csv(csv_path, index=False)

    if "canonical_signal" not in snapshot.columns:
        snapshot["canonical_signal"] = False
    if "stage1_bucket" not in snapshot.columns:
        snapshot["stage1_bucket"] = "DATA_INCOMPLETE"
    if "allowed_review_type" not in snapshot.columns:
        snapshot["allowed_review_type"] = "none"
    if "home_prob_raw" not in snapshot.columns:
        snapshot["home_prob_raw"] = pd.NA
    if "home_odds" not in snapshot.columns:
        snapshot["home_odds"] = pd.NA
    if "away_odds" not in snapshot.columns:
        snapshot["away_odds"] = pd.NA

    summary = {
        "games": int(len(snapshot)),
        "canonical_model_signals": int(snapshot["canonical_signal"].fillna(False).astype(bool).sum()),
        "raw_underdog_temptations": int((snapshot["stage1_bucket"] == "RAW_UNDERDOG_TEMPTATION").sum()),
        "market_gap_reviews": int((snapshot["stage1_bucket"] == "MODEL_MARKET_GAP_REVIEW").sum()),
        "live_watch_only": int((snapshot["stage1_bucket"] == "LIVE_WATCH_ONLY").sum()),
        "no_value_skips": int((snapshot["stage1_bucket"] == "NO_VALUE_SKIP").sum()),
    }
    compact_games = [
        {
            "game": f"{r.home_team} vs {r.away_team}",
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_prob_raw": r.home_prob_raw,
            "home_odds": r.home_odds,
            "away_odds": r.away_odds,
            "canonical_signal": bool(r.canonical_signal),
            "stage1_bucket": r.stage1_bucket,
            "allowed_review_type": r.allowed_review_type,
        }
        for r in snapshot.itertuples(index=False)
    ]
    json_payload = {"target_date": target_date, "summary": summary, "games": compact_games, "manifest": manifest}
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    snapshot.to_csv(output_dir / "stage1_daily_snapshot_latest.csv", index=False)
    (output_dir / "stage1_daily_snapshot_latest.json").write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")
    (output_dir / "stage1_manifest_latest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
