"""Validate Script 7 case loading and archived reconciliation inputs.

This is a normalization/parity check only. It does not alter model settings,
choose strategy parameters, or unlock canonical betting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from nba2027.legacy.script7_cases import (
    _extract_matchup_teams,
    _normalize_agent_training_record,
    _normalize_team_phrase,
    load_betting_agent_training_cases,
)


ROOT = Path(__file__).resolve().parents[1]
LIGHTGBM = ROOT / "data/baseline_2026/LightGBM"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        records.append(_normalize_agent_training_record(record, path, line_no))
    return records


def file_summary(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        rows.append(
            {
                "file": path.name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "sha256": sha256(path),
            }
        )
    return rows


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/script7_parity/latest_validation.json")
    args = parser.parse_args()

    alias_expectations = {
        "Knicks": "NYK",
        "New York Knicks": "NYK",
        "LA Lakers": "LAL",
        "San Antonio Spurs": "SAS",
        "Phoenix Suns": "PHX",
        "BKN": "BRK",
        "CHA": "CHO",
    }
    alias_results = {raw: _normalize_team_phrase(raw) for raw in alias_expectations}
    matchup_results = {
        "SAS @ NYK": _extract_matchup_teams("SAS @ NYK"),
        "New York Knicks vs San Antonio Spurs": _extract_matchup_teams("New York Knicks vs San Antonio Spurs"),
        "LAL at BOS": _extract_matchup_teams("LAL at BOS"),
    }

    jsonl_files = sorted(LIGHTGBM.glob("betting_agent_training_cases*.jsonl"))
    latest_cases = load_betting_agent_training_cases([LIGHTGBM])
    all_case_rows = []
    per_jsonl = []
    for path in jsonl_files:
        rows = read_jsonl_records(path)
        all_case_rows.extend(rows)
        per_jsonl.append({"file": path.name, "rows": len(rows), "sha256": sha256(path)})
    all_cases = pd.DataFrame(all_case_rows)

    model_files = sorted(p for p in LIGHTGBM.glob("model_vs_actual_vs_user_bets*.csv") if p.is_file())
    script11_files = sorted(p for p in LIGHTGBM.glob("script11_watchlist_history*.csv") if p.is_file())
    latest_model = pd.read_csv(LIGHTGBM / "model_vs_actual_vs_user_bets_latest.csv")
    latest_script11 = pd.read_csv(LIGHTGBM / "script11_watchlist_history_latest.csv")

    latest_model["date"] = pd.to_datetime(latest_model["date"], errors="coerce").dt.normalize()
    latest_model["home_team_norm"] = latest_model["home_team"].map(_normalize_team_phrase)
    latest_model["away_team_norm"] = latest_model["away_team"].map(_normalize_team_phrase)
    latest_cases_for_merge = latest_cases.copy()
    merged = latest_cases_for_merge.merge(
        latest_model[["date", "home_team_norm", "away_team_norm", "actual_winner", "script11_decision", "decision_bucket"]],
        left_on=["date", "home_team", "away_team"],
        right_on=["date", "home_team_norm", "away_team_norm"],
        how="left",
        validate="one_to_one",
    )

    checks = {
        "team_aliases_match_expected": alias_results == alias_expectations,
        "latest_jsonl_loaded": int(len(latest_cases)) == 4,
        "all_jsonl_rows_reconstruct": int(len(all_cases)) == sum(item["rows"] for item in per_jsonl),
        "latest_cases_have_game_keys": bool(latest_cases["game_key"].astype(str).str.len().gt(0).all()),
        "latest_cases_match_latest_snapshot": int(merged["actual_winner"].notna().sum()) == int(len(latest_cases)),
        "latest_snapshot_has_script11_columns": all(
            col in latest_model.columns
            for col in ["script11_decision", "script11_stage2_candidate_type", "script11_blocked_by", "decision_bucket"]
        ),
        "script11_latest_has_rule_columns": all(
            col in latest_script11.columns
            for col in ["params_chosen", "stage2_candidate_type", "canonical_signal", "allowed_review_type"]
        ),
    }

    report = {
        "scope": "Script 7 parity / data normalization validation only",
        "model_unlock": False,
        "execution_unlock": False,
        "source_root": str(LIGHTGBM),
        "checks": checks,
        "passed": all(checks.values()),
        "team_alias_results": alias_results,
        "matchup_parse_results": {k: list(v) for k, v in matchup_results.items()},
        "jsonl_files": per_jsonl,
        "latest_cases": {
            "rows": int(len(latest_cases)),
            "unique_game_keys": int(latest_cases["game_key"].nunique()),
            "canonical_decisions": latest_cases["canonical_decision"].fillna("NA").value_counts().to_dict(),
            "watchlist_decisions": latest_cases["watchlist_decision"].fillna("NA").value_counts().to_dict(),
        },
        "case_reconstruction_against_latest_snapshot": {
            "cases": int(len(merged)),
            "matched": int(merged["actual_winner"].notna().sum()),
            "unmatched": int(merged["actual_winner"].isna().sum()),
            "matched_keys": json_records(merged[["date", "home_team", "away_team", "canonical_decision", "watchlist_decision", "script11_decision", "decision_bucket"]]
            .assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
            ),
        },
        "historical_snapshot_handling": {
            "model_vs_actual_snapshot_files": len(model_files),
            "script11_watchlist_snapshot_files": len(script11_files),
            "model_vs_actual_latest": file_summary([LIGHTGBM / "model_vs_actual_vs_user_bets_latest.csv"])[0],
            "script11_watchlist_latest": file_summary([LIGHTGBM / "script11_watchlist_history_latest.csv"])[0],
            "model_vs_actual_latest_unique_game_keys": int(latest_model["game_key"].nunique()),
            "script11_latest_unique_game_keys": int(latest_script11["game_key"].nunique()),
        },
        "notes": [
            "Latest JSONL loader intentionally prefers betting_agent_training_cases_latest.jsonl when present.",
            "Daily snapshot files are counted as archived snapshots, not concatenated independent games.",
            "This validates normalization and reconstruction only; profitability and model unlock remain out of scope.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
