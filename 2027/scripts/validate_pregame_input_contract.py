"""Validate the fresh 2027 pregame input contract.

This is an input-boundary check only. It proves realistic new-season rows fail
closed with reason codes when freshness, timestamps, played status, tipoff
window, or required fields are invalid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-10-20T12:00:01Z"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_row() -> dict:
    return {
        "game_date": "2026-10-20",
        "home_team": "BOS",
        "away_team": "NYK",
        "home_team_prob": 0.63,
        "prob_iso": 0.63,
        "home_win_rate": 0.70,
        "odds_1": 1.70,
        "odds_2": 2.20,
        "tipoff_utc": "2026-10-20T23:00:00Z",
        "as_of_utc": "2026-10-20T11:00:00Z",
        "is_played": False,
    }


def contract_rows() -> list[dict]:
    rows = []
    rows.append({**base_row(), "case_id": "valid_fresh_complete"})
    rows.append({**base_row(), "case_id": "stale_input", "home_team": "LAL", "away_team": "GSW", "as_of_utc": "2026-10-18T11:00:00Z"})
    rows.append({**base_row(), "case_id": "timezone_missing", "home_team": "DAL", "away_team": "PHX", "tipoff_utc": "2026-10-20T23:00:00"})
    rows.append({**base_row(), "case_id": "already_played", "home_team": "MIA", "away_team": "ORL", "is_played": True})
    rows.append({**base_row(), "case_id": "outside_36h_window", "home_team": "MIL", "away_team": "CHI", "tipoff_utc": "2026-10-22T01:00:02Z"})
    incomplete = {**base_row(), "case_id": "missing_required_odds", "home_team": "DEN", "away_team": "UTA"}
    incomplete.pop("odds_1")
    rows.append(incomplete)
    rows.append({**base_row(), "case_id": "game_started", "home_team": "SAC", "away_team": "POR", "tipoff_utc": "2026-10-20T11:30:00Z"})
    rows.append({**base_row(), "case_id": "future_input", "home_team": "TOR", "away_team": "CLE", "as_of_utc": "2026-10-20T13:00:00Z"})
    rows.append({**base_row(), "case_id": "postgame_input", "home_team": "ATL", "away_team": "WAS", "tipoff_utc": "2026-10-20T13:30:00Z", "as_of_utc": "2026-10-20T13:31:00Z"})
    rows.append({**base_row(), "case_id": "invalid_probability", "home_team": "HOU", "away_team": "SAS", "home_team_prob": 1.20})
    return rows


EXPECTED_REASON = {
    "stale_input": "STALE_OR_FUTURE_INPUT",
    "timezone_missing": "MISSING_OR_INVALID_TIMESTAMPS",
    "already_played": "PLAYED_STATUS_NOT_CONFIRMED",
    "outside_36h_window": "OUTSIDE_PREGAME_WINDOW",
    "missing_required_odds": "DATA_INCOMPLETE",
    "game_started": "GAME_STARTED",
    "future_input": "STALE_OR_FUTURE_INPUT",
    "postgame_input": "POSTGAME_INPUT",
    "invalid_probability": "INVALID_PROBABILITY",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/input_contract/20261020T120001000000Z",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = args.output_dir / "pregame_contract_fixture.json"
    cli_output_dir = args.output_dir / "cli_run"
    report_path = args.output_dir / "validation.json"
    fixture_path.write_text(json.dumps(contract_rows(), indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/nba-pregame"),
            "--input",
            str(fixture_path),
            "--now",
            NOW,
            "--output-dir",
            str(cli_output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_summary = json.loads(completed.stdout)
    run_dir = Path(cli_summary["run_dir"])
    candidates = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))
    strategy_evidence = json.loads((run_dir / "strategy_evidence.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    by_case = {row["home_team"]: row for row in candidates}
    source_by_home = {row["home_team"]: row["case_id"] for row in contract_rows()}
    normalized = {source_by_home[home]: out for home, out in by_case.items()}
    valid = normalized["valid_fresh_complete"]

    checks = {
        "cli_completed": cli_summary["rows"] == len(contract_rows()),
        "no_bets_created": cli_summary["bets"] == 0,
        "execution_disabled": cli_summary["execution_enabled"] is False,
        "valid_row_no_input_reason_codes": valid["canonical_decision"] == "NO_BET"
        and valid["blocked_by"] == ["ENGINE_NO_BET"]
        and valid["watch_label"] == "PASS_FILTERS_ONLY",
        "invalid_rows_no_bet": all(normalized[case]["canonical_decision"] == "NO_BET" for case in EXPECTED_REASON),
        "invalid_rows_have_expected_reasons": all(
            expected in normalized[case]["blocked_by"] for case, expected in EXPECTED_REASON.items()
        ),
        "no_history_fallback": manifest["history_sha256"] is None
        and strategy_evidence["reason"] == "frozen June baseline; no history supplied",
        "paper_mode_manifest": manifest["mode"] == "paper",
        "input_hash_recorded": manifest["input_sha256"] == sha256(fixture_path),
    }
    report = {
        "scope": "Fresh 2027 input dry-run contract validation",
        "model_unlock": False,
        "execution_unlock": False,
        "profitability_claim": False,
        "fixture": str(fixture_path),
        "cli_run_dir": str(run_dir),
        "checks": checks,
        "passed": all(checks.values()),
        "expected_reason_codes": EXPECTED_REASON,
        "case_results": [
            {
                "case_id": source_by_home[row["home_team"]],
                "game_key": row["game_key"],
                "canonical_decision": row["canonical_decision"],
                "watch_label": row["watch_label"],
                "blocked_by": row["blocked_by"],
            }
            for row in candidates
        ],
        "notes": [
            "The valid row remains NO_BET because the baseline engine is still frozen.",
            "Invalid rows return reason codes in candidate output; the CLI does not crash.",
            "No history file was supplied and no historical fallback was used.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(report_path), "passed": report["passed"], "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
