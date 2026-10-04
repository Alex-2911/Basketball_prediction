"""Validate evidence-only outcome reconciliation for paper candidates.

This check proves settled results can be joined after the fact without changing
pregame decisions, configs, or execution state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=ROOT / "outputs/replay_validations/20261020T120001000000Z",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/outcome_reconciliation/20261020T120001000000Z",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "settled_results_fixture.csv"
    reconciled_path = args.output_dir / "reconciled_cases.csv"
    report_path = args.output_dir / "validation.json"

    case_path = args.cases_dir / "betting_agent_training_cases_latest.jsonl"
    candidate_path = args.cases_dir / "candidates.json"
    manifest_path = args.cases_dir / "manifest.json"
    config_path = ROOT / "configs/baseline.json"

    before_hashes = {
        "case_sha256": sha256(case_path),
        "candidate_sha256": sha256(candidate_path),
        "manifest_sha256": sha256(manifest_path),
        "config_sha256": sha256(config_path),
    }

    candidate_rows = json.loads(candidate_path.read_text(encoding="utf-8"))
    result_rows = [
        {
            "date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "win": 1,
        }
        for row in candidate_rows
    ]
    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    if reconciled_path.exists():
        reconciled_path.unlink()

    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/reconcile_cases.py"),
            "--cases-dir",
            str(args.cases_dir),
            "--results",
            str(results_path),
            "--output",
            str(reconciled_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    reconcile_summary = json.loads(completed.stdout)
    reconciled = pd.read_csv(reconciled_path)
    original_cases = [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    after_hashes = {
        "case_sha256": sha256(case_path),
        "candidate_sha256": sha256(candidate_path),
        "manifest_sha256": sha256(manifest_path),
        "config_sha256": sha256(config_path),
    }

    checks = {
        "reconciler_matched_all_cases": reconcile_summary["settled_matches"] == reconcile_summary["cases"],
        "reconciler_left_no_unmatched_cases": reconcile_summary["unmatched"] == 0,
        "reconciled_file_has_win_column": "win" in reconciled.columns,
        "original_cases_remain_pending": all(
            record.get("outcome", {}).get("result") == "PENDING" for record in original_cases
        ),
        "original_case_files_unchanged": before_hashes == after_hashes,
        "canonical_decisions_unchanged": set(reconciled["canonical_decision"].dropna()) == {"NO_BET"},
        "actions_remain_paper_observation": set(reconciled["actual_action"].dropna()) == {"PAPER_OBSERVATION"},
        "stakes_remain_zero": float(pd.to_numeric(reconciled["stake"], errors="coerce").fillna(0).sum()) == 0.0,
    }

    report = {
        "scope": "Outcome reconciliation validation only",
        "model_unlock": False,
        "execution_unlock": False,
        "profitability_claim": False,
        "checks": checks,
        "passed": all(checks.values()),
        "cases_dir": str(args.cases_dir),
        "results_fixture": str(results_path),
        "reconciled_output": str(reconciled_path),
        "reconcile_summary": reconcile_summary,
        "hashes": before_hashes,
        "reconciled_decisions": reconciled[
            ["date", "home_team", "away_team", "canonical_decision", "watchlist_decision", "actual_action", "stake", "result", "win"]
        ].to_dict("records"),
        "notes": [
            "Settled result is joined into a separate reconciliation output only.",
            "Original JSONL case remains PENDING and unchanged.",
            "Pregame candidate output, manifest, and baseline config hashes are unchanged.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(report_path), "passed": report["passed"], "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
