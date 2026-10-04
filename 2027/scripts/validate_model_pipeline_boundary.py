"""Validate that nba-pregame consumes supplied predictions only.

This check proves the pregame shell does not fetch data, generate model
predictions, retrain, backfill, or use archived history unless --history is
explicitly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-10-20T12:00:01Z"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_cli(input_path: Path, output_dir: Path, history_path: Path | None = None) -> tuple[dict, dict, dict]:
    command = [
        str(ROOT / ".venv/bin/nba-pregame"),
        "--input",
        str(input_path),
        "--now",
        NOW,
        "--output-dir",
        str(output_dir),
    ]
    if history_path is not None:
        command.extend(["--history", str(history_path)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads(completed.stdout)
    run_dir = Path(summary["run_dir"])
    candidates = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((run_dir / "strategy_evidence.json").read_text(encoding="utf-8"))
    return summary, {"run_dir": str(run_dir), "candidates": candidates}, {"manifest": manifest, "evidence": evidence}


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


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


def runtime_source_scan() -> dict:
    src_files = sorted((ROOT / "src/nba2027").rglob("*.py"))
    banned_network = re.compile(r"\b(requests|httpx|urllib|aiohttp|socket|selenium|playwright|nba_api)\b")
    banned_process = re.compile(r"\b(subprocess|os\.system|popen)\b")
    model_generation = re.compile(r"\b(LGBMClassifier|LGBMRegressor|lightgbm\.train|\.fit\(|\.predict\()\b")
    findings = {
        "network_or_browser_imports": [],
        "process_spawning": [],
        "model_generation_markers": [],
    }
    allowed_model_files = {
        str(ROOT / "src/nba2027/strategy.py"),
        str(ROOT / "src/nba2027/legacy/live_oos_proxy.py"),
        str(ROOT / "src/nba2027/legacy/script11.py"),
    }
    for path in src_files:
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(ROOT))
        if banned_network.search(text):
            findings["network_or_browser_imports"].append(rel)
        if banned_process.search(text):
            findings["process_spawning"].append(rel)
        if model_generation.search(text) and str(path) not in allowed_model_files:
            findings["model_generation_markers"].append(rel)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/model_pipeline_boundary/20261020T120001000000Z",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    supplied_input = args.output_dir / "explicit_prediction_rows.json"
    model_less_input = args.output_dir / "model_less_rows.json"
    stale_model_less_input = args.output_dir / "stale_model_less_rows.json"
    history_path = ROOT / "data/processed/combined_predictions_2026.parquet"
    report_path = args.output_dir / "validation.json"

    row = base_row()
    write_json(supplied_input, [row])
    model_less = {k: v for k, v in row.items() if k not in {"home_team_prob", "prob_iso", "home_win_rate"}}
    model_less.update({"home_team": "LAL", "away_team": "GSW"})
    write_json(model_less_input, [model_less])
    stale_model_less = {**model_less, "home_team": "DAL", "away_team": "PHX", "as_of_utc": "2026-10-18T11:00:00Z"}
    write_json(stale_model_less_input, [stale_model_less])

    no_history_summary, no_history_out, no_history_meta = run_cli(
        supplied_input,
        args.output_dir / "cli_no_history",
    )
    model_less_summary, model_less_out, model_less_meta = run_cli(
        model_less_input,
        args.output_dir / "cli_model_less",
    )
    stale_model_less_summary, stale_model_less_out, stale_model_less_meta = run_cli(
        stale_model_less_input,
        args.output_dir / "cli_stale_model_less",
    )
    history_summary, history_out, history_meta = run_cli(
        supplied_input,
        args.output_dir / "cli_explicit_history",
        history_path=history_path,
    )
    scan = runtime_source_scan()

    no_history_candidate = no_history_out["candidates"][0]
    model_less_candidate = model_less_out["candidates"][0]
    stale_candidate = stale_model_less_out["candidates"][0]
    history_candidate = history_out["candidates"][0]

    checks = {
        "runtime_has_no_network_or_browser_imports": scan["network_or_browser_imports"] == [],
        "runtime_has_no_process_spawning": scan["process_spawning"] == [],
        "runtime_has_no_model_generation_markers_outside_legacy_history_rules": scan["model_generation_markers"] == [],
        "no_history_run_does_not_use_archived_history": no_history_meta["manifest"]["history_sha256"] is None
        and no_history_meta["evidence"]["reason"] == "frozen June baseline; no history supplied",
        "explicit_history_run_records_history_hash": history_meta["manifest"]["history_sha256"] == sha256(history_path),
        "explicit_history_is_opt_in_only": history_meta["manifest"]["history_sha256"] != no_history_meta["manifest"]["history_sha256"],
        "manifest_records_prediction_input_hashes": no_history_meta["manifest"]["input_sha256"] == sha256(supplied_input)
        and model_less_meta["manifest"]["input_sha256"] == sha256(model_less_input),
        "supplied_prediction_row_is_consumed_not_generated": "prob_used" in no_history_candidate
        and no_history_candidate["canonical_decision"] == "NO_BET"
        and no_history_candidate["blocked_by"] == ["ENGINE_NO_BET"],
        "model_less_input_fails_closed": model_less_candidate["canonical_decision"] == "NO_BET"
        and "DATA_INCOMPLETE" in model_less_candidate["blocked_by"],
        "stale_model_less_input_fails_closed_with_reasons": stale_candidate["canonical_decision"] == "NO_BET"
        and "DATA_INCOMPLETE" in stale_candidate["blocked_by"]
        and "STALE_OR_FUTURE_INPUT" in stale_candidate["blocked_by"],
        "explicit_history_still_does_not_unlock_execution": history_summary["execution_enabled"] is False
        and history_summary["bets"] == 0
        and history_candidate["canonical_decision"] == "NO_BET",
    }
    report = {
        "scope": "Model-pipeline boundary validation",
        "model_unlock": False,
        "execution_unlock": False,
        "profitability_claim": False,
        "checks": checks,
        "passed": all(checks.values()),
        "source_scan": scan,
        "runs": {
            "no_history": {
                "summary": no_history_summary,
                "run_dir": no_history_out["run_dir"],
                "history_sha256": no_history_meta["manifest"]["history_sha256"],
                "input_sha256": no_history_meta["manifest"]["input_sha256"],
                "candidate": no_history_candidate,
            },
            "model_less": {
                "summary": model_less_summary,
                "run_dir": model_less_out["run_dir"],
                "candidate": model_less_candidate,
            },
            "stale_model_less": {
                "summary": stale_model_less_summary,
                "run_dir": stale_model_less_out["run_dir"],
                "candidate": stale_candidate,
            },
            "explicit_history": {
                "summary": history_summary,
                "run_dir": history_out["run_dir"],
                "history_sha256": history_meta["manifest"]["history_sha256"],
                "input_sha256": history_meta["manifest"]["input_sha256"],
                "candidate": history_candidate,
                "evidence_keys": sorted(history_meta["evidence"].keys()),
            },
        },
        "notes": [
            "nba-pregame reads supplied JSON/Parquet prediction rows from --input.",
            "Archived 2026 history is used only when --history is supplied.",
            "Model-less and stale model-less inputs fail closed as NO_BET with reason codes.",
            "No network, browser, process-spawn, or model-training path exists in active runtime files outside the explicit history/rule replay helpers.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(report_path), "passed": report["passed"], "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
