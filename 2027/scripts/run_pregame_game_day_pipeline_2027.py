#!/usr/bin/env python3
"""Run the guarded 2027 pregame game-day sequence.

Order: Script 1 previous-game-day stats guard -> Script 2 next-game-day
schedule -> Script 3 predictions with odds -> Script 4 statistics ->
Script 5 strategy gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "pregame_game_day_pipeline"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def run_step(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    payload: dict[str, Any] = {"command": command, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    try:
        payload["json"] = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        payload["json"] = None
    return payload


def report_path_from(step: dict[str, Any]) -> Path | None:
    data = step.get("json") or {}
    report = data.get("report")
    return Path(report) if report else None


def output_csv_from_report(report_path: Path | None) -> str | None:
    if not report_path or not report_path.exists():
        return None
    data = json.loads(report_path.read_text())
    results = data.get("results", {})
    return results.get("output_csv") or results.get("predictions_file")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded 2027 pregame game-day pipeline.")
    parser.add_argument("--run-id", default=_run_id(), help="Shared evidence id for this pipeline run.")
    parser.add_argument("--allow-baseline-2026", action="store_true", help="Allow Script 3 to use 2026 baseline stats before 2027 games exist.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    run_id = args.run_id
    steps: dict[str, Any] = {}

    script1_cmd = [str(PYTHON), "scripts/get_data_previous_game_day_2027.py", "--run-id", run_id]
    steps["script1_previous_game_day"] = run_step(script1_cmd)

    script2_cmd = [str(PYTHON), "scripts/get_data_next_game_day_2027.py", "--run-id", run_id, "--write-csv"]
    steps["script2_next_game_day"] = run_step(script2_cmd)

    script3_cmd = [str(PYTHON), "scripts/run_lightgbm_predictions_2027.py", "--run-id", run_id]
    if args.allow_baseline_2026:
        script3_cmd.append("--allow-baseline-2026")
    steps["script3_predictions"] = run_step(script3_cmd)
    script3_csv = output_csv_from_report(report_path_from(steps["script3_predictions"]))

    script4_cmd = [str(PYTHON), "scripts/calculate_betting_statistics_2027.py", "--run-id", run_id]
    if script3_csv:
        script4_cmd.extend(["--predictions", script3_csv])
    steps["script4_statistics"] = run_step(script4_cmd)
    script4_csv = output_csv_from_report(report_path_from(steps["script4_statistics"]))

    script5_cmd = [str(PYTHON), "scripts/run_isotonic_strategy_2027.py", "--run-id", run_id]
    if script4_csv:
        script5_cmd.extend(["--input", script4_csv])
    steps["script5_strategy_gate"] = run_step(script5_cmd)

    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "script": "run_pregame_game_day_pipeline_2027",
        "run_id": run_id,
        "run_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "steps": steps,
        "safety": {
            "numbered_pipeline_order": ["script1_previous_game_day", "script2_next_game_day", "script3_predictions", "script4_statistics", "script5_strategy_gate"],
            "odds_are_part_of_script3": True,
            "canonical_unlock": False,
            "execution_path_created": False,
            "manifold_path_created": False,
        },
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    failed = [name for name, step in steps.items() if step["returncode"] != 0]
    status = "READY" if not failed else "FAILED"
    print(json.dumps({"status": status, "failed_steps": failed, "report": str(report_path)}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
