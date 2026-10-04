#!/usr/bin/env python3
"""Guarded 2027 rewrite of Script 4 betting statistics.

Script 4 consumes Script 3 prediction rows and prepares betting-statistics
evidence.  It does not fetch odds, invent odds, claim profitability, unlock
canonical bets, or create execution paths.  With preseason 2026-baseline
predictions it records pending outcomes and missing odds explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "script4_betting_statistics"
DEFAULT_PREDICTIONS = ROOT / "outputs" / "script3_model_predictions" / "20261020T120008000000Z" / "baseline_2026_predictions.csv"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_predictions(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, ["MISSING_SCRIPT3_PREDICTIONS"]
    frame = _read_table(path)
    required = {"game_date", "home_team", "away_team", "home_team_prob"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return None, [f"PREDICTIONS_MISSING_{col}" for col in missing]
    frame = frame.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["home_team"] = frame["home_team"].astype(str).str.strip()
    frame["away_team"] = frame["away_team"].astype(str).str.strip()
    frame["home_team_prob"] = pd.to_numeric(frame["home_team_prob"], errors="coerce")
    frame = frame.dropna(subset=["game_date", "home_team", "away_team", "home_team_prob"])
    if frame.empty:
        return None, ["EMPTY_OR_INVALID_SCRIPT3_PREDICTIONS"]
    if not frame["home_team_prob"].between(0, 1).all():
        return None, ["INVALID_PROBABILITY"]
    return frame.drop_duplicates(["game_date", "home_team", "away_team"]).reset_index(drop=True), []


def build_statistics(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["predicted_side"] = out["home_team"].where(out["home_team_prob"] >= 0.5, out["away_team"])
    out["prediction_confidence"] = (out["home_team_prob"] - 0.5).abs() * 2
    if "odds_1" not in out.columns:
        out["odds_1"] = pd.NA
    if "odds_2" not in out.columns:
        out["odds_2"] = pd.NA
    out["odds_1"] = pd.to_numeric(out["odds_1"], errors="coerce")
    out["odds_2"] = pd.to_numeric(out["odds_2"], errors="coerce")
    out["result"] = "PENDING"
    out["accuracy"] = pd.NA
    odds_missing = out[["odds_1", "odds_2"]].isna().any(axis=1)
    out["betting_statistics_status"] = "EVIDENCE_ONLY"
    out["reason_code"] = "ODDS_PRESENT_OUTCOME_PENDING"
    out.loc[odds_missing, "reason_code"] = "ODDS_MISSING_OUTCOME_PENDING"
    out["canonical_decision"] = "NO_BET"
    out["watch_label"] = "PREDICTION_WITH_ODDS_OUTCOME_PENDING"
    out.loc[odds_missing, "watch_label"] = "PREDICTION_ONLY_NO_ODDS"
    out["stake"] = 0
    preferred = [
        "game_date",
        "home_team",
        "away_team",
        "home_team_prob",
        "prob_iso",
        "home_win_rate",
        "predicted_side",
        "prediction_confidence",
        "odds_1",
        "odds_2",
        "result",
        "accuracy",
        "canonical_decision",
        "watch_label",
        "stake",
        "betting_statistics_status",
        "reason_code",
        "model_source",
        "model_issue",
    ]
    return out[[col for col in preferred if col in out.columns]]


def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    prediction_path = Path(args.predictions)
    predictions, issues = load_predictions(prediction_path)
    status = "SKIPPED" if issues else "READY"
    reason_code = issues[0] if issues else "BETTING_STATISTICS_EVIDENCE_READY"
    rows = 0
    output_csv = None
    summary: dict[str, Any] = {
        "home_prob_min": None,
        "home_prob_max": None,
        "home_prob_mean": None,
        "pending_outcomes": 0,
        "odds_missing_rows": 0,
    }

    if predictions is not None:
        statistics = build_statistics(predictions)
        rows = int(len(statistics))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "betting_statistics_evidence.csv"
        statistics.to_csv(output_path, index=False)
        output_csv = str(output_path)
        summary = {
            "home_prob_min": float(statistics["home_team_prob"].min()),
            "home_prob_max": float(statistics["home_team_prob"].max()),
            "home_prob_mean": float(statistics["home_team_prob"].mean()),
            "pending_outcomes": int((statistics["result"] == "PENDING").sum()),
            "odds_missing_rows": int(statistics[["odds_1", "odds_2"]].isna().any(axis=1).sum()),
        }

    return {
        "script": "calculate_betting_statistics_2027",
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "status": status,
        "reason_code": reason_code,
        "issues": issues,
        "inputs": {
            "predictions": str(prediction_path),
            "predictions_exists": prediction_path.exists(),
            "predictions_sha256": _sha256(prediction_path),
        },
        "results": {
            "rows": rows,
            "output_csv": output_csv,
            **summary,
        },
        "safety": {
            "betting_statistics_only": True,
            "network_attempted": False,
            "odds_fetched": False,
            "odds_created": False,
            "outcomes_settled": False,
            "profitability_claim": False,
            "betting_signals_created": False,
            "canonical_unlock": False,
            "execution_path_created": False,
            "manifold_path_created": False,
        },
    }


def write_report(report: dict[str, Any], output_root: Path, run_id: str) -> Path:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded 2027 Script 4 betting-statistics evidence step.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS), help="Script 3 prediction CSV or Parquet.")
    parser.add_argument("--run-id", help="Stable output folder name for evidence. Defaults to current UTC timestamp.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = _utc_now()
    run_id = args.run_id or _run_id(now)
    output_dir = Path(args.output_root) / run_id
    args.output_dir = str(output_dir)
    report = build_report(args, now)
    report_path = write_report(report, Path(args.output_root), run_id)
    print(json.dumps({"status": report["status"], "reason_code": report["reason_code"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
