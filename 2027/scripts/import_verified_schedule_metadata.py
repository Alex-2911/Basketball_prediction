#!/usr/bin/env python3
"""Import one explicit local schedule file as 2027 schedule metadata only.

This adapter is intentionally stricter than folder discovery.  It consumes only
the source file passed by `--source-file`, validates the shape, normalizes team
names through the preserved Script 7 helpers, and writes schedule metadata.  It
does not fetch data, parse statistics, create predictions, create odds, create
betting signals, or unlock canonical betting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nba2027.legacy.script7_cases import _normalize_team_phrase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "verified_schedule_import"
SENSITIVE_COLUMNS = {"home_team_prob", "prob_iso", "home_win_rate", "odds_1", "odds_2"}
TIPOFF_COLUMNS = ["tipoff_utc", "tipoff", "tipoff_raw", "game_time", "start_time", "time", "datetime", "commence_time"]
SEASON_WINDOWS = {
    "2026-27": ("2026-10-01", "2027-07-01"),
}


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


def _choose_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _status_from_issues(issues: list[str]) -> str:
    return "OK" if not issues else "SKIPPED"


def _season_window(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if season not in SEASON_WINDOWS:
        raise ValueError(f"Unsupported season '{season}'. Supported seasons: {', '.join(sorted(SEASON_WINDOWS))}")
    start, end = SEASON_WINDOWS[season]
    return pd.Timestamp(start), pd.Timestamp(end)


def import_schedule_metadata(source_file: Path, output_dir: Path, season: str = "2026-27") -> dict[str, Any]:
    issues: list[str] = []
    imported_file: str | None = None
    rows: list[dict[str, Any]] = []
    duplicate_count = 0
    timezone_checked = False
    teams_normalized = False

    if not source_file.exists():
        issues.append("SOURCE_FILE_NOT_FOUND")
    elif not source_file.is_file():
        issues.append("SOURCE_PATH_IS_NOT_FILE")

    if issues:
        return {
            "status": "SKIPPED",
            "issues": issues,
            "games_found": 0,
            "teams_normalized": False,
            "duplicates": None,
            "timezone_checked": False,
            "imported_file": None,
            "rows": [],
        }

    frame = _read_source(source_file)
    lower_columns = {str(col).lower() for col in frame.columns}
    sensitive = sorted(SENSITIVE_COLUMNS & lower_columns)
    if sensitive:
        issues.append("SOURCE_CONTAINS_MODEL_OR_ODDS_COLUMNS")

    date_col = _choose_column(frame, ["game_date", "date"])
    home_col = _choose_column(frame, ["home_team", "home_team_raw"])
    away_col = _choose_column(frame, ["away_team", "visitor_team", "away_team_raw", "visitor_team_raw"])
    tipoff_col = _choose_column(frame, TIPOFF_COLUMNS)

    if not date_col:
        issues.append("MISSING_GAME_DATE")
    if not home_col:
        issues.append("MISSING_HOME_TEAM")
    if not away_col:
        issues.append("MISSING_AWAY_TEAM")
    if not tipoff_col:
        issues.append("MISSING_TIPOFF_TIME")

    if issues:
        return {
            "status": "SKIPPED",
            "issues": issues,
            "games_found": 0,
            "teams_normalized": False,
            "duplicates": None,
            "timezone_checked": False,
            "imported_file": None,
            "rows": [],
        }

    season_start, season_end = _season_window(season)
    game_dates = pd.to_datetime(frame[date_col], errors="coerce")
    work = pd.DataFrame(
        {
            "game_date": game_dates.dt.strftime("%Y-%m-%d"),
            "home_team_raw": frame[home_col],
            "away_team_raw": frame[away_col],
            "tipoff_raw": frame[tipoff_col],
            "source_file": str(source_file),
        }
    )
    work["home_team"] = work["home_team_raw"].map(_normalize_team_phrase)
    work["away_team"] = work["away_team_raw"].map(_normalize_team_phrase)
    work["tipoff_utc"] = pd.to_datetime(work["tipoff_raw"], errors="coerce", utc=True)
    work["game_key"] = work["game_date"] + "|" + work["home_team"] + "|" + work["away_team"]

    invalid = work[
        work["game_date"].isna()
        | (work["home_team"] == "")
        | (work["away_team"] == "")
        | work["tipoff_utc"].isna()
    ]
    if not invalid.empty:
        issues.append("INVALID_OR_INCOMPLETE_ROWS")

    in_window = game_dates.ge(season_start) & game_dates.lt(season_end)
    if not bool(in_window.all()):
        issues.append("OUT_OF_SEASON_ROWS")

    duplicate_count = int(work.duplicated("game_key").sum())
    timezone_checked = bool(work["tipoff_utc"].notna().all())
    teams_normalized = bool(
        work["home_team"].map(lambda value: isinstance(value, str) and len(value) == 3).all()
        and work["away_team"].map(lambda value: isinstance(value, str) and len(value) == 3).all()
    )
    if not teams_normalized:
        issues.append("TEAM_NORMALIZATION_FAILED")

    if issues:
        return {
            "status": "SKIPPED",
            "issues": issues,
            "games_found": 0,
            "teams_normalized": teams_normalized,
            "duplicates": duplicate_count,
            "timezone_checked": timezone_checked,
            "imported_file": None,
            "rows": [],
        }

    clean = work.drop_duplicates("game_key").sort_values(["game_date", "home_team", "away_team"])
    clean["tipoff_utc"] = clean["tipoff_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "schedule_metadata.csv"
    clean[
        [
            "game_date",
            "home_team_raw",
            "away_team_raw",
            "home_team",
            "away_team",
            "tipoff_utc",
            "game_key",
            "source_file",
        ]
    ].to_csv(output_path, index=False)
    imported_file = str(output_path)
    rows = clean.to_dict("records")

    return {
        "status": _status_from_issues(issues),
        "issues": issues,
        "games_found": int(len(clean)),
        "teams_normalized": teams_normalized,
        "duplicates": duplicate_count,
        "timezone_checked": timezone_checked,
        "imported_file": imported_file,
        "rows": rows,
    }


def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    source_file = Path(args.source_file)
    output_dir = Path(args.output_dir)
    result = import_schedule_metadata(source_file, output_dir, args.season)
    return {
        "scope": "NBA 2027 verified local schedule import adapter - metadata only",
        "created_utc": now.isoformat().replace("+00:00", "Z"),
        "source": "explicit local file",
        "source_file": str(source_file),
        "source_file_exists": source_file.exists(),
        "source_file_sha256": _sha256(source_file),
        "season": args.season,
        "status": result["status"],
        "issues": result["issues"],
        "games_found": result["games_found"],
        "teams_normalized": "OK" if result["teams_normalized"] else "NOT_OK",
        "duplicates": result["duplicates"],
        "timezone_checked": "OK" if result["timezone_checked"] else "NOT_OK",
        "imported_file": result["imported_file"],
        "sample_rows": result["rows"][:10],
        "safety": {
            "schedule_metadata_only": True,
            "source_discovery_used": False,
            "network_attempted": False,
            "stats_parser_executed": False,
            "previous_game_statistics_modified": False,
            "predictions_created": False,
            "odds_created": False,
            "betting_signals_created": False,
            "canonical_unlock": False,
            "execution_path_created": False,
            "manifold_path_created": False,
        },
    }


def write_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import one explicit local schedule file as metadata only.")
    parser.add_argument("--source-file", "--input", dest="source_file", required=True, help="Explicit local CSV or Parquet schedule file.")
    parser.add_argument("--season", default="2026-27", choices=sorted(SEASON_WINDOWS), help="Season window to validate.")
    parser.add_argument("--run-id", help="Stable output folder name for evidence. Defaults to current UTC timestamp.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", help="Exact output directory. Overrides output-root/run-id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = _utc_now()
    run_id = args.run_id or _run_id(now)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_root) / run_id
    args.output_dir = str(output_dir)
    report = build_report(args, now)
    report_path = write_report(report, output_dir)
    print(json.dumps({"status": report["status"], "games_found": report["games_found"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
