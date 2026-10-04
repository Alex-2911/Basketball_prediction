#!/usr/bin/env python3
"""Guarded 2027 rewrite of Script 2 next-game-day schedule metadata.

The old Script 2 found the next NBA game day and wrote `games_df_<date>.csv`.
For the 2027 paper shell, this remains schedule metadata only.  It must not
fetch schedules, create odds, create predictions, create betting signals, or
unlock canonical betting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nba2027.legacy.script7_cases import _normalize_team_phrase


ROOT = Path(__file__).resolve().parents[1]
SEASON_END_YEAR = 2027
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "next_game_day"
DEFAULT_NEXT_GAME_DIR = ROOT / "data" / "raw" / "Gathering_Data" / "Next_Game"
DEFAULT_NEXT_GAME_BACKUP_DIR = ROOT / "data" / "raw" / "Gathering_Data_Backup" / "Next_Game"
DEFAULT_SCHEDULE_METADATA = (
    ROOT
    / "outputs"
    / "verified_schedule_import"
    / "20261020T120005000000Z"
    / "schedule_metadata.csv"
)
SENSITIVE_MODEL_COLUMNS = {"home_team_prob", "prob_iso", "home_win_rate", "odds_1", "odds_2"}


@dataclass(frozen=True)
class NextGameResult:
    status: str
    reason_code: str
    rows: list[dict[str, Any]]
    message: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S%fZ")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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


def load_schedule_metadata(schedule_path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not schedule_path.exists():
        return None, ["NO_VERIFIED_2027_SCHEDULE_METADATA"]

    frame = pd.read_csv(schedule_path, low_memory=False)
    issues: list[str] = []
    lower_columns = {str(col).lower() for col in frame.columns}
    sensitive = sorted(SENSITIVE_MODEL_COLUMNS & lower_columns)
    if sensitive:
        issues.append("SCHEDULE_CONTAINS_MODEL_OR_ODDS_COLUMNS")

    date_col = _choose_column(frame, ["game_date", "date"])
    home_col = _choose_column(frame, ["home_team", "home_team_raw"])
    away_col = _choose_column(frame, ["away_team", "visitor_team", "away_team_raw", "visitor_team_raw"])
    if not date_col or not home_col or not away_col:
        issues.append("SCHEDULE_MISSING_REQUIRED_COLUMNS")
        return None, issues

    work = pd.DataFrame(
        {
            "game_date": pd.to_datetime(frame[date_col], errors="coerce").dt.date,
            "home_team": frame[home_col].map(_normalize_team_phrase),
            "away_team": frame[away_col].map(_normalize_team_phrase),
        }
    )
    tipoff_col = _choose_column(frame, ["tipoff_utc", "tipoff", "tipoff_raw", "game_time", "start_time", "time", "datetime"])
    work["tipoff_raw"] = frame[tipoff_col] if tipoff_col else None
    work = work.dropna(subset=["game_date"])
    work = work[(work["home_team"] != "") & (work["away_team"] != "")]
    work["game_key"] = (
        work["game_date"].astype(str) + "|" + work["home_team"].astype(str) + "|" + work["away_team"].astype(str)
    )

    duplicates = int(work.duplicated("game_key").sum())
    if duplicates:
        issues.append("DUPLICATE_GAME_KEYS")
        work = work.drop_duplicates("game_key")

    return work.sort_values(["game_date", "home_team", "away_team"]), issues


def find_next_game_day(schedule_path: Path, anchor_date: date) -> NextGameResult:
    schedule, issues = load_schedule_metadata(schedule_path)
    if schedule is None:
        return NextGameResult(
            status="SKIPPED",
            reason_code=issues[0] if issues else "NO_VERIFIED_2027_SCHEDULE_METADATA",
            rows=[],
            message="No verified local 2026-27 schedule metadata is available.",
        )

    if "SCHEDULE_CONTAINS_MODEL_OR_ODDS_COLUMNS" in issues:
        return NextGameResult(
            status="SKIPPED",
            reason_code="SCHEDULE_CONTAINS_MODEL_OR_ODDS_COLUMNS",
            rows=[],
            message="Schedule metadata included prediction or odds columns, so it was not used.",
        )

    upcoming = schedule[schedule["game_date"] >= anchor_date]
    if upcoming.empty:
        return NextGameResult(
            status="SKIPPED",
            reason_code="NO_UPCOMING_GAMES_IN_LOCAL_SCHEDULE",
            rows=[],
            message="A schedule file exists, but it has no games on or after the anchor date.",
        )

    next_date = upcoming["game_date"].min()
    day = upcoming[upcoming["game_date"] == next_date]
    rows = [
        {
            "home_team": row.home_team,
            "away_team": row.away_team,
            "game_date": row.game_date.isoformat(),
            "tipoff_raw": None if pd.isna(row.tipoff_raw) else row.tipoff_raw,
        }
        for row in day.itertuples(index=False)
    ]
    return NextGameResult(
        status="READY",
        reason_code="NEXT_GAME_DAY_METADATA_READY",
        rows=rows,
        message=f"Found {len(rows)} games for next local schedule date {next_date.isoformat()}.",
    )


def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    anchor_date = _parse_date(args.date) if args.date else now.date()
    schedule_path = Path(args.schedule_metadata)
    result = find_next_game_day(schedule_path, anchor_date)
    output_csv: str | None = None
    files_created: list[str] = []
    files_copied: list[str] = []

    if result.status == "READY" and args.write_csv:
        out_dir = Path(args.next_game_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"games_df_{anchor_date.isoformat()}.csv"
        pd.DataFrame(result.rows, columns=["home_team", "away_team", "game_date"]).to_csv(csv_path, index=False)
        output_csv = str(csv_path)
        files_created.append(output_csv)
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / csv_path.name
        shutil.copy2(csv_path, backup_path)
        files_copied.append(str(backup_path))

    return {
        "script": "get_data_next_game_day_2027",
        "season_end_year": SEASON_END_YEAR,
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "anchor_date": anchor_date.isoformat(),
        "status": result.status,
        "reason_code": result.reason_code,
        "message": result.message,
        "source": {
            "schedule_metadata": str(schedule_path),
            "schedule_metadata_exists": schedule_path.exists(),
            "schedule_metadata_sha256": _sha256(schedule_path),
        },
        "results": {
            "games_found": len(result.rows),
            "games": result.rows,
            "output_csv": output_csv,
            "files_created": files_created,
            "files_copied": files_copied,
        },
        "safety": {
            "schedule_metadata_only": True,
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


def write_report(report: dict[str, Any], output_root: Path, run_id: str) -> Path:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded 2027 next-game-day schedule metadata entrypoint.")
    parser.add_argument("--date", help="Anchor date in YYYY-MM-DD form. Finds first local schedule date on/after it.")
    parser.add_argument("--schedule-metadata", default=str(DEFAULT_SCHEDULE_METADATA))
    parser.add_argument("--write-csv", action="store_true", help="Write games_df_<anchor>.csv only when metadata is READY.")
    parser.add_argument("--next-game-dir", default=str(DEFAULT_NEXT_GAME_DIR))
    parser.add_argument("--backup-dir", default=str(DEFAULT_NEXT_GAME_BACKUP_DIR))
    parser.add_argument("--run-id", help="Stable output folder name for evidence. Defaults to current UTC timestamp.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = _utc_now()
    run_id = args.run_id or _run_id(now)
    report = build_report(args, now)
    report_path = write_report(report, Path(args.output_root), run_id)
    print(json.dumps({"status": report["status"], "reason_code": report["reason_code"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
