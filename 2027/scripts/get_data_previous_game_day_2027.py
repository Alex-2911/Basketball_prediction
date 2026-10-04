#!/usr/bin/env python3
"""Guarded 2027 rewrite of Script 1 previous-game-day statistics collection.

The 2026 notebook/script ended with the June 14, 2026 statistics snapshot.  This
2027 entrypoint must not rerun that final season state while the new season has
no played-game statistics yet.  It therefore starts with a local guard and exits
cleanly before any network or parsing path can run.

This script is metadata/statistics-only.  It does not create predictions, odds,
betting candidates, canonical BET signals, schedules, Manifold orders, or live
execution paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEASON_END_YEAR = 2027
FINAL_2026_SNAPSHOT = (
    ROOT
    / "data"
    / "baseline_2026"
    / "Gathering_Data"
    / "Whole_Statistic"
    / "nba_games_2026-06-14.csv"
)
STATISTICS_DIR_2027 = ROOT / "data" / "raw" / "Gathering_Data" / "Whole_Statistic"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "previous_game_day"


@dataclass(frozen=True)
class GuardResult:
    status: str
    reason_code: str
    message: str
    allowed_to_collect: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _existing_2027_stat_files(statistics_dir: Path = STATISTICS_DIR_2027) -> list[Path]:
    if not statistics_dir.exists():
        return []
    return sorted(statistics_dir.glob("nba_games_2027-*.csv"))


def resolve_target_collect_date(args: argparse.Namespace, now: datetime) -> date:
    if args.collect_date:
        return _parse_date(args.collect_date)
    current_date = _parse_date(args.date) if args.date else now.date()
    return current_date - timedelta(days=1)


def evaluate_first_game_guard(
    *,
    final_2026_snapshot: Path = FINAL_2026_SNAPSHOT,
    statistics_dir_2027: Path = STATISTICS_DIR_2027,
) -> GuardResult:
    """Return whether Script 1 is allowed to collect new 2027 played games."""

    final_2026_done = final_2026_snapshot.exists()
    existing_2027_stats = _existing_2027_stat_files(statistics_dir_2027)

    if final_2026_done and not existing_2027_stats:
        return GuardResult(
            status="SKIPPED",
            reason_code="WAITING_FOR_FIRST_2027_PLAYED_GAME",
            message=(
                "Final 2026 statistics snapshot is already present and no "
                "2027 played-game statistics snapshot exists yet. The script "
                "will not rerun the last 2026 game state."
            ),
            allowed_to_collect=False,
        )

    return GuardResult(
        status="READY",
        reason_code="FIRST_2027_PLAYED_GAME_GUARD_PASSED",
        message="A 2027 statistics snapshot exists or the final 2026 guard anchor is missing.",
        allowed_to_collect=True,
    )


def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    target_collect_date = resolve_target_collect_date(args, now)
    existing_2027_stats = _existing_2027_stat_files()
    guard = evaluate_first_game_guard()

    status = guard.status
    reason_code = guard.reason_code
    network_attempted = False
    files_created: list[str] = []
    files_modified: list[str] = []
    games_imported = 0
    parser_executed = False

    if guard.allowed_to_collect and not args.allow_network:
        status = "SKIPPED"
        reason_code = "NETWORK_DISABLED_FOR_SAFE_DRY_RUN"

    if guard.allowed_to_collect and args.allow_network:
        status = "SKIPPED"
        reason_code = "FETCH_AND_PARSE_PORT_NOT_ENABLED_YET"

    return {
        "script": "get_data_previous_game_day_2027",
        "season_end_year": SEASON_END_YEAR,
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "target_collect_date": target_collect_date.isoformat(),
        "status": status,
        "reason_code": reason_code,
        "message": guard.message,
        "safety": {
            "metadata_statistics_only": True,
            "network_attempted": network_attempted,
            "parser_executed": parser_executed,
            "predictions_created": False,
            "odds_created": False,
            "betting_signals_created": False,
            "canonical_unlock": False,
            "execution_path_created": False,
            "manifold_path_created": False,
        },
        "guard": {
            "final_2026_snapshot": str(FINAL_2026_SNAPSHOT),
            "final_2026_snapshot_exists": FINAL_2026_SNAPSHOT.exists(),
            "final_2026_snapshot_sha256": _sha256(FINAL_2026_SNAPSHOT),
            "final_2026_snapshot_rows": _csv_row_count(FINAL_2026_SNAPSHOT),
            "statistics_dir_2027": str(STATISTICS_DIR_2027),
            "existing_2027_stat_files": [str(path) for path in existing_2027_stats],
            "existing_2027_stat_file_count": len(existing_2027_stats),
        },
        "results": {
            "games_imported": games_imported,
            "teams_normalized": "NOT_RUN_WAITING_FOR_FIRST_2027_PLAYED_GAME"
            if reason_code == "WAITING_FOR_FIRST_2027_PLAYED_GAME"
            else "NOT_RUN",
            "duplicates_checked": False,
            "timezone_tipoff_checked": False,
            "files_created": files_created,
            "files_modified": files_modified,
        },
    }


def write_report(report: dict[str, Any], output_root: Path, run_id: str) -> Path:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded 2027 previous-game-day statistics collection entrypoint."
    )
    parser.add_argument(
        "--date",
        help="Current date in YYYY-MM-DD form. The target collect date is date minus one day.",
    )
    parser.add_argument(
        "--collect-date",
        help="Exact game date to collect in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Reserved for a later reviewed fetch/parse port. Current script still fails closed.",
    )
    parser.add_argument(
        "--run-id",
        help="Stable output folder name for evidence. Defaults to current UTC timestamp.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder for run evidence.",
    )
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
