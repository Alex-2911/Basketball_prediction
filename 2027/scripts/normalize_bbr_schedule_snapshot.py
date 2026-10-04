#!/usr/bin/env python3
"""Normalize a browser-extracted Basketball-Reference schedule snapshot.

This script performs no network access.  It reads a JSON snapshot already saved
from the user-provided Basketball-Reference page, converts ET tipoff times to
UTC, and writes a local schedule CSV that can be consumed by the verified
schedule metadata adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "bbr_schedule_source"
ET = ZoneInfo("America/New_York")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bbr_datetime(date_text: str, start_et: str) -> datetime:
    clean_time = start_et.strip().lower()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})([ap])", clean_time)
    if not match:
        raise ValueError(f"Unsupported BBR time '{start_et}'")
    hour = int(match.group(1))
    minute = int(match.group(2))
    suffix = match.group(3)
    if suffix == "p" and hour != 12:
        hour += 12
    if suffix == "a" and hour == 12:
        hour = 0
    day = datetime.strptime(date_text, "%a, %b %d, %Y")
    local = day.replace(hour=hour, minute=minute, tzinfo=ET)
    return local.astimezone(timezone.utc)


def normalize_snapshot(snapshot_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(snapshot_path.read_text())
    source_rows = payload.get("rows") or []
    normalized_rows: list[dict[str, Any]] = []
    issues: list[str] = []

    for idx, row in enumerate(source_rows, start=1):
        date_text = str(row.get("date") or "").strip()
        start_et = str(row.get("start_et") or "").strip()
        visitor = str(row.get("visitor") or "").strip()
        home = str(row.get("home") or "").strip()
        if date_text == "Date" or start_et.startswith("Start") or visitor in {"Visitor/Neutral", "Visitor Points"}:
            continue
        if not date_text or not start_et or not visitor or not home:
            issues.append(f"ROW_{idx}_INCOMPLETE")
            continue
        try:
            tipoff_utc = parse_bbr_datetime(date_text, start_et)
        except ValueError:
            issues.append(f"ROW_{idx}_BAD_TIPOFF")
            continue
        game_date = datetime.strptime(date_text, "%a, %b %d, %Y").strftime("%Y-%m-%d")
        normalized_rows.append(
            {
                "game_date": game_date,
                "home_team": home,
                "away_team": visitor,
                "tipoff_utc": tipoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tipoff_et": start_et,
                "arena": row.get("arena") or "",
                "notes": row.get("notes") or "",
                "source_url": row.get("page_url") or payload.get("source_url") or "",
                "source_date_text": date_text,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = output_dir / "bbr_nba_2027_schedule_browser_snapshot.json"
    if snapshot_path.resolve() != raw_copy.resolve():
        shutil.copy2(snapshot_path, raw_copy)

    csv_path = output_dir / "bbr_nba_2027_schedule_explicit_source.csv"
    fieldnames = [
        "game_date",
        "home_team",
        "away_team",
        "tipoff_utc",
        "tipoff_et",
        "arena",
        "notes",
        "source_url",
        "source_date_text",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    duplicate_keys = set()
    seen = set()
    for row in normalized_rows:
        key = (row["game_date"], row["home_team"], row["away_team"])
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)

    report = {
        "scope": "Basketball-Reference 2026-27 schedule browser snapshot normalization",
        "created_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "source_url": payload.get("source_url"),
        "raw_snapshot": str(raw_copy),
        "raw_snapshot_sha256": _sha256(raw_copy),
        "output_csv": str(csv_path),
        "output_csv_sha256": _sha256(csv_path),
        "rows_in_snapshot": len(source_rows),
        "rows_written": len(normalized_rows),
        "date_min": min((row["game_date"] for row in normalized_rows), default=None),
        "date_max": max((row["game_date"] for row in normalized_rows), default=None),
        "duplicates": len(duplicate_keys),
        "issues": issues,
        "safety": {
            "browser_snapshot_input_only": True,
            "network_attempted": False,
            "stats_parser_executed": False,
            "predictions_created": False,
            "odds_created": False,
            "betting_signals_created": False,
            "canonical_unlock": False,
            "execution_path_created": False,
            "manifold_path_created": False,
        },
    }
    report_path = output_dir / "normalization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a BBR browser schedule snapshot to explicit local CSV.")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--run-id", help="Stable output folder name for evidence. Defaults to current UTC timestamp.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", type=Path, help="Exact output directory. Overrides output-root/run-id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = _utc_now()
    output_dir = args.output_dir or Path(args.output_root) / (args.run_id or _run_id(now))
    report = normalize_snapshot(args.snapshot, output_dir)
    print(json.dumps({"rows_written": report["rows_written"], "output_csv": report["output_csv"], "report": str(output_dir / "normalization_report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
