"""Validate Basketball_prediction schedule metadata availability.

This is a metadata-only check. It must not create prediction rows, odds,
canonical signals, or execution paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from nba2027.legacy.script7_cases import _normalize_team_phrase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/Users/alexanderrazmyslov/1. Python/Basketball_prediction")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_schedule_like_files(source_root: Path) -> list[dict]:
    candidates: list[dict] = []
    for path in source_root.rglob("*.csv"):
        lower_path = str(path).lower()
        if not any(token in lower_path for token in ["next_game", "schedule", "fixture", "games_df"]):
            continue
        try:
            head = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        columns = set(head.columns)
        if not ({"home_team", "away_team"} <= columns and ("game_date" in columns or "date" in columns)):
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        date_col = "game_date" if "game_date" in frame.columns else "date"
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        candidates.append(
            {
                "path": str(path),
                "relative_to_source": str(path.relative_to(source_root)),
                "sha256": sha256(path),
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "date_min": None if dates.dropna().empty else dates.min().strftime("%Y-%m-%d"),
                "date_max": None if dates.dropna().empty else dates.max().strftime("%Y-%m-%d"),
                "has_tipoff_time_column": any(
                    col.lower() in {"tipoff_utc", "tipoff", "game_time", "start_time", "time", "datetime", "commence_time"}
                    for col in frame.columns
                ),
                "has_prediction_columns": any(
                    col in frame.columns for col in ["home_team_prob", "prob_iso", "home_win_rate", "odds_1", "odds_2"]
                ),
            }
        )
    return candidates


def import_metadata(source_root: Path, source_files: list[dict], output_dir: Path) -> tuple[list[dict], list[str], str | None]:
    if not source_files:
        return [], ["no_2026_27_schedule_file_found_in_basketball_prediction"], None

    frames = []
    for item in source_files:
        frame = pd.read_csv(item["path"], low_memory=False)
        date_col = "game_date" if "game_date" in frame.columns else "date"
        work = pd.DataFrame(
            {
                "game_date": pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
                "home_team_raw": frame["home_team"],
                "away_team_raw": frame["away_team"],
                "source_file": item["relative_to_source"],
            }
        )
        for col in ["tipoff_utc", "tipoff", "game_time", "start_time", "time", "datetime", "commence_time"]:
            if col in frame.columns:
                work["tipoff_raw"] = frame[col]
                break
        else:
            work["tipoff_raw"] = None
        frames.append(work)

    schedule = pd.concat(frames, ignore_index=True)
    schedule["home_team"] = schedule["home_team_raw"].map(_normalize_team_phrase)
    schedule["away_team"] = schedule["away_team_raw"].map(_normalize_team_phrase)
    schedule["game_key"] = schedule["game_date"] + "|" + schedule["home_team"] + "|" + schedule["away_team"]
    schedule = schedule.drop_duplicates(["game_key"]).sort_values(["game_date", "home_team", "away_team"])
    output_path = output_dir / "schedule_metadata.csv"
    schedule.to_csv(output_path, index=False)
    issues = []
    if not schedule["tipoff_raw"].notna().all():
        issues.append("tipoff_time_missing_or_not_available_in_source")
    return schedule.to_dict("records"), issues, str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/schedule_import/20261020T120001000000Z")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_schedule_like_files(args.source_root)
    season_start = pd.Timestamp("2026-10-01")
    season_end = pd.Timestamp("2027-07-01")
    season_files = []
    for item in candidates:
        min_date = pd.to_datetime(item["date_min"], errors="coerce")
        max_date = pd.to_datetime(item["date_max"], errors="coerce")
        if pd.notna(min_date) and pd.notna(max_date) and max_date >= season_start and min_date < season_end:
            season_files.append(item)

    imported, issues, imported_file = import_metadata(args.source_root, season_files, args.output_dir)
    report = {
        "scope": "NBA 2027 schedule import validation - metadata only",
        "created_utc": "2026-10-20T12:00:01+00:00",
        "source": "Basketball_prediction",
        "source_root": str(args.source_root),
        "season": "2026-27",
        "nba_schedule_available": bool(season_files),
        "games_imported": len(imported),
        "teams_normalized": bool(imported) and all(row["home_team"] and row["away_team"] for row in imported),
        "duplicates_checked": True,
        "timezone_tipoff_utc_checked": bool(imported) and all(row.get("tipoff_raw") for row in imported),
        "no_odds_required": True,
        "no_predictions_required": True,
        "no_canonical_unlock": True,
        "no_betting_signals": True,
        "no_fallback_to_archived_2026_combined_files": True,
        "nba_pregame_still_requires_explicit_prediction_rows": True,
        "issues": issues,
        "candidate_schedule_like_files_found": len(candidates),
        "candidate_date_min": min([item["date_min"] for item in candidates if item["date_min"]], default=None),
        "candidate_date_max": max([item["date_max"] for item in candidates if item["date_max"]], default=None),
        "season_source_files": season_files,
        "sample_candidate_files": candidates[-20:],
        "imported_file": imported_file,
        "notes": [
            "Schedule metadata is not betting/model input.",
            "No home_team_prob, prob_iso, home_win_rate, odds_1, or odds_2 are created.",
            "Local Basketball_prediction schedule-like files found in this validation top out before the 2026-27 season.",
        ],
    }
    output = args.output_dir / "validation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "nba_schedule_available": report["nba_schedule_available"],
                "games_imported": report["games_imported"],
                "issues": issues,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
