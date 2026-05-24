#!/usr/bin/env python3
"""Create independent halftime live-odds capture schedules and launchd jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import plistlib
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


FULL_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}
API_TO_PROJECT_ABBR = {"PHX": "PHO", "CHA": "CHO"}

CAPTURE_TYPES = {
    "CANONICAL_MODEL_SIGNAL",
    "LIVE_WATCH_ONLY",
    "RAW_MODEL_MARKET_GAP_HOME_DOG",
    "LOW_PRICE_NEGATIVE_EV",
}


def project_abbr(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    abbr = FULL_TO_ABBR.get(text, text)
    return API_TO_PROJECT_ABBR.get(abbr, abbr)


@dataclass
class ScheduledGame:
    run_date: str
    game_date: str
    game_key: str
    home_team: str
    away_team: str
    commence_time_utc: str | None
    primary_capture_utc: str | None
    early_capture_utc: str | None
    late_capture_utc: str | None
    source: str
    stage2_candidate_type: str
    blocked_by: str
    live_watch_capture_candidate: bool
    scheduled_status: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().replace(",", ".")
        if not text or text.lower() in {"nan", "none"}:
            return None
        return float(text)
    except ValueError:
        return None


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def source_root() -> Path:
    return Path(os.environ.get("SOURCE_ROOT", Path.cwd())).resolve()


def lightgbm_dir(root: Path) -> Path:
    if os.environ.get("LGBM_DIR"):
        return Path(os.environ["LGBM_DIR"]).resolve()
    if (root / "2026" / "output" / "LightGBM").exists() or (root / "2026").exists():
        return (root / "2026" / "output" / "LightGBM").resolve()
    if (root / "output" / "LightGBM").exists():
        return (root / "output" / "LightGBM").resolve()
    return (root / "LightGBM").resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def candidate_row(row: dict[str, str]) -> bool:
    stage = (row.get("stage2_candidate_type") or "").strip()
    canonical = (row.get("canonical_signal") or "").strip()
    blocked_by = (row.get("blocked_by") or "").strip()
    rules_passed = to_float(row.get("rules_passed")) or 0.0
    prob = to_float(row.get("prob_used")) or to_float(row.get("home_team_prob")) or 0.0
    hwr = to_float(row.get("home_win_rate")) or 0.0

    if truthy(row.get("live_watch_capture_candidate")):
        return True
    if canonical == "CANONICAL_MODEL_SIGNAL":
        return True
    if stage in CAPTURE_TYPES:
        return True
    if stage == "NO_VALUE_SKIP" and (rules_passed >= 3 or prob >= 0.48 or hwr >= 0.45):
        return True
    if rules_passed >= 3 and ("EV" in blocked_by or "PASS" in blocked_by):
        return True
    return False


def latest_by_game(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    chosen: dict[str, dict[str, str]] = {}
    for row in rows:
        game_date = (row.get("game_date") or row.get("date") or "").strip()
        home = (row.get("home_team") or "").strip()
        away = (row.get("away_team") or "").strip()
        if not game_date or not home or not away:
            continue
        key = f"{game_date}__{home}__{away}"
        chosen[key] = row
    return list(chosen.values())


def http_json(url: str, params: dict[str, str], timeout: int = 12) -> Any:
    full_url = f"{url}?{urlencode(params)}"
    with urlopen(full_url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_odds_api_events(api_key: str | None) -> dict[tuple[str, str], datetime]:
    if not api_key:
        return {}
    try:
        data = http_json(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            {"apiKey": api_key},
        )
    except Exception as exc:
        print(f"[WARN] The Odds API events lookup failed: {exc}", file=sys.stderr)
        return {}

    out: dict[tuple[str, str], datetime] = {}
    for ev in data or []:
        home = project_abbr(ev.get("home_team"))
        away = project_abbr(ev.get("away_team"))
        commence = parse_dt(ev.get("commence_time"))
        if home and away and commence:
            out[(home, away)] = commence
    return out


def fetch_espn_schedule(game_dates: set[str]) -> dict[tuple[str, str, str], datetime]:
    out: dict[tuple[str, str, str], datetime] = {}
    for game_date in sorted(game_dates):
        try:
            data = http_json(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                {"dates": game_date.replace("-", "")},
            )
        except Exception as exc:
            print(f"[WARN] ESPN schedule lookup failed for {game_date}: {exc}", file=sys.stderr)
            continue
        for event in data.get("events", []) or []:
            commence = parse_dt(event.get("date"))
            comps = ((event.get("competitions") or [{}])[0].get("competitors") or [])
            home = away = None
            for comp in comps:
                abbr = project_abbr((comp.get("team") or {}).get("abbreviation"))
                if comp.get("homeAway") == "home":
                    home = abbr
                elif comp.get("homeAway") == "away":
                    away = abbr
            if home and away and commence:
                out[(game_date, home, away)] = commence
    return out


def resolve_commence(
    row: dict[str, str],
    odds_events: dict[tuple[str, str], datetime],
    espn_events: dict[tuple[str, str, str], datetime],
) -> tuple[datetime | None, str]:
    direct = parse_dt(row.get("commence_time_utc") or row.get("commence_time"))
    if direct:
        return direct, "watchlist"
    home = (row.get("home_team") or "").strip()
    away = (row.get("away_team") or "").strip()
    game_date = (row.get("game_date") or row.get("date") or "").strip()
    if (home, away) in odds_events:
        return odds_events[(home, away)], "the_odds_api_events"
    if (game_date, home, away) in espn_events:
        return espn_events[(game_date, home, away)], "espn_scoreboard"
    return None, "missing"


def build_schedule(root: Path, lgbm: Path, run_date: str, *, include_past: bool = False) -> list[ScheduledGame]:
    rows = latest_by_game(read_rows(lgbm / "script11_watchlist_history_latest.csv"))
    rows = [r for r in rows if candidate_row(r)]
    if not include_past:
        rows = [
            r
            for r in rows
            if (r.get("game_date") or r.get("date") or "").strip() >= run_date
        ]
    rows_needing_lookup = [
        r for r in rows if parse_dt(r.get("commence_time_utc") or r.get("commence_time")) is None
    ]
    game_dates = {
        (r.get("game_date") or r.get("date") or "").strip()
        for r in rows_needing_lookup
        if r.get("game_date") or r.get("date")
    }
    api_key = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    odds_events = fetch_odds_api_events(api_key) if rows_needing_lookup else {}
    espn_events = fetch_espn_schedule(game_dates) if rows_needing_lookup else {}
    now = utc_now()
    scheduled: list[ScheduledGame] = []

    for row in rows:
        game_date = (row.get("game_date") or row.get("date") or "").strip()
        home = (row.get("home_team") or "").strip()
        away = (row.get("away_team") or "").strip()
        game_key = clean_key(row.get("game_key") or f"{game_date}__{home}__{away}")
        commence, source = resolve_commence(row, odds_events, espn_events)
        primary = commence + timedelta(minutes=75) if commence else None
        early = commence + timedelta(minutes=70) if commence else None
        late = commence + timedelta(minutes=80) if commence else None
        if not commence:
            status = "missing_commence_time"
        elif late and late < now:
            status = "past_due"
        else:
            status = "scheduled"
        scheduled.append(
            ScheduledGame(
                run_date=run_date,
                game_date=game_date,
                game_key=game_key,
                home_team=home,
                away_team=away,
                commence_time_utc=iso_z(commence),
                primary_capture_utc=iso_z(primary),
                early_capture_utc=iso_z(early),
                late_capture_utc=iso_z(late),
                source=source,
                stage2_candidate_type=(row.get("stage2_candidate_type") or "").strip(),
                blocked_by=(row.get("blocked_by") or "").strip(),
                live_watch_capture_candidate=True,
                scheduled_status=status,
            )
        )
    return scheduled


def write_schedule(lgbm: Path, run_date: str, scheduled: list[ScheduledGame]) -> None:
    payload = {
        "run_date": run_date,
        "created_utc": iso_z(utc_now()),
        "count": len(scheduled),
        "games": [asdict(g) for g in scheduled],
    }
    for path in (
        lgbm / "halftime_capture_schedule_latest.json",
        lgbm / f"halftime_capture_schedule_{run_date}.json",
    ):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


def launchd_calendar(dt_utc: datetime) -> dict[str, int]:
    local_dt = dt_utc.astimezone()
    return {
        "Month": local_dt.month,
        "Day": local_dt.day,
        "Hour": local_dt.hour,
        "Minute": local_dt.minute,
    }


def create_launchd_jobs(root: Path, lgbm: Path, scheduled: list[ScheduledGame], *, load: bool) -> list[Path]:
    agents = Path.home() / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    logs = root / "2026" / "logs" if (root / "2026").exists() else root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    uid = os.getuid()
    capture_script = capture_script_path(root)

    for game in scheduled:
        capture_dt = parse_dt(game.primary_capture_utc)
        if not capture_dt or game.scheduled_status != "scheduled":
            continue
        stamp = capture_dt.strftime("%Y%m%dT%H%M%SZ")
        label = f"com.alexander.nba.halftime.{clean_key(game.game_key)}.{stamp}"
        out_log = logs / f"halftime_live_odds_{clean_key(game.game_key)}_{stamp}.out"
        err_log = logs / f"halftime_live_odds_{clean_key(game.game_key)}_{stamp}.err"
        plist = {
            "Label": label,
            "ProgramArguments": [
                sys.executable,
                str(capture_script),
                "--run-date",
                game.run_date,
                "--game-date",
                game.game_date,
                "--game-key",
                game.game_key,
                "--home-team",
                game.home_team,
                "--away-team",
                game.away_team,
                "--mode",
                "once",
                "--markets",
                "h2h,spreads,totals",
                "--snapshot-label",
                "halftime",
            ],
            "StartCalendarInterval": launchd_calendar(capture_dt),
            "StandardOutPath": str(out_log),
            "StandardErrorPath": str(err_log),
            "WorkingDirectory": str(root),
            "EnvironmentVariables": {
                "SOURCE_ROOT": str(root),
                "LGBM_DIR": str(lgbm),
                "PYTHONPATH": str(root),
                "ODDS_API_KEY": os.environ.get("ODDS_API_KEY", ""),
                "THE_ODDS_API_KEY": os.environ.get("THE_ODDS_API_KEY", ""),
                "ODDS_BOOKMAKER_KEY": os.environ.get("ODDS_BOOKMAKER_KEY", ""),
            },
        }
        path = agents / f"{label}.plist"
        with path.open("wb") as f:
            plistlib.dump(plist, f)
        created.append(path)
        print(f"Wrote launchd job {path}")
        if load:
            subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(path)], check=False)
    return created


def capture_script_path(root: Path) -> Path:
    return (
        root / "2026" / "scripts" / "capture_halftime_live_odds.py"
        if (root / "2026" / "scripts").exists()
        else root / "scripts" / "capture_halftime_live_odds.py"
    )


def capture_due_games(
    root: Path,
    lgbm: Path,
    scheduled: list[ScheduledGame],
    *,
    due_window_minutes: int,
    markets: str,
) -> list[str]:
    now = utc_now()
    capture_script = capture_script_path(root)
    captured: list[str] = []
    for game in scheduled:
        early = parse_dt(game.early_capture_utc)
        late = parse_dt(game.late_capture_utc)
        primary = parse_dt(game.primary_capture_utc)
        if not primary:
            continue
        due_start = early or (primary - timedelta(minutes=due_window_minutes))
        due_end = late or (primary + timedelta(minutes=due_window_minutes))
        if not (due_start <= now <= due_end):
            continue
        cmd = [
            sys.executable,
            str(capture_script),
            "--run-date",
            game.run_date,
            "--game-date",
            game.game_date,
            "--game-key",
            game.game_key,
            "--home-team",
            game.home_team,
            "--away-team",
            game.away_team,
            "--mode",
            "once",
            "--markets",
            markets,
            "--snapshot-label",
            "halftime",
            "--source-root",
            str(root),
            "--lightgbm-dir",
            str(lgbm),
        ]
        print(f"Capturing due halftime odds: {game.game_key}")
        subprocess.run(cmd, check=True)
        captured.append(game.game_key)
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description="Schedule independent halftime live-odds captures.")
    parser.add_argument("--run-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--lightgbm-dir", default=None)
    parser.add_argument("--install-launchd", action="store_true")
    parser.add_argument("--load-launchd", action="store_true", help="Bootstrap generated LaunchAgents immediately.")
    parser.add_argument("--include-past", action="store_true", help="Include historical watchlist rows before run_date.")
    parser.add_argument("--capture-due", action="store_true", help="Immediately run capture for games inside halftime window.")
    parser.add_argument("--due-window-minutes", type=int, default=8)
    parser.add_argument("--markets", default="h2h,spreads,totals")
    args = parser.parse_args()

    root = Path(args.source_root).resolve() if args.source_root else source_root()
    lgbm = Path(args.lightgbm_dir).resolve() if args.lightgbm_dir else lightgbm_dir(root)
    scheduled = build_schedule(root, lgbm, args.run_date, include_past=args.include_past)
    write_schedule(lgbm, args.run_date, scheduled)
    if args.install_launchd:
        create_launchd_jobs(root, lgbm, scheduled, load=args.load_launchd)
    if args.capture_due:
        captured = capture_due_games(
            root,
            lgbm,
            scheduled,
            due_window_minutes=args.due_window_minutes,
            markets=args.markets,
        )
        print(f"Captured due games: {len(captured)}")
    print(f"Scheduled candidates: {len(scheduled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
