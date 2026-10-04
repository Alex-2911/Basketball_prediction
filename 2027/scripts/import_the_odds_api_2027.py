#!/usr/bin/env python3
"""Guarded 2027 odds import from The Odds API.

This adapts the archived fetch_odds helper for the 2027 paper shell.  It can
fetch NBA head-to-head moneyline odds only when an API key is explicitly
provided through --api-key or ODDS_API_KEY.  It writes evidence only and never
creates predictions, betting signals, canonical unlocks, or execution paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "the_odds_api_import"
DEFAULT_GAMES = (
    ROOT
    / "outputs"
    / "script4_betting_statistics"
    / "20261020T120009000000Z"
    / "betting_statistics_evidence.csv"
)
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
LEGACY_HARDCODED_KEY_SOURCE = Path("/Users/alexanderrazmyslov/1. Python/1. NBA Script/2026/_3. 25102025_lightgbm_extracted.py")

FULL_TO_ABBREV = {
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
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}
# API/bookmaker aliases seen in older files.
FULL_TO_ABBREV.update({"Phoenix Suns": "PHO", "Brooklyn Nets": "BRK"})
API_ABBREV_TO_LOCAL = {"PHX": "PHO", "CHA": "CHO", "BKN": "BRK"}
LOCAL_TO_API_ABBREV = {"PHO": "PHX", "CHO": "CHA", "BRK": "BKN"}


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


def american_to_decimal(value: object) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price == 0:
        return None
    if price > 0:
        return round(1.0 + price / 100.0, 6)
    return round(1.0 + 100.0 / abs(price), 6)


def canonical_abbrev(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text in FULL_TO_ABBREV:
        return FULL_TO_ABBREV[text]
    upper = text.upper()
    return API_ABBREV_TO_LOCAL.get(upper, upper)


def api_abbrev(value: object) -> str | None:
    local = canonical_abbrev(value)
    if local is None:
        return None
    return LOCAL_TO_API_ABBREV.get(local, local)



def read_legacy_hardcoded_key(path: Path = LEGACY_HARDCODED_KEY_SOURCE) -> str | None:
    """Resolve the local legacy The Odds API key without printing or storing it.

    The 2027 repo must not commit the raw secret.  This supports the local daily
    workflow by reading the existing hard-coded 2026 source at runtime.
    """
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    match = re.search(r"API_KEY\s*=\s*[\"']([^\"']+)[\"']", text)
    if not match:
        return None
    value = match.group(1).strip()
    if not value or value.lower() in {"none", "<redacted>"}:
        return None
    return value


def resolve_api_key(args: argparse.Namespace) -> tuple[str | None, str]:
    explicit = args.api_key
    if explicit:
        return explicit, "cli"
    env_value = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    if env_value:
        return env_value, "environment"
    if getattr(args, "allow_legacy_hardcoded_key", True):
        legacy = read_legacy_hardcoded_key()
        if legacy:
            return legacy, "legacy_2026_hardcoded_source"
    return None, "missing"

def fetch_api_json(api_key: str) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"})
    request = urllib.request.Request(f"{ODDS_URL}?{params}", headers={"User-Agent": "nba2027-paper-shell/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def load_games(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, ["MISSING_GAMES_INPUT"]
    frame = _read_table(path)
    required = {"game_date", "home_team", "away_team"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return None, [f"GAMES_INPUT_MISSING_{col}" for col in missing]
    work = frame.copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work["home_team"] = work["home_team"].map(canonical_abbrev)
    work["away_team"] = work["away_team"].map(canonical_abbrev)
    work = work.dropna(subset=["game_date", "home_team", "away_team"])
    work = work.drop_duplicates(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    if work.empty:
        return None, ["EMPTY_OR_INVALID_GAMES_INPUT"]
    return work[["game_date", "home_team", "away_team"]], []


def fetch_odds(games_df: pd.DataFrame, api_key: str, preferred: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = fetch_api_json(api_key)

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for event in data:
        home = canonical_abbrev(event.get("home_team"))
        away = canonical_abbrev(event.get("away_team"))
        bookmakers = event.get("bookmakers") or []
        if not home or not away or not bookmakers:
            continue

        bookmaker = None
        if preferred:
            for key in preferred:
                bookmaker = next((bm for bm in bookmakers if bm.get("key") == key), None)
                if bookmaker:
                    break
        if bookmaker is None:
            bookmaker = bookmakers[0]

        market = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
        if not market:
            continue

        prices: dict[str, Any] = {}
        for outcome in market.get("outcomes", []):
            abbr = canonical_abbrev(outcome.get("name"))
            if abbr:
                prices[abbr] = outcome.get("price")

        lookup[(home, away)] = {
            "home_odds_american": prices.get(home),
            "away_odds_american": prices.get(away),
            "bookmaker_key": bookmaker.get("key"),
            "bookmaker_title": bookmaker.get("title"),
            "api_commence_time": event.get("commence_time"),
        }

    rows = []
    for _, game in games_df.iterrows():
        home, away = game.home_team, game.away_team
        found = lookup.get((home, away), {})
        home_american = found.get("home_odds_american")
        away_american = found.get("away_odds_american")
        rows.append(
            {
                "game_date": game.game_date,
                "home_team": home,
                "away_team": away,
                "odds_1": american_to_decimal(home_american),
                "odds_2": american_to_decimal(away_american),
                "home_odds_american": home_american,
                "away_odds_american": away_american,
                "bookmaker_key": found.get("bookmaker_key"),
                "bookmaker_title": found.get("bookmaker_title"),
                "api_commence_time": found.get("api_commence_time"),
                "odds_status": "OK" if home_american is not None and away_american is not None else "NO_ODDS_FOUND",
            }
        )

    return pd.DataFrame(rows), {"api_events_seen": len(data), "api_events_matched": len(lookup)}


def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    input_path = Path(args.input)
    games, issues = load_games(input_path)
    api_key, api_key_source = resolve_api_key(args)
    preferred = [x.strip() for x in str(args.preferred_bookmakers or "").split(",") if x.strip()]

    rows = 0
    output_csv = None
    odds_ok_rows = 0
    odds_missing_rows = 0
    network_attempted = False
    api_meta: dict[str, Any] = {"api_events_seen": 0, "api_events_matched": 0}

    if not issues and not api_key:
        issues = ["NO_VERIFIED_ODDS_API_KEY"]

    if not issues and games is not None:
        try:
            network_attempted = True
            odds, api_meta = fetch_odds(games, api_key=str(api_key), preferred=preferred or None)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "the_odds_api_evidence.csv"
            odds.to_csv(output_path, index=False)
            rows = int(len(odds))
            output_csv = str(output_path)
            odds_ok_rows = int((odds["odds_status"] == "OK").sum())
            odds_missing_rows = int((odds["odds_status"] != "OK").sum())
            if odds_ok_rows == 0:
                issues = ["NO_MATCHED_ODDS_FOR_GAMES"]
        except urllib.error.HTTPError as exc:
            issues = [f"ODDS_API_HTTP_ERROR_{exc.code}"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            issues = ["ODDS_API_REQUEST_FAILED"]

    status = "READY" if not issues or rows > 0 else "SKIPPED"
    reason_code = "ODDS_EVIDENCE_READY" if rows > 0 else (issues[0] if issues else "ODDS_EVIDENCE_READY")

    return {
        "script": "import_the_odds_api_2027",
        "legacy_source": "Script 3 fetch_odds / The Odds API h2h",
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "status": status,
        "reason_code": reason_code,
        "issues": issues,
        "inputs": {
            "games": str(input_path),
            "games_exists": input_path.exists(),
            "games_sha256": _sha256(input_path),
            "api_key_supplied": bool(api_key),
            "api_key_source": api_key_source,
            "preferred_bookmakers": preferred,
        },
        "results": {
            "rows": rows,
            "output_csv": output_csv,
            "odds_ok_rows": odds_ok_rows,
            "odds_missing_rows": odds_missing_rows,
            **api_meta,
        },
        "safety": {
            "odds_import_only": True,
            "network_attempted": network_attempted,
            "predictions_created": False,
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
    parser = argparse.ArgumentParser(description="Guarded The Odds API importer for 2027 NBA paper shell.")
    parser.add_argument("--input", default=str(DEFAULT_GAMES), help="Current games CSV or Parquet.")
    parser.add_argument("--api-key", help="The Odds API key. Prefer ODDS_API_KEY env var so the key is not committed.")
    parser.add_argument("--no-legacy-hardcoded-key", dest="allow_legacy_hardcoded_key", action="store_false", help="Do not read the local legacy 2026 hard-coded key.")
    parser.set_defaults(allow_legacy_hardcoded_key=True)
    parser.add_argument("--preferred-bookmakers", default="draftkings,fanduel,betmgm", help="Comma-separated bookmaker keys.")
    parser.add_argument("--run-id", help="Stable output folder name for evidence. Defaults to current UTC timestamp.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = _utc_now()
    run_id = args.run_id or _run_id(now)
    args.output_dir = str(Path(args.output_root) / run_id)
    report = build_report(args, now)
    report_path = write_report(report, Path(args.output_root), run_id)
    print(json.dumps({"status": report["status"], "reason_code": report["reason_code"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
