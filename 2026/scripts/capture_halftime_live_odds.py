from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nba_utils_2026 import normalize_team_code

ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
WATCHLIST_PATH = "2026/output/LightGBM/script11_watchlist_history_latest.csv"
SNAPSHOT_PATH = "2026/output/LightGBM/live_odds_snapshots.csv"
HALFTIME_LEAD_MIN = 60
HALFTIME_LAG_MIN = 90

FULL_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def _read_live_watch_games(repo_root: Path, run_date: str) -> pd.DataFrame:
    watch_path = repo_root / WATCHLIST_PATH
    if not watch_path.exists():
        return pd.DataFrame(columns=["game_date", "home_team", "away_team"])
    df = pd.read_csv(watch_path)
    needed = {"game_date", "home_team", "away_team", "stage2_candidate_type", "run_date"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=["game_date", "home_team", "away_team"])
    mask = (
        df["run_date"].astype(str).eq(run_date)
        & df["stage2_candidate_type"].astype(str).isin(["LIVE_WATCH_ONLY", "RAW_MODEL_MARKET_GAP_HOME_DOG", "DATA_INCOMPLETE"])
    )
    out = df.loc[mask, ["game_date", "home_team", "away_team"]].copy()
    out["home_team"] = out["home_team"].apply(normalize_team_code)
    out["away_team"] = out["away_team"].apply(normalize_team_code)
    return out.drop_duplicates()


def _fetch_events(api_key: str) -> list[dict]:
    resp = requests.get(
        ODDS_URL,
        params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _snapshot_rows(events: list[dict], watch_games: pd.DataFrame, now_utc: datetime) -> pd.DataFrame:
    wanted = {(r.home_team, r.away_team) for r in watch_games.itertuples(index=False)}
    rows = []
    for ev in events:
        home = normalize_team_code(FULL_TO_ABBREV.get(ev.get("home_team")))
        away = normalize_team_code(FULL_TO_ABBREV.get(ev.get("away_team")))
        if (home, away) not in wanted:
            continue
        commence = pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(commence):
            continue
        halftime_start = commence + pd.Timedelta(minutes=HALFTIME_LEAD_MIN)
        halftime_end = commence + pd.Timedelta(minutes=HALFTIME_LAG_MIN)
        in_window = halftime_start <= now_utc <= halftime_end
        home_odds = None
        away_odds = None
        if in_window:
            books = ev.get("bookmakers") or []
            if books:
                market = next((m for m in (books[0].get("markets") or []) if m.get("key") == "h2h"), None)
                if market:
                    prices = {}
                    for out in market.get("outcomes", []):
                        team = normalize_team_code(FULL_TO_ABBREV.get(out.get("name")))
                        if team:
                            prices[team] = out.get("price")
                    home_odds = prices.get(home)
                    away_odds = prices.get(away)

        rows.append(
            {
                "game_date": now_utc.strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "commence_time_utc": commence.isoformat(),
                "halftime_time_utc": (commence + pd.Timedelta(minutes=75)).isoformat(),
                "is_halftime_window": bool(in_window),
                "halftime_home_odds": home_odds,
                "halftime_away_odds": away_odds,
                "live_odds_fetched": bool(in_window and home_odds is not None and away_odds is not None),
                "snapshot_utc": now_utc.isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _next_halftime_start(events: list[dict], watch_games: pd.DataFrame, now_utc: datetime) -> datetime | None:
    wanted = {(r.home_team, r.away_team) for r in watch_games.itertuples(index=False)}
    starts: list[datetime] = []
    for ev in events:
        home = normalize_team_code(FULL_TO_ABBREV.get(ev.get("home_team")))
        away = normalize_team_code(FULL_TO_ABBREV.get(ev.get("away_team")))
        if (home, away) not in wanted:
            continue
        commence = pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(commence):
            continue
        halftime_start = commence + pd.Timedelta(minutes=HALFTIME_LEAD_MIN)
        if halftime_start.to_pydatetime() > now_utc:
            starts.append(halftime_start.to_pydatetime())
    if not starts:
        return None
    return min(starts)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is required.")
    now_utc = datetime.now(timezone.utc)
    run_date = now_utc.strftime("%Y-%m-%d")

    watch_games = _read_live_watch_games(repo_root, run_date)
    if watch_games.empty:
        print(json.dumps({"run_date": run_date, "watch_games": 0, "snapshots_written": 0}))
        return

    events = _fetch_events(api_key)

    next_start = _next_halftime_start(events, watch_games, now_utc)
    if next_start is not None:
        wait_seconds = max(0, int((next_start - now_utc).total_seconds()))
        if wait_seconds > 0:
            print(json.dumps({
                "run_date": run_date,
                "watch_games": int(len(watch_games)),
                "scheduled_halftime_start_utc": next_start.isoformat(),
                "wait_seconds": wait_seconds,
            }))
            time.sleep(wait_seconds)
            now_utc = datetime.now(timezone.utc)
            events = _fetch_events(api_key)

    snap = _snapshot_rows(events, watch_games, now_utc)
    if snap.empty:
        print(json.dumps({"run_date": run_date, "watch_games": int(len(watch_games)), "snapshots_written": 0}))
        return

    out_path = repo_root / SNAPSHOT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        prior = pd.read_csv(out_path)
        snap = pd.concat([prior, snap], ignore_index=True, sort=False)
    snap = snap.drop_duplicates(subset=["game_date", "home_team", "away_team", "snapshot_utc"], keep="last")
    snap.to_csv(out_path, index=False)

    print(json.dumps({
        "run_date": run_date,
        "watch_games": int(len(watch_games)),
        "snapshots_written": int(len(snap)),
        "path": str(out_path),
    }))


if __name__ == "__main__":
    main()
