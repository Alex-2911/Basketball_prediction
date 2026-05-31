#!/usr/bin/env python3
"""One-shot halftime live odds and score capture."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
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

TEAM_ALIASES = {
    "SA": "SAS",
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def norm_team(value: Any) -> str:
    text = str(value or "").strip()
    return TEAM_ALIASES.get(text, text)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def source_root() -> Path:
    return Path(os.environ.get("SOURCE_ROOT", Path.cwd())).resolve()


def lightgbm_dir(root: Path) -> Path:
    return Path(os.environ.get("LGBM_DIR", root / "LightGBM")).resolve()


def load_local_env(root: Path) -> None:
    """Load local API keys for standalone launchd runs."""
    for env_path in (root / ".env.local", root / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    if not (os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")):
        legacy = discover_legacy_odds_api_key(root)
        if legacy:
            os.environ["ODDS_API_KEY"] = legacy


def discover_legacy_odds_api_key(root: Path) -> str | None:
    """Recover the existing local Odds API key from legacy Script 3 artifacts."""
    candidates = [
        root / "_3. 25102025_lightgbm_extracted.py",
        root / "_3. 25102025_lightgbm.ipynb",
    ]
    pattern = re.compile(r'API_KEY\s*=\s*["\']([A-Za-z0-9_-]{20,})["\']')
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def http_json(url: str, params: dict[str, str], timeout: int = 12) -> Any:
    full_url = f"{url}?{urlencode(params)}"
    with urlopen(full_url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def decimal_price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price >= 100:
        return round(1 + price / 100, 4)
    if price <= -100:
        return round(1 + 100 / abs(price), 4)
    if price <= 0:
        return None
    return price


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).replace(",", ".")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        return float(match.group(0))
    except ValueError:
        return None


def fetch_espn_odds(
    *,
    game_date: str,
    home_team: str,
    away_team: str,
    fixture: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if fixture and "espn_scoreboard" in fixture:
        data = fixture["espn_scoreboard"]
    else:
        try:
            data = http_json(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                {"dates": game_date.replace("-", "")},
            )
        except Exception as exc:
            print(f"[WARN] ESPN odds fallback failed: {exc}", file=sys.stderr)
            return []

    rows: list[dict[str, Any]] = []
    for event in data.get("events", []) or []:
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []
        teams: dict[str, dict[str, Any]] = {}
        for c in competitors:
            abbr = norm_team((c.get("team") or {}).get("abbreviation"))
            teams[abbr] = c
        if home_team not in teams or away_team not in teams:
            continue
        for odds in comp.get("odds", []) or []:
            provider = odds.get("provider") or {}
            base = {
                "bookmaker_key": f"espn_{str(provider.get('name') or provider.get('displayName') or 'odds').lower().replace(' ', '_')}",
                "bookmaker_title": provider.get("displayName") or provider.get("name") or "ESPN odds",
            }
            moneyline = odds.get("moneyline") or {}
            home_ml = decimal_price((((moneyline.get("home") or {}).get("close") or {}).get("odds")))
            away_ml = decimal_price((((moneyline.get("away") or {}).get("close") or {}).get("odds")))
            if home_ml is not None or away_ml is not None:
                rows.append({**base, "market_type": "h2h", "home_ml": home_ml, "away_ml": away_ml})

            spread = odds.get("pointSpread") or {}
            home_spread = ((spread.get("home") or {}).get("close") or {})
            away_spread = ((spread.get("away") or {}).get("close") or {})
            home_line = parse_float(home_spread.get("line"))
            away_line = parse_float(away_spread.get("line"))
            if home_line is not None or away_line is not None:
                rows.append(
                    {
                        **base,
                        "market_type": "spreads",
                        "home_spread_line": home_line,
                        "home_spread_price": decimal_price(home_spread.get("odds")),
                        "away_spread_line": away_line,
                        "away_spread_price": decimal_price(away_spread.get("odds")),
                    }
                )

            total = odds.get("total") or {}
            over = ((total.get("over") or {}).get("close") or {})
            under = ((total.get("under") or {}).get("close") or {})
            total_line = parse_float(over.get("line")) or parse_float(under.get("line"))
            if total_line is not None:
                rows.append(
                    {
                        **base,
                        "market_type": "totals",
                        "total_line": total_line,
                        "over_price": decimal_price(over.get("odds")),
                        "under_price": decimal_price(under.get("odds")),
                    }
                )
        return rows
    return rows


def read_watchlist_row(lgbm: Path, game_key: str) -> dict[str, str]:
    path = lgbm / "script11_watchlist_history_latest.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in reversed(rows):
        if (row.get("game_key") or "").strip() == game_key:
            return row
    return {}


def load_fixture(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_odds(
    *,
    api_key: str | None,
    game_date: str,
    home_team: str,
    away_team: str,
    markets: str,
    bookmaker_key: str | None,
    fixture: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if fixture and "odds_events" in fixture:
        data = fixture["odds_events"]
    else:
        if not api_key:
            return fetch_espn_odds(
                game_date=game_date,
                home_team=home_team,
                away_team=away_team,
                fixture=fixture,
            )
        try:
            data = http_json(
                "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
                {
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": markets,
                    "oddsFormat": "decimal",
                },
            )
        except Exception as exc:
            print(f"[WARN] odds fetch failed: {exc}", file=sys.stderr)
            return []

    rows: list[dict[str, Any]] = []
    for event in data or []:
        home = norm_team(FULL_TO_ABBR.get(str(event.get("home_team", "")), event.get("home_team")))
        away = norm_team(FULL_TO_ABBR.get(str(event.get("away_team", "")), event.get("away_team")))
        if home != home_team or away != away_team:
            continue
        bookmakers = event.get("bookmakers") or []
        if bookmaker_key:
            bookmakers = [b for b in bookmakers if b.get("key") == bookmaker_key]
        for bm in bookmakers:
            base = {
                "bookmaker_key": bm.get("key"),
                "bookmaker_title": bm.get("title"),
            }
            market_map = {m.get("key"): m for m in bm.get("markets", []) or []}
            h2h = market_map.get("h2h")
            if h2h:
                row = dict(base, market_type="h2h")
                for out in h2h.get("outcomes", []) or []:
                    abbr = norm_team(FULL_TO_ABBR.get(str(out.get("name", "")), out.get("name")))
                    if abbr == home_team:
                        row["home_ml"] = decimal_price(out.get("price"))
                    elif abbr == away_team:
                        row["away_ml"] = decimal_price(out.get("price"))
                rows.append(row)

            spread = market_map.get("spreads")
            if spread:
                row = dict(base, market_type="spreads")
                for out in spread.get("outcomes", []) or []:
                    abbr = norm_team(FULL_TO_ABBR.get(str(out.get("name", "")), out.get("name")))
                    if abbr == home_team:
                        row["home_spread_line"] = parse_float(out.get("point"))
                        row["home_spread_price"] = decimal_price(out.get("price"))
                    elif abbr == away_team:
                        row["away_spread_line"] = parse_float(out.get("point"))
                        row["away_spread_price"] = decimal_price(out.get("price"))
                rows.append(row)

            total = market_map.get("totals")
            if total:
                row = dict(base, market_type="totals")
                for out in total.get("outcomes", []) or []:
                    name = str(out.get("name", "")).lower()
                    if name == "over":
                        row["total_line"] = parse_float(out.get("point"))
                        row["over_price"] = decimal_price(out.get("price"))
                    elif name == "under":
                        row["total_line"] = row.get("total_line") or parse_float(out.get("point"))
                        row["under_price"] = decimal_price(out.get("price"))
                rows.append(row)
    if not rows and not fixture:
        return fetch_espn_odds(
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            fixture=fixture,
        )
    return rows


def fetch_score(
    *,
    game_date: str,
    home_team: str,
    away_team: str,
    fixture: dict[str, Any] | None,
) -> dict[str, Any]:
    if fixture and "score" in fixture:
        return fixture["score"]
    try:
        data = http_json(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            {"dates": game_date.replace("-", "")},
        )
    except Exception as exc:
        print(f"[WARN] score fetch failed: {exc}", file=sys.stderr)
        return {"score_status": "missing", "score_source": "espn_scoreboard"}

    for event in data.get("events", []) or []:
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []
        parsed: dict[str, Any] = {
            "score_status": "missing",
            "score_source": "espn_scoreboard",
            "period_status": (event.get("status") or {}).get("type", {}).get("description"),
            "period_state": (event.get("status") or {}).get("type", {}).get("state"),
            "period_detail": (event.get("status") or {}).get("type", {}).get("detail"),
            "period": (event.get("status") or {}).get("period"),
            "game_clock": (event.get("status") or {}).get("displayClock"),
        }
        teams: dict[str, dict[str, Any]] = {}
        for c in competitors:
            abbr = norm_team((c.get("team") or {}).get("abbreviation"))
            teams[abbr] = c
        if home_team not in teams or away_team not in teams:
            continue
        home = teams[home_team]
        away = teams[away_team]
        parsed["final_home_score"] = parse_float(home.get("score"))
        parsed["final_away_score"] = parse_float(away.get("score"))
        home_lines = home.get("linescores") or []
        away_lines = away.get("linescores") or []
        if len(home_lines) >= 2 and len(away_lines) >= 2:
            home_ht = sum(parse_float(q.get("value")) or 0 for q in home_lines[:2])
            away_ht = sum(parse_float(q.get("value")) or 0 for q in away_lines[:2])
            period = parse_float(parsed.get("period")) or 0
            status_text = " ".join(
                str(parsed.get(k) or "").lower()
                for k in ("period_status", "period_state", "period_detail")
            )
            is_halftime_or_later = (
                "halftime" in status_text
                or "half" in status_text
                or period >= 3
                or parsed.get("period_state") == "post"
            )
            parsed.update(
                {
                    "halftime_score_home": int(home_ht),
                    "halftime_score_away": int(away_ht),
                    "halftime_margin_home": int(home_ht - away_ht),
                    "score_status": "fetched" if is_halftime_or_later else "waiting_for_halftime",
                }
            )
        return parsed
    return {"score_status": "missing", "score_source": "espn_scoreboard"}


def best_market(rows: list[dict[str, Any]], market_type: str, field: str) -> dict[str, Any]:
    candidates = [r for r in rows if r.get("market_type") == market_type and r.get(field) is not None]
    if not candidates:
        return {}
    return max(candidates, key=lambda r: float(r[field]))


def classify_live_decision(
    *,
    home_team: str,
    away_team: str,
    score: dict[str, Any],
    odds_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    h2h_best = best_market(odds_rows, "h2h", "away_ml")
    spread_best = best_market(odds_rows, "spreads", "away_spread_price")
    away_ml = h2h_best.get("away_ml")
    away_spread_line = spread_best.get("away_spread_line")
    away_spread_price = spread_best.get("away_spread_price")
    margin_home = score.get("halftime_margin_home")
    if score.get("score_status") == "waiting_for_halftime":
        return {
            "live_classification": "WAITING_FOR_HALFTIME",
            "candidate_side": "",
            "candidate_market": "",
            "stake_class": "none",
            "final_decision": "WAIT",
            "notes": "Score feed has not reached halftime yet; retry window remains active.",
        }
    if margin_home is None:
        return {
            "live_classification": "READ_ONLY_SCORE_MISSING",
            "candidate_side": "",
            "candidate_market": "",
            "stake_class": "none",
            "final_decision": "NO_ACTION",
            "notes": "Halftime score unavailable; odds captured without decision label.",
        }
    if not odds_rows:
        return {
            "live_classification": "READ_ONLY_ODDS_MISSING",
            "candidate_side": "",
            "candidate_market": "",
            "stake_class": "none",
            "final_decision": "NO_ACTION",
            "notes": "No exact live odds available.",
        }

    away_margin = -float(margin_home)
    if away_margin <= 0:
        return {
            "live_classification": "LIVE_WATCH_ONLY_NO_ACTION",
            "candidate_side": "",
            "candidate_market": "",
            "stake_class": "none",
            "final_decision": "NO_ACTION",
            "notes": f"{away_team} did not lead at halftime; away-underdog HT-lead setup inactive.",
        }

    threshold = None
    watch_label = "LIVE_WATCH_ONLY_NO_ACTION"
    if 1 <= away_margin <= 3:
        threshold = 2.40
        watch_label = "WATCH_ONLY_SMALL_AWAY_LEAD"
    elif 4 <= away_margin <= 7:
        threshold = 1.95
        watch_label = "WATCH_ONLY_MEDIUM_AWAY_LEAD"
    elif 8 <= away_margin <= 12:
        threshold = 1.60
        watch_label = "WATCH_ONLY_STRONG_AWAY_LEAD"
    elif away_margin >= 13:
        threshold = 1.30
        watch_label = "SHORT_PRICE_SKIP"

    if away_ml is not None and threshold is not None and float(away_ml) >= threshold:
        return {
            "live_classification": "AWAY_LIVE_ML_CANDIDATE",
            "candidate_side": away_team,
            "candidate_market": "away_ml",
            "stake_class": "small_experimental",
            "final_decision": "REVIEW_ONLY",
            "notes": f"{away_team} led by {away_margin:.0f}; away ML {away_ml} cleared threshold {threshold}.",
        }

    if (
        4 <= away_margin <= 12
        and away_spread_line is not None
        and away_spread_line < 0
        and abs(float(away_spread_line)) <= 6.5
        and away_spread_price is not None
        and float(away_spread_price) >= 1.75
    ):
        return {
            "live_classification": "AWAY_LEAD_SPREAD_CANDIDATE",
            "candidate_side": away_team,
            "candidate_market": "away_spread",
            "stake_class": "small_experimental",
            "final_decision": "REVIEW_ONLY",
            "notes": f"{away_team} ML was too short for bucket, but spread {away_spread_line} at {away_spread_price} is a review candidate.",
        }

    return {
        "live_classification": watch_label,
        "candidate_side": "",
        "candidate_market": "",
        "stake_class": "none",
        "final_decision": "NO_ACTION",
        "notes": f"{away_team} led by {away_margin:.0f}; away ML {away_ml} did not clear threshold {threshold}.",
    }


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latest_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def persist(
    *,
    lgbm: Path,
    run_date: str,
    game_date: str,
    game_key: str,
    home_team: str,
    away_team: str,
    snapshot_utc: str,
    snapshot_label: str,
    watchlist: dict[str, str],
    score: dict[str, Any],
    odds_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    base = {
        "run_date": run_date,
        "game_date": game_date,
        "game_key": game_key,
        "matchup": f"{away_team} @ {home_team}",
        "home_team": home_team,
        "away_team": away_team,
        "snapshot_utc": snapshot_utc,
        "snapshot_label": snapshot_label,
        "pregame_model_label": watchlist.get("canonical_signal") or "NO",
        "stage2_candidate_type": watchlist.get("stage2_candidate_type") or "",
        "blocked_by": watchlist.get("blocked_by") or "",
        "halftime_score_home": score.get("halftime_score_home"),
        "halftime_score_away": score.get("halftime_score_away"),
        "halftime_margin_home": score.get("halftime_margin_home"),
        "period_status": score.get("period_status"),
        "period_state": score.get("period_state"),
        "period_detail": score.get("period_detail"),
        "period": score.get("period"),
        "game_clock": score.get("game_clock"),
        "score_source": score.get("score_source"),
        "score_status": score.get("score_status"),
        "live_market_status": "fetched" if odds_rows else "missing",
        "live_odds_fetched": bool(odds_rows),
    }

    snapshot_fields = [
        "run_date",
        "game_date",
        "game_key",
        "home_team",
        "away_team",
        "snapshot_utc",
        "snapshot_label",
        "bookmaker_key",
        "bookmaker_title",
        "market_type",
        "home_ml",
        "away_ml",
        "home_spread_line",
        "home_spread_price",
        "away_spread_line",
        "away_spread_price",
        "total_line",
        "over_price",
        "under_price",
        "halftime_score_home",
        "halftime_score_away",
        "halftime_margin_home",
        "score_status",
    ]
    snapshot_rows = [{**base, **r} for r in odds_rows] or [{**base, "market_type": "none"}]
    append_csv(lgbm / "live_odds_snapshots.csv", snapshot_rows, snapshot_fields)
    append_csv(lgbm / f"live_odds_snapshots_{run_date}.csv", snapshot_rows, snapshot_fields)

    best_h2h_home = best_market(odds_rows, "h2h", "home_ml")
    best_h2h_away = best_market(odds_rows, "h2h", "away_ml")
    best_spread_home = best_market(odds_rows, "spreads", "home_spread_price")
    best_spread_away = best_market(odds_rows, "spreads", "away_spread_price")
    best_total = best_market(odds_rows, "totals", "over_price") or best_market(odds_rows, "totals", "under_price")
    tracking = {
        **base,
        "live_home_ml": best_h2h_home.get("home_ml"),
        "live_away_ml": best_h2h_away.get("away_ml"),
        "live_home_spread_line": best_spread_home.get("home_spread_line"),
        "live_home_spread_price": best_spread_home.get("home_spread_price"),
        "live_away_spread_line": best_spread_away.get("away_spread_line"),
        "live_away_spread_price": best_spread_away.get("away_spread_price"),
        "live_total_line": best_total.get("total_line"),
        "live_over_price": best_total.get("over_price"),
        "live_under_price": best_total.get("under_price"),
        **decision,
    }
    for path in (
        lgbm / "live_decision_tracking_latest.json",
        lgbm / f"live_decision_tracking_{run_date}.json",
    ):
        path.write_text(json.dumps(tracking, indent=2), encoding="utf-8")
        print(f"Wrote {path}")
    write_latest_csv(lgbm / "live_decision_tracking_latest.csv", tracking)
    write_latest_csv(lgbm / f"live_decision_tracking_{run_date}.csv", tracking)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture halftime live odds.")
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--game-date", required=True)
    parser.add_argument("--game-key", required=True)
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--mode", default="once", choices=["once", "retry"])
    parser.add_argument("--markets", default="h2h,spreads,totals")
    parser.add_argument("--bookmaker-key", default=os.environ.get("ODDS_BOOKMAKER_KEY") or None)
    parser.add_argument("--snapshot-label", default="halftime")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--lightgbm-dir", default=None)
    parser.add_argument("--fixture-json", default=None)
    parser.add_argument("--retry-window-minutes", type=float, default=0)
    parser.add_argument("--retry-interval-seconds", type=float, default=120)
    args = parser.parse_args()

    root = Path(args.source_root).resolve() if args.source_root else source_root()
    lgbm = Path(args.lightgbm_dir).resolve() if args.lightgbm_dir else lightgbm_dir(root)
    load_local_env(root)
    fixture = load_fixture(Path(args.fixture_json)) if args.fixture_json else None
    watchlist = read_watchlist_row(lgbm, args.game_key)
    started = time.monotonic()
    max_seconds = max(0.0, args.retry_window_minutes * 60.0) if args.mode == "retry" else 0.0
    attempt = 0

    while True:
        attempt += 1
        snapshot_utc = iso_z(utc_now())
        odds_rows = fetch_odds(
            api_key=os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY"),
            game_date=args.game_date,
            home_team=args.home_team,
            away_team=args.away_team,
            markets=args.markets,
            bookmaker_key=args.bookmaker_key,
            fixture=fixture,
        )
        score = fetch_score(
            game_date=args.game_date,
            home_team=args.home_team,
            away_team=args.away_team,
            fixture=fixture,
        )
        decision = classify_live_decision(
            home_team=args.home_team,
            away_team=args.away_team,
            score=score,
            odds_rows=odds_rows,
        )
        persist(
            lgbm=lgbm,
            run_date=args.run_date,
            game_date=args.game_date,
            game_key=args.game_key,
            home_team=args.home_team,
            away_team=args.away_team,
            snapshot_utc=snapshot_utc,
            snapshot_label=args.snapshot_label,
            watchlist=watchlist,
            score=score,
            odds_rows=odds_rows,
            decision=decision,
        )
        if (
            os.environ.get("HALFTIME_TOTAL_PREDICTION_ENABLED", "true").lower()
            not in {"0", "false", "no", "off"}
            and score.get("score_status") == "fetched"
        ):
            try:
                from halftime_total_prediction import run_from_artifacts

                total_record = run_from_artifacts(lgbm_dir=lgbm)
                print(
                    "halftime_total_prediction="
                    f"{total_record.get('direction')} "
                    f"edge={total_record.get('edge_points')} "
                    f"decision={total_record.get('candidate_decision')} "
                    f"order_status={total_record.get('order_status')}"
                )
            except Exception as exc:
                print(f"[WARN] halftime total prediction failed: {exc}", file=sys.stderr)
        print(
            f"attempt={attempt} {args.away_team} @ {args.home_team}: "
            f"{decision['live_classification']} · {decision['final_decision']} · "
            f"odds_rows={len(odds_rows)} score_status={score.get('score_status')} "
            f"period={score.get('period')} clock={score.get('game_clock')}"
        )
        if args.mode == "once":
            break
        if odds_rows and score.get("score_status") == "fetched":
            print("Halftime odds and score captured; retry loop complete.")
            break
        if time.monotonic() - started >= max_seconds:
            print("Retry window expired before a complete halftime odds+score capture.")
            break
        time.sleep(max(5.0, args.retry_interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
