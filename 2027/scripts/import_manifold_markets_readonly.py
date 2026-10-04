#!/usr/bin/env python3
"""Read-only Manifold NBA market adapter for the 2027 paper shell.

This script is intentionally outside the numbered daily pipeline.  It can fetch
or load Manifold market metadata, classify market types, and write comparison
evidence.  It never creates predictions, odds, betting signals, stakes, orders,
or canonical unlock state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "manifold_market_scan"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
OPENING_DAY_TEAMS = {
    "DET": "Detroit Pistons",
    "BOS": "Boston Celtics",
    "NYK": "New York Knicks",
    "PHI": "Philadelphia 76ers",
    "SAS": "San Antonio Spurs",
    "OKC": "Oklahoma City Thunder",
}
TEAM_NAMES = set(OPENING_DAY_TEAMS.values())
TEAM_WORDS = {
    "pistons": "DET",
    "celtics": "BOS",
    "knicks": "NYK",
    "76ers": "PHI",
    "sixers": "PHI",
    "spurs": "SAS",
    "thunder": "OKC",
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



def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def fetch_manifold_search(term: str, limit: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"term": term, "limit": limit, "filter": "open", "sort": "newest"})
    request = urllib.request.Request(f"{MANIFOLD_SEARCH_URL}?{params}", headers={"User-Agent": "nba2027-manifold-readonly/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        markets = payload.get("markets") or payload.get("results") or []
        return [item for item in markets if isinstance(item, dict)]
    return []


def load_markets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if args.input:
        path = Path(args.input)
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return [], {"source": "local_file", "input": str(path), "input_sha256": _sha256(path), "network_attempted": False}, ["LOCAL_MARKET_FILE_INVALID_JSON"]
        if isinstance(payload, dict):
            records = payload.get("markets") or payload.get("results") or []
        else:
            records = payload
        markets = [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []
        return markets, {"source": "local_file", "input": str(path), "input_sha256": _sha256(path), "network_attempted": False}, []

    issues: list[str] = []
    all_markets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in args.search_terms:
        try:
            records = fetch_manifold_search(term, args.limit)
        except urllib.error.HTTPError as exc:
            issues.append(f"MANIFOLD_HTTP_ERROR_{exc.code}")
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            issues.append("MANIFOLD_REQUEST_FAILED")
            continue
        for market in records:
            key = str(market.get("id") or market.get("slug") or market.get("url") or market.get("question"))
            if key in seen:
                continue
            seen.add(key)
            all_markets.append(market)
    return all_markets, {"source": "manifold_search_api", "search_terms": args.search_terms, "network_attempted": True}, issues


def market_url(market: dict[str, Any]) -> str | None:
    url = market.get("url")
    if isinstance(url, str) and url.startswith("http"):
        return url
    slug = market.get("slug")
    creator = market.get("creatorUsername") or market.get("creator", {}).get("username") if isinstance(market.get("creator"), dict) else None
    if creator and slug:
        return f"https://manifold.markets/{creator}/{slug}"
    return None


def probability(market: dict[str, Any]) -> float | None:
    value = market.get("probability")
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return None


def classify_market(question: str, description: str = "") -> tuple[str, list[str], list[str]]:
    text = f"{question} {description}".lower()
    matched_teams = sorted({abbr for word, abbr in TEAM_WORDS.items() if re.search(rf"\b{re.escape(word)}\b", text)})
    reasons: list[str] = []
    if re.search(r"\bcover\b|point spread|\bspread\b|\bby\s+\d+(?:\.\d+)?\s+or\s+more\b", text):
        reasons.append("SPREAD_TERMS")
        return "SPREAD", matched_teams, reasons
    if re.search(r"\bscore\b|points|rebounds|assists|wembanyama|player", text):
        reasons.append("PLAYER_PROP_TERMS")
        return "PLAYER_PROP", matched_teams, reasons
    if re.search(r"championship|playoffs|draft|mvp|season|twenty games|20 games", text):
        reasons.append("FUTURES_TERMS")
        return "FUTURES", matched_teams, reasons
    if len(matched_teams) >= 2 and re.search(r"\bwin\b|defeat|beat|winner", text):
        reasons.append("MONEYLINE_STYLE_TERMS")
        return "MONEYLINE", matched_teams, reasons
    if matched_teams:
        reasons.append("TEAM_MENTION_ONLY")
        return "UNKNOWN", matched_teams, reasons
    return "UNKNOWN", matched_teams, ["NO_OPENING_DAY_TEAM_MATCH"]


def line_value_from_question(question: str) -> float | None:
    cover = re.search(r"(?:cover\s+the\s+)?([+-]\d+(?:\.\d+)?)", question)
    if cover:
        return float(cover.group(1))
    by_more = re.search(r"\bby\s+(\d+(?:\.\d+)?)\s+or\s+more\b", question.lower())
    if by_more:
        return float(by_more.group(1))
    return None


def mapping_status_for(market_type: str, teams: list[str]) -> tuple[bool, str, str]:
    if market_type == "MONEYLINE":
        if len(teams) >= 2:
            return False, "MONEYLINE_REVIEW_REQUIRED", "MANIFOLD_WATCH_ONLY"
        return False, "MONEYLINE_GAME_MATCH_MISSING", "MANIFOLD_WATCH_ONLY"
    if market_type == "SPREAD":
        return False, "UNMAPPED_SPREAD_MARKET", "MANIFOLD_WATCH_ONLY"
    if market_type == "PLAYER_PROP":
        return False, "PROP_MODEL_MISSING", "MANIFOLD_WATCH_ONLY"
    if market_type == "FUTURES":
        return False, "FUTURES_MODEL_MISSING", "MANIFOLD_WATCH_ONLY"
    return False, "UNKNOWN_MARKET_UNMAPPED", "MANIFOLD_WATCH_ONLY"


def normalize_markets(markets: list[dict[str, Any]], run_id: str, as_of_utc: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for market in markets:
        question = str(market.get("question") or market.get("title") or "").strip()
        description = str(market.get("description") or "")
        market_type, teams, reasons = classify_market(question, description)
        mapped_to_model, mapping_status, research_label = mapping_status_for(market_type, teams)
        home_team = teams[0] if teams else None
        away_team = teams[1] if len(teams) > 1 else None
        line_type = market_type if market_type in {"MONEYLINE", "SPREAD", "TOTAL"} else None
        rows.append(
            {
                "run_id": run_id,
                "as_of_utc": as_of_utc,
                "market_url": market_url(market),
                "market_title": question,
                "market_type": market_type,
                "league": "NBA" if "nba" in question.lower() or teams else None,
                "game_date": "2026-10-20" if "october 20, 2026" in question.lower() or "oct 20" in question.lower() else None,
                "home_team": home_team,
                "away_team": away_team,
                "line_type": line_type,
                "line_value": line_value_from_question(question) if market_type == "SPREAD" else None,
                "side": None,
                "manifold_probability": probability(market),
                "liquidity_if_available": market.get("totalLiquidity") or market.get("liquidity"),
                "volume_if_available": market.get("volume"),
                "resolved_status": bool(market.get("isResolved") or market.get("is_resolved", False)),
                "mapped_to_model": mapped_to_model,
                "mapping_status": mapping_status,
                "reason_code": ",".join(reasons),
                "canonical_decision": "NO_BET",
                "research_label": research_label,
                "execution_enabled": False,
            }
        )
    return pd.DataFrame(rows)


def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    markets, source_meta, issues = load_markets(args)
    as_of_utc = now.isoformat().replace("+00:00", "Z")
    frame = normalize_markets(markets, args.run_id_value, as_of_utc)
    if not frame.empty and args.open_only:
        frame = frame[~frame["resolved_status"]].copy()
    type_counts = {} if frame.empty else {str(k): int(v) for k, v in frame["market_type"].value_counts().sort_index().items()}
    status = "READY" if markets and not issues else ("READY_WITH_WARNINGS" if markets else "SKIPPED")
    reason_code = "MANIFOLD_MARKETS_CLASSIFIED_READ_ONLY" if markets else (issues[0] if issues else "NO_MARKETS_FOUND")
    return {
        "script": "import_manifold_markets_readonly",
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "status": status,
        "reason_code": reason_code,
        "issues": issues,
        "inputs": source_meta,
        "results": {
            "markets_seen": int(len(markets)),
            "markets_written": int(len(frame)),
            "market_type_counts": type_counts,
            "moneyline_rows": int((frame.get("market_type", pd.Series(dtype=str)) == "MONEYLINE").sum()) if not frame.empty else 0,
            "spread_rows": int((frame.get("market_type", pd.Series(dtype=str)) == "SPREAD").sum()) if not frame.empty else 0,
            "player_prop_rows": int((frame.get("market_type", pd.Series(dtype=str)) == "PLAYER_PROP").sum()) if not frame.empty else 0,
            "futures_rows": int((frame.get("market_type", pd.Series(dtype=str)) == "FUTURES").sum()) if not frame.empty else 0,
        },
        "safety": {
            "read_only_market_metadata": True,
            "outside_numbered_daily_pipeline": True,
            "prediction_rows_created": False,
            "odds_created": False,
            "betting_signals_created": False,
            "stake_created": False,
            "orders_created": False,
            "canonical_unlock": False,
            "execution_path_created": False,
            "manifold_execution_path_created": False,
        },
        "_frame": frame,
    }


def write_outputs(report: dict[str, Any], output_root: Path, run_id: str) -> Path:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = report.pop("_frame")
    markets_csv = output_dir / "markets.csv"
    markets_json = output_dir / "markets.json"
    validation_json = output_dir / "validation.json"
    readme = output_dir / "README.md"
    frame.to_csv(markets_csv, index=False)
    markets_json.write_text(json.dumps(json_records(frame), indent=2, sort_keys=True, allow_nan=False) + "\n")
    report["results"]["markets_csv"] = str(markets_csv)
    report["results"]["markets_json"] = str(markets_json)
    validation_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    readme.write_text("# Manifold NBA market scan\n\nRead-only research evidence only. These rows do not feed Script 5, do not unlock canonical betting, and do not create orders. Spread, player prop, futures, and unknown markets remain unmapped to the current moneyline-style model.\n")
    return validation_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Manifold NBA market classification for 2027.")
    parser.add_argument("--input", help="Optional local JSON fixture or exported Manifold search result.")
    parser.add_argument("--search-term", dest="search_terms", action="append", default=[], help="Manifold search term. Can be repeated.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-resolved", dest="open_only", action="store_false")
    parser.set_defaults(open_only=True)
    parser.add_argument("--run-id", help="Stable output folder name for evidence. Defaults to current UTC timestamp.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    if not args.search_terms:
        args.search_terms = [
            "NBA",
            "NBA Knicks",
            "NBA Pistons Celtics",
            "NBA Spurs Thunder",
            "NBA 2027 opening night",
        ]
    return args


def main() -> int:
    args = parse_args()
    now = _utc_now()
    run_id = args.run_id or _run_id(now)
    args.run_id_value = run_id
    report = build_report(args, now)
    report_path = write_outputs(report, Path(args.output_root), run_id)
    print(json.dumps({"status": report["status"], "reason_code": report["reason_code"], "validation": str(report_path)}))
    return 0 if report["status"] in {"READY", "READY_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
