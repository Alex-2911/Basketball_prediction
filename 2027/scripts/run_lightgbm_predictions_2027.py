#!/usr/bin/env python3
"""Guarded 2027 rewrite of Script 3 LightGBM prediction step.

The 2026 notebook trained LightGBM inside the daily script and fell back to the
latest available statistics CSV.  For the 2027 paper shell this is intentionally
fail-closed: the script validates the Script 2 next-game slate, verifies whether
real 2027 played-game statistics or an explicit prediction artifact exist, and
does not train, backfill, fetch, or create model probabilities by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEASON_END_YEAR = 2027
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "script3_model_predictions"
DEFAULT_NEXT_GAME_DIR = ROOT / "data" / "raw" / "Gathering_Data" / "Next_Game"
DEFAULT_STATS_DIR = ROOT / "data" / "raw" / "Gathering_Data" / "Whole_Statistic"
DEFAULT_BASELINE_2026_STATS = ROOT / "data" / "processed" / "whole_statistics_2026.parquet"
MODEL_COLUMNS = {"home_team_prob", "prob_iso", "home_win_rate"}
ODDS_COLUMNS = {"odds_1", "odds_2"}
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
LEGACY_HARDCODED_KEY_SOURCE = Path("/Users/alexanderrazmyslov/1. Python/1. NBA Script/2026/_3. 25102025_lightgbm_extracted.py")
FULL_TO_ABBREV = {"Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BRK","Charlotte Hornets":"CHO","Chicago Bulls":"CHI","Cleveland Cavaliers":"CLE","Dallas Mavericks":"DAL","Denver Nuggets":"DEN","Detroit Pistons":"DET","Golden State Warriors":"GSW","Houston Rockets":"HOU","Indiana Pacers":"IND","LA Clippers":"LAC","Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL","Memphis Grizzlies":"MEM","Miami Heat":"MIA","Milwaukee Bucks":"MIL","Minnesota Timberwolves":"MIN","New Orleans Pelicans":"NOP","New York Knicks":"NYK","Oklahoma City Thunder":"OKC","Orlando Magic":"ORL","Philadelphia 76ers":"PHI","Phoenix Suns":"PHO","Portland Trail Blazers":"POR","Sacramento Kings":"SAC","San Antonio Spurs":"SAS","Toronto Raptors":"TOR","Utah Jazz":"UTA","Washington Wizards":"WAS"}
API_ABBREV_TO_LOCAL = {"PHX":"PHO","CHA":"CHO","BKN":"BRK"}
LOCAL_TO_API_ABBREV = {"PHO":"PHX","CHO":"CHA","BRK":"BKN"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S%fZ")


def _parse_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_next_game_slate(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, ["MISSING_NEXT_GAME_SLATE"]
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].copy()
    missing = [col for col in ["home_team", "away_team", "game_date"] if col not in frame.columns]
    if missing:
        return None, [f"MISSING_NEXT_GAME_COLUMN_{col}" for col in missing]
    frame = frame.dropna(subset=["home_team", "away_team", "game_date"]).copy()
    frame["home_team"] = frame["home_team"].astype(str).str.strip()
    frame["away_team"] = frame["away_team"].astype(str).str.strip()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["game_date"]).drop_duplicates(["home_team", "away_team", "game_date"])
    if frame.empty:
        return None, ["EMPTY_NEXT_GAME_SLATE"]
    bad_codes = sorted(
        {
            value
            for value in set(frame["home_team"]).union(frame["away_team"])
            if not isinstance(value, str) or len(value) != 3
        }
    )
    if bad_codes:
        return None, ["INVALID_TEAM_CODES_IN_NEXT_GAME_SLATE"]
    return frame.reset_index(drop=True), []


def list_2027_stat_files(stats_dir: Path) -> list[Path]:
    if not stats_dir.exists():
        return []
    return sorted(stats_dir.glob("nba_games_2027-*.csv"))


def validate_prediction_artifact(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, ["PREDICTION_ARTIFACT_NOT_FOUND"]
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"game_date", "home_team", "away_team"} | MODEL_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        return None, [f"PREDICTION_ARTIFACT_MISSING_{col}" for col in missing]
    return frame, []


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train_baseline_2026_predictions(stats_path: Path, next_games: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    if not stats_path.exists():
        return None, ["BASELINE_2026_STATS_NOT_FOUND"]

    try:
        import lightgbm as lgb
    except Exception:
        return None, ["LIGHTGBM_NOT_AVAILABLE"]

    raw = _read_table(stats_path).copy()
    required = {"date", "team", "team_opp", "home", "won"}
    missing = sorted(required - set(raw.columns))
    if missing:
        return None, [f"BASELINE_STATS_MISSING_{col}" for col in missing]

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "team", "team_opp", "home", "won"]).copy()
    raw["team"] = raw["team"].astype(str).str.strip()
    raw["team_opp"] = raw["team_opp"].astype(str).str.strip()
    raw["home"] = pd.to_numeric(raw["home"], errors="coerce").fillna(0).astype(int)
    raw["won"] = raw["won"].astype(int)
    raw = raw.sort_values(["team", "date"]).reset_index(drop=True)

    exclude = {"season", "date", "won", "target", "team", "team_opp"}
    numeric_cols = [
        col
        for col in raw.select_dtypes(include=["number", "bool"]).columns
        if col not in exclude and not str(col).startswith("index")
    ]
    if not numeric_cols:
        return None, ["NO_NUMERIC_BASELINE_FEATURES"]

    rolling = (
        raw.groupby("team", group_keys=False)[numeric_cols]
        .apply(lambda group: group.shift(1).rolling(9, min_periods=1).mean())
        .add_prefix("roll_")
    )
    model_frame = pd.concat([raw[["date", "team", "team_opp", "home", "won"]], rolling], axis=1)
    feature_cols = list(rolling.columns)
    home_rows = model_frame[model_frame["home"] == 1].copy()
    away_rows = model_frame[model_frame["home"] == 0].copy()
    joined = home_rows.merge(
        away_rows[["date", "team"] + feature_cols],
        left_on=["date", "team_opp"],
        right_on=["date", "team"],
        suffixes=("_home", "_away"),
    )
    home_features = [f"{col}_home" for col in feature_cols]
    away_features = [f"{col}_away" for col in feature_cols]
    train_features = home_features + away_features
    train = joined.dropna(subset=["won"]).copy()
    train_features = [col for col in train_features if col in train.columns and train[col].notna().sum() > 0]
    feature_medians = train[train_features].median(numeric_only=True).fillna(0)
    train[train_features] = train[train_features].fillna(feature_medians).fillna(0)
    if len(train) < 100:
        return None, ["INSUFFICIENT_BASELINE_TRAINING_ROWS"]

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        num_leaves=10,
        learning_rate=0.1,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=10,
        boosting_type="gbdt",
        verbosity=-1,
        random_state=42,
        lambda_l1=0.5,
        lambda_l2=0.5,
        max_depth=7,
        min_child_weight=5,
    )
    model.fit(train[train_features], train["won"].astype(int))

    latest = model_frame.sort_values("date").groupby("team").tail(1).set_index("team")
    pred_rows: list[dict[str, object]] = []
    for row in next_games.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        if home not in latest.index or away not in latest.index:
            pred_rows.append(
                {
                    "game_date": row.game_date,
                    "home_team": home,
                    "away_team": away,
                    "home_team_prob": None,
                    "prob_iso": None,
                    "home_win_rate": None,
                    "model_source": "baseline_2026_lightgbm",
                    "model_issue": "TEAM_BASELINE_HISTORY_MISSING",
                }
            )
            continue
        vector: dict[str, float] = {}
        for col in feature_cols:
            vector[f"{col}_home"] = float(latest.loc[home, col])
            vector[f"{col}_away"] = float(latest.loc[away, col])
        pred_input = pd.DataFrame([vector], columns=train_features)
        pred_input = pred_input.fillna(feature_medians).fillna(0)
        probability = float(model.predict_proba(pred_input)[0, 1])
        pred_rows.append(
            {
                "game_date": row.game_date,
                "home_team": home,
                "away_team": away,
                "home_team_prob": probability,
                "prob_iso": probability,
                "home_win_rate": probability,
                "model_source": "baseline_2026_lightgbm",
                "model_issue": "",
            }
        )

    return pd.DataFrame(pred_rows), []



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


def read_legacy_hardcoded_key(path: Path = LEGACY_HARDCODED_KEY_SOURCE) -> str | None:
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


def resolve_odds_api_key(args: argparse.Namespace) -> tuple[str | None, str]:
    explicit = getattr(args, "odds_api_key", None)
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
    request = urllib.request.Request(f"{ODDS_URL}?{params}", headers={"User-Agent": "nba2027-script3/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_odds_for_predictions(predictions: pd.DataFrame, api_key: str, preferred: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
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

    out = predictions.copy()
    odds_1: list[float | None] = []
    odds_2: list[float | None] = []
    home_raw: list[object] = []
    away_raw: list[object] = []
    bookmaker_key: list[object] = []
    bookmaker_title: list[object] = []
    api_commence_time: list[object] = []
    odds_status: list[str] = []
    for row in out.itertuples(index=False):
        home = canonical_abbrev(getattr(row, "home_team"))
        away = canonical_abbrev(getattr(row, "away_team"))
        found = lookup.get((home, away), {})
        h = found.get("home_odds_american")
        a = found.get("away_odds_american")
        odds_1.append(american_to_decimal(h))
        odds_2.append(american_to_decimal(a))
        home_raw.append(h)
        away_raw.append(a)
        bookmaker_key.append(found.get("bookmaker_key"))
        bookmaker_title.append(found.get("bookmaker_title"))
        api_commence_time.append(found.get("api_commence_time"))
        odds_status.append("OK" if h is not None and a is not None else "NO_ODDS_FOUND")
    out["odds_1"] = odds_1
    out["odds_2"] = odds_2
    out["home_odds_american"] = home_raw
    out["away_odds_american"] = away_raw
    out["bookmaker_key"] = bookmaker_key
    out["bookmaker_title"] = bookmaker_title
    out["api_commence_time"] = api_commence_time
    out["odds_status"] = odds_status
    return out, {"api_events_seen": len(data), "api_events_matched": len(lookup), "odds_ok_rows": int((out["odds_status"] == "OK").sum()), "odds_missing_rows": int((out["odds_status"] != "OK").sum())}


def attach_odds_if_enabled(predictions: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if not getattr(args, "fetch_odds", True):
        return predictions, [], {"odds_fetch_enabled": False, "network_attempted": False, "api_key_source": "disabled", "api_events_seen": 0, "api_events_matched": 0, "odds_ok_rows": 0, "odds_missing_rows": len(predictions)}
    api_key, source = resolve_odds_api_key(args)
    if not api_key:
        return predictions, ["NO_VERIFIED_ODDS_API_KEY"], {"odds_fetch_enabled": True, "network_attempted": False, "api_key_source": source, "api_events_seen": 0, "api_events_matched": 0, "odds_ok_rows": 0, "odds_missing_rows": len(predictions)}
    preferred = [x.strip() for x in str(getattr(args, "preferred_bookmakers", "draftkings,fanduel,betmgm") or "").split(",") if x.strip()]
    try:
        enriched, meta = fetch_odds_for_predictions(predictions, api_key, preferred=preferred or None)
        return enriched, [], {"odds_fetch_enabled": True, "network_attempted": True, "api_key_source": source, **meta}
    except urllib.error.HTTPError as exc:
        return predictions, [f"ODDS_API_HTTP_ERROR_{exc.code}"], {"odds_fetch_enabled": True, "network_attempted": True, "api_key_source": source, "api_events_seen": 0, "api_events_matched": 0, "odds_ok_rows": 0, "odds_missing_rows": len(predictions)}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return predictions, ["ODDS_API_REQUEST_FAILED"], {"odds_fetch_enabled": True, "network_attempted": True, "api_key_source": source, "api_events_seen": 0, "api_events_matched": 0, "odds_ok_rows": 0, "odds_missing_rows": len(predictions)}

def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    anchor_date = _parse_date(args.date) if args.date else now.date()
    next_game_path = Path(args.next_game_file) if args.next_game_file else Path(args.next_game_dir) / f"games_df_{anchor_date.isoformat()}.csv"
    next_games, slate_issues = load_next_game_slate(next_game_path)
    stat_files = list_2027_stat_files(Path(args.stats_dir))
    prediction_artifact = Path(args.prediction_artifact) if args.prediction_artifact else None

    status = "SKIPPED"
    reason_code = "MODEL_PIPELINE_LOCKED"
    issues = list(slate_issues)
    predictions_file: str | None = None
    predictions_rows = 0
    odds_meta: dict[str, Any] = {"odds_fetch_enabled": bool(getattr(args, "fetch_odds", True)), "network_attempted": False, "api_key_source": "not_run", "api_events_seen": 0, "api_events_matched": 0, "odds_ok_rows": 0, "odds_missing_rows": 0}

    if slate_issues:
        reason_code = slate_issues[0]
    elif prediction_artifact:
        artifact, artifact_issues = validate_prediction_artifact(prediction_artifact)
        issues.extend(artifact_issues)
        if artifact_issues:
            reason_code = artifact_issues[0]
        else:
            status = "READY"
            reason_code = "EXPLICIT_PREDICTION_ARTIFACT_VALIDATED"
            artifact, odds_issues, odds_meta = attach_odds_if_enabled(artifact, args)
            issues.extend(odds_issues)
            if odds_issues:
                status = "SKIPPED"
                reason_code = odds_issues[0]
            predictions_rows = int(len(artifact))
            if status == "READY" and args.copy_prediction_artifact:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "validated_predictions.csv"
                artifact.to_csv(out_path, index=False)
                predictions_file = str(out_path)
    elif args.allow_baseline_2026:
        baseline_predictions, baseline_issues = train_baseline_2026_predictions(Path(args.baseline_2026_stats), next_games)
        issues.extend(baseline_issues)
        if baseline_issues:
            reason_code = baseline_issues[0]
        else:
            baseline_predictions, odds_issues, odds_meta = attach_odds_if_enabled(baseline_predictions, args)
            issues.extend(odds_issues)
            if odds_issues:
                reason_code = odds_issues[0]
            else:
                status = "READY"
                reason_code = "BASELINE_2026_PREDICTIONS_WITH_ODDS_CREATED"
                predictions_rows = int(len(baseline_predictions))
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "baseline_2026_predictions.csv"
                baseline_predictions.to_csv(out_path, index=False)
                predictions_file = str(out_path)
    elif not stat_files:
        reason_code = "WAITING_FOR_FIRST_2027_PLAYED_GAME_STATS"
        issues.append("NO_2027_STATISTICS_AVAILABLE")
    else:
        reason_code = "EXPLICIT_MODEL_ARTIFACT_REQUIRED"
        issues.append("TRAINING_DISABLED_BY_DEFAULT")

    games = [] if next_games is None else next_games.to_dict("records")
    return {
        "script": "run_lightgbm_predictions_2027",
        "season_end_year": SEASON_END_YEAR,
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "anchor_date": anchor_date.isoformat(),
        "status": status,
        "reason_code": reason_code,
        "issues": issues,
        "inputs": {
            "next_game_file": str(next_game_path),
            "next_game_file_exists": next_game_path.exists(),
            "next_game_file_sha256": _sha256(next_game_path),
            "stats_dir": str(Path(args.stats_dir)),
            "stat_files_2027": [str(path) for path in stat_files],
            "prediction_artifact": None if prediction_artifact is None else str(prediction_artifact),
            "prediction_artifact_sha256": None if prediction_artifact is None else _sha256(prediction_artifact),
            "baseline_2026_stats": str(Path(args.baseline_2026_stats)),
            "baseline_2026_stats_sha256": _sha256(Path(args.baseline_2026_stats)),
        },
        "results": {
            "next_games_found": len(games),
            "next_games": games,
            "predictions_created": bool(predictions_file),
            "predictions_rows": predictions_rows,
            "predictions_file": predictions_file,
            "output_csv": predictions_file,
            "odds": odds_meta,
        },
        "safety": {
            "model_step_only": True,
            "network_attempted": bool(odds_meta.get("network_attempted")),
            "stats_fallback_to_2026_used": False,
            "training_executed": bool(args.allow_baseline_2026 and status == "READY"),
            "lightgbm_fit_called": bool(args.allow_baseline_2026 and status == "READY"),
            "predictions_generated_by_script": bool(args.allow_baseline_2026 and status == "READY"),
            "baseline_2026_mode": bool(args.allow_baseline_2026),
            "baseline_2026_predictions_are_preseason_proxy": bool(args.allow_baseline_2026 and status == "READY"),
            "odds_created": bool(predictions_file and odds_meta.get("odds_ok_rows", 0) > 0),
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
    parser = argparse.ArgumentParser(description="Guarded 2027 Script 3 LightGBM prediction step.")
    parser.add_argument("--date", help="Anchor date in YYYY-MM-DD form. Defaults to current UTC date.")
    parser.add_argument("--next-game-file", help="Explicit Script 2 games_df file.")
    parser.add_argument("--next-game-dir", default=str(DEFAULT_NEXT_GAME_DIR))
    parser.add_argument("--stats-dir", default=str(DEFAULT_STATS_DIR))
    parser.add_argument("--prediction-artifact", help="Explicit externally validated prediction CSV or Parquet.")
    parser.add_argument("--copy-prediction-artifact", action="store_true")
    parser.add_argument("--allow-baseline-2026", action="store_true", help="Use preserved 2026 stats as an explicit preseason baseline proxy.")
    parser.add_argument("--baseline-2026-stats", default=str(DEFAULT_BASELINE_2026_STATS))
    parser.add_argument("--no-fetch-odds", dest="fetch_odds", action="store_false", help="Disable Script 3 The Odds API fetch.")
    parser.set_defaults(fetch_odds=True)
    parser.add_argument("--odds-api-key", help="The Odds API key. If omitted, Script 3 resolves env or the local legacy 2026 key.")
    parser.add_argument("--no-legacy-hardcoded-key", dest="allow_legacy_hardcoded_key", action="store_false")
    parser.set_defaults(allow_legacy_hardcoded_key=True)
    parser.add_argument("--preferred-bookmakers", default="draftkings,fanduel,betmgm")
    parser.add_argument("--run-id", help="Stable output folder name for evidence. Defaults to current UTC timestamp.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = _utc_now()
    run_id = args.run_id or _run_id(now)
    output_dir = Path(args.output_root) / run_id
    args.output_dir = str(output_dir)
    report = build_report(args, now)
    report_path = write_report(report, Path(args.output_root), run_id)
    print(json.dumps({"status": report["status"], "reason_code": report["reason_code"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
