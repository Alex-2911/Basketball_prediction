#!/usr/bin/env python3
"""Halftime final-total prediction and Manifold candidate generation.

This module is deliberately separate from canonical NBA moneyline selection.
Default behavior is dry-run only: it writes prediction/candidate artifacts and
never places Manifold orders unless explicit live-order flags are enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from manifold_api_client import ManifoldClient
except ImportError:  # pragma: no cover - supports package-style imports in tests.
    from scripts.manifold_api_client import ManifoldClient


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TotalPredictionConfig:
    dry_run: bool = True
    live_order_enabled: bool = False
    max_stake_mana: float = 10.0
    max_market_exposure_mana: float = 25.0
    min_edge_points: float = 5.0
    min_confidence: str = "MEDIUM"
    min_price_decimal: float = 1.70
    baseline_total: float = 218.0
    live_pace_weight: float = 0.55
    market_regression_weight: float = 0.45
    allow_duplicate_order: bool = False
    manifold_market_id: str = ""
    over_outcome: str = "YES"
    under_outcome: str = "NO"

    @classmethod
    def from_env(cls) -> "TotalPredictionConfig":
        return cls(
            dry_run=parse_bool(os.environ.get("MANIFOLD_DRY_RUN"), default=True),
            live_order_enabled=parse_bool(os.environ.get("MANIFOLD_ENABLE_LIVE_ORDERS"), default=False),
            max_stake_mana=float(os.environ.get("MANIFOLD_MAX_STAKE_MANA", "10")),
            max_market_exposure_mana=float(os.environ.get("MANIFOLD_MAX_MARKET_EXPOSURE_MANA", "25")),
            min_edge_points=float(os.environ.get("MANIFOLD_MIN_EDGE_POINTS", "5")),
            min_confidence=os.environ.get("MANIFOLD_MIN_CONFIDENCE", "MEDIUM").upper(),
            min_price_decimal=float(os.environ.get("MANIFOLD_MIN_PRICE_DECIMAL", "1.70")),
            baseline_total=float(os.environ.get("HALFTIME_TOTAL_BASELINE", "218")),
            live_pace_weight=float(os.environ.get("HALFTIME_TOTAL_LIVE_PACE_WEIGHT", "0.55")),
            market_regression_weight=float(os.environ.get("HALFTIME_TOTAL_MARKET_REGRESSION_WEIGHT", "0.45")),
            allow_duplicate_order=parse_bool(os.environ.get("MANIFOLD_ALLOW_DUPLICATE_ORDER"), default=False),
            manifold_market_id=os.environ.get("MANIFOLD_TOTAL_MARKET_ID", ""),
            over_outcome=os.environ.get("MANIFOLD_TOTAL_OVER_OUTCOME", "YES").upper(),
            under_outcome=os.environ.get("MANIFOLD_TOTAL_UNDER_OUTCOME", "NO").upper(),
        )


CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def confidence_meets(actual: str, minimum: str) -> bool:
    return CONFIDENCE_RANK.get(actual, 0) >= CONFIDENCE_RANK.get(minimum, 1)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_snapshot_rows(path: Path, *, game_key: str, snapshot_utc: str | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("game_key") == game_key]
    if snapshot_utc:
        exact = [r for r in rows if r.get("snapshot_utc") == snapshot_utc]
        if exact:
            return exact
    return rows


def best_total_market(rows: list[dict[str, str]]) -> dict[str, Any]:
    total_rows = [r for r in rows if r.get("market_type") == "totals" and parse_float(r.get("total_line")) is not None]
    if not total_rows:
        return {}
    # Prefer a market that has both sides priced and the most recent row order.
    both_sides = [r for r in total_rows if parse_float(r.get("over_price")) and parse_float(r.get("under_price"))]
    row = (both_sides or total_rows)[-1]
    return {
        "market_total": parse_float(row.get("total_line")),
        "over_price": parse_float(row.get("over_price")),
        "under_price": parse_float(row.get("under_price")),
        "bookmaker_key": row.get("bookmaker_key") or "",
        "bookmaker_title": row.get("bookmaker_title") or "",
    }


def compute_total_prediction(
    *,
    halftime_home: float,
    halftime_away: float,
    market_total: float | None,
    config: TotalPredictionConfig,
    q1_total: float | None = None,
    q2_total: float | None = None,
) -> dict[str, Any]:
    halftime_total = halftime_home + halftime_away
    live_pace_projection = halftime_total * 2.0
    regression_target = market_total if market_total is not None else config.baseline_total
    predicted_final_total = (
        config.live_pace_weight * live_pace_projection
        + config.market_regression_weight * regression_target
    )
    projected_second_half_total = predicted_final_total - halftime_total

    edge_points = 0.0
    direction = "NO_EDGE"
    if market_total is not None:
        edge_points = predicted_final_total - market_total
        if edge_points >= config.min_edge_points:
            direction = "OVER"
        elif edge_points <= -config.min_edge_points:
            direction = "UNDER"

    abs_edge = abs(edge_points)
    confidence = "LOW"
    if market_total is not None and abs_edge >= config.min_edge_points:
        confidence = "MEDIUM"
    if market_total is not None and abs_edge >= max(config.min_edge_points + 3.0, config.min_edge_points * 1.6):
        confidence = "HIGH"

    q_delta = None
    if q1_total is not None and q2_total is not None:
        q_delta = q2_total - q1_total
        if abs(q_delta) >= 18 and confidence == "HIGH":
            confidence = "MEDIUM"

    pace_agrees = True
    if market_total is not None and direction != "NO_EDGE":
        pace_edge = live_pace_projection - market_total
        pace_agrees = (direction == "OVER" and pace_edge > 0) or (direction == "UNDER" and pace_edge < 0)

    return {
        "halftime_total": round(halftime_total, 2),
        "live_pace_projection": round(live_pace_projection, 2),
        "regression_target_total": round(regression_target, 2),
        "projected_second_half_total": round(projected_second_half_total, 2),
        "predicted_final_total": round(predicted_final_total, 2),
        "market_total": market_total,
        "model_edge_points": round(edge_points, 2),
        "direction": direction,
        "confidence_bucket": confidence,
        "pace_tendency_agrees": pace_agrees,
        "q1_total": q1_total,
        "q2_total": q2_total,
        "q2_minus_q1_total": q_delta,
    }


def prior_order_exists(log_path: Path, *, game_key: str, market_id: str, direction: str) -> bool:
    if not log_path.exists() or not market_id:
        return False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("game_key") == game_key
            and row.get("manifold_market_id") == market_id
            and row.get("direction") == direction
            and row.get("order_status") in {"dry_run", "placed"}
        ):
            return True
    return False


def decide_candidate(
    *,
    prediction: dict[str, Any],
    total_market: dict[str, Any],
    tracking: dict[str, Any],
    config: TotalPredictionConfig,
    execution_log_path: Path,
) -> dict[str, Any]:
    blocked: list[str] = []
    direction = prediction["direction"]

    if direction == "NO_EDGE":
        blocked.append("no_edge")
    if not confidence_meets(prediction["confidence_bucket"], config.min_confidence):
        blocked.append("low_confidence")
    if abs(float(prediction["model_edge_points"])) < config.min_edge_points:
        blocked.append("edge_below_threshold")
    if not prediction.get("pace_tendency_agrees"):
        blocked.append("pace_contradiction")
    if prediction.get("market_total") is None:
        blocked.append("market_total_missing")

    price_field = "over_price" if direction == "OVER" else "under_price"
    market_price = parse_float(total_market.get(price_field))
    if direction != "NO_EDGE" and (market_price is None or market_price < config.min_price_decimal):
        blocked.append("price_not_acceptable")

    market_id = config.manifold_market_id
    if not market_id:
        blocked.append("missing_manifold_market_id")
    if (
        not config.allow_duplicate_order
        and prior_order_exists(
            execution_log_path,
            game_key=str(tracking.get("game_key") or ""),
            market_id=market_id,
            direction=direction,
        )
    ):
        blocked.append("duplicate_order")

    candidate_decision = "BET_CANDIDATE" if not blocked else "SKIP"
    intended_stake = min(config.max_stake_mana, config.max_market_exposure_mana)
    order_status = "not_attempted"
    if candidate_decision == "BET_CANDIDATE" and config.dry_run:
        order_status = "dry_run"
    elif candidate_decision == "BET_CANDIDATE" and not config.live_order_enabled:
        blocked.append("live_orders_disabled")
        candidate_decision = "SKIP"
        order_status = "not_attempted"

    outcome = ""
    if direction == "OVER":
        outcome = config.over_outcome
    elif direction == "UNDER":
        outcome = config.under_outcome

    return {
        "candidate_decision": candidate_decision,
        "blocked_by": "|".join(blocked),
        "market_price": market_price,
        "manifold_market_id": market_id,
        "manifold_outcome": outcome,
        "intended_stake": intended_stake,
        "order_status": order_status,
        "response_payload_summary": {},
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_record(
    *,
    tracking: dict[str, Any],
    prediction: dict[str, Any],
    candidate: dict[str, Any],
    config: TotalPredictionConfig,
    total_market: dict[str, Any],
    timestamp_utc: str,
) -> dict[str, Any]:
    halftime_score = f"{tracking.get('away_team')} {tracking.get('halftime_score_away')} - {tracking.get('home_team')} {tracking.get('halftime_score_home')}"
    return {
        "timestamp_utc": timestamp_utc,
        "game_key": tracking.get("game_key"),
        "game_date": tracking.get("game_date"),
        "home_team": tracking.get("home_team"),
        "away_team": tracking.get("away_team"),
        "halftime_score": halftime_score,
        "halftime_score_home": tracking.get("halftime_score_home"),
        "halftime_score_away": tracking.get("halftime_score_away"),
        "halftime_total": prediction["halftime_total"],
        "predicted_final_total": prediction["predicted_final_total"],
        "projected_second_half_total": prediction["projected_second_half_total"],
        "market_total": prediction["market_total"],
        "edge_points": prediction["model_edge_points"],
        "direction": prediction["direction"],
        "confidence_bucket": prediction["confidence_bucket"],
        "pace_tendency_agrees": prediction["pace_tendency_agrees"],
        "bookmaker_key": total_market.get("bookmaker_key", ""),
        "bookmaker_title": total_market.get("bookmaker_title", ""),
        "candidate_decision": candidate["candidate_decision"],
        "blocked_by": candidate["blocked_by"],
        "dry_run": config.dry_run,
        "live_order_enabled": config.live_order_enabled,
        "manifold_market_id": candidate["manifold_market_id"],
        "manifold_outcome": candidate["manifold_outcome"],
        "intended_stake": candidate["intended_stake"],
        "order_status": candidate["order_status"],
        "response_payload_summary": candidate["response_payload_summary"],
        "canonical_model_context": tracking.get("pregame_model_label"),
        "stage2_candidate_type": tracking.get("stage2_candidate_type"),
    }


def maybe_place_order(record: dict[str, Any], config: TotalPredictionConfig) -> dict[str, Any]:
    if record["candidate_decision"] != "BET_CANDIDATE" or config.dry_run or not config.live_order_enabled:
        return record
    if not os.environ.get("MANIFOLD_API_KEY"):
        record["candidate_decision"] = "SKIP"
        record["blocked_by"] = "|".join(filter(None, [str(record.get("blocked_by") or ""), "missing_api_key"]))
        record["order_status"] = "not_attempted"
        return record

    client = ManifoldClient()
    response = client.place_bet(
        contract_id=str(record["manifold_market_id"]),
        amount=float(record["intended_stake"]),
        outcome=str(record["manifold_outcome"]),
    )
    record["order_status"] = "placed" if response.ok else "api_error"
    record["response_payload_summary"] = response.summary()
    if not response.ok:
        record["candidate_decision"] = "SKIP"
        record["blocked_by"] = "|".join(filter(None, [str(record.get("blocked_by") or ""), "api_error"]))
    return record


def run_from_artifacts(
    *,
    lgbm_dir: Path,
    tracking_path: Path | None = None,
    snapshots_path: Path | None = None,
    output_dir: Path | None = None,
    config: TotalPredictionConfig | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    lgbm_dir = Path(lgbm_dir)
    tracking = read_json(tracking_path or lgbm_dir / "live_decision_tracking_latest.json")
    run_date = str(tracking.get("run_date") or tracking.get("game_date") or "unknown")
    game_key = str(tracking.get("game_key") or "unknown_game")
    snapshots = read_snapshot_rows(
        snapshots_path or lgbm_dir / f"live_odds_snapshots_{run_date}.csv",
        game_key=game_key,
        snapshot_utc=tracking.get("snapshot_utc"),
    )
    total_market = best_total_market(snapshots)
    cfg = config or TotalPredictionConfig.from_env()
    home_ht = parse_float(tracking.get("halftime_score_home"))
    away_ht = parse_float(tracking.get("halftime_score_away"))
    if home_ht is None or away_ht is None:
        raise ValueError("Halftime score is required for total prediction.")

    prediction = compute_total_prediction(
        halftime_home=home_ht,
        halftime_away=away_ht,
        market_total=total_market.get("market_total"),
        config=cfg,
    )

    out_dir = output_dir or lgbm_dir / "betting_agent_stage1"
    execution_log = out_dir / "manifold_execution_log_latest.jsonl"
    candidate = decide_candidate(
        prediction=prediction,
        total_market=total_market,
        tracking=tracking,
        config=cfg,
        execution_log_path=execution_log,
    )
    record = build_record(
        tracking=tracking,
        prediction=prediction,
        candidate=candidate,
        config=cfg,
        total_market=total_market,
        timestamp_utc=timestamp_utc or utc_now_iso(),
    )
    record = maybe_place_order(record, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    dated_json = out_dir / f"halftime_total_prediction_{run_date}_{game_key}.json"
    latest_json = out_dir / "halftime_total_prediction_latest.json"
    for path in (dated_json, latest_json):
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    dated_csv = out_dir / f"manifold_halftime_order_candidates_{run_date}.csv"
    latest_csv = out_dir / "manifold_halftime_order_candidates_latest.csv"
    write_csv(dated_csv, [record])
    write_csv(latest_csv, [record])
    append_jsonl(execution_log, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict halftime final total and write Manifold candidate artifacts.")
    parser.add_argument("--lightgbm-dir", default=os.environ.get("LGBM_DIR", "LightGBM"))
    parser.add_argument("--tracking-json", default=None)
    parser.add_argument("--snapshots-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifold-market-id", default=None)
    args = parser.parse_args()

    config = TotalPredictionConfig.from_env()
    if args.manifold_market_id:
        config = TotalPredictionConfig(
            **{**config.__dict__, "manifold_market_id": args.manifold_market_id}
        )
    record = run_from_artifacts(
        lgbm_dir=Path(args.lightgbm_dir),
        tracking_path=Path(args.tracking_json) if args.tracking_json else None,
        snapshots_path=Path(args.snapshots_csv) if args.snapshots_csv else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        config=config,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

