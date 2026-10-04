#!/usr/bin/env python3
"""Guarded 2027 rewrite of Script 5 isotonic strategy engine.

The archived Script 5 was the isotonic/Kelly daily betting engine.  For the
2027 paper shell this step is a strategy gate only: it can consume Script 4
prediction/statistics evidence, but it must not fetch odds, infer odds, settle
outcomes, create stakes, unlock canonical betting, or expose execution paths.

Until explicit odds and settled historical replay inputs are validated in a
separate phase, all rows remain NO_BET evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nba2027.legacy import script11


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "script5_isotonic_strategy"
DEFAULT_SCRIPT4_EVIDENCE = (
    ROOT
    / "outputs"
    / "script4_betting_statistics"
    / "20261020T120009000000Z"
    / "betting_statistics_evidence.csv"
)
REQUIRED_COLUMNS = {"game_date", "home_team", "away_team", "home_team_prob"}
OPTIONAL_ODDS_COLUMNS = {"odds_1", "odds_2"}
DEFAULT_BASELINE_CONFIG = ROOT / "configs" / "baseline.json"


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


def canonical_team(value: object) -> str:
    return str(value).strip().upper()


def merge_odds_evidence(evidence: pd.DataFrame, odds_path: Path | None) -> tuple[pd.DataFrame, list[str], str | None]:
    if odds_path is None:
        return evidence, [], None
    if not odds_path.exists():
        return evidence, ["MISSING_ODDS_EVIDENCE"], None
    odds = _read_table(odds_path)
    required = {"game_date", "home_team", "away_team", "odds_1", "odds_2"}
    missing = sorted(required - set(odds.columns))
    if missing:
        return evidence, [f"ODDS_EVIDENCE_MISSING_{col}" for col in missing], None
    odds = odds.copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    odds["home_team"] = odds["home_team"].map(canonical_team)
    odds["away_team"] = odds["away_team"].map(canonical_team)
    odds["odds_1"] = pd.to_numeric(odds["odds_1"], errors="coerce")
    odds["odds_2"] = pd.to_numeric(odds["odds_2"], errors="coerce")
    keep_cols = ["game_date", "home_team", "away_team", "odds_1", "odds_2"]
    for optional in ["home_odds_american", "away_odds_american", "bookmaker_key", "bookmaker_title", "api_commence_time", "odds_status"]:
        if optional in odds.columns:
            keep_cols.append(optional)
    odds = odds[keep_cols].drop_duplicates(["game_date", "home_team", "away_team"])

    work = evidence.copy()
    work["home_team"] = work["home_team"].map(canonical_team)
    work["away_team"] = work["away_team"].map(canonical_team)
    merged = work.drop(columns=[col for col in ["odds_1", "odds_2"] if col in work.columns]).merge(
        odds, on=["game_date", "home_team", "away_team"], how="left"
    )
    return merged, [], _sha256(odds_path)


def load_script4_evidence(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, ["MISSING_SCRIPT4_EVIDENCE"]

    frame = _read_table(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        return None, [f"SCRIPT4_EVIDENCE_MISSING_{col}" for col in missing]

    frame = frame.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["home_team"] = frame["home_team"].astype(str).str.strip()
    frame["away_team"] = frame["away_team"].astype(str).str.strip()
    frame["home_team_prob"] = pd.to_numeric(frame["home_team_prob"], errors="coerce")
    if "prob_iso" in frame.columns:
        frame["prob_iso"] = pd.to_numeric(frame["prob_iso"], errors="coerce")
    else:
        frame["prob_iso"] = frame["home_team_prob"]

    for col in OPTIONAL_ODDS_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["game_date", "home_team", "away_team", "home_team_prob"])
    if frame.empty:
        return None, ["EMPTY_OR_INVALID_SCRIPT4_EVIDENCE"]
    if not frame["home_team_prob"].between(0, 1).all():
        return None, ["INVALID_PROBABILITY"]

    return frame.drop_duplicates(["game_date", "home_team", "away_team"]).reset_index(drop=True), []



def load_baseline_config(path: Path = DEFAULT_BASELINE_CONFIG) -> dict[str, Any]:
    if not path.exists():
        return {
            "baseline_engine_state": "NO_BET",
            "canonical_params": None,
            "watch_reference_params": {
                "home_win_rate_threshold": 0.5,
                "odds_min": 1.3,
                "odds_max": 3.4,
                "prob_threshold": 0.55,
            },
            "min_ev": 0.0,
        }
    return json.loads(path.read_text())


def prepare_upcoming_for_script11(evidence: pd.DataFrame) -> pd.DataFrame:
    out = evidence.copy()
    out["date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["is_played"] = False
    for col in [
        "prob_iso_oos_time",
        "prob_live_oos_proxy",
        "prob_live_safe_pre_clip",
        "prob_base",
        "prob_used",
        "EV_€_per_100",
        "market_implied_p_raw",
        "market_implied_p_devig",
        "model_market_gap",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    defaults = {
        "live_oos_proxy_ready": False,
        "live_oos_proxy_used": False,
        "live_oos_proxy_train_rows": 0,
        "live_oos_proxy_bin_n": 0,
        "live_oos_proxy_bin_winrate": pd.NA,
        "blocked_by": "",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    return out


def build_script5_logic(evidence: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    engine_state = str(config.get("baseline_engine_state", "NO_BET"))
    canonical_params = config.get("canonical_params")
    watch_params = config.get("watch_reference_params") or config.get("last_local_candidate")
    min_ev = float(config.get("min_ev", 0.0))

    upcoming = prepare_upcoming_for_script11(evidence)
    if watch_params:
        watch = script11.build_near_miss_watchlist(upcoming, params_used=watch_params, min_ev=min_ev, top_n=max(len(upcoming), 10))
    else:
        watch = pd.DataFrame()

    if watch.empty:
        logic = upcoming.copy()
        logic["prob_used"] = pd.to_numeric(logic.get("prob_iso"), errors="coerce").fillna(pd.to_numeric(logic.get("home_team_prob"), errors="coerce"))
        logic["EV_€_per_100"] = pd.NA
        logic["rules_passed"] = 0
        logic["blocked_by"] = "NO_WATCH_REFERENCE_PARAMS"
    else:
        logic = upcoming.merge(
            watch,
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_script5"),
        )
        for col in watch.columns:
            alt = f"{col}_script5"
            if alt in logic.columns:
                logic[col] = logic[alt]
                logic = logic.drop(columns=[alt])

    if canonical_params and engine_state == "BET_ENABLED":
        canonical_ready = True
    else:
        canonical_ready = False

    odds_missing = logic[["odds_1", "odds_2"]].isna().any(axis=1)
    pass_filters = logic["blocked_by"].astype(str).eq("PASS")

    logic["strategy_step"] = "SCRIPT5_CORE_LOGIC"
    logic["strategy_variant"] = "script11_flat_shortlist_core_2027"
    logic["isotonic_applied"] = False
    logic["kelly_applied"] = False
    logic["stake"] = 0
    logic["canonical_decision"] = "NO_BET"
    logic["watch_label"] = "DIAGNOSTIC_NEAR_MISS"
    logic.loc[odds_missing, "watch_label"] = "PREDICTION_ONLY_NO_ODDS"
    logic.loc[pass_filters & ~canonical_ready, "watch_label"] = "PASS_FILTERS_ONLY"
    logic.loc[~odds_missing & ~pass_filters, "watch_label"] = "WATCH_RULES_NOT_PASSED"
    logic["reason_code"] = logic["blocked_by"].astype(str)
    logic.loc[odds_missing, "reason_code"] = "ODDS_MISSING_OUTCOME_PENDING"
    logic.loc[pass_filters & ~canonical_ready, "reason_code"] = "ENGINE_NO_BET"
    logic["execution_decision"] = "DISABLED"
    logic["profitability_claim"] = False

    summary = {
        "engine_state": engine_state,
        "canonical_params_present": canonical_params is not None,
        "watch_reference_params_present": watch_params is not None,
        "pass_filters_only_rows": int(((logic["watch_label"] == "PASS_FILTERS_ONLY")).sum()),
        "diagnostic_rows": int(len(logic)),
    }
    return logic, summary

def build_strategy_gate(evidence: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    config = config or load_baseline_config()
    out, _summary = build_script5_logic(evidence, config)

    preferred = [
        "game_date",
        "home_team",
        "away_team",
        "home_team_prob",
        "prob_iso",
        "home_win_rate",
        "odds_1",
        "odds_2",
        "prob_base",
        "prob_used",
        "EV_€_per_100",
        "market_implied_p_raw",
        "market_implied_p_devig",
        "model_market_gap",
        "model_market_gap_flag",
        "live_underdog_upscale_guard_triggered",
        "live_shrink_triggered",
        "rules_passed",
        "blocked_by",
        "margin_hw",
        "margin_odds",
        "margin_prob",
        "margin_ev",
        "canonical_decision",
        "watch_label",
        "reason_code",
        "stake",
        "isotonic_applied",
        "kelly_applied",
        "execution_decision",
        "profitability_claim",
        "strategy_step",
        "strategy_variant",
        "model_source",
        "model_issue",
    ]
    for col in preferred:
        if col not in out.columns:
            out[col] = pd.NA
    return out[preferred]


def build_metrics_snapshot(strategy: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def push(section: str, metric: str, value: Any) -> None:
        if hasattr(value, "item"):
            value = value.item()
        rows.append({"section": section, "metric": metric, "value": value})

    push("meta", "rows", int(len(strategy)))
    push("meta", "engine_state", summary.get("engine_state"))
    push("meta", "canonical_params_present", bool(summary.get("canonical_params_present")))
    push("meta", "watch_reference_params_present", bool(summary.get("watch_reference_params_present")))
    push("decisions", "no_bet_rows", int((strategy["canonical_decision"] == "NO_BET").sum()))
    push("decisions", "pass_filters_only_rows", int((strategy["watch_label"] == "PASS_FILTERS_ONLY").sum()))
    push("decisions", "watch_rules_not_passed_rows", int((strategy["watch_label"] == "WATCH_RULES_NOT_PASSED").sum()))
    push("decisions", "odds_missing_rows", int(strategy[["odds_1", "odds_2"]].isna().any(axis=1).sum()))
    push("safety", "stake_total", float(pd.to_numeric(strategy["stake"], errors="coerce").fillna(0).sum()))
    push("safety", "canonical_bet_rows", int((strategy["canonical_decision"] == "BET").sum()))

    for col, section in [
        ("home_team_prob", "probability"),
        ("prob_used", "probability"),
        ("EV_€_per_100", "ev"),
        ("model_market_gap", "market"),
        ("rules_passed", "rules"),
    ]:
        if col not in strategy.columns:
            continue
        series = pd.to_numeric(strategy[col], errors="coerce")
        push(section, f"{col}_count", int(series.notna().sum()))
        push(section, f"{col}_min", None if series.dropna().empty else float(series.min()))
        push(section, f"{col}_mean", None if series.dropna().empty else float(series.mean()))
        push(section, f"{col}_max", None if series.dropna().empty else float(series.max()))

    if "blocked_by" in strategy.columns:
        counts = strategy["blocked_by"].astype(str).value_counts(dropna=False)
        for reason, count in counts.items():
            push("blocked_by", str(reason), int(count))

    return pd.DataFrame(rows)


def build_report(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    evidence_path = Path(args.input)
    evidence, issues = load_script4_evidence(evidence_path)
    odds_path = Path(args.odds) if getattr(args, "odds", None) else None
    odds_sha256 = None
    if evidence is not None and odds_path is not None:
        evidence, odds_issues, odds_sha256 = merge_odds_evidence(evidence, odds_path)
        issues.extend(odds_issues)
    status = "SKIPPED" if issues else "READY"
    reason_code = issues[0] if issues else "SCRIPT5_STRATEGY_GATE_READY"

    rows = 0
    output_csv = None
    summary: dict[str, Any] = {
        "no_bet_rows": 0,
        "watch_rows": 0,
        "odds_missing_rows": 0,
        "stake_total": 0,
    }

    logic_summary: dict[str, Any] = {}
    if evidence is not None and not issues:
        config = load_baseline_config(Path(getattr(args, "config", DEFAULT_BASELINE_CONFIG)))
        strategy, logic_summary = build_script5_logic(evidence, config)
        strategy = build_strategy_gate(evidence, config)
        rows = int(len(strategy))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "strategy_gate_evidence.csv"
        strategy.to_csv(output_path, index=False)
        output_csv = str(output_path)
        metrics = build_metrics_snapshot(strategy, logic_summary)
        metrics_csv = output_dir / "metrics_snapshot.csv"
        metrics_json = output_dir / "metrics_snapshot.json"
        metrics.to_csv(metrics_csv, index=False)
        metrics.to_json(metrics_json, orient="records", indent=2)
        summary = {
            "no_bet_rows": int((strategy["canonical_decision"] == "NO_BET").sum()),
            "watch_rows": int(strategy["watch_label"].notna().sum()),
            "odds_missing_rows": int(strategy[["odds_1", "odds_2"]].isna().any(axis=1).sum()),
            "stake_total": float(pd.to_numeric(strategy["stake"], errors="coerce").fillna(0).sum()),
            "pass_filters_only_rows": int((strategy["watch_label"] == "PASS_FILTERS_ONLY").sum()),
            "engine_no_bet_rows": int((strategy["reason_code"] == "ENGINE_NO_BET").sum()),
            **logic_summary,
        }

    return {
        "script": "run_isotonic_strategy_2027",
        "legacy_script": "Script 5 isotonic-calibrated betting engine",
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "status": status,
        "reason_code": reason_code,
        "issues": issues,
        "inputs": {
            "script4_evidence": str(evidence_path),
            "script4_evidence_exists": evidence_path.exists(),
            "script4_evidence_sha256": _sha256(evidence_path),
            "odds_evidence": str(odds_path) if odds_path else None,
            "odds_evidence_sha256": odds_sha256,
        },
        "results": {
            "rows": rows,
            "output_csv": output_csv,
            "metrics_snapshot_csv": str(metrics_csv) if evidence is not None and not issues else None,
            "metrics_snapshot_json": str(metrics_json) if evidence is not None and not issues else None,
            **summary,
        },
        "safety": {
            "script5_core_logic_applied": True,
            "metrics_snapshot_only": True,
            "strategy_gate_only": True,
            "network_attempted": False,
            "odds_fetched": False,
            "odds_created": False,
            "outcomes_settled": False,
            "isotonic_live_calibration_applied": False,
            "kelly_staking_applied": False,
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
    parser = argparse.ArgumentParser(description="Guarded 2027 Script 5 isotonic strategy gate.")
    parser.add_argument("--input", default=str(DEFAULT_SCRIPT4_EVIDENCE), help="Script 4 evidence CSV or Parquet.")
    parser.add_argument("--config", default=str(DEFAULT_BASELINE_CONFIG), help="Frozen 2027 baseline config.")
    parser.add_argument("--odds", help="Optional The Odds API evidence CSV to merge before the Script 5 gate.")
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
