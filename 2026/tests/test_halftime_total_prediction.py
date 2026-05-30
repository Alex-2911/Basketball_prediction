from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_module(ROOT / "scripts" / "manifold_api_client.py", "manifold_api_client")
total_mod = load_module(ROOT / "scripts" / "halftime_total_prediction.py", "halftime_total_prediction")

TotalPredictionConfig = total_mod.TotalPredictionConfig
compute_total_prediction = total_mod.compute_total_prediction
decide_candidate = total_mod.decide_candidate
run_from_artifacts = total_mod.run_from_artifacts


class HalftimeTotalPredictionTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "dry_run": True,
            "live_order_enabled": False,
            "max_stake_mana": 10.0,
            "max_market_exposure_mana": 25.0,
            "min_edge_points": 5.0,
            "min_confidence": "MEDIUM",
            "min_price_decimal": 1.70,
            "baseline_total": 218.0,
            "live_pace_weight": 0.55,
            "market_regression_weight": 0.45,
            "allow_duplicate_order": False,
            "manifold_market_id": "mkt_total",
            "over_outcome": "YES",
            "under_outcome": "NO",
        }
        values.update(overrides)
        return TotalPredictionConfig(**values)

    def test_over_candidate_dry_run(self):
        cfg = self.config()
        pred = compute_total_prediction(halftime_home=62, halftime_away=58, market_total=220, config=cfg)
        candidate = decide_candidate(
            prediction=pred,
            total_market={"over_price": 1.85, "under_price": 1.85},
            tracking={"game_key": "g1"},
            config=cfg,
            execution_log_path=Path("/tmp/does-not-exist-total-log.jsonl"),
        )
        self.assertEqual(pred["direction"], "OVER")
        self.assertEqual(candidate["candidate_decision"], "BET_CANDIDATE")
        self.assertEqual(candidate["order_status"], "dry_run")

    def test_under_candidate_dry_run(self):
        cfg = self.config()
        pred = compute_total_prediction(halftime_home=45, halftime_away=44, market_total=224, config=cfg)
        candidate = decide_candidate(
            prediction=pred,
            total_market={"over_price": 1.9, "under_price": 1.82},
            tracking={"game_key": "g2"},
            config=cfg,
            execution_log_path=Path("/tmp/does-not-exist-total-log.jsonl"),
        )
        self.assertEqual(pred["direction"], "UNDER")
        self.assertEqual(candidate["candidate_decision"], "BET_CANDIDATE")
        self.assertEqual(candidate["manifold_outcome"], "NO")

    def test_no_edge(self):
        cfg = self.config()
        pred = compute_total_prediction(halftime_home=54, halftime_away=54, market_total=218, config=cfg)
        candidate = decide_candidate(
            prediction=pred,
            total_market={"over_price": 1.85, "under_price": 1.85},
            tracking={"game_key": "g3"},
            config=cfg,
            execution_log_path=Path("/tmp/does-not-exist-total-log.jsonl"),
        )
        self.assertEqual(pred["direction"], "NO_EDGE")
        self.assertEqual(candidate["candidate_decision"], "SKIP")
        self.assertIn("no_edge", candidate["blocked_by"])

    def test_blocked_by_price(self):
        cfg = self.config()
        pred = compute_total_prediction(halftime_home=62, halftime_away=58, market_total=220, config=cfg)
        candidate = decide_candidate(
            prediction=pred,
            total_market={"over_price": 1.55, "under_price": 1.85},
            tracking={"game_key": "g4"},
            config=cfg,
            execution_log_path=Path("/tmp/does-not-exist-total-log.jsonl"),
        )
        self.assertEqual(candidate["candidate_decision"], "SKIP")
        self.assertIn("price_not_acceptable", candidate["blocked_by"])

    def test_blocked_by_low_confidence(self):
        cfg = self.config(min_confidence="HIGH")
        pred = compute_total_prediction(halftime_home=59, halftime_away=57, market_total=220, config=cfg)
        candidate = decide_candidate(
            prediction=pred,
            total_market={"over_price": 1.9, "under_price": 1.85},
            tracking={"game_key": "g5"},
            config=cfg,
            execution_log_path=Path("/tmp/does-not-exist-total-log.jsonl"),
        )
        self.assertEqual(pred["confidence_bucket"], "MEDIUM")
        self.assertIn("low_confidence", candidate["blocked_by"])

    def test_blocked_by_duplicate_order(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "manifold_execution_log_latest.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "game_key": "g6",
                        "manifold_market_id": "mkt_total",
                        "direction": "OVER",
                        "order_status": "dry_run",
                    }
                )
                + "\n"
            )
            pred = compute_total_prediction(halftime_home=62, halftime_away=58, market_total=220, config=cfg)
            candidate = decide_candidate(
                prediction=pred,
                total_market={"over_price": 1.9, "under_price": 1.85},
                tracking={"game_key": "g6"},
                config=cfg,
                execution_log_path=log,
            )
            self.assertIn("duplicate_order", candidate["blocked_by"])

    def test_run_from_artifacts_writes_dry_run_outputs(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lgbm = root / "LightGBM"
            out = lgbm / "betting_agent_stage1"
            lgbm.mkdir()
            tracking = {
                "run_date": "2026-05-30",
                "game_date": "2026-05-30",
                "game_key": "2026-05-30_OKC_SAS",
                "home_team": "OKC",
                "away_team": "SAS",
                "snapshot_utc": "2026-05-31T01:25:00Z",
                "halftime_score_home": 62,
                "halftime_score_away": 58,
                "pregame_model_label": "NO",
                "stage2_candidate_type": "LOW_PRICE_NEGATIVE_EV",
            }
            (lgbm / "live_decision_tracking_latest.json").write_text(json.dumps(tracking))
            rows = [
                {
                    "game_key": "2026-05-30_OKC_SAS",
                    "snapshot_utc": "2026-05-31T01:25:00Z",
                    "market_type": "totals",
                    "total_line": "220",
                    "over_price": "1.85",
                    "under_price": "1.85",
                    "bookmaker_key": "testbook",
                }
            ]
            with (lgbm / "live_odds_snapshots_2026-05-30.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            record = run_from_artifacts(lgbm_dir=lgbm, output_dir=out, config=cfg, timestamp_utc="2026-05-31T01:26:00Z")
            self.assertEqual(record["order_status"], "dry_run")
            self.assertTrue((out / "halftime_total_prediction_latest.json").exists())
            self.assertTrue((out / "manifold_halftime_order_candidates_latest.csv").exists())
            self.assertTrue((out / "manifold_execution_log_latest.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
