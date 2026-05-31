from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


schedule_mod = load_module(ROOT / "scripts" / "schedule_halftime_live_odds_jobs.py", "schedule_halftime")
capture_mod = load_module(ROOT / "scripts" / "capture_halftime_live_odds.py", "capture_halftime")


class HalftimeLiveOddsTests(unittest.TestCase):
    def test_no_value_skip_candidate_is_included_when_close_to_thresholds(self):
        row = {
            "stage2_candidate_type": "NO_VALUE_SKIP",
            "rules_passed": "3",
            "prob_used": "0.54",
            "home_win_rate": "0.50",
            "blocked_by": "Prob<0.55 | EV<=0",
        }

        self.assertTrue(schedule_mod.candidate_row(row))

    def test_scheduler_uses_watchlist_commence_time_and_preserves_game_date(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            lgbm = base / "LightGBM"
            lgbm.mkdir()
            with (lgbm / "script11_watchlist_history_latest.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "game_key",
                        "game_date",
                        "home_team",
                        "away_team",
                        "stage2_candidate_type",
                        "rules_passed",
                        "blocked_by",
                        "commence_time_utc",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "game_key": "2026-05-23_CLE_NYK",
                        "game_date": "2026-05-23",
                        "home_team": "CLE",
                        "away_team": "NYK",
                        "stage2_candidate_type": "NO_VALUE_SKIP",
                        "rules_passed": "3",
                        "blocked_by": "PASS",
                        "commence_time_utc": "2026-05-24T00:00:00Z",
                    }
                )

            scheduled = schedule_mod.build_schedule(base, lgbm, "2026-05-23")
            self.assertEqual(len(scheduled), 1)
            self.assertEqual(scheduled[0].game_date, "2026-05-23")
            self.assertEqual(scheduled[0].primary_capture_utc, "2026-05-24T01:15:00Z")
            self.assertEqual(scheduled[0].source, "watchlist")

    def test_nyk_cle_fixture_rejects_short_ml_but_flags_spread_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            lgbm = base / "LightGBM"
            lgbm.mkdir()

            with (lgbm / "script11_watchlist_history_latest.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "game_key",
                        "game_date",
                        "home_team",
                        "away_team",
                        "stage2_candidate_type",
                        "blocked_by",
                        "canonical_signal",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "game_key": "2026-05-23_CLE_NYK",
                        "game_date": "2026-05-23",
                        "home_team": "CLE",
                        "away_team": "NYK",
                        "stage2_candidate_type": "NO_VALUE_SKIP",
                        "blocked_by": "PASS",
                        "canonical_signal": "",
                    }
                )

            fixture = {
                "score": {
                    "score_status": "fetched",
                    "score_source": "fixture",
                    "period_status": "Halftime",
                    "game_clock": "0.0",
                    "halftime_score_home": 54,
                    "halftime_score_away": 60,
                    "halftime_margin_home": -6,
                },
                "odds_events": [
                    {
                        "home_team": "Cleveland Cavaliers",
                        "away_team": "New York Knicks",
                        "bookmakers": [
                            {
                                "key": "fixture_book",
                                "title": "Fixture Book",
                                "markets": [
                                    {
                                        "key": "h2h",
                                        "outcomes": [
                                            {"name": "New York Knicks", "price": 1.49},
                                            {"name": "Cleveland Cavaliers", "price": 2.50},
                                        ],
                                    },
                                    {
                                        "key": "spreads",
                                        "outcomes": [
                                            {"name": "New York Knicks", "price": 1.80, "point": -3.5},
                                            {"name": "Cleveland Cavaliers", "price": 1.83, "point": 3.5},
                                        ],
                                    },
                                    {
                                        "key": "totals",
                                        "outcomes": [
                                            {"name": "Over", "price": 1.91, "point": 220.5},
                                            {"name": "Under", "price": 1.91, "point": 220.5},
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
            fixture_path = base / "fixture.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "capture_halftime_live_odds.py"),
                    "--run-date",
                    "2026-05-24",
                    "--game-date",
                    "2026-05-23",
                    "--game-key",
                    "2026-05-23_CLE_NYK",
                    "--home-team",
                    "CLE",
                    "--away-team",
                    "NYK",
                    "--mode",
                    "once",
                    "--markets",
                    "h2h,spreads,totals",
                    "--snapshot-label",
                    "halftime",
                    "--source-root",
                    str(base),
                    "--lightgbm-dir",
                    str(lgbm),
                    "--fixture-json",
                    str(fixture_path),
                ],
                check=True,
            )

            tracking = json.loads((lgbm / "live_decision_tracking_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(tracking["game_date"], "2026-05-23")
            self.assertEqual(tracking["snapshot_label"], "halftime")
            self.assertEqual(tracking["stage2_candidate_type"], "NO_VALUE_SKIP")
            self.assertEqual(tracking["pregame_model_label"], "NO")
            self.assertEqual(tracking["live_away_ml"], 1.49)
            self.assertEqual(tracking["live_away_spread_line"], -3.5)
            self.assertEqual(tracking["live_total_line"], 220.5)
            self.assertEqual(tracking["live_classification"], "AWAY_LEAD_SPREAD_CANDIDATE")
            self.assertEqual(tracking["candidate_market"], "away_spread")

            with (lgbm / "live_odds_snapshots_2026-05-24.csv").open(encoding="utf-8") as f:
                markets = {row["market_type"] for row in csv.DictReader(f)}
            self.assertEqual(markets, {"h2h", "spreads", "totals"})

    def test_small_away_halftime_lead_candidate_threshold_is_240(self):
        decision = capture_mod.classify_live_decision(
            home_team="OKC",
            away_team="SAS",
            score={
                "score_status": "fetched",
                "halftime_margin_home": -3,
            },
            odds_rows=[
                {
                    "market_type": "h2h",
                    "away_ml": 2.45,
                }
            ],
        )

        self.assertEqual(decision["live_classification"], "AWAY_LIVE_ML_CANDIDATE")
        self.assertEqual(decision["candidate_market"], "away_ml")

    def test_small_away_halftime_lead_below_240_stays_watch_only(self):
        decision = capture_mod.classify_live_decision(
            home_team="OKC",
            away_team="SAS",
            score={
                "score_status": "fetched",
                "halftime_margin_home": -3,
            },
            odds_rows=[
                {
                    "market_type": "h2h",
                    "away_ml": 2.35,
                }
            ],
        )

        self.assertEqual(decision["live_classification"], "WATCH_ONLY_SMALL_AWAY_LEAD")
        self.assertEqual(decision["final_decision"], "NO_ACTION")


if __name__ == "__main__":
    unittest.main()
