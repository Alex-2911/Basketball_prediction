import argparse
import importlib.util
import sys
import tempfile
import unittest
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_lightgbm_predictions_2027.py"
SPEC = importlib.util.spec_from_file_location("script3_2027", SCRIPT)
script3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script3
SPEC.loader.exec_module(script3)


class Script3ModelPredictionTests(unittest.TestCase):
    def test_missing_2027_stats_fail_closed_after_valid_slate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            slate = root / "games_df_2026-10-03.csv"
            slate.write_text("home_team,away_team,game_date\nDET,BOS,2026-10-20\n")
            args = argparse.Namespace(
                date="2026-10-03",
                next_game_file=str(slate),
                next_game_dir=str(root),
                stats_dir=str(root / "stats"),
                prediction_artifact=None,
                copy_prediction_artifact=False,
                allow_baseline_2026=False,
                baseline_2026_stats=str(root / "missing.parquet"),
                fetch_odds=False,
                odds_api_key=None,
                allow_legacy_hardcoded_key=False,
                preferred_bookmakers="draftkings",
            )

            report = script3.build_report(args, script3.datetime(2026, 10, 3, tzinfo=script3.timezone.utc))

            self.assertEqual(report["status"], "SKIPPED")
            self.assertEqual(report["reason_code"], "WAITING_FOR_FIRST_2027_PLAYED_GAME_STATS")
            self.assertFalse(report["safety"]["training_executed"])
            self.assertFalse(report["safety"]["stats_fallback_to_2026_used"])
            self.assertFalse(report["safety"]["predictions_generated_by_script"])

    def test_rejects_bad_next_game_slate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            slate = root / "bad.csv"
            slate.write_text("home_team,away_team\nDET,BOS\n")

            frame, issues = script3.load_next_game_slate(slate)

            self.assertIsNone(frame)
            self.assertIn("MISSING_NEXT_GAME_COLUMN_game_date", issues)

    def test_validates_explicit_prediction_artifact_without_generating(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            slate = root / "games_df_2026-10-03.csv"
            artifact = root / "predictions.csv"
            slate.write_text("home_team,away_team,game_date\nDET,BOS,2026-10-20\n")
            artifact.write_text(
                "game_date,home_team,away_team,home_team_prob,prob_iso,home_win_rate\n"
                "2026-10-20,DET,BOS,0.50,0.50,0.50\n"
            )
            args = argparse.Namespace(
                date="2026-10-03",
                next_game_file=str(slate),
                next_game_dir=str(root),
                stats_dir=str(root / "stats"),
                prediction_artifact=str(artifact),
                copy_prediction_artifact=False,
                allow_baseline_2026=False,
                baseline_2026_stats=str(root / "missing.parquet"),
                fetch_odds=False,
                odds_api_key=None,
                allow_legacy_hardcoded_key=False,
                preferred_bookmakers="draftkings",
            )

            report = script3.build_report(args, script3.datetime(2026, 10, 3, tzinfo=script3.timezone.utc))

            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["reason_code"], "EXPLICIT_PREDICTION_ARTIFACT_VALIDATED")
            self.assertEqual(report["results"]["predictions_rows"], 1)
            self.assertFalse(report["safety"]["predictions_generated_by_script"])

    def test_baseline_2026_mode_creates_proxy_predictions(self):
        slate = ROOT / "data" / "raw" / "Gathering_Data" / "Next_Game" / "games_df_2026-10-03.csv"
        stats = ROOT / "data" / "processed" / "whole_statistics_2026.parquet"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = argparse.Namespace(
                date="2026-10-03",
                next_game_file=str(slate),
                next_game_dir=str(root),
                stats_dir=str(root / "stats"),
                prediction_artifact=None,
                copy_prediction_artifact=False,
                allow_baseline_2026=True,
                baseline_2026_stats=str(stats),
                fetch_odds=False,
                odds_api_key=None,
                allow_legacy_hardcoded_key=False,
                preferred_bookmakers="draftkings",
                output_dir=str(root / "out"),
            )

            report = script3.build_report(args, script3.datetime(2026, 10, 3, tzinfo=script3.timezone.utc))

            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["reason_code"], "BASELINE_2026_PREDICTIONS_WITH_ODDS_CREATED")
            self.assertEqual(report["results"]["predictions_rows"], 3)
            self.assertTrue(report["safety"]["training_executed"])
            self.assertTrue(report["safety"]["baseline_2026_predictions_are_preseason_proxy"])
            self.assertFalse(report["safety"]["odds_created"])

    def test_script3_fetches_odds_into_prediction_rows(self):
        predictions = pd.DataFrame([{"game_date": "2026-10-20", "home_team": "NYK", "away_team": "PHI", "home_team_prob": 0.8}])
        fake_events = [
            {
                "home_team": "New York Knicks",
                "away_team": "Philadelphia 76ers",
                "commence_time": "2026-10-20T23:00:00Z",
                "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": [{"key": "h2h", "outcomes": [{"name": "New York Knicks", "price": -185}, {"name": "Philadelphia 76ers", "price": 154}]}]}],
            }
        ]
        with unittest.mock.patch.object(script3, "fetch_api_json", return_value=fake_events):
            enriched, issues, meta = script3.attach_odds_if_enabled(
                predictions,
                argparse.Namespace(fetch_odds=True, odds_api_key="test", allow_legacy_hardcoded_key=False, preferred_bookmakers="draftkings"),
            )

        self.assertEqual(issues, [])
        self.assertEqual(meta["odds_ok_rows"], 1)
        self.assertAlmostEqual(enriched.loc[0, "odds_1"], 1.540541)
        self.assertAlmostEqual(enriched.loc[0, "odds_2"], 2.54)



if __name__ == "__main__":
    unittest.main()
