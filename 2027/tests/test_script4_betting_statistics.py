import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calculate_betting_statistics_2027.py"
SPEC = importlib.util.spec_from_file_location("script4_2027", SCRIPT)
script4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script4
SPEC.loader.exec_module(script4)


class Script4BettingStatisticsTests(unittest.TestCase):
    def test_prediction_rows_become_evidence_only_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            predictions = root / "predictions.csv"
            predictions.write_text(
                "game_date,home_team,away_team,home_team_prob,prob_iso,home_win_rate,model_source\n"
                "2026-10-20,DET,BOS,0.54,0.54,0.54,baseline_2026_lightgbm\n"
            )
            args = argparse.Namespace(predictions=str(predictions), output_dir=str(root / "out"))

            report = script4.build_report(args, script4.datetime(2026, 10, 3, tzinfo=script4.timezone.utc))

            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["reason_code"], "BETTING_STATISTICS_EVIDENCE_READY")
            self.assertEqual(report["results"]["rows"], 1)
            self.assertEqual(report["results"]["pending_outcomes"], 1)
            self.assertEqual(report["results"]["odds_missing_rows"], 1)
            self.assertFalse(report["safety"]["betting_signals_created"])
            self.assertFalse(report["safety"]["canonical_unlock"])

    def test_missing_predictions_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = argparse.Namespace(predictions=str(root / "missing.csv"), output_dir=str(root / "out"))

            report = script4.build_report(args, script4.datetime(2026, 10, 3, tzinfo=script4.timezone.utc))

            self.assertEqual(report["status"], "SKIPPED")
            self.assertEqual(report["reason_code"], "MISSING_SCRIPT3_PREDICTIONS")

    def test_preserves_script3_odds_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            predictions = root / "predictions.csv"
            predictions.write_text(
                "game_date,home_team,away_team,home_team_prob,prob_iso,home_win_rate,odds_1,odds_2,model_source\n"
                "2026-10-20,NYK,PHI,0.87,0.87,0.87,1.54,2.54,baseline_2026_lightgbm\n"
            )
            frame, issues = script4.load_predictions(predictions)
            self.assertEqual(issues, [])
            statistics = script4.build_statistics(frame)
            self.assertAlmostEqual(statistics.loc[0, "odds_1"], 1.54)
            self.assertAlmostEqual(statistics.loc[0, "odds_2"], 2.54)
            self.assertEqual(statistics.loc[0, "reason_code"], "ODDS_PRESENT_OUTCOME_PENDING")
            self.assertEqual(statistics.loc[0, "watch_label"], "PREDICTION_WITH_ODDS_OUTCOME_PENDING")



if __name__ == "__main__":
    unittest.main()
