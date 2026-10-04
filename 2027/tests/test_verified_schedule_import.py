import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_verified_schedule_metadata.py"
SPEC = importlib.util.spec_from_file_location("verified_schedule_import", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


class VerifiedScheduleImportTests(unittest.TestCase):
    def test_imports_explicit_local_schedule_metadata_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "basketball_prediction_schedule.csv"
            output = root / "out"
            source.write_text(
                "game_date,home_team,away_team,tipoff_utc\n"
                "2026-10-21,Boston Celtics,New York Knicks,2026-10-21T23:00:00Z\n"
                "2026-10-22,Los Angeles Lakers,Golden State Warriors,2026-10-22T02:00:00Z\n"
            )

            result = adapter.import_schedule_metadata(source, output, "2026-27")

            self.assertEqual(result["status"], "OK")
            self.assertEqual(result["games_found"], 2)
            self.assertTrue(result["teams_normalized"])
            self.assertEqual(result["duplicates"], 0)
            self.assertTrue(result["timezone_checked"])
            self.assertTrue((output / "schedule_metadata.csv").exists())

    def test_rejects_prediction_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "bad_schedule.csv"
            source.write_text(
                "game_date,home_team,away_team,tipoff_utc,home_team_prob\n"
                "2026-10-21,BOS,NYK,2026-10-21T23:00:00Z,0.7\n"
            )

            result = adapter.import_schedule_metadata(source, root / "out", "2026-27")

            self.assertEqual(result["status"], "SKIPPED")
            self.assertIn("SOURCE_CONTAINS_MODEL_OR_ODDS_COLUMNS", result["issues"])
            self.assertIsNone(result["imported_file"])

    def test_rejects_missing_tipoff(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "bad_schedule.csv"
            source.write_text("game_date,home_team,away_team\n2026-10-21,BOS,NYK\n")

            result = adapter.import_schedule_metadata(source, root / "out", "2026-27")

            self.assertEqual(result["status"], "SKIPPED")
            self.assertIn("MISSING_TIPOFF_TIME", result["issues"])

    def test_rejects_unnormalized_team_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "bad_team_schedule.csv"
            source.write_text(
                "game_date,home_team,away_team,tipoff_utc\n"
                "2026-10-21,Unknown Team,BOS,2026-10-21T23:00:00Z\n"
            )

            result = adapter.import_schedule_metadata(source, root / "out", "2026-27")

            self.assertEqual(result["status"], "SKIPPED")
            self.assertIn("TEAM_NORMALIZATION_FAILED", result["issues"])

    def test_rejects_out_of_season_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "old_schedule.csv"
            source.write_text(
                "game_date,home_team,away_team,tipoff_utc\n"
                "2026-04-12,BOS,NYK,2026-04-12T19:00:00Z\n"
            )

            result = adapter.import_schedule_metadata(source, root / "out", "2026-27")

            self.assertEqual(result["status"], "SKIPPED")
            self.assertIn("OUT_OF_SEASON_ROWS", result["issues"])

    def test_report_records_no_model_or_execution_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "schedule.csv"
            out = root / "out"
            source.write_text(
                "game_date,home_team,away_team,tipoff_utc\n"
                "2026-10-21,BOS,NYK,2026-10-21T23:00:00Z\n"
            )
            args = argparse.Namespace(source_file=str(source), output_dir=str(out), season="2026-27")

            report = adapter.build_report(args, adapter.datetime(2026, 10, 3, tzinfo=adapter.timezone.utc))

            self.assertEqual(report["status"], "OK")
            self.assertFalse(report["safety"]["network_attempted"])
            self.assertFalse(report["safety"]["predictions_created"])
            self.assertFalse(report["safety"]["odds_created"])
            self.assertFalse(report["safety"]["betting_signals_created"])
            self.assertFalse(report["safety"]["canonical_unlock"])
            self.assertFalse(report["safety"]["execution_path_created"])


if __name__ == "__main__":
    unittest.main()
