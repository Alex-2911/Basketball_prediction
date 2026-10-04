import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "get_data_next_game_day_2027.py"
SPEC = importlib.util.spec_from_file_location("script2_2027", SCRIPT)
script2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script2
SPEC.loader.exec_module(script2)


class NextGameDay2027Tests(unittest.TestCase):
    def test_missing_schedule_fails_closed(self):
        result = script2.find_next_game_day(Path("/tmp/does-not-exist-schedule.csv"), script2.date(2026, 10, 3))

        self.assertEqual(result.status, "SKIPPED")
        self.assertEqual(result.reason_code, "NO_VERIFIED_2027_SCHEDULE_METADATA")
        self.assertEqual(result.rows, [])

    def test_local_schedule_selects_first_available_day(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "schedule.csv"
            path.write_text(
                "game_date,home_team,away_team,tipoff_utc\n"
                "2026-10-21,Boston Celtics,New York Knicks,2026-10-21T23:00:00Z\n"
                "2026-10-22,Los Angeles Lakers,Golden State Warriors,2026-10-22T02:00:00Z\n"
            )

            result = script2.find_next_game_day(path, script2.date(2026, 10, 3))

            self.assertEqual(result.status, "READY")
            self.assertEqual(result.reason_code, "NEXT_GAME_DAY_METADATA_READY")
            self.assertEqual(len(result.rows), 1)
            self.assertEqual(result.rows[0]["home_team"], "BOS")
            self.assertEqual(result.rows[0]["away_team"], "NYK")

    def test_schedule_with_prediction_or_odds_columns_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad_schedule.csv"
            path.write_text(
                "game_date,home_team,away_team,home_team_prob\n"
                "2026-10-21,BOS,NYK,0.7\n"
            )

            result = script2.find_next_game_day(path, script2.date(2026, 10, 3))

            self.assertEqual(result.status, "SKIPPED")
            self.assertEqual(result.reason_code, "SCHEDULE_CONTAINS_MODEL_OR_ODDS_COLUMNS")

    def test_safe_dry_run_creates_no_model_or_execution_paths(self):
        args = argparse.Namespace(
            date="2026-10-03",
            schedule_metadata="/tmp/does-not-exist-schedule.csv",
            write_csv=False,
            next_game_dir="/tmp/unused-next-game",
            backup_dir="/tmp/unused-next-game-backup",
        )

        report = script2.build_report(args, script2.datetime(2026, 10, 3, tzinfo=script2.timezone.utc))

        self.assertEqual(report["status"], "SKIPPED")
        self.assertEqual(report["results"]["games_found"], 0)
        self.assertFalse(report["safety"]["network_attempted"])
        self.assertFalse(report["safety"]["predictions_created"])
        self.assertFalse(report["safety"]["odds_created"])
        self.assertFalse(report["safety"]["betting_signals_created"])
        self.assertFalse(report["safety"]["execution_path_created"])


if __name__ == "__main__":
    unittest.main()
