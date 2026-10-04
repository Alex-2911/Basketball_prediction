import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "get_data_previous_game_day_2027.py"
SPEC = importlib.util.spec_from_file_location("script1_2027", SCRIPT)
script1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script1
SPEC.loader.exec_module(script1)


class PreviousGameDay2027Tests(unittest.TestCase):
    def test_final_2026_snapshot_blocks_before_first_2027_stats(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            final = root / "nba_games_2026-06-14.csv"
            stats = root / "stats"
            final.write_text("date,home_team,away_team\n2026-06-14,OKC,IND\n")

            guard = script1.evaluate_first_game_guard(
                final_2026_snapshot=final,
                statistics_dir_2027=stats,
            )

            self.assertFalse(guard.allowed_to_collect)
            self.assertEqual(guard.status, "SKIPPED")
            self.assertEqual(guard.reason_code, "WAITING_FOR_FIRST_2027_PLAYED_GAME")

    def test_existing_2027_snapshot_passes_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            final = root / "nba_games_2026-06-14.csv"
            stats = root / "stats"
            stats.mkdir()
            final.write_text("date,home_team,away_team\n2026-06-14,OKC,IND\n")
            (stats / "nba_games_2027-2026-10-21.csv").write_text(
                "date,home_team,away_team\n2026-10-21,BOS,NYK\n"
            )

            guard = script1.evaluate_first_game_guard(
                final_2026_snapshot=final,
                statistics_dir_2027=stats,
            )

            self.assertTrue(guard.allowed_to_collect)
            self.assertEqual(guard.reason_code, "FIRST_2027_PLAYED_GAME_GUARD_PASSED")

    def test_safe_dry_run_does_not_create_predictions_or_execution(self):
        args = argparse.Namespace(
            date="2026-10-03",
            collect_date=None,
            allow_network=False,
        )

        report = script1.build_report(args, script1.datetime(2026, 10, 3, tzinfo=script1.timezone.utc))

        self.assertEqual(report["status"], "SKIPPED")
        self.assertEqual(report["reason_code"], "WAITING_FOR_FIRST_2027_PLAYED_GAME")
        self.assertFalse(report["safety"]["network_attempted"])
        self.assertFalse(report["safety"]["predictions_created"])
        self.assertFalse(report["safety"]["betting_signals_created"])
        self.assertFalse(report["safety"]["execution_path_created"])
        self.assertEqual(report["results"]["games_imported"], 0)


if __name__ == "__main__":
    unittest.main()
