import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pregame_game_day_pipeline_2027.py"
SPEC = importlib.util.spec_from_file_location("pipeline_2027", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class GameDayPipelineTests(unittest.TestCase):
    def test_report_csv_extractor_reads_nested_results(self):
        path = Path(__file__).with_name("tmp_pipeline_report.json")
        try:
            path.write_text('{"results":{"output_csv":"/tmp/out.csv"}}')
            self.assertEqual(pipeline.output_csv_from_report(path), "/tmp/out.csv")
        finally:
            path.unlink(missing_ok=True)

    def test_source_declares_numbered_daily_pipeline(self):
        text = SCRIPT.read_text()
        self.assertIn("scripts/get_data_previous_game_day_2027.py", text)
        self.assertIn("scripts/get_data_next_game_day_2027.py", text)
        self.assertIn("scripts/run_lightgbm_predictions_2027.py", text)
        self.assertIn("scripts/calculate_betting_statistics_2027.py", text)
        self.assertIn("scripts/run_isotonic_strategy_2027.py", text)
        self.assertIn("odds_are_part_of_script3", text)
        self.assertNotIn("scripts/import_the_odds_api_2027.py", text)


if __name__ == "__main__":
    unittest.main()
