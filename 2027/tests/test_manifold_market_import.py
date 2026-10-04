import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_manifold_markets_readonly.py"
SPEC = importlib.util.spec_from_file_location("manifold_market_import", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class ManifoldMarketImportTests(unittest.TestCase):
    def test_spread_market_is_not_script5_eligible(self):
        market_type, teams, reasons = module.classify_market("NBA: Will the New York Knicks cover the -5.5 point spread against the 76ers on Oct 20?")
        self.assertEqual(market_type, "SPREAD")
        self.assertIn("NYK", teams)
        self.assertIn("PHI", teams)
        self.assertIn("SPREAD_TERMS", reasons)

    def test_moneyline_like_market_still_does_not_create_signal(self):
        frame = module.normalize_markets([
            {"id": "m1", "question": "Will the Boston Celtics beat the Detroit Pistons on Oct 20?", "probability": 0.62, "isResolved": False}
        ], "testrun", "2026-10-03T00:00:00Z")
        self.assertEqual(frame.loc[0, "market_type"], "MONEYLINE")
        self.assertFalse(bool(frame.loc[0, "mapped_to_model"]))
        self.assertEqual(frame.loc[0, "canonical_decision"], "NO_BET")
        self.assertFalse(bool(frame.loc[0, "execution_enabled"]))

    def test_json_records_convert_empty_values_to_null(self):
        frame = module.normalize_markets([
            {"id": "m2", "question": "Will Steph Curry get traded before 2027?", "probability": 0.2, "isResolved": False}
        ], "testrun", "2026-10-03T00:00:00Z")
        records = module.json_records(frame)
        self.assertIsNone(records[0]["game_date"])
        self.assertIsNone(records[0]["line_value"])

    def test_output_contract_columns_are_present(self):
        frame = module.normalize_markets([
            {"id": "m3", "question": "NBA: Will the New York Knicks cover the -5.5 point spread against the 76ers on Oct 20?", "probability": 0.47, "isResolved": False}
        ], "testrun", "2026-10-03T00:00:00Z")
        required = {"run_id", "as_of_utc", "market_url", "market_title", "market_type", "league", "game_date", "home_team", "away_team", "line_type", "line_value", "side", "manifold_probability", "liquidity_if_available", "volume_if_available", "resolved_status", "mapped_to_model", "mapping_status", "reason_code", "canonical_decision", "research_label", "execution_enabled"}
        self.assertTrue(required.issubset(set(frame.columns)))
        self.assertEqual(frame.loc[0, "mapping_status"], "UNMAPPED_SPREAD_MARKET")
        self.assertEqual(frame.loc[0, "canonical_decision"], "NO_BET")

    def test_safety_flags_keep_adapter_outside_pipeline(self):
        text = SCRIPT.read_text()
        self.assertIn("outside_numbered_daily_pipeline", text)
        self.assertIn('"orders_created": False', text)
        self.assertIn('"canonical_unlock": False', text)
        self.assertNotIn("/v0/bet", text)
        self.assertIn("outputs" , text)


if __name__ == "__main__":
    unittest.main()
