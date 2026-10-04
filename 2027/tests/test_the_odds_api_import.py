import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_the_odds_api_2027.py"
SPEC = importlib.util.spec_from_file_location("odds_2027", SCRIPT)
odds = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = odds
SPEC.loader.exec_module(odds)


def fake_api_json(api_key):
    return [
        {
            "home_team": "New York Knicks",
            "away_team": "Philadelphia 76ers",
            "commence_time": "2026-10-20T23:30:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "New York Knicks", "price": -150},
                                {"name": "Philadelphia 76ers", "price": 130},
                            ],
                        }
                    ],
                }
            ],
        }
    ]


class TheOddsApiImportTests(unittest.TestCase):
    def test_missing_api_key_fails_closed_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            games = root / "games.csv"
            games.write_text("game_date,home_team,away_team\n2026-10-20,NYK,PHI\n")
            args = argparse.Namespace(
                input=str(games),
                api_key=None,
                preferred_bookmakers="draftkings",
                output_dir=str(root / "out"),
                allow_legacy_hardcoded_key=False,
            )

            with patch.dict(odds.os.environ, {}, clear=True), patch.object(odds, "fetch_odds") as mocked:
                report = odds.build_report(args, odds.datetime(2026, 10, 3, tzinfo=odds.timezone.utc))

            self.assertEqual(report["status"], "SKIPPED")
            self.assertEqual(report["reason_code"], "NO_VERIFIED_ODDS_API_KEY")
            self.assertFalse(report["safety"]["network_attempted"])
            mocked.assert_not_called()

    def test_fetch_odds_maps_american_to_decimal(self):
        games = odds.pd.DataFrame([{"game_date": "2026-10-20", "home_team": "NYK", "away_team": "PHI"}])
        with patch.object(odds, "fetch_api_json", side_effect=fake_api_json):
            frame, meta = odds.fetch_odds(games, api_key="test", preferred=["draftkings"])

        self.assertEqual(meta["api_events_seen"], 1)
        self.assertEqual(frame.loc[0, "odds_status"], "OK")
        self.assertAlmostEqual(frame.loc[0, "odds_1"], 1.666667)
        self.assertAlmostEqual(frame.loc[0, "odds_2"], 2.3)
        self.assertEqual(frame.loc[0, "home_odds_american"], -150)
        self.assertEqual(frame.loc[0, "away_odds_american"], 130)

    def test_successful_api_import_remains_evidence_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            games = root / "games.csv"
            games.write_text("game_date,home_team,away_team\n2026-10-20,NYK,PHI\n")
            args = argparse.Namespace(
                input=str(games),
                api_key="test",
                preferred_bookmakers="draftkings",
                output_dir=str(root / "out"),
                allow_legacy_hardcoded_key=False,
            )

            with patch.object(odds, "fetch_api_json", side_effect=fake_api_json):
                report = odds.build_report(args, odds.datetime(2026, 10, 3, tzinfo=odds.timezone.utc))

            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["reason_code"], "ODDS_EVIDENCE_READY")
            self.assertEqual(report["results"]["odds_ok_rows"], 1)
            self.assertFalse(report["safety"]["betting_signals_created"])
            self.assertFalse(report["safety"]["canonical_unlock"])
            self.assertFalse(report["safety"]["execution_path_created"])

    def test_legacy_hardcoded_key_can_be_resolved_without_printing_value(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "legacy.py"
            source.write_text("API_KEY = \"secret-test-key\"\n")

            value = odds.read_legacy_hardcoded_key(source)

            self.assertEqual(value, "secret-test-key")



if __name__ == "__main__":
    unittest.main()
