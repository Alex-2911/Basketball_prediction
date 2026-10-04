import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_isotonic_strategy_2027.py"
SPEC = importlib.util.spec_from_file_location("script5_2027", SCRIPT)
script5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script5
SPEC.loader.exec_module(script5)


class Script5IsotonicStrategyTests(unittest.TestCase):
    def test_script4_rows_remain_no_bet_without_odds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "script4.csv"
            evidence.write_text(
                "game_date,home_team,away_team,home_team_prob,prob_iso,home_win_rate,odds_1,odds_2,model_source\n"
                "2026-10-20,DET,BOS,0.54,0.54,0.54,,,baseline_2026_lightgbm\n"
            )
            args = argparse.Namespace(input=str(evidence), output_dir=str(root / "out"))

            report = script5.build_report(args, script5.datetime(2026, 10, 3, tzinfo=script5.timezone.utc))

            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["reason_code"], "SCRIPT5_STRATEGY_GATE_READY")
            self.assertEqual(report["results"]["rows"], 1)
            self.assertEqual(report["results"]["no_bet_rows"], 1)
            self.assertEqual(report["results"]["odds_missing_rows"], 1)
            self.assertEqual(report["results"]["stake_total"], 0)
            self.assertFalse(report["safety"]["kelly_staking_applied"])
            self.assertFalse(report["safety"]["canonical_unlock"])
            self.assertFalse(report["safety"]["execution_path_created"])

    def test_missing_script4_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = argparse.Namespace(input=str(root / "missing.csv"), output_dir=str(root / "out"))

            report = script5.build_report(args, script5.datetime(2026, 10, 3, tzinfo=script5.timezone.utc))

            self.assertEqual(report["status"], "SKIPPED")
            self.assertEqual(report["reason_code"], "MISSING_SCRIPT4_EVIDENCE")
            self.assertFalse(report["safety"]["betting_signals_created"])

    def test_odds_present_still_does_not_unlock_engine(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "script4.csv"
            evidence.write_text(
                "game_date,home_team,away_team,home_team_prob,prob_iso,home_win_rate,odds_1,odds_2\n"
                "2026-10-20,NYK,PHI,0.87,0.87,0.61,1.5,2.7\n"
            )
            frame, issues = script5.load_script4_evidence(evidence)
            self.assertEqual(issues, [])

            strategy = script5.build_strategy_gate(frame)

            self.assertEqual(strategy.loc[0, "canonical_decision"], "NO_BET")
            self.assertEqual(strategy.loc[0, "watch_label"], "WATCH_RULES_NOT_PASSED")
            self.assertNotEqual(strategy.loc[0, "reason_code"], "ENGINE_NO_BET")
            self.assertEqual(strategy.loc[0, "stake"], 0)
            self.assertFalse(bool(strategy.loc[0, "kelly_applied"]))

    def test_odds_evidence_is_merged_before_strategy_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "script4.csv"
            odds_file = root / "odds.csv"
            evidence.write_text(
                "game_date,home_team,away_team,home_team_prob,prob_iso,home_win_rate,odds_1,odds_2\n"
                "2026-10-20,NYK,PHI,0.87,0.87,0.61,,\n"
            )
            odds_file.write_text(
                "game_date,home_team,away_team,odds_1,odds_2,bookmaker_key\n"
                "2026-10-20,NYK,PHI,1.54,2.54,draftkings\n"
            )
            args = argparse.Namespace(input=str(evidence), odds=str(odds_file), output_dir=str(root / "out"))

            report = script5.build_report(args, script5.datetime(2026, 10, 3, tzinfo=script5.timezone.utc))

            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["results"]["odds_missing_rows"], 0)
            self.assertEqual(report["results"]["no_bet_rows"], 1)
            self.assertFalse(report["safety"]["canonical_unlock"])


    def test_passed_filters_remain_noncanonical_when_engine_frozen(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "script4.csv"
            evidence.write_text(
                "game_date,home_team,away_team,home_team_prob,prob_iso,home_win_rate,odds_1,odds_2\n"
                "2026-10-20,NYK,PHI,0.59,0.59,0.80,3.40,1.35\n"
            )
            frame, issues = script5.load_script4_evidence(evidence)
            self.assertEqual(issues, [])

            frozen_config = {
                "baseline_engine_state": "NO_BET",
                "canonical_params": None,
                "watch_reference_params": {
                    "home_win_rate_threshold": 0.5,
                    "odds_min": 1.3,
                    "odds_max": 3.4,
                    "prob_threshold": 0.35,
                },
                "min_ev": 0.0,
            }
            strategy = script5.build_strategy_gate(frame, frozen_config)

            self.assertEqual(strategy.loc[0, "canonical_decision"], "NO_BET")
            self.assertEqual(strategy.loc[0, "watch_label"], "PASS_FILTERS_ONLY")
            self.assertEqual(strategy.loc[0, "reason_code"], "ENGINE_NO_BET")
            self.assertGreaterEqual(int(strategy.loc[0, "rules_passed"]), 4)
            self.assertEqual(strategy.loc[0, "stake"], 0)



if __name__ == "__main__":
    unittest.main()

class Script5MetricsSnapshotTests(unittest.TestCase):
    def test_metrics_snapshot_summarizes_strategy_without_execution(self):
        strategy = script5.pd.DataFrame([
            {"canonical_decision": "NO_BET", "watch_label": "WATCH_RULES_NOT_PASSED", "stake": 0, "odds_1": 1.8, "odds_2": 2.0, "home_team_prob": 0.54, "prob_used": 0.53, "EV_€_per_100": -2.0, "model_market_gap": 0.01, "rules_passed": 2, "blocked_by": "EV<=0.00"},
            {"canonical_decision": "NO_BET", "watch_label": "PASS_FILTERS_ONLY", "stake": 0, "odds_1": 2.1, "odds_2": 1.8, "home_team_prob": 0.60, "prob_used": 0.58, "EV_€_per_100": 5.0, "model_market_gap": 0.02, "rules_passed": 4, "blocked_by": "PASS"},
        ])
        metrics = script5.build_metrics_snapshot(strategy, {"engine_state": "NO_BET", "canonical_params_present": False, "watch_reference_params_present": True})
        pairs = {(row.section, row.metric): row.value for row in metrics.itertuples(index=False)}
        self.assertEqual(pairs[("decisions", "no_bet_rows")], 2)
        self.assertEqual(pairs[("decisions", "pass_filters_only_rows")], 1)
        self.assertEqual(pairs[("safety", "stake_total")], 0.0)
        self.assertEqual(pairs[("safety", "canonical_bet_rows")], 0)
