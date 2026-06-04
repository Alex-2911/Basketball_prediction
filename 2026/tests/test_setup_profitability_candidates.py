from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_profitability_scan.py"
SPEC = importlib.util.spec_from_file_location("setup_profitability_scan", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SetupProfitabilityCandidatesTest(unittest.TestCase):
    def test_excludes_unsettled_games_before_run_date(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-04",
                    "home_team_prob": 0.70,
                    "odds_1": 2.10,
                    "home_win_rate": 0.65,
                    "home_win": None,
                },
                {
                    "date": "2026-06-05",
                    "home_team_prob": 0.70,
                    "odds_1": 2.10,
                    "home_win_rate": 0.65,
                    "home_win": None,
                },
            ]
        )

        selected = frame[MODULE._current_candidate_mask(frame, "2026-06-04")]

        self.assertEqual(selected["date"].tolist(), ["2026-06-05"])


if __name__ == "__main__":
    unittest.main()
