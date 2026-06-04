from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "betting_agent_stage1_snapshot.py"
SPEC = importlib.util.spec_from_file_location("betting_agent_stage1_snapshot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage1TargetDateTest(unittest.TestCase):
    def test_selects_earliest_upcoming_daily_slate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for date in ("2026-06-03", "2026-06-05", "2026-06-08"):
                (root / f"nba_games_predict_{date}.csv").write_text("home_team,away_team\n")

            selected = MODULE._infer_latest_date(root, run_date="2026-06-04")

            self.assertEqual(selected, "2026-06-05")

    def test_does_not_republish_a_past_slate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for date in ("2026-06-01", "2026-06-03"):
                (root / f"nba_games_predict_{date}.csv").write_text("home_team,away_team\n")

            selected = MODULE._infer_latest_date(root, run_date="2026-06-04")

            self.assertEqual(selected, "2026-06-04")


if __name__ == "__main__":
    unittest.main()
