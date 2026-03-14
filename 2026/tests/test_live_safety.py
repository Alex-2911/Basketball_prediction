from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_calibration import apply_live_oos_proxy, build_live_oos_proxy
from live_safety import apply_live_safety


class LiveSafetyTests(unittest.TestCase):
    def test_underdog_gap_guard_caps_or_blocks(self) -> None:
        df = pd.DataFrame(
            {
                "home_team_prob": [0.684],
                "prob_live_oos_proxy": [0.684],
                "odds_1": [2.30],
                "odds_2": [1.65],
            }
        )

        out = apply_live_safety(df, live_oos_proxy_ready=True)
        row = out.iloc[0]

        self.assertTrue(bool(row["model_market_gap_flag"]))
        self.assertTrue(bool(row["live_underdog_upscale_guard_triggered"]))
        self.assertEqual(row["blocked_by"], "MODEL_MARKET_GAP")
        self.assertLessEqual(float(row["prob_used"]), 0.55)

    def test_live_oos_proxy_ready_and_fills_upcoming(self) -> None:
        rows: list[dict] = []
        for i in range(360):
            p = 0.38 + (i % 25) * 0.015
            rows.append(
                {
                    "game_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "home_team": f"H{i}",
                    "away_team": f"A{i}",
                    "home_team_prob": p,
                    "prob_iso_oos_time": p + 0.01,
                    "win": int(i % 2 == 0),
                }
            )

        played = pd.DataFrame(rows)
        proxy = build_live_oos_proxy(
            played,
            prob_source_cols=["prob_iso_oos_time", "home_team_prob"],
            target_col="win",
            n_bins=20,
            min_train_rows=300,
            min_bin_n=10,
            date_col="game_date",
            home_col="home_team",
            away_col="away_team",
        )

        self.assertTrue(proxy.ready)
        self.assertGreaterEqual(proxy.train_rows, 300)

        upcoming = pd.DataFrame(
            {
                "home_team_prob": [0.52, 0.61],
                "home_team": ["ORL", "LAL"],
                "away_team": ["DET", "DEN"],
                "game_date": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-01")],
            }
        )
        out = apply_live_oos_proxy(upcoming, proxy, in_col="home_team_prob")

        self.assertTrue(out["prob_live_oos_proxy"].notna().all())
        self.assertTrue(out["live_oos_proxy_ready"].eq(True).all())
        self.assertTrue(out["live_oos_proxy_train_rows"].eq(proxy.train_rows).all())
        self.assertTrue((out["live_oos_proxy_bin_n"] > 0).all())


if __name__ == "__main__":
    unittest.main()
