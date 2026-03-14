from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from live_oos_proxy import apply_live_oos_proxy, build_live_oos_proxy
from live_safety import UNDERDOG_CAP, apply_live_safety


def test_underdog_gap_guard_caps_or_blocks():
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

    assert bool(row["model_market_gap_flag"]) is True
    assert bool(row["live_underdog_upscale_guard_triggered"]) is True
    assert row["blocked_by"] == "MODEL_MARKET_GAP"
    assert float(row["prob_used"]) <= UNDERDOG_CAP


def test_live_oos_proxy_ready_and_fills_upcoming():
    rng = np.random.default_rng(42)
    n = 420
    p = rng.uniform(0.35, 0.75, size=n)
    y = rng.binomial(1, p)
    played = pd.DataFrame(
        {
            "home_team_prob": p,
            "prob_iso_oos_time": np.clip(p + rng.normal(0, 0.03, size=n), 0.01, 0.99),
            "win": y,
        }
    )

    proxy = build_live_oos_proxy(played, min_train_rows=300)
    assert proxy["ready"] is True

    upcoming = pd.DataFrame({"home_team_prob": [0.41, 0.52, 0.68]})
    out = apply_live_oos_proxy(upcoming, proxy)

    assert out["live_oos_proxy_ready"].all()
    assert out["prob_live_oos_proxy"].notna().all()
