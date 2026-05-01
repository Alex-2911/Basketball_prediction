import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "persist_script11_watchlist_history.py"
spec = importlib.util.spec_from_file_location("persist_script11_watchlist_history", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def test_upsert_and_classify_and_reconcile(tmp_path):
    out = tmp_path / "LightGBM"
    out.mkdir(parents=True)
    run_date = "2026-05-01"
    rows = pd.DataFrame([
        {"game_date": run_date, "home_team": "ORL", "away_team": "DET", "home_win_rate": 0.78, "odds_1": 2.36, "prob_base": 0.622, "prob_used": 0.421, "blocked_by": "MODEL_MARKET_GAP | Prob<0.55 | EV<=0.00", "EV_€_per_100": -2},
        {"game_date": run_date, "home_team": "HOU", "away_team": "LAL", "home_win_rate": 0.73, "odds_1": 1.59, "prob_used": 0.577, "blocked_by": "EV<=0.00", "EV_€_per_100": -1},
    ])
    combined = pd.DataFrame([
        {"game_date": run_date, "home_team": "ORL", "away_team": "DET", "result": "ORL"},
        {"game_date": run_date, "home_team": "HOU", "away_team": "LAL", "result": "LAL"},
    ])
    cpath = out / f"combined_nba_predictions_acc_{run_date}.csv"
    combined.to_csv(cpath, index=False)

    first = mod.persist_script11_watchlist_history(rows, out, run_date, combined_predictions_path=cpath, source="watchlist")
    second = mod.persist_script11_watchlist_history(rows, out, run_date, combined_predictions_path=cpath, source="watchlist")

    assert len(first) == 2
    assert len(second) == 2
    orl = second[second.home_team == "ORL"].iloc[0]
    hou = second[second.home_team == "HOU"].iloc[0]
    assert "MODEL_MARKET_GAP | Prob<0.55" in orl.blocked_by
    assert orl.stage2_candidate_type in {"RAW_MODEL_MARKET_GAP_HOME_DOG", "LIVE_WATCH_ONLY"}
    assert hou.stage2_candidate_type == "LOW_PRICE_NEGATIVE_EV"
    assert orl.home_team_won == 1
    assert hou.home_team_won == 0
    assert float(orl.pnl_ml_100) == (2.36 - 1.0) * 100.0
    assert float(hou.pnl_ml_100) == -100.0
