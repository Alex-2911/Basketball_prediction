import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "betting_agent_stage1_snapshot.py"


def test_missing_daily_file_writes_data_incomplete(tmp_path):
    inp = tmp_path / "LightGBM"
    out = inp / "betting_agent_stage1"
    inp.mkdir(parents=True)
    pd.DataFrame([{"game_date": "2026-05-01", "home_team": "A", "away_team": "B"}]).to_csv(
        inp / "combined_nba_predictions_acc_2026-05-01.csv", index=False
    )
    subprocess.check_call([sys.executable, str(SCRIPT), "--input-dir", str(inp), "--output-dir", str(out), "--target-date", "2026-05-01"])
    manifest = json.loads((out / "stage1_manifest_2026-05-01.json").read_text())
    snap = pd.read_csv(out / "stage1_daily_snapshot_2026-05-01.csv")
    assert manifest["status"] == "data_incomplete_daily_predictions_missing"
    assert manifest["daily_games_found"] == 0
    assert snap.empty


def test_daily_file_preferred(tmp_path):
    inp = tmp_path / "LightGBM"
    out = inp / "betting_agent_stage1"
    inp.mkdir(parents=True)
    daily = pd.DataFrame([
        {"game_date": "2026-05-01", "home_team": "A", "away_team": "B", "odds_1": 2.1, "odds_2": 1.8, "home_team_prob": 0.55},
        {"game_date": "2026-05-01", "home_team": "C", "away_team": "D", "odds_1": 2.2, "odds_2": 1.7, "home_team_prob": 0.52},
        {"game_date": "2026-05-01", "home_team": "E", "away_team": "F", "odds_1": 1.9, "odds_2": 1.9, "home_team_prob": 0.51},
    ])
    daily.to_csv(inp / "nba_games_predict_2026-05-01.csv", index=False)
    subprocess.check_call([sys.executable, str(SCRIPT), "--input-dir", str(inp), "--output-dir", str(out), "--target-date", "2026-05-01"])
    snap = pd.read_csv(out / "stage1_daily_snapshot_2026-05-01.csv")
    assert len(snap) == 3
