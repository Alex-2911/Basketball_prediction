from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest


def _load_module(module_name: str, rel_path: str):
    base = Path(__file__).resolve().parents[1]
    mod_path = base / "src" / rel_path
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


script4 = _load_module("script4_2026", "4_calculate_betting_statistics_2026.py")
script5 = _load_module("script5_2026", "5_Isotonic_based_betting_strategy_2026.py")


def test_script4_upsert_preserves_history_and_prefers_resolved_rows():
    base = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "home_team": ["BOS", "NYK"],
            "away_team": ["LAL", "MIA"],
            "home_team_prob": [0.62, 0.51],
            "odds_1": [1.7, 1.9],
            "odds_2": [2.2, 1.95],
            "result": [np.nan, np.nan],
            "accuracy": [np.nan, np.nan],
        }
    )
    updates = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-03"],
            "home_team": ["NYK", "DEN"],
            "away_team": ["MIA", "PHX"],
            "home_team_prob": [0.51, 0.57],
            "odds_1": [1.9, 1.8],
            "odds_2": [1.95, 2.0],
            "result": ["NYK", np.nan],
            "accuracy": [1.0, np.nan],
        }
    )

    merged = script4.upsert_by_game_key(base, updates)

    assert len(merged) == 3
    row = merged[(merged["date"] == "2026-01-02") & (merged["home_team"] == "NYK")].iloc[0]
    assert row["result"] == "NYK"
    assert float(row["accuracy"]) == 1.0


def test_script5_classifies_insufficient_history_for_local_search():
    assert script5.classify_local_search_history(99, 100) == "insufficient_history"
    assert script5.classify_local_search_history(100, 100) == "ok"


def test_script5_shortlist_uses_effective_probability_threshold_floor():
    params = {
        "home_win_rate_threshold": 0.50,
        "odds_min": 1.20,
        "odds_max": 2.20,
        # Intentionally lower than PROB_CLIP_LO to verify threshold flooring.
        "prob_threshold": 0.10,
    }
    df = pd.DataFrame(
        {
            "result_raw": [0],
            "home_win_rate": [0.65],
            "closing_home_odds": [2.0],
            # Between 0.10 and PROB_CLIP_LO(0.35) -> should be filtered out.
            "prob_used": [0.20],
            "home_team": ["BOS"],
            "away_team": ["LAL"],
            "game_date": ["2026-02-01"],
        }
    )

    shortlist = script5.build_bet_shortlist(df, params, min_ev=-100.0)

    assert shortlist.empty


def test_script5_validate_structured_csv_rejects_duplicate_game_keys(tmp_path):
    csv_path = tmp_path / "combined.csv"
    pd.DataFrame(
        {
            "date": ["2026-02-01", "2026-02-01"],
            "home_team": ["BOS", "BOS"],
            "away_team": ["LAL", "LAL"],
            "prob_used": [0.61, 0.62],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(RuntimeError, match="duplicated key rows"):
        script5.validate_structured_csv(
            csv_path,
            required_cols=["home_team", "away_team", "date", "prob_used"],
            min_data_rows=1,
            unique_key_cols=["date", "home_team", "away_team"],
        )


def test_script4_normalize_prediction_dataframe_backfills_home_prob_from_pred_column():
    df = pd.DataFrame(
        {
            "date": ["2026-04-10"],
            "home_team": ["MIL"],
            "away_team": ["BRK"],
            "pred_home_win_proba": [0.481416],
            "closing_home_odds": [1.21],
            "closing_away_odds": [4.60],
            "result": ["MIL"],
        }
    )
    normalized = script4.normalize_prediction_dataframe(df)
    row = normalized.iloc[0]
    assert float(row["home_team_prob"]) == pytest.approx(0.481416)
    assert float(row["odds_1"]) == pytest.approx(1.21)
    assert float(row["odds_2"]) == pytest.approx(4.60)
