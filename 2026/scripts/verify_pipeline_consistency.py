from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_probability_pipeline import (
    build_probability_chain_config,
    load_active_strategy_params,
    load_required_strategy_params,
    prepare_live_probability_columns,
)


KEY_COLS = ["prob_used", "model_market_gap", "model_market_gap_flag", "blocked_by"]
DEBUG_COLS = [
    "prob_base",
    "prob_live_safe_pre_clip",
    "prob_live_oos_proxy",
    "market_implied_p_raw",
    "market_implied_p_devig",
    "model_market_gap",
    "model_market_gap_flag",
    "live_underdog_upscale_guard_triggered",
    "live_shrink_triggered",
    "live_oos_proxy_ready",
    "live_oos_proxy_train_rows",
    "live_oos_proxy_bin_n",
    "live_oos_proxy_bin_winrate",
    "blocked_by",
]


def _sample_df() -> pd.DataFrame:
    n_played = 420
    n_upcoming = 6
    dates_played = pd.date_range("2025-01-01", periods=n_played, freq="D")
    dates_upcoming = pd.date_range("2026-03-10", periods=n_upcoming, freq="D")

    pred_played = np.clip(np.linspace(0.38, 0.78, n_played), 0.01, 0.99)
    win_played = (pred_played + np.sin(np.linspace(0, 8, n_played)) * 0.05 > 0.56).astype(int)

    played = pd.DataFrame(
        {
            "game_date": dates_played,
            "pred_home_win_proba": pred_played,
            "home_team_prob": pred_played,
            "home_team_won": win_played,
            "result_raw": win_played,
            "odds_1": np.where(pred_played > 0.55, 1.85, 2.45),
            "odds_2": np.where(pred_played > 0.55, 1.98, 1.65),
        }
    )

    pred_upcoming = np.array([0.41, 0.49, 0.55, 0.63, 0.67, 0.72])
    upcoming = pd.DataFrame(
        {
            "game_date": dates_upcoming,
            "pred_home_win_proba": pred_upcoming,
            "home_team_prob": pred_upcoming,
            "home_team_won": np.nan,
            "result_raw": 0,
            "odds_1": np.array([2.7, 2.2, 2.1, 1.95, 2.35, 2.55]),
            "odds_2": np.array([1.52, 1.7, 1.82, 1.96, 1.62, 1.51]),
        }
    )
    return pd.concat([played, upcoming], ignore_index=True)


def verify_probability_parity() -> None:
    base = _sample_df()
    config = build_probability_chain_config(
        date_col="game_date",
        result_col="home_team_won",
        result_raw_col="result_raw",
        pred_proba_col="pred_home_win_proba",
        today_date=pd.Timestamp("2026-03-10").date(),
        tomorrow_date=pd.Timestamp("2026-03-11").date(),
        compute_oos_chain=True,
    )
    script5_view = prepare_live_probability_columns(base, clip_lo=0.35, clip_hi=0.80, config=config)

    ledger_input = script5_view.copy()
    ledger_view = prepare_live_probability_columns(
        ledger_input,
        clip_lo=0.35,
        clip_hi=0.80,
        config=build_probability_chain_config(
            date_col="game_date",
            result_col="home_team_won",
            result_raw_col="result_raw",
            pred_proba_col="pred_home_win_proba",
            compute_oos_chain=False,
        ),
    )

    for c in KEY_COLS:
        a = script5_view[c]
        b = ledger_view[c]
        if c == "blocked_by":
            assert a.fillna("PASS").astype(str).equals(b.fillna("PASS").astype(str)), f"Mismatch in {c}"
        elif a.dtype == bool or b.dtype == bool:
            assert a.fillna(False).astype(bool).equals(b.fillna(False).astype(bool)), f"Mismatch in {c}"
        else:
            assert np.allclose(pd.to_numeric(a, errors="coerce").fillna(-9999), pd.to_numeric(b, errors="coerce").fillna(-9999), atol=1e-12), f"Mismatch in {c}"

    for c in DEBUG_COLS:
        assert c in script5_view.columns, f"Missing debug column {c}"
        assert c in ledger_view.columns, f"Missing debug column {c} in ledger"


def verify_strategy_param_loading() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "2026" / "output" / "LightGBM"
        out.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "params_used_type": "LOCAL",
            "params_used": {
                "home_win_rate_threshold": 0.61,
                "odds_min": 2.15,
                "odds_max": 3.05,
                "prob_threshold": 0.57,
            },
            "local_window_200": {"min_EV_applied": -4.0},
        }
        (out / "metrics_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

        params = load_active_strategy_params(root)
        required = load_required_strategy_params(root)
        assert params["home_win_rate_threshold"] == 0.61
        assert required["home_win_rate_threshold"] == 0.61
        assert params["odds_min"] == 2.15
        assert params["odds_max"] == 3.05
        assert params["prob_threshold"] == 0.57
        assert params["min_ev"] == -4.0




def verify_required_strategy_param_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "2026" / "output" / "LightGBM"
        out.mkdir(parents=True, exist_ok=True)
        (out / "strategy_params.txt").write_text("home_win_rate_threshold=0.6\nodds_min=2.1\n", encoding="utf-8")
        try:
            load_required_strategy_params(root)
        except RuntimeError:
            return
        raise AssertionError("Expected RuntimeError for missing required strategy thresholds")

def verify_strategy_param_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        params = load_active_strategy_params(root)
        assert params == {}, "Expected empty params when no files exist"


if __name__ == "__main__":
    verify_probability_parity()
    verify_strategy_param_loading()
    verify_required_strategy_param_failure()
    verify_strategy_param_fallback()
    print("verify_pipeline_consistency: OK")
