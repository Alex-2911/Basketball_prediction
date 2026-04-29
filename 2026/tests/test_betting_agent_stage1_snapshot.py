from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from betting_agent_stage1_snapshot import build_snapshot


def _write_fixture(root: Path) -> None:
    lgbm = root / "LightGBM"
    (lgbm / "Kelly").mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-04-29", "2026-04-29"],
            "home_team": ["DET", "LAL"],
            "away_team": ["ORL", "HOU"],
            "home_team_prob": [0.6837, 0.52],
            "odds 1": [1.25, 1.55],
            "odds 2": [4.10, 2.60],
        }
    ).to_csv(lgbm / "nba_games_predict_2026-04-29.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2026-04-29", "2026-04-29"],
            "home_team": ["DET", "LAL"],
            "away_team": ["ORL", "HOU"],
            "prob_used": [0.67, 0.51],
            "model_market_gap_flag": [False, False],
            "blocked_by": ["", ""],
        }
    ).to_csv(lgbm / "combined_nba_predictions_acc_2026-04-29.csv", index=False)

    pd.DataFrame({"home_team": ["DET", "LAL"], "home_win_rate": [0.61, 0.54]}).to_csv(
        lgbm / "home_win_rates_sorted_2026-04-29.csv", index=False
    )
    (lgbm / "strategy_params.json").write_text('{"ok": true}', encoding="utf-8")
    (lgbm / "local_matched_games_export_report_2026-04-29.txt").write_text(
        "source_rows=0\nrows_exported=0\nreason_if_nothing_written=skipped due to empty source dataframe\n",
        encoding="utf-8",
    )
    pd.DataFrame(columns=["game_date", "home_team", "away_team"]).to_csv(
        lgbm / "Kelly" / "bet_shortlist_2026-04-29.csv", index=False
    )


def test_stage1_snapshot_outputs(tmp_path: Path):
    _write_fixture(tmp_path)
    input_dir = tmp_path / "LightGBM"
    output_dir = input_dir / "betting_agent_stage1"

    snapshot, manifest = build_snapshot(input_dir, output_dir, "2026-04-29")
    assert manifest["canonical_rows_found"] == 0
    assert manifest["files_found"]["daily_predictions"]

    row = snapshot.iloc[0]
    assert abs(row["home_implied_raw"] - (1 / 1.25)) < 1e-9
    assert abs(row["away_implied_raw"] - (1 / 4.10)) < 1e-9
    assert bool(row["canonical_signal"]) is False
    assert row["canonical_reason"] == "official_shortlist_empty"
    assert row["canonical_export_status"] == "empty_source_dataframe"


def test_cli_writes_files(tmp_path: Path):
    _write_fixture(tmp_path)
    input_dir = tmp_path / "LightGBM"
    output_dir = input_dir / "betting_agent_stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot, manifest = build_snapshot(input_dir, output_dir, "2026-04-29")
    snapshot.to_csv(output_dir / "stage1_daily_snapshot_2026-04-29.csv", index=False)
    (output_dir / "stage1_daily_snapshot_2026-04-29.json").write_text(
        json.dumps({"target_date": "2026-04-29", "summary": {}, "games": [], "manifest": manifest}),
        encoding="utf-8",
    )
    (output_dir / "stage1_manifest_2026-04-29.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert (output_dir / "stage1_daily_snapshot_2026-04-29.csv").exists()
    assert (output_dir / "stage1_daily_snapshot_2026-04-29.json").exists()
    assert (output_dir / "stage1_manifest_2026-04-29.json").exists()
