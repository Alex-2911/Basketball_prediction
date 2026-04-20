from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    src_dir = str(REPO_ROOT / "2026" / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_boxscore_html_validation_handles_blocked_and_valid_variants():
    script1 = _load_module("script1_2026", "2026/src/1_get_data_previous_game_day_2026.py")

    blocked_html = "<html><body>Please verify you are a human</body></html>"
    valid_html = '<html><table id="line_score"></table><div>boxscore</div></html>'

    assert script1.file_is_valid_html_boxscore(blocked_html) is False
    assert script1.file_is_valid_html_boxscore(valid_html) is True


def test_find_games_for_next_day_rolls_across_month_files(tmp_path: Path):
    script2 = _load_module("script2_2026", "2026/src/2_get_data_next_game_day_2026.py")

    month_1 = tmp_path / "october.html"
    month_2 = tmp_path / "november.html"

    month_1.write_text(
        """
        <table id="schedule">
          <tr><th data-stat="date_game">Thu, Oct 30, 2025</th><td></td><td>Lakers</td><td></td><td>Celtics</td></tr>
        </table>
        """,
        encoding="utf-8",
    )

    month_2.write_text(
        """
        <table id="schedule">
          <tr><th data-stat="date_game">Sat, Nov 01, 2025</th><td></td><td>Knicks</td><td></td><td>Bulls</td></tr>
          <tr><th data-stat="date_game">Sat, Nov 01, 2025</th><td></td><td>Heat</td><td></td><td>Nets</td></tr>
          <tr><th data-stat="date_game">Sun, Nov 02, 2025</th><td></td><td>Suns</td><td></td><td>Warriors</td></tr>
        </table>
        """,
        encoding="utf-8",
    )

    rows = script2.find_games_for_next_day(
        target_date=datetime(2025, 10, 31),
        file_paths=[str(month_1), str(month_2)],
    )

    assert len(rows) == 2
    assert rows[0]["date"].strftime("%Y-%m-%d") == "2025-11-01"
    assert {(r["visitor_team"], r["home_team"]) for r in rows} == {
        ("Knicks", "Bulls"),
        ("Heat", "Nets"),
    }


def test_fetch_odds_handles_partial_bookmaker_payload(monkeypatch):
    os.environ.setdefault("ODDS_API_KEY", "dummy-test-key")
    script3 = _load_module("script3_2026", "2026/src/3_predict_games_hybrid_2026.py")

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "home_team": "Boston Celtics",
                    "away_team": "New York Knicks",
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Boston Celtics", "price": -140}
                                        # away outcome intentionally missing
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

    class _FakeSession:
        def get(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(script3, "get_session", lambda: _FakeSession())

    odds = script3.fetch_odds(
        pd.DataFrame([{"home_team": "BOS", "away_team": "NYK"}]),
        api_key="dummy-test-key",
        preferred=["draftkings"],
    )

    assert list(odds.columns) == ["home_team", "away_team", "odds 1", "odds 2"]
    assert len(odds) == 1
    assert pd.isna(odds.loc[0, "odds 2"])
