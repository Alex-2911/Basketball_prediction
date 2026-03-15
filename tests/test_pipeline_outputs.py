from __future__ import annotations

from pathlib import Path
import pandas as pd

PROBABILITY_COLUMNS = [
    "prob_iso",
    "prob_iso_oos_time",
    "prob_live_oos_proxy",
    "prob_live_safe_pre_clip",
    "prob_base",
    "prob_live_safe",
    "prob_used",
]

SHORTLIST_HEADER = [
    "game_date",
    "home_team",
    "away_team",
    "home_team_prob",
    "prob_iso",
    "prob_iso_oos_time",
    "prob_live_oos_proxy",
    "prob_live_safe_pre_clip",
    "prob_base",
    "prob_used",
    "odds_1",
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
    "home_win_rate",
    "EV_€_per_100",
]


def _latest(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise AssertionError(f"Missing files for pattern {pattern} under {path}")
    return files[-1]


def _assert_single_line_header(path: Path) -> None:
    first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if "\n" in first_line or "\r" in first_line:
        raise AssertionError(f"Header is malformed in {path}")


def test_pipeline_outputs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    acc_dir = repo_root / "2026" / "output" / "LightGBM"
    kelly_dir = acc_dir / "Kelly"

    latest_acc = _latest(acc_dir, "combined_nba_predictions_acc_*.csv")
    latest_iso = _latest(kelly_dir, "combined_nba_predictions_iso_*.csv")
    latest_shortlist = _latest(kelly_dir, "bet_shortlist_*.csv")

    _assert_single_line_header(latest_acc)
    _assert_single_line_header(latest_iso)
    _assert_single_line_header(latest_shortlist)

    acc_df = pd.read_csv(latest_acc)
    for col in PROBABILITY_COLUMNS:
        if col not in acc_df.columns:
            raise AssertionError(f"ACC missing required probability column: {col}")
    played_mask = acc_df["result"].notna() & ~acc_df["result"].astype(str).str.strip().isin(["", "0", "nan", "None"])
    if played_mask.any() and acc_df.loc[played_mask, PROBABILITY_COLUMNS].notna().all(axis=1).mean() <= 0:
        raise AssertionError("ACC probability columns are blank on played rows")

    iso_df = pd.read_csv(latest_iso)
    missing_iso = [col for col in PROBABILITY_COLUMNS if col not in iso_df.columns]
    if missing_iso:
        raise AssertionError(f"ISO missing required probability columns: {missing_iso}")

    if latest_acc.name.replace("acc", "iso") != latest_iso.name:
        raise AssertionError(
            f"Latest ACC/ISO date mismatch: {latest_acc.name} vs {latest_iso.name}"
        )

    shortlist_df = pd.read_csv(latest_shortlist)
    if list(shortlist_df.columns) != SHORTLIST_HEADER:
        raise AssertionError(
            f"Shortlist header mismatch.\nExpected: {SHORTLIST_HEADER}\nGot: {list(shortlist_df.columns)}"
        )

    ledger = pd.read_csv(acc_dir / "bet_log_flat_live.csv")
    ledger_dates = pd.to_datetime(ledger["date"], errors="coerce")
    if ledger_dates.max() <= pd.Timestamp("2026-01-31"):
        raise AssertionError("Ledger does not contain any entries after January 2026")


if __name__ == "__main__":
    test_pipeline_outputs()
    print("Pipeline output verification test passed.")
