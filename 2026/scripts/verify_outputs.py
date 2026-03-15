from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

EXPECTED_ACC_PREFIX = [
    "home_team",
    "away_team",
    "home_team_prob",
    "odds_1",
    "odds_2",
    "result",
    "date",
    "accuracy",
]

EXPECTED_ACC_REQUIRED = [
    "prob_iso",
    "prob_iso_oos_time",
    "prob_live_oos_proxy",
    "prob_live_safe_pre_clip",
    "prob_base",
    "prob_live_safe",
    "prob_used",
]

EXPECTED_SHORTLIST_HEADER = [
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


def _latest_by_pattern(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {pattern} under {path}")
    return files[-1]


def _assert_header(path: Path, *, expected_prefix: list[str], required_cols: list[str]) -> pd.DataFrame:
    header_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if "\n" in header_line or "\r" in header_line:
        raise AssertionError(f"Header in {path} is malformed (contains line breaks).")

    df = pd.read_csv(path)
    if list(df.columns)[: len(expected_prefix)] != expected_prefix:
        raise AssertionError(
            f"Header prefix mismatch for {path}.\nExpected: {expected_prefix}\nGot: {list(df.columns)}"
        )

    bad_cols = [c for c in df.columns if (" " in c or "\n" in c or "\r" in c)]
    if bad_cols:
        raise AssertionError(f"Non-canonical column names in {path}: {bad_cols}")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required columns in {path}: {missing}")
    return df


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    acc_dir = repo_root / "2026" / "output" / "LightGBM"
    kelly_dir = acc_dir / "Kelly"

    latest_acc = _latest_by_pattern(acc_dir, "combined_nba_predictions_acc_*.csv")
    latest_iso = _latest_by_pattern(kelly_dir, "combined_nba_predictions_iso_*.csv")
    latest_shortlist = _latest_by_pattern(kelly_dir, "bet_shortlist_*.csv")

    acc_df = _assert_header(latest_acc, expected_prefix=EXPECTED_ACC_PREFIX, required_cols=EXPECTED_ACC_REQUIRED)
    iso_df = _assert_header(latest_iso, expected_prefix=EXPECTED_ACC_PREFIX, required_cols=EXPECTED_ACC_REQUIRED)
    shortlist_df = _assert_header(latest_shortlist, expected_prefix=EXPECTED_SHORTLIST_HEADER, required_cols=EXPECTED_SHORTLIST_HEADER)

    if latest_acc.name.replace("acc", "iso") != latest_iso.name:
        raise AssertionError(
            f"Latest ISO filename does not match latest ACC date: {latest_acc.name} vs {latest_iso.name}"
        )

    ledger_path = acc_dir / "bet_log_flat_live.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Missing ledger file: {ledger_path}")

    ledger = pd.read_csv(ledger_path)
    if not ledger.empty and not shortlist_df.empty:
        ledger_dates = pd.to_datetime(ledger["date"], errors="coerce")
        shortlist_dates = pd.to_datetime(shortlist_df["game_date"], errors="coerce")
        if ledger_dates.max() < shortlist_dates.max():
            raise AssertionError(
                "Ledger last date is older than shortlist max date. "
                f"ledger_max={ledger_dates.max()} shortlist_max={shortlist_dates.max()}"
            )

    print("Verification passed:")
    print(f"  ACC: {latest_acc}")
    print(f"  ISO: {latest_iso}")
    print(f"  Shortlist: {latest_shortlist}")
    print(f"  Ledger: {ledger_path}")
    print(f"  Rows -> acc={len(acc_df)} iso={len(iso_df)} shortlist={len(shortlist_df)} ledger={len(ledger)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
