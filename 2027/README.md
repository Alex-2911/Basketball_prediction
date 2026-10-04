# NBA 2027 — local pregame paper workflow

Season: 2026–27. The original `1. NBA Script/2026` is the verified historical working source. Its next-season note confirms that the local notebooks remain master; `Basketball_prediction/2026/src/nba_utils_2026.py` supplies a shared dependency.

## Baseline preserved

The June 13 Script 11 run selected **NO_BET**, despite a profitable local candidate: only one of the required two stability windows was available. Its watch reference was HWR ≥ 0.50, decimal odds 1.30–3.40, probability ≥ 0.55, EV strictly > 0. The local candidate was HWR 0.50, odds 1.30–2.60, probability 0.50. The April saved strategy differs and remains separately preserved. Passing diagnostic filters does not activate a canonical bet.

The extracted Script 11 probability path, market-gap guard, walk-forward search, coverage and profit/stability functions retain their historical defaults. Script 7 case normalization and JSONL loader are ported as pure helpers. Original reconciliation notebooks, model notebooks, statistics, case JSONL and supporting code are retained under `data/baseline_2026` as reference material. Notebook outputs and embedded credential assignments are removed from reference copies; original and destination hashes are recorded. Archived code still contains old paths and must never be executed.

## Folders

- `data/baseline_2026/`: historical reference copies, never new-season input.
- `data/raw/`: incoming 2027 data; `data/processed/`: Parquet tables.
- `src/nba2027/`: offline candidate generator and isolated legacy rules.
- `configs/`: frozen defaults, provenance and execution gates.
- `scripts/`: migration, conversion and reconciliation tools.
- `notebooks/`: optional new analysis; historical notebooks stay archived.
- `outputs/`: immutable run folders with candidates, cases and evidence.
- `betting_agent/cases/`: new observations; `logs/`, `tests/`, `docs/`.

## Run

Create a Python 3.11 virtual environment here and install `pip install -e .`. Exact validated dependencies are recorded in `requirements.lock.txt`.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/nba-pregame --input data/raw/pregame.json
```

Each JSON input row requires `game_date` (YYYY-MM-DD), `home_team`, `away_team`, `tipoff_utc`, `as_of_utc` (both timezone-qualified timestamps), `is_played: false`, `home_team_prob`, `prob_iso`, `home_win_rate`, `odds_1` and `odds_2`. Probabilities use 0–1; odds are decimal. Data must be at most 24 hours old and tipoff within the next 36 hours. These are conservative new operational checks, separate from preserved strategy thresholds. Uncertain or invalid rows yield NO_BET with reasons. No fallback to combined historical files occurs.

With no history, the frozen June NO_BET state remains active. For paper replay of the original selection rules, add `--history data/processed/combined_predictions_2026.parquet`. Only settled rows from entirely earlier dates are used. Where archived OOS values are absent, calibration is rebuilt with the legacy 50-row minimum and 10-row folds; same-date outcomes are excluded as an additional temporal safeguard. This is recorded in strategy evidence and is not a claim of complete historical model parity. The original local search and stability gates choose canonical parameters, then the original OOS proxy calibrates upcoming probabilities. BET is a paper signal, never an order. `--now` exists solely for reproducible historical replay; it does not make stale data current.

Every run emits canonical BET/NO_BET, separate discretionary watch labels, reason codes, probability/EV diagnostics, Steadivus-compatible JSONL, source/config hashes and strategy evidence. New labels use PREGAME rather than suggesting live action. `PASS_FILTERS_ONLY` is discretionary. Outcomes remain pending until reconciliation.

## Storage and validation limits

CSV originals are retained. Parquet is the first local structured format. Historical source snapshots can repeat; keep their provenance rather than treating concatenated snapshots as independent games. Dataset size is local and modest after normalization; ClickHouse is not justified at this stage.

No acquisition job, scheduler, dashboard export, model retraining or live betting job is started. Model hyperparameters and feature-building source are preserved, but the legacy random train/test split is not paper-validation evidence. Fresh 2027 predictions must come from a separately validated ingestion/model run; this setup accepts predictions and checks them. Full model and full Script 7 notebook parity remain follow-up validation work. Successful rule tests do not prove profitability or that an individual bet is safe.

Manifold execution is disabled and unimplemented. Enabling config flags causes an error. A future execution release requires chronological paper validation, review of reconciliation results, explicit activation and order-specific confirmation.
