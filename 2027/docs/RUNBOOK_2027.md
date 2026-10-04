# NBA 2027 Runbook

This runbook describes the safe local procedure for the 2027 pregame paper shell.

Current state:

- Canonical engine: `NO_BET`
- `canonical_params`: `null`
- `execution_enabled`: `false`
- `paper_only`: `true`
- Manifold/live execution: unavailable

## 1. Prepare `data/raw/pregame.json`

Create `data/raw/pregame.json` from an explicitly supplied prediction artifact. The shell does not create predictions, fetch data, backfill missing values, or train a model.

Each row must contain:

- `game_date` in `YYYY-MM-DD` format
- `home_team`
- `away_team`
- `tipoff_utc`, timezone-qualified
- `as_of_utc`, timezone-qualified
- `is_played: false`
- `home_team_prob`
- `prob_iso`
- `home_win_rate`
- `odds_1`
- `odds_2`

Probabilities must be 0-1. Odds must be decimal and greater than 1. Input data must be no more than 24 hours old, and tipoff must be within the next 36 hours.

Invalid, stale, already-played, postgame, duplicate, or incomplete rows must be expected to return `NO_BET` with reason codes.

## 2. Run Pregame

From the 2027 workspace:

```sh
.venv/bin/nba-pregame --input data/raw/pregame.json
```

For reproducible validation only, a fixed clock may be supplied:

```sh
.venv/bin/nba-pregame --input data/raw/pregame.json --now 2026-10-20T12:00:01Z
```

Historical replay is opt-in only:

```sh
.venv/bin/nba-pregame \
  --input data/raw/pregame.json \
  --history data/processed/combined_predictions_2026.parquet
```

Do not use `--history` for normal fresh operation unless the run is explicitly a replay or validation run.

## 3. Interpret Decisions

`NO_BET` is the default and current canonical state.

`PASS_FILTERS_ONLY` means a row passed diagnostic watch filters. It is not a canonical bet. It must not be treated as permission to bet.

Watch labels such as `PREGAME_WATCH_ONLY`, `PREGAME_MARKET_GAP_WATCH`, `RAW_MODEL_MARKET_GAP_HOME_DOG`, `LOW_PRICE_NEGATIVE_EV`, or `NO_VALUE_SKIP` are diagnostic labels only.

Future canonical `BET` signals may only be enabled after the requirements in `docs/CANONICAL_UNLOCK_CRITERIA.md` are met and a separate explicit config change is reviewed and committed. Even then, `BET` means paper signal only unless a separate execution release is approved later.

## 4. Store Outputs

Every run writes a timestamped output folder containing:

- `candidates.json`
- `candidates.csv`
- `betting_agent_training_cases_latest.jsonl`
- `strategy_evidence.json`
- `manifest.json`

Keep these folders immutable. Do not edit generated outputs in place. If a run is important, commit the output folder under an evidence path such as:

```text
outputs/<validation_name>/<timestamp>/
```

The manifest records input, config, and optional history hashes. Use those hashes to verify what was evaluated.

## 5. Reconcile Outcomes

Reconciliation is a later evidence step. It must not rewrite pregame decisions.

Prepare a settled results CSV with:

- `date`
- `home_team`
- `away_team`
- `win`

Then run:

```sh
.venv/bin/python scripts/reconcile_cases.py \
  --cases-dir outputs/<run_folder> \
  --results path/to/settled_results.csv \
  --output outputs/<reconciliation_folder>/reconciled_cases.csv
```

The original JSONL case should remain `PENDING`. The reconciled output is a separate evidence file.

## 6. Never Do These In This Shell

Do not add or enable:

- schedulers
- automatic acquisition
- web scraping
- API fetching
- model retraining
- prediction generation
- dashboard publishing
- Manifold client code
- live execution
- order submission
- credential imports
- silent archived-history fallback
- config unlock without `docs/CANONICAL_UNLOCK_CRITERIA.md`

Do not treat `PASS_FILTERS_ONLY` as a bet.

Do not change `execution_enabled` from `false`.

Do not set `canonical_params` until the canonical unlock criteria are satisfied, reviewed, and activated by an explicit config commit.

## 7. Routine Validation

Before relying on a run, these checks should still pass:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_pregame_input_contract.py
.venv/bin/python scripts/validate_model_pipeline_boundary.py
```

These checks do not prove profitability. They only prove the shell is behaving safely and locally.
