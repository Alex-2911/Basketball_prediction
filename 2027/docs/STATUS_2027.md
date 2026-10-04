# NBA 2027 Status

Current project state: documented local pregame paper shell.

## Checkpoints

`nba2027-paper-shell-safe-v1 -> 4a6419c`

Baseline/input-contract safe shell.

`nba2027-paper-shell-safe-v2 -> 3a60b85`

Documented paper shell with model-pipeline boundary validation, canonical unlock criteria, safe runbook, and demo pack.

## Commit Chain

- `65f05ae` - safe baseline established
- `885164a` - 2026 replay harness validation evidence
- `6c8c185` - Script 7 normalization parity
- `9e49491` - outcome reconciliation evidence flow
- `4a6419c` - fresh pregame input contract
- `582402e` - model-pipeline boundary validation
- `684b485` - canonical unlock criteria documented
- `4df1244` - safe operating runbook
- `3a60b85` - safe demo run pack

## Frozen State

- Canonical engine: `NO_BET`
- `canonical_params`: `null`
- `execution_enabled`: `false`
- `paper_only`: `true`
- Manifold/live execution: unavailable
- `PASS_FILTERS_ONLY`: diagnostic only, not a bet

## Validated Behavior

- Fresh, complete, timezone-qualified pregame rows are accepted as paper observations.
- Invalid, stale, played, postgame, duplicate, incomplete, or model-less rows fail closed as `NO_BET` with reason codes.
- `nba-pregame` consumes supplied prediction rows; it does not create predictions.
- Archived 2026 history is used only when `--history` is explicitly supplied.
- Input, config, and optional history hashes are recorded in manifests.
- Outcome reconciliation writes separate evidence and does not rewrite pregame decisions.
- Demo pack exists at `outputs/demo_safe_run/`.

## Key Documents

- `docs/RUNBOOK_2027.md`
- `docs/CANONICAL_UNLOCK_CRITERIA.md`
- `docs/SETUP_VERIFICATION.json`
- `outputs/demo_safe_run/README.md`

## Pending Future Work

- Full model pipeline validation beyond boundary checks.
- Full Script 7 notebook parity, if exact notebook parity is still desired.
- Fresh 2027 paper sample collection from explicitly supplied prediction artifacts.
- Chronological paper review before any canonical unlock.
- Drawdown and loss-streak review before any canonical unlock.
- Human review of proposed `canonical_params` before any config change.
- Separate design and approval for any future live/execution path.

## Explicit Non-Goals For Current Shell

- No scheduler.
- No acquisition.
- No dashboard publishing.
- No data fetching.
- No model retraining.
- No prediction generation.
- No Manifold client.
- No live execution.
- No credential imports.
- No automatic config unlock.

Current decision:

`NO_BET until validated historical replay, fresh 2027 paper evidence, reconciliation review, stability review, drawdown review, and explicit manual activation all exist.`
