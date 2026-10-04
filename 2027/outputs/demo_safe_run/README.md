# Demo Safe Run

This folder is a compact reference run for the NBA 2027 local pregame paper shell.

It contains one valid fresh-style input row and one invalid stale input row. The demo uses a fixed clock so the result is reproducible.

## Command

See `command.txt`.

## Expected Result

The valid row should remain a paper-only `NO_BET` because the canonical engine is frozen. It may show `PASS_FILTERS_ONLY`, but that is a diagnostic watch label, not a canonical bet.

Expected valid-row markers:

- `canonical_decision`: `NO_BET`
- `watch_label`: `PASS_FILTERS_ONLY`
- `blocked_by`: `ENGINE_NO_BET`
- `paper_only`: `true`
- `execution_enabled`: `false`

The invalid stale row should also be `NO_BET`, with `STALE_OR_FUTURE_INPUT` in `blocked_by`.

No stake is created. No order is created. Manifold/live execution is unavailable.

## Files

- `pregame_demo_input.json`: the demo input fixture
- `command.txt`: exact command used
- `20261020T120001000000Z/manifest.json`: run manifest with input/config hashes
- `20261020T120001000000Z/candidates.json`: structured candidate output
- `20261020T120001000000Z/betting_agent_training_cases_latest.jsonl`: paper observation JSONL
- `20261020T120001000000Z/strategy_evidence.json`: frozen baseline evidence

This demo does not prove profitability and does not unlock canonical betting.
