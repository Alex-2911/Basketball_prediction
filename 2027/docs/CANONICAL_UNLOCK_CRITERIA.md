# Canonical Unlock Criteria

This document defines the evidence required before the 2027 pregame shell may move from the frozen `NO_BET` baseline to paper canonical `BET` signals.

It does not unlock anything. The current state remains:

- `baseline_engine_state`: `NO_BET`
- `canonical_params`: `null`
- `execution_enabled`: `false`
- `paper_only`: `true`
- Manifold/live order paths: unavailable

`PASS_FILTERS_ONLY`, watch labels, near misses, and discretionary observations are not canonical bets.

## Scope

An unlock can only mean enabling paper canonical `BET` signals inside local outputs. It must not enable staking, Manifold orders, live betting, scheduling, dashboard publishing, or automated execution.

Any later move from paper canonical signals to real orders requires a separate review, a separate config change, and explicit order-specific approval.

## Required Evidence

Before `canonical_params` can be set, the following evidence must exist in committed validation outputs or reviewed local reports.

1. Chronological replay

   A replay must use strictly chronological data. Training, calibration, proxy building, and rule selection must use only rows available before each evaluated game. Same-date leakage is not allowed. The report must include source hashes, cutoff dates, row counts, selected parameters, rejected parameters, and the exact reason canonical rules passed or failed.

2. Fresh 2027 paper sample

   The candidate generator must run on fresh 2027 pregame inputs supplied by an external, explicitly documented prediction artifact. Inputs must be timezone-qualified, unplayed, within the pregame window, structurally complete, and hash-recorded in manifests. Model-less, stale, played, postgame, or incomplete rows must continue to fail closed as `NO_BET`.

3. Minimum sample size

   Do not unlock on a handful of games. Require a reviewed sample large enough to include ordinary favorites, underdogs, low-odds cases, high-gap cases, and no-value skips. A future validation report should define the final numeric threshold before unlock review begins. Until that threshold is defined and met, the answer is `NO_BET`.

4. Stability windows

   The selected parameters must pass the preserved Script 11 stability standard: at least two profitable qualifying windows, each meeting minimum trade-count requirements. A single profitable window is not enough; that was the reason the June baseline stayed `NO_BET`.

5. Walk-forward evidence

   Walk-forward validation must pass with enough active splits, enough total test trades, and no obvious reliance on one isolated outlier period. The report must show split count, active split count, trades, ROI/profit, selected score, and failed alternatives.

6. Drawdown and loss controls

   Paper results must include maximum drawdown, longest losing streak, worst week, worst parameter window, and downside cases where the model looked attractive but lost. The unlock review must define acceptable drawdown limits before judging the results.

7. Reconciliation review

   Pending paper candidates must be reconciled against settled results in a separate output. Reconciliation must not rewrite original pregame decisions. The review must compare canonical decisions, watch labels, outcomes, and skipped candidates.

8. Boundary checks

   The shell must remain a consumer of supplied prediction rows. It must not fetch data, train a model, create predictions, backfill missing model outputs, or silently use archived 2026 data. `--history` use must remain explicit and hash-recorded.

9. Manual review

   A human review must confirm that the evidence supports paper canonical signals. The review must record the proposed `canonical_params`, the exact config change, and why the evidence is sufficient.

10. Explicit activation

   Unlock requires an explicit, separate config change from `canonical_params: null` to reviewed parameters. The commit must say that this enables paper canonical signals only. It must not change `execution_enabled`.

## Non-Requirements

The following are not sufficient by themselves:

- A `PASS_FILTERS_ONLY` label
- A watchlist hit
- One profitable historical window
- A profitable local replay without chronological controls
- An unreconciled paper candidate
- A manually attractive game
- A model probability that looks high against market price
- Any result produced by archived notebooks that has not been ported and validated

## Required Final State After Paper Unlock

If paper canonical signals are ever unlocked, these must still remain true:

- `execution_enabled` is `false`
- `paper_only` is `true`
- Manifold execution remains unavailable
- Orders are not submitted
- Every run writes input hashes, config hashes, and strategy evidence
- `BET` means paper signal only

## Current Decision

Current canonical model decision remains:

`NO_BET until validated historical replay, fresh 2027 paper evidence, reconciliation review, stability review, drawdown review, and explicit manual activation all exist.`
