# Branch D Candidate Rollout Plan Reviews

Plan: `.hermes/plans/2026-08-04_092228-branchd-candidate-rollout.md`

## Review 1 — FAIL

Reviewer: `deleg_f868f6c0`

Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_093221_806250.txt`

Blockers:

1. `ActionEncoder.context_steps=None` was incorrectly accepted despite meaning all prior actions; repeated single-step calls reset its GRU. The plan must accept an exposed context only when exactly `1`, rejecting both `None` and values greater than one.
2. Predictor validation did not preserve complete state shapes or exact metadata tensor/storage identities. Equal cloned `object_ids` or `object_batch` must fail.
3. Exact initial-state and per-step control/state validation was incomplete, including later control-width changes and object metadata alignment.
4. A test incorrectly required changed candidates to produce changed controls/predictions, contradicting the accepted transition-equivalence evidence.
5. Fake-module recurrence coverage was redundant while real `ActionEncoder(context_steps=1)` plus real-predictor compatibility, candidate-expanded object batching, and gradient flow were absent.
6. README correction needed to describe `ActionDecoder` as heuristic nearby-candidate search, not a reliable inverse.

Corrections applied:

- Closed temporal action context to exactly `1` when the attribute exists; added separate `None` and `2` RED cases.
- Required complete initial/predicted state shapes and exact `object_ids`/`object_batch` storage identities.
- Added explicit initial-state, metadata, control-width, predictor-type, and predictor-shape failures.
- Replaced distinguishability assertions with exact candidate-slice/order spies that allow transition equivalence.
- Kept one manual recurrence test and added real action-encoder/predictor, `B=2`, `H>=2`, expanded-object, gradient integration coverage.
- Required safe README wording for the decoder's heuristic, non-inverse behavior.

Review 1 authorized no implementation.

## Review 2 — FAIL

Reviewer: `deleg_eab874be`

Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_093659_377521.txt`

Review 2 confirmed every Review 1 blocker was closed but found one lifecycle blocker: Task 3 introduced signature, output, candidate-order, and real-module tests only after Task 1 had already implemented the behavior, so those tests could pass immediately and violate strict behavioral RED/GREEN.

Correction applied:

- Moved the public signature/output assertions and the real `ActionEncoder(context_steps=1)` plus residual-predictor gradient integration test into Task 1 before any production implementation.
- Added one exact two-node initial RED command whose expected failure is the absent public API, followed by the exact GREEN rerun after minimal implementation.
- Kept all Task 2 contract tests incremental and behavior-specific against an importable Task 1 implementation.
- Reduced Task 3 to documentation only; it adds no after-the-fact tests.

Review 2 authorized no implementation.

## Review 3 — PASS

Reviewer: `deleg_d23d721b`

Review 3 confirmed all prior blockers are closed and authorized implementation, tests, README, and the RED/GREEN record within the plan's exact boundaries. It explicitly did not authorize a commit, push, planner/CLI integration, decoder/CEM changes, tuning, Phase 1, simulator, candidate generation/search, or applicability work.

Authorized sequencing:

1. Add both Task 1 tests before production symbols and record the single missing-public-API RED.
2. Implement the minimal recurrence/public contract and record Task 1 GREEN.
3. Add every Task 2 contract behavior individually, observe a behavior-specific RED against the importable implementation, then minimally fix it.
4. Update documentation only after all behavior is GREEN.

## Implementation Review 1 — FAIL

Reviewer: `deleg_c7f01fe1`

Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_101914_697093.txt`

Blockers:

1. Exact metadata object identity did not detect in-place mutation of the original `object_ids` or `object_batch`; immutable value snapshots and later/final-step mutation regressions are required in addition to `is` checks.
2. The initial implementation did not satisfy the approved one-behavior-at-a-time Task 2 lifecycle. The log admitted that the planned object-row mismatch test passed immediately, and several sections recorded grouped/template commands rather than every exact RED command/failure/fix/GREEN.

The reviewer confirmed all other model behavior, real-module gradients, scope boundaries, README claims, full tests, and static checks passed. Commit was not authorized.

Required correction is an honest implementation restart from the signed base for tracked production/test/docs changes while preserving plan history. The restart must retain the prior attempt as rejected history, introduce Task 1 tests before production, and then add each Task 2 behavior individually with exact RED and GREEN records. No mutation-only reconstruction may be substituted for genuine RED.

## Implementation Review 2 — FAIL

Reviewer: `deleg_ae7e590f`

Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_113851_232115.txt`

The reviewer accepted the honest restart lifecycle and all Task 1–2.33 evidence, including genuine object-row RED, metadata snapshots, exact identity, late mutation checks, gradients, scope, docs, and regressions. Two technical blockers remained:

1. Predictor in-place resize of the original graph/object latent tensors changed the dimensions later used as the supposed baseline, allowing final-step shape mutation. Authoritative pre-recurrence dimension snapshots and focused late in-place graph/object resize regressions are required.
2. A non-Tensor action-encoder result reached `.ndim` and raised uncontrolled `AttributeError`; a genuine RED and deterministic control-type contract are required.

Commit remained unauthorized.

## Implementation Review 3 — PASS

Reviewer: `deleg_5add14c0`

Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_120743_648758.txt`

The reviewer independently confirmed:

- all Review 1 and Review 2 blockers are closed;
- Accepted Restart Attempt 2 contains genuine sequential Task 1–2.36 RED/GREEN evidence;
- final-step in-place metadata and latent-shape mutation, clone, replacement, and reordering are rejected;
- non-Tensor controls fail deterministically;
- context, repeated-candidate, recurrence, ordering, source causality, gradients, public API/export, docs, and scope all conform;
- candidate rollout `38 passed`, graph components plus planner `86 passed`, full core `334 passed`, scoped CLI `256 passed`, and static checks passed.

Review 3 authorized exactly one SSH-signed local commit after parent verification. It did not authorize push, planner/CLI integration, tuning, or Phase 1 work.
