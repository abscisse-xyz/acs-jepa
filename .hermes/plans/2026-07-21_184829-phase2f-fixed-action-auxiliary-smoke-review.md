# Stage 2F Plan Review Record

Plan: `.hermes/plans/2026-07-21_184829-phase2f-fixed-action-auxiliary-smoke.md`
Governing specification: `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2
Prerequisite: signed Stage 2E commit `0a59c11`

## Plan Review 1 — FAIL

Reviewer: independent subagent `deleg_c50ed9bd`

Blocking findings:

1. The inherited MLflow URI targeted unwritable `/home/awalga`; the overlay did not provide a writable fixed URI.
2. The offline producer did not require simulator/recorded source-state and terminal-state equality or recorded-action membership before replay.
3. Problem aliases and sparse trajectory problem indices were unspecified despite the corpus containing 24 parsed aliases and 12 trajectories using indices `1, 3, ..., 23`.
4. `applicability_pos_weight: 4.0` did not match the measured effective class ratio near 3.0.
5. Required applicability example/positive/negative counts were not emitted by the trainer.
6. Producer syntax, standalone-script test seam, eval output, and stale-output policy were underspecified.
7. Held-out training validation and all-corpus `acs-jepa eval` semantics were conflated.
8. Acceptance criteria did not name exact serialized metric keys.

Resolution in revised plan:

- fixed MLflow to a writable absolute SQLite URI under `/opt/data/workspace/acs-jepa-runs`;
- required exact source/terminal fact equality, applicable recorded actions, and contextual rejection;
- specified record-to-file identity, sparse problem indices, internal parsed names, and alias regression coverage;
- set class-balanced positive weight to `3.0` and recorded the deterministic train/validation/all-corpus census;
- added minimal detached applicability count instrumentation and RED/GREEN tests;
- supplied exact producer/train/eval commands, subprocess tests, distinct eval output, and no-stale-run preconditions;
- separated held-out `metrics/eval.jsonl` from all-corpus `eval/eval_summary.json` evidence;
- listed every literal required loss and count metric.

Implementation was blocked pending Plan Review 2 PASS.

## Plan Review 2 — PASS

Reviewer: independent subagent `deleg_ac1e305d`

All eight Plan Review 1 blockers are resolved. The reviewer verified the corrected plan against current code, config interpolation, checkpoint behavior, output-directory semantics, the fixed corpus (24 parsed entries, 12 trajectories, 345 transitions), the exact split (271 train, 38 held-out, 309 all-corpus windows), CUDA availability, offline-oracle isolation, TDD gates, literal metrics, and the Stage 2G boundary. TDD implementation and fixed execution are authorized.

## Implementation and Evidence Review 1 — FAIL

Reviewer: independent subagent `deleg_c8ee0f19`

The producer, fixed configuration, offline-only boundary, immutable-table coverage, all training/held-out/all-corpus metrics, checkpoint states, resolved configuration, evidence hashes, and quality gates were independently verified. One blocker remained: the fixed config-stack regression test asserted module construction and real train/eval behavior but did not directly prove every optional trainable parameter was optimizer-owned exactly once. Existing set-based coverage could not detect duplicate registration.

Correction:

- the fixed config-stack test now collects the optimizer's parameter IDs without deduplication;
- every contrastive-anchor, argument-head, and applicability-head parameter ID must occur exactly once;
- the optimizer's complete parameter-ID list must itself contain no duplicates;
- the corrected focused test, Ruff, `compileall`, and `git diff --check` pass.

A nonblocking review suggestion noted that MLflow's default artifact root was repository-local even though the tracking database was external. Those temporary untracked artifacts were removed and are not committed; the fixed external run evidence remains valid.

Signed commit was blocked pending corrected Implementation and Evidence Review 2 PASS.

## Implementation and Evidence Review 2 — PASS

Reviewer: independent subagent `deleg_82be7f58`

The reviewer confirmed the exactly-once optimizer regression now covers all 24 auxiliary parameter tensors (6 contrastive anchor, 7 argument head, 11 applicability head), each appearing once in an optimizer containing 139 unique parameter tensors. The producer/config tests, full core suite, scoped CLI suite, strict artifact coverage, finite metrics and positive counts, checkpoint states and hashes, Ruff, `compileall`, and `git diff --check` were independently reverified. Creation and verification of the SSH-signed Stage 2F commit is authorized; no push is authorized.
