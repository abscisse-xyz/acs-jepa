# Phase 2D2 Action-Supervision Dataset Plan Review

## Review 1 — FAIL

Reviewer: independent subagent `deleg_c4eedc4a`

Blocking findings:

1. A frozen config containing a copied mutable `dict` was not actually immutable and could change labels between reads/workers.
2. Complete offline oracle actions were not validated against the `ParsedProblem` selected by each key.
3. The `(problem_index, positive atoms)` key needed an explicit supported-state boundary because it cannot distinguish numeric, temporal/running-action, or other hidden simulator state discarded by corpus ingestion.

The plan is revised to specify:

- a pickleable immutable `ActionApplicabilityTable` backed only by canonical tuples/frozensets and logarithmic lookup;
- strict symbolic validation of every keyed state atom and stored applicable action against its keyed parsed problem;
- an explicit `positive_ground_atoms_closed_world_v1` semantics acknowledgment required whenever a table is supplied;
- strict scalar/key typing, mutation-rejection and pickle/spawn tests;
- source-transition-stable content-derived seeding rather than trajectory-index seeding;
- fixed-K and compatible action-schema requirements when supervision is enabled;
- include-static lookup invariance and cross-problem collision tests;
- full batched tensor/dtype assertions.

Coding remains blocked pending re-review.

## Review 2 — PASS

Reviewer: independent subagent `deleg_07eaaea6`

The corrected plan resolves all blockers. The reviewer confirmed the immutable/pickleable table design, keyed symbolic validation, explicit atom-only state semantics, strict scalar/key contracts, fixed-`K` batching, content-stable seeds, PyG/spawn viability, and trusted offline completeness boundary. Coding is authorized.

## Implementation Review 1 — FAIL

Reviewer: independent subagent `deleg_e9d77f31`

The tensor/data integration, tests, and Stage 2E compatibility were substantively correct, but two approved error-reporting contracts were missing:

1. Oracle symbolic-validation errors did not include the complete state key and offending atom/action.
2. Trace-positive contradiction errors included problem and step but not dataset trajectory index.

Both require regression tests, implementation fixes, full verification, and re-review.

## Implementation Review 2 — PASS

Reviewer: independent subagent `deleg_3971fbab`

Both error-context blockers were fixed and covered by regression tests. The reviewer reconfirmed the corrected implementation, focused and full verification, deterministic sampling, immutable offline labels, PyG batching, oracle isolation, standalone scope, and Stage 2E compatibility. No blocking findings remained; commit was authorized.
