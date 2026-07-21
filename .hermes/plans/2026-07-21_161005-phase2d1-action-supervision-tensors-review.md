# Phase 2D1 Action-Supervision Tensor Builder Plan Review

## Review 1 — FAIL

Reviewer: independent subagent `deleg_cae300f3`

Blocking findings:

1. Empty typed-domain handling contradicted itself: random-other sampling skipped unusable schemas while validation rejected the whole problem.
2. Random-category round-robin and attempt-consumption semantics were underspecified, so deterministic output and fairness were not reproducible from the plan.
3. Verification referenced nonexistent `test_graph_dataset.py` instead of `test_pddl_graph.py`.

The plan is revised to:

- validate only the reference action's typed bindings and skip unrelated unusable schemas;
- define one random draw per category visit, including exact budget consumption, schema selection, duplicate/rejection handling, and exhaustion;
- fix the verification path;
- define changed-role semantics across different arities;
- state bounded-attempt shortfall versus proven grounding-space exhaustion;
- document exact-type/no-subtype limitations and later loss-integration empty-mask rules;
- strengthen category-ID and no-enumeration tests.

Coding remains blocked pending re-review.

## Review 2 — PASS

Reviewer: independent subagent `deleg_7c8b68de`

The revised plan resolves all blockers. The reviewer confirmed deterministic and bounded round-robin semantics, unrelated unusable-schema handling, changed-role behavior, exact-type limitations, stable category contracts, corrected verification, no-enumeration coverage, and viable Stage 2D2/2E boundaries. Coding is authorized.
