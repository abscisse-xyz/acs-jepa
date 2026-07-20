# Phase 1A Negative Sampler Plan Review

Plan reviewed: `.hermes/plans/2026-07-20_155945-phase1a-negative-sampler.md`

Verdict: PASS.

Independent review source:

- `deleg_1742cbfe`, completed 2026-07-20.

Blocking issues:

- None.

Spec/research compliance:

- PASS: Scope is limited to reusable negative sampler plus JSON diagnostic output.
- PASS: No learned probes, heads, losses, planner changes, decoder changes, or training behavior changes are planned.
- PASS: Required negative categories are included: `one_arg_substitution`, `role_swap`, `random_same_schema`, and `random_other_schema`.
- PASS: Type-validity is enforced via `ActionDecodingSpace` and tensorization tests.
- PASS: Determinism is covered via local `random.Random(seed)` and same-seed tests.
- PASS: Applicability labels are optional offline diagnostic labels only.
- PASS: Plan avoids production use of `applicable_actions()` as an action generator.
- PASS: TDD flow is present, followed by separate implementation review.

Non-blocking suggestions folded into the plan before coding:

1. Clarify `changed_roles` semantics for `random_other_schema`, especially when arity differs.
2. In the diagnostic script, prefer membership in `applicable_keys(engine)`; do not mutate a shared replay engine with `apply_action`.
3. Add an explicit non-mutation test for true `GroundAction` and parsed problem.
4. Add a duplicate-argument role-swap test to ensure swaps producing the original tuple are skipped.

Conclusion: Coding may proceed.
