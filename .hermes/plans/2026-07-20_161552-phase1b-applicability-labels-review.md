# Phase 1B Applicability Label Plan Review

Plan reviewed: `.hermes/plans/2026-07-20_161552-phase1b-applicability-labels.md`

Verdict: PASS.

Independent review source:

- `deleg_020f7d2a`, completed 2026-07-20.

Blocking issues:

- None.

Spec/research compliance:

- PASS: Scope is limited to offline `(state, grounded_action, applicability_label)` example construction plus JSON diagnostics.
- PASS: Learned `ApplicabilityHead`, BCE loss, trainer wiring, planner integration, decoder scoring, and production behavior changes are deferred.
- PASS: Positives are trace/reference actions.
- PASS: Negatives come from Phase 1A `sample_action_negatives(...)`.
- PASS: Applicability labels are computed from `applicable_keys(engine)` membership, not candidate `apply_action` mutation.
- PASS: `applicable_actions()` remains an offline oracle/label source only, not production action generation.
- PASS: Plan includes TDD, RED/GREEN steps, smoke-run gating, separate implementation review, and post-review commit gate.

Non-blocking suggestions folded into the plan before coding:

1. Call `sample_action_negatives(...)` without `applicability_fn`; apply labels only through `build_applicability_examples(...)` from the injected applicable-key set.
2. Use `load_corpus([args.dataset_dir], strict=True)` to match existing diagnostic scripts.
3. Deduplicate examples by action key globally for the label batch.
4. Include a test that `--no-oracle-labels` emits explicit unknown counts and `applicable: null` for trace and negative examples.

Conclusion: Coding may proceed.
