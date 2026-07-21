# Phase 2 Roadmap / Stage 2A Plan Review

## Review 1 — FAIL

Reviewer: independent subagent `deleg_914eb3ff`

Blocking findings:

1. The proposed `D_a == 1` contract reused `CovarianceLoss`, whose empty off-diagonal mean is NaN. The plan needed an explicit finite behavior.
2. Existing helper zeros for `N == 1` are detached constants, conflicting with the proposed autograd contract.
3. The TDD plan did not include a deterministic numerical oracle for the std and covariance components.
4. An exactly collapsed input cannot demonstrate a useful nonzero gradient; collapse and gradient tests needed separate fixtures.

Accepted aspects:

- Phase decomposition preserves the unchanged Phase 2 requirements and user-mandated gates.
- Stage 2A is appropriately narrow.
- The plan correctly avoids claiming a paired-view invariance objective.
- Rank-3 temporal flattening and SIGReg deferral align with the specification.

Revisions made before re-review:

- Define graph-connected zero boundaries for one sample and one feature without changing shared `GraphVCLoss` behavior.
- Require floating-point inputs.
- Rename component outputs to `std_penalty` and `covariance_penalty`.
- Add deterministic numerical-oracle coverage.
- Separate collapsed-loss and non-degenerate nonzero-gradient tests.
- Document repository covariance normalization.
- Require future applicability-coefficient reconciliation and effective sample-count reporting.

Coding remains blocked until the revised plan receives PASS.

## Review 2 — PASS

Reviewer: independent subagent `deleg_451fca23`

All four blocking findings from Review 1 were resolved. The reviewer confirmed:

- the `D_a == 1` covariance boundary is finite and graph-connected without changing shared graph VC behavior;
- one-sample outputs retain a valid autograd path;
- numerical-oracle and separate non-degenerate-gradient tests are sufficient;
- the roadmap preserves the unchanged Phase 2 specification and Stage 2A remains narrowly scoped.

Accepted non-blocking refinements for implementation:

- include at least one manually expected scalar in numerical-oracle coverage;
- explicitly assert graph connectivity for the one-feature covariance zero;
- distinguish later Phase 2 representation acceptance from decoder/planner criteria deferred to Phases 3 and 4.

Verdict: coding may begin for Stage 2A.
