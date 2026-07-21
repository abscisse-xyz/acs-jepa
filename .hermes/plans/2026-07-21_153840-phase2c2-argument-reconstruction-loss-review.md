# Phase 2C2 Argument-Reconstruction-Loss Plan Review

## Review 1 — FAIL

Reviewer: independent subagent `deleg_80552d24`

Blocking findings:

1. Finite input logits can still produce non-finite CE or margins for opposite-sign extreme values. The plan lacked a compute/output dtype policy, derived-output finite checks, and extreme/dtype TDD.
2. Target range validation ordering was not explicit enough to prevent unsafe gather/index behavior for `-1` or out-of-range active targets.
3. Rank-0 tensor output and normal/empty-branch dtype semantics were unspecified.

Required revision:

- define low-precision compute promotion and exact output dtypes;
- reject non-finite derived CE/diagnostics;
- validate active target range before gather/indexing;
- require rank-0 tensor outputs with identical dtype behavior in normal and empty branches;
- add dtype, extreme-finite, safe singleton, mask-sentinel, permutation, and exact-gradient tests.

Coding remains blocked pending re-review.

## Review 2 — FAIL

Reviewer: independent subagent `deleg_c9b92afe`

The first review's three blockers were resolved, but one test-contract inconsistency remained: float16 maximum finite logits are finite after float32 promotion, so they must not be rejected under a derived-finiteness-only policy.

The plan now requires:

- maximum finite float16 logits to return finite float32 CE and margin;
- corresponding bfloat16, float32, and float64 fixtures to raise only when derived outputs become non-finite;
- no artificial magnitude cutoff.

Coding remains blocked pending re-review.

## Review 3 — PASS

Reviewer: independent subagent `deleg_09e6a367`

The final re-review confirmed that the corrected extreme-value contract is internally coherent and numerically verified:

- maximum finite float16 logits produce finite promoted float32 CE/margin;
- corresponding bfloat16, float32, and float64 fixtures are rejected when derived outputs become non-finite;
- rejection uses no arbitrary magnitude threshold.

All earlier validation-ordering, dtype/output, singleton, empty, masking, gradient, and finite-output blockers are resolved. Coding is authorized.
