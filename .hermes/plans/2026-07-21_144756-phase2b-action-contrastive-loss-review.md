# Phase 2B Contrastive-Loss Plan Review

## Review 1 — FAIL

Reviewer: independent subagent `deleg_95e5f9ba`

Blocking findings:

1. Permitting zero vectors with default `F.normalize` was semantically invalid for cosine similarity and numerically unsafe for float16.
2. Merely requiring a finite positive temperature did not prevent underflow/non-finite scaled logits.
3. The finite-latent contract lacked explicit tests for NaN and both infinities in every input, including masked negatives.

Accepted aspects:

- Cosine-normalized InfoNCE is a justified refinement of the specification's dot-product sketch.
- External anchor semantics and later detach decisions are appropriate for a standalone helper.
- Masking, strict tie handling, unscaled margins, active-negative counting, scope boundaries, and later deterministic hard-negative compatibility were sound.

Revisions made before re-review:

- Reject zero/near-zero vectors using a documented norm threshold.
- Compute normalization/loss in float32 for low-precision/float32 inputs and float64 for float64 inputs.
- Add forward-time temperature representability and finite-logit checks.
- Specify output dtypes.
- Add explicit low-precision zero-vector, tiny-temperature, and NaN/±inf tests, including masked negatives.
- Require all-true mask equivalence and exact zero gradients for masked negatives.
- Require later integration to specify gradient destinations independently and compare cosine and L2 margins.

Coding remains blocked until the revised plan receives PASS.

## Review 2 — FAIL

Reviewer: independent subagent `deleg_305a7cb0`

The original three blockers were resolved, but three additional numerical/API blockers remained:

1. Direct L2 norms can overflow for finite extreme float32/float64 vectors.
2. The compute/output dtype policy lacked valid positive-path tests.
3. `negative_mask` device semantics were unspecified.

Revisions made before Review 3:

- Replace direct normalization with a stable max-scaled L2 normalization algorithm.
- Add extreme-finite float32, bfloat16, and float64 positive-path tests.
- Add valid float16/bfloat16/float32/float64 output-dtype tests.
- Require the mask to share the latent device and test mismatches where CUDA is available.
- Add exact norm/temperature boundary tests and masked zero/near-zero rejection.
- Require grouped validations to be implemented as separate RED → GREEN cycles.

Coding remains blocked until Review 3 returns PASS.

## Review 3 — PASS

Reviewer: independent subagent `deleg_75f2b082`

All prior numerical, dtype, device, masking, and TDD blockers were resolved. The reviewer confirmed:

- max-abs-scaled normalization handles low precision and extreme finite values;
- exact norm/temperature boundaries and finite-logit checks are covered;
- compute/output dtype contracts have positive-path tests;
- masked negatives are validated but excluded from loss, metrics, counts, and gradients;
- cosine InfoNCE remains aligned with Track B;
- the three-file standalone scope is surgical.

Coding may proceed under the approved plan.
