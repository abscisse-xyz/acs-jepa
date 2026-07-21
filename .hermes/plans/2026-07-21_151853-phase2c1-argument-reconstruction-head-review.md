# Phase 2C1 Argument-Reconstruction-Head Plan Review

## Review 1 — PASS

Reviewer: independent subagent `deleg_8398d725`

No blockers. The reviewer confirmed that:

- problem-local candidate classification implements the per-role reconstruction objective;
- the nonlinear additive action/object/role feature model has adequate interaction capacity;
- learned role embeddings provide explicit role sensitivity;
- shared candidate scoring preserves object permutation equivariance;
- role-specific candidate masks support type validity and padding;
- all-masked inactive roles are a correct boundary for Stage 2C2 to handle before CE;
- state-conditioned object latents and action latents provide sufficient conditioning for this primitive;
- the scope is appropriately surgical.

Implementation will include the non-blocking recommendations:

- permute candidate masks in the object-equivariance test;
- explicitly test all-`-inf` inactive roles;
- phrase/test zero gradients per masked role-candidate path, accounting for candidates active in other roles;
- move the module to float64 in float64 preservation tests;
- preserve Stage 2C2 requirements to select active rows before CE, reject missing/masked targets, and remap targets under candidate permutations.

Coding may proceed.
