# Stage 2E Plan Review Record

Plan: `.hermes/plans/2026-07-21_172719-phase2e-action-auxiliary-integration.md`
Governing specification: `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2

## Review 1 — FAIL

Reviewer: independent subagent `deleg_488c36b4`

Blocking findings:

1. Integrated versus legacy applicability routing was inferred partly from `action_supervision` presence. An unrelated contrastive/argument objective could therefore switch a legacy precomputed-latent applicability configuration into integrated Stage 2D2 mode. The trainer also had only one proposed effective coefficient and could not encode the routing choice explicitly.
2. The strict JSON artifact contract did not reject duplicate JSON object member names, permitting last-value-wins ambiguity. It also did not state the policy for duplicate atoms/actions inside list-valued fields.

Resolution in revised plan:

- added distinct `JepaTrainerConfig.integrated_applicability_loss_weight` and retained legacy `applicability_loss_weight`;
- defined the complete coefficient matrix, made modes mutually exclusive, rejected both-positive configurations, and prohibited routing based on unrelated batch keys;
- required duplicate-member detection through `object_pairs_hook` or equivalent at every JSON object level;
- required rejection of duplicate canonical state atoms, applicable actions, and state-key entries;
- adopted accepted refinements for exact anchor construction, term names/count tensors, causal prefix/future tests, permuted object-row tests, checkpoint early-return coverage, absolute artifact paths, and explicit Stage 2F/2G completion gates.

Coding remains blocked pending Review 2 PASS.

## Review 2 — PASS

Reviewer: independent subagent `deleg_8978c18f`

The reviewer confirmed that both Review 1 blockers are fully resolved and that the accepted refinements are explicit, testable, and implementable against the current trainer, rollout, action encoder, Stage 2D2 tensors/table, CLI builders, config merge behavior, and checkpoint path. Shapes, masking/padding, autograd, oracle isolation, optimizer ownership, compatibility, TDD order, and scope passed review. TDD implementation is authorized.

## Implementation Review 1 — FAIL

Reviewer: independent subagent `deleg_c5c80906`

Blocking findings:

1. CLI configuration used nested/non-governing field locations and skipped unconditional disabled-path hyperparameter validation.
2. Integrated applicability object gathering checked only index bounds, not `object_mask`, so an active role could consume an in-range padded slot.
3. Offline JSON errors were not consistently UTF-8/path/entry aware, especially for nested duplicate members and Stage 2D2 symbolic validation.

Resolution:

- replaced nested auxiliary configuration with the exact flat `model.loss` fields and `model.argument_reconstruction_head` contract; moved hard-negative count to `model.loss`; added unconditional finite/range/kind/capacity validation, including `num_objects >= 1`;
- made applicability object gathering require `object_mask` and reject active references to absent problem-local objects;
- introduced pair-preserving JSON decoding, context-aware duplicate conversion, explicit UTF-8 reading, path/entry-aware schema and symbolic errors, and per-entry Stage 2D2 validation;
- strengthened causal prefix/future/padding mutation tests, strict loader/schema/symbolic/missing-state tests, both dataset paths, label-isolation checks, and a real train-step optimizer-update test.

Implementation was blocked pending Implementation Review 2 PASS.

## Implementation Review 2 — PASS

Reviewer: independent subagent `deleg_055f28be`

The reviewer confirmed that all three Implementation Review 1 blockers are resolved and found no new blocking issues. Exact flat configuration fields, unconditional validation, object-presence enforcement, UTF-8/context-aware strict JSON parsing, explicit applicability routing, causal masking, optimizer/checkpoint compatibility, disabled defaults, and oracle isolation all satisfy the approved contract. Focused and full verification passed, apart from the documented unrelated missing tuning-config discovery fixture. Stage 2E is authorized for a signed commit.
