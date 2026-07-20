# Phase 1C Applicability Head Plan Review

Plan reviewed: `.hermes/plans/2026-07-20_162512-phase1c-applicability-head.md`

Final verdict: PASS. Coding may proceed.

Independent review sources:

- `deleg_5bfd8d34`, completed 2026-07-20: FAIL.
- `deleg_04988444`, completed 2026-07-20: FAIL.
- `deleg_5a57e132`, completed 2026-07-20: PASS.

Blocking issues found and resolved:

1. The original plan proposed a masked mean over selected object latents while also requiring role/order-aware argument binding. A plain masked mean is permutation-invariant and cannot distinguish role swaps or argument-position binding differences, which are central to the ACS-JEPA failure mode and Track A rationale.
2. The revised role/order-aware plan still lacked an explicit `max_action_arity`/slot-count constructor argument, which would force dynamic parameters in `forward` or push implementation back toward permutation-invariant pooling.
3. Object-context validation requirements were not explicit enough for the module’s central responsibility.

Corrections applied before coding:

- The proposed API now includes `max_action_arity: int`.
- The plan requires role/order-aware object context via learned argument-role embeddings or equivalent fixed-slot projection.
- The plan explicitly prohibits plain masked mean of raw object latents.
- The plan requires tests for:
  - slot permutation changing logits;
  - masked slots not affecting logits;
  - masked slots receiving zero/no gradient;
  - object latent rank;
  - argument mask rank;
  - graph/action latent last-dimension mismatches;
  - object latent dimension against `latent_dim`;
  - object/mask arity alignment;
  - object/mask arity not exceeding `max_action_arity`;
  - `argument_mask` supplied without `object_latents`;
  - invalid dropout range.

Non-blocking suggestions folded into the plan:

- Permutation tests should use `torch.manual_seed(...)`, `dropout=0.0`, and constructed inputs that make slot sensitivity observable.
- Export from `acs_jepa.__init__` should be deliberate; exporting is acceptable because later slices are expected to consume the head publicly.

Conclusion: proceed with Phase 1C implementation under TDD and block commit until separate code review passes.
