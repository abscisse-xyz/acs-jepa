# Phase 1D Applicability BCE Loss Plan Review

Plan reviewed: `.hermes/plans/2026-07-20_164156-phase1d-applicability-loss.md`

Verdict: PASS. Coding may proceed.

Independent review source:

- `deleg_2dcf8114`, completed 2026-07-20.

Blocking issues:

- None.

Review findings:

- Scope is correctly limited to a standalone BCE-with-logits applicability loss helper plus tests.
- No `GraphJEPALossModule`, trainer, config, planner, decoder, checkpoint, simulator, oracle, or production `applicable_actions()` wiring is planned.
- The helper consumes precomputed logits and labels, preserving the Track A design:
  - `ApplicabilityHead(...) -> scalar logit`
  - `BCEWithLogitsLoss(applicability_logit, applicable_label)`
- Planned tests cover scalar BCE parity, optional masking, masked gradient behavior, empty effective-batch rejection, shape validation, positive/negative diagnostics, absent-class diagnostics, and `pos_weight` support.

Non-blocking suggestions folded into the plan before coding:

- Prefer tests importing from `acs_jepa.losses` unless public package export is intentional.
- Require `example_mask.dtype == torch.bool` instead of silently converting numeric masks.
- Validate label values are in `[0, 1]` to catch dataset bugs.
- Explicitly test that masked logits have zero gradient while unmasked logits receive nonzero gradient.

Conclusion: proceed with Phase 1D implementation under TDD and block commit until separate code review passes.
