# Phase 1E Applicability Trainer Wiring Plan Review

Plan reviewed: `.hermes/plans/2026-07-20_165104-phase1e-applicability-trainer-wiring.md`

Verdict: PASS. Coding may proceed.

Independent review source:

- `deleg_f2473114`, completed 2026-07-20.

Blocking issues:

- None.

Review findings:

- Scope is consistent with Track A / Phase 1: optional supervised applicability objective from offline/precomputed labels.
- Precomputed head-input tensors are acceptable for this trainer-plumbing slice. Deriving graph/action/object latents from rollout/action encoder would pull in data/modeling/action-tensorization complexity that belongs in a later reviewed slice.
- The plan does not introduce CLI/Hydra config, data-loader, checkpoint, planner, decoder, simulator, oracle, or production `applicable_actions()` changes.
- TDD and separate implementation review are required.

Non-blocking suggestions folded into the plan before coding:

- Clarify that `applicability_head_detach=False` only permits gradients through supplied tensors if those tensors are graph-attached; ordinary dataloader-precomputed tensors will not update JEPA/action encoder parameters.
- Ensure `ApplicabilityLossOutput` diagnostics added to `terms` are detached and unweighted, matching existing trainer style.
- Explicitly test that `applicability_loss_weight=0.0` does not require applicability batch keys and preserves JEPA-only behavior even when optional head/loss modules are supplied.
- Keep validation focused: reject negative weight; require head/loss only when weight is positive; leave shape validation to `ApplicabilityHead` and `ApplicabilityLoss`.

Conclusion: proceed with Phase 1E implementation under TDD and block commit until separate code review passes.
