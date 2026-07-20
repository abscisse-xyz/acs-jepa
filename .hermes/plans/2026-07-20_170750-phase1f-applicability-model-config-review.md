# Phase 1F Applicability Model/Config Construction Plan Review

Plan reviewed: `.hermes/plans/2026-07-20_170750-phase1f-applicability-model-config.md`

Verdict: PASS. Coding may proceed.

Independent review source:

- `deleg_a06f7581`, completed 2026-07-20.

Blocking issues:

- None.

Review findings:

- Model/config construction is an appropriate next plumbing slice after Phase 1E trainer wiring and before any dataloader/batch construction.
- The plan preserves the Track A sequence: learned auxiliary applicability supervision and config plumbing first; no planner/decoder behavior changes.
- Defaults are safe: `model.applicability_head.kind: none` and `trainer.applicability_loss_weight: 0.0` mean no applicability batch keys are required.
- The plan avoids production `applicable_actions()` use, simulator/oracle calls, checkpoint/data/planner/decoder changes, and hyperparameter tuning expansion.
- TDD coverage is adequate: default-disabled construction, enabled construction, optimizer inclusion, invalid config handling, and inconsistency rejection.

Non-blocking suggestions folded into the plan before coding:

- Explicitly create `acs-jepa-cli/tests/test_modeling.py` because no modeling test file currently exists.
- Add an explicit default optimizer parameter-count test because the acceptance criteria mentions unchanged default optimizer params.
- Either validate `applicability_pos_weight` whenever it is non-null, even when disabled, or document disabled config as inert; the plan will validate it whenever supplied.
- Explicitly state that checkpoint round-tripping of applicability-head weights is intentionally out of scope because current checkpoints save `jepa` and `goal_head`, not an applicability head.

Conclusion: proceed with Phase 1F implementation under TDD and block commit until separate code review passes.
