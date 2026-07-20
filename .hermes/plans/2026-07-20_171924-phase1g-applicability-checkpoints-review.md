# Phase 1G Applicability Checkpoint Support Plan Review

Plan reviewed: `.hermes/plans/2026-07-20_171924-phase1g-applicability-checkpoints.md`

Verdict: PASS. Coding may proceed.

Independent review source:

- `deleg_efbde5ef`, completed 2026-07-20.

Blocking issues:

- None.

Review findings:

- Phase 1G is correctly scoped as a small checkpointing slice after Phase 1F and before treating enabled applicability training/eval/plan flows as checkpoint-safe.
- The slice aligns with Track A / Phase 1 intent: preserve learned applicability-head state without adding planner scoring, dataloader applicability tensors, simulator/oracle production use, or CLI surface.
- Backward compatibility via `checkpoint.get("applicability_head_state_dict")` is appropriate.
- Saving explicit `None` for disabled configs is reasonable schema hygiene and keeps default-disabled behavior unchanged.
- Extracting `_load_checkpoint_state(bundle, checkpoint)` is justified because JEPA/goal loading is duplicated in `cmd_eval` and `cmd_plan`.

Non-blocking suggestions folded into the plan before coding:

- Add at least one command-path or monkeypatch-style assertion proving `cmd_eval`/`cmd_plan` call the new helper, not only helper unit tests.
- In restore tests, explicitly perturb one applicability head's parameters instead of relying on random initialization differences.
- Consider warning rather than failing when a configured applicability head loads an old checkpoint without `applicability_head_state_dict`, to avoid silent random-head evaluation while preserving backward compatibility.
- Set `bundle.applicability_head.eval()` in `cmd_plan` when present for future-proofing, although planner scoring remains out of scope.

Conclusion: proceed with Phase 1G implementation under TDD and block commit until separate code review passes.
