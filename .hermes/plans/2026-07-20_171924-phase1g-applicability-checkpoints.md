# Phase 1G Applicability Checkpoint Support Implementation Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Extend checkpoint save/load plumbing so enabled `ApplicabilityHead` weights can be saved and restored consistently for eval/plan/resume-like use, without adding applicability dataloader generation or planner/decoder scoring.

## Background

Phase 1F intentionally left checkpoint round-tripping out of scope. The implementation review explicitly noted this should be handled before enabled applicability training is treated as resumable/evaluable from checkpoint.

Current checkpoint behavior in `acs-jepa-cli/src/acs_jepa_cli/cli.py`:

- `_save_checkpoint(...)` stores:
  - `model_state_dict`
  - `goal_head_state_dict`
  - optimizer/scheduler state
  - config/vocab metadata
- `cmd_eval(...)` and `cmd_plan(...)` rebuild the model bundle from checkpoint config and load JEPA + goal-head state dicts.
- There is no `applicability_head_state_dict` key yet.

## Scope

Implement only checkpoint persistence/loading for the optional applicability head.

Allowed changes:

1. Save `applicability_head_state_dict` in `_save_checkpoint(...)` as:
   - `None` when `bundle.applicability_head is None`
   - `bundle.applicability_head.state_dict()` otherwise.
2. Load applicability head state in `cmd_eval(...)` and `cmd_plan(...)` when both:
   - `bundle.applicability_head is not None`
   - checkpoint has non-null `applicability_head_state_dict`.
3. Preserve backward compatibility with old checkpoints that lack the key.
4. Add focused tests in `acs-jepa-cli/tests/test_cli.py` or `acs-jepa-cli/tests/test_modeling.py` for helper/save-load behavior.
5. Keep default-disabled behavior unchanged.

Out of scope:

- Dataloader construction of `applicability_*` tensors.
- CLI commands or flags.
- Planner/decoder use of applicability scores.
- Simulator/oracle/applicable-actions calls.
- Changing checkpoint schema for `ApplicabilityLoss` (it has no trainable state currently).
- Loading optimizer/scheduler state in eval/plan beyond current behavior.
- New full training campaign or smoke run.

## TDD plan

### Test 1 — RED: checkpoint includes applicability head state when enabled

Add a unit-style test that builds a tiny enabled model bundle with `model.applicability_head.kind: mlp`, calls `_save_checkpoint(...)`, loads the file with `torch.load`, and asserts:

- checkpoint has `"applicability_head_state_dict"`.
- value is not `None`.
- keys match `bundle.applicability_head.state_dict().keys()`.
- default-disabled bundle saves the key with `None`.

This test should fail before implementation because `_save_checkpoint` lacks the key.

### Test 2 — RED: loader helper restores applicability head weights

Prefer extracting a small helper to avoid duplicating state-dict loading logic across `cmd_eval` and `cmd_plan`, e.g.:

```python
def _load_checkpoint_state(bundle, checkpoint):
    bundle.jepa.load_state_dict(checkpoint["model_state_dict"])
    if bundle.goal_head is not None and checkpoint.get("goal_head_state_dict") is not None:
        bundle.goal_head.load_state_dict(checkpoint["goal_head_state_dict"])
    if bundle.applicability_head is not None and checkpoint.get("applicability_head_state_dict") is not None:
        bundle.applicability_head.load_state_dict(checkpoint["applicability_head_state_dict"])
```

Test this helper by:

1. Build enabled bundle.
2. Save checkpoint.
3. Build a second enabled bundle and explicitly perturb its applicability-head parameters so the test does not rely on sequential random initialization being different.
4. Call `_load_checkpoint_state(second_bundle, checkpoint)`.
5. Assert second applicability head params now match first bundle.

### Test 3 — Backward compatibility with old checkpoint dicts

Call the same helper with a checkpoint dict that lacks `applicability_head_state_dict` and assert:

- no exception is raised;
- existing JEPA/goal loading still works.

### Test 4 — plan/eval parity if cheap

If existing CLI test helpers make it cheap, add assertions to the train/eval checkpoint smoke that `applicability_head_state_dict` is present and `None` for default config. Do not broaden the smoke to enable applicability training because dataloader keys do not exist yet.

Also add at least one focused command-path or monkeypatch-style assertion showing `cmd_eval` and/or `cmd_plan` routes through `_load_checkpoint_state(...)`, not only direct helper unit tests.

## Implementation tasks

1. Add RED tests and run targeted test command to confirm failure.
2. Patch `_save_checkpoint(...)` in `acs-jepa-cli/src/acs_jepa_cli/cli.py` to include the new key.
3. Add `_load_checkpoint_state(bundle, checkpoint)` helper in `cli.py`.
4. Replace duplicated JEPA/goal loading in `cmd_eval(...)` and `cmd_plan(...)` with the helper.
5. Ensure helper uses `.get("applicability_head_state_dict")` for backward compatibility.
6. If a bundle has an applicability head but the checkpoint lacks `applicability_head_state_dict`, issue a warning rather than failing. This preserves old-checkpoint compatibility while avoiding silent random-head evaluation.
7. In `cmd_plan`, set `bundle.applicability_head.eval()` when present for future-proofing even though planner scoring remains out of scope.
8. Run targeted and adjacent verification.

## Acceptance criteria

- Enabled applicability-head checkpoints save non-null applicability head state dicts.
- Default-disabled checkpoints save `applicability_head_state_dict: None` or otherwise preserve a clearly backward-compatible null state. Prefer saving explicit `None` for schema clarity.
- Eval/plan loading restores applicability head state when configured and present.
- Old checkpoints without the new key still load.
- No dataloader/planner/decoder/simulator/oracle behavior changes.
- No production `applicable_actions()` usage.
- TDD proof exists: targeted test fails before implementation and passes after.

## Verification commands

Use:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_modeling.py acs-jepa-cli/tests/test_cli.py::test_train_and_eval_commands_log_with_mlflow -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_training.py packages/acs-jepa-core/tests/test_applicability_loss.py packages/acs-jepa-core/tests/test_applicability_head.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check acs-jepa-cli/src/acs_jepa_cli/cli.py acs-jepa-cli/tests/test_cli.py acs-jepa-cli/tests/test_modeling.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python -m compileall acs-jepa-cli/src/acs_jepa_cli/cli.py
```

If the broader `test_cli.py` tuning-config path test still fails because `acs-jepa-cli/configs/tuning` is absent while `script/configs/tuning` exists, record it as unrelated unless the diff caused it.

## Code-review gate

After implementation and verification, dispatch independent implementation review. Do not commit until the implementation review returns PASS.
