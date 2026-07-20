# Phase 1F Applicability Model/Config Construction Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Add disabled-by-default CLI/model-construction support for building `ApplicabilityHead` + `ApplicabilityLoss` and passing them into the Phase 1E trainer wiring.

This slice intentionally stops at construction/config plumbing. It does **not** create applicability training examples in the dataloader and does **not** score planner/decoder candidates.

## Background

Previous slices established:

- Phase 1A: deterministic type-valid hard-negative sampler diagnostic.
- Phase 1B: offline applicability label/example builder diagnostic.
- Phase 1C: neural `ApplicabilityHead` scoring `(state latent, action latent, optional argument object latents)`.
- Phase 1D: standalone `ApplicabilityLoss` BCE helper.
- Phase 1E: optional `JepaTrainer` wiring consuming precomputed applicability tensors.

Phase 1F should make the new modules constructible from normal config/model-building paths, while preserving current default training behavior exactly.

## Scope

Implement only:

1. Extend CLI model config defaults with an applicability-head section and trainer loss knobs, all disabled by default.
2. Extend `acs_jepa_cli.modeling.ModelBundle` to carry optional applicability modules.
3. Add `build_applicability_head(...)` or equivalent in `acs_jepa_cli.modeling`.
4. Construct `ApplicabilityHead` and `ApplicabilityLoss` only when `config.model.applicability_head.kind != "none"` or when enabled by the chosen explicit config contract.
5. Include modules in optimizer parameter collection when present.
6. Pass modules and trainer config fields into `JepaTrainer`.
7. Add focused tests for default-disabled construction and enabled construction.

Do **not** implement:

- Dataloader generation of `applicability_*` batch keys.
- CLI commands or diagnostic scripts.
- Checkpoint serialization changes beyond whatever existing full-model checkpoints already capture naturally.
- Planner/decoder applicability scoring.
- Simulator/oracle/applicable-actions calls in production training.
- Automatic derivation of applicability labels from traces.
- Hyperparameter tuning configs beyond one minimal disabled default and tests.

## Proposed config contract

Add under `model` in `script/configs/default.yaml`:

```yaml
  applicability_head:
    # Optional auxiliary applicability head: none or mlp.
    kind: none
    # Hidden width for the applicability head. Null/default may fall back to predictor hidden dim.
    hidden_dim: 64
    # Dropout inside the applicability head MLP.
    dropout: 0.0
```

Add under `trainer`:

```yaml
  # Multiplier for the auxiliary applicability BCE loss. Requires applicability batch keys when > 0.
  applicability_loss_weight: 0.0
  # true trains the applicability head on detached precomputed latents; false permits gradients through graph-attached supplied tensors.
  applicability_head_detach: true
  # Positive scalar class weight for positive applicability labels, or null for unweighted BCE.
  applicability_pos_weight: null
```

Default behavior must remain `kind: none`, `applicability_loss_weight: 0.0`, so current training remains unchanged and no applicability batch keys are required.

## Implementation tasks

### Task 1 — RED tests

Create `acs-jepa-cli/tests/test_modeling.py` because no modeling-specific test file currently exists.

Required tests:

1. Default config builds with no applicability modules:
   - `bundle.applicability_head is None`
   - `bundle.applicability_loss_module is None`
   - `bundle.trainer.config.applicability_loss_weight == 0.0`
   - existing JEPA/goal construction still works.
   - optimizer trainable parameter count is unchanged relative to the pre-applicability default path: it should include JEPA and the configured goal head, but no applicability-head parameters.
2. Enabled config builds applicability modules:
   - set `model.applicability_head.kind = "mlp"`
   - set `trainer.applicability_loss_weight > 0`
   - `bundle.applicability_head` is an `ApplicabilityHead`
   - `bundle.applicability_loss_module` is an `ApplicabilityLoss`
   - trainer receives both modules and weight/detach config.
3. Optimizer includes applicability head parameters when enabled:
   - collect parameter ids from optimizer groups and assert all head trainable parameter ids are included.
4. Invalid config is rejected clearly:
   - unknown applicability head kind raises `ValueError`.
   - negative dropout should be rejected by `ApplicabilityHead`; the test may assert construction raises.
   - non-positive `applicability_pos_weight` should be rejected whenever supplied, even if `model.applicability_head.kind == "none"`, so inert disabled configs do not hide invalid knobs.
5. Weight-positive but kind-none behavior:
   - decide explicitly: either reject at model-building time with clear `ValueError`, or build disabled head only when kind is mlp.
   - Recommended: reject `applicability_loss_weight > 0` when `kind == "none"`, before reaching `JepaTrainer`, so users get a config-level error.

Run these tests before implementation and confirm RED.

### Task 2 — Config/default changes

Patch `script/configs/default.yaml` only.

Add comments that make the experimental nature clear and preserve defaults.

Do not edit the tuning matrix or adaptive configs in this slice.

### Task 3 — Modeling construction

In `acs_jepa_cli.modeling`:

1. Import `ApplicabilityHead` and `ApplicabilityLoss`.
2. Extend `ModelBundle`:

```python
applicability_head: nn.Module | None
applicability_loss_module: nn.Module | None
```

3. Add `build_applicability_head(vocab, config)`:

```python
kind = str(config.model.applicability_head.kind)
if kind == "none": return None
if kind == "mlp": return ApplicabilityHead(
    latent_dim=int(config.model.latent_dim),
    action_dim=int(config.model.action_dim),
    max_action_arity=vocab.max_action_arity,
    hidden_dim=int(config.model.applicability_head.hidden_dim),
    dropout=float(config.model.applicability_head.dropout),
)
raise ValueError(...)
```

4. Add helper for optional `pos_weight`:

```python
pos_weight = None if config.trainer.applicability_pos_weight is None else float(...)
applicability_loss_module = None if applicability_head is None else ApplicabilityLoss(pos_weight=pos_weight)
```

5. Reject inconsistent config:

```python
if config.trainer.applicability_loss_weight > 0 and applicability_head is None:
    raise ValueError("applicability_loss_weight > 0 requires model.applicability_head.kind != 'none'")
```

6. Move modules to device when present.
7. Include `applicability_head` in `build_optimizer([...])`.
8. Pass trainer config fields:
   - `applicability_loss_weight`
   - `applicability_head_detach`
9. Pass trainer modules:
   - `applicability_head`
   - `applicability_loss_module`
10. Return modules on `ModelBundle`.

Checkpoint round-tripping for applicability-head weights remains intentionally out of scope. Current checkpoints save `jepa` and `goal_head`, not a separate applicability head; changing checkpoint schemas belongs in a later reviewed slice.

## Acceptance criteria

- Default config behavior unchanged:
  - no applicability modules are built;
  - optimizer parameter count for default path is unchanged except for no-op dataclass fields;
  - trainer does not require applicability batch keys.
- Enabled config builds modules and passes them to trainer.
- Applicability head parameter tensors are included in optimizer groups.
- Invalid config surfaces clear errors.
- No production simulator/oracle/applicable-actions usage.
- No dataloader/checkpoint/planner/decoder changes.
- TDD proof exists: tests fail before implementation and pass after.

## Verification commands

Use:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_modeling.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_training.py packages/acs-jepa-core/tests/test_applicability_loss.py packages/acs-jepa-core/tests/test_applicability_head.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check acs-jepa-cli/src/acs_jepa_cli/modeling.py acs-jepa-cli/tests/test_modeling.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python -m compileall acs-jepa-cli/src/acs_jepa_cli/modeling.py
```

Also run existing CLI diagnostics tests if any modeling changes appear to affect CLI imports:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py acs-jepa-cli/tests/test_action_negative_sampling.py acs-jepa-cli/tests/test_action_applicability_labels.py -q
```

## Code-review gate

After implementation and verification, dispatch independent implementation review. Do not commit until the implementation review returns PASS.
