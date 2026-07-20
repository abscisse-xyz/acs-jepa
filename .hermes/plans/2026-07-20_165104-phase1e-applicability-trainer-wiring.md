# Phase 1E Applicability Trainer Wiring Implementation Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Wire the Phase 1C `ApplicabilityHead` and Phase 1D `ApplicabilityLoss` into `JepaTrainer` as an optional supervised auxiliary training path over precomputed applicability batches. This slice is trainer-level plumbing and tests only. It does not change CLI configs, data loading, checkpoint format, decoder scoring, planner behavior, or production action generation.

**Motivation:** Track A needs BCE training for the applicability head, but previous slices intentionally stopped at diagnostics/module/loss. This slice adds the smallest in-core integration point so a later CLI/data slice can feed Phase 1B examples into training without redesigning the trainer.

---

## Scope and assumptions

- Repository: `/opt/data/workspace/acs-jepa`.
- Prior local signed commits:
  - `7828091 feat: add action latent statistics diagnostic`
  - `70d813e feat: add action negative sampler diagnostic`
  - `73420c6 feat: add action applicability label diagnostic`
  - `0d7ee78 feat: add applicability head module`
  - `d482ecc feat: add applicability loss helper`
- This slice modifies only core trainer/test exports as needed.
- No Hydra/OmegaConf config schema changes yet.
- No CLI training command changes yet.
- No checkpoint save/load changes yet.
- No planner/decoder scoring integration yet.
- No simulator/oracle code in trainer.
- The trainer receives precomputed tensors in the batch:
  - `applicability_graph_latents` or encoded latent-state source? See design choice below.
  - `applicability_action_latents`
  - optional `applicability_object_latents`
  - optional `applicability_argument_mask`
  - `applicability_labels`
  - optional `applicability_example_mask`

## Design choice to review

Prefer **precomputed head inputs** in this slice rather than deriving them from symbolic actions inside `JepaTrainer`.

Rationale:

- Keeps trainer wiring decoupled from Phase 1B JSON/data-loader details.
- Avoids simulator/oracle imports.
- Avoids action tensorization/gathering complexity before a reviewed data-loader slice.
- Lets unit tests exercise loss wiring with synthetic tensors.

Proposed batch keys:

```python
batch = {
    "states": ...,
    "actions": ...,
    "applicability_graph_latents": Tensor[N, D_z],
    "applicability_action_latents": Tensor[N, D_a],
    "applicability_labels": Tensor[N],
    # optional:
    "applicability_object_latents": Tensor[N, A_max, D_z],
    "applicability_argument_mask": BoolTensor[N, A_max],
    "applicability_example_mask": BoolTensor[N],
}
```

A later Phase 1F data/modeling slice can decide how to produce these tensors from Phase 1B examples, `GraphJEPA.encode`, and `ActionEncoder`.

## Research/spec review checkpoints

This implements the next smallest Track A integration after:

```text
ApplicabilityHead(action_latent, graph_latent, selected object latents)
  -> scalar logit
BCEWithLogitsLoss(applicability_logit, applicable_label)
```

The implementation must preserve:

- Applicability training is optional and disabled by default.
- No production-time `applicable_actions()` use is introduced.
- The learned planner/decoder are not changed in this slice.
- Supervision consumes offline labels from prior diagnostic/data slices.

## Task 0: Plan review gate

**Objective:** Confirm Phase 1E scope before coding.

**Files:**
- Create: `.hermes/plans/2026-07-20_165104-phase1e-applicability-trainer-wiring-review.md`

**Review requirements:**

- No implementation starts until this review is complete.
- Blocking review comments must be folded back into this plan before Task 1.
- Confirm this slice is small enough: trainer accepts optional head/loss and batch tensors; no CLI/data/config/checkpoint/planner wiring.
- Confirm precomputed head-input tensors are acceptable for this slice or require a revised plan.

## Task 1: Add trainer config/module fields

**Files:**
- Modify: `packages/acs-jepa-core/src/acs_jepa/training.py`
- Modify tests: `packages/acs-jepa-core/tests/test_training.py` or new focused test file.

### Step 1: Write failing tests

Tests should cover:

1. `JepaTrainerConfig` gains:
   - `applicability_loss_weight: float = 0.0`
   - `applicability_head_detach: bool = True`
2. `JepaTrainer` accepts optional:
   - `applicability_head: nn.Module | None = None`
   - `applicability_loss_module: nn.Module | None = None`
3. If `applicability_loss_weight == 0`, existing behavior is unchanged and no applicability batch keys are required.
   This must also hold when optional applicability head/loss modules are supplied but the weight remains zero.
4. If `applicability_loss_weight > 0`, `applicability_head` and `applicability_loss_module` are required.
5. Negative `applicability_loss_weight` is rejected.

Run RED targeted tests.

### Step 2: Implement minimal config/module plumbing

- Add config fields with default disabled values.
- Store optional modules on the trainer.
- Include `applicability_head` and `applicability_loss_module` in `_unique_trainable_parameters(...)` for grad clipping only when present.
- Set train/eval mode for the modules in `train_step`/`eval_step`.

## Task 2: Add applicability auxiliary loss path

### Step 1: Write failing tests

Tests should cover:

1. With `applicability_loss_weight > 0`, `train_step` adds applicability loss to `total_loss`.
2. `terms` includes:
   - `applicability`
   - `applicability_bce`
   - `applicability_positive_logit_mean` when present
   - `applicability_negative_logit_mean` when present
   - `applicability_positive_negative_margin` when present
3. `JepaTrainerStepOutput` gains optional `applicability_loss: Tensor | None` or stores it in `terms`; prefer adding a field for symmetry with `goal_loss`.
4. `applicability_head_detach=True` detaches input graph/action/object latents before head scoring so applicability loss does not update upstream producers through those tensors.
5. `applicability_head_detach=False` permits gradients through supplied latent tensors only if those tensors are graph-attached; ordinary dataloader-precomputed tensors will not update JEPA/action encoder parameters in this slice.
6. Missing required applicability batch keys are rejected only when loss weight is positive.
7. Optional object context is passed through when both `applicability_object_latents` and `applicability_argument_mask` exist.
8. Optional `applicability_example_mask` is passed to `ApplicabilityLoss`.

### Step 2: Implement minimal loss path

Proposed helper in `JepaTrainer`:

```python
def _applicability_loss(self, batch: dict[str, object]) -> ApplicabilityLossOutput | None:
    if self.config.applicability_loss_weight == 0:
        return None
    graph_latents = _required(batch, "applicability_graph_latents")
    action_latents = _required(batch, "applicability_action_latents")
    labels = _required(batch, "applicability_labels")
    object_latents = batch.get("applicability_object_latents")
    argument_mask = batch.get("applicability_argument_mask")
    example_mask = batch.get("applicability_example_mask")
    if self.config.applicability_head_detach:
        graph_latents = graph_latents.detach()
        action_latents = action_latents.detach()
        if object_latents is not None:
            object_latents = object_latents.detach()
    logits = self.applicability_head(graph_latents, action_latents, object_latents, argument_mask)
    return self.applicability_loss_module(logits, labels, example_mask=example_mask)
```

Then combine:

```python
applicability_output = self._applicability_loss(batch)
if applicability_output is not None:
    total_loss = total_loss + self.config.applicability_loss_weight * applicability_output.total
```

Add detached diagnostics to `terms`.
Diagnostics from `ApplicabilityLossOutput` should be unweighted and detached, matching existing trainer term style.

Do not:

- import simulator code;
- generate negatives;
- read Phase 1B JSON;
- call action encoders or `ActionDecodingSpace`;
- update CLI config/checkpoint behavior.

## Task 3: Verification

Run:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_training.py packages/acs-jepa-core/tests/test_applicability_loss.py packages/acs-jepa-core/tests/test_applicability_head.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_losses.py packages/acs-jepa-core/tests/test_graph_jepa_components.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py acs-jepa-cli/tests/test_action_negative_sampling.py acs-jepa-cli/tests/test_action_applicability_labels.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check packages/acs-jepa-core/src/acs_jepa/training.py packages/acs-jepa-core/tests/test_training.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core python -m compileall packages/acs-jepa-core/src/acs_jepa/training.py
```

## Task 4: Separate implementation review

Review checklist:

- Scope: trainer plumbing/tests only; no CLI/config/checkpoint/planner/decoder/simulator changes.
- Spec: optional supervised applicability objective over offline/precomputed labels.
- Correctness: disabled default preserves existing behavior; missing keys rejected only when enabled; detach behavior tested; terms are detached; grad clipping includes new modules.
- Coding practice: small helper, no oracle leakage, no shape assumptions beyond delegated head/loss validation.

## Task 5: Commit after review passes

Only after separate code review passes:

```bash
git add .hermes/plans/2026-07-20_165104-phase1e-applicability-trainer-wiring.md \
  .hermes/plans/2026-07-20_165104-phase1e-applicability-trainer-wiring-review.md \
  packages/acs-jepa-core/src/acs_jepa/training.py \
  packages/acs-jepa-core/tests/test_training.py
git commit -S -m "feat: wire applicability loss into trainer"
```
