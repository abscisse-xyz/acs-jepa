# Phase 1D Applicability BCE Loss Helper Implementation Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Add a small supervised loss helper for Track A that computes BCE-with-logits over `ApplicabilityHead` scores. This slice introduces the loss/output data structure and tests only. It does not wire the loss into `GraphJEPALossModule`, the trainer, config files, checkpoint format, decoder scoring, or planner behavior.

**Motivation:** Phase 1B created offline `(state, grounded_action, applicability_label)` diagnostic examples. Phase 1C added a role/order-aware `ApplicabilityHead`. Phase 1D adds the minimal training objective building block, while still avoiding production behavior changes until a later reviewed wiring slice.

---

## Scope and assumptions

- Repository: `/opt/data/workspace/acs-jepa`.
- Prior local commits:
  - `7828091 feat: add action latent statistics diagnostic`
  - `70d813e feat: add action negative sampler diagnostic`
  - `73420c6 feat: add action applicability label diagnostic`
  - `0d7ee78 feat: add applicability head module`
- Add a standalone loss module in `packages/acs-jepa-core/src/acs_jepa/losses.py`.
- Export from `acs_jepa.__init__` only if needed for public use/tests.
- The loss consumes logits that are already produced by `ApplicabilityHead`; it must not encode actions, gather objects, call simulator code, enumerate actions, or load Phase 1B JSON.
- Unknown labels from `--no-oracle-labels` are out of scope for the BCE helper and should be represented by an optional boolean mask or excluded by the caller.

## Research/spec review checkpoints

This implements the Track A training-loss line from `ACTION_LATENT_SOLUTION_SPEC.md`:

```text
BCEWithLogitsLoss(applicability_logit, applicable_label)
```

The implementation must preserve:

- It is a supervised head objective, not an action generator.
- It trains from labels generated offline by Phase 1B.
- It remains independent of production-time `applicable_actions()` use.
- It can report positive/negative diagnostics and margins needed for later acceptance checks.

## Task 0: Plan review gate

**Objective:** Confirm Phase 1D scope before coding.

**Files:**
- Create: `.hermes/plans/2026-07-20_164156-phase1d-applicability-loss-review.md`

**Review requirements:**

- No implementation starts until this review is complete.
- Blocking review comments must be folded back into this plan before Task 1.
- Confirm this slice is small enough: standalone loss helper + tests only, no trainer/config/planner/decoder wiring.

## Task 1: Add tested loss/output helper

**Files:**
- Modify: `packages/acs-jepa-core/src/acs_jepa/losses.py`
- Modify if public export is chosen: `packages/acs-jepa-core/src/acs_jepa/__init__.py`
- Create/modify: `packages/acs-jepa-core/tests/test_applicability_loss.py`

### Step 1: Write failing tests

Tests should cover:

1. `ApplicabilityLoss()` computes scalar `total` matching `torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())`.
2. Output dataclass exposes:
   - `total`
   - `bce`
   - `positive_logit_mean`
   - `negative_logit_mean`
   - `positive_negative_margin`
   - `num_examples`
   - `num_positive`
   - `num_negative`
3. Optional `example_mask` excludes unknown/unlabeled examples before loss/stat computation.
4. Masked-out logits receive zero/no gradient while unmasked logits receive nonzero gradient.
5. The helper rejects empty effective batches after masking.
6. The helper rejects shape mismatches and non-rank-1 logits/labels/masks.
7. Positive/negative means are `None` when that class is absent, but BCE still works for all-positive or all-negative effective batches.
8. Optional `pos_weight` is supported and matches PyTorch BCE behavior.
9. The helper rejects non-bool `example_mask` tensors rather than silently converting numeric masks.
10. The helper rejects labels outside `[0, 1]` to catch malformed diagnostic datasets.

Run RED:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_applicability_loss.py -q
```

Expected: FAIL because `ApplicabilityLoss` does not exist.

### Step 2: Implement minimal module

Proposed API:

```python
@dataclass(frozen=True)
class ApplicabilityLossOutput:
    total: Tensor
    bce: Tensor
    positive_logit_mean: Tensor | None
    negative_logit_mean: Tensor | None
    positive_negative_margin: Tensor | None
    num_examples: int
    num_positive: int
    num_negative: int

class ApplicabilityLoss(nn.Module):
    def __init__(self, *, pos_weight: float | None = None) -> None: ...

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        example_mask: Tensor | None = None,
    ) -> ApplicabilityLossOutput: ...
```

Implementation details:

- Validate `logits.ndim == 1`, `labels.ndim == 1`, and same shape.
- Convert labels to `logits.dtype` for BCE but do not silently reshape.
- If `example_mask` is supplied:
  - require rank-1 and same shape;
  - require `example_mask.dtype == torch.bool`;
  - select only mask-true rows for BCE/statistics;
  - raise `ValueError` if no rows remain.
- Validate all effective labels lie in `[0, 1]`.
- Use `F.binary_cross_entropy_with_logits(...)`.
- If `pos_weight` is supplied, register it as a buffer tensor or create it on the logits device/dtype in forward; reject `pos_weight <= 0`.
- Diagnostics:
  - `positive_logit_mean`: mean logits for labels >= 0.5 if present, else `None`;
  - `negative_logit_mean`: mean logits for labels < 0.5 if present, else `None`;
  - `positive_negative_margin`: positive mean minus negative mean if both exist, else `None`.
- Do not import `ApplicabilityHead` here; the loss is intentionally decoupled from the module that produces logits.

### Step 3: Run GREEN tests

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_applicability_loss.py -q
```

Expected: PASS.

## Task 2: Verification

Run:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_applicability_loss.py packages/acs-jepa-core/tests/test_applicability_head.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_losses.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py acs-jepa-cli/tests/test_action_negative_sampling.py acs-jepa-cli/tests/test_action_applicability_labels.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check packages/acs-jepa-core/src/acs_jepa/losses.py packages/acs-jepa-core/src/acs_jepa/__init__.py packages/acs-jepa-core/tests/test_applicability_loss.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core python -m compileall packages/acs-jepa-core/src/acs_jepa/losses.py packages/acs-jepa-core/src/acs_jepa/__init__.py
```

## Task 3: Separate implementation review

Review checklist:

- Scope: standalone loss helper/tests only; no trainer/config/planner/decoder behavior changes.
- Spec: exactly implements BCE-with-logits for applicability labels and reports useful positive-vs-negative statistics.
- Correctness: masks exclude unlabeled examples; gradients are zero for masked rows; empty effective batch rejected; pos_weight behavior matches PyTorch.
- Coding practice: no simulator/oracle imports, no action enumeration, simple validation, export deliberate.

## Task 4: Commit after review passes

Only after separate code review passes:

```bash
git add .hermes/plans/2026-07-20_164156-phase1d-applicability-loss.md \
  .hermes/plans/2026-07-20_164156-phase1d-applicability-loss-review.md \
  packages/acs-jepa-core/src/acs_jepa/losses.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_applicability_loss.py
git commit -S -m "feat: add applicability loss helper"
```
