# Phase 1C Applicability Head Module Implementation Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Add the first learned-module building block for Track A: an `ApplicabilityHead` that scores `(state, grounded_action)` pairs from graph latent, action latent, and optional selected object latent summaries. This slice adds the neural module and unit tests only. It does not wire the head into training, loss computation, checkpoint loading, decoding, planning, or configs.

**Architecture:** Place the module in `packages/acs-jepa-core/src/acs_jepa/architectures.py` near action-related components and export it from `acs_jepa.__init__` only if needed by tests/importers. The head should be generic enough for later BCE training on Phase 1B examples, but independent of simulator/oracle code.

**Tech Stack:** Python 3.13, PyTorch, ACS-JEPA `JEPALatentState`, pytest, uv.

---

## Scope and assumptions

- Repository: `/opt/data/workspace/acs-jepa`.
- This plan covers only the neural scoring head and focused shape/masking tests.
- No trainer, loss module, config, planner, decoder, or checkpoint changes.
- The head should accept already-computed `action_latent`; it should not call the action encoder itself.
- Object context is optional but useful: it lets the later trained head condition on selected argument object latents rather than only graph/action global vectors.
- Because this slice has no data loader/trainer wiring, no `BCEWithLogitsLoss` integration is added yet; tests should only verify logits are shaped and differentiable.

## Research/spec review checkpoints

This supports Track A from `ACTION_LATENT_SOLUTION_SPEC.md`:

```text
ApplicabilityHead(action_latent, graph_latent, selected object latents)
  -> scalar logit P(applicable | state, action)
```

The implementation must preserve:

- Applicability is a learned state-action relation, not symbolic candidate generation.
- The module must be trainable from positives and hard negatives from earlier Phase 1 slices.
- No production-time `applicable_actions()` use is introduced.
- Object context should be role/order aware enough that later training can learn argument binding distinctions.

## Task 0: Plan review gate

**Objective:** Confirm Phase 1C scope before coding.

**Files:**
- Create: `.hermes/plans/2026-07-20_162512-phase1c-applicability-head-review.md`

**Review requirements:**

- No implementation starts until this review is complete.
- Blocking review comments must be folded back into this plan before Task 1.
- Confirm this slice is small enough: module + tests only, no trainer/loss/planner wiring.

## Task 1: Add tested `ApplicabilityHead`

**Objective:** Implement a differentiable neural head that maps graph/action/object context to one scalar logit per candidate action.

**Files:**
- Modify: `packages/acs-jepa-core/src/acs_jepa/architectures.py`
- Modify if needed: `packages/acs-jepa-core/src/acs_jepa/__init__.py`
- Create/modify: `packages/acs-jepa-core/tests/test_applicability_head.py`

**Step 1: Write failing tests**

Tests should cover:

1. `ApplicabilityHead(latent_dim=8, action_dim=6, max_action_arity=4, hidden_dim=12)` returns logits shaped `[B]` for:
   - `graph_latent`: `[B, D_z]`
   - `action_latent`: `[B, D_a]`
   - no object context.
2. With object context:
   - `object_latents`: `[B, A_max, D_z]`
   - `argument_mask`: `[B, A_max]`
   the head returns logits shaped `[B]`.
3. Role/order-aware object context changes the logit when unmasked object latents change, but ignores masked object slots.
4. Permuting two unmasked argument slots can change logits, proving the object-context path is not a permutation-invariant masked mean.
5. Gradients flow to `graph_latent`, `action_latent`, and unmasked `object_latents`.
6. Masked object slots do not affect logits when changed and receive zero/no gradient.
7. Input validation rejects:
   - rank mismatch for `graph_latent` or `action_latent`;
   - `graph_latent` last-dimension mismatch vs `latent_dim`;
   - `action_latent` last-dimension mismatch vs `action_dim`;
   - batch-size mismatch;
   - object context supplied without matching `argument_mask`;
   - `argument_mask` supplied without `object_latents`;
   - `object_latents.ndim != 3`;
   - `argument_mask.ndim != 2`;
   - object latent dimension mismatch vs `latent_dim`;
   - object/mask arity mismatch;
   - object/mask arity exceeding `max_action_arity`;
   - invalid dropout range;
   - non-positive dimensions in constructor.

Run RED:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_applicability_head.py -q
```

Expected: FAIL because `ApplicabilityHead` does not exist.

**Step 2: Implement minimal module**

Proposed API:

```python
class ApplicabilityHead(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        max_action_arity: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None: ...

    def forward(
        self,
        graph_latent: Tensor,
        action_latent: Tensor,
        object_latents: Tensor | None = None,
        argument_mask: Tensor | None = None,
    ) -> Tensor: ...
```

Suggested internals:

- Validate dimensions early.
- Project object context with a role/order-aware argument summary:
  - if no object context, use a zero vector shaped `[B, D_z]` on graph device/dtype;
  - allocate role/slot embeddings from explicit constructor parameter `max_action_arity`;
  - if context exists, require `argument_mask` shaped `[B, A]` with `A <= max_action_arity`;
  - add a learned argument-role embedding per slot before masking/aggregation, or flatten/project fixed slots with mask handling;
  - do **not** use a plain masked mean of raw object latents, because it is permutation-invariant and cannot distinguish role swaps.
- Concatenate `[graph_latent, action_latent, object_summary, abs(graph_latent - object_summary)]` or a similarly simple state/action/context feature vector.
- Use a small MLP ending in one output unit and return `logits.squeeze(-1)`.
- Keep dropout configurable but default `0.0` so tests are deterministic; tests that compare slot permutations must set `torch.manual_seed(...)`, use `dropout=0.0`, and use constructed inputs that make slot sensitivity observable.

Do not:

- call simulator code;
- call `ActionDecodingSpace`;
- enumerate actions;
- add BCE loss or trainer wiring in this slice.

**Step 3: Run GREEN tests**

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_applicability_head.py -q
```

Expected: PASS.

## Task 2: Export/import check

If tests/import style require public import, export from `packages/acs-jepa-core/src/acs_jepa/__init__.py`.

Test should assert either:

```python
from acs_jepa import ApplicabilityHead
```

or keep direct module import if the project usually tests internals from `acs_jepa.architectures`.

Prefer exporting if this head will be used by CLI/modeling in the next slice.

## Task 3: Verification

Run:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_applicability_head.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py acs-jepa-cli/tests/test_action_negative_sampling.py acs-jepa-cli/tests/test_action_applicability_labels.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_losses.py packages/acs-jepa-core/tests/test_graph_jepa_components.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check packages/acs-jepa-core/src/acs_jepa/architectures.py packages/acs-jepa-core/src/acs_jepa/__init__.py packages/acs-jepa-core/tests/test_applicability_head.py
```

If full component tests are too slow, run the targeted new test plus `test_graph_losses.py` and report the omission explicitly.

## Task 4: Separate implementation review

Review checklist:

- Scope: module/tests only; no training/planner/decoder/config changes.
- Spec: API can score `(state, grounded_action)` via graph latent, action latent, and selected object context.
- Correctness: shape validation, masking behavior, deterministic dropout default, gradient flow.
- Coding practice: simple module, no simulator imports, no hardcoded paths, exports are deliberate.

## Task 5: Commit after review passes

Only after separate code review passes:

```bash
git add .hermes/plans/2026-07-20_162512-phase1c-applicability-head.md \
  .hermes/plans/2026-07-20_162512-phase1c-applicability-head-review.md \
  packages/acs-jepa-core/src/acs_jepa/architectures.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_applicability_head.py
git commit -S -m "feat: add applicability head module"
```
