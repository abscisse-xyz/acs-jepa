# Phase 2C1 Role-Aware Argument Reconstruction Head Plan

Governing documents:

- `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2 and the argument-identifiability objective.
- `.hermes/plans/2026-07-21_143024-phase2-roadmap-stage2a-action-vicreg.md`.

Completed prerequisites:

- Stage 2A action VC/VICReg helper: `2ec072e`.
- Stage 2B same-state contrastive helper: `490d980`.

Phase 2 deliverables and acceptance criteria remain unchanged by splitting argument reconstruction into a head stage (2C1), loss stage (2C2), and later data/integration stages.

## Objective

Add a standalone role-aware head that scores each problem-local candidate object for each action-argument role from an action latent and current-state object latents.

The head creates the trainable scoring primitive for the Phase 2 argument-reconstruction auxiliary objective while leaving targets, argument masks, type-valid candidate masks, loss reduction, data construction, and trainer wiring to later separately reviewed stages.

## Representation contract

```python
class ArgumentReconstructionHead(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        object_dim: int,
        max_action_arity: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None: ...

    def forward(
        self,
        action_latents: Tensor,          # float [B, D_a]
        candidate_object_latents: Tensor,# float [B, O, D_o]
        candidate_mask: Tensor | None = None, # bool [B, R_max, O]
    ) -> Tensor:                         # float [B, R_max, O]
```

`R_max = max_action_arity` and `O` is the padded problem-local object count for the batch.

## Architecture

Constructor-owned modules only:

- `action_projection: Linear(D_a, H)`;
- `object_projection: Linear(D_o, H)`;
- `role_embedding: Embedding(R_max, H)`;
- `scorer: GELU -> Dropout -> Linear(H, 1)`.

For every `(batch, role, candidate object)`:

```text
feature = action_projection(action)
        + object_projection(candidate_object)
        + role_embedding(role)
logit = scorer(feature)
```

The nonlinear scorer makes role/action/object interactions learnable while preserving permutation equivariance over the candidate-object axis. Learned role embeddings make argument order explicit and prevent permutation-invariant role pooling.

`candidate_mask` is role-specific because PDDL type-valid object domains differ by argument role. Invalid or padded candidates receive `-inf` via `masked_fill`; they must not affect active logits or receive gradients through this head.

Actual schema arity and target indices are intentionally absent from the head API. Stage 2C2/later data integration will supply an argument-role mask and ensure each active role has a valid target/candidate. Inactive roles may therefore have all candidates masked and yield all `-inf` logits.

## Tensor validation

- Positive constructor dimensions; `max_action_arity > 0`; finite dropout in `[0,1)`.
- `action_latents` rank 2 `[B,D_a]`; candidates rank 3 `[B,O,D_o]`.
- Matching non-empty `B`; non-empty `O`; exact final dimensions.
- Both latent tensors floating, finite, same dtype, and same device.
- Optional mask must be bool `[B,R_max,O]` on the latent device.
- The head preserves the input floating dtype, including float64. Low-precision behavior follows the underlying configured model/autocast path; this stage does not silently cast.
- No input is detached.

## Scope

Modify only:

- `packages/acs-jepa-core/src/acs_jepa/architectures.py`
- `packages/acs-jepa-core/src/acs_jepa/__init__.py`
- `packages/acs-jepa-core/tests/test_argument_reconstruction_head.py`

## Explicit non-goals

- No cross-entropy/reconstruction loss (Stage 2C2).
- No target-index or argument-mask handling.
- No action/object encoder construction or detach policy.
- No negative generation, applicability labels, simulator/oracle calls, or data collation.
- No `GraphJEPALossModule`, trainer, config, optimizer, or checkpoint changes.
- No decoder/planner changes, smoke retraining, or Phase 2 acceptance claim.

## TDD sequence

Use small vertical RED → GREEN cycles. Passing behavior introduced incidentally is recorded as regression coverage rather than misreported as RED.

1. **Output shape and basic gradients**
   - RED: `[B,D_a]` plus `[B,O,D_o]` produces finite `[B,R_max,O]`; a non-degenerate sum of logits gives finite gradients to actions, objects, and all head parameters.
   - GREEN: implement constructor-owned projections, role embeddings, and scorer.
2. **Object permutation equivariance**
   - RED: permuting the candidate-object axis permutes logits identically and does not otherwise change them.
   - GREEN: retain candidate-wise shared scoring with no object-position embedding.
3. **Role sensitivity**
   - RED: deterministic parameter assignment proves two role slots can produce different logits for the same action/object candidate.
   - GREEN: use fixed constructor-owned role embeddings in every role/candidate feature.
4. **Candidate masking**
   - RED: invalid candidates are `-inf`; changing a masked candidate cannot change active logits; masked candidate-object gradients are exactly zero while active candidates receive gradients.
   - GREEN: validate and apply bool role-specific masking after scoring.
5. **Validation**
   - RED: constructor rejects invalid dimensions/dropout.
   - RED: forward rejects rank/shape/empty mismatch, non-floating or mismatched dtype/device, NaN/±inf in either latent tensor (including masked candidates), and invalid mask dtype/shape/device.
   - GREEN: add narrow public-boundary validation.
6. **Public API**
   - RED: import from `acs_jepa`.
   - GREEN: export `ArgumentReconstructionHead`.

## Verification

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_argument_reconstruction_head.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_applicability_head.py \
         packages/acs-jepa-core/tests/test_action_contrastive_loss.py \
         packages/acs-jepa-core/tests/test_action_vicreg_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/architectures.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_argument_reconstruction_head.py

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  python -m compileall -q packages/acs-jepa-core/src/acs_jepa

git diff --check
python /opt/data/skills/software-development/phase-gated-implementation/scripts/static_diff_scan.py
```

## Stage 2C1 acceptance criteria

- Public role-aware candidate-scoring head satisfies the reviewed tensor contract.
- Candidate scoring is permutation-equivariant over objects but role-sensitive.
- Invalid candidates are excluded with `-inf` and receive zero gradients.
- Constructor/forward validation and public exports are tested.
- Existing applicability and Stage 2A/2B modules remain green.
- No loss, data, training, checkpoint, decoder, planner, or oracle behavior is added.
- Independent implementation review returns PASS before a signed commit.

## Gate

Implementation is blocked until independent plan review returns PASS. Blocking findings require revision and re-review. Any change after implementation-review PASS requires fresh verification and review.
