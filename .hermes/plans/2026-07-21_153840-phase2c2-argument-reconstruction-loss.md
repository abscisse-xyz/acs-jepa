# Phase 2C2 Argument Reconstruction Loss Plan

Governing documents:

- `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2 argument-identifiability objective.
- `.hermes/plans/2026-07-21_143024-phase2-roadmap-stage2a-action-vicreg.md`.
- `.hermes/plans/2026-07-21_151853-phase2c1-argument-reconstruction-head.md`.

Completed prerequisites:

- Stage 2A action VC/VICReg helper: `2ec072e`.
- Stage 2B same-state contrastive helper: `490d980`.
- Stage 2C1 role-aware argument reconstruction head: `e8c44a8`.

Splitting the argument-identifiability module into head (2C1) and loss (2C2) does not change the original Phase 2 deliverables or acceptance criteria.

## Objective

Add a standalone role-aware cross-entropy loss over problem-local candidate-object logits. The loss must select active argument rows before cross-entropy, enforce type-valid candidate and target semantics, exclude inactive roles and invalid candidates from objective/diagnostics/gradients, and provide finite strict-identifiability diagnostics.

Targets, masks, and logits are externally supplied. Data construction and trainer/config/checkpoint integration remain later stages.

## Public contract

```python
@dataclass(frozen=True)
class ArgumentReconstructionLossOutput:
    total: Tensor
    role_accuracy: Tensor
    competitive_role_accuracy: Tensor
    mean_target_margin: Tensor
    num_active_roles: int
    num_competitive_roles: int

class ArgumentReconstructionLoss(nn.Module):
    def forward(
        self,
        logits: Tensor,          # float [B, R, O]
        target_indices: Tensor,  # long [B, R], -1 iff inactive
        argument_mask: Tensor,   # bool [B, R]
        candidate_mask: Tensor,  # bool [B, R, O]
    ) -> ArgumentReconstructionLossOutput: ...
```

`R` is the configured/padded role count and `O` the padded problem-local object count. The loss does not own either dimension.

## Mathematical definition

For active role rows `A = {(b,r): argument_mask[b,r]}`:

1. Gather `logits[A]`, `targets[A]`, and `candidate_mask[A]` before cross-entropy.
2. Set invalid-candidate logits to `-inf` with `masked_fill`.
3. Compute mean categorical cross-entropy over the active roles.

For each active role:

```text
target margin = target logit - hardest active non-target candidate logit
strict correct = target logit > hardest active non-target candidate logit
```

A target tie is incorrect. If the target is the only active candidate, the role is correct by construction and is excluded from competitive accuracy, target-margin reduction, and `num_competitive_roles`. Hardest competitors and target-minus-competitor subtraction are computed only after selecting competitive rows; the implementation must not transiently evaluate `target - (-inf)` and overwrite it later.

`role_accuracy` averages strict correctness over all active roles. `competitive_role_accuracy` and `mean_target_margin` average only over active roles with at least one non-target candidate. If no competitive role exists, both competitive diagnostics are graph-connected finite zeros. Reporting both accuracies prevents singleton roles from overstating learned identifiability.

## Compute and output dtype contract

```text
input float16  -> compute/output float32
input bfloat16 -> compute/output float32
input float32  -> compute/output float32
input float64  -> compute/output float64
```

All tensor outputs are rank-0 scalars in the compute dtype in both normal and empty-active branches. Count fields are Python `int` values.

Promotion improves ordinary low-precision stability but does not make arbitrary finite extremes safe: bfloat16/float32 extremes can overflow float32 derived differences, and opposite-sign float64 extremes can overflow float64. After CE and each reduced diagnostic are computed, every tensor output must be checked for finiteness. Non-finite derived outputs raise `ValueError` rather than escaping as NaN/±inf. Positive-path dtype tests cover every supported input dtype. Opposite-sign maximum finite float16 logits must produce finite promoted float32 outputs; corresponding bfloat16, float32, and float64 fixtures must raise when their derived CE or margin is non-finite. No artificial magnitude threshold is introduced: rejection is based only on derived non-finiteness.

## Empty-active-role behavior

A batch containing no active argument roles is valid, e.g. a batch of zero-arity actions. It returns:

- graph-connected rank-0 scalar zero `total` in the compute dtype. To avoid overflow from summing many extreme finite values, use at most the first valid candidate logit (`compute_logits[candidate_mask][:1].sum() * 0.0`); if the selection is empty, the empty indexed sum remains graph-connected;
- finite rank-0 scalar zero diagnostics in the compute dtype;
- zero Python `int` counts.

No cross-entropy is evaluated on empty or all-`-inf` rows.

## Validation

- `logits` rank 3 `[B,R,O]` with non-empty `B`, `R`, and `O`; floating dtype.
- `target_indices` shape `[B,R]`, dtype `torch.long`.
- `argument_mask` bool `[B,R]`.
- `candidate_mask` bool `[B,R,O]`.
- All four tensors share a device.
- NaN and `+inf` logits are rejected globally, including at mask-false entries.
- Candidate-mask-true logits must be finite. Mask-false logits may be finite or `-inf`; `-inf` is forbidden at mask-true entries. This admits Stage 2C1's sentinel while allowing standalone callers to provide finite raw logits at excluded positions.
- Active roles must have at least one candidate.
- Active targets must lie in `[0,O)` and be candidate-mask true.
- Inactive targets must equal `-1` exactly.
- Candidate masks on inactive roles may be all false; inactive rows are never sent to CE.
- The loss never detaches logits.

Validation ordering is mandatory:

1. validate all ranks, shapes, non-empty dimensions, dtypes, and devices;
2. validate global logit values and mask-position finiteness;
3. validate `target == -1` exactly iff inactive and validate every active target is in `[0,O)`;
4. gather active logits, targets, and candidate masks;
5. only then index gathered candidate masks to verify each target is candidate-mask true;
6. compute masked CE and diagnostics.

No gather, advanced indexing by targets, or target-dependent mask lookup may occur before active target range validation.

## Candidate permutation contract

If candidates are permuted, callers must apply the same permutation to logits and `candidate_mask` and remap every active `target_indices` value through the inverse/new-position mapping. For `new_logits = old_logits[..., permutation]`, require `new_target = inverse_permutation[old_target]`, where `inverse_permutation = argsort(permutation)`. A non-self-inverse permutation TDD oracle will prove every output is invariant.

## Scope

Modify only:

- `packages/acs-jepa-core/src/acs_jepa/losses.py`
- `packages/acs-jepa-core/src/acs_jepa/__init__.py`
- `packages/acs-jepa-core/tests/test_argument_reconstruction_loss.py`

## Explicit non-goals

- No changes to `ArgumentReconstructionHead` after its implementation-review PASS.
- No target/mask data generation or collation.
- No encoder construction or gradient-detach policy.
- No trainer, config, optimizer, checkpoint, CLI, simulator, oracle, decoder, or planner changes.
- No smoke retraining or empirical Phase 2 acceptance claim.

## TDD sequence

Use small vertical RED → GREEN cycles. Passing behavior introduced incidentally becomes regression coverage and is not misreported as an observed RED.

1. **Manual CE oracle and output contract**
   - RED: one active competitive role matches an independently calculated categorical CE, all-role and competitive strict accuracies, competitive target margin, Python counts, rank-0 shapes, and compute dtype while an inactive all-`-inf` row is never evaluated.
   - GREEN: add output dataclass, dtype promotion, and active-row gather before masked CE.
2. **Masking and gradients**
   - RED: invalid candidates cannot affect CE or diagnostics and receive exact zero gradients from `output.total.backward()`; inactive finite rows and inactive all-`-inf` rows receive exact zero gradients; active target/competitor logits in a nondegenerate competitive row receive finite nonzero gradients.
   - GREEN: apply candidate masking after active-row gather and compute diagnostics only from active candidates.
3. **Strict ties and singleton candidates**
   - RED: target/competitor ties fail both strict accuracies with zero margin; singleton target roles are correct in all-role accuracy, finite, excluded from competitive reductions/counts, and may correctly have zero CE gradient.
   - GREEN: select competitive rows before any hardest-competitor max or subtraction; return graph-connected competitive zeros when none exist.
4. **No-active-role boundary**
   - RED: all-inactive batches with finite candidate logits and with all-`-inf` masked logits return graph-connected rank-0 zeros in every supported compute dtype and zero Python counts; backward succeeds with exact zero gradients and no NaN.
   - GREEN: branch before CE using at most one candidate-mask-selected compute logit for connected zero.
5. **Permutation invariance**
   - RED: a non-self-inverse candidate permutation, correspondingly permuted mask, and explicit `new_target = argsort(permutation)[old_target]` preserve every output.
   - GREEN: maintain index-based target semantics without positional assumptions.
6. **Validation and numerical boundaries**
   - RED/GREEN groups for rank/shape/non-empty/dtype/device checks and safe target validation before indexing.
   - RED/GREEN for NaN and `+inf` rejection even at mask-false positions; `-inf` accepted only mask-false; finite mask-false values accepted; active rows with no candidate; active out-of-range/masked targets; and `target == -1` exactly iff inactive.
   - RED/GREEN positive paths for float16, bfloat16, float32, and float64 with exact rank-0 output dtypes in normal and empty branches.
   - RED/GREEN opposite-sign maximum finite inputs: float16 must produce finite promoted float32 CE/margin, while bfloat16, float32, and float64 must raise `ValueError` when their derived CE or competitive margin is non-finite. The helper uses no arbitrary magnitude cutoff.
7. **Public API**
   - RED: imports from `acs_jepa` fail before export.
   - GREEN: export both public types.

## Verification

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_argument_reconstruction_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_argument_reconstruction_head.py \
         packages/acs-jepa-core/tests/test_action_contrastive_loss.py \
         packages/acs-jepa-core/tests/test_action_vicreg_loss.py \
         packages/acs-jepa-core/tests/test_applicability_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/losses.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_argument_reconstruction_loss.py

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  python -m compileall -q packages/acs-jepa-core/src/acs_jepa

git diff --check
python /opt/data/skills/software-development/phase-gated-implementation/scripts/static_diff_scan.py
```

## Stage 2C2 acceptance criteria

- Numerical CE and all-role/competitive diagnostics match independent oracles.
- Normal and empty branches return rank-0 tensors in the documented compute dtype and Python `int` counts.
- Extreme finite inputs either produce finite derived outputs or are rejected before non-finite outputs escape.
- Inactive roles are selected out before CE.
- Invalid candidates and inactive rows contribute no objective, diagnostics, or gradients.
- Strict ties and singleton roles have explicit finite behavior.
- Empty-active batches return graph-connected finite zeros.
- Candidate/target validity and permutation semantics are enforced/tested.
- Existing Stage 2A/2B/2C1 and applicability helpers remain green.
- No data/trainer/config/checkpoint/decoder/planner behavior is added.
- Independent implementation review returns PASS before a signed commit.

## Gate

Implementation is blocked until an independent plan review returns PASS. Blocking findings require plan revision and re-review. Any change after implementation-review PASS requires fresh verification and review.
