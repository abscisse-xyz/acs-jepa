# Phase 2B Same-State Hard-Negative Contrastive Loss Plan

Governing documents:

- `script/ACTION_LATENT_SOLUTION_SPEC.md`, Track B and Phase 2.
- `.hermes/plans/2026-07-21_143024-phase2-roadmap-stage2a-action-vicreg.md`.

Stage 2A is committed as `2ec072e`. Phase 2 deliverables and acceptance criteria remain unchanged by sub-staging.

## Objective

Add a standalone InfoNCE-style loss for one transition anchor, its true grounded-action latent, and explicit same-source-state negative action latents. The helper supplies the action-identifiability objective required by Track B without deciding how anchors or negatives are generated.

The loss is an auxiliary helper only. It will be integrated with deterministic training negatives and an inverse-dynamics/transition-derived anchor in later reviewed stages.

## Research/spec mapping

The governing specification proposes:

```text
positive = q_phi(true_action, state)
negatives = q_phi(negative_action_i, state)
anchor = inverse_dynamics_or_transition_delta(state, next_state)
L = CE(anchor dot [positive, negatives] / tau, label=positive)
```

Stage 2B uses cosine-normalized dot products before temperature scaling. Normalization prevents the objective from being minimized only through latent-norm inflation and complements Stage 2A's variance/covariance geometry. The helper consumes precomputed anchors and action latents and never calls a simulator, sampler, encoder, trainer, or planner.

This stage does not define the eventual anchor producer. The later integration plan must choose and test whether the anchor is the existing inverse-dynamics prediction or another transition-derived representation. It must state separately whether gradients flow into the anchor producer, source/next-state encoders, positive action encoding, and negative action encodings.

## Scope

Modify only:

- `packages/acs-jepa-core/src/acs_jepa/losses.py`
- `packages/acs-jepa-core/src/acs_jepa/__init__.py`
- `packages/acs-jepa-core/tests/test_action_contrastive_loss.py`

Add:

```python
@dataclass(frozen=True)
class ActionContrastiveLossOutput:
    total: Tensor
    positive_similarity_mean: Tensor
    hardest_negative_similarity_mean: Tensor
    positive_negative_margin: Tensor
    top1_accuracy: Tensor
    num_examples: int
    num_negatives: int

class ActionContrastiveLoss(nn.Module):
    def __init__(self, *, temperature: float = 0.1) -> None: ...

    def forward(
        self,
        anchor_latents: Tensor,             # [N, D_a]
        positive_action_latents: Tensor,    # [N, D_a]
        negative_action_latents: Tensor,    # [N, M, D_a]
        negative_mask: Tensor | None = None,# bool [N, M]
    ) -> ActionContrastiveLossOutput: ...
```

## Tensor and metric contract

- All latent tensors must be floating point, contain only finite values, and be on the same device with the same dtype. Non-finite masked negatives are also rejected rather than silently ignored.
- `anchor_latents` and positives must have identical non-empty rank-2 shape `[N,D_a]`.
- Negatives must have non-empty rank-3 shape `[N,M,D_a]` with matching `N,D_a`.
- Optional `negative_mask` must be bool `[N,M]`, reside on the same device as the latent tensors, and retain at least one negative in every row. Device mismatches are rejected rather than copied implicitly.
- Reject every anchor/positive/negative vector whose stable float32/float64 L2 norm is less than or equal to `1e-12`; this applies to masked negatives too. Cosine similarity is undefined for zero vectors and the contrastive objective cannot escape an exactly zero/collapsed normalized state by itself.
- Compute normalization, similarities, logits, CE, and diagnostics in float32 for float16/bfloat16/float32 inputs and in float64 for float64 inputs. This prevents the default normalization epsilon from underflowing in float16 while preserving float64 precision.
- Normalize each non-degenerate vector with a stable scaled-norm algorithm in the compute dtype: divide by its maximum absolute component, compute the L2 norm of the bounded scaled vector, then divide by that norm. This avoids overflow for finite extreme float32/float64 vectors and underflow for low-precision inputs. The resulting vectors must have finite unit norm.
- Build logits `[N,1+M]` with the positive in column zero. Inactive negatives receive `-inf` only for softmax/ranking and no gradient.
- `total` is mean cross entropy with target column zero.
- Similarity diagnostics are unscaled cosine similarities:
  - positive mean;
  - per-row hardest active negative mean;
  - mean `positive - hardest_negative` margin.
- `top1_accuracy` is the mean strict comparison `positive_similarity > hardest_negative_similarity`; ties are failures rather than benefiting from positive column ordering.
- `num_negatives` is the number of active negative pairs, not `M`.
- No tensor is detached by the helper; integration controls detach semantics.
- Constructor validation requires finite positive `temperature`. At forward time, require it to be at least `torch.finfo(compute_dtype).tiny` and verify that every active scaled logit is finite before CE. This makes tiny-temperature behavior explicit for float16/bfloat16/float32/float64 inputs.
- `top1_accuracy` and all other floating outputs use the compute dtype (float32 for low-precision/float32 inputs, float64 for float64 inputs).

## Explicit non-goals

- No negative sampling or applicability labeling.
- No action encoding or anchor construction.
- No `GraphJEPALossModule`, trainer, config, optimizer, dataloader, or checkpoint wiring.
- No argument reconstruction or applicability integration.
- No decoder/planner changes and no production simulator oracle.
- No smoke retraining or Phase 2 acceptance claim.
- Later integration/acceptance diagnostics must compare both cosine and L2 same-schema margins because this loss directly optimizes angular separation while deployed latent decoding may use L2.

## TDD sequence

Each grouped item below is implemented as separate small RED → GREEN cycles; passing behavior discovered incidentally is recorded as regression coverage rather than falsely claimed as RED.

Use vertical RED → GREEN cycles:

1. **Numerical InfoNCE contract**
   - RED: deterministic orthogonal/antipodal fixture matches manually computed cosine logits and cross entropy; output terms/counts have the documented scalar semantics.
   - RED: valid float16 and bfloat16 inputs produce finite float32 loss/diagnostics, float32 remains float32, and float64 produces float64 outputs.
   - RED: extreme finite float32, bfloat16, and float64 vectors normalize to finite unit vectors and produce finite outputs.
   - GREEN: implement output dataclass, cosine logits, and CE.
2. **Masking**
   - RED: changing a masked negative cannot change loss/metrics or produce gradient for that negative; variable active-negative counts are reported correctly.
   - RED: an all-`True` mask is equivalent to no mask, while non-finite masked negatives are still rejected by the public finite-input contract.
   - RED: masked zero/near-zero negatives are rejected; mask/latent device mismatch is rejected on CUDA-capable test environments.
   - GREEN: apply strict bool masking before CE and hardest-negative reduction.
3. **Strict ranking metrics**
   - RED: clear positives yield positive margins/accuracy one; tied positive/hardest-negative rows count as incorrect.
   - GREEN: compute diagnostics from unscaled cosine similarities using strict `>`.
4. **Gradients**
   - RED: a non-degenerate fixture yields finite, nonzero gradients for anchors, positives, and active negatives.
   - GREEN: preserve autograd and avoid detaches.
5. **Validation**
   - RED: reject invalid/non-finite temperature, forward-time temperatures below the compute-dtype bound, rank/shape mismatch, empty dimensions, dtype/device mismatch, non-floating latents, zero/near-zero vectors, non-bool/wrong-shape mask, and rows with no active negative.
   - RED: independently reject NaN, positive infinity, and negative infinity in anchor, positive, and negative tensors, including masked negatives.
   - RED: float16 and bfloat16 zero-vector fixtures fail with the documented `ValueError` rather than producing NaN; tiny-temperature float16/float32 fixtures fail before CE.
   - RED: test norm exactly `1e-12` and immediately above it, and temperature exactly `torch.finfo(compute_dtype).tiny` and immediately below it.
   - GREEN: add narrow public-boundary validation.
6. **Public API**
   - RED: both types import from `acs_jepa`.
   - GREEN: export them.

## Verification

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_action_contrastive_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_action_vicreg_loss.py \
         packages/acs-jepa-core/tests/test_graph_losses.py \
         packages/acs-jepa-core/tests/test_applicability_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/losses.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_action_contrastive_loss.py

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  python -m compileall -q packages/acs-jepa-core/src/acs_jepa

git diff --check
python /opt/data/skills/software-development/phase-gated-implementation/scripts/static_diff_scan.py
```

## Stage 2B acceptance criteria

- Standalone public contrastive loss implements the reviewed same-state positive/negative contract.
- Numerical CE, cosine diagnostics, masking, strict ranking, gradients, and validation are tested.
- Masked negatives cannot affect outputs or gradients.
- Existing Stage 2A and graph/applicability losses remain unchanged and green.
- No future-stage data, training, decoder, or planner behavior is introduced.
- Independent implementation review returns PASS before a signed commit.

## Gate

Implementation is blocked until independent plan review returns PASS. Blocking findings require plan revision and re-review. Any code edit after implementation-review PASS requires fresh verification and implementation review.
