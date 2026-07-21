# Phase 2 Roadmap and Stage 2A Action VICReg Helper Plan

Governing specification: `script/ACTION_LATENT_SOLUTION_SPEC.md`, especially Phase 2 (lines 598–621), the action VICReg/SIGReg rationale, and Tracks A–C.

## Phase 2 objective

Train ACS-JEPA with explicit action-identifiability auxiliaries while leaving the existing JEPA transition objective structurally unchanged. Phase 2 is complete only when the configured component smoke run executes and diagnostics show whether action-latent rank, same-schema margins, role-object recovery, and applicability improve. Splitting Phase 2 does not reduce its original deliverables or acceptance criteria.

## Phase 2 staged roadmap

Every stage follows its own plan → plan review → TDD implementation → implementation review → signed commit gate. A later stage cannot begin implementation until the previous stage passes review and is committed.

1. **2A — standalone action VICReg helper**
   - Add an action-latent variance/covariance regularizer with explicit scalar diagnostics.
   - No config, trainer, data, checkpoint, or planner changes.
2. **2B — standalone same-state hard-negative contrastive helper**
   - Define and test the positive/negative tensor contract and anchor semantics.
   - Keep candidate generation and trainer wiring out of this helper slice.
3. **2C — argument-identifiability auxiliary modules**
   - Plan/review role-aware, problem-local argument reconstruction separately; split head and loss if needed.
   - Reuse the existing applicability head/loss rather than duplicating it.
4. **2D — training-data construction**
   - Build deterministic trace-positive and hard-negative tensors/labels for normal training batches.
   - Simulator applicability remains an offline label oracle only and never a production action generator or model feature.
5. **2E — auxiliary integration**
   - Wire reviewed action VICReg, contrastive, applicability, and argument terms into the loss/trainer/model/config/checkpoint path in small disabled-by-default sub-stages.
   - Add all Phase 2 `model.loss` fields from the governing specification, including disabled `action_sigreg_coeff`; SIGReg implementation remains deferred until evidence warrants it.
   - Reconcile the required `model.loss.applicability_coeff` with the existing `trainer.applicability_loss_weight` path in that stage's reviewed plan so there is one documented authority rather than two silently divergent coefficients.
6. **2F — component configuration and smoke execution**
   - Add the required component test config derived from the inverse-dynamics smoke config.
   - Run the fixed smoke train/eval protocol and rerun Phase 0/1 diagnostics.
   - Report effective action-VICReg sample counts per batch/aggregate because short batches may make the regularizer statistically weak.
7. **2G — Phase 2 acceptance assessment**
   - Compare transition loss, action-latent std/effective rank, same-schema margins, role-object probe accuracy, and applicability AUROC/margins.
   - If global VICReg improves only schema/global geometry, do not declare success: use the reviewed contrastive/applicability/argument terms to target binding identifiability.

Phase 3 decoder scoring, Phase 4 planner manifold regularization, broad tuning, production oracle enumeration, and free-latent planner changes remain out of scope for all Phase 2 stages.

---

# Stage 2A detailed plan: standalone action VICReg helper

## Objective

Add a small reusable `ActionVICRegLoss` that regularizes encoded action-latent samples with VICReg-style per-dimension variance and covariance terms. This is the first implementation step recommended by the research specification because equivalent state-latent machinery already exists and Phase 0 measured severe action-latent anisotropy.

There is no paired-view invariance term in this stage: grounded-action identity has no augmentation pair in the current data contract. The existing JEPA transition prediction objective supplies predictive pressure; later same-state contrastive/applicability/argument auxiliaries supply identifiability pressure. `action_vicreg_coeff` will be the outer integration coefficient in a later stage.

## Research/spec mapping

- VICReg (Bardes et al., arXiv:2105.04906): use hinge variance and off-diagonal covariance penalties to prevent dimensional collapse and anisotropy.
- ACS-JEPA specification: start with action VICReg, instrument it, and judge it by decoding-relevant diagnostics rather than loss decrease alone.
- Phase 0 evidence: effective action-latent rank is approximately 1.5–3 and variance is overwhelmingly schema dominated.
- Known limitation: global VC regularization alone may separate schemas without improving argument bindings. Stage 2A therefore provides infrastructure, not Phase 2 empirical acceptance.
- SIGReg/Weak-SIGReg remains disabled/deferred until VICReg evidence is available.

## Scope

Modify only:

- `packages/acs-jepa-core/src/acs_jepa/losses.py`
- `packages/acs-jepa-core/src/acs_jepa/__init__.py`
- `packages/acs-jepa-core/tests/test_action_vicreg_loss.py`

Add:

```python
@dataclass(frozen=True)
class ActionVICRegLossOutput:
    total: Tensor
    std_penalty: Tensor
    covariance_penalty: Tensor
    num_samples: int

class ActionVICRegLoss(nn.Module):
    def __init__(
        self,
        *,
        std_coeff: float = 1.0,
        cov_coeff: float = 1.0,
        std_margin: float = 1.0,
    ) -> None: ...

    def forward(self, action_latents: Tensor) -> ActionVICRegLossOutput: ...
```

Tensor contract:

- rank 2 `[N, D_a]` is treated as `N` samples;
- rank 3 `[B, K, D_a]` is flattened to `[B*K, D_a]`;
- other ranks, empty sample/feature dimensions, and non-floating tensors are rejected;
- one-sample input is accepted and returns graph-connected zero std/covariance penalties, so `total.backward()` is valid and produces a zero input gradient;
- one-dimensional action features (`D_a == 1`) are accepted: the std penalty remains defined and covariance is a finite graph-connected zero because no off-diagonal entries exist;
- `total = std_coeff * std_penalty + cov_coeff * covariance_penalty`;
- for non-degenerate multi-sample input, gradients flow to non-detached action latents and at least one input-gradient element is finite and nonzero;
- internal coefficients must be finite and non-negative; `std_margin` must be finite and positive.

Reuse `HingeStdLoss`, `CovarianceLoss`, and `_sample_matrix`; do not duplicate formulas. Handle only the two undefined/non-differentiable helper boundaries in the wrapper with input-dependent zeros (`samples.sum() * 0.0`): `N == 1` for both terms and `D_a == 1` for covariance. This leaves existing `GraphVCLoss` behavior unchanged.

`std_penalty` is the VICReg hinge penalty, not an empirical standard-deviation value. `covariance_penalty` follows this repository's existing `CovarianceLoss` normalization: mean squared off-diagonal covariance. Original VICReg uses a different dimension normalization, so paper coefficients are not assumed transferable.

## Explicit non-goals

- No `GraphJEPALossModule` integration.
- No config fields or model-builder changes.
- No dataloader, negative sampler, applicability, contrastive, or argument-reconstruction changes.
- No checkpoint changes.
- No smoke retraining or claims of same-schema improvement.
- No planner/decoder behavior changes.
- No SIGReg implementation.

## TDD sequence

Use vertical RED → GREEN cycles:

1. **Composition/output contract**
   - RED: rank-2 input returns named scalar std/covariance/total terms and sample count; total equals the configured weighted sum.
   - RED numerical oracle: on deterministic data with both nonzero hinge and nonzero off-diagonal covariance, `std_penalty` and `covariance_penalty` equal separately invoked `HingeStdLoss` and `CovarianceLoss`. This catches swapped, duplicated, or accidentally zero terms.
   - GREEN: add the output dataclass and minimal module composition.
2. **Temporal flattening**
   - RED: rank-3 `[B,K,D]` output equals reshaped rank-2 output and records `B*K` samples.
   - GREEN: use `_sample_matrix`.
3. **Collapse and gradients**
   - RED: collapsed multi-sample latents have positive std penalty.
   - RED separately: a non-degenerate multi-sample input produces at least one finite, nonzero action-latent gradient after backward.
   - GREEN: preserve autograd through existing helpers.
4. **Small-batch behavior**
   - RED: one sample yields graph-connected zero std/covariance/total, records one sample, and permits backward with zero input gradient.
   - RED: `D_a == 1` yields finite std/total, graph-connected zero covariance, and permits backward.
   - GREEN: return input-dependent zeros at these wrapper boundaries without changing shared helpers.
5. **Validation**
   - RED: reject ranks other than 2/3, empty dimensions, non-floating inputs, negative/non-finite coefficients, and non-positive/non-finite margin.
   - GREEN: add constructor/forward validation.
6. **Public API**
   - RED: import `ActionVICRegLoss` and output type from `acs_jepa`.
   - GREEN: export both names.

## Verification

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_action_vicreg_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_graph_losses.py \
         packages/acs-jepa-core/tests/test_applicability_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/losses.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_action_vicreg_loss.py

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  python -m compileall -q packages/acs-jepa-core/src/acs_jepa

git diff --check
python /opt/data/skills/software-development/phase-gated-implementation/scripts/static_diff_scan.py
```

## Stage 2A acceptance criteria

- Public standalone action VICReg helper exists with documented rank-2/rank-3 semantics.
- Outputs expose unweighted `std_penalty`/`covariance_penalty` diagnostics and weighted total.
- Validation and small-batch behavior are explicit and tested.
- Single-sample and one-feature edge cases are finite and graph-connected; non-degenerate multi-sample inputs receive finite, nonzero gradients.
- Existing graph-state VC behavior is unchanged.
- Targeted/adjacent tests and static checks pass.
- Independent implementation review returns PASS before a signed commit.

## Gate

No Stage 2A production code may be written until independent plan review returns PASS. Any blocking review finding must be incorporated and re-reviewed. Any code change after implementation-review PASS invalidates that PASS and requires fresh review.
