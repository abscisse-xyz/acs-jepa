# Branch D Candidate Rollout Boundary Implementation Plan

> **For Hermes:** Execute only after an independent plan review returns PASS. Use strict behavioral RED/GREEN and preserve exact failing commands/output in a sibling RED/GREEN record.

**Goal:** Add the first Branch D vertical slice: a public model-level API that rolls out externally supplied grounded candidate-action sequences while keeping learned action latents internal, explicitly non-invertible transition controls.

**Architecture:** Candidate identity, generation, applicability, search, and execution remain outside `GraphJEPA`. The caller supplies already tensorized candidate sequences `[B,H,...]`; at each step `GraphJEPA` encodes the grounded candidate against that candidate's current predicted source state and applies the predictor. The API returns predicted latent states and internal control latents, never recovered grounded actions or validity claims.

**Tech Stack:** Python 3.13, PyTorch, PyTorch Geometric, pytest, Ruff.

---

## 1. Interruption point and governing decision

- Clean signed base: `63534e96ebe02bbb7f60ab66bd05d041830b5ea0`.
- Updated Phase 0 deterministic decision: `BRANCH_D_ABSTRACT_ACTIONS`.
- Evidence motivating Branch D:
  - fixed Phase 2 hard-negative transition-equivalence rate `1.0` over all 44 eligible groups;
  - raw-symbolic and hybrid discrete rankers passed;
  - latent-applicability, role/object, and teacher-forced latent-transition rankers failed the fixed ranking gate.
- No Branch D implementation began after the decision. This plan starts at the first post-decision implementation gate.

The new API is not a planner and does not imply that a continuous latent identifies an executable PDDL action.

## 2. Related-work rationale

Primary-work design patterns support keeping executable choices explicit while using learned dynamics for decision-relevant rollout quantities:

- Grimm et al., *The Value Equivalence Principle for Model-Based Reinforcement Learning*, NeurIPS 2020, arXiv:2011.03506: planning models need preserve decision-relevant value updates, not uniquely reconstruct all transitions.
- Oh et al., *Value Prediction Network*, NeurIPS 2017, arXiv:1707.03497: abstract option-conditioned dynamics predict planning-relevant values.
- Schrittwieser et al., *MuZero*, arXiv:1911.08265: search retains explicit actions while the learned model predicts planning quantities without observation reconstruction.
- Shen et al., *Learning Domain-Independent Planning Heuristics with Hypergraph Networks*, arXiv:1911.13101, and Chen et al., *Learning Domain-Independent Heuristics for Grounded and Lifted Planning*, arXiv:2312.11143: symbolic search owns successor generation and validity while learned models provide heuristic guidance.
- Takata and Fukunaga, *Plausibility-Based Heuristics for Latent Space Classical Planning*, arXiv:2306.11434: latent plans can contain invalid/hallucinatory states, so learned rollout plausibility must not establish symbolic applicability.

These works do not prove this exact ACS-JEPA/PDDL design. They justify the boundary tested by this slice; later search and score-combination choices require separate evidence and review.

## 3. Scope

### In scope

- A public `GraphJEPA.rollout_grounded_candidates(...)` API.
- A frozen output dataclass with:
  - original initial latent state;
  - final predicted latent state;
  - one predicted state per horizon step;
  - internal control latents `[B,H,D_a]`.
- Exact batch/horizon/shape validation.
- Explicit rejection of any stateful action encoder whose temporal action context is not exactly one, including `context_steps=None` (all prior actions) and values greater than one; streaming action-GRU state is not implemented here. Stateless encoders that expose no `context_steps` attribute remain valid.
- Exact source causality: action at step `t` is encoded against predicted state `t`, not against the initial state or an observed future state.
- Object identity and batch-membership preservation checks through every prediction.
- README documentation of the non-invertible control contract.

### Explicitly deferred

- No `ActionDecoder`, nearest-neighbor latent recovery, categorical CEM/MPPI, or free-continuous action optimization changes.
- No `ActionDecodingSpace`, candidate generator, applicability filter/oracle, simulator, search algorithm, beam width, score combination, or CLI planner mode.
- No use of `applicable_actions()` in production code.
- No planner refactor to consume this API.
- No Phase 1 training, new head/loss, model retraining, checkpoint changes, or tuning.
- No claim that predicted latent states are symbolically valid successor states.
- No options/skills, multi-step executor, or latent-to-action inverse.

## 4. Public contract

Add to `packages/acs-jepa-core/src/acs_jepa/jepa.py`:

```python
@dataclass(frozen=True)
class GraphJEPACandidateRolloutOutput:
    initial_state: JEPALatentState
    final_state: JEPALatentState
    predicted_states: tuple[JEPALatentState, ...]
    control_latents: Tensor
```

Add to `GraphJEPA`:

```python
def rollout_grounded_candidates(
    self,
    initial_state: JEPALatentState,
    action_tensors: dict[str, Tensor],
) -> GraphJEPACandidateRolloutOutput:
    ...
```

Input semantics:

- `initial_state` is non-temporal and has candidate batch size `B`.
- Every action tensor has at least two dimensions and identical leading dimensions `[B,H]`.
- `B >= 1`, `H >= 1`.
- Candidate tensors use problem-local IDs; their vocabulary/type correctness remains the external tensorizer's responsibility.
- If several candidates share one observed source state, the caller repeats that encoded source into a candidate batch before this call.
- `action_encoder.context_steps`, when present, must be exactly `1`. Both `None` (all available prior actions during training) and any integer other than one raise a clear `ValueError`; repeated one-step calls would reset the GRU and are not faithful to those checkpoints. Stateless encoders with no `context_steps` attribute are accepted.

Step semantics:

```text
state_0 = initial_state
for t in 0..H-1:
    candidate_t = {name: tensor[:, t] for each action tensor}
    control_t = action_encoder(candidate_t, state_t)
    state_{t+1} = predictor(state_t, control_t)
```

Output semantics:

- `control_latents.shape == [B,H,D_a]`.
- `len(predicted_states) == H`.
- `final_state is predicted_states[-1]`.
- Every predicted state preserves graph shape `[B,D_z]`, object shape `[N_obj,D_z]`, and the exact metadata tensor/storage identities for `object_ids` and `object_batch`; equal cloned metadata is rejected.
- The result contains no `GroundAction`, decoded action, applicability label, or execution result.
- This model method remains differentiable; inference callers own `torch.no_grad()` and evaluation mode.

## 5. Strict RED/GREEN implementation

Create `.hermes/plans/2026-08-04_092228-branchd-candidate-rollout-red-green.md` before tests. Record every command, expected failure, actual failure excerpt, fix, and GREEN output. An import failure because the new symbol is absent is acceptable only for the first API RED; every subsequent RED must target the next concrete behavior.

### Task 1: Establish the public API and exact recurrence

**Files:**
- Modify: `packages/acs-jepa-core/tests/test_graph_jepa_components.py`
- Modify: `packages/acs-jepa-core/src/acs_jepa/jepa.py`
- Modify: `packages/acs-jepa-core/src/acs_jepa/__init__.py`
- Create: `.hermes/plans/2026-08-04_092228-branchd-candidate-rollout-red-green.md`

**RED behaviors:**

Before adding any production symbol or method, add both tests below.

First, add a test with `B=2`, `H=3` and instrumented action encoder/predictor. Assert:

1. the public dataclass and method exist;
2. candidate step `t` is sliced exactly as `tensor[:, t]` for every field;
3. each action-encoder call receives the immediately preceding predicted state;
4. controls stack in candidate/horizon order as `[B,H,D_a]`;
5. predicted-state order and exact recurrence are correct;
6. `final_state is predicted_states[-1]` and `initial_state` is preserved.

The same test also asserts the public method signature accepts only `initial_state` and already tensorized `action_tensors` after `self`, and that the frozen output fields are exactly `initial_state`, `final_state`, `predicted_states`, and `control_latents`.

Second, add the production-module integration test before implementation. It uses the real `ActionEncoder(context_steps=1)` and a real residual latent predictor with `B=2`, `H=3`, candidate-expanded object metadata, and asserts expected control/final-state shapes plus gradient propagation through both modules.

Run:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-core pytest \
  packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_uses_each_predicted_source_and_preserves_order \
  packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_supports_real_action_encoder_predictor_and_gradients -q
```

Expected initial RED: both nodes fail because `GraphJEPACandidateRolloutOutput` / `rollout_grounded_candidates` is absent. This is the one authorized missing-public-API RED. Record the exact collection/runtime failure. Task 2 introduces every subsequent test one behavior at a time against the importable Task 1 implementation, so later REDs must be contract-specific rather than missing-symbol failures.

**GREEN implementation:**

Implement only the dataclass, method, per-step slicing/encoding/prediction, control stacking, and public export needed by these two precommitted tests. Run the same two-node command and record GREEN before adding Task 2 validation tests.

### Task 2: Close shape, context, and identity contracts

**Files:**
- Modify: `packages/acs-jepa-core/tests/test_graph_jepa_components.py`
- Modify: `packages/acs-jepa-core/src/acs_jepa/jepa.py`

Add parametrized behavior tests, each observed RED before its fix:

- empty action mapping;
- non-tensor action value;
- action tensor with fewer than two dimensions;
- mismatched `[B,H]` prefixes between fields;
- candidate batch mismatch with `initial_state.graph_latent`;
- zero batch or zero horizon;
- temporal or empty `initial_state.graph_latent`;
- temporal `initial_state.object_latents`, object-latent width unequal to graph-latent width, or inconsistent object-row count;
- non-rank-one `object_ids`/`object_batch`, metadata lengths unequal to `N_obj`, or `object_batch` values outside `[0,B)`;
- malformed action-encoder output not shaped `[B,D_a]`;
- a later action-encoder call changing control width from the first `[B,D_a]` width;
- non-`JEPALatentState` predictor output;
- malformed predictor graph batch, graph rank, graph width, object rank, object rows, or object width;
- predictor mutation/reordering of `object_ids`;
- predictor mutation/reordering of `object_batch`;
- predictor cloning otherwise equal `object_ids` or `object_batch` instead of preserving exact tensor/storage identity;
- `action_encoder.context_steps is None`;
- `action_encoder.context_steps == 2`.

Required errors are deterministic `TypeError`/`ValueError` messages naming the violated contract. Do not silently reshape, truncate, broadcast, or regenerate identities.

Run the exact new nodes individually for RED, then:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-core pytest \
  packages/acs-jepa-core/tests/test_graph_jepa_components.py -q
```

### Task 3: Document the proven Branch D boundary

**Files:**
- Modify: `README.md`

Task 1 already precommits and drives RED/GREEN for the public signature, exact output fields, exact candidate slicing/order, sole manual recurrence, and real-module gradient integration. Do not add post-implementation tests that can pass immediately. Decoder/planner absence is enforced by that public signature, the bounded diff, and independent implementation review rather than a brittle monkeypatch test for components the method cannot receive.

Update README sections on action encoding, prediction, and planning to state:

- grounded actions may be encoded into internal abstract transition controls;
- candidate identity remains explicit and external;
- the existing `ActionDecoder` searches/selects a nearby type-valid candidate heuristically; it does not provide a reliable inverse, applicability, or candidate-identity guarantee;
- continuous controls have no supported inverse-to-grounded-action guarantee;
- this API supplies rollout predictions only and does not establish applicability or symbolic validity;
- search/generation/execution remain separate future Branch D work.

No README text may claim Branch D planning success.

## 6. Acceptance criteria

All must pass:

1. At least two externally supplied candidate sequences with horizon at least two roll out in one batch.
2. Every step encodes the exact candidate slice against the immediately prior predicted source state.
3. Controls are `[B,H,D_a]`; predicted states have length `H`; final state is exact.
4. Graph/object latent shapes remain exact, while `object_ids` and `object_batch` preserve exact tensor/storage identity across rollout; equal clones fail closed.
5. Invalid initial-state alignment, batch/horizon disagreement, changing control width, malformed predictor outputs, and action contexts `None` or greater than one fail with deterministic contract errors.
6. No public result or argument supports decoding, candidate generation, applicability, simulator execution, or grounded-action recovery.
7. Manual recurrence and API output are bit-exact for deterministic fake modules; a separate real-module test covers `ActionEncoder(context_steps=1)`, a production predictor, candidate-expanded object batching, and gradients.
8. Existing graph-component, planner, core, and scoped CLI tests remain green.
9. Ruff, `compileall`, and `git diff --check` pass.
10. An independent implementation review confirms no planner, decoder, CEM, tuning, Phase 1, or simulator scope entered the diff.

## 7. Verification commands

Focused:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-core pytest \
  packages/acs-jepa-core/tests/test_graph_jepa_components.py \
  packages/acs-jepa-core/tests/test_planner.py -q
```

Full core:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests -q
```

Scoped CLI regression, preserving the known unrelated tuning-YAML exclusion:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-cli pytest acs-jepa-cli/tests -q \
  -k 'not test_tuning_configs_load_and_keep_required_defaults'
```

Static checks:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/jepa.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_graph_jepa_components.py

python -m compileall -q \
  packages/acs-jepa-core/src/acs_jepa/jepa.py \
  packages/acs-jepa-core/src/acs_jepa/__init__.py \
  packages/acs-jepa-core/tests/test_graph_jepa_components.py

git diff --check
git status --short --branch
```

## 8. Lifecycle gates

1. Independent plan review must PASS before production/test implementation.
2. Implementation follows the exact behavioral RED/GREEN sequence above.
3. Official tuning or planner evaluation is not part of this slice.
4. Independent implementation review must PASS before commit.
5. Only one SSH-signed commit may be created after review PASS and explicit signature verification.
6. Push remains unauthorized unless separately approved.
7. Any later symbolic candidate generator, learned heuristic combination, search algorithm, CLI integration, or Phase 1 work requires a separate plan and review.

## 9. Risks and deliberate tradeoffs

- `action_context_steps=None` or any value other than one cannot be faithfully streamed by repeated single-step calls with the current `ActionEncoder`; this slice fails explicitly instead of silently resetting a GRU history.
- Predicted latents may be value/useful-transition-equivalent without being symbolically reconstructable; callers must not treat them as validity proof.
- The API assumes external tensorization has already enforced vocabulary/type compatibility. Pulling grounding into `GraphJEPA` would violate the boundary.
- This is an enabling model contract, not evidence that a particular Branch D planner or score combination works.
