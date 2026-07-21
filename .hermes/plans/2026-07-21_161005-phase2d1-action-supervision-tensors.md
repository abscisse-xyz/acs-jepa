# Phase 2D1 Deterministic Action-Supervision Tensor Builder Plan

Governing documents:

- `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2 explicit action-identifiability losses.
- `.hermes/plans/2026-07-21_143024-phase2-roadmap-stage2a-action-vicreg.md`.
- Completed Stage 2A/2B/2C1/2C2 commits `2ec072e`, `490d980`, `e8c44a8`, and `8962b26`.

Stage 2D is split into independently reviewed slices:

- **2D1 (this plan):** pure deterministic symbolic negative sampling and fixed-shape supervision tensors for one transition;
- **2D2 (later):** dataset/window integration and PyG batching;
- trainer/config/model/checkpoint integration remains Stage 2E.

This decomposition does not change Phase 2 deliverables or acceptance criteria.

## Objective

Add a simulator-independent helper that converts one parsed problem and reference grounded action into:

1. a bounded, deterministic, unique set of type-valid negative grounded actions;
2. fixed-size negative-action tensors and masks suitable for later batching;
3. role-object reconstruction targets and role-specific type-valid candidate masks;
4. optional, explicitly masked offline applicability labels supplied by a caller callback.

The helper must not import a simulator, enumerate production applicable actions, mutate datasets, encode actions, or leak labels into model inputs.

## Public API

Add `acs_jepa.graph.action_supervision` with:

```python
ONE_ARG_SUBSTITUTION = 0
ROLE_SWAP = 1
RANDOM_SAME_SCHEMA = 2
RANDOM_OTHER_SCHEMA = 3
NUM_NEGATIVE_CATEGORIES = 4
NEGATIVE_CATEGORY_NAMES = (
    "one_arg_substitution",
    "role_swap",
    "random_same_schema",
    "random_other_schema",
)

@dataclass(frozen=True)
class SampledActionNegative:
    action: GroundAction
    category_id: int
    changed_roles: tuple[int, ...]

ApplicabilityLabeler = Callable[[GroundAction], bool | None]

def sample_type_valid_action_negatives(
    parsed_problem: ParsedProblem,
    true_action: GroundAction,
    *,
    num_negatives: int,
    seed: int,
    max_random_attempts_per_category: int = 128,
) -> tuple[SampledActionNegative, ...]: ...

def build_action_supervision_tensors(
    parsed_problem: ParsedProblem,
    true_action: GroundAction,
    *,
    max_action_arity: int,
    max_objects: int,
    num_negatives: int,
    seed: int,
    applicability_labeler: ApplicabilityLabeler | None = None,
    max_random_attempts_per_category: int = 128,
) -> dict[str, Tensor]: ...
```

Export these APIs/constants from `acs_jepa.graph`, not the root `acs_jepa` namespace.

## Negative sampling semantics

### Validity and uniqueness

Every emitted negative must:

- use a schema in `parsed_problem.actions`;
- have exactly that schema's arity;
- bind each role to an object whose exact normalized type string matches the schema parameter type;
- differ from the reference grounded action;
- be unique by `(schema name, argument tuple)`.

This is exact-type validity under the current `ParsedProblem` representation, matching `ActionDecodingSpace`. It is not general PDDL subtype-aware validity because `ParsedProblem` stores no type hierarchy.

Validate the true action before sampling: known schema, exact arity, known object names, and role-wise type compatibility.

### Categories

1. **One-argument substitution**
   - Same schema.
   - Exactly one role differs.
   - Replacement is type-valid.
2. **Role swap**
   - Same schema.
   - Swap two distinct reference arguments only when both remain type-valid.
   - Exclude no-op equal-object swaps.
3. **Random same-schema**
   - Draw one object independently from each role's sorted type domain.
   - Exclude the true action and prior negatives.
4. **Random other-schema**
   - Draw a different schema and one object from each sorted role domain.
   - Skip schemas with an empty role domain.

Category candidate order is deterministic under a local `random.Random(seed)` and never mutates global RNG state.

### Bounded construction and exact round-robin algorithm

Do not call `ActionDecodingSpace.enumerate_ground_actions()` or materialize a Cartesian product for random categories.

- One-argument substitutions and pairwise swaps may be exhaustively materialized because their sizes are linear/quadratic in action arity and object domains. Build each finite list, shuffle it once with the local RNG, and maintain a cursor.
- For random-other sampling, precompute the lexicographically sorted list of schemas other than the true schema whose every role has a non-empty exact-type object domain. Zero-arity other schemas are eligible. Unusable unrelated schemas are skipped rather than making the problem invalid.
- Visit categories repeatedly in fixed order: one-argument substitution, role swap, random same-schema, random other-schema.
- A finite-category visit consumes exactly the next list element. The element is consumed even if the global seen set rejects it as a duplicate from an earlier category. A finite category is exhausted when its cursor reaches list length.
- A random-category visit performs exactly one complete grounding draw and increments that category's attempt counter exactly once. True-action draws, duplicates, or any rejected draws still consume the attempt budget and emit nothing.
- A random same-schema draw independently samples one object from each sorted role domain with the local RNG.
- A random other-schema draw first samples uniformly from the precomputed eligible-schema list with the local RNG, then independently samples one object from each of that schema's sorted role domains.
- A random category is exhausted exactly when its attempt counter reaches `max_random_attempts_per_category`. Random-other is immediately exhausted when its eligible-schema list is empty.
- Stop when `num_negatives` unique negatives have been emitted or all four categories are exhausted.

The global seen set initially contains the true action and is updated only when a unique negative is emitted. This exact algorithm makes seed-level output reproducible and prevents easy one-argument substitutions from silently consuming every slot when other categories can emit candidates.

The bounded random procedure may return fewer than `num_negatives` because its attempt budget expired even if unseen valid Cartesian groundings still exist. It does not claim proof of full grounding-space exhaustion.

## Tensor contract

`build_action_supervision_tensors()` returns:

```text
negative_action_id                    long  [M]
negative_action_object_indices        long  [M, R]
negative_action_role_ids              long  [M, R]
negative_action_arg_mask              bool  [M, R]
negative_mask                         bool  [M]
negative_category_id                  long  [M]       (-1 when padded)
negative_changed_role_mask            bool  [M, R]
negative_applicability_label          float32 [M]     (0 when unknown/padded)
negative_applicability_label_mask     bool  [M]
argument_target_indices               long  [R]       (-1 when role inactive)
argument_mask                         bool  [R]
argument_candidate_mask               bool  [R, O]
object_mask                           bool  [O]
```

where `M = num_negatives`, `R = max_action_arity`, and `O = max_objects`.

Active negatives are tensorized using the existing `tensorize_action()` contract. Padded negative rows use:

```text
action_id = 0
object_indices = -1
role_ids = -1
arg_mask = false
category_id = -1
changed_role_mask = false
applicability_label = 0
applicability_label_mask = false
negative_mask = false
```

Padding values are never semantic negatives; later Stage 2D2/2E code must honor `negative_mask` before action encoding/loss use.

For every emitted negative, `changed_roles` is computed over `range(max(len(true.arguments), len(candidate.arguments)))`: an overlapping role is changed when argument names differ, and every non-overlapping role is changed. `max_action_arity >= parsed_problem.max_action_arity` guarantees these roles fit `negative_changed_role_mask`. Different-schema identity is represented separately by `negative_category_id`; therefore two zero-arity actions from different schemas legitimately have an all-false changed-role mask.

### Argument reconstruction tensors

- `argument_target_indices` and `argument_mask` are the true action's problem-local object indices and real-role mask.
- `object_mask[o]` is true exactly for real problem objects and false for global padding.
- For each active role `r`, `argument_candidate_mask[r,o]` is true exactly when object `o` exists and its exact normalized type matches the true schema's role type.
- Inactive/padded roles have all-false candidate rows.
- Every active target must be candidate-mask true by construction.
- Zero-arity actions are valid and produce all-inactive argument tensors.

### Applicability-label boundary

The reference trace action is known positive but is already present in the ordinary action tensors; 2D1 does not duplicate it.

For each active negative only:

- if `applicability_labeler is None`, label remains `0` and label mask is false;
- if callback returns `True`/`False`, store `1.0`/`0.0` and set label mask true;
- if callback returns `None`, label remains `0` and label mask false;
- reject any callback return not exactly `bool` or `None` (including integers).

Never call the callback for padded rows. The callback receives only a symbolic negative action; its output is stored solely in label tensors, never in negative selection, category choice, masks, or model-input tensors. Thus simulator-backed callbacks can be used only by a later offline data-construction boundary without label leakage or production inference dependency.

## Validation

Reject:

- negative `num_negatives`;
- non-positive `max_random_attempts_per_category`;
- `max_action_arity` smaller than the true action or parsed-problem maximum arity;
- `max_objects` smaller than the problem object count;
- malformed true actions as described above.

Unrelated schemas with one or more empty exact-type role domains do not invalidate the problem; they are excluded only from random-other sampling.

`num_negatives == 0` is valid and returns correctly typed empty leading dimensions while still returning argument/object tensors.

All tensors are CPU tensors. Device transfer remains the ordinary batch `_to_device` responsibility.

## Scope

Modify only:

- `packages/acs-jepa-core/src/acs_jepa/graph/action_supervision.py`;
- `packages/acs-jepa-core/src/acs_jepa/graph/__init__.py`;
- `packages/acs-jepa-core/tests/test_action_supervision.py`;
- this plan and review record.

Do not modify the Phase 1 diagnostic sampler in `script/action_negative_sampling.py`; convergence/migration may be considered later after training behavior is verified.

## Explicit non-goals

- No dataset class or `make_torch_dataset()` changes.
- No PyG batching/collation behavior.
- No state or next-state input to the helper.
- No simulator import or `applicable_actions()` call.
- No assumption that a type-valid negative is inapplicable.
- No action/object encoder execution.
- No Stage 2A/2B/2C1/2C2 loss invocation.
- No trainer, config, optimizer, checkpoint, CLI, decoder, planner, or smoke-run changes.
- No full grounding-space enumeration for random categories.
- No broad tuning or Phase 2 empirical acceptance claim.

## Required later integration boundaries

Stage 2D2/2E must preserve these contracts; they are documented now to ensure 2D1 tensors do not force redesign:

- `ActionContrastiveLoss` requires at least one active negative per selected example. Select only transition rows where `negative_mask.any(-1)` before invoking it.
- `ActionContrastiveLoss` validates even masked latent slots. Encode active negative actions only, then fill inactive latent slots with a known finite nonzero latent of matching dtype/device (the corresponding positive action latent is acceptable) before passing `negative_mask`; never encode canonical symbolic padding as though it were an action.
- Run applicability supervision only when at least one negative applicability label is active; otherwise skip the auxiliary term rather than training a positive-only batch or calling `ApplicabilityLoss` with an empty effective negative set. The ordinary trace action supplies the positive label in Stage 2E.
- `ArgumentReconstructionHead` requires positive configured arity and non-empty candidate-object axes. If `R == 0` or `O == 0`, Stage 2E must skip argument reconstruction (or reject enabling it during config validation); 2D1 still represents those symbolic edge cases correctly.

## TDD sequence

Use small RED → GREEN cycles. Passing behavior introduced incidentally is retained as regression coverage but is not misreported as an observed RED.

1. **True action and argument tensors**
   - RED: missing helper/export.
   - GREEN: true action targets, active-role mask, object mask, role-specific type-valid candidate mask, and zero-negative shapes match a manual typed problem oracle.
2. **One-argument substitutions**
   - RED: deterministic same-schema one-role changes are unique, true-action-excluding, type-valid, and correctly categorized/tensorized.
   - GREEN: add validation and finite substitution candidate generation.
3. **Role swaps and round-robin diversity**
   - RED: a type-compatible non-self swap is categorized with both changed roles, no-op swaps are absent, and category diversity appears before one category consumes all slots.
   - GREEN: add swap candidates and round-robin selection.
4. **Bounded random categories**
   - RED: same-seed outputs are exactly identical; one random draw attempt occurs per random-category round-robin visit; rejected/duplicate draws consume budget; random-other chooses from the sorted eligible-schema list; random categories exhaust exactly at the attempt limit; emitted same/other-schema actions are exact-type valid.
   - RED: a large Cartesian grounding fixture completes without enumeration; monkeypatch `ActionDecodingSpace.enumerate_ground_actions()` to fail if invoked and assert sampler attempt counters through a deterministic RNG test double rather than relying on timing.
   - GREEN: add local-RNG bounded random draws and explicit category state.
5. **Padding/exhaustion and zero arity**
   - RED: exhausted spaces return fewer symbolic negatives but exact `[M,...]` tensors with canonical inactive padding; zero-arity true actions and `M=0` remain valid.
   - GREEN: add fixed-shape initialization and masked fill.
6. **Optional applicability labels and isolation**
   - RED: bool labels and unknown labels produce exact value/mask tensors; callback is called once per active negative and never for padding; changing callback outputs does not alter any non-label tensor or sampled action.
   - GREEN: add post-sampling label projection with strict callback return validation.
7. **Validation**
   - RED/GREEN malformed true schema/arity/object/type, invalid capacities, negative M, invalid attempt budget, and acceptance of unrelated unusable schemas that are skipped by random-other sampling.
8. **Public graph API**
   - RED/GREEN imports of helper, dataclass, and category constants from `acs_jepa.graph`; assert exact stable constant values `0,1,2,3`, `NUM_NEGATIVE_CATEGORIES == 4`, and exact `NEGATIVE_CATEGORY_NAMES` ordering.

## Verification

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_action_supervision.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_pddl_graph.py \
         packages/acs-jepa-core/tests/test_action_contrastive_loss.py \
         packages/acs-jepa-core/tests/test_argument_reconstruction_loss.py \
         packages/acs-jepa-core/tests/test_applicability_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/graph/action_supervision.py \
  packages/acs-jepa-core/src/acs_jepa/graph/__init__.py \
  packages/acs-jepa-core/tests/test_action_supervision.py

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  python -m compileall -q packages/acs-jepa-core/src/acs_jepa

git diff --check
python /opt/data/skills/software-development/phase-gated-implementation/scripts/static_diff_scan.py
```

## Acceptance criteria

- Sampling is deterministic, local-RNG-only, bounded, unique, and type-valid.
- No random-category Cartesian grounding enumeration occurs.
- Fixed-shape tensors and canonical padding match the contract.
- Argument targets and role-specific candidate masks are exact and target-valid.
- Applicability labels are explicitly masked and cannot affect sampling/model-input tensors.
- Zero-negative, finite-pool exhaustion, random-attempt-budget exhaustion, and zero-arity cases are valid.
- Existing graph dataset and Stage 2 loss helpers remain green.
- Scope remains a pure symbolic/tensor primitive.
- Independent implementation review returns PASS before a signed commit.

## Gate

Coding is blocked until an independent plan review returns PASS. Blocking findings require plan revision and re-review. Any code edit after an implementation-review PASS invalidates that PASS and requires fresh verification/review.
