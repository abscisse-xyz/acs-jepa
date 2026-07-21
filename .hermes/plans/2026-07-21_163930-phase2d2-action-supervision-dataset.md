# Phase 2D2 Action-Supervision Dataset and PyG Integration Plan

Governing documents:

- `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2.
- Completed Stage 2D1 plan and commit `109eb44`.
- Completed Stage 2A/2B/2C1/2C2 primitives in commits `2ec072e`, `490d980`, `e8c44a8`, and `8962b26`.

Stage 2D remains decomposed without changing its deliverables:

- **2D1:** deterministic single-transition symbolic/tensor builder — complete;
- **2D2 (this plan):** trajectory-dataset integration, fixed time/batch axes, and precomputed offline applicability-label ingestion;
- **2E (later):** disabled-by-default config/model/trainer/checkpoint integration.

## Objective

Integrate Stage 2D1 output into both existing core trajectory dataset variants so later Stage 2E can consume batched supervision regardless of goal-head mode.

The integration must:

1. remain disabled unless an explicit action-supervision config is supplied;
2. produce a nested `action_supervision` tensor dictionary with a leading transition/time axis;
3. use dataset-global action-arity and object capacities so PyG `DataLoader` can batch different problems;
4. deterministically seed each trajectory transition;
5. ingest only a precomputed offline applicability table, never a live simulator callback;
6. keep oracle values solely in label tensors and reject stale oracle tables that contradict trace positives.

## Public API

Add to `acs_jepa.graph.dataset` and export from `acs_jepa.graph`:

```python
ATOM_STATE_APPLICABILITY_SEMANTICS = "positive_ground_atoms_closed_world_v1"

@dataclass(frozen=True, order=True)
class ActionApplicabilityStateKey:
    problem_index: int
    state_atoms: tuple[GroundAtom, ...]

    def __post_init__(self) -> None:
        # exact non-bool int >= 0; exact tuple of exact GroundAtom values;
        # state_atoms already sorted and duplicate-free


def action_applicability_state_key(
    problem_index: int,
    state_atoms: Sequence[GroundAtom],
) -> ActionApplicabilityStateKey:
    # validate exact GroundAtom values and canonicalize with
    # tuple(sorted(set(state_atoms)))


@dataclass(frozen=True)
class ActionApplicabilityTable(
    Mapping[ActionApplicabilityStateKey, frozenset[GroundAction]]
):
    entries: tuple[
        tuple[ActionApplicabilityStateKey, frozenset[GroundAction]], ...
    ]

    @classmethod
    def from_mapping(cls, source: Mapping[...]) -> ActionApplicabilityTable: ...
    def __getitem__(self, key: ActionApplicabilityStateKey) -> frozenset[GroundAction]: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...


@dataclass(frozen=True)
class ActionSupervisionConfig:
    num_negatives: int
    seed: int = 0
    max_random_attempts_per_category: int = 128
    applicable_actions_by_state: (
        Mapping[ActionApplicabilityStateKey, AbstractSet[GroundAction]]
        | ActionApplicabilityTable
        | None
    ) = None
    applicability_state_semantics: str | None = None
```

`ActionApplicabilityTable` is a pickleable immutable mapping:

- direct construction validates that `entries` is an exact tuple of exact two-tuples, sorted by unique exact `ActionApplicabilityStateKey` keys, with exact `frozenset` values containing only exact `GroundAction` values;
- `from_mapping()` copies and canonicalizes keys in sorted order and values to `frozenset`;
- `entries` is a tuple of unique `(key, frozenset)` pairs and contains no mutable cache;
- lookup uses manual binary search over sorted entry keys, remaining `O(log N)` without a mutable `dict`;
- item assignment is unsupported, dataclass field reassignment is frozen, and caller mutations after construction cannot affect it;
- pickle round-trip preserves equality and lookup behavior.

`ActionSupervisionConfig.__post_init__()` must:

- require `type(num_negatives) is int`, `type(seed) is int`, and `type(max_random_attempts_per_category) is int`, excluding booleans;
- reject negative `num_negatives` and non-positive random-attempt limits;
- normalize any input mapping to `ActionApplicabilityTable` via `object.__setattr__`;
- preserve `None` as “no oracle table supplied”;
- require `applicability_state_semantics is None` when no table is supplied;
- require the exact `ATOM_STATE_APPLICABILITY_SEMANTICS` value when a table, including an empty table, is supplied.

The table is tied to the exact `parsed_problems` ordering through `problem_index`. Dataset construction rejects keys whose problem index is outside `[0, len(parsed_problems))`.

## Dataset constructor changes

Add an optional keyword to both:

```python
PDDLTrajectoryDataset(..., action_supervision: ActionSupervisionConfig | None = None)
PDDLAtomTrajectoryDataset(..., action_supervision: ActionSupervisionConfig | None = None)
```

`None` is the disabled default and must preserve current keys, values, and behavior exactly.

When enabled, both classes store:

```text
max_action_arity = maximum schema arity across parsed problems  (existing)
max_objects      = maximum object count across parsed problems
```

They also validate:

- every present trajectory has the same positive action length `K >= 1`; an empty trajectory collection is allowed, but zero-action trajectories and mixed positive lengths are rejected when action supervision is enabled because standard collation cannot form `[B,K,...]` tensors;
- all parsed problems have the same sorted `(action name, parameter-type tuple)` signature, so problem-local action IDs remain semantically batch-compatible;
- every applicability-table key references an in-range problem;
- every keyed state atom has a known predicate, exact predicate arity, known objects, and exact role types for that keyed problem;
- every stored applicable action has a known schema, exact action arity, known objects, and exact role types for that keyed problem.

This symbolic validation catches foreign or stale tables before any label is treated as authoritative. It cannot prove oracle completeness without rerunning the forbidden simulator; completeness remains an explicit trusted assertion made by the offline producer through the semantics token and artifact-generation process.

`num_negatives == 0` remains enabled: the dataset emits argument/object supervision with zero-length negative axes. This is distinct from `action_supervision is None`, which emits no nested supervision key.

No CLI/config selection is added in this stage; Stage 2E will construct this config only when an auxiliary objective is enabled.

## Output contract

For a trajectory/window with `K` actions, add:

```python
sample["action_supervision"] = {
    key: torch.stack(per_transition[key])
    for key in Stage2D1TensorContract
}
```

Shapes become:

```text
negative_action_id                    [K, M]
negative_action_object_indices        [K, M, R]
negative_action_role_ids              [K, M, R]
negative_action_arg_mask              [K, M, R]
negative_mask                         [K, M]
negative_category_id                  [K, M]
negative_changed_role_mask            [K, M, R]
negative_applicability_label          [K, M]
negative_applicability_label_mask     [K, M]
argument_target_indices               [K, R]
argument_mask                         [K, R]
argument_candidate_mask               [K, R, O]
object_mask                           [K, O]
```

where `M = config.num_negatives`, `R = dataset.max_action_arity`, and `O = dataset.max_objects`.

Existing `states`, `actions`, `atom_queries`, `goal`, and `terminal_state` fields remain unchanged.

For fixed-length windows, PyG `DataLoader(batch_size=B)` must collate nested supervision tensors to `[B,K,...]` without custom collation. Different problem object counts are represented by `object_mask` and candidate-mask padding against dataset-global `O`.

## Deterministic transition seeding

Add an internal content-derived helper:

```python
def _action_supervision_seed(
    base_seed: int,
    problem_index: int,
    state_atoms: Sequence[GroundAtom],
    action: GroundAction,
) -> int:
    # BLAKE2b digest over explicit length-prefixed UTF-8 fields:
    # base seed, problem index, canonical sorted unique current atoms,
    # action name, and action arguments.
```

Do not use `repr()`, pickle bytes, delimiter-only concatenation, or Python `hash()`. Each UTF-8 field is prefixed by its fixed-width byte length before digest update, and the unsigned 64-bit digest becomes the Stage 2D1 seed.

Requirements:

- stable across processes and repeated dataset reads;
- stable for the same source transition across sliding-window duplication and trajectory reordering;
- different whenever the keyed problem, represented current state, or action differs, except negligible digest collision risk;
- no mutation of global RNG state;
- action-supervision seeding independent from atom-query sampling RNG.

The helper remains internal because offline applicability tables contain complete state-level action sets and do not need to predict sampled negatives.

## Offline applicability-table boundary

The only accepted oracle input is the precomputed `applicable_actions_by_state` table.

### Supported state semantics

An oracle table is permitted only for classical atom-only domains where action applicability is fully determined by:

- the represented closed-world set of positive `GroundAtom` values for the current state; and
- problem-specific static structure identified by `problem_index`.

Numeric fluents, temporal/running-action state, continuous resources, hidden simulator state, or any other applicability-relevant component discarded by corpus ingestion are unsupported. Supplying a table requires explicit acknowledgment with `ATOM_STATE_APPLICABILITY_SEMANTICS`; the dataset cannot infer this property from the reduced `ParsedProblem` representation.

Offline producers must build every key with `action_applicability_state_key()` from the exact atom state stored in the corpus. Dataset lookup always uses raw `trajectory.states[step_idx]`, so `include_static=True/False` only changes graph construction and never changes oracle identity or labels. `problem_index` prevents equal positive-atom sets from colliding across problems with different static structure.

### Lookup behavior

For transition `step_idx`:

1. Build the key from `trajectory.problem_index` and the **current** symbolic state `trajectory.states[step_idx]`, not the next state.
2. If the table is `None` or the key is absent, pass no labeler to Stage 2D1; all negative labels remain unknown/masked.
3. If the key exists, treat its action set as a complete offline oracle result:
   - first require the observed trace action to be present; otherwise raise `ValueError` identifying the problem/trajectory/step as stale or misaligned data;
   - pass a closure to Stage 2D1 returning `candidate in applicable_set`, yielding explicit true/false labels for every sampled active negative.
4. The closure receives only the sampled symbolic action and performs set membership. It cannot alter sampling.

An empty set is a present, complete oracle entry and therefore contradicts any trace transition; it is not equivalent to a missing key.

No `SimulatorEngine`, `applicable_actions()`, planner, decoder, or PDDL replay import is permitted. Oracle generation occurs before dataset construction in a separately controlled offline process. The dataset only consumes immutable symbolic results.

## Shared implementation

Add one internal helper used by both dataset variants:

```python
def _build_action_supervision_sequence(
    parsed_problem: ParsedProblem,
    trajectory: TrajectorySample,
    *,
    max_action_arity: int,
    max_objects: int,
    config: ActionSupervisionConfig,
) -> dict[str, Tensor]: ...
```

For each `(state_t, action_t)`, it calls `build_action_supervision_tensors()` with the exact global capacities and derived seed, then uses existing `_stack_tensor_dicts()`.

Do not duplicate sampling or tensor logic in dataset code.

## Validation

In addition to strict config/table and dataset-constructor validation above:

- direct state-key construction requires an exact non-bool integer index and an exact tuple containing only exact `GroundAtom` values in sorted duplicate-free order; callers should use the canonicalizing helper;
- state/action symbolic validation errors identify the offending problem index, key, and atom/action;
- a present table entry must contain the trace action for every transition that looks it up;
- malformed sampled action/type/capacity behavior remains delegated to Stage 2D1.

Unused but valid oracle entries are allowed so one precomputed table can be shared across train/validation/test trajectory subsets that retain the same `parsed_problems` ordering.

Disabled datasets retain existing variable/empty-trajectory behavior. Enabled datasets allow an empty collection but require every present trajectory to have the same `K >= 1` and require compatible action signatures for batch semantics.

## Scope

Modify only:

- `packages/acs-jepa-core/src/acs_jepa/graph/dataset.py`;
- `packages/acs-jepa-core/src/acs_jepa/graph/__init__.py`;
- `packages/acs-jepa-core/tests/test_action_supervision_dataset.py`;
- this plan and its review record.

Stage 2D1 source remains unchanged unless a blocking integration defect is independently identified and separately re-reviewed.

## Explicit non-goals

- No simulator or oracle execution.
- No live callback stored on a dataset.
- No applicability leakage into negative selection, masks, states, actions, or candidate tensors.
- No CLI `make_torch_dataset()` changes.
- No trainer, loss invocation, model/head construction, optimizer, config schema, checkpoint, or metrics changes.
- No encoder execution or latent padding.
- No planner/decoder behavior changes.
- No smoke training, retraining, tuning, or empirical acceptance claim.

## Required later Stage 2E boundaries

Stage 2E must still:

- select only rows with `negative_mask.any(-1)` for `ActionContrastiveLoss`;
- encode active symbolic negatives only and fill inactive latent slots with a finite nonzero latent before masked contrastive validation;
- skip applicability training when no negative applicability labels are active, while adding the observed trace action as the positive example;
- skip/reject argument reconstruction when configured `R == 0` or `O == 0` because the current head requires non-empty axes;
- keep every auxiliary objective disabled by default.

## TDD sequence

Use explicit RED → GREEN cycles. Behavior introduced incidentally remains regression coverage and is not misreported as an observed RED.

1. **Disabled compatibility and enabled zero-negative output**
   - RED: missing config/API.
   - GREEN: default datasets have no `action_supervision`; explicit `num_negatives=0` emits exact `[K,0,...]` negative tensors plus argument/object tensors.
2. **Time-axis assembly**
   - RED: a two-transition trajectory matches two direct Stage 2D1 calls using derived seeds; all keys, values, shapes, and dtypes match.
   - GREEN: shared sequence builder and nested output in `PDDLTrajectoryDataset`.
3. **Content-stable deterministic seeds and RNG isolation**
   - RED: repeated reads and independently constructed equal datasets match exactly; the same source transition duplicated into overlapping/reordered windows receives identical supervision; changing represented current state, problem index, or action changes the derived seed; global RNG state is unchanged.
   - GREEN: explicit length-prefixed BLAKE2b seed derivation.
4. **Dataset-global object capacity and PyG batching**
   - RED: two action-schema-compatible problems with different object counts batch to exact `[B,K,M,R/O]` shapes; assert every nested key, value, dtype, object/candidate padding mask, and problem-local active target after `DataLoader(batch_size=2)`.
   - GREEN: global `max_objects` and standard nested-dict collation.
5. **Immutable offline labels use current state**
   - RED: complete table entries label active negatives true/false; identical candidate actions can receive different labels at two current states; identical atom sets under different problem indices do not collide; missing state entries produce unknown masks; `include_static` does not change lookup.
   - RED: mutating the original input mapping/set after config construction has no effect; config/table item assignment and field mutation fail; pickle round-trip preserves exact table/config equality and lookup.
   - GREEN: canonical state keys, immutable tuple/frozenset binary-search table, and state-specific membership closure.
6. **Oracle isolation, semantics, and stale-data rejection**
   - RED: changing only oracle sets changes only the two applicability label tensors; empty/present or trace-action-missing entries raise controlled errors; unsupported/missing semantics acknowledgment rejects table construction; no simulator module is imported or called.
   - RED: dataset construction rejects table keys with out-of-range problem indices, malformed state predicates/arity/objects/types, and stored actions with unknown schemas/wrong arity/foreign objects/wrong role types.
   - GREEN: constructor table validation and trace-positive consistency before Stage 2D1.
7. **Goal-head dataset parity**
   - RED: `PDDLAtomTrajectoryDataset` emits the identical `action_supervision` tensors as `PDDLTrajectoryDataset` for equal trajectory/config while preserving atom-query/goal/terminal outputs and independent atom RNG behavior.
   - GREEN: reuse shared sequence helper from the atom dataset path.
8. **Config/key/dataset validation and exports**
   - RED/GREEN strict non-bool integer scalars, negative M, non-positive attempt limit, malformed/noncanonical direct keys, canonical helper behavior, non-`GroundAction` table entries, fixed nonzero `K`, incompatible action signatures, and public imports/constants.
9. **DataLoader workers**
   - RED/GREEN config/table pickle round-trip and `DataLoader(..., num_workers=1, multiprocessing_context="spawn")` batch match single-process collation for a small top-level fixture.

## Verification

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_action_supervision_dataset.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_action_supervision.py \
         packages/acs-jepa-core/tests/test_pddl_graph.py \
         packages/acs-jepa-core/tests/test_action_contrastive_loss.py \
         packages/acs-jepa-core/tests/test_argument_reconstruction_loss.py \
         packages/acs-jepa-core/tests/test_applicability_loss.py -q

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/graph/dataset.py \
  packages/acs-jepa-core/src/acs_jepa/graph/__init__.py \
  packages/acs-jepa-core/tests/test_action_supervision_dataset.py

UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  python -m compileall -q packages/acs-jepa-core/src/acs_jepa

git diff --check
python /opt/data/skills/software-development/phase-gated-implementation/scripts/static_diff_scan.py
```

Run the entire `acs-jepa-core` test suite before implementation review.

## Acceptance criteria

- Disabled datasets are backward-compatible.
- Enabled datasets expose exact `[K,...]` supervision tensors in both trajectory variants.
- PyG batches fixed-`K`, action-schema-compatible problems with different object counts using global capacity and correct masks/dtypes.
- Transition sampling is content-stable across window reordering and isolated from global/atom-query RNG.
- Applicability labels come only from pickleable immutable precomputed state-level tables whose atoms/actions are symbolically valid for their keyed problem.
- Table use requires explicit atom-only closed-world semantics acknowledgment; unsupported numeric/temporal/hidden state is documented and rejected by policy.
- Current-state lookup, complete-entry false labels, missing-entry unknown labels, and trace-positive consistency are exact.
- No simulator/oracle execution or label leakage is introduced.
- Focused, adjacent, full-core, lint, compile, diff, and static checks pass.
- Independent implementation review returns PASS before a signed commit.

## Gate

Coding is blocked until an independent plan review returns PASS. Blocking findings require revision and re-review. Any code edit after implementation-review PASS invalidates that PASS and requires fresh verification and review.
