# Phase 1A Negative Sampler Implementation Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Add the first Phase 1 slice: reusable type-valid hard-negative action sampling and JSON diagnostics, without changing training, planner, decoder, or model behavior.

**Architecture:** Keep Phase 1A as a measurement/labeling utility layer. Add a pure sampler module that works from `ActionDecodingSpace`, `GroundAction`, and an optional simulator applicability oracle. Add tests with tiny parsed PDDL problems. Defer learned probes/heads to Phase 1B after this sampler is reviewed.

**Tech Stack:** Python 3.13, ACS-JEPA graph/action structures, pytest, uv.

---

## Scope and assumptions

- Repository: `/opt/data/workspace/acs-jepa`.
- This plan covers only Phase 1A negative sampling, not probe training or model heads.
- Generated negatives must remain type-valid according to `ActionDecodingSpace`.
- Applicability labels, if requested, use simulator replay/apply checks only as offline diagnostic labels.
- The sampler must be deterministic under a seed and must not mutate parsed problem, simulator, or reference actions.
- The sampler should categorize negatives as required by the spec:
  - `one_arg_substitution`
  - `role_swap`
  - `random_same_schema`
  - `random_other_schema`

## Research/spec review checkpoints

The plan must preserve these points from the spec and reference papers:

- This is an action-grounding/identifiability support component, not a classical planner replacement.
- Hard negatives should focus on same-schema wrong-object substitutions because root-cause evidence shows schema identity is easy but argument binding is weak.
- Offline applicability labels are allowed for diagnosis/supervision labels, but `applicable_actions()` must not become production action generation.
- The sampler should make later contrastive/applicability objectives possible without adding those objectives yet.

## Task 0: Plan review gate

**Objective:** Confirm Phase 1A scope before coding.

**Files:**
- Create: `.hermes/plans/2026-07-20_155945-phase1a-negative-sampler-review.md`

**Review requirements:**

- No implementation starts until this review is complete.
- Blocking review comments must be folded back into this plan before Task 1.
- Confirm this slice is small enough: reusable sampler + diagnostic output only; no learned probes or loss wiring yet.

## Task 1: Add tested negative sampler helper

**Objective:** Implement deterministic, categorized, type-valid negative sampling around a true action.

**Files:**
- Create: `script/action_negative_sampling.py`
- Create/modify: `acs-jepa-cli/tests/test_action_negative_sampling.py`

**Step 1: Write failing tests**

Tests should cover:

1. `sample_action_negatives(...)` returns no copy of the true action.
2. One-argument substitution negatives preserve schema and change exactly one argument.
3. Role-swap negatives preserve the same multiset of arguments when type-compatible and differ from the true ordered tuple.
4. Random same-schema negatives preserve schema but differ from the true action.
5. Random other-schema negatives use a different action schema.
6. All returned negatives can be tensorized by `ActionDecodingSpace.action_tensors_for_ground_actions(...)`, proving type-validity.
7. Sampling is deterministic for the same seed.
8. Category counts respect requested per-category caps when enough candidates exist.
9. Sampling does not mutate the true `GroundAction` or `ParsedProblem`.
10. Duplicate-argument role swaps that would reproduce the original tuple are skipped.

Run RED:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_negative_sampling.py -q
```

Expected: FAIL because `script/action_negative_sampling.py` does not exist.

**Step 2: Implement minimal sampler**

Create:

```python
@dataclass(frozen=True)
class NegativeActionExample:
    action: GroundAction
    category: str
    changed_roles: tuple[int, ...]
    applicable: bool | None = None
```

Main function:

```python
def sample_action_negatives(
    space: ActionDecodingSpace,
    true_action: GroundAction,
    *,
    per_category: int = 4,
    seed: int = 0,
    applicability_fn: Callable[[GroundAction], bool] | None = None,
) -> tuple[NegativeActionExample, ...]:
    ...
```

Implementation notes:

- Use `random.Random(seed)` only; no global randomness.
- Compute type-compatible object domains from `space.objects_by_type` and `space.parsed_problem.actions[action_name].parameter_types`.
- Deduplicate by `(action.name, action.arguments)` while preserving first category order.
- Category generation order:
  1. one-argument substitutions;
  2. role swaps;
  3. random same-schema;
  4. random other-schema.
- Role swaps are valid only when each swapped object is type-compatible for the target role.
- `changed_roles` is the tuple of true-action role indices whose value changed. For `random_other_schema`, set it to all comparable role indices where arguments differ plus any true-action role index outside the other schema arity; it is diagnostic metadata, not a claim that schemas share semantics.
- `applicability_fn`, if provided, is called after candidate generation and labels only the returned negatives.
- Do not import or call simulator code from this helper.

**Step 3: Run GREEN tests**

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_negative_sampling.py -q
```

Expected: PASS.

## Task 2: Add negative sampler diagnostic script

**Objective:** Produce per-transition JSON labels for sampled negatives using the reusable helper.

**Files:**
- Create: `script/diagnose_action_negatives.py`
- Modify tests if needed: `acs-jepa-cli/tests/test_action_negative_sampling.py`

**Step 1: Write failing import/parser test**

Test that:

- `script/diagnose_action_negatives.py` imports without running main.
- parser accepts:
  - `dataset_dir`
  - `--output`
  - `--split`
  - `--max-transitions`
  - `--per-category`
  - `--label-applicability`
  - `--seed`
- validation rejects non-positive `--per-category` and negative `--max-transitions`.

Run RED:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_negative_sampling.py -q
```

Expected: FAIL because diagnostic script does not exist.

**Step 2: Implement diagnostic CLI**

Runtime behavior:

1. Load corpus from `dataset_dir` with `load_corpus` and config with `load_config(None)`.
2. Select split with existing `select_split` helper.
3. For each transition up to `--max-transitions`:
   - build/cache `ActionDecodingSpace` per parsed problem;
   - optionally build/replay simulator engine once per transition and pass an `applicability_fn` that checks membership in `applicable_keys(engine)`; do not mutate a shared replay engine with `apply_action`;
   - call `sample_action_negatives(...)`;
   - write true action and negatives with category, changed roles, and optional applicability label.
4. Aggregate counts by category and applicability label.
5. Write `summary.json` and `details.json`.

Output shape:

```json
{
  "metrics": {
    "transitions": 2,
    "negatives": 32,
    "labeled_applicability": true
  },
  "category_counts": {
    "one_arg_substitution": 8
  },
  "applicability_counts": {
    "applicable": 1,
    "inapplicable": 31,
    "unknown": 0
  }
}
```

**Step 3: Run GREEN tests**

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_negative_sampling.py -q
```

Expected: PASS.

## Task 3: Optional local diagnostic smoke run

Only if local smoke data exists:

```bash
test -d /opt/data/workspace/acs-jepa-tuning-data/smoke && echo data=yes || echo data=no
```

If yes:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_negatives.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --output /opt/data/workspace/acs-jepa-runs/smoke/diagnostics/action_negatives_val_4 \
  --split val \
  --max-transitions 4 \
  --per-category 4 \
  --label-applicability \
  --seed 0
```

If no, report that artifact-level run is blocked by missing local smoke data and rely on unit/import tests.

## Task 4: Separate implementation review

**Objective:** Review code separately after implementation.

Review checklist:

- Spec compliance: required negative categories are present and type-valid.
- Research compliance: sampler emphasizes same-schema argument near-misses and does not replace planning with oracle enumeration.
- Correctness: no true action appears in negatives; role swaps are type-compatible; deterministic under seed; caps respected.
- Coding practice: helper is pure/testable; diagnostic script reuses helpers; no broad refactors; no hardcoded local paths in code.
- Performance: no exhaustive simulator applicability enumeration unless explicitly requested for offline labels.

Verification commands:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_negative_sampling.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check script/action_negative_sampling.py script/diagnose_action_negatives.py acs-jepa-cli/tests/test_action_negative_sampling.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python -m compileall script/action_negative_sampling.py script/diagnose_action_negatives.py
```

## Task 5: Commit after review passes

Only after separate code review passes:

```bash
git add .hermes/plans/2026-07-20_155945-phase1a-negative-sampler.md \
  .hermes/plans/2026-07-20_155945-phase1a-negative-sampler-review.md \
  script/action_negative_sampling.py \
  script/diagnose_action_negatives.py \
  acs-jepa-cli/tests/test_action_negative_sampling.py
git commit -S -m "feat: add action negative sampler diagnostic"
```
