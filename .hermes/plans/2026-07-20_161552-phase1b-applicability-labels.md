# Phase 1B Applicability Probe Dataset Implementation Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Add the next Track A/Phase 1 slice: reusable offline `(state, grounded_action, applicability_label)` example construction and diagnostic summaries, using the Phase 1A negative sampler. This creates training/evaluation data for a future learned applicability head, but does not yet add the head, loss, trainer wiring, planner scoring, or decoder behavior changes.

**Architecture:** Build a pure-ish labeling module around existing diagnostic helpers. The module should accept reference positives, generated type-valid negatives, and offline simulator-applicability membership labels. The CLI should produce JSON artifacts that can be inspected before adding any supervised neural module.

**Tech Stack:** Python 3.13, ACS-JEPA graph/action structures, Phase 1A `action_negative_sampling.py`, pytest, uv.

---

## Scope and assumptions

- Repository: `/opt/data/workspace/acs-jepa`.
- This plan covers only offline labels/examples and summaries.
- No model, action encoder, planner, decoder, loss, or training config changes in this slice.
- The true trace action is a positive reference example; optional simulator oracle labels can validate whether the trace action is currently applicable after replay.
- Negatives come from `sample_action_negatives(...)` and are type-valid by construction.
- Applicability labels are computed via `applicable_keys(engine)` membership only; do not mutate a replay engine with candidate `apply_action` calls.
- This remains an offline diagnostic/oracle-labeling path, not a production action generator.

## Research/spec review checkpoints

This plan supports Track A from `ACTION_LATENT_SOLUTION_SPEC.md`:

- Positives: trace action at each state.
- Negatives: one-argument substitutions, type-compatible role swaps, random type-valid same-schema and other-schema actions from Phase 1A.
- Labels: offline `SimulatorEngine.applicable_actions()` membership, used only as diagnostics/supervision targets.
- Output should make it easy to verify the spec criterion that invalid same-schema one-argument substitutions are lower-scored than true/applicable actions in a future learned head.

It deliberately does not implement the learned `ApplicabilityHead` yet; that comes after the label pipeline is tested and reviewed.

## Task 0: Plan review gate

**Objective:** Confirm Phase 1B scope before coding.

**Files:**
- Create: `.hermes/plans/2026-07-20_161552-phase1b-applicability-labels-review.md`

**Review requirements:**

- No implementation starts until this review is complete.
- Blocking review comments must be folded back into this plan before Task 1.
- Confirm this is a small enough slice: data/example labeling only, not learned head/loss/planner integration.

## Task 1: Add tested applicability-label helper

**Objective:** Convert a true action plus sampled negatives into JSON-safe labeled examples for one transition.

**Files:**
- Create: `script/action_applicability_labels.py`
- Create/modify: `acs-jepa-cli/tests/test_action_applicability_labels.py`

**Step 1: Write failing tests**

Tests should cover:

1. `build_applicability_examples(...)` emits exactly one true/reference example plus sampled negatives.
2. The true/reference example has fields:
   - `kind == "positive_trace"`
   - `category == "trace"`
   - action payload with name and arguments.
3. Negative examples preserve Phase 1A category names and include `changed_roles`.
4. Applicability membership labels are computed from an injected set of applicable action keys, not by mutating any simulator.
5. `true_action_applicable` summary reports whether the trace action is in the applicable set.
6. Category and label counters are JSON-safe and deterministic.
7. No duplicate action-key records are emitted within one labeled batch.
8. `applicable_action_keys=None` produces explicit `unknown` counts and `applicable: null` for both trace and negative examples.

Run RED:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_applicability_labels.py -q
```

Expected: FAIL because `script/action_applicability_labels.py` does not exist.

**Step 2: Implement minimal helper**

Proposed dataclass:

```python
@dataclass(frozen=True)
class ApplicabilityExampleBatch:
    examples: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
```

Main function:

```python
def build_applicability_examples(
    true_action: GroundAction,
    negatives: Sequence[NegativeActionExample],
    *,
    applicable_action_keys: AbstractSet[tuple[str, tuple[str, ...]]] | None,
) -> ApplicabilityExampleBatch:
    ...
```

Output example shape:

```json
{
  "kind": "negative",
  "category": "one_arg_substitution",
  "action": {"name": "drive", "arguments": ["car0", "j0", "j2"]},
  "changed_roles": [2],
  "applicable": false
}
```

Implementation notes:

- If `applicable_action_keys is None`, set all `applicable` fields to `None` and count them as `unknown`.
- If labels are available, set the trace action label by membership too; do not assume it is true.
- Keep helper pure and independent of simulator imports.
- Deduplicate by action key globally while preserving order; the trace action wins over any duplicate negative.

**Step 3: Run GREEN tests**

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_applicability_labels.py -q
```

Expected: PASS.

## Task 2: Add applicability-label diagnostic CLI

**Objective:** Produce per-transition positive/negative applicability label JSON using replayed states and Phase 1A sampler.

**Files:**
- Create: `script/diagnose_action_applicability_labels.py`
- Modify tests if needed: `acs-jepa-cli/tests/test_action_applicability_labels.py`

**Step 1: Write failing import/parser test**

Test that:

- `script/diagnose_action_applicability_labels.py` imports without running main.
- parser accepts:
  - `dataset_dir`
  - `--output`
  - `--split`
  - `--max-transitions`
  - `--per-category`
  - `--seed`
  - `--no-oracle-labels`
- validation rejects non-positive `--per-category` and negative `--max-transitions`.

Run RED:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_applicability_labels.py -q
```

Expected: FAIL because diagnostic script does not exist.

**Step 2: Implement diagnostic CLI**

Runtime behavior:

1. Load corpus from `dataset_dir` with `load_corpus([args.dataset_dir], strict=True)` and config with `load_config(None)`.
2. Select split with existing `select_split` helper.
3. For each transition up to `--max-transitions`:
   - cache `ActionDecodingSpace` per parsed problem;
   - sample negatives with `sample_action_negatives(...)` without `applicability_fn`;
   - unless `--no-oracle-labels`, replay source state and call `applicable_keys(engine)` once;
   - call `build_applicability_examples(...)`;
   - write transition metadata, trace action, and examples.
4. Aggregate global counts:
   - transitions;
   - examples;
   - positives/negatives;
   - category counts;
   - applicable/inapplicable/unknown counts;
   - true trace applicable rate when labels are available.
5. Write `summary.json` and `details.json`.

**Do not:**

- add MLflow logging in this first label-pipeline slice;
- add a neural head or trainer loss;
- use `applicable_actions()` results as production candidate generation;
- mutate the replay engine with candidate actions.

## Task 3: Optional local diagnostic smoke run

Only if local smoke data exists:

```bash
test -d /opt/data/workspace/acs-jepa-tuning-data/smoke && echo data=yes || echo data=no
```

If yes:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_applicability_labels.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --output /opt/data/workspace/acs-jepa-runs/smoke/diagnostics/action_applicability_labels_val_4 \
  --split val \
  --max-transitions 4 \
  --per-category 4 \
  --seed 0
```

If no, report that artifact-level run is blocked by missing local smoke data and rely on unit/import tests.

## Task 4: Separate implementation review

Review checklist:

- Scope: data labels only; no learned head/loss/planner integration.
- Spec: positive trace actions and hard negatives are emitted; oracle labels are offline only.
- Correctness: labels come from membership in a provided applicable-action set; unknown labels are explicit; counters are deterministic.
- Coding practice: helper is pure/testable; CLI reuses Phase 1A sampler and existing diagnostic helpers; no hardcoded workspace paths in code.

Verification commands:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_applicability_labels.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check script/action_applicability_labels.py script/diagnose_action_applicability_labels.py acs-jepa-cli/tests/test_action_applicability_labels.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python -m compileall script/action_applicability_labels.py script/diagnose_action_applicability_labels.py
```

## Task 5: Commit after review passes

Only after separate code review passes:

```bash
git add .hermes/plans/2026-07-20_161552-phase1b-applicability-labels.md \
  .hermes/plans/2026-07-20_161552-phase1b-applicability-labels-review.md \
  script/action_applicability_labels.py \
  script/diagnose_action_applicability_labels.py \
  acs-jepa-cli/tests/test_action_applicability_labels.py
git commit -S -m "feat: add action applicability label diagnostic"
```
