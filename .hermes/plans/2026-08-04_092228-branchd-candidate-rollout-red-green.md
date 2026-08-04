# Branch D Candidate Rollout RED/GREEN Record

This record is maintained incrementally. Commands are run from `/opt/data/workspace/acs-jepa`.

## Attempt 1 — REJECTED

All original Task 1, Task 2, and final-verification claims below belong to rejected Attempt 1. Implementation Review 1 returned **FAIL** because exact metadata identity did not detect in-place mutation and the approved one-behavior-at-a-time Task 2 lifecycle was not followed; notably, the object-row mismatch test passed immediately. These claims remain unchanged as historical evidence and are not accepted evidence for the restarted implementation.

Rejected tracked files were preserved at `/opt/data/cache/branchd-candidate-rollout-rejected-attempt1/` with hashes in `SHA256SUMS`. The four tracked files were then restored byte-for-byte from HEAD `63534e9` while the plan, review, and this RED/GREEN record remained untracked and preserved.

## Attempt 1 historical record (rejected)

### Task 1 — public API and recurrence

### RED

Command (exit 4):

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_uses_each_predicted_source_and_preserves_order packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_supports_real_action_encoder_predictor_and_gradients -q
```

Expected and observed missing-public-API failure:

```text
ImportError: cannot import name 'GraphJEPACandidateRolloutOutput' from 'acs_jepa.jepa'
ERROR: found no collectors for ...::test_candidate_rollout_uses_each_predicted_source_and_preserves_order
ERROR: found no collectors for ...::test_candidate_rollout_supports_real_action_encoder_predictor_and_gradients
```

Minimal fix: add the frozen result dataclass, exact per-step candidate recurrence method, and package export.

### GREEN

The same command exited 0 with `2 passed`.

## Task 2 — behavioral validation slices

### Empty mapping

- RED: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_empty_action_mapping -q` exited 1: expected `ValueError`, observed `StopIteration` at first-value lookup.
- Fix: explicit nonempty mapping check with `ValueError: action_tensors must not be empty`.
- GREEN: the same focused node exited 0 with `1 passed`.

### Action mapping values, rank, and common prefix

- RED commands: the focused node `test_candidate_rollout_validates_action_tensor_fields[CASE] -q` was run separately for `non-tensor`, `rank`, and `prefix`; each exited 1. Observed failures were respectively `AttributeError: 'list' object has no attribute 'size'`, dimension `IndexError`, and step-slice `IndexError`.
- Fix: deterministic Tensor type, minimum-rank, and common-leading-`[B,H]` validation.
- GREEN: the parameterized focused node exited 0 with `3 passed`.

### Initial latent state, metadata, batch, and horizon

- RED commands: each exact parameter node `test_candidate_rollout_validates_initial_state_and_candidate_extent[CASE] -q` was run for `graph-rank`, `graph-empty`, `object-rank`, `object-width`, `ids-rank`, `batch-rank`, `metadata-length`, `batch-range`, `candidate-batch`, and `zero-horizon`; all exited 1. Failures were uncontrolled `RuntimeError`/`IndexError` or `DID NOT RAISE`, proving each missing contract.
- Fix: validate non-temporal/nonempty graph shape, object rank/width/rows, metadata ranks/lengths/range, candidate batch equality, and positive horizon before module calls.
- Test-defect correction (not RED): the first grouped GREEN exposed unescaped `[]` in `pytest.raises(match=...)`; changed to `re.escape(message)` while retaining exact-message `check`.
- GREEN: the corrected focused node exited 0 with `10 passed`.

### Context, controls, predictor result/shapes, and metadata identity

- RED commands: exact parameter nodes were run individually for exposed contexts `none`/`two`, controls `rank`/`batch`/`changing-width`, predictor non-latent type, graph shapes `rank`/`batch`/`width`, object shapes `rank`/`rows`/`width`, and metadata `ids-clone`/`ids-mutate`/`batch-clone`/`batch-mutate`. Every node exited 1 with `DID NOT RAISE` or an uncontrolled `IndexError`/`RuntimeError`/`AttributeError`.
- Fix: reject exposed context other than exactly one; require controls `[B,D_a]` with fixed width; require `JEPALatentState`; require exact graph/object shapes; require the original metadata tensor objects (therefore storage) at every step. Encoders without a `context_steps` attribute remain accepted by the Task 1 recurrence test.
- GREEN: the five focused nodes exited 0 with `16 passed`.
- Immediate-pass regression coverage (not behavioral RED): added explicit initial object-row mismatch, transition-equivalent/repeated candidate acceptance, and frozen-result assignment checks. These restate already implemented/precommitted contract details and are not claimed as new RED evidence.

### Task 2 aggregate GREEN

Command:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py -q
```

Exit 0: `74 passed`.

## Final verification

- Focused graph components + planner: exit 0, `82 passed` (rerun after final formatting: `82 passed`).
- Full core: exit 0, `330 passed`.
- Scoped CLI with only `test_tuning_configs_load_and_keep_required_defaults` excluded: exit 0, `256 passed`.
- Ruff on the three Python files: `All checks passed!` after wrapping pre-existing overlong lines/import sorting.
- `python -m compileall -q ...`: exit 0.
- `git diff --check`: exit 0.
- Expected upstream dependency deprecation warnings remained; no test failures.

## Accepted Restart Attempt 2 — correction phase A

Implementation Review 1 FAIL report: `/opt/data/cache/delegation/subagent-summary-0-20260804_101914_697093.txt`.

This is a fresh tracked-file restart from HEAD `63534e9`. Only evidence recorded below belongs to Accepted Restart Attempt 2. Scope is Task 1 and the ordered early Task 2 validations through initial object-row mismatch; later Task 2 behaviors, README, aggregate suites, commit, and push remain deferred.

### Task 1 — public API and recurrence

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_uses_each_predicted_source_and_preserves_order packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_supports_real_action_encoder_predictor_and_gradients -q`
- RED exit: `4`.
- Expected/observed excerpt: `ImportError: cannot import name 'GraphJEPACandidateRolloutOutput' from 'acs_jepa.jepa'`; pytest also reported no collectors for both exact nodes.
- Minimal fix: added only the frozen four-field output dataclass, exact per-step slicing/encoding/prediction recurrence, stacked controls, and package export. No Task 2 validator was added.
- GREEN: the exact same two-node command exited `0` with `2 passed`.

### Task 2.1 — empty action mapping

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_empty_action_mapping -q`
- RED exit/excerpt: `1`; `StopIteration` at `next(iter(action_tensors.values()))` instead of the required deterministic `ValueError`.
- Minimal fix: explicit nonempty mapping check with `ValueError: action_tensors must not be empty`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.2 — non-Tensor action value

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_non_tensor_action_value -q`
- RED exit/excerpt: `1`; `AttributeError: 'list' object has no attribute 'size'` instead of the required `TypeError`.
- Minimal fix: validate every mapping value with `isinstance(tensor, Tensor)` and name the offending field.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.3 — action tensor rank below two

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_action_tensor_below_rank_two -q`
- RED exit/excerpt: `1`; `IndexError: Dimension out of range ... got 1` at horizon lookup instead of the required `ValueError`.
- Minimal fix: reject each Tensor with `ndim < 2` before horizon access.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.4 — inconsistent action `[B,H]`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_inconsistent_action_batch_horizon -q`
- RED exit/excerpt: `1`; `IndexError: index 2 is out of bounds for dimension 1 with size 2` during candidate slicing.
- Minimal fix: capture the first `[B,H]` prefix and reject any differing field before recurrence.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.5 — initial graph rank

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_graph_rank -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: stack expects each tensor to be equal size` inside the action encoder.
- Minimal fix: require `initial_state.graph_latent.ndim == 2` before module calls.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.6 — empty initial graph

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_empty_initial_graph -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: stack expects each tensor to be equal size` inside the action encoder.
- Minimal fix: reject an empty batch or latent-width dimension after graph-rank validation.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.7 — initial object rank

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_object_rank -q`
- RED exit/excerpt: `1`; uncontrolled predictor `RuntimeError: The expanded size of the tensor (1) must match ...`.
- Minimal fix: require `initial_state.object_latents.ndim == 2` before module calls.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.8 — initial object width

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_object_width -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>` instead of the required deterministic width mismatch error.
- Minimal fix: require the initial object-latent width to equal the initial graph-latent width, with `ValueError: initial state graph and object latent widths must match`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.9 — initial object-row mismatch

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_object_row_mismatch -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: The expanded size of the tensor (3) must match the existing size (4) at non-singleton dimension 0. Target sizes: [3, 2]. Tensor sizes: [4, 1]` in the predictor.
- Minimal fix: reject an initial object-row count different from `object_batch.numel()` with `ValueError: initial state object rows and metadata lengths must match`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.10 — initial `object_ids` rank

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_object_ids_rank -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: require `initial_state.object_ids.ndim == 1` with `ValueError: initial_state.object_ids must be rank one`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.11 — initial `object_batch` rank

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_object_batch_rank -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: expand(torch.FloatTensor{[4, 1, 1]}, size=[4, 2]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)` in the predictor.
- Minimal fix: require `initial_state.object_batch.ndim == 1` with `ValueError: initial_state.object_batch must be rank one`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.12 — initial metadata lengths mismatch

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_metadata_length_mismatch -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: reject an initial object-row count different from `object_ids.numel()` with `ValueError: initial state object rows and metadata lengths must match`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Correction phase A aggregate verification

- All 14 Accepted Restart Attempt 2 nodes currently present: exit `0`, `14 passed`.
- Entire `packages/acs-jepa-core/tests/test_graph_jepa_components.py`: exit `0`, `54 passed`.
- Ruff on the two Python files modified during correction phase A (`jepa.py` and `test_graph_jepa_components.py`): exit `0`, `All checks passed!`.
- `python -m compileall -q` on those two Python files: exit `0` with no output.
- `git diff --check`: exit `0` with no output.

### Task 2.13 — initial `object_batch` value outside `[0, B)`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_initial_object_batch_out_of_range -q`
- RED exit/excerpt: `1`; uncontrolled `IndexError: index 2 is out of bounds for dimension 0 with size 2` in the predictor.
- Minimal fix: reject any initial `object_batch` value below zero or at least the initial graph batch size with `ValueError: initial_state.object_batch values must be in [0, B)`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.14 — candidate batch differs from initial graph batch

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_candidate_batch_mismatch -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 1` in the action encoder.
- Minimal fix: require the common candidate batch to equal `initial_state.graph_latent.size(0)` with `ValueError: candidate batch must match initial state graph batch`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.15 — zero candidate horizon

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_zero_candidate_horizon -q`
- RED exit/excerpt: `1`; uncontrolled `IndexError: list index out of range` at `final_state=predicted_states[-1]`.
- Minimal fix: reject a zero common candidate horizon with `ValueError: candidate horizon must be positive`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.16 — exposed `action_encoder.context_steps is None`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_exposed_none_action_context -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: when the action encoder exposes `context_steps`, reject `None` with `ValueError: action_encoder.context_steps must be exactly 1`; stateless encoders without the attribute remain accepted.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.17 — exposed `action_encoder.context_steps == 2`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_exposed_multi_step_action_context -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: broaden the exposed-context check from `None` to every value other than exactly `1`, retaining the same deterministic error.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.18 — control rank not `[B, D_a]`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_control_rank -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: expand(torch.FloatTensor{[2, 1, 2]}, size=[2, 2])` in the predictor.
- Minimal fix: reject a control whose rank is not two with `ValueError: action encoder control must have shape [B, D_a]` before prediction.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.19 — control batch mismatch

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_control_batch_mismatch -q`
- RED exit/excerpt: `1`; uncontrolled `IndexError: index 1 is out of bounds for dimension 0 with size 1` in the predictor.
- Minimal fix: require each rank-two control batch to equal the candidate batch with `ValueError: action encoder control batch must match candidate batch`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.20 — control width changes on a later step

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_later_control_width_change -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: stack expects each tensor to be equal size, but got [2, 2] at entry 0 and [2, 3] at entry 1`.
- Minimal fix: after the first valid control, require every later control width to match it with `ValueError: action encoder control width must remain constant`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Candidate extent, context, and control aggregate verification

- All 22 Accepted Restart Attempt 2 nodes currently present (`-k candidate_rollout`): exit `0`, `22 passed`.
- Entire `packages/acs-jepa-core/tests/test_graph_jepa_components.py`: exit `0`, `62 passed`.
- Ruff on `jepa.py` and `test_graph_jepa_components.py`: exit `0`, `All checks passed!`.
- `python -m compileall -q` on those two Python files: exit `0` with no output.
- `git diff --check`: exit `0` with no output.
- Expected upstream dependency deprecation warnings remained; there were no test failures.
- Interruption point: the next unimplemented contract is rejection of a non-`JEPALatentState` predictor output. Predictor shape/metadata validators and immutable metadata snapshots remain intentionally deferred.

### Task 2.21 — predictor returns non-`JEPALatentState`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_non_latent_predictor_output -q`
- RED exit/excerpt: `1`; uncontrolled `AttributeError: 'Tensor' object has no attribute 'graph_latent'` on the next action-encoder call.
- Minimal fix: immediately require every predictor result to be a `JEPALatentState`, with `TypeError: predictor must return JEPALatentState`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.22 — predicted graph rank changes

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_predicted_graph_rank_change -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: stack expects each tensor to be equal size, but got [2] at entry 0 and [2, 2] at entry 1` on the next action-encoder call.
- Minimal fix: immediately require every predicted `graph_latent` to remain rank two, with `ValueError: predictor graph_latent must preserve rank two`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.23 — predicted graph batch changes

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_predicted_graph_batch_change -q`
- RED exit/excerpt: `1`; uncontrolled `RuntimeError: stack expects each tensor to be equal size, but got [2] at entry 0 and [1] at entry 1` on the next action-encoder call.
- Minimal fix: require every predicted graph batch to equal the initial graph batch, with `ValueError: predictor graph_latent must preserve initial batch size`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.24 — predicted graph latent width changes

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_predicted_graph_width_change -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: require every predicted graph width to equal the initial graph latent width, with `ValueError: predictor graph_latent must preserve initial latent width`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.25 — predicted object rank changes

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_predicted_object_rank_change -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: immediately require every predicted `object_latents` tensor to remain rank two, with `ValueError: predictor object_latents must preserve rank two`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.26 — predicted object row count changes

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_predicted_object_row_change -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: require every predicted object row count to equal the initial object row count, with `ValueError: predictor object_latents must preserve initial row count`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.27 — predicted object latent width changes

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_predicted_object_width_change -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: require every predicted object width to equal the initial object latent width, with `ValueError: predictor object_latents must preserve initial latent width`.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Latent-shape interruption point

Predictor output type and all six independent latent-shape dimensions are now behaviorally GREEN. Exact `object_ids` tensor identity replacement/clone is the next unstarted contract. No metadata identity, clone, mutation, or snapshot validator/test has been added in Accepted Restart Attempt 2.

### Predictor type and latent-shape aggregate verification

| Check | Exact command | Exit | Result |
|---|---|---:|---|
| Accepted Restart Attempt 2 nodes | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py -k candidate_rollout -q` | `0` | `29 passed` |
| Whole graph components | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py -q` | `0` | `69 passed` |
| Ruff | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check packages/acs-jepa-core/src/acs_jepa/jepa.py packages/acs-jepa-core/tests/test_graph_jepa_components.py` | `0` | `All checks passed!` |
| Compile | `python -m compileall -q packages/acs-jepa-core/src/acs_jepa/jepa.py packages/acs-jepa-core/tests/test_graph_jepa_components.py` | `0` | no output |
| Diff whitespace | `git diff --check` | `0` | no output |

Expected upstream dependency deprecation warnings remained in pytest output; there were no test failures. README, metadata identity/clone/mutation/snapshot tests, commit, and push remain intentionally untouched/deferred.

### Task 2.28 — replaced/reordered `object_ids` values

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_replaced_object_ids_values -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: compare every predicted `object_ids` value tensor against the original initial metadata, with `ValueError: predictor object_ids must preserve initial values`. No identity or immutable snapshot check was added in this cycle.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.29 — value-equal cloned `object_ids`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_cloned_object_ids -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: after the existing value check, require every predicted `object_ids` tensor to be the exact original tensor object, with `ValueError: predictor object_ids must preserve exact tensor identity`. No immutable snapshot was added in this cycle.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.30 — replaced/reordered `object_batch` values

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_replaced_object_batch_values -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: compare every predicted `object_batch` value tensor against the original initial metadata, with `ValueError: predictor object_batch must preserve initial values`. No identity or immutable snapshot check was added in this cycle.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.31 — value-equal cloned `object_batch`

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_cloned_object_batch -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal fix: after the existing value check, require every predicted `object_batch` tensor to be the exact original tensor object, with `ValueError: predictor object_batch must preserve exact tensor identity`. No immutable snapshot was added in this cycle.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.32 — late/final-step in-place `object_ids` mutation

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_late_in_place_object_ids_mutation -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>` despite the prior per-step value and exact-object checks.
- Test chronology proof: the predictor mutates only on call three of the three-step horizon, returns the original tensor object, and asserts `call_count == 3`, so the failure targets final-step validation rather than only the first prediction.
- Minimal fix: clone an immutable initial `object_ids` value snapshot before recurrence and compare every prediction against that snapshot, retaining the exact-object `is` check and deterministic value-preservation error.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.33 — late/final-step in-place `object_batch` mutation

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_late_in_place_object_batch_mutation -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>` despite the prior per-step value and exact-object checks.
- Test chronology proof: the predictor mutates only on call three of the three-step horizon, returns the original tensor object, and asserts `call_count == 3`, so the failure targets final-step validation rather than only the first prediction.
- Minimal fix: clone an immutable initial `object_batch` value snapshot before recurrence and compare every prediction against that snapshot, retaining the exact-object `is` check and deterministic value-preservation error.
- GREEN: the exact same focused command exited `0` with `1 passed`.

## Accepted Restart Attempt 2 — aggregate and final verification

These are the accepted restart results and supersede every aggregate/final result under rejected Attempt 1.

| Check | Exact command | Exit | Accepted result |
|---|---|---:|---|
| Candidate-rollout nodes | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py -k candidate_rollout -q` | `0` | `35 passed` (collection confirmed `35/75`, `40 deselected`) |
| Graph components + planner | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py packages/acs-jepa-core/tests/test_planner.py -q` | `0` | `83 passed` |
| Full core | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests -q` | `0` | `331 passed` |
| Scoped CLI | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests -q -k 'not test_tuning_configs_load_and_keep_required_defaults'` | `0` | `256 passed`; exactly the known tuning-default node was excluded (`1 deselected`) |
| Ruff final | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check packages/acs-jepa-core/src/acs_jepa/jepa.py packages/acs-jepa-core/src/acs_jepa/__init__.py packages/acs-jepa-core/tests/test_graph_jepa_components.py` | `0` | `All checks passed!` |
| Compile | `python -m compileall -q packages/acs-jepa-core/src/acs_jepa/jepa.py packages/acs-jepa-core/src/acs_jepa/__init__.py packages/acs-jepa-core/tests/test_graph_jepa_components.py` | `0` | no output |
| Diff whitespace | `git diff --check` | `0` | no output |
| Status | `git status --short --branch` | `0` | four tracked files modified; three plan records untracked; no commit/push |

The first combined static-check invocation exposed that the new public result was imported at package scope but absent from `__all__` (`F401`). Adding `GraphJEPACandidateRolloutOutput` to `__all__` closed that public export and the exact final Ruff command above passed. Expected upstream PyG, TorchScript, `pkg_resources`, and MLflow deprecation/future warnings remained; there were no test failures.

Final accepted implementation scope is limited to the candidate-rollout API/export, strict initial/control/predictor contracts, immutable per-rollout metadata snapshots plus exact tensor-object preservation, six focused metadata regressions, approved README boundary wording, and the audit records. No planner/decoder/CEM/tuning/Phase 1/simulator/candidate-generation/search behavior was added, and no planning-success claim was made.

### Task 2.34 — action encoder returns a non-Tensor control

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_non_tensor_control -q`
- RED exit/excerpt: `1`; uncontrolled `AttributeError: 'list' object has no attribute 'ndim'` at the control shape check instead of a deterministic type/shape contract error.
- Minimal fix: before `.ndim`, require the action-encoder result to be a `Tensor`, with `TypeError: action encoder control must be a Tensor with shape [B, D_a]`. No latent-shape snapshot was added in this cycle.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.35 — final-step in-place resize of the original graph latent

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_final_in_place_initial_graph_resize -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>` after the predictor resized the original graph tensor from `(2, 2)` to `(2, 3)` only on call three and returned the same original `JEPALatentState` and tensor.
- Test chronology proof: the three-step fixture asserts `call_count == 3`; metadata tensors and object-latent row/width dimensions remain unchanged, so no earlier metadata or object-shape check intercepts the intended graph-width assertion.
- Minimal fix: snapshot the authoritative initial graph dimensions before recurrence as immutable Python tuple `initial_graph_shape`, then validate every prediction's graph batch and width against that tuple. No object-dimension snapshot was added in this cycle.
- GREEN: the exact same focused command exited `0` with `1 passed`.

### Task 2.36 — final-step in-place resize of the original object latents

- RED command: `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py::test_candidate_rollout_rejects_final_in_place_initial_object_resize -q`
- RED exit/excerpt: `1`; `Failed: DID NOT RAISE <class 'ValueError'>` after the predictor resized the original object-latent tensor from `(4, 2)` to `(4, 3)` only on call three and returned the same original `JEPALatentState` and tensor.
- Test chronology proof: the three-step fixture asserts `call_count == 3`; object rows remain four and both metadata tensors/lengths remain unchanged, so metadata and row checks do not intercept the intended object-width assertion.
- Minimal fix: snapshot the authoritative initial object dimensions before recurrence as immutable Python tuple `initial_object_shape`, then validate every prediction's object rows and width against that tuple. Existing immutable metadata value snapshots and exact tensor-object checks remain unchanged.
- GREEN: the exact same focused command exited `0` with `1 passed`.

## Accepted Restart Attempt 2 — superseding final verification after Tasks 2.34–2.36

These results supersede the earlier Accepted Restart Attempt 2 aggregate/final table above; rejected Attempt 1 remains historical only.

| Check | Exact command | Exit | Superseding result |
|---|---|---:|---|
| Candidate-rollout nodes | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py -k candidate_rollout -q` | `0` | `38 passed` (`38` dots; no failures) |
| Graph components + planner | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests/test_graph_jepa_components.py packages/acs-jepa-core/tests/test_planner.py -q` | `0` | `86 passed` (`86` dots; no failures) |
| Full core | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests -q` | `0` | `334 passed` (`334` dots; no failures) |
| Scoped CLI | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests -q -k 'not test_tuning_configs_load_and_keep_required_defaults'` | `0` | `256 passed`; exactly the approved tuning-default node remained excluded |
| Ruff final | `UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check packages/acs-jepa-core/src/acs_jepa/jepa.py packages/acs-jepa-core/src/acs_jepa/__init__.py packages/acs-jepa-core/tests/test_graph_jepa_components.py` | `0` | `All checks passed!` |
| Compile | `python -m compileall -q packages/acs-jepa-core/src/acs_jepa/jepa.py packages/acs-jepa-core/src/acs_jepa/__init__.py packages/acs-jepa-core/tests/test_graph_jepa_components.py` | `0` | no output |
| Diff whitespace | `git diff --check` | `0` | no output |

Expected upstream PyG, TorchScript, `pkg_resources`, and MLflow deprecation/future warnings remained; there were no test failures. No README change was needed for these corrections. No commit or push was performed.
