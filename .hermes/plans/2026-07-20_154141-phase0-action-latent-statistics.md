# Phase 0 Action Latent Statistics Diagnostic Implementation Plan

> **For Hermes:** Implement this plan task-by-task with separate plan review, implementation, then independent code review.

**Goal:** Add the Phase 0 measurement-only diagnostic required by `script/ACTION_LATENT_SOLUTION_SPEC.md`: action-latent distribution statistics before adding losses or planner behavior changes.

**Architecture:** Keep the first slice narrow and non-invasive. Add pure tensor/statistics helpers plus a CLI diagnostic script that reuses `script/action_diag_common.py` and existing checkpoint/corpus loading. Do not alter training, decoding, or planner behavior in this slice.

**Tech Stack:** Python 3.13, PyTorch, existing ACS-JEPA CLI/package layout, pytest, uv.

---

## Scope and assumptions

- Repository: `/opt/data/workspace/acs-jepa`.
- This plan covers only Phase 0 statistics diagnostics, not Phase 1-4 model/planner changes.
- Offline simulator/applicable-action oracle may be used only for labels/metrics, consistent with the spec.
- The diagnostic should work on CPU for small smoke slices, with `--device cuda` still supported by existing loader helpers.
- Existing scripts already implement candidate enumeration and action latent scoring; this slice should reuse those patterns without refactoring them.

## Research/spec review checkpoints

The implementation must satisfy these spec points:

- Measure global per-dimension action-latent standard deviation.
- Measure covariance off-diagonal penalty.
- Measure effective rank/eigenvalue spectrum.
- Measure per-schema variance where support is sufficient.
- Measure same-schema nearest-wrong margin distribution.
- Include schema-vs-argument variance decomposition if feasible for this slice.
- Avoid production-time dependence on `SimulatorEngine.applicable_actions()`.

## Task 0: Plan review gate

**Objective:** Confirm the implementation plan satisfies the specification before coding starts.

**Files:**
- Create: `.hermes/plans/2026-07-20_154141-phase0-plan-review.md`

**Review requirements:**

- No implementation starts until this review is complete.
- Any blocking review comments are folded back into this plan before Task 1.
- `schema-vs-argument variance decomposition` is required. If a future implementation cannot compute it, `summary.json` must include an explicit `not_available_reason` instead of silently omitting it.
- Same-schema nearest-wrong reporting must include the true/reference action latent to nearest wrong same-schema candidate per transition, not only generic candidate-to-candidate nearest neighbors.
- Cross-checkpoint comparison of baseline, inverse-dynamics, and RNN-action-encoder checkpoints is a follow-up Phase 0 reporting step when those local artifacts are available; this first slice provides the reusable diagnostic needed for that comparison.

Research rationale to preserve:

- VICReg/SIGReg motivate variance/covariance/effective-rank statistics, but this phase must not add those losses yet.
- Latent-action grounding papers motivate separating transition-predictive action latents from grounded-action recoverability; this diagnostic reports representation geometry only.

## Task 1: Add tested tensor statistics helpers

**Objective:** Create small deterministic helpers for action-latent distribution statistics independent of checkpoint loading.

**Files:**
- Create: `script/action_latent_statistics.py`
- Create: `acs-jepa-cli/tests/test_action_latent_statistics.py`

**Step 1: Write failing tests**

Tests should cover:

1. `latent_distribution_stats(latents)` returns:
   - `count`
   - `dim`
   - `std_mean`
   - `std_min`
   - `std_values`
   - `cov_offdiag_mean_sq`
   - `effective_rank`
   - `eigenvalues`
2. A constant latent matrix has zero/std collapse-like values and effective rank `0.0`.
3. A simple diagonal-ish latent matrix has positive std and finite effective rank.
4. `schema_group_stats(latents, schema_ids, min_count=2)` skips under-supported schemas.
5. `same_schema_nearest_wrong_margins(latents, schema_ids, action_keys)` reports the nearest wrong same-schema distance and margin when there are at least two actions for a schema.
6. `reference_same_schema_margins(latents, schema_ids, action_keys, reference_mask, group_ids)` reports true/reference-action margins within each transition group.
7. `schema_argument_variance_decomposition(latents, schema_ids)` returns between-schema and within-schema variance summaries.

Run expected RED command:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py -q
```

Expected: FAIL because `script/action_latent_statistics.py` does not exist yet.

**Step 2: Implement minimal helper module**

Create `script/action_latent_statistics.py` with pure functions only:

- `_as_2d_float(latents: torch.Tensor) -> torch.Tensor`
- `latent_distribution_stats(latents: torch.Tensor) -> dict[str, Any]`
- `schema_group_stats(latents: torch.Tensor, schema_ids: Sequence[str], min_count: int) -> dict[str, Any]`
- `same_schema_nearest_wrong_margins(latents: torch.Tensor, schema_ids: Sequence[str], action_keys: Sequence[Any]) -> dict[str, Any]`
- `reference_same_schema_margins(latents: torch.Tensor, schema_ids: Sequence[str], action_keys: Sequence[Any], reference_mask: Sequence[bool], group_ids: Sequence[Any]) -> dict[str, Any]`
- `schema_argument_variance_decomposition(latents: torch.Tensor, schema_ids: Sequence[str]) -> dict[str, Any]`

Implementation details:

- Use CPU tensors internally; detach inputs.
- Use population covariance for stability on small diagnostic samples.
- Define effective rank as `exp(entropy(normalized_positive_eigenvalues))`; return `0.0` if total eigenvalue mass is zero.
- Use squared L2 distances for nearest-neighbor ordering, and report distances as square roots.
- JSON outputs must be plain Python floats/ints/lists.

**Step 3: Run GREEN tests**

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py -q
```

Expected: PASS.

## Task 2: Add CLI diagnostic script

**Objective:** Add `script/diagnose_action_latent_statistics.py` that encodes candidate grounded actions and writes JSON summaries.

**Files:**
- Create: `script/diagnose_action_latent_statistics.py`
- Modify tests if needed: `tests/test_action_latent_statistics.py`

**Step 1: Write failing CLI/import test**

Test that:

- The script module can be imported without executing `main()`.
- `parse_args()` or a parser factory accepts key options if exposed.
- The script references helper functions rather than duplicating statistics code.

Run expected RED command:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py -q
```

Expected: FAIL because `diagnose_action_latent_statistics.py` does not exist.

**Step 2: Implement minimal diagnostic CLI**

CLI arguments:

```text
dataset_dir
--checkpoint PATH
--output PATH
--device {cpu,cuda,mps}
--split {train,val,test,all}
--max-transitions INT
--max-candidates-per-state INT
--same-schema-only
--min-schema-count INT
--chunk-size INT
--seed INT
```

Runtime behavior:

1. Load checkpoint/corpus with `load_checkpoint_bundle()`.
2. Select split with `select_split()`.
3. For each transition up to `--max-transitions`:
   - Build `ActionDecodingSpace` for the parsed problem.
   - Enumerate grounded actions, optionally filter to the true action schema.
   - Optionally downsample candidates deterministically with `random.Random(seed + transition_index)` while always retaining the true action.
   - Encode source state with `encode_state()`.
   - Encode candidate actions in chunks using the model action encoder and repeated latent state, matching `diagnose_action_latent_geometry.py`.
   - Record schema ids, action keys, transition group ids, and a reference-action mask.
4. Aggregate all candidate latents across selected transitions.
5. Write:
   - `summary.json` with global stats, per-schema stats, generic same-schema nearest-neighbor stats, true/reference-action same-schema margin stats, schema-vs-argument variance decomposition, runtime, sample counts, and command metadata.
   - `details.json` with compact per-transition counts and true action metadata.

Do not add MLflow logging in this first slice unless it is already trivial; JSON artifacts are sufficient for Phase 0.

**Step 3: Run GREEN import/helper tests**

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py -q
```

Expected: PASS.

## Task 3: Smoke-run the diagnostic if data/checkpoint exists

**Objective:** Verify the diagnostic works on available local artifacts without requiring unavailable data.

**Files:**
- No code changes unless a real bug is discovered.

**Step 1: Check artifacts**

```bash
test -d /opt/data/workspace/acs-jepa-tuning-data/smoke && echo data=yes || echo data=no
test -f /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt && echo checkpoint=yes || echo checkpoint=no
```

**Step 2: If both exist, run a bounded CPU diagnostic**

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_latent_statistics.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --output /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/diagnostics/action_latent_statistics_val_2 \
  --device cpu \
  --split val \
  --same-schema-only \
  --max-transitions 2 \
  --max-candidates-per-state 128 \
  --seed 0
```

Expected: command exits 0 and writes `summary.json` and `details.json`.

**Step 3: If artifacts are absent**

Run only unit/import tests and report that full smoke execution is blocked by missing local data/checkpoint.

## Task 4: Separate implementation review

**Objective:** Review the resulting diff independently from the implementation step.

Review checklist:

- Spec compliance: all Phase 0 requested statistics are present or explicitly marked not implemented.
- Research compliance: statistics support VICReg/SIGReg diagnosis without adding training losses prematurely.
- Non-goal compliance: no production planner/decoder behavior changed; applicable-actions oracle not used as production generator.
- Coding practice: helpers are testable, deterministic, JSON-safe, no broad refactors, no hardcoded workspace paths in code.
- Performance: chunked encoding, candidate cap, true action retained under sampling.

Verification commands:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_latent_statistics.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest packages/acs-jepa-core/tests/test_graph_losses.py -q
python -m compileall script/action_latent_statistics.py script/diagnose_action_latent_statistics.py
```

## Task 5: Commit after verification

Only after tests and review pass:

```bash
git add script/action_latent_statistics.py script/diagnose_action_latent_statistics.py acs-jepa-cli/tests/test_action_latent_statistics.py
git commit -m "feat: add action latent statistics diagnostic"
```

Use the configured SSH signing workflow; if signing fails, fix signing rather than committing unsigned.
