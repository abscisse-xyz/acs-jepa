# Stage 2G Plan Review Record

Plan: `2026-07-21_194939-phase2g-action-latent-acceptance.md`
Date: 2026-07-21

## Plan Review 1 — FAIL

Reviewer: independent subagent `deleg_ba7dfec3`

Blocking findings:

1. The untouched trained argument-reconstruction head was not directly assessed.
2. G1 required identical sampled action identities, but the diagnostic did not persist them.
3. Checkpoint/metric/config/corpus identities were incomplete or impossible to validate as written.
4. Compatibility-style optional-head loading could silently evaluate random state; acceptance needs strict restoration.
5. Raw L2 margins could pass through VICReg-induced scale inflation without improved identity geometry.
6. AP/F1/category definitions and edge-case tests were underspecified.
7. CUDA probe fitting was not fully deterministic.
8. Loss direction needed exact formulas, monotonic-step selection, exact keys, and retrospective status.

Nonblocking refinements requested meaningful per-role improvements, explicit treatment of applicable sampled alternatives, compact outputs, and optional uncertainty reporting.

## Corrections

The revised plan:

- adds a separate untouched checkpoint argument-head evaluation using training-time type masks, top-1 accuracy, target-vs-best-wrong margins, and chance references;
- persists and hashes canonical 604-example manifests with reorder/mutation tests;
- pins both checkpoint hashes and Stage 2F metrics/config/split/corpus hashes, requires byte validation, and emits an immutable evidence manifest;
- requires strict present/non-null/compatible loading and eval mode for every configured module, while allowing truly disabled baseline modules;
- makes unit-normalized L2 the primary same-schema geometry metric and retains raw L2 as secondary;
- defines tie-group AP exactly, omits trapezoidal PR-AUC, defines zero-denominator behavior and category subsets, and expands edge-case tests;
- runs fitted probes entirely on deterministic CPU twice per checkpoint and requires exact decision-metric repeatability;
- defines loss decrease, monotonic step selection, exact metric keys, and retrospective interpretation;
- strengthens per-role improvement to at least two percentage points;
- excludes applicable alternatives from negative-margin distributions while retaining their true labels for classification;
- requires compact statistics outputs and permits non-decisional uncertainty reporting.

## Plan Review 2 — FAIL

Reviewer: independent subagent `deleg_b123e63f`

Most prior blockers were corrected, but implementation remained blocked because:

1. The direct argument-head production call and exact training-time tensor/mask ordering were not pinned, and edge-case tests were missing.
2. G8 referred to undefined “target counts” and did not define target-only roles, exact metric keys, or required role presence.
3. The canonical example manifest had no precommitted expected hash/count fixture.
4. Repeat comparison did not define an executable JSON projection or exact exclusions.
5. Pinned Stage 2F hashes were not mapped to exact paths, eight count keys were wildcarded, and the assessor command was absent.

## Review 2 corrections

The plan now:

- pins the production `ArgumentReconstructionHead(action_latents, candidate_object_latents, candidate_mask)` call, sorted object-ID bank, all five training-time tensors/masks, validation rules, exact overall/per-role keys, target-only behavior, and focused edge tests;
- removes undefined target counts, defines active/competitive roles and count reconciliation, and requires all roles 0–3;
- pins canonical example-manifest SHA-256 `bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9`, 117385 bytes, 604 entries, and exact category/label counts;
- defines an allowlisted `decision_projection`, exact top-level exclusions, canonical comparison bytes, and mutation tests;
- maps every pinned hash to its absolute assessor path, enumerates all eight Stage 2F count keys, and provides the concrete assessor invocation.

## Plan Review 3 — FAIL

Reviewer: independent subagent `deleg_597a2207`

The prior tensor, manifest, and source-identity corrections were verified, including all-role competitive feasibility, but implementation remained blocked because:

1. The assessor command still used placeholders and generated diagnostic paths were not fixed.
2. Repeat comparison used semantic subtree descriptions instead of one exact executable projection.
3. G8 lacked literal nested JSON paths, the exact chance formula, and complete count reconciliation.

## Review 3 corrections

- Fixed all six generated summary filenames and replaced every assessor placeholder with an absolute path.
- Defined the exact probe-summary top-level schema, six literal JSON-pointer deletions, retain-all-other-values projection, canonical bytes, unknown-key behavior, and exhaustive mutation/exclusion tests.
- Defined literal G8 overall/per-role roots, exact key sets and distribution schema, top-1/chance/margin formulas, role filtering, target-only handling, per-root count equalities, overall sum reconciliation, and finite-value requirements.

## Plan Review 4 — PASS

Reviewer: independent subagent `deleg_6a29b123`

All Review 3 blockers are resolved. The reviewer confirmed the six fixed paths and placeholder-free assessor command, exact 20-key repeat schema and six-pointer canonical projection, and complete G8 roots/formulas/count contracts are operationally implementable without interpretation and do not contradict prior corrections or the governing specification.

Diagnostic implementation is authorized.

## Implementation and evidence review 1 — FAIL

Reviewer: independent subagent `deleg_3f148e2a`

The empirical values and G1/G2/G4/G5 PASS plus G3/G6/G7/G8 FAIL verdict were independently reproduced, but commit authorization was withheld for two assessor-integrity blockers:

1. The assessor trusted manifest identity declared inside probe summaries without opening the four canonical manifest files, and the evidence manifest omitted those files and four probe `details.json` outputs.
2. G1 allowed configured Phase 2 goal/contrastive modules to be marked disabled; it did not enforce the exact baseline-versus-Phase 2 restoration status map.

### Corrections

Strict RED/GREEN regression tests reproduced both failures. The assessor now:

- requires each declared manifest path to equal the fixed sibling `example_manifest.json` path;
- reads strict UTF-8 JSON, verifies canonical bytes, 604 records, 117385 bytes, and the pinned SHA-256;
- opens/validates each sibling `details.json`;
- adds all four manifests and all four details files to `evidence_manifest.json` (21 evidence files total);
- enforces exact restoration states: baseline JEPA/goal restored and three action auxiliaries disabled; Phase 2 JEPA/goal/contrastive/argument/applicability all restored.

The corrected assessor returned exit 0 with the same valid empirical decision `FAIL`. Implementation and Evidence Review 2 is pending; no assessment commit is authorized yet.

## Implementation and evidence review 2 — PASS

Reviewer: independent subagent `deleg_210f2d38`

Both Review 1 blockers are resolved. The reviewer independently verified all 21 evidence/output hashes and byte counts, strict direct validation of all four canonical manifests and details files, exact baseline and Phase 2 restoration maps, and adversarial rejection of missing/changed manifests and disabled configured modules. The corrected assessor returned exit 0 with the unchanged honest empirical verdict: G1/G2/G4/G5 PASS and G3/G6/G7/G8 FAIL.

Focused tests (53), full core, scoped CLI, Ruff, `compileall`, and `git diff --check` passed. Creation of the SSH-signed Stage 2G assessment commit is authorized; do not push.
