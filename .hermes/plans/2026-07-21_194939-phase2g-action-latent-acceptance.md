# Stage 2G Plan — Empirical action-latent acceptance assessment

Date: 2026-07-21
Status: complete; empirical decision FAIL; Implementation and Evidence Review 2 PASS; ready for signed assessment commit
Governing specification: `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2 acceptance
Prerequisites: signed Stage 2E commit `0a59c11`; signed Stage 2F commit `c1550c0`

## Objective

Run a fixed, apples-to-apples empirical assessment of whether the Phase 2 auxiliary objectives materially repair grounded-action aliasing/collapse on the 44-transition `p166`/`p192` validation slice. Compare the original baseline checkpoint against the Stage 2F checkpoint with identical frozen diagnostics. Evaluate untouched trained applicability and argument-reconstruction heads separately from newly fitted frozen-representation probes. Emit a deterministic PASS/FAIL decision from thresholds fixed before execution.

A Stage 2G FAIL is valid. It keeps broad tuning paused and identifies unsatisfied mechanisms; it must not trigger coefficient search, retraining, decoder/CEM execution, or planning.

## Fixed corpus, checkpoints, and immutable identities

```text
corpus: /opt/data/workspace/acs-jepa-tuning-data/smoke
baseline: /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt
phase2: /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt
split: val (p166, p192; 44 transitions)
latent-statistics device: cuda
probe device: cpu
probe seed: 20260717
probe epochs: 200
probe eval fraction: 0.25
hard negatives per category: 4
```

Pinned inputs:

| Input | SHA-256 |
|---|---|
| baseline checkpoint | `65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e` |
| Phase 2 checkpoint | `7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968` |
| Stage 2F train metrics | `f3e94b0d6c8a38b78ba6ce209f6c0ab31a3cf49bac1c450affcaf60ca30f0e43` |
| Stage 2F held-out metrics | `5ce3ab7aa535bf68990e000b79c8cd29a1bf14b68670db67eb06a19d5f123954` |
| Stage 2F resolved config | `01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5` |
| Stage 2F split manifest | `02aa33b0aa12008142fe08940a16aff81554d7fa3c2345866be6d9b65d9842ae` |
| smoke corpus manifest | `055b5616d7616331e6edbc8f72523f07e8c1808e5aa31089c8420f01aaf0e400` |

The assessor validates bytes, not path strings. It writes an immutable input/output hash manifest. Outputs live under:

`/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/`

No bulky generated diagnostics are committed.

## Historical baseline fixed before execution

- global effective rank: `2.1275684834`;
- global std minimum: `0.0073343054`;
- within-schema variance fraction: `1.4071544e-05`;
- raw true-reference nearest same-schema wrong L2 median: `8.9686942e-05`;
- raw minimum reference margin: `3.3233016e-05`;
- schema probe eval accuracy: `0.5496688485`;
- role/object probe eval accuracy: `0.1332046241`;
- frozen applicability probe eval AUROC: `0.6166666746`;
- frozen applicability median hard-negative margin: `6.8426132e-05`;
- frozen applicability positive prevalence: `15/151 = 0.0993377`.

Baseline diagnostics are rerun after instrumentation so all new metrics use identical code.

## 1. Strict diagnostic checkpoint restoration (TDD)

Update `script/action_diag_common.py` with a strict diagnostic-loading mode:

- every module constructed by the checkpoint's resolved config must have a present, non-null, shape-compatible state dictionary and load successfully;
- missing, null, or incompatible configured states are fatal;
- a module absent because the baseline config disables it is valid;
- JEPA and every restored goal, contrastive anchor, argument head, and applicability head are put in evaluation mode;
- restoration presence/status metadata is emitted.

Tests cover successful restoration, missing/null/incompatible configured states, and baseline-disabled modules. Acceptance evidence must not silently use CLI compatibility warnings or randomly initialized optional heads. No model/training code changes.

## 2. Supervised diagnostic completion (TDD)

Extend `script/diagnose_action_supervised_probes.py`.

### Exact binary metrics

Report threshold-zero accuracy, precision, recall, F1, AUROC, positive prevalence, and average precision (AP). AP is not trapezoidal PR-AUC: group equal scores, sort distinct score groups descending, update cumulative TP/FP after each complete tie group, and compute `sum(delta_recall * precision_at_group_boundary)`.

Ranking metrics are `null` for empty or single-class subsets. When both classes exist but zero-threshold predicts no positives, precision/recall/F1 are `0.0`. Ordinary confusion-matrix values apply to all-positive predictions.

For each negative category, form a subset containing every trace positive plus every candidate in that category, retaining actual oracle labels. Report all binary metrics. Margin distributions compare trace positives only with sampled candidates that are actually inapplicable; applicable alternatives are excluded from “negative” margins.

Tests cover perfect, reversed, imbalanced, empty/single-class, all-positive, all-negative, equal-score tie permutations, and a category with an applicable alternative.

### Canonical example identity

Persist a compact manifest containing group, problem, step, category, action schema/arguments, and applicability label. Canonically sort entries before UTF-8 JSON hashing: input reordering preserves SHA-256, any field mutation changes it. Add reorder/mutation tests. Baseline and Phase 2 manifests must match exactly.

Canonical bytes are UTF-8 `json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\\n"`, after sorting by `(group, category, action.name, tuple(action.arguments), applicability_label, problem, step)`. The precommitted expected 604-entry manifest is:

```text
sha256: bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9
bytes: 117385
category counts: one_arg_substitution=176, random_other_schema=176,
                 random_same_schema=176, role_swap=32, trace=44
label counts: applicable=62, inapplicable=542
```

Add a fixed test against this identity and counts, in addition to generic mutation/reorder tests.

### Separate fitted and restored heads

Keep existing newly fitted schema, role/object, and applicability probes as frozen-representation diagnostics.

Add `checkpoint_applicability_head`:

- baseline: `null` because no module is configured;
- Phase 2: evaluate the untouched strictly restored head on the same group-disjoint train/eval examples;
- report the exact binary/category/margin metrics above;
- never fit or threshold-select this head.

Add `checkpoint_argument_reconstruction_head`:

- baseline: `null`;
- Phase 2: evaluate the untouched restored head on the 44 validation trace actions only;
- call the production contract exactly as `ArgumentReconstructionHead(action_latents, candidate_object_latents, candidate_mask)`;
- construct a problem-local object bank sorted by object ID, so `argument_target_indices` index that bank exactly as training does;
- construct `argument_mask` from active schema roles; construct `object_mask` over real bank entries and false batch padding; construct `argument_candidate_mask` by exact normalized action-parameter/object type equality, intersected with `object_mask`; and pass the dense bank as `candidate_object_latents` and `argument_candidate_mask` as `candidate_mask`;
- validate every active target is nonnegative, inside the real object bank, and true in `argument_candidate_mask`; exclude inactive roles, type-invalid candidates, and padded candidates from logits/argmax/chance;
- report exact overall and per-role (`0`, `1`, `2`, `3`) keys: `active_role_count`, `competitive_role_count`, `top1_accuracy`, `chance_accuracy`, `valid_candidate_count` distribution, and `target_minus_best_wrong_margin` distribution;
- a role is competitive when it has at least one valid wrong candidate. When only the target is valid, it still contributes accuracy/chance but not a margin; its margin is undefined and the distribution count remains zero;
- keep it separate from the newly fitted role/object probe.

Focused tests cover target-in-mask rejection, object-ID bank ordering, type-invalid and padded exclusion, exact valid-candidate chance, correct/wrong top-1, best-wrong margins, inactive roles, and the target-only/no-wrong case.

## 3. Scale-robust geometry diagnostic (TDD)

Extend `script/diagnose_action_latent_statistics.py` to report true-reference same-schema nearest-wrong margins both as raw L2 and after unit-normalizing each action latent (`unit_l2`). Add `--omit-details` so compact summaries exclude multi-million-row detail arrays. Tests cover normalization, zero-norm safety, unchanged raw outputs, and compact mode.

Raw L2 remains secondary because VICReg can change scale. Unit-normalized L2 is the primary geometry gate.

## 4. Determinism

Run supervised probes entirely on CPU with deterministic Torch algorithms. Record Python/Torch versions, backend, deterministic state, and thread settings. Run each checkpoint twice into distinct outputs. Require:

- byte-identical canonical example manifests;
- identical split/group identities;
- exact equality of every decision metric after excluding runtime and path metadata.

Define one shared `decision_projection(summary)` used by diagnostic tests and the assessor. First require the exact top-level key set:

```text
dataset, checkpoint, split, seed, device, per_category, eval_fraction,
epochs, learning_rate, metadata, probe_split, label_counts, category_counts,
example_manifest, checkpoint_restoration, probes,
checkpoint_applicability_head, checkpoint_argument_reconstruction_head,
environment, runtime_seconds
```

Deep-copy the summary and delete exactly these JSON pointers and no others:

```text
/dataset
/checkpoint
/device
/runtime_seconds
/environment
/example_manifest/path
```

All remaining values—including all nested keys under `metadata`, `probes`, both checkpoint-head sections, `checkpoint_restoration`, and any additional nested metric key—form the projection. Unknown top-level keys are invalid; an extra nested key is retained and therefore causes repeat mismatch unless present identically. Serialize with `json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"` and compare bytes. Tests mutate every retained top-level subtree, one leaf in every train/eval/head metric tree, and an added optional nested metric; all must be rejected. Tests separately prove changes only to each of the six deleted pointers are accepted.

The assessor rejects mismatch. Latent-statistics encoding is deterministic eval-only CUDA execution with no stochastic fitting.

## 5. Deterministic assessor (TDD)

Add `script/assess_action_latent_phase2.py` with focused tests. It accepts explicit paths for both checkpoints, corpus/split/config/Stage 2F metrics, baseline and Phase 2 diagnostic repeats, and output. It rejects missing, nonfinite, wrong-hash, wrong-checkpoint, wrong split/count, manifest mismatch, nondeterministic repeat, or malformed head-restoration evidence.

Exact immutable assessor inputs are:

```text
/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/metrics/train.jsonl
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/metrics/eval.jsonl
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/artifacts/split_manifest.json
/opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json
```

It identifies first/final held-out records by validated strictly increasing `step`; requires positive finite first losses; computes relative decrease as `(first - final) / abs(first)`; and checks exact loss/count keys. G2 is retrospective because Stage 2F metrics predate this plan: it is only an activity/integrity gate, never evidence for threshold selection.

It writes PASS or FAIL and returns exit 0 after valid assessment; malformed evidence exits nonzero. It emits compact JSON/Markdown without nearest-neighbor details.

## 6. Fixed diagnostic commands

Run each checkpoint once for compact full same-schema statistics:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_latent_statistics.py DATASET \
  --checkpoint CHECKPOINT --output OUTPUT \
  --device cuda --split val --same-schema-only \
  --chunk-size 2048 --seed 0 --omit-details
```

This covers all 174,780 same-schema candidates and all 44 transitions.

Run each checkpoint twice for probes:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_supervised_probes.py DATASET \
  --checkpoint CHECKPOINT --output OUTPUT \
  --device cpu --split val --per-category 4 \
  --eval-fraction 0.25 --epochs 200 \
  --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```

After all diagnostics, invoke the assessor with no placeholders. Fixed generated summaries are:

```text
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/statistics/summary.json
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/phase2/statistics/summary.json
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/summary.json
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run2/summary.json
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/phase2/probe_run1/summary.json
/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/phase2/probe_run2/summary.json
```

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/assess_action_latent_phase2.py \
  --baseline-checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --phase2-checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
  --train-metrics /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/metrics/train.jsonl \
  --heldout-metrics /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/metrics/eval.jsonl \
  --resolved-config /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml \
  --split-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/artifacts/split_manifest.json \
  --corpus-manifest /opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json \
  --baseline-statistics /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/statistics/summary.json \
  --phase2-statistics /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/phase2/statistics/summary.json \
  --baseline-probe /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/summary.json \
  --baseline-probe-repeat /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run2/summary.json \
  --phase2-probe /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/phase2/probe_run1/summary.json \
  --phase2-probe-repeat /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/phase2/probe_run2/summary.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/assessment
```

## 7. Precommitted conjunctive acceptance gates

### G1 — Evidence identity and completeness

- both statistics runs cover 44 transitions and retain all true actions;
- all four probe runs contain 604 examples, identical 151/453 splits, group lists, category/label counts, and canonical example-manifest SHA-256;
- repeat decision metrics are exactly equal;
- every decision metric is finite when defined;
- all pinned checkpoint, metric, config, split, and corpus hashes match;
- generated evidence manifest pins every diagnostic output.

### G2 — Retrospective auxiliary activity

Using minimum/maximum monotonic held-out steps and exact loss keys `action_vicreg_loss`, `action_contrastive_loss`, `argument_reconstruction_loss`, and `applicability_loss`, require all eight count keys in both selected records:

```text
term/action_vicreg_num_samples
term/action_contrastive_num_examples
term/action_contrastive_num_negatives
term/argument_num_active_roles
term/argument_num_competitive_roles
term/applicability_num_examples
term/applicability_num_positive
term/applicability_num_negative
```

- each loss is non-increasing;
- at least one relative decrease is `>= 0.01`;
- every selected effective count is positive.

### G3 — Collapse/aliasing

Phase 2 must satisfy all:

- global effective rank `>= 4.0`;
- global std minimum `>= 0.02`;
- within-schema variance fraction `>= 0.001`;
- none is below rerun baseline.

### G4 — True-vs-near-miss geometry

- unit-normalized L2 median `>= 1.5x` rerun baseline;
- unit-normalized L2 minimum is no lower than rerun baseline;
- raw L2 median `>= 1.793738847e-04` (2x historical baseline);
- raw L2 minimum `>= 3.323301644e-05`;
- all 44 references have a wrong same-schema comparison.

### G5 — Fitted grounded argument information

- role/object eval accuracy `>= 0.1832046241` (historical baseline +5 points);
- at least three of four per-role accuracies improve by `>= 0.02` over rerun baseline;
- schema eval accuracy is no more than 5 points below rerun baseline.

### G6 — Fitted frozen applicability separability

Eval split:

- AUROC `>= 0.70`;
- AP `>= 0.25` versus approximately 0.0993 prevalence;
- F1 `>= 0.25`;
- overall inapplicable-candidate median margin > 0;
- one-argument substitution, random-same-schema, and role-swap median margins > 0;
- AUROC and AP exceed rerun baseline.

### G7 — Restored trained applicability head

Phase 2 eval split:

- AUROC `>= 0.70`;
- AP `>= 0.25`;
- F1 `>= 0.25`;
- overall and the three hard-category median margins from G6 are > 0.

Baseline must report no checkpoint applicability head.

### G8 — Restored trained argument head

Literal JSON roots are:

```text
checkpoint_argument_reconstruction_head.overall
checkpoint_argument_reconstruction_head.per_role.0
checkpoint_argument_reconstruction_head.per_role.1
checkpoint_argument_reconstruction_head.per_role.2
checkpoint_argument_reconstruction_head.per_role.3
```

Each root contains exactly `active_role_count`, `competitive_role_count`, `top1_accuracy`, `chance_accuracy`, `valid_candidate_count`, and `target_minus_best_wrong_margin`. Each distribution contains exactly `count`, `min`, `median`, `mean`, `max`.

For active rows `i`, let `N_i` be the number of true entries in `argument_candidate_mask[i]`. Define `top1_accuracy = sum(argmax_valid(logits_i) == target_i) / active_role_count`; define `chance_accuracy = sum(1 / N_i) / active_role_count`. For competitive rows (`N_i >= 2`), define margin `logit_i[target_i] - max(logit_i[j] for valid j != target_i)`. Compute the same formulas after filtering by literal role ID for each per-role root.

Phase 2 validation trace actions must satisfy:

- overall type-masked top-1 accuracy `>= 0.25` and `>= 2x` overall chance reference;
- overall target-minus-best-wrong median margin > 0;
- at least three roles exceed their chance reference by `>= 0.02`;
- roles `0`, `1`, `2`, and `3` must all be present with `active_role_count > 0` and `competitive_role_count > 0`;
- at every root, `valid_candidate_count.count == active_role_count` and `target_minus_best_wrong_margin.count == competitive_role_count`;
- overall active and competitive counts equal the respective sums across roles 0–3; overall valid-candidate and margin distribution counts equal the corresponding overall active and competitive counts;
- every scalar and nonempty distribution value is finite. Target-only rows are valid for accuracy/chance but do not contribute to competitive margin counts.

Baseline must report no checkpoint argument head.

## 8. Decision artifacts

Run the assessor once. Write external `phase2g/assessment/summary.json` and `evidence_manifest.json`, plus checked-in `script/ACTION_LATENT_PHASE2_ACCEPTANCE.md` with exact commands, hashes, compact baseline/Phase 2 metrics, all gate verdicts, overall PASS/FAIL, and next allowed action.

- PASS: Phase 2 representation acceptance is complete; broad tuning may be planned separately but does not start here.
- FAIL: tuning remains paused. Report exact mechanisms and only the smallest evidence-driven next architecture/objective phase; do not train or tune.

Optionally report bootstrap sensitivity/intervals for AUROC/AP and the small role-swap subset, but they do not alter fixed gate verdicts.

## Verification and review

Run focused diagnostic/assessor tests, full core, scoped CLI excluding the documented unrelated missing tuning-config test, Ruff, `compileall`, `git diff --check`, and static/security scans. Dispatch independent Stage 2G implementation/evidence review, fix blockers, rerun affected gates, then create and verify an SSH-signed assessment commit. Do not push.

## Non-goals

- no decoder/CEM/planner execution;
- no training, retraining, coefficient changes, sweep, or seed search;
- no use of all-corpus examples to fit or select thresholds;
- no broad-generalization claim from smoke data;
- no threshold reinterpretation after Phase 2 outputs;
- no publication or push.
