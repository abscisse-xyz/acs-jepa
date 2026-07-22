# Updated Phase 0 Falsification Diagnostics Implementation Plan

> **For Hermes:** Execute each stage as an independent `plan -> plan review -> RED/GREEN implementation -> implementation/evidence review` gate. A later stage may use only artifacts accepted by the preceding stage. Do not start Updated Phase 1 from this plan.

**Goal:** Determine, without model training or planner changes, whether the current ACS-JEPA representation contains enough grounded action identity, precondition-relevant state information, and transition discrimination to justify a minimal action-latent repair—or whether the project must select a replacement branch.

**Architecture:** Four measurement-only stages share one immutable 604-record candidate manifest and the fixed `p166`/`p192` 44-transition validation slice. Stage 0A measures schema-residual geometry; 0B performs a controlled applicability recoverability ladder; 0C tests whether nearest invalid same-schema candidates are transition-equivalent; 0D evaluates fixed candidate scorers and emits one deterministic continuation/branch decision. Every checkpoint/module is restored strictly, all learned probes train only on the fixed train groups, all thresholds are preregistered here, and a final assessor validates byte identities and repeat projections.

**Tech stack:** Python 3.13, PyTorch, PyTorch Geometric, ACS-JEPA core/CLI, deterministic CPU probe fitting, CUDA read-only latent extraction, pytest, Ruff.

---

## 1. Governing inputs and invariants

### 1.1 Source authority

- Governing spec: `script/ACTION_LATENT_UPDATED_SPEC.md`
- Governing spec SHA-256: `b4146d21b6082ec085628f7d1c56ff135c9fe606c8307db8b84689e449ec9606`
- Governing Git commit: `94604ab5cfe73038f35f62ba160da215a47dd090`
- Prior empirical record: `script/ACTION_LATENT_PHASE2_ACCEPTANCE.md`

If the spec bytes change, stop and re-plan/re-review; the assessor must reject the changed identity.

### 1.2 Fixed model/data evidence

| Input | Absolute path | SHA-256 |
|---|---|---|
| Baseline checkpoint | `/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt` | `65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e` |
| Baseline config | `/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml` | `f65e2cbb33fb3e7322e0cc0c5e8a8f01e9ca7c408e4594516d50a9735c673193` |
| Phase 2 checkpoint | `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt` | `7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968` |
| Phase 2 config | `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml` | `01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5` |
| Corpus manifest | `/opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json` | `055b5616d7616331e6edbc8f72523f07e8c1808e5aa31089c8420f01aaf0e400` |
| Candidate manifest | `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json` | `bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9` |

Candidate-manifest contract:

- canonical UTF-8 JSON plus one trailing newline;
- exactly 604 records and 117,385 bytes;
- groups are exactly these 44 literal IDs: `p166:0`, `p166:1`, `p166:2`, `p166:3`, `p166:4`, `p166:5`, `p166:6`, `p166:7`, `p166:8`, `p166:9`, `p166:10`, `p166:11`, `p166:12`, `p166:13`, `p166:14`, `p166:15`, `p166:16`, `p166:17`, `p166:18`, `p166:19`, `p166:20`, `p166:21`, `p166:22`, `p192:0`, `p192:1`, `p192:2`, `p192:3`, `p192:4`, `p192:5`, `p192:6`, `p192:7`, `p192:8`, `p192:9`, `p192:10`, `p192:11`, `p192:12`, `p192:13`, `p192:14`, `p192:15`, `p192:16`, `p192:17`, `p192:18`, `p192:19`, `p192:20`;
- categories/counts: trace 44, one-argument substitution 176, random-same-schema 176, role-swap 32, random-other-schema 176;
- labels: 62 applicable, 542 inapplicable;
- one and only one trace record per group;
- each record has exact keys `action`, `applicability_label`, `category`, `group`, `problem`, `step`; `action` has exact keys `name`, `arguments`.

No stage regenerates, resamples, relabels, or copies this manifest. Every stage opens and validates these bytes directly. Labels are immutable offline evidence; no live simulator/oracle is allowed during diagnostic evaluation.

### 1.3 Fixed output root

All generated artifacts live under:

`/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0`

The literal first official command is Stage 0A baseline `run1`. It must require that this root does not exist, create it atomically, and write `root_identity.json` before creating its destination. That marker has the exact schema in §8.2 and pins all immutable inputs. Every later command must validate the marker byte-for-byte, refuse a missing/changed marker, and refuse only its own already-existing destination directory. A failed/partial first command requires deletion of the entire root and a clean restart; no CLI exposes an overwrite/destructive flag. Tests use temporary directories only. Official outputs are never checked into Git; their hashes are pinned by the assessor and summarized in `script/ACTION_LATENT_UPDATED_PHASE0_DECISION.md`.

### 1.4 Prohibited behavior

- no train/eval command for JEPA model parameters;
- no optimizer step outside fixed diagnostic probes;
- no checkpoint mutation;
- no planner, decoder, CEM, rollout campaign, coefficient sweep, or Updated Phase 1 objective;
- no call to `SimulatorEngine.applicable_actions()` or any live simulator in dataset reading, feature extraction, scoring, or assessment;
- no use of labels, categories, problem names, group IDs, or split membership as model input features;
- no post-result threshold, feature, model-capacity, seed, or branch-precedence selection.

## 2. Research basis for the measurements

This plan uses literature to constrain measurement, not to import untested architecture:

1. Garrido et al., **RankMe**, arXiv:2210.02885v3, and Papyan et al., **Prevalence of Neural Collapse**, arXiv:2008.08186v2: effective rank and class-mean/within-class decomposition diagnose collapse. Therefore Stage 0A reports full spectra, pooled/per-schema statistics, and both global-schema and source-state/schema residuals; within-schema collapse is pathological here only because grounded argument identity remains required.
2. Alain & Bengio, **Understanding Intermediate Layers Using Linear Classifier Probes**, arXiv:1610.01644v4, and Hewitt & Liang, **Designing and Interpreting Probes with Control Tasks**, arXiv:1909.03368v1: probe capacity may reveal nonlinear recoverability or merely memorize. Therefore Stage 0B fixes one linear and one small nonlinear probe, identical group splits/budgets, and label-permutation controls.
3. Saito & Rehmsmeier, **The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets**, DOI:10.1371/journal.pone.0118432: sparse applicability makes AUROC insufficient. Therefore AP, prevalence, category margins, and precision/recall remain explicit gates/evidence.
4. Guo et al., **On Calibration of Modern Neural Networks**, arXiv:1706.04599v2: probability calibration and ranking/threshold selection are distinct. Therefore the train-only decision threshold is frozen before evaluation, while NLL/Brier/reliability are reported separately and never claimed to improve score ordering.
5. Gelada et al., **DeepMDP**, arXiv:1906.02736v1, and Zhang et al., **Learning Invariant Representations for RL without Reconstruction**, arXiv:2006.10742v2: transition/bisimulation evidence requires prediction behavior, not latent proximity alone. Therefore Stage 0C jointly tests error against the observed successor and true-vs-wrong predicted-transition separation; invalid actions are not called genuinely behaviorally equivalent merely because JEPA under-specifies them.
6. Hafner et al., **PlaNet**, arXiv:1811.04551v5: latent CEM is established for native continuous controls, not interpolation among sparse symbolic tuples. This supports treating teacher-forced transition scoring as diagnostic only and preserves the ban on a new continuous-planning run in Phase 0.
7. Schrittwieser et al., **MuZero**, arXiv:1911.08265v2: learned models can score explicit discrete actions without requiring invertible continuous action vectors. Therefore Stage 0D measures bounded candidate ranking separately from continuous planner optimization; it does not claim MuZero equivalence or add search.

No paper determines ACS-JEPA thresholds; thresholds below come from the updated spec and preregistered operational definitions.

## 3. Shared deterministic contracts

### 3.1 Checkpoint restoration

Reuse `script/action_diag_common.py::load_checkpoint_bundle(..., include_restoration_metadata=True)`. Baseline must report JEPA and goal head `restored`, with contrastive/argument/applicability modules `disabled`. Phase 2 must report all five modules `restored`. Missing, null, incompatible, or unexpected restoration states invalidate evidence before gate calculation.

Every module is in eval mode; model parameters have `requires_grad=False` during extraction; official extraction runs inside `torch.inference_mode()`.

### 3.2 Group-disjoint split

Do not regenerate the accepted Phase 2G split from manifest encounter order. Pin these lexicographically sorted arrays exactly:

- eval: `p166:0`, `p166:12`, `p166:19`, `p166:8`, `p192:0`, `p192:10`, `p192:13`, `p192:18`, `p192:6`, `p192:7`, `p192:8`;
- train: `p166:1`, `p166:10`, `p166:11`, `p166:13`, `p166:14`, `p166:15`, `p166:16`, `p166:17`, `p166:18`, `p166:2`, `p166:20`, `p166:21`, `p166:22`, `p166:3`, `p166:4`, `p166:5`, `p166:6`, `p166:7`, `p166:9`, `p192:1`, `p192:11`, `p192:12`, `p192:14`, `p192:15`, `p192:16`, `p192:17`, `p192:19`, `p192:2`, `p192:20`, `p192:3`, `p192:4`, `p192:5`, `p192:9`.

The split has exactly 453 train and 151 evaluation records. Its identity bytes are `json.dumps({"eval_groups": eval_groups, "train_groups": train_groups}, sort_keys=True, separators=(",", ":")) + "\n"`; they are 455 bytes with SHA-256 `5397fc5e7820c9fdee3eb38c05278a3b680fb5ca8460d0bbe588ffa7ff22815c`. The validator rejects any order/content/count drift. Both checkpoints, all five feature sets, linear/MLP probes, and all candidate scorers use these arrays directly. Conclusions are explicitly transition-group-disjoint but not problem-disjoint and apply only to the fixed `p166`/`p192` slice.

### 3.3 Probe models

For each feature set in Stage 0B, train exactly two fixed probes:

- `linear`: one `torch.nn.Linear(input_dim, 1)`;
- `mlp`: `Linear(input_dim,64) -> ReLU -> Linear(64,1)`.

Fixed training contract: CPU, float32, Adam, 200 epochs, learning rate `0.001`, no weight decay, full train set in canonical manifest order, seed `20260717`, one thread, deterministic algorithms enabled. Continuous features are standardized using train-only mean/std with zero-std dimensions mapped to zero; binary/mask features remain `{0,1}` and are not standardized. Labels never enter standardization.

Label-permutation control: one deterministic within-train-set permutation generated by `torch.Generator().manual_seed(20260717)`; fit the same MLP for each feature set. It is diagnostic only. A control eval AUROC above `0.70` invalidates the corresponding probe evidence as likely leakage/memorization.

### 3.4 Binary metrics and calibration

Use exact tie-aware AUROC and average precision semantics already implemented in `diagnose_action_supervised_probes.py`. The decision threshold (not probability calibration) is selected on train logits from candidate thresholds `{+inf} U unique(train_logits) U {-inf}` to maximize train F1; ties choose the numerically largest threshold. Freeze it before evaluation. Report threshold, prevalence, accuracy, precision, recall, threshold-selected F1, AUROC, AP, and confusion counts. Separately report sigmoid-logit NLL, Brier score, and an equal-width 10-bin reliability table (`[0,.1),...,[.9,1]`, with count/mean probability/positive rate and null means for empty bins). No temperature or other calibration fit is performed, and these probability metrics are descriptive—not evidence of improved ranking.

For category margin `C`, each inapplicable eval record in `C` receives:

`margin = score(trace record in same group) - score(candidate)`.

Report count/min/median/mean/max. Missing categories produce `count=0` and all scalar values `null`; they cannot pass a positive-margin gate. Per-schema metrics include counts and return AUROC/AP `null` when both labels are not present.

### 3.5 Repeatability

Each official diagnostic command runs twice into literal `run1` and `run2` paths. The assessor requires an exact top-level schema per diagnostic and compares canonical JSON after deleting only these six JSON pointers:

- `/checkpoint`
- `/output`
- `/device`
- `/runtime_seconds`
- `/environment/torch_version`
- `/environment/platform`

No other nested field is excluded. Unknown top-level and nested keys are rejected under §8 before projection; duplicate keys and non-finite values are rejected.

## 4. Stage 0A — Manifest/state adapter and schema-residual geometry

### 4.1 Scope and files

**Create:**

- `script/action_phase0_common.py`
- `script/diagnose_action_schema_residuals.py`
- `acs-jepa-cli/tests/test_action_phase0_common.py`
- `acs-jepa-cli/tests/test_action_schema_residuals.py`

**Modify:**

- `script/action_latent_statistics.py` only for pure residual/statistical helpers reused by the CLI.

No trainer/model/planner production file changes.

### 4.2 Exact latent population and centroids

For each of the 44 recorded source states, enumerate every grounded action sharing the trace action schema using `ActionDecodingSpace.enumerate_ground_actions()` and the same deterministic ordering as Phase 2G. This must reproduce exactly 174,780 `(group, action)` rows. Compute a checkpoint-independent canonical identity stream of records with exact keys `group`, `problem`, `step`, and `action`; both checkpoints and both repeats must report the same count, byte count, and SHA-256 before latent metrics are compared. Separately validate the 604-record manifest and use its labels only when an explicitly invalid comparator is required.

Recover each recorded source state directly from the strict corpus trajectory and verify problem/step/trace identity. Build action tensors with `ActionDecodingSpace.action_tensors_for_ground_actions`, encode each source state once, and encode candidates in chunks in canonical action order.

Report two residual populations over all 174,780 rows:

1. `global_schema_residual`: subtract the centroid over all candidates sharing action schema across all states, as required by the updated spec.
2. `state_schema_residual`: subtract the centroid within each `(group, action schema)` bucket, to remove state-context confounding and isolate within-state argument identity.

Every bucket has at least two candidates; any singleton/missing bucket invalidates official evidence. Let raw full-population action latents be `z_i in R^64`, `N=174780`, grand mean `mu`, schema mean `mu_s`, and state/schema mean `mu_gs`. Define exactly:

- `total_variance = (1/N) * sum_i ||z_i - mu||_2^2`;
- `between_schema_variance = sum_s (n_s/N) * ||mu_s - mu||_2^2`;
- `within_schema_variance = (1/N) * sum_i ||z_i - mu_schema(i)||_2^2`;
- `within_schema_fraction = within_schema_variance / total_variance` and `between_schema_fraction = between_schema_variance / total_variance`.

`total_variance <= torch.finfo(float64).eps` invalidates evidence. Compute all four in float64 with population (`1/N`) denominators and require `abs(total - between - within) <= 1e-10 * max(1,total)`. This decomposition is computed once from raw `z_i`; it is never recomputed on centered residuals. Its exact JSON path is `/metrics/raw_variance_decomposition/{total_variance,between_schema_variance,within_schema_variance,between_schema_fraction,within_schema_fraction,reconstruction_absolute_error}`.

Define `global_schema_residual_i = z_i-mu_schema(i)` and `state_schema_residual_i = z_i-mu_group(i),schema(i)`. For any emitted residual matrix `R` with `M` rows, compute float64 population covariance `C=(R^T R)/M`; thus pooled matrices use `M=N=174780`, while each per-schema matrix uses that schema’s own row count `M=n_s`, never global `N`. Because the relevant global-schema or state/schema bucket means are zero, do not subtract another fitted mean. Sort `eigvalsh(C)` descending, clip values in `[-1e-12,0)` to zero, reject values `<-1e-12`, set `p_j=lambda_j/sum(lambda)`, and define `effective_rank=exp(-sum_{p_j>0} p_j*log(p_j))`. A zero eigenvalue sum invalidates evidence. Numerical rank counts eigenvalues `> 1e-6 * max_eigenvalue`. The assessor’s representation rank is exactly `/metrics/state_schema_residual/pooled/effective_rank`; the baseline/Phase 2 fraction is exactly `/metrics/raw_variance_decomposition/within_schema_fraction`.

For both residual populations, report pooled and per-schema raw residual statistics:

- count, dimension, float64 population (`correction=0`) std min/mean/max and full 64-value std array;
- 64 covariance eigenvalues and normalized eigenvalue spectrum;
- effective rank and numerical rank by the exact formulas above;
- zero-norm count.

For each trace, report two non-tautological distance diagnostics, not “similarity margins”:

- `nearest_wrong_same_schema_raw_l2`: minimum L2 from the trace action latent to every different full-population same-schema action in the same group; ties break canonical action key;
- `nearest_invalid_same_schema_unit_residual_l2`: minimum L2 between unit-normalized **state/schema residuals** for the trace and inapplicable same-schema candidates present in the fixed manifest; ties break canonical action key.

The first comparator may have unknown applicability and is labeled as such. The second is guaranteed inapplicable by the fixed offline manifest. Zero residual norm maps to the all-zero vector and is counted; it never emits NaN/Inf. Distances are descriptive/localization evidence and are not used as an automatic positive-margin continuation clause.

### 4.3 RED/GREEN tests

1. RED: manifest validator rejects changed bytes/hash, duplicate JSON keys/records, wrong counts, noncanonical encoding, bad group/step/trace mapping, unknown keys, and label/category count drift.
2. GREEN: fixed manifest validates to the exact identity above without simulator imports/calls.
3. RED: state/schema centering leaves nonzero bucket means or mixes source-state buckets in a synthetic fixture.
4. GREEN: each bucket mean is zero within `atol=1e-7`; global-schema and state/schema residuals differ on a multi-state fixture.
5. Test exact 174,780 population identity/count reconciliation, zero-norm normalization, repeated action keys, canonical tie-breaks, effective-rank oracle values, and no cross-state nearest-neighbor comparison.
6. Import/CLI parser test pins every default and rejects nonpositive chunk sizes.

### 4.4 Official commands and outputs

Run each command twice (`run1`, then identical command with `run2`):

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_schema_residuals.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/baseline/run1 \
  --device cuda --split val --chunk-size 2048 --seed 20260717
```

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_schema_residuals.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/phase2/run1 \
  --device cuda --split val --chunk-size 2048 --seed 20260717
```

Each output directory contains `summary.json` and `details.json`. `details.json` has exactly 44 records—one per trace—with the selected full-population nearest wrong action, fixed-manifest nearest invalid action, raw/unit residual distances, candidate counts, and zero-norm flags. It does not serialize 174,780 latent vectors; the full population is pinned by canonical identity count/bytes/hash in `summary.json`.

### 4.5 Stage 0A implementation review gate

Independent review must PASS: strict source identity, no oracle/simulator, within-state grouping, centroid formulas, zero-norm behavior, count reconciliation, repeatability, focused/full tests, Ruff, compileall, and `git diff --check`. Only then may Stage 0B plan implementation begin.

## 5. Stage 0B — Applicability recoverability ladder

### 5.1 Scope and files

**Create:**

- `script/diagnose_action_applicability_recoverability.py`
- `acs-jepa-cli/tests/test_action_applicability_recoverability.py`

**Reuse:** `script/action_phase0_common.py` and metric/probe helpers from `script/diagnose_action_supervised_probes.py`; move helpers to the common module only if tests prove no Phase 2G output change.

### 5.2 Exact feature sets

All features are derived from the same source state and grounded candidate; dimensions and ordered feature names are emitted and hashed.

- `A_action` (64): frozen `action_latent` only.
- `B_graph_action` (128): `[graph_latent[64], action_latent[64]]`.
- `C_selected_graph_action` (388): `[graph_latent[64], action_latent[64], flatten(role-ordered selected_object_latents[4,64]), argument_presence_mask[4]]`; inactive roles are exact zeros and mutation-tested.
- `D_raw_symbolic` is exactly 217 binary dimensions in this order:
  1. schema one-hot (7): `build_diagonal_oneway`, `build_straight_oneway`, `car_arrived`, `car_start`, `destroy_road`, `move_car_in_road`, `move_car_out_road`;
  2. active-role mask (4), role indices `0,1,2,3`;
  3. strict upper-triangle role equality (6), lexicographic pairs `(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)`; inactive pairs are zero;
  4. role type one-hot (16), role-major then type order `car,garage,junction,road`; inactive roles are all-zero;
  5. bound-fact indicators (184): predicates in exact order `arrived/2`, `at_car_jun/2`, `at_car_road/2`, `at_garage/2`, `clear/1`, `diagonal/2`, `in_place/1`, `road_connect/3`, `same_line/2`, `starting/2`; within each predicate enumerate role tuples with `itertools.product(range(4), repeat=arity)` in lexicographic order. A feature is one iff every referenced role is active, each bound object type exactly matches the corresponding `PredicateSchema.arg_types`, and the resulting `GroundAtom(predicate, bound_arguments)` belongs to the recorded positive source-state atom set; otherwise zero. Repeated role indices and zero-arity predicates follow that same product rule (CityCar has no zero-arity predicate). Static facts are not synthesized or queried: the strict recorded source-state atoms are authoritative and tests verify that their static predicates match the `include_static=True` state-graph input. Problem-local object IDs/names, precondition trees, all-preconditions-satisfied bits, labels, and simulator outputs are never features.
- `E_hybrid` (605): exact concatenation `[C_selected_graph_action[388], D_raw_symbolic[217]]`.

The ordered feature-name list is fixed by these rules, emitted before fitting, and SHA-256 identified. No parser/schema extension is needed: the adapter uses existing `ParsedProblem.actions`, `types`, `predicates`, `objects`, and recorded `GroundAtom` states. Any live problem whose sorted vocabularies differ from the exact CityCar contract above invalidates official evidence.

### 5.3 RED/GREEN tests

1. RED/GREEN each feature set’s exact tensor composition, role ordering, dimensions, and names.
2. Mutation tests prove padded roles, labels, categories, future states, and split metadata cannot change features.
3. Prefix/source-state mutation changes only expected symbolic/state features; future-state mutation changes none.
4. Raw bound-fact indicators match hand-derived positive, absent, static, repeated-role, inactive-role, and type-incompatible CityCar fixtures and never import/call a simulator.
5. Train-only standardization and decision-threshold selection are frozen before eval; eval-label mutation cannot alter either.
6. Exact AUROC/AP tie groups, all-positive/all-negative per-schema nulls, threshold-selected-F1 tie-break, NLL/Brier/reliability bins, category margins, and count reconciliation.
7. Two invocations produce identical decision projections; control-label probe uses only the deterministic permuted train labels.

### 5.4 Official commands and outputs

Run each twice with `run1`/`run2`:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_applicability_recoverability.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1 \
  --device cpu --split val --epochs 200 \
  --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_applicability_recoverability.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1 \
  --device cpu --split val --epochs 200 \
  --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```

Outputs: `summary.json`, `details.json`, `feature_schema.json`, `split_manifest.json`, and `probe_states.json`.

- `details.json` contains all 604 rows (train and eval), each model’s original-label logit, control-label logit, split, and immutable metadata; Stage 0D therefore has accepted train/eval component scores without refitting.
- `probe_states.json` canonically serializes all ten original-label probes as exact architecture metadata, train-only preprocessing tensors, plus sorted state-dict tensor names, shapes, dtypes, and row-major finite float values. Stage 0D strictly reconstructs preprocessing and these CPU modules, verifies that recomputed logits exactly match `details.json` within `atol=1e-7, rtol=0`, and never trains them again.
- `feature_schema.json` contains exact ordered feature names/dimensions and standardization indices; `split_manifest.json` contains the pinned arrays/identity from §3.2.

All four non-summary artifacts contain no run path, runtime, platform, or version metadata and must be byte-identical between `run1` and `run2`.

### 5.5 Recoverability gate

A feature/model pair is `separable` only if all hold on fixed eval records:

- AUROC `>= 0.80`;
- AP `>= 0.35`;
- role-swap median margin `> 0`;
- one-argument-substitution median margin `> 0`;
- corresponding permuted-label control AUROC `<= 0.70`.

Threshold-selected F1 and per-schema metrics are reported but not added continuation thresholds because the governing spec does not preregister them.

Classify evidence without choosing a branch yet using the fixed MLP (linear results remain separately reported to diagnose organization):

- `latent_separable`: Phase 2 `C_selected_graph_action/mlp` is separable;
- `raw_separable`: Phase 2 `D_raw_symbolic/mlp` is separable;
- `hybrid_separable`: Phase 2 `E_hybrid/mlp` is separable;
- `label_or_sampling_blocker`: `raw_separable` is false;
- `latent_state_bottleneck`: `raw_separable` is true and `latent_separable` is false.

Additionally report whether any A/B/C linear or MLP probe passes, but do not substitute that post-hoc winner into branch predicates.

Independent review must PASS feature causality, no label leakage, controls, calibration, exact metrics, repeatability, and artifacts before Stage 0C.

## 6. Stage 0C — Hard-negative transition equivalence

### 6.1 Scope and files

**Create:**

- `script/diagnose_action_transition_equivalence.py`
- `acs-jepa-cli/tests/test_action_transition_equivalence.py`

No model changes.

### 6.2 Candidate selection and exact calculations

For each of 44 groups and each checkpoint:

1. select the unique trace action;
2. among inapplicable same-schema candidates from categories `one_arg_substitution`, `role_swap`, and `random_same_schema`, select nearest by unit-normalized action-latent L2; ties break by canonical `(action.name, action.arguments)`;
3. encode recorded source and next states with the restored JEPA encoder;
4. compute `pred_true = predictor(source, z_true)` and `pred_wrong = predictor(source, z_wrong)`;
5. in float64 on CPU compute configured weighted graph/object errors `E_true = d(pred_true, target_next)` and `E_wrong = d(pred_wrong, target_next)`, where `d = graph_weight * mean_graph_squared_error + object_weight * mean_object_squared_error`; also serialize unweighted graph/object components;
6. compute configured weighted prediction separation `S = d(pred_wrong, pred_true)` with exact object-ID/order equality asserted;
7. compute `error_ratio = E_wrong / max(E_true, torch.finfo(torch.float64).eps)`, `error_margin = E_wrong - E_true`, and `separation_ratio = S / max(E_true, torch.finfo(torch.float64).eps)`.

A pair is `transition_equivalent` iff both `error_ratio <= 1.10` and `separation_ratio <= 0.25`. Exact/near-zero true-error rows are counted separately. “Mostly transition-equivalent” is fixed as at least 50% of eligible Phase 2 pairs. Report baseline/Phase 2 total and per-category distributions; Phase 2 determines the branch gate. Groups lacking an inapplicable same-schema manifest candidate are skipped and must reconcile; fewer than 40 eligible groups invalidates evidence rather than changing the denominator.

The observed next state is diagnostic evidence only and is never available to production ranking. No effect/applicability simulator query is used.

### 6.3 RED/GREEN tests

- exact nearest-invalid filtering, category set, tie-break, and exclusion of applicable wrong alternatives;
- source/next ordering and object-ID alignment;
- hand-calculated graph/object weighted error, error/separation ratios, inclusive `1.10`/`0.25` boundaries, and float64 epsilon behavior;
- synthetic equivalent/non-equivalent cases and the inclusive 50% “mostly” boundary;
- no cross-state comparison, no simulator import/call, finite validation, count reconciliation, strict restoration, and repeat projection.

### 6.4 Official commands and outputs

Run each twice with `run1`/`run2`:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_transition_equivalence.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run1 \
  --device cuda --split val --chunk-size 2048 --seed 20260717 \
  --equivalence-error-ratio 1.10 --equivalence-separation-ratio 0.25
```

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_transition_equivalence.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/phase2/run1 \
  --device cuda --split val --chunk-size 2048 --seed 20260717 \
  --equivalence-error-ratio 1.10 --equivalence-separation-ratio 0.25
```

Outputs: `summary.json`, `details.json` with exactly one eligible/skipped record per group.

Independent implementation/evidence review must PASS before Stage 0D.

## 7. Stage 0D — Candidate-ranking baselines and deterministic branch decision

### 7.1 Scope and files

**Create:**

- `script/diagnose_action_candidate_ranking.py`
- `script/assess_action_latent_updated_phase0.py`
- `acs-jepa-cli/tests/test_action_candidate_ranking.py`
- `acs-jepa-cli/tests/test_assess_action_latent_updated_phase0.py`
- after official evidence only: `script/ACTION_LATENT_UPDATED_PHASE0_DECISION.md`

No production planner/scorer is added.

### 7.2 Five fixed candidate scores

Fit the role/object probe on train groups only; consume all applicability probe logits/states from the independently accepted Stage 0B artifacts. Evaluate on the fixed 151 eval records:

1. `latent_transition` (teacher-forced/non-deployable): negative configured weighted one-step prediction error `-E(candidate)` to the recorded observed next latent, calculated exactly as Stage 0C for every manifest candidate. It is kept separate and can never satisfy a “deployable ranker” branch predicate.
2. `latent_applicability`: accepted matching-checkpoint Stage 0B `C_selected_graph_action/mlp` logit, strictly reproduced from `probe_states.json` and checked against `details.json`.
3. `role_object`: fit a `RoleObjectProbe(latent_dim=64, action_dim=64, max_action_arity=4, hidden_dim=64)` separately for each checkpoint. Training uses **all active argument roles from all pinned train-manifest candidates**, flattened in manifest then ascending-role order. For each row: graph/action latents are that candidate’s frozen latents; object bank is source-state object latents sorted by numeric `object_id` then zero-padded to the maximum bank size in the train/eval call; `object_mask` is true for every real sorted problem-local object and false only for padding (no type-filter mask); role ID is the active position; target is the candidate argument object’s index in that sorted bank. Optimize full-batch cross-entropy over this all-real-object denominator with CPU/Adam/200 epochs/lr `0.001`/seed `20260717`/no weight decay. Add a new pure `fit_role_object_probe` helper returning both fitted model and metrics; preserve `train_role_probe` as a compatibility wrapper and test its old metrics unchanged. Canonically serialize the fitted model to ranking `role_probe_state.json`. At scoring, recompute each active-role all-real-object logit row, apply `log_softmax(dim=-1)`, select the candidate argument target index, and define `score_role(candidate)=(1/R_active)*sum_role log_probability`; zero-active-role candidates are invalid. No type-filtered `argument_candidate_mask` is used.
4. `raw_symbolic`: accepted matching-checkpoint Stage 0B `D_raw_symbolic/mlp` logit.
5. `hybrid`: accepted matching-checkpoint Stage 0B `E_hybrid/mlp` logit. This is one directly fitted feature-level scorer, not a post-hoc ensemble or a sixth ablation model.

No scorer uses candidate category, group/problem ID, oracle generation, or eval labels. `latent_transition` alone uses the recorded future and is explicitly diagnostic/non-deployable; the other four use source-state/candidate information only. Stage 0D receives `summary.json`, `details.json`, `feature_schema.json`, `split_manifest.json`, and `probe_states.json` from the matching checkpoint’s accepted Stage 0B `run1`, records canonical identities for all five under its summary `settings.recoverability_inputs`, and fails unless their hashes, logits, and schemas reconcile. For every one of the 151 eval rows it copies `latent_applicability`, `raw_symbolic`, and `hybrid` from the exact matching Stage 0B `C_selected_graph_action/mlp`, `D_raw_symbolic/mlp`, and `E_hybrid/mlp` JSON numbers without float reformatting.

### 7.3 Ranking metrics

For every score, report fixed eval binary metrics and category margins plus:

- `top1_applicable_rate`: fraction of eval groups whose highest-scoring candidate is applicable; score ties break canonical action key;
- `mrr_first_applicable`: reciprocal rank of the first applicable candidate per group;
- `pairwise_applicable_accuracy`: fraction of within-group applicable/inapplicable pairs with positive score margin, ties count `0.5`;
- trace-action rank/MRR separately, never used as the applicability acceptance result;
- complete group/candidate/pair counts and per-schema breakdown.

A score `ranks_applicable` iff AUROC `>=0.80`, AP `>=0.35`, role-swap and one-substitution median margins are positive, `top1_applicable_rate >=0.80`, and `pairwise_applicable_accuracy >=0.80`.

### 7.4 Branch/continuation precedence

The assessor first computes these exact Phase 2 booleans from named artifacts:

- `representation_ok`: Phase 2 `/metrics/state_schema_residual/pooled/effective_rank >=4.0`, Phase 2 `/metrics/raw_variance_decomposition/within_schema_fraction >=0.001`, and each Phase 2 value is greater than or equal to the same JSON path in the baseline summary;
- `latent_separable`: Stage 0B Phase 2 `/metrics/features/C_selected_graph_action/mlp` has eval AUROC `>=0.80`, eval average precision `>=0.35`, both named median margins `>0`, **and** `/metrics/features/C_selected_graph_action/control_mlp/eval/auroc <=0.70`;
- `hybrid_separable`: the identical four-gate predicate at `/metrics/features/E_hybrid/{mlp,control_mlp}`;
- `raw_separable`: the identical four-gate predicate at `/metrics/features/D_raw_symbolic/{mlp,control_mlp}`;
- `latent_rank`: `latent_applicability.ranks_applicable OR role_object.ranks_applicable`;
- `raw_rank`: `raw_symbolic.ranks_applicable`;
- `hybrid_rank`: `hybrid.ranks_applicable`;
- `mostly_transition_equivalent`: Stage 0C Phase 2 equivalence rate `>=0.50`;
- `transition_distinguishable`: equivalence rate `<0.50` and median `error_margin >0`.

`latent_transition` is non-deployable and is excluded from all branch predicates. The assessor emits exactly one action in this precedence order:

1. `FIX_DATA_LABEL_CONSTRUCTION` iff `raw_separable` is false. The fixed raw-fact sanity classifier failed, so no representation conclusion is accepted.
2. `BRANCH_D_ABSTRACT_ACTIONS` iff `mostly_transition_equivalent` is true.
3. `CONTINUE_PHASE1_MINIMAL_SCHEMA_RANK` iff `representation_ok AND latent_separable AND hybrid_separable AND latent_rank AND transition_distinguishable`.
4. `BRANCH_B_DISCRETE_CANDIDATE_PLANNING` iff `latent_rank AND transition_distinguishable` but clause 3 did not match because `representation_ok` or `latent_separable` is false. This is the exact “bounded candidate ranking works while the continuous representation is not recoverable” predicate; it does not rely on a new continuous-planning run.
5. `BRANCH_A_EXPLICIT_STATE_ACTION_SCORER` iff `NOT latent_rank AND raw_separable AND hybrid_separable AND hybrid_rank`. This is exactly raw/hybrid recovery with weak deployable latent-only ranking.
6. `BRANCH_C_STATE_ENCODER_REDESIGN` iff `raw_separable AND raw_rank AND NOT hybrid_rank`. Raw recorded facts rank correctly but the hybrid graph/object/action representation does not.
7. `BRANCH_D_ABSTRACT_ACTIONS` otherwise.

These named booleans remove any post-hoc ablation or “off-manifold reasons” interpretation and select exactly one path. `CONTINUE_PHASE1...` authorizes only creation/review of a Phase 1 plan, not implementation. Any branch result requires a branch-selection note and forbids Phase 1 implementation.

### 7.5 RED/GREEN tests

- exact five score constructions and no forbidden features;
- role/object complete-action scoring, masks, sorted object IDs, target-only roles, and inactive-padding mutations;
- accepted Stage 0B probe-state reconstruction, exact train/eval logit reproduction, and no probe refit;
- ranking tie-breaks, pairwise ties, groups with one/no applicable candidate, MRR/top1, margins, and count reconciliation;
- each branch-precedence clause and every exact threshold boundary;
- malformed/unknown/nonfinite JSON; duplicate keys; changed absolute paths/hashes; missing outputs; repeat-projection mutation tests;
- evidence manifest includes every `summary.json`, `details.json`, feature/split schema, the candidate manifest, checkpoints, corpus manifest, governing spec, and generated decision outputs (except self-hash recursion).

### 7.6 Official candidate-ranking commands

Run each checkpoint twice. Baseline run1:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_candidate_ranking.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --recoverability-summary /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/summary.json \
  --recoverability-details /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/details.json \
  --recoverability-feature-schema /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/feature_schema.json \
  --recoverability-split-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/split_manifest.json \
  --recoverability-probe-states /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/probe_states.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run1 \
  --device cpu --split val --epochs 200 --learning-rate 0.001 \
  --hidden-dim 64 --seed 20260717
```

Phase 2 run1:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_candidate_ranking.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --recoverability-summary /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/summary.json \
  --recoverability-details /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/details.json \
  --recoverability-feature-schema /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/feature_schema.json \
  --recoverability-split-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/split_manifest.json \
  --recoverability-probe-states /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/probe_states.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run1 \
  --device cpu --split val --epochs 200 --learning-rate 0.001 \
  --hidden-dim 64 --seed 20260717
```

For each checkpoint, repeat its run1 command by changing only the candidate-ranking `--output .../run1` component to `.../run2`; both ranking repeats deliberately consume the same accepted matching-checkpoint Stage 0B `run1` artifacts so the summary projection and imported scores are identical.

Outputs: `summary.json`, `details.json`, `split_manifest.json`, `role_probe_state.json`. The baseline/Phase 2 summaries share the same schema and scorer set; assessment reports their complete fixed metric maps side by side without a separately serialized delta map, while branch predicates use Phase 2 only.

### 7.7 Assessor command

The assessor CLI must expose and strictly require the literal fixed paths below; parser order is irrelevant, values are not:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/assess_action_latent_updated_phase0.py \
  --updated-spec /opt/data/workspace/acs-jepa/script/ACTION_LATENT_UPDATED_SPEC.md \
  --baseline-checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --baseline-config /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml \
  --phase2-checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
  --phase2-config /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml \
  --corpus-manifest /opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json \
  --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json \
  --baseline-schema-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/baseline/run1/summary.json \
  --baseline-schema-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/baseline/run2/summary.json \
  --phase2-schema-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/phase2/run1/summary.json \
  --phase2-schema-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/phase2/run2/summary.json \
  --baseline-recoverability-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/summary.json \
  --baseline-recoverability-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run2/summary.json \
  --phase2-recoverability-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/summary.json \
  --phase2-recoverability-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run2/summary.json \
  --baseline-transition-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run1/summary.json \
  --baseline-transition-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run2/summary.json \
  --phase2-transition-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/phase2/run1/summary.json \
  --phase2-transition-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/phase2/run2/summary.json \
  --baseline-ranking-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run1/summary.json \
  --baseline-ranking-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run2/summary.json \
  --phase2-ranking-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run1/summary.json \
  --phase2-ranking-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run2/summary.json \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/assessment
```

The assessor discovers each ranking summary’s five bound Stage 0B files from `settings.recoverability_inputs`, requires those paths to be the matching fixed checkpoint’s accepted Stage 0B run1 siblings already supplied to assessment, recomputes all file identities, validates their complete schemas, reconstructs all ten probes plus preprocessing, and reproduces all 604 Stage 0B logits at `atol=1e-7,rtol=0`. It then requires exact JSON-number equality for each of the 151 imported `latent_applicability`, `raw_symbolic`, and `hybrid` ranking scores against the matching Stage 0B details rows. Any checkpoint or accepted-run1 path mismatch, identity mismatch, failed reconstruction, or score mismatch is fatal.

Exit semantics:

- `0`: structurally valid evidence and one emitted research action, regardless of whether that action continues or pivots;
- nonzero: malformed, incomplete, nonfinite, identity-mismatched, non-repeatable, or count-inconsistent evidence.

Assessment outputs: `summary.json`, `summary.md`, `evidence_manifest.json`. The checked-in decision record reproduces compact metrics, every stage verdict, selected action, exact commands, artifact hashes, and why later branches were not selected.

## 8. Exact artifact schemas and repeat contract

Every JSON file is canonical UTF-8 JSON with sorted object keys, separators `(",", ":")`, no NaN/Infinity, and one trailing newline. Duplicate keys and unknown keys at every schema-defined object level are rejected. Unless explicitly nullable below, fields are required and non-null. A `file_identity` has exactly `path:str,bytes:int,sha256:64-lowercase-hex`; a `manifest_identity` adds required `count:int`.

### 8.1 Diagnostic summaries

Every diagnostic `summary.json` has exactly these 16 top-level keys:

`schema_version:str`, `kind:str`, `dataset:str`, `checkpoint:str`, `checkpoint_sha256:str`, `split:str`, `seed:int`, `candidate_manifest:manifest_identity`, `settings:object`, `checkpoint_restoration:object`, `counts:object`, `metrics:object`, `environment:object`, `device:str`, `output:str`, `runtime_seconds:number`.

`kind`/`schema_version` pairs are exactly `schema_residual/action_latent_updated_phase0.schema_residual.v1`, `applicability_recoverability/action_latent_updated_phase0.applicability_recoverability.v1`, `transition_equivalence/action_latent_updated_phase0.transition_equivalence.v1`, and `candidate_ranking/action_latent_updated_phase0.candidate_ranking.v1`; `split` is `val`. `checkpoint_restoration` has exactly the five Phase 2G module keys and each value exactly `state_key:str,status:"restored"|"disabled"`; accepted baseline/Phase 2 maps remain the strict Phase 2G maps. `environment` has exactly `python_version,torch_version,platform,byteorder,num_threads,num_interop_threads,deterministic_algorithms,python_hash_seed,cublas_workspace_config`, with only the last two nullable strings.

Kind-specific nested schemas are:

- `schema_residual`: settings exactly `chunk_size,expected_population_count,residual_centers,numerical_rank_relative_tolerance,zero_norm_policy`; counts exactly `groups,full_population,candidate_manifest_records,schemas,nearest_wrong_rows,nearest_invalid_rows`; metrics exactly `full_population_identity,raw_variance_decomposition,global_schema_residual,state_schema_residual,nearest_wrong_same_schema_raw_l2,nearest_invalid_same_schema_unit_residual_l2`. `full_population_identity` has exactly `count,bytes,sha256` (no path because it identifies a canonical stream, not a saved file). `raw_variance_decomposition` has exactly the six keys/formulas in §4.2. Each residual object has exactly `pooled,per_schema`; `per_schema` has exactly the seven schema names from §5.2; each statistics object has exactly `count,dimension,std_min,std_mean,std_max,std_values,covariance_eigenvalues,normalized_eigenvalue_spectrum,effective_rank,numerical_rank,zero_norm_count`. Distribution objects have exactly `count,min,median,mean,max` with scalars nullable iff count is zero.
- `applicability_recoverability`: settings exactly `epochs,learning_rate,hidden_dim,models,feature_sets,threshold_policy,control_policy,reliability_bins`; counts exactly `records,train_records,eval_records,train_groups,eval_groups,applicable,inapplicable`; metrics exactly `features,verdicts`. `features` has exactly the five feature-set keys; each has exactly `linear,mlp,control_mlp`; each probe object has exactly `train,eval,role_swap_margin,one_arg_substitution_margin,per_schema,threshold`. A binary metric object has exactly `count,positive_count,negative_count,prevalence,accuracy,precision,recall,f1,auroc,average_precision,nll,brier,true_positive,false_positive,true_negative,false_negative,reliability_bins`; AUROC/AP are nullable only for single-class slices, and each of ten reliability-bin records has exactly `lower,upper,upper_inclusive,count,mean_probability,positive_rate`, with last two nullable only when count is zero. `verdicts` has exactly `latent_separable,raw_separable,hybrid_separable,label_or_sampling_blocker,latent_state_bottleneck,any_abc_separable` booleans.
- `transition_equivalence`: settings exactly `chunk_size,error_ratio_threshold,separation_ratio_threshold,mostly_rate_threshold,float64_epsilon`; counts exactly `groups,eligible,skipped,exact_or_near_zero_true_error`; metrics exactly `error_ratio,error_margin,separation_ratio,equivalence_rate,mostly_transition_equivalent,per_category`; each distribution uses the schema above and each category object has exactly `count,error_ratio,error_margin,separation_ratio,equivalence_rate`.
- `candidate_ranking`: settings exactly `epochs,learning_rate,hidden_dim,scorers,ranking_gate,recoverability_inputs`; `recoverability_inputs` has exactly `summary,details,feature_schema,split_manifest,probe_states`, each a `file_identity` of the matching checkpoint/repeat Stage 0B file; counts exactly `eval_records,eval_groups,applicable,inapplicable,within_group_pairs,groups_without_applicable,groups_without_inapplicable`; metrics has exactly the five scorer keys. Each scorer object has exactly `binary,role_swap_margin,one_arg_substitution_margin,top1_applicable_rate,mrr_first_applicable,pairwise_applicable_accuracy,trace_mrr,per_schema,ranks_applicable,deployable`; `binary`, distribution, and nullable per-schema rules are those above.

Settings values/types are fixed, not merely key names: schema settings are `chunk_size:2048`, `expected_population_count:174780`, `residual_centers:["global_schema","state_schema"]`, `numerical_rank_relative_tolerance:1e-6`, `zero_norm_policy:"zero_vector"`; recoverability settings are `epochs:200`, `learning_rate:0.001`, `hidden_dim:64`, `models:["linear","mlp","control_mlp"]`, `feature_sets:["A_action","B_graph_action","C_selected_graph_action","D_raw_symbolic","E_hybrid"]`, `threshold_policy:"max_train_f1_highest_threshold"`, `control_policy:"train_label_permutation_seed_20260717"`, and reliability edges `[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]`; transition settings are `chunk_size:2048`, `error_ratio_threshold:1.10`, `separation_ratio_threshold:0.25`, `mostly_rate_threshold:0.50`, `float64_epsilon:2.220446049250313e-16`; ranking settings are `epochs:200`, `learning_rate:0.001`, `hidden_dim:64`, `scorers:["latent_transition","latent_applicability","role_object","raw_symbolic","hybrid"]`, and `ranking_gate` exactly `auroc:0.80,average_precision:0.35,top1_applicable_rate:0.80,pairwise_applicable_accuracy:0.80,role_swap_margin_strictly_positive:true,one_arg_margin_strictly_positive:true`.

Every `per_schema` dynamic map has exactly the seven literal schema keys in §5.2; a ranking group belongs to its unique trace action schema. Recoverability values use the full binary-metric schema. Ranking values have exactly `count,applicable,inapplicable,auroc,average_precision,top1_applicable_rate,mrr_first_applicable,pairwise_applicable_accuracy`; AUROC/AP are null iff a schema is single-class, group metrics are null iff no eligible group/pair exists. Transition `per_category` has exactly `one_arg_substitution,role_swap,random_same_schema`; zero-count category distributions use the declared null rule. Apart from those rules and skipped transition rows, every numeric metric is finite and non-null; every count is a nonnegative integer, every rate is in `[0,1]`, and every verdict/ranking/deployability field is Boolean.

### 8.2 Details and supporting JSON

- Root `root_identity.json` has exactly `schema_version,updated_spec,baseline_checkpoint,baseline_config,phase2_checkpoint,phase2_config,corpus_manifest,candidate_manifest,split_sha256,created_by`; `schema_version` is exactly `action_latent_updated_phase0.root_identity.v1`; the five model/config/spec identities are `file_identity`, two manifests are `manifest_identity`, `split_sha256` is the §3.2 literal, and `created_by` is exactly `schema_residual/baseline/run1`. Its canonical bytes are deterministic and every command/assessor validates them.

- Schema-residual `details.json` is a list of exactly 44 objects with exactly `group,problem,step,trace_action,full_candidate_count,nearest_wrong_action,nearest_wrong_raw_l2,invalid_manifest_candidate_count,nearest_invalid_action,nearest_invalid_unit_residual_l2,trace_zero_residual_norm,nearest_invalid_zero_residual_norm`.
- Recoverability `details.json` is a list of exactly 604 objects in manifest order with exactly `manifest_index,group,problem,step,action,category,label,split,logits,control_logits`; `logits` has exactly the ten keys `A_action/linear`, `A_action/mlp`, `B_graph_action/linear`, `B_graph_action/mlp`, `C_selected_graph_action/linear`, `C_selected_graph_action/mlp`, `D_raw_symbolic/linear`, `D_raw_symbolic/mlp`, `E_hybrid/linear`, `E_hybrid/mlp`; `control_logits` has exactly the five corresponding `/mlp` keys.
- `feature_schema.json` has exactly `schema_version,candidate_manifest_sha256,feature_sets`; `schema_version` is exactly `action_latent_updated_phase0.feature_schema.v1`; `feature_sets` is a five-item list in A–E order, each exactly `name,dimension,feature_names,binary_indices,standardized_indices`, with dimensions derived in §5.2 and indices sorted/disjoint/reconciling all dimensions.
- `split_manifest.json` has exactly `eval_groups,train_groups`; its bytes/hash are exactly §3.2.
- `probe_states.json` has exactly `schema_version,candidate_manifest_sha256,split_manifest_sha256,training,models`; `schema_version` is exactly `action_latent_updated_phase0.probe_states.v1`; `training` exactly `seed,epochs,learning_rate,hidden_dim,optimizer,dtype`, where optimizer is exactly `Adam(lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)` and dtype is exactly `torch.float32`; `models` is ten objects in A–E then linear/MLP order, each exactly `feature_set,model_kind,input_dim,architecture,preprocessing,state_dict`; for `model_kind:"linear"`, `architecture` has exactly `name:"linear",input_dim:<matching int>,output_dim:1,bias:true`; for `model_kind:"mlp"`, it has exactly `name:"mlp",input_dim:<matching int>,hidden_dim:64,output_dim:1,activation:"relu",bias:true`; `preprocessing` has exactly `mean,std,binary_indices,standardized_indices,zero_std_indices`, where mean/std are full input-dimension finite arrays fitted on train only, binary dimensions use mean `0`/std `1`, and zero-std continuous dimensions are listed and map to zero; `state_dict` is a sorted list of records exactly `name,shape,dtype,values`, and every tensor dtype is exactly `torch.float32`.
- Transition-equivalence `details.json` is a list of exactly 44 objects with exactly `group,problem,step,trace_action,status,skip_reason,wrong_action,wrong_category,wrong_unit_action_l2,true_graph_error,true_object_error,true_total_error,wrong_graph_error,wrong_object_error,wrong_total_error,prediction_separation,error_ratio,error_margin,separation_ratio,transition_equivalent`; `skip_reason` is nullable only when status is `eligible`, and all metric/action fields are nullable only when status is `skipped`.
- Candidate-ranking `details.json` is a list of exactly 151 eval-manifest objects with exactly `manifest_index,group,problem,step,action,category,label,scores,ranks`; `scores` and `ranks` each have exactly the five scorer names. Candidate-ranking `split_manifest.json` is byte-identical to §3.2. `role_probe_state.json` has exactly `schema_version,candidate_manifest_sha256,split_manifest_sha256,training,model`; `schema_version` is exactly `action_latent_updated_phase0.role_probe_state.v1`; `training` has exactly `seed,epochs,learning_rate,hidden_dim,optimizer,dtype,mask_policy,row_order`, with optimizer exactly `Adam(lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)`, dtype exactly `torch.float32`, mask policy exactly `all_real_sorted_object_ids_padding_only`, and row order exactly `canonical_manifest_then_ascending_active_role`; `model` has exactly `architecture,state_dict`; `architecture` has exactly `name:"RoleObjectProbe",latent_dim:64,action_dim:64,max_action_arity:4,hidden_dim:64,role_embedding,query`; `role_embedding` has exactly `num_embeddings:4,embedding_dim:64`; `query` is exactly the ordered list `[{kind:"linear",in_features:192,out_features:64,bias:true},{kind:"gelu",approximate:"none"},{kind:"linear",in_features:64,out_features:64,bias:true}]`; and every sorted `state_dict` record uses the canonical `name,shape,dtype,values` tensor schema with dtype exactly `torch.float32`.

Action objects always have exactly `name:str,arguments:list[str]`. Lists retain canonical manifest/group order; they are never resorted after score computation.

### 8.3 Assessment and evidence schemas

Assessment `summary.json` has exactly `schema_version,kind,input_identities,repeatability,stage_verdicts,decision_booleans,selected_action,precedence_trace,compact_metrics,output`; `schema_version` is exactly `action_latent_updated_phase0.assessment.v1`; `kind` is `updated_phase0_assessment`; `input_identities` has exactly `updated_spec,baseline_checkpoint,baseline_config,phase2_checkpoint,phase2_config,corpus_manifest,candidate_manifest,baseline_schema_run1,baseline_schema_run2,phase2_schema_run1,phase2_schema_run2,baseline_recoverability_run1,baseline_recoverability_run2,phase2_recoverability_run1,phase2_recoverability_run2,baseline_transition_run1,baseline_transition_run2,phase2_transition_run1,phase2_transition_run2,baseline_ranking_run1,baseline_ranking_run2,phase2_ranking_run1,phase2_ranking_run2`, each a `file_identity` except the two manifest entries, which are `manifest_identity`. `repeatability` has exactly `baseline_schema,phase2_schema,baseline_recoverability,phase2_recoverability,baseline_transition,phase2_transition,baseline_ranking,phase2_ranking`; each value has exactly `summary_projection_equal:bool,sibling_files_equal:bool,files_checked:list[str]`. `decision_booleans` has exactly the nine booleans in §7.4. `stage_verdicts` has exactly `evidence,residual,recoverability,transition_equivalence,ranking`, each literal `"PASS"|"FAIL"`: evidence PASS iff every identity/restoration/schema/count/repeat check passes; residual PASS iff `representation_ok`; recoverability PASS iff `raw_separable AND (latent_separable OR hybrid_separable)`; transition-equivalence PASS iff `transition_distinguishable`; ranking PASS iff at least one of the four deployable scorers has `ranks_applicable=true`. `precedence_trace` is one to seven records exactly `clause:int,action:str,predicate:str,matched:bool`, ending at the first true match; predicate is respectively one of these seven exact strings: `not raw_separable`, `mostly_transition_equivalent`, `representation_ok and latent_separable and hybrid_separable and latent_rank and transition_distinguishable`, `latent_rank and transition_distinguishable and (not representation_ok or not latent_separable)`, `not latent_rank and raw_separable and hybrid_separable and hybrid_rank`, `raw_separable and raw_rank and not hybrid_rank`, `otherwise`. `selected_action` is one of the six distinct literal actions in §7.4 and must equal the first matched trace action. `output` is the fixed absolute assessment directory string.

`compact_metrics` has exactly `residual_effective_rank_baseline,residual_effective_rank_phase2,within_schema_fraction_baseline,within_schema_fraction_phase2,latent_auroc,latent_ap,latent_role_swap_margin,latent_one_arg_margin,raw_auroc,raw_ap,hybrid_auroc,hybrid_ap,transition_equivalence_rate,transition_error_margin_median,ranking_baseline,ranking_phase2`; each ranking map has exactly the five scorer names, and each scorer has exactly `auroc,average_precision,top1_applicable_rate,pairwise_applicable_accuracy,ranks_applicable`. These values are copied from exact stage JSON paths, not recomputed under different formulas. Unknown keys are rejected.

`evidence_manifest.json` has exactly `schema_version,entries`; `schema_version` is exactly `action_latent_updated_phase0.evidence_manifest.v1`; entries are sorted by absolute path and each has exactly `path,bytes,sha256,role`, where role is one of `root_identity,governing_spec,corpus_manifest,candidate_manifest,checkpoint,config,diagnostic_summary,diagnostic_details,feature_schema,split_manifest,probe_states,role_probe_state,assessment_summary,assessment_markdown`. It includes `root_identity.json`, governing spec, corpus manifest, candidate manifest, both checkpoints and configs, every diagnostic summary/details/supporting JSON from both repeats, and assessment summary/Markdown; it excludes itself to avoid recursive hashing.

### 8.4 Repeat comparison

For each diagnostic pair, only `summary.json` undergoes the six-pointer projection in §3.5. Required byte-identical sibling inventories are: schema residual `{details.json}`; recoverability `{details.json,feature_schema.json,split_manifest.json,probe_states.json}`; transition equivalence `{details.json}`; candidate ranking `{details.json,split_manifest.json,role_probe_state.json}`. Any difference fails repeatability. The assessor discovers siblings from the fixed summary parent, requires exactly that inventory plus `summary.json`, validates recursive schemas before hashing, and rejects extra JSON files. `root_identity.json` is validated once at the root. Assessment artifacts are produced once and are not projected against themselves.

## 9. Verification and review sequence

For each stage:

1. record the stage-specific implementation plan status;
2. obtain independent plan review `PASS`; correct/re-review all blockers;
3. add tests and record actual RED failures before production code;
4. implement only the reviewed stage;
5. run focused tests, then:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests -q -k 'not tuning_configs_load_and_keep_required_defaults'
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check <all changed Python files>
python3 -m compileall -q <all changed Python files>
git diff --check
```

6. execute official stage commands only after software checks pass;
7. obtain independent implementation/evidence review `PASS`; fix/re-review blockers;
8. create and verify an SSH-signed stage commit before beginning the next stage.

Final Stage 0D review independently recomputes every evidence hash, branch threshold, repeat projection, and selected branch. Push only when explicitly requested.

## 9. Phase completion outcomes

Updated Phase 0 is complete only when Stage 0D Implementation/Evidence Review passes and the signed decision record exists. Then:

- if action is `CONTINUE_PHASE1_MINIMAL_SCHEMA_RANK`, create a separate Updated Phase 1 implementation plan and review it before code;
- otherwise write the selected branch note (the decision record may serve if complete), mark Updated Phase 1 forbidden, and create a separate plan for exactly that branch or label/data remediation;
- broad tuning remains paused in every outcome until the revised specification’s resumption criterion is independently demonstrated.

## 10. Plan review checklist

The plan reviewer must return `PASS` or `FAIL` and explicitly audit:

- fidelity to all four Updated Phase 0 diagnostics and all non-goals;
- whether source-state/schema centering avoids contextual confounding;
- exact raw symbolic feature semantics and absence of direct label leakage;
- probe selectivity/control, calibration, split isolation, and imbalance metrics;
- transition predictor/target/error formulas and equivalence thresholds;
- candidate-score inference availability and branch precedence/exhaustiveness;
- absolute paths, input hashes, output inventory, count reconciliation, repeat projection, and assessor exit behavior;
- practical implementability against current ACS-JEPA APIs;
- whether any threshold or branch rule silently authorizes Phase 1 contrary to the updated spec.

Implementation is not authorized until this review passes.
