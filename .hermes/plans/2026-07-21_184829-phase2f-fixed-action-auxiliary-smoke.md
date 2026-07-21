# Stage 2F Plan — Fixed action-auxiliary component smoke

Date: 2026-07-21
Status: implementation and evidence Review 2 PASS; ready for verified SSH-signed commit
Governing specification: `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2
Prerequisite: Stage 2E commit `0a59c11`

## Objective

Add and actually execute one fixed, reproducible smoke train/eval configuration that exercises the Stage 2 action-identifiability auxiliaries together without changing decoding, planning, sampling, or the JEPA transition loss. Produce a strict offline applicability artifact before training, run the existing smoke split with seed 0, verify finite auxiliary metrics and checkpoint state, and record component-level evidence. Empirical representation acceptance remains Stage 2G.

## Scope

### 1. Offline applicability artifact producer

Add `script/build_action_applicability_table.py` plus focused tests.

The producer:

- accepts exactly one dataset directory and an absolute output path;
- loads the ordinary corpus through `acs_jepa_cli.data.load_corpus(..., strict=True)`;
- iterates every recorded trajectory and every source transition in deterministic corpus/trajectory/time order;
- zips each `TrajectorySample` with its `TrajectoryRecord`, resolves the PDDL file from `record.problem_name`, preserves the trajectory's sparse/unchanged `problem_index`, and serializes `corpus.parsed_problems[problem_index].name` rather than the filename alias;
- creates a fresh offline `SimulatorEngine` for each trajectory and, before every oracle query, requires canonical `engine.current_facts()` to equal `trajectory.states[step]`;
- obtains the complete applicable grounded-action set, requires the recorded action to be present in that set, then advances only by replaying that action with `finish=True`;
- after the final replay, requires canonical simulator facts to equal the recorded terminal state;
- rejects source, action-membership, replay, or terminal mismatches with dataset/problem/trajectory/step context;
- serializes the exact Stage 2E UTF-8 JSON schema and semantics token `positive_ground_atoms_closed_world_v1`;
- uses the trajectory's recorded source atoms and exact corpus `problem_index`/problem name;
- canonical-sorts atoms, actions, and final entries;
- deduplicates equal `(problem_index, canonical state)` keys only when the applicable action sets agree, and rejects contradictions;
- writes atomically and refuses a relative output path;
- never becomes part of dataset reads, model construction, training, evaluation, decoding, or planning.

The producer may use `SimulatorEngine.applicable_actions()` only here as the explicitly allowed controlled offline oracle. It must not infer complete labels from trace positives.

Tests use a tiny PDDL corpus and verify deterministic bytes, exact strict-loader round trip, full action sets rather than trace-only positives, duplicate-state agreement, contradiction rejection via a unit seam, source/terminal/action-membership mismatch rejection, absolute output policy, internal PDDL name differing from filename stem, sparse problem-index preservation, and that production CLI/core modules do not import the producer. Tests invoke the standalone `script/` entry point through a subprocess; `script/` is not treated as an importable package.

### 2. Fixed checked-in component configuration

Add:

`script/configs/adaptive/01_action_decode/action_auxiliary_smoke.yaml`

Stack it after `script/configs/adaptive/base.yaml` and `script/configs/adaptive/00_smoke/default_smoke.yaml`.

Exact component settings:

```yaml
model:
  loss:
    action_vicreg_coeff: 0.1
    action_vicreg_std_coeff: 1.0
    action_vicreg_cov_coeff: 1.0
    action_vicreg_std_margin: 1.0
    action_sigreg_coeff: 0.0
    action_contrastive_coeff: 0.1
    action_contrastive_temperature: 0.1
    action_hard_negatives_per_positive: 4
    argument_reconstruction_coeff: 0.1
    applicability_coeff: 0.1
  argument_reconstruction_head:
    kind: mlp
    hidden_dim: 64
    dropout: 0.0
  applicability_head:
    kind: mlp
    hidden_dim: 64
    dropout: 0.0
trainer:
  applicability_pos_weight: 3.0
data:
  action_supervision_seed: 20260721
  action_negative_max_attempts_per_category: 32
  action_applicability_table_path: ${oc.env:ACS_JEPA_ACTION_APPLICABILITY_TABLE}
tracking:
  mlflow_tracking_uri: sqlite:////opt/data/workspace/acs-jepa-runs/acs-jepa-mlflow.db
```

Tracking metadata identifies `stage: smoke_component_test`, seed 0, and the action-identifiability hypothesis. The tracking URI explicitly overrides the stale `/home/awalga` URI and its parent directory must be created before execution. The artifact environment variable must resolve to an absolute path; absence or relativity must fail before training.

The fixed `3.0` positive class weight is the rounded all-corpus deterministic class ratio for seed `20260721` and four candidates: train `4063/1357 = 2.9941`, held-out validation `563/197 = 2.8579`, and all-corpus `4626/1554 = 2.9768`. Record this census in the evidence document; it is class balancing, not deliberate positive over-weighting.

### 3. Minimal applicability count instrumentation

Before smoke execution, add RED/GREEN coverage in the core trainer for detached scalar terms:

- `applicability_num_examples`;
- `applicability_num_positive`;
- `applicability_num_negative`.

Populate them directly from `ApplicabilityLossOutput` in `_add_applicability_terms()`. Do not change loss values, routing, gradients, or default metrics when applicability is disabled. CLI metric serialization will emit these as `term/applicability_num_*` through the existing generic term path.

### 4. TDD and preflight

RED before GREEN:

1. Add producer tests before the producer.
2. Add config-stack/model/dataset preflight tests before the config. Assert all five auxiliary modules/routes are enabled, each trainable optional module is optimizer-owned exactly once, the strict artifact is propagated, and a real collated train/eval step is finite.
3. Add applicability count-term tests before the three term assignments.
4. Keep all Phase 2E disabled-default tests unchanged and green.

Preflight the actual smoke corpus before GPU execution:

- strict corpus load succeeds;
- artifact strict-loader round trip succeeds;
- all trajectory source states are represented in the table;
- resolved config contains the exact fixed values and absolute artifact identity;
- one CPU collated train/eval step emits every expected Stage 2E term.

### 5. Fixed execution

Use only:

```text
dataset: /opt/data/workspace/acs-jepa-tuning-data/smoke
artifact: /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/action_applicability.json
train output: /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0
eval output: /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/eval
seed: 0
split seed: inherited 20260717
training: inherited default smoke, 3 epochs
batch size: inherited 8
rollout steps: inherited 4
device: cuda
```

Execution order:

1. Require the fixed train output directory not to exist. Do not append to or clean an ambiguous prior run. Create `/opt/data/workspace/acs-jepa-runs` and the new run directory; before training it may contain only the newly generated artifact.
2. Build the offline artifact exactly with:

   ```bash
   UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli \
     python script/build_action_applicability_table.py \
     /opt/data/workspace/acs-jepa-tuning-data/smoke \
     --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/action_applicability.json
   ```

3. Run training exactly with:

   ```bash
   ACS_JEPA_ACTION_APPLICABILITY_TABLE=/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/action_applicability.json \
   UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli acs-jepa train \
     /opt/data/workspace/acs-jepa-tuning-data/smoke \
     --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0 \
     --config script/configs/adaptive/base.yaml \
              script/configs/adaptive/00_smoke/default_smoke.yaml \
              script/configs/adaptive/01_action_decode/action_auxiliary_smoke.yaml \
     --device cuda --seed 0
   ```

4. Require the distinct eval directory not to exist, then run:

   ```bash
   UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli acs-jepa eval \
     /opt/data/workspace/acs-jepa-tuning-data/smoke \
     --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
     --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/eval \
     --device cuda
   ```

   Evaluation uses the saved resolved configuration and absolute table identity.
5. Treat training `metrics/eval.jsonl` as held-out validation evidence for the two-problem `p166`/`p192` split (38 windows). Treat post-checkpoint `eval/eval_summary.json` only as an all-corpus (309-window) checkpoint/load smoke; do not call it held-out validation.
6. Do not run planner/decoder/CEM or tune coefficients.

Capture exact commands, wall time, corpus/table counts, checkpoint path/hash, resolved config hash, train/eval metric file paths, and finite metric summaries in `script/ACTION_AUXILIARY_SMOKE.md`. Do not commit bulky run artifacts or checkpoints.

## Required component evidence

The fixed run passes Stage 2F when:

1. training and evaluation exit zero;
2. literal metric keys `jepa_loss`, `action_vicreg_loss`, `action_contrastive_loss`, `argument_reconstruction_loss`, and `applicability_loss` are present and finite;
3. literal keys `term/action_vicreg_num_samples`, `term/action_contrastive_num_examples`, `term/action_contrastive_num_negatives`, `term/argument_num_active_roles`, `term/applicability_num_examples`, `term/applicability_num_positive`, and `term/applicability_num_negative` are present and positive in train, held-out validation, and all-corpus post-checkpoint evaluation records;
4. latest and best checkpoints exist and contain non-`None` state dictionaries for the contrastive anchor, argument reconstruction head, and applicability head;
5. the saved resolved config has exact fixed coefficients, seed, negative count, and absolute table path;
6. the table covers all 345 recorded transition source states across 12 trajectories while preserving the corpus's 24 parsed entries and sparse trajectory problem indices `1, 3, ..., 23`; missing-state unknowns are not silently used in this smoke;
7. no simulator/planner/decoder/oracle call occurs after artifact generation begins training;
8. no NaN/Inf, crash, or missing auxiliary metric occurs.

This is a component execution gate, not an empirical efficacy claim. Loss direction, latent margins, applicability separation, and collapse/aliasing acceptance are assessed independently in Stage 2G.

## Verification

Before implementation review:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_applicability_table_script.py acs-jepa-cli/tests/test_action_auxiliary_smoke_config.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests -q -k 'not tuning_configs_load_and_keep_required_defaults'
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check <all changed Python files>
python -m compileall -q <all changed Python paths>
git diff --check
```

After the documented run, dispatch an independent Stage 2F implementation/evidence review. Fix blockers, rerun affected verification, then create and verify an SSH-signed Stage 2F commit. Do not push.

## Explicit non-goals

- no coefficient search or tuning;
- no Phase 2G margin/applicability acceptance decision;
- no decoder, planner, CEM, manifold, replay, sampling-category, or JEPA objective change;
- no production-time simulator or oracle;
- no checkpoint publication or push;
- no claim that Phase 2 is complete until Stage 2G passes its own review and evidence gate.
