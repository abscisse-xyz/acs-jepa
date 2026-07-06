# ACS-JEPA CLI

This package is the planned command-line interface for training and evaluating
ACS-JEPA models from PDDL simulation data.

The CLI is intended to sit between:

* `acs-jepa-core`, which provides graph encoders, JEPA modules, losses, action
  encoders, goal heads, and trainer utilities;
* `pddl-simulator`, which replays PDDL plans and persists state/action traces to
  DuckDB.

This document is a feature and implementation sketch. It describes the expected
CLI behavior and data contracts rather than a finalized command surface.

## Dataset Layout

A dataset directory should contain a PDDL problem collection and one simulator
database:

```text
dataset/
  problem/
    domain.pddl
    p01.pddl
    p02.pddl
    ...
  simulation/
    simulation.duckdb
```

The `problem/` directory defines the symbolic planning domain and the problem
instances. The `simulation/simulation.duckdb` database contains solver or
simulator output for those problems.

Multiple dataset directories should be accepted by training and evaluation
commands. The CLI should treat them as one logical corpus after validating that
their schemas and model-relevant domain assumptions are compatible. Because
`run_id` values are local to a single DuckDB file, merged datasets must attach a
stable dataset/source id before grouping trajectories or splitting data.

## Simulator Input Contract

`pddl-cli simulate` writes one DuckDB database per simulation session. The
database contains:

```text
simulation_runs(id, domain_name, problem_name, created_at)
state_snapshots(...)
action_events(...)
trace_entries(...)
state_action_transitions(...)
```

For JEPA training, the main table-like source is the
`state_action_transitions` view. It joins each executed action with its previous
and next state snapshots:

```sql
SELECT
    run_id,
    step_index,
    sim_time,
    phase,
    action_name,
    action_kind,
    arguments,
    duration,
    state_facts,
    state_numeric_values,
    state_running_actions,
    next_state_facts,
    next_state_numeric_values,
    next_state_running_actions
FROM state_action_transitions
ORDER BY run_id, step_index;
```

For model training and evaluation, relevant transitions are those from solved
sequential plans. Join through `simulation_runs` and `planner_attempts` to select
those rows and retain `problem_name`. Within one simulator database,
`(run_id, problem_name)` identifies the trajectory, which is useful for
autoregressive evaluation:

```sql
SELECT
    r.domain_name,
    r.problem_name,
    t.run_id,
    t.step_index,
    t.sim_time,
    t.phase,
    t.action_name,
    t.arguments,
    t.duration,
    t.state_facts,
    t.state_numeric_values,
    t.next_state_facts,
    t.next_state_numeric_values
FROM state_action_transitions t
JOIN simulation_runs r
  ON r.id = t.run_id
JOIN planner_attempts p
  ON r.id = p.run_id
WHERE p.status IN ('SOLVED_SATISFICING', 'SOLVED_OPTIMALLY')
  AND p.plan_topology = 'SequentialPlan'
ORDER BY t.run_id, t.step_index;
```

Simulator columns such as `facts`, `numeric_values`, and `running_actions` are
DuckDB native lists of structs, not JSON strings. The CLI should preserve this
structure until it converts rows into `acs_jepa.graph` tensors.

Additional simulator details are documented in
`dependencies/pddl-generator/packages/pddl-simulator/src/simulator/README.md`.

## Core Workflows

### Train

The training command should build fixed-length trajectory windows from one or
more dataset directories and optimize a `GraphJEPA` model.

Expected responsibilities:

* load each `domain.pddl` and problem file;
* read simulator rows from `state_action_transitions` and group them into runs;
* convert simulator facts into graph states with `build_state_graph`;
* convert executed actions into action tensors with `tensorize_action`;
* construct batched samples compatible with `JepaTrainer.train_step`;
* periodically run validation with `JepaTrainer.eval_step`;
* write checkpoints, training metrics, and configuration metadata.

The command should support at least:

```text
acs-jepa train DATASET_DIR [DATASET_DIR ...]
    --output RUN_DIR
    --config CONFIG_FILE
    --device cpu|cuda|mps
    --seed SEED
```

Use `configs/template.yaml` as the starting config.

### Evaluate

The evaluation command should load a checkpoint and compute losses or downstream
metrics on held-out simulation transitions.

Expected responsibilities:

* load the saved model and training configuration;
* build the same dataset representation used for training;
* run `JepaTrainer.eval_step` or an equivalent no-gradient evaluation loop;
* report aggregate JEPA loss, goal-head loss when configured, and per-term loss
  statistics;
* optionally export predictions, latent rollouts, and decoded actions for
  inspection.

The command should support at least:

```text
acs-jepa eval DATASET_DIR [DATASET_DIR ...]
    --checkpoint CHECKPOINT
    --output EVAL_DIR
    --device cpu|cuda|mps
```

### Plan

The planning command loads a checkpoint, builds a simulator from one domain and
problem file, and runs the latent MPPI `PlannerAgent` with the checkpoint's graph
encoder, state encoder, predictor, action encoder, and configured goal head.

```text
acs-jepa plan
    --checkpoint CHECKPOINT
    --domain DOMAIN_PDDL
    --problem PROBLEM_PDDL
    --output PLAN_DIR
    --device cpu|cuda|mps
    --seed SEED
```

Planning requires `model.goal_head.kind` to be `predicate`, `gaussian`, `gmm`, or
`conditional_sampler`. The command writes `plan_summary.json`; it returns zero
only when the simulator goal is reached.

### Inspect Data

A lightweight inspection command is useful before training.

Expected responsibilities:

* verify that `domain.pddl`, problem files, and `simulation.duckdb` exist;
* list simulation runs and problem names;
* count transitions per run and per dataset;
* report action names, fact counts, and missing before/after snapshots;
* fail early on malformed rows that cannot be converted to graph/action tensors.

Possible command:

```text
acs-jepa inspect-data DATASET_DIR [DATASET_DIR ...]
```

## Model Configuration

The CLI should keep model choices explicit in a config file rather than burying
them in command flags. Important options include:

* graph encoder width, embedding dimension, and number of layers;
* JEPA latent dimension;
* latent action argument encoder: `pooled` or `rnn`;
* latent predictor: MLP or GRU;
* loss weights and variance/covariance regularization settings;
* optional goal head kind and goal loss weight;
* optimizer, learning rate, warmup cosine scheduler settings, batch size, and
  gradient clipping.

The action encoder configuration should map directly to
`build_action_encoder(kind=...)`; action arguments are always encoded from JEPA
representation latents.

## Dataset Conversion Notes

The simulator and ACS-JEPA use different but compatible representations:

* simulator facts are structured records with a predicate name and argument
  names;
* ACS-JEPA graph states use parsed PDDL metadata to map predicates, object
  names, object types, and arguments to tensors;
* simulator actions provide `action_name` and `arguments`;
* ACS-JEPA actions use problem-local action ids, object ids, role ids, and masks.

Object ids are problem-local. The CLI must not assume that the same object id
has the same meaning across different problem instances.

For temporal simulations, `state_action_transitions` may include action phase and
duration. The first training target should be the discrete transition
`state_facts -> next_state_facts`. Numeric fluents, running actions, pending
events, phases, and durations should remain available for later model variants
rather than being silently discarded in the ingestion layer.

## Outputs

A run directory should be self-contained:

```text
run/
  config.yaml
  checkpoints/
    latest.pt
    best.pt
  metrics/
    train.jsonl
    eval.jsonl
  artifacts/
    decoded_actions.parquet
    rollouts.parquet
```

The minimum useful output is a checkpoint plus JSONL metrics. Parquet exports are
optional but natural because simulator data is already stored in analytical
tables.

## Design Decisions

* Multiple dataset directories are merged into one logical corpus, then split by
  problem rather than by transition. Since `run_id` and problem ids are local to
  each dataset directory, the ingestion layer must add a dataset/source id before
  merging.
* Static facts are always included in graph states.
* Temporal fields such as `phase`, `duration`, and `running_actions` are not
  supported by the first CLI training path. They should remain available in the
  loaded rows for future model variants.
* Precondition validity should be checked when decoded actions are inspected at
  plan time. The PDDL simulator can be used to validate candidate action
  sequences.
* Configuration should use YAML with OmegaConf.
