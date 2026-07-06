# ACS-JEPA

ACS-JEPA is a graph-native, action-conditioned JEPA framework for learning
latent transition models from PDDL states and simulator traces.

The project connects three layers:

* a symbolic graph interface for PDDL states, actions, and goals;
* a JEPA transition model over graph and object latents;
* a CLI for training and evaluating models from `pddl-simulator` DuckDB traces,
  with MLflow tracking.

## Acknowledgment

ACS-JEPA is heavily inspired by and hard-forked from Meta's
[EB-JEPA](https://github.com/facebookresearch/eb_jepa/tree/main?tab=readme-ov-file)
project. This repository adapts the action-conditioned JEPA planning ideas from
EB-JEPA to symbolic PDDL factor graphs and grounded actions.

## Packages

```text
packages/acs-jepa-core/
  Core graph encoders, JEPA modules, losses, planners, datasets, and trainer.

acs-jepa-cli/
  Command-line interface for inspecting datasets, training, and evaluation.

dependencies/pddl-generator/packages/pddl-simulator/
  Simulator dependency that writes PDDL state/action traces to DuckDB.
```

The CLI entry point is:

```sh
uv run --package acs-jepa-cli acs-jepa --help
```

## Conceptual Reference

ACS-JEPA follows the action-conditioned JEPA pattern used by the original
EB-JEPA planning code:

$$
z_0 = e(x_0), \qquad z_{t+1} = f_\theta(z_t, a_t).
$$

The original EB-JEPA examples use image/video encoders, continuous action
optimization, and an objective of the form

$$
E_{\text{plan}} = \sum_t \lVert z_g - \hat z_t \rVert^2.
$$

ACS-JEPA keeps the same high-level decomposition but replaces image states with
PDDL factor graphs and grounded symbolic actions:

```text
symbolic state x_t -> graph encoder -> graph/object embeddings
graph/object embeddings -> state encoder -> JEPA latent state z_t
(z_t, grounded action a_t) -> latent predictor -> predicted z_{t+1}
```

For partial goals, a full target latent is often too restrictive. A partial goal
defines a set of acceptable terminal states rather than a single point:

$$
z_T \in \mathcal{G}(g_p).
$$

A planner can therefore optimize an energy

$$
a_{0:T-1}^{*}
=
\arg\min_{a_{0:T-1}}
\left[
E_g(z_T,g_p)
+
\lambda \sum_{t=0}^{T-1} c_a(a_t)
+
\beta \sum_{t=0}^{T-1} c_{\mathrm{reg}}(z_t)
\right],
$$

subject to

$$
z_{t+1}=f_\theta(z_t,a_t).
$$

The goal energy may be implemented as a predicate satisfaction energy,
a conditional distribution over terminal latents, or a factorized latent mask.

## Graph Representation

`acs_jepa.graph.builders` converts parsed PDDL states and grounded actions into
PyTorch tensors for graph neural encoders and JEPA transition models.

### Symbolic State Graphs

A symbolic state is a set of positive grounded atoms over the objects of one
parsed PDDL problem. `build_state_graph(parsed_problem, atoms)` builds a
PyTorch Geometric `Data` object using a factor-node graph:

* object nodes represent PDDL objects;
* atom nodes represent active grounded predicates;
* role-labeled edges connect each atom node to its argument object nodes.

For example,

```text
road_connect(road1, junction0-1, junction1-2)
```

becomes one atom node connected to three object nodes with role ids `0`, `1`,
and `2`. This keeps the graph invariant to atom serialization order while
preserving the argument order of n-ary predicates.

The returned `Data` object contains:

```text
x:          LongTensor[num_nodes, 5]
edge_index: LongTensor[2, num_edges]
edge_attr:  LongTensor[num_edges, 2]
```

Node feature columns:

| Column | Name | Meaning |
| --- | --- | --- |
| 0 | `node_kind` | `OBJECT_NODE` or `ATOM_NODE` |
| 1 | `object_type` | type id for object nodes, `-1` for atom nodes |
| 2 | `predicate_id` | predicate id for atom nodes, `-1` for object nodes |
| 3 | `arity` | atom arity for atom nodes, `0` for object nodes |
| 4 | `object_id` | problem-local object id for object nodes, `-1` for atom nodes |

Edge attribute columns:

| Column | Name | Meaning |
| --- | --- | --- |
| 0 | `role_id` | argument position in the grounded predicate |
| 1 | `edge_direction` | `ATOM_TO_OBJECT` or `OBJECT_TO_ATOM` |

Object nodes are ordered by object name. Atom nodes follow canonical sorted atom
order. Object ids, predicate ids, action ids, and type ids are stable within a
`ParsedProblem`; object ids are intentionally problem-local.

`build_state_graph(..., include_static=False)` can drop atoms whose predicates
are never modified by action effects. The CLI v1 always includes static facts.

### Grounded Action Tensors

`tensorize_action(parsed_problem, action)` converts a grounded PDDL action into
fixed-size tensors:

| Key | Shape | Meaning |
| --- | --- | --- |
| `action_id` | scalar or `[B]` | action-schema id |
| `action_object_indices` | `[max_action_arity]` or `[B, max_action_arity]` | problem-local object ids, padded with `-1` |
| `action_role_ids` | `[max_action_arity]` or `[B, max_action_arity]` | argument positions, padded with `-1` |
| `action_arg_mask` | `[max_action_arity]` or `[B, max_action_arity]` | real arguments versus padding |

The model gathers contextual object embeddings for `action_object_indices`,
combines them with `action_id` and `action_role_ids`, and uses the result to
condition transition prediction.

## Graph JEPA Model

The model separates graph representation, JEPA state encoding, action encoding,
and latent dynamics.

### State Encoder

The graph encoder produces graph-level and object-level embeddings:

$$
h_t = E_{\mathrm{graph}}(x_t).
$$

`StateEncoderF` maps these embeddings into JEPA latent variables:

$$
z_t =
\left(
z_t^{\mathrm{graph}},
\{z_t^{(o)}\}_{o \in \mathcal{O}}
\right).
$$

The same `forward()` handles single states and trajectory windows. Single-state
inputs use graph/object tensors shaped `[B, D]`; trajectory inputs use
`[B, K + 1, D]` for graph latents and `[N_obj, K + 1, D]` for object latents.

### Action Encoder

A grounded action is

$$
\bar a = a_i(o_0,\dots,o_{k_i-1}),
$$

where each argument object has the type required by the action schema. The
encoding must represent both the action schema identity and the objects bound to
its arguments.

The abstract action encoder can be written as

$$
e_a(\bar a, z_t)
=
\rho
\left(
e_{\mathcal{A}}(\operatorname{action\_id}),
\left\{
\left(e_{\mathrm{role}}(j), z_t^{(o_j)}\right)
: 0 \le j < k_i
\right\}
\right),
$$

where $\rho$ is an order-aware or permutation-aware argument composition module.

For heterogeneous action schemas, variable arity can be understood through a
fixed set of typed slots. For action $a_i$ with signature

$$
\sigma_i = (\tau_{i,0},\tau_{i,1},\dots,\tau_{i,k_i-1}),
$$

define the type-local occurrence index

$$
\ell_{i,j}
=
\left|\{h < j : \tau_{i,h}=\tau_{i,j}\}\right|.
$$

The corresponding slot signature is

$$
\Omega_i =
\left(
(\tau_{i,0},\ell_{i,0}),
\dots,
(\tau_{i,k_i-1},\ell_{i,k_i-1})
\right).
$$

The occurrence index is necessary when the same type appears more than once in
one action signature. For example,
`same_line(junction, junction)` uses slots `(junction,0)` and `(junction,1)`.
The implementation uses a simpler schema-ordered padding convention, but the
same information is carried by `action_arg_mask`, `action_object_indices`, and
`action_role_ids`.

ACS-JEPA encodes grounded actions against the current `JEPALatentState`.
`GraphJEPA` runs the graph encoder and state encoder first, then
`LatentActionEncoder` gathers action arguments from
`latent_state.object_latents`.

Both use `PooledArgumentEncoder` or `RNNArgumentEncoder` for role-aware argument
composition. `ActionEncoder.forward()` accepts either one action batch or a
batched temporal action tensor `[B, K, ...]` and returns `[B, D_a]` or
`[B, K, D_a]`. Graph-sourced action encoding was removed because future planner
rollout states have predicted latents, not observed graph encoder object
embeddings:

```python
action_encoder = build_action_encoder(
    kind="pooled",
    num_actions=num_actions,
    max_action_arity=max_action_arity,
    latent_dim=latent_dim,
    action_dim=action_dim,
)
```

### Predictor

The latent predictor consumes current state latents and an action latent:

$$
\hat z_{t+1} = G_\theta(z_t, e_a(\bar a_t,z_t)).
$$

The default predictor is a residual MLP update over graph and object latents.
A GRU-based predictor is also available through `build_latent_predictor`.

### Training Objective

Training uses fixed-length trajectory windows
$(x_0,\bar a_0,x_1,\dots,\bar a_{K-1},x_K)$. The graph encoder is applied to
each state independently, then causal state/action sequence encoders produce
time-aware latents:

$$
z_t=f_\psi(x_{0:t}), \qquad u_t=q_\omega(\bar a_{0:t}).
$$

The predictor is recursively unrolled to compute prediction losses for all
orders $k=1,\dots,K$. Order one is the $k=1$ horizon; higher orders
compare recursively predicted latents against observed target latents:

$$
\mathcal{L}_{\mathrm{pred}}
=
\sum_{k=1}^{K}
\alpha_k
\frac{1}{K-k+1}
\sum_{t=k}^{K}
\ell(\hat z_t^{(k)}, z_t^{(0)}).
$$

The full trajectory objective is:

$$
\mathcal{L}
=
\lambda_{\mathrm{pred}}
\mathcal{L}_{\mathrm{pred}}(\hat z_{1:K}, z_{1:K})
+
\lambda_{\mathrm{vc}}
\mathcal{L}_{\mathrm{vc}}(z_{1:K})
+
\lambda_{\mathrm{sim}}
\mathcal{L}_{\mathrm{sim}}(z_{0:K},\hat z_{1:K})
+
\lambda_{\mathrm{idm}}
\mathcal{L}_{\mathrm{idm}}(z_{0:K},u_{0:K-1}).
$$

The prediction term compares graph and object latents:

$$
\mathcal{L}_{\mathrm{pred}}
=
w_g
\lVert \hat z_{t+1}^{\mathrm{graph}} - z_{t+1}^{\mathrm{graph}}\rVert^2
+
w_o
\frac{1}{|\mathcal{O}|}
\sum_o
\lVert \hat z_{t+1}^{(o)} - z_{t+1}^{(o)} \rVert^2.
$$

The variance/covariance regularizer discourages latent collapse and is averaged
over observed target states in the window. Optional temporal-similarity and
inverse-dynamics terms are averaged over adjacent trajectory steps.

`JepaTrainer` wraps `GraphJEPA.trajectory_rollout()` for optimization and
`JepaTrainer.eval_step()` for no-gradient evaluation.

## Action Decoding

`ActionDecoder` maps an action latent back to a type-valid grounded action. It
builds an `ActionDecodingSpace` from the parsed problem, where a candidate has
the form

$$
(\operatorname{action\_id}, u_0,\dots,u_{K_{\max}-1}).
$$

Decoding is type-constrained: for an action id and role id, only object ids
compatible with the required argument type are legal.

Two decoding modes are implemented:

* exact enumeration over all type-valid grounded actions;
* approximate CEM/MPPI search through `ActionSamplingFamily`.

Each candidate action is tensorized, encoded with the same latent action encoder
against the current `JEPALatentState`, and scored against the target action
latent using either negative squared distance or cosine similarity. The decoder
does not enforce state-dependent preconditions; those must be checked
separately, for example by using the PDDL simulator during plan-time inspection.

## Planning

Inference-time planning is implemented outside `GraphJEPA`. `LatentMPPIPlanner`
depends directly on the `graph_encoder`, `state_encoder`, `predictor`, and a
goal-energy callable compatible with `acs_jepa.goals`:

```text
goal_energy(goal_tensors, terminal_state) -> FloatTensor[N]
```

Continuous MPPI samples action-latent sequences `[N, H, D_a]`, rolls them
through the predictor, and scores terminal states with the goal energy. V1 uses
a constant per-action cost, so the fixed-horizon action-cost term is logged for
diagnostics but does not change candidate ranking.

`PlannerAgent` connects this latent planner to `pddl-simulator`. It builds the
current graph from `engine.current_facts()`, decodes the selected latent prefix,
and applies only the first valid prefix with
`engine.apply_action(..., finish=True)`. The simulator is the validity oracle:
invalid decoded actions raise `ValueError`. If the first decoded action is
invalid, the agent retries MPPI with a rejection penalty up to
`max_decode_attempts`; if a later decoded action is invalid, it replans from the
updated simulator state.

## CLI and Dataset Flow

The CLI trains from `pddl-simulator` DuckDB outputs. A dataset directory has the
layout:

```text
dataset/
  problem/
    domain.pddl
    p01.pddl
    p02.pddl
  simulation/
    simulation.duckdb
```

The relevant simulator rows are solved sequential-plan transitions. Ingestion
groups them by `(dataset_id, run_id, problem_name)`, validates state continuity,
and creates fixed-length sliding trajectory windows:

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

Multiple dataset directories are merged into one logical corpus. Because
`run_id` values are local to a DuckDB file, ingestion attaches a dataset/source
id before grouping trajectories. Splits are by problem, not by individual
trajectory window.

`data.rollout_steps` configures `K`, the number of actions in each training
window. Every emitted sample has exactly `K` actions and `K + 1` states:

```yaml
data:
  rollout_steps: 4
```

DuckDB runs may have different lengths, but the model never receives
variable-length trajectories. After rows are grouped and sorted by `step_index`,
the loader reconstructs the full run as
`states = [state_0, next_state_0, ..., next_state_{N-1}]` and
`actions = [action_0, ..., action_{N-1}]`. Runs with fewer than `K` actions are
dropped. Longer runs become overlapping fixed-size windows:

```text
N actions, K rollout steps -> max(0, N - K + 1) windows

K = 4:
  N = 2  -> 0 windows
  N = 4  -> 1 window
  N = 5  -> 2 windows
  N = 10 -> 7 windows
```

There is no padding or sequence-length masking at the training boundary; window
creation is the alignment step.

Commands:

```sh
uv run --package acs-jepa-cli acs-jepa inspect-data DATASET_DIR [DATASET_DIR ...]

uv run --package acs-jepa-cli acs-jepa train DATASET_DIR [DATASET_DIR ...] \
  --output RUN_DIR \
  --config CONFIG_FILE \
  --device cpu \
  --seed 0

uv run --package acs-jepa-cli acs-jepa eval DATASET_DIR [DATASET_DIR ...] \
  --checkpoint RUN_DIR/checkpoints/latest.pt \
  --output EVAL_DIR \
  --device cpu

uv run --package acs-jepa-cli acs-jepa plan \
  --checkpoint RUN_DIR/checkpoints/latest.pt \
  --domain DATASET_DIR/problem/domain.pddl \
  --problem DATASET_DIR/problem/p01.pddl \
  --output PLAN_DIR \
  --device cpu \
  --seed 0
```

Start from the template config:

```text
acs-jepa-cli/configs/template.yaml
```

Training and evaluation use MLflow. The config supports:

```yaml
optimizer:
  name: adam
  lr: 0.001
  weight_decay: 0.0
  scheduler:
    kind: warmup_cosine
    warmup_ratio: 0.1
    min_lr: 0.00001
    start_factor: 1.0e-8

tracking:
  mlflow_tracking_uri: null
  experiment_name: acs-jepa
  run_name: null
  tags: {}
  log_artifacts: true
```

The CLI writes local outputs and logs them to MLflow:

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
    dataset_summary.json
    split_manifest.json
```

## Design Decisions and Limitations

* Static facts are always included by the CLI v1.
* CLI v1 uses only solved `SequentialPlan` traces.
* Numeric fluents, action duration, phase, and running actions are loaded from
  the simulator but are not modeled by the first training path.
* Object ids are problem-local. Generalization across problem instances should
  rely on type, role, and contextual object embeddings, not raw object ids.
* Type-correct action decoding does not imply precondition satisfaction.
* The current symbolic representation assumes fixed action schemas and bounded
  arity.
* If the PDDL type system uses inheritance or multiple compatible types,
  decoding must use the compatible object set for each required type.

## Development

Run the CLI tests:

```sh
uv run pytest acs-jepa-cli/tests -q
```

Run the core tests:

```sh
uv run pytest packages/acs-jepa-core/tests -q
```

Run the combined test set used during development:

```sh
uv run pytest packages/acs-jepa-core/tests acs-jepa-cli/tests -q
```
