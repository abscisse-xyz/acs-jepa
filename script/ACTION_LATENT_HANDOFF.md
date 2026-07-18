# Action Latent Space Investigation Handoff

Date: 2026-07-17

## Current status

The tuning campaign is paused. The current blocker is not selecting the next
hyperparameter set; it is understanding why the learned action latent space does
not provide a useful enough energy landscape for CEM action decoding and
planning.

The model can train and checkpoint correctly on the smoke subset. The failure
mode appears when a planned latent action is decoded back to a grounded PDDL
action: the decoder often chooses a type-correct but state-invalid action,
usually by selecting wrong object arguments. This makes the planner fail before
it can evaluate meaningful multi-step behavior.

Important correction: inverse dynamics should not be treated as rejected as a
research direction. The enabled inverse-dynamics run proved that the auxiliary
objective can be wired, trained, and can improve the learned latent-action
regression signal. The specific 3-epoch smoke run did not improve the bounded
planner probe, but that result is evidence that the downstream decoding setup
still needs analysis, not evidence against inverse dynamics.

## Relevant workspace paths

- Repository: `/home/awalga/workspace/acs-jepa`
- Derived tuning data: `/home/awalga/workspace/acs-jepa-tuning-data`
- Smoke run outputs: `/home/awalga/workspace/acs-jepa-runs/smoke`
- MLflow DB: `/home/awalga/workspace/acs-jepa-mlflow.db`
- MLflow artifacts: `/home/awalga/workspace/acs-jepa-mlartifacts`
- Campaign log: `script/TUNING_LOG.md`
- Action decoder diagnostic: `script/diagnose_action_decoder.py`

## Fixed smoke data

The reproducible smoke subset is:

- Dataset root: `/home/awalga/workspace/acs-jepa-tuning-data/smoke`
- Split seed: `20260717`
- Train problems: `p010`, `p017`, `p043`, `p056`, `p067`, `p087`, `p129`,
  `p151`, `p165`, `p180`
- Validation problems: `p166`, `p192`
- Smoke transitions: 345 total
- Validation transitions: 44 total

The source dataset was not modified. Derived data was produced by
`script/prepare_tuning_data.py`.

## Reproducing the issue

Run commands from `/home/awalga/workspace/acs-jepa`.

Use:

```bash
UV_CACHE_DIR=/home/awalga/workspace/.uv-cache
```

### 1. Train the baseline smoke model

```bash
UV_CACHE_DIR=/home/awalga/workspace/.uv-cache uv run --package acs-jepa-cli acs-jepa train \
  /home/awalga/workspace/acs-jepa-tuning-data/smoke \
  --output /home/awalga/workspace/acs-jepa-runs/smoke/default_seed0 \
  --config script/configs/adaptive/base.yaml script/configs/adaptive/00_smoke/default_smoke.yaml \
  --device cuda \
  --seed 0
```

Expected baseline training evidence:

- MLflow run: `05d3c30b2db64e16ba47c188e94f469a`
- Checkpoint: `/home/awalga/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt`
- Training succeeds, losses are finite, JEPA loss decreases.

### 2. Run the bounded action-decoder diagnostic

```bash
UV_CACHE_DIR=/home/awalga/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_decoder.py \
  /home/awalga/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /home/awalga/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --output /home/awalga/workspace/acs-jepa-runs/smoke/default_seed0/action_decode_val_cem_fast \
  --device cuda \
  --split val \
  --method cem \
  --decoder-num-samples 64 \
  --decoder-max-iters 8 \
  --seed 0
```

Observed baseline result:

- MLflow run: `7af3e78165d84a55afce77ffb27b141c`
- Transitions: 44
- Exact-match rate: 0.363636
- Applicable-decoded-action rate: 0.363636
- Failure pattern: decoded action names match the true action-name histogram,
  but object arguments are often wrong and the decoded actions are usually
  simulator-invalid.

### 3. Reproduce bounded planning failure

```bash
UV_CACHE_DIR=/home/awalga/workspace/.uv-cache uv run --package acs-jepa-cli acs-jepa plan \
  --checkpoint /home/awalga/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --domain /home/awalga/workspace/acs-jepa-tuning-data/smoke/problem/domain.pddl \
  --problem /home/awalga/workspace/acs-jepa-tuning-data/smoke/problem/p166.pddl \
  --output /home/awalga/workspace/acs-jepa-runs/smoke/default_seed0/planning/p166_fast \
  --config script/configs/adaptive/base.yaml script/configs/adaptive/00_smoke/default_smoke.yaml script/configs/adaptive/00_smoke/planning_probe_fast.yaml \
  --device cuda \
  --seed 0
```

Repeat with `p192.pddl`.

Observed baseline fast planning:

- `p166`: `decode_invalid`, 0 applied actions
- `p192`: `decode_invalid`, 0 applied actions

The default unbounded planner can spend minutes in nested CEM decoding after an
invalid decoded action because rejection penalties decode many candidate first
actions. The bounded probe avoids that loop and exposes first-action validity
directly.

### 4. Reproduce inverse-dynamics component run

```bash
UV_CACHE_DIR=/home/awalga/workspace/.uv-cache uv run --package acs-jepa-cli acs-jepa train \
  /home/awalga/workspace/acs-jepa-tuning-data/smoke \
  --output /home/awalga/workspace/acs-jepa-runs/smoke/inverse_dynamics_seed0 \
  --config script/configs/adaptive/base.yaml script/configs/adaptive/00_smoke/default_smoke.yaml script/configs/adaptive/01_action_decode/inverse_dynamics_smoke.yaml \
  --device cuda \
  --seed 0
```

Observed inverse-dynamics evidence:

- MLflow train run: `eb662c306d7944b8a3c01cf0a4131f36`
- MLflow eval run: `ea58052c414249f88d49ac7ca38edcf1`
- Median inverse-dynamics loss decreased from about 0.026512 to 0.002271.
- This confirms the component is active and trainable.

The bounded CEM decoder metric for this short run did not improve over baseline,
but the correct interpretation is: inverse dynamics improved the latent-action
regression objective, while the action latent geometry/decoder landscape still
needs investigation.

## Why `engine.applicable_actions()` is not the preferred next step

`SimulatorEngine.applicable_actions()` can enumerate grounded applicable actions.
That is useful as a debugging oracle on small problems, but it is not a scalable
solution. On large PDDL instances, grounding/enumerating applicable actions can
be very expensive. If full grounding is affordable, a classical planner or A*
style heuristic search may already be competitive, reducing the value of ACS-JEPA
as a learned planner.

Use `applicable_actions()` only as an offline diagnostic oracle, not as the
primary planning mechanism.

## Main investigation question

Why does the action encoder output fail to induce a useful structure or energy
landscape for CEM decoding?

More concretely:

- Are true action latents separated from near-miss object-argument substitutions?
- Does the action latent encode action schema but not argument identity/order?
- Are different grounded actions collapsed in latent space because the training
  objective only requires good transition prediction?
- Does the CEM decoder objective have many flat or aliased regions?
- Does the action encoder use contextual object embeddings that are not stable
  enough across states for decoding?
- Does the latent action space preserve simulator precondition information, or
  only transition-relevant information after the fact?

## Recommended investigation directions

### 1. Action latent nearest-neighbor analysis

For each validation transition:

- Encode the true action in the true source state.
- Encode a candidate set of type-valid grounded actions in the same state.
- Rank candidates by L2/cosine distance to the true action latent.
- Report:
  - true action rank;
  - top-k exact match;
  - top-k same action name;
  - top-k applicable action;
  - distance margin between the true action and nearest wrong action;
  - whether wrong neighbors differ by object, role order, action schema, or road/car/junction entity class.

This directly tests whether the latent space itself separates correct actions
before CEM optimization is involved.

### 2. CEM landscape diagnostics

Instrument `ActionDecoder._decode_with_ce` on the smoke validation problems:

- Best score per CEM iteration.
- Elite score mean/std.
- Entropy of action-id categorical distribution.
- Entropy of each argument-role categorical distribution.
- Whether action-id entropy collapses before argument entropy.
- Whether the final distribution puts mass on type-valid but state-invalid
  arguments.

The current evidence suggests action names are easier than arguments. This
diagnostic should confirm whether CEM has a usable gradient-free signal for
arguments.

### 3. Compare exact/ranked search on small validation states

Exact decoding is expensive but usable for a small controlled slice if optimized
or chunked. Use it offline for only a few states:

- Compute exact rank of the ground-truth action.
- Compare exact nearest-neighbor failure to CEM failure.
- If exact rank is poor, the latent geometry is poor.
- If exact rank is good but CEM fails, the decoder optimizer/sampling family is
  the bottleneck.

The previous exact diagnostic was interrupted because exhaustive encoding was
slow. Optimize this by caching candidate action tensors and repeated latent
state expansion per problem/state.

### 4. Action latent supervised probes

Train small frozen-latent probes from action encoder output:

- Predict action schema id.
- Predict each argument object id.
- Predict each argument type.
- Predict whether the action is applicable in the source state.

Run probes on baseline and inverse-dynamics checkpoints. If schema is easy but
object roles are hard, this confirms the observed failure mode quantitatively.

### 5. Latent collapse and invariance checks

Measure within-state action latent distances:

- Same action schema, different arguments.
- Same objects, different role order.
- Same road/car/junction entities in nearby topology.
- Applicable versus inapplicable actions.
- Reference action versus one-argument substitutions.

Plot or log distributions. A useful action latent for CEM should give near-miss
substitutions a clear but not flat penalty.

### 6. Architecture hypotheses to test after diagnostics

Do not start these as blind configs. Use the diagnostics above to decide which
failure is real.

Candidate directions:

- Add a contrastive action-latent loss over grounded actions in the same state:
  push true action away from hard negative same-schema/wrong-argument actions.
- Add action-argument reconstruction heads from action latent and state context.
- Add an applicability classifier head trained from sampled type-valid negatives
  and true applicable positives.
- Make inverse dynamics predict or contrast against action identity/arguments,
  not only regress to the current action encoder latent.
- Add role-aware/object-aware scoring for decoding instead of decoding the whole
  action as one latent vector.
- Normalize or temperature-scale action latents so CEM score margins are not too
  small.
- Cache and reuse per-state candidate encodings to make exact/ranked decoder
  diagnostics practical.

## Suggested next artifact

Create a diagnostic script, for example:

`script/diagnose_action_latent_geometry.py`

Minimum useful output:

- JSON summary with top-k and rank metrics.
- JSONL details per transition.
- MLflow run with metrics and artifacts.
- Optional CSV of nearest neighbors for manual inspection.

Recommended first run:

```bash
UV_CACHE_DIR=/home/awalga/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_latent_geometry.py \
  /home/awalga/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /home/awalga/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --output /home/awalga/workspace/acs-jepa-runs/smoke/default_seed0/action_latent_geometry_val \
  --device cuda \
  --split val \
  --top-k 10 \
  --max-transitions 44 \
  --seed 0
```

Then repeat against:

- `/home/awalga/workspace/acs-jepa-runs/smoke/inverse_dynamics_seed0/checkpoints/best.pt`

The comparison should decide whether inverse dynamics improves latent geometry
even if the current CEM decoder does not yet exploit it.

## Stopping criterion for resuming tuning

Resume hyperparameter/model tuning only after one of these is true:

- Exact/ranked latent diagnostics show the true action is usually in top-k and
  CEM is the bottleneck; then tune/replace the decoder.
- Exact/ranked diagnostics show the true action is not separated; then change
  the action representation or add contrastive/reconstruction/applicability
  objectives.
- A bounded planner probe applies valid actions consistently on both fixed smoke
  validation problems without relying on full applicable-action grounding.
