# Stage 2E Plan — Disabled-by-default action auxiliary trainer/config/model/checkpoint integration

Date: 2026-07-21
Status: implemented; independent plan and implementation reviews PASS
Governing specification: `script/ACTION_LATENT_SOLUTION_SPEC.md`, Phase 2
Prerequisites: Stage 2D1 commit `109eb44`; Stage 2D2 commit `2cf3114`

## Objective

Wire the reviewed Stage 2 action-latent primitives and Stage 2D2 nested symbolic supervision into the ordinary training path without changing the JEPA transition objective, decoder, planner, or default behavior.

Stage 2E provides one coherent, disabled-by-default integration gate covering:

1. trainer composition of action VICReg, explicit hard-negative contrastive, role-object argument reconstruction, and applicability losses;
2. deterministic encoding of sampled negative action sequences against the same causal source/history as each positive trace action;
3. CLI model/loss/config/optimizer construction;
4. CLI dataset configuration and strict loading of optional offline applicability artifacts;
5. checkpoint save/load compatibility for newly trainable modules;
6. detached scalar diagnostics for Phase 2F/2G.

No Stage 2F smoke campaign is run here. No learned score is used by decoding or planning.

## Research/spec mapping

- VICReg (Bardes et al., arXiv:2105.04906) motivates action-latent variance/covariance control, but does not replace grounded identity supervision.
- Latent action representation work (Chandak et al., arXiv:1902.00183; CLAM, arXiv:2505.04999) motivates jointly preserving actionable grounding rather than only transition prediction.
- The governing spec, Phase 2 lines 598–621, requires action VICReg, explicit hard negatives, applicability, argument reconstruction, configuration, and a later fixed smoke protocol.
- Stage 2D1/2D2 already define deterministic negatives, target/candidate masks, optional offline labels, fixed capacities, and `[B,K,...]` PyG collation. Stage 2E consumes those contracts; it does not resample or relabel.
- The Stage 2B contrastive primitive requires an anchor independent of the candidate action. Stage 2E uses a dedicated `GraphInverseDynamicsModel(source_state, next_state)` anchor, avoiding the degenerate positive-as-its-own-anchor shortcut.

## Scope

### Core trainer integration

Modify:

- `packages/acs-jepa-core/src/acs_jepa/training.py`
- `packages/acs-jepa-core/tests/test_training.py`
- core exports only if a new public trainer output type is introduced

Add optional trainer modules/configuration:

- `ActionVICRegLoss`;
- `ActionContrastiveLoss`;
- dedicated `GraphInverseDynamicsModel` contrastive anchor;
- `ArgumentReconstructionHead`;
- `ArgumentReconstructionLoss`;
- existing `ApplicabilityHead` and `ApplicabilityLoss`, now able to consume Stage 2D2 tensors while preserving the Phase 1 precomputed-latent fallback.

### CLI model/config/optimizer integration

Modify:

- `acs-jepa-cli/src/acs_jepa_cli/modeling.py`
- `acs-jepa-cli/tests/test_modeling.py`
- `script/configs/default.yaml`

Construct optional modules only when their effective coefficient is positive. Include trainable contrastive-anchor and argument-head parameters in the optimizer exactly once. Keep all defaults disabled.

### CLI dataset/offline-label integration

Modify:

- `acs-jepa-cli/src/acs_jepa_cli/data.py`
- `acs-jepa-cli/tests/test_data.py`

Build `ActionSupervisionConfig` only when contrastive, argument-reconstruction, or integrated applicability supervision is enabled. Load an optional strict JSON applicability table into the immutable Stage 2D2 table contract.

### Checkpoint integration

Modify:

- `acs-jepa-cli/src/acs_jepa_cli/cli.py`
- existing CLI checkpoint tests (currently in `acs-jepa-cli/tests/test_cli.py`, or the exact adjacent test file owning `_save_checkpoint`/`_load_checkpoint_state`)

Persist explicit `None` for disabled new trainable modules, load via `.get(...)` for old-checkpoint compatibility, and warn rather than fail when an enabled new module is absent from an old checkpoint.

## Explicit non-goals

- no simulator import/call in core trainer, dataset reads, model construction, or checkpoint paths;
- no online applicability callback;
- no use of `SimulatorEngine.applicable_actions()` except a future offline producer/smoke preparation step;
- no production candidate generation or oracle-driven sampling;
- no decoder score, CEM marginal, planner, or action-manifold change;
- no SIGReg implementation: expose `action_sigreg_coeff` with default `0.0` and reject nonzero values with a clear not-yet-supported error;
- no tuning, retraining campaign, p166/p192 acceptance run, or broad benchmark;
- no change to the existing JEPA transition loss calculation;
- no silent change to default optimizer parameter sets, sample dictionaries, checkpoints, or metrics;
- no Cartesian action enumeration.

Stage 2E does not complete governing Phase 2 by itself. Stage 2F must still add and execute the fixed component smoke configuration, and Stage 2G must still produce updated empirical diagnostics/acceptance evidence before broad tuning can resume.

## Configuration contract

Add the governing Phase 2 fields under `model.loss`:

```yaml
model:
  loss:
    action_vicreg_coeff: 0.0
    action_vicreg_std_coeff: 1.0
    action_vicreg_cov_coeff: 1.0
    action_vicreg_std_margin: 1.0
    action_sigreg_coeff: 0.0
    action_contrastive_coeff: 0.0
    action_contrastive_temperature: 0.1
    action_hard_negatives_per_positive: 0
    applicability_coeff: 0.0
    argument_reconstruction_coeff: 0.0
  argument_reconstruction_head:
    kind: none
    hidden_dim: 64
    dropout: 0.0
```

Add data controls:

```yaml
data:
  action_supervision_seed: 0
  action_negative_max_attempts_per_category: 32
  action_applicability_table_path: null
```

Compatibility and routing policy:

- preserve existing `trainer.applicability_loss_weight`, `trainer.applicability_head_detach`, and `trainer.applicability_pos_weight` as the legacy precomputed-latent applicability path;
- add a distinct `JepaTrainerConfig.integrated_applicability_loss_weight` populated only from canonical `model.loss.applicability_coeff`;
- the two trainer fields are mode selectors as well as coefficients; routing must never be inferred from whether unrelated `action_supervision` happens to be present;
- exact coefficient matrix:
  - canonical `0`, legacy `0`: applicability disabled;
  - canonical `>0`, legacy `0`: integrated Stage 2D2 applicability mode;
  - canonical `0`, legacy `>0`: legacy precomputed-latent mode, even when contrastive or argument reconstruction independently adds `action_supervision` to the batch;
  - canonical `>0`, legacy `>0`: reject as ambiguous, regardless of whether values are equal;
- `JepaTrainer._applicability_loss(...)` branches only on these explicit config fields: integrated first is impossible to overlap by validation; legacy never reads `action_supervision`;
- the single optional `JepaTrainerStepOutput.applicability_loss` represents whichever mutually exclusive applicability mode is active;
- no existing config must be edited to retain prior behavior.

Validation:

- all coefficients and temperatures/margins must be finite; coefficients non-negative; temperature/margin positive;
- integer capacities/seeds/attempt budgets reject booleans; negatives are non-negative; attempt budget positive;
- nonzero `action_sigreg_coeff` is rejected as deferred;
- contrastive coefficient > 0 requires `action_hard_negatives_per_positive >= 1` and a contrastive anchor;
- argument reconstruction coefficient > 0 requires an enabled argument head, `max_action_arity >= 1`, and at least one object in the model vocabulary;
- canonical integrated applicability coefficient > 0 requires an enabled applicability head, at least one requested negative, and an absolute `action_applicability_table_path` for the CLI-integrated path;
- legacy applicability coefficient > 0 retains its existing enabled-head/module requirements but never requires the Stage 2D2 table;
- canonical and legacy applicability coefficients may not both be positive;
- enabling a head with zero coefficient may be rejected to avoid unused optimizer state, or allowed but excluded from the optimizer; choose one policy and test it consistently. Preferred: allow constructible heads only when the corresponding coefficient is positive and return `None` otherwise.

## Offline applicability JSON contract

The configured artifact is UTF-8 JSON with exact top-level fields:

```json
{
  "semantics": "positive_ground_atoms_closed_world_v1",
  "entries": [
    {
      "problem_index": 0,
      "problem_name": "p166",
      "state_atoms": [
        {"predicate": "clear", "arguments": ["j1"]}
      ],
      "applicable_actions": [
        {"name": "move", "arguments": ["car0", "j0", "j1"]}
      ]
    }
  ]
}
```

Loader rules:

- require `action_applicability_table_path` to be absolute; reject relative paths so a resolved config saved in a checkpoint remains reproducible from any working directory;
- parse with `json.load(..., object_pairs_hook=...)` (or an equivalent pair-preserving decoder) that rejects duplicate member names at every object nesting level before ordinary mapping conversion; last-value-wins JSON behavior is forbidden for top-level, entry, atom, and action objects;
- require an ordinary JSON object; reject missing/unknown fields at every structured level;
- require exact semantics token `positive_ground_atoms_closed_world_v1`;
- require `entries` to be a list;
- require strict non-boolean integer `problem_index` in range;
- verify `problem_name` exactly matches `parsed_problems[problem_index].name`;
- require atom/action names and all arguments to be strings and arrays to be lists;
- reject duplicate canonical `GroundAtom` values within one `state_atoms` list and duplicate canonical `GroundAction` values within one `applicable_actions` list; although the immutable table ultimately uses set semantics, duplicated producer records are treated as malformed rather than silently normalized;
- canonicalize state keys only through `action_applicability_state_key()`;
- reject duplicate keys rather than silently overwrite;
- construct `ActionApplicabilityTable.from_mapping(...)`, thereby reusing Stage 2D2 immutable/canonical/symbolic validation;
- never execute or import a simulator;
- never infer a complete applicable set from trace positives alone;
- keep missing table states unknown through the existing label mask;
- surface file path and entry index in parse errors, while retaining the Stage 2D2 problem/key/offending-symbol context from table validation.

## Trainer tensor contract

The integrated path reads `batch["action_supervision"]`, a mapping with Stage 2D2 tensors:

- negative actions/masks/category/changed-role data `[B,K,M,...]`;
- negative applicability labels and known-label mask `[B,K,M]`;
- argument targets/masks/candidate masks `[B,K,R]`, `[B,K,R,O]`;
- object presence mask `[B,K,O]`.

It also consumes existing batched true actions and the `GraphJEPATrainingOutput`.

Before use:

- require a mapping, required tensor keys, exact rank, exact leading `[B,K]`, bool/long/float dtypes, same device, and compatible `M/R/O` axes;
- keep `negative_mask` authoritative;
- require label-known mask implies `negative_mask`;
- require candidate mask implies object presence and inactive/padded roles stay masked;
- do not interpret padded symbolic values;
- do not require `action_supervision` when all integrated coefficients are zero.

## Causal negative-action encoding

A negative for transition `(b,k)` must be encoded against the same state/action history as the true action. It is incorrect to encode all negatives as independent one-step actions when the configured `ActionEncoder` has causal context, and incorrect to let earlier negative candidates replace the true prefix.

Implement one shared helper with this behavior:

1. Start from `rollout.observed_states` and the existing true action tensor sequence.
2. For every active negative triple `(b,k,m)`, construct the candidate sequence:
   - true actions at steps `0..k-1`;
   - the sampled negative action at step `k`;
   - no future actions.
3. Select/duplicate the matching temporal latent-state prefix `0..k` for each candidate, preserving object ids and rebasing `object_batch` to the candidate-sequence batch.
4. Group candidates by `k` so each group can be encoded in one module call.
5. Call the ordinary `jepa.action_encoder` on each candidate prefix and retain only the final latent at step `k`.
6. Scatter encoded active candidates into `[B,K,M,D_a]` initialized from the corresponding positive latent. Thus inactive padding receives a finite, semantically ignored latent; no padded symbolic row is encoded.
7. Return both encoded negatives and the original bool `negative_mask`; never allow labels/categories to affect encoding.

Tests must compare this batched helper against an explicit per-candidate reference call for context windows greater than one. Changing only oracle labels must leave encoded candidates and every non-label loss unchanged.

## Latent context for heads

Build a dense source-state object bank from the source slice `observed_states[:, 0:K]`:

- graph latents: `[B,K,D_z]`;
- object latents: `[B,K,O,D_z]`, placed by `object_batch` and problem-local `object_ids`;
- validate `object_mask` matches represented object ids and prevents padded-object use.

Gather role argument object latents through target/negative object indices only where their argument masks are true. Never index `-1` padding before masking/normalization.

## Auxiliary objectives

### Action VICReg

- input: `rollout.action_latents [B,K,D_a]`;
- execute only when coefficient > 0;
- use `ActionVICRegLoss` unchanged;
- add weighted scalar to total;
- emit detached terms for total, std penalty, covariance penalty, and sample count (count represented as a scalar tensor on the loss device);
- gradients update the JEPA action encoder through the existing rollout.

### Contrastive hard-negative loss

- anchor: dedicated `GraphInverseDynamicsModel` over source/next observed state slices, producing `[B,K,D_a]`; construct it with `latent_dim=model.latent_dim`, `action_dim=model.action_dim`, and `hidden_dim=model.action_encoder.hidden_dim`;
- positive: `rollout.action_latents`;
- negatives: causal candidate encodings above;
- select only transition rows with at least one active negative;
- pass the selected bool `negative_mask` to `ActionContrastiveLoss`;
- skip the entire term (return `None`, no metric) if a particular batch has no row with an active negative, rather than calling the primitive with an empty effective batch;
- gradients update the dedicated anchor, action encoder, and state encoder unless an existing documented detach contract requires otherwise;
- emit detached total/similarity/margin/top-1/count diagnostics.

### Argument reconstruction

- positive actions only;
- flatten `[B,K]` into examples;
- head input: positive action latent plus dense problem-local source object bank and `argument_candidate_mask`;
- loss input: logits, `argument_target_indices`, `argument_mask`, and candidate mask;
- execute only when coefficient > 0;
- allow a batch with zero active roles to return the primitive’s graph-connected zero; model construction already rejects globally zero role/object capacity;
- emit detached total, role accuracy, competitive role accuracy, mean target margin, and counts;
- gradients update the argument head, action encoder, and state/object encoder.

### Applicability

Integrated Stage 2 path:

- operate only on transitions with at least one active known negative label;
- create one positive trace example per selected transition with label `1`, plus every active known negative with its supplied `0/1` label;
- do not duplicate a positive once per negative;
- use current source graph latent, corresponding positive/negative action latent, and gathered role-ordered argument object latents/masks;
- skip the entire term if no active negative label is known in the batch; do not train positive-only BCE and do not call `ApplicabilityLoss` with an empty effective batch;
- apply existing `applicability_head_detach` to graph/action/object latent inputs, not labels or head parameters;
- labels affect only this objective and diagnostics.

Legacy Phase 1 fallback:

- when integrated `action_supervision` is absent and the legacy coefficient is active, preserve the existing precomputed keys (`applicability_graph_latents`, etc.) and behavior;
- do not require or synthesize Stage 2 symbolic supervision for existing direct trainer callers.

## Total loss and output contract

Extend `JepaTrainerStepOutput` with optional detached scalar fields:

- `action_vicreg_loss`;
- `action_contrastive_loss`;
- `argument_reconstruction_loss`;
- retain `applicability_loss`.

Total:

```text
trainer_total =
    jepa_loss_weight * jepa_loss
  + goal_loss_weight * goal_loss (when present)
  + action_vicreg_coeff * action_vicreg.total (when present)
  + action_contrastive_coeff * action_contrastive.total (when present)
  + argument_reconstruction_coeff * argument_reconstruction.total (when present)
  + integrated_applicability_loss_weight * integrated_applicability.total (when canonical mode is active)
  + applicability_loss_weight * legacy_applicability.total (when legacy mode is active)
```

The two applicability additions are mutually exclusive by constructor validation.

Requirements:

- exact new term names are:
  - `action_vicreg`, `action_vicreg_std`, `action_vicreg_covariance`, `action_vicreg_num_samples`;
  - `action_contrastive`, `action_contrastive_positive_similarity`, `action_contrastive_hardest_negative_similarity`, `action_contrastive_margin`, `action_contrastive_top1_accuracy`, `action_contrastive_num_examples`, `action_contrastive_num_negatives`;
  - `argument_reconstruction`, `argument_role_accuracy`, `argument_competitive_role_accuracy`, `argument_mean_target_margin`, `argument_num_active_roles`, `argument_num_competitive_roles`;
  - existing applicability names remain `applicability`, `applicability_bce`, `applicability_positive_logit_mean`, `applicability_negative_logit_mean`, and `applicability_positive_negative_margin`;
- integer counts are converted to detached rank-0 tensors on the corresponding loss device (use a floating diagnostic dtype compatible with scalar metric extraction), never left as Python values inside `terms`;
- every diagnostic inserted into `terms` is detached and unweighted;
- `trainer_total` is detached only in output, never before backward;
- train/eval paths use the same composition and skip semantics;
- gradient clipping includes each trainable optional module exactly once;
- modules receive `.train()`/`.eval()` consistently;
- disabled defaults preserve existing output fields/terms and do not touch `action_supervision`.

## Model bundle and optimizer contract

Extend `ModelBundle` with explicit optional fields:

- `action_vicreg_loss_module`;
- `action_contrastive_loss_module`;
- `action_contrastive_anchor`;
- `argument_reconstruction_head`;
- `argument_reconstruction_loss_module`;
- existing applicability fields.

Construction:

- stateless loss helpers exist only when enabled;
- dedicated contrastive anchor exists only when contrastive coefficient > 0;
- argument head/loss exist only when argument coefficient > 0;
- applicability constructs the shared head/loss when either mutually exclusive mode is active, while the two distinct trainer config coefficients preserve explicit routing;
- move every enabled module to the selected device;
- optimizer includes only trainable modules (`jepa`, goal head, contrastive anchor, argument head, applicability head), deduplicated by parameter identity;
- disabled default optimizer parameter ids remain exactly the legacy set.

## Checkpoint contract

Save keys:

- `action_contrastive_anchor_state_dict`;
- `argument_reconstruction_head_state_dict`;
- retain existing `applicability_head_state_dict`;
- explicit `None` when disabled.

Load:

- use `checkpoint.get(...)` for each new key;
- if bundle module is disabled, ignore a stored state;
- if module is enabled and state is present, load strictly;
- if module is enabled and key/state is absent, emit one clear `UserWarning` and leave initialized;
- old checkpoints/configs without Stage 2 fields merge with defaults and remain loadable;
- checkpoint evaluation uses the saved merged config and therefore reconstructs the same enabled modules;
- no state is required for stateless loss helpers.

## RED/GREEN TDD sequence

### RED 1 — disabled trainer compatibility

Add a test with an invalid/non-mapping `action_supervision` sentinel while all new coefficients are zero. Assert train/eval never inspect it, optional output fields are `None`, no new terms appear, and the legacy total/parameters match. Run and observe failure only after introducing the expected new output/API assertion.

GREEN: add config/output fields with zero defaults and disabled short-circuiting.

### RED 2 — VICReg composition

Add an eval-step numerical oracle comparing the trainer VICReg term with direct `ActionVICRegLoss(rollout.action_latents)`, including weighted total and detached terms. Assert finite nonzero gradients in an independent train-step fixture.

GREEN: wire VICReg only.

### RED 3 — exact causal negative encoding

Create `B >= 2`, `K >= 2`, `M >= 2` supervision with mixed active/padded candidates and `action_context_steps > 1`. Compare helper output to explicit one-candidate true-prefix reference encodings. Change an earlier true-prefix action and assert a later candidate encoding can change; change only a future true action and assert an earlier candidate encoding cannot change. Change padded symbolic ids and prove active outputs/loss are identical. Prove labels do not affect encoding.

GREEN: implement validated causal candidate encoding and finite positive-latent padding.

### RED 4 — contrastive composition

Use a dedicated deterministic inverse-dynamics anchor and hand/direct primitive comparison where each subterm is nonzero. Assert weighted total, counts, margins, train/eval behavior, and gradients to anchor/action/state modules. Test mixed rows and all-empty effective batch skip.

GREEN: wire contrastive objective.

### RED 5 — dense object bank and argument reconstruction

Use batched problems with different object counts and deliberately permuted/non-contiguous packed object rows/ids. Assert exact dense object placement by problem-local `object_ids`, object padding isolation, candidate masks, direct head/loss numerical equivalence, weighted total, count metrics, and gradients to argument head plus action/object encoders. Test zero active roles and reject globally unsupported `R==0`/`O==0` enablement.

GREEN: wire argument reconstruction.

### RED 6 — integrated applicability

Construct batches with known/unknown labels and multiple known negatives per transition. Assert exactly one positive per selected transition, every known negative included once, direct BCE equivalence, role-ordered object context, detach behavior, no-known-negative skip, and label isolation from all other tensors/losses. Retain existing precomputed-latent fallback tests.

GREEN: wire integrated applicability.

### RED 7 — config/model/optimizer

Add tests for exact disabled defaults, every enabled module, all four canonical/legacy applicability coefficient combinations, legacy routing despite unrelated `action_supervision`, optimizer parameter identity, finite/range validation, nonzero SIGReg rejection, missing-head/capacity errors, and rejection whenever both applicability coefficients are positive.

GREEN: add builders/config validation/default YAML.

### RED 8 — dataset config and JSON artifact

Add tests for disabled sample parity, enabled nested supervision, strict valid artifact lookup, absolute-path enforcement, duplicate JSON member names at every object level, duplicate state atoms/actions, missing-state unknown labels, duplicate entries, out-of-range/wrong-name/malformed fields/unsupported semantics, and no-simulator imports. Include both goal/no-goal dataset variants.

GREEN: add strict artifact loader and `make_torch_dataset` configuration.

### RED 9 — checkpoint compatibility

Add round-trip tests that mutate enabled anchor/head parameters before loading and recover exact saved state. Assert explicit disabled `None`; old checkpoint warning behavior; stored-state ignored when a module is disabled; existing applicability checkpoint behavior unchanged. Independently cover applicability disabled while only the contrastive anchor is enabled and while only the argument head is enabled, and require `_load_checkpoint_state` to process every optional module without an applicability-driven early return.

GREEN: add save/load keys.

### RED 10 — integrated one-step regression

Run one train and eval step through a real PyG DataLoader with all implemented objectives enabled and a tiny complete offline table. Assert finite total/subterms, optimizer updates optional trainable modules, terms are detached, and oracle labels do not alter sampling/model inputs.

GREEN: fix integration defects only; do not add Phase 2F configuration or campaign behavior.

## Verification commands

Focused core:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-core \
  pytest packages/acs-jepa-core/tests/test_training.py \
         packages/acs-jepa-core/tests/test_action_supervision_dataset.py \
         packages/acs-jepa-core/tests/test_action_vicreg_loss.py \
         packages/acs-jepa-core/tests/test_action_contrastive_loss.py \
         packages/acs-jepa-core/tests/test_argument_reconstruction_head.py \
         packages/acs-jepa-core/tests/test_argument_reconstruction_loss.py \
         packages/acs-jepa-core/tests/test_applicability_head.py \
         packages/acs-jepa-core/tests/test_applicability_loss.py -q
```

Focused CLI:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-cli \
  pytest acs-jepa-cli/tests/test_modeling.py \
         acs-jepa-cli/tests/test_data.py \
         acs-jepa-cli/tests/test_cli.py -q
```

Adjacent/core full:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-core pytest packages/acs-jepa-core/tests -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache \
  uv run --package acs-jepa-cli pytest acs-jepa-cli/tests -q
```

Static/compile:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check \
  packages/acs-jepa-core/src/acs_jepa/training.py \
  packages/acs-jepa-core/tests/test_training.py \
  acs-jepa-cli/src/acs_jepa_cli/modeling.py \
  acs-jepa-cli/src/acs_jepa_cli/data.py \
  acs-jepa-cli/src/acs_jepa_cli/cli.py \
  acs-jepa-cli/tests/test_modeling.py \
  acs-jepa-cli/tests/test_data.py \
  acs-jepa-cli/tests/test_cli.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-core \
  python -m compileall -q packages/acs-jepa-core/src/acs_jepa
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli \
  python -m compileall -q acs-jepa-cli/src/acs_jepa_cli
git diff --check
python /opt/data/skills/software-development/phase-gated-implementation/scripts/static_diff_scan.py
```

Use `git add -N` for every untracked test/plan before the diff-based scan.

## Acceptance criteria

1. Every new objective is disabled by default and default datasets/model bundles/optimizer/checkpoints remain behaviorally compatible.
2. Enabled trainer losses use Stage 2D2 masks and tensors exactly; no resampling, oracle-driven candidate choice, or padding semantics leak into training.
3. Contrastive anchors are transition-derived and independent of candidate action latents.
4. Negative candidates preserve the positive true-action prefix under causal action context; active outputs match explicit references.
5. Only rows with active negatives reach contrastive loss; only batches with active known negative labels reach integrated applicability loss.
6. Argument reconstruction uses role-specific candidate masks and problem-local dense object banks with safe padding.
7. All weighted totals match direct numerical oracles; every diagnostic is detached/unweighted; expected modules receive finite nonzero gradients.
8. CLI configuration exposes the governing Phase 2 fields, rejects invalid/inconsistent combinations, and uses explicit mutually exclusive canonical-integrated versus legacy-precomputed applicability modes independent of unrelated batch keys.
9. Offline JSON ingestion is duplicate-member-safe and strict, canonical, immutable after construction, atom-only semantics-acknowledged, absolute-path reproducible, and simulator-free.
10. Optional trainable modules are in the optimizer exactly when enabled and checkpoint round-trip correctly; old checkpoints warn rather than fail when missing new state.
11. Focused, adjacent, full core, and full CLI suites pass; ruff/compile/diff/static checks pass.
12. Independent implementation review returns PASS on the final diff.
13. A verified SSH-signed commit is created only after PASS.

## Commit gate

Implementation remains blocked until independent plan review returns PASS. If review fails, revise this plan and re-review. After implementation, any code edit following implementation-review PASS invalidates that PASS and requires complete verification plus another independent review. Do not begin Stage 2F until the signed Stage 2E commit is verified.
