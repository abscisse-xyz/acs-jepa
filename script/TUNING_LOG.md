# ACS-JEPA Adaptive Tuning Log

This is the durable, append-only decision record for the CityCar tuning
campaign. Update it before and after every data preparation, training,
evaluation, and planning action. Never rewrite an earlier conclusion silently;
add a dated correction if new evidence changes it.

## Campaign objective

Select the model components and hyperparameters that maximize generalization to
held-out PDDL problems. The primary endpoint is held-out planning success.
Representation losses are screening and diagnostic metrics, not the final
objective.

Tie-breakers, in order:

1. Lower excess action count relative to the dataset solution.
2. Lower planning runtime.
3. Lower variance across seeds and problem strata.
4. Lower training and inference cost.

The final test partition is evaluated once, after model and planner selection.

## Experimental protocol

### Reproducibility and tracking

- Record every train, eval, and plan invocation in MLflow.
- Use MLflow experiment `acs-jepa-adaptive-tuning-v1`.
- Use a persistent tracking URI shared by all campaign runs; record the exact URI
  here before the first run.
- Log the resolved configuration, dataset summary, immutable split manifest,
  checkpoints, metrics, runtime, and planning summaries.
- Tag runs with `stage`, `fidelity`, `hypothesis`, `parent_run`, `split_id`,
  `sample_id`, `seed`, `model_family`, and repository commit identifiers.
- A new configuration must cite its parent run and a concrete observation. It
  must change only the variables needed to test that observation.
- Failed and rejected runs remain in MLflow and receive a status/reason tag.

### Data protocol

- Preserve `/home/awalga/workspace/pddl-dataset` as the read-only source of
  truth. Derived CLI-compatible datasets and samples live outside it.
- Split by problem identity, never by transition or sliding window.
- Freeze one stratified split independently of model seeds. Stratify using at
  least reference plan length and problem size; add topology family if the
  manifest supports it without sparsifying strata excessively.
- Reserve an untouched final test set before examining model results.
- Create deterministic fidelity samples from the development partition:
  - `smoke`: a few easy/medium/hard problems for correctness and overfit tests;
  - `development`: approximately 10-30% of development problems;
  - `full`: all non-test development problems.
- Store problem lists, sampling seed, source fingerprint, and source paths as
  MLflow artifacts.

### Adaptive experiment loop

For every proposed run:

1. State the observation from the parent run.
2. State one hypothesis explaining that observation.
3. Craft the smallest experiment that can reject the hypothesis.
4. Define promotion and rejection thresholds before running it.
5. Run at the lowest informative fidelity.
6. Inspect learning curves, component loss scales, stability, memory, runtime,
   and planning behavior.
7. Promote, reject, repeat with another seed, or diagnose. Record the decision
   and evidence here before crafting the next configuration.

Do not launch a directory-wide blind sweep. Existing tuning overlays are a menu
of supported controls and prior ideas, not a mandatory run list.

### Fidelity and promotion

- Smoke fidelity: one seed; validates data, gradients, checkpoint reload, and
  easy-problem planning. It does not establish a winner.
- Development fidelity: two paired seeds on the same fixed subset and split.
  Promote only material, consistent improvements over the incumbent.
- Confirmation fidelity: three to five seeds on full development data for the
  final two or three candidates.
- Borderline results receive another paired seed rather than an arbitrary win.
- Stop runs early for non-finite loss, latent collapse, persistent divergence,
  or clear domination after a minimum evidence window.

### Planned decision sequence

1. Validate and characterize the corpus; create immutable splits and samples.
2. Verify MLflow persistence and execute a tiny overfit/smoke run.
3. Establish a default baseline with a meaningful training horizon.
4. Ablate predictor and action encoder components against the incumbent.
5. Compare goal heads using downstream planning as the deciding metric.
6. Tune optimization based on observed learning curves and gradient stability.
7. Tune rollout/context based on error by prediction order.
8. Increase or reduce capacity only when underfitting/overfitting evidence calls
   for it.
9. Tune planner compute and decoding for only the trained-model finalists.
10. Confirm finalists, lock the winner, then evaluate the untouched test set.

## Decision records

### 2026-07-17 — Initial repository and hardware inspection

Status: protocol established; no training run started.

Evidence:

- Source dataset found at `/home/awalga/workspace/pddl-dataset`.
- Problem collection:
  `/home/awalga/workspace/pddl-dataset/citycar-topology-200-problem`.
- Simulation collection:
  `/home/awalga/workspace/pddl-dataset/citycar-topology-200-simulation`.
- The problem manifest contains 200 CityCar PDDL instances (`p001`-`p200`)
  with generation parameters including topology family and problem dimensions.
- Simulation data includes solved sequential plans, per-problem artifacts, and
  `simulation.duckdb` (12,595,200 bytes).
- The source layout does not directly match the ACS-JEPA CLI contract of one
  root containing `problem/` and `simulation/simulation.duckdb`. A derived,
  non-destructive compatible view is required.
- GPU: NVIDIA GeForce RTX 2080 Ti; 11,264 MiB total VRAM; 10,828 MiB free at
  inspection; driver 610.43.02. GPU checks require host access in this runtime.
- Existing ACS-JEPA tuning guidance promotes configurations mainly by
  `eval/total_loss`. This campaign instead uses that metric for screening and
  held-out planning success for final selection.
- The current CLI split is random by problem and tied to the training seed.
  Comparisons therefore require a frozen external dataset/split construction or
  a small implementation change before trustworthy tuning.

Reasoning:

- With only 200 problems, transition-level random sampling would leak problem
  structure. Problem-level stratification is necessary.
- The 11 GB device makes the existing batch-size calibration relevant, but no
  batch size should be assumed safe until a representative graph-size sample is
  profiled.
- Model-component losses can have different scales, especially across goal-head
  families. Cross-family selection on raw total loss alone would be misleading.
- Preparing deterministic fidelity subsets reduces turnaround while keeping
  early results comparable and traceable.

Next action:

1. Query the manifest and DuckDB for plan-length, topology, object/graph-size,
   trajectory, transition, and action distributions.
2. Define and materialize deterministic CLI-compatible smoke, development, and
   full dataset views without modifying the source dataset.
3. Configure and verify the persistent MLflow backend.
4. Define the first tiny overfit run and its pass/fail thresholds from those
   statistics.

### 2026-07-17 — Dataset preparation and tracking verification

Status: completed; no model training started.

Source characterization:

- Source fingerprint:
  `dff56cad6f9cc781ff338d77bde85f3f3a5a11011976ca8d76a14712cd5a7deb`.
- 200 generated problems; 198 have solved sequential trajectories.
- The database contains 204 simulation runs, 198 solved trajectories, 6 timeout
  attempts, 5,796 solved transitions, and 7 action schemas.
- Reference plan length: minimum 13, first quartile 21, median 26.5, third
  quartile 38.75, maximum 58, mean 29.27.
- Difficulty/topology generation metadata is available in the source manifest.

Derived data:

- Reproducible preparation utility:
  `script/prepare_tuning_data.py`.
- Derived root: `/home/awalga/workspace/acs-jepa-tuning-data`.
- Split seed: `20260717`.
- Stratification: declared difficulty plus global reference-plan-length tercile.
  Topology is retained in each derived manifest but omitted from the sampling
  key because difficulty x length x topology is too sparse for 12- and
  20-problem samples.
- `smoke`: 12 problems, 12 trajectories, 345 transitions.
- `development`: 48 problems, 48 trajectories, 1,366 transitions.
- `full-dev`: 178 problems, 178 trajectories, 5,198 transitions.
- `final-test`: 20 problems, 20 trajectories, 598 transitions.
- All four derived datasets passed `acs-jepa inspect-data` with zero malformed
  rows. Source files were not modified.
- The final-test set is disjoint from full development. Smoke is nested in
  development, which is nested in full development.

Fixed split behavior:

- Added optional `data.split_seed`; null preserves the original behavior of
  reusing the model seed.
- Campaign base config fixes `data.split_seed: 20260717`, so model seeds do not
  change train/validation membership.
- Smoke train problems: `p010`, `p017`, `p043`, `p056`, `p067`, `p087`, `p129`,
  `p151`, `p165`, `p180`.
- Smoke validation problems: `p166`, `p192`.
- Repeated construction produced an identical split manifest.

MLflow:

- Tracking URI: `sqlite:////home/awalga/workspace/acs-jepa-mlflow.db`.
- Artifact root: `file:///home/awalga/workspace/acs-jepa-mlartifacts`.
- Experiment: `acs-jepa-adaptive-tuning-v1` (experiment id `1`).
- Verification run: `843052eb6cbc44c18b9665dd6eeb6304`, status `FINISHED`.
- The verification run logged the source fingerprint, split seed, and campaign
  manifest artifact.

Environment and repository findings:

- Locked CLI environment installed with `uv 0.11.29`; PyTorch `2.11.0+cu130`
  sees the RTX 2080 Ti only under host execution.
- ACS-JEPA pins pddl-generator commit
  `75c20ebb77b0f815fcaf795ec1711567a31df2be`, but the remote reports that commit
  as unavailable. The initialized dependency is current remote `main` at
  `835beaf7bd5e5017cd92a8aeb76478c9cfa59230`. All campaign runs must retain this
  exact deviation in their provenance.
- Production config loading pointed at a nonexistent `acs-jepa-cli/configs`
  directory. It was corrected to the checked-in `script/configs` directory.
- CLI tests now pass 7 of 8 cases. The remaining test only searches the obsolete
  `acs-jepa-cli/configs/tuning` path and does not exercise runtime behavior.
- `inspect-data` reports twice the actual number of parsed problems because the
  loader appends both parsed PDDL names and filename aliases. Trajectory and
  transition counts are correct. Treat this as a known diagnostic issue.

First smoke-learning run specification:

- Parent: campaign setup run `843052eb6cbc44c18b9665dd6eeb6304`.
- Hypothesis: the default model/data pipeline learns finite, decreasing losses
  on the representative smoke corpus without exhausting 11 GB VRAM.
- Dataset: `/home/awalga/workspace/acs-jepa-tuning-data/smoke`.
- Config stack: `script/configs/adaptive/base.yaml`, then
  `script/configs/adaptive/00_smoke/default_smoke.yaml`.
- Model seed: `0`; split seed remains `20260717`.
- Budget: 3 epochs, batch size 8, rollout 4. This is a pipeline learning check,
  not evidence that the default architecture is optimal.

Predeclared pass criteria:

1. Process exits successfully; every logged loss is finite.
2. Latest and best checkpoints exist and reload for evaluation.
3. Median training total loss over the final 20 steps is at least 20% below the
   median over the first 20 steps. If fewer than 40 steps run, compare equal
   first/last windows covering at least 20% of steps.
4. Validation total loss is finite. With only two validation problems, its
   direction is diagnostic and is not a promotion criterion.
5. MLflow run finishes and contains resolved config, split manifest, dataset
   summary, metrics, checkpoints, and runtime artifacts.
6. No CUDA out-of-memory error. If memory is close to the device limit, the next
   experiment tests batch size only; otherwise batch size remains fixed.

Failure response is diagnostic, not a configuration sweep: data/checkpoint
failures are repaired first; divergence triggers learning-rate/gradient
inspection; weak loss reduction triggers a longer same-configuration run before
any component ablation.

Next action: execute this single smoke-learning run, evaluate its best
checkpoint, inspect its MLflow evidence, and record the decision before crafting
another configuration.

### 2026-07-17 — Smoke default run and planning validity probe

Status: completed; the default smoke learner passes the training-pipeline checks
but fails the first planning-action validity check.

Training run:

- MLflow run: `05d3c30b2db64e16ba47c188e94f469a`, status `FINISHED`.
- Dataset: `/home/awalga/workspace/acs-jepa-tuning-data/smoke`.
- Config stack: `script/configs/adaptive/base.yaml`,
  `script/configs/adaptive/00_smoke/default_smoke.yaml`.
- Output: `/home/awalga/workspace/acs-jepa-runs/smoke/default_seed0`.
- Runtime: 29.8786 seconds, 102 optimizer steps, 813 examples,
  3.414 steps/s, 27.21 examples/s.
- Checkpoints: `checkpoints/best.pt` and `checkpoints/latest.pt` present.
- All logged losses were finite.

Learning evidence:

- Training JEPA loss median first 20 steps: 1.07717.
- Training JEPA loss median final 20 steps: 0.527808.
- Prediction loss median first 20 steps: 0.123743.
- Prediction loss median final 20 steps: 0.002832.
- Regularization loss median first 20 steps: 0.953425.
- Regularization loss median final 20 steps: 0.525018.
- Validation total moved from 0.398 at step 20 to -5.7673 at step 102.
- Validation JEPA moved from 0.908 to 0.52758.
- The total loss is dominated by the GMM goal-head likelihood and becomes
  negative. It is useful within this run, but raw total loss is not a reliable
  cross-component selection metric.

Best-checkpoint evaluation:

- MLflow run: `1d69dace06384a01a430f1e161cf770e`, status `FINISHED`.
- Output: `acs-jepa-runs/smoke/default_seed0/eval_best`.
- Full-smoke evaluation metrics: total -5.7654, goal -6.2930, JEPA 0.527597,
  prediction 0.002743, regularization 0.524854.
- Runtime: 9.397 seconds.
- Interpretation: checkpoint reload and evaluation path are valid; this is not
  a held-out estimate because it evaluates the smoke corpus.

Goal-head diagnostic:

- Best checkpoint GMM log variance reaches the configured clamp minimum.
- Graph logvar: min -8, median -2.8276, max 2.7602.
- Object logvar: min -8, median -2.8947, max 1.8986.
- Interpretation: some mixture components are overconfident or collapsing.
  Planning evidence is needed before deciding whether to change the goal head or
  its variance constraints.

Auxiliary-loss finding:

- The config exposes nonzero `similarity_coeff` and
  `inverse_dynamics_coeff`, but the current `acs-jepa-cli/modeling.py` path does
  not construct or pass the temporal-similarity or inverse-dynamics loss modules
  into `GraphJEPALossModule`.
- Therefore these coefficients are inert in current runs. Existing tuning
  configs that vary them do not actually test those components until the wiring
  is fixed.

Planning diagnostics:

- Default `p166` planning run was interrupted manually after several minutes.
  It produced only `planning/p166/config.yaml` and no `plan_summary.json`.
- Stack trace showed time spent in nested categorical cross-entropy decoding
  while applying the rejection penalty after an invalid decoded action:
  `PlannerAgent._rejection_penalty -> action_decoder.decode ->
  cross_entropy_optimize`.
- A bounded diagnostic config was added:
  `script/configs/adaptive/00_smoke/planning_probe_fast.yaml`.
- The probe keeps the trained model fixed and reduces only planning budget:
  horizon 2, 32 MPPI samples, 10 MPPI iterations, max 16 actions,
  one decode attempt, decoder 64 samples and 8 iterations. One decode attempt
  avoids the rejection-penalty loop and measures first-action validity directly.

Fast planning probe results:

- `p166`: MLflow run `980ffe9c06404f5fbc41daafc0648ddf`, status `FINISHED`;
  success 0, attempts 1, total actions 0, runtime 2.833699 seconds,
  failure reason `decode_invalid`.
- `p192`: MLflow run `4b33d0f42f294417a4b24a6efcfed24d`, status `FINISHED`;
  success 0, attempts 1, total actions 0, runtime 2.798036 seconds,
  failure reason `decode_invalid`.

Decision:

- The smoke training pipeline is viable, but this checkpoint is not yet a
  planner baseline because it cannot decode a valid first action on either fixed
  smoke validation problem under a bounded diagnostic budget.
- The next experiment should not scale to the development set yet.
- The next action is to test action validity more directly: run a small
  teacher-forced/ground-truth action decoding diagnostic on smoke validation, or
  add the missing auxiliary-loss wiring if the diagnostic confirms the action
  latent space is not invertible enough for planning.

### 2026-07-17 — Ground-truth action decoder diagnostic

Status: completed; the action latent can recover action schemas more reliably
than action arguments, which explains the planner's invalid first actions.

Diagnostic script:

- Added `script/diagnose_action_decoder.py`.
- It loads a checkpoint, recreates the configured train/validation/test split,
  encodes each ground-truth action in its true source state, decodes the action
  latent back to a ground action, and tests whether that decoded action is
  applicable after replaying the reference prefix in the simulator.
- Exact decoding was attempted first, but interrupted because exhaustive search
  spent more than 90 seconds on only a few validation transitions. Stack trace
  showed repeated action-encoder object lookup inside `_gather_by_object_id`.
- The diagnostic was extended with bounded CEM decoder flags and rerun with
  64 samples and 8 iterations, matching the fast planning probe budget.

Sanity run:

- MLflow run: `05abbd534dbe4489a64956e1cfb69345`, status `FINISHED`.
- Scope: first 5 smoke-validation transitions.
- Exact-match rate: 0.6.
- Applicable-decoded-action rate: 0.6.
- Runtime: 6.6596 seconds.

Full smoke-validation run:

- MLflow run: `7af3e78165d84a55afce77ffb27b141c`, status `FINISHED`.
- Scope: all 44 transitions from validation problems `p166` and `p192`.
- Exact-match rate: 0.363636.
- Applicable-decoded-action rate: 0.363636.
- Runtime: 29.8180 seconds.
- Per-problem exact-match rates: `p166` 0.391304, `p192` 0.333333.
- True action-name counts and decoded action-name counts are identical across
  the full diagnostic: build_straight_oneway 4, build_diagonal_oneway 7,
  car_start 6, move_car_in_road 10, move_car_out_road 10, car_arrived 6,
  destroy_road 1.
- The failures are mostly wrong object arguments for the right action schema,
  and those wrong arguments are simulator-invalid.

Decision:

- The next model experiment should focus on action-argument identifiability,
  not on scaling data, changing the goal head, or increasing planner budget.
- The existing inverse-dynamics and temporal-similarity coefficients are
  configured but inert. Before testing coefficient values, wire the missing
  auxiliary loss modules so the intended model components are actually active.

### 2026-07-17 — Inverse-dynamics component test

Status: completed; inverse dynamics is now wired and trainable, but this
configuration is not promoted.

Implementation:

- `acs-jepa-cli/src/acs_jepa_cli/modeling.py` now constructs
  `GraphEncodedActionInverseDynamicsLoss` and `GraphTemporalSimilarityLoss`
  when explicitly enabled.
- Added `model.loss.enable_auxiliary_terms`, default `false`, to preserve
  strict loading of checkpoints trained before auxiliary module parameters
  existed.
- Added `script/configs/adaptive/01_action_decode/inverse_dynamics_smoke.yaml`.
  It enables auxiliary terms, sets `inverse_dynamics_coeff: 0.1`, and sets
  `similarity_coeff: 0.0` to isolate the action-identifiability component.
- Focused check passed: old baseline checkpoint reloads, and the new config
  emits an `inverse_dynamics` loss term.
- Test slice result with coverage addopts disabled: 13 passed, 1 failed. The
  failure is the known stale test that searches the obsolete
  `acs-jepa-cli/configs/tuning` path.

Training:

- MLflow run: `eb662c306d7944b8a3c01cf0a4131f36`, status `FINISHED`.
- Output: `/home/awalga/workspace/acs-jepa-runs/smoke/inverse_dynamics_seed0`.
- Runtime: 30.3190 seconds, 102 steps, 813 examples.
- Median training JEPA loss first 20 steps: 1.082256.
- Median training JEPA loss final 20 steps: 0.528290.
- Median training inverse-dynamics loss first 20 steps: 0.026512.
- Median training inverse-dynamics loss final 20 steps: 0.002271.
- The auxiliary term trains down cleanly, but the main JEPA metrics are nearly
  unchanged from the baseline.

Best-checkpoint evaluation:

- MLflow run: `ea58052c414249f88d49ac7ca38edcf1`, status `FINISHED`.
- Full-smoke evaluation metrics: total -5.511289, goal -6.039415,
  JEPA 0.528126, prediction 0.002821, regularization 0.525082,
  inverse dynamics 0.002228.
- Runtime: 9.2249 seconds.

Action-decoder diagnostic:

- MLflow run: `3af6fdc4c19947aa8b3012458b69b6bd`, status `FINISHED`.
- Same validation split and bounded CEM decoder budget as baseline:
  44 transitions, 64 samples, 8 iterations.
- Exact-match rate: 0.318182.
- Applicable-decoded-action rate: 0.318182.
- Baseline under the same diagnostic was 0.363636, so this component decreased
  the measured action-decoding quality.

Fast planning probe:

- `p166`: MLflow run `8c9e1ce863524762a3f796a4162c7fbb`, status `FINISHED`;
  success 0, attempts 1, total actions 0, runtime 2.807950 seconds,
  failure reason `decode_invalid`.
- `p192`: MLflow run `0144c284f64748dd993fecd32a5da4ff`, status `FINISHED`;
  success 0, attempts 2, total actions 1, runtime 3.800966 seconds,
  failure reason `decode_invalid`.

Decision:

- Do not treat this as evidence against inverse dynamics. The component is
  active and trainable, and the inverse-dynamics objective improved strongly.
- The short smoke run did not improve the bounded CEM decoder or planner probes,
  so the unresolved problem is the action latent geometry and decoder energy
  landscape, not simply whether the auxiliary objective can learn.
- The next work should investigate latent-action structure before more tuning.

### 2026-07-17 — RNN action-argument encoder and structured planner probes

Status: completed; both probes are rejected.

RNN action encoder:

- Added `script/configs/adaptive/01_action_decode/action_encoder_rnn_smoke.yaml`.
- Hypothesis: using recurrent argument composition would improve ordered
  object-argument decoding versus the pooled action encoder.
- MLflow training run: `5a3757c29a814a5bb224dc6897134c10`, status `FINISHED`.
- MLflow eval run: `64db7449c71a446990c25ce550f56030`, status `FINISHED`.
- Training ran 102 steps.
- Median training JEPA loss first 20 steps: 1.082027.
- Median training JEPA loss final 20 steps: 0.504683.
- Full-smoke eval JEPA loss: 0.504693, lower than baseline 0.527597.
- Full-smoke eval prediction loss: 0.004237, worse than baseline 0.002743.
- Full-smoke eval regularization: 0.500456, lower than baseline 0.524854.

RNN action-decoder diagnostic:

- MLflow run: `43884f5f8c8b4cec90c45ae85e5506a3`, status `FINISHED`.
- Same validation split and bounded CEM decoder budget as baseline:
  44 transitions, 64 samples, 8 iterations.
- Exact-match rate: 0.227273.
- Applicable-decoded-action rate: 0.272727.
- This is worse than baseline exact/applicable 0.363636 and worse than the
  inverse-dynamics run on exact/applicable 0.318182.

RNN fast planning probe:

- `p166`: MLflow run `2a3902b774f94025a5d52f9ad900e427`, status `FINISHED`;
  success 0, attempts 1, total actions 0, failure reason `decode_invalid`.
- `p192`: MLflow run `eb09539a08f843858345eab50a939243`, status `FINISHED`;
  success 0, attempts 1, total actions 0, failure reason `decode_invalid`.

Decision on RNN action encoder:

- Do not promote `model.action_encoder.kind: rnn`. It improves the aggregate
  JEPA scalar mostly by reducing regularization, but it worsens the
  action-decoding diagnostic and still fails planning.

Structured CE planner probe:

- Added `script/configs/adaptive/00_smoke/planning_probe_structured_fast.yaml`.
- Hypothesis: direct typed grounded-action CE could bypass the continuous
  latent-action decoder that produced invalid object arguments.
- Baseline checkpoint, `p166`: MLflow run
  `206357581552405592d15054b3fbd91d`, status `FINISHED`; success 0,
  attempts 1, total actions 0, runtime 3.730396 seconds, failure reason
  `decode_invalid`.
- Baseline checkpoint, `p192`: MLflow run
  `92647f7f821249698d8b8adf86bca6ab`, status `FINISHED`; success 0,
  attempts 1, total actions 0, runtime 3.517669 seconds, failure reason
  `decode_invalid`.

Structured planner finding:

- `structured_ce` removes latent-action decoding, but still samples
  type-correct actions without a learned state-applicability signal. The
  simulator exposes `SimulatorEngine.applicable_actions()`, but that method is
  a small-problem diagnostic oracle rather than a scalable planner component:
  on large problems, grounding/enumerating applicable actions can be expensive,
  and if full grounding is available a classical search baseline may already be
  competitive.

Current promotion state:

- Tuning/training is paused. The current blocker is understanding why the
  learned action encoder output does not provide enough structure or energy
  margin for CEM action decoding.
- Best operational checkpoint for reproducing the issue remains the baseline
  pooled action encoder smoke run. The inverse-dynamics checkpoint should be
  retained as a comparison point for latent-geometry diagnostics, not discarded.
- No configuration should be scaled to the development set until the action
  latent geometry and CEM decoder failure mode are understood.

### 2026-07-17 — Pause and investigation handoff

Status: completed.

- Created `script/ACTION_LATENT_HANDOFF.md`.
- The handoff includes fixed data paths, exact reproduction commands, observed
  baseline and inverse-dynamics evidence, why `engine.applicable_actions()` is
  not a scalable solution, and concrete investigation directions.
- Recommended next artifact:
  `script/diagnose_action_latent_geometry.py`.
