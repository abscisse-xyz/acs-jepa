# Action latent updated research specification

Date: 2026-07-22
Status: proposed replacement for the Phase 2 prescription after Phase 2G FAIL
Scope: research/specification only; no implementation or tuning is authorized by this document

## 1. Purpose

Phase 2 showed that the initial action-identifiability prescription was implementable and active, but not effective. This updated specification therefore changes the research posture:

- do not add another auxiliary head or coefficient sweep on top of the same unstable foundation;
- first decide whether the current ACS-JEPA action-latent formulation is recoverable at all;
- if it is recoverable, make only the smallest causal intervention needed to test that claim;
- if it is not recoverable, pivot to a different planning/action-grounding formulation rather than hiding the failure behind more complexity.

The governing question is no longer “which auxiliary loss should be added next?” It is:

> Can a continuous action latent learned primarily for JEPA transition prediction also be a reliable interface for grounded, applicable CityCar actions, or should action choice be represented and optimized as an explicitly discrete/state-conditioned candidate-ranking problem?

## 2. Evidence forcing this update

Phase 2F passed only the component/activity gate: all four auxiliary terms were finite, had positive counts, checkpointed/restored, and showed small loss decreases. Phase 2G then failed the efficacy gates on the fixed 44-transition `p166`/`p192` validation slice.

Key Phase 2G results:

- global effective rank regressed from `2.127568` to `1.761690`, below the required `>= 4`;
- global minimum action-latent std remained below the floor: `0.013615 < 0.02`;
- within-schema variance fraction collapsed further from `1.40715e-5` to `3.69487e-6`;
- scale-normalized nearest-wrong distances improved, so the auxiliary terms did move local geometry, but not in the way needed for action decoding;
- frozen role/object probes improved modestly, but applicability remained weak: AUROC `0.638725`, AP `0.182121`, F1 `0.0`;
- the restored trained applicability head was worse than the fitted diagnostic expectation: AUROC `0.583824`, AP `0.147178`, F1 `0.214286`;
- the restored argument head had above-chance top-1 accuracy (`0.315068`) but negative median target-minus-best-wrong margin (`-0.00576854`).

Interpretation:

- The first specification correctly identified schema-vs-argument aliasing and applicability sparsity as the core symptoms.
- Its Phase 2 prescription did not repair the governing representation failure.
- The local improvement in same-schema distance is insufficient because action-latent rank and within-schema variance worsened, and applicability is still not separable.
- The next step must be falsification-oriented, not an expanded multi-head training recipe.

## 3. Review of the initial specification

### 3.1 What remains valid

The following parts of `ACTION_LATENT_SOLUTION_SPEC.md` remain supported by evidence:

1. The production failure is a binding/applicability failure, not primarily schema classification.
2. Type-valid object sampling is too weak for CityCar because applicable actions are a tiny subset of the typed product space.
3. Whole-action latent distance is a brittle decoder score when same-schema substitutions are near-tied.
4. Continuous planner latents can drift away from the encoded grounded-action manifold.
5. `SimulatorEngine.applicable_actions()` should remain an offline oracle/label source, not the production planner.
6. Exact/debug match to one trace is a diagnostic label, not a production planning metric.

### 3.2 What Phase 2 invalidated or weakened

The initial spec treated explicit action identifiability auxiliaries as the natural next implementation phase: action VICReg, contrastive hard negatives, applicability head, and argument reconstruction head. Phase 2G weakens that prescription in four ways.

1. Global action VICReg did not prevent action-latent collapse.
   - Effective rank regressed.
   - Within-schema variance fraction regressed sharply.
   - This is evidence that global action-latent spread is either too weak, misaligned, or satisfied in dimensions that do not encode grounded arguments.

2. The contrastive/argument auxiliaries were not sufficient to create positive decision margins.
   - Frozen role/object probes improved, but the trained argument head still had negative median target-vs-best-wrong margin.
   - Better role information exists locally, but is not strong or organized enough for a reliable decoder.

3. Applicability was not learned as a separable relation.
   - Both fitted and restored applicability heads failed AUROC/AP/F1/margin gates.
   - This suggests one of: the latent state lacks precondition-relevant facts, negative labels are not aligned with the needed decision boundary, the current head/feature interface is inadequate, or applicability is too sparse for the current training protocol.

4. “Loss decreases” were not meaningful efficacy evidence.
   - All auxiliary losses decreased only slightly over the fixed smoke run.
   - The decreases showed wiring/activity, not causal correction of action decoding.

### 3.3 What the initial spec left under-tested

The initial spec did not force the following decisive questions before implementation:

1. Is applicability linearly/nonlinearly recoverable from the current latent state and encoded action at all?
2. Is action identity recoverable if the action latent is compared only within a fixed schema, after removing schema-centroid information?
3. Are true actions distinguishable from hard negatives using raw symbolic facts but not using JEPA latents? If yes, the latent state is discarding necessary precondition information.
4. Are many hard negatives transition-equivalent under the JEPA target? If yes, transition prediction may actively permit or reward aliasing.
5. Does continuous latent action optimization provide any advantage over selecting/ranking discrete grounded candidates with the learned model as a scorer?

Those questions become Phase 0 in the updated specification.

## 4. Updated root-cause hypotheses

Treat these as hypotheses to falsify, not assumptions to code around.

### H1: Schema-dominated action latent collapse

The action latent primarily encodes operator/schema identity. Argument identity survives only weakly. Global regularizers and small hard-negative losses can move distances without creating a stable schema-conditioned representation.

Prediction:

- subtracting per-schema centroids will leave very low effective rank and poor role/object recoverability;
- same-schema margins will remain fragile even if global nearest-wrong distances improve.

### H2: State latent loses precondition-relevant facts

Applicability may not be separable because `graph_latent`/`object_latents` discard facts needed to decide whether an action is currently applicable.

Prediction:

- a probe over raw symbolic/state features can classify applicability far better than a probe over frozen JEPA state/action latents;
- adding more action-latent losses will not fix applicability unless state encoding or the scoring interface changes.

### H3: Transition-predictive equivalence conflicts with grounded-action invertibility

Some wrong-argument substitutions may be equivalent or nearly equivalent under the JEPA transition target, especially if the target latent is insensitive to the exact object binding. In that case, forcing all trace actions apart may conflict with the learned dynamics abstraction.

Prediction:

- hard negatives with tiny action-latent margins also have small predicted/observed next-state differences;
- increasing separation may harm JEPA transition loss or separate actions that are valid alternatives.

### H4: Continuous action-latent planning is the wrong interface for sparse symbolic actions

The off-manifold planner-latent failure may not be a secondary bug; it may indicate a mismatch between continuous latent optimization and sparse symbolic action applicability.

Prediction:

- even if teacher-forced candidate ranking improves, optimized continuous latents still decode to invalid candidates unless constrained so strongly that planning becomes discrete candidate search in disguise.

## 5. Updated research program

The next work should be organized as decision gates. Each gate can stop the current line of work.

### Phase 0: Falsification diagnostics before new training

Purpose: determine whether the current representation contains the information required for grounded action decoding.

No model training, no planner/CEM tuning, and no new auxiliary objectives are allowed in this phase.

Required diagnostics:

1. Schema-residual latent statistics
   - Compute per-schema centroids of encoded candidate action latents.
   - Analyze residuals `z_action - centroid(schema)`.
   - Report residual effective rank, residual per-dimension std, same-schema nearest-wrong margins, and within-schema variance fraction.
   - Compare baseline and Phase 2 checkpoints.

2. Applicability recoverability ladder
   - Train deterministic frozen probes with identical splits on increasing feature sets:
     a. action latent only;
     b. graph latent + action latent;
     c. object latents for selected arguments + graph latent + action latent;
     d. raw symbolic/fact-derived features + grounded action tuple;
     e. hybrid raw-symbolic + latent features.
   - Use the same positive/negative candidate manifest for all feature sets.
   - Report AUROC, AP, calibrated F1, role-swap margin, one-argument-substitution margin, and per-schema breakdown.

3. Transition-equivalence check for hard negatives
   - For true action and nearest wrong same-schema candidates, compare predicted next latent and observed next latent deltas.
   - Identify cases where wrong candidates are nearly transition-equivalent under the JEPA target.
   - Report whether invalid hard negatives are actually distinguishable by the current transition objective.

4. Candidate-ranking oracle baselines
   - On the fixed validation slice, evaluate candidate scoring baselines that do not use continuous planned latents:
     a. latent nearest-neighbor score only;
     b. fitted applicability score only;
     c. fitted role/object score only;
     d. simple symbolic/fact-derived applicability features;
     e. hybrid score.
   - This is still diagnostic; it must not call `applicable_actions()` as a production generator.

Acceptance to continue with current latent-action line:

- schema-residual effective rank and within-schema variance are nontrivial enough to support argument identity, or a clear representation-level bottleneck is localized;
- latent or hybrid features can reach applicability AUROC `>= 0.80`, AP `>= 0.35`, and positive role-swap/one-substitution median margins on the fixed slice;
- hard negatives are not mostly transition-equivalent under the JEPA target;
- a candidate ranker can put applicable actions above hard invalid negatives without using exhaustive applicable-action enumeration at inference.

Stop / pivot criteria:

- raw symbolic features classify applicability well but latent features do not: pivot to a state/action scoring interface that preserves explicit facts or modifies the state encoder; do not add action-latent losses.
- even raw symbolic features perform poorly under the sampled manifest: the label/negative construction is flawed; fix the data/label problem first.
- hard negatives are transition-equivalent under the JEPA target: do not force arbitrary separation inside the transition latent; introduce a separate action-grounding objective or abandon invertible continuous action latents.
- candidate ranking works but continuous latent planning remains off-manifold: pivot planning to discrete candidate search/ranking or strongly anchored candidate perturbations.

### Phase 1: Minimal causal intervention, only if Phase 0 passes

Purpose: test whether the current action encoder can be repaired by a narrowly targeted schema-conditioned anti-alias objective.

This phase is not a broad training run and not a decoder/planner change.

Allowed model change:

- Add a candidate-level, schema-conditioned ranking objective over trace actions and deterministic hard negatives.
- The objective must operate within the same source state and same schema wherever possible.
- It should compare true/applicable action candidates against one-argument substitutions, role swaps, and random same-schema negatives.
- The loss should be normalized within schema/residual space so it cannot pass merely by increasing global schema separation or latent norm.

Recommended first objective:

```text
z_pos = q(state, true_action)
z_neg_i = q(state, hard_negative_i)
c_s = stopgrad(mean_schema_centroid(schema(true_action)))
r_pos = normalize(z_pos - c_s)
r_neg_i = normalize(z_neg_i - c_s)

L_schema_rank = CE([sim(anchor, r_pos), sim(anchor, r_neg_1), ...], target=pos)
```

Anchor options, in order of preference:

1. a stop-gradient transition-delta/inverse-dynamics anchor if Phase 0 shows it contains useful argument information;
2. a learned small anchor trained only for this ranking task;
3. direct pairwise margin between `r_pos` and `r_neg_i` only if no reliable anchor exists.

Do not change:

- planner behavior;
- CEM sampling;
- applicability or argument head architecture;
- broad JEPA architecture;
- loss coefficients beyond the one preregistered intervention.

Acceptance:

- residual/schema-conditioned effective rank improves and does not regress below baseline;
- within-schema variance fraction improves by at least 10x over Phase 2 or reaches a preregistered floor;
- same-schema target-minus-best-wrong median margin is positive in raw and normalized space;
- fitted applicability does not regress;
- JEPA transition loss does not materially degrade on held-out smoke;
- all assessment artifacts are byte-identified as in Phase 2G.

Failure interpretation:

- If local margins improve but applicability remains weak, the blocker is state/action applicability representation, not action-latent anti-aliasing.
- If rank/std remain collapsed, the current action encoder/training interface is not recoverable by small regularizers.
- If JEPA loss degrades materially, grounded invertibility conflicts with the current transition abstraction.

### Phase 2: Choose one branch, not all branches

Phase 2 is deliberately a branch point. Select exactly one path based on Phase 0/1 evidence.

#### Branch A: Explicit state-action applicability/ranking model

Choose this only if raw/hybrid features show applicability is recoverable but latent-only features are weak.

Specification:

- Build a state-conditioned candidate scorer that consumes explicit symbolic/fact-derived features plus selected latent features.
- Score complete grounded actions, not independent role marginals only.
- Use offline simulator labels for training/evaluation but not as a production generator.
- Decode by ranking sampled/type-generated candidate actions under this scorer and the JEPA rollout score.

This branch admits that pure continuous action-latent decoding is not sufficient.

#### Branch B: Discrete candidate planning with JEPA as a value/rollout scorer

Choose this if candidate ranking works but continuous planned latents remain off-manifold.

Specification:

- Stop optimizing arbitrary continuous action vectors for the first action.
- Generate a bounded candidate set using type constraints plus learned/symbolic filters.
- Encode each candidate action and use JEPA rollout/goal energy to rank sequences.
- Treat this as learned heuristic search over candidates, not classical `applicable_actions()` enumeration.

This branch changes direction away from continuous latent-action CEM.

#### Branch C: State encoder redesign

Choose this if applicability is recoverable from raw facts but not from graph/object latents even with selected argument object latents.

Specification:

- Revisit what `GraphEncoder`/`StateEncoderF` preserve about preconditions, topology, and object-role relations.
- Add diagnostics proving precondition predicates/topological relations are decodable from object/state latents before touching action losses.
- Only after those diagnostics pass should action decoding be retried.

This branch treats the action failure as a state-representation bottleneck.

#### Branch D: Abandon exact grounded-action recovery from JEPA action latents

Choose this if hard negatives are transition-equivalent under the JEPA target or if all candidate-level interventions fail.

Specification:

- Treat JEPA action latents as abstract transition controls, not invertible grounded actions.
- Use ACS-JEPA for heuristic evaluation, subgoal scoring, or rollout priors inside a separate symbolic/candidate planner.
- Stop optimizing for direct latent-to-PDDL action recovery unless the model objective is reformulated.

This is a valid research outcome, not a failure to add enough machinery.

## 6. Non-goals

- No broad hyperparameter tuning while Phase 0/1 gates are unresolved.
- No larger CEM budget as a claimed fix.
- No new decoder stack combining all possible scores before knowing which score contains information.
- No production use of `SimulatorEngine.applicable_actions()` as an action generator.
- No acceptance based on auxiliary-loss decrease alone.
- No acceptance based on exact match to a single reference trace alone.
- No unbounded architectural complexity to preserve the original direction.

## 7. Revised decision criteria for resuming tuning

Broad tuning may resume only if one of these is demonstrated with fixed, reproducible evidence:

1. Current latent-action path is recoverable:
   - schema-residual rank/variance and same-schema margins pass;
   - applicability is separable from latent/hybrid features;
   - candidate ranking prefers applicable hard positives over invalid near-misses;
   - first-action planning can be constrained without collapsing into invalid off-manifold latents.

2. A branch replacement is selected and validated:
   - explicit state-action scorer, discrete candidate planner, or state-encoder redesign has a small deterministic acceptance result superior to baseline and Phase 2;
   - the acceptance result includes applicability, margins, and no JEPA-loss degradation that invalidates the world-model objective.

3. The research conclusion is negative:
   - evidence shows that continuous JEPA action latents are not a viable direct interface for grounded CityCar actions under the current objective;
   - the project pivots to using JEPA as a heuristic/model component rather than a direct action decoder.

## 8. Immediate next artifact

The next artifact should be a Phase 0 plan, not code. It should specify exact commands and outputs for:

- schema-residual latent statistics for baseline and Phase 2 checkpoints;
- applicability recoverability ladder using one fixed candidate manifest;
- transition-equivalence analysis for hard negatives;
- candidate-ranking diagnostic baselines.

The plan must include PASS/FAIL review before implementation. If Phase 0 fails the continuation criteria, do not implement Phase 1; write a branch-selection note instead.
