# Action Latent Failure Mode and Solution Specification

Date: 2026-07-18

## Purpose

This note turns the action-latent failure investigation into an architecture
review and implementation specification. The immediate goal is not another blind
hyperparameter sweep. The goal is to change the model and decoder so that a
planned latent action maps to a grounded PDDL action that is both useful for
latent rollout and applicable in the current simulator state.

## Failure mode summary

The trained ACS-JEPA model learns enough about CityCar action schemas to recover
which operator should be used, but it does not learn a sufficiently separated or
applicability-aware representation of the operator arguments.

The observed production failure is:

1. The planner optimizes a continuous latent action sequence toward low terminal
   goal energy.
2. The first latent action is decoded back to a grounded symbolic action.
3. The decoder usually selects the right action schema but wrong object
   arguments.
4. The selected action is type-correct but simulator-invalid, so planning fails
   before meaningful multi-step behavior can be evaluated.

This is a binding/applicability failure, not primarily a schema-classification
failure.

## Architecture reviewed

The current pipeline is:

```text
PDDL state facts
  -> GraphEncoder
  -> StateEncoderF
  -> JEPALatentState(graph_latent, object_latents)

grounded action + source JEPALatentState
  -> ActionEncoder(LatentActionEncoder)
  -> action_latent

JEPALatentState + action_latent
  -> latent predictor G
  -> predicted next JEPALatentState

planner continuous latent sequence
  -> ActionDecoder
  -> grounded PDDL action
  -> simulator apply_action(...)
```

Key implementation points:

- `GraphEncoder` embeds object ids, object types, predicate ids, predicate
  arity, and role-labeled atom/object edges. It returns both graph-level and
  object-level embeddings.
- `StateEncoderF` projects graph/object embeddings into JEPA graph/object
  latents, optionally with causal GRUs over trajectory windows.
- `LatentActionEncoder` represents an action as an action-schema embedding plus
  contextual source-state object latents gathered by object id. Argument
  composition is currently either:
  - `pooled`: role-shifted masked mean pooling; or
  - `rnn`: schema-ordered GRU over `[action, arg_0, arg_1, ...]`.
- `ActionDecoder` scores candidate grounded actions by re-encoding them in the
  same source latent state and comparing candidate latents to the target latent
  with L2/cosine. Its CEM sampler only enforces type compatibility, not state
  applicability.
- `LatentGMMMPPIPlanner` seeds continuous latent planning from encoded sampled
  type-valid actions, but then optimizes continuous GMM components that can
  drift off the encoded grounded-action manifold.
- The current inverse-dynamics auxiliary loss regresses from consecutive graph
  latents to the existing encoded action latent. It does not directly classify
  schema, reconstruct arguments, or classify applicability.

## Root-cause mapping

### RC1: Schema identity dominates object-role binding

Evidence:

- Bounded CEM diagnostics recover the same schema for all validation transitions,
  while exact/applicable action recovery remains low.
- Same-schema wrong-argument substitutions are nearest neighbors of the true
  action with tiny L2 margins around `1e-05`.

Architectural cause:

- The action schema has a dedicated embedding table, while argument identity is
  represented indirectly through contextual object latents.
- The default pooled action encoder compresses all arguments into one pooled
  vector. Even with role embeddings, this makes one-argument substitutions and
  role-specific binding errors cheap to alias.
- The transition loss only asks the action latent to help predict the next latent
  state. It does not require invertible grounded-action identity or separation
  from invalid near-miss arguments.

### RC2: Type-valid decoding is far too weak for CityCar

Evidence:

- Type-valid grounded actions per smoke validation state average about 21,960.
- Simulator-applicable actions average about 108.9, only about 0.5% of the
  type-valid set.
- Move schemas are orders of magnitude sparser than the full typed space.

Architectural cause:

- `ActionSamplingFamily` samples only from type-compatible object domains.
- It has no access to precondition satisfaction, static topology constraints, or
  learned applicability scores.
- CEM therefore searches a large product space where almost every tuple is
  invalid despite being type-correct.

### RC3: CEM confidently collapses to aliased invalid basins

Evidence:

- CEM action entropy and argument entropies collapse near zero even when the
  decoded tuple is simulator-invalid.
- Increasing inverse-dynamics signal slightly improves latent regression but not
  bounded decoding/applicability at the tested smoke budget.

Architectural cause:

- The decoder score is a single whole-action latent distance. If same-schema
  argument substitutions are near-ties, CEM has no robust signal to distinguish
  them.
- The sampling family factorizes argument-role distributions conditioned only on
  action id. It cannot model constraints like “junction A must be adjacent to
  junction B and road R must connect both” except indirectly through the weak
  latent score.

### RC4: Continuous planner latents drift off the grounded-action manifold

Evidence:

- First planned latent diagnostics produce nearest encoded grounded-action scores
  around `-25`, far from near-zero teacher-forced action scores.
- Top nearest grounded actions to planned latents are often invalid; applicable
  actions appear only as near-tied later neighbors.

Architectural cause:

- Continuous MPPI/GMM-MPPI optimizes in unconstrained `R^D_a` after seeding.
- The latent predictor and goal energy can reward latent directions that do not
  correspond to any grounded action.
- Decoding then becomes a projection from an off-manifold vector into a sparse
  symbolic action space.

## Related research context

The next step should include a literature review before implementation. The
failure has close analogues in latent-action world models, learned action
representations for large discrete action sets, non-contrastive collapse
regularization, and neuro-symbolic planning constraints.

### Latent action models and grounding

Relevant papers:

- Chandak et al., `Learning Action Representations for Reinforcement Learning`,
  arXiv:1902.00183. This paper frames the core action-representation problem:
  large finite action sets need a learned low-dimensional action space, but that
  representation must preserve similarity of action outcomes and be usable by a
  policy-to-action decoder. ACS-JEPA has the same issue in symbolic form: the
  latent can recover coarse schema/outcome similarity while failing to preserve
  exact grounded argument identity.
- Liang et al., `CLAM: Continuous Latent Action Models for Robot Learning from
  Unlabeled Demonstrations`, arXiv:2505.04999. CLAM explicitly argues that
  continuous latent actions need a jointly trained action decoder so latent
  labels can be grounded to real actions. This is directly relevant: ACS-JEPA
  currently optimizes continuous planned latents and then projects them through a
  weak decoder. The decoder-grounding path should be a first-class training
  target, not only a post-hoc nearest-neighbor/CEM procedure.
- Klepach et al., `Object-Centric Latent Action Learning`, arXiv:2502.09680.
  This paper motivates object-centric latent actions when distractors or
  irrelevant dynamics can corrupt proxy action labels. For ACS-JEPA, the analogue
  is that object identity, role, and topology must remain explicit enough for
  action binding; a graph/object latent alone is not guaranteed to make argument
  substitution separable.
- Zhang et al., `DiLA: Disentangled Latent Action World Models`,
  arXiv:2605.15725. DiLA identifies a trade-off between action abstraction and
  generation/prediction fidelity. ACS-JEPA sees a symbolic version: abstract
  schema-level latents are useful for transition prediction, but too much
  abstraction destroys grounded-action invertibility and applicability.

Implication for ACS-JEPA:

- Treat action grounding as part of the representation-learning problem. A good
  latent transition model is insufficient if the action latent is not decodable
  into a legal action under current-state preconditions.
- Separate evaluation of `transition-predictive usefulness` from `grounded-action
  recoverability`. The current failure is exactly a mismatch between these two
  objectives.

### JEPA and anti-collapse regularization

Relevant papers:

- Assran et al., `Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture`, arXiv:2301.08243. I-JEPA shows that joint embedding
  prediction can learn useful semantic latents without reconstruction, but the
  design of the target/context prediction task strongly shapes what information
  survives in the embedding.
- Bardes et al., `VICReg: Variance-Invariance-Covariance Regularization for
  Self-Supervised Learning`, arXiv:2105.04906. VICReg addresses representation
  collapse by combining an invariance/prediction term with per-dimension variance
  lower bounds and covariance decorrelation. ACS-JEPA already uses a related
  variance/covariance regularizer for graph/object state latents, but not for
  action latents.
- Balestriero and LeCun, `LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics`, arXiv:2511.08544. LeJEPA introduces Sketched
  Isotropic Gaussian Regularization (SIGReg), targeting an isotropic Gaussian
  embedding distribution for JEPA-style objectives. Its argument is directly
  relevant to action latents: if a latent distribution is anisotropic or
  collapsed, distance-based decoding can become ill-conditioned.
- Akbar, `Weak-SIGReg: Covariance Regularization for Stable Deep Learning`,
  arXiv:2603.05924. Weak-SIGReg adapts SIGReg as a lower-cost covariance-style
  stabilizer and views collapse as stochastic drift that can be controlled by
  pushing representation statistics toward isotropy.
- Wu et al., `VISReg: Variance-Invariance-Sketching Regularization for JEPA
  training`, arXiv:2606.02572. VISReg explicitly compares VICReg-style
  variance/covariance control with SIGReg-style distributional sketching and
  argues that variance terms give useful collapse gradients while sketching
  better constrains distributional shape.

Implication for ACS-JEPA:

- Add action-latent regularization as a diagnostic and likely model change.
  Current regularization only targets state latents through `GraphVCLoss` over
  graph/object latents. The action-latent distribution can still collapse or
  become extremely anisotropic even when state latents are well-regularized.
- However, regularization alone is not enough. VICReg/SIGReg can make action
  latents non-collapsed and better conditioned, but they do not by themselves
  guarantee that same-schema wrong-argument actions are separated or that decoded
  actions satisfy simulator preconditions. They should be paired with
  hard-negative and applicability objectives.

### Neuro-symbolic planning and applicability constraints

The PDDL setting adds a constraint that many latent-action papers avoid: a
continuous or learned action representation must eventually be projected into a
symbolic action whose preconditions hold in the current state. This makes the
ACS-JEPA issue closer to constrained/neuro-symbolic planning than pure latent
video prediction.

Implication for ACS-JEPA:

- Type-correctness is a weak syntactic constraint. Applicability is a semantic
  state-action relation and should be learned or approximated during decoding.
- Full symbolic applicability enumeration is useful as an offline oracle for
  labels and evaluation, but using it as the production action generator changes
  the algorithm into classical grounded search.

## Action-latent regularization: VICReg/SIGReg discussion

The root-cause note describes same-schema action latents separated by distances
around `1e-05`. That is not necessarily full action-latent collapse, because the
true action still ranks first in teacher-forced nearest-neighbor tests by
construction. But it is strong evidence of local aliasing and poor conditioning
for distance-based decoding. A dedicated action-latent regularizer should be
added to test whether better action-latent statistics improve CEM margins.

### Why state-latent regularization is insufficient

Current `GraphJEPALossModule` applies `GraphVCLoss` to predicted/observed state
latents, with configurable target `graph`, `object`, or `both`. The action
latents are only constrained indirectly by transition prediction and optional
inverse-dynamics regression. Therefore:

- action dimensions can have low variance even if object/state dimensions do not;
- schema embeddings can dominate variance while argument-binding dimensions have
  tiny variance;
- covariance anisotropy can make L2 distance mostly measure schema identity;
- inverse dynamics can regress to the collapsed/aliased action latent and still
  report a low loss.

### Proposed action VICReg-style regularizer

Apply variance/covariance regularization directly to the batch of encoded action
latents `action_latents` with shape `[B, K, D_a]`, flattened to `[B*K, D_a]`.

Recommended first form:

```text
L_action_vicreg =
    lambda_std * mean_d max(0, gamma - std(action_latents[:, d]))
  + lambda_cov * mean_{i != j} Cov(action_latents)_ij^2
```

This mirrors the existing state-latent VC loss, but the reporting must be
split by action schema:

- global action-latent std/cov;
- per-schema action-latent std/cov where enough samples exist;
- variance attributable to schema id versus argument substitutions.

A global variance term can be satisfied by separating schemas while still
collapsing same-schema arguments. Therefore the diagnostic should include a
same-schema variant:

```text
L_same_schema_std = mean_schema VICReg(action_latents[action_id == schema])
```

Use this carefully on small batches; aggregate across batches or enable it only
for schemas with at least a minimum sample count.

Expected benefit:

- Larger usable score margins for CEM.
- Less anisotropic action-latent geometry.
- Reduced dominance of schema dimensions over argument-binding dimensions.

Limitations:

- It does not teach which wrong argument is invalid.
- It may push apart actions that are genuinely interchangeable alternatives.
- It can conflict with transition prediction if many actions produce equivalent
  next-state latents. That conflict is informative and should be measured rather
  than hidden.

### Proposed action SIGReg-style regularizer

SIGReg/LeJEPA targets an isotropic Gaussian embedding distribution using random
sketches of the representation distribution. For action latents, the objective
would push the empirical action-latent distribution toward a well-conditioned
isotropic reference:

```text
L_action_sigreg = distance_sketch(action_latents, N(0, I))
```

For ACS-JEPA this is appealing because CEM and nearest-neighbor decoding depend
on metric geometry. An isotropic action-latent distribution should make distances
less dominated by a few schema dimensions.

But use a conditional interpretation:

- global SIGReg may encourage separation across action schemas;
- same-schema or residual SIGReg is more relevant to the observed failure;
- an action latent may naturally live on a structured/manifold-like distribution,
  not exactly an isotropic Gaussian. If strict SIGReg harms transition loss or
  valid alternative actions, prefer VICReg/Weak-SIGReg or conditional SIGReg.

Recommended implementation order:

1. Start with action VICReg because the code already has analogous VC machinery
   for state latents and it is easy to instrument.
2. Add metrics first: action-latent per-dimension std, covariance off-diagonal,
   effective rank, schema-conditional variance, and same-schema nearest-wrong
   margins.
3. If VICReg improves global statistics but not same-schema margins, add
   same-schema hard-negative contrastive loss rather than simply increasing the
   regularizer.
4. Test SIGReg/Weak-SIGReg as a second-stage regularizer if action-latent
   anisotropy remains severe or VICReg produces brittle scale tuning.

### Regularizer acceptance criteria

An action-latent regularizer is useful only if it improves decoding-relevant
metrics, not merely if the regularizer loss decreases.

Required metrics:

- action-latent per-dimension std does not collapse globally;
- effective rank of action latents increases or remains healthy;
- same-schema nearest-wrong margins increase on the fixed smoke validation slice;
- bounded CEM applicable-action rate improves or at least does not regress;
- first planned latent nearest-manifold scores improve when combined with the
  planner manifold penalty.

If the regularizer improves global variance but same-schema invalid neighbors
remain near-tied, the root cause is not generic collapse; it is missing
argument/applicability supervision.

## Recommended solution program

The solution should combine literature-grounded representation learning,
training-time identifiability, decoding-time applicability awareness, and
planning-time manifold constraints. No single larger CEM budget should be treated
as sufficient unless it passes the validation criteria below.

### Track A: Add applicability-aware supervision

Add a learned applicability head that scores `(state, grounded_action)` pairs.
It should be trained from positives and sampled negatives:

- positives:
  - reference trace action at each state;
  - optional offline `SimulatorEngine.applicable_actions()` samples on small
    states, used only as labels/oracle data;
- negatives:
  - same-schema one-argument substitutions;
  - role swaps where type-compatible;
  - sampled type-valid actions that `apply_action` rejects;
  - hard negatives from current decoder/planner failures.

Proposed module:

```text
ApplicabilityHead(action_latent, graph_latent, selected object latents)
  -> scalar logit P(applicable | state, action)
```

Training loss:

```text
BCEWithLogitsLoss(applicability_logit, applicable_label)
```

Integrate the score into decoding as:

```text
score(action) = latent_similarity(action, target)
              + lambda_app * log_sigmoid(applicability_logit)
```

For CEM this preserves learned decoding without requiring full applicable-action
enumeration at inference. For exact/ranked diagnostics it provides a direct way
to measure whether invalid near-neighbors are suppressed.

Acceptance criteria:

- On smoke validation diagnostic states, the top-1 decoded action should be
  applicable in at least 80% of transitions with the bounded CEM budget.
- Invalid same-schema one-argument substitutions should receive lower
  applicability logits than the true/applicable action with a positive median
  margin.
- The bounded planner should apply at least one valid action on both fixed smoke
  validation problems without using production-time `applicable_actions()`.

### Track B: Add contrastive same-state hard-negative action loss

Add a contrastive objective over actions in the same source state. For each
training transition, encode the true action and a small set of hard negatives.
Use negatives that are maximally confusing under the current diagnostics:

- same schema, one argument changed;
- same schema, role order swapped if type-valid;
- same road/car/junction entity class but wrong topology;
- currently decoded invalid action for that state, when available.

A practical first loss is InfoNCE-style over the predictor-conditioned target:

```text
positive = q_phi(true_action, state)
negatives = q_phi(negative_action_i, state)
anchor = inverse_dynamics_or_transition_delta(state, next_state)
L = CE(anchor dot [positive, negatives] / tau, label=positive)
```

If using only action-latent geometry, enforce a margin:

```text
L = mean_i max(0, margin + d(true_latent, anchor) - d(neg_i, anchor))
```

Important: the current inverse-dynamics loss regresses to `q_phi(action)`, so it
inherits any aliasing in `q_phi`. The contrastive loss must include explicit
negative actions, otherwise it will not guarantee argument identifiability.

Acceptance criteria:

- Same-schema nearest-wrong L2/cosine margins increase by at least an order of
  magnitude over the current `1e-05` scale on the fixed smoke diagnostic slice,
  or by a statistically stable positive margin if the metric is rescaled.
- The nearest wrong same-schema action is less often invalid, or appears below
  applicable alternatives after adding applicability scoring.
- CEM entropy should not collapse to invalid argument tuples; if it collapses,
  the modal tuple should usually be applicable.

### Track C: Replace whole-action latent decoding with role-aware scoring

The decoder should not rely only on a single vector distance after compressing
schema and all arguments. Add a role-aware decoder score that decomposes action
selection and argument binding:

```text
score(a_i(o_0, ..., o_k)) =
    schema_score(a_i, target, state)
  + sum_r role_score(a_i, r, o_r, target, state)
  + pairwise/topology_score(a_i, o_0, ..., o_k, state)
  + applicability_score(a_i, o_0, ..., o_k, state)
```

Minimal implementation path:

1. Keep `LatentActionEncoder` for transition prediction.
2. Add auxiliary projection heads from the action latent:
   - schema classifier;
   - per-role object classifier over problem-local objects, masked by type;
   - optional pairwise role compatibility scorer.
3. Train them teacher-forced from trace actions plus hard negatives.
4. Use their logits to initialize or bias `ActionSamplingFamily` CEM marginals.

This directly addresses the finding that action names are easy but object roles
are hard. It also gives the decoder a non-flat signal for argument identities.

Acceptance criteria:

- Frozen-latent probes or trained heads show high schema accuracy and improving
  per-role object accuracy on validation.
- CEM action-id entropy may collapse early, but argument entropies should retain
  uncertainty until there is score evidence; collapse to invalid tuples should
  decrease.

### Track D: Constrain planner latents to the encoded action manifold

Because production planning can create off-manifold latents, add an action
manifold term during latent optimization.

Options, from least invasive to most structural:

1. GMM component drift penalty:

```text
score -= lambda_manifold * min_j ||a_latent_t - encoded_pool_j||^2
```

where `encoded_pool_j` are the per-state seed action latents already computed by
`LatentGMMMPPIPlanner`.

2. Component-anchored GMM:

Keep each sampled latent as:

```text
latent = encoded_component + bounded_delta
```

and penalize or clamp `||bounded_delta||`. This retains local continuous search
but prevents large drift to arbitrary latent regions.

3. Structured symbolic planning path:

Use `StructuredCEPlanner` over grounded action samples for bounded diagnostics or
small states, but augment it with the applicability/role-aware scores above. Do
not rely on full `applicable_actions()` enumeration in production.

Acceptance criteria:

- First planned latent nearest encoded-action score should move from around `-25`
  toward the teacher-forced near-zero range, or at least top nearest neighbors
  should include applicable actions at rank 1-3.
- Bounded planner should stop failing with `decode_invalid` at the first action
  on `p166` and `p192`.

## Prioritized implementation sequence

### Phase 0: Research and measurement before implementation

1. Build a short related-work note from the papers listed above and any newer
   latent-action / JEPA regularization papers found during implementation.
2. Add pure diagnostics for action-latent statistics before adding losses:
   - global per-dimension std;
   - covariance off-diagonal penalty;
   - effective rank / eigenvalue spectrum;
   - per-schema variance where support is sufficient;
   - same-schema nearest-wrong margin distribution;
   - schema-vs-argument variance decomposition.
3. Compare baseline, inverse-dynamics, and RNN-action-encoder checkpoints on
   these metrics. This decides whether the immediate bottleneck is generic
   action-latent collapse/anisotropy or missing argument/applicability
   supervision.

Deliverable:

- `script/diagnose_action_latent_statistics.py` or equivalent extension to
  `diagnose_action_latent_geometry.py`.
- A JSON/markdown related-work summary with implications for ACS-JEPA.

### Phase 1: Add labels and probes before changing planner behavior

1. Create reusable negative sampler:
   - input: parsed problem, current state/action, random seed;
   - output: type-valid negatives categorized as one-argument substitution,
     role swap, random same-schema, random other-schema;
   - optional label: simulator applicability via offline `apply_action` check.
2. Add supervised probes/heads:
   - schema id from action latent;
   - role object id from action latent and state context;
   - applicability from `(state, action)`.
3. Train/evaluate on baseline and inverse-dynamics checkpoints without changing
   planning. This separates representation limitations from planner limitations.

Deliverable:

- `script/diagnose_action_supervised_probes.py` or equivalent training/eval
  script.
- JSON metrics for schema accuracy, per-role object accuracy, applicability AUROC
  or accuracy, and hard-negative margins.

### Phase 2: Train with explicit action identifiability losses

1. Add config fields under `model.loss`:
   - `action_vicreg_coeff`;
   - `action_vicreg_std_coeff`;
   - `action_vicreg_cov_coeff`;
   - `action_vicreg_std_margin`;
   - `action_sigreg_coeff` for later experiments, default disabled;
   - `action_contrastive_coeff`;
   - `action_contrastive_temperature`;
   - `action_hard_negatives_per_positive`;
   - `applicability_coeff`;
   - `argument_reconstruction_coeff`.
2. Extend `GraphJEPALossModule` or a sibling auxiliary module to consume sampled
   negative action tensors and labels.
3. Keep the JEPA transition loss unchanged initially; add the new losses as
   auxiliary terms.
4. Run the fixed smoke train/eval protocol.

Deliverable:

- A component test config similar to
  `script/configs/adaptive/01_action_decode/inverse_dynamics_smoke.yaml`.
- Updated diagnostics showing margin/applicability improvement.

### Phase 3: Use learned scores in decoding

1. Extend `ActionDecoder` with optional `applicability_head` and role-aware
   scorer.
2. Add decoder config:

```yaml
planning:
  action_decoder:
    score_terms:
      latent_similarity: 1.0
      applicability: 1.0
      role_binding: 0.5
```

3. Bias CEM initialization from schema/object logits when heads are available.
4. Preserve exact/ranked diagnostics to compare latent-only vs augmented scores.

Deliverable:

- Bounded action-decoder diagnostic improves applicable rate on the 44-transition
  smoke validation split.

### Phase 4: Add planner manifold regularization

1. In `LatentGMMMPPIPlanner`, retain the encoded seed action pool per state.
2. Add a manifold-distance penalty in the MPPI score function.
3. Optionally constrain GMM component deltas around encoded components.
4. Re-run first-latent diagnostics and bounded planner probes.

Deliverable:

- First planned latent is near the encoded action manifold.
- `p166` and `p192` fast bounded planning no longer fail at zero applied actions.

## Non-goals and cautions

- Do not use `SimulatorEngine.applicable_actions()` as the production action
  generator. It is acceptable only as an offline diagnostic/label oracle on
  controlled slices.
- Do not interpret exact/debug match against one planner trace as the planning
  metric. It is useful for invertibility diagnosis only.
- Do not reject inverse dynamics. The current inverse-dynamics target is too
  weak because it regresses to an already-aliased latent. Strengthen it with
  explicit action identity, argument, contrastive, or applicability targets.
- Do not assume more CEM samples solve the issue. Larger budgets may help
  teacher-forced decoding but do not prevent off-manifold planned latents.

## Decision criteria for resuming broad tuning

Resume broad tuning only when at least one condition is met:

1. Augmented decoding shows high applicability with teacher-forced action
   latents, and first-latent diagnostics show planner latents are near the
   grounded-action manifold.
2. Representation diagnostics show true/applicable actions separated from
   same-schema invalid near-misses with stable margins.
3. The bounded planner applies valid first actions consistently on both fixed
   smoke validation problems without production-time full applicable-action
   grounding.

Until then, tune only targeted components: negative sampling, applicability
losses, contrastive/action reconstruction losses, decoder scoring, and manifold
regularization.
