# Action/state dual-latent JEPA causal study specification

Date: 2026-08-04
Status: draft research specification; implementation, training, tuning, planner integration, and promotion are not authorized
Scope: bounded causal study of separately represented state and action latents, transition/applicability energy, and anti-collapse regularization
Governing predecessor: `script/ACTION_LATENT_UPDATED_SPEC.md`
Accepted predecessor decision: `script/ACTION_LATENT_UPDATED_PHASE0_DECISION.md`

## 1. Purpose

Updated Phase 0 showed that the tested ACS-JEPA checkpoint could not use its nearest action-latent alternatives as a reliable grounded-action interface. That result selected `BRANCH_D_ABSTRACT_ACTIONS` operationally, but it did not causally establish that every separately learned state/action latent formulation must fail.

The current evidence is restricted by:

- 12 training problems and 345 transitions;
- a 44-transition validation slice from CityCar `p166` and `p192`;
- three epochs and 102 optimizer steps;
- batch size 8;
- one principal 64-dimensional action-latent setting;
- the repository's variance/covariance-only action VC regularizer at one coefficient;
- no implemented action SIGReg experiment;
- nearest-in-action-latent invalid hard negatives rather than a complete semantic pair taxonomy;
- no causal localization of information loss across raw facts, graph embeddings, final state latents, action latents, and predictor outputs.

This specification defines the smallest bounded study needed to answer:

> Can ACS-JEPA jointly learn a state space and a distinct action space such that valid state-action-successor triples have low transition compatibility energy, incompatible or invalid action bindings receive reliably higher energy, planning-critical state facts remain represented, and both spaces avoid collapse without sacrificing transition prediction?

The study must distinguish a failed training recipe from a structural incompatibility. It must not become an open-ended coefficient sweep or silently resume broad Phase 1 tuning.

## 2. Theoretical contract and necessary correction

### 2.1 JEPA energy

For source state `s`, grounded action `a`, and candidate successor `s'`, define:

```text
z_s      = F_shared(s)
u_a      = Q(a, s)
z_target = F_shared(s')          # evaluation value; current training target is not detached
z_pred   = G(z_s, u_a)
E_trans(s, a, s') = D(z_pred, z_target)
```

`D` has separately normalized graph and object components. For one aligned transition triple:

```text
E_g = mean over graph latent coordinates of (z_pred_g - z_target_g)^2
E_o = mean over all aligned object rows and coordinates of (z_pred_o - z_target_o)^2
S_g = mean over development target graphs of MSE(z_target_g, mean_development_target_graph)
S_o = mean over development target object rows of MSE(z_target_o, mean_development_target_object)
E_trans = 0.5 * E_g / S_g + 0.5 * E_o / S_o
```

The development populations include every legal own-successor target selected by the frozen manifest, one record per source/action, and all latent coordinates. Two scale pairs are distinguished:

- `S_g_train,S_o_train` are computed once from the initialization/frozen baseline state encoder before any run and are used inside every `L_gap`, including frozen-state DL2–DL5 and trainable-state DL6. They are common to paired treatments and never updated.
- `S_g_eval,S_o_eval` are recomputed from each completed checkpoint on development targets and frozen for that checkpoint's development/test metrics.

All four are float64 scalars with bytes and hashes. If any required scale is nonfinite or `<=1e-8`, that run has malformed evidence and cannot pass; no epsilon fallback is used. Raw `E_g`, `E_o`, and repository `E_g+E_o` remain reported. `L_gap` uses train scales; official evaluation thresholds use eval scales. This removes any circular dependence on final checkpoint statistics while preventing latent-scale changes from satisfying evaluation margins.

This equation is the study's formal compatibility contract, not a description of the current implementation. Current `GraphJEPA.trajectory_rollout()` uses one trainable graph/state encoder for observed source and target states; `GraphJEPALossModule` does not stop gradients through target slices and has no EMA target encoder. DL1 audits this target path only. Stop-gradient/EMA training is not authorized by this specification; a positive target-path flag requires a separate reviewed state-space addendum.

A positive observed transition `(s, a+, s+)` should have low `E_trans(s, a+, s+)`.

The JEPA/energy-based framework permits defining compatibility through low energy, but the current positive-only prediction term does not guarantee high energy for an invalid action. An invalid action has no canonical legal successor, and anti-collapse regularization alone does not assign semantic meaning to invalid bindings.

Accordingly, this study must separately test:

1. positive transition prediction;
2. action-successor incompatibility energy;
3. state-action applicability energy;
4. state- and action-space collapse/conditioning.

The study must not infer applicability solely from Gaussian density, VICReg, SIGReg, or distance from the valid-state latent distribution.

### 2.2 Explicit transition-triple incompatibility energy

For two legal actions from the same source, `(s, a_i, s_i')` and `(s, a_j, s_j')`, with distinct canonical successors, define:

```text
Delta_E_distinct = E_trans(s, a_i, s_j') - E_trans(s, a_j, s_j')
L_gap = mean max(0, 0.20 - Delta_E_distinct)
lambda_gap = 0.10
```

`a_j` is the action that actually produces `s_j'`. Margin and coefficient are fixed; there is no scale matching or sweep. This objective ranks legal action-successor compatibility; it does not classify action validity.

Energy labels follow exact symbolic successors:

- legal action with own successor: positive triple;
- another legal action with a distinct canonical successor: negative triple;
- legal action with the exact same canonical successor: effect-equivalent positive for transition energy; no separation requirement;
- same effect signature but different full canonical successor: reported as its own ambiguity category and excluded from official positive/negative loss until its semantics are resolved;
- invalid action: no transition target and no primary `L_gap` label.

A secondary invalid-action successor-bank diagnostic may report:

```text
E_bank(s, a_invalid) = min_{s' in B_legal(s)} normalized_E_trans(s, a_invalid, s')
```

where `B_legal(s)` is the deduplicated set of exact legal one-step successors for that source state. The manifest must define complete versus sampled coverage, hashes, deduplication, and per-state weighting. `E_bank` supports only the claim that the invalid action is unlike the evaluated legal successors; it is never treated as a global validity score or official PASS predicate.

### 2.3 Applicability energy

Define applicability with the repository head exactly:

```text
ell_app = ApplicabilityHead(
    latent_dim=64, action_dim=64,
    max_action_arity=manifest.max_action_arity,
    hidden_dim=128, dropout=0.0,
)(
    graph_latent=z_s.graph,
    action_latent=u_final,
    object_latents=z_s.objects[gathered_argument_rows],
    argument_mask=active_role_mask,
)
E_app      = -ell_app
L_app      = BCEWithLogits(ell_app, y_app, pos_weight=N_negative/N_positive)
lambda_app = 0.10
```

`u_final` is A0's current output, A1's normalized output, A2/A3's identity output, or A4/A4_SHAM's fused output. Argument rows are padded to manifest maximum arity with zeros and mask `false`; real roles follow schema order. The same frozen-state object latents/masks are supplied in every architecture, including A3, because applicability is explicitly pair-conditional. Head parameters use the local module-seed rule and train only in O2/O3.

`N_positive` and `N_negative` are training-manifest counts before batching. Both must be nonzero. `pos_weight` is float64, hash-frozen, and unclipped. Batches preserve manifest proportions; no class resampling is applied. Development F1 uses the smallest distinct observed development logit attaining maximum problem-macro F1, with candidates `{-inf} U observed_logits U {+inf}`; the threshold is frozen for test. AUROC/AP and groupwise ranking use logits directly.

`E_app` is the sole primary validity measure. Exact offline symbolic/simulator labels are allowed only for frozen construction/evaluation, never production candidate generation.

The fixed objective ablation is:

```text
O0 = positive transition prediction with signed frozen state encoder; state VC diagnostic only; no action geometry regularizer
O1 = O0 + legal transition-triple L_gap
O2 = O0 + applicability energy
O3 = O0 + L_gap + applicability energy
```

These four cells use identical architecture, matched initialization, minibatch order, examples, optimizer budget, and seeds. The state encoder is frozen and action geometry regularization is off during this ablation; action anti-collapse is studied only afterward.

## 3. Related-work basis and limits

The study is grounded in:

- LeCun et al., "A Tutorial on Energy-Based Learning" (2006), and Dawid and LeCun, "Introduction to Latent Variable Energy-Based Models", arXiv:2306.02572: compatible configurations receive lower energy than incompatible configurations only when the loss or constrained architecture supplies that pressure; an energy architecture alone is insufficient.
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022): JEPA predicts representations and frames prediction discrepancy as compatibility energy while emphasizing regularized architectures to avoid collapse.
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA), arXiv:2301.08243: non-contrastive positive target prediction can learn semantic representations with architectural asymmetry and target-encoder dynamics, but it does not establish invalid-action energy in an action-conditioned symbolic domain.
- Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (V-JEPA), arXiv:2404.08471: positive latent regression without explicit negatives further demonstrates that JEPA prediction energy and negative compatibility supervision are separate design choices.
- Bardes et al., "V-JEPA 2", arXiv:2506.09985: action-conditioned latent prediction and goal-distance planning energy do not by themselves provide an applicability energy; candidate actions remain distribution-constrained.
- Bardes, Ponce, and LeCun, "VICReg", arXiv:2105.04906: variance and covariance terms reduce representational collapse and redundancy, but do not assign grounded-action semantics.
- Balestriero and LeCun, "LeJEPA", arXiv:2511.08544: SIGReg's isotropic-Gaussian distributional regularization motivates a reviewed action/state variant; its scoped geometric or downstream-risk guarantees do not establish action identity, applicability, or high invalid-action energy.
- Sobal et al., "Joint Embedding Predictive Architectures Focus on Slow Features", arXiv:2211.10831: a non-collapsed VICReg-trained action-conditioned predictor can preserve static shortcuts and ignore controllable/action-sensitive content, motivating explicit action interventions and Jacobian tests.
- Chandak et al., "Learning Action Representations for Reinforcement Learning", arXiv:1902.00183: action-representation identifiability depends on explicit transition/action reconstruction assumptions; those assumptions are not inherited automatically by JEPA prediction.
- Grimm et al., "The Value Equivalence Principle for Model-Based Reinforcement Learning", arXiv:2011.03506: a learned model may preserve decision-relevant quantities without reconstructing every observation or action identity; this motivates explicitly testing which equivalences are harmless for planning.

These sources motivate the hypotheses and losses. None guarantees that the proposed ACS-JEPA dual-space architecture will work.

## 4. Representation hypotheses

### H1: State-information preservation

Planning-critical facts must remain decodable through the state pathway:

```text
raw symbolic facts
→ graph/object encoder outputs
→ final state latents z_s
→ predicted successor latents z_pred
```

If a fact is present in raw and graph representations but absent from final `z_s`, action regularization cannot repair that loss. State-space representation or prediction must be addressed first.

### H2: Action-space conditioning

A distinct action space can retain schema, role, and object-binding information at usable scale while allowing effect-equivalent actions to share transition behavior.

Healthy global effective rank is insufficient. The action study must measure:

- global variance/rank;
- schema-residual variance/rank;
- within-schema scale relative to schema scale;
- role/object binding retrieval;
- state-conditioned applicability;
- stability across states and problems.

### H3: Fusion/predictor sensitivity

Even if `u_a` distinguishes actions, `G(z_s, u_a)` may ignore those differences. Predictor action sensitivity must be measured separately from action-encoder geometry.

### H4: Explicit negative pressure

If invalid actions are off-support, positive transition prediction need not reject them. Explicit incompatibility or applicability supervision may be necessary even with VICReg/SIGReg.

### H5: Conditional anti-collapse

Global VC/SIGReg may be satisfied by separating action schemas while same-schema bindings remain aliased. Schema-residual or candidate-conditional regularization is more relevant to grounded identity.

### H6: Benign versus harmful equivalence

Transition-equivalent legal actions need not be separated by transition energy. Aliasing is harmful when it merges:

- valid and invalid bindings;
- legal actions with distinct symbolic successors;
- actions whose distinction changes goal reachability or planning cost.

## 5. Matched architecture ladder

The current repository already has different tensors and configurable widths for state and action latents, but `LatentActionEncoder` constructs action arguments directly from final state object latents. The study therefore uses cumulative, matched ablations rather than treating a wholesale baseline-to-two-tower replacement as causal evidence for “separation.”

### A0: Current coupled baseline

```text
z_s = F(s)
u_a = Q_current(action id, selected z_s object latents)
z_pred = G(z_s, u_a)
```

### A1: Geometry-only decoupling

Keep `D_s=D_a=64`, action IDs, selected `z_s` inputs, the pinned baseline `action_encoder.kind='pooled'` composer, predictor, and every trainable parameter unchanged. Apply `LayerNorm(64, elementwise_affine=False, eps=1e-5)` to the final action vector immediately before predictor/applicability use. It adds zero trainable parameters. This isolates action-space normalization; width is not varied in this specification.

### Frozen descriptor tensor

A2–A4 use one literal descriptor per grounded argument object:

```text
d_obj = concat(
  one_hot(object_type, sorted training-domain type vocabulary),
  sinusoidal_PE16(canonical object ordinal),
  static_incidence
)
```

The canonical ordinal is the zero-based index after UTF-8 bytewise sorting object names within a problem. For `k=0..7`, `sinusoidal_PE16[2k]=sin(ordinal/10000^(2k/16))` and `[2k+1]=cos(ordinal/10000^(2k/16))`, computed float64 then cast float32. A predicate is static iff it appears in no add or delete effect of any parsed domain action schema. `static_incidence` has one float32 coordinate for every `(static predicate, argument position)` pair sorted bytewise; its value is the count of true static atoms containing the object in that position divided by the maximum training-manifest count for that coordinate, with zero when the maximum is zero. Dynamic predicates and successor facts are excluded. Unknown test types/static coordinates are a preflight FAIL rather than silently mapped. Each argument also receives a one-hot role position of width equal to maximum domain action arity. Descriptor schema, maxima, vocabularies, object ordinals, and bytes are manifest-hashed.

All new linear weights use Xavier-uniform gain `1.0`; biases are zero. Each new module uses a local CPU `torch.Generator` seeded by the first unsigned 64 little-endian bits of `SHA256(UTF8(str(run_seed)+":"+fully_qualified_module_path))`; initialization order cannot affect another module. Existing modules load byte-identical paired baseline weights. All architecture cells use `D_a=64`, `action_encoder.kind='pooled'`, hidden width `64`, dropout `0`, argument tensors in action-schema role order, and the same `ResidualMLPLatentPredictorG` fusion.

### A2: Descriptors plus current state conditioning

For each argument role, replace the current object-projector input `z_s_object` by `concat(z_s_object, d_obj, one_hot(role))`; change only that projector input width. Schema embedding, argument order, pooled composer, output width, predictor, and losses stay fixed.

### A3: Separate identity tower

Use the identical A2 module and parameter shapes, but replace the `z_s_object` segment with a zero tensor before the projector:

```text
u_identity = Q_A2(schema, role, concat(zeros(D_s), d_obj, one_hot(role)))
z_pred = G(z_s, u_identity)
```

Because A2 and A3 have byte-identical parameter shapes and paired initialization, A2→A3 isolates removal of final-state object content while preserving explicit grounded descriptors.

### A4: Identity plus state-conditioned effect context

Keep A3 `u_identity`. A separate context composer uses the original A0 selected `z_s` object inputs and pooled composer to produce `u_context in R^64`. Fusion is:

```text
u_fused = LayerNorm(64, elementwise_affine=False, eps=1e-5)(Linear_128_to_64(concat(u_identity, u_context)))
z_pred  = G(z_s, u_fused)
```

The applicability head receives `z_s.graph_latent` and `u_fused`; identity retrieval reads `u_identity`. A matched `A4_SHAM` has identical modules, parameter shapes, initialization, and optimizer but replaces `u_context` by zeros at fusion. A4 is evaluated only by paired A4−A4_SHAM differences.

A4/A4_SHAM are authorized only when A3 development complete-binding retrieval is at least `0.80`, A3 positive prediction ratio versus A0 is at most `1.05`, and either A3 legal-distinct pairwise accuracy or applicability groupwise top-1 is below `0.80`. Otherwise both are recorded `NOT_AUTHORIZED`.

Every adjacent comparison uses paired seeds, data order, optimizer budget, initialization for unchanged modules, literal tensors above, and reported parameter counts. No unspecified capacity-matching module is permitted.

### Explicit exclusions

This study does not authorize a planner, continuous latent optimization, inverse-decoder promotion, broad Phase 1 training, a production candidate generator, production simulator access, option/skill learning, or simultaneous changes to state encoder, action encoder, predictor, regularizer, and dataset.

## 6. Frozen action and triple taxonomy

The manifest separates state-action validity from transition-triple compatibility.

Action records:

1. `APPLICABLE_RECORDED`.
2. `APPLICABLE_ALTERNATIVE`.
3. `INAPPLICABLE_ONE_ROLE`.
4. `INAPPLICABLE_ROLE_SWAP`.
5. `INAPPLICABLE_RANDOM_TYPE_VALID`.
6. `ARGUMENT_DISTANCE_CONTROL`.

Triple records:

1. `LEGAL_OWN_SUCCESSOR`: `(s, a_i, canonical_successor(s,a_i))`.
2. `LEGAL_DISTINCT_MISMATCH`: `(s, a_i, canonical_successor(s,a_j))`, where both actions are legal and full canonical successors differ.
3. `LEGAL_EXACT_SHARED_SUCCESSOR`: two legal actions whose full canonical successor is byte-equivalent after canonicalization.
4. `LEGAL_SHARED_EFFECT_DIFFERENT_STATE`: same canonical effect signature but different full canonical successor; diagnostic only until semantics are reviewed.
5. `INAPPLICABLE_NO_SUCCESSOR`: `(s,a_invalid)` with no transition-triple target and applicability labels only.

Canonical state bytes are UTF-8 JSON arrays of fully grounded atoms. Each atom is `[predicate,arg1,...]` with original case-sensitive strings; atoms are sorted by their compact JSON UTF-8 bytes. The outer JSON uses `ensure_ascii=false`, separators `(',',':')`, and one final newline. Effect bytes are exactly `{"add":[...],"delete":[...]}` with keys in that order and atom arrays encoded/sorted identically. Full-successor equality is canonical-state byte equality; effect-signature equality is never substituted.

For each source, the offline oracle enumerates all applicable actions for evidence only, executes each once, groups by full-successor bytes, sorts groups by successor bytes, and sorts actions within each group by canonical action bytes. Every action contributes one `LEGAL_OWN_SUCCESSOR`. For each unordered pair of different successor groups, the lower-successor-byte group is `L`, the higher is `H`, and their first actions are `a_L,a_H`; form both ordered mismatch directions. Every mismatch record stores source bytes/hash, `action_i_bytes`, `successor_i_bytes/hash`, `action_j_bytes`, `successor_j_bytes/hash`, both schema IDs, and effect-relation category. Thus `E_mismatch=E_trans(s,a_i,s_j')` and `E_own=E_trans(s,a_j,s_j')` always have an explicit comparator. Exact-shared-successor actions are exclusion controls and never enter `L_gap`.

Each source keeps at most 128 ordered training directions. Direction pairs are indivisible. A pair unit has orientation-independent hash `SHA256(source_hash || successor_L_hash || action_L_bytes || successor_H_hash || action_H_bytes)`. Its stratum key is compact UTF-8 JSON `[min(schema_L,schema_H),max(schema_L,schema_H),effect_relation]`. Strata are traversed in ascending stratum-key bytes; units within each stratum are ascending pair-unit hash; round-robin takes one unit per nonempty stratum until 64 units or exhaustion. Evaluation uses the same ordering with a 2,048-unit cap.  Manifests record total/retained units and payload hashes. Losses average directions within source then sources; metrics macro-average sources then problems. No successor-group-size weighting is applied.

Action bytes are UTF-8 compact JSON `[schema,arg1,...]` with original case-sensitive strings and one final newline. Invalid-action records never enter triple construction. Nearest-latent negatives remain diagnostic only.

## 7. Study phases and fixed treatment designs

### Common optimization protocol

Official seeds are exactly `[0,1,2]`. PyTorch deterministic algorithms are enabled; cuDNN benchmark and TF32 are disabled. Every official smoke cell uses Adam (`lr=1e-3`, betas=`(0.9,0.999)`, eps=`1e-8`, weight_decay=`0`), batch size `24`, rollout length `4`, 102 optimizer steps, no gradient clipping, 10-step linear warmup from `1e-8*lr`, then cosine decay to `1e-5`. There is no early stopping or best-checkpoint selection: step 102 is evaluated. Data order for each complete pass/cycle is ascending `SHA256(UTF8(str(seed)+":"+str(cycle_index)+":"+record_hash))`; batches cycle until step 102. Validation runs only after the final step.

DL3/DL4 cells load the same DL0-pinned state/action/predictor weights; the state encoder is frozen, while action/predictor/new heads train. DL4 cells do not warm-start from earlier architecture cells. DL5 cells for a seed all load the same selected DL4 final checkpoint for that seed and train 102 additional matched steps. DL6 baseline and winner both reload the same pinned initialization, unfreeze state, and train exactly 1,000 steps with the same optimizer/scheduler shape rescaled to 100 warmup steps; final step only is evaluated. Any OOM or nondeterministic-kernel workaround changes the protocol and requires spec review rather than silent substitution.

DL2 uses Adam with the same scalar hyperparameters, no scheduler, one full-manifest batch, and exactly 2,000 steps; final step only is assessed. The pinned state encoder is frozen. For each required schema, the micro manifest selects by SHA-256 the first four source states having at least two same-schema applicable actions with distinct full successors and two same-schema inapplicable one-role substitutions. It retains the first two legal and first two invalid actions by action-byte order, both ordered legal mismatch triples, and all four own/applicability records. If exact-shared-successor alternatives exist for that schema in the oracle pool, the first such pair is added as an exclusion control. A required schema lacking the core four-state construction makes DL2 evidence FAIL.

### DL0: Evidence, split, and power preflight

Deliverables:

- immutable train/development/untouched-test problem splits;
- schema and action-cardinality-stratum coverage in every split;
- frozen action/triple manifests and exact canonicalization audit;
- per-schema independent sample counts and candidate counts;
- parameter, optimizer-step, runtime, and memory accounting;
- exact SHA-256 pin of the accepted Updated-Phase-0 baseline checkpoint referenced by `ACTION_LATENT_UPDATED_PHASE0_DECISION.md`; no validation-based checkpoint reselection;
- development-only variance estimates and a preregistered detectable-effect/power analysis defining the minimum number of independent test problems.

The inferential unit is an independent PDDL problem. DL0 requires at least 12 development and 12 untouched-test problems, with no shared generator seed/template lineage when that metadata exists. Training cardinality quartiles are computed from median type-valid action count per problem using deterministic NumPy `quantile(method='linear')`; each development/test quartile must contain at least two problems. A schema is `required` only if it occurs in at least four development and four test problems; other schemas are reported as sparse and cannot support a schema-specific PASS.

The preregistered inferential policy is:

```text
alpha = 0.05 family-wise, one-sided for four directional DL3 contrasts
confidence intervals = 95% two-sided
power = 0.80
minimum detectable macro-metric difference = 0.10
randomization unit = problem
seed aggregation = arithmetic mean of the three paired-seed problem metrics
```

The four hypothesis IDs, in tie order, are `H1=O1-O0` and `H2=O3-O2` for legal-distinct pairwise accuracy, and `H3=O2-O0` and `H4=O3-O1` for applicability groupwise top-1. For each `h`, the statistic is the arithmetic mean paired problem difference. For `n<=20`, enumerate all `2^n` sign flips and set `p_h=count(T_perm>=T_obs)/2^n`. For larger `n`, draw exactly 1,000,000 PCG64(0) sign vectors and set `p_h=(1+count(T_perm>=T_obs))/(1+1_000_000)`. Sort by `(p_h,hypothesis_ID)` and compute Holm adjusted values `p_adj(k)=max_{j<=k} min(1,(4-j+1)*p_(j))`. Eligibility requires strict `p_adj<0.05` plus numeric gates.

For each contrast separately, compute sample standard deviation `sigma_dev_h` with `ddof=1` over development problem differences and:

```text
n_power_h = ceil(((1.6448536269514715 + 0.8416212335729144) * sigma_dev_h / 0.10)^2)
n_required = max(12, n_power_H1, n_power_H2, n_power_H3, n_power_H4)
```

Any nonfinite variance or fewer than two development problem differences is malformed evidence. If test problems are fewer than `n_required`, `evidence_ok=false`. For each contrast, 95% intervals use exactly 100,000 PCG64(0) problem-block bootstrap resamples, recomputing the three-seed mean, with NumPy `quantile(method='linear')` endpoints `[0.025,0.975]`. `p166`/`p192` remain historical. Test runs once after freeze.

DL0 PASS requires deterministic repeat manifests, zero split overlap, complete hashes, all required cardinality strata, the minimum problem counts, and independent review of computed lineage/power evidence.

### DL1: No-training causal localization

Run frozen probes at raw symbolic state, graph encoder output, projected graph/object features, final `z_s`, raw action descriptors, pre-projection action features, final `u_a`, `z_pred`, and observed targets.

Required outputs:

- precondition/topology decodability with raw controls;
- exact-successor separation by triple category;
- fixed same-state action substitution/permutation interventions;
- schema/role/object retrieval;
- predictor action Jacobian spectrum;
- source/target gradient flow and fixed-batch target-detach intervention;
- loss-gradient norms/cosines;
- effective rank, std, covariance, and within-schema scale at every boundary;
- a fixed-capacity oracle substitution probe matrix trained only on development records:

```text
learned z_s + learned action features
raw/oracle state facts + learned action features
learned z_s + structured oracle action descriptors
raw/oracle state facts + structured oracle action descriptors
```

The probe architecture, optimizer, examples, and seeds are identical across cells. It is diagnostic only and does not alter JEPA checkpoints. This matrix localizes state, action, and predictor/probe capacity ceilings.

The localization assessor emits independent Boolean flags for state-information loss, action-information loss, predictor insensitivity, target-path shortcut, and invalid-action support gap. These are recommendations only; Branch D remains operationally active.

All frozen probes use training records only for fitting and never tune hyperparameters. Boolean fact probes standardize each input coordinate with training mean/std (zero-std coordinates become zero) and fit one scikit-learn `LogisticRegression(C=1, penalty='l2', solver='liblinear', class_weight='balanced', tol=1e-6, max_iter=10000, random_state=0)` per required fact. The action-binding probe is exactly `ArgumentReconstructionHead(action_dim=64, object_dim=64, max_action_arity=manifest.max_action_arity, hidden_dim=64, dropout=0)`. It receives detached `u_final`, all detached source-state object latents as candidates, and a Boolean `[B,A,O]` mask true only for active roles/type-compatible objects. The head alone trains with Adam lr `1e-3`, weight decay 0, summed active-role cross-entropy divided by active-role count, exactly 2,000 steps, seed matched to the checkpoint, no early stopping. The four oracle-substitution cells use two independent two-layer MLPs `[input -> 64 -> ReLU -> 1]`, one for applicability labels and one for own-successor (`1`) versus legal-distinct-mismatch (`0`) triple labels. Each uses Xavier initialization under the module seed rule, class-weighted BCE as in Section 2.3, Adam lr `1e-3`, batch 256, exactly 2,000 steps, and identical record orders. Probe nonconvergence or nonfinite output is malformed evidence, not a reason to change the probe.

### DL2: Deterministic micro-overfit capacity gate

Use one seed and balanced tiny datasets containing own-successor triples, legal distinct mismatches, exact shared successors, and inapplicable substitutions. Test A0–A3 across O0–O3: exactly 16 micro-fit runs. Determinism is checked by replaying the saved checkpoint/evidence generation twice, not by retraining. If A4 is later authorized in DL4, A4 and A4_SHAM each receive four micro-fit runs before smoke use, for an absolute DL2 cap of 24 training runs.

For each architecture, run all four O0–O3 objective cells. The state encoder is frozen and repository state VC is diagnostic/no-grad; action geometry regularization is R0/off.

Micro-overfit PASS requires all of:

- at least 99% reduction from initial positive transition loss and final normalized positive loss at most `1e-4`;
- complete-binding training retrieval exactly `1.0`;
- at least 99% of legal-distinct training pairs satisfy the preregistered margin under O1/O3;
- applicability classification and groupwise ranking exactly `1.0` under O2/O3;
- exact-shared-successor pairs are excluded from `L_gap` and receive no forced-separation label;
- exact repeat output under the same seed.

Failure stops that architecture before smoke training.

### DL3: Objective ablation — fixed 12-checkpoint design

DL3 is authorized only if `state_ok_dev_baseline` is true: the single DL0-pinned checkpoint must satisfy the `state_ok` predicate/fact clauses on development, without the later three-seed clause. That checkpoint's graph/state encoder is frozen for every DL3–DL5 seed; transition, applicability, and action-geometry gradients cannot update it. This isolates action/objective effects before joint co-adaptation.

Use A0, state VC as a fixed diagnostic, action geometry R0/off, and four cells:

```text
O0 positive prediction with frozen state encoder
O1 O0 + legal transition-triple L_gap
O2 O0 + E_app
O3 O0 + L_gap + E_app
```

Run three paired seeds: exactly 12 checkpoints. For each seed, unchanged modules share initialization, data order, and minibatches. Primary paired estimands are:

- O1−O0 and O3−O2 legal-distinct pairwise energy accuracy;
- O2−O0 and O3−O1 applicability groupwise top-1;
- each treatment's positive prediction-loss ratio to O0.

Report per-seed paired differences, exact problem-level randomization intervals, and Holm-adjusted tests for the four primary contrasts. O3 is the only promotable objective because the final claim requires both triple and applicability semantics. O3 is eligible for DL4 only if O3−O2 and O3−O1 each have adjusted `p<0.05`, mean paired gain at least `0.10`, at least two of three seeds improve, both O3 development metrics reach `0.80`, and positive prediction-loss ratio to O0 is at most `1.05`. O1/O2 remain causal ablations. If O3 fails, later architecture/regularizer phases are not run.

### DL4: Matched architecture ladder — maximum 15 new checkpoints

Hold O3 fixed. Compare A0→A1→A2→A3 sequentially with three paired seeds; A0 checkpoints are reused, so at most nine new checkpoints. If authorized, A4 and A4_SHAM add exactly six checkpoints after their DL2 micro-fit gate.

The closed DL4 primary set is `{complete_binding_retrieval, normalized_binding_margin, legal_distinct_pairwise_accuracy, applicability_AUROC, applicability_groupwise_top1, positive_prediction_loss_ratio}`. A1–A3 are tested in order. A cell is retained only if complete-binding retrieval or normalized binding margin gains at least `0.10` versus its immediate predecessor, at least two paired seeds improve that same metric, each of the three semantic metrics regresses by at most `0.02`, and its positive-loss ratio to A0 is `<=1.05`. Exact ties within `1e-12` do not count as improvement. First failure stops A1–A3 and selects the last retained predecessor.

A4 is authorized only when selected A3 has complete-binding retrieval `>=0.80`, positive prediction-loss ratio to A0 `<=1.05`, and either applicability groupwise top-1 or legal-distinct accuracy is `<0.80` on development. A4 and A4_SHAM are then both run. A4 is retained only if every deficient authorization metric gains `>=0.10` versus A4_SHAM and reaches `>=0.80`, at least two seeds improve, the remaining semantic metrics regress by at most `0.02`, and positive-loss ratio is `<=1.05`; otherwise A3 remains selected. If A3 was not retained, A4 is not run.

### DL5: Anti-collapse study — maximum 12 checkpoints

Hold the DL3 objective and DL4 architecture fixed. `GraphVCLoss` remains unchanged throughout this action-regularizer study. Compare three paired-seed cells:

```text
R0 no action geometry regularizer
R1 repository action VC (variance + covariance; no invariance term)
R2 schema-residual action VC
```

This is nine checkpoints. One SIGReg cell `R3` may add three checkpoints only after its estimator contract in Section 8 passes independent implementation review. No coefficient sweep is permitted.

A regularizer is eligible only if its mean three-seed problem-macro gain in action residual scale or normalized binding margin is at least `0.10` relative to R0, at least two seeds improve, applicability/triple ranking does not regress more than `0.02`, and positive prediction-loss ratio remains at most `1.05`. Improving only global rank is FAIL. Select the eligible cell with the largest arithmetic mean of its two gains `(residual-scale gain + binding-margin gain)/2`; ties within `1e-12` prefer simpler cells in order R1, R2, R3. If no cell is eligible, select R0 and set `geometry_ok=false`.

A state-regularizer intervention is not crossed with DL5. If DL1 identifies state-information loss, a separate state-space addendum is required.

### DL6: One joint-learning confirmation and untouched test

Promote exactly one frozen winner plus A0/O0 baseline, three seeds each. In DL6 only, unfreeze the graph/state encoder for both configurations using matched initialization, data order, state VC, optimizer budget, and gradient accounting. This is the sole official test that JEPA can co-learn the state and action spaces after the frozen-state causal ladder. No fallback winner is selected after test observation. The confirmation adds fixed-cardinality candidate diagnostics without a planner: top-k retrieval at fixed `K`, candidate ranking, query-latent perturbations, and performance versus action cardinality.

The untouched test runs once after code, manifests, coefficients, and assessor hashes are frozen.

## 8. Loss and regularizer contracts

The executable O0 core is frozen as:

```text
GraphLatentPredictionLoss: graph_weight=1.0, object_weight=1.0
prediction_coeff=1.0
GraphVCLoss: target=both, std_coeff=1.0, cov_coeff=1.0, std_margin=1.0
regularization_coeff=1.0 when state encoder is trainable; diagnostic-only when frozen
similarity_coeff=0.0
inverse_dynamics_coeff=0.0
action contrastive/argument reconstruction/action VC/SIGReg coefficients=0.0
applicability coefficient=0.0 except O2/O3
gap coefficient=0.0 except O1/O3
goal-head loss excluded from all official cells
rollout-order weights=uniform
```

The decomposed trainable loss is:

```text
L_total =
    1.0 * L_positive_transition
  + 1.0 * L_state_vc                 # joint DL6 only; diagnostic/no-grad in DL2–DL5
  + 0.10 * L_gap                     # O1/O3 only
  + 0.10 * L_app                     # O2/O3 only
  + lambda_action_geom * L_action_geom  # DL5 selected cell only
```

State VC is computed but detached/diagnostic during frozen-state DL3–DL5. There is no development scale matching and no coefficient selection algorithm: every coefficient is literal above. `L_gap` is defined only over legal transition triples. `L_app` is binary state-action BCE. Test data cannot change any coefficient.

### 8.1 Repository VC regularizers

The existing action and graph losses are named `VC`, not full VICReg, because they contain no paired-view invariance term:

```text
VC(X) =
  mean_d max(0, gamma - sqrt(Var(X_d) + epsilon))
  + mean_{i != j} Cov(X)_ij^2
```

Official constants are the accepted repository values:

```text
gamma = 1.0
epsilon = 1e-4
std coefficient = 1.0
covariance coefficient = 1.0
overall action geometry coefficient = 0.1
```

The eligible geometry population is exactly the deduplicated set of legal `LEGAL_OWN_SUCCESSOR` training-manifest records from every required schema, keyed by `(source_hash,action_bytes)`; invalid, mismatch, sparse-schema, development, and test records are excluded. Let `C` be the number of required schemas and `m=ceil((4*D_a)/C)`. Each required schema must contain at least `m` unique eligible records. For each seed, every step uses the same precomputed batch manifest across R0–R3: each schema queue is sorted by `SHA256(UTF8(seed:step)||source_hash||action_bytes)`, and the first `m` records are taken. Thus all cells receive exactly `C*m >= 4*D_a` records, equal schema counts, no duplicates, and no undersampling discrepancy. Missing schema/count/hash consistency makes the run malformed. R0 computes diagnostics only; R1 applies repository VC to the whole batch; R2 partitions that identical batch by schema.

For each schema `c`, with residual matrix `R_c = U_c - mean(U_c,axis=0)` and `n_c=m`:

```text
Cov_c = R_c^T R_c / (n_c - 1)
Cov_equal_schema = mean_c Cov_c
std_penalty = mean_d max(0, 1 - sqrt(Cov_equal_schema[d,d] + 1e-4))
cov_penalty = mean_{i!=j} Cov_equal_schema[i,j]^2
R2 = std_penalty + cov_penalty
```

Thus schemas, not records, receive equal weight. The identical geometry-batch manifest is itself hashed and is an input to every R0–R3 artifact.

No paired-view invariance term is claimed. A future genuine VICReg action-invariance term requires a separate specification of semantically valid action views.

`GraphVCLoss` is literal repository `target='both'`: observed target timesteps `1..K` are flattened, graph and object rows are concatenated on the sample axis, variance uses PyTorch unbiased `var`, covariance denominator is `N-1`, `std_coeff=cov_coeff=std_margin=regularization_coeff=1.0`, and gradients flow through the shared graph/state encoder only in DL6. DL2–DL5 compute the value under `no_grad` as a diagnostic.

State geometry is also evaluated separately on graph and object target matrices. `state_geometry_ok` requires, for each matrix, at least 90% of dimensions with std `>=0.50`, covariance effective rank `>=0.50*D_s`, and off-diagonal correlation RMS `<=0.20` in at least two of three seeds.

### 8.2 SIGReg cell R3

R3 is a deterministic sketched isotropic-Gaussian discrepancy over schema-residual action samples. For fixed unit projection vectors `v_j` and frequencies `t in {0.5, 1.0, 2.0}`:

```text
phi_hat(j,t) = mean_n exp(i * t * dot(v_j, r_n))
phi_N(t)     = exp(-t^2 / 2)
L_SIG = mean_{j,t} |phi_hat(j,t) - phi_N(t)|^2
```

Contracts:

- `J=256`; projection matrix is generated once as `numpy.random.Generator(numpy.random.PCG64(0)).standard_normal((256,D_a), dtype=float64)`, row-normalized in float64, saved as little-endian NumPy `.npy`, and SHA-256 hashed;
- frequencies are exactly float64 `[0.5,1.0,2.0]`;
- no batch standardization;
- schema centroids and residuals are computed independently in each geometry batch;
- characteristic functions are equally weighted by schema:

```text
phi_hat(j,t) = mean_c mean_{n in c} exp(i*t*dot(v_j,r_cn))
```

- R3 uses the exact same hashed `C*m` geometry batch as R0/R1/R2, where `m=ceil((4*D_a)/C)`;
- coefficient is exactly `0.10`; no sweep;
- gradients flow through current action samples and their batch centroids;
- real/imaginary sums and loss accumulate in float64; the scalar is cast to model dtype only when added to `L_total`;
- CPU repeat must be bitwise equal; row permutation changes loss by at most `1e-12`; analytic versus central-finite-difference gradients use `rtol=1e-5, atol=1e-7`; a fixed `N(0,I)` fixture of 4096 samples must have loss `<0.02`; an all-zero fixture must have loss `>0.10`; nonfinite, insufficient-count, and duplicate-record fixtures must raise;
- independent implementation/evidence review is required before R3 training.

The R3 claim is limited to the defined estimator. It does not claim that standard-normal geometry implies applicability or grounded identity.

### 8.3 Gradient accounting

Every run reports each loss magnitude, parameter-group gradient norm, and pairwise gradient cosine. Required nonzero finite gradients are: predictor/action encoder for transition cells; applicability head for O2/O3; action encoder from `L_gap` for O1/O3; action encoder from geometry loss for R1/R2/R3; graph/state encoder from transition/state-VC only in trainable-state DL6. Frozen-state DL2–DL5 graph/state gradients must be exactly absent/zero. Disabled heads/losses must be absent/zero. Violation is malformed evidence.

## 9. Required metrics

### State representation

- predicate/topology probe AUROC, AP, and F1;
- exact fact accuracy where labels are balanced;
- raw versus graph versus final-state probe gap;
- legal distinct-successor latent distance distribution;
- legal same-successor control distribution;
- state effective rank, per-dimension std, and covariance.

### Action representation

- global and schema-residual effective rank;
- absolute within-schema variance and fraction of total variance;
- schema, role, and complete-binding retrieval;
- normalized true-versus-hard-negative margin;
- stability of `u_identity` across source states where action identity is semantically comparable;
- action-space per-dimension std and covariance;
- sample-count and covariance-rank adequacy.

### Energy and applicability

- `E_positive` by problem/schema;
- `Delta_E_distinct` and legal-distinct pairwise accuracy;
- exact problem-level effects and randomization intervals, never transition-only resampling;
- applicability AUROC, AP, F1, calibration error, and positive/negative energy margins;
- groupwise top-1 and top-k applicable ranking;
- distinct-successor matched-energy confusion;
- exact-shared-successor false-separation diagnostics;
- secondary `E_bank` reported separately with no validity claim.

### Predictor and optimization

- action Jacobian singular spectrum or fixed directional sensitivity summary;
- transition error against each action's own exact legal successor;
- gradient norms and pairwise gradient cosines;
- training stability across seeds;
- parameter count, examples, optimizer steps, GPU time, and peak memory.

### Metric, aggregation, and tie contracts

- Effective rank everywhere uses float64 sample covariance `C=(X-mean(X))^T(X-mean(X))/(N-1)` with `N>=2` and participation ratio `r_eff=(sum eigenvalues)^2/(sum eigenvalues^2)`, using float64 symmetric eigendecomposition and eigenvalues clipped below at zero. A zero denominator gives rank zero. Off-diagonal correlation uses covariance divided by outer standard deviations; any dimension with std `<=1e-8` makes the geometry predicate fail before RMS computation.
- AUROC and AP use the pinned scikit-learn `roc_auc_score` and `average_precision_score` on logits/labels. Calibration uses sigmoid probabilities and 15 equal-width bins on `[0,1]`, left-closed/right-open except the final closed bin; ECE is sample-count-weighted `abs(mean_probability-mean_label)`. Single-class populations are malformed for branch-critical metrics.
- `required precondition predicates` are every `(predicate, argument-position)` Boolean fact used in any action-schema precondition and having at least 20 positive and 20 negative labels in both development and test. Probe candidates, labels, and splits are manifest-frozen.
- Role retrieval uses the detached representation and post-hoc action-binding probe fitted exactly in DL1. It scores every type-compatible object in the source state. A role is correct only when the true object has strictly greater score than every wrong object. Complete-binding retrieval is `1` only when every active role is correct; zero-arity actions are excluded. A tie with a wrong object is failure.
- The normalized binding margin is `(score_true-max_wrong)/(abs(score_true)+abs(max_wrong)+1e-8)` per role, minimum over roles per action, mean over actions per source, then mean sources per problem. Positive means every winning role is strictly correct.
- Legal-distinct pairwise energy accuracy is `1[E_mismatch > E_own]`; equality is failure. It is averaged over triples within source and sources within problem.
- An applicability candidate group contains all frozen applicable and inapplicable action records for one source state. Groupwise top-1 is success only if every action tied for maximum logit is applicable. Top-k uses descending logit and canonical action-byte order for ties.
- Every seed metric is first averaged over source states within problem, then macro-averaged over problems. “At least two of three seeds” is evaluated on those per-seed problem-macro values. Treatment tests instead use the arithmetic mean of the three paired-seed differences within each problem, as specified in DL0.
- A `valid run/seed` means process exit zero plus complete, hash-valid, schema-complete, finite artifacts generated by the preregistered code/config. A converged but below-threshold run is valid evidence and a performance FAIL. Any malformed run sets `evidence_ok=false`; all three official seeds must be valid.
- Action residual scale is `median_schema sqrt(mean_d Var(u_d|schema)) / (sqrt(mean_d Var(u_d))+1e-8)`, with schemas equally weighted. The DL5 gain is an absolute difference in this ratio or in normalized binding margin.

Query-latent robustness uses no planner. For every recorded positive latent `u+`, construct: zero vector; same-state same-schema hard-negative swap; interpolation `(1-alpha)u+ + alpha*u-` for `alpha in {0.25,0.50,0.75}`; and Gaussian perturbations `u+ + sigma*r*epsilon` for `sigma in {0.01,0.05}`, where `r` is development median per-dimension action std and `epsilon` comes from PCG64 keyed by the record SHA-256. Decode only by nearest Euclidean action latent among the frozen explicit candidate set for that state, with action-byte tie order. Query top-1 applicable rate macro-averages all perturbation categories, states, then problems.

For cardinality, candidates are scored only by the selected O3 applicability logit `ell_app`; descending score and canonical action-byte tie order define top K. Source-level applicable Recall@K is the fraction of oracle-applicable actions in top K for `K in {8,32,128}`. A source with zero oracle-applicable actions, missing candidates, or undefined logits makes transfer evidence malformed. Source `required_K` is the smallest listed K with Recall `>=0.90`, else `type_valid_cardinality+1`. The problem point is `(median source type_valid_cardinality using NumPy quantile(method='higher'), 90th-percentile source required_K using method='higher')`. Fit OLS to the log problem points. The scaling upper bound is the 97.5th percentile from 100,000 PCG64(0) problem bootstraps. At least 12 nonempty problems spanning at least three frozen cardinality quartiles are required; otherwise `transfer_ok=false` and `evidence_ok=false`. Any empty population for a branch-critical metric is malformed evidence; sparse-schema reporting metrics are the only skippable populations.

### Cardinality transfer

- candidate count and full type-valid cardinality;
- top-k applicable and trace/competitive-action recall for fixed `K`;
- performance versus object/action cardinality;
- no scalability claim if the required-K exponent upper bound is `>=1.0`.

## 10. Deterministic acceptance and branch decision

All predicates are computed first on development for selection and once on untouched test for confirmation. Missing required metrics, insufficient test power, manifest mismatch, nonfinite values, or unavailable required schema strata set `evidence_ok=false`.

### 10.1 Boolean predicates

```text
evidence_ok =
  DL0 manifest/split/power PASS
  and every required official run/hash exists
  and all 3 of 3 official seeds are valid evidence

state_ok =
  for every required precondition predicate:
    final-z_s AUROC >= 0.80
    and final-z_s AUROC >= graph-feature AUROC - 0.05
  and required predicates cover >=80% of held-out labeled examples
  and >=2 of 3 seeds meet all clauses

state_geometry_ok =
  graph and object target matrices each satisfy:
    >=90% dimensions with std >=0.50
    effective rank >=0.50*D_s
    off-diagonal correlation RMS <=0.20
  in >=2 of 3 seeds

triple_ok =
  legal-distinct pairwise energy accuracy >= 0.80
  and problem-macro accuracy >= 0.80
  and >=2 of 3 seeds meet both

app_ok =
  applicability AUROC >= 0.80
  and groupwise top-1 applicable rate >= 0.80
  and >=2 of 3 seeds meet both

identity_ok =
  complete-binding retrieval >= 0.80
  and problem-macro normalized binding margin > 0
  and >=2 of 3 seeds meet both

geometry_ok =
  one of R1/R2/R3 improves schema-residual scale or normalized binding margin by >=0.10 over R0
  and no semantic metric regresses by >0.02
  and >=2 of 3 seeds improve

prediction_ok =
  held-out positive normalized transition-loss ratio to paired A0/O0 <= 1.05
  in >=2 of 3 seeds

transfer_ok =
  in at least 2 of 3 official seeds independently:
    query/perturbed-latent groupwise top-1 applicable rate >=0.80
    and problem-macro applicable Recall@128 >=0.90
    and that seed's required-K OLS scaling-exponent 97.5% bootstrap upper bound <1.0
```

For exact-shared-successor triples, no separation threshold exists; they must be excluded from `L_gap`. The secondary invalid successor-bank energy never enters these predicates.

```text
dual_space_pass =
  evidence_ok
  and state_ok
  and state_geometry_ok
  and triple_ok
  and app_ok
  and identity_ok
  and geometry_ok
  and prediction_ok
  and transfer_ok
```

### 10.2 Numeric localization recommendations

These flags may co-occur and do not override branch selection:

```text
state_representation_flag =
  raw and graph AUROC >=0.80
  and (final-z_s AUROC <0.65 or graph-to-final drop >0.15)

action_representation_flag =
  state_ok and complete-binding retrieval <0.80

predictor_fusion_flag =
  identity retrieval >=0.80
  and legal-distinct pairwise predictor-energy accuracy <0.60

negative_support_flag =
  O2/O3 improves applicability top-1 by >=0.10 over its paired no-applicability cell
  and reaches >=0.80

target_path_flag =
  fixed-batch target detach changes positive prediction loss or state-encoder gradient norm by >=20%
  and the change repeats in >=2 of 3 fixed seeds
```

Each true flag recommends a separately reviewed next specification; none silently authorizes a redesign.

### 10.3 Ordered branch precedence

The deterministic assessor applies exactly:

```text
1. if not evidence_ok:
     INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D
2. else if dual_space_pass:
     REOPEN_BOUNDED_DUAL_SPACE_DECODING
3. else:
     KEEP_BRANCH_D_ABSTRACT_ACTIONS
```

Localization flags are attached to the record after this selection. `BRANCH_D_ABSTRACT_ACTIONS` remains the operational default unless clause 2 fires. A clause-2 PASS authorizes only a new bounded decoding/planning specification, not planner implementation, tuning, commit, or promotion. Clause 3 terminates this rescue study; no further coefficient or architecture sweep is authorized.

## 11. Evidence, determinism, and review

Every phase requires:

- immutable JSON/JSONL manifests with bytes and SHA-256 hashes;
- fixed seeds, sorted records, and deterministic tie handling;
- exact commands and environment metadata;
- separate development and untouched test outputs;
- atomic publication with no stale artifact reuse;
- behavioral RED/GREEN for every new production boundary;
- independent plan review PASS before implementation;
- independent readiness PASS before official training;
- independent implementation/evidence PASS before branch selection;
- explicit accounting for exclusions, failed runs, and rejected attempts.

Official branch selection must be performed by a deterministic assessor encoding Section 10 precedence. Manual interpretation cannot override it.

## 12. Resource budget

Expected caps, recalibrated in DL0:

```text
DL0–DL1 diagnostics: CPU-bound; several hours
DL2 micro-overfit:   <=1 smoke-equivalent GPU-hour
DL3 objective ablation: exactly 12 smoke checkpoints, about 0.83 GPU-hours at the recorded 0.069/checkpoint
DL4 architecture ladder: <=15 new smoke checkpoints, recorded-runtime lower estimate 1.04 GPU-hours
DL5 anti-collapse: <=12 smoke checkpoints including conditional SIGReg, about 0.83 GPU-hours
DL6 confirmation: baseline + one winner, 3 seeds each, estimated 2–4 GPU-hours
Total GPU cap: 8.0 GPU-hours; DL0 recalibrates per-run estimates and stops before any design would exceed the cap
Engineering/review: approximately 3–6 focused days before any separately specified state redesign
```

These are caps, not entitlements. Sequential FAIL stops later cells. A state-encoder redesign, planner, or candidate generator requires a new specification.

## 13. Deliverables

1. Related-work/theoretical-contract note with primary citations.
2. Power analysis, immutable splits, and frozen action/triple manifests.
3. Layerwise state/action localization and target-path report.
4. Exact-successor energy, applicability, and equivalence report.
5. Predictor intervention, Jacobian, and gradient-conflict report.
6. DL2 micro-overfit record.
7. Fixed DL3 four-cell objective-ablation evidence.
8. Matched DL4 architecture-ladder evidence.
9. DL5 VC/schema-residual-VC and conditional SIGReg evidence.
10. One frozen DL6 confirmation and untouched-test report.
11. Deterministic assessor and human-readable decision record.
12. Independent review history through final PASS or explicit FAIL.

## 14. Non-claims

This specification does not assume or claim that:

- JEPA prediction error automatically assigns high energy to invalid actions;
- VICReg/SIGReg creates grounded semantics;
- isotropic embeddings are valid-state density models;
- every grounded action must have a unique transition latent;
- transition-equivalent legal actions must be separated;
- a decoder is scalable merely because it avoids explicit full enumeration;
- the existing `p166`/`p192` slice establishes generalization;
- a successful representation probe establishes a working planner.

The study is successful if it produces a causal branch decision, including a well-supported negative result.