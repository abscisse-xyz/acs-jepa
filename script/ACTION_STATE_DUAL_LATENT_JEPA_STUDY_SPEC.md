# Action/state dual-latent JEPA causal study specification

Date: 2026-08-05 revision
Status: revised draft awaiting fresh independent review; the prior PASS applied only to the superseded SHA; implementation, training, tuning, planner integration, and promotion are not authorized
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

- `S_g_train,S_o_train` are computed once from the initialization/frozen baseline state encoder before any run and are used inside every `L_gap`, including frozen-state DL2–DL4 and trainable-state DL5A/DL5B/DL6. They are common to paired treatments and never updated.
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
    latent_dim=D_s, action_dim=D_a,  # D_s=D_a=64 outside DL5B
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

`u_final` is the sole post-temporal consumer latent: A0/A2/A3 use the existing temporal-GRU output, A1 applies its fixed LayerNorm to that output, and A4/A4_SHAM feed their fused/sham per-step composer tensor through that same temporal GRU. Predictor, applicability, transition losses, and action LocalSIGReg consume only this `u_final`; no pre-GRU tensor is silently substituted. Argument rows are padded to manifest maximum arity with zeros and mask `false`; real roles follow schema order. The same frozen-state object latents/masks are supplied in every architecture, including A3, because applicability is explicitly pair-conditional. Head parameters use the local module-seed rule and train only in O2/O3.

`N_positive` and `N_negative` are training-manifest counts before batching. Both must be nonzero. `pos_weight` is float64, hash-frozen, and unclipped. Batches preserve manifest proportions; no class resampling is applied. Development F1 uses the smallest distinct observed development logit attaining maximum problem-macro F1, with candidates `{-inf} U observed_logits U {+inf}`; the threshold is frozen for test. AUROC/AP and groupwise ranking use logits directly.

`E_app` is the sole primary inference-time validity score and is never directly minimized. Applicability supervision in every O2/O3 training objective is exclusively class-weighted `L_app`; “adding applicability” means adding `0.10*L_app`, never adding raw `E_app`. Exact offline symbolic/simulator labels are allowed only for frozen construction/evaluation, never production candidate generation.

The fixed objective ablation is:

```text
O0 = positive transition prediction with signed frozen state encoder; state LocalSIGReg diagnostic only; default action LocalSIGReg enabled
O1 = O0 + 0.10 * legal transition-triple L_gap
O2 = O0 + 0.10 * supervised applicability BCE L_app
O3 = O0 + 0.10 * L_gap + 0.10 * supervised applicability BCE L_app
```

These four cells use identical architecture, matched initialization, minibatch order, examples, optimizer budget, seeds, and default action LocalSIGReg. The state encoder is frozen and state LocalSIGReg is diagnostic/no-grad during this ablation. The later regularizer phase tests the necessity and locality of the default; it does not choose the default post hoc.

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
q_t = Q_current(action id, selected z_s object latents)
h_t = ActionEncoder.temporal_GRU(q_t, h_{t-1})
u_final = h_t
z_pred = G(z_s, u_final)
```

### A1: Geometry-only decoupling

Keep `D_s=D_a=64`, action IDs, selected `z_s` inputs, the pinned baseline `action_encoder.kind='pooled'` per-step composer, temporal `ActionEncoder` GRU, predictor, and every trainable parameter unchanged. Apply `LayerNorm(64, elementwise_affine=False, eps=1e-5)` to the temporal-GRU output immediately before predictor/applicability use: `u_final=LayerNorm(h_t)`. It adds zero trainable parameters. This isolates action-space normalization; widths vary only in the separately controlled DL5 sensitivity matrix.

### Frozen descriptor tensor

A2–A4 use one literal descriptor per grounded argument object:

```text
d_obj = concat(
  one_hot(object_type, sorted training-domain type vocabulary),
  sinusoidal_PE16(canonical object ordinal),
  static_incidence
)
```

The canonical ordinal is the zero-based index after UTF-8 bytewise sorting object names within a problem. For `k=0..7`, `sinusoidal_PE16[2k]=sin(ordinal/10000^(2k/16))` and `[2k+1]=cos(ordinal/10000^(2k/16))`, computed float64 then cast float32. A predicate is static iff it appears in no add or delete effect of any parsed domain action schema. `static_incidence` has one float32 coordinate for every `(static predicate, argument position)` pair sorted bytewise; its value is the count of true static atoms containing the object in that position divided by the maximum training-manifest count for that coordinate, with zero when the maximum is zero. Dynamic predicates and successor facts are excluded. Unknown development types/static coordinates are a preflight FAIL; an unknown test coordinate discovered only after the frozen holdout command sets `test_evidence_ok=false` and cannot alter the vocabulary. Each argument also receives a one-hot role position of width equal to maximum domain action arity. Descriptor schema, maxima, vocabularies, object ordinals, and bytes are manifest-hashed.

Outside DL5B, every newly instantiated trainable parameter uses the exhaustive class-specific initializer in the DL5B subsection below. Its local CPU generator key is the fully qualified state-dict parameter path (plus `:r`, `:z`, or `:n` for a GRU gate), seeded by the first unsigned 64 little-endian bits of `SHA256(UTF8(str(run_seed)+":"+key))`; biases/LayerNorm/GINE `eps` use the exact constants there. Existing modules load byte-identical paired baseline weights, and any new parameter not consumed exactly once by that initializer is malformed. This rule includes production/control applicability heads and post-hoc argument heads, so embeddings and recurrent parameters never retain constructor defaults. All DL4 architecture-ladder cells use `D_s=D_a=64`, `action_encoder.kind='pooled'`, hidden width `64`, dropout `0`, argument tensors in action-schema order, and parameter counts are reported. No recurrent argument encoder is allowed in the ladder.

### A2: Descriptors plus current state conditioning

For each argument role, replace the current per-step object-projector input `z_s_object` by `concat(z_s_object, d_obj, one_hot(role))`; change only that projector input width. Schema embedding, argument order, pooled composer, its per-step output width, the existing temporal GRU, predictor, and losses stay fixed. The resulting composer output `q_A2,t` enters the unchanged temporal GRU; its output is `u_final`.

### A3: Separate identity tower

Use the identical A2 module and parameter shapes, but replace the `z_s_object` segment with a zero tensor before the projector:

```text
q_identity,t = Q_A2(schema, role, concat(zeros(D_s), d_obj, one_hot(role)))
u_final = ActionEncoder.temporal_GRU(q_identity,t, h_{t-1})
z_pred = G(z_s, u_final)
```

Because A2 and A3 have byte-identical parameter shapes and paired initialization, A2→A3 isolates removal of final-state object content while preserving explicit grounded descriptors.

### A4: Identity plus state-conditioned effect context

Keep A3 per-step `q_identity,t`. A separate context composer uses the original A0 selected `z_s` object inputs and pooled composer to produce `q_context,t in R^64`. Fusion occurs before the unchanged temporal GRU:

```text
q_fused,t = LayerNorm(64, elementwise_affine=False, eps=1e-5)(Linear_128_to_64(concat(q_identity,t, q_context,t)))
u_final = ActionEncoder.temporal_GRU(q_fused,t, h_{t-1})
z_pred  = G(z_s, u_final)
```

The predictor, applicability head, transition/applicability energies, and action LocalSIGReg always consume `u_final`, the post-temporal consumer latent. Primary complete-binding retrieval also reads `u_final`; a separately named identity-tower localization diagnostic reads `q_identity,t`. A matched `A4_SHAM` has identical modules, parameter shapes, initialization, optimizer, and temporal GRU but replaces `q_context,t` by zeros before fusion. A4 is evaluated only by paired A4−A4_SHAM differences.

A4/A4_SHAM are authorized only when A3 development complete-binding retrieval is at least `0.80`, A3 positive prediction ratio versus A0 is at most `1.05`, and either A3 legal-distinct pairwise accuracy or applicability groupwise top-1 is below `0.80`. Otherwise both are recorded `NOT_AUTHORIZED`.

Every adjacent comparison uses paired seeds, data order, optimizer budget, initialization for unchanged modules, literal tensors above, and reported parameter counts. No unspecified capacity-matching module is permitted.

### Explicit exclusions

This study does not authorize a planner, continuous latent optimization, inverse-decoder promotion, broad Phase 1 training, a production candidate generator, production simulator access, option/skill learning, or simultaneous changes to state encoder, action encoder, predictor, regularizer, and dataset.

## 6. Frozen action and triple taxonomy

The manifest separates state-action validity from transition-triple compatibility.

Action records and one diagnostic relation:

1. `APPLICABLE_RECORDED`.
2. `APPLICABLE_ALTERNATIVE`.
3. `INAPPLICABLE_ONE_ROLE`.
4. `INAPPLICABLE_ROLE_SWAP`.
5. `INAPPLICABLE_RANDOM_TYPE_VALID`.
6. `ARGUMENT_DISTANCE_CONTROL` (pairwise diagnostic tag only; never a third applicability label or additional BCE row).

Triple records:

1. `LEGAL_OWN_SUCCESSOR`: `(s, a_i, canonical_successor(s,a_i))`.
2. `LEGAL_DISTINCT_MISMATCH`: `(s, a_i, canonical_successor(s,a_j))`, where both actions are legal and full canonical successors differ.
3. `LEGAL_EXACT_SHARED_SUCCESSOR`: two legal actions whose full canonical successor is byte-equivalent after canonicalization.
4. `LEGAL_SHARED_EFFECT_DIFFERENT_STATE`: same canonical effect signature but different full canonical successor; diagnostic only until semantics are reviewed.
5. `INAPPLICABLE_NO_SUCCESSOR`: `(s,a_invalid)` with no transition-triple target and applicability labels only.

For each source state, enumerate the complete type-valid grounded-action universe in canonical action-byte order using declared parameter types, then obtain the Boolean applicability label from the offline oracle. Construction is exact:

1. `APPLICABLE_RECORDED` is every applicable action appearing as the trace action for that canonical source; duplicate windows do not duplicate the action.
2. `APPLICABLE_ALTERNATIVE` is every other oracle-applicable action.
3. `INAPPLICABLE_ONE_ROLE` candidates are generated from every applicable anchor by replacing exactly one active role with each different type-compatible object, retaining only oracle-inapplicable results.
4. `INAPPLICABLE_ROLE_SWAP` candidates are generated from every applicable anchor by swapping each ordered pair of distinct role positions `i<j` when both objects satisfy the opposite role type, retaining only changed, oracle-inapplicable results.
5. `INAPPLICABLE_RANDOM_TYPE_VALID` is the remaining oracle-inapplicable type-valid universe after categories 3 and 4; “random” is historical naming only.

Deduplicate by canonical action bytes before caps. Applicability truth has precedence over any generator tag; among positive tags `RECORDED > ALTERNATIVE` (`positive_tag_rank` 0 and 1 respectively), and among invalid tags `ONE_ROLE > ROLE_SWAP > RANDOM_TYPE_VALID`. A collision with inconsistent oracle labels is malformed. Within each capped class (one combined positive class and three separate invalid classes), create one bucket per `action_schema_id`; sort positive buckets by `(positive_tag_rank,SHA256(source_hash||UTF8(category)||action_bytes),action_bytes)` and invalid buckets by `(SHA256(source_hash||UTF8(category)||action_bytes),action_bytes)`; sort nonempty schema IDs bytewise; and repeatedly take one action from each schema bucket in that order until the class cap or exhaustion. Training retains at most 32 positives total and 32 actions from each invalid category per source; development/test retain at most 256 positives and 256 from each invalid category per source. If a class has fewer records, retain all; no backfilling transfers quota between classes. Manifests store pre-cap and post-cap counts by source/schema/category plus payload hashes. `N_positive`, `N_negative`, BCE weighting, and applicability candidate groups use only these deduplicated retained Boolean action rows; every source used for branch-critical applicability metrics must retain at least one positive and one negative.

After retention, `ARGUMENT_DISTANCE_CONTROL` contains no new action row. For every retained positive/negative pair with the same schema, record active-role Hamming distance and `pair_hash=SHA256(source_hash||min(action_pos_bytes,action_neg_bytes)||max(action_pos_bytes,action_neg_bytes))`. For each `(source,schema,distance,negative_category)`, retain the lowest pair hash for diagnostics of score versus argument distance. These pairs never enter `L_app`, `L_gap`, AUROC/AP, or candidate-group counts independently; their underlying action rows retain their ordinary Boolean labels.

Canonical state bytes are UTF-8 JSON arrays of fully grounded atoms. Each atom is `[predicate,arg1,...]` with original case-sensitive strings; atoms are sorted by their compact JSON UTF-8 bytes. The outer JSON uses `ensure_ascii=false`, separators `(',',':')`, and one final newline. Effect bytes are exactly `{"add":[...],"delete":[...]}` with keys in that order and atom arrays encoded/sorted identically. Full-successor equality is canonical-state byte equality; effect-signature equality is never substituted.

For each source, the offline oracle enumerates all applicable actions for evidence only, executes each once, groups by full-successor bytes, sorts groups by successor bytes, and sorts actions within each group by canonical action bytes. Every action contributes one `LEGAL_OWN_SUCCESSOR`. For each unordered pair of different successor groups, the lower-successor-byte group is `L`, the higher is `H`, and their first actions are `a_L,a_H`; form both ordered mismatch directions. Every mismatch record stores source bytes/hash, `action_i_bytes`, `successor_i_bytes/hash`, `action_j_bytes`, `successor_j_bytes/hash`, both schema IDs, and effect-relation category. Thus `E_mismatch=E_trans(s,a_i,s_j')` and `E_own=E_trans(s,a_j,s_j')` always have an explicit comparator. Exact-shared-successor actions are exclusion controls and never enter `L_gap`.

Each source keeps at most 128 ordered training directions. Direction pairs are indivisible. A pair unit has orientation-independent hash `SHA256(source_hash || successor_L_hash || action_L_bytes || successor_H_hash || action_H_bytes)`. Its stratum key is compact UTF-8 JSON `[min(schema_L,schema_H),max(schema_L,schema_H),effect_relation]`. Strata are traversed in ascending stratum-key bytes; units within each stratum are ascending pair-unit hash; round-robin takes one unit per nonempty stratum until 64 units or exhaustion. Evaluation uses the same ordering with a 2,048-unit cap.  Manifests record total/retained units and payload hashes. Losses average directions within source then sources; metrics macro-average sources then problems. No successor-group-size weighting is applied.

Action bytes are UTF-8 compact JSON `[schema,arg1,...]` with original case-sensitive strings and one final newline. Invalid-action records never enter triple construction. Nearest-latent negatives remain diagnostic only.

Temporal ownership is explicit. Primary positive-transition rollout windows use the existing temporal GRU recurrence over their ordered action sequence with zero initial hidden state at each canonical window start. Every source-local applicability row, gap unit, candidate-group metric, and oracle/probe action forward is a separate one-step candidate evaluation with the temporal hidden state reset to the all-zero tensor; all actions compared at one source therefore share the identical zero predecessor hidden state. A4 fusion occurs before this temporal step and A1 normalization after it. No hidden state is carried between source records, classes, batches, problems, or auxiliary updates. Manifests persist the reset flag, zero-tensor shape/dtype/hash, action order, and resulting A5/A6 hashes. The temporal trace over primary windows and the reset candidate trace are reported separately; neither may be substituted for the other.

## 7. Study phases and fixed treatment designs

### Common optimization protocol

Official seeds are exactly `[0,1,2]`. PyTorch deterministic algorithms are enabled; cuDNN benchmark and TF32 are disabled. Every official bounded training cell uses `torch.optim.Adam(lr=lr_base=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0,amsgrad=false,maximize=false,foreach=false,capturable=false,differentiable=false,fused=false)`, rollout length `4`, and no gradient clipping; batch size is `24` except for the registered DL5B batch factor. DL2 alone uses its stated constant `lr_base` with no schedule. Every other trained cell preflights `N_steps>=3`, sets `W=ceil(0.10*N_steps)` and requires `1<=W<=N_steps-2`, and assigns the learning rate used by zero-based optimizer update `q` before that update's forward/backward/`optimizer.step()`; no scheduler `.step()` is called afterward:

```text
lr_start = 1e-8 * lr_base
lr_end   = 1e-5
if 0 <= q < W:
    lr(q) = lr_base                                      if W == 1
            else lr_start + (lr_base-lr_start)*q/(W-1)
if W <= q < N_steps:
    lr(q) = lr_end + 0.5*(lr_base-lr_end) *
            (1 + cos(pi*(q-W)/(N_steps-W-1)))
```

Thus update `W-1` and the first decay update `W` both use exactly `lr_base`, and update `N_steps-1` uses exactly `lr_end`. Set every Adam parameter group's `lr` to float64-evaluated `lr(q)` cast to that group's Python float before the forward; persist the complete LR array and its hash. There is no early stopping or best-checkpoint selection: the final update is evaluated.

The canonical training-window manifest contains every valid rollout window from all 130 training problems, sorted first by problem ID and then record hash. One bounded smoke-equivalent pass consumes every manifest record exactly once with no replacement or `drop_last`; problem-balanced round-robin interleaves problem queues ordered by `SHA256(UTF8(str(seed)+":"+record_hash))`. `N_steps=ceil(N_records/batch_size)`. The final short batch is retained and its size recorded. Every development evaluation consumes every valid window from all 48 development problems exactly once in canonical order. A cell that omits a problem or record is malformed evidence.

Auxiliary losses have independent deterministic producers tied to those same `N_steps`. For applicability, sort every retained training action row by its canonical manifest bytes; let `N_app` be that nonzero row count and assign update `q=0..N_steps-1` the half-open slice `[floor(q*N_app/N_steps),floor((q+1)*N_app/N_steps))`. Load each row's canonical source state and grounded action, compute its production applicability logit, and use `L_app_step(q)=(N_steps/N_app)*sum_row BCEWithLogits(logit,label,pos_weight)` with the frozen global `pos_weight`; an empty slice is a connected scalar zero. For gap supervision, one indivisible unit is one source state together with all retained ordered legal-distinct directions for that source. Sort units by canonical source bytes; let `N_gap` be the nonzero unit count, use the same floor partition, compute each unit loss as the mean hinge over all its retained directions, and use `L_gap_step(q)=(N_steps/N_gap)*sum_unit unit_loss`. Thus the arithmetic mean over a pass is exactly the manifest row-mean applicability loss and equal-source gap loss specified in Section 6 despite unequal final slices.

At each enabled update, load the minimal deduplicated union of canonical source/action/successor records for that auxiliary slice and perform one attached auxiliary forward under current parameters; identical encoded states/actions within that forward may be memoized only when the resulting tensor is reused without detach or recomputation. O2/O3 attach `L_app_step`; O1/O3 attach `L_gap_step`; disabled cells do not instantiate the corresponding production head/loss. Every enabled auxiliary row/source is consumed exactly once per complete pass, and DL6 repeats the same partition in each of its four passes. Persist per-pass `{loss_kind,q,start,end,record_or_source_hashes,N_app,N_gap,N_steps,forward_hash}` before the update. Missing/duplicate records, changed ordering, cycling, resampling, source splitting, an omitted final slice, or a manifest mismatch is malformed. In Sections 2 and 8, per-update `L_app` and `L_gap` mean these scheduled step scalars; their arithmetic pass means equal the declared full-manifest objectives.

DL3/DL4 cells load the same DL0-pinned state/action/predictor weights; the state encoder is frozen, while action/predictor/new heads train for one complete canonical pass. DL4 cells do not warm-start from earlier architecture cells. After DL4 selects only an architecture identity, every DL5A S0-S3 cell reconstructs that architecture from the same DL0-pinned pre-DL3 state/action/predictor weights and identical module-seed initialization; it loads no trained DL3/DL4 action, head, or predictor weight. DL5A then unfreezes state/action/predictor together and trains one complete matched pass, so no comparator inherits S0 treatment. DL5B uses its own dimension-derived matched initialization.

DL6 baseline and winner both reload the same pinned initialization, unfreeze state, and train one uninterrupted optimizer trajectory of exactly four complete passes. Let `M=ceil(N_records/24)` and `N_steps=4*M`; the single LR array in the common protocol is constructed once over global updates `q=0..4*M-1`, and model, Adam moments, LR phase, and global update index never reset at a pass boundary. For pass `p in {0,1,2,3}`, order each problem queue by `SHA256(compact_json([seed,p,record_hash])+"\n")`, apply the same problem-balanced round-robin, consume every canonical training window exactly once, retain that pass's final short batch, and map local batch `j=0..M-1` to `q=p*M+j`. Winner and baseline use byte-identical pass orders for each paired seed. The applicability/gap producer repeats its canonical floor partition with local `j` once per pass, consumes every auxiliary row/source exactly once in that pass, and persists `p,j,q`; its losses retain the Section 7 one-pass normalization, not an extra factor four. The geometry scheduler likewise receives `N_steps=M` separately for each pass and its existing key includes `p`; it covers every eligible group in every pass, while its update artifact also records global `q`. Boundary trace percentage updates use total `N_steps=4*M`. Only the checkpoint after `q=4*M-1` is evaluated. Any reset, one-pass LR replay, pass-independent transition hash, omitted/repeated pass record, or alternative local/global indexing is malformed. Any OOM, record filtering, gradient accumulation, or nondeterministic-kernel workaround changes the protocol and requires spec review rather than silent substitution.

DL2 uses Adam with the same scalar hyperparameters, no scheduler, one full merged-micro transition batch, and exactly 2,000 steps; final step only is assessed. The pinned state encoder is frozen. Construct exactly one `DL2_MICRO_MANIFEST`, shared byte-for-byte by all architecture/objective runs, rather than one run per schema. Iterate required schemas in UTF-8 byte order. For each schema, traverse source states by SHA-256 and retain the first four having at least two same-schema applicable actions with distinct full successors and two same-schema inapplicable one-role substitutions; retain the first two legal and first two invalid actions per source by action-byte order, both ordered legal mismatch triples, and all own/applicability records. If exact-shared-successor alternatives exist for that schema in the oracle pool, append the first pair by canonical pair bytes as an exclusion control. Prefix every selected record with its required-schema ID, serialize it under the Section 6 record formats, concatenate schema lists in schema-byte order, then sort the merged records by full record bytes. Byte-identical records deduplicate once while retaining the sorted provenance-schema array; two payloads for the same canonical record key or one record carrying conflicting labels/successor hashes are malformed. A required schema lacking the complete construction makes the single merged manifest malformed and all DL2 cells FAIL. The merged manifest must contain, for every required schema, at least one eligible `(problem,schema)` action group with four distinct legal atoms and its eligible `(domain,schema)` group with four distinct legal atoms. Eligibility and PASS metrics are computed both on the merged manifest and separately for every required schema; global success cannot hide a per-schema failure. These rules preserve exactly 16 A0-A3 architecture×objective runs, with at most eight later A4/A4_SHAM runs, never multiplied by schema count.

### DL0: Evidence, split, and power-code preflight

Deliverables:

- immutable train/development/untouched-test problem splits;
- schema and action-cardinality-stratum coverage in every split;
- frozen action/triple manifests and exact canonicalization audit;
- per-schema independent sample counts and candidate counts;
- parameter, optimizer-step, runtime, and memory accounting;
- exact SHA-256 pin of the accepted Updated-Phase-0 baseline checkpoint referenced by `ACTION_LATENT_UPDATED_PHASE0_DECISION.md`; no validation-based checkpoint reselection;
- a preregistered detectable-effect/power formula whose development contrasts are computed only after DL3, before DL4 authorization.

The inferential unit is an independent PDDL problem. The study universe is the complete 198-problem campaign inventory under `/opt/data/workspace/acs-jepa-tuning-data` (the root may relocate only when file bytes/hashes remain identical), not the historical 12-problem smoke subset. Authoritative manifest SHA-256 values are:

```text
campaign_manifest.json   eef4ec9c26422b87e57386c57634e40cbeff12f1298bbb60f86e1a4b3722d989
full-dev/manifest.json   8ac0f6f99d8e376ae8da50e3087327cbc1f5324c4716195734f7805613970453
development/manifest.json 9d5767bace6728dcc193bdf41519714a1ee0910c7322621347abbb6a24aa29d5
final-test/manifest.json  7a8ba83ed35618b170754f29ffa6a4fadb83f4a1a90791093c802599398434d9
```

Split ownership is literal:

```text
untouched test = every one of the 20 problems in final-test/manifest.json
selection/development holdout = every one of the 48 problems in development/manifest.json
training = the 130 problems in full-dev/manifest.json minus those 48 development problems
```

The three sets must be pairwise disjoint, their union must equal the 198 campaign problems, and their manifest/source fingerprints must match the campaign manifest. Any absent, extra, duplicated, unreadable, or silently filtered pretest problem makes `pretest_evidence_ok=false`; the corresponding defect discovered only in the frozen test command makes `test_evidence_ok=false`. “Smoke” henceforth describes a bounded optimizer/runtime budget only: every smoke cell trains over all available windows from all 130 training problems and evaluates every development metric on all 48 development problems. No cell may use only `p166`, `p192`, the old 12-problem smoke manifest, or a convenience subset as official evidence. Every final-test metric is evaluated once on all 20 untouched problems; no test problem may influence architecture, width, batch size, coefficient, threshold, checkpoint, or branch selection. Within-problem action/triple caps from Section 6 remain permitted, but problem-level omission does not.

Before test access, freeze and hash the implementation commit, environment lock, train/development manifests, canonicalization code, projection/group artifacts, baseline and selected checkpoints, architecture/objective/regularizer/width/batch choices, all probes/metrics/thresholds, deterministic assessor, and exact final command. Preflight may read final-test manifest membership/counts/hashes only; training, development, debugging, power estimation, and selection processes must not load final-test PDDL contents or generated labels. After test metrics are observed, a failed run cannot be repaired/retried unless independent review verifies that the failure occurred before any metric was materialized and that the correction cannot affect selection; otherwise the decision is `INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D`.

Training cardinality quartiles are computed from median type-valid action count per training problem using deterministic NumPy `quantile(method='linear')`; development must contain at least two problems per quartile. A schema is `required` only if it occurs in at least four training and four development problems. These sets are frozen before test access; test content may not remove a required schema. If final test lacks a required schema or three training-cardinality quartiles, `test_evidence_ok=false` rather than redefining coverage.

DL0 freezes the following inferential policy and code before any objective-ablation result exists; it does not pretend that the DL3 contrast variances are already available:

```text
alpha = 0.05 family-wise, one-sided for four directional DL3 contrasts
confidence intervals = 95% two-sided
power = 0.80
minimum detectable macro-metric difference = 0.10
randomization unit = problem
seed aggregation = arithmetic mean of the three paired-seed problem metrics
```

The four hypothesis IDs, in tie order, are `H1=O1-O0` and `H2=O3-O2` for legal-distinct pairwise accuracy, and `H3=O2-O0` and `H4=O3-O1` for matched control-probe applicability groupwise top-1. For each `h`, the statistic is the arithmetic mean paired problem difference. For `n<=20`, enumerate all `2^n` sign flips and set `p_h=count(T_perm>=T_obs)/2^n`. For larger `n`, draw exactly 1,000,000 PCG64(0) sign vectors and set `p_h=(1+count(T_perm>=T_obs))/(1+1_000_000)`. Sort by `(p_h,hypothesis_ID)` and compute Holm adjusted values `p_adj(k)=max_{j<=k} min(1,(4-j+1)*p_(j))`. Eligibility requires strict `p_adj<0.05` plus numeric gates.

For each contrast separately, compute sample standard deviation `sigma_dev_h` with `ddof=1` over development problem differences and:

```text
n_power_h = ceil(((1.6448536269514715 + 0.8416212335729144) * sigma_dev_h / 0.10)^2)
n_required = max(12, n_power_H1, n_power_H2, n_power_H3, n_power_H4)
```

Immediately after all 12 DL3 checkpoints exist, and before DL4 authorization, a mandatory `DL3P` gate computes the four development problem-difference vectors, `sigma_dev_h`, `n_power_h`, and `n_required` with the frozen DL0 code. Any nonfinite variance or fewer than two development problem differences is malformed evidence. `power_ok = (48 >= n_required)`, because these objective contrasts are decided only on the 48-problem development split and are never recomputed on test. If false, later cells and test are not run and the decision is `INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D`. For each contrast, 95% intervals use exactly 100,000 PCG64(0) problem-block bootstrap resamples, recomputing the three-seed mean, with NumPy `quantile(method='linear')` endpoints `[0.025,0.975]`. `p166`/`p192` remain historical. Test runs once after freeze.

DL0 PASS requires deterministic repeat manifests, zero split overlap, complete hashes, all required development cardinality strata, exact 130/48/20 membership, and independent review of lineage plus the frozen power implementation/fixtures. It does not require unavailable DL3 effect estimates. DL3P PASS later requires complete hash-valid contrast artifacts and `power_ok=true`.

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
- a fixed-capacity oracle substitution probe matrix fitted on training records and evaluated on every development problem:

```text
learned z_s + learned action features
raw/oracle state facts + learned action features
learned z_s + structured oracle action descriptors
raw/oracle state facts + structured oracle action descriptors
```

The probe architecture, optimizer, examples, and seeds are identical across cells. It is diagnostic only and does not alter JEPA checkpoints. This matrix localizes state, action, and predictor/probe capacity ceilings.

#### Mandatory component-boundary collapse trace

Instrumentation must capture these exact tensors without changing forward values:

```text
S0 raw symbolic node/edge/fact controls
GN0kind/GN0type/GN0pred/GN0arity/GN0object node categorical-embedding outputs
GN1sum summed node embeddings; GN2ln node_projection LayerNorm; GN3linear node_projection Linear; GN4relu node_projection ReLU output
GE0role/GE0direction edge categorical-embedding outputs; GE1sum summed edge embeddings
GE2ln edge_projection LayerNorm; GE3linear edge_projection Linear; GE4relu edge_projection ReLU output
GM[k]0 GINEConv output before residual; GM[k]1 residual-add output; GM[k]2 LayerNorm output; GM[k]3 ReLU output, for every message-passing layer k
S3 GraphEncoder.output_projection node rows
S4mask object-node Boolean mask; S4g global_mean_pool graph_embedding; S4o object-mask-selected object_embeddings
SPg_ln/SPo_ln GraphStateProjector graph/object MLP LayerNorm outputs; SPg_l1/SPo_l1 first Linear outputs; SPg_relu/SPo_relu ReLU outputs
S5g/S5o GraphStateProjector second-Linear outputs immediately before temporal GRUs; S6g/S6o StateEncoderF graph/object GRU outputs
A0 action_embedding rows
A1 gathered final-state object argument rows
A2in per-role object-projector input, including exact state/zero/descriptor/role segments; A2linear object-projector Linear output
AT[t]role role-embedding output; AT[t]argmod role-modulated argument rows; AT[t]mask cast mask; AT[t]sum/AT[t]count/AT[t]pool masked sum/count/mean; AT[t]concat head-plus-pool concatenation
AT[t]ln/AT[t]l1/AT[t]relu composer output-MLP LayerNorm/first-Linear/ReLU outputs, for each executed tower t in {A3single,A4id,A4ctx}
A3/A4id/A4ctx respective tower second-Linear outputs; A3 exists in A0-A3 only and A4id/A4ctx exist in A4/A4_SHAM only
A4fuse_concat identity/context concatenation; A4fuse_linear fusion Linear output; A4fuse_ln fusion LayerNorm output when present
A5 ActionEncoder temporal-GRU output before optional A1 normalization
A6ln A1-only post-GRU LayerNorm output; A6 sole post-temporal consumer latent (`u_final`; A6=A6ln in A1 and A6=A5 otherwise)
P0g/P0o exact graph/object concatenated predictor inputs
P1g_ln/P1o_ln predictor MLP LayerNorm outputs; P1g_l1/P1o_l1 first Linear outputs; P1g_relu/P1o_relu ReLU outputs; P1g_l2/P1o_l2 second Linear delta outputs
P2g/P2o residual-add predicted successor graph/object outputs for every rollout order
Tg/To observed target final-state graph/object latents
Happ0g/Happ0a/Happ0o raw applicability graph/action/gathered-object inputs
Happ1maskcast argument mask after device/dtype cast; Happ1mask cast mask after `unsqueeze(-1)`
Happ1slotids integer `arange(arity)`; Happ1slotembed slot-embedding output; Happ1slotdtype post-dtype-conversion rows; Happ1slotunsq post-`unsqueeze(0)` rows; Happ1slot expanded slot rows
Happ1slottanh `tanh(Happ1slot)`; Happ1objmod exact elementwise `Happ0o*Happ1slottanh`; Happ1role exact `concat(Happ0o,Happ1slot,Happ1objmod)`
Happ1ctx_ln/Happ1ctx_l1/Happ1ctx_relu/Happ1ctx_l2 object_context MLP outputs
Happ1masked exact `Happ1ctx_l2*Happ1mask`; Happ1masksum raw mask sum; Happ1count post-`clamp_min(1.0)` count; Happ1projsum projected-row sum; Happ1pool exact `Happ1projsum/Happ1count` object summary
Happ1diffsub exact `Happ0g-Happ1pool`; Happ1diffabs exact `abs(Happ1diffsub)`; Happ1scorein exact `concat(Happ0g,Happ0a,Happ1pool,Happ1diffabs)`
Happ1linear scorer first Linear; Happ1gelu GELU; Happ1drop dropout output; Happ2linear final Linear `[B,1]`; Happ2 post-`squeeze(-1)` scalar logit
Harg0a action-projection rows; Harg0o object-projection rows; Harg0r role-embedding rows; Harg1sum broadcast sum; Harg1gelu GELU; Harg1drop dropout; Harg2linear pre-mask role/object scores; Harg2mask post-mask scores
L0g[k]/L0o[k] per-rollout-order graph/object MSE; L0go[k] weighted graph+object order scalar; L0w[k] order-weighted scalar; L0sum weighted-order numerator; L0den weight denominator; L0pred normalized positive-transition scalar
L1own/L1mismatch energies; L1delta signed gap; L1hinge per-direction hinge; L1source equal-direction source mean; L1step scheduled equal-source scalar
L2logit/L2bce per-row applicability logit/BCE; L2step scheduled manifest-row scalar
L3component/L3group/L3step state-SIGReg component, group, and scheduled-wrapper scalars; L4component/L4group/L4step corresponding action-SIGReg scalars
L5pred/L5gap/L5app/L5stategeom/L5actiongeom coefficient-weighted objective terms; L5total final scalar; L5grad parameter-group gradients
QUERY each perturbed/query action latent when query diagnostics run; swap/interpolation QUERY IDs alone may carry exact `NOT_APPLICABLE_NO_SAME_SCHEMA_NEGATIVE` under Section 9
```

Every ID above is a separate persisted tensor/scalar, not a slash-combined implementation hook. Module forward hooks capture module outputs; side-effect-free identity taps capture functional sums, residual additions, masks, pooling/division, concatenations, loss arithmetic, and ReLUs without detach, clone-based substitution, or changed forward values. Indexed IDs emit one artifact per actual layer/order/component. An absent optional architecture tensor is `NOT_APPLICABLE`; an edge-path row may be `NOT_APPLICABLE_EMPTY` only for a graph with exactly zero edges, while its shape/count and the aggregate nonempty edge population remain mandatory. `Harg2mask` must be finite at every true candidate-mask coordinate and exactly `-inf` at false coordinates; those intentional sentinels are persisted and excluded from vector geometry rather than treated as nonfinite collapse. Forward hooks and taps must be keyed to these named module/operation paths and fail if a required boundary is absent, duplicated, shape-incompatible, conflated with another ID, or renamed without a reviewed registry update. The trace manifest uses every development problem. At initialization and after updates `{1,ceil(.10N),ceil(.25N),ceil(.50N),ceil(.75N),N}` (deduplicated and sorted), it selects the first 16 windows per development problem by record hash; the final checkpoint additionally traces every valid development window. After the holdout is opened, the frozen final baseline/winner traces every valid window from every one of the 20 test problems exactly once, with no intermediate test trace. Sample identity, problem, state, object, schema, role, rollout order, and checkpoint/update hashes are retained so rows remain aligned across boundaries.

For every numeric boundary globally and under its applicable local groups, emit: row/dimension counts; finite fraction; coordinate mean/std quantiles; fraction with std below `{1e-3,0.05,0.50}`; covariance eigenspectrum; participation-ratio effective rank and `rank/D`; off-diagonal correlation RMS; mean norm; median pairwise distance; mean off-diagonal cosine; unique-row fraction at float32 byte equality; LocalSIGReg value; and semantic probe metrics appropriate to that boundary. Local groups are problem for graph/state/predictor graph rows, `(problem,state_hash,object_type)` and `(problem,object_type)` for object rows, and both `(problem,schema)` and `(domain,schema)` for action rows. Also emit linear CKA for each aligned adjacent pair and the predecessor-to-successor probe drop.

Boundary flags are literal:

```text
complete_collapse(b) =
  nonfinite values
  or rank_ratio <=0.10
  or fraction(std<=1e-3) >=0.90
  or median_pairwise_distance <=1e-3
  or sqrt(trace(C_global)/D) <=1e-8

dimensional_collapse(b) =
  rank_ratio <=0.50
  or fraction(std<=0.05) >=0.25

conditional_residual_scale(b,f) =
  sqrt(trace(C_equal_group(b,f))/D)/(global_scale+1e-8)

conditional_collapse(b,f) =
  participation_rank(C_equal_group(b,f))/D <=0.25
  or conditional_residual_scale(b,f) <=0.10

conditional_collapse(b) = OR_f conditional_collapse(b,f)

where X_g is the float64 matrix of canonical rows in group g,
      R_g = X_g - 1*mean_rows(X_g),
      C_equal_group(b,f) = mean_g (R_g^T R_g/(n_g-1)) over eligible n_g>=4 groups in family f,
      C_global is the float64 sample covariance over all global rows at boundary b,
      and global_scale = sqrt(trace(C_global)/D)

semantic_drop(prev,b) =
  prev required-fact/binding AUROC >=0.80
  and (b AUROC <=0.65 or prev-to-b drop >=0.15)
```

Registered semantic-drop paths are exact. Aligned node/fact probes follow `S0->GN1sum->GN2ln->GN3linear->GN4relu->GM[k]0->GM[k]1->GM[k]2->GM[k]3->S3`, with each indexed message layer continuing into the next. From S3, the graph path is `S3->S4g->SPg_ln->SPg_l1->SPg_relu->S5g->S6g` and the object path is `S3->S4o->SPo_ln->SPo_l1->SPo_relu->S5o->S6o`; S3-to-S4 comparisons use the persisted graph-pool/object-mask ownership map rather than pretending row identity. Individual categorical node embeddings and every GE edge boundary receive direct category/role/direction retrieval and finite/geometry diagnostics, not an invented aligned fact predecessor. Action paths are `A3->A5->A6` for A0/A2/A3, `A3->A5->A6ln->A6` for A1, and `A2linear->A4id->A4fuse_concat->A4fuse_linear->A4fuse_ln->A5->A6` for A4/A4_SHAM, using the fixed binding probe with absent tensors `NOT_APPLICABLE`; A4ctx is context-diagnostic.

Applicability semantic tracing follows the exact executed DAG. `Happ1maskcast->Happ1mask`; `Happ1slotids->Happ1slotembed->Happ1slotdtype->Happ1slotunsq->Happ1slot->Happ1slottanh`; `(Happ0o,Happ1slottanh)->Happ1objmod`; and `(Happ0o,Happ1slot,Happ1objmod)->Happ1role->Happ1ctx_ln->Happ1ctx_l1->Happ1ctx_relu->Happ1ctx_l2`. Then `(Happ1ctx_l2,Happ1mask)->Happ1masked->Happ1projsum`, `Happ1mask->Happ1masksum->Happ1count`, and `(Happ1projsum,Happ1count)->Happ1pool`. The scorer DAG is `(Happ0g,Happ1pool)->Happ1diffsub->Happ1diffabs`; `(Happ0g,Happ0a,Happ1pool,Happ1diffabs)->Happ1scorein->Happ1linear->Happ1gelu->Happ1drop->Happ2linear->Happ2`. Every arrow is a separate hook or functional identity tap and every named multi-input tuple is persisted with exact input hashes. If both repository optional object inputs are literally `None`, `Happ1pool` is the direct finite zero tensor and IDs `{Happ1maskcast,Happ1mask,Happ1slotids,Happ1slotembed,Happ1slotdtype,Happ1slotunsq,Happ1slot,Happ1slottanh,Happ1objmod,Happ1role,Happ1ctx_ln,Happ1ctx_l1,Happ1ctx_relu,Happ1ctx_l2,Happ1masked,Happ1masksum,Happ1count,Happ1projsum}` carry sole reason `NOT_APPLICABLE_NO_OBJECT_INPUT`; this reason is forbidden when either object input is supplied. For a supplied zero-arity `[B,0,D]` object tensor and `[B,0]` mask, the object DAG executes: empty role-row tensors persist exact shapes/counts and are omitted only from row-geometry denominators, while `Happ1masksum`, `Happ1count`, `Happ1projsum`, `Happ1pool`, and the complete scorer DAG remain mandatory finite tensors. Predictor, target, Harg, scalar, and query boundaries use their direct transition-sensitivity, target-fact, role-retrieval, finite/gradient, and perturbation checks rather than `semantic_drop`.

Threshold equality takes the failing branch. `complete_collapse` and `dimensional_collapse` are computed once from the global population at each vector boundary; per-group versions are reported diagnostically but never directly control the branch because group sample rank is support-limited. Required conditional families are: `{problem}` for graph/state/predictor graph rows; `{(problem,state_hash,object_type),(problem,object_type)}` for object rows; and `{(problem,schema),(domain,schema)}` for action rows and action-conditioned Happ rows. A family with no eligible group is malformed at a branch-required boundary. `conditional_collapse(b)` is the OR across all required families for `b`; family values and the controlling family are persisted. Boundaries with no listed local family record conditional collapse as `NOT_APPLICABLE`. `semantic_drop(prev,b)` is branch-controlling only for an explicitly aligned predecessor/boundary probe pair in the registry; otherwise it is `NOT_APPLICABLE`. For scalar logits/losses or boundaries without meaningful vector geometry, geometry flags are `NOT_APPLICABLE`, never PASS; their registered finite, discrimination, retrieval, intervention, and gradient checks remain mandatory. The assessor records the earliest boundary where each applicable flag first becomes true and whether it existed at initialization or emerged during optimization. Section 10.1 is the sole authoritative source for `boundary_trace_ok_dev` and `boundary_final_ok(split)` and applies these flags/checks to the exact boundary registry above. Dimensional-collapse flags are always reported and feed the batch/width sensitivity analysis even where Section 10.1 does not make them independently branch-failing.

The localization assessor emits independent Boolean flags for state-information loss, action-information loss, predictor insensitivity, target-path shortcut, and invalid-action support gap. These are recommendations only; Branch D remains operationally active.

All frozen probes use training records only for fitting and never tune hyperparameters. Boolean fact probes standardize each input coordinate with training mean/std (zero-std coordinates become zero) and fit one scikit-learn `LogisticRegression(C=1, penalty='l2', solver='liblinear', class_weight='balanced', tol=1e-6, max_iter=10000, random_state=0)` per required fact.

The action-binding probe is exactly `ArgumentReconstructionHead(action_dim=D_a, object_dim=D_s, max_action_arity=manifest.max_action_arity, hidden_dim=64, dropout=0)` (`D_s=D_a=64` outside DL5B). It receives detached `u_final`, all detached source-state object latents as candidates, and a Boolean `[B,A,O]` mask true only for active roles/type-compatible objects. The head alone trains with Adam lr `1e-3`, weight decay 0, summed active-role cross-entropy divided by active-role count, exactly 2,000 steps, seed matched to the checkpoint, no early stopping.

DL3 additionally fits a matched `ControlApplicabilityProbe` after each O0-O3 final checkpoint so applicability contrasts exist without inventing an O0/O1 production head. It is byte-identical to `ApplicabilityHead` in Section 2.3 but lives under module path `evaluation.control_applicability_probe`, receives detached checkpoint state/action/argument latents, initializes from the same run-seed rule in every O cell, and trains only its own parameters on the complete retained training applicability manifest with Adam lr `1e-3`, weight decay 0, batch 256, class-weighted BCE from the frozen manifest, exactly 2,000 steps, no scheduler or early stopping. Its development logits define only the explicitly named DL3 control-probe contrasts. O0/O1 have no production applicability head or production applicability logit; O2/O3 train the production head normally, and all DL4-DL6/final `app_ok` metrics use that production head. Missing or nonfinite control-probe evidence invalidates DL3.

The four oracle-substitution cells use two independent two-layer MLPs `[input -> 64 -> ReLU -> 1]`, one for applicability labels and one for own-successor (`1`) versus legal-distinct-mismatch (`0`) triple labels. Each uses Xavier initialization under the module seed rule, class-weighted BCE as in Section 2.3, Adam lr `1e-3`, batch 256, exactly 2,000 steps, and identical record orders.

Every 2,000-step neural probe above uses the same literal producer. One probe record is `compact_json([problem_id,source_state_hash,canonical_action_bytes,target_kind,target_payload])+"\n"`, where `target_kind` is one of `binding`, `applicability`, `oracle_applicability`, or `oracle_triple`; `target_payload` is respectively the schema-ordered true-object-name array, Boolean applicability label, Boolean oracle applicability label, or `[Boolean_own_successor,canonical_successor_hash]`. Define `probe_key` as the first four fields for `binding`, `applicability`, and `oracle_applicability`, and as those fields plus `canonical_successor_hash` for `oracle_triple`; repeated candidates with byte-identical full records collapse to one, while different payloads for the same `probe_key` are malformed. Sort the resulting distinct records once by their UTF-8 bytes. Batch size is 256 for every neural probe. Every named probe `Adam` is `torch.optim.Adam(lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0,amsgrad=false,maximize=false,foreach=false,capturable=false,differentiable=false,fused=false)` with no scheduler or clipping. Each epoch traverses contiguous batches from index 0 without shuffle; the final batch is retained even when shorter than 256; the next optimizer step starts the next epoch again at index 0. Stop immediately after update 2,000 and assess that checkpoint only. Empty manifests, nonzero process exit, missing updates/records, or nonfinite losses/parameters/logits are malformed. There is no convergence or early-stopping predicate. Record bytes/order, epoch, start/end indices, final-short-batch flags, optimizer state, and hashes are persisted. Paired cells use byte-identical record order and labels; only the named checkpoint/oracle input representation differs.

### DL2: Deterministic micro-overfit capacity gate

Use seed `0` and balanced tiny datasets containing own-successor triples, legal distinct mismatches, exact shared successors, and inapplicable substitutions. Test A0–A3 across O0–O3: exactly 16 micro-fit runs. Determinism is checked by replaying the saved checkpoint/evidence generation twice, not by retraining. If A4 is later authorized in DL4, A4 and A4_SHAM each receive four micro-fit runs before smoke use, for an absolute DL2 cap of 24 training runs.

For each architecture, run all four O0–O3 objective cells. The state encoder is frozen and state LocalSIGReg is diagnostic/no-grad; default action LocalSIGReg is enabled with coefficient `0.10`.

DL2 has one closed, deterministic micro-only loss contract and no other interpretation. Its `DL2_MICRO_MANIFEST` overrides the common one-pass floor partition for all DL2 auxiliary and geometry losses; Sections 6/7 floor slices remain unchanged for DL3-DL6. At every one of the exactly 2,000 updates, first evaluate the entire merged-micro positive-transition batch. For O2/O3, evaluate every retained merged-micro applicability row in one `DL2_AUX_FORWARD`, apply the frozen merged-micro `pos_weight=N_negative/N_positive`, and set `L_app_micro` to the ordinary mean of per-row `BCEWithLogits`; for O1/O3, the same attached forward evaluates every legal-distinct source unit and sets `L_gap_micro` to the equal-source mean of each unit's mean directional hinge. The auxiliary forward uses the complete deduplicated source/action/successor union in canonical record-byte order, one-step zero temporal resets, current parameters, and no stale activation; disabled losses instantiate no head/path. Add the usual `0.10` coefficients. Full auxiliary records are deliberately replayed at every update—never floor-partitioned, pass-weighted, cycled, or left as empty connected zeros. Persist at each update `{q,objective,ordered_app_record_hashes,ordered_gap_source_and_direction_hashes,N_positive,N_negative,pos_weight,L_app_micro,L_gap_micro,reset_hash,forward_hash}`; changed order/support, an omitted or duplicate row/unit, nonfinite output, or a forward count other than one when any auxiliary loss is enabled is malformed.

For micro geometry, freeze every distinct legal `(problem_id,canonical_action_bytes)` semantic atom from that same merged manifest, all of its raw occurrences, and all backing windows; group the atoms separately by `(problem,schema)` and `(domain,schema)`. Every required schema must have the eligible groups mandated above; sparse additional groups are recorded and omitted exactly as Section 8.1 specifies. This DL2-only support rule replaces both the campaign-wide 130-problem requirement and the `4*D` batch-closure requirement; it changes neither rule in DL3-DL6. Sort windows and atoms by their canonical hashes once. At each update, after the transition and optional auxiliary forward, run exactly one additional attached `GraphJEPA.trajectory_rollout()` over the full deduplicated micro-geometry backing-window union with source-local temporal resets; aggregate every frozen occurrence to its attached semantic-atom mean, evaluate all eligible groups, and set `L_action_local_SIG=0.5*(mean_{all eligible problem/schema groups} L_group + mean_{one required eligible domain/schema group per required schema} L_group)`. Add `0.10*L_action_local_SIG`, perform one joint backward/step over every enabled loss, and never cycle, subsample, invoke `r_bg`, or reuse stale activations. The state terms from this geometry forward are persisted under `no_grad` for diagnostics only. The merged manifest, provenance, family/group/atom/occurrence/window hashes, per-schema and merged metrics, per-update loss components, attached-forward counts, and reset flags are persisted; a missing occurrence, changed order, unsupported required schema, nonfinite value, or any access to a non-micro training problem makes DL2 malformed. DL2 therefore tests per-schema and merged micro-capacity under the default action regularizer but makes no all-problem or campaign-performance claim.

Micro-overfit PASS requires all of:

- at least 99% reduction from initial positive transition loss and final normalized positive loss at most `1e-4`;
- complete-binding training retrieval exactly `1.0`;
- at least 99% of legal-distinct training pairs satisfy the preregistered margin under O1/O3;
- applicability classification and groupwise ranking exactly `1.0` under O2/O3;
- exact-shared-successor pairs are excluded from `L_gap` and receive no forced-separation label;
- exact repeat output under the same seed.

Failure stops that architecture before smoke training.

### DL3: Objective ablation — fixed 12-checkpoint design

For every branch-changing DL3/DL4 clause, aggregation is fixed. An absolute metric for one cell is the arithmetic mean of its three per-seed problem-macro values. A paired gain or regression is the arithmetic mean over development problems of the arithmetic mean over seeds `[0,1,2]` of the paired within-problem difference. A positive-prediction-loss ratio is the arithmetic three-seed problem-macro numerator divided by the corresponding arithmetic three-seed problem-macro comparator denominator; a zero or nonfinite denominator is malformed. Thus every numeric authorization, eligibility, retention, regression, and `0.80`/`1.05` threshold in DL3/DL4 uses these reductions. “At least two seeds improve” is a separate additional sign clause and never substitutes for the arithmetic reduction.

DL3 is authorized only if `state_ok_dev_baseline` is true: the single DL0-pinned checkpoint must satisfy the `state_ok` predicate/fact clauses on development, without the later three-seed clause. That checkpoint's graph/state encoder is frozen for every DL3–DL4 seed; transition, applicability, and action-geometry gradients cannot update it. This isolates action/objective effects before the explicitly matched joint regularizer and sensitivity interventions in DL5.

Use A0, state LocalSIGReg as a fixed diagnostic, default action LocalSIGReg, and four cells:

```text
O0 positive prediction with frozen state encoder
O1 O0 + 0.10*legal transition-triple L_gap
O2 O0 + 0.10*L_app
O3 O0 + 0.10*L_gap + 0.10*L_app
```

Run three paired seeds: exactly 12 checkpoints. For each seed, unchanged modules share initialization, data order, and minibatches. Primary paired estimands are:

- O1−O0 and O3−O2 legal-distinct pairwise energy accuracy;
- O2−O0 and O3−O1 control-probe applicability groupwise top-1;
- each treatment's positive prediction-loss ratio to O0.

Report per-seed paired differences, exact problem-level randomization intervals, and Holm-adjusted tests for the four primary contrasts. O3 is the only promotable objective because the final claim requires both triple and applicability semantics. O3 is eligible for DL4 only if O3−O2 legal-distinct accuracy and O3−O1 control-probe top-1 each have adjusted `p<0.05`, mean paired gain at least `0.10`, at least two of three seeds improve, O3 legal-distinct accuracy, control-probe top-1, and production-head top-1 each reach `0.80`, and positive prediction-loss ratio to O0 is at most `1.05`. O1/O2 remain causal ablations. If O3 fails, later architecture/regularizer phases are not run.

### DL4: Matched architecture ladder — maximum 15 new checkpoints

Hold O3 fixed. Compare A0→A1→A2→A3 sequentially with three paired seeds; A0 checkpoints are reused, so at most nine new checkpoints. If authorized, A4 and A4_SHAM add exactly six checkpoints after their DL2 micro-fit gate.

The closed DL4 primary set is `{complete_binding_retrieval, normalized_binding_margin, legal_distinct_pairwise_accuracy, applicability_AUROC, applicability_groupwise_top1, positive_prediction_loss_ratio}`. A1–A3 are tested in order. A cell is retained only if complete-binding retrieval or normalized binding margin gains at least `0.10` versus its immediate predecessor, at least two paired seeds improve that same metric, each of the three semantic metrics regresses by at most `0.02`, and its positive-loss ratio to A0 is `<=1.05`. Exact ties within `1e-12` do not count as improvement. First failure stops A1–A3 and selects the last retained predecessor.

A4 is authorized only when selected A3 has complete-binding retrieval `>=0.80`, positive prediction-loss ratio to A0 `<=1.05`, and either applicability groupwise top-1 or legal-distinct accuracy is `<0.80` on development. A4 and A4_SHAM are then both run. A4 is retained only if every deficient authorization metric gains `>=0.10` versus A4_SHAM and reaches `>=0.80`, at least two seeds improve, the remaining semantic metrics regress by at most `0.02`, and positive-loss ratio is `<=1.05`; otherwise A3 remains selected. If A3 was not retained, A4 is not run.

### DL5A: Default regularizer necessity/locality — exactly 12 checkpoints

Hold O3, the selected DL4 architecture, `D_s=D_a=64`, and batch size 24 fixed. Run S0–S3 from Section 8.2 with three paired seeds, unfreezing state, action, and predictor in all four cells so both default state and action regularizers are causally tested. S0 is the default, not a candidate selected from test performance. `geometry_ok` is the literal S0 acceptance predicate in Section 8.2. A FAIL stops before joint confirmation and keeps Branch D; it cannot switch the default to VC or tune coefficients.

### DL5B: Batch-size and latent-width sensitivity — exactly 39 checkpoints

After S0 passes DL5A, run a preregistered diagnostic sensitivity matrix with O3, the selected architecture, default LocalSIGReg, all encoders/predictor trainable for one complete canonical training pass, all 130 training problems, and all 48 development problems. These cells cannot promote a model or replace the default; they test whether conclusions are artifacts of batch or latent width.

```text
batch_size B in {8,16,24}
latent pairs:
  L0=(D_s=64, D_a=64) reference
  L1=(32,64)           state-width low
  L2=(128,64)          state-width high
  L3=(64,32)           action-width low
  L4=(64,128)          action-width high
  L5=(32,32)           jointly low
  L6=(128,128)         jointly high

screen:  all 3*7=21 cells at seed 0
confirm: seeds 1 and 2 for B=24 at L0-L6 and for (B,L)=(8,L0),(16,L0)
maximum: 21 + 2*9 = 39 checkpoints
```

Every batch cell consumes the identical canonical record set exactly once; only optimizer-update count changes as `ceil(N_records/B)`. No gradient accumulation is used. Width cells use `graph_embed_dim=64`, `latent_dim=D_s`, `action_dim=D_a`, state/action GRU output widths equal their respective latent widths, predictor hidden width `64`, applicability hidden width `128`, and action-probe hidden width `64`; only dimension-required input/output shapes change. All modules are initialized from the DL5B parameter rule below rather than truncating, padding, projecting, or loading any 64-dimensional checkpoint. The `(64,64,24)` sensitivity cell is independently initialized under this rule and is not reused from DL4.

DL5B initialization is exhaustive and overrides the general DL3/DL4 baseline-loading rule. Construct every selected-architecture module on CPU in float32, then overwrite every named parameter before any forward. A local generator for parameter or GRU gate key `k` uses the first unsigned 64 little-endian bits of `SHA256(UTF8(str(run_seed)+":"+k))`; `k` is the fully qualified state-dict path plus `:r`, `:z`, or `:n` for a GRU gate. Batch size is absent from `k`, so the three B cells at the same `(seed,D_s,D_a)` have byte-identical initial state dicts; a shape-invariant path also has byte-identical values across width cells.

Initialization by parameter class is literal:

```text
Linear.weight:   torch.nn.init.xavier_uniform_(gain=1.0,generator=local_generator)
Linear.bias:     zeros
Embedding.weight: torch.nn.init.xavier_uniform_(gain=1.0,generator=local_generator);
                  then overwrite the declared padding row with exact zeros
LayerNorm.weight: ones
LayerNorm.bias:   zeros
GRU.weight_ih_l0: split in PyTorch (r,z,n) order and Xavier-uniform each gate matrix
GRU.weight_hh_l0: split in PyTorch (r,z,n) order and Xavier-uniform each gate matrix
GRU.bias_ih_l0 and GRU.bias_hh_l0: zeros
trainable GINEConv.eps, if present: zero
```

The gate matrices are initialized separately with their gate-key generators and concatenated contiguously in `(r,z,n)` order; all initial GRU hidden states are exact zeros. This rule covers GraphEncoder embeddings/projections/GINE MLPs, GraphStateProjector, state GRUs, action/schema/role embeddings, action object projectors/composers/fusion/temporal GRU/normalization, predictor, applicability head including slot embeddings, argument head including role embeddings, and every authorized A2-A4 module. Dropout and fixed descriptor/projection artifacts have no trainable parameters. Any named parameter not consumed exactly once by one rule, any unexpected recurrent suffix/layer/direction, or any constructor value surviving without an explicit rule is malformed. Persist the ordered `(path,shape,dtype,initializer,key,parameter_sha256)` table and whole-training-model state-dict hash before optimizer construction; replay must be byte-identical. The post-hoc action-binding probe is instantiated only after checkpoint freezing but applies the same per-parameter rules and run seed to its `evaluation.argument_reconstruction_head.*` paths and persists its own pre-probe-optimizer table/state hash; it is not silently included in the training-model hash. Thus initialization is fully defined for widths 32/64/128, and the 39 cells vary only the registered batch/width/seed factors and their consequent tensor shapes/update counts.

For every boundary and primary semantic metric, report all 39 run values, problem-macro values, and—where three seeds exist—mean/std and coefficient of variation when the absolute mean exceeds `1e-8`. DL5B's sole loss metric is `within_cell_positive_loss_ratio`: for each run and development problem, evaluate normalized positive transition loss on that problem's complete positive manifest both immediately after initialization and at the final checkpoint, reduce records to source within each evaluation, and divide the final problem mean by the same problem's initialization mean. A zero/nonfinite problem denominator, changed record set, or missing problem is malformed. The run scalar is the arithmetic mean of the resulting 48 problem ratios, and those problem ratios are the rows used by the categorical decomposition and seed reductions. The same-cell, same-width, same-architecture, same-seed initial problem loss is the only denominator in DL5B; no DL3/DL4 checkpoint, A0/O0 cell, later DL6 baseline, or cross-width loss is a comparator.

The required seed-0 categorical decomposition is exact and descriptive only. For every reported scalar primary semantic/boundary metric having one finite problem value in every one of the 48×21 seed-0 problem/cell rows, sort rows by `(problem_id,B,L)` with `B=[24,8,16]` treatment order and `L=[L0,L1,...,L6]`. Build float64 design matrices with an intercept, 47 problem treatment dummies (first UTF-8 problem ID is reference), batch dummies `{B8,B16}` (`B24` reference), latent dummies `{L1,...,L6}` (`L0` reference), and all 12 products in batch-major then latent-major order. Fit with `numpy.linalg.lstsq(X,y,rcond=None)` and require the expected full rank. Report the full coefficient vector and sequential balanced-design sums of squares `SS_batch=SSE(problem)-SSE(problem+batch)`, `SS_latent=SSE(problem+batch)-SSE(problem+batch+latent)`, and `SS_interaction=SSE(problem+batch+latent)-SSE(full)`, with df `{2,6,12}` and `partial_eta2=SS_term/(SS_term+SSE_full)`; a negative value beyond float64 tolerance `1e-10`, rank loss, or nonfinite result is malformed.

Also report named equal-cell-weight contrasts from the 48 problem-block cell means: batch `{B8-B24,B16-B24}` averaged over seven latent pairs; latent `{Lj-L0,j=1..6}` averaged over three batches; and 12 interactions `(B, Lj)-(B,L0)-(B24,Lj)+(B24,L0)` for `B in {8,16}` and `j=1..6`. Precompute exactly 10,000 bootstrap index rows with `numpy.random.Generator(numpy.random.PCG64(0)).integers(0,48,size=(10000,48),endpoint=False)`, reuse the identical matrix for every metric, retain all 21 paired cells for each sampled problem occurrence, recompute only these named contrasts, and report `[0.025,0.975]` percentile intervals using `numpy.quantile(method='linear')`. Persist the row/design column names, matrices, ranks, coefficients, SSE/SS/df/effect sizes, contrasts, bootstrap-index bytes/hash, replicate contrasts, and interval bytes. This fully instantiates the closed `metric ~ problem + batch + latent_pair + batch:latent_pair` report; no alternative coding, ANOVA type, seed pooling, transition resampling, bootstrap count, or RNG is allowed.

The full seed-0 screen uses this decomposition. Three-seed sentinel contrasts report batch effects `(8,L0)-(24,L0)` and `(16,L0)-(24,L0)`, state-width effects `L1/L2-L0`, action-width effects `L3/L4-L0`, and joint-width effects `L5/L6-L0`. Transitions are never treated as independent.

Acceptance uses seed 0 only for the 21-cell descriptive screen and arithmetic three-seed means of problem-macro metrics for every sentinel comparison. Higher is better for legal-distinct accuracy, applicability top-1, and complete-binding retrieval; lower is better for `within_cell_positive_loss_ratio`. `sensitivity_ok` requires: all 21 screen cells and all 18 confirmation runs/artifacts are valid; no seed-0 screen cell has complete collapse at a branch-critical boundary; all nine sentinel cells satisfy `boundary_trace_cell_ok(cell)` under its explicit two-of-three-seeds rule; at L0, the max-minus-min of the three batch-cell three-seed means is `<=0.05` for each metric; the `(24,L0)` three-seed mean is no more than `0.02` below the maximum higher-is-better mean and its `within_cell_positive_loss_ratio` is no more than `0.02` above the minimum such ratio; and no B=24 L1-L6 three-seed-mean vector is at least as good as `(24,L0)` in all four oriented metrics with at least one strict oriented gain `>0.02`, where the fourth metric is that within-cell ratio oriented lower-is-better. Any seed-0 or sentinel axial cell with complete collapse identifies the responsible state/action width and fails sensitivity. No width or batch is reselected after development or untouched-test observation.

### DL6: One joint-learning confirmation and untouched test

Promote exactly the frozen development winner plus A0/O0 baseline, three seeds each, with default state/action LocalSIGReg. In DL6 only, unfreeze the graph/state encoder for both configurations using matched initialization, data order, optimizer budget, and gradient accounting. Train the six checkpoints on training only, evaluate them first on all 48 development problems, and freeze them before holdout access. This is the sole official efficacy test that JEPA can co-learn the state and action spaces after the frozen-state causal ladder. No fallback winner is selected after test observation. The confirmation adds fixed-cardinality candidate diagnostics without a planner: top-k retrieval at fixed `K`, candidate ranking, query-latent perturbations, and performance versus action cardinality.

Before opening test, mandatory `DL6P` uses the frozen DL0 power code on four winner development problem vectors oriented so higher is better. For each development problem `p`, the first three entries are respectively `mean_seed legal_distinct_accuracy(p,seed)-0.80`, `mean_seed production_applicability_top1(p,seed)-0.80`, and `mean_seed complete_binding_retrieval(p,seed)-0.80`, using paired winner seeds `[0,1,2]` after the registered within-problem source reduction. The fourth entry is `1.05 - mean_seed(winner_positive_loss(p,seed)/paired_A0_O0_baseline_positive_loss(p,seed))`, where each numerator and denominator is that problem's records-to-source mean normalized positive transition loss at the final paired DL6 checkpoint. A zero/nonfinite baseline denominator, missing paired seed/problem, changed record set, or nonfinite ratio is malformed; ratio-of-seed-means and macro-mean ratios are forbidden substitutes. Each vector therefore has exactly 48 finite problem entries; compute `sigma_dev` with sample standard deviation (`numpy.std(ddof=1)`). For each vector compute `n_power=ceil(((1.6448536269514715+0.8416212335729144)*sigma_dev/0.10)^2)`, and `n_test_required=max(12,n_power_1,...,n_power_4)`. Nonfinite variance, fewer than two problem values, or `20<n_test_required` yields `INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D` without opening test. DL6P fixes only adequacy; all development performance gates remain literal and no observed effect changes the `0.10` target.

The untouched test runs once after code, manifests, coefficients, sensitivity matrix, and assessor hashes are frozen. Every final checkpoint is evaluated on every one of the 20 `final-test` problems; any omitted problem invalidates evidence.

## 8. Loss and regularizer contracts

The executable O0 core is frozen as:

```text
GraphLatentPredictionLoss: graph_weight=1.0, object_weight=1.0
prediction_coeff=1.0
regularization.kind=local_sigreg
encoder_graph/object and state_graph/object boundary weights=0.25 each
state_local_sigreg_coeff=1.0 when state path is trainable; diagnostic-only when frozen
action problem-schema/domain-schema boundary weights=0.5 each
action_local_sigreg_coeff=0.10 in every official O/A cell
repository GraphVC/ActionVICReg coefficients=0.0 except comparator S3
similarity_coeff=0.0
inverse_dynamics_coeff=0.0
action contrastive/argument reconstruction coefficients=0.0
applicability coefficient=0.0 except O2/O3
gap coefficient=0.0 except O1/O3
goal-head loss excluded from all official cells
rollout-order weights=uniform
```

The decomposed trainable loss is:

```text
L_total =
    1.0 * L_positive_transition
  + 1.0 * L_state_local_SIG          # wrapper's four encoder/final-state components; trainable DL5A/B/DL6
  + 0.10 * L_action_local_SIG        # wrapper's problem/domain action sum; default in all O/A cells
  + 0.10 * L_gap                    # O1/O3 only
  + 0.10 * L_app                    # O2/O3 only
```

State LocalSIGReg is computed but detached/diagnostic during frozen-state DL2–DL4 and is trainable in DL5A/DL5B/DL6; action LocalSIGReg is trainable by default throughout. There is no development scale matching and no coefficient selection algorithm: every coefficient is literal above. `L_gap` is defined only over legal transition triples. `L_app` is binary state-action BCE. Test data cannot change any coefficient.

### 8.1 Default local SIGReg module

The default state and action regularizer is a new generic `LocalSIGRegLoss`; repository VC is no longer the default. The implementation contract is:

```text
LocalSIGRegLoss.forward(
    samples: FloatTensor[N,D],
    atom_ids: Int64Tensor[N],
    group_ids: Int64Tensor[N],
    sample_hashes: UInt8Tensor[N,32],
    projection: Float64Tensor[256,D],
) -> {total, group_losses, group_counts, distinct_atom_counts}
```

Overlapping rollout windows first aggregate repeated semantic atoms in float64 while preserving autograd: sort by canonical atom bytes and replace all rows for atom `a` by `x_bar_a=mean_{i:atom_i=a} x_i`. An atom assigned to more than one group is malformed. Atom/group definitions are exact:

- `encoder_graph` and `state_graph`: atom `(problem_id,canonical_state_hash)`, group `(problem_id)`;
- `encoder_object` and `state_object`: atom `(problem_id,canonical_state_hash,object_name)`, local group `(problem_id,canonical_state_hash,object_type_id)`;
- `action_local`: atom `(problem_id,canonical_action_bytes)`, group `(problem_id,action_schema_id)`;
- `action_domain`: the same aggregated action atoms, group `(domain_hash,action_schema_id)`.

`encoder_graph/object` are observed target outputs from `GraphEncoder`; `state_graph/object` are final observed target outputs from `StateEncoderF`. Invalid actions, mismatches, padding, masked objects, and predicted states are excluded from SIGReg populations. The action objective is `0.5*L_action_local+0.5*L_action_domain`, so it must preserve grounded variation both inside a problem and across problems without relying only on schema means. Predictor inputs/deltas/outputs and all other intermediate boundaries are traced under Section 9 but are not independently regularized.

Canonical atom/group bytes are compact UTF-8 JSON arrays with original case-sensitive identifiers and one final newline. IDs are zero-based indices after bytewise sorting all eligible training bytes; mappings, raw-row counts, aggregated-atom counts, and population hashes are persisted. A raw `sample_hash` is exactly `SHA256(compact_json([problem_id,source_record_path,window_start,time_index,boundary_id,canonical_row_id])+"\n")`; repeated semantic atoms from different raw occurrences therefore have different sample hashes and are aggregated by `atom_id`, while a duplicated raw occurrence has the same sample hash and is malformed. Development/test atoms are diagnostic only and never passed to the loss.

This regularizes local residual geometry while allowing problem topology, object type, state, and action schema means to remain semantically distinct.

For each width `D`, projection rows are generated once with `numpy.random.Generator(numpy.random.PCG64(0)).standard_normal((256,D),dtype=float64)`, normalized to unit length in float64, saved as little-endian contiguous `.npy`, and SHA-256 hashed. Checkpoint loading verifies the stored projection bytes rather than regenerating silently. Frequencies are float64 `[0.5,1.0,2.0]`. For a group with `n>=4` distinct aggregated atoms:

```text
mean_g   = mean_n x_bar_gn
r_gn     = sqrt(n/(n-1)) * (x_bar_gn - mean_g)
real_gjt = mean_n cos(t*dot(v_j,r_gn))
imag_gjt = mean_n sin(t*dot(v_j,r_gn))
phi_t    = exp(-t^2/2)
L_group(g) = mean_{j,t} ((real_gjt-phi_t)^2 + imag_gjt^2)
L_local_SIG = mean_g L_group(g)
```

The `sqrt(n/(n-1))` correction restores unit marginal variance in expectation after centering a Gaussian group. No finite-sample bias correction, clamping, batch standardization, or complex tensor implementation is permitted. Atom means, group means, and group losses remain attached to autograd. Distinct groups receive equal weight independent of raw or aggregated row count.

A deterministic geometry manifest is separate from the transition minibatch. The population builder records every canonical group and marks groups with fewer than four distinct atoms `INELIGIBLE_SPARSE`; such groups receive zero weight, are never passed to `LocalSIGRegLoss`, and do not cause component-weight renormalization. In campaign-performance cells DL3-DL6, strict official mode requires every training problem to have at least one eligible group at each applicable boundary and every geometry batch to provide at least `4*D` distinct atoms; otherwise the component/run is malformed. DL2 is solely the explicitly defined micro-capacity fixture below and uses its own closed micro-geometry contract; it is not permitted to claim campaign coverage or use the 130-problem scheduler.

Scheduling is literal, persistent, and preserves the equal-group estimand. At the start of complete training pass `p`, let `N_steps` be the already frozen optimizer-step count. For each boundary and cycle `c=0,1,...`, sort all eligible groups by `SHA256(compact_json([seed,p,c,boundary_id])+canonical_group_bytes)`; within each group sort atoms by `SHA256(compact_json([seed,p,c,boundary_id])+canonical_atom_bytes)` and select the first `min(16,n_g)`. A selected group is indivisible: append all of its selected atoms, then close the batch if its distinct-atom count is at least `4*D`; never split a group across batches. Only after every eligible group has been consumed may the stream continue with cycle `c+1`. If a next-cycle occurrence has a canonical group already present in the still-open batch, defer that occurrence to the front of the next batch and continue scanning; failure to reach `4*D` with distinct canonical groups is malformed. Precompute exactly `N_steps` closed batches for each component, extending through complete cycles and a final deterministic prefix as needed; preflight requires the first complete coverage to close within `N_steps`. Let `G_b` be the number of eligible groups for component `b`, and let `r_bg>=1` be the number of scheduled occurrences of group `g` in those `N_steps` batches. For occurrence `(g,c)`, the attached group loss uses that cycle's selected atoms and the scheduler assigns an occurrence-specific `loss_group_id=ordinal(canonical_group_id,c)` for weighting and checkpointing; the defer rule guarantees that each `loss_group_id` and canonical group occur at most once per geometry batch. At optimizer step `q`, the component contribution is not a per-batch group mean but

```text
L_scheduled_b(q) = (N_steps/G_b) * sum_{(g,c) in batch_q} L_group(g,c)/r_bg
```

Therefore the arithmetic mean of `L_scheduled_b(q)` over all `N_steps` is exactly `mean_g mean_{c in scheduled occurrences(g)} L_group(g,c)`, so uneven group counts, atom-count closure, and the final cycle prefix cannot reweight groups. `LocalSIGRegLoss` returns attached `group_losses`; in scheduled calls its `group_ids` argument is the occurrence-specific `loss_group_ids`, while the wrapper retains the one-to-one canonical-group mapping for support and weights. `ACSJEPALocalSIGRegLoss` applies these persisted weights without within-batch renormalization. Optimizer step `q` consumes scheduled batch `q` independently for each of the six components. For each selected semantic atom include every distinct raw occurrence and every backing window listed for that atom in the frozen geometry manifest, then form one geometry-forward window batch as the deduplicated union across atoms and all six components sorted by window hash. Component/sample-hash masks select exactly those occurrences; extra rows emitted by a shared window are excluded. The attached float64 atom mean therefore averages all manifest occurrences, never one representative occurrence. Persist `{pass,cycle,component,group_cursor,atom_ids,group_ids,sample_hashes,canonical_group_ids,loss_group_ids,r_bg,G_b,N_steps,union_window_hashes}` for the entire schedule before the pass and verify the step slice on resume. Thus, in every DL3-DL6 campaign-performance pass, every eligible group—and therefore every one of the 130 training problems—contributes at least once per applicable boundary. Repeated raw rows for one atom are averaged; conflicting atom metadata, repeated canonical atom IDs after aggregation, a split group, a cursor/weight mismatch, insufficient support, skipped problems, or a manifest mismatch makes the run malformed.

Default coefficients and the ACS-JEPA wrapper are literal:

```text
L_state_local_SIG = 0.25 * (
    L_encoder_graph + L_encoder_object + L_state_graph + L_state_object
)
L_action_local_SIG = 0.5 * (L_action_problem_schema + L_action_domain_schema)
state_local_sigreg_coeff = 1.0   # trainable in DL5A/DL5B/DL6; diagnostic when frozen
action_local_sigreg_coeff = 0.10 # trainable in every O0-O3/A0-A4 official cell
```

`ACSJEPALocalSIGRegLoss` receives one dedicated geometry forward's observed target `GraphEncoderOutput`, final observed target `JEPALatentState`, legal trace action latents, masks, and canonical atom/group metadata; invokes the generic module six times; returns every component plus the two weighted sums; and verifies that graph/object temporal slices are exactly target timesteps `1..K` while action slices are source/action timesteps `0..K-1`. At each optimizer update, first run the ordinary transition minibatch forward, then load the geometry manifest's backing windows across problems and run `GraphJEPA.trajectory_rollout()` exactly once on that geometry batch under the current parameters. Deduplicate/aggregate its attached outputs by semantic atom, compute all enabled geometry terms, add them to the transition/objective loss, call one joint backward, and take one optimizer step. There is no cross-step autograd accumulation and no reuse requirement when a transition window also appears in the geometry batch. Frozen state boundaries run under `no_grad` and emit detached diagnostics; trainable state/action geometry paths remain attached. S1 executes the identical geometry forward under `no_grad` for diagnostics and matched forward exposure but adds no geometry term; S0/S2/S3 use the same geometry windows and attach their configured loss. `GraphJEPA.trajectory_rollout()` must expose each forward's already-computed stacked `GraphEncoderOutput` and canonical problem/state/object/action metadata without recomputing a boundary inside that forward or changing forward values. Manifests separately record transition and geometry examples, forwards, FLOPs/runtime, peak memory, and hashes; DL0 resource calibration includes this second forward. No dynamic reweighting is allowed when a component lacks support: strict official mode raises and invalidates evidence.

Required module tests precede any training:

- exact float64 comparison to a scalar reference implementation for `D in {32,64,128}`, uneven group sizes, and repeated raw rows aggregated to semantic atoms; an uneven-batch/final-prefix schedule fixture must verify that the arithmetic mean of persisted `L_scheduled_b(q)` equals the `mean_g mean_c L_group(g,c)` scheduled equal-canonical-group reference and that no group is split or duplicated within a batch;
- bitwise CPU repeat; invariance within `1e-12` to row permutation and group-ID relabeling;
- analytic versus central-finite-difference gradients at `rtol=1e-5, atol=1e-7`, including centroid gradients;
- fixed `N(0,I)` groups of at least 256 atoms each and at least 4096 total atoms have loss `<0.02`; all-zero groups have loss `>0.10`;
- group-local mean shifts leave the loss unchanged within `1e-12`, the gradient sum within each group is zero within `1e-10`, and rank-one/duplicated-atom fixtures are detected by the boundary-collapse assessor;
- nonfinite values, `n_g<4`, duplicate sample hashes, projection-width mismatch, absent required problems, and malformed group IDs raise rather than silently drop rows;
- optimizer-ownership tests require exactly one registration of every enabled SIGReg parameter dependency and zero gradients on frozen boundaries.

Configuration defaults after implementation are `regularization.kind: local_sigreg`, the coefficients above, `J=256`, and frequencies `[0.5,1.0,2.0]`. Existing `GraphVCLoss` and `ActionVICRegLoss` remain available only as explicitly named comparators. The claim is limited to local residual anti-collapse; Gaussian geometry does not imply applicability, binding identity, or symbolic validity.

### 8.2 Regularizer ablations and comparators

All regularizer cells use byte-identical transition batches, S0-built geometry schedules/windows, projections, architecture, objective O3, widths, seeds, and optimizer exposure. S1 runs the same geometry forward without a term. S2 does not build an impossible one-group scheduler: at each step it pools exactly that component's S0-scheduled atoms across canonical groups, assigns one global `loss_group_id`, and uses the ordinary global group loss; its reported estimand is the arithmetic mean of those `N_steps` step losses. S3 consumes the same step matrices and ignores group IDs. Thus S0-S3 differ in geometry objective, not sampled atoms or forward exposure:

```text
S0 LOCAL_SIGREG_DEFAULT: state LocalSIGReg + action LocalSIGReg
S1 NO_GEOMETRY:          both disabled
S2 GLOBAL_SIGREG:        same estimator but one global group per boundary
S3 REPOSITORY_VC_FORMULA: same six populations, replacing each SIGReg term with repository variance+covariance formula
```

S0 is the preregistered default and remains the production candidate regardless of comparator performance; S1-S3 estimate necessity and locality and cannot silently replace it. All comparator summaries use arithmetic three-seed means of problem-macro metrics. For DL5A only, define `dl5a_positive_loss_ratio(seed)` as the final S0 normalized positive-transition loss divided by the final paired S1 loss after each loss is reduced over the identical complete development manifest in order `records -> source -> problem -> arithmetic problem macro`; a zero/nonfinite S1 seed denominator or changed/missing problem set is malformed. The branch scalar is the arithmetic mean of these three seed ratios—never a final/initial ratio, ratio of three-seed macro means, DL3/DL4 ratio, or DL5B metric. Define `effective_rank_ratio(b,seed)=participation_rank(C_global(b,seed))/D_b` and `geometric_collapse(b,seed)=complete_collapse(b,seed) OR dimensional_collapse(b,seed) OR OR_f conditional_collapse(b,f,seed)`; semantic drop is not part of this geometry-only flag. `geometry_ok` requires all of: there exists one fixed coordinate—either a boundary `b` effective-rank ratio or a boundary/family `(b,f)` conditional residual scale—for which paired `S0-S1>=0.10` in the same at least two seeds; for every registered boundary `b`, fewer than two S0 seeds have `geometric_collapse(b,seed)` unless at least two S1 seeds also have it; each higher-is-better semantic metric `{complete_binding_retrieval, normalized_binding_margin, legal_distinct_pairwise_accuracy, applicability_AUROC, applicability_groupwise_top1}` has an arithmetic three-seed problem-macro S0 mean no more than `0.02` below the maximum corresponding S1-S3 mean; and `mean_seed dl5a_positive_loss_ratio(seed)<=1.05`. The winning geometry coordinate, every family value, seed Booleans, losses, denominators, ratios, and arithmetic reductions are persisted. Failure sets `geometry_ok=false` and keeps Branch D; it does not trigger coefficient tuning or fallback to VC.

Repository VC comparator constants remain `std_coeff=1.0`, `cov_coeff=1.0`, `std_margin=1.0`, `epsilon=1e-4`, state coefficient `1.0`, and action coefficient `0.10`. For each of the same six deduplicated S0 sample matrices `X` (group IDs ignored), it computes `Xc=X-mean(X)`, `std=sqrt(torch.var(Xc,dim=0,correction=1)+1e-4)`, `L_std=mean(relu(1-std))`, `C=Xc.T@Xc/(N-1)`, `L_cov=mean(off_diagonal(C)^2)` (connected zero when `D=1`), and `L_VC=L_std+L_cov`; `N<=1` is malformed in official cells. The four state and two action component weights are exactly the S0 wrapper weights. It is explicitly a variance/covariance comparator, not full VICReg, because no paired-view invariance term is defined.

### 8.3 Gradient accounting

Every run reports each loss magnitude, active-hinge count, parameter-group gradient norm, whether `torch.autograd.grad(loss,group,allow_unused=True)` returned `None`, and pairwise gradient cosine. Before any official run, ownership fixtures instantiate the actual full forward with fixed nonstationary positive-transition, misclassified-applicability, unsatisfied-gap (`Delta_E=0`), non-Gaussian SIGReg, and collapsed-VC inputs; each fixture must yield finite nonzero gradients for every differentiable dependency named below and absent/zero gradients only for frozen or unrelated groups. During official runs, every enabled nonempty scheduled loss must remain connected (`grad is not None`) to all of its currently trainable dependencies, but its numeric norm may become zero at a stationary solution. In particular, `L_gap` may have a connected zero scalar/gradient whenever its active-hinge count is zero; this is successful margin satisfaction, not malformed evidence.

The ownership map follows the actual non-detached graph phase by phase:

```text
positive transition -> predictor + action encoder + graph/state encoder when trainable
L_gap               -> predictor + action encoder + shared source/target graph/state encoder when trainable
L_app               -> applicability head + action encoder + source graph/state encoder when trainable
action SIGReg/VC    -> action encoder; additionally graph/state encoder when trainable only for A0/A1/A2 and genuine A4, whose action input consumes nonzero state/object context; no graph/state dependency in A3 or A4_SHAM
state SIGReg/VC     -> graph/state encoder when trainable
post-hoc probes     -> probe parameters only; checkpoint latents are detached
```

The shared successor target remains attached, so its graph/state dependency is mandatory for positive transition and `L_gap` whenever state is unfrozen. Frozen-state DL2–DL4 graph/state gradients must be absent/zero although diagnostics are emitted; in DL5A/DL5B/DL6 every graph/state dependency listed above is enabled. S1 geometry gradients and disabled head/loss gradients must be absent/zero. Missing graph connection, unexpected detachment, nonfinite gradients, or ownership outside this map is malformed; a finite connected zero after fixture PASS is recorded, not rejected.

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
- identity-tower localization stability of `q_identity,t` across source states where action identity is semantically comparable (A3/A4 only; `NOT_APPLICABLE` otherwise);
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

### Boundary collapse and sensitivity

- every separate GN/GE/GM, S0/S3-S6, A0-A6, P0-P2, T, Happ, Harg, L0-L5, and QUERY registry ID above at every required trace update, including each indexed layer/order/component and every explicit `NOT_APPLICABLE` reason;
- earliest optimization update and exact predecessor where complete, dimensional, conditional collapse or registered semantic drop first appears;
- adjacent-boundary CKA and semantic-probe deltas;
- LocalSIGReg/VC loss and gradient contribution at each regularized boundary;
- all 21 seed-0 `3x7` batch/latent-pair screen effects and interactions, plus the nine three-seed sentinel contrasts including separate state/action axial effects;
- per-problem and per-seed variance, batch-span, width-normalized effective rank, and primary-metric Pareto relation;
- complete problem/window coverage counts for training, development, and test.

### Metric, aggregation, and tie contracts

- Effective rank everywhere uses float64 sample covariance `C=(X-mean(X))^T(X-mean(X))/(N-1)` with `N>=2` and participation ratio `r_eff=(sum eigenvalues)^2/(sum eigenvalues^2)`, using float64 symmetric eigendecomposition and eigenvalues clipped below at zero. A zero denominator gives rank zero. Off-diagonal correlation uses covariance divided by outer standard deviations; any dimension with std `<=1e-8` makes the geometry predicate fail before RMS computation. Equal-group conditional covariance uses the literal `C_equal_group` definition in DL1.
- Pairwise distance/cosine diagnostics use all rows when `N<=4096`; otherwise use exactly the first 4096 distinct semantic atoms ordered by `SHA256(boundary_id||checkpoint_hash||canonical_atom_bytes)` and all unordered pairs among them. The sample manifest and hash are persisted.
- Adjacent-boundary linear CKA uses aligned semantic atoms, float64 column-centered matrices, and `||X^T Y||_F^2/(||X^T X||_F*||Y^T Y||_F)`; a denominator `<=1e-12` records CKA zero and `complete_collapse=true`.
- AUROC and AP use the pinned scikit-learn `roc_auc_score` and `average_precision_score` on logits/labels. Calibration uses sigmoid probabilities and 15 equal-width bins on `[0,1]`, left-closed/right-open except the final closed bin; ECE is sample-count-weighted `abs(mean_probability-mean_label)`. Single-class populations are malformed for branch-critical metrics.
- `required precondition predicates` are every `(predicate, argument-position)` Boolean fact used in any action-schema precondition and having at least 20 positive and 20 negative labels in both training and development. Probe candidates, labels, and splits are manifest-frozen before test inference. Every development-required predicate remains required on untouched test; test labels may not remove one. If a required predicate has fewer than one positive or one negative test label, its AUROC is undefined and `test_evidence_ok=false` rather than silently changing the predicate set.
- Role retrieval uses the detached representation and post-hoc action-binding probe fitted exactly in DL1. It scores every type-compatible object in the source state. A role is correct only when the true object has strictly greater score than every wrong object. Complete-binding retrieval is `1` only when every active role is correct; zero-arity actions are excluded. A tie with a wrong object is failure.
- The normalized binding margin is `(score_true-max_wrong)/(abs(score_true)+abs(max_wrong)+1e-8)` per role, minimum over roles per action, mean over actions per source, then mean sources per problem. Positive means every winning role is strictly correct.
- Legal-distinct pairwise energy accuracy is `1[E_mismatch > E_own]`; equality is failure. It is averaged over triples within source and sources within problem.
- An applicability candidate group contains all frozen applicable and inapplicable action records for one source state. Groupwise top-1 is success only if every action tied for maximum logit is applicable. Top-k uses descending logit and canonical action-byte order for ties.
- Every seed metric is first averaged over source states within problem, then macro-averaged over problems. “At least two of three seeds” is evaluated on those per-seed problem-macro values. Treatment tests instead use the arithmetic mean of the three paired-seed differences within each problem, as specified in DL0.
- A `valid run/seed` means process exit zero plus complete, hash-valid, schema-complete, finite artifacts generated by the preregistered code/config. A converged but below-threshold run is valid evidence and a performance FAIL. Any malformed pretest run sets `pretest_evidence_ok=false`; a malformed frozen test run sets `test_evidence_ok=false`; every seed required by that cell (seed 0 alone for an explicitly descriptive screen, otherwise all `[0,1,2]`) must be valid.
- Action residual scale is `median_schema sqrt(mean_d Var(u_d|schema)) / (sqrt(mean_d Var(u_d))+1e-8)`, with schemas equally weighted. The DL5 gain is an absolute difference in this ratio or in normalized binding margin.

Query-latent robustness uses no planner and is bound exclusively to `CARDINALITY_FULL_UNIVERSE`, never the capped Section 6 rows. For each official seed and each distinct source in the evaluated split, encode every complete-universe type-valid action exactly once with the source-local zero temporal reset, persist its float32 `A6` latent/logit/label/action-byte hash, and cast the latent to float64 for query construction and decoding. Every oracle-applicable universe action is one positive `u+`; the decode candidate set is exactly all universe actions for that same source, with no filtering.

For a positive action, eligible hard negatives are exactly oracle-inapplicable universe actions at the same source with byte-identical schema name. Select `u-` by minimum float64 squared Euclidean distance to `u+`, breaking exact distance ties by canonical action bytes. The seven equal-weight query categories are: all-zero vector; selected `u-`; `(1-alpha)u+ + alpha*u-` for ordered `alpha=[0.25,0.50,0.75]`; and two Gaussian queries `u+ + sigma*r_k*epsilon` for ordered `sigma=[0.01,0.05]`. If no eligible hard negative exists, record `NOT_APPLICABLE_NO_SAME_SCHEMA_NEGATIVE` for the swap/interpolation QUERY tensors and assign success `0` to those four categories; zero/Gaussian categories remain evaluated. This is the only permitted missing-query reason and is a conservative performance result, not malformed evidence.

For seed `k`, compute the Gaussian scale before holdout access from that seed's frozen winner development checkpoint: concatenate in canonical `(problem,source,action_bytes)` order every development full-universe candidate `A6` row, compute float64 sample std (`correction=1`) per action coordinate, and set `r_k=numpy.quantile(std_coordinates,0.5,method='linear')`; fewer than two rows, `r_k<=0`, or a nonfinite value is malformed. Test uses this frozen development `r_k` unchanged. For positive query record bytes `R=compact_json([split_name,k,problem_id,source_state_hash,canonical_action_bytes])+"\n"`, set `h=SHA256(b"query-epsilon-v1"||R)`, interpret `h[0:16]` as one unsigned little-endian 128-bit integer, initialize `numpy.random.Generator(numpy.random.PCG64(integer))`, and draw exactly `D_a` values with `standard_normal(size=D_a,dtype=float64)` in coordinate order. Reuse that one epsilon vector for both sigma values and compute every query in float64.

Decode each query by minimum float64 squared Euclidean distance to the complete same-source candidate latents; exact ties use canonical action-byte order. Success is the stored oracle-applicability Boolean of the decoded candidate. For each source/category, mean successes over all oracle-applicable positives; then take equal arithmetic means over the seven categories, distinct canonical source states within problem, and problems. A source with no positive, a missing universe candidate/latent/label, duplicated action bytes, nonfinite distance/query, changed development scale, or any population other than the complete universe is malformed. Persist positive/negative eligibility, selected-negative distances/ties, missing-negative flags, scale inputs/value, RNG integer, epsilon/query bytes, candidate distances, decoded bytes/labels, every reduction denominator, and hashes. This problem-macro query top-1 applicable rate is the value used by `transfer_ok`.

Cardinality transfer uses a separate uncapped, read-only `CARDINALITY_FULL_UNIVERSE` artifact, never the capped applicability-training/evaluation rows from Section 6. For each distinct canonical source state appearing in the complete evaluated-split window manifest, it contains every type-valid grounded action enumerated before category caps, sorted by canonical action bytes, with the offline-oracle Boolean applicability label and source/action payload hashes. Training/development artifacts are frozen before holdout access; the identical frozen generator produces the test artifact once inside the final command. This universe is evidence-only and never becomes a production generator or grants simulator access. The frozen winner encodes/scores every action exactly once by the selected action encoder and O3 applicability logit `ell_app`; this is the same persisted A6/logit forward consumed by query robustness, not a second encoding. It runs in contiguous action-byte chunks of 1024 with a retained final short chunk and the source-local zero temporal reset from Section 6; chunk boundaries, logits, labels, counts, and hashes are persisted. Missing, duplicate, filtered, or nonfinite rows make the corresponding split evidence malformed.

Descending score and canonical action-byte tie order define top K over this complete universe. Source-level applicable Recall@K is the fraction of all oracle-applicable actions in the first `min(K,type_valid_cardinality)` rows for `K in {8,32,128}`. A source with zero oracle-applicable actions, an incomplete universe, or undefined logits is malformed. Source `required_K` is the smallest listed K with Recall `>=0.90`, else `type_valid_cardinality+1`. The problem point is `(median source type_valid_cardinality using NumPy quantile(method='higher'), 90th-percentile source required_K using method='higher')`. Use one equal-weight point per problem, `x=ln(max(1,cardinality_point))`, `y=ln(max(1,required_K_point))`, and ordinary least squares with an intercept; zero full-split `Var(x)` is malformed. For exactly 100,000 bootstrap replicates, `numpy.random.Generator(numpy.random.PCG64(0))` samples the split's problem indices with replacement at the original problem count and refits the same slope; a replicate with zero `Var(x)` receives conservative slope `1.0`. The scaling upper bound is `numpy.quantile(slopes,0.975,method='linear')`. Every problem in the evaluated split—48 for development or 20 for untouched test—must contribute a nonempty point and the split must span at least three frozen training-cardinality quartiles. A malformed development cardinality artifact sets `pretest_evidence_ok=false`; a malformed untouched-test artifact sets `test_evidence_ok=false`. A complete finite but below-threshold recall or scaling result sets only `transfer_ok(split)=false` and is scientific performance evidence. Sparse-schema reporting metrics are the only skippable populations.

### Cardinality transfer

- candidate count and full type-valid cardinality;
- top-k applicable and trace/competitive-action recall for fixed `K`;
- performance versus object/action cardinality;
- no scalability claim if the required-K exponent upper bound is `>=1.0`.

## 10. Deterministic acceptance and branch decision

Development-only selection gates are frozen before test access. `geometry_ok` consumes only DL5A development S0-S3 artifacts; `sensitivity_ok` consumes only the 39 DL5B development artifacts; DL3P consumes only O0-O3 development contrasts. None is recomputed on test. The untouched command evaluates only the frozen DL6 winner and A0/O0 baseline, three seeds each, on all 20 problems and computes the parameterized confirmation predicates below. Unless a predicate explicitly names a paired baseline ratio, it applies to the frozen winner; baseline outputs remain mandatory finite comparators but do not have to satisfy winner thresholds. Missing required metrics, insufficient power, manifest mismatch, nonfinite values, or unavailable required schema strata make the corresponding evidence predicate false.

### 10.1 Boolean predicates

```text
sequential_pretest_artifacts_ok =
  evaluate gates only when authorized by every earlier gate
  and every reached gate has complete hash-valid finite artifacts and every required seed
  and every reached pretest evidence gate (including DL3P when reached) passes
  and a reached, complete, adequately powered performance-threshold FAIL is a valid scientific stop
  and every artifact after the first valid scientific stop is absent/NOT_AUTHORIZED
  and no missing artifact occurs before that stop

pretest_evidence_ok =
  DL0 manifest/split PASS
  and exact pretest problem coverage is training=130 and development=48
  and every required training/development problem/window appears in its reached phase artifact
  and sequential_pretest_artifacts_ok
```

The ordered pretest gates are DL1 `state_ok_dev_baseline`; the required A0/O0-O3 DL2 capacity gate; DL3P then O3 eligibility; DL4 winner freezing; DL5A `geometry_ok`; DL5B `sensitivity_ok`; and development confirmation/boundary predicates. DL3P failure, malformed data, missing support, nonfinite output, hash mismatch, or pretest resource infeasibility is an evidence failure and makes `pretest_evidence_ok=false`. DL6P occurs only after a development candidate exists and is owned exclusively by `holdout_readiness_ok`; its failure reaches branch-precedence clause 3 rather than rewriting pretest evidence. In contrast, a complete valid below-threshold result at `state_ok_dev_baseline`, required A0 DL2 capacity, O3 eligibility, `geometry_ok`, `sensitivity_ok`, or development confirmation is a scientific-performance stop: `pretest_evidence_ok` remains true, downstream cells stay `NOT_AUTHORIZED`, and `development_candidate_ok=false`. A failed optional A1-A4 micro-fit/retention gate merely excludes that architecture under DL4's rules and is not a whole-study stop. This reached-gate rule overrides no thresholds; it only distinguishes valid negative evidence from absent or inadequate evidence. Any `NOT_AUTHORIZED` conjunct in `development_candidate_ok` evaluates Boolean false without changing `pretest_evidence_ok`.

```text
test_evidence_ok =
  exactly the frozen DL6 winner and A0/O0 baseline are evaluated
  and every one of the 20 untouched-test problems/windows appears
  and all 6 checkpoint-seed artifacts and hashes are valid and finite

state_ok(split) =
  for every required precondition predicate:
    final-z_s AUROC >= 0.80
    and final-z_s AUROC >= graph-feature AUROC - 0.05
  and required predicates cover >=80% of held-out labeled examples
  and >=2 of 3 seeds meet all clauses

state_geometry_ok(split) =
  graph and object target matrices each satisfy:
    >=90% dimensions with std >=0.50
    effective rank >=0.50*D_s
    off-diagonal correlation RMS <=0.20
  in >=2 of 3 seeds

triple_ok(split) =
  per-seed problem-macro legal-distinct pairwise energy accuracy >= 0.80
  in >=2 of 3 seeds

app_ok(split) =
  applicability AUROC >= 0.80
  and groupwise top-1 applicable rate >= 0.80
  and >=2 of 3 seeds meet both

identity_ok(split) =
  complete-binding retrieval >= 0.80
  and problem-macro normalized binding margin > 0
  and >=2 of 3 seeds meet both

geometry_ok =
  the literal three-seed, problem-macro S0 predicate in Section 8.2 passes

boundary_trace_seed_ok(cell,seed) =
  for every separate registry ID: present, finite, shape-valid, supported, and hash-valid,
  or else carrying that ID's sole permitted explicit NOT_APPLICABLE reason;
  and no complete_collapse or dimensional_collapse at branch-semantic vector boundaries
  S4g/S4o; S5g/S5o; S6g/S6o; A6; P1g_l2/P1o_l2; P2g/P2o; Tg/To;
  Happ1pool/Happ1scorein/Happ1linear/Happ1gelu/Happ1drop; Harg1sum/Harg1gelu/Harg1drop
  and no semantic_drop at any registered predecessor/boundary pair
  and no conditional_collapse at S6g/S6o, A6, P2g/P2o, Happ1pool, or Happ1scorein
  and Happ2linear/Happ2/Harg2linear scores are finite, Harg2mask obeys its exact finite/`-inf` mask contract, and every separate L0-L5 scalar is finite
  with required class support
  and every required adjacent-boundary CKA/intervention/gradient record is present

boundary_trace_cell_ok(cell) =
  boundary_trace_seed_ok(cell,seed) in >=2 of 3 seeds

boundary_trace_ok_dev =
  boundary_trace_cell_ok(cell) passes for the DL5A S0 cell,
  every one of the nine DL5B three-seed sentinel cells, and the DL6 winner cell;
  comparator/baseline collapse remains causal evidence but cannot itself fail the winner

sensitivity_ok =
  all 39 DL5B runs and boundary traces are valid
  and the literal batch-span/default-near-best/Pareto/axial clauses in DL5B pass

boundary_final_seed_ok(split,seed) =
  at the frozen final winner checkpoint every separate registry ID is present/valid or explicitly NOT_APPLICABLE;
  no complete_collapse or dimensional_collapse at the branch-semantic vector boundaries above;
  no registered semantic_drop;
  no conditional_collapse at S6g/S6o, A6, P2g/P2o, Happ1pool, or Happ1scorein;
  and all registered head/logit/loss/intervention outputs are finite and supported

boundary_final_ok(split) =
  boundary_final_seed_ok(split,seed) in >=2 of 3 seeds

prediction_ok(split) =
  held-out positive normalized transition-loss ratio to paired A0/O0 <= 1.05
  in >=2 of 3 seeds

transfer_ok(split) =
  in at least 2 of 3 official seeds independently:
    query/perturbed-latent groupwise top-1 applicable rate >=0.80
    and problem-macro applicable Recall@128 >=0.90
    and that seed's required-K OLS scaling-exponent 97.5% bootstrap upper bound <1.0
```

For exact-shared-successor triples, no separation threshold exists; they must be excluded from `L_gap`. The secondary invalid successor-bank energy never enters these predicates.

```text
confirmation_ok(split) =
  state_ok(split)
  and state_geometry_ok(split)
  and triple_ok(split)
  and app_ok(split)
  and identity_ok(split)
  and boundary_final_ok(split)
  and prediction_ok(split)
  and transfer_ok(split)

development_candidate_ok =
  pretest_evidence_ok
  and O3 eligibility passed and the DL4 winner was frozen
  and geometry_ok
  and boundary_trace_ok_dev
  and sensitivity_ok
  and confirmation_ok(development)

holdout_readiness_ok =
  DL6P artifact is complete, hash-valid, and test_power_ok=true

dual_space_pass =
  development_candidate_ok
  and holdout_readiness_ok
  and test_evidence_ok
  and confirmation_ok(untouched_test)
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
1. if not pretest_evidence_ok:
     INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D
2. else if not development_candidate_ok:
     KEEP_BRANCH_D_ABSTRACT_ACTIONS
3. else if not holdout_readiness_ok:
     INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D
4. else if not test_evidence_ok:
     INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D
5. else if dual_space_pass:
     REOPEN_BOUNDED_DUAL_SPACE_DECODING
6. else:
     KEEP_BRANCH_D_ABSTRACT_ACTIONS
```

Localization flags are attached to the record after this selection. `BRANCH_D_ABSTRACT_ACTIONS` remains the operational default unless clause 5 fires. A clause-5 PASS authorizes only a new bounded decoding/planning specification, not planner implementation, tuning, commit, or promotion. Clauses 2 and 6 terminate this rescue study on performance evidence; no further coefficient or architecture sweep is authorized.

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

Expected hard caps, recalibrated in DL0 against one complete 130-problem pass:

```text
DL0–DL1 manifests/probes/traces: CPU-bound; <=24 wall-clock hours
DL2 micro-overfit:              <=2 GPU-hours including 2,000 transition + geometry and enabled auxiliary forwards per run
DL3 objective ablation:         exactly 12 one-pass checkpoints
DL4 architecture ladder:        <=15 one-pass checkpoints
DL5A regularizer causality:     exactly 12 one-pass checkpoints
DL5B batch/width sensitivity:   exactly 39 one-pass checkpoints
DL6 confirmation:               baseline + one winner, 3 seeds each, 4 passes
Total GPU cap:                  64 GPU-hours
```

DL0 measures one full-pass runtime and peak memory for widths 32/64/128 before authorizing the matrix. If the preregistered design would exceed 64 GPU-hours, this is pretest resource infeasibility: it sets `pretest_evidence_ok=false` and the authoritative Section 10.3 clause 1 returns `INSUFFICIENT_EVIDENCE_KEEP_BRANCH_D`. There is no separate resource-specific assessor verdict. Problems, seeds, cells, trace boundaries, and untouched-test coverage may not be silently reduced. These are caps, not entitlements. Sequential scientific FAIL stops later cells. A state-encoder redesign, planner, candidate generator, larger width/batch grid, or coefficient sweep requires a new specification.

## 13. Deliverables

1. Related-work/theoretical-contract note with primary citations.
2. Complete 198-problem inventory audit; immutable 130/48/20 train/development/test split; power analysis; frozen action/triple manifests.
3. Reviewed `LocalSIGRegLoss` state/action module contract, deterministic projection/group manifests, fixtures, and default configuration.
4. Full separate GN/GE/GM/S/A/P/T/Happ/Harg/L/QUERY registry traces over every development problem, including every indexed layer/order/component, with earliest-boundary localization.
5. Exact-successor energy, applicability, equivalence, predictor Jacobian, and gradient-conflict report.
6. DL2 micro-overfit record.
7. Fixed DL3 four-cell objective-ablation evidence.
8. Matched DL4 architecture-ladder evidence.
9. DL5A S0–S3 default-SIGReg necessity/locality evidence.
10. DL5B 39-checkpoint batch/width sensitivity and variance report.
11. One frozen DL6 confirmation evaluated once on all 20 untouched-test problems.
12. Deterministic assessor and human-readable decision record.
13. Independent review history through final PASS or explicit FAIL.

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