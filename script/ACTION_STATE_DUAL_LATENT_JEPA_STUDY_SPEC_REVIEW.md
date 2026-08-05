# Action/state dual-latent JEPA study specification review history

Date: 2026-08-04
Specification: `script/ACTION_STATE_DUAL_LATENT_JEPA_STUDY_SPEC.md`

## Review 1

Reviewer: independent adversarial research-specification review `deleg_2a6b8f33`
Result: FAIL
Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_204018_494950.txt`

Blocking findings:

1. The four objective conditions were named but not implemented as an identifiable fixed ablation.
2. The 9–18-checkpoint fractional design could not identify five binary factors.
3. The repository's VC-only losses were incorrectly/ambiguously called VICReg; state/action samples, formulas, and SIGReg were underspecified.
4. Invalid-action transition energy was not a valid primary applicability measure.
5. The taxonomy labeled actions where transition energy requires labeled triples and conflated exact successors with effect signatures.
6. M0/M1/M2 changed too many architecture properties to identify the effect of distinct state/action spaces.
7. Acceptance terms and overlapping branch outcomes were not executable by a deterministic assessor.
8. Held-out power, schema coverage, and untouched-test policy were insufficient.
9. The stated scope could not fit the causal questions within its checkpoint budget.

Corrections:

- Added fixed O0–O3 objective cells with exactly 12 paired-seed checkpoints and defined paired estimands.
- Replaced the unidentifiable factorial with sequential DL3 objective, DL4 matched architecture, DL5 anti-collapse, and DL6 confirmation gates with per-stage caps.
- Renamed existing losses VC, specified state/action ownership, formulas, constants, sample adequacy, schema-residual VC, and a concrete deterministic characteristic-function SIGReg cell.
- Made `E_app` the sole primary invalid-action validity measure; restricted `L_gap` to legal transition triples and made successor-bank energy secondary only.
- Split action and triple taxonomies; separated full canonical successors from effect signatures; specified deduplication and equal-state/category weighting.
- Replaced wholesale architectures with cumulative A0–A4 matched ablations, parameter matching, and numeric A4 authorization.
- Added numeric acceptance predicates, missing-data/evidence failure, localization flags, and ordered Branch D precedence.
- Added development-only power analysis, schema/cardinality coverage, and one-shot untouched test.
- Recalibrated the budget to explicit checkpoint counts and approximately 5–7 total GPU-hours.
- Added current target-path/gradient-flow caveat, action intervention tests, normalized energy, and primary JEPA/VICReg/SIGReg/action-representation citations.

Review 1 did not authorize implementation or training.

## Review 2

Reviewer: independent implementation-readiness review `deleg_8ce4416d`
Result: FAIL
Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_204858_804764.txt`

Blocking findings:

1. Combined normalized graph/object energy, gap margin/coefficient, applicability sign/BCE, and exact O0 auxiliary terms were underdefined.
2. Architecture tensors, target-path treatment, and objective promotion were not fully identifiable.
3. Ordered legal mismatch generation and canonical state/effect serialization were incomplete.
4. VC/SIGReg weighting, sampling, gradients, and state-geometry acceptance were ambiguous.
5. Power, randomization, hierarchy, and multiple-testing policies were deferred.
6. Retrieval, margins, valid-run semantics, perturbations, required K, tie policy, and cardinality strata were undefined.
7. DL2/DL4 counts and the total resource cap were inconsistent.

Corrections:

- Defined separately normalized graph/object transition energy, zero-scale failure, fixed gap margin/coefficient, applicability logit-to-energy sign, class-weighted BCE, calibration, and literal O0 coefficients with temporal similarity/inverse dynamics disabled.
- Removed stop-gradient/EMA as an authorized treatment; retained only a DL1 target-path diagnostic and addendum gate.
- Defined A1–A4/A4_SHAM tensors, descriptor vocabularies/serialization, initialization, dimensions, fusion, matched masking/sham controls, and O3-only promotion.
- Defined UTF-8 canonical state/action/effect bytes, both ordered mismatch directions, caps, hash selection, deduplication, and state-level weighting.
- Defined equal-schema R2 covariance, deterministic geometry batches, literal GraphVC target/population/gradient path, concrete PCG64 SIGReg projections/statistics/tests, and state-geometry thresholds.
- Fixed problem-level inference: minimum 12 development/test problems, quartile/schema support, alpha/power/MDE, exact sign flips, Holm hypotheses, problem bootstrap, and seed hierarchy.
- Defined predicates, complete-binding retrieval, normalized margins, all tie rules, valid evidence versus performance failure, query perturbations, Recall@K, required-K OLS scaling, and deterministic architecture/regularizer selection.
- Fixed DL2 at 16 plus at most 8 conditional micro-runs, DL4 at at most 15 new checkpoints, and the total cap at 8 GPU-hours.

Review 2 did not authorize implementation or training.

## Review 3

Reviewer: independent deterministic-readiness review `deleg_468314c4`
Result: FAIL
Reviewed SHA-256: `fb1204f190d0e27081d169353ed48e5d486da8c00777fbea75c2c158dbec4bf1`
Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_210308_932576.txt`

Blocking findings:

1. Legal mismatch records omitted the explicit comparator `a_j`.
2. State/action probes and exact applicability-head tensor inputs were not frozen.
3. State effective rank/correlation details were undefined.
4. R1/R2/R3 used inconsistent populations and batch sizes.
5. Power was not maximized over four contrasts and sign-flip boundaries were unspecified.
6. Architecture/transfer metrics, scores, empty populations, and source-to-problem aggregation were incomplete.
7. Optimizer/update/batch/checkpoint-selection budgets were not fixed.

Corrections:

- Mismatch records now store both action/successor payloads and pair-unit hashes; canonical comparator selection, paired caps, and averaging are literal.
- Frozen logistic state probes, detached action-binding probes, oracle substitution probes, and exact repository `ApplicabilityHead` arguments/dimensions/masks are specified.
- Participation-ratio effective rank, float64 eigendecomposition, clipping, zero variance, and state graph/object geometry gates are literal.
- R0–R3 now use one identical hash-frozen equal-schema geometry population/batch of at least `4*D_a` legal records.
- Per-hypothesis development variance/power, max required N, exact/Monte-Carlo p-values, Holm ordering, and strict boundaries are defined.
- The DL4 metric set/A4 sham rule, query perturbation decoder, O3 applicability score, source/problem required-K aggregation, OLS bootstrap, ties, and empty-population failure are defined.
- Seeds, Adam settings, batch/rollout/update counts, scheduler, data order, initialization, final-checkpoint rule, micro manifest, frozen/joint training boundaries, and 8-GPU-hour stop cap are fixed.
- Optional Review 3 polish was also addressed: F1 candidate thresholds, AUROC/AP/ECE, canonical bytes, gradient ownership, and SIGReg fixture metadata.

Review 3 did not authorize implementation or training.

## Review 4

Reviewer: independent readiness review `deleg_606c6611`
Result: FAIL
Reviewed SHA-256: `ae209ea36abc610da5fbc9a5a4b3fd7890010334bdd17611761804f31ed39204`
Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_211107_240385.txt`

Remaining blockers were orientation-dependent mismatch-pair ordering, inconsistent A4 authorization predicates, unspecified transfer seed aggregation, and contradictory DL2 state-VC gradients.

## Review 5

Reviewer: independent final-readiness review `deleg_91a12453`
Result: FAIL
Reviewed SHA-256: `2f6a4143d834be267beedb4ed813f7ee8d98330a6453630514162b1c2c2bbfed`
Report: `/opt/data/cache/delegation/subagent-summary-0-20260804_211303_679500.txt`

Review 5 confirmed all Review 3 blockers except two contradictions: DL2 state-encoder/VC gradient ownership and A4's ordinary-versus-query applicability authorization metric. It also confirmed the core scientific theory and bounded Branch D precedence.

Corrections after Reviews 4–5:

- Successor groups, pair orientation, orientation-independent pair hash, stratum key/order, round-robin traversal, and train/evaluation pair-unit caps are now canonical.
- Section 5 and DL4 now share one A4 predicate: A3 binding `>=0.80`, positive-loss ratio `<=1.05`, and low ordinary applicability top-1 or legal-distinct accuracy.
- `transfer_ok` now requires all three transfer clauses independently in at least two of three seeds.
- DL2 is consistently frozen-state: state VC is diagnostic/no-grad throughout DL2–DL5 and trainable only in joint DL6.

Neither Review 4 nor Review 5 authorized implementation or training.

## Review 6

Reviewer: independent final research-specification review `deleg_e1b3758c`
Result: **PASS**
Reviewed specification SHA-256: `5c880621fb87d965f84ffec5d9b5df6606fea73b166839dda24510f5fbddbb1e`
Live transcript: `/opt/data/cache/delegation/live/deleg_e1b3758c/task-0.log`

Review 6 verified that:

- canonical orientation-independent pair-unit selection is deterministic;
- Sections 5 and DL4 use one authoritative A4 authorization predicate;
- `transfer_ok` requires all three clauses independently in at least two of three seeds;
- DL2 freezes the state encoder and treats state VC as diagnostic/no-grad, while state-VC gradients are enabled only in joint DL6;
- all seven Review 3 blockers remain closed;
- no new implementation-changing contradiction was introduced; and
- the study is scientifically coherent, falsifiable, bounded, and deterministic enough to authorize a separate strict-TDD implementation plan.

Authorization boundary: this PASS authorizes planning only. It does not authorize implementation, training, tuning, planner integration, or promotion. `BRANCH_D_ABSTRACT_ACTIONS` remains the operational default.

## Review 20

Date: 2026-08-05
Reviewer: independent adversarial exact-SHA review `deleg_513931ae`
Result: **PASS**
Reviewed specification SHA-256: `ada8fb118e52e876a0bda47d69ea663d9cac74c5cac44f1e05505affa9fabb58`
Live transcript: `/opt/data/cache/delegation/live/deleg_513931ae/task-0.log`

Review 20 independently inspected the exact locked bytes and verified:

- complete graph, object, state, action, predictor, target, applicability, argument, loss, and query boundary coverage against the repository implementation;
- the exact applicability DAG, including slot/object multiplicative interaction and graph/object-summary absolute-difference paths;
- architecture, objective, loss, and gradient ownership;
- deterministic populations, estimators, schedules, reductions, and initialization;
- the complete `130/48/20` campaign protocol, `3x7`/39-checkpoint sensitivity design, resource cap, sequential gates, and branch precedence;
- locally conditioned SIGReg as the preregistered default with matched causal controls;
- transition compatibility/applicability separation and untouched-test isolation; and
- no implementation-changing blocker in the reviewed bytes.

Authorization boundary: this exact-SHA PASS authorizes only preparation of a separate strict-TDD implementation plan. It does not authorize implementation, training, tuning, evidence generation, planner integration, promotion, commit, or push. `BRANCH_D_ABSTRACT_ACTIONS` remains active.
