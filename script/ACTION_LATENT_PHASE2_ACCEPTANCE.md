# Phase 2 action-latent empirical acceptance

Date: 2026-07-21
Decision: **FAIL**
Scope: fixed Stage 2G assessment on the 44-transition `p166`/`p192` validation slice
Training/tuning performed in this stage: none
Planner/decoder/CEM execution: none

## Fixed inputs

| Artifact | SHA-256 |
|---|---|
| baseline `best.pt` | `65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e` |
| Phase 2 `best.pt` | `7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968` |
| Stage 2F train metrics | `f3e94b0d6c8a38b78ba6ce209f6c0ab31a3cf49bac1c450affcaf60ca30f0e43` |
| Stage 2F held-out metrics | `5ce3ab7aa535bf68990e000b79c8cd29a1bf14b68670db67eb06a19d5f123954` |
| Stage 2F resolved config | `01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5` |
| Stage 2F split manifest | `02aa33b0aa12008142fe08940a16aff81554d7fa3c2345866be6d9b65d9842ae` |
| smoke corpus manifest | `055b5616d7616331e6edbc8f72523f07e8c1808e5aa31089c8420f01aaf0e400` |

The assessor validated these identities from bytes. Its complete generated evidence manifest is:

`/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/assessment/evidence_manifest.json`

## Commands

The placeholder-free commands and absolute output paths were preregistered in:

`.hermes/plans/2026-07-21_194939-phase2g-action-latent-acceptance.md`

Executed without modification:

1. full same-schema latent statistics for baseline and Phase 2 checkpoints, each covering all 174,780 candidates and 44 validation transitions;
2. two deterministic CPU supervised-diagnostic runs per checkpoint;
3. one deterministic G1–G8 assessment over the fixed outputs.

Common statistics invocation:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_latent_statistics.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint CHECKPOINT --output FIXED_OUTPUT \
  --device cuda --split val --same-schema-only \
  --chunk-size 2048 --seed 0 --omit-details
```

Common probe invocation, run twice per checkpoint:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_supervised_probes.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint CHECKPOINT --output FIXED_OUTPUT \
  --device cpu --split val --per-category 4 \
  --eval-fraction 0.25 --epochs 200 \
  --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```

The assessor command is reproduced in full in the approved plan. It returned `0` for a valid assessment and printed `FAIL`; malformed evidence would return nonzero.

Approximate wall times were 96.8 seconds per statistics run and 32.1–32.4 seconds per probe run.

## Identity and determinism

- each probe run: 604 examples, 453 probe-train and 151 probe-eval;
- canonical manifest: 117,385 bytes;
- manifest SHA-256 in all four runs: `bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9`;
- categories: 176 one-argument substitutions, 176 random-other-schema, 176 random-same-schema, 32 role swaps, 44 traces;
- labels: 62 applicable and 542 inapplicable;
- repeat decision projections were byte-identical for each checkpoint;
- baseline correctly restored JEPA/goal states and constructed no disabled auxiliary heads;
- Phase 2 strictly restored JEPA, goal, contrastive anchor, argument head, and applicability head; no missing/null/random head state was accepted.

## Compact results

### Latent collapse and geometry

| Metric | Baseline | Phase 2 | Required | Result |
|---|---:|---:|---:|---|
| global effective rank | 2.127568 | 1.761690 | >= 4 and no regression | FAIL |
| global minimum dimension std | 0.007334 | 0.013615 | >= 0.02 and no regression | FAIL |
| within-schema variance fraction | 1.40715e-5 | 3.69487e-6 | >= 0.001 and no regression | FAIL |
| raw nearest wrong median L2 | 8.97198e-5 | 2.55976e-4 | >= 1.79374e-4 | PASS |
| raw nearest wrong minimum L2 | 3.32683e-5 | 4.60731e-5 | >= 3.32330e-5 | PASS |
| unit-L2 nearest wrong median | 7.51907e-5 | 2.04703e-4 | >= 1.5x baseline | PASS |
| unit-L2 nearest wrong minimum | 2.75618e-5 | 4.21338e-5 | no regression | PASS |

Near-miss distances improved after scale normalization, so the G4 result is not explained only by latent norm inflation. However, global rank regressed and within-schema variance fraction fell by approximately 74%, so collapse/aliasing is not repaired.

### Fitted frozen-representation probes

| Metric | Baseline | Phase 2 | Result |
|---|---:|---:|---|
| schema eval accuracy | 0.549669 | 0.662252 | PASS/no regression |
| role/object eval accuracy | 0.133205 | 0.183398 | PASS |
| applicability AUROC | 0.615686 | 0.638725 | FAIL (< 0.70) |
| applicability AP | 0.175704 | 0.182121 | FAIL (< 0.25) |
| applicability F1 | 0.000000 | 0.000000 | FAIL (< 0.25) |
| applicability overall median margin | 6.65188e-5 | 0.00146103 | positive |
| applicability role-swap median margin | -1.40190e-4 | -1.07288e-5 | FAIL (not positive) |

Phase 2 role accuracies were 0.165563, 0.218543, 0.142857, and 0.216867 for roles 0–3. The preregistered G5 gate passed, but applicability separation remained weak and the zero-threshold classifier predicted no positives.

### Untouched trained applicability head

| Metric | Phase 2 |
|---|---:|
| accuracy | 0.708609 |
| precision | 0.146341 |
| recall | 0.400000 |
| F1 | 0.214286 |
| AUROC | 0.583824 |
| AP | 0.147178 |
| overall median margin | 9.28938e-5 |
| role-swap median margin | -3.93689e-5 |

The direct head failed AUROC, AP, F1, and role-swap margin thresholds. Its performance is not hidden by the fitted-probe protocol.

### Untouched trained argument head

| Metric | Phase 2 |
|---|---:|
| active roles | 146 |
| competitive roles | 146 |
| top-1 accuracy | 0.315068 |
| chance reference | 0.144863 |
| median target-minus-best-wrong margin | -0.00576854 |

All four roles were present and competitive. Overall retrieval accuracy exceeded both 0.25 and twice chance, but the median target margin was negative; G8 therefore failed.

### Stage 2F activity check

All four held-out auxiliary losses decreased from the first to final monotonic step:

| Loss | Relative decrease |
|---|---:|
| Action VICReg | 2.4079% |
| action contrastive | 0.2243% |
| applicability | 1.9762% |
| argument reconstruction | 0.3591% |

All eight effective counts stayed positive. This retrospective activity gate passed but is not efficacy evidence.

## Gate decision

| Gate | Verdict |
|---|---|
| G1 evidence identity/completeness | PASS |
| G2 retrospective auxiliary activity | PASS |
| G3 collapse/aliasing | **FAIL** |
| G4 true-vs-near-miss geometry | PASS |
| G5 fitted grounded argument information | PASS |
| G6 fitted applicability separability | **FAIL** |
| G7 restored trained applicability head | **FAIL** |
| G8 restored trained argument head | **FAIL** |

Overall: **FAIL** because all gates are conjunctive.

Machine-readable decision:

`/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/assessment/summary.json`

## Interpretation and next allowed action

The Phase 2 auxiliaries produce useful local changes: scale-robust near-miss distances and frozen role/object retrieval improve. They do not repair the governing failure:

- action-latent effective rank regresses;
- within-schema argument/state variation becomes an even smaller share of total variation;
- both fitted and trained applicability heads fail separation gates;
- the trained argument head has negative median target-vs-best-wrong margin.

Broad tuning remains paused. The smallest evidence-driven next phase is a separately planned **candidate-level, schema-conditioned anti-alias objective** over trace actions plus the already deterministic same-schema hard negatives. It should test SIGReg/VICReg-style spread on candidate action latents and normalized ranking margins while leaving the applicability and argument heads fixed, so the causal effect on G3/G4 can be isolated before changing heads or tuning coefficients. No such implementation or training begins in Stage 2G.
