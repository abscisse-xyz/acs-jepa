# Updated Phase 0 assessment

## Compact metrics

- `residual_effective_rank_baseline`: `6.371300938481334`
- `residual_effective_rank_phase2`: `10.334873163207176`
- `within_schema_fraction_baseline`: `1.4071546829893256e-05`
- `within_schema_fraction_phase2`: `3.694867485187534e-06`
- `latent_auroc`: `0.8446078431372549`
- `latent_ap`: `0.37933323482921644`
- `latent_role_swap_margin`: `0.15346449613571167`
- `latent_one_arg_margin`: `0.5543402433395386`
- `raw_auroc`: `0.9936274509803922`
- `raw_ap`: `0.9358169608169609`
- `hybrid_auroc`: `0.9715686274509804`
- `hybrid_ap`: `0.8437950992362758`
- `transition_equivalence_rate`: `1.0`
- `transition_error_margin_median`: `-7.279481795032915e-08`
- `ranking_baseline`: `{"hybrid": {"auroc": 0.9754901960784313, "average_precision": 0.8708616831197477, "pairwise_applicable_accuracy": 0.9775280898876404, "ranks_applicable": true, "top1_applicable_rate": 1.0}, "latent_applicability": {"auroc": 0.8578431372549019, "average_precision": 0.424452132493965, "pairwise_applicable_accuracy": 0.8932584269662921, "ranks_applicable": false, "top1_applicable_rate": 0.5454545454545454}, "latent_transition": {"auroc": 0.578921568627451, "average_precision": 0.14981284009344284, "pairwise_applicable_accuracy": 0.449438202247191, "ranks_applicable": false, "top1_applicable_rate": 0.09090909090909091}, "raw_symbolic": {"auroc": 0.9936274509803922, "average_precision": 0.9358169608169609, "pairwise_applicable_accuracy": 0.9775280898876404, "ranks_applicable": true, "top1_applicable_rate": 0.9090909090909091}, "role_object": {"auroc": 0.4730392156862745, "average_precision": 0.10377833634176827, "pairwise_applicable_accuracy": 0.5786516853932584, "ranks_applicable": false, "top1_applicable_rate": 0.0}}`
- `ranking_phase2`: `{"hybrid": {"auroc": 0.9715686274509804, "average_precision": 0.8437950992362758, "pairwise_applicable_accuracy": 0.9775280898876404, "ranks_applicable": true, "top1_applicable_rate": 1.0}, "latent_applicability": {"auroc": 0.8446078431372549, "average_precision": 0.37933323482921644, "pairwise_applicable_accuracy": 0.8764044943820225, "ranks_applicable": false, "top1_applicable_rate": 0.36363636363636365}, "latent_transition": {"auroc": 0.5794117647058824, "average_precision": 0.12963634062559148, "pairwise_applicable_accuracy": 0.5168539325842697, "ranks_applicable": false, "top1_applicable_rate": 0.0}, "raw_symbolic": {"auroc": 0.9936274509803922, "average_precision": 0.9358169608169609, "pairwise_applicable_accuracy": 0.9775280898876404, "ranks_applicable": true, "top1_applicable_rate": 0.9090909090909091}, "role_object": {"auroc": 0.5627450980392157, "average_precision": 0.16422531495312, "pairwise_applicable_accuracy": 0.7078651685393258, "ranks_applicable": false, "top1_applicable_rate": 0.18181818181818182}}`

## Stage verdicts

- `evidence`: **PASS**
- `residual`: **FAIL**
- `recoverability`: **PASS**
- `transition_equivalence`: **FAIL**
- `ranking`: **PASS**

## Decision booleans

- `representation_ok`: `false`
- `latent_separable`: `true`
- `hybrid_separable`: `true`
- `raw_separable`: `true`
- `latent_rank`: `false`
- `raw_rank`: `true`
- `hybrid_rank`: `true`
- `mostly_transition_equivalent`: `true`
- `transition_distinguishable`: `false`

## Evaluated precedence

1. `FIX_DATA_LABEL_CONSTRUCTION` — `not raw_separable` = `false` (not selected because predicate is false).
2. `BRANCH_D_ABSTRACT_ACTIONS` — `mostly_transition_equivalent` = `true` (selected; later clauses skipped).
3. **SKIPPED** — later clause not evaluated after clause 2 selected the action.
4. **SKIPPED** — later clause not evaluated after clause 2 selected the action.
5. **SKIPPED** — later clause not evaluated after clause 2 selected the action.
6. **SKIPPED** — later clause not evaluated after clause 2 selected the action.
7. **SKIPPED** — later clause not evaluated after clause 2 selected the action.

Selected research action: `BRANCH_D_ABSTRACT_ACTIONS`

## Exact commands

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python script/diagnose_action_candidate_ranking.py /opt/data/workspace/acs-jepa-tuning-data/smoke --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json --recoverability-summary /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/summary.json --recoverability-details /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/details.json --recoverability-feature-schema /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/feature_schema.json --recoverability-split-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/split_manifest.json --recoverability-probe-states /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/probe_states.json --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run1 --device cpu --split val --epochs 200 --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```
```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python script/diagnose_action_candidate_ranking.py /opt/data/workspace/acs-jepa-tuning-data/smoke --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json --recoverability-summary /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/summary.json --recoverability-details /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/details.json --recoverability-feature-schema /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/feature_schema.json --recoverability-split-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/split_manifest.json --recoverability-probe-states /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/probe_states.json --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run2 --device cpu --split val --epochs 200 --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```
```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python script/diagnose_action_candidate_ranking.py /opt/data/workspace/acs-jepa-tuning-data/smoke --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json --recoverability-summary /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/summary.json --recoverability-details /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/details.json --recoverability-feature-schema /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/feature_schema.json --recoverability-split-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/split_manifest.json --recoverability-probe-states /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/probe_states.json --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run1 --device cpu --split val --epochs 200 --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```
```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python script/diagnose_action_candidate_ranking.py /opt/data/workspace/acs-jepa-tuning-data/smoke --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json --recoverability-summary /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/summary.json --recoverability-details /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/details.json --recoverability-feature-schema /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/feature_schema.json --recoverability-split-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/split_manifest.json --recoverability-probe-states /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/probe_states.json --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run2 --device cpu --split val --epochs 200 --learning-rate 0.001 --hidden-dim 64 --seed 20260717
```
```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python script/assess_action_latent_updated_phase0.py --updated-spec /opt/data/workspace/acs-jepa/script/ACTION_LATENT_UPDATED_SPEC.md --baseline-checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt --baseline-config /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml --phase2-checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt --phase2-config /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml --corpus-manifest /opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json --candidate-manifest /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json --baseline-schema-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/baseline/run1/summary.json --baseline-schema-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/baseline/run2/summary.json --phase2-schema-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/phase2/run1/summary.json --phase2-schema-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/phase2/run2/summary.json --baseline-recoverability-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/summary.json --baseline-recoverability-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run2/summary.json --phase2-recoverability-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/summary.json --phase2-recoverability-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run2/summary.json --baseline-transition-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run1/summary.json --baseline-transition-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run2/summary.json --phase2-transition-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/phase2/run1/summary.json --phase2-transition-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/phase2/run2/summary.json --baseline-ranking-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run1/summary.json --baseline-ranking-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run2/summary.json --phase2-ranking-run1 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run1/summary.json --phase2-ranking-run2 /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run2/summary.json --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/assessment
```

## Input artifacts

- `/opt/data/workspace/acs-jepa/script/ACTION_LATENT_UPDATED_SPEC.md` — `b4146d21b6082ec085628f7d1c56ff135c9fe606c8307db8b84689e449ec9606` (19273 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt` — `65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e` (3724372 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml` — `f65e2cbb33fb3e7322e0cc0c5e8a8f01e9ca7c408e4594516d50a9735c673193` (2441 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt` — `7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968` (4409140 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml` — `01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5` (3354 bytes)
- `/opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json` — `055b5616d7616331e6edbc8f72523f07e8c1808e5aa31089c8420f01aaf0e400` (4679 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json` — `bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9` (117385 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/baseline/run1/summary.json` — `f428905cacf2fd809a6f5504be8b3a7f525893f30f4ec32daf1e69d148ce1843` (75970 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/baseline/run2/summary.json` — `3b600377cf1d504dc9f96a5268318eeae9652011735abcddb1951548d443d558` (75970 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/phase2/run1/summary.json` — `031da2b75aad2e843c41c890b9e33733dff2c01f5622588190881386b445c49b` (75990 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/schema_residual/phase2/run2/summary.json` — `f947895e0851ba17e4db60cd6ded288b8826576bb9d58cb0884c744fdaa14fd6` (75989 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1/summary.json` — `ffc198ebec8c918574330ffc187c729f571a4e4f2610872dfbdfd7d79e1a8c4f` (209570 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run2/summary.json` — `8b119febbcc889ba4224bd3bd7653dde1519672b001ae35fb2a8f3d9491a34af` (209570 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run1/summary.json` — `0001ccfc4a4906386873471a1d4013a3c2b9d5856664596d4b9669cfc60c3ea7` (209318 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/phase2/run2/summary.json` — `253d85726cce96a2bb6b50720c09a3d3e15c747d3184e29044396819161b2b1a` (209317 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run1/summary.json` — `60eeb8ca1a7f3d2de4fb388d076cdf1204bba0534b42d45a4916ec2ddd033f09` (4071 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run2/summary.json` — `e0b0800daedd6e2fead47d70e9ec3cd1ced32c1d08892e2646f02cdfa543ced7` (4070 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/phase2/run1/summary.json` — `bbb089690974877d160bb877865b92c005bc6cfa1a5cfc390a022fc92e623cda` (4074 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/phase2/run2/summary.json` — `33814b4992840c7c1271667ba0ed2beef6e1be6cbb067fb9b212c32288741204` (4074 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run1/summary.json` — `9d12451109f0541a75a45805ab6097ab2d1925fbfb9d7b70e36332843385dd78` (27069 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/baseline/run2/summary.json` — `c8cf77b1888f2c41554782fa0b4216fbbd773442f79acb878e670967292c8eec` (27069 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run1/summary.json` — `732569ce0e8a06de9761003cd6de5b6b53c55f6bba45f6153b23cad2fdbb836a` (26939 bytes)
- `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/candidate_ranking/phase2/run2/summary.json` — `5458e87b1c592509e70a9acafded373b7dc6bdb3a34facd34ae3c4f1df2991ba` (26939 bytes)
