# Phase 2F fixed action-auxiliary component smoke

Date: 2026-07-21
Status: execution PASS; independent implementation/evidence Review 2 PASS
Prerequisite: signed Stage 2E commit `0a59c11d5a8830fb25611b6a9b23759739db9e8a`

## Scope

This is the fixed Phase 2F component execution gate. It verifies that the Stage 2 action-identifiability auxiliaries can be enabled together, trained, evaluated, serialized, and restored without NaN/Inf or online simulator/planner/oracle use. It is not a Phase 2G efficacy or representation-acceptance claim.

No planner, decoder, CEM, replay callback, tuning search, or broad campaign was run.

## Fixed inputs

- Dataset: `/opt/data/workspace/acs-jepa-tuning-data/smoke`
- Seed: `0`
- Split seed: `20260717`
- Training: 3 epochs, batch size 8, rollout length 4, CUDA
- Config stack:
  1. `script/configs/adaptive/base.yaml`
  2. `script/configs/adaptive/00_smoke/default_smoke.yaml`
  3. `script/configs/adaptive/01_action_decode/action_auxiliary_smoke.yaml`
- Output: `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0`
- Evaluation output: `/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/eval`
- GPU: NVIDIA GeForce RTX 2080 Ti, 11,264 MiB

Fixed auxiliary coefficients were 0.1 for Action VICReg, action contrastive, argument reconstruction, and integrated applicability; SIGReg remained disabled. Four deterministic candidates were requested per positive. `applicability_pos_weight` was 3.0, the rounded deterministic all-corpus candidate ratio (train 4063/1357 = 2.9941, held-out 563/197 = 2.8579, all-corpus 4626/1554 = 2.9768).

## Offline applicability artifact

Successful command:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli \
  python script/build_action_applicability_table.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/action_applicability.json
```

Result:

- wall time: 97.76 seconds
- parsed problem entries: 24
- trajectories: 12
- transitions/source states: 345
- strict table entries: 345
- missing source states: 0
- trajectory problem indices: `1, 3, 5, ..., 23`
- artifact size: 15,591,079 bytes
- SHA-256: `acfb600d0236c6e970e059a8b1b738edc1528d9984beedb861f071b799d482a1`
- semantics: `positive_ground_atoms_closed_world_v1`

Every trajectory was replayed in a fresh offline simulator. Before each applicability query, simulator facts matched the recorded source facts, the recorded action belonged to the complete applicable-action set, and final simulator facts matched the recorded terminal facts. The strict Stage 2E loader accepted the generated table.

A first timing wrapper invocation stopped before producer execution because `/usr/bin/time` was unavailable. The new output directory was verified empty, then the successful command above was run using the shell `time -p` builtin. No stale metrics or partial artifact were present.

## CPU preflight

The exact three-config stack and generated artifact were loaded against the real corpus. A real collated batch completed train and eval steps on CPU with all auxiliary terms finite. The three optional trainable modules contributed 24 parameters, and every one was optimizer-owned exactly once. The resulting dataset contained 309 windows.

## Fixed GPU training

Successful command:

```bash
ACS_JEPA_ACTION_APPLICABILITY_TABLE=/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/action_applicability.json \
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli acs-jepa train \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0 \
  --config script/configs/adaptive/base.yaml \
           script/configs/adaptive/00_smoke/default_smoke.yaml \
           script/configs/adaptive/01_action_decode/action_auxiliary_smoke.yaml \
  --device cuda --seed 0
```

Result:

- exit: 0
- wall time: 158.95 seconds
- measured training runtime: 150.0345 seconds
- optimizer steps: 102
- training examples across epochs: 813
- train windows per epoch: 271
- held-out windows: 38
- held-out problems: `p166`, `p192`
- train metric records: 102
- held-out metric records: 8

Final training record:

| Metric | Value |
|---|---:|
| `jepa_loss` | 0.5294389129 |
| `action_vicreg_loss` | 0.9363151789 |
| `action_contrastive_loss` | 1.6435067654 |
| `argument_reconstruction_loss` | 2.3908569813 |
| `applicability_loss` | 1.0238683224 |
| `term/action_vicreg_num_samples` | 28.0 |
| `term/action_contrastive_num_examples` | 28.0 |
| `term/action_contrastive_num_negatives` | 112.0 |
| `term/argument_num_active_roles` | 86.0 |
| `term/applicability_num_examples` | 140.0 |
| `term/applicability_num_positive` | 38.0 |
| `term/applicability_num_negative` | 102.0 |

Final held-out validation record:

| Metric | Value |
|---|---:|
| `jepa_loss` | 0.5290471554 |
| `action_vicreg_loss` | 0.9299906015 |
| `action_contrastive_loss` | 1.5737522125 |
| `argument_reconstruction_loss` | 2.2233963966 |
| `applicability_loss` | 1.0186666846 |
| `term/action_vicreg_num_samples` | 30.4 |
| `term/action_contrastive_num_examples` | 30.4 |
| `term/action_contrastive_num_negatives` | 121.6 |
| `term/argument_num_active_roles` | 103.2 |
| `term/applicability_num_examples` | 152.0 |
| `term/applicability_num_positive` | 39.4 |
| `term/applicability_num_negative` | 112.6 |

Every required loss was present and finite, and every required count was positive in every training and held-out record.

## Post-checkpoint all-corpus evaluation

Successful command:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli acs-jepa eval \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt \
  --output /opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/eval \
  --device cuda
```

Result:

- exit: 0
- wall time: 58.95 seconds
- measured evaluation runtime: 49.9408 seconds
- examples: 309
- scope: complete corpus, not held-out validation

| Metric | Value |
|---|---:|
| `jepa_loss` | 0.5291080200 |
| `action_vicreg_loss` | 0.9321032166 |
| `action_contrastive_loss` | 1.5965533960 |
| `argument_reconstruction_loss` | 2.3064594850 |
| `applicability_loss` | 1.0146016066 |
| `term/action_vicreg_num_samples` | 31.6923 |
| `term/action_contrastive_num_examples` | 31.6923 |
| `term/action_contrastive_num_negatives` | 126.7692 |
| `term/argument_num_active_roles` | 107.7436 |
| `term/applicability_num_examples` | 158.4615 |
| `term/applicability_num_positive` | 39.8462 |
| `term/applicability_num_negative` | 118.6154 |

All required losses were finite and all required counts were positive.

## Checkpoints and resolved configuration

Both checkpoints contain non-`None` states for:

- `action_contrastive_anchor_state_dict`
- `argument_reconstruction_head_state_dict`
- `applicability_head_state_dict`

Artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/best.pt` | 4,409,140 | `7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968` |
| `checkpoints/latest.pt` | 4,429,642 | `a17fd519bb184faaa65d32d664f9b5126b507a29f65d5c1af0ee4f0b8b3594ea` |
| `config.yaml` | 3,354 | `01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5` |

The saved configuration contains the exact fixed coefficients, seed, four-negative count, writable MLflow URI, and resolved absolute applicability artifact path. CLI evaluation restored that configuration without re-exporting the environment variable.

## Test and quality gates

- producer/config focused CLI tests: 11 passed
- focused core trainer tests: 18 passed
- full core suite: 296 passed
- CLI suite excluding the documented unrelated missing tuning-config discovery test: 74 passed
- Ruff on every changed Python file: passed
- `compileall` on every changed Python file: passed
- `git diff --check`: passed
- static scan: the only new subprocess call is a test using an argv list with `shell=False`; no dynamic `eval`, `exec`, unsafe YAML load, or production oracle import was introduced

## Phase 2F conclusion

The fixed component smoke passes its execution criteria: all objectives ran together, emitted finite losses and positive effective counts, persisted and restored optional module states, and covered every recorded source state with immutable offline labels. Phase 2G remains required to assess loss direction, latent margins, action applicability separation, object-binding aliasing, and collapse.
