# Phase 1H Supervised Action Probe Diagnostic Plan

Goal: complete the original Phase 1 deliverable by adding a diagnostic script that trains/evaluates supervised action probes for schema id, per-role object id, and applicability from smoke data/checkpoints, without changing planner/decoder behavior or the JEPA training loop.

## Scope

Implement:

- `script/diagnose_action_supervised_probes.py` or equivalent.
- Probe training/eval over checkpoint-derived frozen representations.
- Metrics JSON for:
  - schema accuracy from action latent;
  - per-role object accuracy from action latent + state context;
  - applicability accuracy/AUROC when labels contain both classes;
  - hard-negative applicability logit margins, especially same-schema negatives.
- Use smoke validation p166/p192 / 44 transitions by default; allow larger slices via args.
- Use offline oracle labels only, e.g. `applicable_keys(engine)` membership during diagnostic data construction.

Do not implement:

- broad tuning;
- full JEPA retraining;
- decoder/planner score integration;
- action VICReg/SIGReg;
- contrastive auxiliary training loss;
- production `applicable_actions()` generation;
- new checkpoint schema changes.

## Inputs

Default target artifacts:

- Dataset: `/opt/data/workspace/acs-jepa-tuning-data/smoke`
- Checkpoints:
  - `default_seed0/checkpoints/best.pt`
  - `inverse_dynamics_seed0/checkpoints/best.pt`
  - `action_encoder_rnn_seed0/checkpoints/best.pt`
- Split: `val`
- Seed: `20260717`

## Design

1. Replay selected transitions with existing diagnostic helpers.
2. For each state/action:
   - encode source state with checkpoint JEPA;
   - encode true action and sampled negatives with checkpoint action encoder;
   - gather selected object latents and masks for candidate arguments;
   - label applicability with offline `applicable_keys(engine)` membership.
3. Train lightweight diagnostic probes on frozen tensors:
   - schema probe: action latent -> action schema id;
   - role-object probe: action latent + role slot/state context -> problem-local object id per role, masked by real roles;
   - applicability probe/head: graph latent + action latent + object slot latents -> binary applicability.
4. Split examples deterministically into disjoint probe-train and probe-eval partitions and record the split seed/counts in JSON. Report train and eval metrics separately. Since Phase 1 targets surfacing limitations, the 44 p166/p192 transitions are acceptable.
5. Define each hard-negative margin as `positive applicability logit - negative applicability logit`; report aggregate and per-category distributions, including same-schema one-argument substitutions.
6. Emit JSON summary and details. Keep models ephemeral unless a small output state dict is useful for inspection.

## TDD plan

Add tests under `acs-jepa-cli/tests/` for reusable script helpers, using tiny synthetic tensors/actions:

1. RED: example builder preserves true positives, same-schema negatives, labels, object slots, and masks without exposing oracle labels as probe input features.
2. RED: schema probe training reports accuracy in `[0,1]` and overfits a tiny separable fixture.
3. RED: role probe reports per-role accuracy, ignores masked/padded roles, and handles problem-local object vocabularies with different object counts.
4. RED: applicability metrics report accuracy, optional AUROC, and positive-minus-negative logit margin; absent-class AUROC is `None`.
5. RED: CLI arg validation rejects empty epochs/batch sizes/negative counts and writes `summary.json`/`details.json` on a tiny fixture path where practical.

## Acceptance criteria

- Produces the original Phase 1 deliverable: supervised probe diagnostic script + JSON metrics.
- Reports schema, role-object, applicability, and hard-negative margin metrics.
- Uses a deterministic, recorded train/eval split over the 44-transition p166/p192 smoke validation slice and reports both partitions separately.
- Completes smoke runs for baseline and inverse-dynamics checkpoints; RNN is optional comparative evidence.
- Uses offline oracle labels only inside diagnostics.
- Leaves planner/decoder/trainer production behavior unchanged.
- Verification includes targeted tests, ruff, compileall, and at least one smoke checkpoint probe run if runtime allows.

## Verification commands

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli pytest acs-jepa-cli/tests/test_action_supervised_probes.py -q
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --dev ruff check script/diagnose_action_supervised_probes.py acs-jepa-cli/tests/test_action_supervised_probes.py
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python -m compileall script/diagnose_action_supervised_probes.py
```

Smoke run, adapted per checkpoint:

```bash
UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python \
  script/diagnose_action_supervised_probes.py \
  /opt/data/workspace/acs-jepa-tuning-data/smoke \
  --checkpoint /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt \
  --output /opt/data/workspace/acs-jepa-runs/smoke/default_seed0/action_supervised_probes_val \
  --device auto \
  --split val \
  --seed 20260717
```

## Code-review gate

Plan review must PASS before coding. Implementation review must PASS before commit.
