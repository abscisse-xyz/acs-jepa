# Action Latent Failure Root-Cause Note

Date: 2026-07-17

## Executive summary

The failure is not primarily action-schema selection. The current checkpoints
usually identify the correct action schema, but they do not provide a robust
object-binding or applicability-aware energy landscape. CityCar makes this
especially brittle: applicable actions are a tiny subset of type-valid grounded
actions, and many same-schema wrong-argument substitutions sit almost on top of
the true action in latent space.

`exact_match_rate` is not a production metric. It compares a decoded action to
one reference planner trace, but production planning does not have that trace and
many valid alternatives may exist. In this investigation it is retained only as a
teacher-forced invertibility/debug label.

The strongest root-cause assessment is:

1. The action encoder learns schema identity much more strongly than object
   role bindings.
2. The decoder samples only type-compatible arguments, while simulator
   applicability is extremely sparse inside that typed space.
3. The CEM decoder sees a nearly flat/aliased score landscape among same-schema
   object substitutions and collapses confidently to invalid tuples.
4. The planner can optimize continuous latents that are far from the encoded
   grounded-action manifold; decoding then becomes a projection onto a sparse,
   mostly invalid symbolic action space.

## Evidence

### Existing diagnostic output

Saved bounded CEM action-decoder diagnostics on the 44-transition smoke
validation split show schema recovery is easy and argument recovery is hard:

| checkpoint | exact/debug | applicable | same-schema |
| --- | ---: | ---: | ---: |
| `default_seed0` | 16/44 | 16/44 | 44/44 |
| `inverse_dynamics_seed0` | 14/44 | 14/44 | 44/44 |
| `action_encoder_rnn_seed0` | 10/44 exact, 12/44 applicable | 12/44 | 44/44 |

The failures are almost entirely same-schema wrong-argument decodes. For the
baseline, all 28 invalid decodes in that run were same-schema substitutions.

### Applicability sparsity

New diagnostic:
`script/diagnose_action_applicability_space.py`

Smoke validation first 12 transitions:

- Average type-valid grounded actions per state: `21,960`.
- Average simulator-applicable actions per state: `108.9`.
- Average applicable fraction: `0.00496`.
- The reference action was applicable in every replayed state.
- Move schemas are much sparser:
  - `move_car_in_road`: `2 / 86,400` accumulated candidates.
  - `move_car_out_road`: `2 / 86,400` accumulated candidates.

Full-dev spot check, first 20 transitions:

- Average type-valid grounded actions per state: `9,732`.
- Average simulator-applicable actions per state: `157.55`.
- Average applicable fraction: `0.01619`.
- Move schemas remain extremely sparse:
  - `move_car_in_road`: `5 / 57,600`.
  - `move_car_out_road`: `3 / 57,600`.

This confirms the failure mode is not an artifact of only two smoke validation
problems. Type-correct decoding is far too weak a constraint in CityCar.

### Teacher-forced latent margins

New diagnostic:
`script/diagnose_action_latent_geometry.py`

For teacher-forced targets, the true action rank is expected to be 1 because the
target latent is computed by the same action encoder used to score candidates.
Therefore exact nearest-neighbor rank is not the useful metric. The useful
signal is the margin to nearest wrong same-schema actions.

Baseline, smoke validation first 12 transitions, same-schema candidates only:

- True action rank-1 rate: `1.0`.
- Nearest wrong same-schema rate: `1.0`.
- Median nearest-wrong L2 distance: `5.19e-05`.
- Minimum nearest-wrong L2 distance: `3.33e-05`.
- Nearest wrong action was applicable only `2/12` times.

Inverse-dynamics checkpoint on the same slice:

- True action rank-1 rate: `1.0`.
- Nearest wrong same-schema rate: `1.0`.
- Median nearest-wrong L2 distance: `6.49e-05`.
- Minimum nearest-wrong L2 distance: `4.17e-05`.
- Nearest wrong action was applicable only `2/12` times.

The auxiliary inverse-dynamics run slightly increases this small margin on the
slice but does not change the downstream CEM/applicability outcome at the tested
budget.

Example baseline nearest-neighbor failures:

- True `build_straight_oneway(junction2-2, junction3-2, road2)`.
  Nearest wrong: `build_straight_oneway(junction2-1, junction3-2, road2)`,
  one argument changed, invalid, distance `3.42e-05`.
- True `build_diagonal_oneway(junction3-4, junction2-3, road1)`.
  Nearest wrong: `build_diagonal_oneway(junction3-4, junction2-1, road1)`,
  one argument changed, invalid, distance `5.68e-05`.

These are not large semantic margins. They are near ties.

### CEM decoder landscape

New diagnostic:
`script/diagnose_cem_action_landscape.py`

Baseline, smoke validation first 12 transitions, 64 samples and 8 iterations:

- Same-schema rate: `12/12`.
- Applicable rate: `4/12`.
- Exact/debug rate: `2/12`.

Inverse-dynamics checkpoint on the same slice:

- Same-schema rate: `12/12`.
- Applicable rate: `4/12`.
- Exact/debug rate: `3/12`.

CEM entropy traces show confident collapse despite invalid object tuples. Example
baseline move failures:

- True `move_car_in_road(junction2-2, junction3-2, car1, road2)`.
  Decoded `move_car_in_road(junction3-0, junction2-0, car1, road2)`,
  invalid. Final action entropy was about `0.009` bits; argument entropies were
  also near zero.
- True `move_car_out_road(junction2-2, junction3-2, car1, road2)`.
  Decoded `move_car_out_road(junction2-2, junction3-0, car1, road2)`,
  invalid. Final action entropy was about `0.009` bits.

The optimizer is not failing by remaining uncertain. It is finding a
high-scoring same-schema basin and collapsing to it, even when the tuple is
state-invalid.

### Planner first-latent behavior

New diagnostic:
`script/diagnose_planner_first_latent.py`

This uses the production-style `gmm_mppi` planner path and analyzes the first
planned latent before applying simulator actions.

Baseline `p166` with the fast planning probe config:

- CEM-decoded first action:
  `build_diagonal_oneway(junction1-0, junction2-3, road5)`.
- The decoded action is not applicable.
- The nearest encoded grounded-action scores to the planned latent are around
  `-25`, far from the near-zero scores seen for teacher-forced encoded actions.
- Top neighbors are mostly invalid; ranks 7 and 10 were applicable but nearly
  tied with invalid neighbors.

Baseline `p192` with the same probe config:

- CEM-decoded first action was applicable in this diagnostic run.
- The top 10 exact nearest encoded neighbors were all invalid.
- Scores were again around `-25`, indicating the planned latent is far from the
  encoded action manifold.

This supports a second root cause beyond teacher-forced decoding: continuous
planning can produce latents that are off the grounded-action manifold, and the
current decoder has no applicability-aware projection.

## Root-cause assessment

### Confirmed

- Action schema identity is much easier than object binding. All saved 44-step
  bounded decoder diagnostics recover the true schema histogram exactly.
- Type-compatible object sampling is insufficient. Applicable actions are
  usually less than a few percent of type-valid actions, and move actions are
  orders of magnitude sparser.
- Same-schema near-miss actions are almost tied in latent score. The nearest
  wrong same-schema candidates are one-argument substitutions with L2 distances
  on the order of `1e-05`.
- CEM collapses confidently to wrong object tuples; this is not just a failure
  to decide among schemas.
- The current inverse-dynamics objective does not solve the decoder failure at
  the tested smoke budget because it regresses to the existing action encoder
  latent, which can itself be aliased over object arguments.

### Likely

- The JEPA transition objective permits action-latent aliasing when multiple
  object substitutions have similar local transition-prediction effects or when
  the downstream loss does not punish invalid preconditions.
- The factorized categorical decoder has a hard search problem: it must recover
  a rare valid tuple from independent role distributions using a weak margin
  signal.
- The continuous planner needs an action-manifold or applicability constraint.
  Otherwise it can optimize useful-looking latent rollouts whose decoded
  nearest grounded actions are invalid.

### Not established

- It is not established that exact exhaustive decoding of teacher-forced action
  latents fails. In fact, exact rank is expected to be 1 by construction for
  teacher-forced targets.
- It is not established that more CEM samples alone will solve production
  planning. Larger budgets may improve teacher-forced recovery, but planned
  latents can be off-manifold.
- It is not established that inverse dynamics is a bad direction. The tested
  inverse-dynamics target is too weak to guarantee argument identifiability.

## Implications for next work

Do not resume broad hyperparameter tuning yet. The next change should target one
of the confirmed failure mechanisms:

- Add an applicability-aware objective or classifier using sampled type-valid
  negatives and true/applicable positives.
- Add contrastive action-latent losses over same-state, same-schema hard
  negatives.
- Decode with role-aware/object-aware scoring rather than a single whole-action
  latent distance.
- Constrain or regularize planner latents toward the encoded grounded-action
  manifold.
- Use `SimulatorEngine.applicable_actions()` only as an offline oracle for
  diagnostics and labels, not as the planner's production action generator.

## Artifacts produced

- `script/diagnose_action_applicability_space.py`
- `script/diagnose_action_latent_geometry.py`
- `script/diagnose_cem_action_landscape.py`
- `script/diagnose_planner_first_latent.py`
- `script/action_diag_common.py`

Representative outputs:

- `acs-jepa-runs/smoke/default_seed0/diagnostics/applicability_val_12`
- `acs-jepa-runs/smoke/default_seed0/diagnostics/geometry_same_schema_val_12`
- `acs-jepa-runs/smoke/default_seed0/diagnostics/cem_val_12`
- `acs-jepa-runs/smoke/inverse_dynamics_seed0/diagnostics/geometry_same_schema_val_12`
- `acs-jepa-runs/smoke/inverse_dynamics_seed0/diagnostics/cem_val_12`
- `acs-jepa-runs/smoke/default_seed0/diagnostics/applicability_full_dev_20`
- `acs-jepa-runs/smoke/default_seed0/diagnostics/planner_first_latent_p166_fast`
- `acs-jepa-runs/smoke/default_seed0/diagnostics/planner_first_latent_p192_fast`
