# ACS-JEPA tuning campaign configs

These configs implement the staged tuning plan. Each file is an overlay on
`configs/default.yaml`. `--config` accepts one or more overlays and merges them
left-to-right, so later files override earlier files.

Run one early-stage variant with:

```bash
uv run --package acs-jepa-cli acs-jepa train DATASET_DIR \
  --output runs/tuning/<variant> \
  --config acs-jepa-cli/configs/tuning/<stage>/<variant>.yaml \
  --device cuda \
  --seed 0
```

Compose promoted winners automatically by stacking overlays:

```bash
uv run --package acs-jepa-cli acs-jepa train DATASET_DIR \
  --output runs/tuning/stage2_size96_layers2 \
  --config \
    acs-jepa-cli/configs/tuning/01_composition/backbone_rnn_gru.yaml \
    acs-jepa-cli/configs/tuning/01_composition/gmm_detach_false.yaml \
    acs-jepa-cli/configs/tuning/02_capacity/size96_layers2.yaml \
  --device cuda \
  --seed 0
```

MLflow keeps `tracking.run_name`, `tracking.tags.stage`, and
`tracking.tags.variant` from the last overlay that sets them. It also adds
derived tags for every tuning overlay in the stack, for example:

```text
stage=capacity
variant=size96_layers2
tuning.composition=backbone_rnn_gru+gmm_detach_false
tuning.capacity=size96_layers2
tuning.stack=composition:backbone_rnn_gru|composition:gmm_detach_false|capacity:size96_layers2
```

Recommended order:

1. `00_calibration`: choose the largest stable batch size, expected `24`.
2. `01_composition`: compare goal heads, then backbone and detach variants.
3. `02_capacity`: compare width/layer profiles for the promoted composition.
4. `03_rollout`: compare rollout lengths `4`, `6`, `8`, then context variants.
5. `04_loss_opt`: tune auxiliary losses, optimizer profile, and goal loss weight.
6. `05_planning`: evaluate finalists only; all variants use non-exact decoding and
   `planning.max_iters >= 60`.

Use MLflow experiment `acs-jepa-tuning-v1` and rank model runs primarily by
`eval/total_loss`, with `eval/jepa_loss`, `eval/goal_loss`,
`eval/term/inverse_dynamics`, and `runtime/seconds` as secondary signals.

Planning variants are overlays for trained checkpoints:

```bash
uv run --package acs-jepa-cli acs-jepa plan \
  --checkpoint runs/tuning/<finalist>/checkpoints/latest.pt \
  --domain DATASET_DIR/problem/domain.pddl \
  --problem DATASET_DIR/problem/p01.pddl \
  --output runs/tuning/planning/<variant> \
  --config \
    acs-jepa-cli/configs/tuning/04_loss_opt/<final-training-overlay>.yaml \
    acs-jepa-cli/configs/tuning/05_planning/<variant>.yaml \
  --device cuda \
  --seed 0
```
