# EB-JEPA Session Notes

## Building Blocks for Action-Conditioned Planning

Based on the paper (ICLR 2026 Workshop on World Models) and the code in `ebj_jepa/examples/ac_video_jepa/`.

### Components

**Encoder `fθ`** — `ImpalaEncoder` (`eb_jepa/architectures.py`)
- Input `[B, C, T, H, W]` → Output `[B, D, T, 1, 1]`
- 3 ResNet stacks + global spatial pooling + MLP. Produces a global per-frame representation.

**Predictor `gϕ`** — `RNNPredictor` (`eb_jepa/architectures.py`)
- Input: state `[B, D, 1, 1, 1]` + action `[B, A, 1]` → Output `[B, D, 1, 1, 1]`
- GRU with `hidden_size=embed_dim`, `input_size=action_dim`. `is_rnn=True` triggers autoregressive unrolling.

**JEPA Unroller** — `JEPA.unroll()` (`eb_jepa/jepa.py`)
- Central engine for training and planning.
- `unroll_mode="autoregressive"`: step-by-step, feeds predicted state back — used for both training (with loss) and planning (without loss).

**Regularizer** — `VC_IDM_Sim_Regularizer` (`eb_jepa/losses.py`)
- Combines 4 terms (paper Eq. 13): variance `Lvar` (α=16), covariance `Lcov` (β=8), temporal similarity `Lsim` (δ=12), inverse dynamics `LIDM` (ω=1).
- IDM is critical: without it the model collapses to 1% planning success (Table 4) due to spurious correlations.

**Planning Objective** — `ReprTargetDistMPCObjective` (`eb_jepa/planning.py`)
- Implements paper Eq. (14): `E_plan = sum_t ||f_theta(x_g) - z_hat_t||^2`
- `sum_all_diffs=True` (cumulative cost over all timesteps) outperforms final-state-only by 8%.

**Planners** — `MPPIPlanner` / `CEMPlanner` (`eb_jepa/planning.py`)
- Both optimize continuous action sequences. MPPI uses importance-weighted elite update, CEM uses sample mean.
- Config: H=90, N=200 samples, J=20 iterations, τ=0.005 → 97% success on Two Rooms.

**Planning Agent** — `GCAgent` (`eb_jepa/planning.py`)
- Wires encoder, predictor, planner. `act(x_t)` → calls `MPPIPlanner.plan()` → returns first action of optimized sequence, then replans.

### Data flow

```
Training:
  (x_t, a_t) -> ImpalaEncoder -> z_t
  (z_t, a_t) -> RNNPredictor  -> z_hat_{t+1}
  Loss = MSE(z_hat_{t+1}, z_{t+1}) + VC_IDM_Sim_Regularizer(z_{1:T})

Planning:
  x_0, x_g  -> encode -> z_0, z_g
  Optimize {a_0:H} via MPPI to minimize sum_t ||z_g - z_hat_t||^2
  Execute a_0, observe x_1, replan
```
---
