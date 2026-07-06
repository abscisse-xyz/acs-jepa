# Planning

Unified planning solution:

$$
\begin{aligned}
a_{0:T-1}^*
&=
\arg\min_{a_{0:T-1}}
\left[
E_g(z_T,g_p)
+
\lambda
\sum_t c_a(a_t)
\right]
\end{aligned}
$$

We find $a_{0:T-1}^*$ with continuous MPPI in latent action space and receding
horizon execution.

## V1 implementation

Planning lives outside `GraphJEPA`. `GraphJEPA` remains the trajectory-training
wrapper; inference rollout belongs to `LatentMPPIPlanner`, which depends
directly on:

```text
graph_encoder, state_encoder, predictor, goal_energy
```

At each simulator state:

1. `PlannerAgent` reads `engine.current_facts()`.
2. It builds the current PyG state graph.
3. `LatentMPPIPlanner` encodes the graph once.
4. Continuous MPPI samples action-latent sequences `[N, H, D_a]`.
5. The planner rolls samples forward with the predictor.
6. It scores the terminal latent with any `acs_jepa.goals`-compatible energy:

```text
goal_energy(goal_tensors, terminal_state) -> FloatTensor[N]
```

The action cost is constant in v1:

```text
c_a(a_t) = constant_action_cost
```

For fixed horizon this term is diagnostic only; it does not change sample
ranking.

## Execution policy

The selected latent sequence is decoded one step at a time. `PlannerAgent`
executes decoded actions with:

```text
engine.apply_action(action_name, arguments, finish=True)
```

The simulator is the validity oracle: if a decoded action is not executable,
`apply_action` raises `ValueError`.

The agent applies only the first valid prefix, up to `apply_steps`, then
replans from the updated simulator state.

If the first decoded action is invalid, the simulator is not mutated. The agent
adds that decoded action to a rejection set, reruns MPPI with a rejection
penalty, and retries up to `max_decode_attempts`. If a later action in the
prefix is invalid after at least one action has been applied, the agent stops
the prefix and replans from the new state.

The loop terminates when `engine.goals_satisfied()` is true or
`max_total_actions` is reached.
