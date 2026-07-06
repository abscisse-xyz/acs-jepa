Now that action encoding is latent-only, the problem is cleaner:

```text
grounded action a_t
current latent state z_t
u_t = q(a_t, z_t)
z_{t+1} = g(z_t, u_t)
```

So the “real” action-latent set is state-dependent:

$$
\mathcal U(z_t) = \{q(a, z_t) \mid a \in \mathcal A_{\text{typed}}\}
$$

The current diagonal Gaussian MPPI samples arbitrary vectors in $\mathbb R^{D_a}$, then later decodes them back to nearest grounded actions. That is cheap, but it can sample off-manifold. Alternatives mostly differ in how tightly they stay on $\mathcal U(z_t)$.

**1. Categorical MPPI Over Grounded Actions**
Sample grounded action sequences directly:

```text
a_{0:H-1} ~ categorical prior
u_t = q(a_t, z_t)
z_{t+1} = g(z_t, u_t)
score = -goal_energy(z_H)
```

This removes the continuous decode mismatch entirely. The planner outputs grounded actions, not latent vectors. The categorical prior can be over action schema plus typed object arguments, similar to `ActionSamplingFamily`.

Best when grounded action space is moderate. Harder when many objects make action grounding huge.

**2. Exact Enumerate First Step, Continuous/Categorical Tail**
For each valid/type-valid first action:

```text
u_0 = q(a_0, z_0)
rollout tail with MPPI
score full sequence
```

Then execute the best first action. This guarantees the executed action was truly optimized, while keeping the tail cheap. It fits receding-horizon MPC nicely because only the first action is applied by default.

This is a very attractive v2 compromise.

**3. Action-Manifold MPPI**
Instead of sampling arbitrary `u`, sample grounded actions, encode them, then add small local noise:

$$
u_t = q(a_t, z_t) + \epsilon
$$

The base point is real, but noise allows local continuous exploration. Decoding can still map back to the anchor action or nearest action. This can help if the predictor benefits from smooth local control variation, but it reintroduces some off-manifold behavior.

**4. Mixture of Gaussians Centered on Encoded Actions**
Build a proposal from actual encoded grounded actions:

$$
p(u \mid z_t) = \sum_a \pi_a \mathcal N(q(a,z_t), \sigma_a^2 I)
$$

For a sequence, sample an action anchor per timestep, then sample around its encoded latent. Elite updates adjust mixture weights and maybe variances. This handles multimodality much better than one diagonal Gaussian.

More expensive, but conceptually aligned with the latent-only refactor.

**5. Learned Proposal Network**
Train a proposal:

$$
p_\phi(a_t \mid z_t, goal)
$$

or

$$
p_\phi(u_t \mid z_t, goal)
$$

Then MPPI samples from that proposal instead of a fixed Gaussian. This is powerful but adds training burden and policy bias. I would not do this before proving simpler action-aware proposals are insufficient.

**6. Cross-Entropy Over Encoded Action Sequences**
Use CE/CEM directly on discrete action samples. Elites update categorical distributions over action ids and object roles. Each sampled sequence is encoded through `q(a_t, z_t)` during rollout.

This is probably the most principled replacement for diagonal Gaussian MPPI in the current architecture. It uses the same latent predictor and goal energy, but the optimizer’s search space is grounded actions.

**7. Hybrid Latent Prior From Action Embedding Statistics**
At each state, enumerate or sample grounded actions, compute their encoded latents, then fit a Gaussian:

$$
\mu_t, \Sigma_t \approx \text{stats}\{q(a,z_t)\}
$$

Use diagonal or low-rank covariance from actual action encodings. This keeps continuous MPPI but initializes it from the real action manifold. Better than zero-mean isotropic prior, still not as clean as discrete action sampling.

My ranking for this codebase:

1. **Best next step:** categorical CE/MPPI over grounded action sequences encoded by `LatentActionEncoder`.
2. **Best pragmatic MPC compromise:** exact or sampled grounded first action, continuous tail.
3. **Best continuous upgrade:** mixture centered on real `q(a,z_t)` encodings.
4. **Keep current diagonal Gaussian only as v1 baseline**, because it is simple but not action-manifold aware.

The latent-only refactor makes these alternatives much more natural because future-step action encoding can now use predicted `z_t`. That was the missing piece.