
---

## Partial goal as a latent constraint or energy function

The point-target condition

$$
z_T \approx z_g
$$

is replaced by the set-membership condition

$$
z_T \in \mathcal{G}(g_p)
$$

where $\mathcal{G}(g_p)$ is the set of latent states satisfying the partial goal.

The feasible goal set may be defined as

$$
\begin{aligned}
\mathcal{G}(g_p)
&=
\{z : C(g_p,z)=1\}
\end{aligned}
$$

or, in a soft formulation, by an energy function

$$
E_g(z,g_p) \ge 0
$$

where lower energy corresponds to greater satisfaction of the partial goal.

The resulting planning objective is

$$
\begin{aligned}
a_{0:T-1}^*
&=
\arg\min_{a_{0:T-1}}
\left[
E_g(z_T,g_p)
+
\lambda
\sum_{t=0}^{T-1} c_a(a_t)
+
\beta
\sum_{t=0}^{T-1} c_z(z_t)
\right]
\end{aligned}
$$

subject to:

$$
z_{t+1}=f_\theta(z_t,a_t)
$$

Here:

* $E_g(z_T,g_p)$: terminal goal-violation energy
* $c_a(a_t)$: action cost
* $c_z(z_t)$: optional latent regularizer / safety cost
* $\lambda,\beta$: weights

A partial goal, for example:

> “object A inside container B”

does not require the whole scene to match a fixed target image. It imposes the predicate constraint

$$
\text{inside}(A,B;z_T)=1
$$

A corresponding terminal cost is

$$
\begin{aligned}
E_g(z_T,g_p)
&=
-\log P_\psi(\text{inside}(A,B)=1 \mid z_T)
\end{aligned}
$$

For a goal containing several predicates,

$$
g_p = \{p_1, p_2, \dots, p_K\}
$$

the goal energy can be written as

$$
\begin{aligned}
E_g(z,g_p)
&=
\sum_{k=1}^{K}
w_k
\ell\left(h_{\psi,k}(z), y_k\right)
\end{aligned}
$$

where:

* $h_{\psi,k}(z)$: learned predicate head
* $y_k$: desired truth value of predicate $p_k$
* $w_k$: importance weight
* $\ell$: binary cross-entropy, hinge loss, margin loss, etc.

For binary predicates:

$$
\begin{aligned}
\ell(h,y)
&=
-\left[
y\log h + (1-y)\log(1-h)
\right]
\end{aligned}
$$

Only predicates specified by the partial goal contribute to the terminal cost. Unspecified aspects of the state remain unconstrained by this term.

### Building blocks

The architecture consists of the following maps:

$$
x_t \xrightarrow{e} z_t
$$

$$
(z_t,a_t) \xrightarrow{f_\theta} z_{t+1}
$$

$$
z_t \xrightarrow{h_\psi} \text{predicates / properties}
$$

$$
(z_T,g_p) \xrightarrow{E_g} \text{goal energy}
$$

MPPI or CEM samples action sequences:

$$
a_{0:T-1}^{(i)} \sim q(a_{0:T-1})
$$

Each sequence is rolled forward:

$$
z_{t+1}^{(i)}=f_\theta(z_t^{(i)},a_t^{(i)})
$$

The corresponding cost is

$$
\begin{aligned}
J^{(i)}
&=
E_g(z_T^{(i)},g_p)
+
\lambda
\sum_t c_a(a_t^{(i)})
\end{aligned}
$$

The sampling distribution is then updated toward lower-cost samples.

### Effect on partial goals

A full target latent $z_g$ imposes constraints on every latent dimension:

$$
\lVert z_T-z_g \rVert^2
$$

A partial-goal energy imposes constraints only on the specified properties:

$$
E_g(z_T,g_p)
$$

Consequently, multiple terminal latents may satisfy the same partial goal.

The target is changed from a point constraint

$$
z_T = z_g
$$

to a set constraint

$$
z_T \in \mathcal{G}(g_p)
$$

---

## Partial goal as a distribution over terminal latent states

A partial goal can also be represented as a conditional distribution over terminal latent states:

$$
p(z_g \mid g_p)
$$

Planning then favors terminal states with high probability under this distribution.

The point-target distance

$$
\lVert z_T-z_g \rVert^2
$$

is replaced by the negative log-likelihood

$$
\begin{aligned}
E_g(z_T,g_p)
&=
-\log p_\eta(z_T \mid g_p)
\end{aligned}
$$

The planning objective becomes

$$
\begin{aligned}
a_{0:T-1}^*
&=
\arg\min_{a_{0:T-1}}
\left[
-\log p_\eta(z_T \mid g_p)
+
\lambda
\sum_t c_a(a_t)
\right]
\end{aligned}
$$

subject to:

$$
z_{t+1}=f_\theta(z_t,a_t)
$$

This formulation is equivalent to an energy model, since any distribution induces the energy

$$
\begin{aligned}
E_g(z,g_p)
&=
-\log p_\eta(z\mid g_p)
\end{aligned}
$$

up to an additive constant.

### Gaussian case

A simple model is a Gaussian conditional distribution:

$$
\begin{aligned}
p_\eta(z_g \mid g_p)
&=
\mathcal{N}
\left(
z_g;
\mu_\eta(g_p),
\Sigma_\eta(g_p)
\right)
\end{aligned}
$$

The corresponding negative log-likelihood is

$$
\begin{aligned}
-\log p_\eta(z_T \mid g_p)
&=
\frac{1}{2}
(z_T-\mu_g)^\top
\Sigma_g^{-1}
(z_T-\mu_g)
+
\text{const}
\end{aligned}
$$

where:

$$
\begin{aligned}
\mu_g &= \mu_\eta(g_p), \\
\Sigma_g &= \Sigma_\eta(g_p)
\end{aligned}
$$

The covariance expresses uncertainty about unspecified dimensions.

For an unconstrained latent dimension, a high variance is assigned:

$$
\Sigma_{g,jj} \gg 1
$$

Mismatch along that dimension is weakly penalized.

For a constrained latent dimension, a low variance is assigned:

$$
\Sigma_{g,jj} \ll 1
$$

Mismatch along that dimension is strongly penalized.

For diagonal covariance, the cost becomes

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\frac{1}{2}
\sum_j
\frac{
(z_{T,j}-\mu_{g,j})^2
}{
\sigma_{g,j}^2
}
\end{aligned}
$$

This corresponds to a soft masking mechanism.

### Multimodal case

Partial goals are often multimodal.

For example:

> “Place the cup on the table.”

There may be many valid table positions, making a single Gaussian overly restrictive.

A mixture model can represent multiple valid modes:

$$
\begin{aligned}
p_\eta(z_g\mid g_p)
&=
\sum_{m=1}^{M}
\pi_m(g_p)
\mathcal{N}
\left(
z_g;
\mu_m(g_p),
\Sigma_m(g_p)
\right)
\end{aligned}
$$

The induced energy is

$$
\begin{aligned}
E_g(z_T,g_p)
&=
-\log
\sum_{m=1}^{M}
\pi_m(g_p)
\mathcal{N}
\left(
z_T;
\mu_m(g_p),
\Sigma_m(g_p)
\right)
\end{aligned}
$$

The planner may therefore converge to any valid mode.

### Conditional generative model

Rather than evaluating likelihood directly, a conditional generator can be trained:

$$
\epsilon \sim \mathcal{N}(0,I)
$$

$$
z_g = G_\eta(g_p,\epsilon)
$$

This defines samples from

$$
p_\eta(z_g\mid g_p)
$$

A practical planning objective is

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\min_{m=1,\dots,M}
\lVert z_T-\tilde z_g^{(m)} \rVert^2
\end{aligned}
$$

where:

$$
\tilde z_g^{(m)} \sim p_\eta(z_g\mid g_p)
$$

This objective selects the closest sampled completion of the partial goal.

### Building blocks

The system contains the following maps:

$$
x_t \xrightarrow{e} z_t
$$

$$
(z_t,a_t) \xrightarrow{f_\theta} z_{t+1}
$$

$$
g_p \xrightarrow{p_\eta} p(z_g\mid g_p)
$$

$$
z_T \xrightarrow{-\log p_\eta(z_T\mid g_p)} \text{terminal cost}
$$

The planner optimizes the action sequence according to

$$
\begin{aligned}
a_{0:T-1}^{*}
&=
\arg\min
\left[
-\log p_\eta(z_T\mid g_p)
+
\lambda
\sum_t c_a(a_t)
\right]
\end{aligned}
$$

### Key interpretation

The partial goal is not represented as the point constraint

$$
z_T=z_g
$$

but as a distributional condition

$$
z_T \sim p(z_g\mid g_p)
$$

or equivalently as membership in a high-density region:

$$
z_T \in \text{high-density region of } p(z_g\mid g_p)
$$

This representation is useful when hidden or unspecified components of the state are required to remain realistic.

---

## Factorized latent space: constrained and free components

A third approach designs or learns a latent representation in which partial goals constrain only selected components.

Let:

$$
z = (z^c, z^u)
$$

where:

* $z^c$: goal-constrained latent components
* $z^u$: unconstrained or nuisance components

The partial goal maps only to the constrained component:

$$
g_p \mapsto z_g^c
$$

The terminal cost is then

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\lVert z_T^c-z_g^c \rVert^2
\end{aligned}
$$

rather than the full latent distance

$$
\lVert z_T-z_g \rVert^2
$$

The planner minimizes

$$
\begin{aligned}
a_{0:T-1}^*
&=
\arg\min_{a_{0:T-1}}
\left[
\lVert z_T^c-z_g^c \rVert^2
+
\lambda
\sum_t c_a(a_t)
+
\beta R(z_T^u)
\right]
\end{aligned}
$$

subject to:

$$
z_{t+1}=f_\theta(z_t,a_t)
$$

The optional term $R(z_T^u)$ keeps the unconstrained component in-distribution.

For example,

$$
\begin{aligned}
R(z_T^u)
&=
-\log p(z_T^u \mid z_T^c)
\end{aligned}
$$

This matches the constrained goal variables while preserving plausibility of the unspecified variables.

### Masked latent version

Rather than explicitly splitting $z$, one may define a mask:

$$
m(g_p) \in \{0,1\}^d
$$

where:

* $m_j=1$: latent dimension $j$ is constrained
* $m_j=0$: latent dimension $j$ is unspecified

The masked objective is

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\lVert m(g_p)\odot(z_T-z_g) \rVert^2
\end{aligned}
$$

Equivalently,

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\sum_{j=1}^{d}
m_j(g_p)
(z_{T,j}-z_{g,j})^2
\end{aligned}
$$

Only specified dimensions are penalized by this objective.

A soft version uses continuous weights:

$$
m_j(g_p)\in[0,1]
$$

The corresponding objective is

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\sum_j
m_j(g_p)
(z_{T,j}-z_{g,j})^2
\end{aligned}
$$

This is equivalent to the diagonal Gaussian case from section 2 under the correspondence

$$
\begin{aligned}
m_j(g_p)
&\propto
\frac{1}{\sigma_{g,j}^2}
\end{aligned}
$$

Thus, factorized latents and distributional goals are closely related.

### Object-centric factorization

For structured environments, object latents may be used:

$$
\begin{aligned}
z
&=
(z^{(1)}, z^{(2)}, \dots, z^{(N)})
\end{aligned}
$$

where each $z^{(i)}$ represents one object or entity.

A partial goal may constrain only object $k$:

$$
g_p: \quad \text{object } k \text{ at location } y
$$

The corresponding cost is

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\left\lVert
r(z_T^{(k)}) - y
\right\rVert^2
\end{aligned}
$$

where $r$ extracts the relevant property, e.g. position.

For a relational goal,

$$
g_p: \quad \text{object } i \text{ left of object } j
$$

the cost may be written as

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\ell
\left(
h_\psi(z_T^{(i)},z_T^{(j)}),
1
\right)
\end{aligned}
$$

where $h_\psi$ predicts whether the relation is true.

### Building blocks

The factorized setup uses either

$$
x_t \xrightarrow{e} z_t=(z_t^c,z_t^u)
$$

or:

$$
x_t \xrightarrow{e} z_t=(z_t^{(1)},\dots,z_t^{(N)})
$$

The dynamics may be represented as

$$
z_{t+1}=f_\theta(z_t,a_t)
$$

or in object-factorized form:

$$
\begin{aligned}
z_{t+1}^{(i)}
&=
f_\theta^{(i)}
\left(
z_t^{(1:N)},a_t
\right)
\end{aligned}
$$

The goal encoder produces either

$$
g_p \xrightarrow{q_\eta} z_g^c
$$

or:

$$
g_p \xrightarrow{q_\eta} (m,z_g)
$$

The planner then minimizes

$$
\begin{aligned}
E_g(z_T,g_p)
&=
\lVert m\odot(z_T-z_g) \rVert^2
\end{aligned}
$$

plus action and regularization costs.

### Effect of factorization

A standard latent target imposes

$$
\text{all latent factors must match}
$$

A factorized partial goal imposes

$$
\text{only these factors must match}
$$

This avoids enforcing arbitrary completions of hidden or unspecified state.

---

## Unifying view

All three approaches have the common form

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
+
\beta
\sum_t c_{\text{reg}}(z_t)
\right]
\end{aligned}
$$

with:

$$
z_{t+1}=f_\theta(z_t,a_t)
$$

They differ in the definition of $E_g$.

| Approach | Goal model | Terminal cost |
| --- | --- | --- |
| Constraint / energy | $z_T \in \mathcal{G}(g_p)$ | $E_g(z_T,g_p)$ |
| Distribution | $z_T \sim p(z\mid g_p)$ | $-\log p(z_T\mid g_p)$ |
| Factorized latent | only part of $z_T$ is constrained | $\lVert m\odot(z_T-z_g) \rVert^2$ |

The most general formulation is the energy model

$$
E_g(z,g_p)
$$

because both the distributional and factorized versions can be expressed as energies:

$$
E_g(z,g_p)=-\log p(z\mid g_p)
$$

and:

$$
E_g(z,g_p)=\lVert m(g_p)\odot(z-z_g) \rVert^2
$$

Thus, an action-conditioned JEPA planner can be formulated as latent dynamics learned with JEPA combined with planning against a **partial-goal energy**, rather than a full latent target.

## Planner implementation note

`GraphJEPA` is the training wrapper. Receding-horizon inference rollout is
implemented by `LatentMPPIPlanner`, which depends directly on the graph encoder,
state encoder, predictor, and any goal-energy callable with signature:

```text
goal_energy(goal_tensors, terminal_state) -> FloatTensor[N]
```

The planner samples continuous action-latent sequences `[N, H, D_a]`, rolls
them forward through the predictor, and minimizes terminal goal energy. V1 uses
a constant action cost. `PlannerAgent` decodes only the selected prefix,
executes the first valid decoded actions in the simulator, and replans after
partial-prefix execution or decode retry exhaustion.
