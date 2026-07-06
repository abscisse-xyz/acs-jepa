
---

## Action Encoder/Decoder

This note describes a fixed-width representation for grounded symbolic actions
with heterogeneous argument signatures. The representation is intended to be
used with contextual object embeddings from the state graph.


### Symbols and Domains

Let $\mathcal{A}$ be the set of action schemas. Each action schema
$a_i \in \mathcal{A}$ has an ordered signature

$$
\sigma_i = (\tau_{i,0}, \tau_{i,1}, \dots, \tau_{i,k_i-1}),
$$

where $\tau_{i,j}$ is the type required at argument position $j$ of action
$a_i$. For each type $\tau$, let $\mathcal{O}_{\tau}$ denote the ordered set of
objects of that type in the current problem instance.

A grounded action is

$$
\bar a = a_i(o_0,\dots,o_{k_i-1}),
$$

with

$$
o_j \in \mathcal{O}_{\tau_{i,j}}.
$$

The ordering of object sets must be deterministic, for example lexicographic by
object name. Object ids are problem-local; they should not be interpreted as
global object identities across problem instances.

### Universal Slot Set

Variable action arity can be converted into a fixed-width representation by
constructing a universal set of typed argument slots.

For action $a_i$, define the type-local occurrence index

$$
\begin{aligned}
\ell_{i,j}
&=
\left|
\{h < j : \tau_{i,h}=\tau_{i,j}\}
\right|
\end{aligned}
$$

Then define its slot signature as

$$
\begin{aligned}
\Omega_i
&=
\left(
(\tau_{i,0},\ell_{i,0}),
(\tau_{i,1},\ell_{i,1}),
\dots,
(\tau_{i,k_i-1},\ell_{i,k_i-1})
\right)
\end{aligned}
$$

The second component is the occurrence index of that type within the action
signature. It is required because repeated types must remain distinguishable.
For example, the two junction arguments in
$\text{same\_line}(\text{junction}, \text{junction})$ occupy slots
$(\text{junction},0)$ and $(\text{junction},1)$.

The global slot set is the ordered union

$$
\begin{aligned}
\Omega
&=
\operatorname{sort}
\left(
\bigcup_{a_i \in \mathcal{A}}
\{(\tau_{i,j}, \ell_{i,j}) : 0 \le j < k_i\}
\right)
\end{aligned}
$$

Let $K = |\Omega|$. Each action schema $a_i$ induces a binary mask

$$
m_i \in \{0,1\}^{K},
$$

where $m_{i,r}=1$ iff slot $r$ belongs to $\Omega_i$.

This mask records which positions of the global slot set are active for a
particular action schema.

### Encoding a Grounded Action

A grounded action should encode both:

1. the action schema identity;
2. the objects bound to the active argument slots.

The schema identity is necessary because two different actions can have the same
typed signature. A mask alone cannot distinguish such actions.

Let $\operatorname{id}(a_i)$ be the action-schema id. Let
$\operatorname{slot}(\tau,\ell)$ be the index of $(\tau,\ell)$ in $\Omega$.
The fixed slot representation of

$$
\bar a = a_i(o_0,\dots,o_{k_i-1})
$$

is given by:

$$
\operatorname{action\_id} = \operatorname{id}(a_i),
$$

$$
m_r =
\begin{cases}
1, & r = \operatorname{slot}(\tau_{i,j},\ell_{i,j})
     \text{ for some } j, \\
0, & \text{otherwise},
\end{cases}
$$

and

$$
u_r =
\begin{cases}
\operatorname{id}(o_j),
& r = \operatorname{slot}(\tau_{i,j},\ell_{i,j}), \\
-1,
& m_r = 0.
\end{cases}
$$

Here $u_r$ is an object id or a padding value. The absolute argument position
$j$ should still be retained as a role id, because the type-local slot index
does not by itself encode the original argument position. In an embedding model,
the object id should be used to gather the corresponding contextual object
embedding from the state graph.

The resulting action encoder can be written abstractly as

$$
\begin{aligned}
e_a(\bar a, z_t)
&=
\rho
\left(
e_{\mathcal{A}}(\operatorname{action\_id}),
\left\{
\left(e_{\Omega}(r), e_{\mathrm{role}}(j_r), z_t^{(u_r)}\right)
: m_r=1
\right\}
\right)
\end{aligned}
$$

where:

* $e_{\mathcal{A}}$: action-schema embedding;
* $e_{\Omega}$: global slot embedding;
* $e_{\mathrm{role}}$: embedding of the original argument position;
* $j_r$: argument position associated with active slot $r$ for this action;
* $z_t^{(u_r)}$: contextual object embedding for object $u_r$ in state $z_t$;
* $\rho$: permutation-aware or order-aware aggregation module.

Because $\Omega$ is ordered, $\rho$ may also be implemented as a fixed-order
sequence model or MLP over padded slots. The mask $m$ must be used to ignore
inactive slots.

### Decoder

A decoder maps a model output back to a valid grounded action. It must recover:

1. an action schema $a_i$;
2. one object for each argument position in $\sigma_i$.

Given predicted action logits and slot/object logits:

$$
\hat a = \arg\max_{a_i \in \mathcal{A}} p(a_i),
$$

then for each argument position $j$ of $\hat a$:

$$
\begin{aligned}
\hat o_j
&=
\arg\max_{o \in \mathcal{O}_{\tau_{\hat a,j}}}
p(o \mid \hat a, j, z_t)
\end{aligned}
$$

The type restriction $o \in \mathcal{O}_{\tau_{\hat a,j}}$ is part of decoding.
It prevents illegal grounded actions such as placing a garage object in a
junction argument position.

The decoded action is

$$
\hat{\bar a} = \hat a(\hat o_0,\dots,\hat o_{k_{\hat a}-1}).
$$

If additional action preconditions are available, they should be applied after
type-correct decoding to mask out actions that are type-correct but invalid in
the current state.

---

## Implementation in `acs_jepa`

The current implementation follows the structure above, with a schema-ordered
padding convention rather than an explicit global slot tensor. The mathematical
slot set $\Omega$ is useful for specifying the representation, but the code
stores only the active action-local argument positions up to
`max_action_arity`.

The concrete action tensor dictionary is produced by `tensorize_action`:

```text
action_id              LongTensor[] or LongTensor[B]
action_object_indices  LongTensor[max_action_arity] or LongTensor[B, max_action_arity]
action_role_ids        LongTensor[max_action_arity] or LongTensor[B, max_action_arity]
action_arg_mask        BoolTensor[max_action_arity] or BoolTensor[B, max_action_arity]
```

Here `action_object_indices` contains problem-local object ids, padded with
`-1`; `action_role_ids` contains the original action-local argument position,
padded with `-1`; and `action_arg_mask` marks the real arguments. This is
equivalent to using the active entries of the fixed-width representation, with
the role id carrying the original argument position.

Action encoders operate on the current `JEPALatentState`. `GraphJEPA` first runs
the graph encoder and state encoder, then `LatentActionEncoder` gathers argument
vectors from `latent_state.object_latents`. Graph-sourced action encoding was
removed because latent planning has predicted intermediate states, not observed
graph encoder object embeddings.

Both encoders use the same composition pattern:

1. batch-normalize the action tensor shapes;
2. embed the action schema id with `action_embedding`;
3. gather the contextual object vector for each active argument;
4. project object vectors into the action latent dimension;
5. add role embeddings and compose the arguments with either
   `PooledArgumentEncoder` or `RNNArgumentEncoder`;
6. return an action latent in $\mathbb{R}^{D_a}$.

Trajectory training uses the same `ActionEncoder.forward()` as single-step
encoding. With temporal action tensors batched as `[B, K, ...]`, the base
encoder produces contextual action embeddings for all timesteps in one call and
a causal GRU maps them to time-aware action latents:

$$
u_t = q_\omega(\bar a_{0:t}).
$$

`model.action_context_steps` can limit the causal history; `null` uses all
previous actions in the current rollout window.

The action encoder is always latent-space:

```python
action_encoder = build_action_encoder(
    kind="pooled",
    num_actions=num_actions,
    max_action_arity=max_action_arity,
    latent_dim=latent_dim,
    action_dim=action_dim,
)
```

`ActionEncoder` is rank-aware: single-step inputs return `[B, D_a]`, temporal
inputs return `[B, K, D_a]`. Per-action embeddings are produced by
`LatentActionEncoder`.

The implemented decoder is `ActionDecoder`. It builds an `ActionDecodingSpace`
from the parsed problem, where a candidate sample has the form

$$
(\operatorname{action\_id}, u_0, \dots, u_{K_{\max}-1}).
$$

Decoding is type-constrained. For an action id and role id, only object ids
compatible with the required argument type are legal. The decoder supports two
search modes:

* exact enumeration over all type-valid grounded actions;
* approximate CEM/MPPI search through `ActionSamplingFamily`.

In both modes, each candidate grounded action is tensorized, encoded with the
same action encoder against the current `JEPALatentState`, and scored against
the target action latent using either negative squared distance or cosine
similarity. Exact decoding repeats the latent state for the enumerated candidate
batch. CEM/MPPI repeats the same latent state for sampled candidate batches.

This decoder recovers type-valid grounded actions, but it does not enforce
state-dependent preconditions. Preconditions must be added as an additional
candidate mask or feasibility model if the decoded action must be executable in
the current state.

In the v1 planner, continuous MPPI optimizes action latents first. Only the
selected execution prefix is decoded. `PlannerAgent` checks each decoded action
against the simulator's applicable-action set, applies the first valid prefix,
and replans. If the first decoded action is invalid, MPPI is retried with that
decoded action penalized in the prior update.

### Example

Consider the action schemas

```text
move(car, junction)
same_line(junction, junction)
busy(garage)
```

with object types

```text
car, garage, junction
```

The per-action slot signatures are

$$
\begin{aligned}
\Omega_{\text{move}}
&=
\left((\text{car},0),(\text{junction},0)\right)
\end{aligned}
$$

$$
\begin{aligned}
\Omega_{\text{same\_line}}
&=
\left((\text{junction},0),(\text{junction},1)\right)
\end{aligned}
$$

$$
\begin{aligned}
\Omega_{\text{busy}}
&=
\left((\text{garage},0)\right)
\end{aligned}
$$

Using lexicographic ordering over slots, the global slot set is

$$
\begin{aligned}
\Omega
&=
\left(
(\text{car},0),
(\text{garage},0),
(\text{junction},0),
(\text{junction},1)
\right)
\end{aligned}
$$

The corresponding masks are

$$
m_{\text{move}} = (1,0,1,0),
$$

$$
m_{\text{same\_line}} = (0,0,1,1),
$$

$$
m_{\text{busy}} = (0,1,0,0).
$$

For example, if

$$
\bar a = \text{move}(\text{car0}, \text{junction3}),
$$

then

$$
\operatorname{action\_id}=\operatorname{id}(\text{move}),
$$

$$
m = (1,0,1,0),
$$

and

$$
\begin{aligned}
u
&=
\left(
\operatorname{id}(\text{car0}),
-1,
\operatorname{id}(\text{junction3}),
-1
\right)
\end{aligned}
$$

---

## Consistency Checks and Limitations

The construction above fixes several ambiguities in the initial sketch:

* $\mathcal{O}_{\tau}$ denotes objects of a type, not an action-specific object
  domain $\mathcal{O}_i$.
* $\sigma_i$ denotes an action signature; $\Omega_i$ denotes the corresponding
  typed slots. These should not be conflated.
* Slot entries must include both type and occurrence index. Type alone is
  insufficient when the same type appears multiple times in one signature.
* The original argument position should still be retained as a role id. The
  occurrence index aligns same-type arguments across schemas; it is not a full
  substitute for the action-local role.
* The action id must be encoded separately. Two action schemas with the same
  signature otherwise receive the same mask.
* A fixed global slot set is a padding convention, not a semantic domain. It
  makes batching convenient but does not by itself define action semantics.

The main limitations are:

* The number of global slots grows with the variety of typed occurrences in the
  action schemas. This can create sparse encodings for domains with many
  schemas.
* Object ids are problem-local. Generalization across problem instances should
  rely on object type embeddings and contextual object embeddings, not on raw
  ids.
* Type-correct decoding does not guarantee precondition satisfaction. State
  dependent validity constraints require an additional precondition mask or
  learned feasibility model.
* The representation assumes fixed action schemas. Domains with dynamically
  generated actions or unbounded arity require a different representation.
* If the type system has inheritance or multiple compatible types, the decoder
  must use the set of objects compatible with the required type, not only
  objects whose declared type is exactly equal to it.
