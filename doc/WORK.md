
---

## Single transtion to trajectory

current dataset is centered around a transition (current_state, action, next_state) and single step rollout while training and inference.

we want to change this approach to a k step rollout while traing and inference. this requires changing the dataset from being centered around transitions to trajectories. a training instance becomes a sequence of state (graphs) and a sequence of actions (a time axis can be added to identify nodes and edges at a give time point in the sequence).
the graph encoder produces embeddings for nodes, predicts and the graph, the graph encoder is independant of time (sequence).
the node and graph latent encoder is time dependent and the dynamic changes to:
    $$z_0 = e_\eta(x_0)$$ 
    $$\qquad z_t = f_\psi(x_{t-v:t})$$ 
    $$\qquad u_t = q_\omega(a_{t-u:t})$$
    $$\hat z_{t+1} = g_\theta(z_{t-w:t}, u_{t-w:t}).$$

with $e$ being the graph encoder(embeddings), $f$ state latent space encoder, $q$ action latent space encoder, and $g$ latent space predictor.

At training time, we compute k-step rollout losses for latent space predication $L_k$ for $k = 1,...,K$
where $L_1$ is the single-step loss. The order of a prediction is the
number of calls to the predictor function required to obtain it from a groundtruth representation. For
a predicted representation $z^{(k)}_t$, we denote the timestep it corresponds to as $t$ and its prediction order
as $k$, with $z(0) = z= f_\theta(x)$. For $k≥1, L_k$ is defined as:
$$L_k =\sum_{t=1}^{T-k} \lVert g_\theta(z^{(k−1)}_{t−w:t} ,u_{t−w:t})−z_{t+1}\rVert $$

where $z^{(k)}_t$ is obtained by recursively unrolling the predictor for all $t≤T$, as
$$z^{(k)}_{t+1} = g_\theta(z^{(k−1)}_{t−w:t} ,u_{t−w:t}), \qquad z^{(0)}_t = f_\psi(x_{t−v:t}).$$

Other loss stays the same.

---

## Predictor Context Note

The current implementation uses a Markov latent transition:

$$\hat z_{t+1} = g_\theta(z_t, u_t).$$

This is the cleaner baseline for graph JEPA because a full PDDL state graph plus
a grounded action is expected to determine the next state. In that setting,
`g` learns a reusable latent transition function and recursive rollout is simply
repeated application of the same transition model.

This does not mean the model is fully memoryless. Temporal context can already
enter through the encoders:

$$z_t = f_\psi(x_{t-v:t}), \qquad u_t = q_\omega(a_{t-u:t}).$$

So the preferred first mechanism for history is to make `StateEncoderF` and the
action encoder produce contextual latents, while keeping the predictor contract
simple:

$$\hat z_{t+1} = g_\theta(z_t, u_t).$$

An RNN/windowed predictor,

$$\hat z_{t+1} = g_\theta(z_{t-w:t}, u_{t-w:t}),$$

can be useful if the single latent state is not sufficiently Markov, for example
when the latent bottleneck drops object-level details or long-horizon losses show
systematic drift. The tradeoff is training stability: during training the window
can contain clean observed latents, while during inference it increasingly
contains predicted latents. This teacher-forcing mismatch can make rollout error
compound more sharply.

The Markov predictor is also easier to batch and cheaper to run. A windowed
predictor needs sliding-window construction or hidden-state bookkeeping across
recursive orders, which increases memory movement and object-alignment surface
area. Therefore, the recommended path is to keep `g(z_t, u_t)` as the v1
predictor and only introduce a context-window predictor if experiments show that
better temporal state/action encoders, latent capacity, or predictor capacity do
not fix long-horizon drift.
    
