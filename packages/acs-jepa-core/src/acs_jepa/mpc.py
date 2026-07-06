"""Cross-entropy optimization on finite action spaces.

This module implements the CE method for the finite-domain maximization problem

    x* in argmax_{x in D} S(x),    D = D_1 x ... x D_k,

where each coordinate domain is finite, ``|D_i| = n_i``. In the setting
discussed earlier,

    S(x) = <f(x), u>,    with f : D -> R^m and u in R^m.

The public API and internal sampling families both use 0-based integer
categories, ``x_i in {0, ..., n_i - 1}``, matching PyTorch sampling operators.

Exact CE update equations
=========================

Let ``p_theta`` be a parametric sampling family on ``D``. At iteration ``t``,
sample ``N`` candidates ``X^(1), ..., X^(N) ~ p_{theta^(t-1)}``, score them
with

    s_r = S(X^(r)),

and define the elite set ``E_t`` as either the top ``N_elite`` samples or,
equivalently, those samples with score above the elite threshold ``gamma_t``.

The generic CE update is the elite-sample maximum-likelihood problem

    theta_hat^(t)
        = argmax_theta sum_{r in E_t} w_r log p_theta(X^(r)),

where ``w_r >= 0`` are elite weights summing to 1. Uniform weights recover the
standard CE update; score-proportional and exponential weights recover
weighted-CE / MPPI-style variants.

Default factorized categorical family
-------------------------------------

By default, the module uses the product family

    p_theta(x) = prod_{i=1}^k theta_{i, x_i},

with per-coordinate simplex constraints

    theta_{i, j} >= 0,    sum_{j=1}^{n_i} theta_{i, j} = 1.

For this family the weighted MLE decomposes coordinate-wise:

    theta_hat^(t)_{i, j}
        = sum_{r in E_t} w_r 1{X_i^(r) = j}.

Instead of ad-hoc clipping, the implementation supports Dirichlet-prior
smoothing with pseudocount ``kappa >= 0``:

    theta_tilde^(t)_{i, j}
        = (sum_{r in E_t} w_r 1{X_i^(r) = j} + kappa)
          / (sum_{r in E_t} w_r + n_i kappa).

Then a standard CE smoothing step is applied:

    theta^(t)_{i, :}
        = alpha theta_tilde^(t)_{i, :}
          + (1 - alpha) theta^(t-1)_{i, :},

with ``alpha in (0, 1]``.

Dependent sampling families
---------------------------

The independence assumption can be too weak for structured problems. To support
dependent samplers, the optimizer is written against a modular ``SamplingFamily``
protocol. Any family that implements

    sample(num_samples, generator)
    update(elite_samples, elite_weights)
    marginals()
    mode()
    is_degenerate(tol)

can be plugged in. The generic CE update still solves

    theta_hat^(t) = argmax_theta sum_{r in E_t} w_r log p_theta(X^(r)),

but the concrete update is delegated to the chosen family.

This module ships with two families:

1. ``FactorizedCategorical``:
   independent per-coordinate categoricals.
2. ``JointCategorical``:
   a full joint categorical over the entire product space, which can model
   arbitrary dependence but scales as ``prod_i n_i``.

Stopping rule
=============

Following the CE tutorial more closely than the original baseline, the optimizer
supports a quantile-stability stopping rule. Let ``gamma_hat_t`` be the elite
threshold at iteration ``t``. We stop when ``gamma_hat_t`` is unchanged within a
tolerance for ``d`` consecutive iterations, or when the sampling family becomes
degenerate, or when FACE-style adaptive sample sizing stalls at the maximum
population size.

Pseudocode
==========

```text
input:
    domain sizes n_1, ..., n_k
    batched score function S
    sampling family p_theta
    sample size N
    elite fraction rho
    smoothing alpha
    Dirichlet prior kappa
    max_iters T

initialize:
    theta <- initial sampling-family parameters
    best_x <- None
    best_score <- -inf
    gamma_history <- []

for t in 1, ..., T:
    sample X^(1), ..., X^(N) ~ p_theta
    optionally inject the best-ever sample (elitism)
    evaluate scores s_r = S(X^(r)) in batched form
    validate scores; if non-finite:
        either raise or mask them out, depending on nan_policy

    choose elite set E_t from the top rho fraction of samples
    gamma_t <- worst elite score
    record population-best and elite-mean scores
    update best_x, best_score using the best sample seen so far

    compute elite weights w_r
    update the sampling family by solving the elite weighted MLE
        theta_hat^(t) = argmax_theta sum_{r in E_t} w_r log p_theta(X^(r))

    if gamma_t has been stable for d consecutive iterations:
        stop with reason "quantile_stable"
    if the sampling family is degenerate:
        stop with reason "degenerate"
    if FACE is enabled and elite performance has stalled:
        grow N up to N_max; if already at N_max and still stalled, stop

evaluate mode_x = mode(theta)
evaluate mode_score = S(mode_x)

return:
    best sampled x, best score, mode x, mode score,
    final marginals, threshold history, and stop metadata
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import torch

BatchedScoreFunction = Callable[[torch.Tensor], torch.Tensor]
"""Contract: takes a ``[B, k]`` int64 tensor (0-based categories) and
returns a ``[B]`` float tensor of scores. Must be batch-vectorized."""

ContinuousScoreFunction = Callable[[torch.Tensor], torch.Tensor]
"""Contract: takes a floating sample tensor ``[B, ...]`` and returns scores ``[B]``."""


class EliteWeighting(Protocol):
    """Maps elite scores to non-negative weights summing to 1."""

    def __call__(self, elite_scores: torch.Tensor) -> torch.Tensor: ...


class UniformWeighting:
    """Standard CE weighting: every elite sample contributes equally."""

    def __call__(self, elite_scores: torch.Tensor) -> torch.Tensor:
        return torch.full_like(elite_scores, 1.0 / elite_scores.numel())


class ScoreProportionalWeighting:
    """Weight by ``psi(S) = S - min(S) + eps``"""

    def __init__(self, eps: float = 1e-12):
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.eps = float(eps)

    def __call__(self, elite_scores: torch.Tensor) -> torch.Tensor:
        shifted = elite_scores - elite_scores.min() + self.eps
        return shifted / shifted.sum()


class SoftmaxWeighting:
    """MPPI-style exponential weights: ``w ∝ exp((S - max S) / tau)``."""

    def __init__(self, temperature: float):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def __call__(self, elite_scores: torch.Tensor) -> torch.Tensor:
        logits = (elite_scores - elite_scores.max()) / self.temperature
        weights = torch.exp(logits)
        return weights / weights.sum()


class SamplingFamily(Protocol):
    """Parametric sampling family over the product domain.

    Indexing is 0-based throughout. Implementations own their parameter tensors
    and mutate them in-place during ``update``.
    """

    domain_sizes: tuple[int, ...]

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor: ...

    def update(self, elite_samples: torch.Tensor, elite_weights: torch.Tensor) -> None: ...

    def marginals(self) -> tuple[torch.Tensor, ...]: ...

    def mode(self) -> torch.Tensor: ...

    def is_degenerate(self, tol: float) -> bool: ...


class FactorizedCategorical:
    """Independent categorical per coordinate with Dirichlet-prior smoothing.

    Parameters
    ----------
    domain_sizes
        Per-coordinate domain sizes.
    smoothing
        Convex combination with previous parameters:
        ``theta <- alpha * theta_hat + (1 - alpha) * theta_prev``.
        Default ``0.7``.
    dirichlet_prior
        Pseudocount ``kappa >= 0`` added to each category. ``kappa = 0``
        recovers the plain MLE, ``kappa = 1`` is Laplace smoothing.
    initial_probs
        Optional per-coordinate starting probabilities.
    device
        Torch device for parameter storage.
    """

    def __init__(
        self,
        domain_sizes: Sequence[int],
        *,
        smoothing: float = 0.7,
        dirichlet_prior: float = 0.0,
        initial_probs: Optional[Sequence[Sequence[float] | torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
    ):
        if not domain_sizes:
            raise ValueError("domain_sizes must contain at least one coordinate")
        if any(int(n) <= 0 for n in domain_sizes):
            raise ValueError("each domain size must be positive")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("smoothing must lie in (0, 1]")
        if dirichlet_prior < 0.0:
            raise ValueError("dirichlet_prior must be non-negative")

        self.domain_sizes = tuple(int(n) for n in domain_sizes)
        self.smoothing = float(smoothing)
        self.dirichlet_prior = float(dirichlet_prior)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._params = self._init_params(initial_probs)

    def _init_params(
        self,
        initial_probs: Optional[Sequence[Sequence[float] | torch.Tensor]],
    ) -> list[torch.Tensor]:
        if initial_probs is None:
            return [torch.full((n,), 1.0 / n, dtype=torch.float32, device=self.device) for n in self.domain_sizes]

        if len(initial_probs) != len(self.domain_sizes):
            raise ValueError("initial_probs must match the number of coordinates")

        params: list[torch.Tensor] = []
        for idx, (n, probs) in enumerate(zip(self.domain_sizes, initial_probs, strict=True)):
            tensor = torch.as_tensor(probs, dtype=torch.float32, device=self.device)
            if tensor.ndim != 1 or tensor.numel() != n:
                raise ValueError(f"initial_probs[{idx}] must be a length-{n} probability vector")
            if torch.any(tensor < 0):
                raise ValueError("initial_probs must be non-negative")
            total = tensor.sum()
            if total <= 0:
                raise ValueError("initial_probs must contain positive mass")
            params.append(tensor / total)
        return params

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        coords = [
            torch.multinomial(param, num_samples, replacement=True, generator=generator) for param in self._params
        ]
        return torch.stack(coords, dim=1)

    def update(self, elite_samples: torch.Tensor, elite_weights: torch.Tensor) -> None:
        total_w = elite_weights.sum()
        for coord, (n, prev) in enumerate(zip(self.domain_sizes, self._params, strict=True)):
            counts = torch.bincount(
                elite_samples[:, coord],
                weights=elite_weights,
                minlength=n,
            ).to(torch.float32)
            posterior = (counts + self.dirichlet_prior) / (total_w + n * self.dirichlet_prior)
            updated = self.smoothing * posterior + (1.0 - self.smoothing) * prev
            self._params[coord] = updated / updated.sum()

    def marginals(self) -> tuple[torch.Tensor, ...]:
        return tuple(param.detach().clone() for param in self._params)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(param) for param in self._params])

    def is_degenerate(self, tol: float) -> bool:
        return all((1.0 - param.max().item()) <= tol for param in self._params)


class JointCategorical:
    """Full joint categorical family over the entire product space.

    This sampler supports arbitrary dependence between coordinates. It is
    intended for small domains because its parameter count is
    ``prod_i n_i``.
    """

    def __init__(
        self,
        domain_sizes: Sequence[int],
        *,
        smoothing: float = 0.7,
        dirichlet_prior: float = 0.0,
        initial_probs: Optional[Sequence[float] | torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
    ):
        if not domain_sizes:
            raise ValueError("domain_sizes must contain at least one coordinate")
        if any(int(n) <= 0 for n in domain_sizes):
            raise ValueError("each domain size must be positive")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("smoothing must lie in (0, 1]")
        if dirichlet_prior < 0.0:
            raise ValueError("dirichlet_prior must be non-negative")

        self.domain_sizes = tuple(int(n) for n in domain_sizes)
        self.smoothing = float(smoothing)
        self.dirichlet_prior = float(dirichlet_prior)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        flat_size = math.prod(self.domain_sizes)
        if initial_probs is None:
            self._joint = torch.full(
                (flat_size,),
                1.0 / flat_size,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            tensor = torch.as_tensor(initial_probs, dtype=torch.float32, device=self.device)
            if tensor.ndim != 1 or tensor.numel() != flat_size:
                raise ValueError(f"initial_probs must be a length-{flat_size} probability vector")
            if torch.any(tensor < 0):
                raise ValueError("initial_probs must be non-negative")
            total = tensor.sum()
            if total <= 0:
                raise ValueError("initial_probs must contain positive mass")
            self._joint = tensor / total

        self._radices = self._build_radices()

    def _build_radices(self) -> torch.Tensor:
        radices = []
        running = 1
        for size in reversed(self.domain_sizes[1:]):
            running *= size
            radices.append(running)
        return torch.tensor(
            [math.prod(self.domain_sizes[i + 1 :]) for i in range(len(self.domain_sizes))],
            dtype=torch.int64,
            device=self.device,
        )

    def _flatten(self, samples: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(samples.shape[0], dtype=torch.int64, device=self.device)
        for coord, size in enumerate(self.domain_sizes):
            stride = math.prod(self.domain_sizes[coord + 1 :]) if coord + 1 < len(self.domain_sizes) else 1
            flat = flat + samples[:, coord] * stride
        return flat

    def _unflatten(self, flat_indices: torch.Tensor) -> torch.Tensor:
        coords = []
        remainder = flat_indices.clone()
        for coord, size in enumerate(self.domain_sizes):
            stride = math.prod(self.domain_sizes[coord + 1 :]) if coord + 1 < len(self.domain_sizes) else 1
            value = remainder // stride
            remainder = remainder % stride
            coords.append(value)
        return torch.stack(coords, dim=1)

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        flat = torch.multinomial(self._joint, num_samples, replacement=True, generator=generator)
        return self._unflatten(flat)

    def update(self, elite_samples: torch.Tensor, elite_weights: torch.Tensor) -> None:
        flat = self._flatten(elite_samples)
        counts = torch.bincount(
            flat,
            weights=elite_weights,
            minlength=self._joint.numel(),
        ).to(torch.float32)
        total_w = elite_weights.sum()
        posterior = (counts + self.dirichlet_prior) / (total_w + self._joint.numel() * self.dirichlet_prior)
        updated = self.smoothing * posterior + (1.0 - self.smoothing) * self._joint
        self._joint = updated / updated.sum()

    def marginals(self) -> tuple[torch.Tensor, ...]:
        table = self._joint.view(*self.domain_sizes)
        marginals: list[torch.Tensor] = []
        for coord in range(len(self.domain_sizes)):
            axes = tuple(i for i in range(len(self.domain_sizes)) if i != coord)
            marginals.append(table.sum(dim=axes))
        return tuple(m.detach().clone() for m in marginals)

    def mode(self) -> torch.Tensor:
        flat = int(torch.argmax(self._joint).item())
        return self._unflatten(torch.tensor([flat], device=self.device))[0]

    def is_degenerate(self, tol: float) -> bool:
        return (1.0 - self._joint.max().item()) <= tol


class ScoreEstimator(Protocol):
    """Callable that maps ``[B, k]`` zero-indexed samples to a ``[B]`` score tensor."""

    def __call__(self, samples: torch.Tensor) -> torch.Tensor: ...


class DeterministicScore:
    """Thin wrapper around a batched deterministic score function."""

    def __init__(self, score_fn: BatchedScoreFunction):
        self.score_fn = score_fn

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.score_fn(samples)


class AveragedScore:
    """Average ``num_evals`` independent evaluations per sample.

    For stochastic / simulation-based ``S``. The underlying ``score_fn`` is
    called once on a batch of size ``num_evals * B`` built by
    ``repeat_interleave`` — memory scales linearly with ``num_evals``.
    """

    def __init__(self, score_fn: BatchedScoreFunction, num_evals: int):
        if num_evals <= 0:
            raise ValueError("num_evals must be positive")
        self.score_fn = score_fn
        self.num_evals = int(num_evals)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        batch_size = samples.shape[0]
        repeated = samples.repeat_interleave(self.num_evals, dim=0)
        scores = self.score_fn(repeated)
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score_fn must return a torch.Tensor, got {type(scores).__name__}")
        return scores.view(batch_size, self.num_evals).mean(dim=1)


@dataclass
class IterationState:
    """Per-iteration snapshot passed to the optional callback."""

    iteration: int
    num_samples: int
    threshold: float
    elite_mean: float
    population_best: float
    best_ever: float
    probabilities: tuple[torch.Tensor, ...]


@dataclass
class CategoricalCEResult:
    """Result of CE optimization on a finite product domain."""

    best_x: tuple[int, ...]
    best_score: float
    mode_x: tuple[int, ...]
    mode_score: float
    probabilities: tuple[torch.Tensor, ...]
    thresholds: list[float]
    elite_mean_scores: list[float]
    population_best_scores: list[float]
    iterations: int
    converged: bool
    stop_reason: str


@dataclass
class ContinuousIterationState:
    """Per-iteration diagnostics for continuous MPPI."""

    iteration: int
    num_samples: int
    threshold: float
    elite_mean: float
    population_best: float
    best_ever: float
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class ContinuousMPPIResult:
    """Result of diagonal-Gaussian continuous MPPI optimization."""

    best_x: torch.Tensor
    best_score: float
    mode_x: torch.Tensor
    mode_score: float
    mean: torch.Tensor
    std: torch.Tensor
    thresholds: list[float]
    elite_mean_scores: list[float]
    population_best_scores: list[float]
    iterations: int
    converged: bool
    stop_reason: str


@dataclass
class ContinuousGMMIterationState:
    """Per-iteration diagnostics for GMM-seeded continuous MPPI.

    Shapes:
        component_probs: ``FloatTensor[H, M]``.
        means/stds: ``FloatTensor[H, M, D]``.
    """

    iteration: int
    num_samples: int
    threshold: float
    elite_mean: float
    population_best: float
    best_ever: float
    component_probs: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor


@dataclass
class ContinuousGMMMPPIResult:
    """Result of MPPI over a per-timestep Gaussian mixture.

    ``best_x`` and ``mode_x`` are latent action sequences shaped
    ``FloatTensor[H, D]``. ``means`` and ``stds`` keep the drifting component
    parameters as ``FloatTensor[H, M, D]``.
    """

    best_x: torch.Tensor
    best_score: float
    best_components: torch.Tensor
    mode_x: torch.Tensor
    mode_score: float
    mode_components: torch.Tensor
    component_probs: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor
    thresholds: list[float]
    elite_mean_scores: list[float]
    population_best_scores: list[float]
    iterations: int
    converged: bool
    stop_reason: str


class ContinuousGaussianMPPI:
    """MPPI-style optimizer over a diagonal Gaussian sampling distribution."""

    def __init__(
        self,
        *,
        num_samples: int = 256,
        elite_frac: float = 0.1,
        max_iters: int = 50,
        temperature: float = 1.0,
        smoothing: float = 0.7,
        noise_std: float = 1.0,
        convergence_tol: float = 1e-4,
        quantile_window: int = 5,
        elitism: bool = True,
        face_adaptive: bool = False,
        num_samples_max: Optional[int] = None,
        face_stall_iters: int = 3,
        nan_policy: str = "raise",
        callback: Optional[Callable[[ContinuousIterationState], None]] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0.0 < elite_frac <= 1.0):
            raise ValueError("elite_frac must lie in (0, 1]")
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("smoothing must lie in (0, 1]")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if nan_policy not in ("raise", "filter"):
            raise ValueError("nan_policy must be 'raise' or 'filter'")
        if quantile_window < 1:
            raise ValueError("quantile_window must be >= 1")
        if convergence_tol < 0.0:
            raise ValueError("convergence_tol must be non-negative")
        if face_adaptive:
            if num_samples_max is None:
                num_samples_max = 4 * num_samples
            if num_samples_max < num_samples:
                raise ValueError("num_samples_max must be >= num_samples")
            if face_stall_iters < 1:
                raise ValueError("face_stall_iters must be >= 1")

        self.num_samples = int(num_samples)
        self.elite_frac = float(elite_frac)
        self.max_iters = int(max_iters)
        self.temperature = float(temperature)
        self.smoothing = float(smoothing)
        self.noise_std = float(noise_std)
        self.convergence_tol = float(convergence_tol)
        self.quantile_window = int(quantile_window)
        self.elitism = bool(elitism)
        self.face_adaptive = bool(face_adaptive)
        self.num_samples_max = int(num_samples_max) if num_samples_max is not None else None
        self.face_stall_iters = int(face_stall_iters)
        self.nan_policy = nan_policy
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.callback = callback
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def optimize(
        self,
        score_fn: ContinuousScoreFunction,
        initial_mean: torch.Tensor,
        initial_std: torch.Tensor | None = None,
    ) -> ContinuousMPPIResult:
        """Optimize a batched continuous score function by maximizing scores."""

        mean = initial_mean.to(device=self.device, dtype=torch.float32).clone()
        if initial_std is None:
            std = torch.full_like(mean, self.noise_std)
        else:
            std = initial_std.to(device=self.device, dtype=torch.float32).clone()
        if std.shape != mean.shape:
            raise ValueError(f"initial_std shape {tuple(std.shape)} must match initial_mean {tuple(mean.shape)}")
        if torch.any(std <= 0):
            raise ValueError("initial_std must be strictly positive")

        thresholds: list[float] = []
        elite_mean_scores: list[float] = []
        population_best_scores: list[float] = []
        best_score = float("-inf")
        best_x: torch.Tensor | None = None
        current_num_samples = self.num_samples
        face_stall_count = 0
        stop_reason = "max_iters"
        weighting = SoftmaxWeighting(self.temperature)

        for iteration in range(self.max_iters):
            samples = self._sample(mean, std, current_num_samples)
            if self.elitism and best_x is not None:
                samples = torch.cat([best_x.unsqueeze(0), samples], dim=0)

            scores = self._score(score_fn, samples)
            num_elites = max(1, int(round(self.elite_frac * samples.shape[0])))
            elite_scores, elite_indices = self._select_elites(scores, num_elites)
            elite_samples = samples.index_select(0, elite_indices)

            threshold = float(elite_scores[-1].item())
            thresholds.append(threshold)
            elite_mean_scores.append(float(elite_scores.mean().item()))
            population_best = float(elite_scores[0].item())
            population_best_scores.append(population_best)

            if population_best > best_score:
                best_score = population_best
                best_x = elite_samples[0].detach().clone()

            weights = weighting(elite_scores).to(device=self.device, dtype=torch.float32)
            weight_sum = weights.sum()
            if not torch.isfinite(weight_sum) or weight_sum <= 0:
                raise RuntimeError("elite weights must sum to a positive finite value")
            weights = weights / weight_sum

            mean, std = self._update_distribution(mean, std, elite_samples, weights)

            if self.callback is not None:
                self.callback(
                    ContinuousIterationState(
                        iteration=iteration + 1,
                        num_samples=int(samples.shape[0]),
                        threshold=threshold,
                        elite_mean=elite_mean_scores[-1],
                        population_best=population_best,
                        best_ever=best_score,
                        mean=mean.detach().clone(),
                        std=std.detach().clone(),
                    )
                )

            if self._quantile_stable(thresholds):
                stop_reason = "quantile_stable"
                break
            if self._is_degenerate(std):
                stop_reason = "degenerate"
                break

            if self.face_adaptive:
                improved = len(population_best_scores) < 2 or (population_best > max(population_best_scores[:-1]))
                assert self.num_samples_max is not None
                if improved:
                    face_stall_count = 0
                elif current_num_samples < self.num_samples_max:
                    current_num_samples = min(2 * current_num_samples, self.num_samples_max)
                    face_stall_count = 0
                else:
                    face_stall_count += 1
                    if face_stall_count >= self.face_stall_iters:
                        stop_reason = "face_stall"
                        break

        if best_x is None:
            raise RuntimeError("optimization did not produce any samples")

        mode_score = float(self._score(score_fn, mean.unsqueeze(0)).item())
        return ContinuousMPPIResult(
            best_x=best_x.detach().cpu(),
            best_score=best_score,
            mode_x=mean.detach().cpu(),
            mode_score=mode_score,
            mean=mean.detach().cpu(),
            std=std.detach().cpu(),
            thresholds=thresholds,
            elite_mean_scores=elite_mean_scores,
            population_best_scores=population_best_scores,
            iterations=len(thresholds),
            converged=stop_reason in ("quantile_stable", "degenerate"),
            stop_reason=stop_reason,
        )

    def _sample(self, mean: torch.Tensor, std: torch.Tensor, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, *mean.shape), generator=self.generator, device=self.device)
        return mean.unsqueeze(0) + noise * std.unsqueeze(0)

    def _score(self, score_fn: ContinuousScoreFunction, samples: torch.Tensor) -> torch.Tensor:
        scores = score_fn(samples)
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score_fn must return a torch.Tensor, got {type(scores).__name__}")
        scores = scores.to(device=self.device, dtype=torch.float32)
        if scores.ndim != 1 or scores.shape[0] != samples.shape[0]:
            raise ValueError(
                f"score_fn must return a 1-D tensor of length {samples.shape[0]}, got shape {tuple(scores.shape)}"
            )
        nonfinite = ~torch.isfinite(scores)
        if nonfinite.any():
            n_bad = int(nonfinite.sum().item())
            if self.nan_policy == "raise":
                raise ValueError(
                    f"score_fn returned {n_bad} non-finite value(s); " "set nan_policy='filter' to mask them as -inf"
                )
            scores = torch.where(nonfinite, torch.tensor(float("-inf"), device=self.device), scores)
        return scores

    def _select_elites(self, scores: torch.Tensor, num_elites: int) -> tuple[torch.Tensor, torch.Tensor]:
        finite_count = int(torch.isfinite(scores).sum().item())
        if finite_count == 0:
            raise RuntimeError("all samples produced non-finite scores")
        return torch.topk(scores, k=min(num_elites, finite_count), largest=True)

    def _update_distribution(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        elite_samples: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_shape = (weights.numel(),) + (1,) * mean.ndim
        weights_view = weights.reshape(view_shape)
        empirical_mean = (elite_samples * weights_view).sum(dim=0)
        empirical_var = ((elite_samples - empirical_mean.unsqueeze(0)).pow(2) * weights_view).sum(dim=0)
        empirical_std = empirical_var.clamp_min(1e-12).sqrt()
        smoothing = self.smoothing
        updated_mean = smoothing * empirical_mean + (1.0 - smoothing) * mean
        updated_std = smoothing * empirical_std + (1.0 - smoothing) * std
        return updated_mean, updated_std.clamp_min(1e-6)

    def _quantile_stable(self, thresholds: list[float]) -> bool:
        if len(thresholds) < self.quantile_window + 1:
            return False
        recent = thresholds[-(self.quantile_window + 1) :]
        return all(abs(value - recent[0]) <= self.convergence_tol for value in recent)

    def _is_degenerate(self, std: torch.Tensor) -> bool:
        return bool(torch.all(std <= self.convergence_tol).item())


class ContinuousGMMMPPI:
    """MPPI optimizer over a drifting diagonal-Gaussian mixture.

    The mixture is horizon-aware: each timestep ``t`` owns ``M`` component
    means and standard deviations. Sampling first draws component indices
    ``LongTensor[N, H]``, then draws latent actions ``FloatTensor[N, H, D]``
    from the selected diagonal Gaussian at each timestep.
    """

    def __init__(
        self,
        *,
        num_samples: int = 256,
        elite_frac: float = 0.1,
        max_iters: int = 50,
        temperature: float = 1.0,
        smoothing: float = 0.7,
        noise_std: float = 1.0,
        min_std: float = 1e-6,
        max_std: float | None = None,
        convergence_tol: float = 1e-4,
        quantile_window: int = 5,
        elitism: bool = True,
        face_adaptive: bool = False,
        num_samples_max: Optional[int] = None,
        face_stall_iters: int = 3,
        nan_policy: str = "raise",
        callback: Optional[Callable[[ContinuousGMMIterationState], None]] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0.0 < elite_frac <= 1.0):
            raise ValueError("elite_frac must lie in (0, 1]")
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("smoothing must lie in (0, 1]")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if min_std <= 0:
            raise ValueError("min_std must be positive")
        if max_std is not None and max_std < min_std:
            raise ValueError("max_std must be >= min_std")
        if nan_policy not in ("raise", "filter"):
            raise ValueError("nan_policy must be 'raise' or 'filter'")
        if quantile_window < 1:
            raise ValueError("quantile_window must be >= 1")
        if convergence_tol < 0.0:
            raise ValueError("convergence_tol must be non-negative")
        if face_adaptive:
            if num_samples_max is None:
                num_samples_max = 4 * num_samples
            if num_samples_max < num_samples:
                raise ValueError("num_samples_max must be >= num_samples")
            if face_stall_iters < 1:
                raise ValueError("face_stall_iters must be >= 1")

        self.num_samples = int(num_samples)
        self.elite_frac = float(elite_frac)
        self.max_iters = int(max_iters)
        self.temperature = float(temperature)
        self.smoothing = float(smoothing)
        self.noise_std = float(noise_std)
        self.min_std = float(min_std)
        self.max_std = None if max_std is None else float(max_std)
        self.convergence_tol = float(convergence_tol)
        self.quantile_window = int(quantile_window)
        self.elitism = bool(elitism)
        self.face_adaptive = bool(face_adaptive)
        self.num_samples_max = int(num_samples_max) if num_samples_max is not None else None
        self.face_stall_iters = int(face_stall_iters)
        self.nan_policy = nan_policy
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.callback = callback
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def optimize(
        self,
        score_fn: ContinuousScoreFunction,
        initial_means: torch.Tensor,
        initial_stds: torch.Tensor | None = None,
        initial_component_probs: torch.Tensor | None = None,
    ) -> ContinuousGMMMPPIResult:
        """Optimize a score over latent action sequences.

        Args:
            score_fn: Batched maximization objective accepting
                ``FloatTensor[N, H, D]`` and returning ``FloatTensor[N]``.
            initial_means: Component centers ``FloatTensor[H, M, D]``.
            initial_stds: Optional component stds ``FloatTensor[H, M, D]``.
            initial_component_probs: Optional mixture weights
                ``FloatTensor[H, M]``.
        """

        means = initial_means.to(device=self.device, dtype=torch.float32).clone()
        if means.ndim != 3:
            raise ValueError(f"initial_means must have shape [H, M, D], got {tuple(means.shape)}")
        horizon, num_components, _ = means.shape
        if initial_stds is None:
            stds = torch.full_like(means, self.noise_std)
        else:
            stds = initial_stds.to(device=self.device, dtype=torch.float32).clone()
        if stds.shape != means.shape:
            raise ValueError(f"initial_stds shape {tuple(stds.shape)} must match initial_means {tuple(means.shape)}")
        if torch.any(stds <= 0):
            raise ValueError("initial_stds must be strictly positive")

        if initial_component_probs is None:
            component_probs = torch.full(
                (horizon, num_components),
                1.0 / num_components,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            component_probs = initial_component_probs.to(device=self.device, dtype=torch.float32).clone()
        if component_probs.shape != (horizon, num_components):
            raise ValueError(
                f"initial_component_probs must have shape {(horizon, num_components)}, "
                f"got {tuple(component_probs.shape)}"
            )
        if torch.any(component_probs < 0) or torch.any(component_probs.sum(dim=1) <= 0):
            raise ValueError("initial_component_probs must be non-negative with positive per-timestep mass")
        component_probs = component_probs / component_probs.sum(dim=1, keepdim=True)

        thresholds: list[float] = []
        elite_mean_scores: list[float] = []
        population_best_scores: list[float] = []
        best_score = float("-inf")
        best_x: torch.Tensor | None = None
        best_components: torch.Tensor | None = None
        current_num_samples = self.num_samples
        face_stall_count = 0
        stop_reason = "max_iters"
        weighting = SoftmaxWeighting(self.temperature)

        for iteration in range(self.max_iters):
            samples, components = self._sample(means, stds, component_probs, current_num_samples)
            if self.elitism and best_x is not None and best_components is not None:
                samples = torch.cat([best_x.unsqueeze(0), samples], dim=0)
                components = torch.cat([best_components.unsqueeze(0), components], dim=0)

            scores = self._score(score_fn, samples)
            num_elites = max(1, int(round(self.elite_frac * samples.shape[0])))
            elite_scores, elite_indices = self._select_elites(scores, num_elites)
            elite_samples = samples.index_select(0, elite_indices)
            elite_components = components.index_select(0, elite_indices)

            threshold = float(elite_scores[-1].item())
            thresholds.append(threshold)
            elite_mean_scores.append(float(elite_scores.mean().item()))
            population_best = float(elite_scores[0].item())
            population_best_scores.append(population_best)

            if population_best > best_score:
                best_score = population_best
                best_x = elite_samples[0].detach().clone()
                best_components = elite_components[0].detach().clone()

            weights = weighting(elite_scores).to(device=self.device, dtype=torch.float32)
            weight_sum = weights.sum()
            if not torch.isfinite(weight_sum) or weight_sum <= 0:
                raise RuntimeError("elite weights must sum to a positive finite value")
            weights = weights / weight_sum

            means, stds, component_probs = self._update_distribution(
                means,
                stds,
                component_probs,
                elite_samples,
                elite_components,
                weights,
            )

            if self.callback is not None:
                self.callback(
                    ContinuousGMMIterationState(
                        iteration=iteration + 1,
                        num_samples=int(samples.shape[0]),
                        threshold=threshold,
                        elite_mean=elite_mean_scores[-1],
                        population_best=population_best,
                        best_ever=best_score,
                        component_probs=component_probs.detach().clone(),
                        means=means.detach().clone(),
                        stds=stds.detach().clone(),
                    )
                )

            if self._quantile_stable(thresholds):
                stop_reason = "quantile_stable"
                break
            if self._is_degenerate(stds, component_probs):
                stop_reason = "degenerate"
                break

            if self.face_adaptive:
                improved = len(population_best_scores) < 2 or (population_best > max(population_best_scores[:-1]))
                assert self.num_samples_max is not None
                if improved:
                    face_stall_count = 0
                elif current_num_samples < self.num_samples_max:
                    current_num_samples = min(2 * current_num_samples, self.num_samples_max)
                    face_stall_count = 0
                else:
                    face_stall_count += 1
                    if face_stall_count >= self.face_stall_iters:
                        stop_reason = "face_stall"
                        break

        if best_x is None or best_components is None:
            raise RuntimeError("optimization did not produce any samples")

        mode_components = torch.argmax(component_probs, dim=1)
        mode_x = means[torch.arange(horizon, device=self.device), mode_components]
        mode_score = float(self._score(score_fn, mode_x.unsqueeze(0)).item())
        return ContinuousGMMMPPIResult(
            best_x=best_x.detach().cpu(),
            best_score=best_score,
            best_components=best_components.detach().cpu(),
            mode_x=mode_x.detach().cpu(),
            mode_score=mode_score,
            mode_components=mode_components.detach().cpu(),
            component_probs=component_probs.detach().cpu(),
            means=means.detach().cpu(),
            stds=stds.detach().cpu(),
            thresholds=thresholds,
            elite_mean_scores=elite_mean_scores,
            population_best_scores=population_best_scores,
            iterations=len(thresholds),
            converged=stop_reason in ("quantile_stable", "degenerate"),
            stop_reason=stop_reason,
        )

    def _sample(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        component_probs: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = component_probs.size(0)
        components = torch.stack(
            [
                torch.multinomial(component_probs[t], num_samples, replacement=True, generator=self.generator)
                for t in range(horizon)
            ],
            dim=1,
        ).to(self.device)
        time_index = torch.arange(horizon, device=self.device).unsqueeze(0).expand(num_samples, -1)
        selected_means = means[time_index, components]
        selected_stds = stds[time_index, components]
        noise = torch.randn(selected_means.shape, generator=self.generator, device=self.device)
        return selected_means + noise * selected_stds, components

    def _score(self, score_fn: ContinuousScoreFunction, samples: torch.Tensor) -> torch.Tensor:
        scores = score_fn(samples)
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score_fn must return a torch.Tensor, got {type(scores).__name__}")
        scores = scores.to(device=self.device, dtype=torch.float32)
        if scores.ndim != 1 or scores.shape[0] != samples.shape[0]:
            raise ValueError(
                f"score_fn must return a 1-D tensor of length {samples.shape[0]}, got shape {tuple(scores.shape)}"
            )
        nonfinite = ~torch.isfinite(scores)
        if nonfinite.any():
            n_bad = int(nonfinite.sum().item())
            if self.nan_policy == "raise":
                raise ValueError(
                    f"score_fn returned {n_bad} non-finite value(s); " "set nan_policy='filter' to mask them as -inf"
                )
            scores = torch.where(nonfinite, torch.tensor(float("-inf"), device=self.device), scores)
        return scores

    def _select_elites(self, scores: torch.Tensor, num_elites: int) -> tuple[torch.Tensor, torch.Tensor]:
        finite_count = int(torch.isfinite(scores).sum().item())
        if finite_count == 0:
            raise RuntimeError("all samples produced non-finite scores")
        return torch.topk(scores, k=min(num_elites, finite_count), largest=True)

    def _update_distribution(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        component_probs: torch.Tensor,
        elite_samples: torch.Tensor,
        elite_components: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        updated_means = means.clone()
        updated_stds = stds.clone()
        updated_probs = component_probs.clone()
        horizon, num_components, _ = means.shape
        for t in range(horizon):
            counts = torch.bincount(elite_components[:, t], weights=weights, minlength=num_components).to(torch.float32)
            # Component probabilities track which seed actions survive elite rollout scoring.
            posterior_probs = counts / counts.sum().clamp_min(1e-12)
            probs = self.smoothing * posterior_probs + (1.0 - self.smoothing) * component_probs[t]
            updated_probs[t] = probs / probs.sum().clamp_min(1e-12)
            for component_id in range(num_components):
                mask = elite_components[:, t] == component_id
                if not bool(mask.any()):
                    continue
                component_weights = weights[mask]
                component_weights = component_weights / component_weights.sum().clamp_min(1e-12)
                values = elite_samples[mask, t]
                mean = (values * component_weights[:, None]).sum(dim=0)
                var = ((values - mean) ** 2 * component_weights[:, None]).sum(dim=0)
                std = var.clamp_min(self.min_std * self.min_std).sqrt()
                updated_means[t, component_id] = self.smoothing * mean + (1.0 - self.smoothing) * means[t, component_id]
                updated_stds[t, component_id] = self.smoothing * std + (1.0 - self.smoothing) * stds[t, component_id]
        if self.max_std is not None:
            updated_stds = updated_stds.clamp(min=self.min_std, max=self.max_std)
        else:
            updated_stds = updated_stds.clamp_min(self.min_std)
        return updated_means, updated_stds, updated_probs

    def _quantile_stable(self, thresholds: list[float]) -> bool:
        if len(thresholds) < self.quantile_window + 1:
            return False
        recent = thresholds[-(self.quantile_window + 1) :]
        return all(abs(value - recent[0]) <= self.convergence_tol for value in recent)

    def _is_degenerate(self, stds: torch.Tensor, component_probs: torch.Tensor) -> bool:
        std_done = bool(torch.all(stds <= self.convergence_tol).item())
        mixture_done = bool(torch.all((1.0 - component_probs.max(dim=1).values) <= self.convergence_tol).item())
        return std_done and mixture_done


class ProductCategoricalCrossEntropy:
    """Cross-entropy optimizer for finite product domains."""

    def __init__(
        self,
        domain_sizes: Sequence[int],
        *,
        num_samples: Optional[int] = None,
        elite_frac: float = 0.1,
        max_iters: int = 50,
        sampling_family: Optional[SamplingFamily] = None,
        elite_weighting: Optional[EliteWeighting] = None,
        nan_policy: str = "raise",
        quantile_window: int = 5,
        convergence_tol: float = 1e-4,
        elitism: bool = True,
        face_adaptive: bool = False,
        num_samples_max: Optional[int] = None,
        face_stall_iters: int = 3,
        callback: Optional[Callable[[IterationState], None]] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ):
        if not domain_sizes:
            raise ValueError("domain_sizes must contain at least one coordinate")
        self.domain_sizes = tuple(int(n) for n in domain_sizes)
        if any(n <= 0 for n in self.domain_sizes):
            raise ValueError("each domain size must be positive")

        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if num_samples is None:
            num_samples = max(256, 5 * sum(self.domain_sizes))
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0.0 < elite_frac <= 1.0):
            raise ValueError("elite_frac must lie in (0, 1]")
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if nan_policy not in ("raise", "filter"):
            raise ValueError("nan_policy must be 'raise' or 'filter'")
        if quantile_window < 1:
            raise ValueError("quantile_window must be >= 1")
        if convergence_tol < 0.0:
            raise ValueError("convergence_tol must be non-negative")
        if face_adaptive:
            if num_samples_max is None:
                num_samples_max = 4 * num_samples
            if num_samples_max < num_samples:
                raise ValueError("num_samples_max must be >= num_samples")
            if face_stall_iters < 1:
                raise ValueError("face_stall_iters must be >= 1")

        self.num_samples = int(num_samples)
        self.elite_frac = float(elite_frac)
        self.max_iters = int(max_iters)
        self.sampling_family: SamplingFamily = sampling_family or FactorizedCategorical(
            self.domain_sizes,
            device=self.device,
        )
        if self.sampling_family.domain_sizes != self.domain_sizes:
            raise ValueError("sampling_family.domain_sizes must match optimizer domain_sizes")
        self.elite_weighting: EliteWeighting = elite_weighting or UniformWeighting()
        self.nan_policy = nan_policy
        self.quantile_window = int(quantile_window)
        self.convergence_tol = float(convergence_tol)
        self.elitism = bool(elitism)
        self.face_adaptive = bool(face_adaptive)
        self.num_samples_max = int(num_samples_max) if num_samples_max is not None else None
        self.face_stall_iters = int(face_stall_iters)
        self.callback = callback

        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def _score(self, estimator: ScoreEstimator, samples: torch.Tensor) -> torch.Tensor:
        scores = estimator(samples)
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score estimator must return a torch.Tensor, got {type(scores).__name__}")
        scores = scores.to(device=self.device, dtype=torch.float32)
        if scores.ndim != 1 or scores.shape[0] != samples.shape[0]:
            raise ValueError(
                f"score estimator must return a 1-D tensor of length {samples.shape[0]}, "
                f"got shape {tuple(scores.shape)}"
            )

        nonfinite = ~torch.isfinite(scores)
        if nonfinite.any():
            n_bad = int(nonfinite.sum().item())
            if self.nan_policy == "raise":
                raise ValueError(
                    f"score estimator returned {n_bad} non-finite value(s); "
                    "set nan_policy='filter' to mask them as -inf"
                )
            scores = torch.where(
                nonfinite,
                torch.tensor(float("-inf"), device=self.device),
                scores,
            )
        return scores

    def _select_elites(
        self,
        scores: torch.Tensor,
        num_elites: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        finite_count = int(torch.isfinite(scores).sum().item())
        if finite_count == 0:
            raise RuntimeError("all samples produced non-finite scores")
        k = min(num_elites, finite_count)
        return torch.topk(scores, k=k, largest=True)

    def _quantile_stable(self, thresholds: list[float]) -> bool:
        if len(thresholds) < self.quantile_window + 1:
            return False
        recent = thresholds[-(self.quantile_window + 1) :]
        return all(abs(value - recent[0]) <= self.convergence_tol for value in recent)

    def optimize(
        self,
        score_fn: BatchedScoreFunction | ScoreEstimator,
    ) -> CategoricalCEResult:
        """Run CE optimization with a batched score function or score estimator."""

        if isinstance(score_fn, (DeterministicScore, AveragedScore)):
            estimator: ScoreEstimator = score_fn
        else:
            estimator = DeterministicScore(score_fn)

        thresholds: list[float] = []
        elite_mean_scores: list[float] = []
        population_best_scores: list[float] = []
        best_score = float("-inf")
        best_x_zero: Optional[torch.Tensor] = None

        current_num_samples = self.num_samples
        face_stall_count = 0
        stop_reason = "max_iters"

        for iteration in range(self.max_iters):
            samples_zero = self.sampling_family.sample(current_num_samples, self.generator).to(self.device)

            if self.elitism and best_x_zero is not None:
                samples_zero = torch.cat([best_x_zero.unsqueeze(0), samples_zero], dim=0)

            scores = self._score(estimator, samples_zero)

            num_elites = max(1, int(round(self.elite_frac * samples_zero.shape[0])))
            elite_scores, elite_indices = self._select_elites(scores, num_elites)

            threshold = float(elite_scores[-1].item())
            thresholds.append(threshold)
            elite_mean_scores.append(float(elite_scores.mean().item()))
            population_best = float(elite_scores[0].item())
            population_best_scores.append(population_best)

            if population_best > best_score:
                best_score = population_best
                best_x_zero = samples_zero[int(elite_indices[0].item())].detach().clone()

            weights = self.elite_weighting(elite_scores).to(
                device=self.device,
                dtype=torch.float32,
            )
            weight_sum = weights.sum()
            if not torch.isfinite(weight_sum) or weight_sum <= 0:
                raise RuntimeError("elite weights must sum to a positive finite value")
            weights = weights / weight_sum

            elite_samples = samples_zero.index_select(0, elite_indices)
            self.sampling_family.update(elite_samples, weights)

            if self.callback is not None:
                self.callback(
                    IterationState(
                        iteration=iteration + 1,
                        num_samples=int(samples_zero.shape[0]),
                        threshold=threshold,
                        elite_mean=elite_mean_scores[-1],
                        population_best=population_best,
                        best_ever=best_score,
                        probabilities=self.sampling_family.marginals(),
                    )
                )

            if self._quantile_stable(thresholds):
                stop_reason = "quantile_stable"
                break
            if self.sampling_family.is_degenerate(self.convergence_tol):
                stop_reason = "degenerate"
                break

            if self.face_adaptive:
                improved = len(population_best_scores) < 2 or (population_best > max(population_best_scores[:-1]))
                assert self.num_samples_max is not None
                if improved:
                    face_stall_count = 0
                elif current_num_samples < self.num_samples_max:
                    current_num_samples = min(2 * current_num_samples, self.num_samples_max)
                    face_stall_count = 0
                else:
                    face_stall_count += 1
                    if face_stall_count >= self.face_stall_iters:
                        stop_reason = "face_stall"
                        break

        if best_x_zero is None:
            raise RuntimeError("optimization did not produce any samples")

        best_x = tuple(int(v) for v in best_x_zero.tolist())
        mode_zero = self.sampling_family.mode().to(self.device)
        mode_x = tuple(int(v) for v in mode_zero.tolist())
        mode_score = float(self._score(estimator, mode_zero.unsqueeze(0)).item())

        return CategoricalCEResult(
            best_x=best_x,
            best_score=best_score,
            mode_x=mode_x,
            mode_score=mode_score,
            probabilities=tuple(prob.detach().cpu() for prob in self.sampling_family.marginals()),
            thresholds=thresholds,
            elite_mean_scores=elite_mean_scores,
            population_best_scores=population_best_scores,
            iterations=len(thresholds),
            converged=stop_reason in ("quantile_stable", "degenerate"),
            stop_reason=stop_reason,
        )


class StructuredSamplingFamily(Protocol):
    """Protocol for categorical samplers with structured tensor samples."""

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        """Draw samples with leading batch dimension ``N``."""
        ...

    def update(self, elite_samples: torch.Tensor, elite_weights: torch.Tensor) -> None:
        """Update distribution parameters from weighted elite samples."""
        ...
    def marginals(self) -> tuple[torch.Tensor, ...]:
        """Return diagnostic marginal probabilities."""
        ...
    def mode(self) -> torch.Tensor:
        """Return the current modal structured sample without a batch axis."""
        ...
    def is_degenerate(self, tol: float) -> bool:
        """Return true when the distribution has collapsed under ``tol``."""
        ...

@dataclass
class StructuredCEResult:
    """Cross-entropy result for non-flat categorical samples."""

    best_x: torch.Tensor
    best_score: float
    mode_x: torch.Tensor
    mode_score: float
    probabilities: tuple[torch.Tensor, ...]
    thresholds: list[float]
    elite_mean_scores: list[float]
    population_best_scores: list[float]
    iterations: int
    converged: bool
    stop_reason: str


class StructuredCrossEntropy:
    """Cross-entropy optimizer for structured categorical sampling families.

    Unlike :class:`ProductCategoricalCrossEntropy`, this class does not assume
    that samples are flat product coordinates. The sampling family owns the
    structured tensor layout; the optimizer only expects a leading sample axis.
    """

    def __init__(
        self,
        *,
        sampling_family: StructuredSamplingFamily,
        num_samples: int = 256,
        elite_frac: float = 0.1,
        max_iters: int = 50,
        elite_weighting: Optional[EliteWeighting] = None,
        nan_policy: str = "raise",
        quantile_window: int = 5,
        convergence_tol: float = 1e-4,
        elitism: bool = True,
        face_adaptive: bool = False,
        num_samples_max: Optional[int] = None,
        face_stall_iters: int = 3,
        callback: Optional[Callable[[IterationState], None]] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0.0 < elite_frac <= 1.0):
            raise ValueError("elite_frac must lie in (0, 1]")
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if nan_policy not in ("raise", "filter"):
            raise ValueError("nan_policy must be 'raise' or 'filter'")
        if quantile_window < 1:
            raise ValueError("quantile_window must be >= 1")
        if convergence_tol < 0.0:
            raise ValueError("convergence_tol must be non-negative")
        if face_adaptive:
            if num_samples_max is None:
                num_samples_max = 4 * num_samples
            if num_samples_max < num_samples:
                raise ValueError("num_samples_max must be >= num_samples")
            if face_stall_iters < 1:
                raise ValueError("face_stall_iters must be >= 1")
        self.sampling_family = sampling_family
        self.num_samples = int(num_samples)
        self.elite_frac = float(elite_frac)
        self.max_iters = int(max_iters)
        self.elite_weighting: EliteWeighting = elite_weighting or UniformWeighting()
        self.nan_policy = nan_policy
        self.quantile_window = int(quantile_window)
        self.convergence_tol = float(convergence_tol)
        self.elitism = bool(elitism)
        self.face_adaptive = bool(face_adaptive)
        self.num_samples_max = int(num_samples_max) if num_samples_max is not None else None
        self.face_stall_iters = int(face_stall_iters)
        self.callback = callback
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def optimize(self, score_fn: BatchedScoreFunction | ScoreEstimator) -> StructuredCEResult:
        """Run CE optimization over structured samples."""

        estimator: ScoreEstimator = (
            score_fn if isinstance(score_fn, (DeterministicScore, AveragedScore)) else DeterministicScore(score_fn)
        )
        thresholds: list[float] = []
        elite_mean_scores: list[float] = []
        population_best_scores: list[float] = []
        best_score = float("-inf")
        best_x: torch.Tensor | None = None
        current_num_samples = self.num_samples
        face_stall_count = 0
        stop_reason = "max_iters"

        for iteration in range(self.max_iters):
            samples = self.sampling_family.sample(current_num_samples, self.generator).to(self.device)
            if self.elitism and best_x is not None:
                samples = torch.cat([best_x.unsqueeze(0), samples], dim=0)
            scores = self._score(estimator, samples)
            num_elites = max(1, int(round(self.elite_frac * samples.shape[0])))
            elite_scores, elite_indices = self._select_elites(scores, num_elites)
            elite_samples = samples.index_select(0, elite_indices)

            threshold = float(elite_scores[-1].item())
            thresholds.append(threshold)
            elite_mean_scores.append(float(elite_scores.mean().item()))
            population_best = float(elite_scores[0].item())
            population_best_scores.append(population_best)
            if population_best > best_score:
                best_score = population_best
                best_x = elite_samples[0].detach().clone()

            weights = self.elite_weighting(elite_scores).to(device=self.device, dtype=torch.float32)
            weights = weights / weights.sum().clamp_min(1e-12)
            self.sampling_family.update(elite_samples, weights)

            if self.callback is not None:
                self.callback(
                    IterationState(
                        iteration=iteration + 1,
                        num_samples=int(samples.shape[0]),
                        threshold=threshold,
                        elite_mean=elite_mean_scores[-1],
                        population_best=population_best,
                        best_ever=best_score,
                        probabilities=self.sampling_family.marginals(),
                    )
                )

            if self._quantile_stable(thresholds):
                stop_reason = "quantile_stable"
                break
            if self.sampling_family.is_degenerate(self.convergence_tol):
                stop_reason = "degenerate"
                break

            if self.face_adaptive:
                improved = len(population_best_scores) < 2 or (population_best > max(population_best_scores[:-1]))
                assert self.num_samples_max is not None
                if improved:
                    face_stall_count = 0
                elif current_num_samples < self.num_samples_max:
                    current_num_samples = min(2 * current_num_samples, self.num_samples_max)
                    face_stall_count = 0
                else:
                    face_stall_count += 1
                    if face_stall_count >= self.face_stall_iters:
                        stop_reason = "face_stall"
                        break

        if best_x is None:
            raise RuntimeError("optimization did not produce any samples")
        mode_x = self.sampling_family.mode().to(self.device)
        mode_score = float(self._score(estimator, mode_x.unsqueeze(0)).item())
        return StructuredCEResult(
            best_x=best_x.detach().cpu(),
            best_score=best_score,
            mode_x=mode_x.detach().cpu(),
            mode_score=mode_score,
            probabilities=tuple(prob.detach().cpu() for prob in self.sampling_family.marginals()),
            thresholds=thresholds,
            elite_mean_scores=elite_mean_scores,
            population_best_scores=population_best_scores,
            iterations=len(thresholds),
            converged=stop_reason in ("quantile_stable", "degenerate"),
            stop_reason=stop_reason,
        )

    def _score(self, estimator: ScoreEstimator, samples: torch.Tensor) -> torch.Tensor:
        scores = estimator(samples)
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"score estimator must return a torch.Tensor, got {type(scores).__name__}")
        scores = scores.to(device=self.device, dtype=torch.float32)
        if scores.ndim != 1 or scores.shape[0] != samples.shape[0]:
            raise ValueError(
                f"score estimator must return a 1-D tensor of length {samples.shape[0]}, "
                f"got shape {tuple(scores.shape)}"
            )
        nonfinite = ~torch.isfinite(scores)
        if nonfinite.any():
            n_bad = int(nonfinite.sum().item())
            if self.nan_policy == "raise":
                raise ValueError(
                    f"score estimator returned {n_bad} non-finite value(s); "
                    "set nan_policy='filter' to mask them as -inf"
                )
            scores = torch.where(nonfinite, torch.tensor(float("-inf"), device=self.device), scores)
        return scores

    def _select_elites(self, scores: torch.Tensor, num_elites: int) -> tuple[torch.Tensor, torch.Tensor]:
        finite_count = int(torch.isfinite(scores).sum().item())
        if finite_count == 0:
            raise RuntimeError("all samples produced non-finite scores")
        return torch.topk(scores, k=min(num_elites, finite_count), largest=True)

    def _quantile_stable(self, thresholds: list[float]) -> bool:
        if len(thresholds) < self.quantile_window + 1:
            return False
        recent = thresholds[-(self.quantile_window + 1) :]
        return all(abs(value - recent[0]) <= self.convergence_tol for value in recent)


def cross_entropy_optimize(
    domain_sizes: Sequence[int],
    score_fn: BatchedScoreFunction | ScoreEstimator,
    *,
    num_samples: Optional[int] = None,
    elite_frac: float = 0.1,
    max_iters: int = 50,
    smoothing: float = 0.7,
    dirichlet_prior: float = 0.0,
    sampling_family: Optional[SamplingFamily] = None,
    elite_weighting: Optional[EliteWeighting] = None,
    nan_policy: str = "raise",
    quantile_window: int = 5,
    convergence_tol: float = 1e-4,
    elitism: bool = True,
    face_adaptive: bool = False,
    num_samples_max: Optional[int] = None,
    face_stall_iters: int = 3,
    callback: Optional[Callable[[IterationState], None]] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    initial_probs: Optional[Sequence[Sequence[float] | torch.Tensor]] = None,
) -> CategoricalCEResult:
    """Convenience wrapper around :class:`ProductCategoricalCrossEntropy`.

    When ``sampling_family`` is ``None``, the default
    :class:`FactorizedCategorical` family is built from ``smoothing``,
    ``dirichlet_prior``, and ``initial_probs``. When a custom family is passed,
    those arguments must be configured on the family directly.
    """

    resolved_device = torch.device(device) if device is not None else torch.device("cpu")

    if sampling_family is None:
        sampling_family = FactorizedCategorical(
            domain_sizes,
            smoothing=smoothing,
            dirichlet_prior=dirichlet_prior,
            initial_probs=initial_probs,
            device=resolved_device,
        )
    elif initial_probs is not None:
        raise ValueError(
            "initial_probs is only supported with the default FactorizedCategorical; "
            "configure your custom sampling_family directly."
        )

    optimizer = ProductCategoricalCrossEntropy(
        domain_sizes=domain_sizes,
        num_samples=num_samples,
        elite_frac=elite_frac,
        max_iters=max_iters,
        sampling_family=sampling_family,
        elite_weighting=elite_weighting,
        nan_policy=nan_policy,
        quantile_window=quantile_window,
        convergence_tol=convergence_tol,
        elitism=elitism,
        face_adaptive=face_adaptive,
        num_samples_max=num_samples_max,
        face_stall_iters=face_stall_iters,
        callback=callback,
        seed=seed,
        device=resolved_device,
    )
    return optimizer.optimize(score_fn)
