import itertools
import math

import pytest
import torch
from acs_jepa.mpc import (
    AveragedScore,
    CategoricalCEResult,
    ContinuousGaussianMPPI,
    ContinuousGMMMPPI,
    ContinuousGMMMPPIResult,
    ContinuousIterationState,
    ContinuousMPPIResult,
    DeterministicScore,
    FactorizedCategorical,
    IterationState,
    JointCategorical,
    ProductCategoricalCrossEntropy,
    ScoreProportionalWeighting,
    SoftmaxWeighting,
    StructuredCEResult,
    StructuredCrossEntropy,
    UniformWeighting,
    cross_entropy_optimize,
)


def _linear_problem():
    domain_sizes = [2, 3, 2]
    u = torch.tensor([1.5, 2.0], dtype=torch.float32)

    def features(samples: torch.Tensor) -> torch.Tensor:
        x1 = samples[:, 0].to(torch.float32)
        x2 = samples[:, 1].to(torch.float32)
        x3 = samples[:, 2].to(torch.float32)
        return torch.stack(
            [
                2.0 * x1 + 0.5 * x2 - 1.5 * x3 + 0.75 * x1 * x2,
                -1.0 * x1 + 3.0 * x2 + 0.25 * x1 * x3 - 0.5 * x2 * x3,
            ],
            dim=1,
        )

    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        return (features(samples) * u).sum(dim=1)

    all_points = list(itertools.product(range(2), range(3), range(2)))
    all_samples = torch.tensor(all_points, dtype=torch.int64)
    all_scores = score_fn(all_samples)
    idx = int(torch.argmax(all_scores).item())
    best_x = tuple(int(v) for v in all_samples[idx].tolist())
    best_score = float(all_scores[idx].item())
    return domain_sizes, score_fn, best_x, best_score


def _target_problem(target):
    target_t = torch.tensor(target)

    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        return -(samples - target_t).abs().sum(dim=1).to(torch.float32)

    return score_fn


def test_product_categorical_ce_finds_bruteforce_optimum():
    domain_sizes, score_fn, brute_force_x, brute_force_score = _linear_problem()

    result = cross_entropy_optimize(
        domain_sizes=domain_sizes,
        score_fn=score_fn,
        num_samples=256,
        elite_frac=0.1,
        max_iters=40,
        smoothing=0.8,
        seed=0,
    )

    assert isinstance(result, CategoricalCEResult)
    assert result.best_x == brute_force_x
    assert result.best_score == brute_force_score
    assert isinstance(result.mode_score, float)
    assert len(result.probabilities) == len(domain_sizes)
    for probs, n in zip(result.probabilities, domain_sizes, strict=True):
        assert probs.shape == (n,)
        assert torch.isclose(probs.sum(), torch.tensor(1.0))
        assert torch.all(probs >= 0.0)


def test_score_function_must_be_batched():
    def bad_score(samples: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0)

    with pytest.raises(ValueError, match="1-D tensor"):
        cross_entropy_optimize([3, 3], bad_score, max_iters=2, seed=0)


def test_score_function_returning_wrong_length_raises():
    def bad_score(samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros(samples.shape[0] + 1)

    with pytest.raises(ValueError, match="1-D tensor"):
        cross_entropy_optimize([3, 3], bad_score, max_iters=2, seed=0)


def test_score_function_must_return_tensor():
    def bad_score(samples: torch.Tensor):
        return [0.0] * samples.shape[0]

    with pytest.raises(TypeError, match="torch.Tensor"):
        cross_entropy_optimize([3, 3], bad_score, max_iters=2, seed=0)


def test_num_samples_auto_sized_from_domain():
    optimizer = ProductCategoricalCrossEntropy(domain_sizes=[50, 50, 50])
    assert optimizer.num_samples == max(256, 5 * 150)


def test_default_smoothing_is_less_than_one():
    family = FactorizedCategorical(domain_sizes=[3, 3])
    assert 0.4 <= family.smoothing <= 0.9


def test_nonfinite_scores_raise_by_default():
    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(samples.shape[0], dtype=torch.float32)
        out[0] = float("nan")
        return out

    with pytest.raises(ValueError, match="non-finite"):
        cross_entropy_optimize([3, 3], score_fn, max_iters=3, seed=0)


def test_nonfinite_scores_can_be_filtered():
    target_score = _target_problem([2, 2])

    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        out = target_score(samples)
        out[0] = float("nan")
        return out

    result = cross_entropy_optimize(
        [3, 3],
        score_fn,
        max_iters=15,
        seed=0,
        nan_policy="filter",
    )
    assert result.best_score > float("-inf")
    assert math.isfinite(result.best_score)


def test_all_nonfinite_scores_raise_even_with_filter():
    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        return torch.full((samples.shape[0],), float("nan"))

    with pytest.raises(RuntimeError, match="non-finite"):
        cross_entropy_optimize(
            [3, 3],
            score_fn,
            max_iters=3,
            seed=0,
            nan_policy="filter",
        )


def test_quantile_stability_is_a_stop_reason():
    domain_sizes, score_fn, _, _ = _linear_problem()
    result = cross_entropy_optimize(
        domain_sizes=domain_sizes,
        score_fn=score_fn,
        max_iters=50,
        quantile_window=3,
        seed=0,
    )
    assert result.stop_reason in ("quantile_stable", "degenerate")
    assert result.converged is True
    assert result.iterations < 50


def test_max_iters_stop_reason():
    score_fn = _target_problem([1, 1])
    result = cross_entropy_optimize(
        [3, 3],
        score_fn,
        max_iters=2,
        seed=0,
        quantile_window=100,
        convergence_tol=1e-12,
    )
    assert result.stop_reason == "max_iters"
    assert result.converged is False
    assert result.iterations == 2


def test_elitism_preserves_best_ever_monotonicity():
    score_fn = _target_problem([1, 1, 1])
    histories: list[list[float]] = []

    def recorder(history: list[float]):
        def callback(state: IterationState) -> None:
            history.append(state.best_ever)

        return callback

    for elitism in (False, True):
        history: list[float] = []
        histories.append(history)
        cross_entropy_optimize(
            [5, 5, 5],
            score_fn,
            max_iters=15,
            seed=1,
            elitism=elitism,
            callback=recorder(history),
        )

    for history in histories:
        for earlier, later in zip(history, history[1:]):
            assert later >= earlier


def test_dirichlet_prior_replaces_min_prob_clipping():
    family = FactorizedCategorical(
        domain_sizes=[4],
        smoothing=1.0,
        dirichlet_prior=1.0,
    )
    elite_samples = torch.tensor([[0], [0], [0], [0]])
    elite_weights = torch.full((4,), 0.25)
    family.update(elite_samples, elite_weights)
    probs = family.marginals()[0]
    assert probs.min() > 0.0
    assert torch.allclose(probs, torch.tensor([2, 1, 1, 1], dtype=torch.float32) / 5.0)


def test_face_adaptive_grows_population_size():
    pop_sizes: list[int] = []

    def flat_score(samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros(samples.shape[0], dtype=torch.float32)

    def callback(state: IterationState) -> None:
        pop_sizes.append(state.num_samples)

    cross_entropy_optimize(
        domain_sizes=[4, 4],
        score_fn=flat_score,
        num_samples=16,
        num_samples_max=64,
        face_adaptive=True,
        face_stall_iters=2,
        max_iters=10,
        seed=0,
        callback=callback,
    )
    assert max(pop_sizes) > pop_sizes[0]


def test_face_stall_stop_reason():
    def flat_score(samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros(samples.shape[0], dtype=torch.float32)

    result = cross_entropy_optimize(
        domain_sizes=[4, 4],
        score_fn=flat_score,
        num_samples=16,
        num_samples_max=32,
        face_adaptive=True,
        face_stall_iters=2,
        max_iters=50,
        seed=0,
    )
    assert result.stop_reason in ("face_stall", "degenerate", "quantile_stable")


def test_uniform_weighting_sums_to_one():
    weights = UniformWeighting()(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.allclose(weights, torch.full((4,), 0.25))


def test_score_proportional_weighting_favours_higher_scores():
    weights = ScoreProportionalWeighting()(torch.tensor([1.0, 2.0, 5.0]))
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[2] > weights[1] > weights[0]


def test_softmax_weighting_temperature_sharpens():
    scores = torch.tensor([0.0, 1.0, 2.0])
    hot_weights = SoftmaxWeighting(temperature=10.0)(scores)
    cold_weights = SoftmaxWeighting(temperature=0.1)(scores)
    assert cold_weights[-1] > hot_weights[-1]
    assert torch.isclose(hot_weights.sum(), torch.tensor(1.0))
    assert torch.isclose(cold_weights.sum(), torch.tensor(1.0))


def test_softmax_weighting_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        SoftmaxWeighting(temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        SoftmaxWeighting(temperature=-1.0)


def test_optimizer_accepts_softmax_weighting():
    domain_sizes, score_fn, brute_force_x, brute_force_score = _linear_problem()

    result = cross_entropy_optimize(
        domain_sizes=domain_sizes,
        score_fn=score_fn,
        max_iters=40,
        seed=0,
        elite_weighting=SoftmaxWeighting(temperature=1.0),
    )

    assert result.best_x == brute_force_x
    assert result.best_score == pytest.approx(brute_force_score)


def test_built_in_joint_categorical_supports_dependent_sampler():
    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        target = torch.tensor([2, 3])
        return -((samples - target).abs().sum(dim=1).to(torch.float32))

    result = cross_entropy_optimize(
        domain_sizes=[4, 5],
        score_fn=score_fn,
        sampling_family=JointCategorical([4, 5], smoothing=0.8),
        max_iters=30,
        seed=0,
    )
    assert result.best_x == (2, 3)


def test_mode_score_is_evaluated_post_hoc():
    score_fn = _target_problem([2, 2, 2])
    result = cross_entropy_optimize(
        domain_sizes=[3, 3, 3],
        score_fn=score_fn,
        max_iters=20,
        seed=0,
    )
    assert result.mode_score == pytest.approx(0.0)


def test_averaged_score_supports_noisy_objectives():
    target = torch.tensor([2, 3, 2])

    def noisy_score(samples: torch.Tensor) -> torch.Tensor:
        base = -(samples - target).abs().sum(dim=1).to(torch.float32)
        return base + 0.2 * torch.randn(samples.shape[0])

    torch.manual_seed(0)
    result = cross_entropy_optimize(
        domain_sizes=[4, 5, 4],
        score_fn=AveragedScore(noisy_score, num_evals=8),
        max_iters=30,
        seed=0,
    )
    assert result.best_x == (2, 3, 2)


def test_averaged_score_reduces_variance():
    target = torch.tensor([2, 3, 2])

    def noisy_score(samples: torch.Tensor) -> torch.Tensor:
        base = -(samples - target).abs().sum(dim=1).to(torch.float32)
        return base + torch.randn(samples.shape[0])

    samples = torch.tensor([[2, 3, 2]] * 200)
    torch.manual_seed(0)
    deterministic_scores = noisy_score(samples)
    torch.manual_seed(0)
    averaged_scores = AveragedScore(noisy_score, num_evals=16)(samples)

    assert deterministic_scores.shape == averaged_scores.shape == (200,)
    assert averaged_scores.var().item() < deterministic_scores.var().item()


def test_deterministic_score_passthrough():
    domain_sizes, score_fn, brute_force_x, _ = _linear_problem()
    result = cross_entropy_optimize(
        domain_sizes,
        DeterministicScore(score_fn),
        max_iters=40,
        seed=0,
    )
    assert result.best_x == brute_force_x


def test_optimizer_rejects_family_domain_mismatch():
    family = FactorizedCategorical([3, 3])
    with pytest.raises(ValueError, match="domain_sizes"):
        ProductCategoricalCrossEntropy(
            domain_sizes=[4, 4],
            sampling_family=family,
        )


class _OrderedPairFamily:
    def __init__(self, domain_sizes, device=None):
        n1, n2 = domain_sizes
        self.domain_sizes = (int(n1), int(n2))
        self.device = torch.device(device) if device else torch.device("cpu")
        self.joint = torch.full((n1 * n2,), 1.0 / (n1 * n2), device=self.device)
        self._smoothing = 0.7

    def sample(self, num_samples, generator):
        flat = torch.multinomial(
            self.joint,
            num_samples,
            replacement=True,
            generator=generator,
        )
        n1, n2 = self.domain_sizes
        x1 = flat // n2
        x2 = flat % n2
        return torch.stack([x1, x2], dim=1)

    def update(self, elite_samples, elite_weights):
        n1, n2 = self.domain_sizes
        flat = elite_samples[:, 0] * n2 + elite_samples[:, 1]
        counts = torch.bincount(flat, weights=elite_weights, minlength=n1 * n2).to(torch.float32)
        empirical = counts / counts.sum()
        new = self._smoothing * empirical + (1.0 - self._smoothing) * self.joint
        self.joint = new / new.sum()

    def marginals(self):
        n1, n2 = self.domain_sizes
        table = self.joint.view(n1, n2)
        return (table.sum(dim=1), table.sum(dim=0))

    def mode(self):
        _, n2 = self.domain_sizes
        flat = int(torch.argmax(self.joint).item())
        return torch.tensor([flat // n2, flat % n2])

    def is_degenerate(self, tol):
        return (1.0 - self.joint.max().item()) <= tol


def test_custom_coupled_sampling_family_is_accepted():
    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        target = torch.tensor([2, 3])
        match = (samples[:, 0] == target[0]) & (samples[:, 1] == target[1])
        return torch.where(
            match,
            torch.tensor(10.0),
            -((samples - target).abs().sum(dim=1).to(torch.float32)),
        )

    result = cross_entropy_optimize(
        domain_sizes=[4, 5],
        score_fn=score_fn,
        sampling_family=_OrderedPairFamily(domain_sizes=[4, 5]),
        max_iters=30,
        seed=0,
    )

    assert result.best_x == (2, 3)
    assert len(result.probabilities) == 2
    assert result.probabilities[0].shape == (4,)
    assert result.probabilities[1].shape == (5,)


def test_custom_family_rejects_initial_probs_via_wrapper():
    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros(samples.shape[0])

    with pytest.raises(ValueError, match="initial_probs"):
        cross_entropy_optimize(
            domain_sizes=[3, 3],
            score_fn=score_fn,
            sampling_family=_OrderedPairFamily(domain_sizes=[3, 3]),
            initial_probs=[[1 / 3] * 3, [1 / 3] * 3],
        )


def test_callback_invoked_every_iteration_with_consistent_state():
    domain_sizes, score_fn, _, _ = _linear_problem()
    states: list[IterationState] = []

    cross_entropy_optimize(
        domain_sizes=domain_sizes,
        score_fn=score_fn,
        max_iters=5,
        seed=0,
        quantile_window=100,
        convergence_tol=1e-12,
        callback=states.append,
    )

    assert len(states) == 5
    for i, state in enumerate(states, start=1):
        assert state.iteration == i
        assert state.elite_mean <= state.population_best
        assert state.best_ever >= state.population_best or i > 1
        assert len(state.probabilities) == len(domain_sizes)

    for earlier, later in zip(states, states[1:]):
        assert later.best_ever >= earlier.best_ever


def test_result_reports_mode_and_mode_score():
    domain_sizes, score_fn, brute_force_x, brute_force_score = _linear_problem()
    result = cross_entropy_optimize(
        domain_sizes=domain_sizes,
        score_fn=score_fn,
        max_iters=40,
        seed=0,
    )
    assert result.mode_x == brute_force_x
    assert result.mode_score == pytest.approx(brute_force_score)


def test_continuous_mppi_optimizes_quadratic_objective():
    target = torch.tensor([[1.0, -2.0], [0.5, 0.25]])

    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        return -((samples - target) ** 2).sum(dim=(1, 2))

    optimizer = ContinuousGaussianMPPI(
        num_samples=512,
        elite_frac=0.1,
        max_iters=35,
        temperature=0.5,
        smoothing=0.9,
        noise_std=1.0,
        seed=0,
    )

    result = optimizer.optimize(score_fn, torch.zeros_like(target), torch.full_like(target, 2.0))

    assert isinstance(result, ContinuousMPPIResult)
    assert result.best_score > -0.1
    assert torch.allclose(result.best_x, target, atol=0.35)
    assert result.mean.shape == target.shape
    assert result.std.shape == target.shape
    assert result.iterations <= 35


def test_continuous_mppi_filters_nonfinite_scores_and_reports_diagnostics():
    states: list[ContinuousIterationState] = []
    target = torch.tensor([0.75, -0.25])

    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        scores = -((samples - target) ** 2).sum(dim=1)
        scores[0] = float("nan")
        return scores

    optimizer = ContinuousGaussianMPPI(
        num_samples=64,
        elite_frac=0.2,
        max_iters=5,
        nan_policy="filter",
        quantile_window=100,
        callback=states.append,
        seed=1,
    )

    result = optimizer.optimize(score_fn, torch.zeros_like(target))

    assert math.isfinite(result.best_score)
    assert len(states) == result.iterations == 5
    assert all(state.mean.shape == target.shape and state.std.shape == target.shape for state in states)
    assert all(later.best_ever >= earlier.best_ever for earlier, later in zip(states, states[1:]))


def test_continuous_mppi_face_adaptive_grows_population_size():
    pop_sizes: list[int] = []

    def flat_score(samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros(samples.shape[0], dtype=torch.float32)

    optimizer = ContinuousGaussianMPPI(
        num_samples=8,
        num_samples_max=32,
        face_adaptive=True,
        face_stall_iters=2,
        max_iters=8,
        quantile_window=100,
        callback=lambda state: pop_sizes.append(state.num_samples),
        seed=0,
    )

    result = optimizer.optimize(flat_score, torch.zeros(2))

    assert max(pop_sizes) > pop_sizes[0]
    assert result.stop_reason in ("face_stall", "degenerate", "quantile_stable", "max_iters")


def test_continuous_mppi_rejects_bad_score_shapes_and_nonfinite_by_default():
    optimizer = ContinuousGaussianMPPI(max_iters=2, seed=0)

    with pytest.raises(ValueError, match="1-D tensor"):
        optimizer.optimize(lambda samples: torch.zeros(samples.shape[0], 1), torch.zeros(2))

    def nan_score(samples: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(samples.shape[0])
        scores[0] = float("nan")
        return scores

    with pytest.raises(ValueError, match="non-finite"):
        optimizer.optimize(nan_score, torch.zeros(2))


def test_continuous_gmm_mppi_drifts_seeded_components_to_quadratic_target():
    target = torch.tensor([[1.0, -1.0], [0.25, 0.75]])
    initial_means = torch.tensor(
        [
            [[0.8, -0.8], [-2.0, 2.0]],
            [[0.0, 0.5], [2.0, -2.0]],
        ],
        dtype=torch.float32,
    )

    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        return -((samples - target) ** 2).sum(dim=(1, 2))

    optimizer = ContinuousGMMMPPI(
        num_samples=256,
        elite_frac=0.15,
        max_iters=25,
        temperature=0.5,
        smoothing=0.8,
        noise_std=0.5,
        seed=0,
    )
    result = optimizer.optimize(score_fn, initial_means, torch.full_like(initial_means, 0.8))

    assert isinstance(result, ContinuousGMMMPPIResult)
    assert result.best_score > -0.2
    assert torch.allclose(result.best_x, target, atol=0.35)
    assert result.means.shape == initial_means.shape
    assert result.component_probs.shape == initial_means.shape[:2]
    assert torch.allclose(result.component_probs.sum(dim=1), torch.ones(2), atol=1e-6)


class _TinyStructuredFamily:
    def __init__(self):
        self.probs = torch.full((2, 3), 1.0 / 3.0)

    def sample(self, num_samples, generator):
        steps = [
            torch.multinomial(self.probs[step], num_samples, replacement=True, generator=generator) for step in range(2)
        ]
        return torch.stack(steps, dim=1).unsqueeze(-1)

    def update(self, elite_samples, elite_weights):
        flat = elite_samples.squeeze(-1)
        for step in range(2):
            counts = torch.bincount(flat[:, step], weights=elite_weights, minlength=3).to(torch.float32)
            posterior = counts / counts.sum().clamp_min(1e-12)
            self.probs[step] = 0.8 * posterior + 0.2 * self.probs[step]
            self.probs[step] = self.probs[step] / self.probs[step].sum()

    def marginals(self):
        return (self.probs[0], self.probs[1])

    def mode(self):
        return torch.argmax(self.probs, dim=1).unsqueeze(-1)

    def is_degenerate(self, tol):
        return bool(torch.all((1.0 - self.probs.max(dim=1).values) <= tol).item())


def test_structured_cross_entropy_keeps_non_flat_sample_shape():
    target = torch.tensor([2, 1])

    def score_fn(samples: torch.Tensor) -> torch.Tensor:
        flat = samples.squeeze(-1)
        return -(flat - target).abs().sum(dim=1).to(torch.float32)

    result = StructuredCrossEntropy(
        sampling_family=_TinyStructuredFamily(),
        num_samples=128,
        max_iters=30,
        seed=0,
    ).optimize(score_fn)

    assert isinstance(result, StructuredCEResult)
    assert result.best_x.shape == (2, 1)
    assert tuple(result.best_x.squeeze(-1).tolist()) == (2, 1)
