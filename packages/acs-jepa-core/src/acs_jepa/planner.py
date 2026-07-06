"""Receding-horizon latent MPPI planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from acs_jepa.architectures import (
    ActionDecodingSpace,
    ActionSamplingFamily,
    JEPALatentState,
    StructuredActionSequenceSamplingFamily,
)
from acs_jepa.graph import GroundAction, GroundAtom, ParsedProblem, build_state_graph
from acs_jepa.graph.encoders import GraphEncoderOutput
from acs_jepa.mpc import (
    ContinuousGaussianMPPI,
    ContinuousGMMMPPI,
    ContinuousGMMMPPIResult,
    ContinuousMPPIResult,
    SoftmaxWeighting,
    StructuredCEResult,
    StructuredCrossEntropy,
)

GoalEnergy = Callable[[dict[str, Tensor], JEPALatentState], Tensor]
RejectionPenalty = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class LatentRolloutOutput:
    """Planner-side autoregressive rollout result."""

    initial_state: JEPALatentState
    final_state: JEPALatentState
    predicted_states: tuple[JEPALatentState, ...]
    action_latents: Tensor


@dataclass(frozen=True)
class LatentMPPIConfig:
    """Configuration for latent-action MPPI planning."""

    horizon: int
    action_dim: int
    num_samples: int = 256
    elite_frac: float = 0.1
    max_iters: int = 50
    temperature: float = 1.0
    smoothing: float = 0.7
    noise_std: float = 1.0
    convergence_tol: float = 1e-4
    quantile_window: int = 5
    elitism: bool = True
    face_adaptive: bool = False
    num_samples_max: int | None = None
    face_stall_iters: int = 3
    nan_policy: str = "raise"
    apply_steps: int = 1
    max_total_actions: int = 64
    constant_action_cost: float = 0.0
    max_decode_attempts: int = 3
    invalid_action_penalty: float = 1_000.0
    initial_std: float = 1.0
    device: torch.device | str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class LatentMPPIResult:
    """One latent MPPI planning result for a single simulator state."""

    action_latents: Tensor
    optimizer_result: ContinuousMPPIResult
    rollout: LatentRolloutOutput
    graph_output: GraphEncoderOutput
    initial_state: JEPALatentState
    constant_action_cost: float


@dataclass(frozen=True)
class LatentGMMMPPIConfig(LatentMPPIConfig):
    """Configuration for GMM-seeded latent MPPI.

    The optimizer keeps a drifting component bank of shape ``[H, M, D_a]``,
    where ``M == action_pool_size`` and components are initialized from encoded
    grounded actions sampled in the current latent state.
    """

    action_pool_size: int = 512
    component_std: float | None = None
    min_std: float = 1e-6
    max_std: float | None = None
    dirichlet_prior: float = 0.0


@dataclass(frozen=True)
class LatentGMMMPPIResult:
    """One GMM-seeded latent MPPI result for a single simulator state."""

    action_latents: Tensor
    optimizer_result: ContinuousGMMMPPIResult
    rollout: LatentRolloutOutput
    graph_output: GraphEncoderOutput
    initial_state: JEPALatentState
    action_pool: tuple[GroundAction, ...]
    constant_action_cost: float


@dataclass(frozen=True)
class StructuredCEPlannerConfig:
    """Configuration for CE over typed grounded action sequences."""

    horizon: int
    num_samples: int = 256
    elite_frac: float = 0.1
    max_iters: int = 50
    temperature: float | None = None
    smoothing: float = 0.7
    dirichlet_prior: float = 0.0
    convergence_tol: float = 1e-4
    quantile_window: int = 5
    elitism: bool = True
    face_adaptive: bool = False
    num_samples_max: int | None = None
    face_stall_iters: int = 3
    nan_policy: str = "raise"
    apply_steps: int = 1
    max_total_actions: int = 64
    constant_action_cost: float = 0.0
    max_decode_attempts: int = 3
    invalid_action_penalty: float = 1_000.0
    device: torch.device | str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class StructuredCEPlanResult:
    """One structured CE planning result for a single simulator state."""

    actions: tuple[GroundAction, ...]
    action_samples: Tensor
    optimizer_result: StructuredCEResult
    rollout: LatentRolloutOutput
    graph_output: GraphEncoderOutput
    initial_state: JEPALatentState
    constant_action_cost: float


@dataclass(frozen=True)
class PlannerAppliedAction:
    """A grounded action applied by ``PlannerAgent``."""

    name: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class PlannerRunResult:
    """Summary of a receding-horizon simulator run."""

    success: bool
    applied_actions: tuple[PlannerAppliedAction, ...]
    total_actions: int
    attempts: int
    failure_reason: str | None = None


class LatentMPPIPlanner:
    """Optimize continuous action-latent sequences against a goal energy."""

    def __init__(
        self,
        *,
        graph_encoder: nn.Module,
        state_encoder: nn.Module,
        predictor: nn.Module,
        goal_energy: GoalEnergy,
        config: LatentMPPIConfig,
    ) -> None:
        if config.horizon < 1:
            raise ValueError("horizon must be positive")
        if config.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if config.apply_steps < 1:
            raise ValueError("apply_steps must be positive")
        if config.max_total_actions < 1:
            raise ValueError("max_total_actions must be positive")
        if config.max_decode_attempts < 1:
            raise ValueError("max_decode_attempts must be positive")
        if config.initial_std <= 0:
            raise ValueError("initial_std must be positive")
        self.graph_encoder = graph_encoder
        self.state_encoder = state_encoder
        self.predictor = predictor
        self.goal_energy = goal_energy
        self.config = config
        self.device = torch.device(config.device) if config.device is not None else torch.device("cpu")

    def encode_graph(self, state_graph: Data) -> tuple[GraphEncoderOutput, JEPALatentState]:
        """Encode one current state graph for planning."""

        graph = state_graph.to(self.device)
        graph_output = self.graph_encoder(graph)
        latent_state = self.state_encoder(graph_output)
        if latent_state.graph_latent.ndim != 2 or latent_state.graph_latent.size(0) != 1:
            raise ValueError("LatentMPPIPlanner expects one current graph at a time")
        return graph_output, latent_state

    def rollout_from_state(self, initial_state: JEPALatentState, action_latents: Tensor) -> LatentRolloutOutput:
        """Roll out ``[N, H, D_a]`` action latents from one encoded state."""

        action_latents = action_latents.to(self.device, dtype=initial_state.graph_latent.dtype)
        if action_latents.ndim != 3:
            raise ValueError(f"Expected action_latents shape [N, H, D_a], got {tuple(action_latents.shape)}")
        if action_latents.size(1) != self.config.horizon or action_latents.size(2) != self.config.action_dim:
            raise ValueError(
                "action_latents must match planner horizon/action_dim: "
                f"got {tuple(action_latents.shape)}, expected [N, {self.config.horizon}, {self.config.action_dim}]"
            )
        state = _repeat_latent_state_for_samples(initial_state, action_latents.size(0), device=self.device)
        predicted = []
        for step_idx in range(action_latents.size(1)):
            state = self.predictor(state, action_latents[:, step_idx])
            predicted.append(state)
        return LatentRolloutOutput(
            initial_state=initial_state,
            final_state=state,
            predicted_states=tuple(predicted),
            action_latents=action_latents,
        )

    def plan(
        self,
        state_graph: Data,
        goal_tensors: dict[str, Tensor],
        *,
        initial_mean: Tensor | None = None,
        initial_std: Tensor | None = None,
        rejection_penalty: RejectionPenalty | None = None,
    ) -> LatentMPPIResult:
        """Optimize a latent action sequence for the current state graph."""

        graph_output, initial_state = self.encode_graph(state_graph)
        mean = self._initial_mean(initial_mean)
        std = self._initial_std(initial_std)
        goal_tensors = _move_tensor_dict(goal_tensors, self.device)
        optimizer = self._build_optimizer()
        constant_cost = float(self.config.constant_action_cost) * float(self.config.horizon)

        def score_fn(samples: Tensor) -> Tensor:
            with torch.no_grad():
                rollout = self.rollout_from_state(initial_state, samples)
                energy = self.goal_energy(goal_tensors, rollout.final_state).to(self.device, dtype=torch.float32)
                if energy.ndim != 1 or energy.size(0) != samples.size(0):
                    raise ValueError(
                        f"goal_energy must return shape [N], got {tuple(energy.shape)} for N={samples.size(0)}"
                    )
                score = -energy - constant_cost
                if rejection_penalty is not None:
                    score = score - rejection_penalty(samples).to(self.device, dtype=torch.float32)
                return score

        result = optimizer.optimize(score_fn, mean, std)
        best_action_latents = result.best_x.to(self.device)
        rollout = self.rollout_from_state(initial_state, best_action_latents.unsqueeze(0))
        return LatentMPPIResult(
            action_latents=best_action_latents.detach().cpu(),
            optimizer_result=result,
            rollout=rollout,
            graph_output=graph_output,
            initial_state=initial_state,
            constant_action_cost=constant_cost,
        )

    def _initial_mean(self, value: Tensor | None) -> Tensor:
        if value is None:
            return torch.zeros((self.config.horizon, self.config.action_dim), dtype=torch.float32, device=self.device)
        mean = value.to(device=self.device, dtype=torch.float32)
        if mean.shape != (self.config.horizon, self.config.action_dim):
            raise ValueError(f"initial_mean must have shape {(self.config.horizon, self.config.action_dim)}")
        return mean

    def _initial_std(self, value: Tensor | None) -> Tensor:
        if value is None:
            return torch.full(
                (self.config.horizon, self.config.action_dim),
                float(self.config.initial_std),
                dtype=torch.float32,
                device=self.device,
            )
        std = value.to(device=self.device, dtype=torch.float32)
        if std.shape != (self.config.horizon, self.config.action_dim):
            raise ValueError(f"initial_std must have shape {(self.config.horizon, self.config.action_dim)}")
        return std

    def _build_optimizer(self) -> ContinuousGaussianMPPI:
        return ContinuousGaussianMPPI(
            num_samples=self.config.num_samples,
            elite_frac=self.config.elite_frac,
            max_iters=self.config.max_iters,
            temperature=self.config.temperature,
            smoothing=self.config.smoothing,
            noise_std=self.config.noise_std,
            convergence_tol=self.config.convergence_tol,
            quantile_window=self.config.quantile_window,
            elitism=self.config.elitism,
            face_adaptive=self.config.face_adaptive,
            num_samples_max=self.config.num_samples_max,
            face_stall_iters=self.config.face_stall_iters,
            nan_policy=self.config.nan_policy,
            seed=self.config.seed,
            device=self.device,
        )


class LatentGMMMPPIPlanner:
    """Optimize latent sequences from a GMM seeded by real encoded actions."""

    def __init__(
        self,
        *,
        graph_encoder: nn.Module,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        goal_energy: GoalEnergy,
        action_space: ActionDecodingSpace,
        config: LatentGMMMPPIConfig,
    ) -> None:
        if config.action_pool_size < 1:
            raise ValueError("action_pool_size must be positive")
        if config.component_std is not None and config.component_std <= 0:
            raise ValueError("component_std must be positive when provided")
        self.graph_encoder = graph_encoder
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.goal_energy = goal_energy
        self.action_space = action_space
        self.config = config
        self.device = torch.device(config.device) if config.device is not None else torch.device("cpu")

    def encode_graph(self, state_graph: Data) -> tuple[GraphEncoderOutput, JEPALatentState]:
        """Encode one current state graph for planning."""

        graph = state_graph.to(self.device)
        graph_output = self.graph_encoder(graph)
        latent_state = self.state_encoder(graph_output)
        if latent_state.graph_latent.ndim != 2 or latent_state.graph_latent.size(0) != 1:
            raise ValueError("LatentGMMMPPIPlanner expects one current graph at a time")
        return graph_output, latent_state

    def rollout_from_state(self, initial_state: JEPALatentState, action_latents: Tensor) -> LatentRolloutOutput:
        """Roll out ``[N, H, D_a]`` action latents from one encoded state."""

        action_latents = action_latents.to(self.device, dtype=initial_state.graph_latent.dtype)
        if action_latents.ndim != 3:
            raise ValueError(f"Expected action_latents shape [N, H, D_a], got {tuple(action_latents.shape)}")
        if action_latents.size(1) != self.config.horizon or action_latents.size(2) != self.config.action_dim:
            raise ValueError(
                "action_latents must match planner horizon/action_dim: "
                f"got {tuple(action_latents.shape)}, expected [N, {self.config.horizon}, {self.config.action_dim}]"
            )
        state = _repeat_latent_state_for_samples(initial_state, action_latents.size(0), device=self.device)
        predicted = []
        for step_idx in range(action_latents.size(1)):
            state = self.predictor(state, action_latents[:, step_idx])
            predicted.append(state)
        return LatentRolloutOutput(
            initial_state=initial_state,
            final_state=state,
            predicted_states=tuple(predicted),
            action_latents=action_latents,
        )

    def plan(
        self,
        state_graph: Data,
        goal_tensors: dict[str, Tensor],
        *,
        initial_mean: Tensor | None = None,
        initial_std: Tensor | None = None,
        initial_component_probs: Tensor | None = None,
        rejection_penalty: RejectionPenalty | None = None,
    ) -> LatentGMMMPPIResult:
        """Optimize a latent action sequence from subset-seeded GMM components."""

        graph_output, initial_state = self.encode_graph(state_graph)
        generator = torch.Generator(device=self.device)
        if self.config.seed is not None:
            generator.manual_seed(self.config.seed)
        family = ActionSamplingFamily(
            self.action_space,
            smoothing=self.config.smoothing,
            dirichlet_prior=self.config.dirichlet_prior,
            device=self.device,
        )
        pool_samples = family.sample(self.config.action_pool_size, generator)
        action_pool = tuple(self.action_space.sample_to_ground_action(sample) for sample in pool_samples)
        action_tensors = self.action_space.samples_to_action_tensors(pool_samples, device=self.device)
        with torch.no_grad():
            repeated_state = _repeat_latent_state_for_samples(
                initial_state,
                pool_samples.size(0),
                device=self.device,
            )
            seed_latents = self.action_encoder(action_tensors, repeated_state)
        if seed_latents.ndim != 2 or seed_latents.size(1) != self.config.action_dim:
            raise ValueError(
                f"action_encoder must return [M, {self.config.action_dim}], got {tuple(seed_latents.shape)}"
            )

        seeded_means = seed_latents.unsqueeze(0).expand(self.config.horizon, -1, -1).contiguous()
        means = seeded_means if initial_mean is None else initial_mean.to(self.device, dtype=torch.float32)
        if means.shape != seeded_means.shape:
            raise ValueError(f"initial_mean must have shape {tuple(seeded_means.shape)}, got {tuple(means.shape)}")
        std_value = self.config.initial_std if self.config.component_std is None else self.config.component_std
        default_stds = torch.full_like(seeded_means, float(std_value))
        stds = default_stds if initial_std is None else initial_std.to(self.device, dtype=torch.float32)
        if stds.shape != seeded_means.shape:
            raise ValueError(f"initial_std must have shape {tuple(seeded_means.shape)}, got {tuple(stds.shape)}")

        goal_tensors = _move_tensor_dict(goal_tensors, self.device)
        optimizer = ContinuousGMMMPPI(
            num_samples=self.config.num_samples,
            elite_frac=self.config.elite_frac,
            max_iters=self.config.max_iters,
            temperature=self.config.temperature,
            smoothing=self.config.smoothing,
            noise_std=self.config.noise_std,
            min_std=self.config.min_std,
            max_std=self.config.max_std,
            convergence_tol=self.config.convergence_tol,
            quantile_window=self.config.quantile_window,
            elitism=self.config.elitism,
            face_adaptive=self.config.face_adaptive,
            num_samples_max=self.config.num_samples_max,
            face_stall_iters=self.config.face_stall_iters,
            nan_policy=self.config.nan_policy,
            seed=self.config.seed,
            device=self.device,
        )
        constant_cost = float(self.config.constant_action_cost) * float(self.config.horizon)

        def score_fn(samples: Tensor) -> Tensor:
            with torch.no_grad():
                rollout = self.rollout_from_state(initial_state, samples)
                energy = self.goal_energy(goal_tensors, rollout.final_state).to(self.device, dtype=torch.float32)
                if energy.ndim != 1 or energy.size(0) != samples.size(0):
                    raise ValueError(
                        f"goal_energy must return shape [N], got {tuple(energy.shape)} for N={samples.size(0)}"
                    )
                score = -energy - constant_cost
                if rejection_penalty is not None:
                    score = score - rejection_penalty(samples).to(self.device, dtype=torch.float32)
                return score

        result = optimizer.optimize(score_fn, means, stds, initial_component_probs)
        best_action_latents = result.best_x.to(self.device)
        rollout = self.rollout_from_state(initial_state, best_action_latents.unsqueeze(0))
        return LatentGMMMPPIResult(
            action_latents=best_action_latents.detach().cpu(),
            optimizer_result=result,
            rollout=rollout,
            graph_output=graph_output,
            initial_state=initial_state,
            action_pool=action_pool,
            constant_action_cost=constant_cost,
        )


class StructuredCEPlanner:
    """Plan by optimizing typed grounded action sequences directly."""

    def __init__(
        self,
        *,
        graph_encoder: nn.Module,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        goal_energy: GoalEnergy,
        action_space: ActionDecodingSpace,
        config: StructuredCEPlannerConfig,
    ) -> None:
        if config.horizon < 1:
            raise ValueError("horizon must be positive")
        if config.apply_steps < 1:
            raise ValueError("apply_steps must be positive")
        if config.max_total_actions < 1:
            raise ValueError("max_total_actions must be positive")
        if config.max_decode_attempts < 1:
            raise ValueError("max_decode_attempts must be positive")
        self.graph_encoder = graph_encoder
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.goal_energy = goal_energy
        self.action_space = action_space
        self.config = config
        self.device = torch.device(config.device) if config.device is not None else torch.device("cpu")

    def encode_graph(self, state_graph: Data) -> tuple[GraphEncoderOutput, JEPALatentState]:
        """Encode one current state graph for planning."""

        graph = state_graph.to(self.device)
        graph_output = self.graph_encoder(graph)
        latent_state = self.state_encoder(graph_output)
        if latent_state.graph_latent.ndim != 2 or latent_state.graph_latent.size(0) != 1:
            raise ValueError("StructuredCEPlanner expects one current graph at a time")
        return graph_output, latent_state

    def plan(
        self,
        state_graph: Data,
        goal_tensors: dict[str, Tensor],
        *,
        rejected_actions: set[tuple[str, tuple[str, ...]]] | None = None,
    ) -> StructuredCEPlanResult:
        """Optimize a compact grounded action sequence for the current state."""

        graph_output, initial_state = self.encode_graph(state_graph)
        goal_tensors = _move_tensor_dict(goal_tensors, self.device)
        family = StructuredActionSequenceSamplingFamily(
            self.action_space,
            self.config.horizon,
            smoothing=self.config.smoothing,
            dirichlet_prior=self.config.dirichlet_prior,
            device=self.device,
        )
        weighting = SoftmaxWeighting(self.config.temperature) if self.config.temperature is not None else None
        optimizer = StructuredCrossEntropy(
            sampling_family=family,
            num_samples=self.config.num_samples,
            elite_frac=self.config.elite_frac,
            max_iters=self.config.max_iters,
            elite_weighting=weighting,
            nan_policy=self.config.nan_policy,
            quantile_window=self.config.quantile_window,
            convergence_tol=self.config.convergence_tol,
            elitism=self.config.elitism,
            face_adaptive=self.config.face_adaptive,
            num_samples_max=self.config.num_samples_max,
            face_stall_iters=self.config.face_stall_iters,
            seed=self.config.seed,
            device=self.device,
        )
        rejected_actions = rejected_actions or set()
        constant_cost = float(self.config.constant_action_cost) * float(self.config.horizon)

        def score_fn(samples: Tensor) -> Tensor:
            with torch.no_grad():
                state = _repeat_latent_state_for_samples(initial_state, samples.size(0), device=self.device)
                for step_idx in range(samples.size(1)):
                    action_tensors = self.action_space.samples_to_action_tensors(
                        samples[:, step_idx], device=self.device
                    )
                    action_latent = self.action_encoder(action_tensors, state)
                    state = self.predictor(state, action_latent)
                energy = self.goal_energy(goal_tensors, state).to(self.device, dtype=torch.float32)
                score = -energy - constant_cost
                if rejected_actions:
                    penalty = torch.zeros((samples.size(0),), dtype=torch.float32, device=self.device)
                    for sample_idx in range(samples.size(0)):
                        action = self.action_space.sample_to_ground_action(samples[sample_idx, 0])
                        if (action.name, action.arguments) in rejected_actions:
                            penalty[sample_idx] = float(self.config.invalid_action_penalty)
                    score = score - penalty
                return score

        result = optimizer.optimize(score_fn)
        best_samples = result.best_x.to(self.device)
        actions = tuple(
            self.action_space.sample_to_ground_action(best_samples[step_idx])
            for step_idx in range(best_samples.size(0))
        )
        state = initial_state
        predicted = []
        action_latents = []
        for step_idx in range(best_samples.size(0)):
            action_tensors = self.action_space.samples_to_action_tensors(best_samples[step_idx], device=self.device)
            action_latent = self.action_encoder(action_tensors, state)
            action_latents.append(action_latent.squeeze(0))
            state = self.predictor(state, action_latent)
            predicted.append(state)
        rollout = LatentRolloutOutput(
            initial_state=initial_state,
            final_state=state,
            predicted_states=tuple(predicted),
            action_latents=torch.stack(action_latents, dim=0).unsqueeze(0),
        )
        return StructuredCEPlanResult(
            actions=actions,
            action_samples=best_samples.detach().cpu(),
            optimizer_result=result,
            rollout=rollout,
            graph_output=graph_output,
            initial_state=initial_state,
            constant_action_cost=constant_cost,
        )


class PlannerAgent:
    """Receding-horizon simulator agent using latent MPPI and action decoding."""

    def __init__(
        self,
        *,
        planner: LatentMPPIPlanner | LatentGMMMPPIPlanner,
        action_decoder,
        parsed_problem: ParsedProblem,
        include_static: bool = True,
    ) -> None:
        self.planner = planner
        self.action_decoder = action_decoder
        self.parsed_problem = parsed_problem
        self.include_static = include_static

    def run(self, engine, goal_tensors: dict[str, Tensor]) -> PlannerRunResult:
        """Run receding-horizon planning against a simulator engine."""

        applied_actions: list[PlannerAppliedAction] = []
        total_attempts = 0
        prior_mean: Tensor | None = None
        prior_std: Tensor | None = None
        prior_component_probs: Tensor | None = None

        while not engine.goals_satisfied() and len(applied_actions) < self.planner.config.max_total_actions:
            rejected: set[tuple[str, tuple[str, ...]]] = set()
            applied_this_cycle = 0

            for _ in range(self.planner.config.max_decode_attempts):
                total_attempts += 1
                state_graph = self._state_graph(engine)
                latent_state = self._action_latent_state(state_graph)
                if isinstance(self.planner, LatentGMMMPPIPlanner):
                    plan = self.planner.plan(
                        state_graph,
                        goal_tensors,
                        initial_mean=prior_mean,
                        initial_std=prior_std,
                        initial_component_probs=prior_component_probs,
                        rejection_penalty=self._rejection_penalty(latent_state, rejected) if rejected else None,
                    )
                    prior_mean = plan.optimizer_result.means
                    prior_std = plan.optimizer_result.stds
                    prior_component_probs = plan.optimizer_result.component_probs
                else:
                    plan = self.planner.plan(
                        state_graph,
                        goal_tensors,
                        initial_mean=prior_mean,
                        initial_std=prior_std,
                        rejection_penalty=self._rejection_penalty(latent_state, rejected) if rejected else None,
                    )
                    prior_mean = plan.optimizer_result.mean
                    prior_std = plan.optimizer_result.std
                    prior_component_probs = None

                remaining_actions = self.planner.config.max_total_actions - len(applied_actions)
                applied_this_attempt, invalid = self._apply_valid_prefix(
                    engine,
                    plan.action_latents,
                    max_actions=remaining_actions,
                )
                applied_actions.extend(applied_this_attempt)
                applied_this_cycle += len(applied_this_attempt)
                if applied_this_attempt:
                    prior_mean, prior_std, prior_component_probs = self._shift_prior(
                        prior_mean,
                        prior_std,
                        len(applied_this_attempt),
                        component_probs=prior_component_probs,
                    )
                    break
                if invalid is None:
                    return PlannerRunResult(
                        success=engine.goals_satisfied(),
                        applied_actions=tuple(applied_actions),
                        total_actions=len(applied_actions),
                        attempts=total_attempts,
                        failure_reason="no_decoded_action",
                    )
                rejected.add((invalid.name, invalid.arguments))

            if engine.goals_satisfied():
                break
            if applied_this_cycle == 0:
                return PlannerRunResult(
                    success=False,
                    applied_actions=tuple(applied_actions),
                    total_actions=len(applied_actions),
                    attempts=total_attempts,
                    failure_reason="decode_invalid",
                )

        success = engine.goals_satisfied()
        return PlannerRunResult(
            success=success,
            applied_actions=tuple(applied_actions),
            total_actions=len(applied_actions),
            attempts=total_attempts,
            failure_reason=None if success else "max_total_actions",
        )

    def _apply_valid_prefix(
        self,
        engine,
        action_latents: Tensor,
        *,
        max_actions: int,
    ) -> tuple[list[PlannerAppliedAction], GroundAction | None]:
        applied: list[PlannerAppliedAction] = []
        max_steps = min(self.planner.config.apply_steps, action_latents.size(0), max_actions)
        for step_idx in range(max_steps):
            latent_state = self._action_latent_state(self._state_graph(engine))
            action = self.action_decoder.decode(action_latents[step_idx].to(self.planner.device), latent_state)
            try:
                engine.apply_action(action.name, action.arguments, finish=True)
            except ValueError:
                return applied, action
            applied.append(PlannerAppliedAction(name=action.name, arguments=action.arguments))
            if engine.goals_satisfied():
                break
        return applied, None

    def _rejection_penalty(
        self,
        latent_state: JEPALatentState,
        rejected: set[tuple[str, tuple[str, ...]]],
    ) -> RejectionPenalty:
        def penalty(samples: Tensor) -> Tensor:
            penalties = torch.zeros((samples.size(0),), dtype=torch.float32, device=samples.device)
            for sample_idx in range(samples.size(0)):
                action = self.action_decoder.decode(samples[sample_idx, 0], latent_state)
                if (action.name, action.arguments) in rejected:
                    penalties[sample_idx] = float(self.planner.config.invalid_action_penalty)
            return penalties

        return penalty

    def _shift_prior(
        self,
        mean: Tensor,
        std: Tensor,
        steps: int,
        *,
        component_probs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        steps = min(int(steps), self.planner.config.horizon)
        shifted_mean = torch.zeros_like(mean)
        shifted_std = torch.full_like(std, float(self.planner.config.initial_std))
        shifted_component_probs = (
            None
            if component_probs is None
            else torch.full_like(
                component_probs,
                1.0 / component_probs.size(-1),
            )
        )
        if steps < self.planner.config.horizon:
            shifted_mean[:-steps] = mean[steps:]
            shifted_std[:-steps] = std[steps:]
            if shifted_component_probs is not None:
                shifted_component_probs[:-steps] = component_probs[steps:]
        return shifted_mean, shifted_std, shifted_component_probs

    def _state_graph(self, engine) -> Data:
        atoms = [GroundAtom(predicate=item[0], arguments=tuple(item[1:])) for item in engine.current_facts()]
        return build_state_graph(self.parsed_problem, atoms, include_static=self.include_static)

    def _action_latent_state(self, state_graph: Data) -> JEPALatentState:
        _, latent_state = self.planner.encode_graph(state_graph)
        return latent_state


class GroundedPlannerAgent:
    """Receding-horizon simulator agent for planners returning grounded actions."""

    def __init__(
        self,
        *,
        planner: StructuredCEPlanner,
        parsed_problem: ParsedProblem,
        include_static: bool = True,
    ) -> None:
        self.planner = planner
        self.parsed_problem = parsed_problem
        self.include_static = include_static

    def run(self, engine, goal_tensors: dict[str, Tensor]) -> PlannerRunResult:
        """Run structured CE planning against a simulator engine."""

        applied_actions: list[PlannerAppliedAction] = []
        total_attempts = 0

        while not engine.goals_satisfied() and len(applied_actions) < self.planner.config.max_total_actions:
            rejected: set[tuple[str, tuple[str, ...]]] = set()
            applied_this_cycle = 0
            for _ in range(self.planner.config.max_decode_attempts):
                total_attempts += 1
                plan = self.planner.plan(
                    self._state_graph(engine),
                    goal_tensors,
                    rejected_actions=rejected,
                )
                remaining_actions = self.planner.config.max_total_actions - len(applied_actions)
                applied_this_attempt, invalid = self._apply_valid_prefix(
                    engine,
                    plan.actions,
                    max_actions=remaining_actions,
                )
                applied_actions.extend(applied_this_attempt)
                applied_this_cycle += len(applied_this_attempt)
                if applied_this_attempt:
                    break
                if invalid is None:
                    return PlannerRunResult(
                        success=engine.goals_satisfied(),
                        applied_actions=tuple(applied_actions),
                        total_actions=len(applied_actions),
                        attempts=total_attempts,
                        failure_reason="no_decoded_action",
                    )
                rejected.add((invalid.name, invalid.arguments))

            if engine.goals_satisfied():
                break
            if applied_this_cycle == 0:
                return PlannerRunResult(
                    success=False,
                    applied_actions=tuple(applied_actions),
                    total_actions=len(applied_actions),
                    attempts=total_attempts,
                    failure_reason="decode_invalid",
                )

        success = engine.goals_satisfied()
        return PlannerRunResult(
            success=success,
            applied_actions=tuple(applied_actions),
            total_actions=len(applied_actions),
            attempts=total_attempts,
            failure_reason=None if success else "max_total_actions",
        )

    def _apply_valid_prefix(
        self,
        engine,
        actions: tuple[GroundAction, ...],
        *,
        max_actions: int,
    ) -> tuple[list[PlannerAppliedAction], GroundAction | None]:
        applied: list[PlannerAppliedAction] = []
        max_steps = min(self.planner.config.apply_steps, len(actions), max_actions)
        for step_idx in range(max_steps):
            action = actions[step_idx]
            try:
                engine.apply_action(action.name, action.arguments, finish=True)
            except ValueError:
                return applied, action
            applied.append(PlannerAppliedAction(name=action.name, arguments=action.arguments))
            if engine.goals_satisfied():
                break
        return applied, None

    def _state_graph(self, engine) -> Data:
        atoms = [GroundAtom(predicate=item[0], arguments=tuple(item[1:])) for item in engine.current_facts()]
        return build_state_graph(self.parsed_problem, atoms, include_static=self.include_static)


def _repeat_latent_state_for_samples(
    latent_state: JEPALatentState,
    num_samples: int,
    *,
    device: torch.device | str,
) -> JEPALatentState:
    if latent_state.graph_latent.ndim != 2 or latent_state.graph_latent.size(0) != 1:
        raise ValueError("Expected one non-temporal latent state to repeat for planning")
    object_count = latent_state.object_ids.numel()
    return JEPALatentState(
        graph_latent=latent_state.graph_latent.to(device).expand(num_samples, -1),
        object_latents=latent_state.object_latents.to(device).repeat(num_samples, 1),
        object_ids=latent_state.object_ids.to(device).repeat(num_samples),
        object_batch=torch.arange(num_samples, device=device).repeat_interleave(object_count),
    )


def _move_tensor_dict(values: dict[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in values.items()}
