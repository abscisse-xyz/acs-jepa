from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from acs_jepa.architectures import ActionDecodingSpace, JEPALatentState
from acs_jepa.graph.encoders import GraphEncoderOutput
from acs_jepa.graph.schemas import ActionSchema, GroundAction, ObjectInfo, ParsedProblem, PredicateSchema
from acs_jepa.mpc import ContinuousMPPIResult
from acs_jepa.planner import (
    GroundedPlannerAgent,
    LatentGMMMPPIConfig,
    LatentGMMMPPIPlanner,
    LatentMPPIConfig,
    LatentMPPIPlanner,
    LatentMPPIResult,
    LatentRolloutOutput,
    PlannerAgent,
    StructuredCEPlanner,
    StructuredCEPlannerConfig,
)
from torch_geometric.data import Data


def test_latent_planner_rollout_matches_repeated_single_step() -> None:
    planner = _tiny_latent_planner()
    _, initial_state = planner.encode_graph(Data())
    actions = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.25]],
            [[-1.0, 0.5], [0.25, -0.75]],
        ]
    )

    batched = planner.rollout_from_state(initial_state, actions)
    first = planner.rollout_from_state(initial_state, actions[:1])
    second = planner.rollout_from_state(initial_state, actions[1:2])

    assert torch.allclose(batched.final_state.graph_latent[0], first.final_state.graph_latent[0])
    assert torch.allclose(batched.final_state.graph_latent[1], second.final_state.graph_latent[0])
    assert len(batched.predicted_states) == 2
    assert batched.final_state.object_ids.tolist() == [0, 0]
    assert batched.final_state.object_batch.tolist() == [0, 1]


def test_latent_planner_accepts_custom_goal_energy() -> None:
    planner = _tiny_latent_planner(target=torch.tensor([1.0, -1.0]))
    result = planner.plan(Data(), goal_tensors={})

    assert result.action_latents.shape == (2, 2)
    assert result.optimizer_result.best_score > -1.0
    assert result.constant_action_cost == 0.0


def test_planner_agent_retries_after_invalid_first_decode() -> None:
    planner = _FakePlanner()
    decoder = _SequenceDecoder(
        [
            GroundAction("bad", ()),
            GroundAction("good", ()),
        ]
    )
    engine = _FakeEngine(valid_actions=[GroundAction("good", ())], goal_after=1)
    agent = PlannerAgent(planner=planner, action_decoder=decoder, parsed_problem=_parsed_problem())

    result = agent.run(engine, goal_tensors={})

    assert result.success is True
    assert result.attempts == 2
    assert planner.saw_rejection_penalty is True
    assert [action.name for action in result.applied_actions] == ["good"]


def test_planner_agent_applies_partial_valid_prefix_then_replans() -> None:
    planner = _FakePlanner(apply_steps=2, max_total_actions=3)
    decoder = _SequenceDecoder(
        [
            GroundAction("good", ()),
            GroundAction("bad", ()),
            GroundAction("good", ()),
        ]
    )
    engine = _FakeEngine(valid_actions=[GroundAction("good", ())], goal_after=2)
    agent = PlannerAgent(planner=planner, action_decoder=decoder, parsed_problem=_parsed_problem())

    result = agent.run(engine, goal_tensors={})

    assert result.success is True
    assert result.attempts == 2
    assert [action.name for action in result.applied_actions] == ["good", "good"]


def test_planner_agent_reports_decode_retry_exhaustion() -> None:
    planner = _FakePlanner(max_decode_attempts=2)
    decoder = _SequenceDecoder(
        [
            GroundAction("bad", ()),
            GroundAction("bad", ()),
        ]
    )
    engine = _FakeEngine(valid_actions=[GroundAction("good", ())], goal_after=1)
    agent = PlannerAgent(planner=planner, action_decoder=decoder, parsed_problem=_parsed_problem())

    result = agent.run(engine, goal_tensors={})

    assert result.success is False
    assert result.failure_reason == "decode_invalid"
    assert result.attempts == 2
    assert result.applied_actions == ()


def test_planner_agent_uses_apply_action_as_validity_oracle() -> None:
    planner = _FakePlanner()
    decoder = _SequenceDecoder([GroundAction("bad", ()), GroundAction("good", ())])
    engine = _FakeEngine(valid_actions=[GroundAction("good", ())], goal_after=1)
    agent = PlannerAgent(planner=planner, action_decoder=decoder, parsed_problem=_parsed_problem())

    result = agent.run(engine, goal_tensors={})

    assert result.success is True
    assert result.failure_reason is None
    assert result.attempts == 2


def test_structured_ce_planner_agent_applies_grounded_action_without_decoder() -> None:
    parsed_problem = _parsed_problem()

    def goal_energy(_goal_tensors, terminal_state: JEPALatentState) -> torch.Tensor:
        target = torch.tensor([[1.0, 0.0]], dtype=terminal_state.graph_latent.dtype)
        return ((terminal_state.graph_latent - target) ** 2).sum(dim=-1)

    planner = StructuredCEPlanner(
        graph_encoder=_TinyGraphEncoder(),
        state_encoder=_TinyStateEncoder(),
        action_encoder=_TinyActionEncoder(),
        predictor=_AdditivePredictor(),
        goal_energy=goal_energy,
        action_space=ActionDecodingSpace.from_parsed_problem(parsed_problem),
        config=StructuredCEPlannerConfig(
            horizon=1,
            num_samples=64,
            max_iters=15,
            seed=0,
            max_total_actions=1,
        ),
    )
    engine = _FakeEngine(valid_actions=[GroundAction("good", ())], goal_after=1)
    agent = GroundedPlannerAgent(planner=planner, parsed_problem=parsed_problem)

    result = agent.run(engine, goal_tensors={})

    assert result.success is True
    assert [action.name for action in result.applied_actions] == ["good"]


def test_gmm_mppi_planner_seeds_components_from_encoded_grounded_actions() -> None:
    parsed_problem = _parsed_problem()

    def goal_energy(_goal_tensors, terminal_state: JEPALatentState) -> torch.Tensor:
        target = torch.tensor([[1.0, 0.0]], dtype=terminal_state.graph_latent.dtype)
        return ((terminal_state.graph_latent - target) ** 2).sum(dim=-1)

    planner = LatentGMMMPPIPlanner(
        graph_encoder=_TinyGraphEncoder(),
        state_encoder=_TinyStateEncoder(),
        action_encoder=_TinyActionEncoder(),
        predictor=_AdditivePredictor(),
        goal_energy=goal_energy,
        action_space=ActionDecodingSpace.from_parsed_problem(parsed_problem),
        config=LatentGMMMPPIConfig(
            horizon=1,
            action_dim=2,
            action_pool_size=4,
            num_samples=64,
            max_iters=8,
            initial_std=0.5,
            seed=0,
        ),
    )

    result = planner.plan(Data(), goal_tensors={})

    assert result.action_latents.shape == (1, 2)
    assert result.optimizer_result.means.shape == (1, 4, 2)
    assert len(result.action_pool) == 4
    assert result.rollout.final_state.graph_latent.shape == (1, 2)


class _TinyGraphEncoder(nn.Module):
    def forward(self, _graph: Data) -> GraphEncoderOutput:
        return GraphEncoderOutput(
            graph_embedding=torch.zeros(1, 2),
            object_embeddings=torch.zeros(1, 2),
            object_ids=torch.tensor([0]),
            object_batch=torch.tensor([0]),
        )


class _TinyStateEncoder(nn.Module):
    def forward(self, graph_output: GraphEncoderOutput) -> JEPALatentState:
        return JEPALatentState(
            graph_latent=graph_output.graph_embedding,
            object_latents=graph_output.object_embeddings,
            object_ids=graph_output.object_ids,
            object_batch=graph_output.object_batch,
        )


class _AdditivePredictor(nn.Module):
    def forward(self, latent_state: JEPALatentState, action_latent: torch.Tensor) -> JEPALatentState:
        object_action = action_latent[latent_state.object_batch]
        return JEPALatentState(
            graph_latent=latent_state.graph_latent + action_latent,
            object_latents=latent_state.object_latents + object_action,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
        )


class _TinyActionEncoder(nn.Module):
    def forward(self, action_tensors: dict[str, torch.Tensor], _latent_state: JEPALatentState) -> torch.Tensor:
        action_id = action_tensors["action_id"].to(torch.float32)
        return torch.stack([action_id, torch.zeros_like(action_id)], dim=1)


def _tiny_latent_planner(target: torch.Tensor | None = None) -> LatentMPPIPlanner:
    target = torch.zeros(2) if target is None else target

    def goal_energy(_goal_tensors, terminal_state: JEPALatentState) -> torch.Tensor:
        return ((terminal_state.graph_latent - target) ** 2).sum(dim=-1)

    return LatentMPPIPlanner(
        graph_encoder=_TinyGraphEncoder(),
        state_encoder=_TinyStateEncoder(),
        predictor=_AdditivePredictor(),
        goal_energy=goal_energy,
        config=LatentMPPIConfig(
            horizon=2,
            action_dim=2,
            num_samples=128,
            max_iters=12,
            initial_std=1.5,
            seed=0,
        ),
    )


class _FakePlanner:
    def __init__(self, *, apply_steps: int = 1, max_total_actions: int = 8, max_decode_attempts: int = 3) -> None:
        self.config = LatentMPPIConfig(
            horizon=2,
            action_dim=2,
            apply_steps=apply_steps,
            max_total_actions=max_total_actions,
            max_decode_attempts=max_decode_attempts,
        )
        self.device = torch.device("cpu")
        self.saw_rejection_penalty = False

    def encode_graph(self, _graph: Data) -> tuple[GraphEncoderOutput, JEPALatentState]:
        graph_output = GraphEncoderOutput(
            graph_embedding=torch.zeros(1, 2),
            object_embeddings=torch.zeros(1, 2),
            object_ids=torch.tensor([0]),
            object_batch=torch.tensor([0]),
        )
        latent_state = JEPALatentState(
            graph_latent=torch.zeros(1, 2),
            object_latents=torch.zeros(1, 2),
            object_ids=torch.tensor([0]),
            object_batch=torch.tensor([0]),
        )
        return graph_output, latent_state

    def plan(
        self,
        _state_graph: Data,
        _goal_tensors: dict[str, torch.Tensor],
        *,
        initial_mean=None,
        initial_std=None,
        rejection_penalty=None,
    ) -> LatentMPPIResult:
        self.saw_rejection_penalty = self.saw_rejection_penalty or rejection_penalty is not None
        mean = torch.zeros(2, 2) if initial_mean is None else initial_mean
        std = torch.ones(2, 2) if initial_std is None else initial_std
        optimizer_result = ContinuousMPPIResult(
            best_x=mean,
            best_score=0.0,
            mode_x=mean,
            mode_score=0.0,
            mean=mean,
            std=std,
            thresholds=[0.0],
            elite_mean_scores=[0.0],
            population_best_scores=[0.0],
            iterations=1,
            converged=False,
            stop_reason="max_iters",
        )
        graph_output, initial_state = self.encode_graph(_state_graph)
        rollout = LatentRolloutOutput(
            initial_state=initial_state,
            final_state=initial_state,
            predicted_states=(),
            action_latents=mean.unsqueeze(0),
        )
        return LatentMPPIResult(
            action_latents=mean,
            optimizer_result=optimizer_result,
            rollout=rollout,
            graph_output=graph_output,
            initial_state=initial_state,
            constant_action_cost=0.0,
        )


class _SequenceDecoder:
    def __init__(self, actions: Sequence[GroundAction]) -> None:
        self.actions = list(actions)

    def decode(self, _target_action_latent: torch.Tensor, _latent_state: JEPALatentState) -> GroundAction:
        if not self.actions:
            raise AssertionError("decoder called more often than expected")
        return self.actions.pop(0)


class _FakeEngine:
    def __init__(self, *, valid_actions: Sequence[GroundAction], goal_after: int) -> None:
        self._valid_actions = {(action.name, action.arguments) for action in valid_actions}
        self._goal_after = goal_after
        self.applied: list[GroundAction] = []

    def current_facts(self) -> tuple[tuple[str, ...], ...]:
        return (("at", "o0"),)

    def apply_action(self, action_name: str, arguments: tuple[str, ...] = (), *, finish: bool = True) -> None:
        if (action_name, arguments) not in self._valid_actions:
            raise ValueError(f"Action {action_name}{arguments!r} is not applicable")
        self.applied.append(GroundAction(action_name, arguments))

    def goals_satisfied(self) -> bool:
        return len(self.applied) >= self._goal_after


def _parsed_problem() -> ParsedProblem:
    return ParsedProblem(
        name="tiny",
        types=("obj",),
        predicates={"at": PredicateSchema("at", ("obj",))},
        objects={"o0": ObjectInfo("o0", "obj")},
        actions={"good": ActionSchema("good", (), ("at",)), "bad": ActionSchema("bad", (), ("at",))},
        initial_atoms=(),
        goal_atoms=(),
        static_predicates=frozenset(),
    )
