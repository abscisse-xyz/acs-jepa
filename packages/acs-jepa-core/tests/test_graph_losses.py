from __future__ import annotations

from pathlib import Path

import torch
from acs_jepa.architectures import (
    ActionEncoder,
    JEPALatentState,
    LatentActionEncoder,
    ResidualMLPLatentPredictorG,
    StateEncoderF,
)
from acs_jepa.graph import (
    GroundAction,
    GraphEncoder,
    build_state_graph,
    parse_domain_problem,
    tensorize_action,
)
from acs_jepa.jepa import GraphJEPA
from acs_jepa.losses import (
    GraphEncodedActionInverseDynamicsLoss,
    GraphInverseDynamicsModel,
    GraphJEPALossModule,
    GraphLatentPredictionLoss,
    GraphTemporalSimilarityLoss,
    GraphVCLoss,
)

DOMAIN = """
(define (domain tiny-city)
  (:requirements :typing :negative-preconditions)
  (:types car junction road)
  (:predicates
    (same_line ?a - junction ?b - junction)
    (clear ?j - junction)
    (at_car_jun ?c - car ?j - junction)
    (road_connect ?r - road ?from - junction ?to - junction)
  )

  (:action move
    :parameters (?c - car ?from - junction ?to - junction ?r - road)
    :precondition (and
      (at_car_jun ?c ?from)
      (road_connect ?r ?from ?to)
      (clear ?to)
    )
    :effect (and
      (not (at_car_jun ?c ?from))
      (at_car_jun ?c ?to)
      (clear ?from)
      (not (clear ?to))
    )
  )
)
"""


PROBLEM = """
(define (problem tiny-city-1)
  (:domain tiny-city)
  (:objects
    car0 - car
    road0 - road
    j0 j1 - junction
  )
  (:init
    (same_line j0 j1)
    (clear j1)
    (at_car_jun car0 j0)
    (road_connect road0 j0 j1)
  )
  (:goal (and
    (at_car_jun car0 j1)
  ))
)
"""


def test_graph_latent_prediction_loss_returns_scalar_terms(tmp_path: Path) -> None:
    current, predicted, target, _ = _states_and_action(tmp_path)
    loss_fn = GraphLatentPredictionLoss(graph_weight=2.0, object_weight=3.0)

    output = loss_fn(predicted, target)

    assert output.total.ndim == 0
    assert output.graph.ndim == 0
    assert output.object.ndim == 0
    assert torch.allclose(output.total, 2.0 * output.graph + 3.0 * output.object)
    assert current.graph_latent.shape == predicted.graph_latent.shape


def test_graph_vc_loss_supports_graph_object_and_both(tmp_path: Path) -> None:
    current, _, _, _ = _states_and_action(tmp_path, batch_size=2)

    graph_loss = GraphVCLoss(target="graph")(current)
    object_loss = GraphVCLoss(target="object")(current)
    both_loss = GraphVCLoss(target="both")(current)

    assert graph_loss.ndim == 0
    assert object_loss.ndim == 0
    assert both_loss.ndim == 0


def test_graph_temporal_similarity_loss_is_zero_for_identical_states(tmp_path: Path) -> None:
    current, _, _, _ = _states_and_action(tmp_path)
    loss = GraphTemporalSimilarityLoss()(current, current)

    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_encoded_action_inverse_dynamics_predicts_action_latent_and_detaches_target(tmp_path: Path) -> None:
    current, _, target, action_latent = _states_and_action(tmp_path)
    action_latent.retain_grad()
    idm_model = GraphInverseDynamicsModel(latent_dim=6, action_dim=6, hidden_dim=10)
    loss_fn = GraphEncodedActionInverseDynamicsLoss(idm_model)

    prediction = idm_model(current, target)
    loss = loss_fn(current, target, action_latent)
    loss.backward()

    assert prediction.shape == action_latent.shape
    assert loss.ndim == 0
    assert action_latent.grad is None
    assert idm_model.model[-1].weight.grad is not None


def test_composite_graph_jepa_loss_includes_enabled_terms(tmp_path: Path) -> None:
    current, predicted, target, action_latent = _states_and_action(tmp_path, batch_size=2)
    idm_model = GraphInverseDynamicsModel(latent_dim=6, action_dim=6, hidden_dim=10)
    loss_module = GraphJEPALossModule(
        prediction_loss=GraphLatentPredictionLoss(),
        regularization_loss=GraphVCLoss(),
        temporal_similarity_loss=GraphTemporalSimilarityLoss(),
        inverse_dynamics_loss=GraphEncodedActionInverseDynamicsLoss(idm_model),
        regularization_coeff=0.1,
        similarity_coeff=0.2,
        inverse_dynamics_coeff=0.3,
    )

    output = loss_module(
        observed_states=_temporal_state(current, target),
        predicted_states_by_order={1: _temporal_state(predicted)},
        action_latents=action_latent.unsqueeze(1),
    )

    assert output.total.ndim == 0
    assert output.similarity is not None
    assert output.inverse_dynamics is not None
    assert output.regularization.ndim == 0
    assert "total" in output.terms
    assert "inverse_dynamics" in output.terms
    assert "regularization" in output.terms


def test_graph_jepa_training_step_uses_loss_module(tmp_path: Path) -> None:
    parsed, state_graph, next_graph = _graphs(tmp_path)
    graph_encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)
    idm_model = GraphInverseDynamicsModel(latent_dim=6, action_dim=6, hidden_dim=10)
    model = GraphJEPA(
        graph_encoder=graph_encoder,
        state_encoder=StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10),
        action_encoder=ActionEncoder(
            LatentActionEncoder(
                num_actions=len(parsed.actions),
                max_action_arity=parsed.max_action_arity,
                latent_dim=6,
                action_dim=6,
                hidden_dim=10,
            ),
            action_dim=6,
        ),
        predictor=ResidualMLPLatentPredictorG(latent_dim=6, action_dim=6, hidden_dim=10),
        loss_module=GraphJEPALossModule(
            prediction_loss=GraphLatentPredictionLoss(),
            regularization_loss=GraphVCLoss(),
            temporal_similarity_loss=GraphTemporalSimilarityLoss(),
            inverse_dynamics_loss=GraphEncodedActionInverseDynamicsLoss(idm_model),
            regularization_coeff=0.1,
            similarity_coeff=0.2,
            inverse_dynamics_coeff=0.3,
        ),
    )
    batch = {
        "states": (state_graph, next_graph),
        "actions": _single_action_window(
            tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
        ),
    }

    output = model(batch["states"], batch["actions"])
    output.loss.total.backward()

    assert output.loss.regularization.ndim == 0
    assert output.loss.similarity is not None
    assert output.loss.inverse_dynamics is not None
    assert output.loss.terms["total"].ndim == 0
    assert idm_model.model[-1].weight.grad is not None


def _states_and_action(tmp_path: Path, batch_size: int = 1):
    parsed, state_graph, next_graph = _graphs(tmp_path)
    graphs = [state_graph for _ in range(batch_size)]
    next_graphs = [next_graph for _ in range(batch_size)]
    if batch_size > 1:
        from torch_geometric.loader import DataLoader

        state_graph = next(iter(DataLoader(graphs, batch_size=batch_size)))
        next_graph = next(iter(DataLoader(next_graphs, batch_size=batch_size)))

    graph_encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    predictor = ResidualMLPLatentPredictorG(latent_dim=6, action_dim=6, hidden_dim=10)

    graph_output = graph_encoder(state_graph)
    current = state_encoder(graph_output)
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    if batch_size > 1:
        action = {key: value.unsqueeze(0).expand(batch_size, *value.shape) for key, value in action.items()}
    action_latent = action_encoder(action, current)
    predicted = predictor(current, action_latent)
    target = state_encoder(graph_encoder(next_graph))
    return current, predicted, target, action_latent


def _temporal_state(*states: JEPALatentState) -> JEPALatentState:
    first = states[0]
    return JEPALatentState(
        graph_latent=torch.stack([state.graph_latent for state in states], dim=1),
        object_latents=torch.stack([state.object_latents for state in states], dim=1),
        object_ids=first.object_ids,
        object_batch=first.object_batch,
    )


def _single_action_window(action_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.reshape(1, 1, *value.shape) for key, value in action_tensors.items()}


def _graphs(tmp_path: Path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)
    parsed = parse_domain_problem(domain_path, problem_path)
    return parsed, build_state_graph(parsed, parsed.initial_atoms), build_state_graph(parsed, parsed.goal_atoms)
