from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from acs_jepa.architectures import (
    ActionDecoder,
    ActionDecodingSpace,
    ActionEncoder,
    ActionSamplingFamily,
    GraphStateProjector,
    GRULatentPredictorG,
    JEPALatentState,
    LatentActionEncoder,
    ResidualMLPLatentPredictorG,
    StateEncoderF,
    build_action_encoder,
    build_latent_predictor,
)
from acs_jepa.goals import PredicateEvaluator, build_predicate_evaluator
from acs_jepa.graph import (
    GraphEncoder,
    GroundAction,
    build_state_graph,
    parse_domain_problem,
    tensorize_action,
    tensorize_predicate,
)
from acs_jepa.jepa import GraphJEPA, GraphJEPACandidateRolloutOutput
from acs_jepa.losses import GraphJEPALossModule, GraphLatentPredictionLoss, GraphVCLoss
from acs_jepa.planner import LatentMPPIConfig, LatentMPPIPlanner
from torch_geometric.loader import DataLoader

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

  (:action build
    :parameters (?r - road ?from - junction ?to - junction)
    :precondition (same_line ?from ?to)
    :effect (road_connect ?r ?from ?to)
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


def test_state_encoder_projects_graph_output_to_latent_state(tmp_path: Path) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    state_encoder = GraphStateProjector(embedding_dim=8, latent_dim=6, hidden_dim=10)

    latent_state = state_encoder(graph_output)

    assert latent_state.graph_latent.shape == (1, 6)
    assert latent_state.object_latents.shape == (4, 6)
    assert latent_state.object_ids.tolist() == graph_output.object_ids.tolist()
    assert latent_state.object_batch.tolist() == graph_output.object_batch.tolist()
    assert parsed.object_to_id["car0"] in latent_state.object_ids.tolist()


def test_state_encoder_single_frame_matches_singleton_sequence(tmp_path: Path) -> None:
    _, graph_output, _ = _encoded_graph(tmp_path)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)

    single = state_encoder(graph_output)
    sequence_single = state_encoder(_temporal_graph_output(graph_output))

    assert sequence_single.graph_latent.shape == (1, 1, 6)
    assert sequence_single.object_latents.shape == (4, 1, 6)
    assert torch.allclose(single.graph_latent, sequence_single.graph_latent[:, 0])
    assert torch.allclose(single.object_latents, sequence_single.object_latents[:, 0])
    assert torch.equal(single.object_ids, sequence_single.object_ids)
    assert torch.equal(single.object_batch, sequence_single.object_batch)


@pytest.mark.parametrize("argument_encoder", ["pooled", "rnn"])
def test_action_encoder_uses_latent_object_embeddings_for_arguments(tmp_path: Path, argument_encoder: str) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
        argument_encoder=argument_encoder,
    )
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    context = _action_context(graph_output)

    action_latent = action_encoder(action, context)

    assert action_latent.shape == (1, 6)


def test_temporal_action_encoder_single_frame_matches_singleton_sequence(tmp_path: Path) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    action_encoder = ActionEncoder(
        LatentActionEncoder(
            num_actions=len(parsed.actions),
            max_action_arity=parsed.max_action_arity,
            latent_dim=6,
            action_dim=6,
            hidden_dim=10,
        ),
        action_dim=6,
    )
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    context = state_encoder(graph_output)

    single = action_encoder(action, context)
    temporal_graph_output = _temporal_graph_output(graph_output)
    temporal_context = state_encoder(temporal_graph_output)
    sequence_single = action_encoder(_single_action_window(action), temporal_context)

    assert sequence_single.shape == (1, 1, 6)
    assert torch.allclose(single, sequence_single[:, 0])


def test_temporal_action_encoder_rejects_unbatched_action_sequence(tmp_path: Path) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    action_encoder = ActionEncoder(
        LatentActionEncoder(
            num_actions=len(parsed.actions),
            max_action_arity=parsed.max_action_arity,
            latent_dim=6,
            action_dim=6,
            hidden_dim=10,
        ),
        action_dim=6,
    )
    temporal_graph_output = _temporal_graph_output(graph_output, graph_output)
    temporal_context = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)(temporal_graph_output)
    actions = _stack_tensor_dict(
        [
            tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))),
            tensorize_action(parsed, GroundAction("build", ("road0", "j0", "j1"))),
        ]
    )

    with pytest.raises(ValueError, match="Temporal action tensors must be batched"):
        action_encoder(actions, temporal_context)


@pytest.mark.parametrize("argument_encoder", ["pooled", "rnn"])
def test_latent_action_encoder_uses_latent_object_embeddings_for_arguments(
    tmp_path: Path,
    argument_encoder: str,
) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    latent_state = state_encoder(graph_output)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
        argument_encoder=argument_encoder,
    )
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    action_latent = action_encoder(action, latent_state)

    assert action_latent.shape == (1, 6)


def test_pddl_action_sampling_family_samples_type_valid_actions(tmp_path: Path) -> None:
    parsed, _, _ = _encoded_graph(tmp_path)
    space = ActionDecodingSpace.from_parsed_problem(parsed)
    family = ActionSamplingFamily(space, device="cpu")

    samples = family.sample(64, torch.Generator().manual_seed(0))

    assert samples.shape == (64, 1 + parsed.max_action_arity)
    for sample in samples:
        action = space.sample_to_ground_action(sample)
        schema = parsed.actions[action.name]
        assert len(action.arguments) == schema.arity
        for object_name, type_name in zip(action.arguments, schema.parameter_types, strict=True):
            assert parsed.objects[object_name].type == type_name


def test_exact_action_latent_decoder_reconstructs_encoded_action(tmp_path: Path) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    torch.manual_seed(0)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    target_action = GroundAction("move", ("car0", "j0", "j1", "road0"))
    target_tensors = tensorize_action(parsed, target_action)
    context = _action_context(graph_output)
    target_latent = action_encoder(target_tensors, context)
    decoder = ActionDecoder(
        parsed_problem=parsed,
        action_encoder=action_encoder,
        method="exact",
    )

    decoded = decoder.decode(target_latent, context)

    assert decoded == target_action


def test_exact_action_latent_decoder_reconstructs_latent_encoded_action(tmp_path: Path) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    torch.manual_seed(0)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    context = state_encoder(graph_output)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    target_action = GroundAction("move", ("car0", "j0", "j1", "road0"))
    target_tensors = tensorize_action(parsed, target_action)
    target_latent = action_encoder(target_tensors, context)
    decoder = ActionDecoder(
        parsed_problem=parsed,
        action_encoder=action_encoder,
        method="exact",
    )

    decoded = decoder.decode(target_latent, context)

    assert decoded == target_action


def test_cem_action_latent_decoder_matches_exact_on_tiny_domain(tmp_path: Path) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    torch.manual_seed(0)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    target_action = GroundAction("build", ("road0", "j0", "j1"))
    target_tensors = tensorize_action(parsed, target_action)
    context = _action_context(graph_output)
    target_latent = action_encoder(target_tensors, context)
    decoder = ActionDecoder(
        parsed_problem=parsed,
        action_encoder=action_encoder,
        method="cem",
        num_samples=128,
        max_iters=20,
        seed=0,
    )

    decoded = decoder.decode(target_latent, context)

    assert decoded == target_action


@pytest.mark.parametrize("argument_encoder", ["pooled", "rnn"])
def test_predicate_evaluator_scores_latent_predicate_queries(tmp_path: Path, argument_encoder: str) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    evaluator = PredicateEvaluator(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        hidden_dim=10,
        argument_encoder=argument_encoder,
    )
    latent_state = state_encoder(graph_output)
    predicate = tensorize_predicate(parsed, parsed.goal_atoms[0])

    logits = evaluator(predicate, latent_state)

    assert logits.shape == (1,)


@pytest.mark.parametrize("predictor_cls", [ResidualMLPLatentPredictorG, GRULatentPredictorG])
def test_latent_predictor_preserves_object_identity_tensors(tmp_path: Path, predictor_cls) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    predictor = predictor_cls(latent_dim=6, action_dim=6, hidden_dim=10)
    latent_state = state_encoder(graph_output)
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    action_latent = action_encoder(action, latent_state)

    next_state = predictor(latent_state, action_latent)

    assert next_state.graph_latent.shape == latent_state.graph_latent.shape
    assert next_state.object_latents.shape == latent_state.object_latents.shape
    assert next_state.object_ids.data_ptr() == latent_state.object_ids.data_ptr()
    assert next_state.object_batch.data_ptr() == latent_state.object_batch.data_ptr()
    assert not torch.allclose(next_state.graph_latent, latent_state.graph_latent)


@pytest.mark.parametrize("predictor_cls", [ResidualMLPLatentPredictorG, GRULatentPredictorG])
def test_latent_predictor_temporal_path_matches_single_step(tmp_path: Path, predictor_cls) -> None:
    parsed, graph_output, _ = _encoded_graph(tmp_path)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    predictor = predictor_cls(latent_dim=6, action_dim=6, hidden_dim=10)
    latent_state = state_encoder(graph_output)
    action_a = action_encoder(
        tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))), latent_state
    )
    action_b = action_encoder(tensorize_action(parsed, GroundAction("build", ("road0", "j0", "j1"))), latent_state)
    temporal_state = _temporal_latent_state(latent_state, latent_state)
    temporal_actions = torch.stack([action_a, action_b], dim=1)

    temporal_next = predictor(temporal_state, temporal_actions)
    single_a = predictor(latent_state, action_a)
    single_b = predictor(latent_state, action_b)

    assert torch.allclose(temporal_next.graph_latent[:, 0], single_a.graph_latent)
    assert torch.allclose(temporal_next.graph_latent[:, 1], single_b.graph_latent)
    assert torch.allclose(temporal_next.object_latents[:, 0], single_a.object_latents)
    assert torch.allclose(temporal_next.object_latents[:, 1], single_b.object_latents)
    assert temporal_next.object_ids.data_ptr() == latent_state.object_ids.data_ptr()
    assert temporal_next.object_batch.data_ptr() == latent_state.object_batch.data_ptr()


@pytest.mark.parametrize("argument_encoder", ["pooled", "rnn"])
def test_components_handle_batched_action_and_predicate_tensors(tmp_path: Path, argument_encoder: str) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    batch = next(iter(DataLoader([graph, graph], batch_size=2)))
    graph_encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)
    graph_output = graph_encoder(batch)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
        argument_encoder=argument_encoder,
    )
    evaluator = PredicateEvaluator(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        hidden_dim=10,
        argument_encoder=argument_encoder,
    )
    action = _stack_tensor_dict(
        [
            tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))),
            tensorize_action(parsed, GroundAction("build", ("road0", "j0", "j1"))),
        ]
    )
    predicate = _stack_tensor_dict([tensorize_predicate(parsed, parsed.goal_atoms[0])] * 2)

    latent_state = state_encoder(graph_output)
    action_latent = action_encoder(action, latent_state)
    logits = evaluator(predicate, latent_state)

    assert action_latent.shape == (2, 6)
    assert logits.shape == (2,)


@pytest.mark.parametrize("argument_encoder", ["pooled", "rnn"])
@pytest.mark.parametrize("predictor_cls", [ResidualMLPLatentPredictorG, GRULatentPredictorG])
def test_gradients_flow_through_action_encoder_and_latent_components(
    tmp_path: Path,
    argument_encoder: str,
    predictor_cls,
) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    graph_encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)
    graph_output = graph_encoder(graph)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
        argument_encoder=argument_encoder,
    )
    evaluator = PredicateEvaluator(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        hidden_dim=10,
        argument_encoder=argument_encoder,
    )
    predictor = predictor_cls(latent_dim=6, action_dim=6, hidden_dim=10)
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    predicate = tensorize_predicate(parsed, parsed.goal_atoms[0])

    latent_state = state_encoder(graph_output)
    action_latent = action_encoder(action, latent_state)
    next_state = predictor(latent_state, action_latent)
    loss = action_latent.sum() + evaluator(predicate, next_state).sum() + next_state.graph_latent.sum()
    loss.backward()

    assert graph_encoder.object_id_embedding.embedding.weight.grad is not None
    assert graph_encoder.object_id_embedding.embedding.weight.grad.abs().sum().item() > 0
    assert state_encoder.base_encoder.graph_projector[-1].weight.grad is not None
    assert evaluator.predicate_embedding.weight.grad is not None
    assert any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in predictor.parameters())


def test_latent_action_encoder_backpropagates_into_state_encoder_object_projector(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    graph_encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)
    graph_output = graph_encoder(graph)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10)
    action_encoder = LatentActionEncoder(
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    latent_state = state_encoder(graph_output)

    loss = action_encoder(action, latent_state).sum()
    loss.backward()

    assert state_encoder.base_encoder.object_projector[-1].weight.grad is not None
    assert state_encoder.base_encoder.object_projector[-1].weight.grad.abs().sum().item() > 0


def test_component_factories_select_latent_action_variant(tmp_path: Path) -> None:
    parsed, _, _ = _encoded_graph(tmp_path)

    latent_action = build_action_encoder(
        kind="pooled",
        num_actions=len(parsed.actions),
        max_action_arity=parsed.max_action_arity,
        latent_dim=6,
        action_dim=6,
        hidden_dim=10,
    )
    rnn_predicate = build_predicate_evaluator(
        kind="rnn",
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        hidden_dim=10,
    )
    mlp_predictor = build_latent_predictor(kind="mlp", latent_dim=6, action_dim=6, hidden_dim=10)
    gru_predictor = build_latent_predictor(kind="gru", latent_dim=6, action_dim=6, hidden_dim=10)

    assert isinstance(latent_action, LatentActionEncoder)
    assert isinstance(rnn_predicate, PredicateEvaluator)
    assert isinstance(mlp_predictor, ResidualMLPLatentPredictorG)
    assert isinstance(gru_predictor, GRULatentPredictorG)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (
            build_action_encoder,
            {
                "num_actions": 1,
                "max_action_arity": 1,
                "latent_dim": 6,
                "action_dim": 6,
            },
        ),
        (
            build_predicate_evaluator,
            {
                "num_predicates": 1,
                "max_predicate_arity": 1,
                "latent_dim": 6,
            },
        ),
        (build_latent_predictor, {"latent_dim": 6}),
    ],
)
def test_component_factories_reject_unknown_kinds(factory, kwargs) -> None:
    with pytest.raises(ValueError):
        factory(kind="unknown", **kwargs)


def test_graph_jepa_requires_loss_module(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    del graph

    with pytest.raises(ValueError, match="loss_module"):
        GraphJEPA(
            graph_encoder=GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2),
            state_encoder=StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10),
            action_encoder=LatentActionEncoder(
                num_actions=len(parsed.actions),
                max_action_arity=parsed.max_action_arity,
                latent_dim=6,
                action_dim=6,
                hidden_dim=10,
            ),
            predictor=ResidualMLPLatentPredictorG(latent_dim=6, action_dim=6, hidden_dim=10),
            loss_module=None,
        )


def test_graph_jepa_forward_handles_k_one_trajectory(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    next_graph = build_state_graph(parsed, parsed.goal_atoms)
    model = _build_graph_jepa(parsed)

    output = model(
        (graph, next_graph),
        _single_action_window(tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))),
    )

    assert output.observed_states.graph_latent.shape == (1, 2, 6)
    assert output.predicted_states_by_order[1].graph_latent.shape == (1, 1, 6)
    assert output.action_latents.shape == (1, 1, 6)
    assert output.loss.total.ndim == 0


def test_graph_jepa_encode_action_constructs_action_context(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    model = _build_graph_jepa(parsed)
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))

    action_latent = model.encode_action(action, graph)

    assert action_latent.shape == (1, 6)


def test_graph_jepa_trajectory_rollout_handles_batched_k_one_windows(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    next_graph = build_state_graph(parsed, parsed.goal_atoms)
    graph_batch = next(iter(DataLoader([graph, graph], batch_size=2)))
    next_graph_batch = next(iter(DataLoader([next_graph, next_graph], batch_size=2)))
    action = _stack_tensor_dict(
        [
            tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))),
            tensorize_action(parsed, GroundAction("build", ("road0", "j0", "j1"))),
        ]
    )
    action = {key: value.unsqueeze(1) for key, value in action.items()}
    model = _build_graph_jepa(parsed)

    output = model.trajectory_rollout(
        (graph_batch, next_graph_batch),
        action,
    )

    assert output.observed_states.graph_latent.shape == (2, 2, 6)
    assert output.predicted_states_by_order[1].graph_latent.shape == (2, 1, 6)
    assert output.action_latents.shape == (2, 1, 6)
    assert output.loss.graph_prediction.ndim == 0
    assert output.loss.regularization.ndim == 0


def test_graph_jepa_trajectory_rollout_supports_latent_action_encoder(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    next_graph = build_state_graph(parsed, parsed.goal_atoms)
    action = tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0")))
    model = _build_graph_jepa(
        parsed,
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
    )

    output = model.trajectory_rollout((graph, next_graph), _single_action_window(action))

    assert output.observed_states.graph_latent.shape == (1, 2, 6)
    assert output.predicted_states_by_order[1].graph_latent.shape == (1, 1, 6)
    assert output.action_latents.shape == (1, 1, 6)


def test_graph_jepa_trajectory_rollout_returns_k_step_losses(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    next_graph = build_state_graph(parsed, parsed.goal_atoms)
    model = _build_graph_jepa(
        parsed,
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
    )
    actions = _stack_tensor_dict(
        [
            tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))),
            tensorize_action(parsed, GroundAction("move", ("car0", "j1", "j0", "road0"))),
        ]
    )

    output = model.trajectory_rollout((graph, next_graph, graph), _batch_action_sequence(actions))

    assert output.observed_states.graph_latent.shape == (1, 3, 6)
    assert output.action_latents.shape == (1, 2, 6)
    assert output.predicted_states_by_order[1].graph_latent.shape == (1, 2, 6)
    assert output.predicted_states_by_order[2].graph_latent.shape == (1, 1, 6)
    assert "prediction/order_1" in output.loss.terms
    assert "prediction/order_2" in output.loss.terms
    assert output.loss.total.ndim == 0


def test_graph_jepa_recursive_predictions_call_predictor_once_per_order(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    next_graph = build_state_graph(parsed, parsed.goal_atoms)
    predictor = _CountingPredictor()
    model = _build_graph_jepa(parsed, predictor=predictor)
    actions = _batch_action_sequence(
        _stack_tensor_dict(
            [
                tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))),
                tensorize_action(parsed, GroundAction("move", ("car0", "j1", "j0", "road0"))),
            ]
        )
    )

    output = model.trajectory_rollout((graph, next_graph, graph), actions)

    assert predictor.call_count == 2
    assert output.predicted_states_by_order[1].graph_latent.shape == (1, 2, 6)
    assert output.predicted_states_by_order[2].graph_latent.shape == (1, 1, 6)


def test_candidate_rollout_uses_each_predicted_source_and_preserves_order(tmp_path: Path) -> None:
    parsed, _, _ = _encoded_graph(tmp_path)
    action_encoder = _RecordingCandidateActionEncoder()
    predictor = _RecordingCandidatePredictor()
    model = _build_graph_jepa(parsed, action_encoder=action_encoder, predictor=predictor)
    object_ids = torch.tensor([0, 1, 0, 1])
    object_batch = torch.tensor([0, 0, 1, 1])
    initial_state = JEPALatentState(
        graph_latent=torch.zeros(2, 2),
        object_latents=torch.zeros(4, 2),
        object_ids=object_ids,
        object_batch=object_batch,
    )
    action_tensors = {
        "action_id": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "arguments": torch.arange(24).reshape(2, 3, 4),
    }

    output = model.rollout_grounded_candidates(initial_state, action_tensors)

    assert tuple(inspect.signature(GraphJEPA.rollout_grounded_candidates).parameters) == (
        "self",
        "initial_state",
        "action_tensors",
    )
    assert tuple(field.name for field in fields(GraphJEPACandidateRolloutOutput)) == (
        "initial_state",
        "final_state",
        "predicted_states",
        "control_latents",
    )
    with pytest.raises(FrozenInstanceError):
        setattr(output, "initial_state", initial_state)
    assert output.initial_state is initial_state
    assert output.final_state is output.predicted_states[-1]
    assert len(output.predicted_states) == 3
    assert action_encoder.states == [initial_state, *output.predicted_states[:-1]]
    assert predictor.states == [initial_state, *output.predicted_states[:-1]]
    for step, candidate in enumerate(action_encoder.actions):
        for name, tensor in action_tensors.items():
            assert torch.equal(candidate[name], tensor[:, step])
    expected_controls = torch.tensor(
        [[[1.0, 0.0], [2.0, 1.0], [3.0, 3.0]], [[4.0, 0.0], [5.0, 4.0], [6.0, 9.0]]]
    )
    assert torch.equal(output.control_latents, expected_controls)
    assert [state.graph_latent[:, 0].tolist() for state in output.predicted_states] == [
        [1.0, 4.0],
        [3.0, 9.0],
        [6.0, 15.0],
    ]


def test_candidate_rollout_supports_real_action_encoder_predictor_and_gradients(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    graph_batch = next(iter(DataLoader([graph, graph], batch_size=2)))
    action_encoder = ActionEncoder(
        LatentActionEncoder(
            num_actions=len(parsed.actions),
            max_action_arity=parsed.max_action_arity,
            latent_dim=6,
            action_dim=6,
            hidden_dim=10,
        ),
        action_dim=6,
        context_steps=1,
    )
    predictor = ResidualMLPLatentPredictorG(latent_dim=6, action_dim=6, hidden_dim=10)
    model = _build_graph_jepa(parsed, action_encoder=action_encoder, predictor=predictor)
    initial_state = model.encode(graph_batch)
    candidates = [
        tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))),
        tensorize_action(parsed, GroundAction("build", ("road0", "j0", "j1"))),
        tensorize_action(parsed, GroundAction("move", ("car0", "j1", "j0", "road0"))),
    ]
    sequence = _stack_tensor_dict(candidates)
    action_tensors = {name: tensor.unsqueeze(0).expand(2, *tensor.shape) for name, tensor in sequence.items()}

    output = model.rollout_grounded_candidates(initial_state, action_tensors)
    (output.control_latents.sum() + output.final_state.graph_latent.sum()).backward()

    assert output.control_latents.shape == (2, 3, 6)
    assert output.final_state.graph_latent.shape == (2, 6)
    assert output.final_state.object_latents.shape == (8, 6)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in action_encoder.parameters()
    )
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in predictor.parameters())


def test_candidate_rollout_rejects_empty_action_mapping(tmp_path: Path) -> None:
    model, initial_state, _ = _candidate_rollout_fixture(tmp_path)

    with pytest.raises(ValueError, match="^action_tensors must not be empty$"):
        model.rollout_grounded_candidates(initial_state, {})


def test_candidate_rollout_rejects_non_tensor_action_value(tmp_path: Path) -> None:
    model, initial_state, _ = _candidate_rollout_fixture(tmp_path)

    with pytest.raises(TypeError, match="^action tensor 'action_id' must be a Tensor$"):
        model.rollout_grounded_candidates(initial_state, {"action_id": [[1, 2], [3, 4]]})


def test_candidate_rollout_rejects_action_tensor_below_rank_two(tmp_path: Path) -> None:
    model, initial_state, _ = _candidate_rollout_fixture(tmp_path)

    message = "action tensor 'action_id' must have at least two dimensions"
    with pytest.raises(ValueError, match=f"^{message}$"):
        model.rollout_grounded_candidates(initial_state, {"action_id": torch.tensor([1, 2])})


def test_candidate_rollout_rejects_inconsistent_action_batch_horizon(tmp_path: Path) -> None:
    model, initial_state, _ = _candidate_rollout_fixture(tmp_path)
    action_tensors = {
        "action_id": torch.ones(2, 3),
        "arguments": torch.ones(2, 2, 4),
    }

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(initial_state, action_tensors)
    assert str(error.value) == "all action tensors must share leading [B, H] dimensions"


def test_candidate_rollout_rejects_initial_graph_rank(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent.unsqueeze(1),
        object_latents=state.object_latents,
        object_ids=state.object_ids,
        object_batch=state.object_batch,
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial_state.graph_latent must have shape [B, D_z]"


def test_candidate_rollout_rejects_empty_initial_graph(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=torch.empty(0, 2),
        object_latents=state.object_latents,
        object_ids=state.object_ids,
        object_batch=state.object_batch,
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial_state.graph_latent dimensions must be nonempty"


def test_candidate_rollout_rejects_initial_object_rank(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent,
        object_latents=state.object_latents.unsqueeze(1),
        object_ids=state.object_ids,
        object_batch=state.object_batch,
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial_state.object_latents must have shape [N_obj, D_z]"


def test_candidate_rollout_rejects_initial_object_width(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent,
        object_latents=torch.zeros(4, 3),
        object_ids=state.object_ids,
        object_batch=state.object_batch,
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial state graph and object latent widths must match"


def test_candidate_rollout_rejects_initial_object_row_mismatch(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent,
        object_latents=state.object_latents[:3],
        object_ids=state.object_ids,
        object_batch=state.object_batch,
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial state object rows and metadata lengths must match"


def test_candidate_rollout_rejects_initial_object_ids_rank(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent,
        object_latents=state.object_latents,
        object_ids=state.object_ids.unsqueeze(1),
        object_batch=state.object_batch,
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial_state.object_ids must be rank one"


def test_candidate_rollout_rejects_initial_object_batch_rank(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent,
        object_latents=state.object_latents,
        object_ids=state.object_ids,
        object_batch=state.object_batch.unsqueeze(1),
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial_state.object_batch must be rank one"


def test_candidate_rollout_rejects_initial_metadata_length_mismatch(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent,
        object_latents=state.object_latents,
        object_ids=state.object_ids[:-1],
        object_batch=state.object_batch,
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial state object rows and metadata lengths must match"


def test_candidate_rollout_rejects_initial_object_batch_out_of_range(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    state = JEPALatentState(
        graph_latent=state.graph_latent,
        object_latents=state.object_latents,
        object_ids=state.object_ids,
        object_batch=torch.tensor([0, 0, 1, 2]),
    )

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "initial_state.object_batch values must be in [0, B)"


def test_candidate_rollout_rejects_candidate_batch_mismatch(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    action_tensors = {name: tensor[:1] for name, tensor in action_tensors.items()}

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "candidate batch must match initial state graph batch"


def test_candidate_rollout_rejects_zero_candidate_horizon(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    action_tensors = {name: tensor[:, :0] for name, tensor in action_tensors.items()}

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "candidate horizon must be positive"


def test_candidate_rollout_rejects_exposed_none_action_context(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.action_encoder.context_steps = None

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "action_encoder.context_steps must be exactly 1"


def test_candidate_rollout_rejects_exposed_multi_step_action_context(tmp_path: Path) -> None:
    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.action_encoder.context_steps = 2

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "action_encoder.context_steps must be exactly 1"


def test_candidate_rollout_rejects_non_tensor_control(tmp_path: Path) -> None:
    class ListControlEncoder(nn.Module):
        def forward(self, action_tensors, latent_state):
            return [[0.0, 0.0], [0.0, 0.0]]

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.action_encoder = ListControlEncoder()

    with pytest.raises(TypeError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "action encoder control must be a Tensor with shape [B, D_a]"


def test_candidate_rollout_rejects_control_rank(tmp_path: Path) -> None:
    class RankThreeControlEncoder(nn.Module):
        def forward(self, action_tensors, latent_state):
            return torch.zeros(latent_state.graph_latent.size(0), 1, 2)

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.action_encoder = RankThreeControlEncoder()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "action encoder control must have shape [B, D_a]"


def test_candidate_rollout_rejects_control_batch_mismatch(tmp_path: Path) -> None:
    class WrongBatchControlEncoder(nn.Module):
        def forward(self, action_tensors, latent_state):
            return torch.zeros(1, 2)

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.action_encoder = WrongBatchControlEncoder()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "action encoder control batch must match candidate batch"


def test_candidate_rollout_rejects_later_control_width_change(tmp_path: Path) -> None:
    class ChangingWidthControlEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0

        def forward(self, action_tensors, latent_state):
            self.call_count += 1
            width = 2 if self.call_count == 1 else 3
            return torch.zeros(latent_state.graph_latent.size(0), width)

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.action_encoder = ChangingWidthControlEncoder()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "action encoder control width must remain constant"


def test_candidate_rollout_rejects_non_latent_predictor_output(tmp_path: Path) -> None:
    class TensorPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return latent_state.graph_latent

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = TensorPredictor()

    with pytest.raises(TypeError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor must return JEPALatentState"


def test_candidate_rollout_rejects_predicted_graph_rank_change(tmp_path: Path) -> None:
    class GraphRankPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent.unsqueeze(1),
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = GraphRankPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor graph_latent must preserve rank two"


def test_candidate_rollout_rejects_predicted_graph_batch_change(tmp_path: Path) -> None:
    class GraphBatchPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent[:1],
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = GraphBatchPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor graph_latent must preserve initial batch size"


def test_candidate_rollout_rejects_predicted_graph_width_change(tmp_path: Path) -> None:
    class GraphWidthPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=torch.cat((latent_state.graph_latent, torch.zeros(2, 1)), dim=1),
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = GraphWidthPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor graph_latent must preserve initial latent width"


def test_candidate_rollout_rejects_final_in_place_initial_graph_resize(tmp_path: Path) -> None:
    class FinalGraphResizePredictor(nn.Module):
        def __init__(self, original_state: JEPALatentState) -> None:
            super().__init__()
            self.original_state = original_state
            self.call_count = 0

        def forward(self, latent_state, action_latent):
            self.call_count += 1
            if self.call_count == 3:
                self.original_state.graph_latent.resize_(2, 3)
            return self.original_state

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    predictor = FinalGraphResizePredictor(state)
    model.predictor = predictor

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor graph_latent must preserve initial latent width"
    assert predictor.call_count == 3


def test_candidate_rollout_rejects_predicted_object_rank_change(tmp_path: Path) -> None:
    class ObjectRankPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents.unsqueeze(1),
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = ObjectRankPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_latents must preserve rank two"


def test_candidate_rollout_rejects_predicted_object_row_change(tmp_path: Path) -> None:
    class ObjectRowPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents[:-1],
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = ObjectRowPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_latents must preserve initial row count"


def test_candidate_rollout_rejects_predicted_object_width_change(tmp_path: Path) -> None:
    class ObjectWidthPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=torch.cat((latent_state.object_latents, torch.zeros(4, 1)), dim=1),
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = ObjectWidthPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_latents must preserve initial latent width"


def test_candidate_rollout_rejects_final_in_place_initial_object_resize(tmp_path: Path) -> None:
    class FinalObjectResizePredictor(nn.Module):
        def __init__(self, original_state: JEPALatentState) -> None:
            super().__init__()
            self.original_state = original_state
            self.call_count = 0

        def forward(self, latent_state, action_latent):
            self.call_count += 1
            if self.call_count == 3:
                self.original_state.object_latents.resize_(4, 3)
            return self.original_state

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    predictor = FinalObjectResizePredictor(state)
    model.predictor = predictor

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_latents must preserve initial latent width"
    assert predictor.call_count == 3


def test_candidate_rollout_rejects_replaced_object_ids_values(tmp_path: Path) -> None:
    class ReorderedObjectIdsPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids.flip(0),
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = ReorderedObjectIdsPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_ids must preserve initial values"


def test_candidate_rollout_rejects_cloned_object_ids(tmp_path: Path) -> None:
    class ClonedObjectIdsPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids.clone(),
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = ClonedObjectIdsPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_ids must preserve exact tensor identity"


def test_candidate_rollout_rejects_replaced_object_batch_values(tmp_path: Path) -> None:
    class ReorderedObjectBatchPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch.flip(0),
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = ReorderedObjectBatchPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_batch must preserve initial values"


def test_candidate_rollout_rejects_cloned_object_batch(tmp_path: Path) -> None:
    class ClonedObjectBatchPredictor(nn.Module):
        def forward(self, latent_state, action_latent):
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch.clone(),
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    model.predictor = ClonedObjectBatchPredictor()

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_batch must preserve exact tensor identity"


def test_candidate_rollout_rejects_late_in_place_object_ids_mutation(tmp_path: Path) -> None:
    class LateObjectIdsMutationPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0

        def forward(self, latent_state, action_latent):
            self.call_count += 1
            if self.call_count == 3:
                latent_state.object_ids.copy_(latent_state.object_ids.flip(0))
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    predictor = LateObjectIdsMutationPredictor()
    model.predictor = predictor

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_ids must preserve initial values"
    assert predictor.call_count == 3


def test_candidate_rollout_rejects_late_in_place_object_batch_mutation(tmp_path: Path) -> None:
    class LateObjectBatchMutationPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0

        def forward(self, latent_state, action_latent):
            self.call_count += 1
            if self.call_count == 3:
                latent_state.object_batch.copy_(latent_state.object_batch.flip(0))
            return JEPALatentState(
                graph_latent=latent_state.graph_latent,
                object_latents=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
            )

    model, state, action_tensors = _candidate_rollout_fixture(tmp_path)
    predictor = LateObjectBatchMutationPredictor()
    model.predictor = predictor

    with pytest.raises(ValueError) as error:
        model.rollout_grounded_candidates(state, action_tensors)
    assert str(error.value) == "predictor object_batch must preserve initial values"
    assert predictor.call_count == 3


def test_planner_rollout_accepts_action_latents(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    model = _build_graph_jepa(parsed)
    planner = _build_latent_planner(model)
    action_latents = torch.randn(1, 3, 6)
    _, initial_state = planner.encode_graph(graph)

    output = planner.rollout_from_state(
        initial_state,
        action_latents,
    )

    assert output.initial_state.graph_latent.shape == (1, 6)
    assert output.final_state.graph_latent.shape == (1, 6)
    assert output.final_state.object_latents.shape == (4, 6)
    assert len(output.predicted_states) == 3
    assert output.action_latents.data_ptr() == action_latents.data_ptr()


def test_planner_rollout_does_not_call_action_encoder(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    model = _build_graph_jepa(parsed, action_encoder=_ExplodingActionEncoder())
    planner = _build_latent_planner(model)
    _, initial_state = planner.encode_graph(graph)

    output = planner.rollout_from_state(
        initial_state,
        torch.randn(1, 3, 6),
    )

    assert output.final_state.graph_latent.shape == (1, 6)
    assert len(output.predicted_states) == 3


def test_planner_rollout_backpropagates_through_predictor(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    model = _build_graph_jepa(parsed)
    planner = _build_latent_planner(model)
    _, initial_state = planner.encode_graph(graph)

    output = planner.rollout_from_state(
        initial_state,
        torch.randn(1, 3, 6),
    )
    loss = output.final_state.graph_latent.sum()
    loss.backward()

    assert any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in model.predictor.parameters())


def test_planner_rollout_rejects_invalid_action_latent_shape(tmp_path: Path) -> None:
    parsed, _, graph = _encoded_graph(tmp_path)
    model = _build_graph_jepa(parsed)
    planner = _build_latent_planner(model)
    _, initial_state = planner.encode_graph(graph)

    with pytest.raises(ValueError, match="action_latents"):
        planner.rollout_from_state(initial_state, torch.randn(1, 6))


class _RecordingCandidateActionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actions: list[dict[str, torch.Tensor]] = []
        self.states: list[JEPALatentState] = []

    def forward(self, action_tensors: dict[str, torch.Tensor], latent_state: JEPALatentState) -> torch.Tensor:
        self.actions.append(action_tensors)
        self.states.append(latent_state)
        return torch.stack((action_tensors["action_id"].float(), latent_state.graph_latent[:, 0]), dim=-1)


class _RecordingCandidatePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.states: list[JEPALatentState] = []

    def forward(self, latent_state: JEPALatentState, action_latent: torch.Tensor) -> JEPALatentState:
        self.states.append(latent_state)
        graph_delta = action_latent[:, :1].expand_as(latent_state.graph_latent)
        object_delta = action_latent[latent_state.object_batch, :1].expand_as(latent_state.object_latents)
        return JEPALatentState(
            graph_latent=latent_state.graph_latent + graph_delta,
            object_latents=latent_state.object_latents + object_delta,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
        )


def _candidate_rollout_fixture(
    tmp_path: Path,
) -> tuple[GraphJEPA, JEPALatentState, dict[str, torch.Tensor]]:
    parsed, _, _ = _encoded_graph(tmp_path)
    model = _build_graph_jepa(
        parsed,
        action_encoder=_RecordingCandidateActionEncoder(),
        predictor=_RecordingCandidatePredictor(),
    )
    initial_state = JEPALatentState(
        graph_latent=torch.zeros(2, 2),
        object_latents=torch.zeros(4, 2),
        object_ids=torch.tensor([0, 1, 0, 1]),
        object_batch=torch.tensor([0, 0, 1, 1]),
    )
    action_tensors = {"action_id": torch.tensor([[1, 2, 3], [4, 5, 6]])}
    return model, initial_state, action_tensors


class _ExplodingActionEncoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("autoregressive rollout must not call q_phi")


class _CountingPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, latent_state: JEPALatentState, action_latent: torch.Tensor) -> JEPALatentState:
        self.call_count += 1
        return JEPALatentState(
            graph_latent=latent_state.graph_latent + action_latent[..., : latent_state.graph_latent.size(-1)],
            object_latents=latent_state.object_latents,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
        )


def _build_graph_jepa(
    parsed,
    action_encoder: nn.Module | None = None,
    state_encoder: nn.Module | None = None,
    predictor: nn.Module | None = None,
) -> GraphJEPA:
    return GraphJEPA(
        graph_encoder=GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2),
        state_encoder=state_encoder
        if state_encoder is not None
        else StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10),
        action_encoder=action_encoder
        if action_encoder is not None
        else ActionEncoder(
            LatentActionEncoder(
                num_actions=len(parsed.actions),
                max_action_arity=parsed.max_action_arity,
                latent_dim=6,
                action_dim=6,
                hidden_dim=10,
            ),
            action_dim=6,
        ),
        predictor=predictor
        if predictor is not None
        else ResidualMLPLatentPredictorG(latent_dim=6, action_dim=6, hidden_dim=10),
        loss_module=GraphJEPALossModule(
            prediction_loss=GraphLatentPredictionLoss(),
            regularization_loss=GraphVCLoss(),
        ),
    )


def _build_latent_planner(model: GraphJEPA) -> LatentMPPIPlanner:
    def goal_energy(_goal_tensors, terminal_state: JEPALatentState) -> torch.Tensor:
        return terminal_state.graph_latent.square().sum(dim=-1)

    return LatentMPPIPlanner(
        graph_encoder=model.graph_encoder,
        state_encoder=model.state_encoder,
        predictor=model.predictor,
        goal_energy=goal_energy,
        config=LatentMPPIConfig(
            horizon=3,
            action_dim=6,
            num_samples=8,
            max_iters=2,
            seed=0,
        ),
    )


def _encoded_graph(tmp_path: Path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)
    parsed = parse_domain_problem(domain_path, problem_path)
    graph = build_state_graph(parsed, parsed.initial_atoms)
    graph_encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)
    return parsed, graph_encoder(graph), graph


def _action_context(graph_output) -> JEPALatentState:
    state_encoder = StateEncoderF(
        embedding_dim=graph_output.graph_embedding.size(-1),
        latent_dim=6,
        hidden_dim=10,
    )
    return state_encoder(graph_output)


def _max_predicate_arity(parsed) -> int:
    return max(len(predicate.arg_types) for predicate in parsed.predicates.values())


def _stack_tensor_dict(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _batch_action_sequence(action_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in action_tensors.items()}


def _single_action_window(action_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.reshape(1, 1, *value.shape) for key, value in action_tensors.items()}


def _temporal_latent_state(*states: JEPALatentState) -> JEPALatentState:
    first = states[0]
    return JEPALatentState(
        graph_latent=torch.stack([state.graph_latent for state in states], dim=1),
        object_latents=torch.stack([state.object_latents for state in states], dim=1),
        object_ids=first.object_ids,
        object_batch=first.object_batch,
    )


def _temporal_graph_output(*graph_outputs):
    first = graph_outputs[0]
    return first.__class__(
        graph_embedding=torch.stack([output.graph_embedding for output in graph_outputs], dim=1),
        object_embeddings=torch.stack([output.object_embeddings for output in graph_outputs], dim=1),
        object_ids=first.object_ids,
        object_batch=first.object_batch,
    )
