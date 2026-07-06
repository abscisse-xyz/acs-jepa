from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from acs_jepa.architectures import (
    ActionEncoder,
    GRULatentPredictorG,
    GraphStateProjector,
    ActionDecodingSpace,
    ActionDecoder,
    ActionSamplingFamily,
    JEPALatentState,
    LatentActionEncoder,
    ResidualMLPLatentPredictorG,
    StateEncoderF,
    build_action_encoder,
    build_latent_predictor,
)
from acs_jepa.goals import PredicateEvaluator, build_predicate_evaluator
from acs_jepa.graph import (
    GroundAction,
    GraphEncoder,
    build_state_graph,
    parse_domain_problem,
    tensorize_action,
    tensorize_predicate,
)
from acs_jepa.jepa import GraphJEPA
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
    action_a = action_encoder(tensorize_action(parsed, GroundAction("move", ("car0", "j0", "j1", "road0"))), latent_state)
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
        state_encoder=state_encoder if state_encoder is not None else StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=10),
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
        predictor=predictor if predictor is not None else ResidualMLPLatentPredictorG(latent_dim=6, action_dim=6, hidden_dim=10),
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
