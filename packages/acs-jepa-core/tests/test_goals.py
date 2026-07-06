from __future__ import annotations

from pathlib import Path

import torch
from acs_jepa import (
    ConditionalSampleGeneratorLoss,
    ConditionalSampleTerminalLatentGeneratorG,
    DiagonalGaussianTerminalLatentDistributionP,
    DistributionalGoalEnergy,
    GaussianMixtureTerminalLatentDistributionP,
    GeneratedTerminalLatentSamples,
    PartialGoalEncoder,
    PredicateEvaluator,
    SampleSetGoalEnergy,
    build_predicate_evaluator,
)
from acs_jepa.architectures import JEPALatentState, StateEncoderF
from acs_jepa.graph import GraphEncoder, build_state_graph, parse_domain_problem, tensorize_goal_atoms

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


def test_goal_exports_preserve_top_level_predicate_evaluator() -> None:
    evaluator = build_predicate_evaluator(
        num_predicates=1,
        max_predicate_arity=1,
        latent_dim=4,
    )

    assert isinstance(evaluator, PredicateEvaluator)


def test_partial_goal_encoder_handles_batched_goal_tensors(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    goal_tensors = _stack_tensor_dict([tensorize_goal_atoms(parsed, parsed.goal_atoms, max_atoms=2)] * 2)
    latent_state = _random_latent_state(batch_size=2, num_objects=len(parsed.objects), latent_dim=6)
    encoder = PartialGoalEncoder(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        goal_dim=6,
        hidden_dim=8,
    )

    context = encoder(goal_tensors, latent_state)

    assert context.shape == (2, 6)


def test_gaussian_goal_energy_scores_terminal_latents_and_backpropagates(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    goal_tensors = tensorize_goal_atoms(parsed, parsed.goal_atoms)
    terminal_state = _random_latent_state(batch_size=2, num_objects=len(parsed.objects), latent_dim=6)
    model = DiagonalGaussianTerminalLatentDistributionP(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        hidden_dim=8,
    )

    params = model(goal_tensors, terminal_state)
    energy = DistributionalGoalEnergy(model)(goal_tensors, terminal_state)
    energy.sum().backward()

    assert params.graph_mean.shape == (2, 6)
    assert params.object_mean.shape == (8, 6)
    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()
    assert terminal_state.graph_latent.grad is not None
    assert terminal_state.object_latents.grad is not None


def test_gmm_goal_energy_scores_terminal_latents(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    goal_tensors = _stack_tensor_dict([tensorize_goal_atoms(parsed, parsed.goal_atoms)] * 2)
    terminal_state = _random_latent_state(batch_size=2, num_objects=len(parsed.objects), latent_dim=6)
    model = GaussianMixtureTerminalLatentDistributionP(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        num_components=3,
        hidden_dim=8,
    )

    params = model(goal_tensors, terminal_state)
    energy = DistributionalGoalEnergy(model)(goal_tensors, terminal_state)

    assert params.mixture_logits.shape == (2, 3)
    assert params.graph_mean.shape == (2, 3, 6)
    assert params.object_mean.shape == (8, 3, 6)
    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


def test_sample_generator_shapes_and_deterministic_noise(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    goal_tensors = _stack_tensor_dict([tensorize_goal_atoms(parsed, parsed.goal_atoms)] * 2)
    reference_state = _random_latent_state(batch_size=2, num_objects=len(parsed.objects), latent_dim=6)
    generator = ConditionalSampleTerminalLatentGeneratorG(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        num_samples=4,
        noise_dim=5,
        hidden_dim=8,
    )
    eps = torch.zeros(2, 4, 5)

    samples = generator(goal_tensors, reference_state, eps=eps)
    repeated = generator(goal_tensors, reference_state, eps=eps)

    assert samples.graph_latents.shape == (2, 4, 6)
    assert samples.object_latents.shape == (8, 4, 6)
    assert torch.allclose(samples.graph_latents, repeated.graph_latents)
    assert torch.allclose(samples.object_latents, repeated.object_latents)


def test_sample_set_energy_picks_nearest_completion() -> None:
    target_state = _random_latent_state(batch_size=2, num_objects=3, latent_dim=4)
    graph_samples = torch.stack(
        [target_state.graph_latent + 10.0, target_state.graph_latent],
        dim=1,
    )
    object_samples = torch.stack(
        [target_state.object_latents + 10.0, target_state.object_latents],
        dim=1,
    )
    samples = GeneratedTerminalLatentSamples(
        graph_latents=graph_samples,
        object_latents=object_samples,
        object_ids=target_state.object_ids,
        object_batch=target_state.object_batch,
    )

    energy = SampleSetGoalEnergy()(goal_tensors={}, terminal_state=target_state, samples=samples)

    assert torch.allclose(energy, torch.zeros_like(energy))


def test_conditional_sample_generator_loss_backpropagates(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    goal_tensors = _stack_tensor_dict([tensorize_goal_atoms(parsed, parsed.goal_atoms)] * 2)
    target_state = _random_latent_state(batch_size=2, num_objects=len(parsed.objects), latent_dim=6)
    generator = ConditionalSampleTerminalLatentGeneratorG(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        num_samples=3,
        hidden_dim=8,
    )
    loss_module = ConditionalSampleGeneratorLoss(generator)

    loss = loss_module(goal_tensors, target_state, eps=torch.zeros(2, 3, 6))
    loss.backward()

    assert loss.ndim == 0
    assert any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in generator.parameters())


def test_goal_models_score_encoded_terminal_state_integration(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    goal_tensors = tensorize_goal_atoms(parsed, parsed.goal_atoms)
    terminal_graph = build_state_graph(parsed, parsed.goal_atoms)
    graph_encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)
    state_encoder = StateEncoderF(embedding_dim=8, latent_dim=6, hidden_dim=8)
    terminal_state = state_encoder(graph_encoder(terminal_graph))
    gaussian = DiagonalGaussianTerminalLatentDistributionP(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        hidden_dim=8,
    )
    gmm = GaussianMixtureTerminalLatentDistributionP(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        num_components=2,
        hidden_dim=8,
    )
    generator = ConditionalSampleTerminalLatentGeneratorG(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        num_samples=2,
        hidden_dim=8,
    )

    loss = (
        DistributionalGoalEnergy(gaussian)(goal_tensors, terminal_state).sum()
        + DistributionalGoalEnergy(gmm)(goal_tensors, terminal_state).sum()
        + ConditionalSampleGeneratorLoss(generator)(
            goal_tensors,
            terminal_state,
            eps=torch.zeros(1, 2, 6),
        )
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in graph_encoder.parameters())


def _parsed_problem(tmp_path: Path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)
    return parse_domain_problem(domain_path, problem_path)


def _random_latent_state(batch_size: int, num_objects: int, latent_dim: int) -> JEPALatentState:
    object_ids = torch.arange(num_objects).repeat(batch_size)
    object_batch = torch.arange(batch_size).repeat_interleave(num_objects)
    return JEPALatentState(
        graph_latent=torch.randn(batch_size, latent_dim, requires_grad=True),
        object_latents=torch.randn(batch_size * num_objects, latent_dim, requires_grad=True),
        object_ids=object_ids,
        object_batch=object_batch,
    )


def _max_predicate_arity(parsed) -> int:
    return max(len(predicate.arg_types) for predicate in parsed.predicates.values())


def _stack_tensor_dict(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}
