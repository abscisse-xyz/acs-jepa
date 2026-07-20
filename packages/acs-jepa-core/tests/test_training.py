from __future__ import annotations

from pathlib import Path

import pytest
import torch
from acs_jepa import (
    ApplicabilityHead,
    ApplicabilityLoss,
    ConditionalSampleTerminalLatentGeneratorG,
    DiagonalGaussianTerminalLatentDistributionP,
    GaussianMixtureTerminalLatentDistributionP,
    JepaTrainer,
    JepaTrainerConfig,
    PredicateEvaluator,
)
from acs_jepa.architectures import ActionEncoder, LatentActionEncoder, ResidualMLPLatentPredictorG, StateEncoderF
from acs_jepa.graph import (
    GraphEncoder,
    GroundAction,
    PDDLAtomTrajectoryDataset,
    TrajectorySample,
    parse_domain_problem,
)
from acs_jepa.jepa import GraphJEPA
from acs_jepa.losses import GraphJEPALossModule, GraphLatentPredictionLoss, GraphVCLoss
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


def test_jepa_trainer_runs_jepa_only_step(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam(jepa.parameters(), lr=0.001),
    )

    output = trainer.train_step(batch)

    assert output.goal_loss is None
    assert output.total_loss.ndim == 0
    assert torch.isfinite(output.total_loss)
    assert "prediction" in output.terms


def test_jepa_trainer_eval_step_does_not_update_parameters_or_gradients(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam(jepa.parameters(), lr=0.001),
    )
    before = [parameter.detach().clone() for parameter in jepa.parameters()]

    output = trainer.eval_step(batch)

    assert output.goal_loss is None
    assert output.total_loss.ndim == 0
    assert torch.isfinite(output.total_loss)
    assert not jepa.training
    assert all(parameter.grad is None for parameter in jepa.parameters())
    assert all(torch.equal(parameter, initial) for parameter, initial in zip(jepa.parameters(), before))


def test_jepa_trainer_predicate_goal_uses_detached_target_state(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    evaluator = _predicate_evaluator(parsed)
    trainer = JepaTrainer(
        jepa=jepa,
        goal_head=evaluator,
        optimizer=torch.optim.Adam([*jepa.parameters(), *evaluator.parameters()], lr=0.001),
        config=JepaTrainerConfig(
            goal_head_kind="predicate",
            jepa_loss_weight=0.0,
            goal_head_detach=True,
        ),
    )

    output = trainer.train_step(batch)

    assert output.goal_loss is not None
    assert torch.isfinite(output.goal_loss)
    assert not _has_nonzero_grad(jepa)
    assert _has_nonzero_grad(evaluator)


def test_jepa_trainer_predicate_goal_can_train_jointly(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    evaluator = _predicate_evaluator(parsed)
    trainer = JepaTrainer(
        jepa=jepa,
        goal_head=evaluator,
        optimizer=torch.optim.Adam([*jepa.parameters(), *evaluator.parameters()], lr=0.001),
        config=JepaTrainerConfig(
            goal_head_kind="predicate",
            jepa_loss_weight=0.0,
            goal_head_detach=False,
        ),
    )

    output = trainer.train_step(batch)

    assert output.goal_loss is not None
    assert _has_nonzero_grad(jepa)
    assert _has_nonzero_grad(evaluator)


def test_jepa_trainer_runs_distributional_goal_steps(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    for kind, goal_head in (
        (
            "gaussian",
            DiagonalGaussianTerminalLatentDistributionP(
                num_predicates=len(parsed.predicates),
                max_predicate_arity=_max_predicate_arity(parsed),
                latent_dim=6,
                hidden_dim=8,
            ),
        ),
        (
            "gmm",
            GaussianMixtureTerminalLatentDistributionP(
                num_predicates=len(parsed.predicates),
                max_predicate_arity=_max_predicate_arity(parsed),
                latent_dim=6,
                num_components=2,
                hidden_dim=8,
            ),
        ),
    ):
        jepa = _build_graph_jepa(parsed)
        trainer = JepaTrainer(
            jepa=jepa,
            goal_head=goal_head,
            optimizer=torch.optim.Adam([*jepa.parameters(), *goal_head.parameters()], lr=0.001),
            config=JepaTrainerConfig(goal_head_kind=kind, jepa_loss_weight=0.0),
        )

        output = trainer.train_step(batch)

        assert output.goal_loss is not None
        assert torch.isfinite(output.goal_loss)
        assert _has_nonzero_grad(goal_head)


def test_jepa_trainer_runs_conditional_sampler_goal_step(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    generator = ConditionalSampleTerminalLatentGeneratorG(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        num_samples=2,
        hidden_dim=8,
    )
    trainer = JepaTrainer(
        jepa=jepa,
        goal_head=generator,
        optimizer=torch.optim.Adam([*jepa.parameters(), *generator.parameters()], lr=0.001),
        config=JepaTrainerConfig(goal_head_kind="conditional_sampler", jepa_loss_weight=0.0),
    )

    output = trainer.train_step(batch)

    assert output.goal_loss is not None
    assert torch.isfinite(output.goal_loss)
    assert _has_nonzero_grad(generator)


def test_jepa_trainer_applicability_disabled_preserves_jepa_only_behavior(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    head = ApplicabilityHead(latent_dim=6, action_dim=6, max_action_arity=parsed.max_action_arity, hidden_dim=8)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam([*jepa.parameters(), *head.parameters()], lr=0.001),
        config=JepaTrainerConfig(applicability_loss_weight=0.0),
        applicability_head=head,
        applicability_loss_module=ApplicabilityLoss(),
    )

    output = trainer.train_step(batch)

    assert output.applicability_loss is None
    assert "applicability" not in output.terms
    assert torch.isfinite(output.total_loss)


def test_jepa_trainer_applicability_loss_adds_detached_auxiliary_terms(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    batch.update(_applicability_batch(requires_grad=True))
    jepa = _build_graph_jepa(parsed)
    head = ApplicabilityHead(latent_dim=6, action_dim=6, max_action_arity=4, hidden_dim=8)
    loss_module = ApplicabilityLoss()
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam([*jepa.parameters(), *head.parameters()], lr=0.001),
        config=JepaTrainerConfig(jepa_loss_weight=0.0, applicability_loss_weight=2.0),
        applicability_head=head,
        applicability_loss_module=loss_module,
    )

    output = trainer.train_step(batch)

    assert output.goal_loss is None
    assert output.applicability_loss is not None
    assert torch.allclose(output.total_loss, 2.0 * output.applicability_loss)
    assert "applicability" in output.terms
    assert "applicability_bce" in output.terms
    assert "applicability_positive_logit_mean" in output.terms
    assert "applicability_negative_logit_mean" in output.terms
    assert "applicability_positive_negative_margin" in output.terms
    assert all(not value.requires_grad for value in output.terms.values())
    assert _has_nonzero_grad(head)
    assert batch["applicability_graph_latents"].grad is None
    assert batch["applicability_action_latents"].grad is None
    assert batch["applicability_object_latents"].grad is None


def test_jepa_trainer_applicability_loss_can_backpropagate_to_attached_inputs(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    batch.update(_applicability_batch(requires_grad=True))
    jepa = _build_graph_jepa(parsed)
    head = ApplicabilityHead(latent_dim=6, action_dim=6, max_action_arity=4, hidden_dim=8)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam([*jepa.parameters(), *head.parameters()], lr=0.001),
        config=JepaTrainerConfig(
            jepa_loss_weight=0.0,
            applicability_loss_weight=1.0,
            applicability_head_detach=False,
        ),
        applicability_head=head,
        applicability_loss_module=ApplicabilityLoss(),
    )

    output = trainer.train_step(batch)

    assert output.applicability_loss is not None
    assert batch["applicability_graph_latents"].grad is not None
    assert batch["applicability_action_latents"].grad is not None
    assert batch["applicability_object_latents"].grad is not None


def test_jepa_trainer_applicability_validation_and_required_keys(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    head = ApplicabilityHead(latent_dim=6, action_dim=6, max_action_arity=4, hidden_dim=8)
    with pytest.raises(ValueError, match="applicability_loss_weight must be non-negative"):
        JepaTrainer(
            jepa=jepa,
            optimizer=torch.optim.Adam(jepa.parameters()),
            config=JepaTrainerConfig(applicability_loss_weight=-1.0),
        )
    with pytest.raises(ValueError, match="applicability_head and applicability_loss_module are required"):
        JepaTrainer(
            jepa=jepa,
            optimizer=torch.optim.Adam(jepa.parameters()),
            config=JepaTrainerConfig(applicability_loss_weight=1.0),
        )
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam([*jepa.parameters(), *head.parameters()], lr=0.001),
        config=JepaTrainerConfig(jepa_loss_weight=0.0, applicability_loss_weight=1.0),
        applicability_head=head,
        applicability_loss_module=ApplicabilityLoss(),
    )
    with pytest.raises(KeyError, match="applicability_graph_latents"):
        trainer.train_step(batch)


def _trainer_batch(tmp_path: Path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)
    parsed = parse_domain_problem(domain_path, problem_path)
    trajectory = TrajectorySample(
        problem_index=0,
        states=(parsed.initial_atoms, parsed.goal_atoms),
        actions=(GroundAction("move", ("car0", "j0", "j1", "road0")),),
        terminal_atoms=parsed.goal_atoms,
    )
    dataset = PDDLAtomTrajectoryDataset(
        [parsed],
        [trajectory],
        num_positive_atoms=2,
        num_negative_atoms=2,
        include_goal=True,
        include_terminal_state=True,
        seed=5,
    )
    return parsed, next(iter(DataLoader(dataset, batch_size=1)))


def _applicability_batch(*, requires_grad: bool) -> dict[str, torch.Tensor]:
    return {
        "applicability_graph_latents": torch.randn(3, 6, requires_grad=requires_grad),
        "applicability_action_latents": torch.randn(3, 6, requires_grad=requires_grad),
        "applicability_object_latents": torch.randn(3, 4, 6, requires_grad=requires_grad),
        "applicability_argument_mask": torch.tensor(
            [[True, True, False, False], [True, False, True, False], [True, True, True, False]]
        ),
        "applicability_labels": torch.tensor([1.0, 0.0, 1.0]),
        "applicability_example_mask": torch.tensor([True, True, True]),
    }


def _build_graph_jepa(parsed) -> GraphJEPA:
    return GraphJEPA(
        graph_encoder=GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2),
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
        ),
    )


def _predicate_evaluator(parsed) -> PredicateEvaluator:
    return PredicateEvaluator(
        num_predicates=len(parsed.predicates),
        max_predicate_arity=_max_predicate_arity(parsed),
        latent_dim=6,
        hidden_dim=8,
    )


def _max_predicate_arity(parsed) -> int:
    return max(len(predicate.arg_types) for predicate in parsed.predicates.values())


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in module.parameters())
