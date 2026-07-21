from __future__ import annotations

from pathlib import Path
from typing import cast

import acs_jepa.training as training_module
import pytest
import torch
from acs_jepa import (
    ActionContrastiveLoss,
    ActionVICRegLoss,
    ApplicabilityHead,
    ApplicabilityLoss,
    ArgumentReconstructionHead,
    ArgumentReconstructionLoss,
    ConditionalSampleTerminalLatentGeneratorG,
    DiagonalGaussianTerminalLatentDistributionP,
    GaussianMixtureTerminalLatentDistributionP,
    JepaTrainer,
    JepaTrainerConfig,
    PredicateEvaluator,
)
from acs_jepa.architectures import (
    ActionEncoder,
    JEPALatentState,
    LatentActionEncoder,
    ResidualMLPLatentPredictorG,
    StateEncoderF,
)
from acs_jepa.graph import (
    GraphEncoder,
    GroundAction,
    PDDLAtomTrajectoryDataset,
    TrajectorySample,
    parse_domain_problem,
)
from acs_jepa.jepa import GraphJEPA, GraphJEPATrainingOutput
from acs_jepa.losses import (
    GraphInverseDynamicsModel,
    GraphJEPALossModule,
    GraphLatentPredictionLoss,
    GraphVCLoss,
)
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
    batch["action_supervision"] = object()
    jepa = _build_graph_jepa(parsed)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam(jepa.parameters(), lr=0.001),
    )

    output = trainer.train_step(batch)

    assert output.goal_loss is None
    assert output.action_vicreg_loss is None
    assert output.action_contrastive_loss is None
    assert output.argument_reconstruction_loss is None
    assert output.applicability_loss is None
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


def test_jepa_trainer_action_vicreg_matches_direct_loss(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    jepa = _build_graph_jepa(parsed)
    loss_module = ActionVICRegLoss(std_coeff=0.7, cov_coeff=0.2, std_margin=0.5)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam(jepa.parameters(), lr=0.001),
        config=JepaTrainerConfig(jepa_loss_weight=0.0, action_vicreg_loss_weight=1.75),
        action_vicreg_loss_module=loss_module,
    )

    output = trainer.eval_step(batch)
    expected = loss_module(output.rollout.action_latents)

    assert output.action_vicreg_loss is not None
    assert torch.allclose(output.action_vicreg_loss, expected.total)
    assert torch.allclose(output.total_loss, 1.75 * expected.total)
    assert torch.allclose(output.terms["action_vicreg_std"], expected.std_penalty)
    assert torch.allclose(output.terms["action_vicreg_covariance"], expected.covariance_penalty)
    assert output.terms["action_vicreg_num_samples"].item() == expected.num_samples
    assert all(not value.requires_grad for value in output.terms.values())


def test_causal_negative_encoding_matches_true_prefix_reference() -> None:
    torch.manual_seed(4)
    encoder = ActionEncoder(
        LatentActionEncoder(
            num_actions=2,
            max_action_arity=2,
            latent_dim=3,
            action_dim=3,
            hidden_dim=5,
        ),
        action_dim=3,
        context_steps=3,
    )
    jepa = type("Jepa", (), {"action_encoder": encoder})()
    observed = JEPALatentState(
        graph_latent=torch.randn(2, 3, 3),
        object_latents=torch.randn(4, 3, 3),
        object_ids=torch.tensor([1, 0, 0, 1]),
        object_batch=torch.tensor([0, 0, 1, 1]),
    )
    actions = {
        "action_id": torch.tensor([[0, 1], [1, 0]]),
        "action_object_indices": torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
        "action_role_ids": torch.tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]]),
        "action_arg_mask": torch.ones(2, 2, 2, dtype=torch.bool),
    }
    positives = encoder(
        actions,
        JEPALatentState(
            graph_latent=observed.graph_latent[:, :2],
            object_latents=observed.object_latents[:, :2],
            object_ids=observed.object_ids,
            object_batch=observed.object_batch,
        ),
    )
    supervision = {
        "negative_action_id": torch.tensor([[[1, 0], [0, 1]], [[0, 0], [1, 0]]]),
        "negative_action_object_indices": torch.tensor(
            [[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]]
        ),
        "negative_action_role_ids": torch.zeros(2, 2, 2, 2, dtype=torch.long),
        "negative_action_arg_mask": torch.ones(2, 2, 2, 2, dtype=torch.bool),
        "negative_mask": torch.tensor([[[True, False], [True, True]], [[False, True], [True, False]]]),
    }
    supervision["negative_action_role_ids"][..., 1] = 1

    encoded, mask = training_module._encode_causal_negative_actions(
        jepa, actions, observed, positives, supervision
    )

    assert torch.equal(mask, supervision["negative_mask"])
    for b, k, m in mask.nonzero(as_tuple=False).tolist():
        candidate = {name: value[b : b + 1, : k + 1].clone() for name, value in actions.items()}
        candidate["action_id"][:, k] = supervision["negative_action_id"][b, k, m]
        candidate["action_object_indices"][:, k] = supervision[
            "negative_action_object_indices"
        ][b, k, m]
        candidate["action_role_ids"][:, k] = supervision["negative_action_role_ids"][b, k, m]
        candidate["action_arg_mask"][:, k] = supervision["negative_action_arg_mask"][b, k, m]
        object_rows = observed.object_batch == b
        state = JEPALatentState(
            graph_latent=observed.graph_latent[b : b + 1, : k + 1],
            object_latents=observed.object_latents[object_rows, : k + 1],
            object_ids=observed.object_ids[object_rows],
            object_batch=torch.zeros(int(object_rows.sum()), dtype=torch.long),
        )
        expected = encoder(candidate, state)[:, -1]
        assert torch.allclose(encoded[b, k, m], expected[0])
    assert torch.equal(encoded[~mask], positives.unsqueeze(2).expand_as(encoded)[~mask])

    changed_prefix = {name: value.clone() for name, value in actions.items()}
    changed_prefix["action_id"][:, 0] = 1 - changed_prefix["action_id"][:, 0]
    prefix_encoded, _ = training_module._encode_causal_negative_actions(
        jepa, changed_prefix, observed, positives, supervision
    )
    later_active = mask[:, 1]
    assert not torch.allclose(encoded[:, 1][later_active], prefix_encoded[:, 1][later_active])

    changed_future = {name: value.clone() for name, value in actions.items()}
    changed_future["action_id"][:, 1] = 1 - changed_future["action_id"][:, 1]
    future_encoded, _ = training_module._encode_causal_negative_actions(
        jepa, changed_future, observed, positives, supervision
    )
    assert torch.allclose(encoded[:, 0][mask[:, 0]], future_encoded[:, 0][mask[:, 0]])

    changed_padding = {name: value.clone() for name, value in supervision.items()}
    changed_padding["negative_action_id"][~mask] = 99
    changed_padding["negative_action_object_indices"][~mask] = 99
    changed_padding["negative_action_role_ids"][~mask] = 99
    padded_encoded, _ = training_module._encode_causal_negative_actions(
        jepa, actions, observed, positives, changed_padding
    )
    assert torch.equal(encoded, padded_encoded)


def test_dense_object_bank_places_permuted_problem_local_object_ids() -> None:
    object_latents = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    observed = JEPALatentState(
        graph_latent=torch.zeros(2, 3, 3),
        object_latents=object_latents,
        object_ids=torch.tensor([2, 1, 0, 0]),
        object_batch=torch.tensor([1, 0, 1, 0]),
    )
    rollout = type(
        "Rollout",
        (),
        {"observed_states": observed, "action_latents": torch.zeros(2, 2, 3)},
    )()
    object_mask = torch.tensor(
        [
            [[True, True, False], [True, True, False]],
            [[True, False, True], [True, False, True]],
        ]
    )

    bank = training_module._dense_source_object_bank(
        cast(GraphJEPATrainingOutput, rollout), {"object_mask": object_mask}
    )

    assert torch.equal(bank[0, :, 0], object_latents[3])
    assert torch.equal(bank[0, :, 1], object_latents[1])
    assert torch.equal(bank[1, :, 0], object_latents[2])
    assert torch.equal(bank[1, :, 2], object_latents[0])
    assert torch.count_nonzero(bank[~object_mask]) == 0


def test_gather_object_latents_rejects_active_padded_object_ids() -> None:
    bank = torch.randn(1, 1, 3, 4)
    indices = torch.tensor([[[2]]], dtype=torch.long)
    argument_mask = torch.tensor([[[True]]])
    object_mask = torch.tensor([[[True, True, False]]])

    with pytest.raises(ValueError, match="represented object"):
        training_module._gather_object_latents(
            bank,
            indices,
            argument_mask,
            object_mask,
        )



def test_jepa_trainer_contrastive_matches_direct_loss_and_skips_empty(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    batch["action_supervision"] = _integrated_supervision()
    jepa = _build_graph_jepa(parsed)
    anchor = GraphInverseDynamicsModel(latent_dim=6, action_dim=6, hidden_dim=8)
    loss_module = ActionContrastiveLoss(temperature=0.4)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam([*jepa.parameters(), *anchor.parameters()], lr=0.001),
        config=JepaTrainerConfig(jepa_loss_weight=0.0, action_contrastive_loss_weight=1.7),
        action_contrastive_anchor=anchor,
        action_contrastive_loss_module=loss_module,
    )

    output = trainer.eval_step(batch)
    negatives, mask = training_module._encode_causal_negative_actions(
        jepa,
        batch["actions"],
        output.rollout.observed_states,
        output.rollout.action_latents,
        batch["action_supervision"],
    )
    source = training_module.latent_time_slice(output.rollout.observed_states, 0, 1)
    target = training_module.latent_time_slice(output.rollout.observed_states, 1, 2)
    expected = loss_module(
        anchor(source, target).reshape(-1, 6),
        output.rollout.action_latents.reshape(-1, 6),
        negatives.reshape(-1, 2, 6),
        mask.reshape(-1, 2),
    )

    assert output.action_contrastive_loss is not None
    assert torch.allclose(output.action_contrastive_loss, expected.total)
    assert torch.allclose(output.total_loss, 1.7 * expected.total)
    assert output.terms["action_contrastive_num_examples"].item() == 1
    assert output.terms["action_contrastive_num_negatives"].item() == 2

    batch["action_supervision"]["negative_applicability_label"].fill_(1.0)
    batch["action_supervision"]["negative_applicability_label_mask"].zero_()
    label_changed = trainer.eval_step(batch)
    assert torch.equal(label_changed.action_contrastive_loss, output.action_contrastive_loss)

    batch["action_supervision"]["negative_mask"].zero_()
    skipped = trainer.eval_step(batch)
    assert skipped.action_contrastive_loss is None
    assert "action_contrastive" not in skipped.terms


def test_jepa_trainer_argument_reconstruction_matches_direct_loss(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    supervision = _integrated_supervision()
    batch["action_supervision"] = supervision
    jepa = _build_graph_jepa(parsed)
    head = ArgumentReconstructionHead(
        action_dim=6, object_dim=6, max_action_arity=4, hidden_dim=8
    )
    loss_module = ArgumentReconstructionLoss()
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam([*jepa.parameters(), *head.parameters()], lr=0.001),
        config=JepaTrainerConfig(
            jepa_loss_weight=0.0, argument_reconstruction_loss_weight=2.3
        ),
        argument_reconstruction_head=head,
        argument_reconstruction_loss_module=loss_module,
    )

    output = trainer.eval_step(batch)
    bank = training_module._dense_source_object_bank(output.rollout, supervision)
    logits = head(
        output.rollout.action_latents.reshape(-1, 6),
        bank.reshape(-1, 4, 6),
        supervision["argument_candidate_mask"].reshape(-1, 4, 4),
    )
    expected = loss_module(
        logits,
        supervision["argument_target_indices"].reshape(-1, 4),
        supervision["argument_mask"].reshape(-1, 4),
        supervision["argument_candidate_mask"].reshape(-1, 4, 4),
    )

    assert output.argument_reconstruction_loss is not None
    assert torch.allclose(output.argument_reconstruction_loss, expected.total)
    assert torch.allclose(output.total_loss, 2.3 * expected.total)
    assert output.terms["argument_num_active_roles"].item() == 4
    assert all(not value.requires_grad for value in output.terms.values())


def test_jepa_trainer_integrated_applicability_uses_one_positive_and_known_negatives(
    tmp_path: Path,
) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    supervision = _integrated_supervision()
    batch["action_supervision"] = supervision
    jepa = _build_graph_jepa(parsed)
    head = _RecordingApplicabilityHead(
        ApplicabilityHead(latent_dim=6, action_dim=6, max_action_arity=4, hidden_dim=8)
    )
    loss_module = _RecordingApplicabilityLoss()
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam([*jepa.parameters(), *head.parameters()], lr=0.001),
        config=JepaTrainerConfig(
            jepa_loss_weight=0.0,
            integrated_applicability_loss_weight=1.9,
            applicability_head_detach=True,
        ),
        applicability_head=head,
        applicability_loss_module=loss_module,
    )

    output = trainer.eval_step(batch)

    assert output.applicability_loss is not None
    assert torch.allclose(output.total_loss, 1.9 * output.applicability_loss)
    assert loss_module.labels is not None
    assert loss_module.labels.tolist() == [1.0, 0.0, 1.0]
    assert head.argument_mask is not None
    assert head.argument_mask.shape == (3, 4)
    supervision["negative_applicability_label_mask"][0, 0, 1] = False
    unknown_skipped = trainer.eval_step(batch)
    assert unknown_skipped.applicability_loss is not None
    assert loss_module.labels is not None
    assert loss_module.labels.tolist() == [1.0, 0.0]
    supervision["negative_applicability_label_mask"].zero_()
    skipped = trainer.eval_step(batch)
    assert skipped.applicability_loss is None
    assert "applicability" not in skipped.terms


def test_jepa_trainer_enabled_action_auxiliaries_backpropagate(tmp_path: Path) -> None:
    parsed, batch = _trainer_batch(tmp_path)
    batch["action_supervision"] = _integrated_supervision()
    jepa = _build_graph_jepa(parsed)
    anchor = GraphInverseDynamicsModel(latent_dim=6, action_dim=6, hidden_dim=8)
    argument_head = ArgumentReconstructionHead(
        action_dim=6, object_dim=6, max_action_arity=4, hidden_dim=8
    )
    applicability_head = ApplicabilityHead(
        latent_dim=6, action_dim=6, max_action_arity=4, hidden_dim=8
    )
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=torch.optim.Adam(
            [
                *jepa.parameters(),
                *anchor.parameters(),
                *argument_head.parameters(),
                *applicability_head.parameters(),
            ],
            lr=0.001,
        ),
        config=JepaTrainerConfig(
            jepa_loss_weight=0.0,
            action_contrastive_loss_weight=1.0,
            argument_reconstruction_loss_weight=1.0,
            integrated_applicability_loss_weight=1.0,
        ),
        action_contrastive_anchor=anchor,
        action_contrastive_loss_module=ActionContrastiveLoss(),
        argument_reconstruction_head=argument_head,
        argument_reconstruction_loss_module=ArgumentReconstructionLoss(),
        applicability_head=applicability_head,
        applicability_loss_module=ApplicabilityLoss(),
    )

    output = trainer.train_step(batch)

    assert torch.isfinite(output.total_loss)
    assert _has_nonzero_grad(anchor)
    assert _has_nonzero_grad(argument_head)
    assert _has_nonzero_grad(applicability_head)
    assert _has_nonzero_grad(jepa.action_encoder)
    assert _has_nonzero_grad(jepa.state_encoder)



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
    batch["action_supervision"] = object()
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
    with pytest.raises(ValueError, match="mutually exclusive"):
        JepaTrainer(
            jepa=jepa,
            optimizer=torch.optim.Adam(jepa.parameters()),
            config=JepaTrainerConfig(
                applicability_loss_weight=1.0,
                integrated_applicability_loss_weight=1.0,
            ),
        )
    with pytest.raises(ValueError, match="action_contrastive_loss_weight must be finite"):
        JepaTrainer(
            jepa=jepa,
            optimizer=torch.optim.Adam(jepa.parameters()),
            config=JepaTrainerConfig(action_contrastive_loss_weight=float("nan")),
        )
    with pytest.raises(ValueError, match="integrated_applicability_loss_weight > 0"):
        JepaTrainer(
            jepa=jepa,
            optimizer=torch.optim.Adam(jepa.parameters()),
            config=JepaTrainerConfig(integrated_applicability_loss_weight=1.0),
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


def _integrated_supervision() -> dict[str, torch.Tensor]:
    return {
        "negative_action_id": torch.zeros(1, 1, 2, dtype=torch.long),
        "negative_action_object_indices": torch.tensor([[[[0, 3, 2, 1], [0, 2, 2, 1]]]]),
        "negative_action_role_ids": torch.tensor([[[[0, 1, 2, 3], [0, 1, 2, 3]]]]),
        "negative_action_arg_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
        "negative_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "negative_category_id": torch.zeros(1, 1, 2, dtype=torch.long),
        "negative_changed_role_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
        "negative_applicability_label": torch.tensor([[[0.0, 1.0]]]),
        "negative_applicability_label_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "argument_target_indices": torch.tensor([[[0, 2, 3, 1]]]),
        "argument_mask": torch.ones(1, 1, 4, dtype=torch.bool),
        "argument_candidate_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        "object_mask": torch.ones(1, 1, 4, dtype=torch.bool),
    }


class _RecordingApplicabilityHead(torch.nn.Module):
    def __init__(self, head: torch.nn.Module) -> None:
        super().__init__()
        self.head = head
        self.argument_mask: torch.Tensor | None = None

    def forward(self, graph, action, objects, argument_mask):
        self.argument_mask = argument_mask.detach().clone()
        return self.head(graph, action, objects, argument_mask)


class _RecordingApplicabilityLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = ApplicabilityLoss()
        self.labels: torch.Tensor | None = None

    def forward(self, logits, labels, example_mask=None):
        self.labels = labels.detach().clone()
        return self.loss(logits, labels, example_mask=example_mask)


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
