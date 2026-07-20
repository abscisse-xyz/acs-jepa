from __future__ import annotations

from pathlib import Path

import pytest
import torch
from acs_jepa import ApplicabilityHead, ApplicabilityLoss
from acs_jepa.graph import parse_domain_problem
from acs_jepa_cli.config import load_config
from acs_jepa_cli.modeling import build_model_bundle

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


def test_build_model_bundle_default_has_no_applicability_modules(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(tmp_path)

    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))

    assert bundle.applicability_head is None
    assert bundle.applicability_loss_module is None
    assert bundle.trainer.config.applicability_loss_weight == 0.0
    assert bundle.trainer.applicability_head is None
    assert bundle.trainer.applicability_loss_module is None
    assert _optimizer_parameter_ids(bundle.optimizer) == _module_parameter_ids(bundle.jepa, bundle.goal_head)


def test_build_model_bundle_constructs_enabled_applicability_modules(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
model:
  applicability_head:
    kind: mlp
    hidden_dim: 9
    dropout: 0.1
trainer:
  applicability_loss_weight: 0.5
  applicability_head_detach: false
  applicability_pos_weight: 2.0
""",
    )

    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))

    assert isinstance(bundle.applicability_head, ApplicabilityHead)
    assert isinstance(bundle.applicability_loss_module, ApplicabilityLoss)
    assert bundle.trainer.applicability_head is bundle.applicability_head
    assert bundle.trainer.applicability_loss_module is bundle.applicability_loss_module
    assert bundle.trainer.config.applicability_loss_weight == 0.5
    assert bundle.trainer.config.applicability_head_detach is False
    assert _module_parameter_ids(bundle.applicability_head) <= _optimizer_parameter_ids(bundle.optimizer)


def test_build_model_bundle_rejects_invalid_applicability_config(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)

    with pytest.raises(ValueError, match="Unknown applicability head kind"):
        build_model_bundle(
            (parsed,),
            _load_small_config(
                tmp_path,
                extra="""
model:
  applicability_head:
    kind: bogus
""",
            ),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="dropout"):
        build_model_bundle(
            (parsed,),
            _load_small_config(
                tmp_path,
                extra="""
model:
  applicability_head:
    kind: mlp
    dropout: -0.1
""",
            ),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="pos_weight must be positive"):
        build_model_bundle(
            (parsed,),
            _load_small_config(
                tmp_path,
                extra="""
trainer:
  applicability_pos_weight: 0.0
""",
            ),
            device=torch.device("cpu"),
        )


def test_build_model_bundle_requires_head_when_applicability_weight_is_positive(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
trainer:
  applicability_loss_weight: 0.5
""",
    )

    with pytest.raises(ValueError, match="applicability_loss_weight > 0 requires"):
        build_model_bundle((parsed,), config, device=torch.device("cpu"))


def _parsed_problem(tmp_path: Path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)
    return parse_domain_problem(domain_path, problem_path)


def _load_small_config(tmp_path: Path, *, extra: str = ""):
    path = tmp_path / "config-base.yaml"
    path.write_text(
        """
model:
  graph_hidden_dim: 8
  graph_embed_dim: 8
  graph_layers: 1
  latent_dim: 6
  action_dim: 6
  action_encoder:
    hidden_dim: 8
  predictor:
    hidden_dim: 8
  loss:
    regularization_coeff: 0.0
    similarity_coeff: 0.0
    inverse_dynamics_coeff: 0.0
  goal_head:
    kind: gmm
    hidden_dim: 8
    num_components: 2
trainer:
  goal_loss_weight: 0.25
optimizer:
  scheduler:
    kind: none
"""
    )
    if not extra:
        return load_config(path)
    extra_path = tmp_path / f"config-extra-{abs(hash(extra))}.yaml"
    extra_path.write_text(extra)
    return load_config([path, extra_path])


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def _module_parameter_ids(*modules: torch.nn.Module | None) -> set[int]:
    return {
        id(parameter)
        for module in modules
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    }
