from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import acs_jepa_cli.cli as cli
import pytest
import torch
from acs_jepa import (
    ActionContrastiveLoss,
    ActionVICRegLoss,
    ApplicabilityHead,
    ApplicabilityLoss,
    ArgumentReconstructionHead,
    ArgumentReconstructionLoss,
    GraphInverseDynamicsModel,
)
from acs_jepa.graph import parse_domain_problem
from acs_jepa_cli.config import load_config
from acs_jepa_cli.modeling import build_model_bundle, compute_vocab_sizes

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
    assert bundle.action_vicreg_loss_module is None
    assert bundle.action_contrastive_loss_module is None
    assert bundle.action_contrastive_anchor is None
    assert bundle.argument_reconstruction_head is None
    assert bundle.argument_reconstruction_loss_module is None
    assert bundle.trainer.config.applicability_loss_weight == 0.0
    assert bundle.trainer.config.integrated_applicability_loss_weight == 0.0
    assert bundle.trainer.config.action_vicreg_loss_weight == 0.0
    assert bundle.trainer.config.action_contrastive_loss_weight == 0.0
    assert bundle.trainer.config.argument_reconstruction_loss_weight == 0.0
    assert config.model.loss.action_vicreg_std_coeff == 1.0
    assert config.model.loss.action_vicreg_cov_coeff == 1.0
    assert config.model.loss.action_vicreg_std_margin == 1.0
    assert config.model.loss.action_contrastive_temperature == 0.1
    assert config.model.loss.action_hard_negatives_per_positive == 0
    assert config.model.argument_reconstruction_head.kind == "none"
    assert bundle.trainer.applicability_head is None
    assert bundle.trainer.applicability_loss_module is None
    assert _optimizer_parameter_ids(bundle.optimizer) == _module_parameter_ids(bundle.jepa, bundle.goal_head)


def test_build_model_bundle_does_not_construct_zero_weight_applicability_head(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
model:
  applicability_head:
    kind: mlp
""",
    )
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))

    assert bundle.applicability_head is None
    assert bundle.applicability_loss_module is None



def test_build_model_bundle_constructs_action_auxiliary_modules(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    table_path = tmp_path / "applicability.json"
    table_path.write_text("{}")
    config = _load_small_config(
        tmp_path,
        extra=f"""
model:
  loss:
    action_vicreg_coeff: 0.2
    action_contrastive_coeff: 0.3
    action_hard_negatives_per_positive: 2
    argument_reconstruction_coeff: 0.4
    applicability_coeff: 0.5
  argument_reconstruction_head:
    kind: mlp
  applicability_head:
    kind: mlp
data:
  action_applicability_table_path: {table_path}
""",
    )

    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))

    assert isinstance(bundle.action_vicreg_loss_module, ActionVICRegLoss)
    assert isinstance(bundle.action_contrastive_loss_module, ActionContrastiveLoss)
    assert isinstance(bundle.action_contrastive_anchor, GraphInverseDynamicsModel)
    assert isinstance(bundle.argument_reconstruction_head, ArgumentReconstructionHead)
    assert isinstance(bundle.argument_reconstruction_loss_module, ArgumentReconstructionLoss)
    assert isinstance(bundle.applicability_head, ApplicabilityHead)
    assert bundle.trainer.config.integrated_applicability_loss_weight == 0.5
    assert _module_parameter_ids(
        bundle.action_contrastive_anchor,
        bundle.argument_reconstruction_head,
        bundle.applicability_head,
    ) <= _optimizer_parameter_ids(bundle.optimizer)
    assert _optimizer_parameter_ids(bundle.optimizer) == _module_parameter_ids(
        bundle.jepa,
        bundle.goal_head,
        bundle.action_contrastive_anchor,
        bundle.argument_reconstruction_head,
        bundle.applicability_head,
    )



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


def test_build_model_bundle_rejects_action_auxiliary_config_conflicts(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    table_path = tmp_path / "labels.json"
    table_path.write_text("{}")
    invalid_cases = [
        (
            f"""
model:
  loss:
    applicability_coeff: 0.2
    action_hard_negatives_per_positive: 1
  applicability_head:
    kind: mlp
trainer:
  applicability_loss_weight: 0.3
data:
  action_applicability_table_path: {table_path}
""",
            "mutually exclusive",
        ),
        (
            """
model:
  loss:
    action_sigreg_coeff: 0.1
""",
            "action_sigreg_coeff.*must remain zero",
        ),
        (
            """
model:
  loss:
    action_contrastive_coeff: 0.1
""",
            "require action hard negatives",
        ),
        (
            """
model:
  loss:
    applicability_coeff: 0.1
    action_hard_negatives_per_positive: 1
  applicability_head:
    kind: mlp
data:
  action_applicability_table_path: relative.json
""",
            "must be absolute",
        ),
        (
            """
model:
  loss:
    action_hard_negatives_per_positive: true
""",
            "non-negative integer",
        ),
        (
            """
model:
  loss:
    action_vicreg_std_margin: 0.0
""",
            "action_vicreg_std_margin.*positive",
        ),
        (
            """
model:
  loss:
    action_vicreg_std_coeff: -1.0
""",
            "action_vicreg_std_coeff.*non-negative",
        ),
        (
            """
model:
  loss:
    action_contrastive_temperature: 0.0
""",
            "action_contrastive_temperature.*positive",
        ),
        (
            """
model:
  loss:
    argument_reconstruction_coeff: 0.1
""",
            "requires an enabled argument head",
        ),
        (
            """
model:
  argument_reconstruction_head:
    dropout: -0.1
""",
            "argument reconstruction head dropout",
        ),
    ]
    for extra, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            build_model_bundle(
                (parsed,),
                _load_small_config(tmp_path, extra=extra),
                device=torch.device("cpu"),
            )



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


def test_argument_reconstruction_requires_nonempty_object_vocabulary(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
model:
  loss:
    argument_reconstruction_coeff: 0.5
  argument_reconstruction_head:
    kind: mlp
""",
    )
    empty_vocab = replace(compute_vocab_sizes((parsed,)), num_objects=0)

    with pytest.raises(ValueError, match="at least one object"):
        build_model_bundle(
            (parsed,),
            config,
            device=torch.device("cpu"),
            vocab_sizes=empty_vocab,
        )


def test_checkpoint_saves_and_restores_applicability_head_state(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
model:
  applicability_head:
    kind: mlp
trainer:
  applicability_loss_weight: 0.5
""",
    )
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoint.pt"

    cli._save_checkpoint(checkpoint_path, bundle, config, epoch=0, step=3, best_eval=1.5)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["applicability_head_state_dict"] is not None
    assert checkpoint["applicability_head_state_dict"].keys() == bundle.applicability_head.state_dict().keys()

    restored = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in restored.applicability_head.parameters():
            parameter.add_(10.0)

    cli._load_checkpoint_state(restored, checkpoint)

    for name, value in bundle.applicability_head.state_dict().items():
        assert torch.allclose(value, restored.applicability_head.state_dict()[name])


def test_checkpoint_saves_and_restores_action_auxiliary_modules_without_applicability(
    tmp_path: Path,
) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
model:
  loss:
    action_contrastive_coeff: 0.3
    action_hard_negatives_per_positive: 1
    argument_reconstruction_coeff: 0.4
  argument_reconstruction_head:
    kind: mlp
""",
    )
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    checkpoint_path = tmp_path / "action-auxiliary.pt"
    cli._save_checkpoint(checkpoint_path, bundle, config, epoch=0, step=2, best_eval=1.0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["action_contrastive_anchor_state_dict"] is not None
    assert checkpoint["argument_reconstruction_head_state_dict"] is not None

    restored = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    assert bundle.action_contrastive_anchor is not None
    assert bundle.argument_reconstruction_head is not None
    assert restored.action_contrastive_anchor is not None
    assert restored.argument_reconstruction_head is not None
    with torch.no_grad():
        for module in (
            restored.action_contrastive_anchor,
            restored.argument_reconstruction_head,
        ):
            for parameter in module.parameters():
                parameter.add_(10.0)
    cli._load_checkpoint_state(restored, checkpoint)
    for source, target in (
        (bundle.action_contrastive_anchor, restored.action_contrastive_anchor),
        (bundle.argument_reconstruction_head, restored.argument_reconstruction_head),
    ):
        for name, value in source.state_dict().items():
            assert torch.allclose(value, target.state_dict()[name])



def test_checkpoint_saves_disabled_applicability_state_as_none(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(tmp_path)
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoint.pt"

    cli._save_checkpoint(checkpoint_path, bundle, config, epoch=0, step=3, best_eval=1.5)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["applicability_head_state_dict"] is None
    assert checkpoint["action_contrastive_anchor_state_dict"] is None
    assert checkpoint["argument_reconstruction_head_state_dict"] is None


def test_checkpoint_loader_warns_but_allows_old_missing_applicability_state(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
model:
  applicability_head:
    kind: mlp
trainer:
  applicability_loss_weight: 0.5
""",
    )
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    checkpoint = {
        "model_state_dict": bundle.jepa.state_dict(),
        "goal_head_state_dict": bundle.goal_head.state_dict(),
    }

    with pytest.warns(UserWarning, match="applicability_head_state_dict"):
        cli._load_checkpoint_state(bundle, checkpoint)


def test_checkpoint_loader_warns_for_old_missing_action_auxiliary_states(tmp_path: Path) -> None:
    parsed = _parsed_problem(tmp_path)
    config = _load_small_config(
        tmp_path,
        extra="""
model:
  loss:
    action_contrastive_coeff: 0.3
    action_hard_negatives_per_positive: 1
    argument_reconstruction_coeff: 0.4
  argument_reconstruction_head:
    kind: mlp
""",
    )
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    assert bundle.goal_head is not None
    checkpoint = {
        "model_state_dict": bundle.jepa.state_dict(),
        "goal_head_state_dict": bundle.goal_head.state_dict(),
    }
    with pytest.warns(UserWarning) as records:
        cli._load_checkpoint_state(bundle, checkpoint)
    messages = "\n".join(str(record.message) for record in records)
    assert "action_contrastive_anchor_state_dict" in messages
    assert "argument_reconstruction_head_state_dict" in messages



def test_cmd_eval_routes_checkpoint_loading_through_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_dir = tmp_path / "eval"
    checkpoint = {
        "config": {},
        "vocab_sizes": {
            "num_types": 1,
            "num_predicates": 1,
            "num_objects": 1,
            "num_actions": 1,
            "max_arity": 1,
            "max_action_arity": 1,
            "max_predicate_arity": 1,
        },
    }
    calls: list[object] = []
    fake_bundle = SimpleNamespace(trainer=object())
    fake_corpus = SimpleNamespace(parsed_problems=(), trajectories=(), records=())

    checkpoint_path.write_bytes(b"placeholder")
    monkeypatch.setattr(cli.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(cli, "save_resolved_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_corpus", lambda *args, **kwargs: fake_corpus)
    monkeypatch.setattr(cli, "make_torch_dataset", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(cli, "build_model_bundle", lambda *args, **kwargs: fake_bundle)
    monkeypatch.setattr(cli, "_load_checkpoint_state", lambda bundle, loaded: calls.append((bundle, loaded)))
    monkeypatch.setattr(cli, "_evaluate", lambda *args, **kwargs: {"total_loss": 1.0})
    monkeypatch.setattr(cli, "_runtime_metrics", lambda *args, **kwargs: {"seconds": 1.0})
    monkeypatch.setattr(cli, "configure_mlflow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "log_config_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "corpus_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli.mlflow, "log_artifact", lambda *args, **kwargs: None)

    assert cli.cmd_eval(
        SimpleNamespace(dataset_dirs=[tmp_path], checkpoint=checkpoint_path, output=output_dir, device="cpu")
    ) == 0

    assert calls == [(fake_bundle, checkpoint)]


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
