from __future__ import annotations

from pathlib import Path

import pytest
import torch
from acs_jepa import ApplicabilityHead, ArgumentReconstructionHead, GraphInverseDynamicsModel
from acs_jepa.graph import GroundAction, TrajectorySample, parse_domain_problem
from acs_jepa_cli.config import load_config
from acs_jepa_cli.data import LoadedCorpus, make_torch_dataset
from acs_jepa_cli.modeling import build_model_bundle
from omegaconf.errors import InterpolationResolutionError
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (
    ROOT / "script/configs/adaptive/base.yaml",
    ROOT / "script/configs/adaptive/00_smoke/default_smoke.yaml",
    ROOT / "script/configs/adaptive/01_action_decode/action_auxiliary_smoke.yaml",
)

DOMAIN = """
(define (domain tiny)
  (:requirements :typing)
  (:types item place)
  (:predicates (at ?x - item ?p - place) (clear ?p - place))
  (:action move
    :parameters (?x - item ?from - place ?to - place)
    :precondition (and (at ?x ?from) (clear ?to))
    :effect (and (not (at ?x ?from)) (at ?x ?to)))
)
"""

PROBLEM = """
(define (problem tiny-1)
  (:domain tiny)
  (:objects box - item p0 p1 - place)
  (:init (at box p0) (clear p1))
  (:goal (at box p1)))
"""


def test_action_auxiliary_smoke_config_has_exact_fixed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = (tmp_path / "applicability.json").resolve()
    monkeypatch.setenv("ACS_JEPA_ACTION_APPLICABILITY_TABLE", str(artifact))

    config = load_config(list(CONFIGS))

    loss = config.model.loss
    assert loss.action_vicreg_coeff == 0.1
    assert loss.action_vicreg_std_coeff == 1.0
    assert loss.action_vicreg_cov_coeff == 1.0
    assert loss.action_vicreg_std_margin == 1.0
    assert loss.action_sigreg_coeff == 0.0
    assert loss.action_contrastive_coeff == 0.1
    assert loss.action_contrastive_temperature == 0.1
    assert loss.action_hard_negatives_per_positive == 4
    assert loss.argument_reconstruction_coeff == 0.1
    assert loss.applicability_coeff == 0.1
    assert config.model.argument_reconstruction_head.kind == "mlp"
    assert config.model.argument_reconstruction_head.hidden_dim == 64
    assert config.model.argument_reconstruction_head.dropout == 0.0
    assert config.model.applicability_head.kind == "mlp"
    assert config.model.applicability_head.hidden_dim == 64
    assert config.model.applicability_head.dropout == 0.0
    assert config.trainer.applicability_pos_weight == 3.0
    assert config.data.action_supervision_seed == 20260721
    assert config.data.action_negative_max_attempts_per_category == 32
    assert Path(config.data.action_applicability_table_path) == artifact
    assert (
        config.tracking.mlflow_tracking_uri
        == "sqlite:////opt/data/workspace/acs-jepa-runs/acs-jepa-mlflow.db"
    )
    assert config.tracking.tags.stage == "smoke_component_test"
    assert config.tracking.tags.variant == "action_auxiliary"


def test_action_auxiliary_smoke_config_requires_artifact_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ACS_JEPA_ACTION_APPLICABILITY_TABLE", raising=False)
    config = load_config(list(CONFIGS))

    with pytest.raises(InterpolationResolutionError):
        _ = config.data.action_applicability_table_path


def test_action_auxiliary_smoke_stack_runs_real_collated_train_and_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN, encoding="utf-8")
    problem_path.write_text(PROBLEM, encoding="utf-8")
    parsed = parse_domain_problem(domain_path, problem_path)
    artifact = tmp_path / "applicability.json"
    artifact.write_text(
        '{"semantics":"positive_ground_atoms_closed_world_v1","entries":['
        '{"problem_index":0,"problem_name":"tiny-1","state_atoms":['
        '{"predicate":"at","arguments":["box","p0"]},'
        '{"predicate":"clear","arguments":["p1"]}],"applicable_actions":['
        '{"name":"move","arguments":["box","p0","p1"]}]}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("ACS_JEPA_ACTION_APPLICABILITY_TABLE", str(artifact.resolve()))
    config = load_config(list(CONFIGS))
    config.model.goal_head.kind = "none"
    config.data.rollout_steps = 1
    action = GroundAction("move", ("box", "p0", "p1"))
    corpus = LoadedCorpus(
        parsed_problems=(parsed,),
        trajectories=(
            TrajectorySample(
                problem_index=0,
                states=(parsed.initial_atoms, parsed.goal_atoms),
                actions=(action,),
                terminal_atoms=parsed.goal_atoms,
            ),
        ),
        records=(),
        dataset_summaries=(),
        malformed_rows=(),
    )
    dataset = make_torch_dataset(corpus, config)
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))

    assert isinstance(bundle.action_contrastive_anchor, GraphInverseDynamicsModel)
    assert isinstance(bundle.argument_reconstruction_head, ArgumentReconstructionHead)
    assert isinstance(bundle.applicability_head, ApplicabilityHead)
    optimizer_parameter_ids = [
        id(parameter)
        for group in bundle.optimizer.param_groups
        for parameter in group["params"]
    ]
    auxiliary_parameter_ids = [
        id(parameter)
        for module in (
            bundle.action_contrastive_anchor,
            bundle.argument_reconstruction_head,
            bundle.applicability_head,
        )
        if module is not None
        for parameter in module.parameters()
    ]
    assert auxiliary_parameter_ids
    assert all(
        optimizer_parameter_ids.count(parameter_id) == 1
        for parameter_id in auxiliary_parameter_ids
    )
    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))
    assert dataset.action_supervision is not None
    assert dataset.action_supervision.num_negatives == 4
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    train_output = bundle.trainer.train_step(batch)
    eval_output = bundle.trainer.eval_step(batch)
    for output in (train_output, eval_output):
        assert torch.isfinite(output.total_loss)
        assert output.action_vicreg_loss is not None
        assert output.action_contrastive_loss is not None
        assert output.argument_reconstruction_loss is not None
        assert output.applicability_loss is not None
        assert output.terms["applicability_num_positive"].item() > 0
        assert output.terms["applicability_num_negative"].item() > 0
