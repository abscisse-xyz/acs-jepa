from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import acs_jepa_cli.data as data_module
import pytest
import torch
from acs_jepa.graph import GroundAction, TrajectorySample, parse_domain_problem
from acs_jepa_cli.config import load_config
from acs_jepa_cli.data import LoadedCorpus, load_action_applicability_table, make_torch_dataset
from acs_jepa_cli.modeling import build_model_bundle
from torch_geometric.loader import DataLoader

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


def test_strict_applicability_json_loads_and_configures_dataset(tmp_path: Path) -> None:
    parsed = _parsed(tmp_path)
    action = GroundAction("move", ("box", "p0", "p1"))
    artifact = tmp_path / "labels.json"
    artifact.write_text(
        json.dumps(
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [
                    {
                        "problem_index": 0,
                        "problem_name": parsed.name,
                        "state_atoms": [
                            {"predicate": atom.predicate, "arguments": list(atom.arguments)}
                            for atom in parsed.initial_atoms
                        ],
                        "applicable_actions": [
                            {"name": action.name, "arguments": list(action.arguments)}
                        ],
                    }
                ],
            }
        )
    )
    table = load_action_applicability_table(artifact, (parsed,))
    assert len(table) == 1

    config = load_config(None)
    config.model.goal_head.kind = "none"
    config.model.loss.argument_reconstruction_coeff = 1.0
    config.model.argument_reconstruction_head.kind = "mlp"
    config.data.rollout_steps = 1
    config.model.loss.action_hard_negatives_per_positive = 1
    config.data.action_supervision_seed = 17
    config.data.action_negative_max_attempts_per_category = 4
    config.data.action_applicability_table_path = str(artifact)
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
    assert dataset.action_supervision is not None
    assert dataset.action_supervision.num_negatives == 1
    assert dataset.action_supervision.seed == 17
    assert "action_supervision" in dataset[0]


@pytest.mark.parametrize("goal_kind", ["none", "predicate"])
def test_missing_applicability_state_stays_unknown_for_both_dataset_paths(
    tmp_path: Path,
    goal_kind: str,
) -> None:
    parsed = _parsed(tmp_path)
    action = GroundAction("move", ("box", "p0", "p1"))
    artifact = tmp_path / f"partial-{goal_kind}.json"
    artifact.write_text(
        json.dumps(
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [
                    {
                        "problem_index": 0,
                        "problem_name": parsed.name,
                        "state_atoms": [
                            {"predicate": atom.predicate, "arguments": list(atom.arguments)}
                            for atom in parsed.initial_atoms
                        ],
                        "applicable_actions": [
                            {"name": action.name, "arguments": list(action.arguments)}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus = LoadedCorpus(
        parsed_problems=(parsed,),
        trajectories=(
            TrajectorySample(
                problem_index=0,
                states=(parsed.initial_atoms, parsed.goal_atoms, parsed.initial_atoms),
                actions=(action, action),
                terminal_atoms=parsed.initial_atoms,
            ),
        ),
        records=(),
        dataset_summaries=(),
        malformed_rows=(),
    )
    config = load_config(None)
    config.model.goal_head.kind = goal_kind
    config.model.loss.applicability_coeff = 1.0
    config.model.loss.action_hard_negatives_per_positive = 1
    config.data.rollout_steps = 1
    config.data.action_applicability_table_path = str(artifact)

    dataset = make_torch_dataset(corpus, config)
    missing_state_supervision = dataset[1]["action_supervision"]
    assert not bool(
        missing_state_supervision["negative_applicability_label_mask"].any().item()
    )


def test_enabled_action_auxiliary_bundle_runs_real_eval_step(tmp_path: Path) -> None:
    parsed = _parsed(tmp_path)
    action = GroundAction("move", ("box", "p0", "p1"))
    artifact = tmp_path / "labels-eval.json"
    artifact.write_text(
        json.dumps(
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [
                    {
                        "problem_index": 0,
                        "problem_name": parsed.name,
                        "state_atoms": [
                            {"predicate": atom.predicate, "arguments": list(atom.arguments)}
                            for atom in parsed.initial_atoms
                        ],
                        "applicable_actions": [
                            {"name": action.name, "arguments": list(action.arguments)}
                        ],
                    }
                ],
            }
        )
    )
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
    config = load_config(None)
    config.model.goal_head.kind = "none"
    config.model.applicability_head.kind = "mlp"
    config.model.loss.action_vicreg_coeff = 0.2
    config.model.loss.action_contrastive_coeff = 0.3
    config.model.loss.argument_reconstruction_coeff = 0.4
    config.model.argument_reconstruction_head.kind = "mlp"
    config.model.loss.applicability_coeff = 0.5
    config.data.rollout_steps = 1
    config.model.loss.action_hard_negatives_per_positive = 1
    config.data.action_applicability_table_path = str(artifact)

    dataset = make_torch_dataset(corpus, config)
    bundle = build_model_bundle((parsed,), config, device=torch.device("cpu"))
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    output = bundle.trainer.eval_step(batch)

    assert torch.isfinite(output.total_loss)
    assert output.action_vicreg_loss is not None
    assert output.action_contrastive_loss is not None
    assert output.argument_reconstruction_loss is not None
    assert output.applicability_loss is not None

    relabeled = copy.deepcopy(batch)
    supervision = relabeled["action_supervision"]
    known = supervision["negative_applicability_label_mask"]
    supervision["negative_applicability_label"][known] = 1.0 - supervision[
        "negative_applicability_label"
    ][known]
    relabeled_output = bundle.trainer.eval_step(relabeled)
    assert torch.equal(output.rollout.action_latents, relabeled_output.rollout.action_latents)
    assert torch.equal(output.action_vicreg_loss, relabeled_output.action_vicreg_loss)
    assert torch.equal(output.action_contrastive_loss, relabeled_output.action_contrastive_loss)
    assert torch.equal(
        output.argument_reconstruction_loss,
        relabeled_output.argument_reconstruction_loss,
    )

    trainable_modules = (
        bundle.action_contrastive_anchor,
        bundle.argument_reconstruction_head,
        bundle.applicability_head,
    )
    before = []
    for module in trainable_modules:
        assert module is not None
        before.append([parameter.detach().clone() for parameter in module.parameters()])
    train_output = bundle.trainer.train_step(batch)
    assert torch.isfinite(train_output.total_loss)
    for module, previous in zip(trainable_modules, before, strict=True):
        assert module is not None
        assert any(
            not torch.equal(old, new.detach())
            for old, new in zip(previous, module.parameters(), strict=True)
        )



def test_strict_applicability_json_rejects_duplicate_members_and_values(tmp_path: Path) -> None:
    parsed = _parsed(tmp_path)
    duplicate_member = tmp_path / "duplicate-member.json"
    duplicate_member.write_text(
        '{"semantics":"positive_ground_atoms_closed_world_v1",'
        '"semantics":"positive_ground_atoms_closed_world_v1","entries":[]}'
    )
    with pytest.raises(ValueError, match="duplicate JSON member.*semantics") as duplicate_error:
        load_action_applicability_table(duplicate_member, (parsed,))
    assert str(duplicate_member) in str(duplicate_error.value)

    nested_duplicate = tmp_path / "nested-duplicate-member.json"
    nested_duplicate.write_text(
        '{"semantics":"positive_ground_atoms_closed_world_v1","entries":['
        '{"problem_index":0,"problem_index":0,"problem_name":"tiny-1",'
        '"state_atoms":[],"applicable_actions":[]}]}'
    )
    with pytest.raises(ValueError, match="duplicate JSON member.*problem_index") as nested_error:
        load_action_applicability_table(nested_duplicate, (parsed,))
    assert str(nested_duplicate) in str(nested_error.value)
    assert "artifact.entries[0]" in str(nested_error.value)

    duplicate_atom = tmp_path / "duplicate-atom.json"
    atom = {"predicate": "at", "arguments": ["box", "p0"]}
    duplicate_atom.write_text(
        json.dumps(
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [
                    {
                        "problem_index": 0,
                        "problem_name": parsed.name,
                        "state_atoms": [atom, atom],
                        "applicable_actions": [],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate state atom"):
        load_action_applicability_table(duplicate_atom, (parsed,))

    action = {"name": "move", "arguments": ["box", "p0", "p1"]}
    duplicate_action = tmp_path / "duplicate-action.json"
    duplicate_action.write_text(
        json.dumps(
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [
                    {
                        "problem_index": 0,
                        "problem_name": parsed.name,
                        "state_atoms": [],
                        "applicable_actions": [action, action],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate applicable action"):
        load_action_applicability_table(duplicate_action, (parsed,))

    duplicate_entry = tmp_path / "duplicate-entry.json"
    entry = {
        "problem_index": 0,
        "problem_name": parsed.name,
        "state_atoms": [],
        "applicable_actions": [],
    }
    duplicate_entry.write_text(
        json.dumps(
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [entry, entry],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicates applicability state key"):
        load_action_applicability_table(duplicate_entry, (parsed,))


def test_action_supervision_is_disabled_by_default_and_paths_are_absolute(tmp_path: Path) -> None:
    parsed = _parsed(tmp_path)
    config = load_config(None)
    config.model.goal_head.kind = "none"
    config.data.rollout_steps = 1
    corpus = LoadedCorpus(
        parsed_problems=(parsed,),
        trajectories=(),
        records=(),
        dataset_summaries=(),
        malformed_rows=(),
    )
    dataset = make_torch_dataset(corpus, config)
    assert dataset.action_supervision is None

    with pytest.raises(ValueError, match="must be absolute"):
        load_action_applicability_table(Path("relative.json"), (parsed,))


def test_data_loading_contains_no_simulator_planner_or_online_oracle_calls() -> None:
    source = inspect.getsource(data_module)
    assert "SimulatorEngine" not in source
    assert ".applicable_actions(" not in source
    assert "solve_plan(" not in source


def test_strict_applicability_json_errors_include_path_entry_and_symbol(
    tmp_path: Path,
) -> None:
    parsed = _parsed(tmp_path)
    artifact = tmp_path / "invalid-symbol.json"
    artifact.write_text(
        json.dumps(
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [
                    {
                        "problem_index": 0,
                        "problem_name": parsed.name,
                        "state_atoms": [],
                        "applicable_actions": [
                            {"name": "not-an-action", "arguments": []}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_action_applicability_table(artifact, (parsed,))
    message = str(error.value)
    assert str(artifact) in message
    assert "artifact.entries[0]" in message
    assert "not-an-action" in message


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"semantics": "wrong", "entries": []}, "semantics"),
        ({"semantics": "positive_ground_atoms_closed_world_v1"}, "exactly"),
        (
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": {},
            },
            "entries must be a list",
        ),
        (
            {
                "semantics": "positive_ground_atoms_closed_world_v1",
                "entries": [
                    {
                        "problem_index": True,
                        "problem_name": "tiny-1",
                        "state_atoms": [],
                        "applicable_actions": [],
                    }
                ],
            },
            "problem_index",
        ),
    ],
)
def test_strict_applicability_json_rejects_schema_identity_and_semantics(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    parsed = _parsed(tmp_path)
    artifact = tmp_path / "invalid-schema.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match) as error:
        load_action_applicability_table(artifact, (parsed,))
    assert str(artifact) in str(error.value)


def _parsed(tmp_path: Path):
    domain = tmp_path / "domain.pddl"
    problem = tmp_path / "problem.pddl"
    domain.write_text(DOMAIN)
    problem.write_text(PROBLEM)
    return parse_domain_problem(domain, problem)
