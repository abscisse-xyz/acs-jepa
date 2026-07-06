from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from acs_jepa.graph import (
    GroundAtom,
    GroundAction,
    PDDLAtomTrajectoryDataset,
    PDDLGraphDataset,
    PDDLTrajectoryDataset,
    TrajectorySample,
    build_state_graph,
    parse_domain_problem,
    tensorize_action,
    tensorize_goal_atoms,
    tensorize_predicate,
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


def test_parse_domain_problem_extracts_symbols(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)

    parsed = parse_domain_problem(domain_path, problem_path)

    assert parsed.types == ("car", "junction", "road")
    assert sorted(parsed.predicates) == ["at_car_jun", "clear", "road_connect", "same_line"]
    assert parsed.predicates["road_connect"].arg_types == ("road", "junction", "junction")
    assert parsed.objects["car0"].type == "car"
    assert parsed.objects["j0"].type == "junction"
    assert {atom.predicate for atom in parsed.initial_atoms} == {
        "same_line",
        "clear",
        "at_car_jun",
        "road_connect",
    }
    assert parsed.goal_atoms[0].predicate == "at_car_jun"
    assert parsed.actions["move"].arity == 4
    assert parsed.static_predicates == frozenset({"same_line"})


def test_build_state_graph_uses_factor_nodes_for_n_ary_atoms(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)

    graph = build_state_graph(parsed, parsed.initial_atoms)

    assert graph.num_objects == 4
    assert graph.num_atoms == 4
    assert graph.x.shape == (8, 5)
    assert graph.edge_index.shape == (2, 16)
    assert graph.edge_attr.shape == (16, 2)
    assert "atom_is_static" not in graph.keys()

    road_connect_id = parsed.predicate_to_id["road_connect"]
    road_connect_node = int(torch.where(graph.predicate_id == road_connect_id)[0][0])
    incident = graph.edge_index[0] == road_connect_node
    role_ids = sorted(graph.edge_attr[incident, 0].tolist())
    directions = graph.edge_attr[incident, 1].tolist()

    assert role_ids == [0, 1, 2]
    assert directions == [0, 0, 0]


def test_build_state_graph_can_drop_static_atoms(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)

    graph = build_state_graph(parsed, parsed.initial_atoms, include_static=False)

    assert graph.num_atoms == 3
    assert parsed.predicate_to_id["same_line"] not in graph.atom_predicate_id.tolist()


def test_tensorize_action_uses_action_and_argument_roles(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)

    tensors = tensorize_action(
        parsed,
        GroundAction("move", ("car0", "j0", "j1", "road0")),
    )

    assert tensors["action_id"].item() == parsed.action_to_id["move"]
    assert tensors["action_object_indices"].tolist() == [
        parsed.object_to_id["car0"],
        parsed.object_to_id["j0"],
        parsed.object_to_id["j1"],
        parsed.object_to_id["road0"],
    ]
    assert tensors["action_role_ids"].tolist() == [0, 1, 2, 3]
    assert tensors["action_arg_mask"].tolist() == [True, True, True, True]


def test_tensorize_predicate_uses_predicate_and_argument_roles(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)

    tensors = tensorize_predicate(
        parsed,
        parsed.goal_atoms[0],
    )

    assert tensors["predicate_id"].item() == parsed.predicate_to_id["at_car_jun"]
    assert tensors["predicate_object_indices"].tolist() == [
        parsed.object_to_id["car0"],
        parsed.object_to_id["j1"],
        -1,
    ]
    assert tensors["predicate_role_ids"].tolist() == [0, 1, -1]
    assert tensors["predicate_arg_mask"].tolist() == [True, True, False]
    assert "predicate_is_static" not in tensors


def test_tensorize_goal_atoms_pads_masks_and_weights(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)

    tensors = tensorize_goal_atoms(
        parsed,
        parsed.goal_atoms,
        max_atoms=3,
        weights=[2.0],
    )

    assert tensors["goal_predicate_id"].tolist() == [parsed.predicate_to_id["at_car_jun"], -1, -1]
    assert tensors["goal_object_indices"].tolist() == [
        [parsed.object_to_id["car0"], parsed.object_to_id["j1"], -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    assert tensors["goal_arg_mask"].tolist() == [
        [True, True, False],
        [False, False, False],
        [False, False, False],
    ]
    assert tensors["goal_atom_mask"].tolist() == [True, False, False]
    assert tensors["goal_weight"].tolist() == [2.0, 0.0, 0.0]
    assert tensors["goal_truth"].tolist() == [True, False, False]


def test_problem_and_trajectory_datasets_return_graph_samples(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)

    problem_dataset = PDDLGraphDataset(domain_path, [problem_path])
    problem_sample = problem_dataset[0]

    assert len(problem_dataset) == 1
    assert problem_sample["state"].num_atoms == 4
    assert problem_sample["goal"].num_atoms == 1

    problem_batch = next(iter(DataLoader(problem_dataset, batch_size=1)))
    assert problem_batch["state"].num_graphs == 1

    trajectory = TrajectorySample(
        problem_index=0,
        states=(parsed.initial_atoms, parsed.goal_atoms),
        actions=(GroundAction("move", ("car0", "j0", "j1", "road0")),),
        terminal_atoms=parsed.goal_atoms,
    )
    trajectory_dataset = PDDLTrajectoryDataset([parsed], [trajectory])
    trajectory_sample = trajectory_dataset[0]

    assert len(trajectory_dataset) == 1
    assert trajectory_sample["states"][0].num_atoms == 4
    assert trajectory_sample["states"][1].num_atoms == 1
    assert trajectory_sample["actions"]["action_id"].tolist() == [parsed.action_to_id["move"]]

    trajectory_batch = next(iter(DataLoader(trajectory_dataset, batch_size=1)))
    assert trajectory_batch["states"][1].num_graphs == 1


def test_trajectory_datasets_return_time_indexed_samples(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)
    middle_atoms = parsed.goal_atoms
    final_atoms = parsed.initial_atoms
    trajectory = TrajectorySample(
        problem_index=0,
        states=(parsed.initial_atoms, middle_atoms, final_atoms),
        actions=(
            GroundAction("move", ("car0", "j0", "j1", "road0")),
            GroundAction("move", ("car0", "j1", "j0", "road0")),
        ),
        terminal_atoms=final_atoms,
    )

    dataset = PDDLTrajectoryDataset([parsed], [trajectory])
    sample = dataset[0]

    assert len(sample["states"]) == 3
    assert sample["actions"]["action_id"].shape == (2,)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert len(batch["states"]) == 3
    assert batch["states"][0].num_graphs == 1
    assert batch["actions"]["action_id"].shape == (1, 2)

    atom_dataset = PDDLAtomTrajectoryDataset(
        [parsed],
        [trajectory],
        num_positive_atoms=2,
        num_negative_atoms=2,
        include_goal=True,
        include_terminal_state=True,
        seed=41,
    )
    atom_sample = atom_dataset[0]

    assert atom_sample["atom_queries"]["atom_predicate_id"].shape == (2, 4)
    assert atom_sample["terminal_state"].num_atoms == len(final_atoms)


def test_atom_trajectory_dataset_prioritizes_satisfied_goal_positives(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)
    goal_atom = parsed.goal_atoms[0]
    satisfied_goal_atom = GroundAtom("clear", ("j0",))
    parsed = replace(parsed, goal_atoms=(goal_atom, satisfied_goal_atom))
    next_atoms = (
        goal_atom,
        satisfied_goal_atom,
        GroundAtom("road_connect", ("road0", "j0", "j1")),
    )
    trajectory = TrajectorySample(
        problem_index=0,
        states=(parsed.initial_atoms, next_atoms),
        actions=(GroundAction("move", ("car0", "j0", "j1", "road0")),),
        terminal_atoms=next_atoms,
    )
    dataset = PDDLAtomTrajectoryDataset(
        [parsed],
        [trajectory],
        num_positive_atoms=2,
        num_negative_atoms=2,
        seed=7,
    )

    sample = dataset[0]
    atom_queries = {key: value[0] for key, value in sample["atom_queries"].items()}
    atoms = _atoms_from_atom_queries(parsed, atom_queries)
    truths = atom_queries["atom_truth"].tolist()
    masks = atom_queries["atom_sample_mask"].tolist()
    positives = [atom for atom, truth, mask in zip(atoms, truths, masks, strict=True) if truth and mask]
    negatives = [atom for atom, truth, mask in zip(atoms, truths, masks, strict=True) if not truth and mask]

    assert len(positives) == 2
    assert sum(atom in parsed.goal_atoms for atom in positives) == 1
    assert all(atom not in next_atoms for atom in negatives)


def test_atom_trajectory_dataset_negative_sources_can_contribute(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)
    source_next_atoms = {
        "random": parsed.goal_atoms,
        "corrupt_positive": parsed.goal_atoms,
        "action_modified": parsed.goal_atoms,
        "unsatisfied_goal": parsed.initial_atoms,
    }

    for seed, source in enumerate(source_next_atoms, start=11):
        trajectory = TrajectorySample(
            problem_index=0,
            states=(parsed.initial_atoms, source_next_atoms[source]),
            actions=(GroundAction("move", ("car0", "j0", "j1", "road0")),),
            terminal_atoms=source_next_atoms[source],
        )
        dataset = PDDLAtomTrajectoryDataset(
            [parsed],
            [trajectory],
            num_positive_atoms=0,
            num_negative_atoms=1,
            negative_source_weights={source: 1.0},
            seed=seed,
            negative_attempts_per_atom=200,
        )

        sample = dataset[0]
        atom_queries = {key: value[0] for key, value in sample["atom_queries"].items()}
        atoms = _atoms_from_atom_queries(parsed, atom_queries)
        masks = atom_queries["atom_sample_mask"].tolist()
        negatives = [atom for atom, mask in zip(atoms, masks, strict=True) if mask]

        assert len(negatives) == 1
        assert negatives[0] not in source_next_atoms[source]
        if source == "action_modified":
            assert negatives[0].predicate in parsed.actions["move"].modified_predicates
        if source == "unsatisfied_goal":
            assert negatives[0] == parsed.goal_atoms[0]


def test_atom_trajectory_dataset_filters_static_atoms_and_pads(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
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
        num_positive_atoms=4,
        num_negative_atoms=4,
        negative_source_weights={"random": 1.0},
        include_static=False,
        seed=23,
        negative_attempts_per_atom=200,
    )

    sample = dataset[0]
    atom_queries = {key: value[0] for key, value in sample["atom_queries"].items()}
    atoms = _atoms_from_atom_queries(parsed, atom_queries)
    masks = atom_queries["atom_sample_mask"].tolist()

    assert atom_queries["atom_sample_mask"][:4].tolist().count(True) == 1
    assert all(atom is None or atom.predicate != "same_line" for atom, mask in zip(atoms, masks, strict=True) if mask)


def test_atom_trajectory_dataset_is_deterministic_and_batchable(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
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
        seed=31,
    )

    first = dataset[0]["atom_queries"]
    second = dataset[0]["atom_queries"]
    for key in first:
        assert torch.equal(first[key], second[key])

    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch["states"][1].num_graphs == 1
    assert batch["atom_queries"]["atom_predicate_id"].shape == (1, 1, 4)


def test_atom_trajectory_dataset_batches_variable_goal_counts(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)
    longer_goal = (
        parsed.goal_atoms[0],
        GroundAtom("clear", ("j0",)),
    )
    parsed_with_longer_goal = replace(parsed, name="tiny-city-2", goal_atoms=longer_goal)
    trajectories = [
        TrajectorySample(
            problem_index=0,
            states=(parsed.initial_atoms, parsed.goal_atoms),
            actions=(GroundAction("move", ("car0", "j0", "j1", "road0")),),
            terminal_atoms=parsed.goal_atoms,
        ),
        TrajectorySample(
            problem_index=1,
            states=(parsed.initial_atoms, parsed_with_longer_goal.goal_atoms),
            actions=(GroundAction("move", ("car0", "j0", "j1", "road0")),),
            terminal_atoms=parsed_with_longer_goal.goal_atoms,
        ),
    ]
    dataset = PDDLAtomTrajectoryDataset(
        [parsed, parsed_with_longer_goal],
        trajectories,
        num_positive_atoms=1,
        num_negative_atoms=1,
        include_goal=True,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["goal"]["goal_predicate_id"].shape == (2, 2)
    assert batch["goal"]["goal_atom_mask"].tolist() == [[True, False], [True, True]]


def test_atom_trajectory_dataset_can_include_goal_and_terminal_state(tmp_path: Path) -> None:
    domain_path, problem_path = _write_pddl(tmp_path)
    parsed = parse_domain_problem(domain_path, problem_path)
    first_terminal = (
        GroundAtom("clear", ("j0",)),
        GroundAtom("at_car_jun", ("car0", "j1")),
    )
    final_terminal = (
        GroundAtom("clear", ("j1",)),
        GroundAtom("at_car_jun", ("car0", "j0")),
        GroundAtom("road_connect", ("road0", "j0", "j1")),
    )
    trajectory = TrajectorySample(
        problem_index=0,
        states=(parsed.initial_atoms, first_terminal, final_terminal),
        actions=(
            GroundAction("move", ("car0", "j0", "j1", "road0")),
            GroundAction("move", ("car0", "j1", "j0", "road0")),
        ),
        terminal_atoms=final_terminal,
    )
    dataset = PDDLAtomTrajectoryDataset(
        [parsed],
        [trajectory],
        num_positive_atoms=1,
        num_negative_atoms=1,
        include_goal=True,
        include_terminal_state=True,
        max_goal_atoms=2,
    )

    sample = dataset[0]
    terminal_state = sample["terminal_state"]

    assert sample["goal"]["goal_predicate_id"].tolist() == [parsed.predicate_to_id["at_car_jun"], -1]
    assert terminal_state.num_atoms == len(final_terminal)
    assert terminal_state.atom_predicate_id.tolist().count(parsed.predicate_to_id["road_connect"]) == 1


def _atoms_from_atom_queries(parsed, atom_queries: dict[str, torch.Tensor]) -> list[GroundAtom | None]:
    predicate_by_id = {idx: name for name, idx in parsed.predicate_to_id.items()}
    object_by_id = {idx: name for name, idx in parsed.object_to_id.items()}
    atoms: list[GroundAtom | None] = []
    for predicate_id, object_indices, arg_mask, sample_mask in zip(
        atom_queries["atom_predicate_id"].tolist(),
        atom_queries["atom_object_indices"].tolist(),
        atom_queries["atom_arg_mask"].tolist(),
        atom_queries["atom_sample_mask"].tolist(),
        strict=True,
    ):
        if not sample_mask:
            atoms.append(None)
            continue
        arguments = tuple(
            object_by_id[object_id] for object_id, is_arg in zip(object_indices, arg_mask, strict=True) if is_arg
        )
        atoms.append(GroundAtom(predicate_by_id[predicate_id], arguments))
    return atoms


def _write_pddl(tmp_path: Path) -> tuple[Path, Path]:
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)
    return domain_path, problem_path
