"""Torch Geometric datasets for PDDL graph samples."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch_geometric.data import Dataset

from acs_jepa.graph.builders import build_state_graph, tensorize_action, tensorize_goal_atoms, tensorize_predicate
from acs_jepa.graph.parsing import parse_domain_problem
from acs_jepa.graph.schemas import GroundAction, GroundAtom, ParsedProblem

_NEGATIVE_SOURCES = ("random", "corrupt_positive", "action_modified", "unsatisfied_goal")
_DEFAULT_NEGATIVE_SOURCE_WEIGHTS = {
    "random": 0.40,
    "corrupt_positive": 0.30,
    "action_modified": 0.20,
    "unsatisfied_goal": 0.10,
}


@dataclass(frozen=True)
class TrajectorySample:
    """One symbolic trajectory window before graph conversion.

    ``states`` contains canonical tuples of positive grounded atoms, with
    ``len(states) == len(actions) + 1``. ``terminal_atoms`` is the terminal
    state of the full source trajectory, which may be later than the window end.
    """

    problem_index: int
    states: tuple[tuple[GroundAtom, ...], ...]
    actions: tuple[GroundAction, ...]
    terminal_atoms: tuple[GroundAtom, ...]

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("TrajectorySample requires len(states) == len(actions) + 1")


class PDDLGraphDataset(Dataset):
    """Problem-level dataset returning initial and goal graphs.

    Each sample is a dictionary:

    ``state``
        PyG ``Data`` graph for the problem initial state.
    ``goal``
        PyG ``Data`` graph for the goal atoms.

    Parsed metadata remains available through ``dataset.parsed_problems``.
    """

    def __init__(
        self,
        domain_path: str | Path,
        problem_paths: Sequence[str | Path],
        *,
        include_static: bool = True,
        transform=None,
        pre_transform=None,
    ) -> None:
        super().__init__(None, transform, pre_transform)
        self.include_static = include_static
        self.parsed_problems = tuple(parse_domain_problem(domain_path, problem_path) for problem_path in problem_paths)

    def len(self) -> int:
        return len(self.parsed_problems)

    def get(self, idx: int) -> dict[str, object]:
        parsed_problem = self.parsed_problems[idx]
        return {
            "state": build_state_graph(
                parsed_problem,
                parsed_problem.initial_atoms,
                include_static=self.include_static,
            ),
            "goal": build_state_graph(
                parsed_problem,
                parsed_problem.goal_atoms,
                include_static=True,
            ),
        }


class PDDLTrajectoryDataset(Dataset):
    """Trajectory dataset returning state-graph and action sequences.

    Each sample is a dictionary:

    ``states``
        List of ``K + 1`` PyG ``Data`` graphs.
    ``actions``
        Fixed-size action tensor dictionary with a leading time axis ``K``.

    The state graphs use the factor-node encoding documented in
    ``acs_jepa.graph.builders``.
    """

    def __init__(
        self,
        parsed_problems: Sequence[ParsedProblem],
        trajectories: Sequence[TrajectorySample],
        *,
        include_static: bool = True,
        transform=None,
        pre_transform=None,
    ) -> None:
        super().__init__(None, transform, pre_transform)
        self.parsed_problems = tuple(parsed_problems)
        self.trajectories = tuple(trajectories)
        self.include_static = include_static
        self.max_action_arity = _max_action_arity(self.parsed_problems)

    def len(self) -> int:
        return len(self.trajectories)

    def get(self, idx: int) -> dict[str, object]:
        trajectory = self.trajectories[idx]
        parsed_problem = self.parsed_problems[trajectory.problem_index]
        return {
            "states": [
                build_state_graph(parsed_problem, atoms, include_static=self.include_static)
                for atoms in trajectory.states
            ],
            "actions": _tensorize_action_sequence(
                parsed_problem,
                trajectory.actions,
                max_arity=self.max_action_arity,
            ),
        }


class PDDLAtomTrajectoryDataset(Dataset):
    """Trajectory dataset with fixed-size positive and negative atom queries.

    Positive atom queries are true in each observed next state. Negative atom
    queries are sampled lazily from typed predicate/object combinations and are
    rejected if they are true in that next state.
    """

    def __init__(
        self,
        parsed_problems: Sequence[ParsedProblem],
        trajectories: Sequence[TrajectorySample],
        *,
        num_positive_atoms: int,
        num_negative_atoms: int,
        negative_source_weights: Mapping[str, float] | None = None,
        goal_positive_fraction: float = 0.5,
        include_static: bool = True,
        include_goal: bool = False,
        include_terminal_state: bool = False,
        max_goal_atoms: int | None = None,
        seed: int = 0,
        negative_attempts_per_atom: int = 50,
        transform=None,
        pre_transform=None,
    ) -> None:
        super().__init__(None, transform, pre_transform)
        if num_positive_atoms < 0:
            raise ValueError("num_positive_atoms must be non-negative")
        if num_negative_atoms < 0:
            raise ValueError("num_negative_atoms must be non-negative")
        if not 0.0 <= goal_positive_fraction <= 1.0:
            raise ValueError("goal_positive_fraction must be between 0 and 1")
        if negative_attempts_per_atom < 1:
            raise ValueError("negative_attempts_per_atom must be positive")

        self.parsed_problems = tuple(parsed_problems)
        self.trajectories = tuple(trajectories)
        self.num_positive_atoms = num_positive_atoms
        self.num_negative_atoms = num_negative_atoms
        self.negative_source_weights = _normalize_negative_source_weights(negative_source_weights, num_negative_atoms)
        self.goal_positive_fraction = goal_positive_fraction
        self.include_static = include_static
        self.include_goal = include_goal
        self.include_terminal_state = include_terminal_state
        self.max_action_arity = _max_action_arity(self.parsed_problems)
        self.max_predicate_arity = _max_predicate_arity_for_problems(self.parsed_problems)
        self.max_goal_atoms = _max_goal_atoms(self.parsed_problems) if max_goal_atoms is None else max_goal_atoms
        self.seed = seed
        self.negative_attempts_per_atom = negative_attempts_per_atom

    def len(self) -> int:
        return len(self.trajectories)

    def get(self, idx: int) -> dict[str, object]:
        trajectory = self.trajectories[idx]
        parsed_problem = self.parsed_problems[trajectory.problem_index]
        rng = random.Random(self.seed + idx)
        atom_queries = []
        for step_idx, action in enumerate(trajectory.actions):
            step = _TransitionView(action=action, next_atoms=trajectory.states[step_idx + 1])
            positive_atoms = self._sample_positive_atoms(parsed_problem, step, rng)
            negative_atoms = self._sample_negative_atoms(parsed_problem, step, rng)
            atom_queries.append(
                _tensorize_atom_queries(
                    parsed_problem,
                    positive_atoms,
                    negative_atoms,
                    num_positive_atoms=self.num_positive_atoms,
                    num_negative_atoms=self.num_negative_atoms,
                    max_arity=self.max_predicate_arity,
                )
            )
        sample = {
            "states": [
                build_state_graph(parsed_problem, atoms, include_static=self.include_static)
                for atoms in trajectory.states
            ],
            "actions": _tensorize_action_sequence(
                parsed_problem,
                trajectory.actions,
                max_arity=self.max_action_arity,
            ),
            "atom_queries": _stack_tensor_dicts(atom_queries),
        }
        if self.include_goal:
            sample["goal"] = tensorize_goal_atoms(
                parsed_problem,
                parsed_problem.goal_atoms,
                max_atoms=self.max_goal_atoms,
                max_arity=self.max_predicate_arity,
            )
        if self.include_terminal_state:
            sample["terminal_state"] = build_state_graph(
                parsed_problem,
                trajectory.terminal_atoms,
                include_static=self.include_static,
            )
        return sample

    def _sample_positive_atoms(
        self,
        parsed_problem: ParsedProblem,
        transition: "_TransitionView",
        rng: random.Random,
    ) -> list[GroundAtom]:
        next_atoms = _filter_static(parsed_problem, transition.next_atoms, include_static=self.include_static)
        next_atom_set = set(next_atoms)
        goal_atoms = _filter_static(parsed_problem, parsed_problem.goal_atoms, include_static=self.include_static)
        goal_candidates = sorted(next_atom_set.intersection(goal_atoms))
        goal_capacity = math.floor(self.num_positive_atoms * self.goal_positive_fraction)
        selected = _sample_without_replacement(rng, goal_candidates, goal_capacity)

        remaining_count = self.num_positive_atoms - len(selected)
        goal_candidate_set = set(goal_candidates)
        other_candidates = [atom for atom in sorted(next_atom_set) if atom not in goal_candidate_set]
        selected.extend(_sample_without_replacement(rng, other_candidates, remaining_count))
        return selected

    def _sample_negative_atoms(
        self,
        parsed_problem: ParsedProblem,
        transition: "_TransitionView",
        rng: random.Random,
    ) -> list[GroundAtom]:
        if self.num_negative_atoms == 0:
            return []

        true_atoms = set(_filter_static(parsed_problem, transition.next_atoms, include_static=self.include_static))
        selected: list[GroundAtom] = []
        source_counts = _allocate_source_counts(self.negative_source_weights, self.num_negative_atoms)
        for source in _NEGATIVE_SOURCES:
            needed = source_counts[source]
            selected.extend(
                self._sample_negative_source(
                    source,
                    parsed_problem,
                    transition,
                    true_atoms,
                    set(selected),
                    needed,
                    rng,
                )
            )

        remaining_count = self.num_negative_atoms - len(selected)
        if remaining_count > 0:
            selected.extend(
                self._sample_negative_source(
                    "random",
                    parsed_problem,
                    transition,
                    true_atoms,
                    set(selected),
                    remaining_count,
                    rng,
                )
            )
        return selected

    def _sample_negative_source(
        self,
        source: str,
        parsed_problem: ParsedProblem,
        transition: "_TransitionView",
        true_atoms: set[GroundAtom],
        already_selected: set[GroundAtom],
        count: int,
        rng: random.Random,
    ) -> list[GroundAtom]:
        if count <= 0:
            return []
        if source == "unsatisfied_goal":
            return _sample_unsatisfied_goal_negatives(
                parsed_problem,
                true_atoms,
                already_selected,
                count,
                rng,
                include_static=self.include_static,
            )

        objects_by_type = _objects_by_type(parsed_problem)
        if source == "random":
            predicate_names = _eligible_predicate_names(
                parsed_problem,
                objects_by_type,
                include_static=self.include_static,
            )
        elif source == "action_modified":
            action = parsed_problem.actions.get(transition.action.name)
            modified_predicates = () if action is None else action.modified_predicates
            predicate_names = _eligible_predicate_names(
                parsed_problem,
                objects_by_type,
                include_static=self.include_static,
                predicate_names=modified_predicates,
            )
        elif source == "corrupt_positive":
            return _sample_corrupt_positive_negatives(
                parsed_problem,
                true_atoms,
                already_selected,
                count,
                rng,
                objects_by_type,
                include_static=self.include_static,
                attempts_per_atom=self.negative_attempts_per_atom,
            )
        else:
            raise ValueError(f"Unknown negative source: {source}")

        return _sample_random_ground_negatives(
            parsed_problem,
            true_atoms,
            already_selected,
            count,
            rng,
            objects_by_type,
            predicate_names,
            attempts_per_atom=self.negative_attempts_per_atom,
        )


@dataclass(frozen=True)
class _TransitionView:
    action: GroundAction
    next_atoms: tuple[GroundAtom, ...]


def _tensorize_action_sequence(
    parsed_problem: ParsedProblem,
    actions: Sequence[GroundAction],
    *,
    max_arity: int | None = None,
) -> dict[str, torch.Tensor]:
    if not actions:
        raise ValueError("Trajectory samples must contain at least one action")
    return _stack_tensor_dicts([tensorize_action(parsed_problem, action, max_arity=max_arity) for action in actions])


def _stack_tensor_dicts(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("Cannot stack an empty tensor dictionary sequence")
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _normalize_negative_source_weights(
    negative_source_weights: Mapping[str, float] | None,
    num_negative_atoms: int,
) -> dict[str, float]:
    raw_weights = _DEFAULT_NEGATIVE_SOURCE_WEIGHTS if negative_source_weights is None else negative_source_weights
    unknown_sources = set(raw_weights) - set(_NEGATIVE_SOURCES)
    if unknown_sources:
        raise ValueError(f"Unknown negative source weights: {sorted(unknown_sources)}")

    weights = {source: float(raw_weights.get(source, 0.0)) for source in _NEGATIVE_SOURCES}
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("negative_source_weights must be non-negative")
    if num_negative_atoms > 0 and sum(weights.values()) <= 0.0:
        raise ValueError("At least one negative source weight must be positive")
    return weights


def _allocate_source_counts(weights: Mapping[str, float], count: int) -> dict[str, int]:
    if count == 0:
        return {source: 0 for source in _NEGATIVE_SOURCES}

    total_weight = sum(weights.values())
    raw_counts = {source: count * weights[source] / total_weight for source in _NEGATIVE_SOURCES}
    source_counts = {source: math.floor(raw_counts[source]) for source in _NEGATIVE_SOURCES}
    remainder = count - sum(source_counts.values())
    fractional_sources = sorted(
        _NEGATIVE_SOURCES,
        key=lambda source: (-(raw_counts[source] - source_counts[source]), _NEGATIVE_SOURCES.index(source)),
    )
    for source in fractional_sources[:remainder]:
        source_counts[source] += 1
    return source_counts


def _tensorize_atom_queries(
    parsed_problem: ParsedProblem,
    positive_atoms: Sequence[GroundAtom],
    negative_atoms: Sequence[GroundAtom],
    *,
    num_positive_atoms: int,
    num_negative_atoms: int,
    max_arity: int | None = None,
) -> dict[str, torch.Tensor]:
    max_arity = _max_predicate_arity(parsed_problem) if max_arity is None else max_arity
    total_atoms = num_positive_atoms + num_negative_atoms
    predicate_ids = torch.full((total_atoms,), -1, dtype=torch.long)
    object_indices = torch.full((total_atoms, max_arity), -1, dtype=torch.long)
    role_ids = torch.full((total_atoms, max_arity), -1, dtype=torch.long)
    arg_mask = torch.zeros((total_atoms, max_arity), dtype=torch.bool)
    truth = torch.zeros((total_atoms,), dtype=torch.bool)
    sample_mask = torch.zeros((total_atoms,), dtype=torch.bool)

    for atom_idx, atom in enumerate(positive_atoms[:num_positive_atoms]):
        _fill_atom_query(
            parsed_problem,
            atom,
            atom_idx,
            True,
            max_arity,
            predicate_ids,
            object_indices,
            role_ids,
            arg_mask,
            truth,
            sample_mask,
        )
    negative_start = num_positive_atoms
    for atom_offset, atom in enumerate(negative_atoms[:num_negative_atoms]):
        atom_idx = negative_start + atom_offset
        _fill_atom_query(
            parsed_problem,
            atom,
            atom_idx,
            False,
            max_arity,
            predicate_ids,
            object_indices,
            role_ids,
            arg_mask,
            truth,
            sample_mask,
        )

    return {
        "atom_predicate_id": predicate_ids,
        "atom_object_indices": object_indices,
        "atom_role_ids": role_ids,
        "atom_arg_mask": arg_mask,
        "atom_truth": truth,
        "atom_sample_mask": sample_mask,
    }


def _max_action_arity(parsed_problems: Sequence[ParsedProblem]) -> int:
    if not parsed_problems:
        return 0
    return max(parsed.max_action_arity for parsed in parsed_problems)


def _max_predicate_arity_for_problems(parsed_problems: Sequence[ParsedProblem]) -> int:
    return max(
        (len(predicate.arg_types) for parsed in parsed_problems for predicate in parsed.predicates.values()),
        default=0,
    )


def _max_goal_atoms(parsed_problems: Sequence[ParsedProblem]) -> int:
    return max((len(parsed.goal_atoms) for parsed in parsed_problems), default=0)


def _fill_atom_query(
    parsed_problem: ParsedProblem,
    atom: GroundAtom,
    atom_idx: int,
    atom_truth: bool,
    max_arity: int,
    predicate_ids: torch.Tensor,
    object_indices: torch.Tensor,
    role_ids: torch.Tensor,
    arg_mask: torch.Tensor,
    truth: torch.Tensor,
    sample_mask: torch.Tensor,
) -> None:
    predicate_tensors = tensorize_predicate(parsed_problem, atom, max_arity=max_arity)
    predicate_ids[atom_idx] = predicate_tensors["predicate_id"]
    object_indices[atom_idx] = predicate_tensors["predicate_object_indices"]
    role_ids[atom_idx] = predicate_tensors["predicate_role_ids"]
    arg_mask[atom_idx] = predicate_tensors["predicate_arg_mask"]
    truth[atom_idx] = atom_truth
    sample_mask[atom_idx] = True


def _filter_static(
    parsed_problem: ParsedProblem,
    atoms: Sequence[GroundAtom],
    *,
    include_static: bool,
) -> tuple[GroundAtom, ...]:
    if include_static:
        return tuple(atoms)
    return tuple(atom for atom in atoms if atom.predicate not in parsed_problem.static_predicates)


def _sample_without_replacement(rng: random.Random, atoms: Sequence[GroundAtom], count: int) -> list[GroundAtom]:
    if count <= 0 or not atoms:
        return []
    sample_count = min(count, len(atoms))
    return rng.sample(list(atoms), sample_count)


def _sample_unsatisfied_goal_negatives(
    parsed_problem: ParsedProblem,
    true_atoms: set[GroundAtom],
    already_selected: set[GroundAtom],
    count: int,
    rng: random.Random,
    *,
    include_static: bool,
) -> list[GroundAtom]:
    candidates = [
        atom
        for atom in sorted(_filter_static(parsed_problem, parsed_problem.goal_atoms, include_static=include_static))
        if atom not in true_atoms and atom not in already_selected
    ]
    return _sample_without_replacement(rng, candidates, count)


def _sample_random_ground_negatives(
    parsed_problem: ParsedProblem,
    true_atoms: set[GroundAtom],
    already_selected: set[GroundAtom],
    count: int,
    rng: random.Random,
    objects_by_type: Mapping[str, Sequence[str]],
    predicate_names: Sequence[str],
    *,
    attempts_per_atom: int,
) -> list[GroundAtom]:
    selected: list[GroundAtom] = []
    selected_set = set(already_selected)
    attempts = max(1, count * attempts_per_atom)
    for _ in range(attempts):
        if len(selected) >= count:
            break
        atom = _sample_ground_atom(parsed_problem, objects_by_type, predicate_names, rng)
        if atom is None or atom in true_atoms or atom in selected_set:
            continue
        selected.append(atom)
        selected_set.add(atom)
    return selected


def _sample_corrupt_positive_negatives(
    parsed_problem: ParsedProblem,
    true_atoms: set[GroundAtom],
    already_selected: set[GroundAtom],
    count: int,
    rng: random.Random,
    objects_by_type: Mapping[str, Sequence[str]],
    *,
    include_static: bool,
    attempts_per_atom: int,
) -> list[GroundAtom]:
    candidates = [
        atom
        for atom in sorted(true_atoms)
        if _can_corrupt_atom(parsed_problem, atom, objects_by_type)
        and (include_static or atom.predicate not in parsed_problem.static_predicates)
    ]
    selected: list[GroundAtom] = []
    selected_set = set(already_selected)
    attempts = max(1, count * attempts_per_atom)
    for _ in range(attempts):
        if len(selected) >= count:
            break
        if not candidates:
            break
        atom = _corrupt_atom(parsed_problem, rng.choice(candidates), objects_by_type, rng)
        if atom is None or atom in true_atoms or atom in selected_set:
            continue
        selected.append(atom)
        selected_set.add(atom)
    return selected


def _sample_ground_atom(
    parsed_problem: ParsedProblem,
    objects_by_type: Mapping[str, Sequence[str]],
    predicate_names: Sequence[str],
    rng: random.Random,
) -> GroundAtom | None:
    if not predicate_names:
        return None
    predicate_name = rng.choice(list(predicate_names))
    predicate = parsed_problem.predicates[predicate_name]
    arguments = tuple(rng.choice(list(objects_by_type[arg_type])) for arg_type in predicate.arg_types)
    return GroundAtom(predicate=predicate_name, arguments=arguments)


def _corrupt_atom(
    parsed_problem: ParsedProblem,
    atom: GroundAtom,
    objects_by_type: Mapping[str, Sequence[str]],
    rng: random.Random,
) -> GroundAtom | None:
    predicate = parsed_problem.predicates.get(atom.predicate)
    if predicate is None:
        return None
    corruptible_roles = [
        role_id
        for role_id, arg_type in enumerate(predicate.arg_types)
        if any(object_name != atom.arguments[role_id] for object_name in objects_by_type.get(arg_type, ()))
    ]
    if not corruptible_roles:
        return None

    role_id = rng.choice(corruptible_roles)
    arg_type = predicate.arg_types[role_id]
    replacements = [
        object_name for object_name in objects_by_type.get(arg_type, ()) if object_name != atom.arguments[role_id]
    ]
    arguments = list(atom.arguments)
    arguments[role_id] = rng.choice(replacements)
    return GroundAtom(predicate=atom.predicate, arguments=tuple(arguments))


def _can_corrupt_atom(
    parsed_problem: ParsedProblem,
    atom: GroundAtom,
    objects_by_type: Mapping[str, Sequence[str]],
) -> bool:
    predicate = parsed_problem.predicates.get(atom.predicate)
    if predicate is None:
        return False
    return any(
        any(object_name != atom.arguments[role_id] for object_name in objects_by_type.get(arg_type, ()))
        for role_id, arg_type in enumerate(predicate.arg_types)
    )


def _objects_by_type(parsed_problem: ParsedProblem) -> dict[str, tuple[str, ...]]:
    objects_by_type: dict[str, list[str]] = {type_name: [] for type_name in parsed_problem.types}
    for object_name, object_info in parsed_problem.objects.items():
        objects_by_type.setdefault(object_info.type, []).append(object_name)
    return {type_name: tuple(sorted(object_names)) for type_name, object_names in objects_by_type.items()}


def _eligible_predicate_names(
    parsed_problem: ParsedProblem,
    objects_by_type: Mapping[str, Sequence[str]],
    *,
    include_static: bool,
    predicate_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    names = parsed_problem.predicates if predicate_names is None else predicate_names
    return tuple(
        sorted(
            predicate_name
            for predicate_name in names
            if predicate_name in parsed_problem.predicates
            and (include_static or predicate_name not in parsed_problem.static_predicates)
            and all(objects_by_type.get(arg_type) for arg_type in parsed_problem.predicates[predicate_name].arg_types)
        )
    )


def _max_predicate_arity(parsed_problem: ParsedProblem) -> int:
    if not parsed_problem.predicates:
        return 0
    return max(len(predicate.arg_types) for predicate in parsed_problem.predicates.values())
