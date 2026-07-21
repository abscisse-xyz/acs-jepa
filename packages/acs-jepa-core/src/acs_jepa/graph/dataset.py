"""Torch Geometric datasets for PDDL graph samples."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Dataset

from acs_jepa.graph.action_supervision import build_action_supervision_tensors
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
ATOM_STATE_APPLICABILITY_SEMANTICS = "positive_ground_atoms_closed_world_v1"


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


@dataclass(frozen=True, order=True)
class ActionApplicabilityStateKey:
    """Canonical atom-only state identity for an offline applicability table."""

    problem_index: int
    state_atoms: tuple[GroundAtom, ...]

    def __post_init__(self) -> None:
        if type(self.problem_index) is not int or self.problem_index < 0:
            raise ValueError("problem_index must be a non-negative integer")
        if type(self.state_atoms) is not tuple or any(
            type(atom) is not GroundAtom for atom in self.state_atoms
        ):
            raise ValueError("state_atoms must be an exact tuple of GroundAtom values")
        if self.state_atoms != tuple(sorted(set(self.state_atoms))):
            raise ValueError("state_atoms must be sorted and duplicate-free")


def action_applicability_state_key(
    problem_index: int,
    state_atoms: Sequence[GroundAtom],
) -> ActionApplicabilityStateKey:
    """Canonicalize a represented positive-atom state for oracle lookup."""

    if any(type(atom) is not GroundAtom for atom in state_atoms):
        raise ValueError("state_atoms must contain only exact GroundAtom values")
    return ActionApplicabilityStateKey(
        problem_index=problem_index,
        state_atoms=tuple(sorted(set(state_atoms))),
    )


@dataclass(frozen=True)
class ActionApplicabilityTable(
    Mapping[ActionApplicabilityStateKey, frozenset[GroundAction]]
):
    """Pickleable immutable state-to-applicable-actions mapping."""

    entries: tuple[
        tuple[ActionApplicabilityStateKey, frozenset[GroundAction]], ...
    ]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise ValueError("entries must be an exact tuple")
        previous_key: ActionApplicabilityStateKey | None = None
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("each table entry must be an exact key/value tuple")
            key, actions = entry
            if type(key) is not ActionApplicabilityStateKey:
                raise ValueError("table keys must be exact ActionApplicabilityStateKey values")
            if type(actions) is not frozenset or any(
                type(action) is not GroundAction for action in actions
            ):
                raise ValueError("table values must be exact frozensets of GroundAction values")
            if previous_key is not None and key <= previous_key:
                raise ValueError("table entries must have sorted unique keys")
            previous_key = key

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[ActionApplicabilityStateKey, Set[GroundAction]],
    ) -> ActionApplicabilityTable:
        if not isinstance(source, Mapping):
            raise ValueError("applicable_actions_by_state must be a mapping")
        entries = []
        for key, actions in source.items():
            if type(key) is not ActionApplicabilityStateKey:
                raise ValueError("table keys must be exact ActionApplicabilityStateKey values")
            if not isinstance(actions, Set) or any(
                type(action) is not GroundAction for action in actions
            ):
                raise ValueError("table values must be sets of exact GroundAction values")
            entries.append((key, frozenset(actions)))
        return cls(tuple(sorted(entries, key=lambda entry: entry[0])))

    def __getitem__(self, key: ActionApplicabilityStateKey) -> frozenset[GroundAction]:
        low = 0
        high = len(self.entries)
        while low < high:
            middle = (low + high) // 2
            middle_key = self.entries[middle][0]
            if middle_key < key:
                low = middle + 1
            else:
                high = middle
        if low >= len(self.entries) or self.entries[low][0] != key:
            raise KeyError(key)
        return self.entries[low][1]

    def __iter__(self) -> Iterator[ActionApplicabilityStateKey]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class ActionSupervisionConfig:
    """Disabled-by-default trajectory action-supervision settings."""

    num_negatives: int
    seed: int = 0
    max_random_attempts_per_category: int = 128
    applicable_actions_by_state: (
        Mapping[ActionApplicabilityStateKey, Set[GroundAction]]
        | ActionApplicabilityTable
        | None
    ) = None
    applicability_state_semantics: str | None = None

    def __post_init__(self) -> None:
        for name in ("num_negatives", "seed", "max_random_attempts_per_category"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.num_negatives < 0:
            raise ValueError("num_negatives must be non-negative")
        if self.max_random_attempts_per_category <= 0:
            raise ValueError("max_random_attempts_per_category must be positive")
        table = self.applicable_actions_by_state
        if table is None:
            if self.applicability_state_semantics is not None:
                raise ValueError("applicability_state_semantics requires an oracle table")
            return
        if self.applicability_state_semantics != ATOM_STATE_APPLICABILITY_SEMANTICS:
            raise ValueError(
                "oracle tables require positive_ground_atoms_closed_world_v1 semantics"
            )
        if type(table) is not ActionApplicabilityTable:
            table = ActionApplicabilityTable.from_mapping(table)
            object.__setattr__(self, "applicable_actions_by_state", table)


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
        action_supervision: ActionSupervisionConfig | None = None,
        transform=None,
        pre_transform=None,
    ) -> None:
        super().__init__(None, transform, pre_transform)
        self.parsed_problems = tuple(parsed_problems)
        self.trajectories = tuple(trajectories)
        self.include_static = include_static
        self.max_action_arity = _max_action_arity(self.parsed_problems)
        self.max_objects = _max_objects(self.parsed_problems)
        self.action_supervision = action_supervision
        if action_supervision is not None:
            _validate_action_supervision_dataset(
                self.parsed_problems,
                self.trajectories,
                action_supervision,
            )

    def len(self) -> int:
        return len(self.trajectories)

    def get(self, idx: int) -> dict[str, object]:
        trajectory = self.trajectories[idx]
        parsed_problem = self.parsed_problems[trajectory.problem_index]
        sample: dict[str, object] = {
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
        if self.action_supervision is not None:
            sample["action_supervision"] = _build_action_supervision_sequence(
                parsed_problem,
                trajectory,
                trajectory_index=idx,
                max_action_arity=self.max_action_arity,
                max_objects=self.max_objects,
                config=self.action_supervision,
            )
        return sample


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
        action_supervision: ActionSupervisionConfig | None = None,
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
        self.max_objects = _max_objects(self.parsed_problems)
        self.action_supervision = action_supervision
        if action_supervision is not None:
            _validate_action_supervision_dataset(
                self.parsed_problems,
                self.trajectories,
                action_supervision,
            )
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
        if self.action_supervision is not None:
            sample["action_supervision"] = _build_action_supervision_sequence(
                parsed_problem,
                trajectory,
                trajectory_index=idx,
                max_action_arity=self.max_action_arity,
                max_objects=self.max_objects,
                config=self.action_supervision,
            )
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


def _max_objects(parsed_problems: Sequence[ParsedProblem]) -> int:
    return max((len(parsed.objects) for parsed in parsed_problems), default=0)


def _action_schema_signature(parsed_problem: ParsedProblem) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (name, parsed_problem.actions[name].parameter_types)
        for name in sorted(parsed_problem.actions)
    )


def _validate_ground_atom_for_problem(
    parsed_problem: ParsedProblem,
    atom: GroundAtom,
    *,
    key: ActionApplicabilityStateKey,
) -> None:
    context = (
        f"oracle validation for problem {key.problem_index}, "
        f"key {key!r}, atom {atom!r}"
    )
    if atom.predicate not in parsed_problem.predicates:
        raise ValueError(f"{context}: unknown predicate {atom.predicate!r}")
    schema = parsed_problem.predicates[atom.predicate]
    if len(atom.arguments) != len(schema.arg_types):
        raise ValueError(f"{context}: predicate {atom.predicate!r} has wrong arity")
    for role_id, (object_name, required_type) in enumerate(
        zip(atom.arguments, schema.arg_types, strict=True)
    ):
        if object_name not in parsed_problem.objects:
            raise ValueError(f"{context}: unknown object {object_name!r}")
        if parsed_problem.objects[object_name].type != required_type:
            raise ValueError(
                f"{context}: object {object_name!r} has wrong type at role {role_id}"
            )


def _validate_ground_action_for_problem(
    parsed_problem: ParsedProblem,
    action: GroundAction,
    *,
    key: ActionApplicabilityStateKey,
) -> None:
    context = (
        f"oracle validation for problem {key.problem_index}, "
        f"key {key!r}, action {action!r}"
    )
    if action.name not in parsed_problem.actions:
        raise ValueError(f"{context}: unknown action schema {action.name!r}")
    schema = parsed_problem.actions[action.name]
    if len(action.arguments) != len(schema.parameter_types):
        raise ValueError(f"{context}: action {action.name!r} has wrong arity")
    for role_id, (object_name, required_type) in enumerate(
        zip(action.arguments, schema.parameter_types, strict=True)
    ):
        if object_name not in parsed_problem.objects:
            raise ValueError(f"{context}: unknown object {object_name!r}")
        if parsed_problem.objects[object_name].type != required_type:
            raise ValueError(
                f"{context}: object {object_name!r} has wrong type at role {role_id}"
            )


def _validate_action_supervision_dataset(
    parsed_problems: Sequence[ParsedProblem],
    trajectories: Sequence[TrajectorySample],
    config: ActionSupervisionConfig,
) -> None:
    action_lengths = {len(trajectory.actions) for trajectory in trajectories}
    if 0 in action_lengths:
        raise ValueError("action supervision requires every trajectory to contain an action")
    if len(action_lengths) > 1:
        raise ValueError("action supervision requires fixed-length trajectories")
    for trajectory_index, trajectory in enumerate(trajectories):
        if not 0 <= trajectory.problem_index < len(parsed_problems):
            raise ValueError(
                f"trajectory {trajectory_index} has out-of-range problem index"
            )

    signatures = {_action_schema_signature(parsed) for parsed in parsed_problems}
    if len(signatures) > 1:
        raise ValueError("action supervision requires compatible action schemas")

    table = config.applicable_actions_by_state
    if table is None:
        return
    if type(table) is not ActionApplicabilityTable:
        raise TypeError("ActionSupervisionConfig did not normalize its oracle table")
    for key, applicable_actions in table.items():
        if key.problem_index >= len(parsed_problems):
            raise ValueError(
                f"oracle key has out-of-range problem index {key.problem_index}"
            )
        parsed_problem = parsed_problems[key.problem_index]
        for atom in key.state_atoms:
            _validate_ground_atom_for_problem(
                parsed_problem,
                atom,
                key=key,
            )
        for action in applicable_actions:
            _validate_ground_action_for_problem(
                parsed_problem,
                action,
                key=key,
            )


def _action_supervision_seed(
    base_seed: int,
    problem_index: int,
    state_atoms: Sequence[GroundAtom],
    action: GroundAction,
) -> int:
    digest = hashlib.blake2b(digest_size=8)

    def update_field(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    update_field(str(base_seed))
    update_field(str(problem_index))
    canonical_atoms = tuple(sorted(set(state_atoms)))
    update_field(str(len(canonical_atoms)))
    for atom in canonical_atoms:
        update_field(atom.predicate)
        update_field(str(len(atom.arguments)))
        for argument in atom.arguments:
            update_field(argument)
    update_field(action.name)
    update_field(str(len(action.arguments)))
    for argument in action.arguments:
        update_field(argument)
    return int.from_bytes(digest.digest(), "big")


def _build_action_supervision_sequence(
    parsed_problem: ParsedProblem,
    trajectory: TrajectorySample,
    *,
    trajectory_index: int,
    max_action_arity: int,
    max_objects: int,
    config: ActionSupervisionConfig,
) -> dict[str, torch.Tensor]:
    table = config.applicable_actions_by_state
    outputs: list[dict[str, torch.Tensor]] = []
    for step_index, true_action in enumerate(trajectory.actions):
        current_atoms = trajectory.states[step_index]
        applicability_labeler = None
        if table is not None:
            state_key = action_applicability_state_key(
                trajectory.problem_index,
                current_atoms,
            )
            try:
                applicable_actions = table[state_key]
            except KeyError:
                pass
            else:
                if true_action not in applicable_actions:
                    raise ValueError(
                        "offline applicability table contradicts trace action "
                        f"at problem {trajectory.problem_index}, "
                        f"trajectory {trajectory_index}, step {step_index}"
                    )
                applicability_labeler = applicable_actions.__contains__
        outputs.append(
            build_action_supervision_tensors(
                parsed_problem,
                true_action,
                max_action_arity=max_action_arity,
                max_objects=max_objects,
                num_negatives=config.num_negatives,
                seed=_action_supervision_seed(
                    config.seed,
                    trajectory.problem_index,
                    current_atoms,
                    true_action,
                ),
                applicability_labeler=applicability_labeler,
                max_random_attempts_per_category=(
                    config.max_random_attempts_per_category
                ),
            )
        )
    return _stack_tensor_dicts(outputs)


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
