"""Type-valid hard-negative action sampling for action-latent diagnostics."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa.graph.schemas import GroundAction

ONE_ARG_SUBSTITUTION = "one_arg_substitution"
ROLE_SWAP = "role_swap"
RANDOM_SAME_SCHEMA = "random_same_schema"
RANDOM_OTHER_SCHEMA = "random_other_schema"


@dataclass(frozen=True)
class NegativeActionExample:
    """One sampled type-valid negative around a reference grounded action."""

    action: GroundAction
    category: str
    changed_roles: tuple[int, ...]
    applicable: bool | None = None


def sample_action_negatives(
    space: ActionDecodingSpace,
    true_action: GroundAction,
    *,
    per_category: int = 4,
    seed: int = 0,
    applicability_fn: Callable[[GroundAction], bool] | None = None,
) -> tuple[NegativeActionExample, ...]:
    """Return deterministic categorized type-valid negatives for ``true_action``."""

    if per_category <= 0:
        raise ValueError("per_category must be positive")
    if true_action.name not in space.parsed_problem.actions:
        raise ValueError(f"Unknown true action schema: {true_action.name}")
    schema = space.parsed_problem.actions[true_action.name]
    if len(true_action.arguments) != schema.arity:
        raise ValueError(f"Action {true_action.name} expects {schema.arity} arguments")

    rng = random.Random(seed)
    true_key = _action_key(true_action)
    seen: set[tuple[str, tuple[str, ...]]] = {true_key}
    examples: list[NegativeActionExample] = []

    def add_candidates(category: str, candidates: Iterable[GroundAction]) -> None:
        candidates_list = list(candidates)
        rng.shuffle(candidates_list)
        emitted = 0
        for candidate in candidates_list:
            if emitted >= per_category:
                break
            key = _action_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                NegativeActionExample(
                    action=candidate,
                    category=category,
                    changed_roles=_changed_roles(true_action, candidate),
                    applicable=None if applicability_fn is None else bool(applicability_fn(candidate)),
                )
            )
            emitted += 1

    add_candidates(ONE_ARG_SUBSTITUTION, _one_arg_substitutions(space, true_action))
    add_candidates(ROLE_SWAP, _role_swaps(space, true_action))
    add_candidates(RANDOM_SAME_SCHEMA, _same_schema_candidates(space, true_action))
    add_candidates(RANDOM_OTHER_SCHEMA, _other_schema_candidates(space, true_action))
    return tuple(examples)


def _one_arg_substitutions(space: ActionDecodingSpace, action: GroundAction) -> list[GroundAction]:
    schema = space.parsed_problem.actions[action.name]
    candidates: list[GroundAction] = []
    for role_id, type_name in enumerate(schema.parameter_types):
        for object_name in space.objects_by_type.get(type_name, ()):  # already sorted by ActionDecodingSpace
            if object_name == action.arguments[role_id]:
                continue
            arguments = list(action.arguments)
            arguments[role_id] = object_name
            candidates.append(GroundAction(action.name, tuple(arguments)))
    return candidates


def _role_swaps(space: ActionDecodingSpace, action: GroundAction) -> list[GroundAction]:
    schema = space.parsed_problem.actions[action.name]
    candidates: list[GroundAction] = []
    for left in range(schema.arity):
        for right in range(left + 1, schema.arity):
            if action.arguments[left] == action.arguments[right]:
                continue
            swapped = list(action.arguments)
            swapped[left], swapped[right] = swapped[right], swapped[left]
            if tuple(swapped) == action.arguments:
                continue
            left_valid = _argument_is_type_compatible(space, schema.parameter_types[left], swapped[left])
            right_valid = _argument_is_type_compatible(
                space,
                schema.parameter_types[right],
                swapped[right],
            )
            if left_valid and right_valid:
                candidates.append(GroundAction(action.name, tuple(swapped)))
    return candidates


def _same_schema_candidates(space: ActionDecodingSpace, action: GroundAction) -> list[GroundAction]:
    return [
        candidate
        for candidate in space.enumerate_ground_actions()
        if candidate.name == action.name and candidate != action
    ]


def _other_schema_candidates(space: ActionDecodingSpace, action: GroundAction) -> list[GroundAction]:
    return [candidate for candidate in space.enumerate_ground_actions() if candidate.name != action.name]


def _argument_is_type_compatible(space: ActionDecodingSpace, type_name: str, object_name: str) -> bool:
    return object_name in set(space.objects_by_type.get(type_name, ()))


def _changed_roles(true_action: GroundAction, candidate: GroundAction) -> tuple[int, ...]:
    max_comparable = min(len(true_action.arguments), len(candidate.arguments))
    changed = [
        role_id
        for role_id in range(max_comparable)
        if true_action.arguments[role_id] != candidate.arguments[role_id]
    ]
    changed.extend(range(max_comparable, len(true_action.arguments)))
    return tuple(changed)


def _action_key(action: GroundAction) -> tuple[str, tuple[str, ...]]:
    return action.name, tuple(action.arguments)
