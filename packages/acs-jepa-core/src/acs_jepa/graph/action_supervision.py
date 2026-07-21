"""Deterministic symbolic supervision tensors for grounded actions."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from acs_jepa.graph.builders import tensorize_action
from acs_jepa.graph.schemas import GroundAction, ParsedProblem

ONE_ARG_SUBSTITUTION = 0
ROLE_SWAP = 1
RANDOM_SAME_SCHEMA = 2
RANDOM_OTHER_SCHEMA = 3
NUM_NEGATIVE_CATEGORIES = 4
NEGATIVE_CATEGORY_NAMES = (
    "one_arg_substitution",
    "role_swap",
    "random_same_schema",
    "random_other_schema",
)

ApplicabilityLabeler = Callable[[GroundAction], bool | None]


@dataclass(frozen=True)
class SampledActionNegative:
    """One type-valid negative action and its diagnostic category metadata."""

    action: GroundAction
    category_id: int
    changed_roles: tuple[int, ...]


def sample_type_valid_action_negatives(
    parsed_problem: ParsedProblem,
    true_action: GroundAction,
    *,
    num_negatives: int,
    seed: int,
    max_random_attempts_per_category: int = 128,
) -> tuple[SampledActionNegative, ...]:
    """Return a bounded deterministic set of exact-type-valid negatives."""

    _validate_request(
        parsed_problem,
        true_action,
        num_negatives=num_negatives,
        max_random_attempts_per_category=max_random_attempts_per_category,
    )
    if num_negatives == 0:
        return ()

    rng = random.Random(seed)
    objects_by_type = _objects_by_type(parsed_problem)
    finite_candidates = {
        ONE_ARG_SUBSTITUTION: _one_argument_substitutions(
            parsed_problem, true_action, objects_by_type
        ),
        ROLE_SWAP: _role_swaps(parsed_problem, true_action),
    }
    for candidates in finite_candidates.values():
        rng.shuffle(candidates)
    finite_cursors = {category_id: 0 for category_id in finite_candidates}

    true_schema = parsed_problem.actions[true_action.name]
    true_domains = tuple(
        objects_by_type[type_name] for type_name in true_schema.parameter_types
    )
    eligible_other_schemas = tuple(
        schema_name
        for schema_name in sorted(parsed_problem.actions)
        if schema_name != true_action.name
        and all(
            objects_by_type.get(type_name, ())
            for type_name in parsed_problem.actions[schema_name].parameter_types
        )
    )
    random_attempts = {RANDOM_SAME_SCHEMA: 0, RANDOM_OTHER_SCHEMA: 0}
    true_key = _action_key(true_action)
    seen = {true_key}
    sampled: list[SampledActionNegative] = []

    while len(sampled) < num_negatives:
        visited_category = False
        for category_id in range(NUM_NEGATIVE_CATEGORIES):
            if len(sampled) >= num_negatives:
                break
            candidate: GroundAction | None = None
            if category_id in finite_candidates:
                cursor = finite_cursors[category_id]
                candidates = finite_candidates[category_id]
                if cursor >= len(candidates):
                    continue
                candidate = candidates[cursor]
                finite_cursors[category_id] = cursor + 1
                visited_category = True
            elif random_attempts[category_id] >= max_random_attempts_per_category:
                continue
            elif category_id == RANDOM_OTHER_SCHEMA and not eligible_other_schemas:
                continue
            else:
                random_attempts[category_id] += 1
                visited_category = True
                if category_id == RANDOM_SAME_SCHEMA:
                    arguments = tuple(rng.choice(domain) for domain in true_domains)
                    candidate = GroundAction(true_action.name, arguments)
                else:
                    schema_name = rng.choice(eligible_other_schemas)
                    schema = parsed_problem.actions[schema_name]
                    arguments = tuple(
                        rng.choice(objects_by_type[type_name])
                        for type_name in schema.parameter_types
                    )
                    candidate = GroundAction(schema_name, arguments)

            if candidate is None or _action_key(candidate) in seen:
                continue
            seen.add(_action_key(candidate))
            sampled.append(
                SampledActionNegative(
                    action=candidate,
                    category_id=category_id,
                    changed_roles=_changed_roles(true_action, candidate),
                )
            )
        if not visited_category:
            break
    return tuple(sampled)


def build_action_supervision_tensors(
    parsed_problem: ParsedProblem,
    true_action: GroundAction,
    *,
    max_action_arity: int,
    max_objects: int,
    num_negatives: int,
    seed: int,
    applicability_labeler: ApplicabilityLabeler | None = None,
    max_random_attempts_per_category: int = 128,
) -> dict[str, Tensor]:
    """Build fixed-shape negative-action and role-object supervision tensors."""

    _validate_request(
        parsed_problem,
        true_action,
        num_negatives=num_negatives,
        max_random_attempts_per_category=max_random_attempts_per_category,
    )
    if max_action_arity < parsed_problem.max_action_arity:
        raise ValueError("max_action_arity must cover the parsed-problem maximum")
    if max_objects < len(parsed_problem.objects):
        raise ValueError("max_objects must cover every problem object")

    negatives = sample_type_valid_action_negatives(
        parsed_problem,
        true_action,
        num_negatives=num_negatives,
        seed=seed,
        max_random_attempts_per_category=max_random_attempts_per_category,
    )
    true_tensors = tensorize_action(
        parsed_problem,
        true_action,
        max_arity=max_action_arity,
    )
    object_mask = torch.zeros(max_objects, dtype=torch.bool)
    object_mask[: len(parsed_problem.objects)] = True
    argument_candidate_mask = torch.zeros(
        (max_action_arity, max_objects), dtype=torch.bool
    )
    schema = parsed_problem.actions[true_action.name]
    object_to_id = parsed_problem.object_to_id
    for role_id, type_name in enumerate(schema.parameter_types):
        for object_name, object_info in parsed_problem.objects.items():
            if object_info.type == type_name:
                argument_candidate_mask[role_id, object_to_id[object_name]] = True

    output = _empty_supervision_tensors(
        num_negatives=num_negatives,
        max_action_arity=max_action_arity,
    )
    output.update(
        {
            "argument_target_indices": true_tensors[
                "action_object_indices"
            ].clone(),
            "argument_mask": true_tensors["action_arg_mask"].clone(),
            "argument_candidate_mask": argument_candidate_mask,
            "object_mask": object_mask,
        }
    )
    _fill_negative_tensors(
        output,
        parsed_problem,
        negatives,
        max_action_arity=max_action_arity,
        applicability_labeler=applicability_labeler,
    )
    return output


def _objects_by_type(parsed_problem: ParsedProblem) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {type_name: [] for type_name in parsed_problem.types}
    for object_name, object_info in parsed_problem.objects.items():
        grouped.setdefault(object_info.type, []).append(object_name)
    return {
        type_name: tuple(sorted(object_names))
        for type_name, object_names in grouped.items()
    }


def _one_argument_substitutions(
    parsed_problem: ParsedProblem,
    true_action: GroundAction,
    objects_by_type: dict[str, tuple[str, ...]],
) -> list[GroundAction]:
    schema = parsed_problem.actions[true_action.name]
    candidates: list[GroundAction] = []
    for role_id, type_name in enumerate(schema.parameter_types):
        for object_name in objects_by_type[type_name]:
            if object_name == true_action.arguments[role_id]:
                continue
            arguments = list(true_action.arguments)
            arguments[role_id] = object_name
            candidates.append(GroundAction(true_action.name, tuple(arguments)))
    return candidates


def _role_swaps(
    parsed_problem: ParsedProblem,
    true_action: GroundAction,
) -> list[GroundAction]:
    schema = parsed_problem.actions[true_action.name]
    candidates: list[GroundAction] = []
    for left in range(schema.arity):
        for right in range(left + 1, schema.arity):
            left_object = true_action.arguments[left]
            right_object = true_action.arguments[right]
            if left_object == right_object:
                continue
            if parsed_problem.objects[right_object].type != schema.parameter_types[left]:
                continue
            if parsed_problem.objects[left_object].type != schema.parameter_types[right]:
                continue
            arguments = list(true_action.arguments)
            arguments[left], arguments[right] = arguments[right], arguments[left]
            candidates.append(GroundAction(true_action.name, tuple(arguments)))
    return candidates


def _changed_roles(
    true_action: GroundAction,
    candidate: GroundAction,
) -> tuple[int, ...]:
    changed: list[int] = []
    max_arity = max(len(true_action.arguments), len(candidate.arguments))
    for role_id in range(max_arity):
        if role_id >= len(true_action.arguments) or role_id >= len(candidate.arguments):
            changed.append(role_id)
        elif true_action.arguments[role_id] != candidate.arguments[role_id]:
            changed.append(role_id)
    return tuple(changed)


def _action_key(action: GroundAction) -> tuple[str, tuple[str, ...]]:
    return action.name, tuple(action.arguments)


def _validate_request(
    parsed_problem: ParsedProblem,
    true_action: GroundAction,
    *,
    num_negatives: int,
    max_random_attempts_per_category: int,
) -> None:
    if num_negatives < 0:
        raise ValueError("num_negatives must be non-negative")
    if max_random_attempts_per_category <= 0:
        raise ValueError("max_random_attempts_per_category must be positive")
    if true_action.name not in parsed_problem.actions:
        raise ValueError(f"unknown true action schema: {true_action.name}")
    schema = parsed_problem.actions[true_action.name]
    if len(true_action.arguments) != schema.arity:
        raise ValueError(
            f"true action {true_action.name} expects {schema.arity} arguments"
        )
    for role_id, (object_name, expected_type) in enumerate(
        zip(true_action.arguments, schema.parameter_types, strict=True)
    ):
        if object_name not in parsed_problem.objects:
            raise ValueError(f"unknown true action object: {object_name}")
        actual_type = parsed_problem.objects[object_name].type
        if actual_type != expected_type:
            raise ValueError(
                f"true action role {role_id} expects type {expected_type}, got {actual_type}"
            )


def _empty_supervision_tensors(
    *,
    num_negatives: int,
    max_action_arity: int,
) -> dict[str, Tensor]:
    return {
        "negative_action_id": torch.zeros(num_negatives, dtype=torch.long),
        "negative_action_object_indices": torch.full(
            (num_negatives, max_action_arity), -1, dtype=torch.long
        ),
        "negative_action_role_ids": torch.full(
            (num_negatives, max_action_arity), -1, dtype=torch.long
        ),
        "negative_action_arg_mask": torch.zeros(
            (num_negatives, max_action_arity), dtype=torch.bool
        ),
        "negative_mask": torch.zeros(num_negatives, dtype=torch.bool),
        "negative_category_id": torch.full(
            (num_negatives,), -1, dtype=torch.long
        ),
        "negative_changed_role_mask": torch.zeros(
            (num_negatives, max_action_arity), dtype=torch.bool
        ),
        "negative_applicability_label": torch.zeros(
            num_negatives, dtype=torch.float32
        ),
        "negative_applicability_label_mask": torch.zeros(
            num_negatives, dtype=torch.bool
        ),
    }


def _fill_negative_tensors(
    output: dict[str, Tensor],
    parsed_problem: ParsedProblem,
    negatives: tuple[SampledActionNegative, ...],
    *,
    max_action_arity: int,
    applicability_labeler: ApplicabilityLabeler | None,
) -> None:
    for index, negative in enumerate(negatives):
        tensors = tensorize_action(
            parsed_problem,
            negative.action,
            max_arity=max_action_arity,
        )
        output["negative_action_id"][index] = tensors["action_id"]
        output["negative_action_object_indices"][index] = tensors[
            "action_object_indices"
        ]
        output["negative_action_role_ids"][index] = tensors["action_role_ids"]
        output["negative_action_arg_mask"][index] = tensors["action_arg_mask"]
        output["negative_mask"][index] = True
        output["negative_category_id"][index] = negative.category_id
        for role_id in negative.changed_roles:
            output["negative_changed_role_mask"][index, role_id] = True
        if applicability_labeler is not None:
            label = applicability_labeler(negative.action)
            if label is not None and type(label) is not bool:
                raise ValueError("applicability_labeler must return bool or None")
            if label is not None:
                output["negative_applicability_label"][index] = float(label)
                output["negative_applicability_label_mask"][index] = True
