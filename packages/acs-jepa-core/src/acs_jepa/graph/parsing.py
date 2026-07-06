"""Parse PDDL domain/problem files into normalized symbolic structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unified_planning.io import PDDLReader

from acs_jepa.graph.schemas import (
    ActionSchema,
    GroundAtom,
    ObjectInfo,
    ParsedProblem,
    PredicateSchema,
)


def parse_domain_problem(domain_path: str | Path, problem_path: str | Path) -> ParsedProblem:
    """Parse a PDDL domain/problem pair with unified-planning."""

    problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))

    types = tuple(sorted(_type_name(t) for t in problem.user_types))
    predicates = {
        fluent.name: PredicateSchema(
            name=fluent.name,
            arg_types=tuple(_type_name(param.type) for param in fluent.signature),
        )
        for fluent in problem.fluents
        if _is_bool_type(fluent.type)
    }
    objects = {
        obj.name: ObjectInfo(name=obj.name, type=_type_name(obj.type))
        for obj in sorted(problem.all_objects, key=lambda item: item.name)
    }

    actions: dict[str, ActionSchema] = {}
    modified_predicates: set[str] = set()
    for action in problem.actions:
        action_modified = sorted(_modified_bool_predicates(action))
        modified_predicates.update(action_modified)
        actions[action.name] = ActionSchema(
            name=action.name,
            parameter_types=tuple(_type_name(param.type) for param in action.parameters),
            modified_predicates=tuple(action_modified),
        )

    initial_atoms = tuple(sorted(_initial_atoms(problem)))
    goal_atoms = tuple(sorted(atom for goal in problem.goals for atom in _atoms_from_expression(goal)))
    static_predicates = frozenset(set(predicates) - modified_predicates)

    return ParsedProblem(
        name=problem.name,
        types=types,
        predicates=predicates,
        objects=objects,
        actions=actions,
        initial_atoms=initial_atoms,
        goal_atoms=goal_atoms,
        static_predicates=static_predicates,
    )


def _type_name(type_expr: Any) -> str:
    return str(getattr(type_expr, "name", type_expr))


def _is_bool_type(type_expr: Any) -> bool:
    is_bool_type = getattr(type_expr, "is_bool_type", None)
    return bool(is_bool_type()) if callable(is_bool_type) else str(type_expr) == "bool"


def _initial_atoms(problem: Any) -> list[GroundAtom]:
    atoms = []
    for fluent_exp, value in problem.initial_values.items():
        if not _truth_value(value):
            continue
        atoms.extend(_atoms_from_expression(fluent_exp))
    return atoms


def _truth_value(value: Any) -> bool:
    is_true = getattr(value, "is_true", None)
    if callable(is_true):
        return bool(is_true())
    bool_constant_value = getattr(value, "bool_constant_value", None)
    if callable(bool_constant_value):
        return bool(bool_constant_value())
    return bool(value)


def _atoms_from_expression(expression: Any) -> list[GroundAtom]:
    if expression.is_and():
        return [atom for arg in expression.args for atom in _atoms_from_expression(arg)]
    if expression.is_fluent_exp():
        fluent = expression.fluent()
        if not _is_bool_type(fluent.type):
            return []
        return [
            GroundAtom(
                predicate=fluent.name,
                arguments=tuple(_object_name(arg) for arg in expression.args),
            )
        ]
    raise ValueError(f"Expected a positive grounded atom or conjunction, got: {expression}")


def _object_name(expression: Any) -> str:
    if expression.is_object_exp():
        return expression.object().name
    raise ValueError(f"Expected a grounded object argument, got: {expression}")


def _modified_bool_predicates(action: Any) -> set[str]:
    modified = set()
    for effect in action.effects:
        fluent_exp = effect.fluent
        if not fluent_exp.is_fluent_exp():
            continue
        fluent = fluent_exp.fluent()
        if _is_bool_type(fluent.type):
            modified.add(fluent.name)
    return modified
