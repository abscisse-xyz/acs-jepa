"""Small symbolic data structures for PDDL graph conversion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class GroundAtom:
    """A positive grounded predicate instance."""

    predicate: str
    arguments: tuple[str, ...]


@dataclass(frozen=True, order=True)
class GroundAction:
    """A grounded action application."""

    name: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class PredicateSchema:
    name: str
    arg_types: tuple[str, ...]


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    type: str


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameter_types: tuple[str, ...]
    modified_predicates: tuple[str, ...]

    @property
    def arity(self) -> int:
        return len(self.parameter_types)


@dataclass(frozen=True)
class ParsedProblem:
    """Normalized PDDL schema and problem facts."""

    name: str
    types: tuple[str, ...]
    predicates: dict[str, PredicateSchema]
    objects: dict[str, ObjectInfo]
    actions: dict[str, ActionSchema]
    initial_atoms: tuple[GroundAtom, ...]
    goal_atoms: tuple[GroundAtom, ...]
    static_predicates: frozenset[str]

    @property
    def type_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.types)}

    @property
    def predicate_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(sorted(self.predicates))}

    @property
    def action_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(sorted(self.actions))}

    @property
    def object_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(sorted(self.objects))}

    @property
    def max_action_arity(self) -> int:
        if not self.actions:
            return 0
        return max(action.arity for action in self.actions.values())
