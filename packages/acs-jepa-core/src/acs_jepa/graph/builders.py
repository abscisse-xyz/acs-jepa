"""Build PyTorch Geometric factor graphs from grounded PDDL atoms.

Graph encoding
--------------
Each symbolic state is a set of positive grounded atoms over a problem-local
object set. The graph uses a factor-node representation:

* object nodes represent PDDL objects;
* atom nodes represent grounded predicates;
* role-labeled edges connect each atom node to its argument object nodes.

For example, ``road_connect(road0, j0, j1)`` becomes one atom node connected to
three object nodes with role ids ``0``, ``1``, and ``2``. This keeps the state
invariant to atom serialization order while preserving n-ary predicate argument
order.

The returned :class:`torch_geometric.data.Data` object has:

``x: LongTensor[num_nodes, 5]``
    Node feature table with columns:

    0. ``node_kind``: ``OBJECT_NODE`` or ``ATOM_NODE``.
    1. ``object_type``: type id for object nodes, ``-1`` for atom nodes.
    2. ``predicate_id``: predicate id for atom nodes, ``-1`` for object nodes.
    3. ``arity``: atom arity for atom nodes, ``0`` for object nodes.
    4. ``object_id``: problem-local object id for object nodes, ``-1`` for
       atom nodes.

``edge_index: LongTensor[2, num_edges]``
    Bidirectional edges between atom nodes and their argument object nodes.

``edge_attr: LongTensor[num_edges, 2]``
    Edge feature table with columns:

    0. ``role_id``: argument position in the predicate.
    1. ``edge_direction``: ``ATOM_TO_OBJECT`` or ``OBJECT_TO_ATOM``.

The object ids, predicate ids, action ids, and type ids are stable within a
``ParsedProblem``. Object ids are intentionally problem-local because PDDL
object names are not global identities across problem instances.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from acs_jepa.graph.schemas import GroundAction, GroundAtom, ParsedProblem

OBJECT_NODE = 0
ATOM_NODE = 1
ATOM_TO_OBJECT = 0
OBJECT_TO_ATOM = 1


def build_state_graph(
    parsed_problem: ParsedProblem,
    atoms: tuple[GroundAtom, ...] | list[GroundAtom],
    *,
    include_static: bool = True,
) -> Data:
    """Build a factor-node graph for one symbolic state.

    Args:
        parsed_problem: Parsed domain/problem metadata containing object,
            predicate, type, and static-predicate vocabularies.
        atoms: Positive grounded atoms active in the state.
        include_static: If false, atoms whose predicates are never modified by
            any action effect are excluded from the state graph.

    Returns:
        A PyG ``Data`` object following the module-level graph encoding. Object
        nodes are ordered first by object name; atom nodes follow in canonical
        sorted atom order.
    """

    object_names = sorted(parsed_problem.objects)
    object_to_node = {name: idx for idx, name in enumerate(object_names)}
    type_to_id = parsed_problem.type_to_id
    predicate_to_id = parsed_problem.predicate_to_id

    state_atoms = tuple(sorted(_filter_atoms(parsed_problem, atoms, include_static=include_static)))
    num_objects = len(object_names)
    num_nodes = num_objects + len(state_atoms)

    x = torch.full((num_nodes, 5), -1, dtype=torch.long)
    for node_idx, object_name in enumerate(object_names):
        object_info = parsed_problem.objects[object_name]
        x[node_idx] = torch.tensor(
            [
                OBJECT_NODE,
                type_to_id[object_info.type],
                -1,
                0,
                parsed_problem.object_to_id[object_name],
            ],
            dtype=torch.long,
        )

    edges: list[tuple[int, int]] = []
    edge_attrs: list[tuple[int, int]] = []
    atom_predicate_ids = []
    atom_arities = []

    for atom_offset, atom in enumerate(state_atoms):
        node_idx = num_objects + atom_offset
        predicate_id = predicate_to_id[atom.predicate]
        arity = len(atom.arguments)
        x[node_idx] = torch.tensor(
            [ATOM_NODE, -1, predicate_id, arity, -1],
            dtype=torch.long,
        )
        atom_predicate_ids.append(predicate_id)
        atom_arities.append(arity)

        for role_id, object_name in enumerate(atom.arguments):
            object_node = object_to_node[object_name]
            edges.append((node_idx, object_node))
            edge_attrs.append((role_id, ATOM_TO_OBJECT))
            edges.append((object_node, node_idx))
            edge_attrs.append((role_id, OBJECT_TO_ATOM))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_kind=x[:, 0],
        object_type=x[:, 1],
        predicate_id=x[:, 2],
        atom_arity=torch.tensor(atom_arities, dtype=torch.long),
        atom_predicate_id=torch.tensor(atom_predicate_ids, dtype=torch.long),
        num_objects=num_objects,
        num_atoms=len(state_atoms),
    )


def tensorize_action(
    parsed_problem: ParsedProblem,
    action: GroundAction,
    *,
    max_arity: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a grounded action to fixed-size tensors for batching.

    The action encoding is meant to be consumed together with object embeddings
    from ``build_state_graph``. A model can gather contextual object embeddings
    with ``action_object_indices`` and combine them with ``action_id`` and
    ``action_role_ids``.

    Returns:
        ``action_id``:
            Scalar action-schema id.
        ``action_object_indices``:
            ``LongTensor[max_arity]`` of problem-local object ids. Unused padded
            entries are ``-1``.
        ``action_role_ids``:
            ``LongTensor[max_arity]`` containing argument positions. Unused
            padded entries are ``-1``.
        ``action_arg_mask``:
            ``BoolTensor[max_arity]`` indicating which entries are real action
            arguments.
    """

    if action.name not in parsed_problem.actions:
        raise KeyError(f"Unknown action: {action.name}")

    action_schema = parsed_problem.actions[action.name]
    if len(action.arguments) != action_schema.arity:
        raise ValueError(f"Action {action.name} expects {action_schema.arity} arguments, got {len(action.arguments)}")

    arity = parsed_problem.max_action_arity if max_arity is None else max_arity
    if len(action.arguments) > arity:
        raise ValueError(f"Action {action.name} has arity {len(action.arguments)}, larger than max_arity={arity}")

    object_to_id = parsed_problem.object_to_id
    object_indices = torch.full((arity,), -1, dtype=torch.long)
    role_ids = torch.full((arity,), -1, dtype=torch.long)
    mask = torch.zeros((arity,), dtype=torch.bool)
    for role_id, object_name in enumerate(action.arguments):
        object_indices[role_id] = object_to_id[object_name]
        role_ids[role_id] = role_id
        mask[role_id] = True

    return {
        "action_id": torch.tensor(parsed_problem.action_to_id[action.name], dtype=torch.long),
        "action_object_indices": object_indices,
        "action_role_ids": role_ids,
        "action_arg_mask": mask,
    }


def tensorize_predicate(
    parsed_problem: ParsedProblem,
    atom: GroundAtom,
    *,
    max_arity: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a grounded predicate atom to fixed-size tensors for batching.

    The predicate encoding is meant to be consumed together with object
    embeddings from ``build_state_graph``. A model can gather contextual object
    embeddings with ``predicate_object_indices`` and combine them with
    ``predicate_id`` and ``predicate_role_ids``. This is especially useful for
    goal/fact satisfaction heads, where a partial PDDL goal should be treated
    as a set of grounded predicate queries instead of as a complete state.

    Returns:
        ``predicate_id``:
            Scalar predicate-schema id.
        ``predicate_object_indices``:
            ``LongTensor[max_arity]`` of problem-local object ids. Unused
            padded entries are ``-1``.
        ``predicate_role_ids``:
            ``LongTensor[max_arity]`` containing argument positions. Unused
            padded entries are ``-1``.
        ``predicate_arg_mask``:
            ``BoolTensor[max_arity]`` indicating which entries are real
            predicate arguments.
    """

    if atom.predicate not in parsed_problem.predicates:
        raise KeyError(f"Unknown predicate: {atom.predicate}")

    predicate_schema = parsed_problem.predicates[atom.predicate]
    if len(atom.arguments) != len(predicate_schema.arg_types):
        raise ValueError(
            f"Predicate {atom.predicate} expects {len(predicate_schema.arg_types)} arguments, got {len(atom.arguments)}"
        )

    arity = _max_predicate_arity(parsed_problem) if max_arity is None else max_arity
    if len(atom.arguments) > arity:
        raise ValueError(f"Predicate {atom.predicate} has arity {len(atom.arguments)}, larger than max_arity={arity}")

    object_to_id = parsed_problem.object_to_id
    object_indices = torch.full((arity,), -1, dtype=torch.long)
    role_ids = torch.full((arity,), -1, dtype=torch.long)
    mask = torch.zeros((arity,), dtype=torch.bool)
    for role_id, object_name in enumerate(atom.arguments):
        object_indices[role_id] = object_to_id[object_name]
        role_ids[role_id] = role_id
        mask[role_id] = True

    return {
        "predicate_id": torch.tensor(parsed_problem.predicate_to_id[atom.predicate], dtype=torch.long),
        "predicate_object_indices": object_indices,
        "predicate_role_ids": role_ids,
        "predicate_arg_mask": mask,
    }


def tensorize_goal_atoms(
    parsed_problem: ParsedProblem,
    atoms: tuple[GroundAtom, ...] | list[GroundAtom],
    *,
    max_atoms: int | None = None,
    max_arity: int | None = None,
    weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a positive conjunctive partial goal to padded tensors.

    The returned tensors represent a set of grounded predicate queries, not a
    complete state graph. They are intended for latent-space goal scorers that
    condition on partial goals while scoring a full terminal ``JEPALatentState``.

    Returns:
        ``goal_predicate_id``:
            ``LongTensor[max_atoms]`` predicate-schema ids, padded with ``-1``.
        ``goal_object_indices``:
            ``LongTensor[max_atoms, max_arity]`` problem-local object ids.
        ``goal_role_ids``:
            ``LongTensor[max_atoms, max_arity]`` argument positions.
        ``goal_arg_mask``:
            ``BoolTensor[max_atoms, max_arity]`` marking real arguments.
        ``goal_atom_mask``:
            ``BoolTensor[max_atoms]`` marking real goal atoms.
        ``goal_weight``:
            ``FloatTensor[max_atoms]`` atom weights, zero for padding.
        ``goal_truth``:
            ``BoolTensor[max_atoms]`` desired truth values. V1 supports
            positive atoms, so real atoms are ``True``.
    """

    goal_atoms = tuple(atoms)
    num_atoms = len(goal_atoms)
    atom_capacity = num_atoms if max_atoms is None else max_atoms
    if num_atoms > atom_capacity:
        raise ValueError(f"Goal has {num_atoms} atoms, larger than max_atoms={atom_capacity}")

    arity = _max_predicate_arity(parsed_problem) if max_arity is None else max_arity
    atom_weights = _goal_weights(num_atoms, weights)

    predicate_ids = torch.full((atom_capacity,), -1, dtype=torch.long)
    object_indices = torch.full((atom_capacity, arity), -1, dtype=torch.long)
    role_ids = torch.full((atom_capacity, arity), -1, dtype=torch.long)
    arg_mask = torch.zeros((atom_capacity, arity), dtype=torch.bool)
    atom_mask = torch.zeros((atom_capacity,), dtype=torch.bool)
    goal_weight = torch.zeros((atom_capacity,), dtype=torch.float32)
    goal_truth = torch.zeros((atom_capacity,), dtype=torch.bool)

    for atom_idx, atom in enumerate(goal_atoms):
        predicate_tensors = tensorize_predicate(parsed_problem, atom, max_arity=arity)
        predicate_ids[atom_idx] = predicate_tensors["predicate_id"]
        object_indices[atom_idx] = predicate_tensors["predicate_object_indices"]
        role_ids[atom_idx] = predicate_tensors["predicate_role_ids"]
        arg_mask[atom_idx] = predicate_tensors["predicate_arg_mask"]
        atom_mask[atom_idx] = True
        goal_weight[atom_idx] = atom_weights[atom_idx]
        goal_truth[atom_idx] = True

    return {
        "goal_predicate_id": predicate_ids,
        "goal_object_indices": object_indices,
        "goal_role_ids": role_ids,
        "goal_arg_mask": arg_mask,
        "goal_atom_mask": atom_mask,
        "goal_weight": goal_weight,
        "goal_truth": goal_truth,
    }


def _filter_atoms(
    parsed_problem: ParsedProblem,
    atoms: tuple[GroundAtom, ...] | list[GroundAtom],
    *,
    include_static: bool,
) -> list[GroundAtom]:
    if include_static:
        return list(atoms)
    return [atom for atom in atoms if atom.predicate not in parsed_problem.static_predicates]


def _max_predicate_arity(parsed_problem: ParsedProblem) -> int:
    if not parsed_problem.predicates:
        return 0
    return max(len(predicate.arg_types) for predicate in parsed_problem.predicates.values())


def _goal_weights(num_atoms: int, weights: torch.Tensor | list[float] | tuple[float, ...] | None) -> torch.Tensor:
    if weights is None:
        return torch.ones((num_atoms,), dtype=torch.float32)
    result = torch.as_tensor(weights, dtype=torch.float32)
    if result.shape != (num_atoms,):
        raise ValueError(f"Expected {num_atoms} goal weights, got shape {tuple(result.shape)}")
    return result
