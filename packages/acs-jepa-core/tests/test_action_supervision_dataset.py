from __future__ import annotations

import random
from dataclasses import FrozenInstanceError, replace
from multiprocessing.reduction import ForkingPickler

import pytest
import torch
from acs_jepa.graph.schemas import (
    ActionSchema,
    GroundAction,
    GroundAtom,
    ObjectInfo,
    ParsedProblem,
    PredicateSchema,
)
from torch_geometric.loader import DataLoader


def _problem(*, name: str = "problem-0", extra_location: bool = False) -> ParsedProblem:
    objects = {
        "item0": ObjectInfo("item0", "item"),
        "l0": ObjectInfo("l0", "location"),
        "l1": ObjectInfo("l1", "location"),
    }
    if extra_location:
        objects["l2"] = ObjectInfo("l2", "location")
    initial_atoms = (
        GroundAtom("at", ("item0", "l0")),
        GroundAtom("clear", ("l1",)),
    )
    return ParsedProblem(
        name=name,
        types=("item", "location"),
        predicates={
            "at": PredicateSchema("at", ("item", "location")),
            "clear": PredicateSchema("clear", ("location",)),
        },
        objects=objects,
        actions={
            "move": ActionSchema(
                "move", ("item", "location", "location"), ("at", "clear")
            ),
            "wait": ActionSchema("wait", (), ()),
        },
        initial_atoms=initial_atoms,
        goal_atoms=(GroundAtom("at", ("item0", "l1")),),
        static_predicates=frozenset(),
    )


def _trajectory(problem_index: int = 0):
    from acs_jepa.graph import TrajectorySample

    state0 = (
        GroundAtom("at", ("item0", "l0")),
        GroundAtom("clear", ("l1",)),
    )
    state1 = (
        GroundAtom("at", ("item0", "l1")),
        GroundAtom("clear", ("l0",)),
    )
    state2 = state0
    return TrajectorySample(
        problem_index=problem_index,
        states=(state0, state1, state2),
        actions=(
            GroundAction("move", ("item0", "l0", "l1")),
            GroundAction("move", ("item0", "l1", "l0")),
        ),
        terminal_atoms=state2,
    )


def test_action_supervision_dataset_disabled_and_zero_negative_contract() -> None:
    from acs_jepa.graph import (
        ActionSupervisionConfig,
        PDDLTrajectoryDataset,
    )

    problem = _problem()
    trajectory = _trajectory()
    disabled = PDDLTrajectoryDataset([problem], [trajectory])
    assert "action_supervision" not in disabled[0]

    enabled = PDDLTrajectoryDataset(
        [problem],
        [trajectory],
        action_supervision=ActionSupervisionConfig(num_negatives=0),
    )
    supervision = enabled[0]["action_supervision"]

    assert supervision["negative_action_id"].shape == (2, 0)
    assert supervision["negative_action_object_indices"].shape == (2, 0, 3)
    assert supervision["negative_action_role_ids"].shape == (2, 0, 3)
    assert supervision["negative_action_arg_mask"].shape == (2, 0, 3)
    assert supervision["negative_mask"].shape == (2, 0)
    assert supervision["negative_category_id"].shape == (2, 0)
    assert supervision["negative_changed_role_mask"].shape == (2, 0, 3)
    assert supervision["negative_applicability_label"].shape == (2, 0)
    assert supervision["negative_applicability_label_mask"].shape == (2, 0)
    assert supervision["argument_target_indices"].shape == (2, 3)
    assert supervision["argument_mask"].shape == (2, 3)
    assert supervision["argument_candidate_mask"].shape == (2, 3, 3)
    assert supervision["object_mask"].shape == (2, 3)
    assert supervision["negative_action_id"].dtype == torch.long
    assert supervision["negative_mask"].dtype == torch.bool
    assert supervision["negative_applicability_label"].dtype == torch.float32


def _assert_tensor_dict_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def _all_ground_actions(problem: ParsedProblem) -> frozenset[GroundAction]:
    locations = tuple(
        name for name, info in problem.objects.items() if info.type == "location"
    )
    return frozenset(
        {GroundAction("wait", ())}
        | {
            GroundAction("move", ("item0", source, target))
            for source in locations
            for target in locations
        }
    )


def test_time_axis_matches_direct_builder_and_content_seed_is_stable() -> None:
    from acs_jepa.graph import (
        ActionSupervisionConfig,
        PDDLTrajectoryDataset,
        build_action_supervision_tensors,
    )
    from acs_jepa.graph.dataset import _action_supervision_seed

    problem = _problem()
    trajectory = _trajectory()
    config = ActionSupervisionConfig(num_negatives=5, seed=13)
    before = random.getstate()
    dataset = PDDLTrajectoryDataset(
        [problem], [trajectory, trajectory], action_supervision=config
    )
    actual = dataset[0]["action_supervision"]
    assert random.getstate() == before
    _assert_tensor_dict_equal(actual, dataset[1]["action_supervision"])

    expected_steps = []
    for step, action in enumerate(trajectory.actions):
        expected_steps.append(
            build_action_supervision_tensors(
                problem,
                action,
                max_action_arity=3,
                max_objects=3,
                num_negatives=5,
                seed=_action_supervision_seed(
                    13, trajectory.problem_index, trajectory.states[step], action
                ),
            )
        )
    expected = {
        key: torch.stack([step[key] for step in expected_steps])
        for key in expected_steps[0]
    }
    _assert_tensor_dict_equal(actual, expected)
    first_seed = _action_supervision_seed(
        13, 0, trajectory.states[0], trajectory.actions[0]
    )
    seeds = {
        first_seed,
        _action_supervision_seed(13, 0, trajectory.states[1], trajectory.actions[0]),
        _action_supervision_seed(13, 0, trajectory.states[0], trajectory.actions[1]),
        _action_supervision_seed(13, 1, trajectory.states[0], trajectory.actions[0]),
    }
    assert len(seeds) == 4


def test_nested_supervision_batches_different_object_counts() -> None:
    from acs_jepa.graph import ActionSupervisionConfig, PDDLTrajectoryDataset

    problems = [_problem(), _problem(name="problem-1", extra_location=True)]
    trajectories = [_trajectory(0), _trajectory(1)]
    dataset = PDDLTrajectoryDataset(
        problems,
        trajectories,
        action_supervision=ActionSupervisionConfig(num_negatives=4, seed=5),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    supervision = batch["action_supervision"]
    assert supervision["negative_action_id"].shape == (2, 2, 4)
    assert supervision["negative_action_object_indices"].shape == (2, 2, 4, 3)
    assert supervision["argument_target_indices"].shape == (2, 2, 3)
    assert supervision["argument_candidate_mask"].shape == (2, 2, 3, 4)
    assert supervision["object_mask"].shape == (2, 2, 4)
    assert torch.equal(supervision["object_mask"][0, 0], torch.tensor([1, 1, 1, 0], dtype=torch.bool))
    assert supervision["object_mask"].dtype == torch.bool
    assert supervision["negative_category_id"].dtype == torch.long
    assert supervision["negative_applicability_label"].dtype == torch.float32
    assert set(supervision) == {
        "negative_action_id",
        "negative_action_object_indices",
        "negative_action_role_ids",
        "negative_action_arg_mask",
        "negative_mask",
        "negative_category_id",
        "negative_changed_role_mask",
        "negative_applicability_label",
        "negative_applicability_label_mask",
        "argument_target_indices",
        "argument_mask",
        "argument_candidate_mask",
        "object_mask",
    }
    for sample_index in range(2):
        expected = dataset[sample_index]["action_supervision"]
        for key in supervision:
            assert torch.equal(supervision[key][sample_index], expected[key]), key


def test_immutable_offline_table_labels_current_state_without_leakage() -> None:
    from acs_jepa.graph import (
        ATOM_STATE_APPLICABILITY_SEMANTICS,
        ActionApplicabilityTable,
        ActionSupervisionConfig,
        PDDLTrajectoryDataset,
        action_applicability_state_key,
    )

    problem = _problem()
    trajectory = _trajectory()
    state0_key = action_applicability_state_key(
        0, tuple(reversed(trajectory.states[0]))
    )
    state1_key = action_applicability_state_key(0, trajectory.states[1])
    source = {
        state0_key: {trajectory.actions[0]},
        state1_key: set(_all_ground_actions(problem)),
    }
    config = ActionSupervisionConfig(
        num_negatives=4,
        seed=3,
        applicable_actions_by_state=source,
        applicability_state_semantics=ATOM_STATE_APPLICABILITY_SEMANTICS,
    )
    assert type(config.applicable_actions_by_state) is ActionApplicabilityTable
    source[state0_key].add(GroundAction("wait", ()))
    restored = ForkingPickler.loads(ForkingPickler.dumps(config))  # type: ignore[attr-defined]
    assert restored == config
    with pytest.raises(TypeError):
        config.applicable_actions_by_state[state0_key] = frozenset()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        config.seed = 99  # type: ignore[misc]

    dataset = PDDLTrajectoryDataset([problem], [trajectory], action_supervision=config)
    supervision = dataset[0]["action_supervision"]
    masks = supervision["negative_applicability_label_mask"]
    labels = supervision["negative_applicability_label"]
    assert masks[0].equal(supervision["negative_mask"][0])
    assert masks[1].equal(supervision["negative_mask"][1])
    assert not labels[0, masks[0]].any()
    assert labels[1, masks[1]].all()

    unknown = PDDLTrajectoryDataset(
        [problem],
        [trajectory],
        action_supervision=ActionSupervisionConfig(num_negatives=4, seed=3),
    )[0]["action_supervision"]
    for key in supervision:
        if key not in {
            "negative_applicability_label",
            "negative_applicability_label_mask",
        }:
            assert torch.equal(supervision[key], unknown[key]), key

    no_static = PDDLTrajectoryDataset(
        [problem], [trajectory], include_static=False, action_supervision=config
    )[0]["action_supervision"]
    _assert_tensor_dict_equal(supervision, no_static)


def test_atom_trajectory_dataset_emits_identical_action_supervision() -> None:
    from acs_jepa.graph import (
        ActionSupervisionConfig,
        PDDLAtomTrajectoryDataset,
        PDDLTrajectoryDataset,
    )

    problem = _problem()
    trajectory = _trajectory()
    config = ActionSupervisionConfig(num_negatives=4, seed=19)
    plain = PDDLTrajectoryDataset(
        [problem], [trajectory], action_supervision=config
    )[0]
    atom_kwargs = {
        "num_positive_atoms": 1,
        "num_negative_atoms": 1,
        "include_goal": True,
        "include_terminal_state": True,
        "seed": 17,
    }
    atom = PDDLAtomTrajectoryDataset(
        [problem],
        [trajectory],
        **atom_kwargs,
        action_supervision=config,
    )[0]
    disabled_atom = PDDLAtomTrajectoryDataset(
        [problem], [trajectory], **atom_kwargs
    )[0]
    assert "action_supervision" not in disabled_atom
    _assert_tensor_dict_equal(atom["atom_queries"], disabled_atom["atom_queries"])
    _assert_tensor_dict_equal(atom["actions"], disabled_atom["actions"])
    _assert_tensor_dict_equal(
        plain["action_supervision"], atom["action_supervision"]
    )
    assert "atom_queries" in atom
    assert "goal" in atom
    assert "terminal_state" in atom


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_negatives": True}, "num_negatives"),
        ({"num_negatives": 0, "seed": False}, "seed"),
        (
            {"num_negatives": 0, "max_random_attempts_per_category": 0},
            "positive",
        ),
        ({"num_negatives": -1}, "non-negative"),
    ],
)
def test_action_supervision_config_rejects_invalid_scalars(
    kwargs: dict, message: str
) -> None:
    from acs_jepa.graph import ActionSupervisionConfig

    with pytest.raises(ValueError, match=message):
        ActionSupervisionConfig(**kwargs)


def test_oracle_key_table_and_semantics_validation() -> None:
    from acs_jepa.graph import (
        ATOM_STATE_APPLICABILITY_SEMANTICS,
        ActionApplicabilityStateKey,
        ActionApplicabilityTable,
        ActionSupervisionConfig,
        action_applicability_state_key,
    )

    atom = GroundAtom("clear", ("l0",))
    with pytest.raises(ValueError, match="problem_index"):
        ActionApplicabilityStateKey(True, (atom,))
    with pytest.raises(ValueError, match="exact tuple"):
        ActionApplicabilityStateKey(0, [atom])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sorted and duplicate-free"):
        ActionApplicabilityStateKey(0, (atom, atom))
    key = action_applicability_state_key(0, (atom, atom))
    assert key.state_atoms == (atom,)

    with pytest.raises(ValueError, match="sorted unique"):
        ActionApplicabilityTable(((key, frozenset()), (key, frozenset())))
    with pytest.raises(ValueError, match="GroundAction"):
        ActionApplicabilityTable.from_mapping({key: {"bad"}})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="semantics"):
        ActionSupervisionConfig(num_negatives=0, applicable_actions_by_state={})
    with pytest.raises(ValueError, match="requires an oracle table"):
        ActionSupervisionConfig(
            num_negatives=0,
            applicability_state_semantics=ATOM_STATE_APPLICABILITY_SEMANTICS,
        )


def test_dataset_rejects_stale_or_malformed_complete_oracle() -> None:
    from acs_jepa.graph import (
        ATOM_STATE_APPLICABILITY_SEMANTICS,
        ActionSupervisionConfig,
        PDDLTrajectoryDataset,
        action_applicability_state_key,
    )

    problem = _problem()
    trajectory = _trajectory()

    def config_for(table: dict) -> ActionSupervisionConfig:
        return ActionSupervisionConfig(
            num_negatives=2,
            applicable_actions_by_state=table,
            applicability_state_semantics=ATOM_STATE_APPLICABILITY_SEMANTICS,
        )

    out_of_range = {
        action_applicability_state_key(1, trajectory.states[0]): {
            trajectory.actions[0]
        }
    }
    with pytest.raises(ValueError, match="out-of-range problem index"):
        PDDLTrajectoryDataset(
            [problem], [trajectory], action_supervision=config_for(out_of_range)
        )

    bad_state = {
        action_applicability_state_key(
            0, (GroundAtom("unknown", ("item0",)),)
        ): {trajectory.actions[0]}
    }
    with pytest.raises(ValueError, match="unknown predicate"):
        PDDLTrajectoryDataset(
            [problem], [trajectory], action_supervision=config_for(bad_state)
        )

    malformed_states = [
        (GroundAtom("at", ("item0",)), "wrong arity"),
        (GroundAtom("at", ("ghost", "l0")), "unknown object"),
        (GroundAtom("at", ("l0", "l1")), "wrong type"),
    ]
    for atom, message in malformed_states:
        table = {
            action_applicability_state_key(0, (atom,)): {
                trajectory.actions[0]
            }
        }
        with pytest.raises(ValueError, match=message):
            PDDLTrajectoryDataset(
                [problem], [trajectory], action_supervision=config_for(table)
            )

    malformed_actions = [
        (GroundAction("unknown", ()), "unknown action schema"),
        (GroundAction("move", ("item0", "l0")), "wrong arity"),
        (GroundAction("move", ("item0", "ghost", "l1")), "unknown object"),
        (GroundAction("move", ("l0", "item0", "l1")), "wrong type"),
    ]
    for action, message in malformed_actions:
        table = {
            action_applicability_state_key(0, trajectory.states[0]): {action}
        }
        with pytest.raises(ValueError, match=message):
            PDDLTrajectoryDataset(
                [problem], [trajectory], action_supervision=config_for(table)
            )

    for incompatible_actions in (set(), {GroundAction("wait", ())}):
        contradictory = {
            action_applicability_state_key(0, trajectory.states[0]): (
                incompatible_actions
            )
        }
        dataset = PDDLTrajectoryDataset(
            [problem], [trajectory], action_supervision=config_for(contradictory)
        )
        with pytest.raises(ValueError, match="contradicts trace action"):
            dataset[0]


def test_enabled_dataset_validates_fixed_k_and_action_compatibility() -> None:
    from acs_jepa.graph import (
        ActionSupervisionConfig,
        PDDLTrajectoryDataset,
        TrajectorySample,
    )

    problem = _problem()
    trajectory = _trajectory()
    short = TrajectorySample(
        problem_index=0,
        states=trajectory.states[:2],
        actions=trajectory.actions[:1],
        terminal_atoms=trajectory.terminal_atoms,
    )
    empty = TrajectorySample(
        problem_index=0,
        states=(trajectory.states[0],),
        actions=(),
        terminal_atoms=trajectory.terminal_atoms,
    )
    config = ActionSupervisionConfig(num_negatives=0)
    with pytest.raises(ValueError, match="fixed-length"):
        PDDLTrajectoryDataset(
            [problem], [trajectory, short], action_supervision=config
        )
    with pytest.raises(ValueError, match="contain an action"):
        PDDLTrajectoryDataset([problem], [empty], action_supervision=config)
    assert len(PDDLTrajectoryDataset([problem], [], action_supervision=config)) == 0

    incompatible = replace(
        _problem(name="different"),
        actions={"wait": ActionSchema("wait", (), ())},
    )
    with pytest.raises(ValueError, match="compatible action schemas"):
        PDDLTrajectoryDataset(
            [problem, incompatible], [], action_supervision=config
        )


def test_problem_index_disambiguates_identical_atom_states() -> None:
    from acs_jepa.graph import (
        ActionApplicabilityTable,
        action_applicability_state_key,
    )

    state = _trajectory().states[0]
    key0 = action_applicability_state_key(0, state)
    key1 = action_applicability_state_key(1, state)
    assert key0 != key1
    table = ActionApplicabilityTable.from_mapping(
        {key0: {GroundAction("wait", ())}, key1: set()}
    )
    assert table[key0] == frozenset({GroundAction("wait", ())})
    assert table[key1] == frozenset()


def test_spawn_worker_collation_matches_single_process() -> None:
    from acs_jepa.graph import ActionSupervisionConfig, PDDLTrajectoryDataset

    dataset = PDDLTrajectoryDataset(
        [_problem()],
        [_trajectory()],
        action_supervision=ActionSupervisionConfig(num_negatives=2, seed=23),
    )
    direct = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))[
        "action_supervision"
    ]
    spawned = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                num_workers=1,
                multiprocessing_context="spawn",
            )
        )
    )["action_supervision"]
    _assert_tensor_dict_equal(direct, spawned)


def test_oracle_validation_errors_include_full_key_value_and_trajectory_context() -> None:
    from acs_jepa.graph import (
        ATOM_STATE_APPLICABILITY_SEMANTICS,
        ActionSupervisionConfig,
        PDDLTrajectoryDataset,
        action_applicability_state_key,
    )

    problem = _problem()
    trajectory = _trajectory()

    bad_atom = GroundAtom("unknown", ("item0",))
    bad_state_key = action_applicability_state_key(0, (bad_atom,))
    bad_state_config = ActionSupervisionConfig(
        num_negatives=1,
        applicable_actions_by_state={
            bad_state_key: {trajectory.actions[0]},
        },
        applicability_state_semantics=ATOM_STATE_APPLICABILITY_SEMANTICS,
    )
    with pytest.raises(ValueError) as state_error:
        PDDLTrajectoryDataset(
            [problem], [trajectory], action_supervision=bad_state_config
        )
    assert repr(bad_state_key) in str(state_error.value)
    assert repr(bad_atom) in str(state_error.value)

    good_state_key = action_applicability_state_key(0, trajectory.states[0])
    bad_action = GroundAction("unknown", ())
    bad_action_config = ActionSupervisionConfig(
        num_negatives=1,
        applicable_actions_by_state={good_state_key: {bad_action}},
        applicability_state_semantics=ATOM_STATE_APPLICABILITY_SEMANTICS,
    )
    with pytest.raises(ValueError) as action_error:
        PDDLTrajectoryDataset(
            [problem], [trajectory], action_supervision=bad_action_config
        )
    assert repr(good_state_key) in str(action_error.value)
    assert repr(bad_action) in str(action_error.value)

    contradiction_config = ActionSupervisionConfig(
        num_negatives=1,
        applicable_actions_by_state={good_state_key: {GroundAction("wait", ())}},
        applicability_state_semantics=ATOM_STATE_APPLICABILITY_SEMANTICS,
    )
    contradiction_dataset = PDDLTrajectoryDataset(
        [problem, problem],
        [replace(trajectory, problem_index=1), trajectory],
        action_supervision=contradiction_config,
    )
    with pytest.raises(ValueError) as contradiction_error:
        contradiction_dataset[1]
    message = str(contradiction_error.value)
    assert "problem 0" in message
    assert "trajectory 1" in message
    assert "step 0" in message
