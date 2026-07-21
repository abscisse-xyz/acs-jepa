from __future__ import annotations

import random

import pytest
import torch
from acs_jepa.graph.schemas import (
    ActionSchema,
    GroundAction,
    ObjectInfo,
    ParsedProblem,
)


def _problem() -> ParsedProblem:
    return ParsedProblem(
        name="typed-city",
        types=("car", "junction", "garage"),
        predicates={},
        objects={
            "car0": ObjectInfo("car0", "car"),
            "car1": ObjectInfo("car1", "car"),
            "j0": ObjectInfo("j0", "junction"),
            "j1": ObjectInfo("j1", "junction"),
            "j2": ObjectInfo("j2", "junction"),
        },
        actions={
            "drive": ActionSchema(
                "drive", ("car", "junction", "junction"), ()
            ),
            "park": ActionSchema("park", ("car", "garage"), ()),
            "wait": ActionSchema("wait", (), ()),
        },
        initial_atoms=(),
        goal_atoms=(),
        static_predicates=frozenset(),
    )


def test_action_supervision_builds_argument_tensors_without_negatives() -> None:
    from acs_jepa.graph.action_supervision import build_action_supervision_tensors

    tensors = build_action_supervision_tensors(
        _problem(),
        GroundAction("drive", ("car0", "j0", "j1")),
        max_action_arity=3,
        max_objects=7,
        num_negatives=0,
        seed=11,
    )

    assert torch.equal(tensors["argument_target_indices"], torch.tensor([0, 2, 3]))
    assert torch.equal(tensors["argument_mask"], torch.tensor([True, True, True]))
    assert torch.equal(
        tensors["argument_candidate_mask"],
        torch.tensor(
            [
                [True, True, False, False, False, False, False],
                [False, False, True, True, True, False, False],
                [False, False, True, True, True, False, False],
            ]
        ),
    )
    assert torch.equal(
        tensors["object_mask"],
        torch.tensor([True, True, True, True, True, False, False]),
    )
    assert tensors["negative_action_id"].shape == (0,)
    assert tensors["negative_action_object_indices"].shape == (0, 3)
    assert tensors["negative_action_role_ids"].shape == (0, 3)
    assert tensors["negative_action_arg_mask"].shape == (0, 3)
    assert tensors["negative_mask"].shape == (0,)
    assert tensors["negative_category_id"].shape == (0,)
    assert tensors["negative_changed_role_mask"].shape == (0, 3)
    assert tensors["negative_applicability_label"].shape == (0,)
    assert tensors["negative_applicability_label_mask"].shape == (0,)
    assert tensors["negative_action_id"].dtype == torch.long
    assert tensors["negative_action_object_indices"].dtype == torch.long
    assert tensors["negative_action_role_ids"].dtype == torch.long
    assert tensors["negative_category_id"].dtype == torch.long
    assert tensors["argument_target_indices"].dtype == torch.long
    assert tensors["negative_action_arg_mask"].dtype == torch.bool
    assert tensors["negative_mask"].dtype == torch.bool
    assert tensors["negative_changed_role_mask"].dtype == torch.bool
    assert tensors["negative_applicability_label_mask"].dtype == torch.bool
    assert tensors["argument_mask"].dtype == torch.bool
    assert tensors["argument_candidate_mask"].dtype == torch.bool
    assert tensors["object_mask"].dtype == torch.bool
    assert tensors["negative_applicability_label"].dtype == torch.float32


def test_action_supervision_samples_one_argument_substitution() -> None:
    from acs_jepa.graph.action_supervision import (
        ONE_ARG_SUBSTITUTION,
        build_action_supervision_tensors,
        sample_type_valid_action_negatives,
    )

    problem = _problem()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    negatives = sample_type_valid_action_negatives(
        problem,
        true_action,
        num_negatives=1,
        seed=7,
    )

    assert len(negatives) == 1
    negative = negatives[0]
    assert negative.category_id == ONE_ARG_SUBSTITUTION
    assert negative.action.name == true_action.name
    assert negative.action != true_action
    assert len(negative.changed_roles) == 1
    changed_role = negative.changed_roles[0]
    assert negative.action.arguments[changed_role] != true_action.arguments[changed_role]
    schema = problem.actions[negative.action.name]
    for object_name, expected_type in zip(
        negative.action.arguments, schema.parameter_types, strict=True
    ):
        assert problem.objects[object_name].type == expected_type

    tensors = build_action_supervision_tensors(
        problem,
        true_action,
        max_action_arity=3,
        max_objects=5,
        num_negatives=1,
        seed=7,
    )
    assert tensors["negative_mask"].tolist() == [True]
    assert tensors["negative_category_id"].tolist() == [ONE_ARG_SUBSTITUTION]
    assert tensors["negative_action_id"].tolist() == [problem.action_to_id["drive"]]
    assert tensors["negative_action_arg_mask"].tolist() == [[True, True, True]]
    assert tensors["negative_changed_role_mask"].sum().item() == 1


def test_action_supervision_round_robin_includes_swap_and_other_schema() -> None:
    from acs_jepa.graph.action_supervision import (
        ONE_ARG_SUBSTITUTION,
        RANDOM_OTHER_SCHEMA,
        RANDOM_SAME_SCHEMA,
        ROLE_SWAP,
        sample_type_valid_action_negatives,
    )

    problem = _problem()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    negatives = sample_type_valid_action_negatives(
        problem,
        true_action,
        num_negatives=4,
        seed=0,
        max_random_attempts_per_category=4,
    )

    assert [negative.category_id for negative in negatives] == [
        ONE_ARG_SUBSTITUTION,
        ROLE_SWAP,
        RANDOM_SAME_SCHEMA,
        RANDOM_OTHER_SCHEMA,
    ]
    swap = negatives[1]
    assert swap.action.arguments == ("car0", "j1", "j0")
    assert swap.changed_roles == (1, 2)
    other = negatives[3]
    assert other.action == GroundAction("wait", ())
    assert other.changed_roles == (0, 1, 2)
    assert all(negative.action.name != "park" for negative in negatives)
    assert len({_action_key(negative.action) for negative in negatives}) == 4


def test_action_supervision_is_deterministic_and_exact_type_valid() -> None:
    from acs_jepa.graph.action_supervision import sample_type_valid_action_negatives

    problem = _problem()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    first = sample_type_valid_action_negatives(
        problem, true_action, num_negatives=8, seed=19
    )
    second = sample_type_valid_action_negatives(
        problem, true_action, num_negatives=8, seed=19
    )

    assert first == second
    assert true_action not in {negative.action for negative in first}
    assert len({_action_key(negative.action) for negative in first}) == len(first)
    for negative in first:
        schema = problem.actions[negative.action.name]
        assert len(negative.action.arguments) == schema.arity
        for object_name, expected_type in zip(
            negative.action.arguments, schema.parameter_types, strict=True
        ):
            assert problem.objects[object_name].type == expected_type

    previous_global_state = random.getstate()
    random.seed(90210)
    global_state = random.getstate()
    sample_type_valid_action_negatives(
        problem, true_action, num_negatives=8, seed=31
    )
    assert random.getstate() == global_state
    random.setstate(previous_global_state)


def test_action_supervision_bounds_random_attempts_without_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import acs_jepa.graph.action_supervision as module
    from acs_jepa.architectures import ActionDecodingSpace
    from acs_jepa.graph.action_supervision import sample_type_valid_action_negatives

    original_random = random.Random
    instances: list[TrackingRandom] = []

    class TrackingRandom(original_random):
        def __init__(self, seed: int) -> None:
            super().__init__(seed)
            self.choice_calls = 0
            instances.append(self)

        def choice(self, seq):  # noqa: ANN001, ANN201
            self.choice_calls += 1
            return super().choice(seq)

    def fail_enumeration(self):  # noqa: ANN001, ANN201
        raise AssertionError("Cartesian action enumeration is forbidden")

    monkeypatch.setattr(module.random, "Random", TrackingRandom)
    monkeypatch.setattr(
        ActionDecodingSpace, "enumerate_ground_actions", fail_enumeration
    )
    problem = ParsedProblem(
        name="singleton",
        types=("item",),
        predicates={},
        objects={"only": ObjectInfo("only", "item")},
        actions={"use": ActionSchema("use", ("item",), ())},
        initial_atoms=(),
        goal_atoms=(),
        static_predicates=frozenset(),
    )

    negatives = sample_type_valid_action_negatives(
        problem,
        GroundAction("use", ("only",)),
        num_negatives=2,
        seed=3,
        max_random_attempts_per_category=3,
    )

    assert negatives == ()
    assert instances[0].choice_calls == 3

    large_objects = {
        f"item{index:03d}": ObjectInfo(f"item{index:03d}", "item")
        for index in range(100)
    }
    large_problem = ParsedProblem(
        name="large-cartesian",
        types=("item",),
        predicates={},
        objects=large_objects,
        actions={"combine": ActionSchema("combine", ("item",) * 6, ())},
        initial_atoms=(),
        goal_atoms=(),
        static_predicates=frozenset(),
    )
    large_negatives = sample_type_valid_action_negatives(
        large_problem,
        GroundAction("combine", tuple(f"item{index:03d}" for index in range(6))),
        num_negatives=4,
        seed=4,
        max_random_attempts_per_category=3,
    )
    assert len(large_negatives) == 4
    assert instances[1].choice_calls == 6


def test_action_supervision_pads_exhausted_space_and_handles_zero_arity() -> None:
    from acs_jepa.graph.action_supervision import build_action_supervision_tensors

    problem = ParsedProblem(
        name="zero-arity",
        types=(),
        predicates={},
        objects={},
        actions={"wait": ActionSchema("wait", (), ())},
        initial_atoms=(),
        goal_atoms=(),
        static_predicates=frozenset(),
    )
    tensors = build_action_supervision_tensors(
        problem,
        GroundAction("wait", ()),
        max_action_arity=0,
        max_objects=0,
        num_negatives=3,
        seed=5,
        applicability_labeler=lambda _: pytest.fail(
            "labeler must not run for padded rows"
        ),
        max_random_attempts_per_category=2,
    )

    assert tensors["negative_mask"].tolist() == [False, False, False]
    assert tensors["negative_action_id"].tolist() == [0, 0, 0]
    assert tensors["negative_category_id"].tolist() == [-1, -1, -1]
    assert tensors["negative_action_object_indices"].shape == (3, 0)
    assert tensors["negative_action_role_ids"].shape == (3, 0)
    assert tensors["argument_target_indices"].shape == (0,)
    assert tensors["argument_mask"].shape == (0,)
    assert tensors["argument_candidate_mask"].shape == (0, 0)
    assert tensors["object_mask"].shape == (0,)


def test_action_supervision_zero_arity_cross_schema_has_no_changed_roles() -> None:
    from acs_jepa.graph.action_supervision import (
        RANDOM_OTHER_SCHEMA,
        sample_type_valid_action_negatives,
    )

    problem = ParsedProblem(
        name="zero-arity-pair",
        types=(),
        predicates={},
        objects={},
        actions={
            "idle": ActionSchema("idle", (), ()),
            "wait": ActionSchema("wait", (), ()),
        },
        initial_atoms=(),
        goal_atoms=(),
        static_predicates=frozenset(),
    )
    negatives = sample_type_valid_action_negatives(
        problem,
        GroundAction("wait", ()),
        num_negatives=1,
        seed=0,
        max_random_attempts_per_category=1,
    )

    assert len(negatives) == 1
    assert negatives[0].action == GroundAction("idle", ())
    assert negatives[0].category_id == RANDOM_OTHER_SCHEMA
    assert negatives[0].changed_roles == ()


def test_action_supervision_projects_optional_labels_without_leakage() -> None:
    from acs_jepa.graph.action_supervision import build_action_supervision_tensors

    problem = _problem()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    calls: list[GroundAction] = []

    def labeler(action: GroundAction) -> bool | None:
        calls.append(action)
        if action.name == "wait":
            return None
        return action.arguments[0] == "car1"

    first = build_action_supervision_tensors(
        problem,
        true_action,
        max_action_arity=3,
        max_objects=5,
        num_negatives=6,
        seed=0,
        applicability_labeler=labeler,
        max_random_attempts_per_category=4,
    )
    second = build_action_supervision_tensors(
        problem,
        true_action,
        max_action_arity=3,
        max_objects=5,
        num_negatives=6,
        seed=0,
        applicability_labeler=lambda _: False,
        max_random_attempts_per_category=4,
    )

    assert len(calls) == int(first["negative_mask"].sum().item())
    assert first["negative_applicability_label_mask"].sum().item() == len(calls) - 1
    for name in first:
        if name not in {
            "negative_applicability_label",
            "negative_applicability_label_mask",
        }:
            assert torch.equal(first[name], second[name])

    with pytest.raises(ValueError, match="bool or None"):
        build_action_supervision_tensors(
            problem,
            true_action,
            max_action_arity=3,
            max_objects=5,
            num_negatives=1,
            seed=0,
            applicability_labeler=lambda _: 1,  # type: ignore[return-value]
        )


@pytest.mark.parametrize(
    ("true_action", "error"),
    [
        (GroundAction("missing", ()), "unknown true action schema"),
        (GroundAction("drive", ("car0", "j0")), "expects 3 arguments"),
        (GroundAction("drive", ("car0", "j0", "missing")), "unknown true action object"),
        (GroundAction("drive", ("j0", "j1", "j2")), "expects type car"),
    ],
)
def test_action_supervision_rejects_malformed_true_action(
    true_action: GroundAction, error: str
) -> None:
    from acs_jepa.graph.action_supervision import sample_type_valid_action_negatives

    with pytest.raises(ValueError, match=error):
        sample_type_valid_action_negatives(
            _problem(), true_action, num_negatives=1, seed=0
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"num_negatives": -1}, "num_negatives must be non-negative"),
        ({"max_random_attempts_per_category": 0}, "must be positive"),
        ({"max_action_arity": 2}, "max_action_arity"),
        ({"max_objects": 4}, "max_objects"),
    ],
)
def test_action_supervision_rejects_invalid_capacities(
    overrides: dict[str, int], error: str
) -> None:
    from acs_jepa.graph.action_supervision import build_action_supervision_tensors

    kwargs = {
        "max_action_arity": 3,
        "max_objects": 5,
        "num_negatives": 1,
        "seed": 0,
        "max_random_attempts_per_category": 2,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=error):
        build_action_supervision_tensors(
            _problem(), GroundAction("drive", ("car0", "j0", "j1")), **kwargs
        )


def test_action_supervision_public_graph_exports_are_stable() -> None:
    from acs_jepa.graph import (
        NEGATIVE_CATEGORY_NAMES,
        NUM_NEGATIVE_CATEGORIES,
        ONE_ARG_SUBSTITUTION,
        RANDOM_OTHER_SCHEMA,
        RANDOM_SAME_SCHEMA,
        ROLE_SWAP,
        ApplicabilityLabeler,
        SampledActionNegative,
        build_action_supervision_tensors,
        sample_type_valid_action_negatives,
    )

    assert (
        ONE_ARG_SUBSTITUTION,
        ROLE_SWAP,
        RANDOM_SAME_SCHEMA,
        RANDOM_OTHER_SCHEMA,
    ) == (0, 1, 2, 3)
    assert NUM_NEGATIVE_CATEGORIES == 4
    assert NEGATIVE_CATEGORY_NAMES == (
        "one_arg_substitution",
        "role_swap",
        "random_same_schema",
        "random_other_schema",
    )
    assert SampledActionNegative is not None
    assert ApplicabilityLabeler is not None
    assert callable(build_action_supervision_tensors)
    assert callable(sample_type_valid_action_negatives)


def _action_key(action: GroundAction) -> tuple[str, tuple[str, ...]]:
    return action.name, action.arguments
