from __future__ import annotations

import copy
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa.graph.schemas import ActionSchema, GroundAction, ObjectInfo, ParsedProblem

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
NEGATIVE_SAMPLING_PATH = SCRIPT_DIR / "action_negative_sampling.py"
DIAGNOSTIC_PATH = SCRIPT_DIR / "diagnose_action_negatives.py"


def _load_module(path: Path, name: str):
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _negative_module():
    return _load_module(NEGATIVE_SAMPLING_PATH, "action_negative_sampling")


def _parsed_problem() -> ParsedProblem:
    return ParsedProblem(
        name="tiny-citycar",
        types=("car", "junction", "garage"),
        predicates={},
        objects={
            "car0": ObjectInfo("car0", "car"),
            "car1": ObjectInfo("car1", "car"),
            "j0": ObjectInfo("j0", "junction"),
            "j1": ObjectInfo("j1", "junction"),
            "j2": ObjectInfo("j2", "junction"),
            "g0": ObjectInfo("g0", "garage"),
        },
        actions={
            "drive": ActionSchema("drive", ("car", "junction", "junction"), ()),
            "load": ActionSchema("load", ("car", "garage"), ()),
            "wait": ActionSchema("wait", (), ()),
        },
        initial_atoms=(),
        goal_atoms=(),
        static_predicates=frozenset(),
    )


def _space() -> ActionDecodingSpace:
    return ActionDecodingSpace.from_parsed_problem(_parsed_problem())


def _by_category(examples):
    grouped: dict[str, list] = {}
    for example in examples:
        grouped.setdefault(example.category, []).append(example)
    return grouped


def test_negative_sampler_returns_type_valid_non_true_actions() -> None:
    negative_sampling = _negative_module()
    space = _space()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))

    examples = negative_sampling.sample_action_negatives(space, true_action, per_category=3, seed=7)

    assert examples
    assert all(example.action != true_action for example in examples)
    assert all(example.applicable is None for example in examples)
    space.action_tensors_for_ground_actions([example.action for example in examples])


def test_one_arg_substitution_preserves_schema_and_changes_exactly_one_argument() -> None:
    negative_sampling = _negative_module()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))

    examples = _by_category(negative_sampling.sample_action_negatives(_space(), true_action, per_category=10, seed=0))[
        "one_arg_substitution"
    ]

    assert examples
    for example in examples:
        assert example.action.name == true_action.name
        changed = tuple(
            idx
            for idx, (before, after) in enumerate(zip(true_action.arguments, example.action.arguments))
            if before != after
        )
        assert changed == example.changed_roles
        assert len(changed) == 1


def test_role_swap_uses_type_compatible_swaps_and_skips_original_tuple() -> None:
    negative_sampling = _negative_module()
    true_action = GroundAction("drive", ("car0", "j0", "j0"))

    examples = _by_category(negative_sampling.sample_action_negatives(_space(), true_action, per_category=10, seed=0))

    assert "role_swap" not in examples

    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    examples = _by_category(negative_sampling.sample_action_negatives(_space(), true_action, per_category=10, seed=0))[
        "role_swap"
    ]
    assert examples
    for example in examples:
        assert example.action.name == true_action.name
        assert example.action.arguments != true_action.arguments
        assert sorted(example.action.arguments) == sorted(true_action.arguments)
        assert example.changed_roles == (1, 2)


def test_random_same_schema_and_other_schema_categories_are_present_and_capped() -> None:
    negative_sampling = _negative_module()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))

    examples = negative_sampling.sample_action_negatives(_space(), true_action, per_category=2, seed=5)
    counts = Counter(example.category for example in examples)

    assert counts["random_same_schema"] == 2
    assert counts["random_other_schema"] == 2
    assert all(count <= 2 for count in counts.values())
    for example in examples:
        if example.category == "random_same_schema":
            assert example.action.name == true_action.name
            assert example.action != true_action
        if example.category == "random_other_schema":
            assert example.action.name != true_action.name
            assert example.changed_roles


def test_negative_sampling_is_deterministic_and_does_not_mutate_inputs() -> None:
    negative_sampling = _negative_module()
    parsed = _parsed_problem()
    parsed_before = copy.deepcopy(parsed)
    space = ActionDecodingSpace.from_parsed_problem(parsed)
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    true_before = copy.deepcopy(true_action)

    first = negative_sampling.sample_action_negatives(space, true_action, per_category=3, seed=13)
    second = negative_sampling.sample_action_negatives(space, true_action, per_category=3, seed=13)

    assert first == second
    assert true_action == true_before
    assert parsed == parsed_before


def test_applicability_function_labels_returned_negatives_only() -> None:
    negative_sampling = _negative_module()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    calls: list[GroundAction] = []

    def applicability_fn(action: GroundAction) -> bool:
        calls.append(action)
        return action.name == "load"

    examples = negative_sampling.sample_action_negatives(
        _space(),
        true_action,
        per_category=2,
        seed=2,
        applicability_fn=applicability_fn,
    )

    assert calls == [example.action for example in examples]
    assert {example.applicable for example in examples} == {False, True}


def test_diagnostic_script_imports_and_validates_arguments() -> None:
    module = _load_module(DIAGNOSTIC_PATH, "diagnose_action_negatives")
    args = module.build_parser().parse_args(
        [
            "data",
            "--output",
            "out",
            "--split",
            "val",
            "--max-transitions",
            "3",
            "--per-category",
            "4",
            "--label-applicability",
            "--seed",
            "9",
        ]
    )

    module.validate_args(args)
    assert args.max_transitions == 3
    assert args.per_category == 4
    assert args.label_applicability is True

    bad_per_category = module.build_parser().parse_args(["data", "--output", "out", "--per-category", "0"])
    with pytest.raises(ValueError, match="--per-category must be positive"):
        module.validate_args(bad_per_category)

    bad_max_transitions = module.build_parser().parse_args(["data", "--output", "out", "--max-transitions", "-1"])
    with pytest.raises(ValueError, match="--max-transitions must be non-negative"):
        module.validate_args(bad_max_transitions)
