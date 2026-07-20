from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from acs_jepa.graph.schemas import GroundAction

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
LABELS_PATH = SCRIPT_DIR / "action_applicability_labels.py"
DIAGNOSTIC_PATH = SCRIPT_DIR / "diagnose_action_applicability_labels.py"


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
    return _load_module(SCRIPT_DIR / "action_negative_sampling.py", "action_negative_sampling")


def _labels_module():
    return _load_module(LABELS_PATH, "action_applicability_labels")


def _negatives():
    negative_sampling = _negative_module()
    return (
        negative_sampling.NegativeActionExample(
            GroundAction("drive", ("car0", "j0", "j2")),
            "one_arg_substitution",
            (2,),
        ),
        negative_sampling.NegativeActionExample(
            GroundAction("load", ("car0", "g0")),
            "random_other_schema",
            (1, 2),
        ),
    )


def test_build_applicability_examples_emits_trace_and_labeled_negatives() -> None:
    labels = _labels_module()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    applicable = {
        ("drive", ("car0", "j0", "j1")),
        ("load", ("car0", "g0")),
    }

    batch = labels.build_applicability_examples(true_action, _negatives(), applicable_action_keys=applicable)

    assert [example["kind"] for example in batch.examples] == ["positive_trace", "negative", "negative"]
    assert batch.examples[0] == {
        "kind": "positive_trace",
        "category": "trace",
        "action": {"name": "drive", "arguments": ["car0", "j0", "j1"]},
        "changed_roles": [],
        "applicable": True,
    }
    assert batch.examples[1]["category"] == "one_arg_substitution"
    assert batch.examples[1]["changed_roles"] == [2]
    assert batch.examples[1]["applicable"] is False
    assert batch.examples[2]["category"] == "random_other_schema"
    assert batch.examples[2]["applicable"] is True
    assert batch.summary["true_action_applicable"] is True
    assert batch.summary["kind_counts"] == {"negative": 2, "positive_trace": 1}
    assert batch.summary["applicability_counts"] == {"applicable": 2, "inapplicable": 1, "unknown": 0}


def test_build_applicability_examples_reports_unknown_without_oracle_labels() -> None:
    labels = _labels_module()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))

    batch = labels.build_applicability_examples(true_action, _negatives(), applicable_action_keys=None)

    assert batch.summary["true_action_applicable"] is None
    assert batch.summary["applicability_counts"] == {"applicable": 0, "inapplicable": 0, "unknown": 3}
    assert all(example["applicable"] is None for example in batch.examples)


def test_build_applicability_examples_deduplicates_by_action_key_with_trace_winning() -> None:
    labels = _labels_module()
    negative_sampling = _negative_module()
    true_action = GroundAction("drive", ("car0", "j0", "j1"))
    negatives = (
        negative_sampling.NegativeActionExample(true_action, "random_same_schema", (2,)),
        negative_sampling.NegativeActionExample(
            GroundAction("drive", ("car0", "j0", "j2")),
            "one_arg_substitution",
            (2,),
        ),
        negative_sampling.NegativeActionExample(
            GroundAction("drive", ("car0", "j0", "j2")),
            "random_same_schema",
            (2,),
        ),
    )

    batch = labels.build_applicability_examples(true_action, negatives, applicable_action_keys=set())

    assert len(batch.examples) == 2
    assert batch.examples[0]["kind"] == "positive_trace"
    assert batch.examples[1]["category"] == "one_arg_substitution"


def test_diagnostic_script_imports_and_validates_arguments() -> None:
    module = _load_module(DIAGNOSTIC_PATH, "diagnose_action_applicability_labels")
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
            "--no-oracle-labels",
            "--seed",
            "9",
        ]
    )

    module.validate_args(args)
    assert args.max_transitions == 3
    assert args.per_category == 4
    assert args.no_oracle_labels is True

    bad_per_category = module.build_parser().parse_args(["data", "--output", "out", "--per-category", "0"])
    with pytest.raises(ValueError, match="--per-category must be positive"):
        module.validate_args(bad_per_category)

    bad_max_transitions = module.build_parser().parse_args(["data", "--output", "out", "--max-transitions", "-1"])
    with pytest.raises(ValueError, match="--max-transitions must be non-negative"):
        module.validate_args(bad_max_transitions)
