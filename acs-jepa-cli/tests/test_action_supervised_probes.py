from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
DIAGNOSTIC_PATH = SCRIPT_DIR / "diagnose_action_supervised_probes.py"


def _load_module():
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("diagnose_action_supervised_probes", DIAGNOSTIC_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_group_split_is_deterministic_disjoint_and_recorded() -> None:
    probes = _load_module()

    first = probes.deterministic_group_split(["t0", "t0", "t1", "t2", "t3"], eval_fraction=0.25, seed=7)
    second = probes.deterministic_group_split(["t0", "t0", "t1", "t2", "t3"], eval_fraction=0.25, seed=7)

    assert first == second
    assert set(first.train_groups).isdisjoint(first.eval_groups)
    assert set(first.train_groups) | set(first.eval_groups) == {"t0", "t1", "t2", "t3"}
    assert first.train_examples + first.eval_examples == 5
    assert first.eval_groups


def test_applicability_metrics_report_auroc_and_category_margins() -> None:
    import torch

    probes = _load_module()
    logits = torch.tensor([3.0, 2.0, -1.0, -2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    group_ids = ["t0", "t1", "t0", "t1"]
    categories = ["trace", "trace", "one_arg_substitution", "random_same_schema"]

    metrics = probes.applicability_metrics(logits, labels, group_ids=group_ids, categories=categories)

    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["margin"]["count"] == 2
    assert metrics["margin"]["median"] == 4.0
    assert metrics["margin_by_category"]["one_arg_substitution"]["median"] == 4.0
    assert metrics["margin_by_category"]["random_same_schema"]["median"] == 4.0


def test_applicability_metrics_return_none_auroc_for_one_class() -> None:
    import torch

    probes = _load_module()
    metrics = probes.applicability_metrics(
        torch.tensor([2.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        group_ids=["t0", "t1"],
        categories=["trace", "trace"],
    )

    assert metrics["auroc"] is None
    assert metrics["margin"]["count"] == 0


def test_binary_metrics_cover_threshold_ranking_and_tied_average_precision() -> None:
    import pytest
    import torch

    probes = _load_module()

    perfect = probes.binary_metrics(torch.tensor([2.0, 1.0, -1.0]), torch.tensor([1.0, 1.0, 0.0]))
    assert perfect == {
        "count": 3,
        "positive_count": 2,
        "negative_count": 1,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "auroc": 1.0,
        "positive_prevalence": pytest.approx(2 / 3),
        "average_precision": 1.0,
    }
    reversed_metrics = probes.binary_metrics(
        torch.tensor([-2.0, -1.0, 1.0]), torch.tensor([1.0, 1.0, 0.0])
    )
    assert reversed_metrics["auroc"] == 0.0
    assert reversed_metrics["average_precision"] == pytest.approx(7 / 12)
    assert reversed_metrics["precision"] == reversed_metrics["recall"] == reversed_metrics["f1"] == 0.0

    tied_a = probes.binary_metrics(torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 1.0]))
    tied_b = probes.binary_metrics(torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 1.0]))
    assert tied_a["average_precision"] == tied_b["average_precision"] == pytest.approx(7 / 12)


def test_binary_metrics_define_empty_single_class_and_threshold_edges() -> None:
    import torch

    probes = _load_module()
    empty = probes.binary_metrics(torch.tensor([]), torch.tensor([]))
    assert empty["count"] == 0
    assert empty["accuracy"] is empty["precision"] is empty["recall"] is empty["f1"] is None
    assert empty["positive_prevalence"] is empty["auroc"] is empty["average_precision"] is None

    all_positive = probes.binary_metrics(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
    assert all_positive["accuracy"] == all_positive["precision"] == all_positive["recall"] == 1.0
    assert all_positive["f1"] == 1.0
    assert all_positive["auroc"] is all_positive["average_precision"] is None

    all_negative = probes.binary_metrics(torch.tensor([-1.0, -2.0]), torch.tensor([0.0, 0.0]))
    assert all_negative["accuracy"] == 1.0
    assert all_negative["precision"] == all_negative["recall"] == all_negative["f1"] == 0.0
    assert all_negative["positive_prevalence"] == 0.0
    assert all_negative["auroc"] is all_negative["average_precision"] is None


def test_category_metrics_include_trace_positives_and_retain_applicable_alternatives() -> None:
    import torch

    probes = _load_module()
    metrics = probes.applicability_metrics(
        torch.tensor([3.0, 2.0, 1.0, -1.0]),
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
        group_ids=["g0", "g1", "g0", "g1"],
        categories=["trace", "trace", "role_swap", "role_swap"],
    )

    category = metrics["per_category"]["role_swap"]
    assert category["count"] == 4
    assert category["positive_count"] == 3
    assert category["negative_count"] == 1
    assert category["auroc"] == category["average_precision"] == 1.0
    # The applicable alternative is retained in binary metrics but excluded from negative margins.
    assert metrics["margin_by_category"]["role_swap"]["count"] == 1


def test_schema_probe_overfits_separable_frozen_latents() -> None:
    import torch

    probes = _load_module()
    latents = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])

    result = probes.train_schema_probe(
        latents,
        labels,
        latents,
        labels,
        num_classes=2,
        epochs=80,
        learning_rate=0.05,
        seed=3,
        device=torch.device("cpu"),
    )

    assert result.train_metrics["accuracy"] == 1.0
    assert result.eval_metrics["accuracy"] == 1.0


def test_role_object_probe_masks_problem_local_padding() -> None:
    import torch

    probes = _load_module()
    probe = probes.RoleObjectProbe(latent_dim=2, action_dim=2, max_action_arity=3, hidden_dim=4)
    graph = torch.zeros((2, 2))
    actions = torch.zeros((2, 2))
    objects = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [99.0, 99.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        ]
    )
    object_mask = torch.tensor([[True, True, False], [True, True, True]])
    roles = torch.tensor([0, 2])

    logits = probe(graph, actions, objects, object_mask, roles)

    assert logits.shape == (2, 3)
    assert torch.isneginf(logits[0, 2])
    assert torch.isfinite(logits[1]).all()


def test_applicability_inputs_do_not_include_oracle_labels() -> None:
    import torch

    probes = _load_module()
    example = probes.ProbeExample(
        group_id="t0",
        problem="p0",
        category="trace",
        action_name="move",
        action_arguments=("car0", "j0"),
        schema_id=0,
        applicability_label=True,
        graph_latent=torch.tensor([1.0, 2.0]),
        action_latent=torch.tensor([3.0, 4.0]),
        selected_object_latents=torch.zeros((2, 2)),
        argument_mask=torch.tensor([True, True]),
        object_bank=torch.zeros((3, 2)),
        argument_targets=torch.tensor([0, 1]),
    )

    inputs, labels = probes.stack_applicability_examples([example])

    assert set(inputs) == {"graph_latents", "action_latents", "object_latents", "argument_mask"}
    assert "label" not in inputs
    assert labels.tolist() == [1.0]


def test_cli_validation_rejects_invalid_probe_training_values() -> None:
    import pytest

    probes = _load_module()
    args = probes.build_parser().parse_args(["data", "--checkpoint", "model.pt", "--output", "out", "--epochs", "0"])

    with pytest.raises(ValueError, match="--epochs must be positive"):
        probes.validate_args(args)


def test_role_probe_overfits_problem_local_targets() -> None:
    import torch

    probes = _load_module()
    graph = torch.zeros((4, 2))
    actions = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    objects = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 4)
    masks = torch.ones((4, 2), dtype=torch.bool)
    roles = torch.zeros(4, dtype=torch.long)
    targets = torch.tensor([0, 0, 1, 1])

    result = probes.train_role_probe(
        (graph, actions, objects, masks, roles, targets),
        (graph, actions, objects, masks, roles, targets),
        max_action_arity=2,
        hidden_dim=8,
        epochs=100,
        learning_rate=0.05,
        seed=5,
        device=torch.device("cpu"),
    )

    assert result.train_metrics["accuracy"] == 1.0
    assert result.eval_metrics["accuracy"] == 1.0


def test_applicability_probe_overfits_separable_examples() -> None:
    import torch

    probes = _load_module()
    graph = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    actions = graph.clone()
    objects = torch.zeros((4, 1, 2))
    masks = torch.ones((4, 1), dtype=torch.bool)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    groups = ["t0", "t1", "t2", "t3"]
    categories = ["one_arg_substitution", "random_same_schema", "trace", "trace"]
    data = (
        {
            "graph_latents": graph,
            "action_latents": actions,
            "object_latents": objects,
            "argument_mask": masks,
        },
        labels,
        groups,
        categories,
    )

    result = probes.train_applicability_probe(
        data,
        data,
        hidden_dim=8,
        epochs=100,
        learning_rate=0.05,
        seed=5,
        device=torch.device("cpu"),
    )

    assert result.train_metrics["accuracy"] == 1.0
    assert result.eval_metrics["accuracy"] == 1.0


def test_argument_features_follow_problem_local_object_ids() -> None:
    import torch
    from acs_jepa import JEPALatentState

    probes = _load_module()
    state = JEPALatentState(
        graph_latent=torch.zeros((1, 2)),
        object_latents=torch.tensor([[20.0, 20.0], [0.0, 0.0], [10.0, 10.0]]),
        object_ids=torch.tensor([2, 0, 1]),
        object_batch=torch.zeros(3, dtype=torch.long),
    )
    action_tensors = {
        "action_object_indices": torch.tensor([[1, 2, -1]]),
        "action_arg_mask": torch.tensor([[True, True, False]]),
    }

    selected, argument_mask, object_bank, targets = probes.argument_features(state, action_tensors)

    assert object_bank.tolist() == [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
    assert selected[0].tolist() == [[10.0, 10.0], [20.0, 20.0], [0.0, 0.0]]
    assert argument_mask.tolist() == [[True, True, False]]
    assert targets.tolist() == [[1, 2, -1]]


def test_canonical_manifest_bytes_are_order_invariant_and_field_sensitive() -> None:
    import hashlib

    probes = _load_module()
    records = [
        {
            "group": "p0:1",
            "problem": "p0",
            "step": 1,
            "category": "trace",
            "action": {"name": "move", "arguments": ["c0", "a"]},
            "applicability_label": True,
        },
        {
            "group": "p0:0",
            "problem": "p0",
            "step": 0,
            "category": "role_swap",
            "action": {"name": "move", "arguments": ["c0", "b"]},
            "applicability_label": False,
        },
    ]

    canonical = probes.canonical_manifest_bytes(records)
    assert canonical == probes.canonical_manifest_bytes(list(reversed(records)))
    assert canonical.endswith(b"\n")
    assert hashlib.sha256(canonical).hexdigest() == probes.manifest_identity(records)["sha256"]
    mutated = [dict(records[0]), dict(records[1])]
    mutated[0]["step"] = 2
    assert probes.canonical_manifest_bytes(mutated) != canonical


def test_decision_projection_requires_exact_schema_and_deletes_only_six_pointers() -> None:
    import copy

    import pytest

    probes = _load_module()
    summary = {
        key: {"retained": key}
        for key in probes.SUMMARY_KEYS
    }
    summary.update(
        {
            "dataset": "data-a",
            "checkpoint": "checkpoint-a",
            "device": "cpu",
            "runtime_seconds": 1.0,
            "environment": {"python": "3.x"},
            "example_manifest": {"path": "manifest-a", "sha256": "abc"},
        }
    )
    expected = probes.decision_projection(summary)

    for key, changed in (
        ("dataset", "data-b"),
        ("checkpoint", "checkpoint-b"),
        ("device", "cuda"),
        ("runtime_seconds", 99.0),
        ("environment", {"python": "other"}),
    ):
        candidate = copy.deepcopy(summary)
        candidate[key] = changed
        assert probes.decision_projection(candidate) == expected
    candidate = copy.deepcopy(summary)
    candidate["example_manifest"]["path"] = "manifest-b"
    assert probes.decision_projection(candidate) == expected

    retained = copy.deepcopy(summary)
    retained["probes"]["extra_metric"] = 2
    assert probes.decision_projection(retained) != expected
    for key in probes.SUMMARY_KEYS - {
        "dataset",
        "checkpoint",
        "device",
        "runtime_seconds",
        "environment",
        "example_manifest",
    }:
        retained = copy.deepcopy(summary)
        retained[key] = {"changed": key}
        assert probes.decision_projection(retained) != expected
    retained = copy.deepcopy(summary)
    retained["example_manifest"]["sha256"] = "def"
    assert probes.decision_projection(retained) != expected
    invalid = copy.deepcopy(summary)
    invalid["unknown"] = 1
    with pytest.raises(ValueError, match="top-level"):
        probes.decision_projection(invalid)


def test_argument_head_metrics_apply_exact_masks_targets_and_formulas() -> None:
    import pytest
    import torch

    probes = _load_module()
    logits = torch.tensor(
        [
            [[1.0, 3.0, 99.0, 99.0], [99.0, 99.0, 4.0, 99.0], [99.0] * 4, [99.0] * 4],
            [[99.0] * 4, [1.0, 3.0, 99.0, 99.0], [99.0] * 4, [99.0] * 4],
        ]
    )
    targets = torch.tensor([[1, 2, -1, -1], [-1, 0, -1, -1]])
    argument_mask = torch.tensor(
        [[True, True, False, False], [False, True, False, False]]
    )
    candidate_mask = torch.tensor(
        [
            [[True, True, False, False], [False, False, True, False], [False] * 4, [False] * 4],
            [[False] * 4, [True, True, False, False], [False] * 4, [False] * 4],
        ]
    )

    metrics = probes.argument_head_metrics(logits, targets, argument_mask, candidate_mask)
    overall = metrics["overall"]
    assert set(overall) == {
        "active_role_count",
        "competitive_role_count",
        "top1_accuracy",
        "chance_accuracy",
        "valid_candidate_count",
        "target_minus_best_wrong_margin",
    }
    assert overall["active_role_count"] == 3
    assert overall["competitive_role_count"] == 2
    assert overall["top1_accuracy"] == pytest.approx(2 / 3)
    assert overall["chance_accuracy"] == pytest.approx(2 / 3)
    assert overall["valid_candidate_count"]["count"] == 3
    assert overall["target_minus_best_wrong_margin"]["count"] == 2
    assert metrics["per_role"]["0"]["top1_accuracy"] == 1.0
    assert metrics["per_role"]["1"]["top1_accuracy"] == 0.5
    assert metrics["per_role"]["1"]["target_minus_best_wrong_margin"]["count"] == 1
    assert metrics["per_role"]["2"]["active_role_count"] == 0

    invalid = candidate_mask.clone()
    invalid[0, 0, 1] = False
    with pytest.raises(ValueError, match="active target"):
        probes.argument_head_metrics(logits, targets, argument_mask, invalid)


def test_checkpoint_argument_head_receives_dense_sorted_banks_and_exact_candidate_mask() -> None:
    import torch
    import torch.nn as nn

    probes = _load_module()

    class RecordingHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received = None

        def forward(self, action_latents, candidate_object_latents, candidate_mask):
            self.received = (action_latents.clone(), candidate_object_latents.clone(), candidate_mask.clone())
            scores = candidate_object_latents[..., 0].unsqueeze(1).expand_as(candidate_mask)
            return torch.where(candidate_mask, scores, -torch.inf)

    head = RecordingHead()
    action_latents = torch.tensor([[1.0], [2.0]])
    object_banks = [torch.tensor([[10.0], [20.0]]), torch.tensor([[30.0]])]
    targets = [torch.tensor([1, -1]), torch.tensor([0, -1])]
    argument_masks = [torch.tensor([True, False]), torch.tensor([True, False])]
    candidate_masks = [
        torch.tensor([[True, True], [False, False]]),
        torch.tensor([[True], [False]]),
    ]

    metrics = probes.evaluate_checkpoint_argument_head(
        head,
        action_latents,
        object_banks,
        targets,
        argument_masks,
        candidate_masks,
    )

    assert head.received is not None
    _, dense_banks, dense_mask = head.received
    assert dense_banks.tolist() == [[[10.0], [20.0]], [[30.0], [0.0]]]
    assert dense_mask.tolist() == [
        [[True, True], [False, False]],
        [[True, False], [False, False]],
    ]
    assert metrics["overall"]["active_role_count"] == 2
    assert metrics["overall"]["competitive_role_count"] == 1


def test_argument_candidate_masks_use_sorted_ids_exact_types_and_inactive_roles() -> None:
    from types import SimpleNamespace

    import torch
    from acs_jepa import JEPALatentState

    probes = _load_module()
    parsed = SimpleNamespace(
        object_to_id={"loc_b": 2, "car": 0, "loc_a": 1},
        objects={
            "loc_b": SimpleNamespace(type="location"),
            "car": SimpleNamespace(type="vehicle"),
            "loc_a": SimpleNamespace(type="location"),
        },
        actions={"move": SimpleNamespace(parameter_types=("vehicle", "location"))},
    )
    state = JEPALatentState(
        graph_latent=torch.zeros((1, 1)),
        object_latents=torch.zeros((3, 1)),
        object_ids=torch.tensor([2, 0, 1]),
        object_batch=torch.zeros(3, dtype=torch.long),
    )
    argument_mask = torch.tensor([[True, True, False]])

    masks = probes.argument_candidate_masks(parsed, state, ["move"], argument_mask)

    # Sorted object-ID bank is car(0), loc_a(1), loc_b(2).
    assert masks.tolist() == [[[True, False, False], [False, True, True], [False, False, False]]]


def test_fixed_smoke_manifest_matches_precommitted_stage2g_identity() -> None:
    from collections import Counter
    from pathlib import Path

    import pytest

    probes = _load_module()
    dataset = Path("/opt/data/workspace/acs-jepa-tuning-data/smoke")
    checkpoint = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt")
    if not dataset.exists() or not checkpoint.exists():
        pytest.skip("fixed Stage 2G smoke fixtures are not installed")
    args = probes.build_parser().parse_args(
        [
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--output",
            "/tmp/stage2g-fixed-manifest-test",
            "--split",
            "val",
            "--per-category",
            "4",
            "--seed",
            "20260717",
        ]
    )

    examples, metadata, device, _, _ = probes.collect_probe_examples(args)
    records = probes._manifest_records(examples)
    identity = probes.manifest_identity(records)

    assert str(device) == "cpu"
    assert metadata["transitions"] == 44
    assert identity == {
        "count": 604,
        "bytes": 117385,
        "sha256": "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9",
    }
    assert Counter(example.category for example in examples) == {
        "one_arg_substitution": 176,
        "random_other_schema": 176,
        "random_same_schema": 176,
        "role_swap": 32,
        "trace": 44,
    }
    assert Counter(example.applicability_label for example in examples) == {True: 62, False: 542}
