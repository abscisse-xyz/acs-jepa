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
