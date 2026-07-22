from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from acs_jepa import JEPALatentState
from acs_jepa.graph import GroundAtom, build_state_graph

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "script" / "diagnose_action_applicability_recoverability.py"


def _load(name: str = "diagnose_action_applicability_recoverability"):
    script_dir = str(ROOT / "script")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stage0b_module_bootstraps_with_fixed_parser_defaults() -> None:
    module = _load()
    args = module.build_parser().parse_args(
        ["data", "--checkpoint", "model.pt", "--candidate-manifest", "manifest.json", "--output", "out"]
    )
    assert (args.device, args.split, args.epochs, args.learning_rate, args.hidden_dim, args.seed) == (
        "cpu",
        "val",
        200,
        0.001,
        64,
        20260717,
    )


def _citycar_fixture():
    predicates = {
        name: SimpleNamespace(name=name, arg_types=types)
        for name, types in (
            ("arrived", ("car", "junction")),
            ("at_car_jun", ("car", "junction")),
            ("at_car_road", ("car", "road")),
            ("at_garage", ("garage", "junction")),
            ("clear", ("junction",)),
            ("diagonal", ("junction", "junction")),
            ("in_place", ("road",)),
            ("road_connect", ("road", "junction", "junction")),
            ("same_line", ("junction", "junction")),
            ("starting", ("car", "garage")),
        )
    }
    return SimpleNamespace(
        actions={
            name: SimpleNamespace(parameter_types=("car", "junction"))
            for name in (
                "build_diagonal_oneway",
                "build_straight_oneway",
                "car_arrived",
                "car_start",
                "destroy_road",
                "move_car_in_road",
                "move_car_out_road",
            )
        },
        types=("car", "garage", "junction", "road"),
        predicates=predicates,
        objects={
            "c": SimpleNamespace(type="car"),
            "g": SimpleNamespace(type="garage"),
            "j": SimpleNamespace(type="junction"),
            "r": SimpleNamespace(type="road"),
        },
    )


@pytest.fixture(scope="module")
def actual_source_context():
    module = _load("recoverability_actual_source_context")
    records, _identity = module.load_and_validate_candidate_manifest(module.FIXED_CANDIDATE_MANIFEST)
    config, corpus, _bundle, device, _restoration = module.load_checkpoint_bundle(
        module.DATASET,
        module.BASELINE_CHECKPOINT,
        device_name="cpu",
        include_restoration_metadata=True,
    )
    selected = module.select_split(corpus, config, "val", seed=20260717)
    sources = module.reconcile_manifest_source_states(records, selected)
    return module, records, sources, device


class _DeterministicJEPA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoded_graphs = []
        self.action_encoder = self._encode_actions

    def encode(self, graph):
        self.encoded_graphs.append(graph)
        object_rows = graph.x[:, 0] == 0
        object_ids = graph.x[object_rows, 4]
        graph_value = float(graph.x[:, 2].clamp_min(0).sum() + graph.edge_attr[:, 0].sum())
        return JEPALatentState(
            graph_latent=torch.full((1, 64), graph_value),
            object_latents=object_ids.to(torch.float32)[:, None].expand(-1, 64).contiguous(),
            object_ids=object_ids,
            object_batch=torch.zeros(object_ids.numel(), dtype=torch.long),
        )

    @staticmethod
    def _encode_actions(tensors, _state):
        values = tensors["action_id"].to(torch.float32) + tensors["action_object_indices"].clamp_min(0).sum(dim=1)
        return values[:, None].expand(-1, 64).contiguous()


def _deterministic_bundle():
    return SimpleNamespace(
        jepa=_DeterministicJEPA(),
        goal_head=None,
        action_contrastive_anchor=None,
        argument_reconstruction_head=None,
        applicability_head=None,
    )


def test_exact_feature_names_partitions_and_composition_a_through_e() -> None:
    module = _load("recoverability_feature_schema")
    schemas = module.feature_schemas()
    assert [(row["name"], row["dimension"]) for row in schemas] == [
        ("A_action", 64),
        ("B_graph_action", 128),
        ("C_selected_graph_action", 388),
        ("D_raw_symbolic", 217),
        ("E_hybrid", 605),
    ]
    assert schemas[0]["feature_names"] == [f"action_latent[{i}]" for i in range(64)]
    assert schemas[2]["feature_names"][128:384] == [
        f"selected_object_latent[{role}][{coordinate}]" for role in range(4) for coordinate in range(64)
    ]
    assert schemas[2]["feature_names"][384:] == [f"argument_present[{i}]" for i in range(4)]
    assert schemas[3]["feature_names"][-1] == "fact:starting(3,3)"
    assert len([name for name in schemas[3]["feature_names"] if name.startswith("fact:")]) == 184
    expected_raw = (
        [f"schema={name}" for name in module.SCHEMAS]
        + [f"role_active[{role}]" for role in range(4)]
        + [f"role_equal[{left},{right}]" for left, right in itertools.combinations(range(4), 2)]
        + [f"role_type[{role}]={type_name}" for role in range(4) for type_name in module.TYPES]
        + [
            f"fact:{predicate}({','.join(map(str, roles))})"
            for predicate, arg_types in module.PREDICATES
            for roles in itertools.product(range(4), repeat=len(arg_types))
        ]
    )
    assert schemas[3]["feature_names"] == expected_raw
    assert [(row["standardized_indices"], row["binary_indices"]) for row in schemas] == [
        (list(range(64)), []),
        (list(range(128)), []),
        (list(range(384)), list(range(384, 388))),
        ([], list(range(217))),
        (list(range(384)), list(range(384, 605))),
    ]

    action = SimpleNamespace(name="car_start", arguments=("c", "j"))
    graph = torch.arange(64, dtype=torch.float32)
    action_latent = torch.arange(64, dtype=torch.float32) + 100
    selected = torch.ones(4, 64)
    selected[2:] = 99
    mask = torch.tensor([True, True, False, False])
    features = module.compose_feature_sets(
        graph, action_latent, selected, mask, module.raw_symbolic_features(_citycar_fixture(), (), action)
    )
    assert torch.equal(features["A_action"], action_latent)
    assert torch.equal(features["B_graph_action"], torch.cat((graph, action_latent)))
    assert torch.equal(features["C_selected_graph_action"][128:384].reshape(4, 64)[2:], torch.zeros(2, 64))
    assert torch.equal(
        features["E_hybrid"], torch.cat((features["C_selected_graph_action"], features["D_raw_symbolic"]))
    )


def test_labels_categories_future_states_and_split_metadata_cannot_change_a_through_e() -> None:
    module = _load("recoverability_feature_leakage")
    parsed = _citycar_fixture()
    action = SimpleNamespace(name="car_start", arguments=("c", "j"))
    graph = torch.arange(64, dtype=torch.float32)
    action_latent = graph + 100
    selected = torch.arange(256, dtype=torch.float32).reshape(4, 64)
    mask = torch.tensor([True, True, False, False])
    atoms = {GroundAtom("clear", ("j",)), GroundAtom("starting", ("c", "g"))}

    def extract(metadata):
        # Metadata is deliberately not accepted by either feature constructor.
        assert set(metadata) == {"label", "category", "future_state", "split"}
        return module.compose_feature_sets(
            graph,
            action_latent,
            selected,
            mask,
            module.raw_symbolic_features(parsed, atoms, action),
        )

    original = extract({"label": True, "category": "trace", "future_state": {"x"}, "split": "train"})
    mutated = extract(
        {"label": False, "category": "role_swap", "future_state": {"entirely", "different"}, "split": "eval"}
    )
    assert original.keys() == mutated.keys()
    assert all(torch.equal(original[name], mutated[name]) for name in original)


def test_raw_symbolic_facts_cover_static_repeated_inactive_and_type_mismatch() -> None:
    module = _load("recoverability_symbolic")
    parsed = _citycar_fixture()
    action = SimpleNamespace(name="car_start", arguments=("c", "j"))
    atoms = {GroundAtom("clear", ("j",)), GroundAtom("diagonal", ("j", "j")), GroundAtom("starting", ("c", "g"))}
    values = module.raw_symbolic_features(parsed, atoms, action)
    names = module.feature_schemas()[3]["feature_names"]
    feature = dict(zip(names, values.tolist(), strict=True))
    assert feature["schema=car_start"] == 1
    assert feature["role_active[0]"] == feature["role_active[1]"] == 1
    assert feature["role_active[2]"] == 0
    assert feature["fact:clear(1)"] == 1
    assert feature["fact:diagonal(1,1)"] == 1
    assert feature["fact:diagonal(0,1)"] == 0
    assert feature["fact:clear(2)"] == 0
    assert feature["fact:starting(0,1)"] == 0


@pytest.mark.parametrize("mutation", ["changed_arg_types", "extra_predicate"])
def test_raw_symbolic_rejects_any_predicate_vocabulary_or_arg_type_drift(mutation: str) -> None:
    module = _load(f"recoverability_predicate_contract_{mutation}")
    parsed = copy.deepcopy(_citycar_fixture())
    if mutation == "changed_arg_types":
        parsed.predicates["starting"].arg_types = ("car", "junction")
    else:
        parsed.predicates["unexpected"] = SimpleNamespace(name="unexpected", arg_types=())

    with pytest.raises(ValueError, match="predicate vocabulary"):
        module.raw_symbolic_features(parsed, (), SimpleNamespace(name="car_start", arguments=("c", "j")))


def test_static_source_facts_must_match_graph_input_without_synthesis() -> None:
    module = _load("recoverability_static")
    parsed = SimpleNamespace(static_predicates=frozenset({"diagonal", "road_connect"}))
    source = {GroundAtom("diagonal", ("j", "j")), GroundAtom("clear", ("j",))}
    module.validate_static_fact_match(parsed, source, tuple(source))
    with pytest.raises(ValueError, match="static facts"):
        module.validate_static_fact_match(parsed, source, (GroundAtom("clear", ("j",)),))


def test_collect_features_invokes_validated_source_extraction_adapter(monkeypatch) -> None:
    module = _load("recoverability_static_collection")
    bundle = SimpleNamespace(
        jepa=None,
        goal_head=None,
        action_contrastive_anchor=None,
        argument_reconstruction_head=None,
        applicability_head=None,
    )
    source = SimpleNamespace(
        parsed=SimpleNamespace(),
        source_atoms=(GroundAtom("diagonal", ("j", "j")),),
        manifest_records=(),
    )
    monkeypatch.setattr(
        module,
        "load_checkpoint_bundle",
        lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace(), bundle, torch.device("cpu"), {}),
    )
    monkeypatch.setattr(module, "select_split", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "reconcile_manifest_source_states", lambda *args, **kwargs: [source])

    def reject(*args, **kwargs):
        raise ValueError("sentinel source extraction")

    monkeypatch.setattr(module, "_extract_source_features", reject)
    with pytest.raises(ValueError, match="sentinel source extraction"):
        module._collect_features(
            SimpleNamespace(dataset_dir=Path("data"), checkpoint=Path("model"), device="cpu", split="val", seed=0), []
        )


def _mutate_state_graph_edges(graph, mutation: str):
    mutated = graph.clone()
    edges = mutated.edge_index.t().clone()
    attributes = mutated.edge_attr.clone()
    forward_index = next(index for index, value in enumerate(attributes.tolist()) if value[1] == 0)
    atom_node, object_node = edges[forward_index].tolist()
    role = int(attributes[forward_index, 0])
    reverse_index = next(
        index
        for index, (edge, attribute) in enumerate(zip(edges.tolist(), attributes.tolist(), strict=True))
        if edge == [object_node, atom_node] and attribute == [role, 1]
    )

    if mutation == "unmatched_reverse_only":
        edges = torch.cat((edges, torch.tensor([[object_node, atom_node]])), dim=0)
        attributes = torch.cat((attributes, torch.tensor([[99, 1]])), dim=0)
    elif mutation == "unmatched_forward_only":
        keep = torch.arange(edges.size(0)) != reverse_index
        edges, attributes = edges[keep], attributes[keep]
    elif mutation == "duplicate_bidirectional_pair":
        edges = torch.cat((edges, edges[[forward_index, reverse_index]]), dim=0)
        attributes = torch.cat((attributes, attributes[[forward_index, reverse_index]]), dim=0)
    elif mutation == "invalid_direction":
        edges = torch.cat((edges, torch.tensor([[atom_node, object_node]])), dim=0)
        attributes = torch.cat((attributes, torch.tensor([[role, 2]])), dim=0)
    elif mutation == "invalid_atom_to_atom_connectivity":
        edges = torch.cat((edges, torch.tensor([[atom_node, atom_node]])), dim=0)
        attributes = torch.cat((attributes, torch.tensor([[role, 0]])), dim=0)
    elif mutation == "invalid_object_to_object_connectivity":
        edges = torch.cat((edges, torch.tensor([[object_node, object_node]])), dim=0)
        attributes = torch.cat((attributes, torch.tensor([[role, 1]])), dim=0)
    elif mutation == "arity_role_mismatch":
        attributes[forward_index, 0] = int(mutated.x[atom_node, 3])
        attributes[reverse_index, 0] = int(mutated.x[atom_node, 3])
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    mutated.edge_index = edges.t().contiguous()
    mutated.edge_attr = attributes
    return mutated


@pytest.mark.parametrize(
    "mutation",
    [
        "unmatched_reverse_only",
        "unmatched_forward_only",
        "duplicate_bidirectional_pair",
        "invalid_direction",
        "invalid_atom_to_atom_connectivity",
        "invalid_object_to_object_connectivity",
        "arity_role_mismatch",
    ],
)
def test_graph_atom_reconstruction_requires_exact_bidirectional_edge_multiset(mutation, actual_source_context) -> None:
    module, _records, sources, _device = actual_source_context
    source = next(item for item in sources if item.group == "p166:0")
    graph = build_state_graph(source.parsed, source.source_atoms, include_static=True)
    reconstructed = module.reconstruct_graph_atoms(source.parsed, graph)
    assert len(reconstructed) == len(source.source_atoms) == 128
    assert set(reconstructed) == set(source.source_atoms)

    with pytest.raises(ValueError, match="state graph"):
        module.reconstruct_graph_atoms(source.parsed, _mutate_state_graph_edges(graph, mutation))


def test_actual_graph_static_reconstruction_detects_filtering_and_encodes_same_validated_data(
    monkeypatch, actual_source_context
) -> None:
    module, _records, sources, device = actual_source_context
    source = next(item for item in sources if item.group == "p166:0")
    source = copy.copy(source)
    source.manifest_records = source.manifest_records[:1]
    bundle = _deterministic_bundle()
    built = []

    def filtered_builder(parsed, atoms, *, include_static=True):
        graph = build_state_graph(parsed, atoms, include_static=False)
        built.append(graph)
        return graph

    monkeypatch.setattr(module, "build_state_graph", filtered_builder)
    with pytest.raises(ValueError, match="static facts"):
        module._extract_source_features(bundle, source, device=device)
    assert bundle.jepa.encoded_graphs == []

    def correct_builder(parsed, atoms, *, include_static=True):
        graph = build_state_graph(parsed, atoms, include_static=include_static)
        built.append(graph)
        return graph

    monkeypatch.setattr(module, "build_state_graph", correct_builder)
    extracted = module._extract_source_features(bundle, source, device=device)
    assert len(extracted) == 1
    assert bundle.jepa.encoded_graphs == [built[-1]]


def test_actual_source_extraction_boundary_has_exact_metadata_and_prefix_causality(actual_source_context) -> None:
    module, _records, sources, device = actual_source_context
    source = copy.copy(next(item for item in sources if item.group == "p166:0"))
    selected = next(
        row
        for row in source.manifest_records
        if row["category"] == "one_arg_substitution"
        and row["action"]["arguments"] == ["junction0-0", "junction3-2", "road2"]
    )
    source.manifest_records = (selected,)

    original = module._extract_source_features(_deterministic_bundle(), source, device=device)[0]
    metadata_source = copy.copy(source)
    metadata_row = copy.deepcopy(selected)
    metadata_row["applicability_label"] = not metadata_row["applicability_label"]
    metadata_row["category"] = "random_same_schema"
    metadata_row["future_state"] = ["inert", "future", "atoms"]
    metadata_row["split"] = "inert-eval"
    metadata_source.manifest_records = (metadata_row,)
    metadata_mutated = module._extract_source_features(_deterministic_bundle(), metadata_source, device=device)[0]
    assert all(torch.equal(original[name], metadata_mutated[name]) for name in original)

    removed = GroundAtom("clear", ("junction3-2",))
    assert removed in source.source_atoms and removed.predicate not in source.parsed.static_predicates
    prefix_source = copy.copy(source)
    prefix_source.source_atoms = tuple(atom for atom in source.source_atoms if atom != removed)
    prefix_mutated = module._extract_source_features(_deterministic_bundle(), prefix_source, device=device)[0]

    assert torch.equal(original["A_action"], prefix_mutated["A_action"])
    assert not torch.equal(original["B_graph_action"][:64], prefix_mutated["B_graph_action"][:64])
    assert torch.equal(original["B_graph_action"][64:], prefix_mutated["B_graph_action"][64:])
    assert not torch.equal(original["C_selected_graph_action"][:64], prefix_mutated["C_selected_graph_action"][:64])
    assert torch.equal(original["C_selected_graph_action"][64:], prefix_mutated["C_selected_graph_action"][64:])
    raw_names = module.feature_schemas()[3]["feature_names"]
    changed_raw = torch.where(original["D_raw_symbolic"] != prefix_mutated["D_raw_symbolic"])[0].tolist()
    expected_raw = raw_names.index("fact:clear(1)")
    assert changed_raw == [expected_raw]
    changed_hybrid = torch.where(original["E_hybrid"] != prefix_mutated["E_hybrid"])[0].tolist()
    assert changed_hybrid == [*range(64), 388 + expected_raw]
    assert torch.equal(original["E_hybrid"][:388], original["C_selected_graph_action"])
    assert torch.equal(prefix_mutated["E_hybrid"][:388], prefix_mutated["C_selected_graph_action"])
    assert torch.equal(original["E_hybrid"][388:], original["D_raw_symbolic"])
    assert torch.equal(prefix_mutated["E_hybrid"][388:], prefix_mutated["D_raw_symbolic"])


def test_preprocessing_is_train_only_float64_population_std_with_exact_zero_policy() -> None:
    module = _load("recoverability_preprocessing")
    train = torch.tensor([[1.0, 5.0, 0.0], [3.0, 5.0, 1.0]], dtype=torch.float32)
    evaluation = torch.tensor([[101.0, 99.0, 1.0]])
    state = module.fit_preprocessing(train, standardized_indices=[0, 1], binary_indices=[2])
    assert state == {
        "mean": [2.0, 5.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "binary_indices": [2],
        "standardized_indices": [0, 1],
        "zero_std_indices": [1],
    }
    transformed = module.apply_preprocessing(evaluation, state)
    assert transformed.dtype == torch.float32
    assert transformed.tolist() == [[99.0, 0.0, 1.0]]
    assert module.apply_preprocessing(train, state)[:, 1].tolist() == [0.0, 0.0]


def test_pinned_split_and_single_control_permutation_are_exact_and_local_rng_only() -> None:
    module = _load("recoverability_split_control")
    split = module.split_manifest()
    assert len(split["train_groups"]) == 33 and len(split["eval_groups"]) == 11
    assert (
        module.canonical_json_bytes(split) == (json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    labels = torch.arange(453, dtype=torch.float32)
    torch.manual_seed(99)
    before = torch.random.get_rng_state().clone()
    first = module.control_permutation(labels, seed=20260717)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert torch.equal(
        first, labels[torch.randperm(453, generator=torch.Generator(device="cpu").manual_seed(20260717))]
    )


def test_threshold_binary_metrics_calibration_and_arithmetic_margin_boundaries() -> None:
    module = _load("recoverability_metrics")
    logits = torch.tensor([2.0, 2.0, 0.0, -2.0], dtype=torch.float64)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    threshold = module.select_f1_threshold(logits, labels)
    assert threshold == 0.0
    metrics = module.binary_metrics(logits, labels, threshold)
    assert set(metrics) == {
        "count",
        "positive_count",
        "negative_count",
        "prevalence",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "average_precision",
        "nll",
        "brier",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "reliability_bins",
    }
    assert metrics["auroc"] == pytest.approx(0.625)
    assert metrics["average_precision"] == pytest.approx(7 / 12)
    assert len(metrics["reliability_bins"]) == 10
    assert module.distribution([1.0, 3.0])["median"] == 2.0
    single = module.binary_metrics(torch.tensor([1.0, 2.0]), torch.ones(2), 0.0)
    assert single["auroc"] is single["average_precision"] is None
    empty = module.distribution([])
    assert empty == {"count": 0, "min": None, "median": None, "mean": None, "max": None}


def test_fit_inventory_order_steps_rng_reset_and_all_fifteen_states_reconstruct(monkeypatch) -> None:
    module = _load("recoverability_fit")
    names = [row["name"] for row in module.feature_schemas()]
    rows = (
        (0.0, 1.0, 4.0, 9.0),
        (1.0, 4.0, 9.0, 0.0),
        (4.0, 9.0, 0.0, 1.0),
        (9.0, 0.0, 1.0, 4.0),
        (0.0, 4.0, 1.0, 9.0),
    )
    features = {name: torch.tensor(rows[index]).unsqueeze(1) for index, name in enumerate(names)}
    preprocessing = {
        name: module.fit_preprocessing(value, standardized_indices=[0], binary_indices=[])
        for name, value in features.items()
    }
    expected_features = {}
    for name, value in features.items():
        values = value.to(torch.float64)
        expected_features[name] = ((values - values.mean(dim=0)) / values.std(dim=0, correction=0)).to(torch.float32)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    control = labels.flip(0)
    seeds = []
    optimizers = []
    step_counts = {}
    forward_calls = []
    target_calls = []
    hook_handles = []
    original_seed = torch.manual_seed
    original_step = torch.optim.Adam.step
    original_bce = torch.nn.functional.binary_cross_entropy_with_logits
    original_probe = module._probe
    monkeypatch.setattr(torch, "manual_seed", lambda seed: (seeds.append(seed), original_seed(seed))[1])

    def counted_step(optimizer, *args, **kwargs):
        if not any(item is optimizer for item in optimizers):
            optimizers.append(optimizer)
            step_counts[id(optimizer)] = 0
        step_counts[id(optimizer)] += 1
        return original_step(optimizer, *args, **kwargs)

    def inspected_probe(*args, **kwargs):
        model_index = len(hook_handles)
        model = original_probe(*args, **kwargs)
        hook_handles.append(
            model.register_forward_pre_hook(
                lambda _model, inputs, model_index=model_index: forward_calls.append(
                    (model_index, inputs[0].detach().clone())
                )
            )
        )
        return model

    def inspected_bce(logits, targets, *args, **kwargs):
        target_calls.append(targets.detach().clone())
        return original_bce(logits, targets, *args, **kwargs)

    def assert_exact_calls(actual_forward, actual_targets):
        expected_forward = [
            (model_index, expected_features[name])
            for model_index, name in enumerate(name for name in names for _kind in range(3))
            for _epoch in range(2)
        ]
        expected_targets = [
            target
            for _name in names
            for target in (labels, labels, control)
            for _epoch in range(2)
        ]
        assert len(actual_forward) == len(expected_forward) == 15 * 2
        assert len(actual_targets) == len(expected_targets) == 15 * 2
        for call_index, ((actual_model, actual_x), (expected_model, expected_x)) in enumerate(
            zip(actual_forward, expected_forward, strict=True)
        ):
            assert actual_model == expected_model
            assert torch.equal(actual_x, expected_x), f"forward call {call_index} changed value or canonical row order"
        for call_index, (actual_target, expected_target) in enumerate(
            zip(actual_targets, expected_targets, strict=True)
        ):
            assert torch.equal(actual_target, expected_target), (
                f"BCE call {call_index} changed target value or row order"
            )

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    monkeypatch.setattr(module, "_probe", inspected_probe)
    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", inspected_bce)
    fitted = module.fit_all_probes(
        features, preprocessing, labels, control, epochs=2, learning_rate=0.001, hidden_dim=64, seed=20260717
    )
    assert [(row.feature_set, row.model_kind) for row in fitted] == [
        (name, kind) for name in names for kind in ("linear", "mlp", "control_mlp")
    ]
    assert seeds == [20260717] * 15
    assert len(optimizers) == 15
    assert step_counts == {id(optimizer): 2 for optimizer in optimizers}
    assert sum(step_counts.values()) == 15 * 2
    assert_exact_calls(forward_calls, target_calls)
    for handle in hook_handles:
        handle.remove()
    states = module.serialize_probe_states(
        fitted, preprocessing, candidate_sha256="a" * 64, epochs=2, learning_rate=0.001, hidden_dim=64, seed=20260717
    )
    assert len(states["models"]) == 15
    with monkeypatch.context() as reconstruction:
        reconstruction.setattr(module, "_probe", original_probe)
        for fitted_probe, state in zip(fitted, states["models"], strict=True):
            rebuilt = module.reconstruct_probe(state)
            x = module.apply_preprocessing(features[fitted_probe.feature_set], state["preprocessing"])
            assert torch.allclose(rebuilt(x), fitted_probe.model(x), atol=1e-7, rtol=0)

    mutant_forward_calls = []
    mutant_target_calls = []

    def duplicate_row_zero(value):
        mutated = value.clone()
        mutated[1] = mutated[0]
        return mutated

    def mutant_probe(*args, **kwargs):
        model_index = len(mutant_forward_calls) // 2
        model = original_probe(*args, **kwargs)
        model.register_forward_pre_hook(lambda _model, inputs: (duplicate_row_zero(inputs[0]),))
        model.register_forward_pre_hook(
            lambda _model, inputs, model_index=model_index: mutant_forward_calls.append(
                (model_index, inputs[0].detach().clone())
            )
        )
        return model

    def mutant_bce(logits, targets, *args, **kwargs):
        mutated = duplicate_row_zero(targets)
        mutant_target_calls.append(mutated.detach().clone())
        return original_bce(logits, mutated, *args, **kwargs)

    with monkeypatch.context() as mutation:
        mutation.setattr(torch, "manual_seed", original_seed)
        mutation.setattr(torch.optim.Adam, "step", original_step)
        mutation.setattr(module, "_probe", mutant_probe)
        mutation.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", mutant_bce)
        module.fit_all_probes(
            features,
            preprocessing,
            labels,
            control,
            epochs=2,
            learning_rate=0.001,
            hidden_dim=64,
            seed=20260717,
        )
    with pytest.raises(AssertionError, match="forward call 0 changed value or canonical row order"):
        assert_exact_calls(mutant_forward_calls, target_calls)
    with pytest.raises(AssertionError, match="BCE call 0 changed target value or row order"):
        assert_exact_calls(forward_calls, mutant_target_calls)


def test_root_checkpoint_destination_binding_and_dependency_boundary(tmp_path: Path) -> None:
    module = _load("recoverability_binding")
    root = tmp_path / "updated_phase0"
    module.validate_run_binding(module.BASELINE_CHECKPOINT, root / "recoverability" / "baseline" / "run1")
    module.validate_run_binding(module.PHASE2_CHECKPOINT, root / "recoverability" / "phase2" / "run2")
    with pytest.raises(ValueError, match="checkpoint/output binding"):
        module.validate_run_binding(module.PHASE2_CHECKPOINT, root / "recoverability" / "baseline" / "run1")
    with pytest.raises(ValueError, match="run1 or run2"):
        module.validate_run_binding(module.BASELINE_CHECKPOINT, root / "recoverability" / "baseline" / "other")
    imports = {
        ast.unparse(node)
        for node in ast.walk(ast.parse(SCRIPT.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    forbidden = (
        "action_diag_common",
        "acs_jepa_cli.cli",
        "diagnose_action_supervised_probes",
        "simulator",
        "replay",
        "oracle",
        "applicable_actions",
    )
    assert all(not any(name in statement for name in forbidden) for statement in imports)


def test_fresh_runtime_import_graph_loads_no_simulator_replay_or_oracle_modules() -> None:
    code = f"""
import importlib.util
import json
import sys
sys.path.insert(0, {str(ROOT / "script")!r})
spec = importlib.util.spec_from_file_location('recoverability_runtime_boundary', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
forbidden = ('simulator', 'replay', 'oracle', 'applicable_actions')
project_prefixes = ('acs_jepa', 'action_', 'diagnose_action')
loaded = sorted(
    name for name in sys.modules
    if name.startswith(project_prefixes) and any(token in name.lower() for token in forbidden)
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_probe_report_reuses_global_threshold_and_control_eval_uses_original_labels() -> None:
    module = _load("recoverability_report")
    rows = [
        {"group": "g0", "category": "trace", "schema": "car_start"},
        {"group": "g0", "category": "role_swap", "schema": "car_start"},
        {"group": "g1", "category": "trace", "schema": "car_arrived"},
        {"group": "g1", "category": "one_arg_substitution", "schema": "car_arrived"},
    ]
    train_logits = torch.tensor([2.0, -1.0, 1.0, -2.0])
    original = torch.tensor([1.0, 0.0, 1.0, 0.0])
    permuted = 1 - original
    report = module.probe_report(train_logits, train_logits, original, original, rows, rows, threshold_labels=permuted)
    assert report["threshold"] == module.select_f1_threshold(train_logits, permuted)
    assert report["eval"]["auroc"] == 1.0
    assert (
        report["per_schema"]["car_start"]["accuracy"]
        == module.binary_metrics(train_logits[:2], original[:2], report["threshold"])["accuracy"]
    )
    assert report["role_swap_margin"]["median"] == 3.0
    assert report["one_arg_substitution_margin"]["median"] == 3.0


def test_real_cli_repeats_emit_exact_recursive_artifacts_and_refuse_unsafe_destinations(tmp_path: Path) -> None:
    module = _load("recoverability_real_cli_contract")
    records, candidate_identity = module.load_and_validate_candidate_manifest(module.FIXED_CANDIDATE_MANIFEST)
    phase_root = tmp_path / "updated_phase0"
    phase_root.mkdir()
    (phase_root / "root_identity.json").write_bytes(
        module.canonical_json_bytes(module._root_identity(candidate_identity))
    )

    def command(output: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            str(module.DATASET),
            "--checkpoint",
            str(module.BASELINE_CHECKPOINT),
            "--candidate-manifest",
            str(module.FIXED_CANDIDATE_MANIFEST),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--split",
            "val",
            "--epochs",
            "200",
            "--learning-rate",
            "0.001",
            "--hidden-dim",
            "64",
            "--seed",
            "20260717",
        ]

    outputs = [phase_root / "recoverability" / "baseline" / repeat for repeat in ("run1", "run2")]
    for output in outputs:
        subprocess.run(command(output), cwd=ROOT, check=True, capture_output=True, text=True)

    run1, run2 = outputs
    siblings = ("details.json", "feature_schema.json", "split_manifest.json", "probe_states.json")
    assert sorted(path.name for path in run1.iterdir()) == [*sorted(siblings), "summary.json"]
    assert all((run1 / name).read_bytes() == (run2 / name).read_bytes() for name in siblings)
    summaries = [json.loads((output / "summary.json").read_bytes()) for output in outputs]
    for summary in summaries:
        for key in ("checkpoint", "output", "device", "runtime_seconds"):
            del summary[key]
        del summary["environment"]["torch_version"]
        del summary["environment"]["platform"]
    assert summaries[0] == summaries[1]

    summary = json.loads((run1 / "summary.json").read_bytes())
    details = json.loads((run1 / "details.json").read_bytes())
    feature_artifact = json.loads((run1 / "feature_schema.json").read_bytes())
    split = json.loads((run1 / "split_manifest.json").read_bytes())
    states = json.loads((run1 / "probe_states.json").read_bytes())
    assert set(summary) == {
        "schema_version",
        "kind",
        "dataset",
        "checkpoint",
        "checkpoint_sha256",
        "split",
        "seed",
        "candidate_manifest",
        "settings",
        "checkpoint_restoration",
        "counts",
        "metrics",
        "environment",
        "device",
        "output",
        "runtime_seconds",
    }
    assert set(summary["settings"]) == {
        "epochs",
        "learning_rate",
        "hidden_dim",
        "models",
        "feature_sets",
        "threshold_policy",
        "control_policy",
        "reliability_bins",
    }
    assert set(summary["counts"]) == {
        "records",
        "train_records",
        "eval_records",
        "train_groups",
        "eval_groups",
        "applicable",
        "inapplicable",
    }
    assert set(summary["metrics"]) == {"features", "verdicts"}
    assert set(summary["metrics"]["verdicts"]) == {
        "latent_separable",
        "raw_separable",
        "hybrid_separable",
        "label_or_sampling_blocker",
        "latent_state_bottleneck",
        "any_abc_separable",
    }
    metric_keys = {
        "count",
        "positive_count",
        "negative_count",
        "prevalence",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "average_precision",
        "nll",
        "brier",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "reliability_bins",
    }
    distribution_keys = {"count", "min", "median", "mean", "max"}
    feature_names = [row["name"] for row in module.feature_schemas()]
    assert list(summary["metrics"]["features"]) == feature_names
    for feature in feature_names:
        assert set(summary["metrics"]["features"][feature]) == {"linear", "mlp", "control_mlp"}
        for probe in summary["metrics"]["features"][feature].values():
            assert set(probe) == {
                "train",
                "eval",
                "role_swap_margin",
                "one_arg_substitution_margin",
                "per_schema",
                "threshold",
            }
            assert set(probe["train"]) == set(probe["eval"]) == metric_keys
            assert set(probe["role_swap_margin"]) == set(probe["one_arg_substitution_margin"]) == distribution_keys
            assert list(probe["per_schema"]) == list(module.SCHEMAS)
            assert all(set(value) == metric_keys for value in probe["per_schema"].values())

    assert len(details) == 604
    detail_keys = {
        "manifest_index",
        "group",
        "problem",
        "step",
        "action",
        "category",
        "label",
        "split",
        "logits",
        "control_logits",
    }
    original_keys = {f"{feature}/{kind}" for feature in feature_names for kind in ("linear", "mlp")}
    control_keys = {f"{feature}/mlp" for feature in feature_names}
    for index, (detail, record) in enumerate(zip(details, records, strict=True)):
        assert set(detail) == detail_keys and detail["manifest_index"] == index
        assert {key: detail[key] for key in ("group", "problem", "step", "action", "category")} == {
            key: record[key] for key in ("group", "problem", "step", "action", "category")
        }
        assert detail["label"] is record["applicability_label"]
        assert set(detail["logits"]) == original_keys
        assert set(detail["control_logits"]) == control_keys

    assert set(feature_artifact) == {"schema_version", "candidate_manifest_sha256", "feature_sets"}
    assert feature_artifact["feature_sets"] == module.feature_schemas()
    assert split == module.split_manifest()
    assert len((run1 / "split_manifest.json").read_bytes()) == 455
    assert hashlib.sha256((run1 / "split_manifest.json").read_bytes()).hexdigest() == module.SPLIT_SHA256
    assert set(states) == {"schema_version", "candidate_manifest_sha256", "split_manifest_sha256", "training", "models"}
    assert set(states["training"]) == {"seed", "epochs", "learning_rate", "hidden_dim", "optimizer", "dtype"}
    assert [(state["feature_set"], state["model_kind"]) for state in states["models"]] == [
        (feature, kind) for feature in feature_names for kind in ("linear", "mlp", "control_mlp")
    ]
    for state in states["models"]:
        assert set(state) == {"feature_set", "model_kind", "input_dim", "architecture", "preprocessing", "state_dict"}
        assert set(state["preprocessing"]) == {
            "mean",
            "std",
            "binary_indices",
            "standardized_indices",
            "zero_std_indices",
        }
        assert all(set(tensor) == {"name", "shape", "dtype", "values"} for tensor in state["state_dict"])
        assert [tensor["name"] for tensor in state["state_dict"]] == sorted(
            tensor["name"] for tensor in state["state_dict"]
        )
        assert module.reconstruct_probe(state) is not None

    extraction_args = SimpleNamespace(
        dataset_dir=module.DATASET,
        checkpoint=module.BASELINE_CHECKPOINT,
        device="cpu",
        split="val",
        seed=20260717,
    )
    extracted, _restoration, _bundle = module._collect_features(extraction_args, records)
    for state in states["models"]:
        model = module.reconstruct_probe(state)
        inputs = module.apply_preprocessing(extracted[state["feature_set"]], state["preprocessing"])
        with torch.no_grad():
            reconstructed = model(inputs).flatten()
        if state["model_kind"] == "control_mlp":
            expected = torch.tensor([detail["control_logits"][f"{state['feature_set']}/mlp"] for detail in details])
        else:
            expected = torch.tensor(
                [detail["logits"][f"{state['feature_set']}/{state['model_kind']}"] for detail in details]
            )
        assert torch.allclose(reconstructed, expected, atol=1e-7, rtol=0)

    existing = subprocess.run(command(run1), cwd=ROOT, capture_output=True, text=True)
    assert existing.returncode != 0 and "destination already exists" in existing.stderr
    bad_root = tmp_path / "bad_marker" / "updated_phase0"
    bad_root.mkdir(parents=True)
    (bad_root / "root_identity.json").write_text("{}\n")
    bad = subprocess.run(
        command(bad_root / "recoverability" / "baseline" / "run1"), cwd=ROOT, capture_output=True, text=True
    )
    assert bad.returncode != 0 and "root identity marker" in bad.stderr
