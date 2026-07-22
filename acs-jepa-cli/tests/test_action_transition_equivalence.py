from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from acs_jepa import JEPALatentState

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "script" / "diagnose_action_transition_equivalence.py"


def _load(name: str = "diagnose_action_transition_equivalence"):
    script_dir = str(ROOT / "script")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row(name, arguments, *, label=False, category="one_arg_substitution"):
    return {
        "action": {"name": name, "arguments": list(arguments)},
        "applicability_label": label,
        "category": category,
    }


def test_filter_hard_negatives_preserves_manifest_order_and_rejects_duplicate_actions() -> None:
    module = _load("transition_filter")
    trace = _row("move", ("trace",), label=True, category="trace")
    rows = [
        trace,
        _row("move", ("first",), category="random_same_schema"),
        _row("move", ("applicable",), label=True),
        _row("other", ("wrong-schema",)),
        _row("move", ("wrong-category",), category="random_other_schema"),
        _row("move", ("second",), category="role_swap"),
    ]

    selected = module.filter_hard_negatives(rows, trace)

    assert [row["action"]["arguments"] for row in selected] == [["first"], ["second"]]
    with pytest.raises(ValueError, match="unique"):
        module.filter_hard_negatives(rows + [_row("move", ("first",), category="role_swap")], trace)


def test_nearest_wrong_uses_all_64_cpu_float64_coordinates_zero_policy_and_canonical_ties() -> None:
    module = _load("transition_distance")
    true_native = torch.zeros((1, 64), dtype=torch.float32)
    first_native = torch.zeros((1, 64), dtype=torch.float32)
    first_native[0, 63] = 1
    canonical_native = -first_native
    rows = [_row("move", ("z",)), _row("move", ("a",))]

    selected, distance, copies = module.select_nearest_wrong(
        true_native, [(rows[0], first_native), (rows[1], canonical_native)]
    )

    assert selected is rows[1]
    assert distance == pytest.approx(1.0)
    assert all(value.device.type == "cpu" and value.dtype == torch.float64 and value.shape == (64,) for value in copies)
    assert torch.equal(true_native, torch.zeros((1, 64), dtype=torch.float32))
    with pytest.raises(ValueError, match="64"):
        module.select_nearest_wrong(torch.zeros(1, 63), [(rows[0], torch.zeros(1, 63))])
    with pytest.raises(ValueError, match="finite"):
        bad = true_native.clone()
        bad[0, 0] = torch.nan
        module.select_nearest_wrong(bad, [(rows[0], first_native)])


def _state(graph, objects, ids=None, batch=None):
    count = len(objects)
    graph = [*graph, *([0.0] * (64 - len(graph)))]
    objects = [[*row, *([0.0] * (64 - len(row)))] for row in objects]
    return JEPALatentState(
        graph_latent=torch.tensor([graph]),
        object_latents=torch.tensor(objects),
        object_ids=torch.arange(count) if ids is None else torch.tensor(ids),
        object_batch=torch.zeros(count, dtype=torch.long) if batch is None else torch.tensor(batch),
    )


def _eligible_detail(module, index: int = 0, *, category: str = "one_arg_substitution"):
    problem, step_text = module.GROUPS[index].split(":")
    rows = _fixed_rows_by_group()[module.GROUPS[index]]
    trace = next(row for row in rows if row["category"] == "trace")
    eligible = [
        row
        for row in rows
        if row["applicability_label"] is False
        and row["action"]["name"] == trace["action"]["name"]
        and row["category"] in ("one_arg_substitution", "role_swap", "random_same_schema")
    ]
    matching = [row for row in eligible if row["category"] == category]
    wrong = (matching or eligible)[0]
    return {
        "group": module.GROUPS[index],
        "problem": problem,
        "step": int(step_text),
        "trace_action": copy.deepcopy(trace["action"]),
        "status": "eligible",
        "skip_reason": None,
        "wrong_action": copy.deepcopy(wrong["action"]),
        "wrong_category": wrong["category"],
        "wrong_unit_action_l2": 0.5,
        "true_graph_error": 1.0,
        "true_object_error": 2.0,
        "true_total_error": 3.0,
        "wrong_graph_error": 2.0,
        "wrong_object_error": 2.0,
        "wrong_total_error": 4.0,
        "prediction_graph_separation": 0.25,
        "prediction_object_separation": 0.25,
        "prediction_separation": 0.5,
        "error_ratio": 4.0 / 3.0,
        "error_margin": 1.0,
        "separation_ratio": 1.0 / 6.0,
        "transition_equivalent": False,
    }


def _valid_details(module):
    return [_eligible_detail(module, index, category=module.CANDIDATE_CATEGORIES[index % 3]) for index in range(44)]


@cache
def _fixed_rows_by_group():
    path = Path(
        "/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json"
    )
    grouped = {}
    for row in json.loads(path.read_text()):
        grouped.setdefault(row["group"], []).append(row)
    return grouped


def _valid_summary(module, details, *, checkpoint=None, output=None):
    checkpoint = module.BASELINE_CHECKPOINT if checkpoint is None else checkpoint
    output = Path("/tmp/updated_phase0/transition_equivalence/baseline/run1") if output is None else output
    counts, metrics = module.aggregate_details(details)
    baseline = checkpoint == module.BASELINE_CHECKPOINT
    state_keys = {
        "jepa": "model_state_dict",
        "goal_head": "goal_head_state_dict",
        "action_contrastive_anchor": "action_contrastive_anchor_state_dict",
        "argument_reconstruction_head": "argument_reconstruction_head_state_dict",
        "applicability_head": "applicability_head_state_dict",
    }
    return {
        "schema_version": "action_latent_updated_phase0.transition_equivalence.v1",
        "kind": "transition_equivalence",
        "dataset": str(module.DATASET),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": module.FIXED_SHA256[checkpoint],
        "split": "val",
        "seed": 20260717,
        "candidate_manifest": {
            "path": str(module.FIXED_CANDIDATE_MANIFEST),
            "bytes": 117385,
            "sha256": "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9",
            "count": 604,
        },
        "settings": {
            "chunk_size": 2048,
            "error_ratio_threshold": 1.10,
            "separation_ratio_threshold": 0.25,
            "mostly_rate_threshold": 0.50,
            "float64_epsilon": torch.finfo(torch.float64).eps,
            "candidate_categories": list(module.CANDIDATE_CATEGORIES),
            "distance_policy": "raw_action_latent_cpu_float64_unit_l2_64d",
            "zero_norm_policy": "zero_vector",
            "graph_weight": 1.0,
            "object_weight": 1.0,
            "near_zero_true_error_policy": "true_total_error_lte_float64_epsilon",
            "detail_order": "literal_fixed_group_order",
        },
        "checkpoint_restoration": {
            name: {
                "state_key": key,
                "status": ("restored" if name in {"jepa", "goal_head"} or not baseline else "disabled"),
            }
            for name, key in state_keys.items()
        },
        "counts": counts,
        "metrics": metrics,
        "environment": {
            "python_version": "3.13.5",
            "torch_version": "2.x",
            "platform": "test",
            "byteorder": "little",
            "num_threads": 1,
            "num_interop_threads": 1,
            "deterministic_algorithms": True,
            "python_hash_seed": None,
            "cublas_workspace_config": ":4096:8",
        },
        "device": "cuda:0",
        "output": str(output),
        "runtime_seconds": 1.0,
    }


def test_transition_components_are_cpu_float64_weighted_and_metadata_aligned() -> None:
    module = _load("transition_components")
    predicted = _state([1.0, 3.0], [[1.0, 2.0], [4.0, 8.0]], ids=[4, 9])
    target = _state([3.0, 7.0], [[2.0, 4.0], [8.0, 10.0]], ids=[4, 9])
    parsed = type("Parsed", (), {"objects": {"b": object(), "a": object()}, "object_to_id": {"a": 4, "b": 9}})()

    graph, objects, total = module.transition_components(predicted, target, parsed, graph_weight=1.0, object_weight=1.0)

    assert graph == pytest.approx(20.0 / 64)
    assert objects == pytest.approx(25.0 / 128)
    assert total == pytest.approx(graph + objects)
    for mutation in (
        _state([3.0, 7.0], [[8.0, 10.0], [2.0, 4.0]], ids=[9, 4]),
        _state([3.0, 7.0], [[2.0, 4.0], [8.0, 10.0]], ids=[9, 4]),
        _state([3.0, 7.0], [[2.0, 4.0], [8.0, 10.0]], ids=[4, 4]),
        _state([3.0, 7.0], [[2.0, 4.0]], ids=[4]),
        _state([3.0, 7.0], [[2.0, 4.0], [8.0, 10.0]], ids=[4, 9], batch=[0, 1]),
    ):
        with pytest.raises(ValueError, match="canonical|metadata"):
            module.transition_components(predicted, mutation, parsed, graph_weight=1.0, object_weight=1.0)
    with pytest.raises(ValueError, match="1.0"):
        module.transition_components(predicted, target, parsed, graph_weight=2.0, object_weight=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata-copy regression requires CUDA")
def test_transition_components_validate_native_cuda_metadata_through_cpu_long_copies() -> None:
    module = _load("transition_cuda_metadata")
    parsed = type("Parsed", (), {"objects": {"a": object()}, "object_to_id": {"a": 0}})()
    left = _state([1.0], [[2.0]])
    right = _state([1.0], [[2.0]])
    left = JEPALatentState(
        graph_latent=left.graph_latent.cuda(),
        object_latents=left.object_latents.cuda(),
        object_ids=left.object_ids.cuda(),
        object_batch=left.object_batch.cuda(),
    )
    right = JEPALatentState(
        graph_latent=right.graph_latent.cuda(),
        object_latents=right.object_latents.cuda(),
        object_ids=right.object_ids.cuda(),
        object_batch=right.object_batch.cuda(),
    )
    assert module.transition_components(left, right, parsed, graph_weight=1.0, object_weight=1.0) == (0.0, 0.0, 0.0)


def test_row_metrics_use_float64_epsilon_inclusive_boundaries_and_exact_aggregations() -> None:
    module = _load("transition_metrics")
    eps = torch.finfo(torch.float64).eps
    row = module.derived_metrics(eps, 1.10 * eps, 0.25 * eps, error_threshold=1.10, separation_threshold=0.25)
    assert {key: row[key] for key in ("error_ratio", "error_margin", "separation_ratio")} == pytest.approx(
        {"error_ratio": 1.10, "error_margin": 0.10 * eps, "separation_ratio": 0.25}
    )
    assert row["transition_equivalent"] is True
    assert row["near_zero"] is True
    assert (
        module.derived_metrics(
            eps,
            1.10 * eps,
            torch.nextafter(torch.tensor(0.25 * eps), torch.tensor(float("inf"))).item(),
            error_threshold=1.10,
            separation_threshold=0.25,
        )["transition_equivalent"]
        is False
    )

    details = [
        {
            "status": "eligible",
            "wrong_category": "one_arg_substitution",
            "error_ratio": 1.0,
            "error_margin": -1.0,
            "separation_ratio": 0.1,
            "transition_equivalent": True,
            "true_total_error": 0.0,
        },
        {
            "status": "eligible",
            "wrong_category": "role_swap",
            "error_ratio": 2.0,
            "error_margin": 3.0,
            "separation_ratio": 0.9,
            "transition_equivalent": False,
            "true_total_error": 1.0,
        },
        {
            "status": "skipped",
            "wrong_category": None,
            "error_ratio": None,
            "error_margin": None,
            "separation_ratio": None,
            "transition_equivalent": None,
            "true_total_error": None,
        },
    ]
    counts, metrics = module.aggregate_details(details, minimum_eligible=2)
    assert counts == {"groups": 3, "eligible": 2, "skipped": 1, "exact_or_near_zero_true_error": 1}
    assert metrics["error_margin"]["median"] == pytest.approx(1.0)
    assert metrics["equivalence_rate"] == pytest.approx(0.5)
    assert metrics["mostly_transition_equivalent"] is True
    assert metrics["per_category"]["random_same_schema"]["equivalence_rate"] is None
    with pytest.raises(ValueError, match="fewer than 40"):
        module.aggregate_details(details, minimum_eligible=40)


def test_default_aggregate_gate_rejects_exactly_39_eligible_rows() -> None:
    module = _load("transition_minimum_eligible")
    details = [_eligible_detail(module, index % 44) for index in range(39)]
    with pytest.raises(ValueError, match="fewer than 40"):
        module.aggregate_details(details)


def test_reconcile_transition_enforces_exact_successor_and_trace_without_reading_successor() -> None:
    module = _load("transition_reconcile")
    trace = SimpleNamespace(name="move", arguments=("a",))
    trajectory = SimpleNamespace(problem_index=0, actions=(trace,), states=("source", "successor"))
    corpus = SimpleNamespace(
        trajectories=(trajectory,), records=(SimpleNamespace(problem_name="p"),), parsed_problems=(SimpleNamespace(),)
    )
    records = [
        {**_row("move", ("a",), label=True, category="trace"), "group": "p:0", "problem": "p", "step": 0},
        {**_row("move", ("b",)), "group": "p:0", "problem": "p", "step": 0},
    ]

    reconciled = module.reconcile_transitions(records, corpus)

    assert len(reconciled) == 1
    assert reconciled[0].source_atoms == "source"
    assert reconciled[0].successor_index == 1
    assert not hasattr(reconciled[0], "successor_atoms")
    bad = SimpleNamespace(problem_index=0, actions=(trace,), states=("source",))
    with pytest.raises(ValueError, match="states.*actions"):
        module.reconcile_transitions(
            records,
            SimpleNamespace(trajectories=(bad,), records=corpus.records, parsed_problems=corpus.parsed_problems),
        )


def test_parser_and_binding_pin_every_stage0c_literal() -> None:
    module = _load("transition_parser")
    assert module.CANDIDATE_CATEGORIES == ("one_arg_substitution", "role_swap", "random_same_schema")
    assert module.SETTINGS["distance_policy"] == "raw_action_latent_cpu_float64_unit_l2_64d"
    args = module.build_parser().parse_args(
        ["data", "--checkpoint", "model", "--candidate-manifest", "manifest", "--output", "out"]
    )
    assert (
        args.device,
        args.split,
        args.chunk_size,
        args.seed,
        args.equivalence_error_ratio,
        args.equivalence_separation_ratio,
    ) == ("cuda", "val", 2048, 20260717, 1.10, 0.25)
    args.chunk_size = 0
    with pytest.raises(ValueError, match="fixed"):
        module.validate_args(args)


def test_recursive_artifact_schema_and_repeat_projection_reject_mutations() -> None:
    module = _load("transition_schema")
    details = _valid_details(module)
    module.validate_details(details)
    for mutation in ("unknown", "nonfinite", "partial_skip"):
        changed = copy.deepcopy(details)
        if mutation == "unknown":
            changed[0]["unknown"] = 1
        elif mutation == "nonfinite":
            changed[0]["step"] = float("nan")
        else:
            changed[0]["status"] = "skipped"
        with pytest.raises(ValueError):
            module.validate_details(changed)

    summary = _valid_summary(module, details)
    repeated = copy.deepcopy(summary)
    repeated.update(
        {"output": str(Path(summary["output"]).with_name("run2")), "device": "cuda:1", "runtime_seconds": 9.0}
    )
    repeated["environment"].update(torch_version="y", platform="y")
    assert module.repeat_projection(summary) == module.repeat_projection(repeated)
    repeated["environment"]["num_threads"] = 2
    assert module.repeat_projection(summary) != module.repeat_projection(repeated)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].__setitem__("group", "p166:1"),
        lambda rows: rows[0].__setitem__("problem", False),
        lambda rows: rows[0].__setitem__("step", True),
        lambda rows: rows[0]["trace_action"].__setitem__("name", 7),
        lambda rows: rows[0]["wrong_action"].__setitem__("arguments", ["ok", 7]),
        lambda rows: rows[0]["wrong_action"].__setitem__("name", "car_start"),
        lambda rows: rows[0].__setitem__("wrong_category", "NOT_ALLOWED"),
        lambda rows: rows[0].__setitem__("wrong_unit_action_l2", -0.1),
        lambda rows: rows[0].__setitem__("transition_equivalent", 7),
        lambda rows: rows[0].__setitem__("true_total_error", -1.0),
        lambda rows: rows[0].__setitem__("error_ratio", 99.0),
    ],
)
def test_details_validator_rejects_exact_order_types_ranges_and_recomputed_values(mutation) -> None:
    module = _load("transition_recursive_details")
    details = _valid_details(module)
    module.validate_details(details)
    mutation(details)
    with pytest.raises(ValueError):
        module.validate_details(details)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].__setitem__("wrong_unit_action_l2", -0.01),
        lambda rows: rows[0].__setitem__("wrong_unit_action_l2", 2.01),
        lambda rows: rows[0].__setitem__("wrong_action", copy.deepcopy(rows[0]["trace_action"])),
        lambda rows: rows[0].__setitem__(
            "wrong_action", {"name": rows[0]["trace_action"]["name"], "arguments": ["absent-object"]}
        ),
        lambda rows: rows[0]["trace_action"]["arguments"].__setitem__(0, "corrupted-trace-argument"),
        lambda rows: rows[0]["wrong_action"]["arguments"].__setitem__(0, "corrupted-wrong-argument"),
    ],
)
def test_details_validator_binds_actions_and_unit_distance_to_fixed_manifest(mutation) -> None:
    module = _load("transition_semantic_manifest_binding")
    details = _valid_details(module)
    module.validate_details(details)
    mutation(details)
    with pytest.raises(ValueError):
        module.validate_details(details)


def test_summary_validation_inherits_fixed_manifest_semantics() -> None:
    module = _load("transition_summary_semantic_binding")
    details = _valid_details(module)
    summary = _valid_summary(module, details)
    details[0]["wrong_action"] = copy.deepcopy(details[0]["trace_action"])
    with pytest.raises(ValueError):
        module.validate_summary(summary, details)


def test_details_and_summary_bind_reconciled_transition_identities_when_available() -> None:
    module = _load("transition_reconciled_semantic_binding")
    details = _valid_details(module)
    reconciled = [
        SimpleNamespace(
            group=row["group"],
            problem=row["problem"],
            step=row["step"],
            trace_action=SimpleNamespace(
                name=row["trace_action"]["name"], arguments=tuple(row["trace_action"]["arguments"])
            ),
        )
        for row in details
    ]
    module.validate_details(details, reconciled_transitions=reconciled)
    summary = _valid_summary(module, details)
    module.validate_summary(summary, details, reconciled_transitions=reconciled)
    reconciled[0].step += 1
    with pytest.raises(ValueError, match="reconciled"):
        module.validate_summary(summary, details, reconciled_transitions=reconciled)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["settings"].__setitem__("distance_policy", "WRONG_POLICY"),
        lambda value: value["settings"].__setitem__(
            "candidate_categories", ["role_swap", "one_arg_substitution", "random_same_schema"]
        ),
        lambda value: value["counts"].__setitem__("eligible", -1),
        lambda value: value["counts"].__setitem__("eligible", 43),
        lambda value: value["metrics"].__setitem__("equivalence_rate", 2.5),
        lambda value: value["metrics"].__setitem__("mostly_transition_equivalent", True),
        lambda value: value["metrics"]["error_margin"].__setitem__("mean", 99.0),
        lambda value: value["metrics"]["per_category"]["role_swap"].__setitem__("count", 0),
        lambda value: value["checkpoint_restoration"]["action_contrastive_anchor"].__setitem__("status", "restored"),
        lambda value: value["candidate_manifest"].__setitem__("count", 603),
        lambda value: value["environment"].__setitem__("num_threads", True),
    ],
)
def test_summary_validator_rejects_settings_restoration_counts_rates_and_distributions(mutation) -> None:
    module = _load("transition_recursive_summary")
    details = _valid_details(module)
    summary = _valid_summary(module, details)
    module.validate_summary(summary, details)
    mutation(summary)
    with pytest.raises(ValueError):
        module.validate_summary(summary, details)


def test_phase2_restoration_and_repeat_projection_are_fully_validated() -> None:
    module = _load("transition_repeat_full_validation")
    details = _valid_details(module)
    output = Path("/tmp/updated_phase0/transition_equivalence/phase2/run1")
    summary = _valid_summary(module, details, checkpoint=module.PHASE2_CHECKPOINT, output=output)
    module.validate_summary(summary, details)
    assert module.repeat_projection(summary) == module.repeat_projection(copy.deepcopy(summary))
    changed = copy.deepcopy(summary)
    changed["settings"]["distance_policy"] = "WRONG_POLICY"
    with pytest.raises(ValueError):
        module.repeat_projection(changed)


def test_runtime_environment_materializes_plain_json_scalar_types() -> None:
    module = _load("transition_environment_types")
    environment = module._environment()
    assert type(environment["torch_version"]) is str
    assert type(environment["python_version"]) is str


def test_no_simulator_or_oracle_dependency_and_fixed_binding() -> None:
    module = _load("transition_no_sim")
    tree = ast.parse(SCRIPT.read_text())
    imports = {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))}
    assert all(
        token not in statement for statement in imports for token in ("simulator", "action_diag_common", "replay")
    )
    source = SCRIPT.read_text()
    assert "applicable_actions(" not in source and "replay_trajectory(" not in source
    root = Path("/tmp/root")
    module.validate_run_binding(module.BASELINE_CHECKPOINT, root / "transition_equivalence/baseline/run1")
    module.validate_run_binding(module.PHASE2_CHECKPOINT, root / "transition_equivalence/phase2/run2")
    with pytest.raises(ValueError, match="binding"):
        module.validate_run_binding(module.PHASE2_CHECKPOINT, root / "transition_equivalence/baseline/run1")


def test_root_marker_and_existing_partial_destination_are_refused_before_reuse(tmp_path: Path) -> None:
    module = _load("transition_root_refusal")
    identity = {"fixed": True}
    root = tmp_path / "updated_phase0"
    root.mkdir()
    (root / "root_identity.json").write_bytes(module.canonical_json_bytes(identity))
    destination = root / "transition_equivalence" / "baseline" / "run1"
    module.prepare_output_directory(root, destination, identity, first_command=False)
    (destination / "partial.tmp").write_text("partial")
    with pytest.raises(FileExistsError, match="destination already exists"):
        module.prepare_output_directory(root, destination, identity, first_command=False)
    changed = root / "transition_equivalence" / "baseline" / "run2"
    with pytest.raises(ValueError, match="root identity"):
        module.prepare_output_directory(root, changed, {"fixed": False}, first_command=False)
    assert not changed.exists()


def test_actual_fixed_manifest_reconciles_all_44_exact_successors() -> None:
    module = _load("transition_actual_reconcile")
    records, _ = module.load_and_validate_candidate_manifest(module.FIXED_CANDIDATE_MANIFEST)
    config, corpus, _bundle, _device, _restoration = module.load_checkpoint_bundle(
        module.DATASET, module.BASELINE_CHECKPOINT, device_name="cpu", include_restoration_metadata=True
    )
    selected = module.select_split(corpus, config, "val", seed=20260717)
    transitions = module.reconcile_transitions(records, selected)
    assert tuple(row.group for row in transitions) == module.GROUPS
    assert len(transitions) == 44
    assert all(row.successor_index == row.step + 1 for row in transitions)
    assert (
        sum(
            bool(
                module.filter_hard_negatives(
                    row.records, next(item for item in row.records if item["category"] == "trace")
                )
            )
            for row in transitions
        )
        >= 40
    )


def test_evaluate_transition_predicts_with_native_latents_before_exact_successor_encode(monkeypatch) -> None:
    module = _load("transition_sequence")
    events = []
    parsed = SimpleNamespace(objects={"a": object(), "b": object()}, object_to_id={"a": 0, "b": 1})

    class Graph:
        def __init__(self, atoms):
            self.atoms = atoms

        def to(self, device):
            events.append(("graph_to", self.atoms, str(device)))
            return self

    class States:
        def __init__(self, source, successor):
            self.source = source
            self.successor = successor

        def __getitem__(self, index):
            events.append(("state_read", index))
            return self.source if index == 0 else self.successor

    class JEPA:
        def encode(self, graph):
            events.append(("encode", graph.atoms))
            value = float(graph.atoms)
            return _state([value], [[value], [value + 1]], ids=[0, 1])

        def predictor(self, source, action):
            assert action.device == source.graph_latent.device
            assert action.dtype == source.graph_latent.dtype == torch.float32
            events.append(("predict", float(action[0, 0])))
            value = float(source.graph_latent[0, 0] + action[0, 0])
            return _state([value], [[value], [value + 1]], ids=[0, 1])

    monkeypatch.setattr(
        module,
        "build_state_graph",
        lambda parsed, atoms, include_static: events.append(("build", atoms, include_static)) or Graph(atoms),
    )
    monkeypatch.setattr(module.ActionDecodingSpace, "from_parsed_problem", lambda parsed: object())

    def encode_action(bundle, space, row, source, device):
        source_value = float(source.graph_latent[0, 0])
        events.append(("action", row["action"]["arguments"], source_value))
        value = source_value + (0.0 if row["category"] == "trace" else 1.0)
        return torch.full((1, 64), value, dtype=source.graph_latent.dtype, device=source.graph_latent.device)

    monkeypatch.setattr(module, "_encode_action", encode_action)
    records = (
        _row("move", ("trace",), label=True, category="trace"),
        _row("move", ("wrong",), category="role_swap"),
        _row("move", ("ignored",), category="random_other_schema"),
    )

    def execute(successor, *, source=1.0, rows=records):
        events.clear()
        item = module.ReconciledTransition(
            "g",
            "p",
            0,
            parsed,
            source,
            SimpleNamespace(name="move", arguments=("trace",)),
            rows,
            SimpleNamespace(states=States(source, successor)),
            1,
        )
        result = module.evaluate_transition(
            item,
            SimpleNamespace(jepa=JEPA()),
            device=torch.device("cpu"),
            graph_weight=1.0,
            object_weight=1.0,
            error_threshold=1.10,
            separation_threshold=0.25,
        )
        return result, list(events)

    first, first_events = execute(2.0)
    second, second_events = execute(5.0)
    first_successor_read = first_events.index(("state_read", 1))
    assert [event[0] for event in first_events[:first_successor_read]].count("predict") == 2
    assert all(event[2] is True for event in first_events if event[0] == "build")
    assert first["wrong_action"] == second["wrong_action"]
    assert first["wrong_unit_action_l2"] == second["wrong_unit_action_l2"]
    assert [event for event in first_events if event[0] in {"action", "predict"}] == [
        event for event in second_events if event[0] in {"action", "predict"}
    ]
    assert first["true_total_error"] != second["true_total_error"]

    mutated_source, source_events = execute(2.0, source=7.0)
    assert [event for event in first_events if event == ("encode", 2.0)] == [
        event for event in source_events if event == ("encode", 2.0)
    ]
    assert first["wrong_action"] == mutated_source["wrong_action"]
    assert [event for event in first_events if event[0] == "action"] != [
        event for event in source_events if event[0] == "action"
    ]
    assert [event for event in first_events if event[0] == "predict"] != [
        event for event in source_events if event[0] == "predict"
    ]
    assert first["true_total_error"] != mutated_source["true_total_error"]

    irrelevant_mutation = tuple(copy.deepcopy(row) for row in records)
    irrelevant_mutation[2]["applicability_label"] = True
    irrelevant_mutation[2]["category"] = "trace"
    filtered, filtered_events = execute(2.0, rows=irrelevant_mutation)
    assert filtered["wrong_action"] == first["wrong_action"]
    assert [event for event in filtered_events if event[0] in {"action", "predict"}] == [
        event for event in first_events if event[0] in {"action", "predict"}
    ]


def test_actual_cpu_source_mutation_changes_native_action_and_prediction_but_not_recorded_target() -> None:
    module = _load("transition_actual_source_causality")
    records, _ = module.load_and_validate_candidate_manifest(module.FIXED_CANDIDATE_MANIFEST)
    config, corpus, bundle, _device, _restoration = module.load_checkpoint_bundle(
        module.DATASET, module.BASELINE_CHECKPOINT, device_name="cpu", include_restoration_metadata=True
    )
    selected_corpus = module.select_split(corpus, config, "val", seed=20260717)
    item = module.reconcile_transitions(records, selected_corpus)[0]
    trace_row = next(row for row in item.records if row["category"] == "trace")
    wrong_row = module.filter_hard_negatives(item.records, trace_row)[0]
    shared_action_arguments = set(item.trace_action.arguments).intersection(wrong_row["action"]["arguments"])
    assert shared_action_arguments
    removed = next(
        atom
        for atom in item.source_atoms
        if atom.predicate not in item.parsed.static_predicates and shared_action_arguments.intersection(atom.arguments)
    )
    mutated_atoms = tuple(atom for atom in item.source_atoms if atom != removed)
    assert len(mutated_atoms) + 1 == len(item.source_atoms)

    device = torch.device("cpu")
    space = module.ActionDecodingSpace.from_parsed_problem(item.parsed)
    successor_graph = module.build_state_graph(
        item.parsed, item.trajectory.states[item.successor_index], include_static=True
    ).to(device)
    bundle.jepa.eval()
    with torch.inference_mode():
        recorded_target = bundle.jepa.encode(successor_graph)

        def source_path(atoms):
            source_graph = module.build_state_graph(item.parsed, atoms, include_static=True).to(device)
            source = bundle.jepa.encode(source_graph)
            trace_action = module._encode_action(bundle, space, trace_row, source, device)
            wrong_action = module._encode_action(bundle, space, wrong_row, source, device)
            return (
                source,
                trace_action,
                wrong_action,
                bundle.jepa.predictor(source, trace_action),
                bundle.jepa.predictor(source, wrong_action),
            )

        original = source_path(item.source_atoms)
        mutated = source_path(mutated_atoms)
        target_again = bundle.jepa.encode(successor_graph)

    assert not torch.equal(original[0].graph_latent, mutated[0].graph_latent)
    assert not torch.equal(original[1], mutated[1])
    assert not torch.equal(original[2], mutated[2])
    assert not torch.equal(original[3].graph_latent, mutated[3].graph_latent)
    assert not torch.equal(original[4].graph_latent, mutated[4].graph_latent)
    assert torch.equal(recorded_target.graph_latent, target_again.graph_latent)
    assert torch.equal(recorded_target.object_latents, target_again.object_latents)
    assert torch.equal(recorded_target.object_ids, target_again.object_ids)
    assert torch.equal(recorded_target.object_batch, target_again.object_batch)
