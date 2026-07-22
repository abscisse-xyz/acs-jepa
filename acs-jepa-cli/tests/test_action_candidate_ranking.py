# ruff: noqa: E501
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
RANKING_PATH = SCRIPT_DIR / "diagnose_action_candidate_ranking.py"
OWNER_PATH = SCRIPT_DIR / "action_role_object_probe.py"


def _load(path: Path, name: str):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _owner(name="stage0d_role_owner"):
    return _load(OWNER_PATH, name)


def _ranking(name="stage0d_ranking"):
    return _load(RANKING_PATH, name)


def _rows():
    return [
        {"manifest_index": 0, "group": "g0", "problem": "p", "step": 0, "action": {"name": "car_start", "arguments": ["b"]}, "category": "trace", "label": True, "scores": {"s": 1.0}},
        {"manifest_index": 1, "group": "g0", "problem": "p", "step": 0, "action": {"name": "car_start", "arguments": ["a"]}, "category": "role_swap", "label": False, "scores": {"s": 1.0}},
        {"manifest_index": 2, "group": "g0", "problem": "p", "step": 0, "action": {"name": "car_start", "arguments": ["c"]}, "category": "one_arg_substitution", "label": True, "scores": {"s": 0.0}},
        {"manifest_index": 3, "group": "g1", "problem": "p", "step": 1, "action": {"name": "car_arrived", "arguments": ["z"]}, "category": "trace", "label": True, "scores": {"s": -1.0}},
        {"manifest_index": 4, "group": "g1", "problem": "p", "step": 1, "action": {"name": "car_arrived", "arguments": ["x"]}, "category": "one_arg_substitution", "label": False, "scores": {"s": 2.0}},
    ]


def test_role_owner_masks_padding_and_validates_roles() -> None:
    owner = _owner("stage0d_owner_forward")
    model = owner.RoleObjectProbe(latent_dim=2, action_dim=2, max_action_arity=2, hidden_dim=3)
    logits = model(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [99.0, 99.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]),
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.tensor([0, 1]),
    )
    assert logits.shape == (2, 3)
    assert torch.isneginf(logits[0, 2]) and torch.isfinite(logits[1]).all()
    with pytest.raises(ValueError, match="bool"):
        model(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 1, 2), torch.ones(1, 1), torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="role"):
        model(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), torch.tensor([2]))


def test_role_fit_exact_full_batch_seed_steps_tensors_and_same_shape_mutant(monkeypatch) -> None:
    owner = _owner("stage0d_owner_fit")
    train = (
        torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 3),
        torch.ones(3, 2, dtype=torch.bool),
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 0]),
    )
    evaluation = tuple(value.clone() for value in train)
    calls = []
    targets = []
    adam_parameters = []
    optimizer_steps = []
    original = owner.RoleObjectProbe.forward
    original_ce = torch.nn.functional.cross_entropy
    original_adam = torch.optim.Adam

    def recording_adam(parameters, *args, **kwargs):
        parameters = list(parameters)
        optimizer = original_adam(parameters, *args, **kwargs)
        adam_parameters.extend(parameters)
        original_step = optimizer.step

        def recording_step(*step_args, **step_kwargs):
            optimizer_steps.append(tuple(id(parameter) for group in optimizer.param_groups for parameter in group["params"]))
            return original_step(*step_args, **step_kwargs)

        optimizer.step = recording_step
        return optimizer

    def recording(self, *values):
        calls.append(tuple(value.detach().clone() for value in values))
        return original(self, *values)

    monkeypatch.setattr(owner.RoleObjectProbe, "forward", recording)
    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    monkeypatch.setattr(
        torch.nn.functional,
        "cross_entropy",
        lambda logits, target, *args, **kwargs: (
            targets.append(target.detach().clone()) or original_ce(logits, target, *args, **kwargs)
        ),
    )
    result = owner.fit_role_object_probe(
        train, evaluation, max_action_arity=2, hidden_dim=4, epochs=200,
        learning_rate=0.001, seed=20260717, device=torch.device("cpu")
    )
    assert result.optimizer_steps == 200
    assert len(calls) == 202
    assert all(all(torch.equal(actual, expected) for actual, expected in zip(call, train[:5], strict=True)) for call in calls[:200])
    assert len(targets) == 200 and all(torch.equal(target, train[5]) for target in targets)
    expected_parameter_ids = sorted(id(parameter) for parameter in result.model.parameters())
    assert sorted(id(parameter) for parameter in adam_parameters) == expected_parameter_ids
    assert len(optimizer_steps) == 200 and all(sorted(step) == expected_parameter_ids for step in optimizer_steps)
    assert all(len(step) == len(set(step)) for step in optimizer_steps)
    assert torch.equal(result.train_tensors[5], train[5])
    assert all(value.device.type == "cpu" and (not value.is_floating_point() or value.dtype == torch.float32) for value in result.train_tensors)
    calls.clear()
    mutant = tuple(torch.cat((value[:1], value[:1], value[2:]), dim=0) for value in train)
    owner.fit_role_object_probe(
        mutant, evaluation, max_action_arity=2, hidden_dim=4, epochs=1,
        learning_rate=0.001, seed=20260717, device=torch.device("cpu")
    )
    assert not torch.equal(calls[0][0], train[0])
    assert not torch.equal(calls[0][1], train[1])

    def assert_execution_trace(steps, parameter_ids):
        assert len(steps) == 200
        assert all(sorted(step) == sorted(parameter_ids) for step in steps)
        assert all(len(step) == len(set(step)) for step in steps)

    assert_execution_trace(optimizer_steps[:200], expected_parameter_ids)
    with pytest.raises(AssertionError):
        assert_execution_trace(optimizer_steps[:199], expected_parameter_ids)
    with pytest.raises(AssertionError):
        assert_execution_trace([(*step, step[0]) for step in optimizer_steps[:200]], expected_parameter_ids)
    with pytest.raises(AssertionError):
        assert_execution_trace([step[:-1] for step in optimizer_steps[:200]], expected_parameter_ids)


def test_role_fit_is_reproducible_and_rejects_non_cpu_or_bad_population() -> None:
    owner = _owner("stage0d_owner_repro")
    tensors = (
        torch.zeros(2, 2), torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 2), torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([0, 1]), torch.tensor([0, 1]),
    )
    first = owner.fit_role_object_probe(tensors, tensors, max_action_arity=2, hidden_dim=4, epochs=2, learning_rate=.01, seed=7, device=torch.device("cpu"))
    second = owner.fit_role_object_probe(tensors, tensors, max_action_arity=2, hidden_dim=4, epochs=2, learning_rate=.01, seed=7, device=torch.device("cpu"))
    assert all(torch.equal(first.model.state_dict()[key], second.model.state_dict()[key]) for key in first.model.state_dict())
    assert first.train_metrics == second.train_metrics
    bad = list(tensors)
    bad[3] = torch.ones(2, 2)
    with pytest.raises(ValueError, match="bool"):
        owner.fit_role_object_probe(tuple(bad), tensors, max_action_arity=2, hidden_dim=4, epochs=1, learning_rate=.01, seed=7, device=torch.device("cpu"))


def test_stack_role_candidates_uses_canonical_roles_sorted_ids_and_partition_bank_scopes() -> None:
    module = _ranking("stage0d_role_rows")
    candidates = [
        SimpleNamespace(graph_latent=torch.tensor([1.0, 2.0]), action_latent=torch.tensor([3.0, 4.0]), object_ids=torch.tensor([9, 2]), object_latents=torch.tensor([[90.0, 91.0], [20.0, 21.0]]), argument_mask=torch.tensor([True, False, True, False]), argument_object_ids=torch.tensor([9, -1, 2, -1])),
        SimpleNamespace(graph_latent=torch.tensor([5.0, 6.0]), action_latent=torch.tensor([7.0, 8.0]), object_ids=torch.tensor([4]), object_latents=torch.tensor([[40.0, 41.0]]), argument_mask=torch.tensor([True, False, False, False]), argument_object_ids=torch.tensor([4, -1, -1, -1])),
    ]
    tensors, slices = module.stack_role_candidates(candidates)
    assert slices == [(0, 2), (2, 3)]
    graph, action, banks, masks, roles, targets = tensors
    assert graph.tolist() == [[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]]
    assert action.tolist() == [[3.0, 4.0], [3.0, 4.0], [7.0, 8.0]]
    assert banks[0].tolist() == [[20.0, 21.0], [90.0, 91.0]]
    assert masks.tolist() == [[True, True], [True, True], [True, False]]
    assert roles.tolist() == [0, 2, 0] and targets.tolist() == [1, 0, 0]
    with pytest.raises(ValueError, match="zero active"):
        module.stack_role_candidates([copy.copy(SimpleNamespace(**{**candidates[1].__dict__, "argument_mask": torch.zeros(4, dtype=torch.bool)}))])


def test_complete_action_role_score_is_mean_target_log_probability_and_state_reconstructs() -> None:
    module = _ranking("stage0d_role_scores")
    owner = _owner("stage0d_role_scores_owner")
    model = owner.RoleObjectProbe(latent_dim=64, action_dim=64, max_action_arity=4, hidden_dim=64)
    objects = torch.zeros(3, 2, 64)
    objects[:, 0, 0] = 1.0
    objects[:, 1, 1] = 1.0
    tensors = (torch.zeros(3, 64), torch.zeros(3, 64), objects, torch.ones(3, 2, dtype=torch.bool), torch.tensor([0, 1, 0]), torch.tensor([0, 1, 1]))
    scores = module.role_candidate_scores(model, tensors, [(0, 2), (2, 3)])
    with torch.no_grad():
        expected = model(*tensors[:5]).log_softmax(-1)[torch.arange(3), tensors[5]]
    assert scores == pytest.approx([float(expected[:2].mean()), float(expected[2])])
    state = module.serialize_role_probe(model, candidate_sha256="a" * 64, train_rows=1549, eval_rows=518, optimizer_steps=200)
    rebuilt = module.reconstruct_role_probe(state)
    assert module.role_candidate_scores(rebuilt, tensors, [(0, 2), (2, 3)]) == pytest.approx(scores, abs=1e-7)
    changed = copy.deepcopy(state)
    changed["training"]["train_role_rows"] = 1548
    with pytest.raises(ValueError):
        module.reconstruct_role_probe(changed)
    for mutation in (
        lambda value: value.__setitem__("candidate_manifest_sha256", "not-a-hash"),
        lambda value: value["model"]["architecture"].__setitem__("latent_dim", 2),
        lambda value: value["model"]["architecture"].__setitem__("action_dim", 2),
    ):
        changed = copy.deepcopy(state)
        mutation(changed)
        with pytest.raises(ValueError):
            module.reconstruct_role_probe(changed)


def test_one_based_total_ranks_ties_and_threshold_free_global_per_schema_metrics() -> None:
    module = _ranking("stage0d_metrics")
    rows = module.rank_details(_rows(), scorers=("s",))
    # Equal scores use action key: a before b; raw-score pair gets half credit.
    assert [row["ranks"]["s"] for row in rows[:3]] == [2, 1, 3]
    assert sorted(row["ranks"]["s"] for row in rows[:3]) == [1, 2, 3]
    report = module.ranking_report(rows, "s", deployable=True)
    assert set(report) == {"binary", "role_swap_margin", "one_arg_substitution_margin", "top1_applicable_rate", "mrr_first_applicable", "pairwise_applicable_accuracy", "trace_mrr", "per_schema", "ranks_applicable", "deployable"}
    assert set(report["binary"]) == {"count", "positive_count", "negative_count", "prevalence", "auroc", "average_precision"}
    assert report["top1_applicable_rate"] == pytest.approx(0.0)
    assert report["mrr_first_applicable"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert report["trace_mrr"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert report["pairwise_applicable_accuracy"] == pytest.approx((0.5 + 0 + 0) / 3)
    assert list(report["per_schema"]) == list(module.SCHEMAS)
    assert report["per_schema"]["build_diagonal_oneway"]["groups"] == 0
    assert report["per_schema"]["build_diagonal_oneway"]["top1_applicable_rate"] is None
    assert report["deployable"] is True
    module.validate_rank_reconciliation(rows, {"s": report}, scorers=("s",), deployable={"s": True})
    rows[0]["ranks"]["s"] = 1
    with pytest.raises(ValueError, match="rank"):
        module.validate_rank_reconciliation(rows, {"s": report}, scorers=("s",), deployable={"s": True})


def test_no_applicable_group_contributes_zero_and_duplicate_trace_or_action_is_fatal() -> None:
    module = _ranking("stage0d_group_edges")
    rows = _rows()
    rows[3]["label"] = rows[4]["label"] = False
    ranked = module.rank_details(rows, scorers=("s",))
    report = module.ranking_report(ranked, "s", deployable=False)
    assert report["top1_applicable_rate"] == 0.0
    assert report["mrr_first_applicable"] == pytest.approx(0.25)
    assert report["deployable"] is False
    duplicate = copy.deepcopy(_rows())
    duplicate[1]["action"] = copy.deepcopy(duplicate[0]["action"])
    with pytest.raises(ValueError, match="unique"):
        module.rank_details(duplicate, scorers=("s",))
    missing_trace = copy.deepcopy(_rows())
    missing_trace[0]["category"] = "role_swap"
    with pytest.raises(ValueError, match="trace"):
        module.ranking_report(module.rank_details(missing_trace, scorers=("s",)), "s", deployable=True)


def test_fixed_cli_and_checkpoint_recoverability_binding() -> None:
    module = _ranking("stage0d_parser")
    args = module.build_parser().parse_args(["data", "--checkpoint", "c", "--candidate-manifest", "m", "--recoverability-summary", "a", "--recoverability-details", "b", "--recoverability-feature-schema", "d", "--recoverability-split-manifest", "e", "--recoverability-probe-states", "f", "--output", "o"])
    assert (args.device, args.split, args.epochs, args.learning_rate, args.hidden_dim, args.seed) == ("cpu", "val", 200, .001, 64, 20260717)
    args.epochs = 199
    with pytest.raises(ValueError, match="fixed"):
        module.validate_args(args)


def test_import_boundary_has_pytorch_owner_and_no_forbidden_transitive_modules() -> None:
    source = OWNER_PATH.read_text()
    tree = ast.parse(source)
    imports = {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))}
    assert all(token not in " ".join(imports) for token in ("diagnose_action_supervised_probes", "action_diag_common", "simulator", "replay"))
    before = set(sys.modules)
    module = _ranking("stage0d_boundary")
    loaded = set(sys.modules) - before
    assert "diagnose_action_supervised_probes" not in loaded and "action_diag_common" not in loaded
    ranking_source = RANKING_PATH.read_text()
    assert "applicable_actions(" not in ranking_source and "replay_trajectory(" not in ranking_source
    assert module.RoleObjectProbe.__module__ == "action_role_object_probe"


def test_score_vector_composition_uses_exact_imported_numbers_and_metadata_is_not_a_feature() -> None:
    module = _ranking("stage0d_scores")
    rows = [{"manifest_index": 3, "group": "g", "problem": "p", "step": 1, "action": {"name": "car_start", "arguments": ["c", "g"]}, "category": "trace", "label": True, "logits": {"C_selected_graph_action/mlp": 1.25, "D_raw_symbolic/mlp": -2.5, "E_hybrid/mlp": 3.75}}]
    scores = module.compose_score_vectors(rows, role_scores=[4.5], transition_scores=[-6.0])
    assert scores == [{"latent_transition": -6.0, "latent_applicability": 1.25, "role_object": 4.5, "raw_symbolic": -2.5, "hybrid": 3.75}]
    changed = copy.deepcopy(rows)
    changed[0].update(category="role_swap", label=False)
    assert module.compose_score_vectors(changed, role_scores=[4.5], transition_scores=[-6.0]) == scores
    bad = copy.deepcopy(rows)
    bad[0]["group"] = "other"
    with pytest.raises(ValueError, match="metadata"):
        module.validate_recoverability_alignment(rows, bad)


def test_ranking_recursive_validators_kill_extra_type_category_order_and_metric_mutants() -> None:
    module = _ranking("stage0d_recursive_ranking")
    base = _rows()
    for row in base:
        row["scores"] = {name: float(row["scores"]["s"]) for name in module.SCORERS}
    rows = module.rank_details(base)
    metrics = {name: module.ranking_report(rows, name) for name in module.SCORERS}
    module.validate_ranking_details(rows, manifest_rows=[{k: row[k] for k in ("manifest_index", "group", "problem", "step", "action", "category", "label")} for row in rows], expected_count=5)
    module.validate_ranking_metrics(rows, metrics)
    mutants = []
    changed = copy.deepcopy(rows)
    changed[0]["EXTRA"] = 1
    mutants.append(changed)
    changed = copy.deepcopy(rows)
    changed[0]["label"] = 1
    mutants.append(changed)
    changed = copy.deepcopy(rows)
    changed[0]["category"] = "bogus"
    mutants.append(changed)
    changed = copy.deepcopy(rows)
    changed.reverse()
    mutants.append(changed)
    for changed in mutants:
        with pytest.raises(ValueError):
            module.validate_ranking_details(changed, manifest_rows=[{k: row[k] for k in ("manifest_index", "group", "problem", "step", "action", "category", "label")} for row in rows], expected_count=5)
    changed_metrics = copy.deepcopy(metrics)
    changed_metrics["hybrid"]["binary"]["count"] += 1
    with pytest.raises(ValueError):
        module.validate_ranking_metrics(rows, changed_metrics)


def test_ranking_artifact_validator_rejects_nested_summary_and_state_mutants() -> None:
    module = _ranking("stage0d_artifact_schema")
    model = module.RoleObjectProbe(latent_dim=64, action_dim=64, max_action_arity=4, hidden_dim=64)
    rows = _rows()
    for row in rows:
        row["scores"] = {name: float(row["scores"]["s"]) for name in module.SCORERS}
    details = module.rank_details(rows)
    metrics = {name: module.ranking_report(details, name) for name in module.SCORERS}
    summary = module.ranking_summary_fixture(metrics=metrics, details=details)
    artifacts = {"summary.json": summary, "details.json": details, "split_manifest.json": {"eval_groups": list(module.EVAL_GROUPS), "train_groups": list(module.TRAIN_GROUPS)}, "role_probe_state.json": module.serialize_role_probe(model, candidate_sha256=module.CANDIDATE_SHA256, train_rows=1549, eval_rows=518, optimizer_steps=200)}
    manifest = [{key: row[key] for key in ("manifest_index", "group", "problem", "step", "action", "category", "label")} for row in details]
    module.validate_ranking_artifacts(artifacts, manifest_rows=manifest, expected_count=5)
    mutations = (
        lambda value: value["summary.json"]["settings"].__setitem__("EXTRA", 1),
        lambda value: value["summary.json"]["counts"].__setitem__("eval_records", True),
        lambda value: value["role_probe_state.json"]["model"]["state_dict"][0]["shape"].__setitem__(0, 999),
        lambda value: value["summary.json"].__setitem__("dataset", "/wrong"),
        lambda value: value["summary.json"].__setitem__("checkpoint", str(module.PHASE2_CHECKPOINT)),
        lambda value: value["summary.json"].__setitem__("checkpoint_sha256", "b" * 64),
        lambda value: value["summary.json"].__setitem__("split", "train"),
        lambda value: value["summary.json"].__setitem__("seed", True),
        lambda value: value["summary.json"].__setitem__("device", "cuda"),
        lambda value: value["summary.json"].__setitem__("runtime_seconds", True),
        lambda value: value["summary.json"]["candidate_manifest"].__setitem__("path", "/wrong"),
        lambda value: value["summary.json"]["candidate_manifest"].__setitem__("count", 603),
        lambda value: value["summary.json"]["candidate_manifest"].__setitem__("sha256", "b" * 64),
        lambda value: value["summary.json"]["settings"]["ranking_gate"].__setitem__("auroc", .79),
        lambda value: value["summary.json"]["settings"].__setitem__("ranking_order", "wrong"),
        lambda value: value["summary.json"]["settings"].__setitem__("pairwise_tie_credit", 1.0),
        lambda value: value["summary.json"]["settings"].__setitem__("mrr_no_applicable_contribution", 1.0),
        lambda value: value["summary.json"]["settings"]["role_training"].__setitem__("threads", True),
        lambda value: value["summary.json"]["settings"]["role_training"].__setitem__("row_order", "wrong"),
        lambda value: value["summary.json"]["settings"]["recoverability_inputs"]["details"].__setitem__("path", "/alternate/details.json"),
        lambda value: value["summary.json"]["checkpoint_restoration"]["jepa"].__setitem__("state_key", "wrong"),
        lambda value: value["summary.json"]["counts"].__setitem__("eval_groups", 99),
        lambda value: value["summary.json"]["counts"].__setitem__("within_group_pairs", 99),
        lambda value: value["summary.json"]["counts"].__setitem__("groups_without_applicable", 99),
        lambda value: value["summary.json"]["counts"].__setitem__("groups_without_inapplicable", 99),
        lambda value: value["summary.json"]["counts"].__setitem__("train_role_rows", 1548),
        lambda value: value["summary.json"]["environment"].__setitem__("byteorder", "middle"),
        lambda value: value["summary.json"]["environment"].__setitem__("num_threads", True),
        lambda value: value["summary.json"]["environment"].__setitem__("deterministic_algorithms", 1),
        lambda value: value["role_probe_state.json"].__setitem__("candidate_manifest_sha256", "b" * 64),
    )
    for mutation_index, mutation in enumerate(mutations):
        changed = copy.deepcopy(artifacts)
        mutation(changed)
        try:
            module.validate_ranking_artifacts(changed, manifest_rows=manifest, expected_count=5)
        except ValueError:
            pass
        else:
            pytest.fail(f"ranking artifact mutation survived at index {mutation_index}")


def _accepted_recoverability_fixture(module, variant="baseline"):
    root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability") / variant / "run1"
    artifacts = {
        name: json.loads((root / filename).read_text())
        for name, filename in {
            "summary": "summary.json", "details": "details.json", "feature_schema": "feature_schema.json",
            "split_manifest": "split_manifest.json", "probe_states": "probe_states.json",
        }.items()
    }
    records, _identity = module.load_and_validate_candidate_manifest(module.FIXED_CANDIDATE_MANIFEST)
    checkpoint = module.BASELINE_CHECKPOINT if variant == "baseline" else module.PHASE2_CHECKPOINT
    return artifacts, records, root / "summary.json", checkpoint


def _recoverability_validation_kwargs(module, summary_path, checkpoint):
    return {
        "expected_summary_path": summary_path, "expected_checkpoint": checkpoint,
        "dataset_dir": module.DATASET, "device": "cpu", "split": "val", "seed": 20260717,
    }


RECOVERABILITY_REVIEW_MUTATIONS = (
    ("feature-schema-version", lambda value: value["feature_schema"].__setitem__("schema_version", "wrong")),
    ("feature-schema-candidate-hash", lambda value: value["feature_schema"].__setitem__("candidate_manifest_sha256", "b" * 64)),
    ("probe-candidate-hash", lambda value: value["probe_states"].__setitem__("candidate_manifest_sha256", "b" * 64)),
    ("probe-optimizer", lambda value: value["probe_states"]["training"].__setitem__("optimizer", "SGD")),
    ("summary-epochs", lambda value: value["summary"]["settings"].__setitem__("epochs", 199)),
    ("summary-record-count", lambda value: value["summary"]["counts"].__setitem__("records", 603)),
    ("summary-manifest-path", lambda value: value["summary"]["candidate_manifest"].__setitem__("path", "/wrong")),
    ("train-detail-problem", lambda value: value["details"][0].__setitem__("problem", "wrong")),
    ("mlp-activation", lambda value: value["probe_states"]["models"][1]["architecture"].__setitem__("activation", "sigmoid")),
    ("preprocessing-mean", lambda value: value["probe_states"]["models"][1]["preprocessing"]["mean"].__setitem__(0, value["probe_states"]["models"][1]["preprocessing"]["mean"][0] + 0.25)),
)


@pytest.mark.parametrize("mutation_name,mutate", RECOVERABILITY_REVIEW_MUTATIONS, ids=[item[0] for item in RECOVERABILITY_REVIEW_MUTATIONS])
def test_public_recoverability_validator_kills_review_mutation(mutation_name, mutate) -> None:
    module = _ranking(f"stage0d_recoverability_{mutation_name}")
    artifacts, records, summary_path, checkpoint = _accepted_recoverability_fixture(module)
    module.validate_recoverability_artifacts(
        artifacts, records, **_recoverability_validation_kwargs(module, summary_path, checkpoint),
    )
    changed = copy.deepcopy(artifacts)
    mutate(changed)
    with pytest.raises(ValueError, match="recoverability|probe|feature|manifest|preprocessing|architecture"):
        module.validate_recoverability_artifacts(
            changed, records, **_recoverability_validation_kwargs(module, summary_path, checkpoint),
        )


@pytest.mark.parametrize("variant", ("baseline", "phase2"))
def test_public_recoverability_validator_rejects_coordinated_preprocessing_mean_mutation(variant) -> None:
    module = _ranking(f"stage0d_recoverability_coordinated_mean_{variant}")
    artifacts, records, summary_path, checkpoint = _accepted_recoverability_fixture(module, variant)
    module.validate_recoverability_artifacts(
        artifacts, records, **_recoverability_validation_kwargs(module, summary_path, checkpoint),
    )
    changed = copy.deepcopy(artifacts)
    feature_set = "A_action"
    standardized_index = next(
        schema["standardized_indices"][0]
        for schema in module.recoverability_module.feature_schemas()
        if schema["name"] == feature_set
    )
    changed_models = [
        model for model in changed["probe_states"]["models"]
        if model["feature_set"] == feature_set
    ]
    assert len(changed_models) == 3
    for model in changed_models:
        model["preprocessing"]["mean"][standardized_index] += 0.25
    with pytest.raises(ValueError, match="recoverability preprocessing"):
        module.validate_recoverability_artifacts(
            changed, records, **_recoverability_validation_kwargs(module, summary_path, checkpoint),
        )


def test_private_recoverability_path_runs_public_validator_before_feature_extraction(monkeypatch) -> None:
    module = _ranking("stage0d_recoverability_first_step")
    artifacts, records, summary_path, checkpoint = _accepted_recoverability_fixture(module)
    args = SimpleNamespace(
        recoverability_summary=summary_path, checkpoint=checkpoint,
        dataset_dir=module.DATASET, device="cpu", split="val", seed=20260717,
    )
    monkeypatch.setattr(
        module, "validate_recoverability_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("recursive-validator-first")),
    )
    monkeypatch.setattr(
        module.recoverability_module, "_collect_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("feature-extraction-ran-first")),
    )
    with pytest.raises(RuntimeError, match="recursive-validator-first"):
        module._validate_recoverability_evidence(args, records, artifacts)


def test_extract_candidates_enforces_the_live_causal_boundary(monkeypatch) -> None:
    module = _ranking("stage0d_live_causal_boundary")
    events: list[tuple[str, float]] = []

    class Graph:
        def __init__(self, value): self.value = float(value)
        def to(self, _device): return self

    class States(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            events.append(("successor-read", float(value)))
            return value

    class JEPA:
        def encode(self, graph):
            events.append(("encode", graph.value))
            value = graph.value
            return module.JEPALatentState(
                graph_latent=torch.tensor([[value, value + 0.25]]),
                object_latents=torch.tensor([[value + 1.0, value + 2.0], [value + 3.0, value + 4.0], [value + 5.0, value + 6.0], [value + 7.0, value + 8.0]]),
                object_ids=torch.tensor([7, 3, 5, 6]), object_batch=torch.tensor([0, 0, 0, 0]),
            )

        def action_encoder(self, tensors, state):
            values = tensors["encoded"].to(torch.float32)
            return torch.stack((values + state.graph_latent[:, 0], values * 2), dim=1)

        def predictor(self, state, action):
            events.append(("predict", float(action[0, 0])))
            return module.JEPALatentState(
                graph_latent=state.graph_latent + action[:, :1],
                object_latents=state.object_latents + action[0, 1],
                object_ids=state.object_ids.clone(), object_batch=state.object_batch.clone(),
            )

    class Space:
        @classmethod
        def from_parsed_problem(cls, _parsed): return cls()
        def action_tensors_for_ground_actions(self, actions, *, device):
            del device
            encoded = torch.tensor([float(action.arguments[0]) for action in actions])
            return {
                "encoded": encoded,
                "action_arg_mask": torch.tensor([[True, False], [True, False]]),
                "action_object_indices": torch.tensor([[int(value), -1] for value in encoded]),
            }

    base_records = [
        {"manifest_index": 0, "group": "g", "problem": "p", "step": 0, "action": {"name": "a", "arguments": ["3"]}, "category": "trace", "applicability_label": True},
        {"manifest_index": 1, "group": "g", "problem": "p", "step": 0, "action": {"name": "a", "arguments": ["5"]}, "category": "role_swap", "applicability_label": False},
    ]

    def execute(*, source_value=1.0, successor_value=9.0, records=None):
        events.clear()
        supplied = copy.deepcopy(base_records if records is None else records)
        trajectory = SimpleNamespace(states=States([source_value, successor_value]))
        transition = SimpleNamespace(
            parsed=object(), source_atoms=source_value, records=supplied,
            trajectory=trajectory, successor_index=1,
        )
        monkeypatch.setattr(module, "load_checkpoint_bundle", lambda *a, **k: (object(), object(), SimpleNamespace(jepa=JEPA()), torch.device("cpu"), {"restored": True}))
        monkeypatch.setattr(module, "select_split", lambda *a, **k: object())

        def reconcile(actual, _selected):
            # Synthetic adapter preserves Stage 0C's identity/order rejection while
            # intentionally allowing non-computational label/category metadata.
            def identity(row):
                return row["manifest_index"], row["group"], row["problem"], row["step"]

            if [identity(row) for row in actual] != [identity(row) for row in base_records]:
                raise ValueError("strict transition identity/order drift")
            transition.records = actual
            return [transition]

        monkeypatch.setattr(module, "reconcile_transitions", reconcile)
        monkeypatch.setattr(module, "build_state_graph", lambda _parsed, atoms, include_static: Graph(atoms))
        monkeypatch.setattr(module, "ActionDecodingSpace", Space)
        monkeypatch.setattr(
            module, "transition_components",
            lambda prediction, target, parsed, graph_weight, object_weight: (
                0.0, 0.0, float(torch.sum((prediction.graph_latent - target.graph_latent) ** 2)),
            ),
        )
        args = SimpleNamespace(dataset_dir=Path("data"), checkpoint=Path("checkpoint"), seed=20260717)
        candidates, _ = module._extract_candidates(args, supplied)
        return candidates, list(events)

    def computational(candidate):
        return {
            "graph": candidate.graph_latent.clone(), "action": candidate.action_latent.clone(),
            "object_ids": candidate.object_ids.clone(), "objects": candidate.object_latents.clone(),
            "argument_mask": candidate.argument_mask.clone(), "argument_ids": candidate.argument_object_ids.clone(),
        }

    base, base_events = execute()
    successor_position = next(i for i, item in enumerate(base_events) if item[0] == "successor-read")
    assert [item[0] for item in base_events[:successor_position]].count("predict") == 2

    successor_changed, successor_events = execute(successor_value=12.0)
    assert all(torch.equal(left, right) for a, b in zip(base, successor_changed, strict=True) for left, right in zip(computational(a).values(), computational(b).values(), strict=True))
    assert [item[0] for item in successor_events].index("successor-read") > max(i for i, item in enumerate(successor_events) if item[0] == "predict")
    assert [item.transition_score for item in base] != [item.transition_score for item in successor_changed]

    source_changed, source_events = execute(source_value=2.0)
    assert [item for item in base_events if item[0] == "encode"][-1] == [item for item in source_events if item[0] == "encode"][-1]
    assert not torch.equal(base[0].graph_latent, source_changed[0].graph_latent)
    assert not torch.equal(base[0].object_latents, source_changed[0].object_latents)
    assert not torch.equal(base[0].action_latent, source_changed[0].action_latent)
    assert torch.equal(base[0].argument_object_ids, source_changed[0].argument_object_ids)

    action_records = copy.deepcopy(base_records)
    action_records[1]["action"]["arguments"] = ["6"]
    action_changed, _ = execute(records=action_records)
    assert computational(base[0]).keys() == computational(action_changed[0]).keys()
    assert all(torch.equal(a, b) for a, b in zip(computational(base[0]).values(), computational(action_changed[0]).values(), strict=True))
    assert torch.equal(base[1].graph_latent, action_changed[1].graph_latent)
    assert torch.equal(base[1].object_latents, action_changed[1].object_latents)
    assert not torch.equal(base[1].action_latent, action_changed[1].action_latent)
    assert not torch.equal(base[1].argument_object_ids, action_changed[1].argument_object_ids)

    metadata_records = copy.deepcopy(base_records)
    metadata_records[1].update(category="random_same_schema", applicability_label=True)
    metadata_changed, _ = execute(records=metadata_records)
    assert all(torch.equal(left, right) for a, b in zip(base, metadata_changed, strict=True) for left, right in zip(computational(a).values(), computational(b).values(), strict=True))
    assert [item.transition_score for item in base] == [item.transition_score for item in metadata_changed]

    class BoundaryRoleScorer(torch.nn.Module):
        def forward(self, graph, action, objects, mask, roles):
            query = graph[:, :1] + action[:, :1] + roles.to(torch.float32).unsqueeze(1)
            return (query.unsqueeze(1) * objects[:, :, :1]).sum(-1).masked_fill(~mask, float("-inf"))

    def assembled(candidates):
        recoverability_rows = []
        for candidate in candidates:
            record = candidate.manifest_record
            source_candidate = float(candidate.graph_latent.sum() + candidate.action_latent.sum())
            raw_candidate = float(candidate.action_latent.sum() + candidate.argument_object_ids[candidate.argument_mask].sum())
            hybrid_candidate = float(source_candidate + candidate.object_latents.sum())
            recoverability_rows.append({
                "manifest_index": record["manifest_index"], "group": record["group"], "problem": record["problem"],
                "step": record["step"], "action": record["action"], "category": record["category"],
                "label": record["applicability_label"],
                "logits": {"C_selected_graph_action/mlp": source_candidate, "D_raw_symbolic/mlp": raw_candidate, "E_hybrid/mlp": hybrid_candidate},
            })
        tensors, slices = module.stack_role_candidates(candidates)
        role_scores = module.role_candidate_scores(BoundaryRoleScorer(), tensors, slices)
        score_vectors = module.compose_score_vectors(
            recoverability_rows, role_scores=role_scores,
            transition_scores=[candidate.transition_score for candidate in candidates],
        )
        rows = []
        for recoverability_row, scores in zip(recoverability_rows, score_vectors, strict=True):
            rows.append({**{key: recoverability_row[key] for key in ("manifest_index", "group", "problem", "step", "action", "category", "label")}, "scores": scores})
        ranked = module.rank_details(rows)
        metrics = {name: module.ranking_report(ranked, name) for name in module.SCORERS}
        return ranked, metrics

    base_ranked, base_metrics = assembled(base)
    successor_ranked, _ = assembled(successor_changed)
    for scorer in module.SCORERS[1:]:
        assert [row["scores"][scorer] for row in successor_ranked] == [row["scores"][scorer] for row in base_ranked]
        assert [row["ranks"][scorer] for row in successor_ranked] == [row["ranks"][scorer] for row in base_ranked]
    assert [row["scores"]["latent_transition"] for row in successor_ranked] != [row["scores"]["latent_transition"] for row in base_ranked]

    source_ranked, _ = assembled(source_changed)
    for scorer in ("latent_applicability", "role_object", "raw_symbolic", "hybrid"):
        assert [row["scores"][scorer] for row in source_ranked] != [row["scores"][scorer] for row in base_ranked]
    action_ranked, _ = assembled(action_changed)
    assert action_ranked[0]["scores"] == base_ranked[0]["scores"]
    for scorer in module.SCORERS:
        assert action_ranked[1]["scores"][scorer] != base_ranked[1]["scores"][scorer]

    category_records = copy.deepcopy(base_records)
    category_records[1]["category"] = "random_same_schema"
    category_changed, _ = execute(records=category_records)
    category_ranked, category_metrics = assembled(category_changed)
    assert [row["scores"] for row in category_ranked] == [row["scores"] for row in base_ranked]
    assert [row["ranks"] for row in category_ranked] == [row["ranks"] for row in base_ranked]
    for scorer in module.SCORERS:
        assert category_metrics[scorer]["binary"] == base_metrics[scorer]["binary"]
        assert category_metrics[scorer]["top1_applicable_rate"] == base_metrics[scorer]["top1_applicable_rate"]
        assert category_metrics[scorer]["role_swap_margin"] != base_metrics[scorer]["role_swap_margin"]

    label_records = copy.deepcopy(base_records)
    label_records[1]["applicability_label"] = True
    label_changed, _ = execute(records=label_records)
    label_ranked, label_metrics = assembled(label_changed)
    assert [row["scores"] for row in label_ranked] == [row["scores"] for row in base_ranked]
    assert [row["ranks"] for row in label_ranked] == [row["ranks"] for row in base_ranked]
    assert label_metrics != base_metrics

    for mutant in (list(reversed(base_records)), [{**base_records[0], "group": "other"}, base_records[1]]):
        with pytest.raises(ValueError, match="identity/order"):
            execute(records=mutant)


def test_fresh_subprocess_import_boundary_has_no_forbidden_modules() -> None:
    import subprocess

    code = f'''import sys; sys.path.insert(0, {str(SCRIPT_DIR)!r}); import diagnose_action_candidate_ranking, assess_action_latent_updated_phase0; forbidden={{name for name in sys.modules if name in ("diagnose_action_supervised_probes", "action_diag_common") or name.startswith(("acs_jepa.simulator", "acs_jepa.replay", "acs_jepa_cli.simulator", "acs_jepa_cli.replay"))}}; print(sorted(forbidden)); raise SystemExit(bool(forbidden))'''
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fixed_manifest_proves_candidate_and_active_role_populations() -> None:
    module = _ranking("stage0d_fixed_populations")
    records, _identity = module.load_and_validate_candidate_manifest(module.FIXED_CANDIDATE_MANIFEST)
    train = [row for row in records if row["group"] in module.TRAIN_GROUPS]
    evaluation = [row for row in records if row["group"] in module.EVAL_GROUPS]
    assert (len(train), len(evaluation)) == (453, 151)
    assert (sum(len(row["action"]["arguments"]) for row in train), sum(len(row["action"]["arguments"]) for row in evaluation)) == (1549, 518)
