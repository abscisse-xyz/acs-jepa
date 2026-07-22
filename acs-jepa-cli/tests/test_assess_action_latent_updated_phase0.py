# ruff: noqa: E501, E702, E731
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
ASSESSOR_PATH = SCRIPT_DIR / "assess_action_latent_updated_phase0.py"


def _load(name="stage0d_assessor"):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, ASSESSOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _summary_inputs():
    residual_base = {"metrics": {"state_schema_residual": {"pooled": {"effective_rank": 3.0}}, "raw_variance_decomposition": {"within_schema_fraction": .002}}}
    residual_phase = {"metrics": {"state_schema_residual": {"pooled": {"effective_rank": 4.0}}, "raw_variance_decomposition": {"within_schema_fraction": .003}}}
    feature = lambda good=True: {"eval": {"auroc": .8 if good else .79, "average_precision": .35}, "role_swap_margin": {"median": .1}, "one_arg_substitution_margin": {"median": .1}}
    control = {"eval": {"auroc": .7}}
    recovery = {"metrics": {"features": {key: {"mlp": feature(), "control_mlp": control} for key in ("C_selected_graph_action", "D_raw_symbolic", "E_hybrid")}}}
    transition = {"metrics": {"equivalence_rate": .49, "error_margin": {"median": .01}, "mostly_transition_equivalent": False}}
    ranking = {"metrics": {
        "latent_transition": {"ranks_applicable": True, "deployable": False},
        "latent_applicability": {"ranks_applicable": True, "deployable": True},
        "role_object": {"ranks_applicable": False, "deployable": True},
        "raw_symbolic": {"ranks_applicable": True, "deployable": True},
        "hybrid": {"ranks_applicable": True, "deployable": True},
    }}
    return residual_base, residual_phase, recovery, transition, ranking


def test_nine_decision_booleans_use_exact_paths_and_inclusive_boundaries() -> None:
    module = _load("stage0d_decision_bools")
    baseline, phase2, recovery, transition, ranking = _summary_inputs()
    values = module.compute_decision_booleans(baseline, phase2, recovery, transition, ranking)
    assert values == {
        "representation_ok": True,
        "latent_separable": True,
        "hybrid_separable": True,
        "raw_separable": True,
        "latent_rank": True,
        "raw_rank": True,
        "hybrid_rank": True,
        "mostly_transition_equivalent": False,
        "transition_distinguishable": True,
    }
    transition["metrics"]["mostly_transition_equivalent"] = True
    with pytest.raises(ValueError, match="mostly"):
        module.compute_decision_booleans(baseline, phase2, recovery, transition, ranking)
    transition["metrics"].update(equivalence_rate=.5, mostly_transition_equivalent=True)
    values = module.compute_decision_booleans(baseline, phase2, recovery, transition, ranking)
    assert values["mostly_transition_equivalent"] is True
    assert values["transition_distinguishable"] is False


def test_all_seven_precedence_clauses_and_trace_end_at_first_match() -> None:
    module = _load("stage0d_precedence")
    base = {key: False for key in module.DECISION_BOOLEAN_KEYS}
    cases = [
        ({**base, "raw_separable": False}, "FIX_DATA_LABEL_CONSTRUCTION", 1),
        ({**base, "raw_separable": True, "mostly_transition_equivalent": True}, "BRANCH_D_ABSTRACT_ACTIONS", 2),
        ({**base, "raw_separable": True, "representation_ok": True, "latent_separable": True, "hybrid_separable": True, "latent_rank": True, "transition_distinguishable": True}, "CONTINUE_PHASE1_MINIMAL_SCHEMA_RANK", 3),
        ({**base, "raw_separable": True, "latent_rank": True, "transition_distinguishable": True}, "BRANCH_B_DISCRETE_CANDIDATE_PLANNING", 4),
        ({**base, "raw_separable": True, "hybrid_separable": True, "hybrid_rank": True}, "BRANCH_A_EXPLICIT_STATE_ACTION_SCORER", 5),
        ({**base, "raw_separable": True, "raw_rank": True, "hybrid_rank": False}, "BRANCH_C_STATE_ENCODER_REDESIGN", 6),
        ({**base, "raw_separable": True}, "BRANCH_D_ABSTRACT_ACTIONS", 7),
    ]
    for booleans, expected, clause in cases:
        action, trace = module.select_action(booleans)
        assert action == expected and trace[-1]["clause"] == clause and trace[-1]["matched"] is True
        assert len(trace) == clause


def test_strict_json_loader_rejects_duplicates_nonfinite_noncanonical_and_unknown_keys(tmp_path: Path) -> None:
    module = _load("stage0d_json")
    path = tmp_path / "x.json"
    path.write_bytes(b'{"a":1}\n')
    assert module.load_canonical_json(path, expected_keys={"a"}) == {"a": 1}
    for raw, message in ((b'{"a":1,"a":2}\n', "duplicate"), (b'{"a":NaN}\n', "finite"), (b'{"a":1}', "canonical"), (b'{"a":1,"b":2}\n', "keys")):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match=message):
            module.load_canonical_json(path, expected_keys={"a"})


def test_repeat_projection_deletes_only_six_pointers_and_siblings_are_byte_exact(tmp_path: Path) -> None:
    module = _load("stage0d_repeat")
    first = {"checkpoint": "a", "output": "run1", "device": "cuda:0", "runtime_seconds": 1.0, "environment": {"torch_version": "a", "platform": "a", "num_threads": 1}, "retained": {"x": 1}}
    second = copy.deepcopy(first)
    second.update(checkpoint="b", output="run2", device="cuda:1", runtime_seconds=2.0)
    second["environment"].update(torch_version="b", platform="b")
    assert module.repeat_projection(first) == module.repeat_projection(second)
    second["environment"]["num_threads"] = 2
    assert module.repeat_projection(first) != module.repeat_projection(second)
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    run1.mkdir(); run2.mkdir()
    for root in (run1, run2):
        (root / "summary.json").write_text("{}\n")
        (root / "details.json").write_text("[]\n")
    assert module.validate_sibling_inventory(run1, run2, ("details.json",)) == ["details.json"]
    (run2 / "details.json").write_text("[1]\n")
    with pytest.raises(ValueError, match="repeat"):
        module.validate_sibling_inventory(run1, run2, ("details.json",))
    (run2 / "extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="inventory"):
        module.validate_sibling_inventory(run1, run2, ("details.json",))


def test_independent_score_rank_metric_reconciliation_rejects_each_vector_mutation() -> None:
    module = _load("stage0d_reconcile")
    ranking = module.ranking_module
    base_rows = [
        {"manifest_index": 0, "group": "g", "problem": "p", "step": 0, "action": {"name": "car_start", "arguments": ["a"]}, "category": "trace", "label": True},
        {"manifest_index": 1, "group": "g", "problem": "p", "step": 0, "action": {"name": "car_start", "arguments": ["b"]}, "category": "role_swap", "label": False},
    ]
    names = tuple(ranking.SCORERS)
    vectors = {name: [1.0, 0.0] for name in names}
    rows = copy.deepcopy(base_rows)
    for index, row in enumerate(rows):
        row["scores"] = {name: vectors[name][index] for name in names}
    rows = ranking.rank_details(rows)
    metrics = {name: ranking.ranking_report(rows, name, deployable=name != "latent_transition") for name in names}
    module.reconcile_ranking(rows, metrics, vectors)
    for name in names:
        changed = copy.deepcopy(vectors)
        changed[name][0] += .25
        with pytest.raises(ValueError, match="score"):
            module.reconcile_ranking(rows, metrics, changed)
    hacked = copy.deepcopy(metrics)
    hacked["raw_symbolic"]["binary"]["auroc"] = 0.0
    with pytest.raises(ValueError):
        module.reconcile_ranking(rows, hacked, vectors)


def test_atomic_directory_prevalidates_refuses_existing_and_cleans_failed_staging(tmp_path: Path) -> None:
    module = _load("stage0d_atomic")
    destination = tmp_path / "assessment"
    module.atomic_write_directory(destination, {"summary.json": {"ok": True}, "summary.md": "ok\n"}, validator=lambda root: json.loads((root / "summary.json").read_text()))
    assert destination.is_dir() and sorted(path.name for path in destination.iterdir()) == ["summary.json", "summary.md"]
    with pytest.raises(FileExistsError):
        module.atomic_write_directory(destination, {"summary.json": {}}, validator=lambda root: None)
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="synthetic"):
        module.atomic_write_directory(failed, {"summary.json": {}}, validator=lambda root: (_ for _ in ()).throw(ValueError("synthetic")))
    assert not failed.exists() and not list(tmp_path.glob(".failed.staging-*"))


def test_evidence_manifest_is_sorted_complete_and_excludes_self(tmp_path: Path) -> None:
    module = _load("stage0d_manifest")
    files = []
    for name, role in (("b.json", "diagnostic_summary"), ("a.pt", "checkpoint"), ("summary.md", "assessment_markdown")):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append((path, role))
    manifest = module.build_evidence_manifest(files)
    assert manifest["schema_version"] == "action_latent_updated_phase0.evidence_manifest.v1"
    assert [entry["path"] for entry in manifest["entries"]] == sorted(str(path.resolve()) for path, _ in files)
    assert all(set(entry) == {"path", "bytes", "sha256", "role"} for entry in manifest["entries"])
    with pytest.raises(ValueError, match="self"):
        module.build_evidence_manifest(files + [(tmp_path / "evidence_manifest.json", "diagnostic_summary")])
    module.validate_evidence_manifest(manifest)
    for mutation in (
        lambda value: value["entries"][0].__setitem__("role", "bogus"),
        lambda value: value["entries"][0].__setitem__("sha256", "not-a-hash"),
        lambda value: value["entries"].reverse(),
    ):
        changed = copy.deepcopy(manifest); mutation(changed)
        with pytest.raises(ValueError):
            module.validate_evidence_manifest(changed)


def test_assessor_parser_requires_all_literal_inputs_and_output() -> None:
    module = _load("stage0d_parser")
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    options = {action.dest for action in parser._actions if action.required}
    assert options == set(module.REQUIRED_PATH_ARGUMENTS) | {"output"}


def test_assessor_import_boundary_uses_pure_role_owner_only() -> None:
    source = ASSESSOR_PATH.read_text()
    assert "diagnose_action_supervised_probes" not in source
    assert "action_diag_common" not in source
    assert "applicable_actions(" not in source and "replay_trajectory(" not in source
    before = set(sys.modules)
    module = _load("stage0d_assessor_boundary")
    loaded = set(sys.modules) - before
    assert "diagnose_action_supervised_probes" not in loaded and "action_diag_common" not in loaded
    assert module.RoleObjectProbe.__module__ == "action_role_object_probe"


def test_closed_recursive_stage_validators_reject_nested_unknown_keys_on_real_artifacts() -> None:
    module = _load("stage0d_real_recursive")
    root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0")
    cases = (
        ("schema_residual/baseline/run1", "schema_residual"),
        ("recoverability/baseline/run1", "applicability_recoverability"),
        ("transition_equivalence/baseline/run1", "transition_equivalence"),
    )
    for relative, kind in cases:
        artifacts = module.load_diagnostic_artifacts(root / relative / "summary.json", kind)
        module.validate_diagnostic_artifacts(kind, artifacts)
        changed = copy.deepcopy(artifacts)
        changed["summary.json"]["metrics"]["EXTRA"] = 1
        with pytest.raises(ValueError):
            module.validate_diagnostic_artifacts(kind, changed)
        if kind == "applicability_recoverability":
            for mutator in (
                lambda value: value["summary.json"]["metrics"]["features"]["A_action"]["linear"].__setitem__("EXTRA", 1),
                lambda value: value["probe_states.json"]["models"][0]["preprocessing"].__setitem__("EXTRA", 1),
                lambda value: value["probe_states.json"]["models"][0]["state_dict"][0].__setitem__("dtype", "torch.float64"),
            ):
                changed = copy.deepcopy(artifacts); mutator(changed)
                with pytest.raises(ValueError):
                    module.validate_diagnostic_artifacts(kind, changed)
        if kind == "schema_residual":
            changed = copy.deepcopy(artifacts)
            changed["summary.json"]["metrics"]["state_schema_residual"]["pooled"]["EXTRA"] = 1
            with pytest.raises(ValueError):
                module.validate_diagnostic_artifacts(kind, changed)


def test_assessor_delegates_recoverability_schema_semantics_to_ranking_validator(monkeypatch) -> None:
    module = _load("stage0d_assessor_recoverability_delegate")
    root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1")
    artifacts = module.load_diagnostic_artifacts(root / "summary.json", "applicability_recoverability")
    records, _identity = module.load_and_validate_candidate_manifest(module.ranking_module.FIXED_CANDIDATE_MANIFEST)
    seen = []

    def delegated(actual_artifacts, actual_records, **kwargs):
        seen.append((actual_artifacts, actual_records, kwargs))
        raise RuntimeError("shared-recoverability-validator")

    monkeypatch.setattr(module.ranking_module, "validate_recoverability_artifacts", delegated, raising=False)
    with pytest.raises(RuntimeError, match="shared-recoverability-validator"):
        module._validate_recoverability(artifacts, summary_path=root / "summary.json", records=records)
    assert seen and seen[0][1] is records
    assert seen[0][2] == {
        "expected_summary_path": root / "summary.json",
        "expected_checkpoint": module.ranking_module.BASELINE_CHECKPOINT,
        "dataset_dir": module.ranking_module.DATASET,
        "device": "cpu", "split": "val", "seed": 20260717,
    }


@pytest.mark.parametrize("mutation", ("preprocessing-mean", "mlp-activation"))
def test_assessor_rejects_recoverability_semantic_mutation_before_reconstruction(monkeypatch, mutation) -> None:
    module = _load(f"stage0d_assessor_recoverability_{mutation}")
    root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability/baseline/run1")
    artifacts = module.load_diagnostic_artifacts(root / "summary.json", "applicability_recoverability")
    records, _identity = module.load_and_validate_candidate_manifest(module.ranking_module.FIXED_CANDIDATE_MANIFEST)
    if mutation == "preprocessing-mean":
        artifacts["probe_states.json"]["models"][1]["preprocessing"]["mean"][0] += 0.25
    else:
        artifacts["probe_states.json"]["models"][1]["architecture"]["activation"] = "sigmoid"
    monkeypatch.setattr(
        module.recoverability_module, "reconstruct_probe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reconstruction ran before semantic rejection")),
    )
    with pytest.raises(ValueError):
        module._validate_recoverability(artifacts, summary_path=root / "summary.json", records=records)


def test_assessment_and_manifest_recursive_schemas_and_markdown_are_complete(tmp_path: Path) -> None:
    module = _load("stage0d_assessment_schema")
    baseline, phase2, recovery, transition, ranking = _summary_inputs()
    booleans = module.compute_decision_booleans(baseline, phase2, recovery, transition, ranking)
    selected, trace = module.select_action(booleans)
    summary = module.assessment_schema_fixture(booleans=booleans, selected=selected, precedence=trace, output=tmp_path / "assessment")
    markdown = module.build_summary_markdown(summary)
    assert all(key in markdown for key in module.DECISION_BOOLEAN_KEYS)
    assert all(key in markdown for key in ("evidence", "residual", "recoverability", "transition_equivalence", "ranking"))
    assert markdown.count("diagnose_action_candidate_ranking.py") == 4
    assert markdown.count("assess_action_latent_updated_phase0.py") == 1
    module.validate_assessment_summary(summary)
    changed = copy.deepcopy(summary); changed["decision_booleans"]["latent_rank"] = 1
    with pytest.raises(ValueError):
        module.validate_assessment_summary(changed)


def test_ranking_bound_recoverability_paths_must_equal_assessor_supplied_run1() -> None:
    module = _load("stage0d_bound_paths")
    expected = Path("/fixed/recoverability/baseline/run1/summary.json")
    identities = {name: {"path": str(expected.parent / filename)} for name, filename in {
        "summary": "summary.json", "details": "details.json", "feature_schema": "feature_schema.json",
        "split_manifest": "split_manifest.json", "probe_states": "probe_states.json",
    }.items()}
    module.validate_ranking_recoverability_paths(identities, expected)
    changed = copy.deepcopy(identities); changed["details"]["path"] = "/alternate/details.json"
    with pytest.raises(ValueError):
        module.validate_ranking_recoverability_paths(changed, expected)


def test_ranking_reconstruction_binds_complete_recoverability_namespace(monkeypatch) -> None:
    module = _load("stage0d_ranking_recoverability_namespace")
    root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0")
    ranking_summary = root / "candidate_ranking/baseline/run1/summary.json"
    recoverability_summary = root / "recoverability/baseline/run1/summary.json"

    def validate_namespace(extraction_args, _records, _artifacts):
        assert vars(extraction_args) == {
            "dataset_dir": module.ranking_module.DATASET,
            "checkpoint": module.ranking_module.BASELINE_CHECKPOINT,
            "device": "cpu",
            "split": "val",
            "seed": 20260717,
            "recoverability_summary": recoverability_summary,
        }
        raise RuntimeError("complete-recoverability-namespace")

    monkeypatch.setattr(module.ranking_module, "_validate_recoverability_evidence", validate_namespace)
    with pytest.raises(RuntimeError, match="complete-recoverability-namespace"):
        module._ranking_vectors_from_bound_artifacts(ranking_summary, recoverability_summary)


def test_pair_and_ranking_validators_receive_canonical_source_context(monkeypatch, tmp_path: Path) -> None:
    module = _load("stage0d_context_threading")
    first, second = tmp_path / "run1" / "summary.json", tmp_path / "run2" / "summary.json"
    first.parent.mkdir(); second.parent.mkdir()
    seen = []
    original_validate = module.validate_diagnostic_artifacts
    monkeypatch.setattr(module, "load_canonical_json", lambda path: {"kind": "transition_equivalence"})
    monkeypatch.setattr(module, "load_diagnostic_artifacts", lambda path, kind: {"summary.json": {}, "details.json": []})
    monkeypatch.setattr(module, "validate_diagnostic_artifacts", lambda kind, artifacts, **kwargs: seen.append(kwargs))
    monkeypatch.setattr(module, "repeat_projection", lambda value: {})
    monkeypatch.setattr(module, "validate_sibling_inventory", lambda *args: ["details.json"])
    records, transitions = [{"fixed": True}], [object()]
    module._validate_pair(first, second, ("details.json",), manifest_records=records, reconciled_transitions=transitions)
    assert len(seen) == 2
    assert all(item["manifest_records"] is records and item["reconciled_transitions"] is transitions for item in seen)

    captured = {}
    monkeypatch.setattr(module, "validate_diagnostic_artifacts", original_validate)
    monkeypatch.setattr(module, "_validate_common_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.ranking_module, "validate_ranking_artifacts", lambda artifacts, **kwargs: captured.update(kwargs))
    ranking_artifacts = {"summary.json": {}, "details.json": [], "split_manifest.json": {}, "role_probe_state.json": {}}
    module.validate_diagnostic_artifacts("candidate_ranking", ranking_artifacts, manifest_records=records)
    assert captured == {"manifest_rows": records}


def test_assessment_validator_kills_same_shaped_nested_identity_repeat_metric_and_output_mutants(tmp_path: Path) -> None:
    module = _load("stage0d_assessment_mutants")
    baseline, phase2, recovery, transition, ranking = _summary_inputs()
    booleans = module.compute_decision_booleans(baseline, phase2, recovery, transition, ranking)
    selected, trace = module.select_action(booleans)
    summary = module.assessment_schema_fixture(booleans=booleans, selected=selected, precedence=trace, output=tmp_path / "assessment")
    module.validate_assessment_summary(summary)
    mutations = (
        lambda value: value["compact_metrics"]["ranking_phase2"]["hybrid"].__setitem__("ranks_applicable", 1),
        lambda value: value["input_identities"]["candidate_manifest"].__setitem__("bytes", True),
        lambda value: value["input_identities"]["candidate_manifest"].__setitem__("sha256", "g" * 64),
        lambda value: value["repeatability"]["phase2_ranking"]["files_checked"].pop(),
        lambda value: value.__setitem__("output", "relative/assessment"),
    )
    for mutate in mutations:
        changed = copy.deepcopy(summary); mutate(changed)
        with pytest.raises(ValueError):
            module.validate_assessment_summary(changed)


def test_manifest_expected_roles_and_markdown_exact_commands_are_closed(tmp_path: Path) -> None:
    module = _load("stage0d_manifest_roles")
    path = tmp_path / "summary.json"; path.write_text("{}\n")
    manifest = module.build_evidence_manifest([(path, "diagnostic_summary")])
    expected_path = str(path.resolve())
    module.validate_evidence_manifest(manifest, expected_paths={expected_path}, expected_roles={expected_path: "diagnostic_summary"})
    changed = copy.deepcopy(manifest); changed["entries"][0]["role"] = "checkpoint"
    with pytest.raises(ValueError, match="role"):
        module.validate_evidence_manifest(changed, expected_paths={expected_path}, expected_roles={expected_path: "diagnostic_summary"})

    baseline, phase2, recovery, transition, ranking = _summary_inputs()
    booleans = module.compute_decision_booleans(baseline, phase2, recovery, transition, ranking)
    selected, trace = module.select_action(booleans)
    summary = module.assessment_schema_fixture(booleans=booleans, selected=selected, precedence=trace, output=tmp_path / "assessment")
    markdown = module.build_summary_markdown(summary)
    assert markdown == module.build_summary_markdown(copy.deepcopy(summary))
    assert markdown.count("UV_CACHE_DIR=/opt/data/workspace/.uv-cache") == 5
    assert "**SKIPPED**" in markdown
    assert all(identity["path"] in markdown and identity["sha256"] in markdown for identity in summary["input_identities"].values())


def test_stage0c_source_identity_mutant_only_fails_with_assessor_reconciled_context() -> None:
    module = _load("stage0d_stage0c_context")
    root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/transition_equivalence/baseline/run1")
    artifacts = module.load_diagnostic_artifacts(root / "summary.json", "transition_equivalence")
    records, _identity = module.load_and_validate_candidate_manifest(module.ranking_module.FIXED_CANDIDATE_MANIFEST)
    config, corpus, _bundle, _device, _restoration = module.transition_module.load_checkpoint_bundle(
        module.ranking_module.DATASET, module.ranking_module.BASELINE_CHECKPOINT,
        device_name="cpu", include_restoration_metadata=True,
    )
    selected = module.transition_module.select_split(corpus, config, "val", seed=20260717)
    transitions = module.transition_module.reconcile_transitions(records, selected)
    changed = copy.deepcopy(artifacts)
    module.transition_module.validate_summary(changed["summary.json"], changed["details.json"])
    transitions[0] = dataclasses.replace(transitions[0], trace_action=transitions[1].trace_action)
    with pytest.raises(ValueError, match="source|reconciled|identity|trace|manifest"):
        module.validate_diagnostic_artifacts(
            "transition_equivalence", changed, manifest_records=records, reconciled_transitions=transitions,
        )


def test_reconstruction_derives_every_decision_verdict_and_compact_metric(tmp_path: Path) -> None:
    module = _load("stage0d_reconstruction")
    baseline, phase2, recovery, transition, ranking = _summary_inputs()
    ranking_any: Any = ranking
    for scorer in ranking_any["metrics"].values():
        scorer.update(
            binary={"auroc": .81, "average_precision": .36},
            top1_applicable_rate=.82,
            pairwise_applicable_accuracy=.83,
        )
    booleans = module.compute_decision_booleans(baseline, phase2, recovery, transition, ranking)
    selected, trace = module.select_action(booleans)
    shell = module.assessment_schema_fixture(booleans=booleans, selected=selected, precedence=trace, output=tmp_path / "assessment")
    summaries = {
        "baseline_schema": baseline, "phase2_schema": phase2,
        "phase2_recoverability": recovery, "phase2_transition": transition,
    }
    rebuilt = module.reconstruct_assessment(
        summaries, ranking, ranking, shell["input_identities"], shell["repeatability"], tmp_path / "assessment",
    )
    module.validate_assessment_summary(rebuilt)
    assert rebuilt["decision_booleans"] == booleans
    assert rebuilt["selected_action"] == selected and rebuilt["precedence_trace"] == trace
    assert rebuilt["stage_verdicts"] == {"evidence": "PASS", "residual": "PASS", "recoverability": "PASS", "transition_equivalence": "PASS", "ranking": "PASS"}
    assert rebuilt["compact_metrics"]["ranking_phase2"]["hybrid"]["auroc"] == .81
    for path, replacement in (
        (("decision_booleans", "latent_rank"), False),
        (("stage_verdicts", "ranking"), "FAIL"),
        (("compact_metrics", "latent_auroc"), .79),
        (("input_identities", "candidate_manifest", "sha256"), "b" * 64),
    ):
        changed = copy.deepcopy(rebuilt)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        assert changed != module.reconstruct_assessment(
            summaries, ranking, ranking, shell["input_identities"], shell["repeatability"], tmp_path / "assessment",
        )
