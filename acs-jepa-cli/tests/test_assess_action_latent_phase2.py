from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "script" / "assess_action_latent_phase2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("assess_action_latent_phase2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dist(value: float, count: int = 4) -> dict[str, float | int]:
    return {"count": count, "min": value, "median": value, "mean": value, "max": value}


def _binary(auroc: float, ap: float, f1: float, margin: float) -> dict[str, object]:
    return {
        "count": 100,
        "positive_count": 10,
        "negative_count": 90,
        "accuracy": 0.9,
        "precision": 0.5,
        "recall": 0.5,
        "f1": f1,
        "auroc": auroc,
        "average_precision": ap,
        "positive_prevalence": 0.1,
        "margin": _dist(margin),
        "margin_by_category": {
            "one_arg_substitution": _dist(margin),
            "random_same_schema": _dist(margin),
            "role_swap": _dist(margin),
        },
        "per_category": {},
    }


def _argument_root(top1: float, chance: float, *, active: int = 4, competitive: int = 4) -> dict[str, object]:
    return {
        "active_role_count": active,
        "competitive_role_count": competitive,
        "top1_accuracy": top1,
        "chance_accuracy": chance,
        "valid_candidate_count": _dist(2.0, active),
        "target_minus_best_wrong_margin": _dist(0.5, competitive),
    }


def _probe(*, baseline: bool = False) -> dict[str, object]:
    binary = _binary(0.80 if not baseline else 0.60, 0.35 if not baseline else 0.15, 0.4, 0.3)
    argument = None
    if not baseline:
        roles = {str(role): _argument_root(0.5, 0.2) for role in range(4)}
        argument = {
            "overall": _argument_root(0.5, 0.2, active=16, competitive=16),
            "per_role": roles,
        }
    return {
        "dataset": "/different/runtime/path",
        "checkpoint": "/different/runtime/checkpoint",
        "split": "val",
        "seed": 20260717,
        "device": "cpu",
        "per_category": 4,
        "eval_fraction": 0.25,
        "epochs": 200,
        "learning_rate": 0.001,
        "metadata": {"transitions": 44, "examples": 604},
        "probe_split": {
            "train_groups": ["p166:0"],
            "eval_groups": ["p192:0"],
            "train_examples": 453,
            "eval_examples": 151,
        },
        "label_counts": {"applicable": 62, "inapplicable": 542},
        "category_counts": {
            "one_arg_substitution": 176,
            "random_other_schema": 176,
            "random_same_schema": 176,
            "role_swap": 32,
            "trace": 44,
        },
        "example_manifest": {
            "path": "/different/runtime/manifest.json",
            "sha256": "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9",
            "bytes": 117385,
            "count": 604,
        },
        "checkpoint_restoration": {
            "jepa": {"status": "restored", "state_key": "model_state_dict"},
            "goal_head": {"status": "restored", "state_key": "goal_head_state_dict"},
            "action_contrastive_anchor": {
                "status": "disabled" if baseline else "restored",
                "state_key": "action_contrastive_anchor_state_dict",
            },
            "applicability_head": {
                "status": "disabled" if baseline else "restored",
                "state_key": "applicability_head_state_dict",
            },
            "argument_reconstruction_head": {
                "status": "disabled" if baseline else "restored",
                "state_key": "argument_reconstruction_head_state_dict",
            },
        },
        "probes": {
            "schema": {"train_metrics": {"accuracy": 0.8}, "eval_metrics": {"accuracy": 0.60}},
            "role_object": {
                "train_metrics": {"accuracy": 0.8, "per_role_accuracy": {str(i): 0.5 for i in range(4)}},
                "eval_metrics": {
                    "accuracy": 0.30 if not baseline else 0.10,
                    "per_role_accuracy": {str(i): 0.30 if not baseline else 0.10 for i in range(4)},
                },
            },
            "applicability": {"train_metrics": binary, "eval_metrics": binary},
        },
        "checkpoint_applicability_head": None
        if baseline
        else {"train_metrics": binary, "eval_metrics": binary},
        "checkpoint_argument_reconstruction_head": argument,
        "environment": {"python": "3.x"},
        "runtime_seconds": 1.0,
    }


def _stats(*, baseline: bool = False) -> dict[str, object]:
    factor = 1.0 if baseline else 2.0
    return {
        "checkpoint": "ignored here",
        "dataset": "ignored here",
        "split": "val",
        "max_transitions": None,
        "max_candidates_per_state": None,
        "same_schema_only": True,
        "seed": 0,
        "checkpoint_restoration": copy.deepcopy(_probe(baseline=baseline)["checkpoint_restoration"]),
        "metrics": {"transitions": 44, "retained_true_action_rate": 1.0},
        "global": {"effective_rank": 4.5 if not baseline else 2.0, "std_min": 0.03 if not baseline else 0.01},
        "schema_argument_variance": {"within_fraction": 0.002 if not baseline else 0.0001},
        "reference_same_schema_margins": {
            "raw_l2": {
                "count": 44,
                "skipped_no_wrong_count": 0,
                "nearest_wrong_distance_min": 0.00004,
                "nearest_wrong_distance_median": 0.0002,
            },
            "unit_l2": {
                "count": 44,
                "skipped_no_wrong_count": 0,
                "nearest_wrong_distance_min": 0.1 * factor,
                "nearest_wrong_distance_median": 0.2 * factor,
            },
        },
    }


def _heldout() -> list[dict[str, float | int]]:
    losses = [
        "action_vicreg_loss",
        "action_contrastive_loss",
        "argument_reconstruction_loss",
        "applicability_loss",
    ]
    counts = [
        "term/action_vicreg_num_samples",
        "term/action_contrastive_num_examples",
        "term/action_contrastive_num_negatives",
        "term/argument_num_active_roles",
        "term/argument_num_competitive_roles",
        "term/applicability_num_examples",
        "term/applicability_num_positive",
        "term/applicability_num_negative",
    ]
    first: dict[str, float | int] = {"step": 1, **{key: 1.0 for key in losses}, **{key: 1 for key in counts}}
    final: dict[str, float | int] = {"step": 2, **{key: 0.9 for key in losses}, **{key: 1 for key in counts}}
    return [first, final]


def test_decision_projection_has_exact_schema_and_six_exclusions() -> None:
    assessor = _load_module()
    probe = _probe()
    projected = assessor.decision_projection(probe)
    assert set(projected) == assessor.PROBE_TOP_LEVEL_KEYS - {
        "dataset",
        "checkpoint",
        "device",
        "runtime_seconds",
        "environment",
    }
    assert "path" not in projected["example_manifest"]

    for pointer, replacement in [
        (("dataset",), "new"),
        (("checkpoint",), "new"),
        (("device",), "cuda"),
        (("runtime_seconds",), 999.0),
        (("environment",), {"new": True}),
        (("example_manifest", "path"), "new"),
    ]:
        changed = copy.deepcopy(probe)
        target = changed
        for key in pointer[:-1]:
            target = target[key]  # type: ignore[index]
        target[pointer[-1]] = replacement  # type: ignore[index]
        assert assessor.decision_projection_bytes(changed) == assessor.decision_projection_bytes(probe)


def test_decision_projection_rejects_unknown_top_level_and_retains_nested_mutations() -> None:
    assessor = _load_module()
    probe = _probe()
    changed = copy.deepcopy(probe)
    changed["probes"]["applicability"]["eval_metrics"]["new_metric"] = 1  # type: ignore[index]
    assert assessor.decision_projection_bytes(changed) != assessor.decision_projection_bytes(probe)
    probe["unknown"] = 1
    with pytest.raises(ValueError, match="top-level keys"):
        assessor.decision_projection(probe)


def test_g2_passes_and_uses_monotonic_endpoint_steps() -> None:
    assessor = _load_module()
    gate, metrics = assessor.gate_g2(_heldout())
    assert gate["pass"] is True
    assert metrics["first_step"] == 1
    assert metrics["final_step"] == 2
    assert metrics["relative_decrease"]["action_vicreg_loss"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.reverse(), "strictly increasing"),
        (lambda rows: rows[-1].pop("applicability_loss"), "exact loss keys"),
        (lambda rows: rows[-1].update({"term/applicability_num_negative": 0}), "positive"),
        (lambda rows: rows[-1].update({"action_vicreg_loss": 1.1}), "non-increasing"),
        (lambda rows: [row.update({key: 0.995 for key in assessor_loss_keys()}) for row in rows[-1:]], "0.01"),
    ],
)
def test_g2_fails_each_activity_contract(mutation, message: str) -> None:
    assessor = _load_module()
    rows = _heldout()
    mutation(rows)
    gate, _ = assessor.gate_g2(rows)
    assert gate["pass"] is False
    assert message in " ".join(gate["reasons"])


def assessor_loss_keys() -> tuple[str, ...]:
    return (
        "action_vicreg_loss",
        "action_contrastive_loss",
        "argument_reconstruction_loss",
        "applicability_loss",
    )


def test_g3_through_g8_pass_and_each_can_fail() -> None:
    assessor = _load_module()
    baseline_stats, phase_stats = _stats(baseline=True), _stats()
    baseline_probe, phase_probe = _probe(baseline=True), _probe()
    gates = [
        assessor.gate_g3(baseline_stats, phase_stats),
        assessor.gate_g4(baseline_stats, phase_stats),
        assessor.gate_g5(baseline_probe, phase_probe),
        assessor.gate_g6(baseline_probe, phase_probe),
        assessor.gate_g7(baseline_probe, phase_probe),
        assessor.gate_g8(baseline_probe, phase_probe),
    ]
    assert all(gate["pass"] for gate in gates)

    failures = []
    bad = copy.deepcopy(phase_stats)
    bad["global"]["effective_rank"] = 3.9  # type: ignore[index]
    failures.append(assessor.gate_g3(baseline_stats, bad))
    bad = copy.deepcopy(phase_stats)
    bad["reference_same_schema_margins"]["raw_l2"]["nearest_wrong_distance_median"] = 0.0  # type: ignore[index]
    failures.append(assessor.gate_g4(baseline_stats, bad))
    bad = copy.deepcopy(phase_probe)
    bad["probes"]["role_object"]["eval_metrics"]["accuracy"] = 0.1  # type: ignore[index]
    failures.append(assessor.gate_g5(baseline_probe, bad))
    bad = copy.deepcopy(phase_probe)
    bad["probes"]["applicability"]["eval_metrics"]["auroc"] = 0.69  # type: ignore[index]
    failures.append(assessor.gate_g6(baseline_probe, bad))
    bad = copy.deepcopy(phase_probe)
    bad["checkpoint_applicability_head"]["eval_metrics"]["average_precision"] = 0.24  # type: ignore[index]
    failures.append(assessor.gate_g7(baseline_probe, bad))
    bad = copy.deepcopy(phase_probe)
    bad["checkpoint_argument_reconstruction_head"]["per_role"]["3"]["competitive_role_count"] = 0  # type: ignore[index]
    failures.append(assessor.gate_g8(baseline_probe, bad))
    assert all(gate["pass"] is False for gate in failures)


def test_g8_rejects_schema_count_mismatch_and_nonfinite() -> None:
    assessor = _load_module()
    phase = _probe()
    phase["checkpoint_argument_reconstruction_head"]["overall"]["active_role_count"] = 15  # type: ignore[index]
    assert assessor.gate_g8(_probe(baseline=True), phase)["pass"] is False
    phase = _probe()
    phase["checkpoint_argument_reconstruction_head"]["overall"]["top1_accuracy"] = float("nan")  # type: ignore[index]
    with pytest.raises(ValueError, match="nonfinite"):
        assessor.gate_g8(_probe(baseline=True), phase)


def test_g1_validates_identity_repeats_and_manifest() -> None:
    assessor = _load_module()
    baseline = _probe(baseline=True)
    phase = _probe()
    gate = assessor.gate_g1(
        _stats(baseline=True),
        _stats(),
        baseline,
        copy.deepcopy(baseline),
        phase,
        copy.deepcopy(phase),
    )
    assert gate["pass"] is True
    repeat = copy.deepcopy(phase)
    repeat["metadata"]["new"] = 1  # type: ignore[index]
    assert assessor.gate_g1(_stats(baseline=True), _stats(), baseline, baseline, phase, repeat)["pass"] is False
    wrong_settings = _probe()
    wrong_settings["seed"] = 0
    assert assessor.gate_g1(
        _stats(baseline=True), _stats(), baseline, baseline, wrong_settings, copy.deepcopy(wrong_settings)
    )["pass"] is False


def test_g1_rejects_malformed_restoration_evidence() -> None:
    assessor = _load_module()
    phase = _probe()
    phase["checkpoint_restoration"]["jepa"].pop("state_key")  # type: ignore[index]
    with pytest.raises(ValueError, match="malformed checkpoint restoration"):
        assessor.gate_g1(
            _stats(baseline=True),
            _stats(),
            _probe(baseline=True),
            _probe(baseline=True),
            phase,
            copy.deepcopy(phase),
        )


def test_g1_requires_every_configured_phase2_module_to_be_restored() -> None:
    assessor = _load_module()
    baseline = _probe(baseline=True)
    phase = _probe()
    phase_stats = _stats()
    phase_stats["checkpoint_restoration"]["action_contrastive_anchor"]["status"] = "disabled"  # type: ignore[index]
    gate = assessor.gate_g1(
        _stats(baseline=True),
        phase_stats,
        baseline,
        copy.deepcopy(baseline),
        phase,
        copy.deepcopy(phase),
    )
    assert gate["pass"] is False
    assert "action_contrastive_anchor status must be restored" in " ".join(gate["reasons"])


def test_probe_artifact_validation_hashes_declared_manifest_and_details(tmp_path: Path, monkeypatch) -> None:
    assessor = _load_module()
    run = tmp_path / "probe"
    run.mkdir()
    summary_path = run / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    manifest_path = run / "example_manifest.json"
    manifest_path.write_text('[{"example":1}]\n', encoding="utf-8")
    details_path = run / "details.json"
    details_path.write_text('{"probe_split":{}}\n', encoding="utf-8")
    probe = _probe()
    probe["example_manifest"] = {
        "path": str(manifest_path),
        "sha256": assessor.sha256_file(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "count": 1,
    }
    monkeypatch.setattr(assessor, "MANIFEST_SHA256", assessor.sha256_file(manifest_path))
    monkeypatch.setattr(assessor, "MANIFEST_BYTES", manifest_path.stat().st_size)
    monkeypatch.setattr(assessor, "MANIFEST_COUNT", 1)

    evidence = assessor.validate_probe_artifacts([("baseline_probe", summary_path, probe)])
    assert evidence == {
        "baseline_probe_details": details_path,
        "baseline_probe_example_manifest": manifest_path,
    }

    probe["example_manifest"]["path"] = str(tmp_path / "missing.json")  # type: ignore[index]
    with pytest.raises(ValueError, match="manifest path"):
        assessor.validate_probe_artifacts([("baseline_probe", summary_path, probe)])


def test_jsonl_and_json_loaders_reject_malformed_nonfinite_and_duplicate_keys(tmp_path: Path) -> None:
    assessor = _load_module()
    path = tmp_path / "bad.jsonl"
    path.write_text('{"step": 1, "step": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        assessor.load_jsonl(path)
    path.write_text('{"step": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="constant"):
        assessor.load_jsonl(path)
    path.write_text("{}\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank"):
        assessor.load_jsonl(path)


def test_write_outputs_hashes_all_evidence_and_is_compact(tmp_path: Path) -> None:
    assessor = _load_module()
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"details": [1, 2, 3]}\n', encoding="utf-8")
    output = tmp_path / "assessment"
    summary = {"decision": "FAIL", "gates": {"G1": {"pass": False, "reasons": ["x"]}}, "metrics": {}}
    assessor.write_outputs(output, summary, {"probe": evidence})
    written = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert written == summary
    assert set(manifest["evidence"]) == {"probe"}
    assert set(manifest["outputs"]) == {"summary.json", "summary.md"}
    assert manifest["evidence"]["probe"]["sha256"] == assessor.sha256_file(evidence)
    assert "details" not in (output / "summary.json").read_text(encoding="utf-8")


def test_validate_args_requires_literal_approved_paths() -> None:
    assessor = _load_module()
    parser = assessor.build_parser()
    values: list[str] = []
    for option, path in assessor.REQUIRED_PATHS.items():
        values.extend([f"--{option.replace('_', '-')}", str(path)])
    args = parser.parse_args(values)
    assessor.validate_args(args)
    args.split_manifest = Path("relative.json")
    with pytest.raises(ValueError, match="approved path"):
        assessor.validate_args(args)
