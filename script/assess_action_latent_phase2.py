#!/usr/bin/env python3
"""Deterministically assess Stage 2 action-latent acceptance evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE = Path("/opt/data/workspace")
RUN = BASE / "acs-jepa-runs/smoke/action_auxiliary_seed0"
PHASE2G = RUN / "phase2g"
REQUIRED_PATHS = {
    "baseline_checkpoint": BASE / "acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt",
    "phase2_checkpoint": RUN / "checkpoints/best.pt",
    "train_metrics": RUN / "metrics/train.jsonl",
    "heldout_metrics": RUN / "metrics/eval.jsonl",
    "resolved_config": RUN / "config.yaml",
    "split_manifest": RUN / "artifacts/split_manifest.json",
    "corpus_manifest": BASE / "acs-jepa-tuning-data/smoke/manifest.json",
    "baseline_statistics": PHASE2G / "baseline/statistics/summary.json",
    "phase2_statistics": PHASE2G / "phase2/statistics/summary.json",
    "baseline_probe": PHASE2G / "baseline/probe_run1/summary.json",
    "baseline_probe_repeat": PHASE2G / "baseline/probe_run2/summary.json",
    "phase2_probe": PHASE2G / "phase2/probe_run1/summary.json",
    "phase2_probe_repeat": PHASE2G / "phase2/probe_run2/summary.json",
    "output": PHASE2G / "assessment",
}
PINNED_HASHES = {
    "baseline_checkpoint": "65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e",
    "phase2_checkpoint": "7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968",
    "train_metrics": "f3e94b0d6c8a38b78ba6ce209f6c0ab31a3cf49bac1c450affcaf60ca30f0e43",
    "heldout_metrics": "5ce3ab7aa535bf68990e000b79c8cd29a1bf14b68670db67eb06a19d5f123954",
    "resolved_config": "01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5",
    "split_manifest": "02aa33b0aa12008142fe08940a16aff81554d7fa3c2345866be6d9b65d9842ae",
    "corpus_manifest": "055b5616d7616331e6edbc8f72523f07e8c1808e5aa31089c8420f01aaf0e400",
}
PROBE_TOP_LEVEL_KEYS = {
    "dataset",
    "checkpoint",
    "split",
    "seed",
    "device",
    "per_category",
    "eval_fraction",
    "epochs",
    "learning_rate",
    "metadata",
    "probe_split",
    "label_counts",
    "category_counts",
    "example_manifest",
    "checkpoint_restoration",
    "probes",
    "checkpoint_applicability_head",
    "checkpoint_argument_reconstruction_head",
    "environment",
    "runtime_seconds",
}
LOSS_KEYS = (
    "action_vicreg_loss",
    "action_contrastive_loss",
    "argument_reconstruction_loss",
    "applicability_loss",
)
COUNT_KEYS = (
    "term/action_vicreg_num_samples",
    "term/action_contrastive_num_examples",
    "term/action_contrastive_num_negatives",
    "term/argument_num_active_roles",
    "term/argument_num_competitive_roles",
    "term/applicability_num_examples",
    "term/applicability_num_positive",
    "term/applicability_num_negative",
)
EXPECTED_LABEL_COUNTS = {"applicable": 62, "inapplicable": 542}
EXPECTED_CATEGORY_COUNTS = {
    "one_arg_substitution": 176,
    "random_other_schema": 176,
    "random_same_schema": 176,
    "role_swap": 32,
    "trace": 44,
}
MANIFEST_SHA256 = "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9"
MANIFEST_BYTES = 117385
MANIFEST_COUNT = 604
DISTRIBUTION_KEYS = {"count", "min", "median", "mean", "max"}
ARGUMENT_ROOT_KEYS = {
    "active_role_count",
    "competitive_role_count",
    "top1_accuracy",
    "chance_accuracy",
    "valid_candidate_count",
    "target_minus_best_wrong_margin",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in REQUIRED_PATHS:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name, approved in REQUIRED_PATHS.items():
        actual = getattr(args, name)
        if actual != approved:
            raise ValueError(f"--{name.replace('_', '-')} must use approved path {approved}, got {actual}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _loads(text: str, *, source: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_pairs_without_duplicates, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed JSON in {source}: {exc}") from exc
    validate_finite(value, source)
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = _loads(path.read_text(encoding="utf-8"), source=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"empty JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line {line_number}: {path}")
        value = _loads(line, source=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def validate_finite(value: Any, pointer: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite value at {pointer}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            validate_finite(child, f"{pointer}/{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            validate_finite(child, f"{pointer}/{index}")
        return
    raise ValueError(f"unsupported JSON value at {pointer}: {type(value).__name__}")


def decision_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    if set(summary) != PROBE_TOP_LEVEL_KEYS:
        missing = sorted(PROBE_TOP_LEVEL_KEYS - set(summary))
        extra = sorted(set(summary) - PROBE_TOP_LEVEL_KEYS)
        raise ValueError(f"invalid probe top-level keys; missing={missing}, extra={extra}")
    projected = copy.deepcopy(dict(summary))
    for key in ("dataset", "checkpoint", "device", "runtime_seconds", "environment"):
        del projected[key]
    manifest = projected.get("example_manifest")
    if not isinstance(manifest, dict) or "path" not in manifest:
        raise ValueError("example_manifest.path is required")
    del manifest["path"]
    validate_finite(projected, "decision_projection")
    return projected


def decision_projection_bytes(summary: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(decision_projection(summary), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _gate(reasons: list[str]) -> dict[str, Any]:
    return {"pass": not reasons, "reasons": reasons}


def _require_number(value: Any, pointer: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected number at {pointer}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite value at {pointer}")
    return number


def _path(root: Mapping[str, Any], *keys: str) -> Any:
    value: Any = root
    traversed = "$"
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing required value at {traversed}/{key}")
        value = value[key]
        traversed += f"/{key}"
    return value


def gate_g2(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("held-out metrics require at least two records")
    steps = [_require_number(row.get("step"), f"heldout/{index}/step") for index, row in enumerate(rows)]
    monotonic = all(second > first for first, second in zip(steps, steps[1:]))
    reasons: list[str] = []
    if not monotonic:
        reasons.append("held-out step must be strictly increasing")
    first, final = rows[0], rows[-1]
    expected = set(LOSS_KEYS) | set(COUNT_KEYS)
    for label, record in (("first", first), ("final", final)):
        present_losses = set(record) & set(LOSS_KEYS)
        if present_losses != set(LOSS_KEYS):
            reasons.append(f"{label} record must contain exact loss keys")
        present_counts = set(record) & set(COUNT_KEYS)
        if present_counts != set(COUNT_KEYS):
            reasons.append(f"{label} record must contain exact count keys")
    relative: dict[str, float] = {}
    if expected <= set(first) and expected <= set(final):
        for key in LOSS_KEYS:
            initial = _require_number(first[key], f"first/{key}")
            ending = _require_number(final[key], f"final/{key}")
            if initial <= 0.0:
                reasons.append(f"first {key} must be positive")
            elif ending > initial:
                reasons.append(f"{key} must be non-increasing")
            else:
                relative[key] = (initial - ending) / abs(initial)
        for label, record in (("first", first), ("final", final)):
            for key in COUNT_KEYS:
                if _require_number(record[key], f"{label}/{key}") <= 0.0:
                    reasons.append(f"{label} effective count {key} must be positive")
        if relative and max(relative.values()) < 0.01:
            reasons.append("at least one relative decrease must be >= 0.01")
    metrics = {"first_step": steps[0], "final_step": steps[-1], "relative_decrease": relative}
    return _gate(reasons), metrics


def gate_g3(baseline: Mapping[str, Any], phase2: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "effective rank": (_path(baseline, "global", "effective_rank"), _path(phase2, "global", "effective_rank"), 4.0),
        "global std minimum": (_path(baseline, "global", "std_min"), _path(phase2, "global", "std_min"), 0.02),
        "within-schema variance fraction": (
            _path(baseline, "schema_argument_variance", "within_fraction"),
            _path(phase2, "schema_argument_variance", "within_fraction"),
            0.001,
        ),
    }
    reasons = []
    for name, (base, phase, threshold) in values.items():
        base_value, phase_value = _require_number(base, name), _require_number(phase, name)
        if phase_value < threshold:
            reasons.append(f"{name} {phase_value} < {threshold}")
        if phase_value < base_value:
            reasons.append(f"{name} is below rerun baseline")
    return _gate(reasons)


def _margin_root(stats: Mapping[str, Any], scale: str) -> Mapping[str, Any]:
    margins = _path(stats, "reference_same_schema_margins")
    if not isinstance(margins, Mapping):
        raise ValueError("reference_same_schema_margins must be an object")
    if scale == "raw_l2" and "raw_l2" not in margins:
        root = margins
    else:
        root = _path(margins, scale)
    if not isinstance(root, Mapping):
        raise ValueError(f"reference_same_schema_margins.{scale} must be an object")
    return root


def gate_g4(baseline: Mapping[str, Any], phase2: Mapping[str, Any]) -> dict[str, Any]:
    base_unit, phase_unit = _margin_root(baseline, "unit_l2"), _margin_root(phase2, "unit_l2")
    phase_raw = _margin_root(phase2, "raw_l2")
    bu_med = _require_number(_path(base_unit, "nearest_wrong_distance_median"), "baseline unit median")
    pu_med = _require_number(_path(phase_unit, "nearest_wrong_distance_median"), "phase2 unit median")
    bu_min = _require_number(_path(base_unit, "nearest_wrong_distance_min"), "baseline unit minimum")
    pu_min = _require_number(_path(phase_unit, "nearest_wrong_distance_min"), "phase2 unit minimum")
    raw_med = _require_number(_path(phase_raw, "nearest_wrong_distance_median"), "phase2 raw median")
    raw_min = _require_number(_path(phase_raw, "nearest_wrong_distance_min"), "phase2 raw minimum")
    reasons = []
    if pu_med < 1.5 * bu_med:
        reasons.append("unit-normalized median is below 1.5x baseline")
    if pu_min < bu_min:
        reasons.append("unit-normalized minimum is below baseline")
    if raw_med < 1.793738847e-4:
        reasons.append("raw L2 median is below 1.793738847e-04")
    if raw_min < 3.323301644e-5:
        reasons.append("raw L2 minimum is below 3.323301644e-05")
    for scale, root in (("raw_l2", phase_raw), ("unit_l2", phase_unit)):
        if _require_number(_path(root, "count"), f"{scale} count") != 44:
            reasons.append(f"{scale} must contain 44 wrong-schema comparisons")
        if _require_number(_path(root, "skipped_no_wrong_count"), f"{scale} skipped") != 0:
            reasons.append(f"{scale} has references without a wrong same-schema comparison")
    return _gate(reasons)


def gate_g5(baseline: Mapping[str, Any], phase2: Mapping[str, Any]) -> dict[str, Any]:
    base_role = _path(baseline, "probes", "role_object", "eval_metrics")
    phase_role = _path(phase2, "probes", "role_object", "eval_metrics")
    role_accuracy = _require_number(_path(phase_role, "accuracy"), "role/object eval accuracy")
    base_per_role = _path(base_role, "per_role_accuracy")
    phase_per_role = _path(phase_role, "per_role_accuracy")
    if not isinstance(base_per_role, Mapping) or not isinstance(phase_per_role, Mapping):
        raise ValueError("per_role_accuracy must be an object")
    improved = sum(
        _require_number(phase_per_role.get(role), f"phase role {role}")
        - _require_number(base_per_role.get(role), f"baseline role {role}")
        >= 0.02
        for role in ("0", "1", "2", "3")
    )
    base_schema = _require_number(_path(baseline, "probes", "schema", "eval_metrics", "accuracy"), "baseline schema")
    phase_schema = _require_number(_path(phase2, "probes", "schema", "eval_metrics", "accuracy"), "phase schema")
    reasons = []
    if role_accuracy < 0.1832046241:
        reasons.append("role/object eval accuracy is below 0.1832046241")
    if improved < 3:
        reasons.append("fewer than three role accuracies improve by 0.02")
    if phase_schema < base_schema - 0.05:
        reasons.append("schema eval accuracy is more than 0.05 below baseline")
    return _gate(reasons)


def _applicability_gate(metrics: Mapping[str, Any], *, baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    reasons = []
    for key, threshold in (("auroc", 0.70), ("average_precision", 0.25), ("f1", 0.25)):
        if _require_number(_path(metrics, key), key) < threshold:
            reasons.append(f"{key} is below {threshold}")
    if _require_number(_path(metrics, "margin", "median"), "overall margin median") <= 0.0:
        reasons.append("overall inapplicable-candidate median margin must be positive")
    for category in ("one_arg_substitution", "random_same_schema", "role_swap"):
        median = _require_number(_path(metrics, "margin_by_category", category, "median"), f"{category} margin")
        if median <= 0.0:
            reasons.append(f"{category} median margin must be positive")
    if baseline is not None:
        for key in ("auroc", "average_precision"):
            if _require_number(_path(metrics, key), key) <= _require_number(_path(baseline, key), f"baseline {key}"):
                reasons.append(f"{key} must exceed rerun baseline")
    return _gate(reasons)


def gate_g6(baseline: Mapping[str, Any], phase2: Mapping[str, Any]) -> dict[str, Any]:
    base = _path(baseline, "probes", "applicability", "eval_metrics")
    phase = _path(phase2, "probes", "applicability", "eval_metrics")
    if not isinstance(base, Mapping) or not isinstance(phase, Mapping):
        raise ValueError("applicability eval metrics must be objects")
    return _applicability_gate(phase, baseline=base)


def gate_g7(baseline: Mapping[str, Any], phase2: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    if baseline.get("checkpoint_applicability_head") is not None:
        reasons.append("baseline checkpoint applicability head must be null")
    head = phase2.get("checkpoint_applicability_head")
    if not isinstance(head, Mapping):
        raise ValueError("Phase 2 checkpoint applicability head evidence is malformed or absent")
    metrics = _path(head, "eval_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("checkpoint applicability eval_metrics must be an object")
    result = _applicability_gate(metrics, baseline=None)
    reasons.extend(result["reasons"])
    return _gate(reasons)


def _validate_argument_root(root: Any, label: str) -> tuple[int, int, float, float, list[str]]:
    if not isinstance(root, Mapping) or set(root) != ARGUMENT_ROOT_KEYS:
        raise ValueError(f"{label} must contain exact argument metric keys")
    active = int(_require_number(root["active_role_count"], f"{label}/active_role_count"))
    competitive = int(_require_number(root["competitive_role_count"], f"{label}/competitive_role_count"))
    reasons = []
    if active < 0 or competitive < 0 or competitive > active:
        reasons.append(f"invalid active/competitive counts at {label}")
    for distribution_name, expected_count in (
        ("valid_candidate_count", active),
        ("target_minus_best_wrong_margin", competitive),
    ):
        distribution = root[distribution_name]
        if not isinstance(distribution, Mapping) or set(distribution) != DISTRIBUTION_KEYS:
            raise ValueError(f"{label}/{distribution_name} must contain exact distribution keys")
        count = int(_require_number(distribution["count"], f"{label}/{distribution_name}/count"))
        if count != expected_count:
            reasons.append(f"{label}/{distribution_name}.count does not reconcile")
        for key in ("min", "median", "mean", "max"):
            value = distribution[key]
            if count == 0:
                if value is not None:
                    reasons.append(f"empty distribution value must be null at {label}/{distribution_name}/{key}")
            else:
                _require_number(value, f"{label}/{distribution_name}/{key}")
    top1 = _require_number(root["top1_accuracy"], f"{label}/top1_accuracy")
    chance = _require_number(root["chance_accuracy"], f"{label}/chance_accuracy")
    return active, competitive, top1, chance, reasons


def gate_g8(baseline: Mapping[str, Any], phase2: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    if baseline.get("checkpoint_argument_reconstruction_head") is not None:
        reasons.append("baseline checkpoint argument head must be null")
    head = phase2.get("checkpoint_argument_reconstruction_head")
    if not isinstance(head, Mapping) or set(head) != {"overall", "per_role"}:
        raise ValueError("Phase 2 argument head must contain exact overall/per_role roots")
    per_role = head["per_role"]
    if not isinstance(per_role, Mapping) or set(per_role) != {"0", "1", "2", "3"}:
        raise ValueError("per_role must contain literal roles 0, 1, 2, and 3")
    overall = _validate_argument_root(head["overall"], "overall")
    roles = [_validate_argument_root(per_role[str(role)], f"per_role/{role}") for role in range(4)]
    reasons.extend(overall[4])
    for role in roles:
        reasons.extend(role[4])
    if overall[0] != sum(role[0] for role in roles):
        reasons.append("overall active_role_count does not equal per-role sum")
    if overall[1] != sum(role[1] for role in roles):
        reasons.append("overall competitive_role_count does not equal per-role sum")
    if any(active <= 0 or competitive <= 0 for active, competitive, _, _, _ in roles):
        reasons.append("all roles must have positive active and competitive counts")
    if overall[2] < 0.25:
        reasons.append("overall top1_accuracy is below 0.25")
    if overall[2] < 2.0 * overall[3]:
        reasons.append("overall top1_accuracy is below 2x chance")
    margin = _require_number(
        _path(head["overall"], "target_minus_best_wrong_margin", "median"), "overall argument margin median"
    )
    if margin <= 0.0:
        reasons.append("overall target-minus-best-wrong median margin must be positive")
    if sum(top1 - chance >= 0.02 for _, _, top1, chance, _ in roles) < 3:
        reasons.append("fewer than three roles exceed chance by 0.02")
    return _gate(reasons)


def _probe_identity(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _path(summary, "metadata", "transitions"),
        _path(summary, "metadata", "examples"),
        summary.get("probe_split"),
        summary.get("label_counts"),
        summary.get("category_counts"),
        {key: value for key, value in _path(summary, "example_manifest").items() if key != "path"},
    )


def validate_probe_artifacts(
    probes: Sequence[tuple[str, Path, Mapping[str, Any]]],
) -> dict[str, Path]:
    """Validate and return every generated artifact accompanying probe summaries."""

    evidence: dict[str, Path] = {}
    for name, summary_path, summary in probes:
        expected_manifest = summary_path.parent / "example_manifest.json"
        expected_details = summary_path.parent / "details.json"
        manifest = _path(summary, "example_manifest")
        if not isinstance(manifest, Mapping) or set(manifest) != {"path", "sha256", "bytes", "count"}:
            raise ValueError(f"{name} has malformed example_manifest metadata")
        if manifest["path"] != str(expected_manifest):
            raise ValueError(f"{name} example manifest path must be {expected_manifest}")
        raw = expected_manifest.read_bytes()
        if len(raw) != MANIFEST_BYTES or sha256_file(expected_manifest) != MANIFEST_SHA256:
            raise ValueError(f"{name} canonical example manifest bytes or SHA-256 mismatch")
        if (
            manifest["bytes"] != len(raw)
            or manifest["sha256"] != MANIFEST_SHA256
            or manifest["count"] != MANIFEST_COUNT
        ):
            raise ValueError(f"{name} declared example manifest identity mismatch")
        records = _loads(raw.decode("utf-8"), source=str(expected_manifest))
        if not isinstance(records, list) or len(records) != MANIFEST_COUNT:
            raise ValueError(f"{name} canonical example manifest must contain {MANIFEST_COUNT} records")
        if _canonical_json(records) != raw:
            raise ValueError(f"{name} example manifest is not canonical JSON")
        load_json(expected_details)
        evidence[f"{name}_example_manifest"] = expected_manifest
        evidence[f"{name}_details"] = expected_details
    return evidence


RESTORATION_KEYS = {
    "jepa": "model_state_dict",
    "goal_head": "goal_head_state_dict",
    "action_contrastive_anchor": "action_contrastive_anchor_state_dict",
    "argument_reconstruction_head": "argument_reconstruction_head_state_dict",
    "applicability_head": "applicability_head_state_dict",
}


def _restoration_is_valid(summary: Mapping[str, Any]) -> bool:
    restoration = summary.get("checkpoint_restoration")
    if not isinstance(restoration, Mapping) or set(restoration) != set(RESTORATION_KEYS):
        return False
    for module, state_key in RESTORATION_KEYS.items():
        state = restoration[module]
        if (
            not isinstance(state, Mapping)
            or set(state) != {"status", "state_key"}
            or state.get("status") not in {"restored", "disabled"}
            or state.get("state_key") != state_key
        ):
            return False
    return restoration["jepa"]["status"] == "restored"


def _check_restoration_status(summary: Mapping[str, Any], label: str, *, baseline: bool) -> list[str]:
    expected = {
        "jepa": "restored",
        "goal_head": "restored",
        "action_contrastive_anchor": "disabled" if baseline else "restored",
        "argument_reconstruction_head": "disabled" if baseline else "restored",
        "applicability_head": "disabled" if baseline else "restored",
    }
    restoration = summary["checkpoint_restoration"]
    return [
        f"{label} {module} status must be {status}"
        for module, status in expected.items()
        if restoration[module]["status"] != status
    ]


def gate_g1(
    baseline_stats: Mapping[str, Any],
    phase2_stats: Mapping[str, Any],
    baseline_probe: Mapping[str, Any],
    baseline_repeat: Mapping[str, Any],
    phase2_probe: Mapping[str, Any],
    phase2_repeat: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    for label, stats in (("baseline", baseline_stats), ("phase2", phase2_stats)):
        if _path(stats, "split") != "val":
            reasons.append(f"{label} statistics split must be val")
        expected_stats_settings = {
            "max_transitions": None,
            "max_candidates_per_state": None,
            "same_schema_only": True,
            "seed": 0,
        }
        for key, expected in expected_stats_settings.items():
            if stats.get(key) != expected:
                reasons.append(f"{label} statistics {key} must be {expected!r}")
        if not _restoration_is_valid(stats):
            raise ValueError(f"{label} statistics have malformed checkpoint restoration evidence")
        reasons.extend(_check_restoration_status(stats, f"{label} statistics", baseline=label == "baseline"))
        if _require_number(_path(stats, "metrics", "transitions"), f"{label} transitions") != 44:
            reasons.append(f"{label} statistics must cover 44 transitions")
        if _require_number(_path(stats, "metrics", "retained_true_action_rate"), f"{label} retained rate") != 1.0:
            reasons.append(f"{label} statistics must retain every true action")
    probes = (baseline_probe, baseline_repeat, phase2_probe, phase2_repeat)
    identities = []
    for index, probe in enumerate(probes):
        decision_projection(probe)
        expected_probe_settings = {
            "split": "val",
            "seed": 20260717,
            "device": "cpu",
            "per_category": 4,
            "eval_fraction": 0.25,
            "epochs": 200,
            "learning_rate": 0.001,
        }
        for key, expected in expected_probe_settings.items():
            if probe.get(key) != expected:
                reasons.append(f"probe {index} {key} must be {expected!r}")
        if _path(probe, "metadata", "transitions") != 44 or _path(probe, "metadata", "examples") != 604:
            reasons.append(f"probe {index} must contain 44 transitions and 604 examples")
        split = _path(probe, "probe_split")
        if _path(split, "train_examples") != 453 or _path(split, "eval_examples") != 151:
            reasons.append(f"probe {index} must have 453/151 example split")
        if probe["label_counts"] != EXPECTED_LABEL_COUNTS or probe["category_counts"] != EXPECTED_CATEGORY_COUNTS:
            reasons.append(f"probe {index} category/label counts mismatch")
        manifest = probe["example_manifest"]
        if (
            _path(manifest, "sha256") != MANIFEST_SHA256
            or _path(manifest, "bytes") != MANIFEST_BYTES
            or _path(manifest, "count") != MANIFEST_COUNT
        ):
            reasons.append(f"probe {index} canonical example manifest mismatch")
        if not _restoration_is_valid(probe):
            raise ValueError(f"probe {index} has malformed checkpoint restoration evidence")
        identities.append(_probe_identity(probe))
    if any(identity != identities[0] for identity in identities[1:]):
        reasons.append("probe identities, split groups, or manifests mismatch")
    if decision_projection_bytes(baseline_probe) != decision_projection_bytes(baseline_repeat):
        reasons.append("baseline probe repeat decision projection mismatch")
    if decision_projection_bytes(phase2_probe) != decision_projection_bytes(phase2_repeat):
        reasons.append("Phase 2 probe repeat decision projection mismatch")
    if baseline_probe["checkpoint_applicability_head"] is not None:
        reasons.append("baseline applicability head must be null")
    if baseline_probe["checkpoint_argument_reconstruction_head"] is not None:
        reasons.append("baseline argument head must be null")
    if phase2_probe["checkpoint_applicability_head"] is None:
        reasons.append("Phase 2 applicability head must be present")
    if phase2_probe["checkpoint_argument_reconstruction_head"] is None:
        reasons.append("Phase 2 argument head must be present")
    reasons.extend(_check_restoration_status(baseline_probe, "baseline probe", baseline=True))
    reasons.extend(_check_restoration_status(phase2_probe, "phase2 probe", baseline=False))
    return _gate(reasons)


def _validate_hashes(paths: Mapping[str, Path]) -> None:
    for name, expected in PINNED_HASHES.items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")


def _validate_source_identity(
    baseline_stats: Mapping[str, Any], phase2_stats: Mapping[str, Any], probes: Sequence[Mapping[str, Any]]
) -> None:
    expected_dataset = str(REQUIRED_PATHS["corpus_manifest"].parent)
    for label, summary, checkpoint in (
        ("baseline statistics", baseline_stats, REQUIRED_PATHS["baseline_checkpoint"]),
        ("phase2 statistics", phase2_stats, REQUIRED_PATHS["phase2_checkpoint"]),
    ):
        if summary.get("checkpoint") != str(checkpoint) or summary.get("dataset") != expected_dataset:
            raise ValueError(f"wrong checkpoint or dataset identity in {label}")
    for index, (probe, checkpoint) in enumerate(
        zip(
            probes,
            (
                REQUIRED_PATHS["baseline_checkpoint"],
                REQUIRED_PATHS["baseline_checkpoint"],
                REQUIRED_PATHS["phase2_checkpoint"],
                REQUIRED_PATHS["phase2_checkpoint"],
            ),
        )
    ):
        if probe.get("checkpoint") != str(checkpoint) or probe.get("dataset") != expected_dataset:
            raise ValueError(f"wrong checkpoint or dataset identity in probe {index}")


def assess(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Path]]:
    _validate_hashes(paths)
    load_jsonl(paths["train_metrics"])
    heldout = load_jsonl(paths["heldout_metrics"])
    baseline_stats = load_json(paths["baseline_statistics"])
    phase2_stats = load_json(paths["phase2_statistics"])
    baseline_probe = load_json(paths["baseline_probe"])
    baseline_repeat = load_json(paths["baseline_probe_repeat"])
    phase2_probe = load_json(paths["phase2_probe"])
    phase2_repeat = load_json(paths["phase2_probe_repeat"])
    probes = (baseline_probe, baseline_repeat, phase2_probe, phase2_repeat)
    probe_artifacts = validate_probe_artifacts(
        [
            ("baseline_probe", paths["baseline_probe"], baseline_probe),
            ("baseline_probe_repeat", paths["baseline_probe_repeat"], baseline_repeat),
            ("phase2_probe", paths["phase2_probe"], phase2_probe),
            ("phase2_probe_repeat", paths["phase2_probe_repeat"], phase2_repeat),
        ]
    )
    _validate_source_identity(baseline_stats, phase2_stats, probes)
    g2, g2_metrics = gate_g2(heldout)
    gates = {
        "G1": gate_g1(baseline_stats, phase2_stats, *probes),
        "G2": g2,
        "G3": gate_g3(baseline_stats, phase2_stats),
        "G4": gate_g4(baseline_stats, phase2_stats),
        "G5": gate_g5(baseline_probe, phase2_probe),
        "G6": gate_g6(baseline_probe, phase2_probe),
        "G7": gate_g7(baseline_probe, phase2_probe),
        "G8": gate_g8(baseline_probe, phase2_probe),
    }
    summary = {
        "decision": "PASS" if all(gate["pass"] for gate in gates.values()) else "FAIL",
        "gates": gates,
        "metrics": {"G2": g2_metrics},
    }
    evidence = {name: path for name, path in paths.items() if name != "output"}
    evidence.update(probe_artifacts)
    return summary, evidence


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_outputs(output: Path, summary: Mapping[str, Any], evidence: Mapping[str, Path]) -> None:
    validate_finite(summary, "summary")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_bytes(_canonical_json(summary))
    lines = [f"# Stage 2G assessment: {summary['decision']}", ""]
    for name in sorted(summary["gates"]):
        gate = summary["gates"][name]
        lines.append(f"- **{name}: {'PASS' if gate['pass'] else 'FAIL'}**")
        lines.extend(f"  - {reason}" for reason in gate["reasons"])
    lines.extend(
        [
            "",
            (
                "Broad tuning may be planned separately; it does not start here."
                if summary["decision"] == "PASS"
                else "Tuning remains paused; use the failed mechanisms to define the smallest next phase."
            ),
            "",
        ]
    )
    markdown_path = output / "summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "evidence": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(evidence.items())
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (summary_path, markdown_path)
        },
    }
    (output / "evidence_manifest.json").write_bytes(_canonical_json(manifest))


def main() -> int:
    try:
        args = build_parser().parse_args()
        validate_args(args)
        paths = {name: getattr(args, name) for name in REQUIRED_PATHS}
        summary, evidence = assess(paths)
        write_outputs(paths["output"], summary, evidence)
        print(summary["decision"])
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
