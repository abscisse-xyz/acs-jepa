"""Stage 0C hard-negative transition-equivalence diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa.graph import GroundAction, build_state_graph
from action_phase0_common import (
    GROUPS,
    SPLIT_SHA256,
    canonical_json_bytes,
    file_identity,
    load_and_validate_candidate_manifest,
    load_checkpoint_bundle,
    prepare_output_directory,
    select_split,
)

CANDIDATE_CATEGORIES = ("one_arg_substitution", "role_swap", "random_same_schema")
UPDATED_SPEC = Path("/opt/data/workspace/acs-jepa/script/ACTION_LATENT_UPDATED_SPEC.md")
BASELINE_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt")
BASELINE_CONFIG = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml")
PHASE2_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt")
PHASE2_CONFIG = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml")
CORPUS_MANIFEST = Path("/opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json")
DATASET = Path("/opt/data/workspace/acs-jepa-tuning-data/smoke")
FIXED_CANDIDATE_MANIFEST = Path(
    "/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json"
)
FIXED_SHA256 = {
    UPDATED_SPEC: "b4146d21b6082ec085628f7d1c56ff135c9fe606c8307db8b84689e449ec9606",
    BASELINE_CHECKPOINT: "65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e",
    BASELINE_CONFIG: "f65e2cbb33fb3e7322e0cc0c5e8a8f01e9ca7c408e4594516d50a9735c673193",
    PHASE2_CHECKPOINT: "7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968",
    PHASE2_CONFIG: "01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5",
    CORPUS_MANIFEST: "055b5616d7616331e6edbc8f72523f07e8c1808e5aa31089c8420f01aaf0e400",
}
NULLABLE_DETAIL_KEYS = (
    "wrong_action",
    "wrong_category",
    "wrong_unit_action_l2",
    "true_graph_error",
    "true_object_error",
    "true_total_error",
    "wrong_graph_error",
    "wrong_object_error",
    "wrong_total_error",
    "prediction_graph_separation",
    "prediction_object_separation",
    "prediction_separation",
    "error_ratio",
    "error_margin",
    "separation_ratio",
    "transition_equivalent",
)
DETAIL_KEYS = frozenset(("group", "problem", "step", "trace_action", "status", "skip_reason", *NULLABLE_DETAIL_KEYS))
SUMMARY_KEYS = frozenset(
    {
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
)


def _action_key(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    action = row["action"]
    return str(action["name"]), tuple(action["arguments"])


def filter_hard_negatives(records: Sequence[Mapping[str, Any]], trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Retain eligible wrong records in immutable manifest order."""

    trace_schema = trace["action"]["name"]
    selected = [
        row
        for row in records
        if row["applicability_label"] is False
        and row["action"]["name"] == trace_schema
        and row["category"] in CANDIDATE_CATEGORIES
    ]
    keys = [_action_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("eligible hard-negative action keys must be unique")
    return selected


@dataclass(frozen=True)
class ReconciledTransition:
    group: str
    problem: str
    step: int
    parsed: Any
    source_atoms: Any
    trace_action: Any
    records: tuple[Mapping[str, Any], ...]
    trajectory: Any
    successor_index: int


def reconcile_transitions(records: Sequence[Mapping[str, Any]], corpus: Any) -> list[ReconciledTransition]:
    """Validate strict trajectory identity while deferring successor value access."""

    by_problem: dict[str, list[Any]] = defaultdict(list)
    for trajectory, record in zip(corpus.trajectories, corpus.records, strict=True):
        by_problem[record.problem_name].append(trajectory)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["group"])].append(row)
    output = []
    group_order = GROUPS if set(grouped) == set(GROUPS) else tuple(grouped)
    for group in group_order:
        rows = grouped[group]
        problem, step = str(rows[0]["problem"]), int(rows[0]["step"])
        matches = [trajectory for trajectory in by_problem.get(problem, ()) if 0 <= step < len(trajectory.actions)]
        if len(matches) != 1:
            raise ValueError(f"group {group} does not map to exactly one strict corpus transition")
        trajectory = matches[0]
        if len(trajectory.states) != len(trajectory.actions) + 1:
            raise ValueError("trajectory states must equal actions plus one")
        if step + 1 >= len(trajectory.states):
            raise ValueError("trajectory has no exact successor state")
        traces = [row for row in rows if row["category"] == "trace" and row["applicability_label"] is True]
        actual = (trajectory.actions[step].name, tuple(trajectory.actions[step].arguments))
        if len(traces) != 1 or _action_key(traces[0]) != actual:
            raise ValueError(f"group {group} trace action does not match strict trajectory")
        output.append(
            ReconciledTransition(
                group=group,
                problem=problem,
                step=step,
                parsed=corpus.parsed_problems[trajectory.problem_index],
                source_atoms=trajectory.states[step],
                trace_action=trajectory.actions[step],
                records=tuple(rows),
                trajectory=trajectory,
                successor_index=step + 1,
            )
        )
    return output


def _distance_copy(value: torch.Tensor) -> torch.Tensor:
    copied = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if copied.shape != (64,):
        raise ValueError("action latent must have exactly 64 coordinates")
    if not bool(torch.isfinite(copied).all()):
        raise ValueError("action latent coordinates must be finite")
    return copied


def _unit(value: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(value * value))
    return value / norm if float(norm) > 0.0 else torch.zeros_like(value)


def select_nearest_wrong(
    true_native: torch.Tensor,
    candidates: Sequence[tuple[Mapping[str, Any], torch.Tensor]],
) -> tuple[Mapping[str, Any], float, list[torch.Tensor]]:
    """Select by exact CPU-float64 unit L2 without mutating native latents."""

    true_copy = _distance_copy(true_native)
    candidate_copies = [_distance_copy(value) for _, value in candidates]
    rows = []
    for (record, _native), copied in zip(candidates, candidate_copies, strict=True):
        distance = float(torch.sqrt(torch.sum((_unit(copied) - _unit(true_copy)) ** 2)))
        if not torch.isfinite(torch.tensor(distance)):
            raise ValueError("action distance must be finite")
        rows.append((distance, _action_key(record), record))
    if not rows:
        raise ValueError("nearest-wrong selection requires a candidate")
    distance, _key, selected = min(rows, key=lambda item: (item[0], item[1]))
    return selected, distance, [true_copy, *candidate_copies]


def _validated_state_metadata(state: Any, parsed: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(state.graph_latent.shape) != (1, 64):
        raise ValueError("graph latent shape must be exactly [1,64]")
    if state.object_latents.ndim != 2 or state.object_latents.shape[1] != 64 or state.object_latents.shape[0] == 0:
        raise ValueError("object latent shape must be nonempty [N,64]")
    if not bool(torch.isfinite(state.graph_latent).all()) or not bool(torch.isfinite(state.object_latents).all()):
        raise ValueError("state latents must be finite")
    ids = state.object_ids.detach().to(device="cpu", dtype=torch.long)
    batch = state.object_batch.detach().to(device="cpu", dtype=torch.long)
    canonical = torch.tensor([parsed.object_to_id[name] for name in sorted(parsed.objects)], dtype=torch.long)
    if not torch.equal(ids, canonical):
        raise ValueError("object IDs do not match canonical sorted problem IDs")
    if not torch.equal(batch, torch.zeros(len(canonical), dtype=torch.long)):
        raise ValueError("object batch metadata must be canonical all-zero CPU-long")
    return ids, batch


def transition_components(
    left: Any,
    right: Any,
    parsed: Any,
    *,
    graph_weight: float,
    object_weight: float,
) -> tuple[float, float, float]:
    """Return exact CPU-float64 graph/object MSEs and configured total."""

    if graph_weight != 1.0 or object_weight != 1.0:
        raise ValueError("accepted prediction graph/object weights must each equal exactly 1.0")
    left_ids, left_batch = _validated_state_metadata(left, parsed)
    right_ids, right_batch = _validated_state_metadata(right, parsed)
    if not torch.equal(left_ids, right_ids) or not torch.equal(left_batch, right_batch):
        raise ValueError("state object metadata does not align")
    if left.graph_latent.shape != right.graph_latent.shape or left.object_latents.shape != right.object_latents.shape:
        raise ValueError("state latent shapes do not align")
    graph = float(
        torch.mean(
            (
                left.graph_latent.detach().to(device="cpu", dtype=torch.float64)
                - right.graph_latent.detach().to(device="cpu", dtype=torch.float64)
            )
            ** 2
        )
    )
    objects = float(
        torch.mean(
            (
                left.object_latents.detach().to(device="cpu", dtype=torch.float64)
                - right.object_latents.detach().to(device="cpu", dtype=torch.float64)
            )
            ** 2
        )
    )
    total = graph_weight * graph + object_weight * objects
    if not all(torch.isfinite(torch.tensor(value)) for value in (graph, objects, total)):
        raise ValueError("transition metrics must be finite")
    return graph, objects, total


FLOAT64_EPSILON = torch.finfo(torch.float64).eps


def derived_metrics(
    true_error: float,
    wrong_error: float,
    separation: float,
    *,
    error_threshold: float,
    separation_threshold: float,
) -> dict[str, Any]:
    denominator = max(true_error, FLOAT64_EPSILON)
    error_ratio = wrong_error / denominator
    separation_ratio = separation / denominator
    values = (error_ratio, wrong_error - true_error, separation_ratio)
    if not all(torch.isfinite(torch.tensor(value, dtype=torch.float64)) for value in values):
        raise ValueError("derived transition metrics must be finite")
    return {
        "error_ratio": error_ratio,
        "error_margin": wrong_error - true_error,
        "separation_ratio": separation_ratio,
        "transition_equivalent": error_ratio <= error_threshold and separation_ratio <= separation_threshold,
        "near_zero": true_error <= FLOAT64_EPSILON,
    }


def distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _require_finite_tree(value: Any) -> None:
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
        raise ValueError("JSON artifact contains a non-finite number")
    if isinstance(value, Mapping):
        for nested in value.values():
            _require_finite_tree(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _require_finite_tree(nested)


def _require_keys(value: Any, keys: set[str] | frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{name} has unknown or missing keys")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
        raise ValueError(f"{name} must be a finite JSON number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return numeric


def _validate_action(value: Any, name: str) -> None:
    action = _require_keys(value, {"name", "arguments"}, name)
    if type(action["name"]) is not str or not action["name"]:
        raise ValueError(f"{name}.name must be a nonempty string")
    if not isinstance(action["arguments"], list) or not all(type(arg) is str for arg in action["arguments"]):
        raise ValueError(f"{name}.arguments must be a list of strings")


def _manifest_groups(
    manifest_records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    if manifest_records is None:
        manifest_records, _identity = load_and_validate_candidate_manifest(FIXED_CANDIDATE_MANIFEST)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in manifest_records:
        grouped[str(record["group"])].append(record)
    if set(grouped) != set(GROUPS):
        raise ValueError("candidate manifest must map every fixed group exactly once")
    return {group: tuple(grouped[group]) for group in GROUPS}


def validate_details(
    details: Sequence[Mapping[str, Any]],
    *,
    expected_groups: int = 44,
    manifest_records: Sequence[Mapping[str, Any]] | None = None,
    reconciled_transitions: Sequence[ReconciledTransition] | None = None,
) -> None:
    if len(details) != expected_groups:
        raise ValueError("transition detail count mismatch")
    expected_order = GROUPS[:expected_groups]
    fixed_groups = _manifest_groups(manifest_records)
    reconciled = None
    if reconciled_transitions is not None:
        reconciled = {item.group: item for item in reconciled_transitions}
        if len(reconciled) != len(reconciled_transitions) or set(reconciled) != set(expected_order):
            raise ValueError("reconciled transitions must uniquely map every validated fixed group")
    for index, row in enumerate(details):
        _require_keys(row, DETAIL_KEYS, "transition detail")
        if row["group"] != expected_order[index]:
            raise ValueError("transition details differ from literal fixed group order")
        problem, step = expected_order[index].split(":")
        if type(row["problem"]) is not str or row["problem"] != problem:
            raise ValueError("transition detail problem does not match group")
        if type(row["step"]) is not int or row["step"] != int(step):
            raise ValueError("transition detail step does not match group")
        _validate_action(row["trace_action"], "trace action")
        fixed_rows = fixed_groups[row["group"]]
        if any(record["problem"] != row["problem"] or record["step"] != row["step"] for record in fixed_rows):
            raise ValueError("fixed group has inconsistent problem/step ownership")
        fixed_traces = [
            record
            for record in fixed_rows
            if record["category"] == "trace" and record["applicability_label"] is True
        ]
        if len(fixed_traces) != 1 or row["trace_action"] != fixed_traces[0]["action"]:
            raise ValueError("trace action does not match the unique fixed manifest trace")
        if reconciled is not None:
            transition = reconciled[row["group"]]
            if (
                transition.problem != row["problem"]
                or transition.step != row["step"]
                or _action_payload(transition.trace_action) != row["trace_action"]
            ):
                raise ValueError("detail identity does not match the reconciled corpus transition")
        eligible_fixed = filter_hard_negatives(fixed_rows, fixed_traces[0])
        skipped = row["status"] == "skipped"
        if skipped:
            if row["skip_reason"] != "no_inapplicable_same_schema_hard_negative" or any(
                row[key] is not None for key in NULLABLE_DETAIL_KEYS
            ):
                raise ValueError("skipped transition detail violates null contract")
            if eligible_fixed:
                raise ValueError("skipped detail has an eligible fixed-manifest hard negative")
        elif row["status"] == "eligible":
            if row["skip_reason"] is not None or any(row[key] is None for key in NULLABLE_DETAIL_KEYS):
                raise ValueError("eligible transition detail violates non-null contract")
            _validate_action(row["wrong_action"], "wrong action")
            if row["wrong_action"]["name"] != row["trace_action"]["name"]:
                raise ValueError("wrong action must have the trace action schema")
            if row["wrong_action"] == row["trace_action"]:
                raise ValueError("wrong action must differ from the trace action")
            if row["wrong_category"] not in CANDIDATE_CATEGORIES:
                raise ValueError("wrong category is not an eligible hard-negative category")
            fixed_matches = [
                record
                for record in eligible_fixed
                if record["action"] == row["wrong_action"] and record["category"] == row["wrong_category"]
            ]
            if len(fixed_matches) != 1:
                raise ValueError("wrong action/category is not a unique eligible fixed-manifest record")
            for key in (
                "wrong_unit_action_l2",
                "true_graph_error",
                "true_object_error",
                "true_total_error",
                "wrong_graph_error",
                "wrong_object_error",
                "wrong_total_error",
                "prediction_graph_separation",
                "prediction_object_separation",
                "prediction_separation",
                "error_ratio",
                "separation_ratio",
            ):
                _require_number(row[key], f"detail.{key}", minimum=0.0)
            if float(row["wrong_unit_action_l2"]) > 2.0:
                raise ValueError("unit-normalized action L2 distance must be in [0,2]")
            _require_number(row["error_margin"], "detail.error_margin")
            if type(row["transition_equivalent"]) is not bool:
                raise ValueError("transition_equivalent must be Boolean")
            expected_true_total = float(row["true_graph_error"]) + float(row["true_object_error"])
            expected_wrong_total = float(row["wrong_graph_error"]) + float(row["wrong_object_error"])
            expected_separation = float(row["prediction_graph_separation"]) + float(row["prediction_object_separation"])
            if float(row["true_total_error"]) != expected_true_total:
                raise ValueError("true total error does not reconcile components")
            if float(row["wrong_total_error"]) != expected_wrong_total:
                raise ValueError("wrong total error does not reconcile components")
            if float(row["prediction_separation"]) != expected_separation:
                raise ValueError("prediction separation does not reconcile components")
            derived = derived_metrics(
                expected_true_total,
                expected_wrong_total,
                expected_separation,
                error_threshold=1.10,
                separation_threshold=0.25,
            )
            for key in ("error_ratio", "error_margin", "separation_ratio", "transition_equivalent"):
                if row[key] != derived[key] or type(row[key]) is not type(derived[key]):
                    raise ValueError(f"detail {key} does not match recomputed value")
        else:
            raise ValueError("transition status must be eligible or skipped")
        _require_finite_tree(row)


SETTINGS = {
    "chunk_size": 2048,
    "error_ratio_threshold": 1.10,
    "separation_ratio_threshold": 0.25,
    "mostly_rate_threshold": 0.50,
    "float64_epsilon": FLOAT64_EPSILON,
    "candidate_categories": list(CANDIDATE_CATEGORIES),
    "distance_policy": "raw_action_latent_cpu_float64_unit_l2_64d",
    "zero_norm_policy": "zero_vector",
    "graph_weight": 1.0,
    "object_weight": 1.0,
    "near_zero_true_error_policy": "true_total_error_lte_float64_epsilon",
    "detail_order": "literal_fixed_group_order",
}
RESTORATION_STATE_KEYS = {
    "jepa": "model_state_dict",
    "goal_head": "goal_head_state_dict",
    "action_contrastive_anchor": "action_contrastive_anchor_state_dict",
    "argument_reconstruction_head": "argument_reconstruction_head_state_dict",
    "applicability_head": "applicability_head_state_dict",
}
ENVIRONMENT_KEYS = frozenset(
    {
        "python_version",
        "torch_version",
        "platform",
        "byteorder",
        "num_threads",
        "num_interop_threads",
        "deterministic_algorithms",
        "python_hash_seed",
        "cublas_workspace_config",
    }
)
DISTRIBUTION_KEYS = frozenset({"count", "min", "median", "mean", "max"})


def _require_literal(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected):
        raise ValueError(f"{name} has the wrong strict type")
    if isinstance(expected, Mapping):
        if set(value) != set(expected):
            raise ValueError(f"{name} has unknown or missing keys")
        for key in expected:
            _require_literal(value[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"{name} has the wrong ordered list length")
        for index, (actual_item, expected_item) in enumerate(zip(value, expected, strict=True)):
            _require_literal(actual_item, expected_item, f"{name}[{index}]")
    elif value != expected:
        raise ValueError(f"{name} differs from the fixed literal")


def _validate_distribution(value: Any, name: str) -> None:
    data = _require_keys(value, DISTRIBUTION_KEYS, name)
    count = _require_int(data["count"], f"{name}.count")
    scalars = [data[key] for key in ("min", "median", "mean", "max")]
    if count == 0:
        if any(item is not None for item in scalars):
            raise ValueError(f"{name} zero-count scalars must be null")
        return
    numbers = [_require_number(item, f"{name} scalar") for item in scalars]
    if not numbers[0] <= numbers[1] <= numbers[3] or not numbers[0] <= numbers[2] <= numbers[3]:
        raise ValueError(f"{name} summary statistics are inconsistent")


def validate_summary(
    summary: Mapping[str, Any],
    details: Sequence[Mapping[str, Any]] | None = None,
    *,
    manifest_records: Sequence[Mapping[str, Any]] | None = None,
    reconciled_transitions: Sequence[ReconciledTransition] | None = None,
) -> None:
    _require_keys(summary, SUMMARY_KEYS, "transition summary")
    _require_literal(
        summary["schema_version"], "action_latent_updated_phase0.transition_equivalence.v1", "schema_version"
    )
    _require_literal(summary["kind"], "transition_equivalence", "kind")
    _require_literal(summary["dataset"], str(DATASET), "dataset")
    checkpoint = Path(summary["checkpoint"]) if type(summary["checkpoint"]) is str else None
    if checkpoint not in (BASELINE_CHECKPOINT, PHASE2_CHECKPOINT):
        raise ValueError("summary checkpoint is not a fixed checkpoint")
    _require_literal(summary["checkpoint_sha256"], FIXED_SHA256[checkpoint], "checkpoint_sha256")
    _require_literal(summary["split"], "val", "split")
    _require_literal(summary["seed"], 20260717, "seed")
    manifest = _require_keys(summary["candidate_manifest"], {"path", "bytes", "sha256", "count"}, "manifest")
    _require_literal(manifest["path"], str(FIXED_CANDIDATE_MANIFEST), "manifest.path")
    _require_literal(manifest["bytes"], 117385, "manifest.bytes")
    _require_literal(manifest["count"], 604, "manifest.count")
    _require_literal(
        manifest["sha256"], "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9", "manifest.sha256"
    )
    _require_literal(summary["settings"], SETTINGS, "settings")
    restoration = _require_keys(summary["checkpoint_restoration"], set(RESTORATION_STATE_KEYS), "restoration")
    baseline = checkpoint == BASELINE_CHECKPOINT
    for module, state_key in RESTORATION_STATE_KEYS.items():
        item = _require_keys(restoration[module], {"state_key", "status"}, f"restoration.{module}")
        expected_status = "restored" if module in {"jepa", "goal_head"} or not baseline else "disabled"
        _require_literal(item, {"state_key": state_key, "status": expected_status}, f"restoration.{module}")
    counts = _require_keys(
        summary["counts"], {"groups", "eligible", "skipped", "exact_or_near_zero_true_error"}, "counts"
    )
    for key in counts:
        _require_int(counts[key], f"counts.{key}")
    if counts["groups"] != 44 or counts["eligible"] < 40 or counts["eligible"] + counts["skipped"] != 44:
        raise ValueError("summary group counts do not reconcile or eligible count is below 40")
    if counts["exact_or_near_zero_true_error"] > counts["eligible"]:
        raise ValueError("near-zero count exceeds eligible count")
    metrics = _require_keys(
        summary["metrics"],
        {
            "error_ratio",
            "error_margin",
            "separation_ratio",
            "equivalence_rate",
            "mostly_transition_equivalent",
            "per_category",
        },
        "metrics",
    )
    for key in ("error_ratio", "error_margin", "separation_ratio"):
        _validate_distribution(metrics[key], f"metrics.{key}")
        if metrics[key]["count"] != counts["eligible"]:
            raise ValueError("global distribution count differs from eligible count")
    rate = _require_number(metrics["equivalence_rate"], "equivalence_rate", minimum=0.0)
    if rate > 1.0 or type(metrics["mostly_transition_equivalent"]) is not bool:
        raise ValueError("equivalence rate/verdict has invalid type or range")
    if metrics["mostly_transition_equivalent"] != (rate >= 0.50):
        raise ValueError("mostly verdict does not match equivalence rate")
    per_category = _require_keys(metrics["per_category"], set(CANDIDATE_CATEGORIES), "per_category")
    category_total = 0
    for category in CANDIDATE_CATEGORIES:
        item = _require_keys(
            per_category[category],
            {"count", "error_ratio", "error_margin", "separation_ratio", "equivalence_rate"},
            f"per_category.{category}",
        )
        category_count = _require_int(item["count"], f"per_category.{category}.count")
        category_total += category_count
        for key in ("error_ratio", "error_margin", "separation_ratio"):
            _validate_distribution(item[key], f"per_category.{category}.{key}")
            if item[key]["count"] != category_count:
                raise ValueError("category distribution count does not reconcile")
        if category_count == 0:
            if item["equivalence_rate"] is not None:
                raise ValueError("empty category equivalence rate must be null")
        else:
            category_rate = _require_number(item["equivalence_rate"], "category equivalence rate", minimum=0.0)
            if category_rate > 1.0:
                raise ValueError("category equivalence rate exceeds one")
    if category_total != counts["eligible"]:
        raise ValueError("category counts do not sum to eligible")
    environment = _require_keys(summary["environment"], ENVIRONMENT_KEYS, "environment")
    for key in ("python_version", "torch_version", "platform", "byteorder"):
        if type(environment[key]) is not str or not environment[key]:
            raise ValueError(f"environment.{key} must be a nonempty string")
    for key in ("num_threads", "num_interop_threads"):
        _require_int(environment[key], f"environment.{key}", minimum=1)
    if type(environment["deterministic_algorithms"]) is not bool:
        raise ValueError("deterministic_algorithms must be Boolean")
    for key in ("python_hash_seed", "cublas_workspace_config"):
        if environment[key] is not None and type(environment[key]) is not str:
            raise ValueError(f"environment.{key} must be a nullable string")
    if type(summary["device"]) is not str or not summary["device"]:
        raise ValueError("device must be a nonempty string")
    if type(summary["output"]) is not str:
        raise ValueError("output must be a string")
    validate_run_binding(checkpoint, Path(summary["output"]))
    _require_number(summary["runtime_seconds"], "runtime_seconds", minimum=0.0)
    if not re.fullmatch(r"[0-9a-f]{64}", summary["checkpoint_sha256"]):
        raise ValueError("checkpoint hash is malformed")
    _require_finite_tree(summary)
    if details is not None:
        validate_details(
            details,
            manifest_records=manifest_records,
            reconciled_transitions=reconciled_transitions,
        )
        expected_counts, expected_metrics = aggregate_details(details)
        if summary["counts"] != expected_counts or summary["metrics"] != expected_metrics:
            raise ValueError("summary counts/metrics do not exactly recompute from details")


def repeat_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    validate_summary(summary)
    projected = dict(summary)
    for key in ("checkpoint", "output", "device", "runtime_seconds"):
        projected.pop(key)
    environment = dict(projected["environment"])
    environment.pop("torch_version")
    environment.pop("platform")
    projected["environment"] = environment
    _require_finite_tree(projected)
    return projected


def aggregate_details(
    details: Sequence[Mapping[str, Any]], *, minimum_eligible: int = 40
) -> tuple[dict[str, int], dict[str, Any]]:
    eligible = [row for row in details if row["status"] == "eligible"]
    if len(eligible) < minimum_eligible:
        raise ValueError(f"fewer than 40 eligible groups: {len(eligible)}")
    counts = {
        "groups": len(details),
        "eligible": len(eligible),
        "skipped": len(details) - len(eligible),
        "exact_or_near_zero_true_error": sum(float(row["true_total_error"]) <= FLOAT64_EPSILON for row in eligible),
    }
    metrics: dict[str, Any] = {
        name: distribution([float(row[name]) for row in eligible])
        for name in ("error_ratio", "error_margin", "separation_ratio")
    }
    equivalent = sum(bool(row["transition_equivalent"]) for row in eligible)
    metrics["equivalence_rate"] = equivalent / len(eligible)
    metrics["mostly_transition_equivalent"] = metrics["equivalence_rate"] >= 0.50
    per_category = {}
    for category in CANDIDATE_CATEGORIES:
        rows = [row for row in eligible if row["wrong_category"] == category]
        per_category[category] = {
            "count": len(rows),
            "error_ratio": distribution([float(row["error_ratio"]) for row in rows]),
            "error_margin": distribution([float(row["error_margin"]) for row in rows]),
            "separation_ratio": distribution([float(row["separation_ratio"]) for row in rows]),
            "equivalence_rate": (sum(bool(row["transition_equivalent"]) for row in rows) / len(rows) if rows else None),
        }
    metrics["per_category"] = per_category
    return counts, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--chunk-size", default=2048, type=int)
    parser.add_argument("--seed", default=20260717, type=int)
    parser.add_argument("--equivalence-error-ratio", default=1.10, type=float)
    parser.add_argument("--equivalence-separation-ratio", default=0.25, type=float)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    actual = (
        args.device,
        args.split,
        args.chunk_size,
        args.seed,
        args.equivalence_error_ratio,
        args.equivalence_separation_ratio,
    )
    if actual != ("cuda", "val", 2048, 20260717, 1.10, 0.25):
        raise ValueError("Stage 0C requires every fixed CLI setting")


def validate_run_binding(checkpoint: Path, output: Path) -> None:
    if len(output.parts) < 3 or output.parts[-3] != "transition_equivalence":
        raise ValueError("output must end in transition_equivalence/{baseline|phase2}/{run1|run2}")
    variant, repeat = output.parts[-2:]
    if repeat not in {"run1", "run2"}:
        raise ValueError("transition-equivalence destination must be run1 or run2")
    expected = {"baseline": BASELINE_CHECKPOINT, "phase2": PHASE2_CHECKPOINT}.get(variant)
    if expected is None or checkpoint != expected:
        raise ValueError("checkpoint/output binding does not match baseline or phase2 fixed evidence")


def _root_identity(candidate_identity: Mapping[str, Any]) -> dict[str, Any]:
    identities = {}
    for path, expected in FIXED_SHA256.items():
        identity = file_identity(path)
        if identity["sha256"] != expected:
            raise ValueError(f"fixed input identity changed: {path}")
        identities[path] = identity
    return {
        "schema_version": "action_latent_updated_phase0.root_identity.v1",
        "updated_spec": identities[UPDATED_SPEC],
        "baseline_checkpoint": identities[BASELINE_CHECKPOINT],
        "baseline_config": identities[BASELINE_CONFIG],
        "phase2_checkpoint": identities[PHASE2_CHECKPOINT],
        "phase2_config": identities[PHASE2_CONFIG],
        "corpus_manifest": {**identities[CORPUS_MANIFEST], "count": 12},
        "candidate_manifest": dict(candidate_identity),
        "split_sha256": SPLIT_SHA256,
        "created_by": "schema_residual/baseline/run1",
    }


def _environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _action_payload(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        action = action["action"]
        return {"name": action["name"], "arguments": list(action["arguments"])}
    return {"name": action.name, "arguments": list(action.arguments)}


def _encode_action(bundle: Any, space: ActionDecodingSpace, row: Mapping[str, Any], source: Any, device: torch.device):
    payload = row["action"]
    action = GroundAction(payload["name"], tuple(payload["arguments"]))
    tensors = space.action_tensors_for_ground_actions([action], device=device)
    latent = bundle.jepa.action_encoder(tensors, source)
    if (
        latent.shape != (1, 64)
        or latent.device != source.graph_latent.device
        or latent.dtype != source.graph_latent.dtype
    ):
        raise ValueError("native action latent must preserve source device/dtype and [1,64] shape")
    return latent


def evaluate_transition(
    item: ReconciledTransition,
    bundle: Any,
    *,
    device: torch.device,
    graph_weight: float,
    object_weight: float,
    error_threshold: float,
    separation_threshold: float,
) -> dict[str, Any]:
    trace_row = next(row for row in item.records if row["category"] == "trace")
    base = {
        "group": item.group,
        "problem": item.problem,
        "step": item.step,
        "trace_action": _action_payload(item.trace_action),
    }
    candidates = filter_hard_negatives(item.records, trace_row)
    if not candidates:
        return {
            **base,
            "status": "skipped",
            "skip_reason": "no_inapplicable_same_schema_hard_negative",
            **{
                key: None
                for key in (
                    "wrong_action",
                    "wrong_category",
                    "wrong_unit_action_l2",
                    "true_graph_error",
                    "true_object_error",
                    "true_total_error",
                    "wrong_graph_error",
                    "wrong_object_error",
                    "wrong_total_error",
                    "prediction_graph_separation",
                    "prediction_object_separation",
                    "prediction_separation",
                    "error_ratio",
                    "error_margin",
                    "separation_ratio",
                    "transition_equivalent",
                )
            },
        }
    source_graph = build_state_graph(item.parsed, item.source_atoms, include_static=True).to(device)
    source = bundle.jepa.encode(source_graph)
    space = ActionDecodingSpace.from_parsed_problem(item.parsed)
    native_true = _encode_action(bundle, space, trace_row, source, device)
    native_candidates = [(row, _encode_action(bundle, space, row, source, device)) for row in candidates]
    selected, distance, _copies = select_nearest_wrong(native_true, native_candidates)
    native_wrong = next(latent for row, latent in native_candidates if row is selected)

    # Both predictions are complete before the successor value is accessed or encoded.
    pred_true = bundle.jepa.predictor(source, native_true)
    pred_wrong = bundle.jepa.predictor(source, native_wrong)
    successor_atoms = item.trajectory.states[item.successor_index]
    next_graph = build_state_graph(item.parsed, successor_atoms, include_static=True).to(device)
    target = bundle.jepa.encode(next_graph)

    true_graph, true_object, true_total = transition_components(
        pred_true, target, item.parsed, graph_weight=graph_weight, object_weight=object_weight
    )
    wrong_graph, wrong_object, wrong_total = transition_components(
        pred_wrong, target, item.parsed, graph_weight=graph_weight, object_weight=object_weight
    )
    sep_graph, sep_object, separation = transition_components(
        pred_wrong, pred_true, item.parsed, graph_weight=graph_weight, object_weight=object_weight
    )
    derived = derived_metrics(
        true_total,
        wrong_total,
        separation,
        error_threshold=error_threshold,
        separation_threshold=separation_threshold,
    )
    return {
        **base,
        "status": "eligible",
        "skip_reason": None,
        "wrong_action": _action_payload(selected),
        "wrong_category": selected["category"],
        "wrong_unit_action_l2": distance,
        "true_graph_error": true_graph,
        "true_object_error": true_object,
        "true_total_error": true_total,
        "wrong_graph_error": wrong_graph,
        "wrong_object_error": wrong_object,
        "wrong_total_error": wrong_total,
        "prediction_graph_separation": sep_graph,
        "prediction_object_separation": sep_object,
        "prediction_separation": separation,
        "error_ratio": derived["error_ratio"],
        "error_margin": derived["error_margin"],
        "separation_ratio": derived["separation_ratio"],
        "transition_equivalent": derived["transition_equivalent"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    validate_args(args)
    validate_run_binding(args.checkpoint, args.output)
    if args.dataset_dir != DATASET or args.candidate_manifest != FIXED_CANDIDATE_MANIFEST:
        raise ValueError("dataset and candidate manifest must use fixed absolute evidence paths")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    records, candidate_identity = load_and_validate_candidate_manifest(args.candidate_manifest)
    root = args.output.parents[2]
    prepare_output_directory(root, args.output, _root_identity(candidate_identity), first_command=False)
    config, corpus, bundle, device, restoration = load_checkpoint_bundle(
        args.dataset_dir, args.checkpoint, device_name=args.device, include_restoration_metadata=True
    )
    selected_corpus = select_split(corpus, config, args.split, seed=args.seed)
    transitions = reconcile_transitions(records, selected_corpus)
    if tuple(item.group for item in transitions) != GROUPS:
        raise ValueError("transition detail order differs from literal fixed group order")
    for module in (
        bundle.jepa,
        bundle.goal_head,
        bundle.action_contrastive_anchor,
        bundle.argument_reconstruction_head,
        bundle.applicability_head,
    ):
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    graph_weight = float(config.model.loss.prediction.graph_weight)
    object_weight = float(config.model.loss.prediction.object_weight)
    if graph_weight != 1.0 or object_weight != 1.0:
        raise ValueError("accepted checkpoint prediction weights must each equal exactly 1.0")
    with torch.inference_mode():
        details = [
            evaluate_transition(
                item,
                bundle,
                device=device,
                graph_weight=graph_weight,
                object_weight=object_weight,
                error_threshold=args.equivalence_error_ratio,
                separation_threshold=args.equivalence_separation_ratio,
            )
            for item in transitions
        ]
    counts, metrics = aggregate_details(details)
    summary = {
        "schema_version": "action_latent_updated_phase0.transition_equivalence.v1",
        "kind": "transition_equivalence",
        "dataset": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": args.split,
        "seed": args.seed,
        "candidate_manifest": candidate_identity,
        "settings": dict(SETTINGS),
        "checkpoint_restoration": restoration,
        "counts": counts,
        "metrics": metrics,
        "environment": _environment(),
        "device": str(device),
        "output": str(args.output),
        "runtime_seconds": time.perf_counter() - started,
    }
    validate_summary(
        summary,
        details,
        manifest_records=records,
        reconciled_transitions=transitions,
    )
    (args.output / "summary.json").write_bytes(canonical_json_bytes(summary))
    (args.output / "details.json").write_bytes(canonical_json_bytes(details))
    return summary


def main() -> int:
    summary = run(build_parser().parse_args())
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
