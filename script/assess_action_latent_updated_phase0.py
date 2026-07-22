"""Strict deterministic assessor for Updated Phase 0 Stage 0D.

The assessor treats every diagnostic artifact as untrusted input: it validates
canonical bytes and repeat inventories, recomputes ranking evidence, and only
then applies the preregistered seven-clause precedence.
"""

from __future__ import annotations

# ruff: noqa: E501, F401, I001

import argparse
import copy
import hashlib
import json
import math
import shutil

import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from action_phase0_common import canonical_json_bytes, load_and_validate_candidate_manifest
from action_role_object_probe import RoleObjectProbe, fit_role_object_probe
import diagnose_action_candidate_ranking as ranking_module
import diagnose_action_applicability_recoverability as recoverability_module
import diagnose_action_schema_residuals as residual_module
import diagnose_action_transition_equivalence as transition_module

DECISION_BOOLEAN_KEYS = (
    "representation_ok", "latent_separable", "hybrid_separable", "raw_separable",
    "latent_rank", "raw_rank", "hybrid_rank", "mostly_transition_equivalent",
    "transition_distinguishable",
)
REQUIRED_PATH_ARGUMENTS = (
    "updated_spec", "baseline_checkpoint", "baseline_config", "phase2_checkpoint", "phase2_config",
    "corpus_manifest", "candidate_manifest", "baseline_schema_run1", "baseline_schema_run2",
    "phase2_schema_run1", "phase2_schema_run2", "baseline_recoverability_run1",
    "baseline_recoverability_run2", "phase2_recoverability_run1", "phase2_recoverability_run2",
    "baseline_transition_run1", "baseline_transition_run2", "phase2_transition_run1",
    "phase2_transition_run2", "baseline_ranking_run1", "baseline_ranking_run2",
    "phase2_ranking_run1", "phase2_ranking_run2",
)
FIXED_PATHS = {
    "updated_spec": Path("/opt/data/workspace/acs-jepa/script/ACTION_LATENT_UPDATED_SPEC.md"),
    "baseline_checkpoint": Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt"),
    "baseline_config": Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml"),
    "phase2_checkpoint": Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt"),
    "phase2_config": Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml"),
    "corpus_manifest": Path("/opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json"),
    "candidate_manifest": ranking_module.FIXED_CANDIDATE_MANIFEST,
}
EVIDENCE_ROLES = {
    "root_identity", "governing_spec", "corpus_manifest", "candidate_manifest", "checkpoint", "config",
    "diagnostic_summary", "diagnostic_details", "feature_schema", "split_manifest", "probe_states",
    "role_probe_state", "assessment_summary", "assessment_markdown",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in REQUIRED_PATH_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _duplicate_guard(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def load_canonical_json(path: Path, *, expected_keys: set[str] | None = None) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_duplicate_guard, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON evidence: {path}") from exc
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"JSON evidence is not canonical: {path}")
    if expected_keys is not None and (not isinstance(value, Mapping) or set(value) != expected_keys):
        raise ValueError("JSON object keys are unknown or missing")
    return value


SUMMARY_KEYS = {
    "schema_version", "kind", "dataset", "checkpoint", "checkpoint_sha256", "split", "seed",
    "candidate_manifest", "settings", "checkpoint_restoration", "counts", "metrics", "environment",
    "device", "output", "runtime_seconds",
}
STAGE_FILES = {
    "schema_residual": ("details.json",),
    "applicability_recoverability": ("details.json", "feature_schema.json", "split_manifest.json", "probe_states.json"),
    "transition_equivalence": ("details.json",),
    "candidate_ranking": ("details.json", "split_manifest.json", "role_probe_state.json"),
}


def _keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} has unknown or missing keys")
    return value


def _finite_tree(value: Any, name: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} is non-finite")


def load_diagnostic_artifacts(summary_path: Path, kind: str) -> dict[str, Any]:
    names = ("summary.json", *STAGE_FILES[kind])
    if {path.name for path in summary_path.parent.iterdir() if path.is_file()} != set(names):
        raise ValueError("diagnostic sibling inventory drift")
    return {name: load_canonical_json(summary_path.parent / name) for name in names}


def _validate_common_summary(summary: Any, kind: str, *, summary_path: Path | None = None) -> str:
    _keys(summary, SUMMARY_KEYS, f"{kind} summary")
    expected_schema = f"action_latent_updated_phase0.{kind}.v1"
    checkpoint_paths = {"baseline": ranking_module.BASELINE_CHECKPOINT, "phase2": ranking_module.PHASE2_CHECKPOINT}
    if type(summary["output"]) is not str:
        raise ValueError("diagnostic output path type drift")
    output = Path(summary["output"])
    if len(output.parts) < 3 or output.parts[-1] not in {"run1", "run2"}:
        raise ValueError("diagnostic output binding drift")
    variant = output.parts[-2]
    checkpoint = checkpoint_paths.get(variant)
    expected_stage = {"schema_residual": "schema_residual", "applicability_recoverability": "recoverability", "transition_equivalence": "transition_equivalence", "candidate_ranking": "candidate_ranking"}[kind]
    if (
        summary["kind"] != kind or summary["schema_version"] != expected_schema
        or summary["dataset"] != str(ranking_module.DATASET) or summary["split"] != "val"
        or type(summary["seed"]) is not int or summary["seed"] != 20260717
        or len(output.parts) < 3 or output.parts[-3] != expected_stage or checkpoint is None
        or summary["checkpoint"] != str(checkpoint) or summary["checkpoint_sha256"] != ranking_module.CHECKPOINT_SHA256[variant]
        or summary_path is not None and output != summary_path.parent
        or type(summary["runtime_seconds"]) not in (int, float) or not math.isfinite(float(summary["runtime_seconds"])) or summary["runtime_seconds"] < 0
    ):
        raise ValueError("diagnostic summary fixed literals/types/path drift")
    manifest = _keys(summary["candidate_manifest"], {"path", "bytes", "sha256", "count"}, "candidate manifest identity")
    _records, actual_manifest = load_and_validate_candidate_manifest(ranking_module.FIXED_CANDIDATE_MANIFEST)
    if manifest != actual_manifest:
        raise ValueError("candidate manifest exact identity drift")
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != summary["checkpoint_sha256"]:
        raise ValueError("checkpoint exact identity drift")
    environment = _keys(summary["environment"], {"python_version", "torch_version", "platform", "byteorder", "num_threads", "num_interop_threads", "deterministic_algorithms", "python_hash_seed", "cublas_workspace_config"}, "environment")
    if (any(type(environment[key]) is not str or not environment[key] for key in ("python_version", "torch_version", "platform"))
            or environment["byteorder"] not in {"little", "big"}
            or any(type(environment[key]) is not int or environment[key] < 1 for key in ("num_threads", "num_interop_threads"))
            or type(environment["deterministic_algorithms"]) is not bool or not environment["deterministic_algorithms"]
            or any(environment[key] is not None and type(environment[key]) is not str for key in ("python_hash_seed", "cublas_workspace_config"))):
        raise ValueError("environment exact literal/type drift")
    restoration = _keys(summary["checkpoint_restoration"], {"jepa", "goal_head", "action_contrastive_anchor", "argument_reconstruction_head", "applicability_head"}, "restoration")
    state_keys = {"jepa": "model_state_dict", "goal_head": "goal_head_state_dict", "action_contrastive_anchor": "action_contrastive_anchor_state_dict", "argument_reconstruction_head": "argument_reconstruction_head_state_dict", "applicability_head": "applicability_head_state_dict"}
    expected_restoration = {
        name: {"state_key": state_key, "status": "restored" if variant == "phase2" or name in {"jepa", "goal_head"} else "disabled"}
        for name, state_key in state_keys.items()
    }
    if restoration != expected_restoration:
        raise ValueError("checkpoint restoration exact map drift")
    expected_device = "cpu" if kind in {"applicability_recoverability", "candidate_ranking"} else "cuda"
    if summary["device"] != expected_device:
        raise ValueError("diagnostic fixed device drift")
    _finite_tree(summary)
    return variant


def _validate_schema_residual(artifacts: Mapping[str, Any], *, summary_path: Path | None = None) -> None:
    summary, details = artifacts["summary.json"], artifacts["details.json"]
    _validate_common_summary(summary, "schema_residual", summary_path=summary_path)
    settings = _keys(summary["settings"], {"chunk_size", "expected_population_count", "residual_centers", "numerical_rank_relative_tolerance", "zero_norm_policy"}, "schema settings")
    if settings != {"chunk_size": 2048, "expected_population_count": 174780, "residual_centers": ["global_schema", "state_schema"], "numerical_rank_relative_tolerance": 1e-6, "zero_norm_policy": "zero_vector"}:
        raise ValueError("schema residual fixed settings drift")
    counts = _keys(summary["counts"], {"groups", "full_population", "candidate_manifest_records", "schemas", "nearest_wrong_rows", "nearest_invalid_rows"}, "schema counts")
    if counts != {"groups": 44, "full_population": 174780, "candidate_manifest_records": 604, "schemas": 7, "nearest_wrong_rows": 44, "nearest_invalid_rows": 44} or any(type(value) is not int for value in counts.values()):
        raise ValueError("schema residual fixed counts drift")
    metrics = _keys(summary["metrics"], {"full_population_identity", "raw_variance_decomposition", "global_schema_residual", "state_schema_residual", "nearest_wrong_same_schema_raw_l2", "nearest_invalid_same_schema_unit_residual_l2"}, "schema metrics")
    population_identity = _keys(metrics["full_population_identity"], {"count", "bytes", "sha256"}, "population identity")
    if population_identity != {"count": 174780, "bytes": 24455400, "sha256": "c6b056d8a976a77c994338aeceedcc519a7506d770e6921d08d00da4753e97d2"}:
        raise ValueError("full population exact identity drift")
    variance = _keys(metrics["raw_variance_decomposition"], {"total_variance", "within_schema_variance", "between_schema_variance", "within_schema_fraction", "between_schema_fraction", "reconstruction_absolute_error"}, "variance decomposition")
    reconstructed_error = abs(variance["total_variance"] - (variance["within_schema_variance"] + variance["between_schema_variance"]))
    if not math.isclose(reconstructed_error, variance["reconstruction_absolute_error"], rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("variance decomposition formula drift")
    if variance["within_schema_fraction"] != variance["within_schema_variance"] / variance["total_variance"] or variance["between_schema_fraction"] != variance["between_schema_variance"] / variance["total_variance"]:
        raise ValueError("variance fraction formula drift")
    stat_keys = {"count", "dimension", "std_min", "std_mean", "std_max", "std_values", "covariance_eigenvalues", "normalized_eigenvalue_spectrum", "effective_rank", "numerical_rank", "zero_norm_count"}
    for residual_name in ("global_schema_residual", "state_schema_residual"):
        residual = _keys(metrics[residual_name], {"pooled", "per_schema"}, residual_name)
        per_schema = _keys(residual["per_schema"], set(ranking_module.SCHEMAS), f"{residual_name}.per_schema")
        for statistics in (residual["pooled"], *per_schema.values()):
            _keys(statistics, stat_keys, "residual statistics")
            if (type(statistics["count"]) is not int or statistics["count"] < 1 or statistics["dimension"] != 64
                    or type(statistics["zero_norm_count"]) is not int or not 0 <= statistics["zero_norm_count"] <= statistics["count"]
                    or len(statistics["std_values"]) != 64 or len(statistics["covariance_eigenvalues"]) != 64
                    or len(statistics["normalized_eigenvalue_spectrum"]) != 64):
                raise ValueError("residual statistic vector dimensions/counts drift")
            std = statistics["std_values"]
            eigen = statistics["covariance_eigenvalues"]
            spectrum = statistics["normalized_eigenvalue_spectrum"]
            if (any(not ranking_module._finite(value) or value < 0 for value in (*std, *eigen, *spectrum))
                    or statistics["std_min"] != min(std) or not math.isclose(statistics["std_mean"], sum(std) / 64, rel_tol=1e-12, abs_tol=1e-15)
                    or statistics["std_max"] != max(std)):
                raise ValueError("residual standard-deviation statistics drift")
            total_eigen = sum(eigen)
            expected_spectrum = [value / total_eigen for value in eigen] if total_eigen else [0.0] * 64
            if any(not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15) for actual, expected in zip(spectrum, expected_spectrum, strict=True)):
                raise ValueError("residual normalized eigen spectrum drift")
            entropy = -sum(value * math.log(value) for value in spectrum if value > 0)
            if not math.isclose(statistics["effective_rank"], math.exp(entropy), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("residual effective-rank entropy drift")
            expected_rank = sum(value > max(eigen) * 1e-6 for value in eigen) if eigen else 0
            if type(statistics["numerical_rank"]) is not int or statistics["numerical_rank"] != expected_rank:
                raise ValueError("residual numerical-rank policy drift")
        if residual["pooled"]["count"] != sum(item["count"] for item in per_schema.values()):
            raise ValueError("residual pooled/per-schema population drift")
    for distribution in ("nearest_wrong_same_schema_raw_l2", "nearest_invalid_same_schema_unit_residual_l2"):
        _keys(metrics[distribution], {"count", "min", "median", "mean", "max"}, distribution)
    detail_keys = {"group", "problem", "step", "trace_action", "full_candidate_count", "nearest_wrong_action", "nearest_wrong_raw_l2", "invalid_manifest_candidate_count", "nearest_invalid_action", "nearest_invalid_unit_residual_l2", "trace_zero_residual_norm", "nearest_invalid_zero_residual_norm"}
    if type(details) is not list or len(details) != 44:
        raise ValueError("schema detail population drift")
    records, _manifest_identity = load_and_validate_candidate_manifest(ranking_module.FIXED_CANDIDATE_MANIFEST)
    grouped_manifest: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped_manifest.setdefault(record["group"], []).append(record)
    if [row.get("group") for row in details] != list(grouped_manifest):
        raise ValueError("schema detail fixed group order drift")
    for row in details:
        _keys(row, detail_keys, "schema detail")
        for key in ("trace_action", "nearest_wrong_action", "nearest_invalid_action"):
            ranking_module._require_action(row[key])
        if (type(row["group"]) is not str or type(row["problem"]) is not str or type(row["step"]) is not int
                or type(row["full_candidate_count"]) is not int or row["full_candidate_count"] < 1
                or type(row["invalid_manifest_candidate_count"]) is not int or row["invalid_manifest_candidate_count"] < 1
                or any(not ranking_module._finite(row[key]) or row[key] < 0 for key in ("nearest_wrong_raw_l2", "nearest_invalid_unit_residual_l2"))
                or row["nearest_invalid_unit_residual_l2"] > 2.0
                or type(row["trace_zero_residual_norm"]) is not bool or type(row["nearest_invalid_zero_residual_norm"]) is not bool):
            raise ValueError("schema detail strict type/domain drift")
        group_records = grouped_manifest[row["group"]]
        trace = [item for item in group_records if item["category"] == "trace"]
        invalid = [item for item in group_records if not item["applicability_label"] and item["action"]["name"] == trace[0]["action"]["name"]]
        if (len(trace) != 1 or row["problem"] != trace[0]["problem"] or row["step"] != trace[0]["step"]
                or row["trace_action"] != trace[0]["action"] or row["invalid_manifest_candidate_count"] != len(invalid)
                or sum(item["action"] == row["nearest_invalid_action"] for item in invalid) != 1
                or row["nearest_wrong_action"] == row["trace_action"]
                or row["nearest_wrong_action"]["name"] != row["trace_action"]["name"]
                or row["nearest_invalid_action"]["name"] != row["trace_action"]["name"]):
            raise ValueError("schema detail manifest/action binding drift")
    for metric_name, detail_key in (("nearest_wrong_same_schema_raw_l2", "nearest_wrong_raw_l2"), ("nearest_invalid_same_schema_unit_residual_l2", "nearest_invalid_unit_residual_l2")):
        if metrics[metric_name] != residual_module._distribution([row[detail_key] for row in details]):
            raise ValueError("schema detail distribution recomputation drift")


def _validate_recoverability(
    artifacts: Mapping[str, Any],
    *,
    summary_path: Path | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    summary, details = artifacts["summary.json"], artifacts["details.json"]
    if records is None:
        records, _identity = load_and_validate_candidate_manifest(ranking_module.FIXED_CANDIDATE_MANIFEST)
    expected_path = summary_path or Path(summary["output"]) / "summary.json"
    ranking_module.validate_recoverability_artifacts(
        artifacts,
        records,
        expected_summary_path=expected_path,
        expected_checkpoint=Path(summary["checkpoint"]),
        dataset_dir=ranking_module.DATASET,
        device="cpu",
        split="val",
        seed=20260717,
    )
    _validate_common_summary(summary, "applicability_recoverability", summary_path=summary_path)
    settings = _keys(summary["settings"], {"epochs", "learning_rate", "hidden_dim", "models", "feature_sets", "threshold_policy", "control_policy", "reliability_bins"}, "recoverability settings")
    expected_settings = {"epochs": 200, "learning_rate": .001, "hidden_dim": 64,
                         "models": ["linear", "mlp", "control_mlp"],
                         "feature_sets": [row["name"] for row in recoverability_module.feature_schemas()],
                         "threshold_policy": "max_train_f1_highest_threshold",
                         "control_policy": "train_label_permutation_seed_20260717",
                         "reliability_bins": [index / 10 for index in range(11)]}
    if settings != expected_settings:
        raise ValueError("recoverability fixed settings drift")
    counts = _keys(summary["counts"], {"records", "train_records", "eval_records", "train_groups", "eval_groups", "applicable", "inapplicable"}, "recoverability counts")
    if counts != {"records": 604, "train_records": 453, "eval_records": 151, "train_groups": 33, "eval_groups": 11, "applicable": 62, "inapplicable": 542} or any(type(value) is not int for value in counts.values()):
        raise ValueError("recoverability fixed counts drift")
    metrics = _keys(summary["metrics"], {"features", "verdicts"}, "recoverability metrics")
    schemas = recoverability_module.feature_schemas()
    feature_names = {str(row["name"]) for row in schemas}
    features = _keys(metrics["features"], feature_names, "recoverability features")
    _keys(metrics["verdicts"], {"latent_separable", "raw_separable", "hybrid_separable", "label_or_sampling_blocker", "latent_state_bottleneck", "any_abc_separable"}, "recoverability verdicts")
    binary_keys = {"count", "positive_count", "negative_count", "prevalence", "accuracy", "precision", "recall", "f1", "auroc", "average_precision", "nll", "brier", "true_positive", "false_positive", "true_negative", "false_negative", "reliability_bins"}
    distribution_keys = {"count", "min", "median", "mean", "max"}
    per_schema_keys = {"count", "positive_count", "negative_count", "prevalence", "accuracy", "precision", "recall", "f1", "auroc", "average_precision", "nll", "brier", "true_positive", "false_positive", "true_negative", "false_negative", "reliability_bins"}
    for feature_name, probes in features.items():
        _keys(probes, {"linear", "mlp", "control_mlp"}, f"feature {feature_name}")
        for kind, probe in probes.items():
            _keys(probe, {"train", "eval", "role_swap_margin", "one_arg_substitution_margin", "per_schema", "threshold"}, f"probe {feature_name}/{kind}")
            for split in ("train", "eval"):
                binary = _keys(probe[split], binary_keys, f"binary {feature_name}/{kind}/{split}")
                if len(binary["reliability_bins"]) != 10:
                    raise ValueError("reliability bin population drift")
                for item in binary["reliability_bins"]:
                    _keys(item, {"lower", "upper", "upper_inclusive", "count", "mean_probability", "positive_rate"}, "reliability bin")
            for margin in ("role_swap_margin", "one_arg_substitution_margin"):
                _keys(probe[margin], distribution_keys, f"margin {margin}")
            per_schema = _keys(probe["per_schema"], set(ranking_module.SCHEMAS), "probe per-schema")
            for item in per_schema.values():
                _keys(item, per_schema_keys, "probe per-schema binary")
    detail_keys = {"manifest_index", "group", "problem", "step", "action", "category", "label", "split", "logits", "control_logits"}
    logits = {f"{feature}/{kind}" for feature in ("A_action", "B_graph_action", "C_selected_graph_action", "D_raw_symbolic", "E_hybrid") for kind in ("linear", "mlp")}
    controls = {f"{feature}/mlp" for feature in ("A_action", "B_graph_action", "C_selected_graph_action", "D_raw_symbolic", "E_hybrid")}
    if type(details) is not list or len(details) != 604:
        raise ValueError("recoverability detail population drift")
    manifest_records, manifest_identity = load_and_validate_candidate_manifest(ranking_module.FIXED_CANDIDATE_MANIFEST)
    for index, row in enumerate(details):
        _keys(row, detail_keys, "recoverability detail")
        _keys(row["logits"], logits, "recoverability logits")
        _keys(row["control_logits"], controls, "recoverability controls")
        ranking_module._require_action(row["action"])
        expected_record = manifest_records[index]
        expected_split = "train" if expected_record["group"] in ranking_module.TRAIN_GROUPS else "eval"
        if (row["manifest_index"] != index or type(row["manifest_index"]) is not int or type(row["label"]) is not bool
                or row["split"] != expected_split or row["group"] != expected_record["group"]
                or row["problem"] != expected_record["problem"] or row["step"] != expected_record["step"]
                or row["action"] != expected_record["action"] or row["category"] != expected_record["category"]
                or row["label"] != expected_record["applicability_label"]
                or any(not ranking_module._finite(value) for value in (*row["logits"].values(), *row["control_logits"].values()))):
            raise ValueError("recoverability detail fixed manifest/type/logit drift")
    feature = _keys(artifacts["feature_schema.json"], {"schema_version", "candidate_manifest_sha256", "feature_sets"}, "feature schema")
    if (feature["schema_version"] != "action_latent_updated_phase0.feature_schema.v1"
            or feature["candidate_manifest_sha256"] != manifest_identity["sha256"]
            or feature["feature_sets"] != recoverability_module.feature_schemas()):
        raise ValueError("feature schema identity/content drift")
    if artifacts["split_manifest.json"] != {"eval_groups": list(ranking_module.EVAL_GROUPS), "train_groups": list(ranking_module.TRAIN_GROUPS)}:
        raise ValueError("recoverability split drift")
    states = _keys(artifacts["probe_states.json"], {"schema_version", "candidate_manifest_sha256", "split_manifest_sha256", "training", "models"}, "probe states")
    training = _keys(states["training"], {"seed", "epochs", "learning_rate", "hidden_dim", "optimizer", "dtype"}, "probe training")
    expected_training = {"seed": 20260717, "epochs": 200, "learning_rate": .001, "hidden_dim": 64,
                         "optimizer": "Adam(lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)",
                         "dtype": "torch.float32"}
    if (states["schema_version"] != "action_latent_updated_phase0.probe_states.v1"
            or states["candidate_manifest_sha256"] != manifest_identity["sha256"]
            or states["split_manifest_sha256"] != ranking_module.SPLIT_SHA256
            or training != expected_training):
        raise ValueError("probe-state identity/training contract drift")
    expected_inventory = [(schema["name"], kind, schema["dimension"]) for schema in schemas for kind in ("linear", "mlp", "control_mlp")]
    if len(states["models"]) != 15 or [(state.get("feature_set"), state.get("model_kind"), state.get("input_dim")) for state in states["models"]] != expected_inventory:
        raise ValueError("probe-state fixed inventory/order/dimensions drift")
    for state in states["models"]:
        _keys(state, {"feature_set", "model_kind", "input_dim", "architecture", "preprocessing", "state_dict"}, "probe state")
        preprocessing = _keys(state["preprocessing"], {"mean", "std", "binary_indices", "standardized_indices", "zero_std_indices"}, "probe preprocessing")
        if any(type(value) is not list for value in preprocessing.values()) or len(preprocessing["mean"]) != state["input_dim"] or len(preprocessing["std"]) != state["input_dim"]:
            raise ValueError("probe preprocessing dimensions drift")
        if state["model_kind"] == "linear":
            expected_architecture = {"name", "input_dim", "output_dim", "bias"}
        else:
            expected_architecture = {"name", "input_dim", "hidden_dim", "output_dim", "activation", "bias"}
        _keys(state["architecture"], expected_architecture, "probe architecture")
        records = state["state_dict"]
        if type(records) is not list or [record.get("name") for record in records] != sorted(record.get("name") for record in records):
            raise ValueError("probe tensor order drift")
        for record in records:
            _keys(record, {"name", "shape", "dtype", "values"}, "probe tensor")
            if record["dtype"] != "torch.float32" or type(record["shape"]) is not list or any(type(size) is not int or size < 0 for size in record["shape"]):
                raise ValueError("probe tensor schema drift")
            if math.prod(record["shape"]) != len(record["values"]) or any(not ranking_module._finite(value) for value in record["values"]):
                raise ValueError("probe tensor values/shape drift")
        recoverability_module.reconstruct_probe(state)

    train_indices = [index for index, row in enumerate(details) if row["split"] == "train"]
    eval_indices = [index for index, row in enumerate(details) if row["split"] == "eval"]
    labels_tensor = torch.tensor([row["label"] for row in details], dtype=torch.float32)
    train_labels, eval_labels = labels_tensor[train_indices], labels_tensor[eval_indices]
    control_labels = recoverability_module.control_permutation(train_labels, seed=20260717)
    compact_rows = [{"group": row["group"], "category": row["category"], "schema": row["action"]["name"]} for row in details]
    train_rows = [compact_rows[index] for index in train_indices]
    eval_rows = [compact_rows[index] for index in eval_indices]
    rebuilt_features: dict[str, Any] = {}
    for feature_name in expected_settings["feature_sets"]:
        rebuilt_features[feature_name] = {}
        for kind in ("linear", "mlp", "control_mlp"):
            field = "control_logits" if kind == "control_mlp" else "logits"
            key = f"{feature_name}/{'mlp' if kind == 'control_mlp' else kind}"
            values = torch.tensor([row[field][key] for row in details], dtype=torch.float64)
            rebuilt_features[feature_name][kind] = recoverability_module.probe_report(
                values[train_indices], values[eval_indices], train_labels, eval_labels,
                train_rows, eval_rows, threshold_labels=control_labels if kind == "control_mlp" else train_labels,
            )
    if rebuilt_features != metrics["features"]:
        raise ValueError("recoverability metrics do not exactly recompute from details")
    latent = recoverability_module._separable(rebuilt_features["C_selected_graph_action"]["mlp"], rebuilt_features["C_selected_graph_action"]["control_mlp"])
    raw = recoverability_module._separable(rebuilt_features["D_raw_symbolic"]["mlp"], rebuilt_features["D_raw_symbolic"]["control_mlp"])
    hybrid = recoverability_module._separable(rebuilt_features["E_hybrid"]["mlp"], rebuilt_features["E_hybrid"]["control_mlp"])
    expected_verdicts = {"latent_separable": latent, "raw_separable": raw, "hybrid_separable": hybrid,
                         "label_or_sampling_blocker": not raw, "latent_state_bottleneck": raw and not latent,
                         "any_abc_separable": any(recoverability_module._separable(rebuilt_features[name][kind], rebuilt_features[name]["control_mlp"]) for name in ("A_action", "B_graph_action", "C_selected_graph_action") for kind in ("linear", "mlp"))}
    if metrics["verdicts"] != expected_verdicts:
        raise ValueError("recoverability verdict formulas drift")


def validate_diagnostic_artifacts(
    kind: str,
    artifacts: Mapping[str, Any],
    *,
    summary_path: Path | None = None,
    manifest_records: Sequence[Mapping[str, Any]] | None = None,
    reconciled_transitions: Sequence[Any] | None = None,
) -> None:
    if set(artifacts) != {"summary.json", *STAGE_FILES[kind]}:
        raise ValueError("diagnostic artifact inventory drift")
    if kind == "schema_residual":
        _validate_schema_residual(artifacts, summary_path=summary_path)
    elif kind == "applicability_recoverability":
        _validate_recoverability(artifacts, summary_path=summary_path, records=manifest_records)
    elif kind == "transition_equivalence":
        _validate_common_summary(artifacts["summary.json"], kind, summary_path=summary_path)
        transition_module.validate_summary(
            artifacts["summary.json"], artifacts["details.json"],
            manifest_records=manifest_records, reconciled_transitions=reconciled_transitions,
        )
    elif kind == "candidate_ranking":
        _validate_common_summary(artifacts["summary.json"], kind, summary_path=summary_path)
        ranking_module.validate_ranking_artifacts(artifacts, manifest_rows=manifest_records)
    else:
        raise ValueError("unknown diagnostic kind")
    _finite_tree(artifacts)


def repeat_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(summary))
    for key in ("checkpoint", "output", "device", "runtime_seconds"):
        projected.pop(key, None)
    environment = projected.get("environment")
    if isinstance(environment, dict):
        environment.pop("torch_version", None)
        environment.pop("platform", None)
    return projected


def validate_sibling_inventory(run1: Path, run2: Path, siblings: Sequence[str]) -> list[str]:
    expected = {"summary.json", *siblings}
    for root in (run1, run2):
        actual = {path.name for path in root.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError(f"repeat sibling inventory mismatch: {root}")
    for name in siblings:
        if (run1 / name).read_bytes() != (run2 / name).read_bytes():
            raise ValueError(f"repeat sibling bytes differ: {name}")
    return list(siblings)


def _separable(recoverability: Mapping[str, Any], feature: str) -> bool:
    probes = recoverability["metrics"]["features"][feature]
    report, control = probes["mlp"], probes["control_mlp"]
    return bool(
        report["eval"]["auroc"] >= .80 and report["eval"]["average_precision"] >= .35
        and report["role_swap_margin"]["median"] > 0
        and report["one_arg_substitution_margin"]["median"] > 0
        and control["eval"]["auroc"] <= .70
    )


def compute_decision_booleans(
    baseline_schema: Mapping[str, Any],
    phase2_schema: Mapping[str, Any],
    phase2_recoverability: Mapping[str, Any],
    phase2_transition: Mapping[str, Any],
    phase2_ranking: Mapping[str, Any],
) -> dict[str, bool]:
    baseline_rank = baseline_schema["metrics"]["state_schema_residual"]["pooled"]["effective_rank"]
    phase2_rank = phase2_schema["metrics"]["state_schema_residual"]["pooled"]["effective_rank"]
    baseline_fraction = baseline_schema["metrics"]["raw_variance_decomposition"]["within_schema_fraction"]
    phase2_fraction = phase2_schema["metrics"]["raw_variance_decomposition"]["within_schema_fraction"]
    transition = phase2_transition["metrics"]
    expected_mostly = transition["equivalence_rate"] >= .50
    if type(transition["mostly_transition_equivalent"]) is not bool or transition["mostly_transition_equivalent"] != expected_mostly:
        raise ValueError("serialized mostly-transition-equivalent verdict is inconsistent")
    ranking = phase2_ranking["metrics"]
    for name in ranking_module.SCORERS:
        expected_deployable = name != "latent_transition"
        if ranking[name]["deployable"] is not expected_deployable:
            raise ValueError("ranking deployability verdict drift")
    return {
        "representation_ok": bool(
            phase2_rank >= 4.0 and phase2_fraction >= .001
            and phase2_rank >= baseline_rank and phase2_fraction >= baseline_fraction
        ),
        "latent_separable": _separable(phase2_recoverability, "C_selected_graph_action"),
        "hybrid_separable": _separable(phase2_recoverability, "E_hybrid"),
        "raw_separable": _separable(phase2_recoverability, "D_raw_symbolic"),
        "latent_rank": bool(
            ranking["latent_applicability"]["ranks_applicable"]
            or ranking["role_object"]["ranks_applicable"]
        ),
        "raw_rank": bool(ranking["raw_symbolic"]["ranks_applicable"]),
        "hybrid_rank": bool(ranking["hybrid"]["ranks_applicable"]),
        "mostly_transition_equivalent": expected_mostly,
        "transition_distinguishable": bool(
            transition["equivalence_rate"] < .50 and transition["error_margin"]["median"] > 0
        ),
    }


def select_action(booleans: Mapping[str, bool]):
    if set(booleans) != set(DECISION_BOOLEAN_KEYS) or any(type(value) is not bool for value in booleans.values()):
        raise ValueError("decision booleans have unknown/missing keys or non-Boolean values")
    clauses = (
        ("FIX_DATA_LABEL_CONSTRUCTION", "not raw_separable", not booleans["raw_separable"]),
        ("BRANCH_D_ABSTRACT_ACTIONS", "mostly_transition_equivalent", booleans["mostly_transition_equivalent"]),
        ("CONTINUE_PHASE1_MINIMAL_SCHEMA_RANK", "representation_ok and latent_separable and hybrid_separable and latent_rank and transition_distinguishable", booleans["representation_ok"] and booleans["latent_separable"] and booleans["hybrid_separable"] and booleans["latent_rank"] and booleans["transition_distinguishable"]),
        ("BRANCH_B_DISCRETE_CANDIDATE_PLANNING", "latent_rank and transition_distinguishable", booleans["latent_rank"] and booleans["transition_distinguishable"]),
        ("BRANCH_A_EXPLICIT_STATE_ACTION_SCORER", "not latent_rank and raw_separable and hybrid_separable and hybrid_rank", not booleans["latent_rank"] and booleans["raw_separable"] and booleans["hybrid_separable"] and booleans["hybrid_rank"]),
        ("BRANCH_C_STATE_ENCODER_REDESIGN", "raw_separable and raw_rank and not hybrid_rank", booleans["raw_separable"] and booleans["raw_rank"] and not booleans["hybrid_rank"]),
        ("BRANCH_D_ABSTRACT_ACTIONS", "otherwise", True),
    )
    trace = []
    for number, (action, predicate, matched) in enumerate(clauses, 1):
        trace.append({"clause": number, "action": action, "predicate": predicate, "matched": bool(matched)})
        if matched:
            return action, trace
    raise AssertionError("exhaustive precedence did not select an action")


def reconcile_ranking(rows, metrics, independently_reconstructed_vectors) -> None:
    if set(independently_reconstructed_vectors) != set(ranking_module.SCORERS):
        raise ValueError("independent score-vector inventory drift")
    if any(len(values) != len(rows) for values in independently_reconstructed_vectors.values()):
        raise ValueError("independent score-vector population drift")
    for index, row in enumerate(rows):
        for scorer in ranking_module.SCORERS:
            expected = independently_reconstructed_vectors[scorer][index]
            if type(expected) not in (int, float) or not math.isfinite(float(expected)):
                raise ValueError("independently reconstructed score is non-finite")
            if row["scores"][scorer] != expected:
                raise ValueError(f"score mismatch for {scorer} at row {index}")
    ranking_module.validate_rank_reconciliation(rows, metrics)


def atomic_write_directory(
    destination: Path,
    artifacts: Mapping[str, Any],
    *,
    validator: Callable[[Path], Any],
) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name, value in artifacts.items():
            path = staging / name
            if isinstance(value, str):
                path.write_text(value, encoding="utf-8")
            else:
                path.write_bytes(canonical_json_bytes(value))
        validator(staging)
        if destination.exists():
            raise FileExistsError(f"destination appeared during staging: {destination}")
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_evidence_manifest(files: Sequence[tuple[Path, str]]) -> dict[str, Any]:
    entries = []
    seen = set()
    for path, role in files:
        if path.name == "evidence_manifest.json":
            raise ValueError("evidence manifest cannot include its self hash")
        if role not in EVIDENCE_ROLES:
            raise ValueError(f"unknown evidence role: {role}")
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError("duplicate evidence-manifest path")
        seen.add(resolved)
        raw = path.read_bytes()
        entries.append({"path": str(resolved), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "role": role})
    entries.sort(key=lambda entry: entry["path"])
    return {"schema_version": "action_latent_updated_phase0.evidence_manifest.v1", "entries": entries}


def validate_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    staged_generated: Mapping[str, bytes] | None = None,
    expected_paths: set[str] | None = None,
    expected_roles: Mapping[str, str] | None = None,
) -> None:
    _keys(manifest, {"schema_version", "entries"}, "evidence manifest")
    if manifest["schema_version"] != "action_latent_updated_phase0.evidence_manifest.v1" or type(manifest["entries"]) is not list:
        raise ValueError("evidence manifest schema drift")
    paths = []
    for entry in manifest["entries"]:
        _keys(entry, {"path", "bytes", "sha256", "role"}, "evidence entry")
        if type(entry["path"]) is not str or type(entry["bytes"]) is not int or entry["bytes"] < 0 or entry["role"] not in EVIDENCE_ROLES:
            raise ValueError("evidence entry type/role drift")
        if type(entry["sha256"]) is not str or len(entry["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in entry["sha256"]):
            raise ValueError("evidence entry hash is malformed")
        if Path(entry["path"]).name == "evidence_manifest.json":
            raise ValueError("evidence manifest self hash is forbidden")
        paths.append(entry["path"])
        raw = (staged_generated or {}).get(entry["path"])
        if raw is None and Path(entry["path"]).is_file():
            raw = Path(entry["path"]).read_bytes()
        if expected_paths is not None and raw is None:
            raise ValueError("expected evidence entry file is missing")
        if raw is not None and (len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]):
            raise ValueError("evidence entry bytes/hash drift")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("evidence entries must be path-sorted and unique")
    if expected_paths is not None and set(paths) != expected_paths:
        raise ValueError("evidence manifest is incomplete or has extra paths")
    if expected_roles is not None and {entry["path"]: entry["role"] for entry in manifest["entries"]} != dict(expected_roles):
        raise ValueError("evidence manifest role inventory drift")


def validate_ranking_recoverability_paths(identities: Mapping[str, Any], expected_summary: Path) -> None:
    filenames = {
        "summary": "summary.json", "details": "details.json", "feature_schema": "feature_schema.json",
        "split_manifest": "split_manifest.json", "probe_states": "probe_states.json",
    }
    if set(identities) != set(filenames):
        raise ValueError("ranking-bound recoverability inventory drift")
    for name, filename in filenames.items():
        item = identities[name]
        if not isinstance(item, Mapping) or type(item.get("path")) is not str or Path(item["path"]) != expected_summary.parent / filename:
            raise ValueError("ranking-bound recoverability path differs from assessor supplied run1")


ASSESSMENT_KEYS = {"schema_version", "kind", "input_identities", "repeatability", "stage_verdicts", "decision_booleans", "selected_action", "precedence_trace", "compact_metrics", "output"}
COMPACT_KEYS = {"residual_effective_rank_baseline", "residual_effective_rank_phase2", "within_schema_fraction_baseline", "within_schema_fraction_phase2", "latent_auroc", "latent_ap", "latent_role_swap_margin", "latent_one_arg_margin", "raw_auroc", "raw_ap", "hybrid_auroc", "hybrid_ap", "transition_equivalence_rate", "transition_error_margin_median", "ranking_baseline", "ranking_phase2"}


def validate_assessment_summary(
    summary: Mapping[str, Any],
    *,
    production_paths: Mapping[str, Path] | None = None,
    production_output: Path | None = None,
) -> None:
    _keys(summary, ASSESSMENT_KEYS, "assessment summary")
    if summary["schema_version"] != "action_latent_updated_phase0.assessment.v1" or summary["kind"] != "updated_phase0_assessment":
        raise ValueError("assessment schema/kind drift")
    if set(summary["input_identities"]) != set(REQUIRED_PATH_ARGUMENTS):
        raise ValueError("assessment input identity inventory drift")
    for name, identity in summary["input_identities"].items():
        required = {"path", "bytes", "sha256", "count"} if name in {"corpus_manifest", "candidate_manifest"} else {"path", "bytes", "sha256"}
        _keys(identity, required, f"input identity {name}")
        if (type(identity["path"]) is not str or not Path(identity["path"]).is_absolute()
                or type(identity["bytes"]) is not int or identity["bytes"] < 0
                or type(identity["sha256"]) is not str or len(identity["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in identity["sha256"])):
            raise ValueError("assessment input identity type/path/hash drift")
        if "count" in identity:
            expected_count = 12 if name == "corpus_manifest" else 604
            if type(identity["count"]) is not int or identity["count"] != expected_count:
                raise ValueError("assessment input identity count drift")
        if production_paths is not None:
            path = production_paths[name]
            expected_count = 12 if name == "corpus_manifest" else 604 if name == "candidate_manifest" else None
            if Path(identity["path"]) != path or _identity(path, count=expected_count) != identity:
                raise ValueError("assessment input identity actual file drift")
    _keys(summary["repeatability"], {"baseline_schema", "phase2_schema", "baseline_recoverability", "phase2_recoverability", "baseline_transition", "phase2_transition", "baseline_ranking", "phase2_ranking"}, "repeatability")
    expected_repeat_files = {
        "baseline_schema": ["details.json"], "phase2_schema": ["details.json"],
        "baseline_recoverability": ["details.json", "feature_schema.json", "split_manifest.json", "probe_states.json"],
        "phase2_recoverability": ["details.json", "feature_schema.json", "split_manifest.json", "probe_states.json"],
        "baseline_transition": ["details.json"], "phase2_transition": ["details.json"],
        "baseline_ranking": ["details.json", "split_manifest.json", "role_probe_state.json"],
        "phase2_ranking": ["details.json", "split_manifest.json", "role_probe_state.json"],
    }
    for name, item in summary["repeatability"].items():
        _keys(item, {"summary_projection_equal", "sibling_files_equal", "files_checked"}, "repeatability item")
        if (item["summary_projection_equal"] is not True or item["sibling_files_equal"] is not True
                or item["files_checked"] != expected_repeat_files[name]):
            raise ValueError("repeatability strict types drift")
    _keys(summary["stage_verdicts"], {"evidence", "residual", "recoverability", "transition_equivalence", "ranking"}, "stage verdicts")
    if any(value not in {"PASS", "FAIL"} for value in summary["stage_verdicts"].values()):
        raise ValueError("stage verdict literal drift")
    if set(summary["decision_booleans"]) != set(DECISION_BOOLEAN_KEYS) or any(type(value) is not bool for value in summary["decision_booleans"].values()):
        raise ValueError("decision Boolean schema drift")
    selected, trace = select_action(summary["decision_booleans"])
    if summary["selected_action"] != selected or summary["precedence_trace"] != trace:
        raise ValueError("assessment precedence drift")
    _keys(summary["compact_metrics"], COMPACT_KEYS, "compact metrics")
    rank_keys = {"auroc", "average_precision", "top1_applicable_rate", "pairwise_applicable_accuracy", "ranks_applicable"}
    for key, value in summary["compact_metrics"].items():
        if key in {"ranking_baseline", "ranking_phase2"}:
            _keys(value, set(ranking_module.SCORERS), key)
            for scorer in value.values():
                _keys(scorer, rank_keys, "compact ranking scorer")
                if any(type(scorer[name]) not in (int, float) or not math.isfinite(float(scorer[name])) for name in rank_keys - {"ranks_applicable"}) or type(scorer["ranks_applicable"]) is not bool:
                    raise ValueError("compact ranking scorer type drift")
        elif type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("compact metric type drift")
    if type(summary["output"]) is not str or not Path(summary["output"]).is_absolute() or (production_output is not None and Path(summary["output"]) != production_output):
        raise ValueError("assessment output literal drift")
    _finite_tree(summary)


def assessment_schema_fixture(*, booleans, selected, precedence, output: Path) -> dict[str, Any]:
    """Return a structurally complete neutral assessment object for validator tests."""
    identities = {}
    for name in REQUIRED_PATH_ARGUMENTS:
        identities[name] = {"path": f"/{name}", "bytes": 1, "sha256": "a" * 64}
        if name in {"corpus_manifest", "candidate_manifest"}:
            identities[name]["count"] = 12 if name == "corpus_manifest" else 604
    repeat_files = {"schema": ["details.json"], "recoverability": ["details.json", "feature_schema.json", "split_manifest.json", "probe_states.json"], "transition": ["details.json"], "ranking": ["details.json", "split_manifest.json", "role_probe_state.json"]}
    repeat = {f"{variant}_{stage}": {"summary_projection_equal": True, "sibling_files_equal": True, "files_checked": repeat_files[stage]} for variant in ("baseline", "phase2") for stage in ("schema", "recoverability", "transition", "ranking")}
    scorer_metrics = {name: {"auroc": 0.0, "average_precision": 0.0, "top1_applicable_rate": 0.0, "pairwise_applicable_accuracy": 0.0, "ranks_applicable": False} for name in ranking_module.SCORERS}
    compact = {key: 0.0 for key in COMPACT_KEYS - {"ranking_baseline", "ranking_phase2"}}
    compact.update(ranking_baseline=copy.deepcopy(scorer_metrics), ranking_phase2=copy.deepcopy(scorer_metrics))
    return {"schema_version": "action_latent_updated_phase0.assessment.v1", "kind": "updated_phase0_assessment", "input_identities": identities, "repeatability": repeat, "stage_verdicts": {name: "PASS" for name in ("evidence", "residual", "recoverability", "transition_equivalence", "ranking")}, "decision_booleans": dict(booleans), "selected_action": selected, "precedence_trace": precedence, "compact_metrics": compact, "output": str(output)}


def _ranking_command(variant: str, repeat: str) -> str:
    checkpoint = ranking_module.BASELINE_CHECKPOINT if variant == "baseline" else ranking_module.PHASE2_CHECKPOINT
    root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0")
    recoverability = root / "recoverability" / variant / "run1"
    return " ".join(("UV_CACHE_DIR=/opt/data/workspace/.uv-cache", "uv run --package acs-jepa-cli python", "script/diagnose_action_candidate_ranking.py", str(ranking_module.DATASET), "--checkpoint", str(checkpoint), "--candidate-manifest", str(ranking_module.FIXED_CANDIDATE_MANIFEST), *sum(([f"--recoverability-{name.replace('_', '-')}", str(recoverability / filename)] for name, filename in (("summary", "summary.json"), ("details", "details.json"), ("feature_schema", "feature_schema.json"), ("split_manifest", "split_manifest.json"), ("probe_states", "probe_states.json"))), []), "--output", str(root / "candidate_ranking" / variant / repeat), "--device cpu --split val --epochs 200 --learning-rate 0.001 --hidden-dim 64 --seed 20260717"))


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    validate_assessment_summary(summary)
    lines = ["# Updated Phase 0 assessment", "", "## Compact metrics", ""]
    for key, value in summary["compact_metrics"].items():
        lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(("", "## Stage verdicts", ""))
    lines.extend(f"- `{key}`: **{value}**" for key, value in summary["stage_verdicts"].items())
    lines.extend(("", "## Decision booleans", ""))
    lines.extend(f"- `{key}`: `{str(value).lower()}`" for key, value in summary["decision_booleans"].items())
    lines.extend(("", "## Evaluated precedence", ""))
    for item in summary["precedence_trace"]:
        reason = "selected; later clauses skipped" if item["matched"] else "not selected because predicate is false"
        lines.append(f"{item['clause']}. `{item['action']}` — `{item['predicate']}` = `{str(item['matched']).lower()}` ({reason}).")
    selected_clause = summary["precedence_trace"][-1]["clause"]
    for clause in range(selected_clause + 1, 8):
        lines.append(f"{clause}. **SKIPPED** — later clause not evaluated after clause {selected_clause} selected the action.")
    lines.extend(("", f"Selected research action: `{summary['selected_action']}`", "", "## Exact commands", ""))
    for variant in ("baseline", "phase2"):
        for repeat in ("run1", "run2"):
            lines.extend(("```bash", _ranking_command(variant, repeat), "```"))
    lines.extend(("```bash", "UV_CACHE_DIR=/opt/data/workspace/.uv-cache uv run --package acs-jepa-cli python script/assess_action_latent_updated_phase0.py " + " ".join(f"--{name.replace('_', '-')} {summary['input_identities'][name]['path']}" for name in REQUIRED_PATH_ARGUMENTS) + f" --output {summary['output']}", "```", "", "## Input artifacts", ""))
    lines.extend(f"- `{identity['path']}` — `{identity['sha256']}` ({identity['bytes']} bytes)" for identity in summary["input_identities"].values())
    return "\n".join(lines) + "\n"


def _identity(path: Path, *, count: int | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    result = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if count is not None:
        result["count"] = count
    return result


def _validate_fixed_paths(args: argparse.Namespace) -> None:
    for name, expected in FIXED_PATHS.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{name} must use the literal fixed path")
    fixed_root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0")
    expected_stage_paths = {
        "baseline_schema_run1": fixed_root / "schema_residual/baseline/run1/summary.json",
        "baseline_schema_run2": fixed_root / "schema_residual/baseline/run2/summary.json",
        "phase2_schema_run1": fixed_root / "schema_residual/phase2/run1/summary.json",
        "phase2_schema_run2": fixed_root / "schema_residual/phase2/run2/summary.json",
        "baseline_recoverability_run1": fixed_root / "recoverability/baseline/run1/summary.json",
        "baseline_recoverability_run2": fixed_root / "recoverability/baseline/run2/summary.json",
        "phase2_recoverability_run1": fixed_root / "recoverability/phase2/run1/summary.json",
        "phase2_recoverability_run2": fixed_root / "recoverability/phase2/run2/summary.json",
        "baseline_transition_run1": fixed_root / "transition_equivalence/baseline/run1/summary.json",
        "baseline_transition_run2": fixed_root / "transition_equivalence/baseline/run2/summary.json",
        "phase2_transition_run1": fixed_root / "transition_equivalence/phase2/run1/summary.json",
        "phase2_transition_run2": fixed_root / "transition_equivalence/phase2/run2/summary.json",
        "baseline_ranking_run1": fixed_root / "candidate_ranking/baseline/run1/summary.json",
        "baseline_ranking_run2": fixed_root / "candidate_ranking/baseline/run2/summary.json",
        "phase2_ranking_run1": fixed_root / "candidate_ranking/phase2/run1/summary.json",
        "phase2_ranking_run2": fixed_root / "candidate_ranking/phase2/run2/summary.json",
    }
    for name, expected in expected_stage_paths.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{name} must use the literal fixed diagnostic path")
    if args.output != fixed_root / "assessment":
        raise ValueError("assessment output must use the literal fixed destination")


def _validate_pair(
    first_path: Path,
    second_path: Path,
    siblings: Sequence[str],
    *,
    manifest_records: Sequence[Mapping[str, Any]] | None = None,
    reconciled_transitions: Sequence[Any] | None = None,
):
    first = load_canonical_json(first_path)
    second = load_canonical_json(second_path)
    kind = first.get("kind") if isinstance(first, Mapping) else None
    if kind not in STAGE_FILES or tuple(siblings) != STAGE_FILES[kind]:
        raise ValueError("diagnostic kind/sibling contract drift")
    first_artifacts = load_diagnostic_artifacts(first_path, kind)
    second_artifacts = load_diagnostic_artifacts(second_path, kind)
    validate_diagnostic_artifacts(kind, first_artifacts, summary_path=first_path, manifest_records=manifest_records, reconciled_transitions=reconciled_transitions)
    validate_diagnostic_artifacts(kind, second_artifacts, summary_path=second_path, manifest_records=manifest_records, reconciled_transitions=reconciled_transitions)
    if repeat_projection(first) != repeat_projection(second):
        raise ValueError("diagnostic summary repeat projection differs")
    checked = validate_sibling_inventory(first_path.parent, second_path.parent, siblings)
    return first, {"summary_projection_equal": True, "sibling_files_equal": True, "files_checked": checked}


def _ranking_vectors_from_bound_artifacts(summary_path: Path, expected_recoverability_summary: Path):
    summary = load_canonical_json(summary_path)
    details = load_canonical_json(summary_path.parent / "details.json")
    state = load_canonical_json(summary_path.parent / "role_probe_state.json")
    recoverability = load_canonical_json(Path(summary["settings"]["recoverability_inputs"]["details"]["path"]))
    eval_recoverability = [row for row in recoverability if row["split"] == "eval"]
    if len(eval_recoverability) != 151:
        raise ValueError("accepted recoverability eval population drift")
    # Rebuild all Stage 0B feature tensors, reconstruct all fifteen probes, and
    # independently verify the complete 604-row accepted logit table.
    checkpoint = Path(summary["checkpoint"])
    extraction_args = argparse.Namespace(
        dataset_dir=ranking_module.DATASET, checkpoint=checkpoint, device="cpu",
        split="val", seed=20260717, recoverability_summary=expected_recoverability_summary,
    )
    records, _candidate_identity = load_and_validate_candidate_manifest(ranking_module.FIXED_CANDIDATE_MANIFEST)
    recoverability_paths = summary["settings"]["recoverability_inputs"]
    validate_ranking_recoverability_paths(recoverability_paths, expected_recoverability_summary)
    recoverability_artifacts = {
        name: load_canonical_json(Path(identity["path"])) for name, identity in recoverability_paths.items()
    }
    for name, identity in recoverability_paths.items():
        if _identity(Path(identity["path"])) != identity:
            raise ValueError(f"ranking-bound recoverability identity changed: {name}")
    ranking_module._validate_recoverability_evidence(extraction_args, records, recoverability_artifacts)

    # Imported score vectors are independently sourced from accepted Stage 0B rows.
    vectors = {
        "latent_applicability": [row["logits"]["C_selected_graph_action/mlp"] for row in eval_recoverability],
        "raw_symbolic": [row["logits"]["D_raw_symbolic/mlp"] for row in eval_recoverability],
        "hybrid": [row["logits"]["E_hybrid/mlp"] for row in eval_recoverability],
    }
    # Strictly restore the checkpoint and independently rebuild candidate,
    # source, object-bank, active-role, and recorded-successor score paths.
    candidates, restoration = ranking_module._extract_candidates(extraction_args, records)
    if restoration != summary["checkpoint_restoration"]:
        raise ValueError("ranking checkpoint restoration drift")
    eval_candidates = [candidate for candidate in candidates if candidate.manifest_record["group"] in ranking_module.EVAL_GROUPS]
    eval_tensors, eval_slices = ranking_module.stack_role_candidates(eval_candidates)
    model = ranking_module.reconstruct_role_probe(state)
    vectors["role_object"] = ranking_module.role_candidate_scores(model, eval_tensors, eval_slices)
    vectors["latent_transition"] = [candidate.transition_score for candidate in eval_candidates]
    reconcile_ranking(details, summary["metrics"], vectors)
    return summary, details


def reconstruct_assessment(
    summaries: Mapping[str, Mapping[str, Any]],
    baseline_ranking: Mapping[str, Any],
    phase2_ranking: Mapping[str, Any],
    identities: Mapping[str, Any],
    repeatability: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Deterministically derive every decision-bearing assessment field."""
    booleans = compute_decision_booleans(
        summaries["baseline_schema"], summaries["phase2_schema"], summaries["phase2_recoverability"],
        summaries["phase2_transition"], phase2_ranking,
    )
    selected, precedence = select_action(booleans)
    stage_verdicts = {
        "evidence": "PASS",
        "residual": "PASS" if booleans["representation_ok"] else "FAIL",
        "recoverability": "PASS" if booleans["raw_separable"] and (booleans["latent_separable"] or booleans["hybrid_separable"]) else "FAIL",
        "transition_equivalence": "PASS" if booleans["transition_distinguishable"] else "FAIL",
        "ranking": "PASS" if any(phase2_ranking["metrics"][name]["ranks_applicable"] for name in ranking_module.SCORERS[1:]) else "FAIL",
    }
    feature = summaries["phase2_recoverability"]["metrics"]["features"]
    compact = {
        "residual_effective_rank_baseline": summaries["baseline_schema"]["metrics"]["state_schema_residual"]["pooled"]["effective_rank"],
        "residual_effective_rank_phase2": summaries["phase2_schema"]["metrics"]["state_schema_residual"]["pooled"]["effective_rank"],
        "within_schema_fraction_baseline": summaries["baseline_schema"]["metrics"]["raw_variance_decomposition"]["within_schema_fraction"],
        "within_schema_fraction_phase2": summaries["phase2_schema"]["metrics"]["raw_variance_decomposition"]["within_schema_fraction"],
        "latent_auroc": feature["C_selected_graph_action"]["mlp"]["eval"]["auroc"],
        "latent_ap": feature["C_selected_graph_action"]["mlp"]["eval"]["average_precision"],
        "latent_role_swap_margin": feature["C_selected_graph_action"]["mlp"]["role_swap_margin"]["median"],
        "latent_one_arg_margin": feature["C_selected_graph_action"]["mlp"]["one_arg_substitution_margin"]["median"],
        "raw_auroc": feature["D_raw_symbolic"]["mlp"]["eval"]["auroc"],
        "raw_ap": feature["D_raw_symbolic"]["mlp"]["eval"]["average_precision"],
        "hybrid_auroc": feature["E_hybrid"]["mlp"]["eval"]["auroc"],
        "hybrid_ap": feature["E_hybrid"]["mlp"]["eval"]["average_precision"],
        "transition_equivalence_rate": summaries["phase2_transition"]["metrics"]["equivalence_rate"],
        "transition_error_margin_median": summaries["phase2_transition"]["metrics"]["error_margin"]["median"],
    }
    for label, ranking in (("ranking_baseline", baseline_ranking), ("ranking_phase2", phase2_ranking)):
        compact[label] = {name: {
            "auroc": ranking["metrics"][name]["binary"]["auroc"],
            "average_precision": ranking["metrics"][name]["binary"]["average_precision"],
            "top1_applicable_rate": ranking["metrics"][name]["top1_applicable_rate"],
            "pairwise_applicable_accuracy": ranking["metrics"][name]["pairwise_applicable_accuracy"],
            "ranks_applicable": ranking["metrics"][name]["ranks_applicable"],
        } for name in ranking_module.SCORERS}
    return {
        "schema_version": "action_latent_updated_phase0.assessment.v1", "kind": "updated_phase0_assessment",
        "input_identities": copy.deepcopy(dict(identities)), "repeatability": copy.deepcopy(dict(repeatability)),
        "stage_verdicts": stage_verdicts, "decision_booleans": booleans, "selected_action": selected,
        "precedence_trace": precedence, "compact_metrics": compact, "output": str(output),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_fixed_paths(args)
    if args.output.exists():
        raise FileExistsError(f"destination already exists: {args.output}")
    if not all(getattr(args, name).is_file() for name in REQUIRED_PATH_ARGUMENTS):
        raise ValueError("assessment input is missing")
    records, candidate_identity = load_and_validate_candidate_manifest(args.candidate_manifest)
    config, corpus, _bundle, _device, _restoration = transition_module.load_checkpoint_bundle(
        ranking_module.DATASET, ranking_module.BASELINE_CHECKPOINT, device_name="cpu", include_restoration_metadata=True,
    )
    selected_corpus = transition_module.select_split(corpus, config, "val", seed=20260717)
    transitions = transition_module.reconcile_transitions(records, selected_corpus)
    eval_manifest_rows = [
        {"manifest_index": index, "group": row["group"], "problem": row["problem"], "step": row["step"],
         "action": row["action"], "category": row["category"], "label": row["applicability_label"]}
        for index, row in enumerate(records) if row["group"] in ranking_module.EVAL_GROUPS
    ]
    root = args.output.parent
    marker = root / "root_identity.json"
    expected_root = recoverability_module._root_identity(candidate_identity)
    if not marker.is_file() or marker.read_bytes() != canonical_json_bytes(expected_root):
        raise ValueError("root identity marker does not match immutable inputs")
    for path, digest in recoverability_module.FIXED_SHA256.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"fixed Phase 0 input identity changed: {path}")
    pair_specs = (
        ("baseline_schema", args.baseline_schema_run1, args.baseline_schema_run2, ("details.json",)),
        ("phase2_schema", args.phase2_schema_run1, args.phase2_schema_run2, ("details.json",)),
        ("baseline_recoverability", args.baseline_recoverability_run1, args.baseline_recoverability_run2, ("details.json", "feature_schema.json", "split_manifest.json", "probe_states.json")),
        ("phase2_recoverability", args.phase2_recoverability_run1, args.phase2_recoverability_run2, ("details.json", "feature_schema.json", "split_manifest.json", "probe_states.json")),
        ("baseline_transition", args.baseline_transition_run1, args.baseline_transition_run2, ("details.json",)),
        ("phase2_transition", args.phase2_transition_run1, args.phase2_transition_run2, ("details.json",)),
        ("baseline_ranking", args.baseline_ranking_run1, args.baseline_ranking_run2, ("details.json", "split_manifest.json", "role_probe_state.json")),
        ("phase2_ranking", args.phase2_ranking_run1, args.phase2_ranking_run2, ("details.json", "split_manifest.json", "role_probe_state.json")),
    )
    summaries, repeatability = {}, {}
    for name, first, second, siblings in pair_specs:
        is_transition = "transition" in name
        is_ranking = "ranking" in name
        summaries[name], repeatability[name] = _validate_pair(
            first, second, siblings,
            manifest_records=eval_manifest_rows if is_ranking else records,
            reconciled_transitions=transitions if is_transition else None,
        )
    baseline_ranking, _ = _ranking_vectors_from_bound_artifacts(args.baseline_ranking_run1, args.baseline_recoverability_run1)
    phase2_ranking, _ = _ranking_vectors_from_bound_artifacts(args.phase2_ranking_run1, args.phase2_recoverability_run1)
    identities = {
        name: _identity(getattr(args, name), count=(12 if name == "corpus_manifest" else 604 if name == "candidate_manifest" else None))
        for name in REQUIRED_PATH_ARGUMENTS
    }
    summary = reconstruct_assessment(summaries, baseline_ranking, phase2_ranking, identities, repeatability, args.output)
    production_paths = {name: getattr(args, name) for name in REQUIRED_PATH_ARGUMENTS}
    validate_assessment_summary(summary, production_paths=production_paths, production_output=args.output)
    markdown = build_summary_markdown(summary)
    evidence_files = [
        (marker, "root_identity"),
        *[
            (getattr(args, name), "candidate_manifest" if name == "candidate_manifest" else "corpus_manifest" if name == "corpus_manifest" else "checkpoint" if "checkpoint" in name else "config" if "config" in name else "governing_spec" if name == "updated_spec" else "diagnostic_summary")
            for name in REQUIRED_PATH_ARGUMENTS
        ],
    ]
    supporting_roles = {
        "details.json": "diagnostic_details", "feature_schema.json": "feature_schema",
        "split_manifest.json": "split_manifest", "probe_states.json": "probe_states",
        "role_probe_state.json": "role_probe_state",
    }
    seen_evidence = {path.resolve() for path, _role in evidence_files}
    for _pair_name, first, second, siblings in pair_specs:
        for parent in (first.parent, second.parent):
            for sibling in siblings:
                path = parent / sibling
                if path.resolve() not in seen_evidence:
                    evidence_files.append((path, supporting_roles[sibling]))
                    seen_evidence.add(path.resolve())

    def validate_staging(root: Path):
        loaded = load_canonical_json(root / "summary.json")
        loaded_manifest = load_canonical_json(root / "evidence_manifest.json")
        reconstructed = reconstruct_assessment(summaries, baseline_ranking, phase2_ranking, identities, repeatability, args.output)
        validate_assessment_summary(loaded, production_paths=production_paths, production_output=args.output)
        validate_evidence_manifest(
            loaded_manifest,
            staged_generated={
                str((args.output / "summary.json").resolve()): (root / "summary.json").read_bytes(),
                str((args.output / "summary.md").resolve()): (root / "summary.md").read_bytes(),
            },
            expected_paths=expected_evidence_paths,
            expected_roles=expected_evidence_roles,
        )
        if (
            loaded != reconstructed
            or loaded_manifest != manifest
            or (root / "summary.md").read_text() != markdown
            or set(path.name for path in root.iterdir()) != {"summary.json", "summary.md", "evidence_manifest.json"}
        ):
            raise ValueError("staged assessment reread mismatch")

    manifest = build_evidence_manifest(evidence_files)
    generated = (
        (args.output / "summary.json", canonical_json_bytes(summary), "assessment_summary"),
        (args.output / "summary.md", markdown.encode("utf-8"), "assessment_markdown"),
    )
    manifest["entries"].extend(
        {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "role": role}
        for path, raw, role in generated
    )
    manifest["entries"].sort(key=lambda entry: entry["path"])
    expected_evidence_paths = {entry["path"] for entry in manifest["entries"]}
    expected_evidence_roles = {
        str(path.resolve()): role for path, role in evidence_files
    } | {str(path.resolve()): role for path, _raw, role in generated}
    validate_evidence_manifest(
        manifest,
        staged_generated={str(path.resolve()): raw for path, raw, _role in generated},
        expected_paths=expected_evidence_paths,
        expected_roles=expected_evidence_roles,
    )
    atomic_write_directory(args.output, {"summary.json": summary, "summary.md": markdown, "evidence_manifest.json": manifest}, validator=validate_staging)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
