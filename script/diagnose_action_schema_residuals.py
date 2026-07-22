"""Measure raw and schema-residual action-latent geometry on fixed offline evidence."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import torch
from acs_jepa.architectures import JEPALatentState
from action_latent_statistics import (
    nearest_candidate,
    raw_variance_decomposition,
    residual_statistics,
    schema_residuals,
    unit_normalize_zero,
)
from action_phase0_common import (
    POPULATION_COUNT,
    SCHEMAS,
    SPLIT_SHA256,
    canonical_json_bytes,
    encode_state,
    enumerate_same_schema_population,
    file_identity,
    load_and_validate_candidate_manifest,
    load_checkpoint_bundle,
    population_identity,
    prepare_output_directory,
    reconcile_manifest_source_states,
    select_split,
)

UPDATED_SPEC = Path("/opt/data/workspace/acs-jepa/script/ACTION_LATENT_UPDATED_SPEC.md")
BASELINE_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt")
BASELINE_CONFIG = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml")
PHASE2_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt")
PHASE2_CONFIG = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/config.yaml")
CORPUS_MANIFEST = Path("/opt/data/workspace/acs-jepa-tuning-data/smoke/manifest.json")
DATASET = Path("/opt/data/workspace/acs-jepa-tuning-data/smoke")
FIXED_CANDIDATE_MANIFEST = Path(
    "/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/"
    "baseline/probe_run1/example_manifest.json"
)
FIXED_SHA256 = {
    UPDATED_SPEC: "b4146d21b6082ec085628f7d1c56ff135c9fe606c8307db8b84689e449ec9606",
    BASELINE_CHECKPOINT: "65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e",
    BASELINE_CONFIG: "f65e2cbb33fb3e7322e0cc0c5e8a8f01e9ca7c408e4594516d50a9735c673193",
    PHASE2_CHECKPOINT: "7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968",
    PHASE2_CONFIG: "01c1ed90c51a89f79abc5097043cfe95cf59b6846f9afbfa50102e00472356a5",
    CORPUS_MANIFEST: "055b5616d7616331e6edbc8f72523f07e8c1808e5aa31089c8420f01aaf0e400",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--chunk-size", default=2048, type=int)
    parser.add_argument("--seed", default=20260717, type=int)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    validate_args(args)
    return args


def configure_determinism(seed: int) -> None:
    """Configure repeatable frozen extraction before any model/device work."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def validate_run_binding(checkpoint: Path, output: Path) -> None:
    """Bind each fixed checkpoint to its exact diagnostic variant and repeat slot."""

    if len(output.parts) < 3 or output.parts[-3] != "schema_residual":
        raise ValueError("output must end in schema_residual/{baseline|phase2}/{run1|run2}")
    variant, repeat = output.parts[-2:]
    if repeat not in {"run1", "run2"}:
        raise ValueError("schema residual destination must be run1 or run2")
    expected = {"baseline": BASELINE_CHECKPOINT, "phase2": PHASE2_CHECKPOINT}.get(variant)
    if expected is None or checkpoint != expected:
        raise ValueError("checkpoint/output binding does not match baseline or phase2 fixed evidence")


def _action_payload(key: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    return {"name": key[0], "arguments": list(key[1])}


def distance_diagnostics(
    raw_latents: torch.Tensor,
    state_residuals: torch.Tensor,
    action_keys: Sequence[tuple[str, tuple[str, ...]]],
    group_ids: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the two preregistered, strictly within-source distance diagnostics."""

    unit_residuals, zero_mask = unit_normalize_zero(state_residuals)
    details = []
    for source in sources:
        group = str(source["group"])
        trace_index = int(source["trace"])
        bucket = [index for index, candidate_group in enumerate(group_ids) if candidate_group == group]
        wrong = [index for index in bucket if action_keys[index] != action_keys[trace_index]]
        wrong_index, wrong_distance = nearest_candidate(raw_latents, trace_index, wrong, action_keys)
        invalid = [int(index) for index in source["invalid"]]
        invalid_index, invalid_distance = nearest_candidate(unit_residuals, trace_index, invalid, action_keys)
        details.append(
            {
                "group": group,
                "problem": str(source["problem"]),
                "step": int(source["step"]),
                "trace_action": _action_payload(action_keys[trace_index]),
                "full_candidate_count": len(bucket),
                "nearest_wrong_action": _action_payload(action_keys[wrong_index]),
                "nearest_wrong_raw_l2": wrong_distance,
                "invalid_manifest_candidate_count": len(invalid),
                "nearest_invalid_action": _action_payload(action_keys[invalid_index]),
                "nearest_invalid_unit_residual_l2": invalid_distance,
                "trace_zero_residual_norm": bool(zero_mask[trace_index]),
                "nearest_invalid_zero_residual_norm": bool(zero_mask[invalid_index]),
            }
        )
    return details


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _repeat_state(state: JEPALatentState, count: int, device: torch.device) -> JEPALatentState:
    return JEPALatentState(
        graph_latent=state.graph_latent.to(device).expand(count, -1).contiguous(),
        object_latents=state.object_latents.to(device).repeat(count, 1),
        object_ids=state.object_ids.to(device).repeat(count),
        object_batch=torch.arange(count, device=device).repeat_interleave(state.object_latents.size(0)),
    )


def _encode_actions(
    bundle: Any,
    space: Any,
    actions: Sequence[Any],
    state: JEPALatentState,
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(actions), chunk_size):
        chunk = tuple(actions[start : start + chunk_size])
        tensors = space.action_tensors_for_ground_actions(chunk, device=device)
        chunks.append(bundle.jepa.action_encoder(tensors, _repeat_state(state, len(chunk), device)).detach().cpu())
    return torch.cat(chunks, dim=0)


def _root_identity(candidate_identity: Mapping[str, Any]) -> dict[str, Any]:
    identities = {}
    for path, expected_sha256 in FIXED_SHA256.items():
        identity = file_identity(path)
        if identity["sha256"] != expected_sha256:
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
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Stage 0A extraction and write its exact two-file artifact."""

    started = time.perf_counter()
    validate_run_binding(args.checkpoint, args.output)
    if args.dataset_dir != DATASET or args.candidate_manifest != FIXED_CANDIDATE_MANIFEST:
        raise ValueError("dataset and candidate manifest must use the fixed absolute evidence paths")
    configure_determinism(args.seed)
    records, candidate_identity = load_and_validate_candidate_manifest(args.candidate_manifest)
    root = args.output.parents[2]
    first_command = args.output.relative_to(root).parts == ("schema_residual", "baseline", "run1")
    prepare_output_directory(root, args.output, _root_identity(candidate_identity), first_command=first_command)

    config, corpus, bundle, device, restoration = load_checkpoint_bundle(
        args.dataset_dir,
        args.checkpoint,
        device_name=args.device,
        include_restoration_metadata=True,
    )
    selected = select_split(corpus, config, args.split, seed=args.seed)
    sources = reconcile_manifest_source_states(records, selected)
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

    latent_chunks: list[torch.Tensor] = []
    schema_ids: list[str] = []
    group_ids: list[str] = []
    action_keys: list[tuple[str, tuple[str, ...]]] = []
    identity_rows: list[dict[str, Any]] = []
    distance_sources: list[dict[str, Any]] = []
    population_buckets = enumerate_same_schema_population(sources)
    with torch.inference_mode():
        for bucket in population_buckets:
            source = bucket.source
            space = bucket.space
            actions = bucket.actions
            state = encode_state(bundle, source.parsed, source.source_atoms, device=device)
            start = len(action_keys)
            latent_chunks.append(
                _encode_actions(bundle, space, actions, state, chunk_size=args.chunk_size, device=device)
            )
            local_keys = [(action.name, tuple(action.arguments)) for action in actions]
            trace_key = (source.trace_action.name, tuple(source.trace_action.arguments))
            key_to_index = {key: start + index for index, key in enumerate(local_keys)}
            invalid_keys = {
                (row["action"]["name"], tuple(row["action"]["arguments"]))
                for row in source.manifest_records
                if not row["applicability_label"] and row["action"]["name"] == source.trace_action.name
            }
            missing = invalid_keys - key_to_index.keys()
            if missing:
                raise ValueError(
                    f"manifest invalid actions missing from full population for {source.group}: {sorted(missing)}"
                )
            distance_sources.append(
                {
                    "group": source.group,
                    "problem": source.problem,
                    "step": source.step,
                    "trace": key_to_index[trace_key],
                    "invalid": [key_to_index[key] for key in sorted(invalid_keys)],
                }
            )
            identity_rows.extend(bucket.identity_rows)
            for action, key in zip(actions, local_keys, strict=True):
                schema_ids.append(action.name)
                group_ids.append(source.group)
                action_keys.append(key)

    latents = torch.cat(latent_chunks, dim=0)
    identity = population_identity(identity_rows)
    if latents.size(0) != POPULATION_COUNT or len(schema_ids) != POPULATION_COUNT:
        raise ValueError("latent/full-population count reconciliation failed")
    global_residual, state_residual = schema_residuals(latents, schema_ids, group_ids)
    details = distance_diagnostics(latents, state_residual, action_keys, group_ids, distance_sources)

    def residual_map(residual: torch.Tensor) -> dict[str, Any]:
        per_schema = {}
        for schema in SCHEMAS:
            indices = [index for index, value in enumerate(schema_ids) if value == schema]
            if not indices:
                raise ValueError(f"full population is missing schema {schema}")
            per_schema[schema] = residual_statistics(residual[indices])
        return {"pooled": residual_statistics(residual), "per_schema": per_schema}

    summary = {
        "schema_version": "action_latent_updated_phase0.schema_residual.v1",
        "kind": "schema_residual",
        "dataset": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": args.split,
        "seed": args.seed,
        "candidate_manifest": candidate_identity,
        "settings": {
            "chunk_size": args.chunk_size,
            "expected_population_count": POPULATION_COUNT,
            "residual_centers": ["global_schema", "state_schema"],
            "numerical_rank_relative_tolerance": 1e-6,
            "zero_norm_policy": "zero_vector",
        },
        "checkpoint_restoration": restoration,
        "counts": {
            "groups": len(sources),
            "full_population": len(action_keys),
            "candidate_manifest_records": len(records),
            "schemas": len(set(schema_ids)),
            "nearest_wrong_rows": len(details),
            "nearest_invalid_rows": len(details),
        },
        "metrics": {
            "full_population_identity": identity,
            "raw_variance_decomposition": raw_variance_decomposition(latents, schema_ids),
            "global_schema_residual": residual_map(global_residual),
            "state_schema_residual": residual_map(state_residual),
            "nearest_wrong_same_schema_raw_l2": _distribution(
                [row["nearest_wrong_raw_l2"] for row in details]
            ),
            "nearest_invalid_same_schema_unit_residual_l2": _distribution(
                [row["nearest_invalid_unit_residual_l2"] for row in details]
            ),
        },
        "environment": _environment(),
        "device": str(device),
        "output": str(args.output),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(args.output / "summary.json", summary)
    _write_json(args.output / "details.json", details)
    return summary


def main() -> int:
    summary = run(parse_args())
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
