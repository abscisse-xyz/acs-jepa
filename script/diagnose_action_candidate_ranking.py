"""Stage 0D fixed candidate-ranking diagnostic.

The command is deliberately measurement-only. Four deployable score paths consume
only a recorded source and candidate; the teacher-forced transition score is kept
in a separately named non-deployable path.
"""

from __future__ import annotations

# ruff: noqa: E501, I001

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from acs_jepa import JEPALatentState
from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa.graph import GroundAction, build_state_graph
from action_phase0_common import (
    SCHEMAS,
    SPLIT_SHA256,
    canonical_json_bytes,
    file_identity,
    load_and_validate_candidate_manifest,
    load_checkpoint_bundle,
    select_split,
    tie_aware_auroc,
    tie_aware_average_precision,
)
from action_role_object_probe import RoleObjectProbe, fit_role_object_probe
import diagnose_action_applicability_recoverability as recoverability_module
from diagnose_action_transition_equivalence import reconcile_transitions, transition_components

SCORERS = ("latent_transition", "latent_applicability", "role_object", "raw_symbolic", "hybrid")
TRAIN_GROUPS = (
    "p166:1", "p166:10", "p166:11", "p166:13", "p166:14", "p166:15", "p166:16", "p166:17",
    "p166:18", "p166:2", "p166:20", "p166:21", "p166:22", "p166:3", "p166:4", "p166:5",
    "p166:6", "p166:7", "p166:9", "p192:1", "p192:11", "p192:12", "p192:14", "p192:15",
    "p192:16", "p192:17", "p192:19", "p192:2", "p192:20", "p192:3", "p192:4", "p192:5", "p192:9",
)
EVAL_GROUPS = (
    "p166:0", "p166:12", "p166:19", "p166:8", "p192:0", "p192:10", "p192:13", "p192:18",
    "p192:6", "p192:7", "p192:8",
)
DATASET = Path("/opt/data/workspace/acs-jepa-tuning-data/smoke")
FIXED_CANDIDATE_MANIFEST = Path(
    "/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/baseline/probe_run1/example_manifest.json"
)
BASELINE_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt")
PHASE2_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt")
CHECKPOINT_SHA256 = {
    "baseline": "65a50ce3b93763e41cfada9c6e4ff717791f654e5b22a9e86526ec0cef7dd84e",
    "phase2": "7379691d246e2dbc4210d5aac28994f7725a3e2b5c257e0f9903ee9515bf5968",
}
CANDIDATE_SHA256 = "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    for name in ("summary", "details", "feature-schema", "split-manifest", "probe-states"):
        parser.add_argument(f"--recoverability-{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--seed", default=20260717, type=int)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (args.device, args.split, args.epochs, args.learning_rate, args.hidden_dim, args.seed) != (
        "cpu", "val", 200, 0.001, 64, 20260717
    ):
        raise ValueError("Stage 0D requires the fixed CPU/split/training/seed contract")
    if args.dataset_dir != DATASET or args.candidate_manifest != FIXED_CANDIDATE_MANIFEST:
        raise ValueError("Stage 0D requires fixed absolute dataset and candidate-manifest paths")
    if len(args.output.parts) < 3 or args.output.parts[-3] != "candidate_ranking":
        raise ValueError("output must end in candidate_ranking/{baseline|phase2}/{run1|run2}")
    variant, repeat = args.output.parts[-2:]
    expected = {"baseline": BASELINE_CHECKPOINT, "phase2": PHASE2_CHECKPOINT}.get(variant)
    if repeat not in {"run1", "run2"} or expected != args.checkpoint:
        raise ValueError("fixed checkpoint/output binding is invalid")
    expected_parent = args.output.parents[2] / "recoverability" / variant / "run1"
    supplied = (
        args.recoverability_summary,
        args.recoverability_details,
        args.recoverability_feature_schema,
        args.recoverability_split_manifest,
        args.recoverability_probe_states,
    )
    expected_names = ("summary.json", "details.json", "feature_schema.json", "split_manifest.json", "probe_states.json")
    if supplied != tuple(expected_parent / name for name in expected_names):
        raise ValueError("recoverability inputs must be matching accepted run1 siblings")


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _action_key(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    action = row["action"]
    return str(action["name"]), tuple(action["arguments"])


def stack_role_candidates(candidates: Sequence[Any]):
    """Flatten candidate roles in candidate/ascending-role order with one partition-wide bank width."""

    if not candidates:
        raise ValueError("at least one role candidate is required")
    max_objects = max(int(candidate.object_ids.numel()) for candidate in candidates)
    rows: list[tuple[Any, int, torch.Tensor, torch.Tensor, int]] = []
    slices = []
    for candidate in candidates:
        order = torch.argsort(candidate.object_ids)
        sorted_ids = candidate.object_ids[order]
        sorted_bank = candidate.object_latents[order]
        positions = {int(object_id): index for index, object_id in enumerate(sorted_ids.tolist())}
        start = len(rows)
        for role, active in enumerate(candidate.argument_mask.tolist()):
            if active:
                object_id = int(candidate.argument_object_ids[role])
                if object_id not in positions:
                    raise ValueError("active argument target is absent from sorted object bank")
                rows.append((candidate, role, sorted_bank, sorted_ids, positions[object_id]))
        if len(rows) == start:
            raise ValueError("zero active-role candidates are invalid")
        slices.append((start, len(rows)))
    latent_dim = int(rows[0][2].size(-1))
    banks = torch.zeros((len(rows), max_objects, latent_dim), dtype=torch.float32)
    masks = torch.zeros((len(rows), max_objects), dtype=torch.bool)
    graph, action, roles, targets = [], [], [], []
    for index, (candidate, role, bank, _ids, target) in enumerate(rows):
        count = bank.size(0)
        banks[index, :count] = bank.detach().to(dtype=torch.float32, device="cpu")
        masks[index, :count] = True
        graph.append(candidate.graph_latent.detach().to(dtype=torch.float32, device="cpu"))
        action.append(candidate.action_latent.detach().to(dtype=torch.float32, device="cpu"))
        roles.append(role)
        targets.append(target)
    return (
        torch.stack(graph), torch.stack(action), banks, masks,
        torch.tensor(roles, dtype=torch.long), torch.tensor(targets, dtype=torch.long),
    ), slices


def role_candidate_scores(model: RoleObjectProbe, tensors, slices: Sequence[tuple[int, int]]) -> list[float]:
    with torch.no_grad():
        logits = model(*tensors[:5])
        selected = logits.log_softmax(dim=-1)[torch.arange(logits.size(0)), tensors[5]]
    return [float(selected[start:end].mean()) for start, end in slices]


def _tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {"name": name, "shape": list(tensor.shape), "dtype": "torch.float32", "values": tensor.flatten().tolist()}


def serialize_role_probe(
    model: RoleObjectProbe,
    *,
    candidate_sha256: str,
    train_rows: int,
    eval_rows: int,
    optimizer_steps: int,
) -> dict[str, Any]:
    hidden_dim = int(model.query[0].out_features)
    latent_dim = int(model.role_embedding.embedding_dim)
    action_dim = int(model.query[0].in_features - 2 * latent_dim)
    return {
        "schema_version": "action_latent_updated_phase0.role_probe_state.v1",
        "candidate_manifest_sha256": candidate_sha256,
        "split_manifest_sha256": SPLIT_SHA256,
        "training": {
            "seed": 20260717, "epochs": 200, "learning_rate": 0.001, "hidden_dim": 64,
            "optimizer": "Adam(lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)",
            "dtype": "torch.float32", "device": "cpu", "num_threads": 1,
            "deterministic_algorithms": True, "mask_policy": "all_real_sorted_object_ids_padding_only",
            "row_order": "canonical_manifest_then_ascending_active_role",
            "train_bank_scope": "all_453_train_candidates", "eval_bank_scope": "all_151_eval_candidates",
            "train_records": 453, "eval_records": 151, "train_role_rows": train_rows,
            "eval_role_rows": eval_rows, "optimizer_steps": optimizer_steps,
        },
        "model": {
            "architecture": {
                "name": "RoleObjectProbe", "latent_dim": latent_dim, "action_dim": action_dim,
                "max_action_arity": model.max_action_arity, "hidden_dim": hidden_dim,
                "role_embedding": {"num_embeddings": model.max_action_arity, "embedding_dim": latent_dim},
                "query": [
                    {"kind": "linear", "in_features": 2 * latent_dim + action_dim, "out_features": hidden_dim, "bias": True},
                    {"kind": "gelu", "approximate": "none"},
                    {"kind": "linear", "in_features": hidden_dim, "out_features": latent_dim, "bias": True},
                ],
            },
            "state_dict": [_tensor_record(name, value) for name, value in sorted(model.state_dict().items())],
        },
    }


def reconstruct_role_probe(state: Mapping[str, Any]) -> RoleObjectProbe:
    if set(state) != {"schema_version", "candidate_manifest_sha256", "split_manifest_sha256", "training", "model"}:
        raise ValueError("role probe state has unknown or missing keys")
    training = state["training"]
    required = {
        "seed": 20260717, "epochs": 200, "learning_rate": 0.001, "hidden_dim": 64,
        "optimizer": "Adam(lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)",
        "dtype": "torch.float32", "device": "cpu", "num_threads": 1, "deterministic_algorithms": True,
        "mask_policy": "all_real_sorted_object_ids_padding_only",
        "row_order": "canonical_manifest_then_ascending_active_role", "train_bank_scope": "all_453_train_candidates",
        "eval_bank_scope": "all_151_eval_candidates", "train_records": 453, "eval_records": 151,
        "train_role_rows": 1549, "eval_role_rows": 518, "optimizer_steps": 200,
    }
    if training != required or state["schema_version"] != "action_latent_updated_phase0.role_probe_state.v1":
        raise ValueError("role probe training contract drift")
    if (
        state["split_manifest_sha256"] != SPLIT_SHA256
        or type(state["candidate_manifest_sha256"]) is not str
        or len(state["candidate_manifest_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in state["candidate_manifest_sha256"])
    ):
        raise ValueError("role probe identity binding drift")
    if not isinstance(state["model"], Mapping) or set(state["model"]) != {"architecture", "state_dict"}:
        raise ValueError("role probe model has unknown or missing keys")
    architecture = state["model"]["architecture"]
    if not isinstance(architecture, Mapping) or set(architecture) != {
        "name", "latent_dim", "action_dim", "max_action_arity", "hidden_dim", "role_embedding", "query"
    }:
        raise ValueError("role probe architecture has unknown or missing keys")
    expected_embedding = {
        "num_embeddings": architecture["max_action_arity"], "embedding_dim": architecture["latent_dim"]
    }
    expected_query = [
        {"kind": "linear", "in_features": 2 * architecture["latent_dim"] + architecture["action_dim"],
         "out_features": architecture["hidden_dim"], "bias": True},
        {"kind": "gelu", "approximate": "none"},
        {"kind": "linear", "in_features": architecture["hidden_dim"],
         "out_features": architecture["latent_dim"], "bias": True},
    ]
    if (
        architecture["name"] != "RoleObjectProbe"
        or architecture["latent_dim"] != 64
        or architecture["action_dim"] != 64
        or architecture["max_action_arity"] != 4
        or architecture["hidden_dim"] != 64
        or architecture["role_embedding"] != expected_embedding
        or architecture["query"] != expected_query
    ):
        raise ValueError("role probe architecture contract drift")
    model = RoleObjectProbe(
        latent_dim=architecture["latent_dim"], action_dim=architecture["action_dim"],
        max_action_arity=architecture["max_action_arity"], hidden_dim=architecture["hidden_dim"],
    )
    restored = {}
    records = state["model"]["state_dict"]
    if [record.get("name") for record in records] != sorted(record.get("name") for record in records):
        raise ValueError("role probe state tensors must be sorted and unique")
    for record in records:
        if set(record) != {"name", "shape", "dtype", "values"} or record["dtype"] != "torch.float32":
            raise ValueError("invalid role probe tensor record")
        tensor = torch.tensor(record["values"], dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()) or tensor.numel() != math.prod(record["shape"]):
            raise ValueError("invalid role probe tensor values/shape")
        if record["name"] in restored:
            raise ValueError("duplicate role probe tensor name")
        restored[record["name"]] = tensor.reshape(record["shape"])
    model.load_state_dict(restored, strict=True)
    model.eval()
    return model


def compose_score_vectors(recoverability_rows, *, role_scores, transition_scores):
    if len(recoverability_rows) != len(role_scores) or len(role_scores) != len(transition_scores):
        raise ValueError("five score vectors must have matching populations")
    output = []
    for row, role, transition in zip(recoverability_rows, role_scores, transition_scores, strict=True):
        logits = row["logits"]
        values = {
            "latent_transition": transition,
            "latent_applicability": logits["C_selected_graph_action/mlp"],
            "role_object": role,
            "raw_symbolic": logits["D_raw_symbolic/mlp"],
            "hybrid": logits["E_hybrid/mlp"],
        }
        if not all(_finite(value) for value in values.values()):
            raise ValueError("candidate scores must be finite")
        output.append(values)
    return output


def validate_recoverability_alignment(manifest_rows, recoverability_rows) -> None:
    keys = ("manifest_index", "group", "problem", "step", "action", "category", "label")
    if len(manifest_rows) != len(recoverability_rows):
        raise ValueError("recoverability metadata population mismatch")
    for expected, actual in zip(manifest_rows, recoverability_rows, strict=True):
        if any(expected.get(key) != actual.get(key) for key in keys):
            raise ValueError("recoverability metadata does not align to fixed manifest")


def rank_details(rows: Sequence[Mapping[str, Any]], scorers: Sequence[str] = SCORERS) -> list[dict[str, Any]]:
    result = copy.deepcopy(list(rows))
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result:
        by_group[row["group"]].append(row)
    for group_rows in by_group.values():
        if len({_action_key(row) for row in group_rows}) != len(group_rows):
            raise ValueError("candidate action keys must be unique within each group")
        for scorer in scorers:
            for row in group_rows:
                if scorer not in row["scores"] or not _finite(row["scores"][scorer]):
                    raise ValueError("every candidate requires every finite score")
            ordered = sorted(group_rows, key=lambda row: (-float(row["scores"][scorer]), *_action_key(row)))
            for rank, row in enumerate(ordered, 1):
                row.setdefault("ranks", {})[scorer] = rank
    return result


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"count": len(ordered), "min": ordered[0], "median": median, "mean": sum(ordered) / len(ordered), "max": ordered[-1]}


def _ranking_slice(rows: Sequence[Mapping[str, Any]], scorer: str) -> dict[str, Any]:
    labels = torch.tensor([bool(row["label"]) for row in rows], dtype=torch.bool)
    scores = torch.tensor([float(row["scores"][scorer]) for row in rows], dtype=torch.float64)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    binary = {
        "count": len(rows), "positive_count": positives, "negative_count": negatives,
        "prevalence": positives / len(rows) if rows else None,
        "auroc": tie_aware_auroc(scores[labels], scores[~labels]) if positives and negatives else None,
        "average_precision": tie_aware_average_precision(scores, labels) if positives and negatives else None,
    }
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)
    top1 = first_mrr = trace_mrr = 0.0
    groups_without_applicable = groups_without_inapplicable = pair_count = 0
    pair_credit = 0.0
    margins = {"role_swap": [], "one_arg_substitution": []}
    for group_rows in by_group.values():
        traces = [row for row in group_rows if row["category"] == "trace"]
        if len(traces) != 1:
            raise ValueError("each ranking group requires exactly one trace")
        applicable = [row for row in group_rows if row["label"]]
        inapplicable = [row for row in group_rows if not row["label"]]
        groups_without_applicable += not applicable
        groups_without_inapplicable += not inapplicable
        if applicable:
            first = min(int(row["ranks"][scorer]) for row in applicable)
            top1 += first == 1
            first_mrr += 1 / first
        trace_mrr += 1 / int(traces[0]["ranks"][scorer])
        for positive in applicable:
            for negative in inapplicable:
                delta = float(positive["scores"][scorer]) - float(negative["scores"][scorer])
                pair_credit += 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
                pair_count += 1
                if negative["category"] in margins:
                    margins[negative["category"]].append(delta)
    group_count = len(by_group)
    return {
        "binary": binary,
        "role_swap_margin": _distribution(margins["role_swap"]),
        "one_arg_substitution_margin": _distribution(margins["one_arg_substitution"]),
        "top1_applicable_rate": top1 / group_count if group_count else None,
        "mrr_first_applicable": first_mrr / group_count if group_count else None,
        "pairwise_applicable_accuracy": pair_credit / pair_count if pair_count else None,
        "trace_mrr": trace_mrr / group_count if group_count else None,
        "counts": {
            "groups": group_count, "groups_without_applicable": int(groups_without_applicable),
            "groups_without_inapplicable": int(groups_without_inapplicable), "within_group_pairs": pair_count,
        },
    }


def ranking_report(rows: Sequence[Mapping[str, Any]], scorer: str, deployable: bool | None = None) -> dict[str, Any]:
    global_values = _ranking_slice(rows, scorer)
    group_schema = {}
    for group in dict.fromkeys(row["group"] for row in rows):
        traces = [row for row in rows if row["group"] == group and row["category"] == "trace"]
        if len(traces) != 1:
            raise ValueError("each ranking group requires exactly one trace")
        group_schema[group] = traces[0]["action"]["name"]
    per_schema = {}
    for schema in SCHEMAS:
        selected = [row for row in rows if group_schema[row["group"]] == schema]
        values = _ranking_slice(selected, scorer)
        counts = values.pop("counts")
        binary = values.pop("binary")
        per_schema[schema] = {
            "count": binary["count"], "applicable": binary["positive_count"],
            "inapplicable": binary["negative_count"], **counts,
            "auroc": binary["auroc"], "average_precision": binary["average_precision"], **values,
        }
    counts = global_values.pop("counts")
    binary = global_values["binary"]
    gate = bool(
        binary["auroc"] is not None and binary["auroc"] >= .80
        and binary["average_precision"] is not None and binary["average_precision"] >= .35
        and global_values["role_swap_margin"]["median"] is not None
        and global_values["role_swap_margin"]["median"] > 0
        and global_values["one_arg_substitution_margin"]["median"] is not None
        and global_values["one_arg_substitution_margin"]["median"] > 0
        and global_values["top1_applicable_rate"] >= .80
        and global_values["pairwise_applicable_accuracy"] is not None
        and global_values["pairwise_applicable_accuracy"] >= .80
    )
    del counts
    return {
        **global_values,
        "per_schema": per_schema,
        "ranks_applicable": gate,
        "deployable": scorer != "latent_transition" if deployable is None else deployable,
    }


def _equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return left == right
    if isinstance(left, dict):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def validate_rank_reconciliation(rows, metrics, *, scorers: Sequence[str] = SCORERS, deployable=None) -> None:
    deployable = deployable or {name: name != "latent_transition" for name in scorers}
    rebuilt = rank_details([{key: value for key, value in row.items() if key != "ranks"} for row in rows], scorers)
    for actual, expected in zip(rows, rebuilt, strict=True):
        if actual.get("ranks") != expected["ranks"]:
            raise ValueError("serialized ranks do not match one-based total order")
    for scorer in scorers:
        expected_report = ranking_report(rows, scorer, deployable=deployable[scorer])
        if not _equal(metrics[scorer], expected_report):
            raise ValueError(f"ranking metric reconciliation failed for {scorer}")


DETAIL_KEYS = {"manifest_index", "group", "problem", "step", "action", "category", "label", "scores", "ranks"}
CATEGORIES = {"trace", "one_arg_substitution", "role_swap", "random_same_schema", "random_other_schema"}


def _require_action(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"name", "arguments"}:
        raise ValueError("action has unknown or missing keys")
    if type(value["name"]) is not str or type(value["arguments"]) is not list or any(type(item) is not str for item in value["arguments"]):
        raise ValueError("action has wrong strict types")


def validate_ranking_details(rows, *, manifest_rows=None, expected_count: int = 151) -> None:
    if type(rows) is not list or len(rows) != expected_count:
        raise ValueError("ranking detail population mismatch")
    if manifest_rows is not None and len(manifest_rows) != len(rows):
        raise ValueError("ranking manifest population mismatch")
    metadata = ("manifest_index", "group", "problem", "step", "action", "category", "label")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != DETAIL_KEYS:
            raise ValueError("ranking detail has unknown or missing keys")
        if type(row["manifest_index"]) is not int or type(row["step"]) is not int or type(row["group"]) is not str or type(row["problem"]) is not str:
            raise ValueError("ranking metadata has wrong strict types")
        if type(row["label"]) is not bool or type(row["category"]) is not str or row["category"] not in CATEGORIES:
            raise ValueError("ranking label/category contract drift")
        _require_action(row["action"])
        if not isinstance(row["scores"], Mapping) or set(row["scores"]) != set(SCORERS):
            raise ValueError("ranking score inventory drift")
        if not isinstance(row["ranks"], Mapping) or set(row["ranks"]) != set(SCORERS):
            raise ValueError("ranking rank inventory drift")
        if any(not _finite(value) for value in row["scores"].values()):
            raise ValueError("ranking scores must be finite numbers")
        if any(type(value) is not int or value < 1 for value in row["ranks"].values()):
            raise ValueError("ranking ranks must be positive strict integers")
        if manifest_rows is not None and any(row[key] != manifest_rows[index][key] or type(row[key]) is not type(manifest_rows[index][key]) for key in metadata):
            raise ValueError("ranking detail order/identity differs from canonical manifest")
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    for group_rows in by_group.values():
        for scorer in SCORERS:
            if sorted(row["ranks"][scorer] for row in group_rows) != list(range(1, len(group_rows) + 1)):
                raise ValueError("ranking ranks are not a one-based group permutation")


def validate_ranking_metrics(rows, metrics) -> None:
    if not isinstance(metrics, Mapping) or set(metrics) != set(SCORERS):
        raise ValueError("ranking metric scorer inventory drift")
    validate_rank_reconciliation(rows, metrics)


def validate_ranking_artifacts(artifacts: Mapping[str, Any], *, manifest_rows=None, expected_count: int = 151) -> None:
    if set(artifacts) != {"summary.json", "details.json", "split_manifest.json", "role_probe_state.json"}:
        raise ValueError("ranking output inventory drift")
    details = artifacts["details.json"]
    if expected_count != 151 and manifest_rows is None:
        raise ValueError("synthetic ranking validation requires explicit manifest rows")
    if expected_count == 151 and manifest_rows is None:
        records, _identity = load_and_validate_candidate_manifest(FIXED_CANDIDATE_MANIFEST)
        manifest_rows = [
            {"manifest_index": index, "group": row["group"], "problem": row["problem"], "step": row["step"],
             "action": row["action"], "category": row["category"], "label": row["applicability_label"]}
            for index, row in enumerate(records) if row["group"] in EVAL_GROUPS
        ]
    validate_ranking_details(details, manifest_rows=manifest_rows, expected_count=expected_count)
    summary = artifacts["summary.json"]
    if not isinstance(summary, Mapping) or set(summary) != {
        "schema_version", "kind", "dataset", "checkpoint", "checkpoint_sha256", "split", "seed",
        "candidate_manifest", "settings", "checkpoint_restoration", "counts", "metrics", "environment",
        "device", "output", "runtime_seconds",
    }:
        raise ValueError("ranking summary has unknown or missing keys")
    if summary["schema_version"] != "action_latent_updated_phase0.candidate_ranking.v1" or summary["kind"] != "candidate_ranking":
        raise ValueError("ranking summary kind/schema drift")
    output = Path(summary["output"]) if type(summary["output"]) is str else Path()
    if len(output.parts) < 3 or output.parts[-3] != "candidate_ranking" or output.parts[-1] not in {"run1", "run2"}:
        raise ValueError("ranking output binding drift")
    variant = output.parts[-2]
    checkpoint = {"baseline": BASELINE_CHECKPOINT, "phase2": PHASE2_CHECKPOINT}.get(variant)
    if (
        checkpoint is None or type(summary["dataset"]) is not str or summary["dataset"] != str(DATASET)
        or type(summary["checkpoint"]) is not str or summary["checkpoint"] != str(checkpoint)
        or type(summary["checkpoint_sha256"]) is not str or summary["checkpoint_sha256"] != CHECKPOINT_SHA256[variant]
        or summary["split"] != "val" or type(summary["seed"]) is not int or summary["seed"] != 20260717
        or summary["device"] != "cpu" or type(summary["runtime_seconds"]) not in (int, float)
        or not math.isfinite(float(summary["runtime_seconds"])) or summary["runtime_seconds"] < 0
    ):
        raise ValueError("ranking fixed identity/runtime contract drift")
    identity = summary["candidate_manifest"]
    if not isinstance(identity, Mapping) or set(identity) != {"path", "bytes", "sha256", "count"}:
        raise ValueError("candidate manifest identity schema drift")
    if (
        identity["path"] != str(FIXED_CANDIDATE_MANIFEST) or type(identity["bytes"]) is not int or identity["bytes"] <= 0
        or identity["sha256"] != CANDIDATE_SHA256 or type(identity["count"]) is not int or identity["count"] != 604
    ):
        raise ValueError("candidate manifest identity drift")
    if expected_count == 151:
        _records, actual_identity = load_and_validate_candidate_manifest(FIXED_CANDIDATE_MANIFEST)
        if identity != actual_identity:
            raise ValueError("candidate manifest bytes/hash drift")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != summary["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes/hash drift")

    settings = summary["settings"]
    if not isinstance(settings, Mapping) or set(settings) != {
        "epochs", "learning_rate", "hidden_dim", "scorers", "ranking_gate", "ranking_order",
        "pairwise_tie_credit", "mrr_no_applicable_contribution", "role_training", "recoverability_inputs",
    }:
        raise ValueError("ranking settings have unknown or missing keys")
    expected_gate = {"auroc": .8, "average_precision": .35, "top1_applicable_rate": .8,
                     "pairwise_applicable_accuracy": .8, "role_swap_margin_strictly_positive": True,
                     "one_arg_margin_strictly_positive": True}
    expected_role = {"train_role_rows": 1549, "eval_role_rows": 518, "optimizer_steps": 200,
                     "row_order": "canonical_manifest_then_ascending_active_role",
                     "train_bank_scope": "all_453_train_candidates", "eval_bank_scope": "all_151_eval_candidates",
                     "threads": 1, "deterministic_algorithms": True, "seed_before_model_construction": 20260717}
    if (
        type(settings["epochs"]) is not int or settings["epochs"] != 200
        or type(settings["learning_rate"]) is not float or settings["learning_rate"] != .001
        or type(settings["hidden_dim"]) is not int or settings["hidden_dim"] != 64
        or settings["scorers"] != list(SCORERS) or not _equal(settings["ranking_gate"], expected_gate)
        or settings["ranking_order"] != "descending_score_then_action_name_then_arguments"
        or type(settings["pairwise_tie_credit"]) is not float or settings["pairwise_tie_credit"] != .5
        or type(settings["mrr_no_applicable_contribution"]) is not float or settings["mrr_no_applicable_contribution"] != 0.0
        or not _equal(settings["role_training"], expected_role)
    ):
        raise ValueError("ranking fixed settings drift")
    recoverability = settings["recoverability_inputs"]
    filenames = {"summary": "summary.json", "details": "details.json", "feature_schema": "feature_schema.json",
                 "split_manifest": "split_manifest.json", "probe_states": "probe_states.json"}
    if not isinstance(recoverability, Mapping) or set(recoverability) != set(filenames):
        raise ValueError("ranking recoverability identity inventory drift")
    expected_parent = output.parents[2] / "recoverability" / variant / "run1"
    for name, filename in filenames.items():
        item = recoverability[name]
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("ranking recoverability identity schema drift")
        if (type(item["path"]) is not str or Path(item["path"]) != expected_parent / filename
                or type(item["bytes"]) is not int or item["bytes"] <= 0
                or type(item["sha256"]) is not str or len(item["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in item["sha256"])):
            raise ValueError("ranking recoverability identity drift")
        if expected_count == 151 and file_identity(Path(item["path"])) != item:
            raise ValueError("ranking recoverability file identity drift")

    statuses = {name: "restored" for name in ("jepa", "goal_head", "action_contrastive_anchor", "argument_reconstruction_head", "applicability_head")}
    if variant == "baseline":
        statuses.update(action_contrastive_anchor="disabled", argument_reconstruction_head="disabled", applicability_head="disabled")
    state_keys = {"jepa": "model_state_dict", "goal_head": "goal_head_state_dict",
                  "action_contrastive_anchor": "action_contrastive_anchor_state_dict",
                  "argument_reconstruction_head": "argument_reconstruction_head_state_dict",
                  "applicability_head": "applicability_head_state_dict"}
    expected_restoration = {name: {"state_key": state_keys[name], "status": statuses[name]} for name in state_keys}
    if summary["checkpoint_restoration"] != expected_restoration:
        raise ValueError("ranking checkpoint restoration drift")

    groups = list(dict.fromkeys(row["group"] for row in details))
    expected_counts = {
        "eval_records": len(details), "eval_groups": len(groups),
        "applicable": sum(row["label"] for row in details), "inapplicable": sum(not row["label"] for row in details),
        "within_group_pairs": sum(sum(row["label"] for row in details if row["group"] == group) * sum(not row["label"] for row in details if row["group"] == group) for group in groups),
        "groups_without_applicable": sum(not any(row["label"] for row in details if row["group"] == group) for group in groups),
        "groups_without_inapplicable": sum(not any(not row["label"] for row in details if row["group"] == group) for group in groups),
        "train_role_rows": 1549, "eval_role_rows": 518, "optimizer_steps": 200,
    }
    if summary["counts"] != expected_counts or any(type(value) is not int for value in summary["counts"].values()):
        raise ValueError("ranking counts do not reconcile details")
    validate_ranking_metrics(details, summary["metrics"])

    environment = summary["environment"]
    if not isinstance(environment, Mapping) or set(environment) != {"python_version", "torch_version", "platform", "byteorder", "num_threads", "num_interop_threads", "deterministic_algorithms", "python_hash_seed", "cublas_workspace_config"}:
        raise ValueError("ranking environment schema drift")
    if (any(type(environment[name]) is not str or not environment[name] for name in ("python_version", "torch_version", "platform"))
            or environment["byteorder"] not in {"little", "big"}
            or any(type(environment[name]) is not int or environment[name] < 1 for name in ("num_threads", "num_interop_threads"))
            or type(environment["deterministic_algorithms"]) is not bool or not environment["deterministic_algorithms"]
            or any(environment[name] is not None and type(environment[name]) is not str for name in ("python_hash_seed", "cublas_workspace_config"))):
        raise ValueError("ranking environment literal/type drift")

    state = artifacts["role_probe_state.json"]
    reconstruct_role_probe(state)
    if state["candidate_manifest_sha256"] != identity["sha256"]:
        raise ValueError("role probe candidate identity drift")
    expected_split = {"eval_groups": list(EVAL_GROUPS), "train_groups": list(TRAIN_GROUPS)}
    if artifacts["split_manifest.json"] != expected_split or state["split_manifest_sha256"] != SPLIT_SHA256:
        raise ValueError("ranking split manifest drift")


def ranking_summary_fixture(*, metrics, details) -> dict[str, Any]:
    """Build a closed synthetic ranking summary used by mutation tests."""
    recoverability_root = Path("/recoverability/baseline/run1")
    identities = {name: {"path": str(recoverability_root / filename), "bytes": 1, "sha256": "a" * 64} for name, filename in {
        "summary": "summary.json", "details": "details.json", "feature_schema": "feature_schema.json",
        "split_manifest": "split_manifest.json", "probe_states": "probe_states.json",
    }.items()}
    groups = {row["group"] for row in details}
    return {
        "schema_version": "action_latent_updated_phase0.candidate_ranking.v1", "kind": "candidate_ranking",
        "dataset": str(DATASET), "checkpoint": str(BASELINE_CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA256["baseline"],
        "split": "val", "seed": 20260717,
        "candidate_manifest": {"path": str(FIXED_CANDIDATE_MANIFEST), "bytes": 117385, "sha256": CANDIDATE_SHA256, "count": 604},
        "settings": {"epochs": 200, "learning_rate": .001, "hidden_dim": 64, "scorers": list(SCORERS),
                     "ranking_gate": {"auroc": .8, "average_precision": .35, "top1_applicable_rate": .8, "pairwise_applicable_accuracy": .8, "role_swap_margin_strictly_positive": True, "one_arg_margin_strictly_positive": True},
                     "ranking_order": "descending_score_then_action_name_then_arguments", "pairwise_tie_credit": .5,
                     "mrr_no_applicable_contribution": 0.0,
                     "role_training": {"train_role_rows": 1549, "eval_role_rows": 518, "optimizer_steps": 200, "row_order": "canonical_manifest_then_ascending_active_role", "train_bank_scope": "all_453_train_candidates", "eval_bank_scope": "all_151_eval_candidates", "threads": 1, "deterministic_algorithms": True, "seed_before_model_construction": 20260717},
                     "recoverability_inputs": identities},
        "checkpoint_restoration": {
            "jepa": {"state_key": "model_state_dict", "status": "restored"},
            "goal_head": {"state_key": "goal_head_state_dict", "status": "restored"},
            "action_contrastive_anchor": {"state_key": "action_contrastive_anchor_state_dict", "status": "disabled"},
            "argument_reconstruction_head": {"state_key": "argument_reconstruction_head_state_dict", "status": "disabled"},
            "applicability_head": {"state_key": "applicability_head_state_dict", "status": "disabled"},
        },
        "counts": {"eval_records": len(details), "eval_groups": len(groups), "applicable": sum(row["label"] for row in details), "inapplicable": sum(not row["label"] for row in details), "within_group_pairs": sum(sum(row["label"] for row in details if row["group"] == group) * sum(not row["label"] for row in details if row["group"] == group) for group in groups), "groups_without_applicable": sum(not any(row["label"] for row in details if row["group"] == group) for group in groups), "groups_without_inapplicable": sum(not any(not row["label"] for row in details if row["group"] == group) for group in groups), "train_role_rows": 1549, "eval_role_rows": 518, "optimizer_steps": 200},
        "metrics": metrics,
        "environment": {"python_version": "x", "torch_version": "x", "platform": "x", "byteorder": "little", "num_threads": 1, "num_interop_threads": 1, "deterministic_algorithms": True, "python_hash_seed": None, "cublas_workspace_config": None},
        "device": "cpu", "output": "/candidate_ranking/baseline/run1", "runtime_seconds": 0.0,
    }


def _repeat_state(state: JEPALatentState, count: int) -> JEPALatentState:
    objects = state.object_latents.size(0)
    return JEPALatentState(
        graph_latent=state.graph_latent.expand(count, -1).contiguous(),
        object_latents=state.object_latents.repeat(count, 1),
        object_ids=state.object_ids.repeat(count),
        object_batch=torch.arange(count).repeat_interleave(objects),
    )


def _extract_candidates(args, records):
    config, corpus, bundle, device, restoration = load_checkpoint_bundle(
        args.dataset_dir, args.checkpoint, device_name="cpu", include_restoration_metadata=True
    )
    selected = select_split(corpus, config, "val", seed=args.seed)
    transitions = reconcile_transitions(records, selected)
    candidates = []
    with torch.inference_mode():
        for source in transitions:
            source_graph = build_state_graph(source.parsed, source.source_atoms, include_static=True).to(device)
            state = bundle.jepa.encode(source_graph)
            space = ActionDecodingSpace.from_parsed_problem(source.parsed)
            actions = [GroundAction(row["action"]["name"], tuple(row["action"]["arguments"])) for row in source.records]
            tensors = space.action_tensors_for_ground_actions(actions, device=device)
            action_latents = bundle.jepa.action_encoder(tensors, _repeat_state(state, len(actions)))
            predictions = [bundle.jepa.predictor(state, action_latents[index:index + 1]) for index in range(len(actions))]
            successor_atoms = source.trajectory.states[source.successor_index]
            target = bundle.jepa.encode(build_state_graph(source.parsed, successor_atoms, include_static=True).to(device))
            order = torch.argsort(state.object_ids)
            object_ids = state.object_ids[order].detach().cpu()
            object_latents = state.object_latents[order].detach().cpu()
            for index, row in enumerate(source.records):
                _g, _o, total = transition_components(
                    predictions[index], target, source.parsed, graph_weight=1.0, object_weight=1.0
                )
                candidates.append(SimpleNamespace(
                    manifest_record=row,
                    graph_latent=state.graph_latent.squeeze(0).detach().cpu(),
                    action_latent=action_latents[index].detach().cpu(),
                    object_ids=object_ids,
                    object_latents=object_latents,
                    argument_mask=tensors["action_arg_mask"][index].detach().cpu().bool(),
                    argument_object_ids=tensors["action_object_indices"][index].detach().cpu().long(),
                    transition_score=-total,
                ))
    return candidates, restoration


def _load_json(path: Path) -> Any:
    raw = path.read_bytes()
    value = json.loads(raw)
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"noncanonical JSON evidence: {path}")
    return value


def _closed_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"recoverability {name} has unknown or missing keys")
    return value


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def validate_recoverability_artifacts(
    artifacts: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_summary_path: Path,
    expected_checkpoint: Path,
    dataset_dir: Path,
    device: str,
    split: str,
    seed: int,
):
    """Validate the complete fixed Stage 0B artifact contract before model use."""

    short_names = {"summary", "details", "feature_schema", "split_manifest", "probe_states"}
    file_names = {"summary.json", "details.json", "feature_schema.json", "split_manifest.json", "probe_states.json"}
    if set(artifacts) == short_names:
        values = dict(artifacts)
    elif set(artifacts) == file_names:
        values = {name.removesuffix(".json"): artifacts[f"{name}.json"] for name in short_names}
    else:
        raise ValueError("recoverability artifact inventory drift")
    summary, details = values["summary"], values["details"]
    feature_schema, split_manifest, probe_states = values["feature_schema"], values["split_manifest"], values["probe_states"]

    canonical_records, manifest_identity = load_and_validate_candidate_manifest(FIXED_CANDIDATE_MANIFEST)
    if not _strict_equal(list(records), canonical_records):
        raise ValueError("recoverability canonical manifest records drift")
    expected_summary_path = Path(expected_summary_path)
    expected_checkpoint = Path(expected_checkpoint)
    dataset_dir = Path(dataset_dir)
    variant = "baseline" if expected_checkpoint == BASELINE_CHECKPOINT else "phase2" if expected_checkpoint == PHASE2_CHECKPOINT else None
    if (variant is None or expected_summary_path != expected_summary_path.parent / "summary.json"
            or (dataset_dir, device, split, seed) != (DATASET, "cpu", "val", 20260717)):
        raise ValueError("recoverability expected checkpoint/summary path drift")
    expected_root = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/updated_phase0/recoverability") / variant
    if expected_summary_path.parent.parent != expected_root or expected_summary_path.parent.name not in {"run1", "run2"}:
        raise ValueError("recoverability accepted summary path drift")

    summary_keys = {
        "schema_version", "kind", "dataset", "checkpoint", "checkpoint_sha256", "split", "seed",
        "candidate_manifest", "settings", "checkpoint_restoration", "counts", "metrics", "environment",
        "device", "output", "runtime_seconds",
    }
    _closed_keys(summary, summary_keys, "summary")
    expected_checkpoint_hash = CHECKPOINT_SHA256[variant]
    if (
        summary["schema_version"] != "action_latent_updated_phase0.applicability_recoverability.v1"
        or summary["kind"] != "applicability_recoverability" or summary["dataset"] != str(DATASET)
        or summary["checkpoint"] != str(expected_checkpoint) or summary["checkpoint_sha256"] != expected_checkpoint_hash
        or summary["split"] != "val" or type(summary["seed"]) is not int or summary["seed"] != 20260717
        or summary["candidate_manifest"] != manifest_identity or summary["device"] != "cpu"
        or summary["output"] != str(expected_summary_path.parent)
        or type(summary["runtime_seconds"]) not in (int, float) or not math.isfinite(float(summary["runtime_seconds"]))
        or summary["runtime_seconds"] < 0
        or hashlib.sha256(expected_checkpoint.read_bytes()).hexdigest() != expected_checkpoint_hash
    ):
        raise ValueError("recoverability summary fixed identity/literal drift")
    expected_settings = {
        "epochs": 200, "learning_rate": .001, "hidden_dim": 64,
        "models": ["linear", "mlp", "control_mlp"],
        "feature_sets": [schema["name"] for schema in recoverability_module.feature_schemas()],
        "threshold_policy": "max_train_f1_highest_threshold",
        "control_policy": "train_label_permutation_seed_20260717",
        "reliability_bins": [index / 10 for index in range(11)],
    }
    settings = _closed_keys(summary["settings"], set(expected_settings), "settings")
    if not _strict_equal(settings, expected_settings):
        raise ValueError("recoverability fixed settings drift")
    expected_counts = {"records": 604, "train_records": 453, "eval_records": 151, "train_groups": 33, "eval_groups": 11, "applicable": 62, "inapplicable": 542}
    counts = _closed_keys(summary["counts"], set(expected_counts), "counts")
    if not _strict_equal(counts, expected_counts):
        raise ValueError("recoverability fixed counts drift")
    statuses = {name: "restored" for name in ("jepa", "goal_head", "action_contrastive_anchor", "argument_reconstruction_head", "applicability_head")}
    if variant == "baseline":
        statuses.update(action_contrastive_anchor="disabled", argument_reconstruction_head="disabled", applicability_head="disabled")
    state_keys = {"jepa": "model_state_dict", "goal_head": "goal_head_state_dict", "action_contrastive_anchor": "action_contrastive_anchor_state_dict", "argument_reconstruction_head": "argument_reconstruction_head_state_dict", "applicability_head": "applicability_head_state_dict"}
    expected_restoration = {name: {"state_key": state_keys[name], "status": statuses[name]} for name in state_keys}
    if not _strict_equal(summary["checkpoint_restoration"], expected_restoration):
        raise ValueError("recoverability checkpoint restoration drift")
    environment = _closed_keys(summary["environment"], {"python_version", "torch_version", "platform", "byteorder", "num_threads", "num_interop_threads", "deterministic_algorithms", "python_hash_seed", "cublas_workspace_config"}, "environment")
    if (any(type(environment[name]) is not str or not environment[name] for name in ("python_version", "torch_version", "platform"))
            or environment["byteorder"] not in {"little", "big"}
            or any(type(environment[name]) is not int or environment[name] < 1 for name in ("num_threads", "num_interop_threads"))
            or type(environment["deterministic_algorithms"]) is not bool or not environment["deterministic_algorithms"]
            or any(environment[name] is not None and type(environment[name]) is not str for name in ("python_hash_seed", "cublas_workspace_config"))):
        raise ValueError("recoverability environment type/literal drift")

    schemas = recoverability_module.feature_schemas()
    schema_by_name = {schema["name"]: schema for schema in schemas}
    feature = _closed_keys(feature_schema, {"schema_version", "candidate_manifest_sha256", "feature_sets"}, "feature schema")
    if feature["schema_version"] != "action_latent_updated_phase0.feature_schema.v1" or feature["candidate_manifest_sha256"] != manifest_identity["sha256"] or not _strict_equal(feature["feature_sets"], schemas):
        raise ValueError("recoverability feature schema identity/content drift")
    expected_split = {"eval_groups": list(EVAL_GROUPS), "train_groups": list(TRAIN_GROUPS)}
    if not _strict_equal(split_manifest, expected_split) or hashlib.sha256(canonical_json_bytes(split_manifest)).hexdigest() != SPLIT_SHA256:
        raise ValueError("recoverability split manifest drift")
    extraction_args = SimpleNamespace(
        dataset_dir=dataset_dir, checkpoint=expected_checkpoint, device=device, split=split, seed=seed,
    )
    collected_features, restoration, bundle = recoverability_module._collect_features(extraction_args, records)
    if not _strict_equal(restoration, summary["checkpoint_restoration"]):
        raise ValueError("recoverability checkpoint restoration drift")
    train_indices = [index for index, row in enumerate(canonical_records) if row["group"] in TRAIN_GROUPS]
    if len(train_indices) != 453:
        raise ValueError("recoverability fixed train feature population drift")
    expected_preprocessing = {
        schema["name"]: recoverability_module.fit_preprocessing(
            collected_features[schema["name"]][train_indices],
            standardized_indices=schema["standardized_indices"],
            binary_indices=schema["binary_indices"],
        )
        for schema in schemas
    }

    detail_keys = {"manifest_index", "group", "problem", "step", "action", "category", "label", "split", "logits", "control_logits"}
    logit_keys = {f"{schema['name']}/{kind}" for schema in schemas for kind in ("linear", "mlp")}
    control_keys = {f"{schema['name']}/mlp" for schema in schemas}
    if type(details) is not list or len(details) != 604:
        raise ValueError("recoverability detail population drift")
    for index, (row, record) in enumerate(zip(details, canonical_records, strict=True)):
        _closed_keys(row, detail_keys, f"detail[{index}]")
        _closed_keys(row["logits"], logit_keys, f"detail[{index}].logits")
        _closed_keys(row["control_logits"], control_keys, f"detail[{index}].control_logits")
        expected_row = {
            "manifest_index": index, "group": record["group"], "problem": record["problem"], "step": record["step"],
            "action": record["action"], "category": record["category"], "label": record["applicability_label"],
            "split": "train" if record["group"] in TRAIN_GROUPS else "eval",
        }
        if any(not _strict_equal(row[name], value) for name, value in expected_row.items()) or any(not _finite(value) for value in (*row["logits"].values(), *row["control_logits"].values())):
            raise ValueError(f"recoverability detail manifest/logit drift at row {index}")

    _closed_keys(probe_states, {"schema_version", "candidate_manifest_sha256", "split_manifest_sha256", "training", "models"}, "probe states")
    expected_training = {"seed": 20260717, "epochs": 200, "learning_rate": .001, "hidden_dim": 64, "optimizer": "Adam(lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)", "dtype": "torch.float32"}
    training = _closed_keys(probe_states["training"], set(expected_training), "probe training")
    if (probe_states["schema_version"] != "action_latent_updated_phase0.probe_states.v1"
            or probe_states["candidate_manifest_sha256"] != manifest_identity["sha256"]
            or probe_states["split_manifest_sha256"] != SPLIT_SHA256
            or not _strict_equal(training, expected_training)):
        raise ValueError("recoverability probe-state identity/training drift")
    expected_inventory = [(schema["name"], kind, schema["dimension"]) for schema in schemas for kind in ("linear", "mlp", "control_mlp")]
    models = probe_states["models"]
    if type(models) is not list or [(model.get("feature_set"), model.get("model_kind"), model.get("input_dim")) for model in models] != expected_inventory:
        raise ValueError("recoverability probe inventory/order/input dimensions drift")
    preprocessing_by_feature: dict[str, Mapping[str, Any]] = {}
    for model_index, model in enumerate(models):
        _closed_keys(model, {"feature_set", "model_kind", "input_dim", "architecture", "preprocessing", "state_dict"}, f"probe model[{model_index}]")
        schema = schema_by_name[model["feature_set"]]
        dimension = schema["dimension"]
        expected_architecture = (
            {"name": "linear", "input_dim": dimension, "output_dim": 1, "bias": True}
            if model["model_kind"] == "linear"
            else {"name": "mlp", "input_dim": dimension, "hidden_dim": 64, "output_dim": 1, "activation": "relu", "bias": True}
        )
        if not _strict_equal(model["architecture"], expected_architecture):
            raise ValueError("recoverability probe architecture drift")
        preprocessing = _closed_keys(model["preprocessing"], {"mean", "std", "binary_indices", "standardized_indices", "zero_std_indices"}, "probe preprocessing")
        if (not all(type(preprocessing[name]) is list for name in preprocessing)
                or len(preprocessing["mean"]) != dimension or len(preprocessing["std"]) != dimension
                or not _strict_equal(preprocessing["binary_indices"], schema["binary_indices"])
                or not _strict_equal(preprocessing["standardized_indices"], schema["standardized_indices"])
                or any(type(index) is not int for index in preprocessing["zero_std_indices"])
                or not set(preprocessing["zero_std_indices"]).issubset(set(schema["standardized_indices"]))
                or set(schema["binary_indices"]) & set(schema["standardized_indices"])
                or sorted([*schema["binary_indices"], *schema["standardized_indices"]]) != list(range(dimension))
                or any(not _finite(value) for value in (*preprocessing["mean"], *preprocessing["std"]))
                or any(preprocessing["mean"][index] != 0.0 or preprocessing["std"][index] != 1.0 for index in schema["binary_indices"])
                or any(preprocessing["std"][index] <= 0 for index in schema["standardized_indices"])
                or any(preprocessing["std"][index] != 1.0 for index in preprocessing["zero_std_indices"])):
            raise ValueError("recoverability preprocessing values/partitions drift")
        previous = preprocessing_by_feature.setdefault(model["feature_set"], preprocessing)
        if not _strict_equal(preprocessing, previous):
            raise ValueError("recoverability preprocessing differs across feature probes")
        if not _strict_equal(preprocessing, expected_preprocessing[model["feature_set"]]):
            raise ValueError("recoverability preprocessing does not match fixed train features")
        expected_tensors = (
            [("bias", [1]), ("weight", [1, dimension])]
            if model["model_kind"] == "linear"
            else [("0.bias", [64]), ("0.weight", [64, dimension]), ("2.bias", [1]), ("2.weight", [1, 64])]
        )
        state_dict = model["state_dict"]
        if type(state_dict) is not list or [(record.get("name"), record.get("shape")) for record in state_dict] != expected_tensors:
            raise ValueError("recoverability probe tensor names/shapes drift")
        for tensor in state_dict:
            _closed_keys(tensor, {"name", "shape", "dtype", "values"}, "probe tensor")
            if (tensor["dtype"] != "torch.float32" or any(type(size) is not int or size < 0 for size in tensor["shape"])
                    or math.prod(tensor["shape"]) != len(tensor["values"])
                    or any(not _finite(value) for value in tensor["values"])):
                raise ValueError("recoverability probe tensor dtype/value drift")
    for model in models:
        recoverability_module.reconstruct_probe(model)

    metrics = _closed_keys(summary["metrics"], {"features", "verdicts"}, "metrics")
    features = _closed_keys(metrics["features"], set(schema_by_name), "metric features")
    verdicts = _closed_keys(metrics["verdicts"], {"latent_separable", "raw_separable", "hybrid_separable", "label_or_sampling_blocker", "latent_state_bottleneck", "any_abc_separable"}, "verdicts")
    if any(type(value) is not bool for value in verdicts.values()):
        raise ValueError("recoverability verdict type drift")
    binary_keys = {"count", "positive_count", "negative_count", "prevalence", "accuracy", "precision", "recall", "f1", "auroc", "average_precision", "nll", "brier", "true_positive", "false_positive", "true_negative", "false_negative", "reliability_bins"}
    for feature_name, probes in features.items():
        _closed_keys(probes, {"linear", "mlp", "control_mlp"}, f"metrics.{feature_name}")
        for kind, report in probes.items():
            _closed_keys(report, {"train", "eval", "role_swap_margin", "one_arg_substitution_margin", "per_schema", "threshold"}, f"metrics.{feature_name}.{kind}")
            for split in ("train", "eval"):
                binary = _closed_keys(report[split], binary_keys, f"metrics.{feature_name}.{kind}.{split}")
                if type(binary["reliability_bins"]) is not list or len(binary["reliability_bins"]) != 10:
                    raise ValueError("recoverability reliability-bin count drift")
                for item in binary["reliability_bins"]:
                    _closed_keys(item, {"lower", "upper", "upper_inclusive", "count", "mean_probability", "positive_rate"}, "reliability bin")
            for margin in ("role_swap_margin", "one_arg_substitution_margin"):
                _closed_keys(report[margin], {"count", "min", "median", "mean", "max"}, f"metrics.{feature_name}.{kind}.{margin}")
            per_schema = _closed_keys(report["per_schema"], set(SCHEMAS), "recoverability per-schema metrics")
            for schema_report in per_schema.values():
                _closed_keys(schema_report, binary_keys, "recoverability per-schema binary metrics")
    if any(isinstance(value, float) and not math.isfinite(value) for value in _flatten_values(summary)):
        raise ValueError("recoverability summary contains non-finite value")
    return collected_features, restoration, bundle


def _flatten_values(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _flatten_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten_values(child)
    else:
        yield value


def _validate_recoverability_evidence(args, records, artifacts) -> None:
    """Reconstruct all fifteen accepted probes and every one of their 604 logits."""

    features, restoration, _bundle = validate_recoverability_artifacts(
        artifacts, records,
        expected_summary_path=args.recoverability_summary,
        expected_checkpoint=args.checkpoint,
        dataset_dir=args.dataset_dir,
        device=args.device,
        split=args.split,
        seed=args.seed,
    )
    summary = artifacts["summary"]
    details = artifacts["details"]
    feature_schema = artifacts["feature_schema"]
    probe_states = artifacts["probe_states"]
    if (
        summary.get("kind") != "applicability_recoverability"
        or len(details) != 604
        or feature_schema.get("feature_sets") != recoverability_module.feature_schemas()
        or probe_states.get("schema_version") != "action_latent_updated_phase0.probe_states.v1"
        or len(probe_states.get("models", ())) != 15
    ):
        raise ValueError("recoverability artifact schema/population drift")
    if restoration != summary["checkpoint_restoration"]:
        raise ValueError("recoverability checkpoint restoration drift")
    expected_order = [
        (feature, kind)
        for feature in (item["name"] for item in recoverability_module.feature_schemas())
        for kind in ("linear", "mlp", "control_mlp")
    ]
    actual_order = [(item["feature_set"], item["model_kind"]) for item in probe_states["models"]]
    if actual_order != expected_order:
        raise ValueError("recoverability probe-state order drift")
    all_logits = {}
    for state in probe_states["models"]:
        model = recoverability_module.reconstruct_probe(state)
        values = recoverability_module.apply_preprocessing(
            features[state["feature_set"]], state["preprocessing"]
        )
        with torch.no_grad():
            logits_tensor = model(values).flatten().to(torch.float64)
            logits = logits_tensor.tolist()
        all_logits[(state["feature_set"], state["model_kind"])] = logits_tensor
        key = f"{state['feature_set']}/{'mlp' if state['model_kind'] == 'control_mlp' else state['model_kind']}"
        field = "control_logits" if state["model_kind"] == "control_mlp" else "logits"
        for index, (actual, row) in enumerate(zip(logits, details, strict=True)):
            if abs(actual - row[field][key]) > 1e-7:
                raise ValueError(f"recoverability logit reconstruction failed at row {index}: {key}")
    train_indices = [index for index, row in enumerate(records) if row["group"] in TRAIN_GROUPS]
    eval_indices = [index for index, row in enumerate(records) if row["group"] in EVAL_GROUPS]
    labels = torch.tensor([row["applicability_label"] for row in records], dtype=torch.float32)
    train_labels, eval_labels = labels[train_indices], labels[eval_indices]
    control_labels = recoverability_module.control_permutation(train_labels, seed=20260717)
    train_rows = [
        {"group": records[index]["group"], "category": records[index]["category"],
         "schema": records[index]["action"]["name"]}
        for index in train_indices
    ]
    eval_rows = [
        {"group": records[index]["group"], "category": records[index]["category"],
         "schema": records[index]["action"]["name"]}
        for index in eval_indices
    ]
    rebuilt_metrics = {name: {} for name in (item["name"] for item in recoverability_module.feature_schemas())}
    for feature, kind in expected_order:
        logits = all_logits[(feature, kind)]
        rebuilt_metrics[feature][kind] = recoverability_module.probe_report(
            logits[train_indices], logits[eval_indices], train_labels, eval_labels,
            train_rows, eval_rows,
            threshold_labels=control_labels if kind == "control_mlp" else train_labels,
        )
    if rebuilt_metrics != summary["metrics"]["features"]:
        raise ValueError("recoverability metric reconstruction failed")
    latent = recoverability_module._separable(
        rebuilt_metrics["C_selected_graph_action"]["mlp"],
        rebuilt_metrics["C_selected_graph_action"]["control_mlp"],
    )
    raw = recoverability_module._separable(
        rebuilt_metrics["D_raw_symbolic"]["mlp"], rebuilt_metrics["D_raw_symbolic"]["control_mlp"]
    )
    hybrid = recoverability_module._separable(
        rebuilt_metrics["E_hybrid"]["mlp"], rebuilt_metrics["E_hybrid"]["control_mlp"]
    )
    rebuilt_verdicts = {
        "latent_separable": latent, "raw_separable": raw, "hybrid_separable": hybrid,
        "label_or_sampling_blocker": not raw, "latent_state_bottleneck": raw and not latent,
        "any_abc_separable": any(
            recoverability_module._separable(rebuilt_metrics[name][kind], rebuilt_metrics[name]["control_mlp"])
            for name in ("A_action", "B_graph_action", "C_selected_graph_action")
            for kind in ("linear", "mlp")
        ),
    }
    if rebuilt_verdicts != summary["metrics"]["verdicts"]:
        raise ValueError("recoverability verdict reconstruction failed")


def _write_atomic(destination: Path, artifacts: Mapping[str, Any], *, manifest_rows=None) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        validate_ranking_artifacts(artifacts, manifest_rows=manifest_rows)
        for name, value in artifacts.items():
            (staging / name).write_bytes(canonical_json_bytes(value))
            if _load_json(staging / name) != value:
                raise ValueError("staged artifact reread mismatch")
        reread = {name: _load_json(staging / name) for name in artifacts}
        validate_ranking_artifacts(reread, manifest_rows=manifest_rows)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    validate_args(args)
    if args.output.exists():
        raise FileExistsError(f"destination already exists: {args.output}")
    records, candidate_identity = load_and_validate_candidate_manifest(args.candidate_manifest)
    root = args.output.parents[2]
    expected_root_identity = recoverability_module._root_identity(candidate_identity)
    marker = root / "root_identity.json"
    if not marker.is_file() or marker.read_bytes() != canonical_json_bytes(expected_root_identity):
        raise ValueError("root identity marker does not match immutable inputs")
    recoverability_paths = {
        "summary": args.recoverability_summary, "details": args.recoverability_details,
        "feature_schema": args.recoverability_feature_schema, "split_manifest": args.recoverability_split_manifest,
        "probe_states": args.recoverability_probe_states,
    }
    recoverability = {name: _load_json(path) for name, path in recoverability_paths.items()}
    if set(path.name for path in args.recoverability_summary.parent.iterdir()) != {
        "summary.json", "details.json", "feature_schema.json", "split_manifest.json", "probe_states.json"
    }:
        raise ValueError("recoverability sibling inventory drift")
    if hashlib.sha256(args.recoverability_split_manifest.read_bytes()).hexdigest() != SPLIT_SHA256:
        raise ValueError("fixed split identity drift")
    _validate_recoverability_evidence(args, records, recoverability)
    candidates, restoration = _extract_candidates(args, records)
    train_candidates = [candidate for candidate in candidates if candidate.manifest_record["group"] in TRAIN_GROUPS]
    eval_candidates = [candidate for candidate in candidates if candidate.manifest_record["group"] in EVAL_GROUPS]
    if (len(train_candidates), len(eval_candidates)) != (453, 151):
        raise ValueError("fixed ranking train/eval candidate count drift")
    train_tensors, _ = stack_role_candidates(train_candidates)
    eval_tensors, eval_slices = stack_role_candidates(eval_candidates)
    if (train_tensors[0].size(0), eval_tensors[0].size(0)) != (1549, 518):
        raise ValueError("fixed role-row population drift")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = fit_role_object_probe(
        train_tensors, eval_tensors, max_action_arity=4, hidden_dim=64, epochs=200,
        learning_rate=.001, seed=20260717, device=torch.device("cpu")
    )
    role_scores = role_candidate_scores(result.model, result.eval_tensors, eval_slices)
    eval_indices = [index for index, row in enumerate(records) if row["group"] in EVAL_GROUPS]
    recovery_eval = [recoverability["details"][index] for index in eval_indices]
    manifest_eval = [
        {"manifest_index": index, "group": records[index]["group"], "problem": records[index]["problem"],
         "step": records[index]["step"], "action": records[index]["action"], "category": records[index]["category"],
         "label": records[index]["applicability_label"]}
        for index in eval_indices
    ]
    validate_recoverability_alignment(manifest_eval, recovery_eval)
    score_vectors = compose_score_vectors(
        recovery_eval, role_scores=role_scores,
        transition_scores=[candidate.transition_score for candidate in eval_candidates],
    )
    details = rank_details([{**row, "scores": scores} for row, scores in zip(manifest_eval, score_vectors, strict=True)])
    metrics = {name: ranking_report(details, name) for name in SCORERS}
    validate_rank_reconciliation(details, metrics)
    role_state = serialize_role_probe(
        result.model, candidate_sha256=candidate_identity["sha256"], train_rows=1549,
        eval_rows=518, optimizer_steps=result.optimizer_steps,
    )
    summary = {
        "schema_version": "action_latent_updated_phase0.candidate_ranking.v1", "kind": "candidate_ranking",
        "dataset": str(args.dataset_dir), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "split": "val", "seed": 20260717,
        "candidate_manifest": candidate_identity,
        "settings": {
            "epochs": 200, "learning_rate": .001, "hidden_dim": 64, "scorers": list(SCORERS),
            "ranking_gate": {"auroc": .80, "average_precision": .35, "top1_applicable_rate": .80,
                             "pairwise_applicable_accuracy": .80, "role_swap_margin_strictly_positive": True,
                             "one_arg_margin_strictly_positive": True},
            "ranking_order": "descending_score_then_action_name_then_arguments", "pairwise_tie_credit": .5,
            "mrr_no_applicable_contribution": 0.0,
            "role_training": {"train_role_rows": 1549, "eval_role_rows": 518, "optimizer_steps": 200,
                              "row_order": "canonical_manifest_then_ascending_active_role",
                              "train_bank_scope": "all_453_train_candidates", "eval_bank_scope": "all_151_eval_candidates",
                              "threads": 1, "deterministic_algorithms": True,
                              "seed_before_model_construction": 20260717},
            "recoverability_inputs": {name: file_identity(path) for name, path in recoverability_paths.items()},
        },
        "checkpoint_restoration": restoration,
        "counts": {
            "eval_records": 151, "eval_groups": 11,
            "applicable": sum(row["label"] for row in details), "inapplicable": sum(not row["label"] for row in details),
            "within_group_pairs": sum(
                sum(r["label"] for r in details if r["group"] == group)
                * sum(not r["label"] for r in details if r["group"] == group)
                for group in EVAL_GROUPS
            ),
            "groups_without_applicable": sum(not any(r["label"] for r in details if r["group"] == group) for group in EVAL_GROUPS),
            "groups_without_inapplicable": sum(not any(not r["label"] for r in details if r["group"] == group) for group in EVAL_GROUPS),
            "train_role_rows": 1549, "eval_role_rows": 518, "optimizer_steps": 200,
        },
        "metrics": metrics,
        "environment": {"python_version": platform.python_version(), "torch_version": str(torch.__version__),
                        "platform": platform.platform(), "byteorder": sys.byteorder, "num_threads": torch.get_num_threads(),
                        "num_interop_threads": torch.get_num_interop_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "device": "cpu", "output": str(args.output), "runtime_seconds": time.perf_counter() - started,
    }
    _write_atomic(args.output, {
        "summary.json": summary, "details.json": details,
        "split_manifest.json": recoverability["split_manifest"], "role_probe_state.json": role_state,
    }, manifest_rows=manifest_eval)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
