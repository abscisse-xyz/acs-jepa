"""Measure deterministic applicability recoverability from fixed offline evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from acs_jepa import JEPALatentState
from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa.graph import GroundAction, GroundAtom, build_state_graph
from action_phase0_common import (
    SPLIT_SHA256,
    canonical_json_bytes,
    file_identity,
    load_and_validate_candidate_manifest,
    load_checkpoint_bundle,
    prepare_output_directory,
    reconcile_manifest_source_states,
    select_split,
    tie_aware_auroc,
    tie_aware_average_precision,
)

SCHEMAS = (
    "build_diagonal_oneway",
    "build_straight_oneway",
    "car_arrived",
    "car_start",
    "destroy_road",
    "move_car_in_road",
    "move_car_out_road",
)
TYPES = ("car", "garage", "junction", "road")
PREDICATES = (
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
ROLE_PAIRS = tuple(itertools.combinations(range(4), 2))
EVAL_GROUPS = (
    "p166:0",
    "p166:12",
    "p166:19",
    "p166:8",
    "p192:0",
    "p192:10",
    "p192:13",
    "p192:18",
    "p192:6",
    "p192:7",
    "p192:8",
)
TRAIN_GROUPS = (
    "p166:1",
    "p166:10",
    "p166:11",
    "p166:13",
    "p166:14",
    "p166:15",
    "p166:16",
    "p166:17",
    "p166:18",
    "p166:2",
    "p166:20",
    "p166:21",
    "p166:22",
    "p166:3",
    "p166:4",
    "p166:5",
    "p166:6",
    "p166:7",
    "p166:9",
    "p192:1",
    "p192:11",
    "p192:12",
    "p192:14",
    "p192:15",
    "p192:16",
    "p192:17",
    "p192:19",
    "p192:2",
    "p192:20",
    "p192:3",
    "p192:4",
    "p192:5",
    "p192:9",
)
BASELINE_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/checkpoints/best.pt")
PHASE2_CHECKPOINT = Path("/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/checkpoints/best.pt")
UPDATED_SPEC = Path("/opt/data/workspace/acs-jepa/script/ACTION_LATENT_UPDATED_SPEC.md")
BASELINE_CONFIG = Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml")
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


def validate_run_binding(checkpoint: Path, output: Path) -> None:
    if len(output.parts) < 3 or output.parts[-3] != "recoverability":
        raise ValueError("output must end in recoverability/{baseline|phase2}/{run1|run2}")
    variant, repeat = output.parts[-2:]
    if repeat not in {"run1", "run2"}:
        raise ValueError("recoverability destination must be run1 or run2")
    expected = {"baseline": BASELINE_CHECKPOINT, "phase2": PHASE2_CHECKPOINT}.get(variant)
    if expected is None or checkpoint != expected:
        raise ValueError("checkpoint/output binding does not match baseline or phase2 fixed evidence")


def _raw_names() -> list[str]:
    return (
        [f"schema={name}" for name in SCHEMAS]
        + [f"role_active[{role}]" for role in range(4)]
        + [f"role_equal[{left},{right}]" for left, right in ROLE_PAIRS]
        + [f"role_type[{role}]={type_name}" for role in range(4) for type_name in TYPES]
        + [
            f"fact:{predicate}({','.join(map(str, roles))})"
            for predicate, arg_types in PREDICATES
            for roles in itertools.product(range(4), repeat=len(arg_types))
        ]
    )


def feature_schemas() -> list[dict[str, object]]:
    action = [f"action_latent[{i}]" for i in range(64)]
    graph = [f"graph_latent[{i}]" for i in range(64)]
    selected = [f"selected_object_latent[{role}][{coordinate}]" for role in range(4) for coordinate in range(64)]
    present = [f"argument_present[{role}]" for role in range(4)]
    raw = _raw_names()
    inventories = (
        ("A_action", action, range(64), ()),
        ("B_graph_action", graph + action, range(128), ()),
        ("C_selected_graph_action", graph + action + selected + present, range(384), range(384, 388)),
        ("D_raw_symbolic", raw, (), range(217)),
        ("E_hybrid", graph + action + selected + present + raw, range(384), range(384, 605)),
    )
    return [
        {
            "name": name,
            "dimension": len(names),
            "feature_names": names,
            "binary_indices": list(binary),
            "standardized_indices": list(standardized),
        }
        for name, names, standardized, binary in inventories
    ]


def raw_symbolic_features(parsed: object, atoms: object, action: object) -> torch.Tensor:
    if tuple(sorted(parsed.actions)) != SCHEMAS or tuple(sorted(parsed.types)) != TYPES:
        raise ValueError("live CityCar action/type vocabulary differs from fixed contract")
    live_predicates = tuple((name, tuple(parsed.predicates[name].arg_types)) for name in sorted(parsed.predicates))
    if live_predicates != PREDICATES:
        raise ValueError("live CityCar predicate vocabulary differs from fixed contract")
    arguments = tuple(action.arguments)
    active = [role < len(arguments) for role in range(4)]
    values: list[float] = [float(action.name == name) for name in SCHEMAS]
    values.extend(map(float, active))
    values.extend(
        float(active[left] and active[right] and arguments[left] == arguments[right]) for left, right in ROLE_PAIRS
    )
    values.extend(
        float(active[role] and parsed.objects[arguments[role]].type == type_name)
        for role in range(4)
        for type_name in TYPES
    )
    atom_set = set(atoms)
    for predicate, arg_types in PREDICATES:
        schema = parsed.predicates[predicate]
        for roles in itertools.product(range(4), repeat=len(arg_types)):
            valid = all(active[role] for role in roles)
            if valid:
                bound = tuple(arguments[role] for role in roles)
                valid = all(
                    parsed.objects[obj].type == expected for obj, expected in zip(bound, schema.arg_types, strict=True)
                )
            values.append(float(valid and GroundAtom(predicate, bound) in atom_set) if valid else 0.0)
    return torch.tensor(values, dtype=torch.float32)


def validate_static_fact_match(parsed, source_atoms, graph_atoms) -> None:
    static = parsed.static_predicates
    source = {atom for atom in source_atoms if atom.predicate in static}
    graphed = {atom for atom in graph_atoms if atom.predicate in static}
    if source != graphed:
        raise ValueError("recorded static facts do not match include_static graph input")


def reconstruct_graph_atoms(parsed, graph) -> tuple[GroundAtom, ...]:
    """Independently recover grounded atoms from a state-graph ``Data`` object."""
    predicate_names = {value: name for name, value in parsed.predicate_to_id.items()}
    object_names = {value: name for name, value in parsed.object_to_id.items()}
    edges = graph.edge_index.t().tolist()
    edge_attributes = graph.edge_attr.tolist()
    actual_edges = Counter(
        (source, target, role, direction)
        for (source, target), (role, direction) in zip(edges, edge_attributes, strict=True)
    )
    object_nodes = {}
    for node, row in enumerate(graph.x.tolist()):
        if row[0] == 0:
            if row[4] not in object_names:
                raise ValueError("state graph contains an unknown object id")
            object_nodes[node] = object_names[row[4]]

    atoms = []
    expected_edges = Counter()
    for node, row in enumerate(graph.x.tolist()):
        if row[0] != 1:
            continue
        predicate_id, arity = row[2], row[3]
        if predicate_id not in predicate_names:
            raise ValueError("state graph contains an unknown predicate id")
        arguments = {}
        for edge, attribute in zip(edges, edge_attributes, strict=True):
            source, target = edge
            role, direction = attribute
            if source != node or direction != 0:
                continue
            if target not in object_nodes or role in arguments:
                raise ValueError("state graph atom has invalid role/object edges")
            arguments[role] = object_nodes[target]
            expected_edges[(node, target, role, 0)] += 1
            expected_edges[(target, node, role, 1)] += 1
        if sorted(arguments) != list(range(arity)):
            raise ValueError("state graph atom roles do not match its arity")
        atoms.append(GroundAtom(predicate_names[predicate_id], tuple(arguments[role] for role in range(arity))))
    if len(atoms) != int(graph.num_atoms):
        raise ValueError("state graph atom count does not match atom nodes")
    if actual_edges != expected_edges:
        raise ValueError("state graph edges are not the exact bidirectional atom-role multiset")
    return tuple(atoms)


def compose_feature_sets(graph_latent, action_latent, selected_object_latents, argument_mask, raw_symbolic):
    selected = selected_object_latents.clone()
    selected[~argument_mask.bool()] = 0
    c = torch.cat((graph_latent, action_latent, selected.flatten(), argument_mask.to(graph_latent.dtype)))
    return {
        "A_action": action_latent,
        "B_graph_action": torch.cat((graph_latent, action_latent)),
        "C_selected_graph_action": c,
        "D_raw_symbolic": raw_symbolic,
        "E_hybrid": torch.cat((c, raw_symbolic)),
    }


def fit_preprocessing(features: torch.Tensor, *, standardized_indices, binary_indices) -> dict[str, object]:
    values = features.detach().to(device="cpu", dtype=torch.float64)
    dimension = values.size(1)
    mean = torch.zeros(dimension, dtype=torch.float64)
    std = torch.ones(dimension, dtype=torch.float64)
    indices = list(standardized_indices)
    if indices:
        mean[indices] = values[:, indices].mean(dim=0)
        raw_std = values[:, indices].std(dim=0, correction=0)
        std[indices] = torch.where(raw_std == 0, torch.ones_like(raw_std), raw_std)
        zero = [index for index, value in zip(indices, raw_std.tolist(), strict=True) if value == 0.0]
    else:
        zero = []
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "binary_indices": list(binary_indices),
        "standardized_indices": indices,
        "zero_std_indices": zero,
    }


def apply_preprocessing(features: torch.Tensor, state: dict[str, object]) -> torch.Tensor:
    values = features.detach().to(device="cpu", dtype=torch.float64)
    mean = torch.tensor(state["mean"], dtype=torch.float64)
    std = torch.tensor(state["std"], dtype=torch.float64)
    indices = state["standardized_indices"]
    values[:, indices] = (values[:, indices] - mean[indices]) / std[indices]
    values[:, state["zero_std_indices"]] = 0.0
    return values.to(torch.float32)


def split_manifest() -> dict[str, list[str]]:
    return {"eval_groups": list(EVAL_GROUPS), "train_groups": list(TRAIN_GROUPS)}


def control_permutation(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return labels[torch.randperm(labels.numel(), generator=generator)]


def select_f1_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.detach().to(device="cpu", dtype=torch.float64).flatten()
    labels = labels.detach().to(device="cpu", dtype=torch.bool).flatten()
    candidates = [float("inf"), *sorted(set(logits.tolist()), reverse=True), float("-inf")]
    best_threshold, best_f1 = candidates[0], -1.0
    for threshold in candidates:
        prediction = logits >= threshold
        tp = int((prediction & labels).sum())
        fp = int((prediction & ~labels).sum())
        fn = int((~prediction & labels).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold


def distribution(values) -> dict[str, object]:
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {
        "count": len(values),
        "min": values[0],
        "median": median,
        "mean": sum(values) / len(values),
        "max": values[-1],
    }


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, object]:
    logits = logits.detach().to(device="cpu", dtype=torch.float64).flatten()
    labels = labels.detach().to(device="cpu", dtype=torch.float64).flatten()
    if (
        logits.shape != labels.shape
        or not bool(torch.isfinite(logits).all())
        or bool(((labels != 0) & (labels != 1)).any())
    ):
        raise ValueError("finite matching logits and binary labels are required")
    positive = labels.bool()
    prediction = logits >= threshold
    count = labels.numel()
    positives = int(positive.sum())
    negatives = count - positives
    tp = int((prediction & positive).sum())
    fp = int((prediction & ~positive).sum())
    tn = int((~prediction & ~positive).sum())
    fn = int((~prediction & positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    probabilities = torch.sigmoid(logits)
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        selected = (probabilities >= lower) & ((probabilities <= upper) if index == 9 else (probabilities < upper))
        bin_count = int(selected.sum())
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "upper_inclusive": index == 9,
                "count": bin_count,
                "mean_probability": float(probabilities[selected].mean()) if bin_count else None,
                "positive_rate": float(labels[selected].mean()) if bin_count else None,
            }
        )
    ranking = positives > 0 and negatives > 0
    return {
        "count": int(count),
        "positive_count": positives,
        "negative_count": negatives,
        "prevalence": positives / count if count else None,
        "accuracy": float((prediction == positive).to(torch.float64).mean()) if count else None,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "auroc": tie_aware_auroc(logits[positive], logits[~positive]) if ranking else None,
        "average_precision": tie_aware_average_precision(logits, positive) if ranking else None,
        "nll": float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)) if count else None,
        "brier": float(((probabilities - labels) ** 2).mean()) if count else None,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "reliability_bins": bins,
    }


@dataclass
class FittedProbe:
    feature_set: str
    model_kind: str
    model: nn.Module


def _probe(input_dim: int, kind: str, hidden_dim: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dim, 1)
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))


def fit_all_probes(features, preprocessing, labels, control_labels, *, epochs, learning_rate, hidden_dim, seed):
    fitted = []
    for feature_set in (row["name"] for row in feature_schemas()):
        x = apply_preprocessing(features[feature_set], preprocessing[feature_set])
        for kind in ("linear", "mlp", "control_mlp"):
            torch.manual_seed(seed)
            model = _probe(x.size(1), kind, hidden_dim).to(device="cpu", dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            targets = (
                (control_labels if kind == "control_mlp" else labels).detach().to(device="cpu", dtype=torch.float32)
            )
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), targets).backward()
                optimizer.step()
            model.eval()
            fitted.append(FittedProbe(feature_set, kind, model))
    return fitted


def _tensor_record(name: str, value: torch.Tensor) -> dict[str, object]:
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {"name": name, "shape": list(value.shape), "dtype": "torch.float32", "values": value.flatten().tolist()}


def serialize_probe_states(fitted, preprocessing, *, candidate_sha256, epochs, learning_rate, hidden_dim, seed):
    models = []
    for item in fitted:
        input_dim = next(item.model.parameters()).shape[1]
        architecture = (
            {"name": "linear", "input_dim": input_dim, "output_dim": 1, "bias": True}
            if item.model_kind == "linear"
            else {
                "name": "mlp",
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": 1,
                "activation": "relu",
                "bias": True,
            }
        )
        models.append(
            {
                "feature_set": item.feature_set,
                "model_kind": item.model_kind,
                "input_dim": input_dim,
                "architecture": architecture,
                "preprocessing": preprocessing[item.feature_set],
                "state_dict": [_tensor_record(name, value) for name, value in sorted(item.model.state_dict().items())],
            }
        )
    return {
        "schema_version": "action_latent_updated_phase0.probe_states.v1",
        "candidate_manifest_sha256": candidate_sha256,
        "split_manifest_sha256": "5397fc5e7820c9fdee3eb38c05278a3b680fb5ca8460d0bbe588ffa7ff22815c",
        "training": {
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "optimizer": "Adam(lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)",
            "dtype": "torch.float32",
        },
        "models": models,
    }


def reconstruct_probe(state) -> nn.Module:
    architecture = state["architecture"]
    model = _probe(state["input_dim"], state["model_kind"], architecture.get("hidden_dim", 64))
    restored = {
        record["name"]: torch.tensor(record["values"], dtype=torch.float32).reshape(record["shape"])
        for record in state["state_dict"]
    }
    model.load_state_dict(restored, strict=True)
    model.eval()
    return model


def _category_margin(logits, labels, rows, category):
    references = {
        row["group"]: float(logit)
        for logit, label, row in zip(logits, labels, rows, strict=True)
        if bool(label) and row["category"] == "trace"
    }
    return distribution(
        references[row["group"]] - float(logit)
        for logit, label, row in zip(logits, labels, rows, strict=True)
        if not bool(label) and row["category"] == category and row["group"] in references
    )


def probe_report(train_logits, eval_logits, train_labels, eval_labels, train_rows, eval_rows, *, threshold_labels=None):
    threshold_source = train_labels if threshold_labels is None else threshold_labels
    threshold = select_f1_threshold(train_logits, threshold_source)
    per_schema = {}
    for schema in SCHEMAS:
        selected = [index for index, row in enumerate(eval_rows) if row["schema"] == schema]
        per_schema[schema] = binary_metrics(eval_logits[selected], eval_labels[selected], threshold)
    return {
        "train": binary_metrics(train_logits, threshold_source, threshold),
        "eval": binary_metrics(eval_logits, eval_labels, threshold),
        "role_swap_margin": _category_margin(eval_logits, eval_labels, eval_rows, "role_swap"),
        "one_arg_substitution_margin": _category_margin(eval_logits, eval_labels, eval_rows, "one_arg_substitution"),
        "per_schema": per_schema,
        "threshold": threshold,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
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
        "cpu",
        "val",
        200,
        0.001,
        64,
        20260717,
    ):
        raise ValueError("Stage 0B requires the fixed CPU/split/training/seed contract")


def _repeat_state(state: JEPALatentState, count: int, device: torch.device) -> JEPALatentState:
    object_count = state.object_latents.size(0)
    return JEPALatentState(
        graph_latent=state.graph_latent.to(device).expand(count, -1).contiguous(),
        object_latents=state.object_latents.to(device).repeat(count, 1),
        object_ids=state.object_ids.to(device).repeat(count),
        object_batch=torch.arange(count, device=device).repeat_interleave(object_count),
    )


def _root_identity(candidate_identity: dict[str, Any]) -> dict[str, Any]:
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
        "candidate_manifest": candidate_identity,
        "split_sha256": SPLIT_SHA256,
        "created_by": "schema_residual/baseline/run1",
    }


def _extract_source_features(bundle, source, *, device: torch.device) -> list[dict[str, torch.Tensor]]:
    state_graph = build_state_graph(source.parsed, source.source_atoms, include_static=True)
    graph_atoms = reconstruct_graph_atoms(source.parsed, state_graph)
    validate_static_fact_match(source.parsed, source.source_atoms, graph_atoms)
    state = bundle.jepa.encode(state_graph.to(device))
    space = ActionDecodingSpace.from_parsed_problem(source.parsed)
    actions = [
        GroundAction(row["action"]["name"], tuple(row["action"]["arguments"]))
        for row in source.manifest_records
    ]
    tensors = space.action_tensors_for_ground_actions(actions, device=device)
    action_latents = bundle.jepa.action_encoder(tensors, _repeat_state(state, len(actions), device))
    masks = tensors["action_arg_mask"].bool()
    object_positions = {int(object_id): position for position, object_id in enumerate(state.object_ids.tolist())}
    selected_objects = state.object_latents.new_zeros((len(actions), 4, state.object_latents.size(-1)))
    for row_index, role_index in masks.nonzero(as_tuple=False).tolist():
        object_id = int(tensors["action_object_indices"][row_index, role_index])
        selected_objects[row_index, role_index] = state.object_latents[object_positions[object_id]]
    graph = state.graph_latent.squeeze(0)
    return [
        compose_feature_sets(
            graph.detach().cpu(),
            action_latents[index].detach().cpu(),
            selected_objects[index].detach().cpu(),
            masks[index].detach().cpu(),
            raw_symbolic_features(source.parsed, source.source_atoms, action),
        )
        for index, action in enumerate(actions)
    ]


def _collect_features(args, records):
    config, corpus, bundle, device, restoration = load_checkpoint_bundle(
        args.dataset_dir, args.checkpoint, device_name=args.device, include_restoration_metadata=True
    )
    selected_corpus = select_split(corpus, config, args.split, seed=args.seed)
    sources = reconcile_manifest_source_states(records, selected_corpus)
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
    record_index = {id(record): index for index, record in enumerate(records)}
    feature_rows: list[dict[str, torch.Tensor] | None] = [None] * len(records)
    with torch.inference_mode():
        for source in sources:
            extracted = _extract_source_features(bundle, source, device=device)
            for record, features in zip(source.manifest_records, extracted, strict=True):
                feature_rows[record_index[id(record)]] = features
    if any(row is None for row in feature_rows):
        raise ValueError("feature extraction did not reconcile all manifest rows")
    feature_names = [schema["name"] for schema in feature_schemas()]
    stacked = {name: torch.stack([row[name] for row in feature_rows if row is not None]) for name in feature_names}
    return stacked, restoration, bundle


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


def _separable(report, control_report) -> bool:
    return bool(
        report["eval"]["auroc"] >= 0.80
        and report["eval"]["average_precision"] >= 0.35
        and report["role_swap_margin"]["count"]
        and report["role_swap_margin"]["median"] > 0
        and report["one_arg_substitution_margin"]["count"]
        and report["one_arg_substitution_margin"]["median"] > 0
        and control_report["eval"]["auroc"] <= 0.70
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    validate_args(args)
    validate_run_binding(args.checkpoint, args.output)
    if args.dataset_dir != DATASET or args.candidate_manifest != FIXED_CANDIDATE_MANIFEST:
        raise ValueError("dataset and candidate manifest must use fixed absolute evidence paths")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    records, candidate_identity = load_and_validate_candidate_manifest(args.candidate_manifest)
    root = args.output.parents[2]
    prepare_output_directory(root, args.output, _root_identity(candidate_identity), first_command=False)
    features, restoration, _bundle = _collect_features(args, records)
    train_indices = [index for index, row in enumerate(records) if row["group"] in TRAIN_GROUPS]
    eval_indices = [index for index, row in enumerate(records) if row["group"] in EVAL_GROUPS]
    if (len(train_indices), len(eval_indices)) != (453, 151):
        raise ValueError("fixed train/eval record count drift")
    labels = torch.tensor([row["applicability_label"] for row in records], dtype=torch.float32)
    train_labels, eval_labels = labels[train_indices], labels[eval_indices]
    control_labels = control_permutation(train_labels, seed=args.seed)
    schemas = feature_schemas()
    preprocessing = {
        schema["name"]: fit_preprocessing(
            features[schema["name"]][train_indices],
            standardized_indices=schema["standardized_indices"],
            binary_indices=schema["binary_indices"],
        )
        for schema in schemas
    }
    train_features = {name: values[train_indices] for name, values in features.items()}
    fitted = fit_all_probes(
        train_features,
        preprocessing,
        train_labels,
        control_labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    train_rows = [
        {"group": records[i]["group"], "category": records[i]["category"], "schema": records[i]["action"]["name"]}
        for i in train_indices
    ]
    eval_rows = [
        {"group": records[i]["group"], "category": records[i]["category"], "schema": records[i]["action"]["name"]}
        for i in eval_indices
    ]
    metrics: dict[str, dict[str, Any]] = {name: {} for name in features}
    all_logits: dict[str, torch.Tensor] = {}
    for item in fitted:
        transformed = apply_preprocessing(features[item.feature_set], preprocessing[item.feature_set])
        with torch.no_grad():
            logits = item.model(transformed).flatten().to(torch.float64)
        all_logits[f"{item.feature_set}/{item.model_kind}"] = logits
        threshold_labels = control_labels if item.model_kind == "control_mlp" else train_labels
        metrics[item.feature_set][item.model_kind] = probe_report(
            logits[train_indices],
            logits[eval_indices],
            train_labels,
            eval_labels,
            train_rows,
            eval_rows,
            threshold_labels=threshold_labels,
        )
    details = []
    for index, row in enumerate(records):
        details.append(
            {
                "manifest_index": index,
                "group": row["group"],
                "problem": row["problem"],
                "step": row["step"],
                "action": row["action"],
                "category": row["category"],
                "label": row["applicability_label"],
                "split": "train" if index in set(train_indices) else "eval",
                "logits": {
                    f"{feature}/{kind}": float(all_logits[f"{feature}/{kind}"][index])
                    for feature in features
                    for kind in ("linear", "mlp")
                },
                "control_logits": {
                    f"{feature}/mlp": float(all_logits[f"{feature}/control_mlp"][index]) for feature in features
                },
            }
        )
    probe_states = serialize_probe_states(
        fitted,
        preprocessing,
        candidate_sha256=candidate_identity["sha256"],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    latent = _separable(metrics["C_selected_graph_action"]["mlp"], metrics["C_selected_graph_action"]["control_mlp"])
    raw = _separable(metrics["D_raw_symbolic"]["mlp"], metrics["D_raw_symbolic"]["control_mlp"])
    hybrid = _separable(metrics["E_hybrid"]["mlp"], metrics["E_hybrid"]["control_mlp"])
    verdicts = {
        "latent_separable": latent,
        "raw_separable": raw,
        "hybrid_separable": hybrid,
        "label_or_sampling_blocker": not raw,
        "latent_state_bottleneck": raw and not latent,
        "any_abc_separable": any(
            _separable(metrics[name][kind], metrics[name]["control_mlp"])
            for name in ("A_action", "B_graph_action", "C_selected_graph_action")
            for kind in ("linear", "mlp")
        ),
    }
    summary = {
        "schema_version": "action_latent_updated_phase0.applicability_recoverability.v1",
        "kind": "applicability_recoverability",
        "dataset": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": args.split,
        "seed": args.seed,
        "candidate_manifest": candidate_identity,
        "settings": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "models": ["linear", "mlp", "control_mlp"],
            "feature_sets": list(features),
            "threshold_policy": "max_train_f1_highest_threshold",
            "control_policy": "train_label_permutation_seed_20260717",
            "reliability_bins": [index / 10 for index in range(11)],
        },
        "checkpoint_restoration": restoration,
        "counts": {
            "records": 604,
            "train_records": 453,
            "eval_records": 151,
            "train_groups": 33,
            "eval_groups": 11,
            "applicable": int(labels.sum()),
            "inapplicable": int((labels == 0).sum()),
        },
        "metrics": {"features": metrics, "verdicts": verdicts},
        "environment": _environment(),
        "device": "cpu",
        "output": str(args.output),
        "runtime_seconds": time.perf_counter() - started,
    }
    feature_artifact = {
        "schema_version": "action_latent_updated_phase0.feature_schema.v1",
        "candidate_manifest_sha256": candidate_identity["sha256"],
        "feature_sets": schemas,
    }
    for name, value in (
        ("summary.json", summary),
        ("details.json", details),
        ("feature_schema.json", feature_artifact),
        ("split_manifest.json", split_manifest()),
        ("probe_states.json", probe_states),
    ):
        (args.output / name).write_bytes(canonical_json_bytes(value))
    return summary


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
