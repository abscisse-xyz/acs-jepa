"""Train and evaluate frozen-representation action supervision probes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Hashable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from acs_jepa import ApplicabilityHead, JEPALatentState
from acs_jepa.architectures import ActionDecodingSpace
from action_diag_common import (
    action_key,
    applicable_keys,
    encode_state,
    iter_transitions,
    load_checkpoint_bundle,
    replay_to_engine,
    select_split,
    write_json,
)
from action_negative_sampling import sample_action_negatives
from action_phase0_common import tie_aware_auroc, tie_aware_average_precision
from torch import Tensor

SUMMARY_KEYS = frozenset(
    {
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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("supervised probes must run on CPU")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")
    if args.per_category <= 0:
        raise ValueError("--per-category must be positive")
    if not 0.0 < args.eval_fraction < 1.0:
        raise ValueError("--eval-fraction must lie in (0, 1)")
    if args.max_transitions is not None and args.max_transitions <= 1:
        raise ValueError("--max-transitions must exceed one")


@dataclass(frozen=True)
class ProbeSplit:
    """Deterministic group-disjoint probe train/eval split."""

    train_groups: tuple[Hashable, ...]
    eval_groups: tuple[Hashable, ...]
    train_examples: int
    eval_examples: int


@dataclass(frozen=True)
class ProbeTrainingResult:
    """Metrics from one frozen-representation probe fit."""

    train_metrics: dict[str, object]
    eval_metrics: dict[str, object]


@dataclass(frozen=True)
class ProbeExample:
    """One frozen state/action example; labels remain separate from probe inputs."""

    group_id: Hashable
    problem: str
    category: str
    action_name: str
    action_arguments: tuple[str, ...]
    schema_id: int
    applicability_label: bool
    graph_latent: Tensor
    action_latent: Tensor
    selected_object_latents: Tensor
    argument_mask: Tensor
    object_bank: Tensor
    argument_targets: Tensor
    step: int = 0
    argument_candidate_mask: Tensor | None = None


class RoleObjectProbe(nn.Module):
    """Retrieve a problem-local argument object from state object latents."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        max_action_arity: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.max_action_arity = int(max_action_arity)
        self.role_embedding = nn.Embedding(max_action_arity, latent_dim)
        self.query = nn.Sequential(
            nn.Linear(latent_dim * 2 + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        graph_latents: Tensor,
        action_latents: Tensor,
        object_latents: Tensor,
        object_mask: Tensor,
        role_ids: Tensor,
    ) -> Tensor:
        if object_mask.dtype != torch.bool:
            raise ValueError("object_mask must be bool")
        if role_ids.numel() and (int(role_ids.min()) < 0 or int(role_ids.max()) >= self.max_action_arity):
            raise ValueError("role_ids exceed max_action_arity")
        role_features = self.role_embedding(role_ids)
        query = self.query(torch.cat([graph_latents, action_latents, role_features], dim=-1))
        logits = torch.einsum("bd,bnd->bn", query, object_latents)
        return logits.masked_fill(~object_mask, float("-inf"))


def deterministic_group_split(
    group_ids: Sequence[Hashable],
    *,
    eval_fraction: float,
    seed: int,
) -> ProbeSplit:
    """Split examples by group so one source transition cannot leak across partitions."""

    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must lie in (0, 1)")
    groups = list(dict.fromkeys(group_ids))
    if len(groups) < 2:
        raise ValueError("at least two groups are required for a train/eval split")
    random.Random(seed).shuffle(groups)
    eval_count = max(1, min(len(groups) - 1, round(len(groups) * eval_fraction)))
    eval_groups = tuple(groups[:eval_count])
    train_groups = tuple(groups[eval_count:])
    counts = Counter(group_ids)
    return ProbeSplit(
        train_groups=train_groups,
        eval_groups=eval_groups,
        train_examples=sum(counts[group] for group in train_groups),
        eval_examples=sum(counts[group] for group in eval_groups),
    )


def argument_features(
    state: JEPALatentState, action_tensors: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gather candidate argument latents and problem-local retrieval targets."""

    object_ids = state.object_ids
    object_latents = state.object_latents
    order = torch.argsort(object_ids)
    object_bank = object_latents[order]
    sorted_ids = object_ids[order]
    id_to_position = {int(object_id): position for position, object_id in enumerate(sorted_ids.tolist())}
    argument_mask = action_tensors["action_arg_mask"].bool()
    object_indices = action_tensors["action_object_indices"].long()
    selected = object_bank.new_zeros((*object_indices.shape, object_bank.size(-1)))
    targets = torch.full_like(object_indices, -1)
    for batch_idx, role_idx in argument_mask.nonzero(as_tuple=False).tolist():
        object_id = int(object_indices[batch_idx, role_idx].item())
        position = id_to_position[object_id]
        selected[batch_idx, role_idx] = object_bank[position]
        targets[batch_idx, role_idx] = position
    return selected, argument_mask, object_bank, targets


def argument_candidate_masks(
    parsed: Any,
    state: JEPALatentState,
    action_names: Sequence[str],
    argument_mask: Tensor,
) -> Tensor:
    """Build exact-type role masks in the state bank's sorted object-ID order."""

    sorted_ids = state.object_ids[torch.argsort(state.object_ids)].tolist()
    id_to_name = {object_id: name for name, object_id in parsed.object_to_id.items()}
    output = torch.zeros(
        (len(action_names), argument_mask.size(1), len(sorted_ids)),
        dtype=torch.bool,
        device=argument_mask.device,
    )
    for batch_index, action_name in enumerate(action_names):
        parameter_types = parsed.actions[action_name].parameter_types
        for role_index, expected_type in enumerate(parameter_types):
            if not bool(argument_mask[batch_index, role_index]):
                continue
            for bank_index, object_id in enumerate(sorted_ids):
                object_name = id_to_name[int(object_id)]
                if parsed.objects[object_name].type == expected_type:
                    output[batch_index, role_index, bank_index] = True
    return output


def stack_applicability_examples(examples: Sequence[ProbeExample]) -> tuple[dict[str, Tensor], Tensor]:
    """Stack applicability features while keeping offline oracle labels separate."""

    if not examples:
        raise ValueError("at least one applicability example is required")
    inputs = {
        "graph_latents": torch.stack([example.graph_latent for example in examples]),
        "action_latents": torch.stack([example.action_latent for example in examples]),
        "object_latents": torch.stack([example.selected_object_latents for example in examples]),
        "argument_mask": torch.stack([example.argument_mask for example in examples]),
    }
    labels = torch.tensor([example.applicability_label for example in examples], dtype=torch.float32)
    return inputs, labels


def train_schema_probe(
    train_latents: Tensor,
    train_labels: Tensor,
    eval_latents: Tensor,
    eval_labels: Tensor,
    *,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> ProbeTrainingResult:
    """Fit a linear schema probe on frozen action latents."""

    torch.manual_seed(seed)
    model = nn.Linear(train_latents.size(-1), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train = train_latents.detach().to(device)
    y_train = train_labels.detach().to(device=device, dtype=torch.long)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return ProbeTrainingResult(
        train_metrics=_classification_metrics(model(x_train), y_train),
        eval_metrics=_classification_metrics(
            model(eval_latents.detach().to(device)),
            eval_labels.detach().to(device=device, dtype=torch.long),
        ),
    )


def _classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, object]:
    predictions = logits.argmax(dim=-1)
    return {
        "count": int(labels.numel()),
        "accuracy": None if labels.numel() == 0 else float((predictions == labels).float().mean().item()),
    }


def train_role_probe(
    train_data: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    eval_data: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    *,
    max_action_arity: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> ProbeTrainingResult:
    """Fit a role-aware object retrieval probe over problem-local object banks."""

    torch.manual_seed(seed)
    train = tuple(tensor.detach().to(device) for tensor in train_data)
    evaluation = tuple(tensor.detach().to(device) for tensor in eval_data)
    model = RoleObjectProbe(
        latent_dim=train[0].size(-1),
        action_dim=train[1].size(-1),
        max_action_arity=max_action_arity,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*train[:5])
        loss = F.cross_entropy(logits, train[5].long())
        loss.backward()
        optimizer.step()
    return ProbeTrainingResult(
        train_metrics=_role_metrics(model(*train[:5]), train[5].long(), train[4].long()),
        eval_metrics=_role_metrics(model(*evaluation[:5]), evaluation[5].long(), evaluation[4].long()),
    )


def _role_metrics(logits: Tensor, targets: Tensor, role_ids: Tensor) -> dict[str, object]:
    metrics = _classification_metrics(logits, targets)
    predictions = logits.argmax(dim=-1)
    metrics["per_role_accuracy"] = {
        str(role): float((predictions[role_ids == role] == targets[role_ids == role]).float().mean().item())
        for role in sorted(set(role_ids.tolist()))
    }
    return metrics


def train_applicability_probe(
    train_data: tuple[dict[str, Tensor], Tensor, Sequence[Hashable], Sequence[str]],
    eval_data: tuple[dict[str, Tensor], Tensor, Sequence[Hashable], Sequence[str]],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> ProbeTrainingResult:
    """Fit the role-aware applicability head on frozen diagnostic features."""

    torch.manual_seed(seed)
    train_inputs, train_labels, train_groups, train_categories = train_data
    eval_inputs, eval_labels, eval_groups, eval_categories = eval_data
    train_features = {key: value.detach().to(device) for key, value in train_inputs.items()}
    eval_features = {key: value.detach().to(device) for key, value in eval_inputs.items()}
    train_targets = train_labels.detach().to(device=device, dtype=torch.float32)
    model = ApplicabilityHead(
        latent_dim=train_features["graph_latents"].size(-1),
        action_dim=train_features["action_latents"].size(-1),
        max_action_arity=train_features["argument_mask"].size(-1),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = _applicability_logits(model, train_features)
        loss = F.binary_cross_entropy_with_logits(logits, train_targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_logits = _applicability_logits(model, train_features)
        eval_logits = _applicability_logits(model, eval_features)
    return ProbeTrainingResult(
        train_metrics=applicability_metrics(
            train_logits,
            train_labels,
            group_ids=train_groups,
            categories=train_categories,
        ),
        eval_metrics=applicability_metrics(
            eval_logits,
            eval_labels,
            group_ids=eval_groups,
            categories=eval_categories,
        ),
    )


def _applicability_logits(model: nn.Module, inputs: dict[str, Tensor]) -> Tensor:
    return model(
        inputs["graph_latents"],
        inputs["action_latents"],
        inputs["object_latents"],
        inputs["argument_mask"],
    )


def binary_metrics(logits: Tensor, labels: Tensor) -> dict[str, object]:
    """Compute threshold-zero and exact tie-group binary ranking metrics."""

    logits = logits.detach().to(dtype=torch.float64, device="cpu").flatten()
    labels = labels.detach().to(dtype=torch.float64, device="cpu").flatten()
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have matching shapes")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("logits and labels must be finite")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("labels must be binary")
    count = int(labels.numel())
    positive_count = int((labels == 1).sum())
    negative_count = count - positive_count
    if count == 0:
        return {
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "positive_prevalence": None,
            "average_precision": None,
        }

    predicted_positive = logits >= 0
    positive_labels = labels.bool()
    true_positive = int((predicted_positive & positive_labels).sum())
    false_positive = int((predicted_positive & ~positive_labels).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / positive_count if positive_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive = logits[positive_labels]
    negative = logits[~positive_labels]
    ranking_defined = positive_count > 0 and negative_count > 0
    return {
        "count": count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "accuracy": float((predicted_positive == positive_labels).to(torch.float64).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _binary_auroc(positive, negative) if ranking_defined else None,
        "positive_prevalence": positive_count / count,
        "average_precision": _average_precision(logits, positive_labels) if ranking_defined else None,
    }


def _average_precision(logits: Tensor, positive_labels: Tensor) -> float:
    """Integrate precision at complete descending equal-score group boundaries."""
    return tie_aware_average_precision(logits, positive_labels)


def applicability_metrics(
    logits: Tensor,
    labels: Tensor,
    *,
    group_ids: Sequence[Hashable],
    categories: Sequence[str],
) -> dict[str, object]:
    """Report binary probe quality and true-minus-hard-negative logit margins."""

    logits = logits.detach().to(dtype=torch.float64, device="cpu").flatten()
    labels = labels.detach().to(dtype=torch.float64, device="cpu").flatten()
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have matching shapes")
    if len(group_ids) != logits.numel() or len(categories) != logits.numel():
        raise ValueError("group_ids and categories must match logits")
    metrics = binary_metrics(logits, labels)

    references: dict[Hashable, float] = {}
    for logit, label, group, category in zip(logits.tolist(), labels.tolist(), group_ids, categories):
        if label == 1.0 and category == "trace":
            references[group] = float(logit)
    margins: list[float] = []
    by_category: dict[str, list[float]] = defaultdict(list)
    for logit, label, group, category in zip(logits.tolist(), labels.tolist(), group_ids, categories):
        if label != 0.0 or group not in references:
            continue
        margin = references[group] - float(logit)
        margins.append(margin)
        by_category[category].append(margin)
    per_category: dict[str, dict[str, object]] = {}
    for category in sorted(set(categories) - {"trace"}):
        selected = [
            index
            for index, value in enumerate(categories)
            if value == category or (value == "trace" and labels[index] == 1)
        ]
        per_category[category] = binary_metrics(logits[selected], labels[selected])
    metrics.update(
        {
            "margin": _distribution(margins),
            "margin_by_category": {category: _distribution(values) for category, values in sorted(by_category.items())},
            "per_category": per_category,
        }
    )
    return metrics


def _binary_auroc(positive: Tensor, negative: Tensor) -> float | None:
    return tie_aware_auroc(positive, negative)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "min": float(tensor.min().item()),
        "median": float(tensor.median().item()),
        "mean": float(tensor.mean().item()),
        "max": float(tensor.max().item()),
    }


def canonical_manifest_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    """Return the exact canonical UTF-8 identity bytes for diagnostic examples."""

    ordered = sorted(
        records,
        key=lambda record: (
            record["group"],
            record["category"],
            record["action"]["name"],
            tuple(record["action"]["arguments"]),
            record["applicability_label"],
            record["problem"],
            record["step"],
        ),
    )
    text = json.dumps(ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return text.encode("utf-8")


def manifest_identity(records: Sequence[dict[str, Any]]) -> dict[str, object]:
    canonical = canonical_manifest_bytes(records)
    return {
        "count": len(records),
        "bytes": len(canonical),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def decision_projection(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain every decision value except the six fixed runtime/path pointers."""

    if set(summary) != SUMMARY_KEYS:
        raise ValueError(f"summary top-level keys must be exactly {sorted(SUMMARY_KEYS)}; got {sorted(summary)}")
    projection = copy.deepcopy(summary)
    for key in ("dataset", "checkpoint", "device", "runtime_seconds", "environment"):
        del projection[key]
    manifest = projection["example_manifest"]
    if not isinstance(manifest, dict) or "path" not in manifest:
        raise ValueError("example_manifest.path is required")
    del manifest["path"]
    return projection


def _argument_metric_subset(
    logits: Tensor,
    targets: Tensor,
    active: Tensor,
    candidate_mask: Tensor,
) -> dict[str, object]:
    valid_counts: list[float] = []
    correct = 0
    chance = 0.0
    margins: list[float] = []
    for batch_index, role_index in active.nonzero(as_tuple=False).tolist():
        valid = candidate_mask[batch_index, role_index]
        valid_count = int(valid.sum())
        target = int(targets[batch_index, role_index])
        valid_counts.append(float(valid_count))
        chance += 1.0 / valid_count
        row = logits[batch_index, role_index].masked_fill(~valid, float("-inf"))
        correct += int(int(row.argmax()) == target)
        if valid_count > 1:
            wrong = valid.clone()
            wrong[target] = False
            margins.append(float(row[target] - row[wrong].max()))
    count = len(valid_counts)
    return {
        "active_role_count": count,
        "competitive_role_count": len(margins),
        "top1_accuracy": None if count == 0 else correct / count,
        "chance_accuracy": None if count == 0 else chance / count,
        "valid_candidate_count": _distribution(valid_counts),
        "target_minus_best_wrong_margin": _distribution(margins),
    }


def argument_head_metrics(
    logits: Tensor,
    targets: Tensor,
    argument_mask: Tensor,
    candidate_mask: Tensor,
) -> dict[str, object]:
    """Compute exact active-role and competitive-role reconstruction diagnostics."""

    logits = logits.detach().cpu()
    targets = targets.detach().to(device="cpu", dtype=torch.long)
    argument_mask = argument_mask.detach().cpu()
    candidate_mask = candidate_mask.detach().cpu()
    if argument_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
        raise ValueError("argument_mask and candidate_mask must be bool")
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or argument_mask.shape != targets.shape:
        raise ValueError("argument tensors must have shapes [B,R,O], [B,R], [B,R]")
    if candidate_mask.shape != logits.shape:
        raise ValueError("candidate_mask must match logits")
    if bool(candidate_mask[~argument_mask].any()):
        raise ValueError("inactive roles must have no candidates")
    batch_indices, role_indices = argument_mask.nonzero(as_tuple=True)
    active_targets = targets[argument_mask]
    if bool(((active_targets < 0) | (active_targets >= logits.size(2))).any()):
        raise ValueError("active target must index the object bank")
    if active_targets.numel() and not bool(candidate_mask[batch_indices, role_indices, active_targets].all()):
        raise ValueError("active target must be true in candidate_mask")
    if not bool(torch.isfinite(logits[candidate_mask]).all()):
        raise ValueError("valid candidate logits must be finite")
    role_grid = torch.arange(targets.size(1))[None, :]
    return {
        "overall": _argument_metric_subset(logits, targets, argument_mask, candidate_mask),
        "per_role": {
            str(role): _argument_metric_subset(logits, targets, argument_mask & (role_grid == role), candidate_mask)
            for role in range(4)
        },
    }


def evaluate_checkpoint_argument_head(
    head: nn.Module,
    action_latents: Tensor,
    object_banks: Sequence[Tensor],
    targets: Sequence[Tensor],
    argument_masks: Sequence[Tensor],
    candidate_masks: Sequence[Tensor],
) -> dict[str, object]:
    """Pad trace examples and invoke the untouched production head exactly once."""

    batch_size = action_latents.size(0)
    if not (len(object_banks) == len(targets) == len(argument_masks) == len(candidate_masks) == batch_size):
        raise ValueError("argument-head inputs must have matching batch sizes")
    max_objects = max(bank.size(0) for bank in object_banks)
    max_roles = targets[0].numel()
    dense_banks = object_banks[0].new_zeros((batch_size, max_objects, object_banks[0].size(-1)))
    dense_targets = torch.full((batch_size, max_roles), -1, dtype=torch.long)
    dense_argument_mask = torch.zeros((batch_size, max_roles), dtype=torch.bool)
    dense_candidate_mask = torch.zeros((batch_size, max_roles, max_objects), dtype=torch.bool)
    for index, (bank, target, active, candidates) in enumerate(
        zip(object_banks, targets, argument_masks, candidate_masks, strict=True)
    ):
        if target.numel() != max_roles or active.shape != target.shape:
            raise ValueError("all argument rows must have the same role capacity")
        if candidates.shape != (max_roles, bank.size(0)):
            raise ValueError("candidate mask must match role and real object counts")
        count = bank.size(0)
        dense_banks[index, :count] = bank
        dense_targets[index] = target
        dense_argument_mask[index] = active
        dense_candidate_mask[index, :, :count] = candidates
    with torch.no_grad():
        logits = head(action_latents, dense_banks, dense_candidate_mask)
    return argument_head_metrics(logits, dense_targets, dense_argument_mask, dense_candidate_mask)


def collect_probe_examples(
    args: argparse.Namespace,
) -> tuple[list[ProbeExample], dict[str, Any], torch.device, Any, dict[str, Any]]:
    """Encode trace actions and hard negatives into frozen probe examples."""

    config, corpus, bundle, device, restoration = load_checkpoint_bundle(
        args.dataset_dir,
        args.checkpoint,
        device_name=args.device,
        include_restoration_metadata=True,
    )
    selected = select_split(corpus, config, args.split, seed=args.seed)
    spaces: dict[int, ActionDecodingSpace] = {}
    examples: list[ProbeExample] = []
    transition_count = 0
    max_action_arity = 0
    schema_names: set[str] = set()

    with torch.no_grad():
        for transition_index, (trajectory, record, step_idx, true_action) in enumerate(
            iter_transitions(selected, max_transitions=args.max_transitions)
        ):
            parsed = selected.parsed_problems[trajectory.problem_index]
            space = spaces.setdefault(trajectory.problem_index, ActionDecodingSpace.from_parsed_problem(parsed))
            negatives = sample_action_negatives(
                space,
                true_action,
                per_category=args.per_category,
                seed=args.seed + transition_index,
            )
            actions = [true_action, *(negative.action for negative in negatives)]
            categories = ["trace", *(negative.category for negative in negatives)]
            engine = replay_to_engine(args.dataset_dir, record.problem_name, trajectory.actions[:step_idx])
            applicable = applicable_keys(engine)
            state = encode_state(bundle, parsed, trajectory.states[step_idx], device=device)
            action_tensors = space.action_tensors_for_ground_actions(actions, device=device)
            repeated_state = _repeat_latent_state(state, len(actions), device=device)
            action_latents = bundle.jepa.action_encoder(action_tensors, repeated_state)
            selected_objects, argument_mask, object_bank, argument_targets = argument_features(state, action_tensors)
            candidate_masks = argument_candidate_masks(
                parsed,
                state,
                [action.name for action in actions],
                argument_mask,
            )
            graph_latent = state.graph_latent.squeeze(0)
            group_id = f"{record.problem_name}:{step_idx}"
            max_action_arity = max(max_action_arity, space.max_action_arity)
            transition_count += 1
            for index, (action, category) in enumerate(zip(actions, categories)):
                schema_names.add(action.name)
                examples.append(
                    ProbeExample(
                        group_id=group_id,
                        problem=record.problem_name,
                        category=category,
                        action_name=action.name,
                        action_arguments=tuple(action.arguments),
                        schema_id=space.action_id(action.name),
                        applicability_label=action_key(action) in applicable,
                        graph_latent=graph_latent.detach().cpu(),
                        action_latent=action_latents[index].detach().cpu(),
                        selected_object_latents=selected_objects[index].detach().cpu(),
                        argument_mask=argument_mask[index].detach().cpu(),
                        object_bank=object_bank.detach().cpu(),
                        argument_targets=argument_targets[index].detach().cpu(),
                        step=step_idx,
                        argument_candidate_mask=candidate_masks[index].detach().cpu(),
                    )
                )
    if not examples:
        raise ValueError("selected split produced no probe examples")
    metadata = {
        "transitions": transition_count,
        "examples": len(examples),
        "schemas": sorted(schema_names),
        "max_action_arity": max_action_arity,
    }
    return examples, metadata, device, bundle, restoration


def _repeat_latent_state(state: JEPALatentState, repeats: int, *, device: torch.device) -> JEPALatentState:
    object_count = state.object_latents.size(0)
    return JEPALatentState(
        graph_latent=state.graph_latent.to(device).expand(repeats, -1).contiguous(),
        object_latents=state.object_latents.to(device).repeat(repeats, 1),
        object_ids=state.object_ids.to(device).repeat(repeats),
        object_batch=torch.arange(repeats, device=device).repeat_interleave(object_count),
    )


def stack_role_examples(
    examples: Sequence[ProbeExample],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Flatten active argument slots and pad problem-local object banks."""

    rows = [
        (example, role) for example in examples for role, active in enumerate(example.argument_mask) if bool(active)
    ]
    if not rows:
        raise ValueError("probe examples contain no active action arguments")
    max_objects = max(example.object_bank.size(0) for example, _ in rows)
    latent_dim = rows[0][0].graph_latent.size(-1)
    banks = torch.zeros((len(rows), max_objects, latent_dim), dtype=rows[0][0].object_bank.dtype)
    bank_mask = torch.zeros((len(rows), max_objects), dtype=torch.bool)
    graph_latents = []
    action_latents = []
    role_ids = []
    targets = []
    for index, (example, role) in enumerate(rows):
        count = example.object_bank.size(0)
        banks[index, :count] = example.object_bank
        bank_mask[index, :count] = True
        graph_latents.append(example.graph_latent)
        action_latents.append(example.action_latent)
        role_ids.append(role)
        targets.append(int(example.argument_targets[role]))
    return (
        torch.stack(graph_latents),
        torch.stack(action_latents),
        banks,
        bank_mask,
        torch.tensor(role_ids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def _applicability_dataset(
    examples: Sequence[ProbeExample],
) -> tuple[dict[str, Tensor], Tensor, list[Hashable], list[str]]:
    inputs, labels = stack_applicability_examples(examples)
    return inputs, labels, [example.group_id for example in examples], [example.category for example in examples]


def _evaluate_checkpoint_applicability_head(
    head: nn.Module,
    train_examples: Sequence[ProbeExample],
    eval_examples: Sequence[ProbeExample],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, examples in (("train_metrics", train_examples), ("eval_metrics", eval_examples)):
        inputs, labels, groups, categories = _applicability_dataset(examples)
        with torch.no_grad():
            logits = _applicability_logits(head, inputs)
        output[name] = applicability_metrics(logits, labels, group_ids=groups, categories=categories)
    return output


def _manifest_records(examples: Sequence[ProbeExample]) -> list[dict[str, Any]]:
    return [
        {
            "group": example.group_id,
            "problem": example.problem,
            "step": example.step,
            "category": example.category,
            "action": {
                "name": example.action_name,
                "arguments": list(example.action_arguments),
            },
            "applicability_label": example.applicability_label,
        }
        for example in examples
    ]


def _environment() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "backend": "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "byteorder": sys.byteorder,
    }


def run_probes(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    examples, metadata, device, bundle, restoration = collect_probe_examples(args)
    split = deterministic_group_split(
        [example.group_id for example in examples],
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    train_group_set = set(split.train_groups)
    eval_group_set = set(split.eval_groups)
    train_examples = [example for example in examples if example.group_id in train_group_set]
    eval_examples = [example for example in examples if example.group_id in eval_group_set]

    schema_train_x = torch.stack([example.action_latent for example in train_examples])
    schema_train_y = torch.tensor([example.schema_id for example in train_examples], dtype=torch.long)
    schema_eval_x = torch.stack([example.action_latent for example in eval_examples])
    schema_eval_y = torch.tensor([example.schema_id for example in eval_examples], dtype=torch.long)
    num_classes = 1 + max(example.schema_id for example in examples)
    schema_result = train_schema_probe(
        schema_train_x,
        schema_train_y,
        schema_eval_x,
        schema_eval_y,
        num_classes=num_classes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    role_result = train_role_probe(
        stack_role_examples(train_examples),
        stack_role_examples(eval_examples),
        max_action_arity=metadata["max_action_arity"],
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    applicability_result = train_applicability_probe(
        _applicability_dataset(train_examples),
        _applicability_dataset(eval_examples),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    checkpoint_applicability = None
    if bundle.applicability_head is not None:
        checkpoint_applicability = _evaluate_checkpoint_applicability_head(
            bundle.applicability_head, train_examples, eval_examples
        )
    checkpoint_argument = None
    if bundle.argument_reconstruction_head is not None:
        trace_examples = [example for example in examples if example.category == "trace"]
        candidate_masks = [example.argument_candidate_mask for example in trace_examples]
        if any(mask is None for mask in candidate_masks):
            raise ValueError("trace argument candidate masks are required")
        checkpoint_argument = evaluate_checkpoint_argument_head(
            bundle.argument_reconstruction_head,
            torch.stack([example.action_latent for example in trace_examples]),
            [example.object_bank for example in trace_examples],
            [example.argument_targets for example in trace_examples],
            [example.argument_mask for example in trace_examples],
            [mask for mask in candidate_masks if mask is not None],
        )
    records = _manifest_records(examples)
    canonical = canonical_manifest_bytes(records)
    manifest_path = args.output / "example_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical)
    example_manifest = {"path": str(manifest_path), **manifest_identity(records)}
    label_counts = Counter("applicable" if example.applicability_label else "inapplicable" for example in examples)
    category_counts = Counter(example.category for example in examples)
    return {
        "dataset": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "per_category": args.per_category,
        "eval_fraction": args.eval_fraction,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "metadata": metadata,
        "probe_split": asdict(split),
        "label_counts": dict(sorted(label_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "example_manifest": example_manifest,
        "checkpoint_restoration": restoration,
        "probes": {
            "schema": asdict(schema_result),
            "role_object": asdict(role_result),
            "applicability": asdict(applicability_result),
        },
        "checkpoint_applicability_head": checkpoint_applicability,
        "checkpoint_argument_reconstruction_head": checkpoint_argument,
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    summary = run_probes(args)
    details = {
        "probe_split": summary["probe_split"],
        "metric_definition": {
            "hard_negative_margin": (
                "trace applicability logit minus inapplicable sampled-action logit within one transition"
            ),
        },
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "details.json", details)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
