"""Train and evaluate frozen-representation action supervision probes."""

from __future__ import annotations

import argparse
import random
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
from torch import Tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
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


def _applicability_logits(model: ApplicabilityHead, inputs: dict[str, Tensor]) -> Tensor:
    return model(
        inputs["graph_latents"],
        inputs["action_latents"],
        inputs["object_latents"],
        inputs["argument_mask"],
    )


def applicability_metrics(
    logits: Tensor,
    labels: Tensor,
    *,
    group_ids: Sequence[Hashable],
    categories: Sequence[str],
) -> dict[str, object]:
    """Report binary probe quality and true-minus-hard-negative logit margins."""

    logits = logits.detach().to(dtype=torch.float32, device="cpu").flatten()
    labels = labels.detach().to(dtype=torch.float32, device="cpu").flatten()
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have matching shapes")
    if len(group_ids) != logits.numel() or len(categories) != logits.numel():
        raise ValueError("group_ids and categories must match logits")
    predictions = logits >= 0
    accuracy = float((predictions == labels.bool()).float().mean().item()) if logits.numel() else None
    positive = logits[labels == 1]
    negative = logits[labels == 0]
    auroc = _binary_auroc(positive, negative)

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
    return {
        "count": int(logits.numel()),
        "positive_count": int(positive.numel()),
        "negative_count": int(negative.numel()),
        "accuracy": accuracy,
        "auroc": auroc,
        "margin": _distribution(margins),
        "margin_by_category": {category: _distribution(values) for category, values in sorted(by_category.items())},
    }


def _binary_auroc(positive: Tensor, negative: Tensor) -> float | None:
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().item())


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


def collect_probe_examples(args: argparse.Namespace) -> tuple[list[ProbeExample], dict[str, Any], torch.device]:
    """Encode trace actions and hard negatives into frozen probe examples."""

    device_name = _automatic_device_name() if args.device == "auto" else args.device
    config, corpus, bundle, device = load_checkpoint_bundle(
        args.dataset_dir,
        args.checkpoint,
        device_name=device_name,
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
    return examples, metadata, device


def _repeat_latent_state(state: JEPALatentState, repeats: int, *, device: torch.device) -> JEPALatentState:
    object_count = state.object_latents.size(0)
    return JEPALatentState(
        graph_latent=state.graph_latent.to(device).expand(repeats, -1).contiguous(),
        object_latents=state.object_latents.to(device).repeat(repeats, 1),
        object_ids=state.object_ids.to(device).repeat(repeats),
        object_batch=torch.arange(repeats, device=device).repeat_interleave(object_count),
    )


def _automatic_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stack_role_examples(
    examples: Sequence[ProbeExample],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Flatten active argument slots and pad problem-local object banks."""

    rows = [
        (example, role)
        for example in examples
        for role, active in enumerate(example.argument_mask)
        if bool(active)
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


def run_probes(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    examples, metadata, device = collect_probe_examples(args)
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
        "probes": {
            "schema": asdict(schema_result),
            "role_object": asdict(role_result),
            "applicability": asdict(applicability_result),
        },
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
