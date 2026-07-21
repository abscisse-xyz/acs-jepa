"""Diagnose action-latent distribution statistics for a trained checkpoint."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from acs_jepa.architectures import ActionDecodingSpace, JEPALatentState
from action_diag_common import (
    action_key,
    action_payload,
    encode_state,
    iter_transitions,
    load_checkpoint_bundle,
    select_split,
    write_json,
)
from action_latent_statistics import (
    latent_distribution_stats,
    reference_same_schema_margins,
    same_schema_nearest_wrong_margins,
    schema_argument_variance_decomposition,
    schema_group_stats,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--max-candidates-per-state", type=int, default=None)
    parser.add_argument("--same-schema-only", action="store_true")
    parser.add_argument("--min-schema-count", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omit-details", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.max_candidates_per_state is not None and args.max_candidates_per_state <= 0:
        raise ValueError("--max-candidates-per-state must be positive when provided")
    if args.min_schema_count <= 0:
        raise ValueError("--min-schema-count must be positive")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config, corpus, bundle, device, restoration = load_checkpoint_bundle(
        args.dataset_dir,
        args.checkpoint,
        device_name=args.device,
        include_restoration_metadata=True,
    )
    selected = select_split(corpus, config, args.split, seed=args.seed)
    space_cache: dict[int, tuple[ActionDecodingSpace, tuple[Any, ...], dict[tuple[str, tuple[str, ...]], int]]] = {}

    latent_chunks: list[torch.Tensor] = []
    schema_ids: list[str] = []
    action_keys: list[tuple[str, tuple[str, ...]]] = []
    transition_group_ids: list[int] = []
    reference_mask: list[bool] = []
    details: list[dict[str, Any]] = []
    total_candidates = 0
    sampled_transitions = 0
    retained_true_count = 0

    with torch.no_grad():
        for transition_index, (trajectory, record, step_idx, true_action) in enumerate(
            iter_transitions(selected, max_transitions=args.max_transitions)
        ):
            parsed = selected.parsed_problems[trajectory.problem_index]
            if trajectory.problem_index not in space_cache:
                space = ActionDecodingSpace.from_parsed_problem(parsed)
                actions = space.enumerate_ground_actions()
                index = {action_key(action): idx for idx, action in enumerate(actions)}
                space_cache[trajectory.problem_index] = (space, actions, index)
            space, all_actions, action_index = space_cache[trajectory.problem_index]
            candidate_indices = _candidate_indices(
                all_actions,
                action_index[action_key(true_action)],
                true_action.name,
                same_schema_only=args.same_schema_only,
                max_candidates=args.max_candidates_per_state,
                seed=args.seed + transition_index,
            )
            candidates = tuple(all_actions[idx] for idx in candidate_indices)
            retained_true = any(action_key(action) == action_key(true_action) for action in candidates)
            retained_true_count += int(retained_true)

            latent_state = encode_state(bundle, parsed, trajectory.states[step_idx], device=device)
            latents = _encode_candidates(
                bundle,
                space,
                candidates,
                latent_state,
                chunk_size=args.chunk_size,
                device=device,
            )
            latent_chunks.append(latents.cpu())
            schema_ids.extend(action.name for action in candidates)
            action_keys.extend(action_key(action) for action in candidates)
            transition_group_ids.extend([transition_index] * len(candidates))
            reference_mask.extend(action_key(action) == action_key(true_action) for action in candidates)
            total_candidates += len(candidates)
            sampled_transitions += 1
            details.append(
                {
                    "problem": record.problem_name,
                    "step_index": step_idx,
                    "true_action": action_payload(true_action),
                    "candidate_count": len(candidates),
                    "full_candidate_count": len(all_actions),
                    "same_schema_only": args.same_schema_only,
                    "retained_true_action": retained_true,
                }
            )

    all_latents = torch.cat(latent_chunks, dim=0) if latent_chunks else torch.empty((0, 0), dtype=torch.float32)
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset_dir),
        "split": args.split,
        "max_transitions": args.max_transitions,
        "max_candidates_per_state": args.max_candidates_per_state,
        "same_schema_only": args.same_schema_only,
        "min_schema_count": args.min_schema_count,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "checkpoint_restoration": restoration,
        "metrics": {
            "transitions": sampled_transitions,
            "total_candidate_latents": total_candidates,
            "retained_true_action_rate": 0.0 if sampled_transitions == 0 else retained_true_count / sampled_transitions,
            "runtime_seconds": time.perf_counter() - started,
        },
        "global": latent_distribution_stats(all_latents),
        "per_schema": schema_group_stats(all_latents, schema_ids, min_count=args.min_schema_count),
        "same_schema_nearest_wrong": same_schema_nearest_wrong_margins(
            all_latents,
            schema_ids,
            action_keys,
            group_ids=transition_group_ids,
        ),
        "reference_same_schema_margins": _reference_same_schema_margin_metrics(
            all_latents,
            schema_ids,
            action_keys,
            reference_mask,
            transition_group_ids,
        ),
        "schema_argument_variance": schema_argument_variance_decomposition(all_latents, schema_ids),
    }
    if args.omit_details:
        summary = _without_details(summary)
    write_json(args.output / "summary.json", summary)
    if not args.omit_details:
        write_json(args.output / "details.json", details)
    print(summary)
    return 0


def _reference_same_schema_margin_metrics(
    latents: torch.Tensor,
    schema_ids: list[str],
    action_keys: list[tuple[str, tuple[str, ...]]],
    reference_mask: list[bool],
    group_ids: list[Any],
) -> dict[str, Any]:
    """Return legacy raw margins plus scale-invariant unit-L2 margins."""

    raw = reference_same_schema_margins(
        latents,
        schema_ids,
        action_keys,
        reference_mask,
        group_ids,
    )
    unit_latents = F.normalize(
        latents.detach().to(dtype=torch.float64, device="cpu"),
        p=2,
        dim=1,
        eps=torch.finfo(torch.float64).tiny,
    )
    raw["unit_l2"] = reference_same_schema_margins(
        unit_latents,
        schema_ids,
        action_keys,
        reference_mask,
        group_ids,
    )
    return raw


def _without_details(value: Any) -> Any:
    """Return a recursive copy with diagnostic detail arrays removed."""

    if isinstance(value, dict):
        return {key: _without_details(item) for key, item in value.items() if key != "details"}
    if isinstance(value, list):
        return [_without_details(item) for item in value]
    return value


def _candidate_indices(
    all_actions: tuple[Any, ...],
    true_index: int,
    true_schema: str,
    *,
    same_schema_only: bool,
    max_candidates: int | None,
    seed: int,
) -> list[int]:
    indices = [idx for idx, action in enumerate(all_actions) if not same_schema_only or action.name == true_schema]
    if max_candidates is None or len(indices) <= max_candidates:
        return indices
    if max_candidates <= 0:
        raise ValueError("--max-candidates-per-state must be positive when provided")
    rng = random.Random(seed)
    pool = [idx for idx in indices if idx != true_index]
    sampled = rng.sample(pool, k=min(len(pool), max_candidates - 1)) if max_candidates > 1 else []
    sampled.append(true_index)
    return sorted(sampled)


def _encode_candidates(
    bundle: Any,
    space: ActionDecodingSpace,
    candidates: tuple[Any, ...],
    latent_state: JEPALatentState,
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        tensors = space.action_tensors_for_ground_actions(chunk, device=device)
        repeated = _repeat_action_latent_state(latent_state, len(chunk), device=device)
        chunks.append(bundle.jepa.action_encoder(tensors, repeated).detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), dtype=torch.float32)


def _repeat_action_latent_state(
    latent_state: JEPALatentState,
    repeats: int,
    *,
    device: torch.device,
) -> JEPALatentState:
    return JEPALatentState(
        graph_latent=latent_state.graph_latent.to(device).expand(repeats, -1).contiguous(),
        object_latents=latent_state.object_latents.to(device).repeat(repeats, 1),
        object_ids=latent_state.object_ids.to(device).repeat(repeats),
        object_batch=torch.arange(repeats, device=device).repeat_interleave(latent_state.object_latents.size(0)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
