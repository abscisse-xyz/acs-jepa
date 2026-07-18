"""Diagnose action-latent margins around teacher-forced true actions."""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from acs_jepa.architectures import ActionDecodingSpace

from action_diag_common import (
    action_key,
    action_payload,
    applicable_keys,
    encode_state,
    iter_transitions,
    load_checkpoint_bundle,
    replay_to_engine,
    select_split,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--metric", default="l2", choices=("l2", "cosine"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--same-schema-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config, corpus, bundle, device = load_checkpoint_bundle(args.dataset_dir, args.checkpoint, device_name=args.device)
    selected = select_split(corpus, config, args.split, seed=args.seed)
    space_cache: dict[int, tuple[ActionDecodingSpace, tuple[Any, ...], dict[tuple[str, tuple[str, ...]], int]]] = {}

    details: list[dict[str, Any]] = []
    totals = Counter()
    nearest_wrong_distances: list[float] = []
    nearest_applicable_ranks: list[int] = []

    with torch.no_grad():
        for trajectory, record, step_idx, true_action in iter_transitions(selected, max_transitions=args.max_transitions):
            parsed = selected.parsed_problems[trajectory.problem_index]
            if trajectory.problem_index not in space_cache:
                space = ActionDecodingSpace.from_parsed_problem(parsed)
                actions = space.enumerate_ground_actions()
                index = {action_key(action): idx for idx, action in enumerate(actions)}
                space_cache[trajectory.problem_index] = (space, actions, index)
            space, all_actions, action_index = space_cache[trajectory.problem_index]
            if args.same_schema_only:
                candidate_indices = [idx for idx, action in enumerate(all_actions) if action.name == true_action.name]
            else:
                candidate_indices = list(range(len(all_actions)))
            candidates = tuple(all_actions[idx] for idx in candidate_indices)
            local_true_index = candidate_indices.index(action_index[action_key(true_action)])

            latent_state = encode_state(bundle, parsed, trajectory.states[step_idx], device=device)
            true_tensors = space.action_tensors_for_ground_actions([true_action], device=device)
            target = bundle.jepa.action_encoder(true_tensors, latent_state)
            scores = _score_candidates(
                bundle,
                space,
                candidates,
                latent_state,
                target,
                metric=args.metric,
                chunk_size=args.chunk_size,
                device=device,
            )
            order = torch.argsort(scores, descending=True).cpu().tolist()
            ranks = {idx: rank + 1 for rank, idx in enumerate(order)}
            true_rank = ranks[local_true_index]
            true_score = float(scores[local_true_index].detach().cpu().item())

            engine = replay_to_engine(args.dataset_dir, record.problem_name, trajectory.actions[:step_idx])
            applicable = applicable_keys(engine)
            applicable_local = [idx for idx, action in enumerate(candidates) if action_key(action) in applicable]
            nearest_applicable_rank = min((ranks[idx] for idx in applicable_local), default=None)
            if nearest_applicable_rank is not None:
                nearest_applicable_ranks.append(int(nearest_applicable_rank))

            wrong_order = [idx for idx in order if idx != local_true_index]
            nearest_wrong_idx = wrong_order[0] if wrong_order else None
            nearest_wrong = candidates[nearest_wrong_idx] if nearest_wrong_idx is not None else None
            nearest_wrong_score = None if nearest_wrong_idx is None else float(scores[nearest_wrong_idx].detach().cpu().item())
            nearest_wrong_distance = _score_to_distance(true_score, nearest_wrong_score, args.metric)
            if nearest_wrong_distance is not None:
                nearest_wrong_distances.append(nearest_wrong_distance)

            top_items = []
            for idx in order[: args.top_k]:
                action = candidates[idx]
                score = float(scores[idx].detach().cpu().item())
                top_items.append(
                    {
                        "rank": ranks[idx],
                        "action": action_payload(action),
                        "score": score,
                        "distance_from_target": _score_to_distance(true_score, score, args.metric),
                        "same_schema": action.name == true_action.name,
                        "applicable": action_key(action) in applicable,
                        "is_true_action": idx == local_true_index,
                        "argument_diff_count": _argument_diff_count(true_action, action),
                    }
                )

            totals["transitions"] += 1
            totals["true_rank_1"] += int(true_rank == 1)
            totals["nearest_wrong_same_schema"] += int(nearest_wrong is not None and nearest_wrong.name == true_action.name)
            totals["nearest_wrong_applicable"] += int(nearest_wrong is not None and action_key(nearest_wrong) in applicable)
            totals["has_applicable_candidate"] += int(bool(applicable_local))
            details.append(
                {
                    "problem": record.problem_name,
                    "step_index": step_idx,
                    "true_action": action_payload(true_action),
                    "candidate_count": len(candidates),
                    "applicable_candidate_count": len(applicable_local),
                    "true_rank": true_rank,
                    "nearest_applicable_rank": nearest_applicable_rank,
                    "nearest_wrong_action": None if nearest_wrong is None else action_payload(nearest_wrong),
                    "nearest_wrong_distance": nearest_wrong_distance,
                    "nearest_wrong_same_schema": nearest_wrong is not None and nearest_wrong.name == true_action.name,
                    "nearest_wrong_applicable": nearest_wrong is not None and action_key(nearest_wrong) in applicable,
                    "top": top_items,
                }
            )

    transition_count = totals["transitions"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset_dir),
        "split": args.split,
        "metric": args.metric,
        "same_schema_only": args.same_schema_only,
        "max_transitions": args.max_transitions,
        "metrics": {
            "transitions": transition_count,
            "true_rank_1_rate": _rate(totals["true_rank_1"], transition_count),
            "nearest_wrong_same_schema_rate": _rate(totals["nearest_wrong_same_schema"], transition_count),
            "nearest_wrong_applicable_rate": _rate(totals["nearest_wrong_applicable"], transition_count),
            "has_applicable_candidate_rate": _rate(totals["has_applicable_candidate"], transition_count),
            "nearest_wrong_distance_median": _median(nearest_wrong_distances),
            "nearest_wrong_distance_min": min(nearest_wrong_distances) if nearest_wrong_distances else None,
            "nearest_applicable_rank_median": _median([float(v) for v in nearest_applicable_ranks]),
            "runtime_seconds": time.perf_counter() - started,
        },
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "details.json", details)
    print(summary)
    return 0


def _score_candidates(
    bundle: Any,
    space: ActionDecodingSpace,
    candidates: tuple[Any, ...],
    latent_state: Any,
    target: torch.Tensor,
    *,
    metric: str,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    scores = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        tensors = space.action_tensors_for_ground_actions(chunk, device=device)
        repeated = _repeat_action_latent_state(latent_state, len(chunk), device=device)
        latents = bundle.jepa.action_encoder(tensors, repeated)
        if metric == "l2":
            scores.append(-((latents - target) ** 2).sum(dim=-1).detach().cpu())
        else:
            cand = torch.nn.functional.normalize(latents, dim=-1)
            tgt = torch.nn.functional.normalize(target, dim=-1)
            scores.append((cand * tgt).sum(dim=-1).detach().cpu())
    return torch.cat(scores, dim=0)


def _repeat_action_latent_state(latent_state: Any, repeats: int, *, device: torch.device) -> Any:
    from acs_jepa.architectures import JEPALatentState

    return JEPALatentState(
        graph_latent=latent_state.graph_latent.to(device).expand(repeats, -1).contiguous(),
        object_latents=latent_state.object_latents.to(device).repeat(repeats, 1),
        object_ids=latent_state.object_ids.to(device).repeat(repeats),
        object_batch=torch.arange(repeats, device=device).repeat_interleave(latent_state.object_latents.size(0)),
    )


def _score_to_distance(true_score: float, score: float | None, metric: str) -> float | None:
    if score is None:
        return None
    if metric == "l2":
        return math.sqrt(max(0.0, true_score - score))
    return true_score - score


def _argument_diff_count(a: Any, b: Any) -> int:
    if a.name != b.name:
        return -1
    return sum(left != right for left, right in zip(a.arguments, b.arguments, strict=True))


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


if __name__ == "__main__":
    raise SystemExit(main())
