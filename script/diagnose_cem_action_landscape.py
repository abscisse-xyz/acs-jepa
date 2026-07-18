"""Trace categorical CEM behavior when decoding true action latents."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from acs_jepa.architectures import ActionSamplingFamily
from acs_jepa.mpc import IterationState, cross_entropy_optimize

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
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--max-transitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config, corpus, bundle, device = load_checkpoint_bundle(args.dataset_dir, args.checkpoint, device_name=args.device)
    selected = select_split(corpus, config, args.split, seed=args.seed)
    details: list[dict[str, Any]] = []
    exact = 0
    applicable_count = 0
    same_schema = 0
    total = 0

    with torch.no_grad():
        for trajectory, record, step_idx, true_action in iter_transitions(selected, max_transitions=args.max_transitions):
            parsed = selected.parsed_problems[trajectory.problem_index]
            latent_state = encode_state(bundle, parsed, trajectory.states[step_idx], device=device)
            from acs_jepa.architectures import ActionDecodingSpace

            space = ActionDecodingSpace.from_parsed_problem(parsed)
            true_tensors = space.action_tensors_for_ground_actions([true_action], device=device)
            target = bundle.jepa.action_encoder(true_tensors, latent_state)
            family = ActionSamplingFamily(space, device=device)
            trace: list[dict[str, Any]] = []

            def callback(state: IterationState) -> None:
                trace.append(
                    {
                        "iteration": state.iteration,
                        "num_samples": state.num_samples,
                        "threshold": state.threshold,
                        "elite_mean": state.elite_mean,
                        "population_best": state.population_best,
                        "best_ever": state.best_ever,
                        "action_entropy": _entropy(state.probabilities[0]),
                        "argument_entropies": [_entropy(item) for item in state.probabilities[1:]],
                    }
                )

            def score_fn(samples: torch.Tensor) -> torch.Tensor:
                action_tensors = space.samples_to_action_tensors(samples, device=device)
                repeated = _repeat_action_latent_state(latent_state, samples.size(0), device=device)
                latents = bundle.jepa.action_encoder(action_tensors, repeated)
                if args.metric == "l2":
                    return -((latents - target) ** 2).sum(dim=-1)
                cand = torch.nn.functional.normalize(latents, dim=-1)
                tgt = torch.nn.functional.normalize(target, dim=-1)
                return (cand * tgt).sum(dim=-1)

            result = cross_entropy_optimize(
                domain_sizes=family.domain_sizes,
                score_fn=score_fn,
                num_samples=args.num_samples,
                elite_frac=args.elite_frac,
                max_iters=args.max_iters,
                sampling_family=family,
                callback=callback,
                seed=args.seed,
                device=device,
            )
            best_sample = torch.tensor(result.best_x, dtype=torch.long, device=device)
            decoded = space.sample_to_ground_action(best_sample)
            engine = replay_to_engine(args.dataset_dir, record.problem_name, trajectory.actions[:step_idx])
            applicable = action_key(decoded) in applicable_keys(engine)
            total += 1
            exact += int(decoded == true_action)
            applicable_count += int(applicable)
            same_schema += int(decoded.name == true_action.name)
            details.append(
                {
                    "problem": record.problem_name,
                    "step_index": step_idx,
                    "true_action": action_payload(true_action),
                    "decoded_action": action_payload(decoded),
                    "same_schema": decoded.name == true_action.name,
                    "exact_match": decoded == true_action,
                    "applicable": applicable,
                    "best_score": result.best_score,
                    "mode_score": result.mode_score,
                    "iterations": result.iterations,
                    "stop_reason": result.stop_reason,
                    "trace": trace,
                }
            )

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset_dir),
        "split": args.split,
        "num_samples": args.num_samples,
        "max_iters": args.max_iters,
        "metrics": {
            "transitions": total,
            "same_schema_rate": 0.0 if total == 0 else same_schema / total,
            "exact_match_rate_diagnostic_only": 0.0 if total == 0 else exact / total,
            "applicable_rate": 0.0 if total == 0 else applicable_count / total,
            "runtime_seconds": time.perf_counter() - started,
        },
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "details.json", details)
    print(summary)
    return 0


def _repeat_action_latent_state(latent_state: Any, repeats: int, *, device: torch.device) -> Any:
    from acs_jepa.architectures import JEPALatentState

    return JEPALatentState(
        graph_latent=latent_state.graph_latent.to(device).expand(repeats, -1).contiguous(),
        object_latents=latent_state.object_latents.to(device).repeat(repeats, 1),
        object_ids=latent_state.object_ids.to(device).repeat(repeats),
        object_batch=torch.arange(repeats, device=device).repeat_interleave(latent_state.object_latents.size(0)),
    )


def _entropy(probs: torch.Tensor) -> float:
    values = probs.detach().float().cpu()
    values = values[values > 0]
    if values.numel() == 0:
        return 0.0
    return float(-(values * values.log()).sum().item() / math.log(2.0))


if __name__ == "__main__":
    raise SystemExit(main())
