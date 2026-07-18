"""Inspect the first latent action produced by the production planner path."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from acs_jepa import LatentGMMMPPIPlanner, LatentMPPIPlanner
from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa.graph import build_state_graph, parse_domain_problem, tensorize_goal_atoms
from omegaconf import OmegaConf

from acs_jepa_cli.cli import (
    _build_action_decoder,
    _build_gmm_planning_config,
    _build_goal_energy,
    _build_planning_config,
    _build_simulator_engine,
    _resolve_device,
    _to_device,
)
from acs_jepa_cli.config import load_config
from acs_jepa_cli.data import load_corpus
from acs_jepa_cli.modeling import build_model_bundle, vocab_sizes_from_dict

from action_diag_common import action_key, action_payload, applicable_keys, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--domain", required=True, type=Path)
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", nargs="+", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_config = OmegaConf.create(checkpoint["config"])
    config = OmegaConf.merge(load_config(None), checkpoint_config)
    if args.config:
        config = OmegaConf.merge(config, load_config(args.config))

    corpus = load_corpus([args.dataset], strict=True)
    vocab_sizes = vocab_sizes_from_dict(checkpoint["vocab_sizes"])
    bundle = build_model_bundle(corpus.parsed_problems, config, device=device, vocab_sizes=vocab_sizes)
    bundle.jepa.load_state_dict(checkpoint["model_state_dict"])
    bundle.jepa.eval()
    if bundle.goal_head is not None:
        bundle.goal_head.load_state_dict(checkpoint["goal_head_state_dict"])
        bundle.goal_head.eval()

    parsed = parse_domain_problem(args.domain, args.problem)
    goal_tensors = tensorize_goal_atoms(
        parsed,
        parsed.goal_atoms,
        max_atoms=None if config.data.max_goal_atoms is None else int(config.data.max_goal_atoms),
        max_arity=vocab_sizes.max_predicate_arity,
    )
    goal_tensors = _to_device(goal_tensors, device)
    goal_energy = _build_goal_energy(bundle.goal_head, str(config.model.goal_head.kind))
    action_space = ActionDecodingSpace.from_parsed_problem(parsed)
    planning_kind = str(getattr(config.planning, "kind", "continuous_mppi"))
    if planning_kind == "gmm_mppi":
        planning_config = _build_gmm_planning_config(config, device=device, seed=args.seed)
        planner = LatentGMMMPPIPlanner(
            graph_encoder=bundle.jepa.graph_encoder,
            state_encoder=bundle.jepa.state_encoder,
            action_encoder=bundle.jepa.action_encoder,
            predictor=bundle.jepa.predictor,
            goal_energy=goal_energy,
            action_space=action_space,
            config=planning_config,
        )
    elif planning_kind == "continuous_mppi":
        planning_config = _build_planning_config(config, device=device, seed=args.seed)
        planner = LatentMPPIPlanner(
            graph_encoder=bundle.jepa.graph_encoder,
            state_encoder=bundle.jepa.state_encoder,
            predictor=bundle.jepa.predictor,
            goal_energy=goal_energy,
            config=planning_config,
        )
    else:
        raise ValueError(f"diagnose_planner_first_latent supports continuous_mppi/gmm_mppi, got {planning_kind}")

    state_graph = _to_device(build_state_graph(parsed, parsed.initial_atoms, include_static=True), device)
    with torch.no_grad():
        plan = planner.plan(state_graph, goal_tensors)
        first_latent = plan.action_latents[0].to(device)
        latent_state = planner.encode_graph(state_graph)[1]
        decoder = _build_action_decoder(parsed, bundle.jepa.action_encoder, config, seed=args.seed)
        decoded = decoder.decode(first_latent, latent_state)
        engine = _build_simulator_engine(args.domain, args.problem)
        applicable = action_key(decoded) in applicable_keys(engine)
        top = _rank_candidates(
            bundle,
            action_space,
            latent_state,
            first_latent,
            applicable_keys(engine),
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            device=device,
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "problem": str(args.problem),
        "planning_kind": planning_kind,
        "decoded_first_action": action_payload(decoded),
        "decoded_first_action_applicable": applicable,
        "optimizer_best_score": getattr(plan.optimizer_result, "best_score", None),
        "optimizer_iterations": getattr(plan.optimizer_result, "iterations", None),
        "runtime_seconds": time.perf_counter() - started,
        "top_decoded_neighbors": top,
    }
    write_json(args.output / "summary.json", summary)
    print(summary)
    return 0


def _rank_candidates(
    bundle: Any,
    space: ActionDecodingSpace,
    latent_state: Any,
    target: torch.Tensor,
    applicable: set[tuple[str, tuple[str, ...]]],
    *,
    top_k: int,
    chunk_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    actions = space.enumerate_ground_actions()
    kept: list[tuple[float, int]] = []
    for start in range(0, len(actions), chunk_size):
        chunk = actions[start : start + chunk_size]
        tensors = space.action_tensors_for_ground_actions(chunk, device=device)
        repeated = _repeat_action_latent_state(latent_state, len(chunk), device=device)
        latents = bundle.jepa.action_encoder(tensors, repeated)
        scores = -((latents - target.unsqueeze(0)) ** 2).sum(dim=-1).detach().cpu()
        values, indices = torch.topk(scores, k=min(top_k, scores.numel()))
        kept.extend((float(score.item()), start + int(idx.item())) for score, idx in zip(values, indices, strict=True))
        kept = sorted(kept, key=lambda item: item[0], reverse=True)[:top_k]
    return [
        {
            "rank": rank,
            "score": score,
            "action": action_payload(actions[index]),
            "applicable": action_key(actions[index]) in applicable,
        }
        for rank, (score, index) in enumerate(kept, start=1)
    ]


def _repeat_action_latent_state(latent_state: Any, repeats: int, *, device: torch.device) -> Any:
    from acs_jepa.architectures import JEPALatentState

    return JEPALatentState(
        graph_latent=latent_state.graph_latent.to(device).expand(repeats, -1).contiguous(),
        object_latents=latent_state.object_latents.to(device).repeat(repeats, 1),
        object_ids=latent_state.object_ids.to(device).repeat(repeats),
        object_batch=torch.arange(repeats, device=device).repeat_interleave(latent_state.object_latents.size(0)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
