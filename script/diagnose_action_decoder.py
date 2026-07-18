"""Diagnose ground-truth action latent decoding for a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from acs_jepa.architectures import ActionDecoder
from acs_jepa.graph import build_state_graph
from omegaconf import OmegaConf

from acs_jepa_cli.cli import _build_simulator_engine, _resolve_device, _to_device
from acs_jepa_cli.config import load_config, save_resolved_config
from acs_jepa_cli.data import load_corpus, split_corpus
from acs_jepa_cli.modeling import build_model_bundle, vocab_sizes_from_dict
from acs_jepa_cli.tracking import configure_mlflow, config_hash, log_config_params, log_metrics, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--method", default="exact", choices=("exact", "cem", "mppi"))
    parser.add_argument("--metric", default="l2", choices=("l2", "cosine"))
    parser.add_argument("--decoder-num-samples", type=int, default=None)
    parser.add_argument("--decoder-max-iters", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = OmegaConf.merge(load_config(None), OmegaConf.create(checkpoint["config"]))
    config.tracking.run_name = f"action_decode_{args.split}_{args.method}_seed{args.seed}"
    config.tracking.tags.stage = "action_decoder_diagnostic"
    config.tracking.tags.hypothesis = "Ground-truth action latents should decode to exact and applicable actions."
    save_resolved_config(config, args.output / "config.yaml")

    split_seed = args.seed if config.data.split_seed is None else int(config.data.split_seed)
    corpus = load_corpus([args.dataset_dir], strict=True)
    if args.split == "all":
        selected = corpus
    else:
        splits = split_corpus(
            corpus,
            val_fraction=float(config.data.val_fraction),
            test_fraction=float(config.data.test_fraction),
            seed=split_seed,
        )
        selected = getattr(splits, args.split)

    vocab_sizes = vocab_sizes_from_dict(checkpoint["vocab_sizes"])
    bundle = build_model_bundle(corpus.parsed_problems, config, device=device, vocab_sizes=vocab_sizes)
    bundle.jepa.load_state_dict(checkpoint["model_state_dict"])
    bundle.jepa.eval()

    details: list[dict[str, Any]] = []
    exact_matches = 0
    applicable_decodes = 0
    total = 0

    with torch.no_grad():
        for trajectory, record in zip(selected.trajectories, selected.records):
            parsed = selected.parsed_problems[trajectory.problem_index]
            problem_path = _problem_path(args.dataset_dir, record.problem_name)
            decoder = ActionDecoder(
                parsed_problem=parsed,
                action_encoder=bundle.jepa.action_encoder,
                method=args.method,
                metric=args.metric,
                num_samples=None
                if args.method == "exact"
                else int(args.decoder_num_samples or config.planning.action_decoder.num_samples),
                elite_frac=float(config.planning.action_decoder.elite_frac),
                max_iters=int(args.decoder_max_iters or config.planning.action_decoder.max_iters),
                smoothing=float(config.planning.action_decoder.smoothing),
                dirichlet_prior=float(config.planning.action_decoder.dirichlet_prior),
                temperature=float(config.planning.action_decoder.temperature),
                decode_chunk_size=(
                    None
                    if getattr(config.planning.action_decoder, "decode_chunk_size", None) is None
                    else int(config.planning.action_decoder.decode_chunk_size)
                ),
                seed=args.seed,
            )
            for step_idx, true_action in enumerate(trajectory.actions):
                if args.max_transitions is not None and total >= args.max_transitions:
                    break
                state_graph = _to_device(build_state_graph(parsed, trajectory.states[step_idx], include_static=True), device)
                latent_state = bundle.jepa.encode(state_graph)
                action_tensors = _to_device(
                    decoder.space.action_tensors_for_ground_actions([true_action]),
                    device,
                )
                target_latent = bundle.jepa.action_encoder(action_tensors, latent_state)
                decoded_action = decoder.decode(target_latent.squeeze(0), latent_state)
                exact = decoded_action == true_action
                applicable = _is_applicable(
                    domain_path=args.dataset_dir / "problem" / "domain.pddl",
                    problem_path=problem_path,
                    replay_actions=trajectory.actions[:step_idx],
                    decoded_action=decoded_action,
                )
                exact_matches += int(exact)
                applicable_decodes += int(applicable)
                total += 1
                details.append(
                    {
                        "problem": record.problem_name,
                        "step_index": step_idx,
                        "true_action": _action_payload(true_action),
                        "decoded_action": _action_payload(decoded_action),
                        "exact_match": exact,
                        "applicable": applicable,
                    }
                )
            if args.max_transitions is not None and total >= args.max_transitions:
                break

    metrics = {
        "transitions": float(total),
        "exact_match_rate": 0.0 if total == 0 else exact_matches / total,
        "applicable_rate": 0.0 if total == 0 else applicable_decodes / total,
        "runtime_seconds": time.perf_counter() - started,
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset_dir),
        "split": args.split,
        "method": args.method,
        "metric": args.metric,
        "decoder_num_samples": args.decoder_num_samples,
        "decoder_max_iters": args.decoder_max_iters,
        "split_seed": split_seed,
        "metrics": metrics,
        "num_trajectories": len(selected.trajectories),
        "num_records": len(selected.records),
    }
    _write_json(args.output / "summary.json", summary)
    _write_json(args.output / "details.json", details)

    configure_mlflow(config)
    with start_run(config, fallback_name="action_decode"):
        log_config_params(
            config,
            extra={
                "command": "diagnose_action_decoder",
                "checkpoint": str(args.checkpoint),
                "dataset": str(args.dataset_dir),
                "split": args.split,
                "method": args.method,
                "metric": args.metric,
                "decoder_num_samples": args.decoder_num_samples,
                "decoder_max_iters": args.decoder_max_iters,
                "seed": args.seed,
                "device": device.type,
                "config_hash": config_hash(config),
            },
        )
        log_metrics("action_decode", metrics)
        mlflow.log_artifact(str(args.output / "config.yaml"))
        mlflow.log_artifact(str(args.output / "summary.json"), artifact_path="action_decode")
        mlflow.log_artifact(str(args.output / "details.json"), artifact_path="action_decode")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _is_applicable(*, domain_path: Path, problem_path: Path, replay_actions: Any, decoded_action: Any) -> bool:
    engine = _build_simulator_engine(domain_path, problem_path)
    for action in replay_actions:
        engine.apply_action(action.name, action.arguments, finish=True)
    try:
        engine.apply_action(decoded_action.name, decoded_action.arguments, finish=True)
    except ValueError:
        return False
    return True


def _problem_path(dataset_dir: Path, problem_name: str) -> Path:
    problem_dir = dataset_dir / "problem"
    direct = problem_dir / problem_name
    if direct.exists():
        return direct
    with_suffix = problem_dir / f"{problem_name}.pddl"
    if with_suffix.exists():
        return with_suffix
    raise FileNotFoundError(f"No PDDL file found for problem {problem_name!r} in {problem_dir}")


def _action_payload(action: Any) -> dict[str, Any]:
    return {"name": action.name, "arguments": list(action.arguments)}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
