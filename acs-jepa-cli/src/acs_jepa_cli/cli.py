"""Argparse entrypoint for ACS-JEPA CLI commands."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from acs_jepa import (
    DistributionalGoalEnergy,
    GroundedPlannerAgent,
    LatentGMMMPPIConfig,
    LatentGMMMPPIPlanner,
    LatentMPPIConfig,
    LatentMPPIPlanner,
    PlannerAgent,
    SampleSetGoalEnergy,
    StructuredCEPlanner,
    StructuredCEPlannerConfig,
)
from acs_jepa.architectures import ActionDecoder, ActionDecodingSpace, JEPALatentState
from acs_jepa.graph import parse_domain_problem, tensorize_goal_atoms
from omegaconf import OmegaConf
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from acs_jepa_cli.config import (
    config_paths_text,
    config_to_container,
    load_config,
    load_config_overrides,
    save_resolved_config,
    tuning_overlay_tags,
)
from acs_jepa_cli.data import corpus_summary, load_corpus, make_torch_dataset, split_corpus
from acs_jepa_cli.modeling import build_model_bundle, vocab_sizes_from_dict, vocab_sizes_to_dict
from acs_jepa_cli.tracking import (
    config_hash,
    configure_mlflow,
    log_config_params,
    log_metrics,
    start_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acs-jepa", description="Train, evaluate, and plan with ACS-JEPA models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="Inspect simulator datasets.")
    inspect_parser.add_argument("dataset_dirs", nargs="+", type=Path)
    inspect_parser.set_defaults(func=cmd_inspect_data)

    train_parser = subparsers.add_parser("train", help="Train an ACS-JEPA model.")
    train_parser.add_argument("dataset_dirs", nargs="+", type=Path)
    train_parser.add_argument("--output", required=True, type=Path)
    train_parser.add_argument("--config", required=True, nargs="+", type=Path)
    train_parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    train_parser.add_argument("--seed", default=0, type=int)
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate an ACS-JEPA checkpoint.")
    eval_parser.add_argument("dataset_dirs", nargs="+", type=Path)
    eval_parser.add_argument("--checkpoint", required=True, type=Path)
    eval_parser.add_argument("--output", required=True, type=Path)
    eval_parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    eval_parser.set_defaults(func=cmd_eval)

    plan_parser = subparsers.add_parser("plan", help="Run a latent MPPI planner from a checkpoint.")
    plan_parser.add_argument("--checkpoint", required=True, type=Path)
    plan_parser.add_argument("--domain", required=True, type=Path)
    plan_parser.add_argument("--problem", required=True, type=Path)
    plan_parser.add_argument("--output", required=True, type=Path)
    plan_parser.add_argument("--config", nargs="+", type=Path)
    plan_parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    plan_parser.add_argument("--seed", default=0, type=int)
    plan_parser.set_defaults(func=cmd_plan)
    return parser


def cmd_inspect_data(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.dataset_dirs, strict=False)
    summary = corpus_summary(corpus)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if corpus.malformed_rows else 0


def cmd_train(args: argparse.Namespace) -> int:
    run_start = time.perf_counter()
    torch.manual_seed(args.seed)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    save_resolved_config(config, output_dir / "config.yaml")
    device = _resolve_device(args.device)

    corpus = load_corpus(args.dataset_dirs, strict=True)
    splits = split_corpus(
        corpus,
        val_fraction=float(config.data.val_fraction),
        test_fraction=float(config.data.test_fraction),
        seed=args.seed if config.data.split_seed is None else int(config.data.split_seed),
    )
    if not splits.train.trajectories:
        raise ValueError("Training split is empty")

    train_dataset = make_torch_dataset(splits.train, config)
    val_dataset = make_torch_dataset(splits.val, config) if splits.val.trajectories else None
    if len(train_dataset) == 0:
        raise ValueError("Training split has no trajectory windows for the configured rollout_steps")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.data.batch_size),
        shuffle=True,
        num_workers=int(config.data.num_workers),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(config.data.batch_size),
            shuffle=False,
            num_workers=int(config.data.num_workers),
        )
        if val_dataset is not None
        else None
    )

    epochs = int(config.training.epochs)
    total_steps = len(train_loader) * epochs
    bundle = build_model_bundle(corpus.parsed_problems, config, device=device, total_steps=total_steps)
    configure_mlflow(config)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    artifact_dir = output_dir / "artifacts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = corpus_summary(corpus)
    _write_json(dataset_summary, artifact_dir / "dataset_summary.json")
    _write_json(splits.manifest, artifact_dir / "split_manifest.json")

    best_eval = float("inf")
    global_step = 0
    with start_run(config, fallback_name="train", extra_tags=tuning_overlay_tags(args.config)):
        log_config_params(
            config,
            extra={
                "command": "train",
                "seed": args.seed,
                "device": device.type,
                "config_files": config_paths_text(args.config),
                "config_hash": config_hash(config),
                **{f"vocab.{key}": value for key, value in vocab_sizes_to_dict(bundle.vocab_sizes).items()},
            },
        )
        if bool(config.tracking.log_artifacts):
            mlflow.log_artifact(str(output_dir / "config.yaml"))
            mlflow.log_artifact(str(artifact_dir / "dataset_summary.json"), artifact_path="data")
            mlflow.log_artifact(str(artifact_dir / "split_manifest.json"), artifact_path="data")

        eval_every_steps = int(config.training.eval_every_steps)
        checkpoint_every_steps = int(config.training.checkpoint_every_steps)
        for epoch in range(epochs):
            for batch in train_loader:
                global_step += 1
                output = bundle.trainer.train_step(_to_device(batch, device))
                if bundle.scheduler is not None:
                    bundle.scheduler.step()
                train_metrics = _step_metrics(output)
                if bundle.scheduler is not None:
                    train_metrics["optim/lr"] = _scheduler_lr(bundle.scheduler)
                log_metrics("train", train_metrics, step=global_step)
                _append_jsonl(metrics_dir / "train.jsonl", {"epoch": epoch, "step": global_step, **train_metrics})

                if val_loader is not None and eval_every_steps > 0 and global_step % eval_every_steps == 0:
                    eval_metrics = _evaluate(bundle.trainer, val_loader, device)
                    log_metrics("eval", eval_metrics, step=global_step)
                    _append_jsonl(metrics_dir / "eval.jsonl", {"epoch": epoch, "step": global_step, **eval_metrics})
                    best_eval = _maybe_save_best(
                        eval_metrics["total_loss"],
                        best_eval,
                        checkpoint_dir / "best.pt",
                        bundle,
                        config,
                        epoch,
                        global_step,
                    )

                if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                    _save_checkpoint(checkpoint_dir / "latest.pt", bundle, config, epoch, global_step, best_eval)

            if val_loader is not None:
                eval_metrics = _evaluate(bundle.trainer, val_loader, device)
                log_metrics("eval", eval_metrics, step=global_step)
                _append_jsonl(metrics_dir / "eval.jsonl", {"epoch": epoch, "step": global_step, **eval_metrics})
                best_eval = _maybe_save_best(
                    eval_metrics["total_loss"],
                    best_eval,
                    checkpoint_dir / "best.pt",
                    bundle,
                    config,
                    epoch,
                    global_step,
                )

        latest = checkpoint_dir / "latest.pt"
        _save_checkpoint(latest, bundle, config, epochs - 1, global_step, best_eval)
        runtime_metrics = _runtime_metrics(run_start, steps=global_step, examples=len(train_dataset) * epochs)
        log_metrics("runtime", runtime_metrics)
        _write_json(runtime_metrics, artifact_dir / "runtime_summary.json")
        if bool(config.tracking.log_artifacts):
            mlflow.log_artifact(str(latest), artifact_path="checkpoints")
            best = checkpoint_dir / "best.pt"
            if best.exists():
                mlflow.log_artifact(str(best), artifact_path="checkpoints")
            for path in metrics_dir.glob("*.jsonl"):
                mlflow.log_artifact(str(path), artifact_path="metrics")
            mlflow.log_artifact(str(artifact_dir / "runtime_summary.json"), artifact_path="runtime")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    run_start = time.perf_counter()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = OmegaConf.merge(load_config(None), OmegaConf.create(checkpoint["config"]))
    save_resolved_config(config, output_dir / "config.yaml")
    corpus = load_corpus(args.dataset_dirs, strict=True)
    dataset = make_torch_dataset(corpus, config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        shuffle=False,
        num_workers=int(config.data.num_workers),
    )
    vocab_sizes = vocab_sizes_from_dict(checkpoint["vocab_sizes"])
    bundle = build_model_bundle(corpus.parsed_problems, config, device=device, vocab_sizes=vocab_sizes)
    _load_checkpoint_state(bundle, checkpoint)

    configure_mlflow(config)
    metrics = _evaluate(bundle.trainer, loader, device)
    runtime_metrics = _runtime_metrics(run_start, examples=len(dataset))
    summary = {
        "checkpoint": str(args.checkpoint),
        "datasets": [str(path) for path in args.dataset_dirs],
        "metrics": metrics,
        "runtime": runtime_metrics,
        "dataset_summary": corpus_summary(corpus),
    }
    _write_json(summary, output_dir / "eval_summary.json")

    with start_run(config, fallback_name="eval"):
        log_config_params(
            config,
            extra={
                "command": "eval",
                "checkpoint": str(args.checkpoint),
                "config_hash": config_hash(config),
            },
        )
        log_metrics("eval", metrics)
        log_metrics("runtime", runtime_metrics)
        if bool(config.tracking.log_artifacts):
            mlflow.log_artifact(str(output_dir / "config.yaml"))
            mlflow.log_artifact(str(output_dir / "eval_summary.json"), artifact_path="eval")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    run_start = time.perf_counter()
    torch.manual_seed(args.seed)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = OmegaConf.merge(load_config(None), OmegaConf.create(checkpoint["config"]))
    if args.config is not None:
        config = OmegaConf.merge(config, load_config_overrides(args.config))
    save_resolved_config(config, output_dir / "config.yaml")

    parsed_problem = parse_domain_problem(args.domain, args.problem)
    vocab_sizes = vocab_sizes_from_dict(checkpoint["vocab_sizes"])
    bundle = build_model_bundle((parsed_problem,), config, device=device, vocab_sizes=vocab_sizes)
    _load_checkpoint_state(bundle, checkpoint)
    bundle.jepa.eval()
    if bundle.goal_head is not None:
        bundle.goal_head.eval()
    if bundle.applicability_head is not None:
        bundle.applicability_head.eval()

    goal_tensors = tensorize_goal_atoms(
        parsed_problem,
        parsed_problem.goal_atoms,
        max_atoms=None if config.data.max_goal_atoms is None else int(config.data.max_goal_atoms),
        max_arity=vocab_sizes.max_predicate_arity,
    )
    goal_tensors = _to_device(goal_tensors, device)
    goal_energy = _build_goal_energy(bundle.goal_head, str(config.model.goal_head.kind))
    planning_kind = str(getattr(config.planning, "kind", "continuous_mppi"))
    action_space = ActionDecodingSpace.from_parsed_problem(parsed_problem)
    if planning_kind == "continuous_mppi":
        planning_config = _build_planning_config(config, device=device, seed=args.seed)
        planner = LatentMPPIPlanner(
            graph_encoder=bundle.jepa.graph_encoder,
            state_encoder=bundle.jepa.state_encoder,
            predictor=bundle.jepa.predictor,
            goal_energy=goal_energy,
            config=planning_config,
        )
        action_decoder = _build_action_decoder(parsed_problem, bundle.jepa.action_encoder, config, seed=args.seed)
        agent = PlannerAgent(
            planner=planner,
            action_decoder=action_decoder,
            parsed_problem=parsed_problem,
            include_static=True,
        )
    elif planning_kind == "gmm_mppi":
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
        action_decoder = _build_action_decoder(parsed_problem, bundle.jepa.action_encoder, config, seed=args.seed)
        agent = PlannerAgent(
            planner=planner,
            action_decoder=action_decoder,
            parsed_problem=parsed_problem,
            include_static=True,
        )
    elif planning_kind == "structured_ce":
        planning_config = _build_structured_ce_planning_config(config, device=device, seed=args.seed)
        planner = StructuredCEPlanner(
            graph_encoder=bundle.jepa.graph_encoder,
            state_encoder=bundle.jepa.state_encoder,
            action_encoder=bundle.jepa.action_encoder,
            predictor=bundle.jepa.predictor,
            goal_energy=goal_energy,
            action_space=action_space,
            config=planning_config,
        )
        agent = GroundedPlannerAgent(
            planner=planner,
            parsed_problem=parsed_problem,
            include_static=True,
        )
    else:
        raise ValueError(f"Unknown planning.kind: {planning_kind}")
    engine = _build_simulator_engine(args.domain, args.problem)
    with torch.no_grad():
        result = agent.run(engine, goal_tensors)

    runtime_metrics = _runtime_metrics(run_start)
    summary = {
        "checkpoint": str(args.checkpoint),
        "domain": str(args.domain),
        "problem": str(args.problem),
        "success": bool(result.success),
        "failure_reason": result.failure_reason,
        "total_actions": result.total_actions,
        "attempts": result.attempts,
        "applied_actions": [
            {"name": action.name, "arguments": list(action.arguments)} for action in result.applied_actions
        ],
        "runtime": runtime_metrics,
        "planning": {
            "kind": planning_kind,
            "horizon": planning_config.horizon,
            "num_samples": planning_config.num_samples,
            "max_iters": planning_config.max_iters,
            "apply_steps": planning_config.apply_steps,
            "max_total_actions": planning_config.max_total_actions,
            "max_decode_attempts": planning_config.max_decode_attempts,
            "action_decoder_method": str(config.planning.action_decoder.method),
        },
    }
    _write_json(summary, output_dir / "plan_summary.json")
    configure_mlflow(config)
    with start_run(config, fallback_name="plan", extra_tags=tuning_overlay_tags(args.config)):
        log_config_params(
            config,
            extra={
                "command": "plan",
                "checkpoint": str(args.checkpoint),
                "override_config": config_paths_text(args.config),
                "domain": str(args.domain),
                "problem": str(args.problem),
                "seed": args.seed,
                "device": device.type,
                "config_hash": config_hash(config),
            },
        )
        log_metrics(
            "plan",
            {
                "success": 1.0 if result.success else 0.0,
                "total_actions": float(result.total_actions),
                "attempts": float(result.attempts),
                "runtime_seconds": runtime_metrics["seconds"],
            },
        )
        if bool(config.tracking.log_artifacts):
            mlflow.log_artifact(str(output_dir / "config.yaml"))
            mlflow.log_artifact(str(output_dir / "plan_summary.json"), artifact_path="plan")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, (Data, Tensor)):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


class _PredicateGoalEnergy(nn.Module):
    """Energy adapter for positive conjunctive goals scored by PredicateEvaluator."""

    def __init__(self, evaluator: nn.Module) -> None:
        super().__init__()
        self.evaluator = evaluator

    def forward(self, goal_tensors: dict[str, Tensor], terminal_state: JEPALatentState) -> Tensor:
        atom_mask = goal_tensors["goal_atom_mask"].bool()
        weights = goal_tensors.get("goal_weight", atom_mask.float()).to(
            device=terminal_state.graph_latent.device,
            dtype=terminal_state.graph_latent.dtype,
        )
        energy = terminal_state.graph_latent.new_zeros((terminal_state.graph_latent.size(0),))
        for atom_idx in atom_mask.nonzero(as_tuple=False).view(-1).tolist():
            query = {
                "predicate_id": goal_tensors["goal_predicate_id"][atom_idx],
                "predicate_object_indices": goal_tensors["goal_object_indices"][atom_idx],
                "predicate_role_ids": goal_tensors["goal_role_ids"][atom_idx],
                "predicate_arg_mask": goal_tensors["goal_arg_mask"][atom_idx],
            }
            logits = self.evaluator(query, terminal_state)
            energy = energy + weights[atom_idx] * F.softplus(-logits)
        return energy


def _build_goal_energy(goal_head: nn.Module | None, kind: str) -> nn.Module:
    if goal_head is None or kind == "none":
        raise ValueError(
            "Planning requires model.goal_head.kind to be predicate, gaussian, gmm, or conditional_sampler"
        )
    if kind == "predicate":
        return _PredicateGoalEnergy(goal_head)
    if kind in {"gaussian", "gmm"}:
        return DistributionalGoalEnergy(goal_head)
    if kind == "conditional_sampler":
        return SampleSetGoalEnergy(goal_head)
    raise ValueError(f"Unknown goal_head kind for planning: {kind}")


def _build_planning_config(config: Any, *, device: torch.device, seed: int) -> LatentMPPIConfig:
    planning = config.planning
    return LatentMPPIConfig(
        horizon=int(planning.horizon),
        action_dim=int(config.model.action_dim),
        num_samples=int(planning.num_samples),
        elite_frac=float(planning.elite_frac),
        max_iters=int(planning.max_iters),
        temperature=float(planning.temperature),
        smoothing=float(planning.smoothing),
        noise_std=float(planning.noise_std),
        convergence_tol=float(planning.convergence_tol),
        quantile_window=int(planning.quantile_window),
        elitism=bool(planning.elitism),
        face_adaptive=bool(planning.face_adaptive),
        num_samples_max=None if planning.num_samples_max is None else int(planning.num_samples_max),
        face_stall_iters=int(planning.face_stall_iters),
        nan_policy=str(planning.nan_policy),
        apply_steps=int(planning.apply_steps),
        max_total_actions=int(planning.max_total_actions),
        constant_action_cost=float(planning.constant_action_cost),
        max_decode_attempts=int(planning.max_decode_attempts),
        invalid_action_penalty=float(planning.invalid_action_penalty),
        initial_std=float(planning.initial_std),
        device=device,
        seed=seed,
    )


def _build_gmm_planning_config(config: Any, *, device: torch.device, seed: int) -> LatentGMMMPPIConfig:
    planning = config.planning
    return LatentGMMMPPIConfig(
        horizon=int(planning.horizon),
        action_dim=int(config.model.action_dim),
        num_samples=int(planning.num_samples),
        elite_frac=float(planning.elite_frac),
        max_iters=int(planning.max_iters),
        temperature=float(planning.temperature),
        smoothing=float(planning.smoothing),
        noise_std=float(planning.noise_std),
        convergence_tol=float(planning.convergence_tol),
        quantile_window=int(planning.quantile_window),
        elitism=bool(planning.elitism),
        face_adaptive=bool(planning.face_adaptive),
        num_samples_max=None if planning.num_samples_max is None else int(planning.num_samples_max),
        face_stall_iters=int(planning.face_stall_iters),
        nan_policy=str(planning.nan_policy),
        apply_steps=int(planning.apply_steps),
        max_total_actions=int(planning.max_total_actions),
        constant_action_cost=float(planning.constant_action_cost),
        max_decode_attempts=int(planning.max_decode_attempts),
        invalid_action_penalty=float(planning.invalid_action_penalty),
        initial_std=float(planning.initial_std),
        action_pool_size=int(getattr(planning, "action_pool_size", 512)),
        component_std=None if getattr(planning, "component_std", None) is None else float(planning.component_std),
        min_std=float(getattr(planning, "min_std", 1e-6)),
        max_std=None if getattr(planning, "max_std", None) is None else float(planning.max_std),
        dirichlet_prior=float(getattr(planning, "dirichlet_prior", 0.0)),
        device=device,
        seed=seed,
    )


def _build_structured_ce_planning_config(config: Any, *, device: torch.device, seed: int) -> StructuredCEPlannerConfig:
    planning = config.planning
    return StructuredCEPlannerConfig(
        horizon=int(planning.horizon),
        num_samples=int(planning.num_samples),
        elite_frac=float(planning.elite_frac),
        max_iters=int(planning.max_iters),
        temperature=None if getattr(planning, "temperature", None) is None else float(planning.temperature),
        smoothing=float(planning.smoothing),
        dirichlet_prior=float(getattr(planning, "dirichlet_prior", 0.0)),
        convergence_tol=float(planning.convergence_tol),
        quantile_window=int(planning.quantile_window),
        elitism=bool(planning.elitism),
        face_adaptive=bool(planning.face_adaptive),
        num_samples_max=None if planning.num_samples_max is None else int(planning.num_samples_max),
        face_stall_iters=int(planning.face_stall_iters),
        nan_policy=str(planning.nan_policy),
        apply_steps=int(planning.apply_steps),
        max_total_actions=int(planning.max_total_actions),
        constant_action_cost=float(planning.constant_action_cost),
        max_decode_attempts=int(planning.max_decode_attempts),
        invalid_action_penalty=float(planning.invalid_action_penalty),
        device=device,
        seed=seed,
    )


def _build_action_decoder(parsed_problem: Any, action_encoder: nn.Module, config: Any, *, seed: int) -> ActionDecoder:
    decoder = config.planning.action_decoder
    return ActionDecoder(
        parsed_problem=parsed_problem,
        action_encoder=action_encoder,
        method=str(decoder.method),
        metric=str(decoder.metric),
        num_samples=None if decoder.num_samples is None else int(decoder.num_samples),
        elite_frac=float(decoder.elite_frac),
        max_iters=int(decoder.max_iters),
        smoothing=float(decoder.smoothing),
        dirichlet_prior=float(decoder.dirichlet_prior),
        temperature=float(decoder.temperature),
        decode_chunk_size=(
            None if getattr(decoder, "decode_chunk_size", None) is None else int(decoder.decode_chunk_size)
        ),
        seed=seed,
    )


def _build_simulator_engine(domain_path: Path, problem_path: Path) -> Any:
    from simulator import SimulatorEngine

    return SimulatorEngine.from_pddl(domain_path, problem_path)


def _step_metrics(output: Any) -> dict[str, float]:
    metrics = {
        "total_loss": float(output.total_loss.detach().cpu().item()),
        "jepa_loss": float(output.jepa_loss.detach().cpu().item()),
    }
    if output.goal_loss is not None:
        metrics["goal_loss"] = float(output.goal_loss.detach().cpu().item())
    for attribute, metric_name in (
        ("action_vicreg_loss", "action_vicreg_loss"),
        ("action_contrastive_loss", "action_contrastive_loss"),
        ("argument_reconstruction_loss", "argument_reconstruction_loss"),
        ("applicability_loss", "applicability_loss"),
    ):
        value = getattr(output, attribute, None)
        if value is not None:
            metrics[metric_name] = float(value.detach().cpu().item())
    for name, value in output.terms.items():
        metrics[f"term/{name.replace('/', '_')}"] = float(value.detach().cpu().item())
    return metrics


def _evaluate(trainer: Any, loader: Iterable[Any], device: torch.device) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        output = trainer.eval_step(_to_device(batch, device))
        for key, value in _step_metrics(output).items():
            totals[key] += value
        count += 1
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {key: value / count for key, value in sorted(totals.items())}


def _load_checkpoint_state(bundle: Any, checkpoint: dict[str, Any]) -> None:
    bundle.jepa.load_state_dict(checkpoint["model_state_dict"])
    if bundle.goal_head is not None and checkpoint.get("goal_head_state_dict") is not None:
        bundle.goal_head.load_state_dict(checkpoint["goal_head_state_dict"])
    applicability_state = checkpoint.get("applicability_head_state_dict")
    if bundle.applicability_head is not None:
        if applicability_state is None:
            warnings.warn(
                "Checkpoint does not contain applicability_head_state_dict; leaving applicability head initialized",
                UserWarning,
                stacklevel=2,
            )
        else:
            bundle.applicability_head.load_state_dict(applicability_state)
    _load_optional_module_state(
        bundle.action_contrastive_anchor,
        checkpoint,
        "action_contrastive_anchor_state_dict",
        "action contrastive anchor",
    )
    _load_optional_module_state(
        bundle.argument_reconstruction_head,
        checkpoint,
        "argument_reconstruction_head_state_dict",
        "argument reconstruction head",
    )


def _load_optional_module_state(
    module: nn.Module | None,
    checkpoint: dict[str, Any],
    key: str,
    label: str,
) -> None:
    if module is None:
        return
    state = checkpoint.get(key)
    if state is None:
        warnings.warn(
            f"Checkpoint does not contain {key}; leaving {label} initialized",
            UserWarning,
            stacklevel=2,
        )
        return
    module.load_state_dict(state)


def _save_checkpoint(path: Path, bundle: Any, config: Any, epoch: int, step: int, best_eval: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": bundle.jepa.state_dict(),
            "goal_head_state_dict": None if bundle.goal_head is None else bundle.goal_head.state_dict(),
            "applicability_head_state_dict": (
                None if bundle.applicability_head is None else bundle.applicability_head.state_dict()
            ),
            "action_contrastive_anchor_state_dict": (
                None
                if bundle.action_contrastive_anchor is None
                else bundle.action_contrastive_anchor.state_dict()
            ),
            "argument_reconstruction_head_state_dict": (
                None
                if bundle.argument_reconstruction_head is None
                else bundle.argument_reconstruction_head.state_dict()
            ),
            "optimizer_state_dict": bundle.optimizer.state_dict(),
            "scheduler_state_dict": None if bundle.scheduler is None else bundle.scheduler.state_dict(),
            "config": config_to_container(config),
            "vocab_sizes": vocab_sizes_to_dict(bundle.vocab_sizes),
            "epoch": int(epoch),
            "step": int(step),
            "best_eval": float(best_eval),
        },
        path,
    )


def _maybe_save_best(
    eval_loss: float,
    best_eval: float,
    path: Path,
    bundle: Any,
    config: Any,
    epoch: int,
    step: int,
) -> float:
    if eval_loss < best_eval:
        _save_checkpoint(path, bundle, config, epoch, step, eval_loss)
        return eval_loss
    return best_eval


def _scheduler_lr(scheduler: Any) -> float:
    return float(scheduler.get_last_lr()[0])


def _runtime_metrics(start_time: float, *, steps: int | None = None, examples: int | None = None) -> dict[str, float]:
    seconds = max(time.perf_counter() - start_time, 1e-9)
    metrics = {"seconds": seconds}
    if steps is not None:
        metrics["steps"] = float(steps)
        metrics["steps_per_second"] = float(steps) / seconds
    if examples is not None:
        metrics["examples"] = float(examples)
        metrics["examples_per_second"] = float(examples) / seconds
    return metrics


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
