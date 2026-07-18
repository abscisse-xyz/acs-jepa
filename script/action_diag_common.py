"""Shared helpers for action-latent diagnostic scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from acs_jepa.graph import build_state_graph
from acs_jepa.graph.schemas import GroundAction
from omegaconf import OmegaConf

from acs_jepa_cli.cli import _build_simulator_engine, _resolve_device, _to_device
from acs_jepa_cli.config import load_config
from acs_jepa_cli.data import LoadedCorpus, load_corpus, split_corpus
from acs_jepa_cli.modeling import ModelBundle, build_model_bundle, vocab_sizes_from_dict


def load_checkpoint_bundle(
    dataset_dir: Path,
    checkpoint_path: Path,
    *,
    device_name: str,
) -> tuple[Any, LoadedCorpus, ModelBundle, torch.device]:
    """Load a checkpoint, corpus, model bundle, and resolved device."""

    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = OmegaConf.merge(load_config(None), OmegaConf.create(checkpoint["config"]))
    corpus = load_corpus([dataset_dir], strict=True)
    vocab_sizes = vocab_sizes_from_dict(checkpoint["vocab_sizes"])
    bundle = build_model_bundle(corpus.parsed_problems, config, device=device, vocab_sizes=vocab_sizes)
    bundle.jepa.load_state_dict(checkpoint["model_state_dict"])
    bundle.jepa.eval()
    if bundle.goal_head is not None:
        bundle.goal_head.eval()
    return config, corpus, bundle, device


def select_split(corpus: LoadedCorpus, config: Any, split: str, *, seed: int) -> LoadedCorpus:
    """Return the selected corpus split."""

    if split == "all":
        return corpus
    split_seed = seed if config.data.split_seed is None else int(config.data.split_seed)
    splits = split_corpus(
        corpus,
        val_fraction=float(config.data.val_fraction),
        test_fraction=float(config.data.test_fraction),
        seed=split_seed,
    )
    return getattr(splits, split)


def problem_path(dataset_dir: Path, problem_name: str) -> Path:
    """Resolve a problem name from simulator metadata to a PDDL path."""

    problem_dir = dataset_dir / "problem"
    direct = problem_dir / problem_name
    if direct.exists():
        return direct
    with_suffix = problem_dir / f"{problem_name}.pddl"
    if with_suffix.exists():
        return with_suffix
    raise FileNotFoundError(f"No PDDL file found for problem {problem_name!r} in {problem_dir}")


def encode_state(bundle: ModelBundle, parsed: Any, atoms: Any, *, device: torch.device) -> Any:
    """Build and encode one symbolic state graph."""

    state_graph = _to_device(build_state_graph(parsed, atoms, include_static=True), device)
    return bundle.jepa.encode(state_graph)


def action_payload(action: Any) -> dict[str, Any]:
    return {"name": action.name, "arguments": list(action.arguments)}


def action_key(action: Any) -> tuple[str, tuple[str, ...]]:
    return action.name, tuple(action.arguments)


def replay_to_engine(dataset_dir: Path, problem_name: str, replay_actions: Any) -> Any:
    """Build a simulator engine and replay a reference prefix."""

    engine = _build_simulator_engine(dataset_dir / "problem" / "domain.pddl", problem_path(dataset_dir, problem_name))
    for action in replay_actions:
        engine.apply_action(action.name, action.arguments, finish=True)
    return engine


def applicable_keys(engine: Any) -> set[tuple[str, tuple[str, ...]]]:
    """Return currently applicable action keys from the simulator oracle."""

    return {(action.name, tuple(action.arguments)) for action in engine.applicable_actions()}


def is_applicable(engine: Any, action: GroundAction) -> bool:
    """Check whether an action is in the current applicable oracle set."""

    return action_key(action) in applicable_keys(engine)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def iter_transitions(corpus: LoadedCorpus, *, max_transitions: int | None = None):
    """Yield trajectory transition views with stable metadata."""

    total = 0
    for trajectory, record in zip(corpus.trajectories, corpus.records):
        for step_idx, action in enumerate(trajectory.actions):
            if max_transitions is not None and total >= max_transitions:
                return
            yield trajectory, record, step_idx, action
            total += 1
