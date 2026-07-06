"""Model construction for the ACS-JEPA CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from acs_jepa import (
    ConditionalSampleTerminalLatentGeneratorG,
    DiagonalGaussianTerminalLatentDistributionP,
    GaussianMixtureTerminalLatentDistributionP,
    GraphJEPA,
    GraphJEPALossModule,
    GraphLatentPredictionLoss,
    GraphVCLoss,
    JepaTrainer,
    JepaTrainerConfig,
    PredicateEvaluator,
    ActionEncoder,
    StateEncoderF,
    build_action_encoder,
    build_latent_predictor,
)
from acs_jepa.graph import GraphEncoder
from acs_jepa.graph.schemas import ParsedProblem
from omegaconf import DictConfig

from acs_jepa_cli.scheduling import NoOpScheduler, WarmupCosineScheduler, build_scheduler


@dataclass(frozen=True)
class VocabSizes:
    num_types: int
    num_predicates: int
    num_objects: int
    num_actions: int
    max_arity: int
    max_action_arity: int
    max_predicate_arity: int


@dataclass(frozen=True)
class ModelBundle:
    jepa: GraphJEPA
    optimizer: torch.optim.Optimizer
    scheduler: NoOpScheduler | WarmupCosineScheduler | None
    trainer: JepaTrainer
    goal_head: nn.Module | None
    vocab_sizes: VocabSizes


def build_model_bundle(
    parsed_problems: tuple[ParsedProblem, ...],
    config: DictConfig,
    *,
    device: torch.device,
    vocab_sizes: VocabSizes | None = None,
    total_steps: int | None = None,
) -> ModelBundle:
    """Build JEPA, optimizer, trainer, and optional goal head from config."""

    resolved_vocab = compute_vocab_sizes(parsed_problems) if vocab_sizes is None else vocab_sizes
    jepa, goal_head = build_jepa(parsed_problems, config, vocab_sizes=resolved_vocab)
    jepa.to(device)
    if goal_head is not None:
        goal_head.to(device)
    optimizer = build_optimizer([jepa, goal_head], config)
    scheduler = None if total_steps is None else build_scheduler(optimizer, config, total_steps=total_steps)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=optimizer,
        config=JepaTrainerConfig(
            goal_head_kind=str(config.model.goal_head.kind),
            jepa_loss_weight=float(config.trainer.jepa_loss_weight),
            goal_loss_weight=float(config.trainer.goal_loss_weight),
            goal_head_detach=bool(config.trainer.goal_head_detach),
            grad_clip_norm=None
            if config.trainer.grad_clip_norm is None
            else float(config.trainer.grad_clip_norm),
        ),
        goal_head=goal_head,
    )
    return ModelBundle(
        jepa=jepa,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        goal_head=goal_head,
        vocab_sizes=resolved_vocab,
    )


def build_jepa(
    parsed_problems: tuple[ParsedProblem, ...],
    config: DictConfig,
    *,
    vocab_sizes: VocabSizes | None = None,
) -> tuple[GraphJEPA, nn.Module | None]:
    """Build a ``GraphJEPA`` and optional goal head."""

    vocab = compute_vocab_sizes(parsed_problems) if vocab_sizes is None else vocab_sizes
    model_cfg = config.model
    graph_encoder = GraphEncoder(
        num_node_kinds=2,
        num_types=vocab.num_types,
        num_predicates=vocab.num_predicates,
        num_objects=vocab.num_objects,
        max_arity=vocab.max_arity,
        max_role_id=max(0, vocab.max_arity - 1),
        hidden_dim=int(model_cfg.graph_hidden_dim),
        embed_dim=int(model_cfg.graph_embed_dim),
        num_layers=int(model_cfg.graph_layers),
    )
    state_encoder = StateEncoderF(
        embedding_dim=int(model_cfg.graph_embed_dim),
        latent_dim=int(model_cfg.latent_dim),
        hidden_dim=int(model_cfg.predictor.hidden_dim),
        context_steps=None if model_cfg.state_context_steps is None else int(model_cfg.state_context_steps),
    )
    action_kwargs: dict[str, Any] = {
        "num_actions": vocab.num_actions,
        "max_action_arity": vocab.max_action_arity,
        "action_dim": int(model_cfg.action_dim),
        "hidden_dim": int(model_cfg.action_encoder.hidden_dim),
        "latent_dim": int(model_cfg.latent_dim),
    }
    base_action_encoder = build_action_encoder(
        kind=str(model_cfg.action_encoder.kind),
        **action_kwargs,
    )
    action_encoder = ActionEncoder(
        base_action_encoder,
        action_dim=int(model_cfg.action_dim),
        context_steps=None if model_cfg.action_context_steps is None else int(model_cfg.action_context_steps),
    )
    predictor = build_latent_predictor(
        kind=str(model_cfg.predictor.kind),
        latent_dim=int(model_cfg.latent_dim),
        action_dim=int(model_cfg.action_dim),
        hidden_dim=int(model_cfg.predictor.hidden_dim),
    )
    loss_cfg = model_cfg.loss
    loss_module = GraphJEPALossModule(
        prediction_loss=GraphLatentPredictionLoss(
            graph_weight=float(loss_cfg.prediction.graph_weight),
            object_weight=float(loss_cfg.prediction.object_weight),
        ),
        regularization_loss=GraphVCLoss(
            std_coeff=float(loss_cfg.regularization.std_coeff),
            cov_coeff=float(loss_cfg.regularization.cov_coeff),
            std_margin=float(loss_cfg.regularization.std_margin),
            target=str(loss_cfg.regularization.target),
        ),
        prediction_coeff=float(loss_cfg.prediction_coeff),
        regularization_coeff=float(loss_cfg.regularization_coeff),
        similarity_coeff=float(loss_cfg.similarity_coeff),
        inverse_dynamics_coeff=float(loss_cfg.inverse_dynamics_coeff),
        rollout_order_weights=None
        if loss_cfg.rollout_order_weights is None
        else [float(weight) for weight in loss_cfg.rollout_order_weights],
    )
    goal_head = build_goal_head(vocab, config)
    return (
        GraphJEPA(
            graph_encoder=graph_encoder,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            predictor=predictor,
            loss_module=loss_module,
        ),
        goal_head,
    )


def build_goal_head(vocab: VocabSizes, config: DictConfig) -> nn.Module | None:
    """Build the configured goal head."""

    goal_cfg = config.model.goal_head
    kind = str(goal_cfg.kind)
    common = {
        "num_predicates": vocab.num_predicates,
        "max_predicate_arity": vocab.max_predicate_arity,
        "latent_dim": int(config.model.latent_dim),
        "hidden_dim": int(goal_cfg.hidden_dim),
    }
    if kind == "none":
        return None
    if kind == "predicate":
        return PredicateEvaluator(
            **common,
            argument_encoder=str(goal_cfg.argument_encoder),
        )
    if kind == "gaussian":
        return DiagonalGaussianTerminalLatentDistributionP(**common)
    if kind == "gmm":
        return GaussianMixtureTerminalLatentDistributionP(
            **common,
            num_components=int(goal_cfg.num_components),
        )
    if kind == "conditional_sampler":
        return ConditionalSampleTerminalLatentGeneratorG(
            **common,
            num_samples=int(goal_cfg.num_samples),
        )
    raise ValueError(f"Unknown goal head kind: {kind}")


def build_optimizer(modules: list[nn.Module | None], config: DictConfig) -> torch.optim.Optimizer:
    """Build the configured optimizer."""

    parameters = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    name = str(config.optimizer.name).lower()
    if name != "adam":
        raise ValueError(f"Only adam optimizer is supported in v1, got {name}")
    return torch.optim.Adam(
        parameters,
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
    )


def compute_vocab_sizes(parsed_problems: tuple[ParsedProblem, ...]) -> VocabSizes:
    """Compute model vocabulary sizes over a full corpus."""

    if not parsed_problems:
        raise ValueError("Cannot build a model without parsed problems")
    max_predicate_arity = max(
        (len(predicate.arg_types) for parsed in parsed_problems for predicate in parsed.predicates.values()),
        default=0,
    )
    max_action_arity = max(parsed.max_action_arity for parsed in parsed_problems)
    max_arity = max(max_predicate_arity, max_action_arity)
    return VocabSizes(
        num_types=max(len(parsed.types) for parsed in parsed_problems),
        num_predicates=max(len(parsed.predicates) for parsed in parsed_problems),
        num_objects=max(len(parsed.objects) for parsed in parsed_problems),
        num_actions=max(len(parsed.actions) for parsed in parsed_problems),
        max_arity=max_arity,
        max_action_arity=max_action_arity,
        max_predicate_arity=max_predicate_arity,
    )


def vocab_sizes_to_dict(vocab_sizes: VocabSizes) -> dict[str, int]:
    return asdict(vocab_sizes)


def vocab_sizes_from_dict(payload: dict[str, Any]) -> VocabSizes:
    return VocabSizes(**{key: int(value) for key, value in payload.items()})
