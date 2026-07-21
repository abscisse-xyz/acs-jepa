"""Model construction for the ACS-JEPA CLI."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from acs_jepa import (
    ActionContrastiveLoss,
    ActionEncoder,
    ActionVICRegLoss,
    ApplicabilityHead,
    ApplicabilityLoss,
    ArgumentReconstructionHead,
    ArgumentReconstructionLoss,
    ConditionalSampleTerminalLatentGeneratorG,
    DiagonalGaussianTerminalLatentDistributionP,
    GaussianMixtureTerminalLatentDistributionP,
    GraphEncodedActionInverseDynamicsLoss,
    GraphInverseDynamicsModel,
    GraphJEPA,
    GraphJEPALossModule,
    GraphLatentPredictionLoss,
    GraphTemporalSimilarityLoss,
    GraphVCLoss,
    JepaTrainer,
    JepaTrainerConfig,
    PredicateEvaluator,
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
    applicability_head: nn.Module | None
    applicability_loss_module: nn.Module | None
    action_vicreg_loss_module: nn.Module | None
    action_contrastive_loss_module: nn.Module | None
    action_contrastive_anchor: nn.Module | None
    argument_reconstruction_head: nn.Module | None
    argument_reconstruction_loss_module: nn.Module | None
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
    _validate_action_auxiliary_config(resolved_vocab, config)
    jepa, goal_head = build_jepa(parsed_problems, config, vocab_sizes=resolved_vocab)
    applicability_enabled = (
        float(config.trainer.applicability_loss_weight) > 0.0
        or float(config.model.loss.applicability_coeff) > 0.0
    )
    applicability_head = (
        build_applicability_head(resolved_vocab, config) if applicability_enabled else None
    )
    applicability_loss_module = (
        build_applicability_loss_module(config) if applicability_enabled else None
    )

    (
        action_vicreg_loss_module,
        action_contrastive_loss_module,
        action_contrastive_anchor,
        argument_reconstruction_head,
        argument_reconstruction_loss_module,
    ) = build_action_auxiliary_modules(resolved_vocab, config)
    jepa.to(device)
    if goal_head is not None:
        goal_head.to(device)
    if applicability_head is not None:
        applicability_head.to(device)
    for module in (action_contrastive_anchor, argument_reconstruction_head):
        if module is not None:
            module.to(device)
    optimizer = build_optimizer(
        [
            jepa,
            goal_head,
            applicability_head,
            action_contrastive_anchor,
            argument_reconstruction_head,
        ],
        config,
    )
    scheduler = None if total_steps is None else build_scheduler(optimizer, config, total_steps=total_steps)
    trainer = JepaTrainer(
        jepa=jepa,
        optimizer=optimizer,
        config=JepaTrainerConfig(
            goal_head_kind=str(config.model.goal_head.kind),
            jepa_loss_weight=float(config.trainer.jepa_loss_weight),
            goal_loss_weight=float(config.trainer.goal_loss_weight),
            goal_head_detach=bool(config.trainer.goal_head_detach),
            applicability_loss_weight=float(config.trainer.applicability_loss_weight),
            integrated_applicability_loss_weight=float(config.model.loss.applicability_coeff),
            applicability_head_detach=bool(config.trainer.applicability_head_detach),
            action_vicreg_loss_weight=float(config.model.loss.action_vicreg_coeff),
            action_contrastive_loss_weight=float(config.model.loss.action_contrastive_coeff),
            argument_reconstruction_loss_weight=float(
                config.model.loss.argument_reconstruction_coeff
            ),
            grad_clip_norm=None
            if config.trainer.grad_clip_norm is None
            else float(config.trainer.grad_clip_norm),
        ),
        goal_head=goal_head,
        applicability_head=applicability_head,
        applicability_loss_module=applicability_loss_module,
        action_vicreg_loss_module=action_vicreg_loss_module,
        action_contrastive_loss_module=action_contrastive_loss_module,
        action_contrastive_anchor=action_contrastive_anchor,
        argument_reconstruction_head=argument_reconstruction_head,
        argument_reconstruction_loss_module=argument_reconstruction_loss_module,
    )
    return ModelBundle(
        jepa=jepa,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        goal_head=goal_head,
        applicability_head=applicability_head,
        applicability_loss_module=applicability_loss_module,
        action_vicreg_loss_module=action_vicreg_loss_module,
        action_contrastive_loss_module=action_contrastive_loss_module,
        action_contrastive_anchor=action_contrastive_anchor,
        argument_reconstruction_head=argument_reconstruction_head,
        argument_reconstruction_loss_module=argument_reconstruction_loss_module,
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
    similarity_coeff = float(loss_cfg.similarity_coeff)
    inverse_dynamics_coeff = float(loss_cfg.inverse_dynamics_coeff)
    enable_auxiliary_terms = bool(getattr(loss_cfg, "enable_auxiliary_terms", False))
    temporal_similarity_loss = (
        GraphTemporalSimilarityLoss() if enable_auxiliary_terms and similarity_coeff != 0.0 else None
    )
    inverse_dynamics_loss = None
    if enable_auxiliary_terms and inverse_dynamics_coeff != 0.0:
        inverse_dynamics_loss = GraphEncodedActionInverseDynamicsLoss(
            GraphInverseDynamicsModel(
                latent_dim=int(model_cfg.latent_dim),
                action_dim=int(model_cfg.action_dim),
                hidden_dim=int(model_cfg.action_encoder.hidden_dim),
            )
        )
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
        temporal_similarity_loss=temporal_similarity_loss,
        inverse_dynamics_loss=inverse_dynamics_loss,
        prediction_coeff=float(loss_cfg.prediction_coeff),
        regularization_coeff=float(loss_cfg.regularization_coeff),
        similarity_coeff=similarity_coeff,
        inverse_dynamics_coeff=inverse_dynamics_coeff,
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


def build_applicability_head(vocab: VocabSizes, config: DictConfig) -> nn.Module | None:
    """Build the configured auxiliary applicability head."""

    head_cfg = config.model.applicability_head
    kind = str(head_cfg.kind)
    if kind == "none":
        return None
    if kind == "mlp":
        return ApplicabilityHead(
            latent_dim=int(config.model.latent_dim),
            action_dim=int(config.model.action_dim),
            max_action_arity=vocab.max_action_arity,
            hidden_dim=int(head_cfg.hidden_dim),
            dropout=float(head_cfg.dropout),
        )
    raise ValueError(f"Unknown applicability head kind: {kind}")


def build_applicability_loss_module(config: DictConfig) -> ApplicabilityLoss | None:
    """Build the auxiliary applicability BCE loss when configured."""

    pos_weight = config.trainer.applicability_pos_weight
    if pos_weight is not None and (
        not math.isfinite(float(pos_weight)) or float(pos_weight) <= 0.0
    ):
        raise ValueError("applicability_pos_weight must be positive and finite when provided")
    if str(config.model.applicability_head.kind) == "none":
        return None
    return ApplicabilityLoss(pos_weight=None if pos_weight is None else float(pos_weight))


def build_action_auxiliary_modules(
    vocab: VocabSizes, config: DictConfig
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, nn.Module | None, nn.Module | None]:
    """Build disabled-by-default Phase 2 action auxiliary modules."""

    loss_cfg = config.model.loss
    vicreg_coeff = float(loss_cfg.action_vicreg_coeff)
    contrastive_coeff = float(loss_cfg.action_contrastive_coeff)
    argument_coeff = float(loss_cfg.argument_reconstruction_coeff)
    vicreg = None
    contrastive = None
    anchor = None
    argument_head = None
    argument_loss = None
    if vicreg_coeff > 0.0:
        vicreg = ActionVICRegLoss(
            std_coeff=float(loss_cfg.action_vicreg_std_coeff),
            cov_coeff=float(loss_cfg.action_vicreg_cov_coeff),
            std_margin=float(loss_cfg.action_vicreg_std_margin),
        )
    if contrastive_coeff > 0.0:
        contrastive = ActionContrastiveLoss(
            temperature=float(loss_cfg.action_contrastive_temperature)
        )
        anchor = GraphInverseDynamicsModel(
            latent_dim=int(config.model.latent_dim),
            action_dim=int(config.model.action_dim),
            hidden_dim=int(config.model.action_encoder.hidden_dim),
        )
    if argument_coeff > 0.0:
        head_cfg = config.model.argument_reconstruction_head
        argument_head = ArgumentReconstructionHead(
            action_dim=int(config.model.action_dim),
            object_dim=int(config.model.latent_dim),
            max_action_arity=vocab.max_action_arity,
            hidden_dim=int(head_cfg.hidden_dim),
            dropout=float(head_cfg.dropout),
        )
        argument_loss = ArgumentReconstructionLoss()
    return vicreg, contrastive, anchor, argument_head, argument_loss


def _validate_action_auxiliary_config(
    vocab: VocabSizes,
    config: DictConfig,
) -> None:
    loss_cfg = config.model.loss
    coefficients = {
        "action_vicreg_coeff": loss_cfg.action_vicreg_coeff,
        "action_contrastive_coeff": loss_cfg.action_contrastive_coeff,
        "argument_reconstruction_coeff": loss_cfg.argument_reconstruction_coeff,
        "applicability_coeff": loss_cfg.applicability_coeff,
        "action_sigreg_coeff": loss_cfg.action_sigreg_coeff,
        "trainer.applicability_loss_weight": config.trainer.applicability_loss_weight,
    }
    for name, raw_value in coefficients.items():
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in ("action_vicreg_std_coeff", "action_vicreg_cov_coeff"):
        value = float(loss_cfg[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in ("action_vicreg_std_margin", "action_contrastive_temperature"):
        value = float(loss_cfg[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if float(loss_cfg.action_sigreg_coeff) != 0.0:
        raise ValueError("action_sigreg_coeff is not implemented and must remain zero")
    integrated = float(loss_cfg.applicability_coeff)
    legacy = float(config.trainer.applicability_loss_weight)
    applicability_kind = str(config.model.applicability_head.kind)
    if applicability_kind not in {"none", "mlp"}:
        raise ValueError(f"Unknown applicability head kind: {applicability_kind}")
    dropout = float(config.model.applicability_head.dropout)
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("applicability head dropout must be finite and in [0, 1)")
    hidden_dim = config.model.applicability_head.hidden_dim
    if isinstance(hidden_dim, bool) or int(hidden_dim) != hidden_dim or int(hidden_dim) <= 0:
        raise ValueError("applicability head hidden_dim must be a positive integer")
    pos_weight = config.trainer.applicability_pos_weight
    if pos_weight is not None and (
        not math.isfinite(float(pos_weight)) or float(pos_weight) <= 0.0
    ):
        raise ValueError("applicability_pos_weight must be positive and finite when provided")
    if integrated > 0.0 and legacy > 0.0:
        raise ValueError("legacy and integrated applicability losses are mutually exclusive")
    if legacy > 0.0 and applicability_kind == "none":
        raise ValueError("applicability_loss_weight > 0 requires model.applicability_head.kind != 'none'")
    if integrated > 0.0 and applicability_kind == "none":
        raise ValueError("applicability_coeff > 0 requires model.applicability_head.kind != 'none'")

    raw_negatives = loss_cfg.action_hard_negatives_per_positive
    if isinstance(raw_negatives, bool) or int(raw_negatives) != raw_negatives or int(raw_negatives) < 0:
        raise ValueError("action_hard_negatives_per_positive must be a non-negative integer")
    for name in ("action_supervision_seed", "action_negative_max_attempts_per_category"):
        raw_value = config.data[name]
        if isinstance(raw_value, bool) or int(raw_value) != raw_value:
            raise ValueError(f"{name} must be an integer")
    if int(config.data.action_negative_max_attempts_per_category) <= 0:
        raise ValueError("action_negative_max_attempts_per_category must be positive")
    num_negatives = int(raw_negatives)
    if (float(loss_cfg.action_contrastive_coeff) > 0.0 or integrated > 0.0) and num_negatives == 0:
        raise ValueError("contrastive and integrated applicability losses require action hard negatives")
    argument_head_cfg = config.model.argument_reconstruction_head
    argument_kind = str(argument_head_cfg.kind)
    if argument_kind not in {"none", "mlp"}:
        raise ValueError(f"Unknown argument reconstruction head kind: {argument_kind}")
    argument_hidden_dim = argument_head_cfg.hidden_dim
    if (
        isinstance(argument_hidden_dim, bool)
        or int(argument_hidden_dim) != argument_hidden_dim
        or int(argument_hidden_dim) <= 0
    ):
        raise ValueError("argument reconstruction head hidden_dim must be a positive integer")
    argument_dropout = float(argument_head_cfg.dropout)
    if not math.isfinite(argument_dropout) or not 0.0 <= argument_dropout < 1.0:
        raise ValueError("argument reconstruction head dropout must be finite and in [0, 1)")
    if float(loss_cfg.argument_reconstruction_coeff) > 0.0:
        if argument_kind == "none":
            raise ValueError("argument_reconstruction_coeff > 0 requires an enabled argument head")
        if vocab.max_action_arity < 1:
            raise ValueError("argument reconstruction requires max_action_arity >= 1")
        if vocab.num_objects < 1:
            raise ValueError("argument reconstruction requires at least one object")

    raw_path = config.data.action_applicability_table_path
    if integrated > 0.0:
        if raw_path is None:
            raise ValueError("integrated applicability requires action_applicability_table_path")
        if not Path(str(raw_path)).is_absolute():
            raise ValueError("action_applicability_table_path must be absolute")


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
