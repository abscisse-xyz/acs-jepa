"""Training helpers for graph-native JEPA models and goal heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from acs_jepa.architectures import JEPALatentState, flatten_temporal_latent_state, latent_time_slice
from acs_jepa.goals import ConditionalSampleGeneratorLoss, PredicateEvaluatorSampleLoss
from acs_jepa.jepa import GraphJEPA, GraphJEPATrainingOutput
from acs_jepa.losses import ApplicabilityLossOutput

GoalHeadKind = Literal["none", "predicate", "gaussian", "gmm", "conditional_sampler"]


@dataclass(frozen=True)
class JepaTrainerConfig:
    """Configuration for trajectory JEPA training with one goal head."""

    goal_head_kind: GoalHeadKind = "none"
    jepa_loss_weight: float = 1.0
    goal_loss_weight: float = 1.0
    goal_head_detach: bool = True
    applicability_loss_weight: float = 0.0
    applicability_head_detach: bool = True
    grad_clip_norm: float | None = None


@dataclass(frozen=True)
class JepaTrainerStepOutput:
    """Detached scalar losses and rollout output from a trainer step."""

    total_loss: Tensor
    jepa_loss: Tensor
    goal_loss: Tensor | None
    applicability_loss: Tensor | None
    terms: dict[str, Tensor]
    rollout: GraphJEPATrainingOutput


class JepaTrainer:
    """Train ``GraphJEPA.trajectory_rollout`` plus at most one goal scoring head."""

    def __init__(
        self,
        *,
        jepa: GraphJEPA,
        optimizer: torch.optim.Optimizer,
        config: JepaTrainerConfig | None = None,
        goal_head: nn.Module | None = None,
        goal_loss_module: nn.Module | None = None,
        applicability_head: nn.Module | None = None,
        applicability_loss_module: nn.Module | None = None,
    ) -> None:
        self.jepa = jepa
        self.optimizer = optimizer
        self.config = JepaTrainerConfig() if config is None else config
        self.goal_head = goal_head
        self.goal_loss_module = self._build_goal_loss_module(goal_loss_module)
        self.applicability_head = applicability_head
        self.applicability_loss_module = applicability_loss_module
        self._validate_config()

    def train_step(self, batch: dict[str, object]) -> JepaTrainerStepOutput:
        """Run one optimization step over a PyG transition batch."""

        self.jepa.train()
        if self.goal_head is not None:
            self.goal_head.train()
        if self.goal_loss_module is not None:
            self.goal_loss_module.train()
        if self.applicability_head is not None:
            self.applicability_head.train()
        if self.applicability_loss_module is not None:
            self.applicability_loss_module.train()

        self.optimizer.zero_grad(set_to_none=True)
        rollout = self.jepa.trajectory_rollout(
            _required(batch, "states"),
            _required(batch, "actions"),
        )
        jepa_loss = rollout.loss.total
        goal_loss = self._goal_loss(batch, rollout)
        applicability_loss = self._applicability_loss(batch)
        total_loss = self.config.jepa_loss_weight * jepa_loss
        if goal_loss is not None:
            total_loss = total_loss + self.config.goal_loss_weight * goal_loss
        if applicability_loss is not None:
            total_loss = total_loss + self.config.applicability_loss_weight * applicability_loss.total

        total_loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                _unique_trainable_parameters(
                    self.jepa,
                    self.goal_head,
                    self.goal_loss_module,
                    self.applicability_head,
                    self.applicability_loss_module,
                ),
                self.config.grad_clip_norm,
            )
        self.optimizer.step()

        terms = {name: value.detach() for name, value in rollout.loss.terms.items()}
        terms["trainer_total"] = total_loss.detach()
        if goal_loss is not None:
            terms["goal"] = goal_loss.detach()
        if applicability_loss is not None:
            _add_applicability_terms(terms, applicability_loss)
        return JepaTrainerStepOutput(
            total_loss=total_loss.detach(),
            jepa_loss=jepa_loss.detach(),
            goal_loss=None if goal_loss is None else goal_loss.detach(),
            applicability_loss=None if applicability_loss is None else applicability_loss.total.detach(),
            terms=terms,
            rollout=rollout,
        )

    def eval_step(self, batch: dict[str, object]) -> JepaTrainerStepOutput:
        """Evaluate one PyG transition batch without gradients or optimizer updates."""

        self.jepa.eval()
        if self.goal_head is not None:
            self.goal_head.eval()
        if self.goal_loss_module is not None:
            self.goal_loss_module.eval()
        if self.applicability_head is not None:
            self.applicability_head.eval()
        if self.applicability_loss_module is not None:
            self.applicability_loss_module.eval()

        with torch.no_grad():
            rollout = self.jepa.trajectory_rollout(
                _required(batch, "states"),
                _required(batch, "actions"),
            )
            jepa_loss = rollout.loss.total
            goal_loss = self._goal_loss(batch, rollout)
            applicability_loss = self._applicability_loss(batch)
            total_loss = self.config.jepa_loss_weight * jepa_loss
            if goal_loss is not None:
                total_loss = total_loss + self.config.goal_loss_weight * goal_loss
            if applicability_loss is not None:
                total_loss = total_loss + self.config.applicability_loss_weight * applicability_loss.total

        terms = {name: value.detach() for name, value in rollout.loss.terms.items()}
        terms["trainer_total"] = total_loss.detach()
        if goal_loss is not None:
            terms["goal"] = goal_loss.detach()
        if applicability_loss is not None:
            _add_applicability_terms(terms, applicability_loss)
        return JepaTrainerStepOutput(
            total_loss=total_loss.detach(),
            jepa_loss=jepa_loss.detach(),
            goal_loss=None if goal_loss is None else goal_loss.detach(),
            applicability_loss=None if applicability_loss is None else applicability_loss.total.detach(),
            terms=terms,
            rollout=rollout,
        )

    def _goal_loss(self, batch: dict[str, object], rollout: GraphJEPATrainingOutput) -> Tensor | None:
        kind = self.config.goal_head_kind
        if kind == "none":
            return None
        if kind == "predicate":
            atom_queries = _required(batch, "atom_queries")
            latent_state = flatten_temporal_latent_state(
                _maybe_detach_latent_state(
                    latent_time_slice(rollout.observed_states, 1, rollout.observed_states.graph_latent.size(1)),
                    self.config.goal_head_detach,
                )
            )
            return self.goal_loss_module(_flatten_time_queries(atom_queries), latent_state)

        terminal_latent = self.jepa.encode(_required(batch, "terminal_state"))
        terminal_latent = _maybe_detach_latent_state(terminal_latent, self.config.goal_head_detach)
        goal_tensors = _required(batch, "goal")
        if kind in {"gaussian", "gmm"}:
            return self.goal_head.negative_log_likelihood(goal_tensors, terminal_latent).mean()
        if kind == "conditional_sampler":
            return self.goal_loss_module(goal_tensors, terminal_latent)
        raise ValueError(f"Unknown goal_head_kind: {kind}")

    def _applicability_loss(self, batch: dict[str, object]) -> ApplicabilityLossOutput | None:
        if self.config.applicability_loss_weight == 0:
            return None
        graph_latents = _required_tensor(batch, "applicability_graph_latents")
        action_latents = _required_tensor(batch, "applicability_action_latents")
        labels = _required_tensor(batch, "applicability_labels")
        object_latents = _optional_tensor(batch, "applicability_object_latents")
        argument_mask = _optional_tensor(batch, "applicability_argument_mask")
        example_mask = _optional_tensor(batch, "applicability_example_mask")
        if self.config.applicability_head_detach:
            graph_latents = graph_latents.detach()
            action_latents = action_latents.detach()
            if object_latents is not None:
                object_latents = object_latents.detach()
        if self.applicability_head is None or self.applicability_loss_module is None:
            raise RuntimeError("applicability modules were not initialized")
        logits = self.applicability_head(graph_latents, action_latents, object_latents, argument_mask)
        return self.applicability_loss_module(logits, labels, example_mask=example_mask)

    def _build_goal_loss_module(self, goal_loss_module: nn.Module | None) -> nn.Module | None:
        kind = self.config.goal_head_kind
        if kind == "none":
            return goal_loss_module
        if goal_loss_module is not None:
            return goal_loss_module
        if kind == "predicate":
            if self.goal_head is None:
                return None
            return PredicateEvaluatorSampleLoss(self.goal_head)
        if kind == "conditional_sampler":
            if self.goal_head is None:
                return None
            return ConditionalSampleGeneratorLoss(self.goal_head)
        return None

    def _validate_config(self) -> None:
        if self.config.jepa_loss_weight < 0:
            raise ValueError("jepa_loss_weight must be non-negative")
        if self.config.goal_loss_weight < 0:
            raise ValueError("goal_loss_weight must be non-negative")
        if self.config.applicability_loss_weight < 0:
            raise ValueError("applicability_loss_weight must be non-negative")
        if self.config.grad_clip_norm is not None and self.config.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.config.applicability_loss_weight > 0 and (
            self.applicability_head is None or self.applicability_loss_module is None
        ):
            raise ValueError(
                "applicability_head and applicability_loss_module are required when applicability_loss_weight > 0"
            )
        if self.config.goal_head_kind == "none":
            if self.goal_head is not None or self.goal_loss_module is not None:
                raise ValueError("goal_head and goal_loss_module require a non-'none' goal_head_kind")
            return
        if self.goal_head is None:
            raise ValueError(f"goal_head is required for goal_head_kind={self.config.goal_head_kind!r}")
        if self.config.goal_head_kind == "predicate" and not isinstance(
            self.goal_loss_module, PredicateEvaluatorSampleLoss
        ):
            raise ValueError("predicate goal_head_kind requires PredicateEvaluatorSampleLoss")
        if self.config.goal_head_kind == "conditional_sampler" and not isinstance(
            self.goal_loss_module,
            ConditionalSampleGeneratorLoss,
        ):
            raise ValueError("conditional_sampler goal_head_kind requires ConditionalSampleGeneratorLoss")
        if self.config.goal_head_kind in {"gaussian", "gmm"} and not hasattr(self.goal_head, "negative_log_likelihood"):
            raise ValueError(f"{self.config.goal_head_kind} goal head must define negative_log_likelihood")


def _maybe_detach_latent_state(latent_state: JEPALatentState, detach: bool) -> JEPALatentState:
    if not detach:
        return latent_state
    return JEPALatentState(
        graph_latent=latent_state.graph_latent.detach(),
        object_latents=latent_state.object_latents.detach(),
        object_ids=latent_state.object_ids,
        object_batch=latent_state.object_batch,
    )


def _required(batch: dict[str, object], key: str):
    if key not in batch:
        raise KeyError(f"Batch is missing required key: {key}")
    return batch[key]


def _required_tensor(batch: dict[str, object], key: str) -> Tensor:
    value = _required(batch, key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Batch key {key!r} must be a Tensor")
    return value


def _optional_tensor(batch: dict[str, object], key: str) -> Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError(f"Batch key {key!r} must be a Tensor")
    return value


def _add_applicability_terms(terms: dict[str, Tensor], output: ApplicabilityLossOutput) -> None:
    terms["applicability"] = output.total.detach()
    terms["applicability_bce"] = output.bce.detach()
    if output.positive_logit_mean is not None:
        terms["applicability_positive_logit_mean"] = output.positive_logit_mean.detach()
    if output.negative_logit_mean is not None:
        terms["applicability_negative_logit_mean"] = output.negative_logit_mean.detach()
    if output.positive_negative_margin is not None:
        terms["applicability_positive_negative_margin"] = output.positive_negative_margin.detach()


def _flatten_time_queries(atom_queries: dict[str, Tensor]) -> dict[str, Tensor]:
    flattened = {}
    for key, value in atom_queries.items():
        if value.ndim < 3:
            raise ValueError(f"Atom query {key} must include batch and time axes, got shape {tuple(value.shape)}")
        flattened[key] = value.reshape(value.size(0) * value.size(1), *value.shape[2:])
    return flattened


def _unique_trainable_parameters(*modules: nn.Module | None) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            parameters.append(parameter)
            seen.add(id(parameter))
    return parameters
