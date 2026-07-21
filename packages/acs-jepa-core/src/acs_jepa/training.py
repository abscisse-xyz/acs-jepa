"""Training helpers for graph-native JEPA models and goal heads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from acs_jepa.architectures import JEPALatentState, flatten_temporal_latent_state, latent_time_slice
from acs_jepa.goals import ConditionalSampleGeneratorLoss, PredicateEvaluatorSampleLoss
from acs_jepa.jepa import GraphJEPA, GraphJEPATrainingOutput
from acs_jepa.losses import (
    ActionContrastiveLossOutput,
    ActionVICRegLossOutput,
    ApplicabilityLossOutput,
    ArgumentReconstructionLossOutput,
)

GoalHeadKind = Literal["none", "predicate", "gaussian", "gmm", "conditional_sampler"]


@dataclass(frozen=True)
class JepaTrainerConfig:
    """Configuration for trajectory JEPA training with one goal head."""

    goal_head_kind: GoalHeadKind = "none"
    jepa_loss_weight: float = 1.0
    goal_loss_weight: float = 1.0
    goal_head_detach: bool = True
    applicability_loss_weight: float = 0.0
    integrated_applicability_loss_weight: float = 0.0
    applicability_head_detach: bool = True
    action_vicreg_loss_weight: float = 0.0
    action_contrastive_loss_weight: float = 0.0
    argument_reconstruction_loss_weight: float = 0.0
    grad_clip_norm: float | None = None


@dataclass(frozen=True)
class JepaTrainerStepOutput:
    """Detached scalar losses and rollout output from a trainer step."""

    total_loss: Tensor
    jepa_loss: Tensor
    goal_loss: Tensor | None
    action_vicreg_loss: Tensor | None
    action_contrastive_loss: Tensor | None
    argument_reconstruction_loss: Tensor | None
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
        action_vicreg_loss_module: nn.Module | None = None,
        action_contrastive_loss_module: nn.Module | None = None,
        action_contrastive_anchor: nn.Module | None = None,
        argument_reconstruction_head: nn.Module | None = None,
        argument_reconstruction_loss_module: nn.Module | None = None,
    ) -> None:
        self.jepa = jepa
        self.optimizer = optimizer
        self.config = JepaTrainerConfig() if config is None else config
        self.goal_head = goal_head
        self.goal_loss_module = self._build_goal_loss_module(goal_loss_module)
        self.applicability_head = applicability_head
        self.applicability_loss_module = applicability_loss_module
        self.action_vicreg_loss_module = action_vicreg_loss_module
        self.action_contrastive_loss_module = action_contrastive_loss_module
        self.action_contrastive_anchor = action_contrastive_anchor
        self.argument_reconstruction_head = argument_reconstruction_head
        self.argument_reconstruction_loss_module = argument_reconstruction_loss_module
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
        if self.action_vicreg_loss_module is not None:
            self.action_vicreg_loss_module.train()
        for module in (
            self.action_contrastive_loss_module,
            self.action_contrastive_anchor,
            self.argument_reconstruction_head,
            self.argument_reconstruction_loss_module,
        ):
            if module is not None:
                module.train()

        self.optimizer.zero_grad(set_to_none=True)
        rollout = self.jepa.trajectory_rollout(
            _required(batch, "states"),
            _required(batch, "actions"),
        )
        jepa_loss = rollout.loss.total
        goal_loss = self._goal_loss(batch, rollout)
        applicability_loss = self._applicability_loss(batch, rollout)
        action_vicreg_loss = self._action_vicreg_loss(rollout)
        action_contrastive_loss = self._action_contrastive_loss(batch, rollout)
        argument_reconstruction_loss = self._argument_reconstruction_loss(batch, rollout)
        total_loss = self._weighted_total(
            jepa_loss,
            goal_loss,
            applicability_loss,
            action_vicreg_loss,
            action_contrastive_loss,
            argument_reconstruction_loss,
        )

        total_loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                _unique_trainable_parameters(
                    self.jepa,
                    self.goal_head,
                    self.goal_loss_module,
                    self.applicability_head,
                    self.applicability_loss_module,
                    self.action_contrastive_anchor,
                    self.argument_reconstruction_head,
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
        if action_vicreg_loss is not None:
            _add_action_vicreg_terms(terms, action_vicreg_loss)
        if action_contrastive_loss is not None:
            _add_action_contrastive_terms(terms, action_contrastive_loss)
        if argument_reconstruction_loss is not None:
            _add_argument_reconstruction_terms(terms, argument_reconstruction_loss)
        return JepaTrainerStepOutput(
            total_loss=total_loss.detach(),
            jepa_loss=jepa_loss.detach(),
            goal_loss=None if goal_loss is None else goal_loss.detach(),
            action_vicreg_loss=(
                None if action_vicreg_loss is None else action_vicreg_loss.total.detach()
            ),
            action_contrastive_loss=_detached_total(action_contrastive_loss),
            argument_reconstruction_loss=_detached_total(argument_reconstruction_loss),
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
        if self.action_vicreg_loss_module is not None:
            self.action_vicreg_loss_module.eval()
        for module in (
            self.action_contrastive_loss_module,
            self.action_contrastive_anchor,
            self.argument_reconstruction_head,
            self.argument_reconstruction_loss_module,
        ):
            if module is not None:
                module.eval()

        with torch.no_grad():
            rollout = self.jepa.trajectory_rollout(
                _required(batch, "states"),
                _required(batch, "actions"),
            )
            jepa_loss = rollout.loss.total
            goal_loss = self._goal_loss(batch, rollout)
            applicability_loss = self._applicability_loss(batch, rollout)
            action_vicreg_loss = self._action_vicreg_loss(rollout)
            action_contrastive_loss = self._action_contrastive_loss(batch, rollout)
            argument_reconstruction_loss = self._argument_reconstruction_loss(batch, rollout)
            total_loss = self._weighted_total(
                jepa_loss,
                goal_loss,
                applicability_loss,
                action_vicreg_loss,
                action_contrastive_loss,
                argument_reconstruction_loss,
            )

        terms = {name: value.detach() for name, value in rollout.loss.terms.items()}
        terms["trainer_total"] = total_loss.detach()
        if goal_loss is not None:
            terms["goal"] = goal_loss.detach()
        if applicability_loss is not None:
            _add_applicability_terms(terms, applicability_loss)
        if action_vicreg_loss is not None:
            _add_action_vicreg_terms(terms, action_vicreg_loss)
        if action_contrastive_loss is not None:
            _add_action_contrastive_terms(terms, action_contrastive_loss)
        if argument_reconstruction_loss is not None:
            _add_argument_reconstruction_terms(terms, argument_reconstruction_loss)
        return JepaTrainerStepOutput(
            total_loss=total_loss.detach(),
            jepa_loss=jepa_loss.detach(),
            goal_loss=None if goal_loss is None else goal_loss.detach(),
            action_vicreg_loss=(
                None if action_vicreg_loss is None else action_vicreg_loss.total.detach()
            ),
            action_contrastive_loss=_detached_total(action_contrastive_loss),
            argument_reconstruction_loss=_detached_total(argument_reconstruction_loss),
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

    def _action_vicreg_loss(
        self, rollout: GraphJEPATrainingOutput
    ) -> ActionVICRegLossOutput | None:
        if self.config.action_vicreg_loss_weight == 0:
            return None
        if self.action_vicreg_loss_module is None:
            raise RuntimeError("action VICReg loss module was not initialized")
        return self.action_vicreg_loss_module(rollout.action_latents)

    def _action_contrastive_loss(
        self, batch: dict[str, object], rollout: GraphJEPATrainingOutput
    ) -> ActionContrastiveLossOutput | None:
        if self.config.action_contrastive_loss_weight == 0:
            return None
        supervision = _required_mapping(batch, "action_supervision")
        negatives, negative_mask = _encode_causal_negative_actions(
            self.jepa,
            _required_mapping(batch, "actions"),
            rollout.observed_states,
            rollout.action_latents,
            supervision,
        )
        active_rows = negative_mask.any(dim=2)
        if not bool(active_rows.any().item()):
            return None
        if self.action_contrastive_anchor is None or self.action_contrastive_loss_module is None:
            raise RuntimeError("action contrastive modules were not initialized")
        num_steps = rollout.action_latents.size(1)
        anchors = self.action_contrastive_anchor(
            latent_time_slice(rollout.observed_states, 0, num_steps),
            latent_time_slice(rollout.observed_states, 1, num_steps + 1),
        )
        return self.action_contrastive_loss_module(
            anchors[active_rows],
            rollout.action_latents[active_rows],
            negatives[active_rows],
            negative_mask[active_rows],
        )

    def _argument_reconstruction_loss(
        self, batch: dict[str, object], rollout: GraphJEPATrainingOutput
    ) -> ArgumentReconstructionLossOutput | None:
        if self.config.argument_reconstruction_loss_weight == 0:
            return None
        if self.argument_reconstruction_head is None or self.argument_reconstruction_loss_module is None:
            raise RuntimeError("argument reconstruction modules were not initialized")
        supervision = _required_mapping(batch, "action_supervision")
        object_bank = _dense_source_object_bank(rollout, supervision)
        targets = _supervision_tensor(supervision, "argument_target_indices", torch.long)
        argument_mask = _supervision_tensor(supervision, "argument_mask", torch.bool)
        candidate_mask = _supervision_tensor(supervision, "argument_candidate_mask", torch.bool)
        batch_size, num_steps, num_objects, latent_dim = object_bank.shape
        if targets.shape != argument_mask.shape or targets.shape[:2] != (batch_size, num_steps):
            raise ValueError("argument targets and mask must have shape [B, K, R]")
        if candidate_mask.shape != (*targets.shape, num_objects):
            raise ValueError("argument_candidate_mask must have shape [B, K, R, O]")
        object_mask = _supervision_tensor(supervision, "object_mask", torch.bool)
        if bool((candidate_mask & ~object_mask.unsqueeze(2)).any().item()):
            raise ValueError("argument candidates must identify represented objects")
        if bool(candidate_mask[~argument_mask].any().item()):
            raise ValueError("inactive argument roles must have no candidates")
        logits = self.argument_reconstruction_head(
            rollout.action_latents.reshape(-1, rollout.action_latents.size(-1)),
            object_bank.reshape(-1, num_objects, latent_dim),
            candidate_mask.reshape(-1, targets.size(2), num_objects),
        )
        return self.argument_reconstruction_loss_module(
            logits,
            targets.reshape(-1, targets.size(2)),
            argument_mask.reshape(-1, targets.size(2)),
            candidate_mask.reshape(-1, targets.size(2), num_objects),
        )

    def _applicability_loss(
        self, batch: dict[str, object], rollout: GraphJEPATrainingOutput
    ) -> ApplicabilityLossOutput | None:
        if self.config.integrated_applicability_loss_weight > 0:
            return self._integrated_applicability_loss(batch, rollout)
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

    def _integrated_applicability_loss(
        self, batch: dict[str, object], rollout: GraphJEPATrainingOutput
    ) -> ApplicabilityLossOutput | None:
        supervision = _required_mapping(batch, "action_supervision")
        negative_mask = _supervision_tensor(supervision, "negative_mask", torch.bool)
        label_mask = _supervision_tensor(
            supervision, "negative_applicability_label_mask", torch.bool
        )
        labels = _supervision_tensor(
            supervision, "negative_applicability_label", torch.float32
        )
        if label_mask.shape != negative_mask.shape or labels.shape != negative_mask.shape:
            raise ValueError("negative applicability tensors must have shape [B, K, M]")
        if bool((label_mask & ~negative_mask).any().item()):
            raise ValueError("known applicability labels must imply negative_mask")
        known = label_mask & negative_mask
        selected = known.any(dim=2)
        if not bool(selected.any().item()):
            return None
        if self.applicability_head is None or self.applicability_loss_module is None:
            raise RuntimeError("applicability modules were not initialized")
        negatives, _ = _encode_causal_negative_actions(
            self.jepa,
            _required_mapping(batch, "actions"),
            rollout.observed_states,
            rollout.action_latents,
            supervision,
        )
        object_bank = _dense_source_object_bank(rollout, supervision)
        positive_indices = _supervision_tensor(
            supervision, "argument_target_indices", torch.long
        )
        positive_mask = _supervision_tensor(supervision, "argument_mask", torch.bool)
        negative_indices = _supervision_tensor(
            supervision, "negative_action_object_indices", torch.long
        )
        negative_argument_mask = _supervision_tensor(
            supervision, "negative_action_arg_mask", torch.bool
        )
        effective_negative_argument_mask = negative_argument_mask & known.unsqueeze(-1)
        object_mask = _supervision_tensor(supervision, "object_mask", torch.bool)
        positive_objects = _gather_object_latents(
            object_bank, positive_indices, positive_mask, object_mask
        )
        negative_objects = _gather_object_latents(
            object_bank,
            negative_indices,
            effective_negative_argument_mask,
            object_mask,
        )
        source_graphs = rollout.observed_states.graph_latent[
            :, : rollout.action_latents.size(1)
        ]
        graph_inputs = torch.cat(
            [
                source_graphs[selected],
                source_graphs.unsqueeze(2)
                .expand(*known.shape, source_graphs.size(-1))[known],
            ],
            dim=0,
        )
        action_inputs = torch.cat(
            [rollout.action_latents[selected], negatives[known]], dim=0
        )
        object_inputs = torch.cat(
            [positive_objects[selected], negative_objects[known]], dim=0
        )
        argument_masks = torch.cat(
            [positive_mask[selected], effective_negative_argument_mask[known]], dim=0
        )
        applicability_labels = torch.cat(
            [labels.new_ones(int(selected.sum().item())), labels[known]], dim=0
        )
        if self.config.applicability_head_detach:
            graph_inputs = graph_inputs.detach()
            action_inputs = action_inputs.detach()
            object_inputs = object_inputs.detach()
        logits = self.applicability_head(
            graph_inputs, action_inputs, object_inputs, argument_masks
        )
        return self.applicability_loss_module(logits, applicability_labels)

    def _weighted_total(
        self,
        jepa_loss: Tensor,
        goal_loss: Tensor | None,
        applicability_loss: ApplicabilityLossOutput | None,
        action_vicreg_loss: ActionVICRegLossOutput | None,
        action_contrastive_loss: ActionContrastiveLossOutput | None,
        argument_reconstruction_loss: ArgumentReconstructionLossOutput | None,
    ) -> Tensor:
        total = self.config.jepa_loss_weight * jepa_loss
        if goal_loss is not None:
            total = total + self.config.goal_loss_weight * goal_loss
        if applicability_loss is not None:
            applicability_weight = (
                self.config.integrated_applicability_loss_weight
                if self.config.integrated_applicability_loss_weight > 0
                else self.config.applicability_loss_weight
            )
            total = total + applicability_weight * applicability_loss.total
        if action_vicreg_loss is not None:
            total = total + self.config.action_vicreg_loss_weight * action_vicreg_loss.total
        if action_contrastive_loss is not None:
            total = total + self.config.action_contrastive_loss_weight * action_contrastive_loss.total
        if argument_reconstruction_loss is not None:
            total = total + self.config.argument_reconstruction_loss_weight * argument_reconstruction_loss.total
        return total

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
        for name in (
            "jepa_loss_weight",
            "goal_loss_weight",
            "applicability_loss_weight",
            "integrated_applicability_loss_weight",
            "action_vicreg_loss_weight",
            "action_contrastive_loss_weight",
            "argument_reconstruction_loss_weight",
        ):
            if not math.isfinite(getattr(self.config, name)):
                raise ValueError(f"{name} must be finite")
        if self.config.jepa_loss_weight < 0:
            raise ValueError("jepa_loss_weight must be non-negative")
        if self.config.goal_loss_weight < 0:
            raise ValueError("goal_loss_weight must be non-negative")
        if self.config.applicability_loss_weight < 0:
            raise ValueError("applicability_loss_weight must be non-negative")
        if self.config.integrated_applicability_loss_weight < 0:
            raise ValueError("integrated_applicability_loss_weight must be non-negative")
        if self.config.action_vicreg_loss_weight < 0:
            raise ValueError("action_vicreg_loss_weight must be non-negative")
        if self.config.action_contrastive_loss_weight < 0:
            raise ValueError("action_contrastive_loss_weight must be non-negative")
        if self.config.argument_reconstruction_loss_weight < 0:
            raise ValueError("argument_reconstruction_loss_weight must be non-negative")
        if (
            self.config.applicability_loss_weight > 0
            and self.config.integrated_applicability_loss_weight > 0
        ):
            raise ValueError("legacy and integrated applicability losses are mutually exclusive")
        if self.config.grad_clip_norm is not None and (
            not math.isfinite(self.config.grad_clip_norm) or self.config.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be finite and positive")
        if self.config.action_vicreg_loss_weight > 0 and self.action_vicreg_loss_module is None:
            raise ValueError("action_vicreg_loss_module is required when action_vicreg_loss_weight > 0")
        if self.config.action_contrastive_loss_weight > 0 and (
            self.action_contrastive_anchor is None or self.action_contrastive_loss_module is None
        ):
            raise ValueError(
                "action_contrastive_anchor and action_contrastive_loss_module are required "
                "when action_contrastive_loss_weight > 0"
            )
        if self.config.argument_reconstruction_loss_weight > 0 and (
            self.argument_reconstruction_head is None
            or self.argument_reconstruction_loss_module is None
        ):
            raise ValueError(
                "argument_reconstruction_head and argument_reconstruction_loss_module are required "
                "when argument_reconstruction_loss_weight > 0"
            )
        if self.config.integrated_applicability_loss_weight > 0 and (
            self.applicability_head is None or self.applicability_loss_module is None
        ):
            raise ValueError(
                "applicability_head and applicability_loss_module are required when "
                "integrated_applicability_loss_weight > 0"
            )
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


def _required_mapping(batch: Mapping[str, object], key: str) -> Mapping[str, Tensor]:
    value = batch.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Batch key {key!r} must be a mapping")
    return value


def _optional_tensor(batch: dict[str, object], key: str) -> Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError(f"Batch key {key!r} must be a Tensor")
    return value


def _add_action_vicreg_terms(
    terms: dict[str, Tensor], output: ActionVICRegLossOutput
) -> None:
    terms["action_vicreg"] = output.total.detach()
    terms["action_vicreg_std"] = output.std_penalty.detach()
    terms["action_vicreg_covariance"] = output.covariance_penalty.detach()
    terms["action_vicreg_num_samples"] = output.total.detach().new_tensor(
        float(output.num_samples)
    )


def _add_action_contrastive_terms(
    terms: dict[str, Tensor], output: ActionContrastiveLossOutput
) -> None:
    terms["action_contrastive"] = output.total.detach()
    terms["action_contrastive_positive_similarity"] = output.positive_similarity_mean.detach()
    terms["action_contrastive_hardest_negative_similarity"] = (
        output.hardest_negative_similarity_mean.detach()
    )
    terms["action_contrastive_margin"] = output.positive_negative_margin.detach()
    terms["action_contrastive_top1_accuracy"] = output.top1_accuracy.detach()
    terms["action_contrastive_num_examples"] = output.total.detach().new_tensor(
        float(output.num_examples)
    )
    terms["action_contrastive_num_negatives"] = output.total.detach().new_tensor(
        float(output.num_negatives)
    )


def _add_argument_reconstruction_terms(
    terms: dict[str, Tensor], output: ArgumentReconstructionLossOutput
) -> None:
    terms["argument_reconstruction"] = output.total.detach()
    terms["argument_role_accuracy"] = output.role_accuracy.detach()
    terms["argument_competitive_role_accuracy"] = output.competitive_role_accuracy.detach()
    terms["argument_mean_target_margin"] = output.mean_target_margin.detach()
    terms["argument_num_active_roles"] = output.total.detach().new_tensor(
        float(output.num_active_roles)
    )
    terms["argument_num_competitive_roles"] = output.total.detach().new_tensor(
        float(output.num_competitive_roles)
    )


def _detached_total(
    output: ActionContrastiveLossOutput | ArgumentReconstructionLossOutput | None,
) -> Tensor | None:
    return None if output is None else output.total.detach()


def _add_applicability_terms(terms: dict[str, Tensor], output: ApplicabilityLossOutput) -> None:
    terms["applicability"] = output.total.detach()
    terms["applicability_bce"] = output.bce.detach()
    if output.positive_logit_mean is not None:
        terms["applicability_positive_logit_mean"] = output.positive_logit_mean.detach()
    if output.negative_logit_mean is not None:
        terms["applicability_negative_logit_mean"] = output.negative_logit_mean.detach()
    if output.positive_negative_margin is not None:
        terms["applicability_positive_negative_margin"] = output.positive_negative_margin.detach()
    terms["applicability_num_examples"] = output.total.detach().new_tensor(
        float(output.num_examples)
    )
    terms["applicability_num_positive"] = output.total.detach().new_tensor(
        float(output.num_positive)
    )
    terms["applicability_num_negative"] = output.total.detach().new_tensor(
        float(output.num_negative)
    )


def _dense_source_object_bank(
    rollout: GraphJEPATrainingOutput, supervision: Mapping[str, Tensor]
) -> Tensor:
    object_mask = _supervision_tensor(supervision, "object_mask", torch.bool)
    batch_size, num_steps = rollout.action_latents.shape[:2]
    if object_mask.ndim != 3 or object_mask.shape[:2] != (batch_size, num_steps):
        raise ValueError("object_mask must have shape [B, K, O]")
    num_objects = object_mask.size(2)
    observed = rollout.observed_states
    if observed.object_latents.ndim != 3 or observed.object_latents.size(1) < num_steps:
        raise ValueError("observed object latents must include every source timestep")
    bank = observed.object_latents.new_zeros(
        (batch_size, num_steps, num_objects, observed.object_latents.size(-1))
    )
    represented = torch.zeros_like(object_mask)
    for row in range(observed.object_latents.size(0)):
        graph_index = int(observed.object_batch[row].item())
        object_index = int(observed.object_ids[row].item())
        if not 0 <= graph_index < batch_size or not 0 <= object_index < num_objects:
            raise ValueError("packed object graph/id lies outside the dense object bank")
        if bool(represented[graph_index, 0, object_index].item()):
            raise ValueError("packed object ids must be unique within each problem")
        bank[graph_index, :, object_index] = observed.object_latents[row, :num_steps]
        represented[graph_index, :, object_index] = True
    if not torch.equal(represented, object_mask):
        raise ValueError("object_mask must exactly match represented problem-local object ids")
    return bank


def _gather_object_latents(
    object_bank: Tensor,
    object_indices: Tensor,
    argument_mask: Tensor,
    object_mask: Tensor,
) -> Tensor:
    if object_indices.shape != argument_mask.shape or object_indices.shape[:2] != object_bank.shape[:2]:
        raise ValueError("object indices and argument mask must share leading [B, K] axes")
    if object_indices.dtype != torch.long or argument_mask.dtype != torch.bool:
        raise ValueError("object indices must be long and argument mask must be bool")
    if object_mask.shape != object_bank.shape[:3] or object_mask.dtype != torch.bool:
        raise ValueError("object_mask must have shape [B, K, O] and dtype bool")
    if any(
        value.device != object_bank.device
        for value in (object_indices, argument_mask, object_mask)
    ):
        raise ValueError("object gather tensors must share the latent device")
    active_indices = object_indices[argument_mask]
    if bool(((active_indices < 0) | (active_indices >= object_bank.size(2))).any().item()):
        raise ValueError("active argument object indices must lie in [0, O)")
    query_shape = object_indices.shape[2:]
    flat_count = 1
    for size in query_shape:
        flat_count *= size
    safe_indices = object_indices.masked_fill(~argument_mask, 0).reshape(
        object_bank.size(0) * object_bank.size(1), flat_count
    )
    represented = object_mask.reshape(-1, object_bank.size(2)).gather(1, safe_indices)
    if bool((argument_mask.reshape_as(represented) & ~represented).any().item()):
        raise ValueError("active argument object indices must identify a represented object")
    flat_bank = object_bank.reshape(-1, object_bank.size(2), object_bank.size(3))
    gathered = flat_bank.gather(
        1, safe_indices.unsqueeze(-1).expand(-1, -1, object_bank.size(3))
    )
    gathered = gathered.reshape(*object_indices.shape, object_bank.size(3))
    return gathered * argument_mask.unsqueeze(-1)


def _encode_causal_negative_actions(
    jepa: GraphJEPA,
    actions: Mapping[str, Tensor],
    observed_states: JEPALatentState,
    positive_action_latents: Tensor,
    supervision: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Encode only active negatives against each transition's true causal prefix."""

    if positive_action_latents.ndim != 3:
        raise ValueError("positive_action_latents must have shape [B, K, D_a]")
    batch_size, num_steps, action_dim = positive_action_latents.shape
    negative_mask = _supervision_tensor(supervision, "negative_mask", torch.bool)
    if negative_mask.ndim != 3 or negative_mask.shape[:2] != (batch_size, num_steps):
        raise ValueError("negative_mask must have shape [B, K, M]")
    if negative_mask.device != positive_action_latents.device:
        raise ValueError("action supervision tensors must be on the rollout device")
    num_negatives = negative_mask.size(2)
    fields = {
        "action_id": _supervision_tensor(supervision, "negative_action_id", torch.long),
        "action_object_indices": _supervision_tensor(
            supervision, "negative_action_object_indices", torch.long
        ),
        "action_role_ids": _supervision_tensor(
            supervision, "negative_action_role_ids", torch.long
        ),
        "action_arg_mask": _supervision_tensor(
            supervision, "negative_action_arg_mask", torch.bool
        ),
    }
    if fields["action_id"].shape != (batch_size, num_steps, num_negatives):
        raise ValueError("negative_action_id must have shape [B, K, M]")
    role_shape = fields["action_object_indices"].shape
    if len(role_shape) != 4 or role_shape[:3] != (batch_size, num_steps, num_negatives):
        raise ValueError("negative action argument tensors must have shape [B, K, M, R]")
    if fields["action_role_ids"].shape != role_shape or fields["action_arg_mask"].shape != role_shape:
        raise ValueError("negative action argument tensor shapes must match")
    if any(value.device != positive_action_latents.device for value in fields.values()):
        raise ValueError("action supervision tensors must be on the rollout device")

    encoded = positive_action_latents.unsqueeze(2).expand(
        batch_size, num_steps, num_negatives, action_dim
    ).clone()
    for step in range(num_steps):
        active = negative_mask[:, step].nonzero(as_tuple=False)
        if active.numel() == 0:
            continue
        source_batches = active[:, 0]
        candidate_actions = {
            name: value.index_select(0, source_batches)[:, : step + 1].clone()
            for name, value in actions.items()
        }
        for action_name, negative_values in fields.items():
            candidate_actions[action_name][:, step] = negative_values[
                source_batches, step, active[:, 1]
            ]

        object_latents: list[Tensor] = []
        object_ids: list[Tensor] = []
        object_batches: list[Tensor] = []
        for candidate_index, source_batch in enumerate(source_batches.tolist()):
            rows = observed_states.object_batch == source_batch
            object_latents.append(observed_states.object_latents[rows, : step + 1])
            object_ids.append(observed_states.object_ids[rows])
            object_batches.append(
                observed_states.object_batch.new_full(
                    (int(rows.sum().item()),), candidate_index
                )
            )
        candidate_states = JEPALatentState(
            graph_latent=observed_states.graph_latent.index_select(0, source_batches)[
                :, : step + 1
            ],
            object_latents=torch.cat(object_latents),
            object_ids=torch.cat(object_ids),
            object_batch=torch.cat(object_batches),
        )
        final_latents = jepa.action_encoder(candidate_actions, candidate_states)[:, -1]
        encoded[source_batches, step, active[:, 1]] = final_latents
    return encoded, negative_mask


def _supervision_tensor(
    supervision: Mapping[str, Tensor], key: str, dtype: torch.dtype
) -> Tensor:
    if key not in supervision:
        raise KeyError(f"action_supervision is missing required key {key!r}")
    value = supervision[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"action_supervision[{key!r}] must be a Tensor")
    if value.dtype != dtype:
        raise ValueError(f"action_supervision[{key!r}] must have dtype {dtype}")
    return value


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
