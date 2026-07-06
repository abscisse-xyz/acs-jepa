"""Latent-space goal scoring modules for graph-native JEPA models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from acs_jepa.architectures import (
    JEPALatentState,
    _as_batch_matrix,
    _as_batch_vector,
    _build_argument_encoder,
    _gather_by_object_id,
    _mlp,
)


@dataclass(frozen=True)
class GaussianTerminalLatentParams:
    """Diagonal Gaussian parameters aligned to a terminal latent state."""

    graph_mean: Tensor
    graph_logvar: Tensor
    object_mean: Tensor
    object_logvar: Tensor


@dataclass(frozen=True)
class GaussianMixtureTerminalLatentParams:
    """Diagonal Gaussian mixture parameters aligned to a terminal latent state."""

    mixture_logits: Tensor
    graph_mean: Tensor
    graph_logvar: Tensor
    object_mean: Tensor
    object_logvar: Tensor


@dataclass(frozen=True)
class GeneratedTerminalLatentSamples:
    """Sampled terminal latent completions aligned to a reference state."""

    graph_latents: Tensor
    object_latents: Tensor
    object_ids: Tensor
    object_batch: Tensor


class PredicateEvaluator(nn.Module):
    """Evaluate grounded predicates over JEPA latents.

    This module implements the learned predicate evaluator ``q_psi``. Unlike
    ``ActionEncoder``, it gathers predicate arguments from
    ``JEPALatentState.object_latents`` so it can score predicted future states
    during planning.
    """

    def __init__(
        self,
        *,
        num_predicates: int,
        max_predicate_arity: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        argument_encoder: str = "pooled",
    ) -> None:
        super().__init__()
        hidden_dim = latent_dim if hidden_dim is None else hidden_dim
        self.predicate_embedding = nn.Embedding(num_predicates, latent_dim)
        self.argument_encoder = _build_argument_encoder(argument_encoder, latent_dim, max_predicate_arity, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, predicate_tensors: dict[str, Tensor], latent_state: JEPALatentState) -> Tensor:
        """Score predicate truth for each graph in a batch."""

        batch_size = latent_state.graph_latent.size(0)
        predicate_id = _as_batch_vector(predicate_tensors["predicate_id"], batch_size).long()
        object_indices = _as_batch_matrix(predicate_tensors["predicate_object_indices"], batch_size).long()
        role_ids = _as_batch_matrix(predicate_tensors["predicate_role_ids"], batch_size).long()
        arg_mask = _as_batch_matrix(predicate_tensors["predicate_arg_mask"], batch_size).bool()

        arg_latents = _gather_by_object_id(
            embeddings=latent_state.object_latents,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
            query_object_ids=object_indices,
            query_mask=arg_mask,
        )
        predicate_features = self.predicate_embedding(predicate_id)
        predicate_context = self.argument_encoder(predicate_features, arg_latents, role_ids, arg_mask)
        features = torch.cat([latent_state.graph_latent, predicate_context, predicate_features], dim=-1)
        return self.output(features).squeeze(-1)


class PredicateEvaluatorSampleLoss(nn.Module):
    """Train ``PredicateEvaluator`` from padded positive/negative atom samples."""

    def __init__(
        self,
        evaluator: PredicateEvaluator,
        *,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unknown reduction: {reduction}")
        self.evaluator = evaluator
        self.reduction = reduction

    def forward(self, atom_queries: dict[str, Tensor], latent_state: JEPALatentState) -> Tensor:
        """Return masked BCE between sampled atom truth labels and evaluator logits."""

        predicate_id = _as_atom_query_vector(atom_queries["atom_predicate_id"]).long()
        object_indices = _as_atom_query_matrix(atom_queries["atom_object_indices"]).long()
        role_ids = _as_atom_query_matrix(atom_queries["atom_role_ids"]).long()
        arg_mask = _as_atom_query_matrix(atom_queries["atom_arg_mask"]).bool()
        truth = _as_atom_query_vector(atom_queries["atom_truth"]).to(dtype=torch.float32)
        sample_mask = _as_atom_query_vector(atom_queries["atom_sample_mask"]).bool()

        if predicate_id.shape != truth.shape or predicate_id.shape != sample_mask.shape:
            raise ValueError("Atom query predicate, truth, and mask tensors must have matching shape")
        if object_indices.shape[:2] != predicate_id.shape or role_ids.shape != object_indices.shape:
            raise ValueError("Atom query argument tensors must align with atom predicate ids")
        if arg_mask.shape != object_indices.shape:
            raise ValueError("Atom query argument mask must align with object indices")

        _, num_queries = predicate_id.shape
        logits = []
        for query_idx in range(num_queries):
            query = {
                "predicate_id": predicate_id[:, query_idx].clamp_min(0),
                "predicate_object_indices": object_indices[:, query_idx],
                "predicate_role_ids": role_ids[:, query_idx],
                "predicate_arg_mask": arg_mask[:, query_idx],
            }
            logits.append(self.evaluator(query, latent_state))

        if logits:
            logit_tensor = torch.stack(logits, dim=1)
        else:
            logit_tensor = latent_state.graph_latent.new_empty((latent_state.graph_latent.size(0), 0))
        loss = F.binary_cross_entropy_with_logits(logit_tensor, truth, reduction="none")
        loss = loss * sample_mask.to(loss.dtype)
        if self.reduction == "none":
            return loss
        return loss.sum() / sample_mask.to(loss.dtype).sum().clamp_min(1.0)


class PartialGoalEncoder(nn.Module):
    """Encode padded grounded predicate goals with JEPA object-latent context."""

    def __init__(
        self,
        *,
        num_predicates: int,
        max_predicate_arity: int,
        latent_dim: int,
        goal_dim: int,
        hidden_dim: int | None = None,
        argument_encoder: str = "pooled",
    ) -> None:
        super().__init__()
        hidden_dim = goal_dim if hidden_dim is None else hidden_dim
        self.predicate_embedding = nn.Embedding(num_predicates, goal_dim)
        self.object_projector = nn.Linear(latent_dim, goal_dim)
        self.argument_encoder = _build_argument_encoder(argument_encoder, goal_dim, max_predicate_arity, hidden_dim)
        self.atom_output = _mlp(goal_dim * 2, hidden_dim, goal_dim)

    def forward(self, goal_tensors: dict[str, Tensor], latent_state: JEPALatentState) -> Tensor:
        """Return one goal context per graph as ``FloatTensor[B, goal_dim]``."""

        batch_count = latent_state.graph_latent.size(0)
        predicate_id = _match_goal_vector(_as_goal_vector(goal_tensors["goal_predicate_id"]).long(), batch_count)
        object_indices = _match_goal_matrix(
            _as_goal_matrix(goal_tensors["goal_object_indices"]).long(),
            batch_count,
        )
        role_ids = _match_goal_matrix(_as_goal_matrix(goal_tensors["goal_role_ids"]).long(), batch_count)
        arg_mask = _match_goal_matrix(_as_goal_matrix(goal_tensors["goal_arg_mask"]).bool(), batch_count)
        atom_mask = _match_goal_vector(_as_goal_vector(goal_tensors["goal_atom_mask"]).bool(), batch_count)
        default_weight = atom_mask.float()
        atom_weight = _match_goal_vector(
            _as_goal_vector(goal_tensors.get("goal_weight", default_weight)).to(dtype=torch.float32),
            batch_count,
        )

        batch_size, num_atoms = predicate_id.shape
        flat_predicate_id = predicate_id.clamp_min(0).reshape(-1)
        flat_role_ids = role_ids.reshape(batch_size * num_atoms, -1)
        flat_arg_mask = arg_mask.reshape(batch_size * num_atoms, -1)

        predicate_features = self.predicate_embedding(flat_predicate_id)
        object_features = _gather_batched_goal_object_latents(latent_state, object_indices, arg_mask)
        object_features = self.object_projector(object_features).reshape(
            batch_size * num_atoms,
            object_indices.size(-1),
            -1,
        )
        atom_context = self.argument_encoder(
            predicate_features,
            object_features,
            flat_role_ids,
            flat_arg_mask,
        )
        atom_features = self.atom_output(torch.cat([atom_context, predicate_features], dim=-1))
        atom_features = atom_features.reshape(batch_size, num_atoms, -1)

        weights = atom_mask.to(atom_features.dtype) * atom_weight.to(atom_features.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (atom_features * weights.unsqueeze(-1)).sum(dim=1) / denom


class DiagonalGaussianTerminalLatentDistributionP(nn.Module):
    """Conditional diagonal Gaussian over terminal graph/object latents."""

    def __init__(
        self,
        *,
        num_predicates: int,
        max_predicate_arity: int,
        latent_dim: int,
        goal_dim: int | None = None,
        hidden_dim: int | None = None,
        argument_encoder: str = "pooled",
        min_logvar: float = -8.0,
        max_logvar: float = 8.0,
    ) -> None:
        super().__init__()
        goal_dim = latent_dim if goal_dim is None else goal_dim
        hidden_dim = goal_dim if hidden_dim is None else hidden_dim
        self.latent_dim = latent_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.goal_encoder = PartialGoalEncoder(
            num_predicates=num_predicates,
            max_predicate_arity=max_predicate_arity,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
            argument_encoder=argument_encoder,
        )
        self.object_projector = nn.Linear(latent_dim, goal_dim)
        self.graph_params = _mlp(goal_dim, hidden_dim, latent_dim * 2)
        self.object_params = _mlp(goal_dim * 2, hidden_dim, latent_dim * 2)

    def forward(
        self,
        goal_tensors: dict[str, Tensor],
        terminal_state: JEPALatentState,
    ) -> GaussianTerminalLatentParams:
        goal_context = _match_goal_context(
            self.goal_encoder(goal_tensors, terminal_state),
            terminal_state.graph_latent.size(0),
        )
        object_context = self._object_context(goal_context, terminal_state)
        graph_mean, graph_logvar = self.graph_params(goal_context).chunk(2, dim=-1)
        object_mean, object_logvar = self.object_params(object_context).chunk(2, dim=-1)
        return GaussianTerminalLatentParams(
            graph_mean=graph_mean,
            graph_logvar=graph_logvar.clamp(self.min_logvar, self.max_logvar),
            object_mean=object_mean,
            object_logvar=object_logvar.clamp(self.min_logvar, self.max_logvar),
        )

    def negative_log_likelihood(
        self,
        goal_tensors: dict[str, Tensor],
        terminal_state: JEPALatentState,
        *,
        graph_weight: float = 1.0,
        object_weight: float = 1.0,
    ) -> Tensor:
        params = self(goal_tensors, terminal_state)
        graph_nll = _diagonal_gaussian_nll(
            terminal_state.graph_latent,
            params.graph_mean,
            params.graph_logvar,
        ).mean(dim=-1)
        object_nll = _diagonal_gaussian_nll(
            terminal_state.object_latents,
            params.object_mean,
            params.object_logvar,
        ).mean(dim=-1)
        object_nll = _batch_mean(object_nll, terminal_state.object_batch, terminal_state.graph_latent.size(0))
        return graph_weight * graph_nll + object_weight * object_nll

    def _object_context(
        self,
        goal_context: Tensor,
        terminal_state: JEPALatentState,
    ) -> Tensor:
        object_features = self.object_projector(terminal_state.object_latents)
        return torch.cat([goal_context[terminal_state.object_batch], object_features], dim=-1)


class GaussianMixtureTerminalLatentDistributionP(nn.Module):
    """Conditional diagonal Gaussian mixture over terminal graph/object latents."""

    def __init__(
        self,
        *,
        num_predicates: int,
        max_predicate_arity: int,
        latent_dim: int,
        num_components: int,
        goal_dim: int | None = None,
        hidden_dim: int | None = None,
        argument_encoder: str = "pooled",
        min_logvar: float = -8.0,
        max_logvar: float = 8.0,
    ) -> None:
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be at least 1")
        goal_dim = latent_dim if goal_dim is None else goal_dim
        hidden_dim = goal_dim if hidden_dim is None else hidden_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.goal_encoder = PartialGoalEncoder(
            num_predicates=num_predicates,
            max_predicate_arity=max_predicate_arity,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
            argument_encoder=argument_encoder,
        )
        self.object_projector = nn.Linear(latent_dim, goal_dim)
        self.mixture_logits = _mlp(goal_dim, hidden_dim, num_components)
        self.graph_params = _mlp(goal_dim, hidden_dim, num_components * latent_dim * 2)
        self.object_params = _mlp(goal_dim * 2, hidden_dim, num_components * latent_dim * 2)

    def forward(
        self,
        goal_tensors: dict[str, Tensor],
        terminal_state: JEPALatentState,
    ) -> GaussianMixtureTerminalLatentParams:
        goal_context = _match_goal_context(
            self.goal_encoder(goal_tensors, terminal_state),
            terminal_state.graph_latent.size(0),
        )
        batch_size = goal_context.size(0)
        object_context = self._object_context(goal_context, terminal_state)
        graph_params = self.graph_params(goal_context).reshape(batch_size, self.num_components, self.latent_dim * 2)
        object_params = self.object_params(object_context).reshape(-1, self.num_components, self.latent_dim * 2)
        graph_mean, graph_logvar = graph_params.chunk(2, dim=-1)
        object_mean, object_logvar = object_params.chunk(2, dim=-1)
        return GaussianMixtureTerminalLatentParams(
            mixture_logits=self.mixture_logits(goal_context),
            graph_mean=graph_mean,
            graph_logvar=graph_logvar.clamp(self.min_logvar, self.max_logvar),
            object_mean=object_mean,
            object_logvar=object_logvar.clamp(self.min_logvar, self.max_logvar),
        )

    def negative_log_likelihood(
        self,
        goal_tensors: dict[str, Tensor],
        terminal_state: JEPALatentState,
        *,
        graph_weight: float = 1.0,
        object_weight: float = 1.0,
    ) -> Tensor:
        params = self(goal_tensors, terminal_state)
        graph_nll = _diagonal_gaussian_nll(
            terminal_state.graph_latent.unsqueeze(1),
            params.graph_mean,
            params.graph_logvar,
        ).mean(dim=-1)
        object_nll = _diagonal_gaussian_nll(
            terminal_state.object_latents.unsqueeze(1),
            params.object_mean,
            params.object_logvar,
        ).mean(dim=-1)
        object_nll = _batch_mean(object_nll, terminal_state.object_batch, terminal_state.graph_latent.size(0))
        component_energy = graph_weight * graph_nll + object_weight * object_nll
        log_weights = torch.log_softmax(params.mixture_logits, dim=-1)
        return -torch.logsumexp(log_weights - component_energy, dim=-1)

    def _object_context(
        self,
        goal_context: Tensor,
        terminal_state: JEPALatentState,
    ) -> Tensor:
        object_features = self.object_projector(terminal_state.object_latents)
        return torch.cat([goal_context[terminal_state.object_batch], object_features], dim=-1)


class ConditionalSampleTerminalLatentGeneratorG(nn.Module):
    """Generate terminal latent completions conditioned on a partial goal."""

    def __init__(
        self,
        *,
        num_predicates: int,
        max_predicate_arity: int,
        latent_dim: int,
        num_samples: int,
        goal_dim: int | None = None,
        noise_dim: int | None = None,
        hidden_dim: int | None = None,
        argument_encoder: str = "pooled",
    ) -> None:
        super().__init__()
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        goal_dim = latent_dim if goal_dim is None else goal_dim
        noise_dim = latent_dim if noise_dim is None else noise_dim
        hidden_dim = goal_dim if hidden_dim is None else hidden_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.num_samples = num_samples
        self.goal_encoder = PartialGoalEncoder(
            num_predicates=num_predicates,
            max_predicate_arity=max_predicate_arity,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
            argument_encoder=argument_encoder,
        )
        self.object_projector = nn.Linear(latent_dim, goal_dim)
        self.graph_generator = _mlp(goal_dim + noise_dim, hidden_dim, latent_dim)
        self.object_generator = _mlp(goal_dim * 2 + noise_dim, hidden_dim, latent_dim)

    def forward(
        self,
        goal_tensors: dict[str, Tensor],
        reference_state: JEPALatentState,
        *,
        eps: Tensor | None = None,
        num_samples: int | None = None,
    ) -> GeneratedTerminalLatentSamples:
        goal_context = _match_goal_context(
            self.goal_encoder(goal_tensors, reference_state),
            reference_state.graph_latent.size(0),
        )
        batch_size = goal_context.size(0)
        sample_count = self.num_samples if num_samples is None else num_samples
        if eps is None:
            eps = goal_context.new_empty((batch_size, sample_count, self.noise_dim)).normal_()
        if eps.shape != (batch_size, sample_count, self.noise_dim):
            raise ValueError(
                f"eps must have shape {(batch_size, sample_count, self.noise_dim)}, got {tuple(eps.shape)}"
            )

        graph_context = goal_context.unsqueeze(1).expand(-1, sample_count, -1)
        graph_input = torch.cat([graph_context, eps], dim=-1)
        graph_latents = self.graph_generator(graph_input.reshape(batch_size * sample_count, -1))
        graph_latents = graph_latents.reshape(batch_size, sample_count, self.latent_dim)

        object_goal = goal_context[reference_state.object_batch].unsqueeze(1).expand(-1, sample_count, -1)
        object_eps = eps[reference_state.object_batch]
        object_features = self.object_projector(reference_state.object_latents)
        object_features = object_features.unsqueeze(1).expand(-1, sample_count, -1)
        object_input = torch.cat([object_goal, object_features, object_eps], dim=-1)
        object_latents = self.object_generator(object_input.reshape(-1, object_input.size(-1)))
        object_latents = object_latents.reshape(reference_state.object_latents.size(0), sample_count, self.latent_dim)
        return GeneratedTerminalLatentSamples(
            graph_latents=graph_latents,
            object_latents=object_latents,
            object_ids=reference_state.object_ids,
            object_batch=reference_state.object_batch,
        )


class DistributionalGoalEnergy(nn.Module):
    """Wrap a Gaussian-style terminal distribution as a planner energy."""

    def __init__(self, distribution: nn.Module, graph_weight: float = 1.0, object_weight: float = 1.0) -> None:
        super().__init__()
        self.distribution = distribution
        self.graph_weight = graph_weight
        self.object_weight = object_weight

    def forward(
        self,
        goal_tensors: dict[str, Tensor],
        terminal_state: JEPALatentState,
    ) -> Tensor:
        return self.distribution.negative_log_likelihood(
            goal_tensors,
            terminal_state,
            graph_weight=self.graph_weight,
            object_weight=self.object_weight,
        )


class SampleSetGoalEnergy(nn.Module):
    """Best-of-samples latent distance energy for generated goal completions."""

    def __init__(
        self,
        generator: ConditionalSampleTerminalLatentGeneratorG | None = None,
        *,
        graph_weight: float = 1.0,
        object_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.graph_weight = graph_weight
        self.object_weight = object_weight

    def forward(
        self,
        goal_tensors: dict[str, Tensor],
        terminal_state: JEPALatentState,
        *,
        eps: Tensor | None = None,
        samples: GeneratedTerminalLatentSamples | None = None,
    ) -> Tensor:
        if samples is None:
            if self.generator is None:
                raise ValueError("SampleSetGoalEnergy requires samples or a generator")
            samples = self.generator(goal_tensors, terminal_state, eps=eps)
        return _best_sample_distance(
            terminal_state,
            samples,
            graph_weight=self.graph_weight,
            object_weight=self.object_weight,
        )


class ConditionalSampleGeneratorLoss(nn.Module):
    """Train a conditional sample generator with best-of-``M`` latent distance."""

    def __init__(
        self,
        generator: ConditionalSampleTerminalLatentGeneratorG,
        *,
        graph_weight: float = 1.0,
        object_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unknown reduction: {reduction}")
        self.generator = generator
        self.energy = SampleSetGoalEnergy(generator, graph_weight=graph_weight, object_weight=object_weight)
        self.reduction = reduction

    def forward(
        self,
        goal_tensors: dict[str, Tensor],
        target_state: JEPALatentState,
        *,
        eps: Tensor | None = None,
    ) -> Tensor:
        loss = self.energy(goal_tensors, target_state, eps=eps)
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_predicate_evaluator(kind: str = "pooled", **kwargs) -> PredicateEvaluator:
    """Build a ``PredicateEvaluator`` with interchangeable argument composition."""

    return PredicateEvaluator(argument_encoder=kind, **kwargs)


def _as_atom_query_vector(value: Tensor) -> Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"Expected rank-1 or rank-2 atom query tensor, got shape {tuple(value.shape)}")
    return value


def _as_atom_query_matrix(value: Tensor) -> Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"Expected rank-2 or rank-3 atom query tensor, got shape {tuple(value.shape)}")
    return value


def _as_goal_vector(value: Tensor) -> Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"Expected rank-1 or rank-2 goal tensor, got shape {tuple(value.shape)}")
    return value


def _as_goal_matrix(value: Tensor) -> Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"Expected rank-2 or rank-3 goal tensor, got shape {tuple(value.shape)}")
    return value


def _diagonal_gaussian_nll(value: Tensor, mean: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * (logvar + (value - mean).pow(2) * torch.exp(-logvar))


def _match_goal_context(goal_context: Tensor, batch_size: int) -> Tensor:
    if goal_context.size(0) == batch_size:
        return goal_context
    if goal_context.size(0) == 1:
        return goal_context.expand(batch_size, -1)
    raise ValueError(f"Goal batch size must be 1 or {batch_size}, got {goal_context.size(0)}")


def _match_goal_vector(value: Tensor, batch_size: int) -> Tensor:
    if value.size(0) == batch_size:
        return value
    if value.size(0) == 1:
        return value.expand(batch_size, -1)
    raise ValueError(f"Goal batch size must be 1 or {batch_size}, got {value.size(0)}")


def _match_goal_matrix(value: Tensor, batch_size: int) -> Tensor:
    if value.size(0) == batch_size:
        return value
    if value.size(0) == 1:
        return value.expand(batch_size, -1, -1)
    raise ValueError(f"Goal batch size must be 1 or {batch_size}, got {value.size(0)}")


def _gather_batched_goal_object_latents(
    latent_state: JEPALatentState,
    query_object_ids: Tensor,
    query_mask: Tensor,
) -> Tensor:
    batch_size, num_atoms, arity = query_object_ids.shape
    gathered = latent_state.object_latents.new_zeros(
        (batch_size, num_atoms, arity, latent_state.object_latents.size(-1))
    )
    for atom_idx in range(num_atoms):
        gathered[:, atom_idx] = _gather_by_object_id(
            embeddings=latent_state.object_latents,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
            query_object_ids=query_object_ids[:, atom_idx],
            query_mask=query_mask[:, atom_idx],
        )
    return gathered


def _batch_mean(values: Tensor, batch: Tensor, batch_size: int) -> Tensor:
    result = values.new_zeros((batch_size, *values.shape[1:]))
    counts = values.new_zeros((batch_size, *([1] * (values.ndim - 1))))
    result.index_add_(0, batch, values)
    count_source = values.new_ones((values.size(0), *([1] * (values.ndim - 1))))
    counts.index_add_(0, batch, count_source)
    return result / counts.clamp_min(1.0)


def _best_sample_distance(
    target_state: JEPALatentState,
    samples: GeneratedTerminalLatentSamples,
    *,
    graph_weight: float,
    object_weight: float,
) -> Tensor:
    graph_distance = (target_state.graph_latent.unsqueeze(1) - samples.graph_latents).pow(2).mean(dim=-1)
    object_distance = (target_state.object_latents.unsqueeze(1) - samples.object_latents).pow(2).mean(dim=-1)
    object_distance = _batch_mean(object_distance, target_state.object_batch, target_state.graph_latent.size(0))
    return (graph_weight * graph_distance + object_weight * object_distance).min(dim=-1).values
