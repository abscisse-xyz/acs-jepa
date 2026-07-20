"""Graph-native JEPA losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from acs_jepa.architectures import JEPALatentState, latent_time_length, latent_time_slice


def square_loss(x: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    """Squared error helper matching the video JEPA loss style.

    Args:
        x: Prediction tensor with arbitrary shape.
        y: Target tensor with the same shape as ``x``.
        reduction: PyTorch MSE reduction, usually ``"mean"`` for scalar loss.

    Returns:
        MSE tensor. With the default reduction this is a scalar
        ``FloatTensor[]``.
    """

    return F.mse_loss(x, y, reduction=reduction)


@dataclass(frozen=True)
class GraphLatentPredictionLossOutput:
    """Scalar graph/object prediction loss terms."""

    total: Tensor
    graph: Tensor
    object: Tensor


@dataclass(frozen=True)
class GraphJEPALossOutput:
    """Composite graph JEPA loss output.

    All fields except optional terms are scalar ``FloatTensor[]`` values.
    ``terms`` contains the named tensors used to form ``total``.
    """

    total: Tensor
    prediction: Tensor
    graph_prediction: Tensor
    object_prediction: Tensor
    regularization: Tensor
    similarity: Tensor | None
    inverse_dynamics: Tensor | None
    terms: dict[str, Tensor]


@dataclass(frozen=True)
class ApplicabilityLossOutput:
    """Scalar applicability BCE loss and diagnostic logit statistics."""

    total: Tensor
    bce: Tensor
    positive_logit_mean: Tensor | None
    negative_logit_mean: Tensor | None
    positive_negative_margin: Tensor | None
    num_examples: int
    num_positive: int
    num_negative: int


class ApplicabilityLoss(nn.Module):
    """Binary cross-entropy objective for applicability logits.

    The module consumes precomputed logits and labels. It deliberately has no
    dependency on action encoders, simulator oracles, or action enumeration.
    """

    def __init__(self, *, pos_weight: float | None = None) -> None:
        super().__init__()
        if pos_weight is not None and pos_weight <= 0.0:
            raise ValueError("pos_weight must be positive")
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        example_mask: Tensor | None = None,
    ) -> ApplicabilityLossOutput:
        """Compute BCE over effective labeled examples.

        Args:
            logits: Applicability logits shaped ``[N]``.
            labels: Binary/probability labels shaped ``[N]`` with values in
                ``[0, 1]``.
            example_mask: Optional bool mask shaped ``[N]``. ``False`` rows are
                excluded from loss/statistics and receive no gradient.

        Returns:
            Loss output with BCE and positive/negative logit diagnostics.
        """

        effective_logits, effective_labels = self._effective_examples(logits, labels, example_mask)
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = effective_logits.new_tensor(self.pos_weight)
        bce = F.binary_cross_entropy_with_logits(
            effective_logits,
            effective_labels.to(dtype=effective_logits.dtype),
            pos_weight=pos_weight,
        )
        positive_mask = effective_labels >= 0.5
        negative_mask = ~positive_mask
        positive_mean = effective_logits[positive_mask].mean() if positive_mask.any() else None
        negative_mean = effective_logits[negative_mask].mean() if negative_mask.any() else None
        margin = positive_mean - negative_mean if positive_mean is not None and negative_mean is not None else None
        return ApplicabilityLossOutput(
            total=bce,
            bce=bce,
            positive_logit_mean=positive_mean,
            negative_logit_mean=negative_mean,
            positive_negative_margin=margin,
            num_examples=int(effective_logits.numel()),
            num_positive=int(positive_mask.sum().item()),
            num_negative=int(negative_mask.sum().item()),
        )

    @staticmethod
    def _effective_examples(logits: Tensor, labels: Tensor, example_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if logits.ndim != 1:
            raise ValueError("logits must have shape [N]")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [N]")
        if logits.shape != labels.shape:
            raise ValueError("logits and labels must have the same shape")
        if example_mask is not None:
            if example_mask.ndim != 1:
                raise ValueError("example_mask must have shape [N]")
            if example_mask.shape != logits.shape:
                raise ValueError("example_mask and logits must have the same shape")
            if example_mask.dtype != torch.bool:
                raise ValueError("example_mask must be bool")
            logits = logits[example_mask]
            labels = labels[example_mask]
        if logits.numel() == 0:
            raise ValueError("ApplicabilityLoss received an empty effective batch")
        if torch.any((labels < 0) | (labels > 1)):
            raise ValueError("labels must lie in [0, 1]")
        return logits, labels


class GraphLatentPredictionLoss(nn.Module):
    """Prediction loss between predicted and target graph/object latents.

    Args:
        graph_weight: Coefficient for graph latent MSE.
        object_weight: Coefficient for object latent MSE.
    """

    def __init__(self, graph_weight: float = 1.0, object_weight: float = 1.0) -> None:
        super().__init__()
        self.graph_weight = graph_weight
        self.object_weight = object_weight

    def forward(self, predicted: JEPALatentState, target: JEPALatentState) -> GraphLatentPredictionLossOutput:
        """Compare predicted and target latent states.

        Args:
            predicted: State with ``graph_latent`` shape ``[B, D_z]`` and
                ``object_latents`` shape ``[N_obj, D_z]``.
            target: State with matching graph/object latent shapes.

        Returns:
            Scalar total, graph, and object prediction losses.
        """

        graph_loss = square_loss(predicted.graph_latent, target.graph_latent)
        object_loss = square_loss(predicted.object_latents, target.object_latents)
        total = self.graph_weight * graph_loss + self.object_weight * object_loss
        return GraphLatentPredictionLossOutput(total=total, graph=graph_loss, object=object_loss)


class HingeStdLoss(nn.Module):
    """Encourage every latent dimension to keep at least ``std_margin`` std.

    Args:
        std_margin: Lower bound margin for per-dimension standard deviation.
    """

    def __init__(self, std_margin: float = 1.0) -> None:
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: Tensor) -> Tensor:
        """Compute the hinge standard-deviation penalty.

        Args:
            x: Sample matrix ``FloatTensor[N, D]``.

        Returns:
            Scalar ``FloatTensor[]``. Returns zero when ``N <= 1``.
        """

        if x.size(0) <= 1:
            return x.new_tensor(0.0)
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        return F.relu(self.std_margin - std).mean()


class CovarianceLoss(nn.Module):
    """Penalize off-diagonal covariance terms."""

    def forward(self, x: Tensor) -> Tensor:
        """Compute off-diagonal covariance penalty.

        Args:
            x: Sample matrix ``FloatTensor[N, D]``.

        Returns:
            Scalar ``FloatTensor[]``. Returns zero when ``N <= 1``.
        """

        if x.size(0) <= 1:
            return x.new_tensor(0.0)
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (x.size(0) - 1)
        return _off_diagonal(cov).pow(2).mean()


class GraphVCLoss(nn.Module):
    """Variance/covariance anti-collapse regularizer for graph-native latents.

    Args:
        std_coeff: Coefficient for :class:`HingeStdLoss`.
        cov_coeff: Coefficient for :class:`CovarianceLoss`.
        std_margin: Target lower bound for latent standard deviation.
        target: Which samples to regularize: ``"graph"``, ``"object"``, or
            ``"both"``. ``"both"`` concatenates graph and object latents along
            the sample dimension before computing VC terms.

    Modules:
        std_loss: Hinge standard-deviation penalty.
        cov_loss: Off-diagonal covariance penalty.
    """

    def __init__(
        self,
        std_coeff: float = 1.0,
        cov_coeff: float = 1.0,
        std_margin: float = 1.0,
        target: str = "both",
    ) -> None:
        super().__init__()
        if target not in {"graph", "object", "both"}:
            raise ValueError(f"Unknown VC target: {target}")
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.target = target
        self.std_loss = HingeStdLoss(std_margin=std_margin)
        self.cov_loss = CovarianceLoss()

    def forward(self, state: JEPALatentState) -> Tensor:
        """Regularize latent samples from a state.

        Args:
            state: Latent state with graph samples ``[B, D_z]`` and object
                samples ``[N_obj, D_z]``.

        Returns:
            Scalar anti-collapse loss.
        """

        samples = []
        if self.target in {"graph", "both"}:
            samples.append(_sample_matrix(state.graph_latent))
        if self.target in {"object", "both"}:
            samples.append(_sample_matrix(state.object_latents))
        x = torch.cat(samples, dim=0)
        return self.std_coeff * self.std_loss(x) + self.cov_coeff * self.cov_loss(x)


class GraphTemporalSimilarityLoss(nn.Module):
    """Penalize excessive graph/object latent movement across one transition.

    Args:
        graph_weight: Coefficient for graph latent movement.
        object_weight: Coefficient for object latent movement.
    """

    def __init__(self, graph_weight: float = 1.0, object_weight: float = 1.0) -> None:
        super().__init__()
        self.graph_weight = graph_weight
        self.object_weight = object_weight

    def forward(self, state: JEPALatentState, next_state: JEPALatentState) -> Tensor:
        """Compute transition smoothness between two latent states.

        Args:
            state: Source state with graph/object latent shapes ``[B, D_z]`` and
                ``[N_obj, D_z]``.
            next_state: Target state with matching shapes.

        Returns:
            Scalar weighted MSE.
        """

        graph_loss = square_loss(state.graph_latent, next_state.graph_latent)
        object_loss = square_loss(state.object_latents, next_state.object_latents)
        return self.graph_weight * graph_loss + self.object_weight * object_loss


class GraphInverseDynamicsModel(nn.Module):
    """Predict an encoded action from consecutive graph latents.

    This small MLP is used by encoded-action inverse dynamics. It predicts the
    action latent produced by ``q_phi`` and does not decode an action id or
    action arguments.

    Args:
        latent_dim: JEPA graph latent size ``D_z``.
        action_dim: Encoded action size ``D_a``.
        hidden_dim: Hidden width of the inverse dynamics MLP. Defaults to
            ``action_dim``.

    Modules:
        model: MLP ``[2 * D_z] -> D_a``.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = action_dim if hidden_dim is None else hidden_dim
        self.model = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: JEPALatentState, next_state: JEPALatentState) -> Tensor:
        """Predict encoded action latents.

        Args:
            state: Current latent state with ``graph_latent`` shape
                ``[B, D_z]``.
            next_state: Next latent state with ``graph_latent`` shape
                ``[B, D_z]``.

        Returns:
            Predicted action latent ``FloatTensor[B, D_a]``.
        """

        return self.model(torch.cat([state.graph_latent, next_state.graph_latent], dim=-1))


class GraphEncodedActionInverseDynamicsLoss(nn.Module):
    """Inverse dynamics loss that regresses to the encoded grounded action.

    Args:
        inverse_dynamics_model: Module with
            ``forward(state, next_state) -> FloatTensor[B, D_a]``.
        detach_target: If ``True``, use ``action_latent.detach()`` as the
            target so inverse dynamics does not update ``q_phi`` through this
            loss term.

    Modules:
        inverse_dynamics_model: Encoded-action regression model.
    """

    def __init__(self, inverse_dynamics_model: nn.Module, detach_target: bool = True) -> None:
        super().__init__()
        self.inverse_dynamics_model = inverse_dynamics_model
        self.detach_target = detach_target

    def forward(self, state: JEPALatentState, next_state: JEPALatentState, action_latent: Tensor) -> Tensor:
        """Compute encoded-action inverse dynamics MSE.

        Args:
            state: Current latent state.
            next_state: Next latent state.
            action_latent: Target action latent ``FloatTensor[B, D_a]``.

        Returns:
            Scalar MSE between predicted and target action latents.
        """

        target = action_latent.detach() if self.detach_target else action_latent
        prediction = self.inverse_dynamics_model(state, next_state)
        return square_loss(prediction, target)


class GraphJEPALossModule(nn.Module):
    """Core graph JEPA loss.

    The total loss is:

    ``prediction_coeff * prediction
    + regularization_coeff * vc
    + similarity_coeff * temporal_similarity
    + inverse_dynamics_coeff * encoded_action_idm``

    Optional terms are included only when the corresponding module is provided
    and the coefficient is non-zero. Predicate losses are intentionally outside
    the core JEPA objective.

    Args:
        prediction_loss: Required latent prediction criterion.
        regularization_loss: Required variance/covariance anti-collapse module.
        temporal_similarity_loss: Optional temporal smoothness module.
        inverse_dynamics_loss: Optional encoded-action inverse dynamics module.
        prediction_coeff: Coefficient for latent prediction loss.
        regularization_coeff: Coefficient for regularization loss.
        similarity_coeff: Coefficient for temporal similarity.
        inverse_dynamics_coeff: Coefficient for inverse dynamics.

    Modules:
        prediction_loss: Graph/object latent prediction loss.
        regularization_loss: Required core regularizer.
        temporal_similarity_loss: Optional transition smoothness regularizer.
        inverse_dynamics_loss: Optional encoded-action IDM loss.
    """

    def __init__(
        self,
        prediction_loss: GraphLatentPredictionLoss,
        regularization_loss: GraphVCLoss,
        *,
        temporal_similarity_loss: GraphTemporalSimilarityLoss | None = None,
        inverse_dynamics_loss: GraphEncodedActionInverseDynamicsLoss | None = None,
        prediction_coeff: float = 1.0,
        regularization_coeff: float = 1.0,
        similarity_coeff: float = 0.0,
        inverse_dynamics_coeff: float = 0.0,
        rollout_order_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.prediction_loss = prediction_loss
        self.regularization_loss = regularization_loss
        self.temporal_similarity_loss = temporal_similarity_loss
        self.inverse_dynamics_loss = inverse_dynamics_loss
        self.prediction_coeff = prediction_coeff
        self.regularization_coeff = regularization_coeff
        self.similarity_coeff = similarity_coeff
        self.inverse_dynamics_coeff = inverse_dynamics_coeff
        if rollout_order_weights is not None and (
            not rollout_order_weights or any(float(weight) < 0.0 for weight in rollout_order_weights)
        ):
            raise ValueError("rollout_order_weights must be a non-empty sequence of non-negative weights")
        self.rollout_order_weights = (
            None if rollout_order_weights is None else tuple(float(w) for w in rollout_order_weights)
        )

    def forward(
        self,
        *,
        observed_states: JEPALatentState,
        predicted_states_by_order: Mapping[int, JEPALatentState],
        action_latents: Tensor,
    ) -> GraphJEPALossOutput:
        """Compute the core graph JEPA trajectory loss.

        Args:
            observed_states: Encoded observed states with temporal latents
                ``[B, K + 1, D_z]`` and ``[N_obj, K + 1, D_z]``.
            predicted_states_by_order: Mapping from rollout order ``k`` to
                temporal predictions for target timesteps ``k ... K``.
            action_latents: Encoded action latents ``FloatTensor[B, K, D_a]``.

        Returns:
            ``GraphJEPALossOutput`` with scalar terms and named ``terms``.
        """

        if observed_states.graph_latent.ndim != 3 or observed_states.object_latents.ndim != 3:
            raise ValueError("observed_states must be temporal rank-3 latent tensors")
        if action_latents.ndim != 3:
            raise ValueError("action_latents must have shape [B, K, D_a]")
        observed_len = latent_time_length(observed_states)
        if observed_len < 2:
            raise ValueError("observed_states must contain at least two states")
        if action_latents.size(1) != observed_len - 1:
            raise ValueError("action_latents time length must be observed_states length - 1")

        order_losses: list[GraphLatentPredictionLossOutput] = []
        weighted_predictions: list[Tensor] = []
        order_weights = self._order_weights(max(predicted_states_by_order))
        for order in sorted(predicted_states_by_order):
            predictions = predicted_states_by_order[order]
            if order < 1:
                raise ValueError("Prediction orders must be positive")
            expected_steps = observed_len - order
            if latent_time_length(predictions) != expected_steps:
                raise ValueError(
                    f"Order {order} expected {expected_steps} predictions, got {latent_time_length(predictions)}"
                )
            target = latent_time_slice(observed_states, order, observed_len)
            order_output = self.prediction_loss(predictions, target)
            order_total = order_output.total
            order_graph = order_output.graph
            order_object = order_output.object
            order_output = GraphLatentPredictionLossOutput(total=order_total, graph=order_graph, object=order_object)
            order_losses.append(order_output)
            weighted_predictions.append(order_weights[order - 1] * order_total)

        if not order_losses:
            raise ValueError("predicted_states_by_order must contain at least one order")

        prediction_total = sum(weighted_predictions) / sum(order_weights[: len(order_losses)])
        graph_prediction = _mean_tensors([loss.graph for loss in order_losses])
        object_prediction = _mean_tensors([loss.object for loss in order_losses])
        regularization = self.regularization_loss(latent_time_slice(observed_states, 1, observed_len))
        terms = {
            "prediction": prediction_total,
            "graph_prediction": graph_prediction,
            "object_prediction": object_prediction,
            "regularization": regularization,
        }
        for order, order_loss in enumerate(order_losses, start=1):
            terms[f"prediction/order_{order}"] = order_loss.total
        total = self.prediction_coeff * prediction_total + self.regularization_coeff * regularization

        similarity = None
        if self.temporal_similarity_loss is not None and self.similarity_coeff != 0:
            order_one = predicted_states_by_order[1]
            similarity = self.temporal_similarity_loss(
                latent_time_slice(observed_states, 0, observed_len - 1),
                order_one,
            )
            terms["similarity"] = similarity
            total = total + self.similarity_coeff * similarity

        inverse_dynamics = None
        if self.inverse_dynamics_loss is not None and self.inverse_dynamics_coeff != 0:
            inverse_dynamics = self.inverse_dynamics_loss(
                latent_time_slice(observed_states, 0, observed_len - 1),
                latent_time_slice(observed_states, 1, observed_len),
                action_latents,
            )
            terms["inverse_dynamics"] = inverse_dynamics
            total = total + self.inverse_dynamics_coeff * inverse_dynamics

        terms["total"] = total
        return GraphJEPALossOutput(
            total=total,
            prediction=prediction_total,
            graph_prediction=graph_prediction,
            object_prediction=object_prediction,
            regularization=regularization,
            similarity=similarity,
            inverse_dynamics=inverse_dynamics,
            terms=terms,
        )

    def _order_weights(self, max_order: int) -> tuple[float, ...]:
        if self.rollout_order_weights is None:
            return (1.0,) * max_order
        if len(self.rollout_order_weights) < max_order:
            raise ValueError(
                f"rollout_order_weights has {len(self.rollout_order_weights)} entries but max order is {max_order}"
            )
        if sum(self.rollout_order_weights[:max_order]) <= 0.0:
            raise ValueError("At least one rollout order weight must be positive")
        return self.rollout_order_weights


def _off_diagonal(x: Tensor) -> Tensor:
    rows, cols = x.shape
    if rows != cols:
        raise ValueError("off-diagonal extraction expects a square matrix")
    return x.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def _sample_matrix(value: Tensor) -> Tensor:
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        return value.reshape(value.size(0) * value.size(1), value.size(2))
    raise ValueError(f"Expected rank-2 or rank-3 latent tensor, got shape {tuple(value.shape)}")


def _mean_tensors(values: Sequence[Tensor]) -> Tensor:
    if not values:
        raise ValueError("Cannot average an empty tensor sequence")
    return torch.stack(list(values)).mean()
