"""Core graph-native JEPA rollout utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from acs_jepa.architectures import JEPALatentState, latent_time_length, latent_time_slice
from acs_jepa.graph.encoders import GraphEncoderOutput
from acs_jepa.losses import GraphJEPALossOutput


@dataclass(frozen=True)
class GraphJEPATrainingOutput:
    """Outputs from observed-trajectory training.

    Shapes:
        observed_states: Encoded observed states with graph latents
            ``FloatTensor[B, K + 1, D_z]``.
        predicted_states_by_order: Recursive predictions by order. Order ``k``
            contains a temporal latent state for target timesteps ``k ... K``.
        action_latents: Encoded action latents ``FloatTensor[B, K, D_a]``.
        loss: Core JEPA scalar loss terms from ``loss_module``.
    """

    observed_states: JEPALatentState
    predicted_states_by_order: Mapping[int, JEPALatentState]
    action_latents: Tensor
    loss: GraphJEPALossOutput


@dataclass(frozen=True)
class GraphJEPACandidateRolloutOutput:
    """Predicted latent rollout for externally supplied grounded candidates."""

    initial_state: JEPALatentState
    final_state: JEPALatentState
    predicted_states: tuple[JEPALatentState, ...]
    control_latents: Tensor


class GraphJEPA(nn.Module):
    """Small core graph JEPA wrapper for trajectory learning and rollout.

    ``GraphJEPA`` owns only the core JEPA modules: graph/state encoders, action
    encoder, predictor, and core loss module. Predicate evaluators are trained
    and composed outside this wrapper.

    Args:
        graph_encoder: Module with ``forward(Data) -> GraphEncoderOutput``.
        state_encoder: Module with ``forward(GraphEncoderOutput) -> JEPALatentState``.
        action_encoder: Module with
            ``forward(action_tensors, JEPALatentState) -> FloatTensor[B, D_a]``
            or ``FloatTensor[B, K, D_a]`` for temporal input.
        predictor: Module with
            ``forward(JEPALatentState, FloatTensor[B, D_a]) -> JEPALatentState``.
        loss_module: Core JEPA loss module with
            ``forward(observed_states, predicted_states_by_order,
            action_latents) -> GraphJEPALossOutput``.
    """

    def __init__(
        self,
        *,
        graph_encoder: nn.Module,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        loss_module: nn.Module,
    ) -> None:
        super().__init__()
        if loss_module is None:
            raise ValueError("GraphJEPA requires a core loss_module")
        self.graph_encoder = graph_encoder
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.loss_module = loss_module

    def encode(self, graph: Data) -> JEPALatentState:
        """Encode state graph(s) into JEPA latent space."""

        return self.state_encoder(self.graph_encoder(graph))

    def encode_action(self, action_tensors: dict[str, Tensor], graph: Data) -> Tensor:
        """Encode grounded action(s) from the current state context."""

        graph_output = self.graph_encoder(graph)
        latent_state = self.state_encoder(graph_output)
        return self.action_encoder(action_tensors, latent_state)

    def rollout_grounded_candidates(
        self,
        initial_state: JEPALatentState,
        action_tensors: dict[str, Tensor],
    ) -> GraphJEPACandidateRolloutOutput:
        """Predict a rollout for externally supplied grounded candidates."""

        if not action_tensors:
            raise ValueError("action_tensors must not be empty")
        leading_shape: tuple[int, int] | None = None
        for name, tensor in action_tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"action tensor '{name}' must be a Tensor")
            if tensor.ndim < 2:
                raise ValueError(f"action tensor '{name}' must have at least two dimensions")
            if leading_shape is None:
                leading_shape = (tensor.size(0), tensor.size(1))
            elif tensor.shape[:2] != leading_shape:
                raise ValueError("all action tensors must share leading [B, H] dimensions")
        assert leading_shape is not None
        if initial_state.graph_latent.ndim != 2:
            raise ValueError("initial_state.graph_latent must have shape [B, D_z]")
        if initial_state.graph_latent.size(0) == 0 or initial_state.graph_latent.size(1) == 0:
            raise ValueError("initial_state.graph_latent dimensions must be nonempty")
        if initial_state.object_latents.ndim != 2:
            raise ValueError("initial_state.object_latents must have shape [N_obj, D_z]")
        if initial_state.object_latents.size(1) != initial_state.graph_latent.size(1):
            raise ValueError("initial state graph and object latent widths must match")
        if initial_state.object_latents.size(0) != initial_state.object_batch.numel():
            raise ValueError("initial state object rows and metadata lengths must match")
        if initial_state.object_ids.ndim != 1:
            raise ValueError("initial_state.object_ids must be rank one")
        if initial_state.object_batch.ndim != 1:
            raise ValueError("initial_state.object_batch must be rank one")
        if initial_state.object_latents.size(0) != initial_state.object_ids.numel():
            raise ValueError("initial state object rows and metadata lengths must match")
        if torch.any(initial_state.object_batch < 0) or torch.any(
            initial_state.object_batch >= initial_state.graph_latent.size(0)
        ):
            raise ValueError("initial_state.object_batch values must be in [0, B)")
        if leading_shape[0] != initial_state.graph_latent.size(0):
            raise ValueError("candidate batch must match initial state graph batch")
        if leading_shape[1] == 0:
            raise ValueError("candidate horizon must be positive")
        if hasattr(self.action_encoder, "context_steps") and self.action_encoder.context_steps != 1:
            raise ValueError("action_encoder.context_steps must be exactly 1")
        initial_graph_shape = tuple(initial_state.graph_latent.shape)
        initial_object_shape = tuple(initial_state.object_latents.shape)
        initial_object_ids = initial_state.object_ids.clone()
        initial_object_batch = initial_state.object_batch.clone()
        current_state = initial_state
        predicted_states: list[JEPALatentState] = []
        controls: list[Tensor] = []
        horizon = leading_shape[1]
        for step in range(horizon):
            candidate = {name: tensor[:, step] for name, tensor in action_tensors.items()}
            control = self.action_encoder(candidate, current_state)
            if not isinstance(control, Tensor):
                raise TypeError("action encoder control must be a Tensor with shape [B, D_a]")
            if control.ndim != 2:
                raise ValueError("action encoder control must have shape [B, D_a]")
            if control.size(0) != leading_shape[0]:
                raise ValueError("action encoder control batch must match candidate batch")
            if controls and control.size(1) != controls[0].size(1):
                raise ValueError("action encoder control width must remain constant")
            current_state = self.predictor(current_state, control)
            if not isinstance(current_state, JEPALatentState):
                raise TypeError("predictor must return JEPALatentState")
            if current_state.graph_latent.ndim != 2:
                raise ValueError("predictor graph_latent must preserve rank two")
            if current_state.graph_latent.size(0) != initial_graph_shape[0]:
                raise ValueError("predictor graph_latent must preserve initial batch size")
            if current_state.graph_latent.size(1) != initial_graph_shape[1]:
                raise ValueError("predictor graph_latent must preserve initial latent width")
            if current_state.object_latents.ndim != 2:
                raise ValueError("predictor object_latents must preserve rank two")
            if current_state.object_latents.size(0) != initial_object_shape[0]:
                raise ValueError("predictor object_latents must preserve initial row count")
            if current_state.object_latents.size(1) != initial_object_shape[1]:
                raise ValueError("predictor object_latents must preserve initial latent width")
            if not torch.equal(current_state.object_ids, initial_object_ids):
                raise ValueError("predictor object_ids must preserve initial values")
            if current_state.object_ids is not initial_state.object_ids:
                raise ValueError("predictor object_ids must preserve exact tensor identity")
            if not torch.equal(current_state.object_batch, initial_object_batch):
                raise ValueError("predictor object_batch must preserve initial values")
            if current_state.object_batch is not initial_state.object_batch:
                raise ValueError("predictor object_batch must preserve exact tensor identity")
            controls.append(control)
            predicted_states.append(current_state)
        return GraphJEPACandidateRolloutOutput(
            initial_state=initial_state,
            final_state=predicted_states[-1],
            predicted_states=tuple(predicted_states),
            control_latents=torch.stack(controls, dim=1),
        )

    def trajectory_rollout(
        self,
        state_graphs: Sequence[Data],
        action_tensors: dict[str, Tensor],
    ) -> GraphJEPATrainingOutput:
        """Process observed trajectory windows over the PyG batch dimension."""

        if len(state_graphs) < 2:
            raise ValueError("trajectory_rollout requires at least two state graphs")
        num_steps = len(state_graphs) - 1
        graph_outputs = tuple(self.graph_encoder(graph) for graph in state_graphs)
        # Stack per-frame encoder outputs into [B, K + 1, D_e] graph tensors
        # and [N_obj, K + 1, D_e] object tensors before temporal state encoding.
        temporal_graph_output = _stack_graph_outputs(graph_outputs)
        observed_states = self.state_encoder(temporal_graph_output)
        # Actions a_t are encoded against source states s_t, so the action
        # context uses observed timesteps [0, K) while targets live in [1, K].
        action_latents = self.action_encoder(action_tensors, latent_time_slice(observed_states, 0, num_steps))
        predicted_states_by_order = self._recursive_predictions(observed_states, action_latents)
        loss = self.loss_module(
            observed_states=observed_states,
            predicted_states_by_order=predicted_states_by_order,
            action_latents=action_latents,
        )
        return GraphJEPATrainingOutput(
            observed_states=observed_states,
            predicted_states_by_order=predicted_states_by_order,
            action_latents=action_latents,
            loss=loss,
        )

    def _recursive_predictions(
        self,
        observed_states: JEPALatentState,
        action_latents: Tensor,
    ) -> dict[int, JEPALatentState]:
        if action_latents.ndim != 3:
            raise ValueError(f"Expected temporal action_latents shape [B, K, D_a], got {tuple(action_latents.shape)}")
        by_order: dict[int, JEPALatentState] = {}
        previous_order = observed_states
        max_order = action_latents.size(1)
        for order in range(1, max_order + 1):
            # Each order consumes the previous order's source sequence and drops
            # its last timestep, giving lengths K, K-1, ..., 1.
            source = latent_time_slice(previous_order, 0, latent_time_length(previous_order) - 1)
            # For order k, the first available source predicts with action
            # a_{k-1}; batching the remaining aligned pairs preserves rollout
            # semantics while calling the predictor once per order.
            predictions = self.predictor(source, action_latents[:, order - 1 :])
            by_order[order] = predictions
            previous_order = predictions
        return by_order

    def forward(
        self,
        state_graphs: Sequence[Data],
        action_tensors: dict[str, Tensor],
    ) -> GraphJEPATrainingOutput:
        """Alias for ``trajectory_rollout``."""

        return self.trajectory_rollout(state_graphs, action_tensors)


def _stack_graph_outputs(graph_outputs: Sequence[GraphEncoderOutput]) -> GraphEncoderOutput:
    if not graph_outputs:
        raise ValueError("graph_outputs must not be empty")
    first = graph_outputs[0]
    for output in graph_outputs[1:]:
        # Object rows are stacked over time, so ids and graph membership must
        # already be aligned across every observed state graph.
        if not torch.equal(output.object_ids, first.object_ids) or not torch.equal(
            output.object_batch, first.object_batch
        ):
            raise ValueError("Object identity and batch tensors must be stable across trajectory states")
    return GraphEncoderOutput(
        # Insert time at dim 1: graph [B, D_e] -> [B, T, D_e],
        # object [N_obj, D_e] -> [N_obj, T, D_e].
        graph_embedding=torch.stack([output.graph_embedding for output in graph_outputs], dim=1),
        object_embeddings=torch.stack([output.object_embeddings for output in graph_outputs], dim=1),
        object_ids=first.object_ids,
        object_batch=first.object_batch,
    )
