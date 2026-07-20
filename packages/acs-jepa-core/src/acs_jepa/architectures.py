"""Neural architecture blocks for graph-native JEPA models."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from acs_jepa.graph.builders import tensorize_action
from acs_jepa.graph.encoders import GraphEncoderOutput
from acs_jepa.graph.schemas import GroundAction, ParsedProblem
from acs_jepa.mpc import SoftmaxWeighting, cross_entropy_optimize


@dataclass(frozen=True)
class JEPALatentState:
    """Latent state passed between graph-native JEPA modules.

    Shapes:
        graph_latent: ``FloatTensor[B, D_z]`` for a single state or
            ``FloatTensor[B, T, D_z]`` for a temporal state sequence.
        object_latents: ``FloatTensor[N_obj, D_z]`` for a single state or
            ``FloatTensor[N_obj, T, D_z]`` for a temporal state sequence.
        object_ids: ``LongTensor[N_obj]`` with problem-local object ids matching
            the graph builder vocabulary.
        object_batch: ``LongTensor[N_obj]`` mapping each object latent to its
            graph index in ``[0, B)``.
    """

    graph_latent: Tensor
    object_latents: Tensor
    object_ids: Tensor
    object_batch: Tensor


def is_temporal_latent_state(state: JEPALatentState) -> bool:
    """Return ``True`` when ``state`` carries a time axis."""

    return state.graph_latent.ndim == 3


def latent_time_length(state: JEPALatentState) -> int:
    """Return temporal length for a rank-3 latent state."""

    if not is_temporal_latent_state(state):
        raise ValueError("Expected temporal latent state with graph_latent shape [B, T, D]")
    return state.graph_latent.size(1)


def latent_time_slice(state: JEPALatentState, start: int, end: int) -> JEPALatentState:
    """Slice a temporal latent state's time axis without changing metadata."""

    if not is_temporal_latent_state(state):
        raise ValueError("Expected temporal latent state")
    return JEPALatentState(
        graph_latent=state.graph_latent[:, start:end],
        object_latents=state.object_latents[:, start:end],
        object_ids=state.object_ids,
        object_batch=state.object_batch,
    )


def latent_state_at(state: JEPALatentState, step_idx: int) -> JEPALatentState:
    """Select one timestep from a temporal latent state."""

    if not is_temporal_latent_state(state):
        return state
    return JEPALatentState(
        graph_latent=state.graph_latent[:, step_idx],
        object_latents=state.object_latents[:, step_idx],
        object_ids=state.object_ids,
        object_batch=state.object_batch,
    )


def flatten_temporal_latent_state(state: JEPALatentState) -> JEPALatentState:
    """Flatten ``[B, T, D]`` latents to pseudo-batch ``[B * T, D]``.

    Object rows are flattened from ``[N_obj, T, D]`` to ``[N_obj * T, D]`` and
    ``object_batch`` is remapped from graph ids ``b`` to graph-time ids
    ``b * T + t``.
    """

    if not is_temporal_latent_state(state):
        return state
    time_steps = latent_time_length(state)
    object_count = state.object_latents.size(0)
    time_ids = torch.arange(time_steps, device=state.object_batch.device)
    object_batch = state.object_batch.repeat_interleave(time_steps) * time_steps + time_ids.repeat(object_count)
    return JEPALatentState(
        graph_latent=state.graph_latent.reshape(-1, state.graph_latent.size(-1)),
        object_latents=state.object_latents.reshape(-1, state.object_latents.size(-1)),
        object_ids=state.object_ids.repeat_interleave(time_steps),
        object_batch=object_batch,
    )


class GraphStateProjector(nn.Module):
    """Project graph encoder embeddings into JEPA latent state variables.

    This module projects time-independent graph encoder output into the latent
    space used by the temporal state encoder.

    Args:
        embedding_dim: Input graph/object embedding size ``D_e``.
        latent_dim: Output JEPA latent size ``D_z``.
        hidden_dim: Hidden width for the graph and object projection MLPs.
            Defaults to ``latent_dim``.

    Modules:
        graph_projector: MLP mapping ``[..., D_e]`` to ``[..., D_z]``.
        object_projector: MLP mapping ``[..., D_e]`` to ``[..., D_z]``.
    """

    def __init__(self, embedding_dim: int, latent_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = latent_dim if hidden_dim is None else hidden_dim
        self.graph_projector = _mlp(embedding_dim, hidden_dim, latent_dim)
        self.object_projector = _mlp(embedding_dim, hidden_dim, latent_dim)

    def forward(self, graph_output: GraphEncoderOutput) -> JEPALatentState:
        """Encode graph/object embeddings into JEPA latents.

        Args:
            graph_output: Output from ``GraphEncoder`` with graph/object
                embeddings shaped ``[B, D_e]``/``[N_obj, D_e]`` or
                ``[B, T, D_e]``/``[N_obj, T, D_e]``.

        Returns:
            ``JEPALatentState`` with matching graph/object rank in ``D_z`` and
            object identity tensors preserved from ``graph_output``.
        """

        return JEPALatentState(
            graph_latent=self.graph_projector(graph_output.graph_embedding),
            object_latents=self.object_projector(graph_output.object_embeddings),
            object_ids=graph_output.object_ids,
            object_batch=graph_output.object_batch,
        )


class StateEncoderF(nn.Module):
    """Causal sequence encoder for JEPA state latents.

    ``GraphEncoder`` remains time-independent. This module first projects each
    graph encoder output with ``GraphStateProjector``, then applies GRUs over the
    graph-latent and object-latent time axes when a time axis is present.

    Shape convention:
        ``B`` is the number of graphs in the PyG batch, ``T_s`` is the number of
        state frames in a trajectory window, ``N_obj`` is the total number of
        object nodes across the batch, ``D_e`` is the graph-encoder embedding
        size, and ``D_z`` is the JEPA latent size.

    ``forward`` accepts either single embeddings ``[B, D_e]``/``[N_obj, D_e]``
    or temporal embeddings ``[B, T_s, D_e]``/``[N_obj, T_s, D_e]``.
    """

    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        *,
        context_steps: int | None = None,
    ) -> None:
        super().__init__()
        if context_steps is not None and context_steps < 1:
            raise ValueError("context_steps must be positive when provided")
        self.base_encoder = GraphStateProjector(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.graph_gru = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.object_gru = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.context_steps = context_steps

    def forward(self, graph_output: GraphEncoderOutput) -> JEPALatentState:
        """Encode single-step or temporal graph encoder output.

        Args:
            graph_output: Graph encoder output with ``graph_embedding`` shaped
                ``[B, D_e]`` or ``[B, T_s, D_e]`` and ``object_embeddings``
                shaped ``[N_obj, D_e]`` or ``[N_obj, T_s, D_e]``.

        Returns:
            Latent state with matching single or temporal rank.
        """

        projected = self.base_encoder(graph_output)
        if projected.graph_latent.ndim == 2:
            temporal = JEPALatentState(
                graph_latent=projected.graph_latent.unsqueeze(1),
                object_latents=projected.object_latents.unsqueeze(1),
                object_ids=projected.object_ids,
                object_batch=projected.object_batch,
            )
            encoded = self._encode_temporal(temporal)
            return latent_state_at(encoded, 0)
        if projected.graph_latent.ndim == 3:
            return self._encode_temporal(projected)
        raise ValueError(f"Expected graph_latent rank 2 or 3, got {projected.graph_latent.ndim}")

    def _encode_temporal(self, projected: JEPALatentState) -> JEPALatentState:
        window = self.context_steps
        if window is not None and window < 1:
            raise ValueError("context_steps must be positive when provided")
        graph_inputs = projected.graph_latent
        object_inputs = projected.object_latents
        if graph_inputs.ndim != 3 or object_inputs.ndim != 3:
            raise ValueError("Temporal state encoding expects graph/object latents with rank 3")
        # Batch-first GRUs preserve [batch, time, hidden]: [B, T_s, D_z].
        graph_outputs_seq = _causal_gru_outputs(self.graph_gru, graph_inputs, window)
        # Objects are treated as the GRU batch dimension: [N_obj, T_s, D_z].
        object_outputs_seq = _causal_gru_outputs(self.object_gru, object_inputs, window)
        return JEPALatentState(
            graph_latent=graph_outputs_seq,
            object_latents=object_outputs_seq,
            object_ids=projected.object_ids,
            object_batch=projected.object_batch,
        )


class ActionEncoder(nn.Module):
    """Causal sequence encoder for grounded action latents.

    The wrapped ``action_encoder`` encodes one grounded action per graph using
    the matching JEPA latent state. This module applies a GRU over the action
    time axis when the wrapped encoder returns temporal latents.

    Shape convention:
        ``B`` is the graph batch size, ``T_a`` is the number of actions in the
        trajectory window, ``A_max`` is the maximum action arity, and ``D_a`` is
        the action latent size.

    Temporal action tensors must be batched as ``[B, T_a]`` for ``action_id``
    and ``[B, T_a, A_max]`` for argument fields.
    """

    def __init__(
        self,
        action_encoder: nn.Module,
        action_dim: int,
        *,
        context_steps: int | None = None,
    ) -> None:
        super().__init__()
        if context_steps is not None and context_steps < 1:
            raise ValueError("context_steps must be positive when provided")
        self.action_encoder = action_encoder
        self.action_gru = nn.GRU(input_size=action_dim, hidden_size=action_dim, batch_first=True)
        self.context_steps = context_steps

    def forward(self, action_tensors: dict[str, Tensor], latent_state: JEPALatentState) -> Tensor:
        """Encode single-step or temporal grounded actions.

        Args:
            action_tensors: Per-step tensors or batched temporal tensors.
            latent_state: Single-step or temporal state latents for the same graph
                batch.

        Returns:
            ``FloatTensor[B, D_a]`` for single-step input or
            ``FloatTensor[B, T_a, D_a]`` for temporal input.
        """

        encoded = self.action_encoder(action_tensors, latent_state)
        if encoded.ndim == 2:
            outputs = self._encode_temporal_actions(encoded.unsqueeze(1))
            return outputs[:, 0]
        if encoded.ndim == 3:
            return self._encode_temporal_actions(encoded)
        raise ValueError(f"Expected action latent rank 2 or 3, got {encoded.ndim}")

    def _encode_temporal_actions(self, encoded: Tensor) -> Tensor:
        window = self.context_steps
        if window is not None and window < 1:
            raise ValueError("context_steps must be positive when provided")
        # Batch-first GRU preserves [batch, time, hidden]: [B, T_a, D_a].
        return _causal_gru_outputs(self.action_gru, encoded, window)


class PooledArgumentEncoder(nn.Module):
    """Compose a head embedding and role-labeled arguments by masked mean pooling.

    The "head" is either an action embedding or predicate embedding. Each
    argument is first shifted by a learned role embedding, then padded arguments
    are ignored by ``arg_mask`` before pooling.

    Args:
        embedding_dim: Shared feature size ``D`` for the head and arguments.
        max_arity: Maximum number of schema arguments. Role ids are expected in
            ``[0, max_arity)`` with ``-1`` reserved for padding.
        hidden_dim: Hidden width of the output MLP. Defaults to ``embedding_dim``.

    Modules:
        role_embedding: Padding-aware embedding for role ids.
        output: MLP mapping concatenated ``[head, pooled_args]`` features from
            ``2 * D`` back to ``D``.
    """

    def __init__(self, embedding_dim: int, max_arity: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = embedding_dim if hidden_dim is None else hidden_dim
        self.role_embedding = _sentinel_embedding(max_arity, embedding_dim)
        self.output = _mlp(embedding_dim * 2, hidden_dim, embedding_dim)

    def forward(self, head_embedding: Tensor, arg_embeddings: Tensor, role_ids: Tensor, arg_mask: Tensor) -> Tensor:
        """Return one composed embedding per item.

        Args:
            head_embedding: ``FloatTensor[B, D]``.
            arg_embeddings: ``FloatTensor[B, A, D]``.
            role_ids: ``LongTensor[B, A]`` with ``-1`` for padded arguments.
            arg_mask: ``BoolTensor[B, A]`` where ``True`` marks valid arguments.

        Returns:
            ``FloatTensor[B, D]``.
        """

        role_features = self.role_embedding(role_ids + 1)
        arg_features = arg_embeddings * (1.0 + torch.tanh(role_features)) + role_features
        pooled_args = _masked_mean(arg_features, arg_mask)
        return self.output(torch.cat([head_embedding, pooled_args], dim=-1))


class RNNArgumentEncoder(nn.Module):
    """Compose a head embedding and schema-ordered arguments with a GRU.

    The GRU sees the sequence ``[head, role_0 + arg_0, role_1 + arg_1, ...]``.
    The order is PDDL schema argument order, not temporal order.

    Args:
        embedding_dim: Shared feature size ``D`` for the head and arguments.
        max_arity: Maximum number of schema arguments. Role ids are expected in
            ``[0, max_arity)`` with ``-1`` reserved for padding.
        hidden_dim: Hidden width of the final output MLP. Defaults to
            ``embedding_dim``.

    Modules:
        role_embedding: Padding-aware embedding for role ids.
        rnn: Batch-first GRU over the head/argument sequence.
        output: MLP mapping the final GRU hidden state to ``D``.
    """

    def __init__(self, embedding_dim: int, max_arity: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = embedding_dim if hidden_dim is None else hidden_dim
        self.role_embedding = _sentinel_embedding(max_arity, embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.output = _mlp(embedding_dim, hidden_dim, embedding_dim)

    def forward(self, head_embedding: Tensor, arg_embeddings: Tensor, role_ids: Tensor, arg_mask: Tensor) -> Tensor:
        """Return one composed embedding per item.

        Args:
            head_embedding: ``FloatTensor[B, D]``.
            arg_embeddings: ``FloatTensor[B, A, D]``.
            role_ids: ``LongTensor[B, A]`` with ``-1`` for padded arguments.
            arg_mask: ``BoolTensor[B, A]`` where ``True`` marks valid arguments.

        Returns:
            ``FloatTensor[B, D]``.
        """

        arg_features = arg_embeddings + self.role_embedding(role_ids + 1)
        sequence = torch.cat([head_embedding.unsqueeze(1), arg_features], dim=1)
        lengths = arg_mask.long().sum(dim=1).add(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return self.output(hidden[-1])


class LatentActionEncoder(nn.Module):
    """Encode grounded actions from action ids and JEPA latent object embeddings.

    This module implements the action encoder ``q_phi``. It gathers action
    argument vectors from ``JEPALatentState.object_latents`` so action encoding
    stays in the same representation space used by the latent predictor.

    Args:
        num_actions: Number of action schemas in the parsed problem.
        max_action_arity: Maximum number of action arguments.
        latent_dim: Input latent object size ``D_z``.
        action_dim: Output action latent size ``D_a``.
        hidden_dim: Hidden width for argument composition MLPs. Defaults to
            ``action_dim``.
        argument_encoder: Argument composition strategy, ``"pooled"`` or
            ``"rnn"``.

    Modules:
        action_embedding: Embedding table ``[num_actions, D_a]``.
        object_projector: Linear projection from latent object size ``D_z`` to
            action size ``D_a``.
        argument_encoder: ``PooledArgumentEncoder`` or ``RNNArgumentEncoder``.
    """

    def __init__(
        self,
        *,
        num_actions: int,
        max_action_arity: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int | None = None,
        argument_encoder: str = "pooled",
    ) -> None:
        super().__init__()
        hidden_dim = action_dim if hidden_dim is None else hidden_dim
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.object_projector = nn.Linear(latent_dim, action_dim)
        self.argument_encoder = _build_argument_encoder(argument_encoder, action_dim, max_action_arity, hidden_dim)

    def forward(self, action_tensors: dict[str, Tensor], latent_state: JEPALatentState) -> Tensor:
        """Encode a batch of grounded actions.

        Args:
            action_tensors: Dictionary produced by ``tensorize_action`` with:
                ``action_id`` as ``LongTensor[]`` or ``LongTensor[B]``;
                ``action_object_indices`` as ``LongTensor[A]`` or
                ``LongTensor[B, A]``; ``action_role_ids`` as ``LongTensor[A]``
                or ``LongTensor[B, A]``; and ``action_arg_mask`` as
                ``BoolTensor[A]`` or ``BoolTensor[B, A]``. With temporal
                context, fields must be batched as ``[B, T]`` or
                ``[B, T, A]``.
            latent_state: Current-state JEPA latents. Its ``object_latents`` are
                gathered by ``object_ids`` and ``object_batch``.

        Returns:
            ``FloatTensor[B, D_a]`` or ``FloatTensor[B, T, D_a]`` action
            latents.
        """

        if latent_state.graph_latent.ndim == 3:
            action_tensors, latent_state, batch_size, time_steps = _flatten_temporal_latent_action_context(
                action_tensors,
                latent_state,
            )
            action_id, object_indices, role_ids, arg_mask = _action_tensor_fields(
                action_tensors,
                batch_size=batch_size * time_steps,
            )
            arg_embeddings = _gather_by_object_id(
                embeddings=latent_state.object_latents,
                object_ids=latent_state.object_ids,
                object_batch=latent_state.object_batch,
                query_object_ids=object_indices,
                query_mask=arg_mask,
            )
            arg_features = self.object_projector(arg_embeddings)
            action_features = self.action_embedding(action_id)
            return self.argument_encoder(action_features, arg_features, role_ids, arg_mask).reshape(
                batch_size, time_steps, -1
            )

        action_id, object_indices, role_ids, arg_mask = _action_tensor_fields(
            action_tensors,
            batch_size=latent_state.graph_latent.size(0),
        )
        arg_embeddings = _gather_by_object_id(
            embeddings=latent_state.object_latents,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
            query_object_ids=object_indices,
            query_mask=arg_mask,
        )
        arg_features = self.object_projector(arg_embeddings)
        action_features = self.action_embedding(action_id)
        return self.argument_encoder(action_features, arg_features, role_ids, arg_mask)


class ApplicabilityHead(nn.Module):
    """Score whether a grounded action is applicable in a latent state.

    The head consumes an already encoded action latent and the current graph
    latent. Optional argument object latents are summarized with learned slot
    embeddings so object context is role/order-aware rather than a plain
    permutation-invariant masked mean.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        max_action_arity: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if max_action_arity < 0:
            raise ValueError("max_action_arity must be non-negative")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must lie in [0, 1)")
        hidden_dim = max(latent_dim + action_dim, latent_dim) if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.max_action_arity = int(max_action_arity)
        self.slot_embedding = nn.Embedding(max(1, max_action_arity), latent_dim)
        self.object_context = _mlp(latent_dim * 3, hidden_dim, latent_dim)
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim * 3 + action_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        graph_latent: Tensor,
        action_latent: Tensor,
        object_latents: Tensor | None = None,
        argument_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one applicability logit per graph/action pair."""

        self._validate_state_action(graph_latent, action_latent)
        object_summary = self._object_summary(graph_latent, object_latents, argument_mask)
        features = torch.cat(
            [
                graph_latent,
                action_latent,
                object_summary,
                (graph_latent - object_summary).abs(),
            ],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)

    def _validate_state_action(self, graph_latent: Tensor, action_latent: Tensor) -> None:
        if graph_latent.ndim != 2:
            raise ValueError("graph_latent must have shape [B, latent_dim]")
        if action_latent.ndim != 2:
            raise ValueError("action_latent must have shape [B, action_dim]")
        if graph_latent.size(0) != action_latent.size(0):
            raise ValueError("graph_latent and action_latent batch size must match")
        if graph_latent.size(-1) != self.latent_dim:
            raise ValueError("graph_latent last dimension must match latent_dim")
        if action_latent.size(-1) != self.action_dim:
            raise ValueError("action_latent last dimension must match action_dim")

    def _object_summary(
        self,
        graph_latent: Tensor,
        object_latents: Tensor | None,
        argument_mask: Tensor | None,
    ) -> Tensor:
        if object_latents is None and argument_mask is None:
            return graph_latent.new_zeros((graph_latent.size(0), self.latent_dim))
        if object_latents is None:
            raise ValueError("object_latents is required when argument_mask is provided")
        if argument_mask is None:
            raise ValueError("argument_mask is required when object_latents is provided")
        if object_latents.ndim != 3:
            raise ValueError("object_latents must have shape [B, A, latent_dim]")
        if argument_mask.ndim != 2:
            raise ValueError("argument_mask must have shape [B, A]")
        if object_latents.size(0) != graph_latent.size(0):
            raise ValueError("object_latents and graph_latent batch size must match")
        if object_latents.size(-1) != self.latent_dim:
            raise ValueError("object_latents last dimension must match latent_dim")
        if argument_mask.shape != object_latents.shape[:2]:
            raise ValueError("object/mask arity dimensions must match")
        if object_latents.size(1) > self.max_action_arity:
            raise ValueError("object/mask arity exceeds max_action_arity")
        mask = argument_mask.to(device=object_latents.device, dtype=object_latents.dtype).unsqueeze(-1)
        arity = object_latents.size(1)
        slot_ids = torch.arange(arity, device=object_latents.device)
        slot_features = self.slot_embedding(slot_ids).to(dtype=object_latents.dtype).unsqueeze(0)
        slot_features = slot_features.expand(object_latents.size(0), -1, -1)
        role_object_features = torch.cat(
            [object_latents, slot_features, object_latents * torch.tanh(slot_features)],
            dim=-1,
        )
        projected = self.object_context(role_object_features) * mask
        counts = mask.sum(dim=1).clamp_min(1.0)
        return projected.sum(dim=1) / counts


@dataclass(frozen=True)
class ActionDecodingSpace:
    """Finite grounding space for action latent decoding.

    Encoded action samples use the compact layout
    ``LongTensor[1 + A_max]``: slot ``0`` stores the action id and slots
    ``1..A_max`` store problem-local object ids for schema arguments. Padded
    argument slots are zero in this compact sample representation and are
    converted to ``-1`` plus a mask by ``samples_to_action_tensors``.
    """

    parsed_problem: ParsedProblem
    action_names: tuple[str, ...]
    object_names: tuple[str, ...]
    objects_by_type: dict[str, tuple[str, ...]]

    @classmethod
    def from_parsed_problem(cls, parsed_problem: ParsedProblem) -> "ActionDecodingSpace":
        object_names = tuple(name for name, _ in sorted(parsed_problem.object_to_id.items(), key=lambda item: item[1]))
        objects_by_type: dict[str, list[str]] = {type_name: [] for type_name in parsed_problem.types}
        for object_name, object_info in parsed_problem.objects.items():
            objects_by_type.setdefault(object_info.type, []).append(object_name)
        return cls(
            parsed_problem=parsed_problem,
            action_names=tuple(
                name for name, _ in sorted(parsed_problem.action_to_id.items(), key=lambda item: item[1])
            ),
            object_names=object_names,
            objects_by_type={type_name: tuple(sorted(names)) for type_name, names in objects_by_type.items()},
        )

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    @property
    def num_objects(self) -> int:
        return len(self.object_names)

    @property
    def max_action_arity(self) -> int:
        return self.parsed_problem.max_action_arity

    @property
    def domain_sizes(self) -> tuple[int, ...]:
        """Return categorical domain sizes for compact action samples.

        Returns:
            Tuple of length ``1 + A_max``. The first entry is ``num_actions``;
            each argument entry is ``num_objects``.
        """

        if self.num_actions <= 0:
            raise ValueError("parsed problem must contain at least one action")
        if self.max_action_arity > 0 and self.num_objects <= 0:
            raise ValueError("actions with arguments require at least one object")
        return (self.num_actions,) + (self.num_objects,) * self.max_action_arity

    def action_name(self, action_id: int) -> str:
        return self.action_names[int(action_id)]

    def action_id(self, action_name: str) -> int:
        return self.parsed_problem.action_to_id[action_name]

    def object_name(self, object_id: int) -> str:
        return self.object_names[int(object_id)]

    def object_id(self, object_name: str) -> int:
        return self.parsed_problem.object_to_id[object_name]

    def valid_object_ids(self, action_id: int, role_id: int, *, device: torch.device | str | None = None) -> Tensor:
        """Return type-compatible object ids for one action argument role.

        Returns:
            ``LongTensor[N_valid]``. The tensor is empty when ``role_id`` is
            outside the selected action schema arity.
        """

        action = self.parsed_problem.actions[self.action_name(action_id)]
        if role_id < 0 or role_id >= action.arity:
            return torch.empty((0,), dtype=torch.long, device=device)
        type_name = action.parameter_types[role_id]
        object_ids = [self.object_id(name) for name in self.objects_by_type.get(type_name, ())]
        return torch.tensor(object_ids, dtype=torch.long, device=device)

    def enumerate_ground_actions(self) -> tuple[GroundAction, ...]:
        actions: list[GroundAction] = []
        for action_name in self.action_names:
            schema = self.parsed_problem.actions[action_name]
            domains = [self.objects_by_type.get(type_name, ()) for type_name in schema.parameter_types]
            if any(len(domain) == 0 for domain in domains):
                continue
            if not domains:
                actions.append(GroundAction(action_name, ()))
                continue
            for arguments in _cartesian_product(domains):
                actions.append(GroundAction(action_name, tuple(arguments)))
        return tuple(actions)

    def ground_action_to_sample(self, action: GroundAction) -> Tensor:
        """Convert a typed ground action to a compact sample.

        Returns:
            ``LongTensor[1 + A_max]`` containing ``[action_id, object_id_0, ...]``.
        """

        schema = self.parsed_problem.actions[action.name]
        if len(action.arguments) != schema.arity:
            raise ValueError(f"Action {action.name} expects {schema.arity} arguments, got {len(action.arguments)}")
        sample = torch.zeros((1 + self.max_action_arity,), dtype=torch.long)
        sample[0] = self.action_id(action.name)
        for role_id, object_name in enumerate(action.arguments):
            sample[1 + role_id] = self.object_id(object_name)
        return sample

    def sample_to_ground_action(self, sample: Sequence[int] | Tensor) -> GroundAction:
        """Convert a compact sample back to a typed ground action.

        Args:
            sample: Sequence or tensor of length ``1 + A_max``.
        """

        values = [int(v) for v in (sample.tolist() if isinstance(sample, Tensor) else sample)]
        if len(values) != 1 + self.max_action_arity:
            raise ValueError(f"Expected sample length {1 + self.max_action_arity}, got {len(values)}")
        action_id = values[0]
        action_name = self.action_name(action_id)
        schema = self.parsed_problem.actions[action_name]
        arguments = tuple(self.object_name(values[1 + role_id]) for role_id in range(schema.arity))
        return GroundAction(action_name, arguments)

    def samples_to_action_tensors(
        self,
        samples: Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Convert compact samples to action encoder tensors.

        Args:
            samples: ``LongTensor[1 + A_max]`` or ``LongTensor[B, 1 + A_max]``.

        Returns:
            Dictionary with ``action_id`` shaped ``[B]`` and
            ``action_object_indices``, ``action_role_ids``, ``action_arg_mask``
            shaped ``[B, A_max]``.
        """

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.ndim != 2 or samples.size(1) != 1 + self.max_action_arity:
            raise ValueError(
                f"Expected samples with shape [B, {1 + self.max_action_arity}], got {tuple(samples.shape)}"
            )

        resolved_device = samples.device if device is None else torch.device(device)
        samples = samples.to(device=resolved_device, dtype=torch.long)
        batch_size = samples.size(0)
        action_ids = samples[:, 0].clone()
        object_indices = torch.full((batch_size, self.max_action_arity), -1, dtype=torch.long, device=resolved_device)
        role_ids = torch.full_like(object_indices, -1)
        arg_mask = torch.zeros((batch_size, self.max_action_arity), dtype=torch.bool, device=resolved_device)

        for batch_idx in range(batch_size):
            action_id = int(action_ids[batch_idx].item())
            action = self.parsed_problem.actions[self.action_name(action_id)]
            for role_id in range(action.arity):
                object_id = samples[batch_idx, 1 + role_id]
                valid_ids = self.valid_object_ids(action_id, role_id, device=resolved_device)
                if not bool((valid_ids == object_id).any()):
                    raise ValueError(
                        f"Object id {int(object_id.item())} is invalid for action {action.name} role {role_id}"
                    )
                object_indices[batch_idx, role_id] = object_id
                role_ids[batch_idx, role_id] = role_id
                arg_mask[batch_idx, role_id] = True

        return {
            "action_id": action_ids,
            "action_object_indices": object_indices,
            "action_role_ids": role_ids,
            "action_arg_mask": arg_mask,
        }

    def action_tensors_for_ground_actions(
        self,
        actions: Sequence[GroundAction],
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Tensorize a batch of typed ground actions.

        Returns:
            Dictionary with ``action_id`` shaped ``[B]`` and argument fields
            shaped ``[B, A_max]``.
        """

        tensors = [tensorize_action(self.parsed_problem, action, max_arity=self.max_action_arity) for action in actions]
        if not tensors:
            raise ValueError("Cannot tensorize an empty action list")
        resolved_device = torch.device(device) if device is not None else None
        stacked = {key: torch.stack([item[key] for item in tensors]) for key in tensors[0]}
        if resolved_device is None:
            return stacked
        return {key: value.to(device=resolved_device) for key, value in stacked.items()}


class ActionSamplingFamily:
    """Conditional categorical sampler for grounded actions.

    The family samples an action id first and then samples each active role from
    the object ids type-compatible with that action-role pair.
    """

    def __init__(
        self,
        decoding_space: ActionDecodingSpace,
        *,
        smoothing: float = 0.7,
        dirichlet_prior: float = 0.0,
        device: torch.device | str | None = None,
    ) -> None:
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("smoothing must lie in (0, 1]")
        if dirichlet_prior < 0.0:
            raise ValueError("dirichlet_prior must be non-negative")
        self.decoding_space = decoding_space
        self.domain_sizes = decoding_space.domain_sizes
        self.smoothing = float(smoothing)
        self.dirichlet_prior = float(dirichlet_prior)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.action_probs = torch.full(
            (decoding_space.num_actions,),
            1.0 / decoding_space.num_actions,
            dtype=torch.float32,
            device=self.device,
        )
        self.valid_object_ids_by_role: dict[tuple[int, int], Tensor] = {}
        self.arg_probs: dict[tuple[int, int], Tensor] = {}
        for action_id, action_name in enumerate(decoding_space.action_names):
            action = decoding_space.parsed_problem.actions[action_name]
            for role_id in range(action.arity):
                valid_ids = decoding_space.valid_object_ids(action_id, role_id, device=self.device)
                if valid_ids.numel() == 0:
                    raise ValueError(f"Action {action.name} role {role_id} has no valid objects")
                key = (action_id, role_id)
                self.valid_object_ids_by_role[key] = valid_ids
                self.arg_probs[key] = torch.full(
                    (valid_ids.numel(),),
                    1.0 / valid_ids.numel(),
                    dtype=torch.float32,
                    device=self.device,
                )

    def sample(self, num_samples: int, generator: torch.Generator) -> Tensor:
        """Draw compact action samples.

        Returns:
            ``LongTensor[N, 1 + A_max]`` where ``N == num_samples``.
        """

        action_ids = torch.multinomial(self.action_probs, num_samples, replacement=True, generator=generator)
        samples = torch.zeros(
            (num_samples, 1 + self.decoding_space.max_action_arity), dtype=torch.long, device=self.device
        )
        samples[:, 0] = action_ids
        for sample_idx, action_id_tensor in enumerate(action_ids):
            action_id = int(action_id_tensor.item())
            action = self.decoding_space.parsed_problem.actions[self.decoding_space.action_name(action_id)]
            for role_id in range(action.arity):
                key = (action_id, role_id)
                local_idx = torch.multinomial(self.arg_probs[key], 1, replacement=True, generator=generator)
                samples[sample_idx, 1 + role_id] = self.valid_object_ids_by_role[key][int(local_idx.item())]
        return samples

    def update(self, elite_samples: Tensor, elite_weights: Tensor) -> None:
        """Update categorical factors from weighted elite compact samples.

        Args:
            elite_samples: ``LongTensor[E, 1 + A_max]``.
            elite_weights: ``FloatTensor[E]``.
        """

        elite_samples = elite_samples.to(self.device)
        elite_weights = elite_weights.to(device=self.device, dtype=torch.float32)
        total_w = elite_weights.sum()
        action_counts = torch.bincount(
            elite_samples[:, 0],
            weights=elite_weights,
            minlength=self.decoding_space.num_actions,
        ).to(torch.float32)
        action_posterior = (action_counts + self.dirichlet_prior) / (
            total_w + self.decoding_space.num_actions * self.dirichlet_prior
        )
        updated_action = self.smoothing * action_posterior + (1.0 - self.smoothing) * self.action_probs
        self.action_probs = updated_action / updated_action.sum()

        for key, previous in list(self.arg_probs.items()):
            action_id, role_id = key
            action_mask = elite_samples[:, 0] == action_id
            if not bool(action_mask.any()):
                continue
            selected = elite_samples[action_mask, 1 + role_id]
            weights = elite_weights[action_mask]
            valid_ids = self.valid_object_ids_by_role[key]
            local_indices = _map_values_to_local_indices(selected, valid_ids)
            counts = torch.bincount(local_indices, weights=weights, minlength=valid_ids.numel()).to(torch.float32)
            role_w = weights.sum()
            posterior = (counts + self.dirichlet_prior) / (role_w + valid_ids.numel() * self.dirichlet_prior)
            updated = self.smoothing * posterior + (1.0 - self.smoothing) * previous
            self.arg_probs[key] = updated / updated.sum()

    def marginals(self) -> tuple[Tensor, ...]:
        """Return current independent marginals for optimizer diagnostics.

        Returns:
            Tuple of length ``1 + A_max``. Entry ``0`` is
            ``FloatTensor[num_actions]``; argument entries are
            ``FloatTensor[num_objects]``.
        """

        marginals: list[Tensor] = [self.action_probs.detach().clone()]
        for role_id in range(self.decoding_space.max_action_arity):
            role_marginal = torch.zeros((self.decoding_space.num_objects,), dtype=torch.float32, device=self.device)
            for action_id, action_name in enumerate(self.decoding_space.action_names):
                action_prob = self.action_probs[action_id]
                action = self.decoding_space.parsed_problem.actions[action_name]
                if role_id >= action.arity:
                    if role_marginal.numel() > 0:
                        role_marginal[0] = role_marginal[0] + action_prob
                    continue
                key = (action_id, role_id)
                role_marginal[self.valid_object_ids_by_role[key]] += action_prob * self.arg_probs[key]
            marginals.append(role_marginal.detach().clone())
        return tuple(marginals)

    def mode(self) -> Tensor:
        """Return the most likely compact action sample.

        Returns:
            ``LongTensor[1 + A_max]``.
        """

        action_id = int(torch.argmax(self.action_probs).item())
        sample = torch.zeros((1 + self.decoding_space.max_action_arity,), dtype=torch.long, device=self.device)
        sample[0] = action_id
        action = self.decoding_space.parsed_problem.actions[self.decoding_space.action_name(action_id)]
        for role_id in range(action.arity):
            key = (action_id, role_id)
            local_idx = int(torch.argmax(self.arg_probs[key]).item())
            sample[1 + role_id] = self.valid_object_ids_by_role[key][local_idx]
        return sample

    def is_degenerate(self, tol: float) -> bool:
        if (1.0 - self.action_probs.max().item()) > tol:
            return False
        action_id = int(torch.argmax(self.action_probs).item())
        action = self.decoding_space.parsed_problem.actions[self.decoding_space.action_name(action_id)]
        return all((1.0 - self.arg_probs[(action_id, role_id)].max().item()) <= tol for role_id in range(action.arity))


class StructuredActionSequenceSamplingFamily:
    """Conditional categorical sampler for fixed-horizon grounded action sequences.

    Samples use compact layout ``LongTensor[N, H, 1 + A_max]``. Each timestep
    owns an independent action categorical, while object-role categoricals are
    conditioned on the selected action id at that timestep.
    """

    def __init__(
        self,
        decoding_space: ActionDecodingSpace,
        horizon: int,
        *,
        smoothing: float = 0.7,
        dirichlet_prior: float = 0.0,
        device: torch.device | str | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("smoothing must lie in (0, 1]")
        if dirichlet_prior < 0.0:
            raise ValueError("dirichlet_prior must be non-negative")
        self.decoding_space = decoding_space
        self.horizon = int(horizon)
        self.smoothing = float(smoothing)
        self.dirichlet_prior = float(dirichlet_prior)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.action_probs = torch.full(
            (self.horizon, decoding_space.num_actions),
            1.0 / decoding_space.num_actions,
            dtype=torch.float32,
            device=self.device,
        )
        self.valid_object_ids_by_role: dict[tuple[int, int], Tensor] = {}
        self.arg_probs: dict[tuple[int, int, int], Tensor] = {}
        for action_id, action_name in enumerate(decoding_space.action_names):
            action = decoding_space.parsed_problem.actions[action_name]
            for role_id in range(action.arity):
                valid_ids = decoding_space.valid_object_ids(action_id, role_id, device=self.device)
                if valid_ids.numel() == 0:
                    raise ValueError(f"Action {action.name} role {role_id} has no valid objects")
                self.valid_object_ids_by_role[(action_id, role_id)] = valid_ids
                for step_idx in range(self.horizon):
                    self.arg_probs[(step_idx, action_id, role_id)] = torch.full(
                        (valid_ids.numel(),),
                        1.0 / valid_ids.numel(),
                        dtype=torch.float32,
                        device=self.device,
                    )

    def sample(self, num_samples: int, generator: torch.Generator) -> Tensor:
        """Draw compact sequence samples.

        Returns:
            ``LongTensor[N, H, 1 + A_max]``.
        """

        samples = torch.zeros(
            (num_samples, self.horizon, 1 + self.decoding_space.max_action_arity),
            dtype=torch.long,
            device=self.device,
        )
        for step_idx in range(self.horizon):
            action_ids = torch.multinomial(
                self.action_probs[step_idx],
                num_samples,
                replacement=True,
                generator=generator,
            )
            samples[:, step_idx, 0] = action_ids
            for sample_idx, action_id_tensor in enumerate(action_ids):
                action_id = int(action_id_tensor.item())
                action = self.decoding_space.parsed_problem.actions[self.decoding_space.action_name(action_id)]
                for role_id in range(action.arity):
                    key = (step_idx, action_id, role_id)
                    local_idx = torch.multinomial(self.arg_probs[key], 1, replacement=True, generator=generator)
                    samples[sample_idx, step_idx, 1 + role_id] = self.valid_object_ids_by_role[(action_id, role_id)][
                        int(local_idx.item())
                    ]
        return samples

    def update(self, elite_samples: Tensor, elite_weights: Tensor) -> None:
        """Update sequence categoricals from weighted elite compact samples.

        Args:
            elite_samples: ``LongTensor[E, H, 1 + A_max]``.
            elite_weights: ``FloatTensor[E]``.
        """

        elite_samples = elite_samples.to(self.device)
        elite_weights = elite_weights.to(device=self.device, dtype=torch.float32)
        total_w = elite_weights.sum()
        for step_idx in range(self.horizon):
            action_counts = torch.bincount(
                elite_samples[:, step_idx, 0],
                weights=elite_weights,
                minlength=self.decoding_space.num_actions,
            ).to(torch.float32)
            action_posterior = (action_counts + self.dirichlet_prior) / (
                total_w + self.decoding_space.num_actions * self.dirichlet_prior
            )
            updated_action = self.smoothing * action_posterior + (1.0 - self.smoothing) * self.action_probs[step_idx]
            self.action_probs[step_idx] = updated_action / updated_action.sum()

            for action_id, action_name in enumerate(self.decoding_space.action_names):
                action = self.decoding_space.parsed_problem.actions[action_name]
                action_mask = elite_samples[:, step_idx, 0] == action_id
                if not bool(action_mask.any()):
                    continue
                for role_id in range(action.arity):
                    key = (step_idx, action_id, role_id)
                    selected = elite_samples[action_mask, step_idx, 1 + role_id]
                    weights = elite_weights[action_mask]
                    valid_ids = self.valid_object_ids_by_role[(action_id, role_id)]
                    local_indices = _map_values_to_local_indices(selected, valid_ids)
                    counts = torch.bincount(local_indices, weights=weights, minlength=valid_ids.numel()).to(
                        torch.float32
                    )
                    role_w = weights.sum()
                    posterior = (counts + self.dirichlet_prior) / (role_w + valid_ids.numel() * self.dirichlet_prior)
                    updated = self.smoothing * posterior + (1.0 - self.smoothing) * self.arg_probs[key]
                    self.arg_probs[key] = updated / updated.sum()

    def marginals(self) -> tuple[Tensor, ...]:
        """Return per-timestep action marginals for diagnostics."""

        return tuple(self.action_probs[step_idx].detach().clone() for step_idx in range(self.horizon))

    def mode(self) -> Tensor:
        """Return the modal compact sequence sample.

        Returns:
            ``LongTensor[H, 1 + A_max]``.
        """

        sample = torch.zeros(
            (self.horizon, 1 + self.decoding_space.max_action_arity),
            dtype=torch.long,
            device=self.device,
        )
        for step_idx in range(self.horizon):
            action_id = int(torch.argmax(self.action_probs[step_idx]).item())
            sample[step_idx, 0] = action_id
            action = self.decoding_space.parsed_problem.actions[self.decoding_space.action_name(action_id)]
            for role_id in range(action.arity):
                key = (step_idx, action_id, role_id)
                local_idx = int(torch.argmax(self.arg_probs[key]).item())
                sample[step_idx, 1 + role_id] = self.valid_object_ids_by_role[(action_id, role_id)][local_idx]
        return sample

    def is_degenerate(self, tol: float) -> bool:
        for step_idx in range(self.horizon):
            if (1.0 - self.action_probs[step_idx].max().item()) > tol:
                return False
            action_id = int(torch.argmax(self.action_probs[step_idx]).item())
            action = self.decoding_space.parsed_problem.actions[self.decoding_space.action_name(action_id)]
            for role_id in range(action.arity):
                if (1.0 - self.arg_probs[(step_idx, action_id, role_id)].max().item()) > tol:
                    return False
        return True


class ActionDecoder:
    """Decode action latents into grounded actions.

    The decoder currently operates on one graph at a time. Candidate grounded
    actions are encoded with the configured action encoder and scored against a
    target action latent.
    """

    def __init__(
        self,
        *,
        parsed_problem: ParsedProblem,
        action_encoder: nn.Module,
        method: Literal["exact", "cem", "mppi"] = "exact",
        metric: Literal["l2", "cosine"] = "l2",
        num_samples: int | None = None,
        elite_frac: float = 0.1,
        max_iters: int = 30,
        smoothing: float = 0.7,
        dirichlet_prior: float = 0.0,
        temperature: float = 1.0,
        decode_chunk_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        if method not in ("exact", "cem", "mppi"):
            raise ValueError(f"Unknown decoder method: {method}")
        if metric not in ("l2", "cosine"):
            raise ValueError(f"Unknown decoder metric: {metric}")
        self.space = ActionDecodingSpace.from_parsed_problem(parsed_problem)
        self.action_encoder = action_encoder
        self.method = method
        self.metric = metric
        self.num_samples = num_samples
        self.elite_frac = float(elite_frac)
        self.max_iters = int(max_iters)
        self.smoothing = float(smoothing)
        self.dirichlet_prior = float(dirichlet_prior)
        self.temperature = float(temperature)
        if decode_chunk_size is not None and decode_chunk_size < 1:
            raise ValueError("decode_chunk_size must be positive when provided")
        self.decode_chunk_size = None if decode_chunk_size is None else int(decode_chunk_size)
        self.seed = seed

    def decode(self, target_action_latent: Tensor, latent_state: JEPALatentState) -> GroundAction:
        """Decode one latent vector to a typed ground action.

        Args:
            target_action_latent: ``FloatTensor[D_a]`` or ``FloatTensor[1, D_a]``.
            latent_state: Single-graph JEPA latent state.
        """

        tensors = self.decode_tensors(target_action_latent, latent_state)
        sample = torch.cat([tensors["action_id"].view(1), tensors["action_object_indices"].view(-1)])
        return self.space.sample_to_ground_action(sample)

    def decode_tensors(self, target_action_latent: Tensor, latent_state: JEPALatentState) -> dict[str, Tensor]:
        """Decode one latent vector to action encoder tensors.

        Args:
            target_action_latent: ``FloatTensor[D_a]`` or ``FloatTensor[1, D_a]``.
            latent_state: Single-graph JEPA latent state.

        Returns:
            Tensor dictionary with ``action_id`` shaped ``[1]`` and argument
            fields shaped ``[1, A_max]``.
        """

        if self.method == "exact":
            return self._decode_exact(target_action_latent, latent_state)
        return self._decode_with_ce(target_action_latent, latent_state)

    def _decode_exact(self, target_action_latent: Tensor, latent_state: JEPALatentState) -> dict[str, Tensor]:
        self._validate_single_graph(latent_state)
        actions = self.space.enumerate_ground_actions()
        if not actions:
            raise RuntimeError("No type-valid grounded actions are available")
        device = target_action_latent.device
        best_idx = 0
        best_score = torch.tensor(float("-inf"), device=device)
        chunk_size = self.decode_chunk_size or len(actions)
        for start in range(0, len(actions), chunk_size):
            chunk = actions[start : start + chunk_size]
            action_tensors = self.space.action_tensors_for_ground_actions(chunk, device=device)
            with torch.no_grad():
                repeated_latent_state = _repeat_action_latent_state(latent_state, len(chunk), device=device)
                latents = self.action_encoder(action_tensors, repeated_latent_state)
                scores = self._score_latents(latents, target_action_latent)
            chunk_score, chunk_idx = torch.max(scores, dim=0)
            if chunk_score > best_score:
                best_score = chunk_score
                best_idx = start + int(chunk_idx.item())
        return self.space.action_tensors_for_ground_actions([actions[best_idx]], device=device)

    def _decode_with_ce(self, target_action_latent: Tensor, latent_state: JEPALatentState) -> dict[str, Tensor]:
        self._validate_single_graph(latent_state)
        device = target_action_latent.device
        family = ActionSamplingFamily(
            self.space,
            smoothing=self.smoothing,
            dirichlet_prior=self.dirichlet_prior,
            device=device,
        )
        elite_weighting = SoftmaxWeighting(self.temperature) if self.method == "mppi" else None

        def score_fn(samples: Tensor) -> Tensor:
            action_tensors = self.space.samples_to_action_tensors(samples, device=device)
            with torch.no_grad():
                repeated_latent_state = _repeat_action_latent_state(latent_state, samples.size(0), device=device)
                latents = self.action_encoder(action_tensors, repeated_latent_state)
                return self._score_latents(latents, target_action_latent)

        result = cross_entropy_optimize(
            domain_sizes=family.domain_sizes,
            score_fn=score_fn,
            num_samples=self.num_samples,
            elite_frac=self.elite_frac,
            max_iters=self.max_iters,
            sampling_family=family,
            elite_weighting=elite_weighting,
            seed=self.seed,
            device=device,
        )
        best_sample = torch.tensor(result.best_x, dtype=torch.long, device=device)
        return self.space.samples_to_action_tensors(best_sample, device=device)

    def _score_latents(self, candidate_latents: Tensor, target_action_latent: Tensor) -> Tensor:
        target = target_action_latent
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if target.ndim != 2 or target.size(0) != 1:
            raise ValueError("target_action_latent must have shape [D] or [1, D]")
        if self.metric == "l2":
            return -((candidate_latents - target) ** 2).sum(dim=-1)
        candidate_norm = torch.nn.functional.normalize(candidate_latents, dim=-1)
        target_norm = torch.nn.functional.normalize(target, dim=-1)
        return (candidate_norm * target_norm).sum(dim=-1)

    @staticmethod
    def _validate_single_graph(latent_state: JEPALatentState) -> None:
        if latent_state.graph_latent.size(0) != 1:
            raise ValueError("ActionLatentDecoder currently decodes one graph at a time")


class ResidualMLPLatentPredictorG(nn.Module):
    """Predict next graph and object latents with residual MLP updates.

    This is the default single-application latent predictor ``g`` used
    recursively by k-step rollouts. It predicts additive deltas for graph and
    object latents conditioned on the encoded action.

    Args:
        latent_dim: JEPA latent size ``D_z``.
        action_dim: Action latent size ``D_a``. Defaults to ``latent_dim``.
        hidden_dim: Hidden width of graph/object update MLPs. Defaults to
            ``latent_dim``.

    Modules:
        graph_delta: MLP ``[D_z + D_a] -> D_z``.
        object_delta: MLP ``[D_z(object) + D_z(graph) + D_a] -> D_z``.
    """

    def __init__(self, latent_dim: int, action_dim: int | None = None, hidden_dim: int | None = None) -> None:
        super().__init__()
        action_dim = latent_dim if action_dim is None else action_dim
        hidden_dim = latent_dim if hidden_dim is None else hidden_dim
        self.graph_delta = _mlp(latent_dim + action_dim, hidden_dim, latent_dim)
        self.object_delta = _mlp(latent_dim * 2 + action_dim, hidden_dim, latent_dim)

    def forward(self, latent_state: JEPALatentState, action_latent: Tensor) -> JEPALatentState:
        """Apply the latent predictor once.

        Args:
            latent_state: Current state with graph latents ``[B, D_z]`` or
                ``[B, T, D_z]`` and object latents ``[N_obj, D_z]`` or
                ``[N_obj, T, D_z]``.
            action_latent: Encoded action ``FloatTensor[B, D_a]`` or
                ``FloatTensor[B, T, D_a]``.

        Returns:
            ``JEPALatentState`` with updated graph/object latents and unchanged
            ``object_ids``/``object_batch`` tensors.
        """

        if latent_state.graph_latent.ndim == 3:
            return self._forward_temporal(latent_state, action_latent)
        graph_delta = self.graph_delta(torch.cat([latent_state.graph_latent, action_latent], dim=-1))
        object_action = action_latent[latent_state.object_batch]
        object_graph = latent_state.graph_latent[latent_state.object_batch]
        object_delta = self.object_delta(torch.cat([latent_state.object_latents, object_graph, object_action], dim=-1))
        return JEPALatentState(
            graph_latent=latent_state.graph_latent + graph_delta,
            object_latents=latent_state.object_latents + object_delta,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
        )

    def _forward_temporal(self, latent_state: JEPALatentState, action_latent: Tensor) -> JEPALatentState:
        if action_latent.ndim != 3:
            raise ValueError("Temporal latent prediction requires action_latent shape [B, T, D_a]")
        graph_delta = self.graph_delta(torch.cat([latent_state.graph_latent, action_latent], dim=-1))
        object_action = action_latent[latent_state.object_batch]
        object_graph = latent_state.graph_latent[latent_state.object_batch]
        object_delta = self.object_delta(torch.cat([latent_state.object_latents, object_graph, object_action], dim=-1))
        return JEPALatentState(
            graph_latent=latent_state.graph_latent + graph_delta,
            object_latents=latent_state.object_latents + object_delta,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
        )


class GRULatentPredictorG(nn.Module):
    """Predict next graph and object latents with GRUCell updates.

    This alternative predictor uses the current latent as the GRU hidden state
    and an action-conditioned input as the GRU input. It is a structured
    single-application predictor used recursively by k-step rollouts.

    Args:
        latent_dim: JEPA latent size ``D_z``.
        action_dim: Action latent size ``D_a``. Defaults to ``latent_dim``.
        hidden_dim: Hidden width of graph/object input MLPs. Defaults to
            ``latent_dim``.

    Modules:
        graph_input: MLP ``[D_z + D_a] -> D_z`` for the graph GRU input.
        object_input: MLP ``[D_z(graph) + D_a] -> D_z`` for each object GRU
            input.
        graph_cell: ``GRUCell(D_z, D_z)`` for graph latents.
        object_cell: ``GRUCell(D_z, D_z)`` for object latents.
    """

    def __init__(self, latent_dim: int, action_dim: int | None = None, hidden_dim: int | None = None) -> None:
        super().__init__()
        action_dim = latent_dim if action_dim is None else action_dim
        hidden_dim = latent_dim if hidden_dim is None else hidden_dim
        self.graph_input = _mlp(latent_dim + action_dim, hidden_dim, latent_dim)
        self.object_input = _mlp(latent_dim + action_dim, hidden_dim, latent_dim)
        self.graph_cell = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.object_cell = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)

    def forward(self, latent_state: JEPALatentState, action_latent: Tensor) -> JEPALatentState:
        """Apply the latent predictor once.

        Args:
            latent_state: Current state with graph latents ``[B, D_z]`` and
                object latents ``[N_obj, D_z]``.
            action_latent: Encoded action ``FloatTensor[B, D_a]``.

        Returns:
            ``JEPALatentState`` with updated graph/object latents and unchanged
            ``object_ids``/``object_batch`` tensors.
        """

        if latent_state.graph_latent.ndim == 3:
            return self._forward_temporal(latent_state, action_latent)
        graph_input = self.graph_input(torch.cat([latent_state.graph_latent, action_latent], dim=-1))
        graph_latent = self.graph_cell(graph_input, latent_state.graph_latent)
        object_action = action_latent[latent_state.object_batch]
        object_graph = graph_latent[latent_state.object_batch]
        object_input = self.object_input(torch.cat([object_graph, object_action], dim=-1))
        object_latents = self.object_cell(object_input, latent_state.object_latents)
        return JEPALatentState(
            graph_latent=graph_latent,
            object_latents=object_latents,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
        )

    def _forward_temporal(self, latent_state: JEPALatentState, action_latent: Tensor) -> JEPALatentState:
        if action_latent.ndim != 3:
            raise ValueError("Temporal latent prediction requires action_latent shape [B, T, D_a]")
        batch_size, time_steps, latent_dim = latent_state.graph_latent.shape
        graph_input = self.graph_input(torch.cat([latent_state.graph_latent, action_latent], dim=-1))
        graph_latent = self.graph_cell(
            graph_input.reshape(batch_size * time_steps, latent_dim),
            latent_state.graph_latent.reshape(batch_size * time_steps, latent_dim),
        ).reshape(batch_size, time_steps, latent_dim)
        object_count = latent_state.object_latents.size(0)
        object_action = action_latent[latent_state.object_batch]
        object_graph = graph_latent[latent_state.object_batch]
        object_input = self.object_input(torch.cat([object_graph, object_action], dim=-1))
        object_latents = self.object_cell(
            object_input.reshape(object_count * time_steps, latent_dim),
            latent_state.object_latents.reshape(object_count * time_steps, latent_dim),
        ).reshape(object_count, time_steps, latent_dim)
        return JEPALatentState(
            graph_latent=graph_latent,
            object_latents=object_latents,
            object_ids=latent_state.object_ids,
            object_batch=latent_state.object_batch,
        )


def build_action_encoder(kind: str = "pooled", **kwargs) -> nn.Module:
    """Build the latent-space grounded-action encoder."""

    return LatentActionEncoder(argument_encoder=kind, **kwargs)


def build_latent_predictor(kind: str = "mlp", **kwargs) -> nn.Module:
    """Build a latent predictor.

    Args:
        kind: ``"mlp"`` for :class:`ResidualMLPLatentPredictorG` or ``"gru"``
            for :class:`GRULatentPredictorG`.
        **kwargs: Constructor arguments forwarded to the selected predictor.

    Returns:
        A predictor with ``forward(latent_state, action_latent)`` returning a
        ``JEPALatentState``.
    """

    if kind == "mlp":
        return ResidualMLPLatentPredictorG(**kwargs)
    if kind == "gru":
        return GRULatentPredictorG(**kwargs)
    raise ValueError(f"Unknown latent predictor kind: {kind}")


def _causal_gru_outputs(gru: nn.GRU, inputs: Tensor, context_steps: int | None) -> Tensor:
    """Run a batch-first GRU with an optional causal rolling context.

    Args:
        gru: GRU with ``batch_first=True``.
        inputs: ``FloatTensor[B, T, D]``.
        context_steps: ``None`` for full prefix context, otherwise the maximum
            number of frames visible to each output.

    Returns:
        ``FloatTensor[B, T, D]`` with output ``t`` depending only on frames
        ``max(0, t + 1 - context_steps)..t``.
    """

    if context_steps is None or context_steps >= inputs.size(1):
        # Full-prefix mode consumes [B, T, D] and returns [B, T, D].
        outputs, _ = gru(inputs)
        return outputs
    outputs = []
    for step_idx in range(inputs.size(1)):
        start_idx = max(0, step_idx + 1 - context_steps)
        # Window input is [B, W_t, D], where W_t <= context_steps.
        window_output, _ = gru(inputs[:, start_idx : step_idx + 1])
        # Keep the last output for this causal window: [B, W_t, D] -> [B, D].
        outputs.append(window_output[:, -1])
    # Reinsert time by stacking per-step outputs: T * [B, D] -> [B, T, D].
    return torch.stack(outputs, dim=1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _sentinel_embedding(max_value: int, embedding_dim: int) -> nn.Embedding:
    return nn.Embedding(max_value + 2, embedding_dim, padding_idx=0)


def _build_argument_encoder(kind: str, embedding_dim: int, max_arity: int, hidden_dim: int) -> nn.Module:
    if kind == "pooled":
        return PooledArgumentEncoder(embedding_dim, max_arity, hidden_dim)
    if kind == "rnn":
        return RNNArgumentEncoder(embedding_dim, max_arity, hidden_dim)
    raise ValueError(f"Unknown argument encoder kind: {kind}")


def _flatten_temporal_latent_action_context(
    action_tensors: dict[str, Tensor],
    latent_state: JEPALatentState,
) -> tuple[dict[str, Tensor], JEPALatentState, int, int]:
    batch_size, time_steps = latent_state.graph_latent.shape[:2]
    return (
        _flatten_temporal_action_tensors(action_tensors, batch_size=batch_size, time_steps=time_steps),
        flatten_temporal_latent_state(latent_state),
        batch_size,
        time_steps,
    )


def _flatten_temporal_action_tensors(
    action_tensors: dict[str, Tensor],
    *,
    batch_size: int,
    time_steps: int,
) -> dict[str, Tensor]:
    action_id = action_tensors["action_id"]
    if action_id.ndim != 2:
        raise ValueError("Temporal action tensors must be batched with action_id shape [B, T]")
    if tuple(action_id.shape) != (batch_size, time_steps):
        raise ValueError(f"Expected action_id shape {(batch_size, time_steps)}, got {tuple(action_id.shape)}")
    flattened = {"action_id": action_id.reshape(batch_size * time_steps)}
    for key in ("action_object_indices", "action_role_ids", "action_arg_mask"):
        value = action_tensors[key]
        if value.ndim != 3 or tuple(value.shape[:2]) != (batch_size, time_steps):
            raise ValueError(f"Expected {key} shape [B, T, A], got {tuple(value.shape)}")
        flattened[key] = value.reshape(batch_size * time_steps, value.size(-1))
    return flattened


def _as_batch_vector(value: Tensor, batch_size: int) -> Tensor:
    """Normalize a scalar or ``[B]`` tensor to ``[B]``."""

    if value.ndim == 0:
        value = value.unsqueeze(0)
    if value.ndim != 1:
        raise ValueError(f"Expected scalar or rank-1 tensor, got shape {tuple(value.shape)}")
    if value.size(0) == 1 and batch_size > 1:
        value = value.expand(batch_size)
    if value.size(0) != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {value.size(0)}")
    return value


def _as_batch_matrix(value: Tensor, batch_size: int) -> Tensor:
    """Normalize an ``[A]`` or ``[B, A]`` tensor to ``[B, A]``."""

    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"Expected rank-1 or rank-2 tensor, got shape {tuple(value.shape)}")
    if value.size(0) == 1 and batch_size > 1:
        value = value.expand(batch_size, -1)
    if value.size(0) != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {value.size(0)}")
    return value


def _action_tensor_fields(
    action_tensors: dict[str, Tensor],
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract normalized action tensor fields.

    Returns:
        ``action_id`` as ``LongTensor[B]``; ``object_indices`` and
        ``role_ids`` as ``LongTensor[B, A_max]``; ``arg_mask`` as
        ``BoolTensor[B, A_max]``.
    """

    action_id = _as_batch_vector(action_tensors["action_id"], batch_size).long()
    object_indices = _as_batch_matrix(action_tensors["action_object_indices"], batch_size).long()
    role_ids = _as_batch_matrix(action_tensors["action_role_ids"], batch_size).long()
    arg_mask = _as_batch_matrix(action_tensors["action_arg_mask"], batch_size).bool()
    return action_id, object_indices, role_ids, arg_mask


def _gather_by_object_id(
    *,
    embeddings: Tensor,
    object_ids: Tensor,
    object_batch: Tensor,
    query_object_ids: Tensor,
    query_mask: Tensor,
) -> Tensor:
    """Gather per-argument object embeddings from a PyG mini-batch.

    Args:
        embeddings: Object features ``FloatTensor[N_obj, D]``.
        object_ids: Problem-local object ids ``LongTensor[N_obj]``.
        object_batch: Graph index per object ``LongTensor[N_obj]``.
        query_object_ids: Requested object ids ``LongTensor[B, A_max]``.
        query_mask: Valid-argument mask ``BoolTensor[B, A_max]``.

    Returns:
        ``FloatTensor[B, A_max, D]``. Masked argument slots are zeros.
    """

    batch_size, arity = query_object_ids.shape
    gathered = embeddings.new_zeros((batch_size, arity, embeddings.size(-1)))
    for batch_idx in range(batch_size):
        for arg_idx in range(arity):
            if not bool(query_mask[batch_idx, arg_idx]):
                continue
            object_id = query_object_ids[batch_idx, arg_idx]
            # Object ids are problem-local, so lookup is scoped by both graph
            # index and object id when operating on PyG mini-batches.
            matches = (object_batch == batch_idx) & (object_ids == object_id)
            match_indices = matches.nonzero(as_tuple=False).flatten()
            if match_indices.numel() != 1:
                raise KeyError(
                    f"Expected one object id {int(object_id.item())} in batch {batch_idx}, got {match_indices.numel()}"
                )
            gathered[batch_idx, arg_idx] = embeddings[match_indices[0]]
    return gathered


def _repeat_action_latent_state(
    latent_state: JEPALatentState,
    batch_size: int,
    *,
    device: torch.device | str,
) -> JEPALatentState:
    """Repeat a single-graph action latent state to score candidate actions.

    Returns:
        Latent state with graph batch size ``batch_size`` and object rows
        repeated so each candidate has its own graph index.
    """

    return _repeat_latent_state(latent_state, batch_size, device=device)


def _repeat_latent_state(
    latent_state: JEPALatentState,
    batch_size: int,
    *,
    device: torch.device | str,
) -> JEPALatentState:
    """Repeat ``JEPALatentState`` from batch size 1 to ``batch_size``.

    Object-level tensors change from ``[N_obj, ...]`` to
    ``[batch_size * N_obj, ...]`` and graph latents become ``[batch_size, D_z]``.
    """

    object_latents = latent_state.object_latents.to(device)
    object_ids = latent_state.object_ids.to(device)
    object_count = object_ids.numel()
    return JEPALatentState(
        graph_latent=latent_state.graph_latent.to(device).expand(batch_size, -1),
        object_latents=object_latents.repeat((batch_size,) + (1,) * (object_latents.ndim - 1)),
        object_ids=object_ids.repeat(batch_size),
        object_batch=torch.arange(batch_size, device=device).repeat_interleave(object_count),
    )


def _map_values_to_local_indices(values: Tensor, valid_values: Tensor) -> Tensor:
    local_indices = torch.empty_like(values)
    for idx, value in enumerate(values):
        matches = (valid_values == value).nonzero(as_tuple=False).flatten()
        if matches.numel() != 1:
            raise ValueError(f"Value {int(value.item())} is not present in valid_values")
        local_indices[idx] = matches[0]
    return local_indices


def _cartesian_product(domains: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(itertools.product(*domains))


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average ``FloatTensor[B, A, D]`` over valid ``BoolTensor[B, A]`` slots."""

    weights = mask.to(values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom
