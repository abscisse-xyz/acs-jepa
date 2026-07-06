"""Neural encoders for PDDL factor graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool

from acs_jepa.graph.builders import OBJECT_NODE
from acs_jepa.graph.schemas import ParsedProblem


@dataclass(frozen=True)
class GraphEncoderOutput:
    """Embeddings produced by :class:`GraphEncoder`.

    Shapes:
        graph_embedding: ``FloatTensor[B, D_e]`` from mean pooling final node
            embeddings per graph.
        object_embeddings: ``FloatTensor[N_obj, D_e]`` filtered from
            final node embeddings.
        object_ids: ``LongTensor[N_obj]`` with problem-local object ids.
        object_batch: ``LongTensor[N_obj]`` mapping each object to a graph in
            ``[0, B)``.
    """

    graph_embedding: Tensor
    object_embeddings: Tensor
    object_ids: Tensor
    object_batch: Tensor


class SentinelEmbedding(nn.Module):
    """Embedding table for integer features where ``-1`` means missing.

    Input feature values in ``[-1, max_value]`` are shifted by ``+1`` before the
    embedding lookup. Index ``0`` is reserved as padding for sentinel ``-1``.

    Args:
        max_value: Maximum non-sentinel feature id.
        embedding_dim: Output embedding size.

    Modules:
        embedding: ``nn.Embedding(max_value + 2, embedding_dim, padding_idx=0)``.
    """

    def __init__(self, max_value: int, embedding_dim: int) -> None:
        super().__init__()
        if max_value < 0:
            raise ValueError("max_value must be non-negative")
        self.embedding = nn.Embedding(max_value + 2, embedding_dim, padding_idx=0)

    def forward(self, values: Tensor) -> Tensor:
        """Embed integer features with ``-1`` mapped to padding.

        Args:
            values: ``LongTensor[...]`` containing values in ``[-1, max_value]``.

        Returns:
            ``FloatTensor[..., embedding_dim]``.
        """

        indices = values.clamp_min(-1) + 1
        return self.embedding(indices)


class GraphEncoder(nn.Module):
    """Edge-aware graph encoder for PDDL factor-node state graphs.

    The expected input is a PyTorch Geometric ``Data`` or mini-batch produced by
    ``build_state_graph``. Nodes are typed as object nodes or atom nodes. Edges
    are directed and bidirectional for every atom/object relation:
    ``OBJECT_TO_ATOM`` lets facts aggregate argument-object information, while
    ``ATOM_TO_OBJECT`` lets objects receive relational context from active
    facts. Predicate semantics remain in node ``predicate_id`` and edge
    ``role_id`` features.

    Node feature columns in ``data.x``:
        ``[node_kind, object_type, predicate_id, arity, object_id]``.

    Edge feature columns in ``data.edge_attr``:
        ``[role_id, edge_direction]``.

    Args:
        num_node_kinds: Number of node kinds. Current builders use object/atom.
        num_types: Number of object types.
        num_predicates: Number of predicate schemas.
        num_objects: Number of problem-local objects.
        max_arity: Maximum predicate/action arity used for arity embeddings.
        max_role_id: Maximum edge role id.
        hidden_dim: Internal message-passing feature size.
        embed_dim: Output embedding size ``D_e``.
        num_layers: Number of GINE message-passing layers.

    Modules:
        *_embedding: Padding-aware feature embeddings for node/edge categorical
            attributes.
        node_projection: MLP applied after summing node feature embeddings.
        edge_projection: MLP applied after summing edge feature embeddings.
        convs: ``num_layers`` edge-aware ``GINEConv`` layers.
        norms: LayerNorm modules paired with ``convs``.
        output_projection: Linear projection from ``hidden_dim`` to ``D_e``.
    """

    def __init__(
        self,
        *,
        num_node_kinds: int,
        num_types: int,
        num_predicates: int,
        num_objects: int,
        max_arity: int,
        max_role_id: int,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.node_kind_embedding = SentinelEmbedding(max(0, num_node_kinds - 1), hidden_dim)
        self.object_type_embedding = SentinelEmbedding(max(0, num_types - 1), hidden_dim)
        self.predicate_embedding = SentinelEmbedding(max(0, num_predicates - 1), hidden_dim)
        self.arity_embedding = SentinelEmbedding(max_arity, hidden_dim)
        self.object_id_embedding = SentinelEmbedding(max(0, num_objects - 1), hidden_dim)

        self.node_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.role_embedding = SentinelEmbedding(max_role_id, hidden_dim)
        self.edge_direction_embedding = SentinelEmbedding(1, hidden_dim)
        self.edge_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_projection = nn.Linear(hidden_dim, embed_dim)

    @classmethod
    def from_parsed_problem(
        cls,
        parsed_problem: ParsedProblem,
        *,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_layers: int = 3,
    ) -> GraphEncoder:
        """Create an encoder with vocabulary sizes from one parsed problem.

        Args:
            parsed_problem: Parsed PDDL domain/problem containing type,
                predicate, action, and object vocabularies.
            hidden_dim: Internal message-passing feature size.
            embed_dim: Output embedding size ``D_e``.
            num_layers: Number of GINE message-passing layers.

        Returns:
            Configured ``GraphEncoder`` for the parsed problem.
        """

        max_predicate_arity = max(
            (len(predicate.arg_types) for predicate in parsed_problem.predicates.values()),
            default=0,
        )
        max_action_arity = parsed_problem.max_action_arity
        max_arity = max(max_predicate_arity, max_action_arity)

        return cls(
            num_node_kinds=2,
            num_types=len(parsed_problem.types),
            num_predicates=len(parsed_problem.predicates),
            num_objects=len(parsed_problem.objects),
            max_arity=max_arity,
            max_role_id=max(0, max_arity - 1),
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

    def forward(self, data: Data) -> GraphEncoderOutput:
        """Encode a single PyG graph or a PyG mini-batch.

        Args:
            data: PyG ``Data``/``Batch`` with ``x`` shape ``[N_node, 5]``,
                ``edge_index`` shape ``[2, N_edge]``, ``edge_attr`` shape
                ``[N_edge, 2]``, and optional ``batch`` shape ``[N_node]``.

        Returns:
            ``GraphEncoderOutput`` containing graph and object embeddings.
            Output embedding dimension is ``embed_dim``.
        """

        x = data.x.long()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        node_features = self._embed_node_features(x)
        edge_features = self._embed_edge_features(data.edge_attr.long())

        node_embeddings = node_features
        for conv, norm in zip(self.convs, self.norms):
            residual = node_embeddings
            node_embeddings = conv(node_embeddings, data.edge_index, edge_features)
            node_embeddings = norm(node_embeddings + residual)
            node_embeddings = torch.relu(node_embeddings)

        node_embeddings = self.output_projection(node_embeddings)
        graph_embedding = global_mean_pool(node_embeddings, batch)

        node_kind = x[:, 0]
        object_mask = node_kind == OBJECT_NODE

        return GraphEncoderOutput(
            graph_embedding=graph_embedding,
            object_embeddings=node_embeddings[object_mask],
            object_ids=x[object_mask, 4],
            object_batch=batch[object_mask],
        )

    def _embed_node_features(self, x: Tensor) -> Tensor:
        """Embed and project node categorical features.

        Args:
            x: Node feature matrix ``LongTensor[N_node, 5]``.

        Returns:
            Hidden node features ``FloatTensor[N_node, hidden_dim]``.
        """

        features = (
            self.node_kind_embedding(x[:, 0])
            + self.object_type_embedding(x[:, 1])
            + self.predicate_embedding(x[:, 2])
            + self.arity_embedding(x[:, 3])
            + self.object_id_embedding(x[:, 4])
        )
        return self.node_projection(features)

    def _embed_edge_features(self, edge_attr: Tensor) -> Tensor:
        """Embed and project edge categorical features.

        Args:
            edge_attr: Edge feature matrix ``LongTensor[N_edge, 2]``.

        Returns:
            Hidden edge features ``FloatTensor[N_edge, hidden_dim]``. Empty
            graphs return an empty tensor with the correct feature dimension.
        """

        if edge_attr.numel() == 0:
            return edge_attr.new_empty((0, self.role_embedding.embedding.embedding_dim), dtype=torch.float)
        features = self.role_embedding(edge_attr[:, 0]) + self.edge_direction_embedding(edge_attr[:, 1])
        return self.edge_projection(features)
