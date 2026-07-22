"""Simulator-free role/object retrieval probe owner for Stage 0D."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

RoleTensors = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class RoleObjectFitResult:
    """Fitted probe, exact detached fit populations, metrics, and execution count."""

    model: nn.Module
    train_metrics: dict[str, object]
    eval_metrics: dict[str, object]
    train_tensors: RoleTensors
    eval_tensors: RoleTensors
    optimizer_steps: int


class RoleObjectProbe(nn.Module):
    """Retrieve a problem-local argument object from all real object latents."""

    def __init__(self, *, latent_dim: int, action_dim: int, max_action_arity: int, hidden_dim: int) -> None:
        super().__init__()
        self.max_action_arity = int(max_action_arity)
        self.role_embedding = nn.Embedding(max_action_arity, latent_dim)
        self.query = nn.Sequential(
            nn.Linear(latent_dim * 2 + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        graph_latents: Tensor,
        action_latents: Tensor,
        object_latents: Tensor,
        object_mask: Tensor,
        role_ids: Tensor,
    ) -> Tensor:
        if object_mask.dtype != torch.bool:
            raise ValueError("object_mask must be bool")
        if role_ids.numel() and (int(role_ids.min()) < 0 or int(role_ids.max()) >= self.max_action_arity):
            raise ValueError("role_ids exceed max_action_arity")
        if graph_latents.ndim != 2 or action_latents.ndim != 2 or object_latents.ndim != 3:
            raise ValueError("role probe tensors must have [B,D], [B,A], and [B,N,D] shapes")
        if object_mask.shape != object_latents.shape[:2] or graph_latents.size(0) != object_latents.size(0):
            raise ValueError("role probe batch/mask shapes do not reconcile")
        if not bool(object_mask.any(dim=1).all()):
            raise ValueError("every role row requires at least one real object")
        role_features = self.role_embedding(role_ids)
        query = self.query(torch.cat((graph_latents, action_latents, role_features), dim=-1))
        logits = torch.einsum("bd,bnd->bn", query, object_latents)
        return logits.masked_fill(~object_mask, float("-inf"))


def _classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, object]:
    predictions = logits.argmax(dim=-1)
    return {
        "count": int(labels.numel()),
        "accuracy": None if labels.numel() == 0 else float((predictions == labels).to(torch.float32).mean()),
    }


def _role_metrics(logits: Tensor, targets: Tensor, role_ids: Tensor) -> dict[str, object]:
    metrics = _classification_metrics(logits, targets)
    predictions = logits.argmax(dim=-1)
    metrics["per_role_accuracy"] = {
        str(role): float((predictions[role_ids == role] == targets[role_ids == role]).to(torch.float32).mean())
        for role in sorted(set(role_ids.tolist()))
    }
    return metrics


def _prepare(data: RoleTensors, device: torch.device) -> RoleTensors:
    if len(data) != 6:
        raise ValueError("role fit requires exactly six tensors")
    if device.type != "cpu":
        raise ValueError("role/object diagnostic fitting requires CPU")
    graph, action, objects, mask, roles, targets = data
    row_count = graph.size(0)
    if row_count == 0 or any(value.size(0) != row_count for value in data):
        raise ValueError("role fit tensors require one nonempty reconciled row population")
    if mask.dtype != torch.bool:
        raise ValueError("object_mask must be bool")
    prepared = (
        graph.detach().to(device=device, dtype=torch.float32).clone(),
        action.detach().to(device=device, dtype=torch.float32).clone(),
        objects.detach().to(device=device, dtype=torch.float32).clone(),
        mask.detach().to(device=device, dtype=torch.bool).clone(),
        roles.detach().to(device=device, dtype=torch.long).clone(),
        targets.detach().to(device=device, dtype=torch.long).clone(),
    )
    if prepared[3].shape != prepared[2].shape[:2]:
        raise ValueError("object mask shape must match object bank")
    if prepared[4].ndim != 1 or prepared[5].ndim != 1:
        raise ValueError("role IDs and targets must be vectors")
    if bool(((prepared[5] < 0) | (prepared[5] >= prepared[2].size(1))).any()):
        raise ValueError("targets must index the padded object bank")
    indices = torch.arange(row_count)
    if not bool(prepared[3][indices, prepared[5]].all()):
        raise ValueError("every target must name a real object")
    return prepared


def fit_role_object_probe(
    train_data: RoleTensors,
    eval_data: RoleTensors,
    *,
    max_action_arity: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> RoleObjectFitResult:
    """Fit exact deterministic full-batch CE retrieval on detached CPU float32 tensors."""

    if epochs <= 0 or learning_rate <= 0 or hidden_dim <= 0 or max_action_arity <= 0:
        raise ValueError("role fit hyperparameters must be positive")
    train = _prepare(train_data, device)
    evaluation = _prepare(eval_data, device)
    torch.manual_seed(seed)
    model = RoleObjectProbe(
        latent_dim=train[0].size(-1),
        action_dim=train[1].size(-1),
        max_action_arity=max_action_arity,
        hidden_dim=hidden_dim,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_steps = 0
    optimizer_step = optimizer.step

    def counted_optimizer_step() -> None:
        nonlocal optimizer_steps
        optimizer_step()
        optimizer_steps += 1

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*train[:5])
        torch.nn.functional.cross_entropy(logits, train[5]).backward()
        counted_optimizer_step()
    model.eval()
    with torch.no_grad():
        train_metrics = _role_metrics(model(*train[:5]), train[5], train[4])
        eval_metrics = _role_metrics(model(*evaluation[:5]), evaluation[5], evaluation[4])
    return RoleObjectFitResult(
        model=model,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        train_tensors=train,
        eval_tensors=evaluation,
        optimizer_steps=optimizer_steps,
    )
