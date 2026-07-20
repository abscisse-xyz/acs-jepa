from __future__ import annotations

import pytest
import torch
from acs_jepa import ApplicabilityHead


def _head() -> ApplicabilityHead:
    torch.manual_seed(123)
    return ApplicabilityHead(latent_dim=8, action_dim=6, max_action_arity=4, hidden_dim=12, dropout=0.0)


def test_applicability_head_scores_graph_action_pairs_without_object_context() -> None:
    head = _head()
    graph_latent = torch.randn(3, 8)
    action_latent = torch.randn(3, 6)

    logits = head(graph_latent, action_latent)

    assert logits.shape == (3,)
    assert logits.dtype == graph_latent.dtype


def test_applicability_head_scores_with_role_order_aware_object_context() -> None:
    head = _head()
    graph_latent = torch.randn(2, 8)
    action_latent = torch.randn(2, 6)
    object_latents = torch.randn(2, 4, 8)
    argument_mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    logits = head(graph_latent, action_latent, object_latents, argument_mask)

    assert logits.shape == (2,)


def test_applicability_head_is_sensitive_to_unmasked_argument_order() -> None:
    head = _head()
    graph_latent = torch.zeros(1, 8)
    action_latent = torch.zeros(1, 6)
    object_latents = torch.zeros(1, 4, 8)
    object_latents[0, 0, 0] = 1.0
    object_latents[0, 1, 1] = 2.0
    swapped = object_latents.clone()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    argument_mask = torch.tensor([[True, True, False, False]])

    original_logit = head(graph_latent, action_latent, object_latents, argument_mask)
    swapped_logit = head(graph_latent, action_latent, swapped, argument_mask)

    assert not torch.allclose(original_logit, swapped_logit)


def test_applicability_head_ignores_masked_slots_for_logits_and_gradients() -> None:
    head = _head()
    graph_latent = torch.randn(1, 8, requires_grad=True)
    action_latent = torch.randn(1, 6, requires_grad=True)
    object_latents = torch.randn(1, 4, 8, requires_grad=True)
    changed_masked = object_latents.detach().clone()
    changed_masked[:, 2:] += 1000.0
    argument_mask = torch.tensor([[True, True, False, False]])

    original_logit = head(graph_latent, action_latent, object_latents, argument_mask)
    changed_logit = head(graph_latent.detach(), action_latent.detach(), changed_masked, argument_mask)
    original_logit.sum().backward()

    assert torch.allclose(original_logit.detach(), changed_logit)
    assert object_latents.grad is not None
    assert torch.count_nonzero(object_latents.grad[:, :2]).item() > 0
    assert torch.count_nonzero(object_latents.grad[:, 2:]).item() == 0


def test_applicability_head_backpropagates_to_state_action_and_unmasked_objects() -> None:
    head = _head()
    graph_latent = torch.randn(2, 8, requires_grad=True)
    action_latent = torch.randn(2, 6, requires_grad=True)
    object_latents = torch.randn(2, 4, 8, requires_grad=True)
    argument_mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    loss = head(graph_latent, action_latent, object_latents, argument_mask).sum()
    loss.backward()

    assert graph_latent.grad is not None and torch.count_nonzero(graph_latent.grad).item() > 0
    assert action_latent.grad is not None and torch.count_nonzero(action_latent.grad).item() > 0
    assert object_latents.grad is not None and torch.count_nonzero(object_latents.grad[argument_mask]).item() > 0


def test_applicability_head_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        ApplicabilityHead(latent_dim=0, action_dim=6, max_action_arity=4)
    with pytest.raises(ValueError, match="action_dim must be positive"):
        ApplicabilityHead(latent_dim=8, action_dim=0, max_action_arity=4)
    with pytest.raises(ValueError, match="max_action_arity must be non-negative"):
        ApplicabilityHead(latent_dim=8, action_dim=6, max_action_arity=-1)
    with pytest.raises(ValueError, match="dropout must lie in"):
        ApplicabilityHead(latent_dim=8, action_dim=6, max_action_arity=4, dropout=1.5)


def test_applicability_head_rejects_invalid_input_shapes() -> None:
    head = _head()
    graph_latent = torch.randn(2, 8)
    action_latent = torch.randn(2, 6)
    object_latents = torch.randn(2, 4, 8)
    argument_mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="graph_latent must have shape"):
        head(torch.randn(2, 1, 8), action_latent)
    with pytest.raises(ValueError, match="action_latent must have shape"):
        head(graph_latent, torch.randn(2, 1, 6))
    with pytest.raises(ValueError, match="graph_latent last dimension"):
        head(torch.randn(2, 7), action_latent)
    with pytest.raises(ValueError, match="action_latent last dimension"):
        head(graph_latent, torch.randn(2, 5))
    with pytest.raises(ValueError, match="batch size"):
        head(torch.randn(3, 8), action_latent)
    with pytest.raises(ValueError, match="argument_mask is required"):
        head(graph_latent, action_latent, object_latents)
    with pytest.raises(ValueError, match="object_latents is required"):
        head(graph_latent, action_latent, argument_mask=argument_mask)
    with pytest.raises(ValueError, match="object_latents must have shape"):
        head(graph_latent, action_latent, torch.randn(2, 8), argument_mask[:, 0])
    with pytest.raises(ValueError, match="argument_mask must have shape"):
        head(graph_latent, action_latent, object_latents, torch.ones(2, 4, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="object_latents last dimension"):
        head(graph_latent, action_latent, torch.randn(2, 4, 7), argument_mask)
    with pytest.raises(ValueError, match="object/mask arity"):
        head(graph_latent, action_latent, object_latents[:, :3], argument_mask)
    with pytest.raises(ValueError, match="exceeds max_action_arity"):
        head(graph_latent, action_latent, torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool))
