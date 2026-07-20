from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from acs_jepa.losses import ApplicabilityLoss


def test_applicability_loss_matches_torch_bce_and_reports_diagnostics() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -0.25], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_fn = ApplicabilityLoss()

    output = loss_fn(logits, labels)

    assert torch.allclose(output.total, F.binary_cross_entropy_with_logits(logits, labels))
    assert torch.allclose(output.bce, output.total)
    assert torch.allclose(output.positive_logit_mean, torch.tensor(1.25))
    assert torch.allclose(output.negative_logit_mean, torch.tensor(-0.625))
    assert torch.allclose(output.positive_negative_margin, torch.tensor(1.875))
    assert output.num_examples == 4
    assert output.num_positive == 2
    assert output.num_negative == 2


def test_applicability_loss_mask_excludes_unknown_labels_and_masked_gradients() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -0.25], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.tensor([True, False, True, False])

    output = ApplicabilityLoss()(logits, labels, example_mask=mask)
    output.total.backward()

    expected = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
    assert torch.allclose(output.total.detach(), expected)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[mask]).item() > 0
    assert torch.count_nonzero(logits.grad[~mask]).item() == 0
    assert output.num_examples == 2
    assert output.num_positive == 2
    assert output.num_negative == 0
    assert output.negative_logit_mean is None
    assert output.positive_negative_margin is None


def test_applicability_loss_rejects_empty_effective_batch() -> None:
    logits = torch.tensor([2.0, -1.0])
    labels = torch.tensor([1.0, 0.0])
    mask = torch.tensor([False, False])

    with pytest.raises(ValueError, match="empty effective batch"):
        ApplicabilityLoss()(logits, labels, example_mask=mask)


def test_applicability_loss_rejects_invalid_shapes_and_mask_dtype() -> None:
    loss_fn = ApplicabilityLoss()
    logits = torch.tensor([2.0, -1.0])
    labels = torch.tensor([1.0, 0.0])

    with pytest.raises(ValueError, match="logits must have shape"):
        loss_fn(logits.unsqueeze(0), labels)
    with pytest.raises(ValueError, match="labels must have shape"):
        loss_fn(logits, labels.unsqueeze(0))
    with pytest.raises(ValueError, match="same shape"):
        loss_fn(logits, torch.tensor([1.0]))
    with pytest.raises(ValueError, match="example_mask must have shape"):
        loss_fn(logits, labels, example_mask=torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="same shape"):
        loss_fn(logits, labels, example_mask=torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError, match="example_mask must be bool"):
        loss_fn(logits, labels, example_mask=torch.ones(2))


def test_applicability_loss_rejects_labels_outside_unit_interval() -> None:
    logits = torch.tensor([2.0, -1.0])

    with pytest.raises(ValueError, match=r"labels must lie in \[0, 1\]"):
        ApplicabilityLoss()(logits, torch.tensor([1.0, -0.1]))
    with pytest.raises(ValueError, match=r"labels must lie in \[0, 1\]"):
        ApplicabilityLoss()(logits, torch.tensor([1.0, 1.1]))


def test_applicability_loss_all_positive_or_all_negative_diagnostics() -> None:
    positive_output = ApplicabilityLoss()(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
    negative_output = ApplicabilityLoss()(torch.tensor([-1.0, -2.0]), torch.tensor([0.0, 0.0]))

    assert positive_output.num_positive == 2
    assert positive_output.num_negative == 0
    assert positive_output.positive_logit_mean is not None
    assert positive_output.negative_logit_mean is None
    assert positive_output.positive_negative_margin is None
    assert negative_output.num_positive == 0
    assert negative_output.num_negative == 2
    assert negative_output.positive_logit_mean is None
    assert negative_output.negative_logit_mean is not None
    assert negative_output.positive_negative_margin is None


def test_applicability_loss_pos_weight_matches_torch_bce() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, -0.25])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_fn = ApplicabilityLoss(pos_weight=3.0)

    output = loss_fn(logits, labels)

    expected = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(3.0))
    assert torch.allclose(output.total, expected)


def test_applicability_loss_rejects_invalid_pos_weight() -> None:
    with pytest.raises(ValueError, match="pos_weight must be positive"):
        ApplicabilityLoss(pos_weight=0.0)
