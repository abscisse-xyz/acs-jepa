from __future__ import annotations

import pytest
import torch


def test_action_contrastive_loss_matches_manual_cosine_infonce() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])

    output = ActionContrastiveLoss(temperature=0.5)(anchors, positives, negatives)

    assert output.total.item() == pytest.approx(0.1429316284998996)
    assert output.positive_similarity_mean.item() == pytest.approx(1.0)
    assert output.hardest_negative_similarity_mean.item() == pytest.approx(0.0)
    assert output.positive_negative_margin.item() == pytest.approx(1.0)
    assert output.top1_accuracy.item() == pytest.approx(1.0)
    assert output.num_examples == 1
    assert output.num_negatives == 2


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_action_contrastive_loss_uses_documented_compute_dtype(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.5]], dtype=input_dtype)
    positives = torch.tensor([[0.8, 0.2]], dtype=input_dtype)
    negatives = torch.tensor([[[-0.5, 1.0]]], dtype=input_dtype)

    output = ActionContrastiveLoss()(anchors, positives, negatives)

    for value in (
        output.total,
        output.positive_similarity_mean,
        output.hardest_negative_similarity_mean,
        output.positive_negative_margin,
        output.top1_accuracy,
    ):
        assert value.dtype == output_dtype
        assert torch.isfinite(value)


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.bfloat16, 3.0e38),
        (torch.float32, 3.0e38),
        (torch.float64, 1.0e308),
    ],
)
def test_action_contrastive_loss_stably_normalizes_extreme_finite_vectors(
    dtype: torch.dtype,
    magnitude: float,
) -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[magnitude, magnitude]], dtype=dtype)
    positives = torch.tensor([[magnitude, magnitude]], dtype=dtype)
    negatives = torch.tensor([[[-magnitude, -magnitude]]], dtype=dtype)

    output = ActionContrastiveLoss(temperature=1.0)(anchors, positives, negatives)

    assert output.positive_similarity_mean.item() == pytest.approx(1.0, abs=1e-5)
    assert output.hardest_negative_similarity_mean.item() == pytest.approx(-1.0, abs=1e-5)
    assert output.positive_negative_margin.item() == pytest.approx(2.0, abs=1e-5)
    assert torch.isfinite(output.total)


def test_action_contrastive_loss_masks_outputs_counts_and_gradients() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[0.8, 0.2]], requires_grad=True)
    negatives = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    mask = torch.tensor([[True, False]])

    output = ActionContrastiveLoss()(anchors, positives, negatives, mask)
    output.total.backward()

    changed_negatives = negatives.detach().clone()
    changed_negatives[0, 1] = torch.tensor([-1.0, 0.0])
    changed = ActionContrastiveLoss()(anchors.detach(), positives.detach(), changed_negatives, mask)

    assert torch.allclose(output.total.detach(), changed.total)
    assert torch.allclose(output.hardest_negative_similarity_mean.detach(), changed.hardest_negative_similarity_mean)
    assert output.num_negatives == 1
    assert negatives.grad is not None
    assert torch.count_nonzero(negatives.grad[0, 0]).item() > 0
    assert torch.equal(negatives.grad[0, 1], torch.zeros_like(negatives.grad[0, 1]))


def test_action_contrastive_loss_counts_variable_active_negatives() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = anchors.clone()
    negatives = torch.tensor(
        [
            [[0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, False, False], [True, True, False]])

    output = ActionContrastiveLoss()(anchors, positives, negatives, mask)

    assert output.num_examples == 2
    assert output.num_negatives == 3
    assert torch.isfinite(output.total)


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf")])
def test_action_contrastive_loss_rejects_invalid_temperature(temperature: float) -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    with pytest.raises(ValueError, match="temperature"):
        ActionContrastiveLoss(temperature=temperature)


def test_action_contrastive_loss_rejects_invalid_shapes() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    loss = ActionContrastiveLoss()
    anchors = torch.ones(2, 3)
    positives = torch.ones(2, 3)
    negatives = torch.ones(2, 2, 3)

    invalid_inputs = [
        (torch.ones(3), positives, negatives),
        (anchors, torch.ones(2, 4), negatives),
        (anchors, positives, torch.ones(2, 3)),
        (anchors, positives, torch.ones(1, 2, 3)),
        (torch.ones(0, 3), torch.ones(0, 3), torch.ones(0, 2, 3)),
        (torch.ones(2, 0), torch.ones(2, 0), torch.ones(2, 2, 0)),
        (anchors, positives, torch.ones(2, 0, 3)),
    ]
    for invalid in invalid_inputs:
        with pytest.raises(ValueError, match="shape"):
            loss(*invalid)


def test_action_contrastive_loss_rejects_nonfloating_or_mismatched_dtypes() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    loss = ActionContrastiveLoss()
    anchors = torch.ones(1, 2)
    positives = torch.ones(1, 2)
    negatives = torch.ones(1, 1, 2)

    with pytest.raises(ValueError, match="floating"):
        loss(anchors.to(torch.int64), positives.to(torch.int64), negatives.to(torch.int64))
    with pytest.raises(ValueError, match="dtype"):
        loss(anchors, positives.to(torch.float64), negatives)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("target", ["anchor", "positive", "negative", "masked_negative"])
def test_action_contrastive_loss_rejects_nonfinite_latents(
    bad_value: float,
    target: str,
) -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.5]])
    positives = torch.tensor([[0.8, 0.2]])
    negatives = torch.tensor([[[0.0, 1.0], [-1.0, 0.5]]])
    mask = torch.tensor([[True, False]])
    if target == "anchor":
        anchors[0, 0] = bad_value
    elif target == "positive":
        positives[0, 0] = bad_value
    elif target == "negative":
        negatives[0, 0, 0] = bad_value
    else:
        negatives[0, 1, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        ActionContrastiveLoss()(anchors, positives, negatives, mask)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_action_contrastive_loss_rejects_zero_vectors(dtype: torch.dtype) -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[0.0, 0.0]], dtype=dtype)
    positives = torch.tensor([[1.0, 0.0]], dtype=dtype)
    negatives = torch.tensor([[[0.0, 1.0]]], dtype=dtype)

    with pytest.raises(ValueError, match="norm"):
        ActionContrastiveLoss()(anchors, positives, negatives)


def test_action_contrastive_loss_rejects_invalid_masks() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    loss = ActionContrastiveLoss()
    anchors = torch.tensor([[1.0, 0.5]])
    positives = torch.tensor([[0.8, 0.2]])
    negatives = torch.tensor([[[0.0, 1.0], [-1.0, 0.5]]])

    with pytest.raises(ValueError, match="bool"):
        loss(anchors, positives, negatives, torch.ones(1, 2))
    with pytest.raises(ValueError, match="shape"):
        loss(anchors, positives, negatives, torch.ones(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="active negative"):
        loss(anchors, positives, negatives, torch.zeros(1, 2, dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cross-device mask")
def test_action_contrastive_loss_rejects_mask_device_mismatch() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.5]], device="cuda")
    positives = torch.tensor([[0.8, 0.2]], device="cuda")
    negatives = torch.tensor([[[0.0, 1.0]]], device="cuda")
    cpu_mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="device"):
        ActionContrastiveLoss()(anchors, positives, negatives, cpu_mask)
    with pytest.raises(ValueError, match="device"):
        ActionContrastiveLoss()(anchors, positives.cpu(), negatives)


def test_action_contrastive_loss_enforces_compute_dtype_temperature_boundary() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[[0.0, 1.0]]])
    tiny = torch.finfo(torch.float32).tiny
    below_tiny = float(
        torch.nextafter(torch.tensor(tiny), torch.tensor(0.0)).item()
    )

    exact = ActionContrastiveLoss(temperature=tiny)(anchors, positives, negatives)
    assert torch.isfinite(exact.total)
    with pytest.raises(ValueError, match="temperature"):
        ActionContrastiveLoss(temperature=below_tiny)(anchors, positives, negatives)


def test_action_contrastive_loss_enforces_vector_norm_boundary_and_masked_negatives() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    threshold = torch.tensor(1.0e-12, dtype=torch.float32)
    above = torch.nextafter(threshold, torch.tensor(float("inf")))
    positives = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="norm"):
        ActionContrastiveLoss()(torch.stack((threshold, torch.tensor(0.0))).unsqueeze(0), positives, negatives)

    accepted = ActionContrastiveLoss()(
        torch.stack((above, torch.tensor(0.0))).unsqueeze(0),
        positives,
        negatives,
    )
    assert torch.isfinite(accepted.total)

    masked_zero = negatives.clone()
    masked_zero[0, 1] = 0.0
    with pytest.raises(ValueError, match="norm"):
        ActionContrastiveLoss()(positives, positives, masked_zero, mask)


def test_action_contrastive_loss_all_true_mask_matches_unmasked_and_ties_fail() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    loss = ActionContrastiveLoss()

    unmasked = loss(anchors, positives, negatives)
    all_true = loss(anchors, positives, negatives, torch.ones(1, 2, dtype=torch.bool))

    assert torch.allclose(unmasked.total, all_true.total)
    assert unmasked.positive_negative_margin.item() == pytest.approx(0.0)
    assert unmasked.top1_accuracy.item() == pytest.approx(0.0)


def test_action_contrastive_loss_backpropagates_to_all_active_inputs() -> None:
    from acs_jepa.losses import ActionContrastiveLoss

    anchors = torch.tensor([[1.0, 0.2]], requires_grad=True)
    positives = torch.tensor([[0.7, 0.6]], requires_grad=True)
    negatives = torch.tensor([[[0.1, 1.0], [-0.8, 0.3]]], requires_grad=True)

    ActionContrastiveLoss()(anchors, positives, negatives).total.backward()

    for value in (anchors, positives, negatives):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad).item() > 0


def test_action_contrastive_loss_is_exported_from_package() -> None:
    import acs_jepa
    from acs_jepa.losses import ActionContrastiveLoss, ActionContrastiveLossOutput

    assert acs_jepa.ActionContrastiveLoss is ActionContrastiveLoss
    assert acs_jepa.ActionContrastiveLossOutput is ActionContrastiveLossOutput
