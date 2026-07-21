from __future__ import annotations

import pytest
import torch


def test_action_vicreg_reports_numerically_correct_weighted_components() -> None:
    from acs_jepa.losses import ActionVICRegLoss, CovarianceLoss, HingeStdLoss

    latents = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], requires_grad=True)
    loss_fn = ActionVICRegLoss(std_coeff=2.0, cov_coeff=3.0, std_margin=2.0)

    output = loss_fn(latents)

    expected_std = HingeStdLoss(std_margin=2.0)(latents)
    expected_covariance = CovarianceLoss()(latents)
    assert output.total.ndim == 0
    assert output.std_penalty.ndim == 0
    assert output.covariance_penalty.ndim == 0
    assert output.num_samples == 3
    assert torch.allclose(output.std_penalty, expected_std)
    assert torch.allclose(output.covariance_penalty, expected_covariance)
    assert output.std_penalty.item() == pytest.approx(0.9999500012499376)
    assert output.covariance_penalty.item() == pytest.approx(1.0)
    assert torch.allclose(output.total, 2.0 * expected_std + 3.0 * expected_covariance)


def test_action_vicreg_flattens_batch_and_time_into_samples() -> None:
    from acs_jepa.losses import ActionVICRegLoss

    temporal = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        ]
    )
    loss_fn = ActionVICRegLoss(std_margin=2.0)

    temporal_output = loss_fn(temporal)
    flat_output = loss_fn(temporal.reshape(-1, temporal.size(-1)))

    assert temporal_output.num_samples == 6
    assert torch.allclose(temporal_output.total, flat_output.total)
    assert torch.allclose(temporal_output.std_penalty, flat_output.std_penalty)
    assert torch.allclose(temporal_output.covariance_penalty, flat_output.covariance_penalty)


def test_action_vicreg_single_sample_returns_graph_connected_zero() -> None:
    from acs_jepa.losses import ActionVICRegLoss

    latents = torch.tensor([[1.0, 2.0]], requires_grad=True)

    output = ActionVICRegLoss()(latents)
    output.total.backward()

    assert output.num_samples == 1
    assert output.std_penalty.item() == 0.0
    assert output.covariance_penalty.item() == 0.0
    assert output.total.item() == 0.0
    assert output.total.requires_grad
    assert latents.grad is not None
    assert torch.equal(latents.grad, torch.zeros_like(latents))


def test_action_vicreg_one_feature_has_graph_connected_zero_covariance() -> None:
    from acs_jepa.losses import ActionVICRegLoss

    latents = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)

    output = ActionVICRegLoss(std_margin=2.0)(latents)
    output.total.backward()

    assert torch.isfinite(output.total)
    assert output.std_penalty.item() > 0.0
    assert output.covariance_penalty.item() == 0.0
    assert output.covariance_penalty.requires_grad
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"std_coeff": -1.0}, "std_coeff"),
        ({"std_coeff": float("nan")}, "std_coeff"),
        ({"cov_coeff": -1.0}, "cov_coeff"),
        ({"cov_coeff": float("inf")}, "cov_coeff"),
        ({"std_margin": 0.0}, "std_margin"),
        ({"std_margin": float("nan")}, "std_margin"),
    ],
)
def test_action_vicreg_rejects_invalid_hyperparameters(kwargs: dict[str, float], message: str) -> None:
    from acs_jepa.losses import ActionVICRegLoss

    with pytest.raises(ValueError, match=message):
        ActionVICRegLoss(**kwargs)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (1, 0, 3), (1, 2, 0)])
def test_action_vicreg_rejects_empty_sample_or_feature_dimensions(shape: tuple[int, ...]) -> None:
    from acs_jepa.losses import ActionVICRegLoss

    with pytest.raises(ValueError, match="non-empty"):
        ActionVICRegLoss()(torch.empty(shape))


def test_action_vicreg_rejects_non_floating_latents() -> None:
    from acs_jepa.losses import ActionVICRegLoss

    with pytest.raises(ValueError, match="floating"):
        ActionVICRegLoss()(torch.tensor([[0, 1], [1, 0]]))


def test_action_vicreg_rejects_non_matrix_or_temporal_ranks() -> None:
    from acs_jepa.losses import ActionVICRegLoss

    with pytest.raises(ValueError, match="rank-2 or rank-3"):
        ActionVICRegLoss()(torch.zeros(3))
    with pytest.raises(ValueError, match="rank-2 or rank-3"):
        ActionVICRegLoss()(torch.zeros(1, 1, 1, 1))


def test_action_vicreg_penalizes_collapsed_multisample_latents() -> None:
    from acs_jepa.losses import ActionVICRegLoss

    output = ActionVICRegLoss(std_margin=1.0)(torch.zeros(4, 3))

    assert output.std_penalty.item() > 0.0
    assert output.covariance_penalty.item() == 0.0


def test_action_vicreg_produces_finite_nonzero_gradients_for_nondegenerate_input() -> None:
    from acs_jepa.losses import ActionVICRegLoss

    latents = torch.tensor(
        [[0.0, 0.0], [0.5, 1.0], [1.5, -0.5], [2.0, 2.0]],
        requires_grad=True,
    )

    ActionVICRegLoss(std_margin=2.0)(latents).total.backward()

    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert torch.count_nonzero(latents.grad).item() > 0


def test_action_vicreg_is_exported_from_package() -> None:
    import acs_jepa
    from acs_jepa.losses import ActionVICRegLoss, ActionVICRegLossOutput

    assert acs_jepa.ActionVICRegLoss is ActionVICRegLoss
    assert acs_jepa.ActionVICRegLossOutput is ActionVICRegLossOutput
