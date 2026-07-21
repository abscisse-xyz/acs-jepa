from __future__ import annotations

import pytest
import torch


def test_argument_reconstruction_head_shape_and_gradients() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    torch.manual_seed(0)
    head = ArgumentReconstructionHead(
        action_dim=3,
        object_dim=4,
        max_action_arity=2,
        hidden_dim=5,
    )
    actions = torch.randn(2, 3, requires_grad=True)
    objects = torch.randn(2, 3, 4, requires_grad=True)

    logits = head(actions, objects)
    logits.square().mean().backward()

    assert logits.shape == (2, 2, 3)
    assert torch.isfinite(logits).all()
    assert actions.grad is not None
    assert objects.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert torch.isfinite(objects.grad).all()
    assert torch.count_nonzero(actions.grad).item() > 0
    assert torch.count_nonzero(objects.grad).item() > 0
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_argument_reconstruction_head_is_object_permutation_equivariant() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    torch.manual_seed(1)
    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    ).eval()
    actions = torch.randn(2, 2)
    objects = torch.randn(2, 4, 3)
    permutation = torch.tensor([2, 0, 3, 1])

    original = head(actions, objects)
    permuted = head(actions, objects[:, permutation])

    assert torch.allclose(permuted, original[:, :, permutation])


def test_argument_reconstruction_head_is_role_sensitive() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    head = ArgumentReconstructionHead(
        action_dim=1,
        object_dim=1,
        max_action_arity=2,
        hidden_dim=1,
    ).eval()
    with torch.no_grad():
        head.action_projection.weight.zero_()
        head.action_projection.bias.zero_()
        head.object_projection.weight.zero_()
        head.object_projection.bias.zero_()
        head.role_embedding.weight.copy_(torch.tensor([[0.0], [1.0]]))
        head.scorer[2].weight.fill_(1.0)
        head.scorer[2].bias.zero_()

    logits = head(torch.ones(1, 1), torch.ones(1, 1, 1))

    assert logits[0, 0, 0].item() == pytest.approx(0.0)
    assert logits[0, 1, 0].item() > logits[0, 0, 0].item()


def test_argument_reconstruction_head_masks_candidates_outputs_and_gradients() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    torch.manual_seed(2)
    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=2,
        max_action_arity=2,
        hidden_dim=4,
    )
    actions = torch.randn(1, 2, requires_grad=True)
    objects = torch.randn(1, 3, 2, requires_grad=True)
    mask = torch.tensor([[[True, True, False], [False, False, False]]])

    logits = head(actions, objects, mask)

    assert torch.isfinite(logits[0, 0, :2]).all()
    assert torch.isneginf(logits[0, 0, 2])
    assert torch.isneginf(logits[0, 1]).all()

    changed_objects = objects.detach().clone()
    changed_objects[0, 2] = torch.tensor([1000.0, -1000.0])
    changed_logits = head(actions.detach(), changed_objects, mask)
    assert torch.equal(changed_logits[~mask], logits.detach()[~mask])
    assert torch.allclose(changed_logits[mask], logits.detach()[mask])

    logits[mask].sum().backward()
    assert objects.grad is not None
    assert torch.count_nonzero(objects.grad[0, :2]).item() > 0
    assert torch.equal(objects.grad[0, 2], torch.zeros_like(objects.grad[0, 2]))


def test_argument_reconstruction_head_masked_permutation_equivariance() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    torch.manual_seed(3)
    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    ).eval()
    actions = torch.randn(1, 2)
    objects = torch.randn(1, 4, 3)
    mask = torch.tensor([[[True, False, True, False], [False, True, True, False]]])
    permutation = torch.tensor([2, 0, 3, 1])

    original = head(actions, objects, mask)
    permuted = head(actions, objects[:, permutation], mask[:, :, permutation])

    assert torch.equal(permuted, original[:, :, permutation])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"action_dim": 0}, "action_dim"),
        ({"object_dim": 0}, "object_dim"),
        ({"max_action_arity": 0}, "max_action_arity"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": float("nan")}, "dropout"),
        ({"dropout": float("inf")}, "dropout"),
    ],
)
def test_argument_reconstruction_head_rejects_invalid_constructor_values(
    override: dict[str, float | int],
    message: str,
) -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    kwargs: dict[str, float | int] = {
        "action_dim": 2,
        "object_dim": 3,
        "max_action_arity": 2,
        "hidden_dim": 4,
        "dropout": 0.0,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        ArgumentReconstructionHead(**kwargs)  # type: ignore[arg-type]


def test_argument_reconstruction_head_rejects_invalid_latent_shapes() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    )
    valid_actions = torch.ones(2, 2)
    valid_objects = torch.ones(2, 3, 3)
    invalid_inputs = [
        (torch.ones(2), valid_objects),
        (valid_actions, torch.ones(2, 3)),
        (torch.ones(1, 2), valid_objects),
        (torch.empty(0, 2), torch.empty(0, 3, 3)),
        (valid_actions, torch.empty(2, 0, 3)),
        (torch.ones(2, 4), valid_objects),
        (valid_actions, torch.ones(2, 3, 4)),
    ]

    for actions, objects in invalid_inputs:
        with pytest.raises(ValueError, match="shape"):
            head(actions, objects)


def test_argument_reconstruction_head_validates_and_preserves_latent_dtype() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    )
    actions = torch.ones(2, 2)
    objects = torch.ones(2, 3, 3)

    with pytest.raises(ValueError, match="floating"):
        head(actions.to(torch.int64), objects)
    with pytest.raises(ValueError, match="floating"):
        head(actions, objects.to(torch.int64))
    with pytest.raises(ValueError, match="dtype"):
        head(actions, objects.to(torch.float64))

    head64 = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    ).to(torch.float64)
    logits64 = head64(actions.to(torch.float64), objects.to(torch.float64))
    assert logits64.dtype == torch.float64
    assert torch.isfinite(logits64).all()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("target", ["action", "object", "masked_object"])
def test_argument_reconstruction_head_rejects_nonfinite_latents(
    bad_value: float,
    target: str,
) -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    )
    actions = torch.ones(1, 2)
    objects = torch.ones(1, 2, 3)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    if target == "action":
        actions[0, 0] = bad_value
    elif target == "object":
        objects[0, 0, 0] = bad_value
    else:
        objects[0, 1, 0] = bad_value
        mask[:, :, 1] = False

    with pytest.raises(ValueError, match="finite"):
        head(actions, objects, mask)


def test_argument_reconstruction_head_rejects_invalid_candidate_masks() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    )
    actions = torch.ones(1, 2)
    objects = torch.ones(1, 3, 3)

    with pytest.raises(ValueError, match="bool"):
        head(actions, objects, torch.ones(1, 2, 3))
    with pytest.raises(ValueError, match="shape"):
        head(actions, objects, torch.ones(1, 3, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        head(actions, objects, torch.ones(1, 2, 2, dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_argument_reconstruction_head_rejects_device_mismatches() -> None:
    from acs_jepa.architectures import ArgumentReconstructionHead

    head = ArgumentReconstructionHead(
        action_dim=2,
        object_dim=3,
        max_action_arity=2,
        hidden_dim=4,
    ).cuda()
    actions = torch.ones(1, 2, device="cuda")
    objects = torch.ones(1, 3, 3, device="cuda")
    cpu_mask = torch.ones(1, 2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="device"):
        head(actions, objects.cpu())
    with pytest.raises(ValueError, match="device"):
        head(actions, objects, cpu_mask)


def test_argument_reconstruction_head_is_exported_from_package() -> None:
    import acs_jepa
    from acs_jepa.architectures import ArgumentReconstructionHead

    assert acs_jepa.ArgumentReconstructionHead is ArgumentReconstructionHead
