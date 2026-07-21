from __future__ import annotations

import math

import pytest
import torch


def test_argument_reconstruction_loss_matches_manual_cross_entropy() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    logits = torch.tensor([[[2.0, 0.0, float("-inf")], [float("-inf")] * 3]])
    targets = torch.tensor([[0, -1]])
    argument_mask = torch.tensor([[True, False]])
    candidate_mask = torch.tensor([[[True, True, False], [False, False, False]]])

    output = ArgumentReconstructionLoss()(logits, targets, argument_mask, candidate_mask)

    assert output.total.item() == pytest.approx(0.1269280110429727)
    assert output.role_accuracy.item() == pytest.approx(1.0)
    assert output.competitive_role_accuracy.item() == pytest.approx(1.0)
    assert output.mean_target_margin.item() == pytest.approx(2.0)
    assert output.num_active_roles == 1
    assert output.num_competitive_roles == 1
    for value in (
        output.total,
        output.role_accuracy,
        output.competitive_role_accuracy,
        output.mean_target_margin,
    ):
        assert value.shape == ()
        assert value.dtype == torch.float32
        assert torch.isfinite(value)


def test_argument_reconstruction_loss_excludes_masked_and_inactive_gradients() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    logits = torch.tensor(
        [[[2.0, 0.0, 100.0], [10.0, 20.0, 30.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, -1]])
    argument_mask = torch.tensor([[True, False]])
    candidate_mask = torch.tensor([[[True, True, False], [True, True, True]]])

    output = ArgumentReconstructionLoss()(logits, targets, argument_mask, candidate_mask)
    output.total.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 0, :2]).item() == 2
    assert logits.grad[0, 0, 2].item() == 0.0
    assert torch.equal(logits.grad[0, 1], torch.zeros_like(logits.grad[0, 1]))

    changed = logits.detach().clone()
    changed[0, 0, 2] = -1000.0
    changed[0, 1] = torch.tensor([-50.0, 0.0, 50.0])
    changed_output = ArgumentReconstructionLoss()(
        changed, targets, argument_mask, candidate_mask
    )
    assert torch.equal(changed_output.total, output.total.detach())
    assert torch.equal(changed_output.role_accuracy, output.role_accuracy)
    assert torch.equal(
        changed_output.competitive_role_accuracy,
        output.competitive_role_accuracy,
    )
    assert torch.equal(changed_output.mean_target_margin, output.mean_target_margin)


def test_argument_reconstruction_loss_strict_ties_and_singleton_roles() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    loss = ArgumentReconstructionLoss()
    tied = loss(
        torch.tensor([[[1.0, 1.0]]]),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        torch.tensor([[[True, True]]]),
    )
    assert tied.role_accuracy.item() == pytest.approx(0.0)
    assert tied.competitive_role_accuracy.item() == pytest.approx(0.0)
    assert tied.mean_target_margin.item() == pytest.approx(0.0)
    assert tied.num_competitive_roles == 1

    singleton = loss(
        torch.tensor([[[5.0, float("-inf")]]]),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        torch.tensor([[[True, False]]]),
    )
    assert singleton.total.item() == pytest.approx(0.0)
    assert singleton.role_accuracy.item() == pytest.approx(1.0)
    assert singleton.competitive_role_accuracy.item() == pytest.approx(0.0)
    assert singleton.mean_target_margin.item() == pytest.approx(0.0)
    assert singleton.num_active_roles == 1
    assert singleton.num_competitive_roles == 0
    for value in (
        singleton.total,
        singleton.role_accuracy,
        singleton.competitive_role_accuracy,
        singleton.mean_target_margin,
    ):
        assert value.shape == ()
        assert torch.isfinite(value)


@pytest.mark.parametrize("all_inf", [False, True])
def test_argument_reconstruction_loss_handles_no_active_roles(all_inf: bool) -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    logits = torch.full(
        (2, 2, 3), float("-inf") if all_inf else 1.0, requires_grad=True
    )
    targets = torch.full((2, 2), -1, dtype=torch.long)
    argument_mask = torch.zeros(2, 2, dtype=torch.bool)
    candidate_mask = torch.full(
        (2, 2, 3), not all_inf, dtype=torch.bool
    )

    output = ArgumentReconstructionLoss()(logits, targets, argument_mask, candidate_mask)
    output.total.backward()

    assert output.num_active_roles == 0
    assert output.num_competitive_roles == 0
    for value in (
        output.total,
        output.role_accuracy,
        output.competitive_role_accuracy,
        output.mean_target_margin,
    ):
        assert value.shape == ()
        assert value.dtype == torch.float32
        assert value.item() == pytest.approx(0.0)
        assert torch.isfinite(value)
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_argument_reconstruction_loss_is_candidate_permutation_invariant() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    logits = torch.tensor([[[0.2, 2.0, -1.0, 0.5]]])
    targets = torch.tensor([[1]])
    argument_mask = torch.tensor([[True]])
    candidate_mask = torch.tensor([[[True, True, False, True]]])
    permutation = torch.tensor([2, 0, 3, 1])
    inverse_permutation = torch.argsort(permutation)
    remapped_targets = inverse_permutation[targets]

    original = ArgumentReconstructionLoss()(
        logits, targets, argument_mask, candidate_mask
    )
    permuted = ArgumentReconstructionLoss()(
        logits[:, :, permutation],
        remapped_targets,
        argument_mask,
        candidate_mask[:, :, permutation],
    )

    assert remapped_targets.item() == 3
    assert original.num_active_roles == permuted.num_active_roles
    assert original.num_competitive_roles == permuted.num_competitive_roles
    for original_value, permuted_value in zip(
        (
            original.total,
            original.role_accuracy,
            original.competitive_role_accuracy,
            original.mean_target_margin,
        ),
        (
            permuted.total,
            permuted.role_accuracy,
            permuted.competitive_role_accuracy,
            permuted.mean_target_margin,
        ),
        strict=True,
    ):
        assert torch.allclose(original_value, permuted_value)


def test_argument_reconstruction_loss_rejects_invalid_shapes() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    loss = ArgumentReconstructionLoss()
    logits = torch.ones(2, 2, 3)
    targets = torch.zeros(2, 2, dtype=torch.long)
    argument_mask = torch.ones(2, 2, dtype=torch.bool)
    candidate_mask = torch.ones(2, 2, 3, dtype=torch.bool)
    invalid = [
        (torch.ones(2, 3), targets, argument_mask, candidate_mask),
        (
            torch.empty(0, 2, 3),
            torch.empty(0, 2, dtype=torch.long),
            torch.empty(0, 2, dtype=torch.bool),
            torch.empty(0, 2, 3, dtype=torch.bool),
        ),
        (
            torch.empty(2, 0, 3),
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(2, 0, dtype=torch.bool),
            torch.empty(2, 0, 3, dtype=torch.bool),
        ),
        (
            torch.empty(2, 2, 0),
            targets,
            argument_mask,
            torch.empty(2, 2, 0, dtype=torch.bool),
        ),
        (logits, torch.zeros(2, 3, dtype=torch.long), argument_mask, candidate_mask),
        (logits, targets, torch.ones(2, 3, dtype=torch.bool), candidate_mask),
        (logits, targets, argument_mask, torch.ones(2, 2, 2, dtype=torch.bool)),
    ]

    for values in invalid:
        with pytest.raises(ValueError, match="shape"):
            loss(*values)


def test_argument_reconstruction_loss_rejects_invalid_dtypes() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    loss = ArgumentReconstructionLoss()
    logits = torch.ones(1, 1, 2)
    targets = torch.zeros(1, 1, dtype=torch.long)
    argument_mask = torch.ones(1, 1, dtype=torch.bool)
    candidate_mask = torch.ones(1, 1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="floating"):
        loss(logits.to(torch.int64), targets, argument_mask, candidate_mask)
    with pytest.raises(ValueError, match="long"):
        loss(logits, targets.to(torch.int32), argument_mask, candidate_mask)
    with pytest.raises(ValueError, match="bool"):
        loss(logits, targets, argument_mask.to(torch.int64), candidate_mask)
    with pytest.raises(ValueError, match="bool"):
        loss(logits, targets, argument_mask, candidate_mask.to(torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_argument_reconstruction_loss_rejects_device_mismatches() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    loss = ArgumentReconstructionLoss()
    logits = torch.ones(1, 1, 2, device="cuda")
    targets = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    argument_mask = torch.ones(1, 1, dtype=torch.bool, device="cuda")
    candidate_mask = torch.ones(1, 1, 2, dtype=torch.bool, device="cuda")

    with pytest.raises(ValueError, match="device"):
        loss(logits, targets.cpu(), argument_mask, candidate_mask)
    with pytest.raises(ValueError, match="device"):
        loss(logits, targets, argument_mask.cpu(), candidate_mask)
    with pytest.raises(ValueError, match="device"):
        loss(logits, targets, argument_mask, candidate_mask.cpu())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("position", [0, 1])
def test_argument_reconstruction_loss_rejects_nonfinite_logits(
    bad_value: float,
    position: int,
) -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    logits = torch.tensor([[[2.0, 0.0]]])
    logits[0, 0, position] = bad_value
    candidate_mask = torch.tensor([[[True, False]]])

    with pytest.raises(ValueError, match="finite"):
        ArgumentReconstructionLoss()(
            logits,
            torch.tensor([[0]]),
            torch.tensor([[True]]),
            candidate_mask,
        )


def test_argument_reconstruction_loss_enforces_negative_infinity_mask_semantics() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    loss = ArgumentReconstructionLoss()
    with pytest.raises(ValueError, match="finite"):
        loss(
            torch.tensor([[[float("-inf"), 0.0]]]),
            torch.tensor([[0]]),
            torch.tensor([[True]]),
            torch.tensor([[[True, False]]]),
        )

    accepted = loss(
        torch.tensor([[[2.0, float("-inf")]]]),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        torch.tensor([[[True, False]]]),
    )
    assert torch.isfinite(accepted.total)


def test_argument_reconstruction_loss_validates_targets_before_indexing() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    loss = ArgumentReconstructionLoss()
    logits = torch.ones(1, 1, 2)
    active = torch.tensor([[True]])
    candidates = torch.ones(1, 1, 2, dtype=torch.bool)

    for invalid_target in (-1, 2, 10):
        with pytest.raises(ValueError, match="target"):
            loss(logits, torch.tensor([[invalid_target]]), active, candidates)

    with pytest.raises(ValueError, match="inactive"):
        loss(
            logits,
            torch.tensor([[0]]),
            torch.tensor([[False]]),
            candidates,
        )
    with pytest.raises(ValueError, match="candidate"):
        loss(
            logits,
            torch.tensor([[1]]),
            active,
            torch.tensor([[[True, False]]]),
        )
    with pytest.raises(ValueError, match="candidate"):
        loss(
            logits,
            torch.tensor([[0]]),
            active,
            torch.zeros(1, 1, 2, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
@pytest.mark.parametrize("empty_all_inf", [False, True])
def test_argument_reconstruction_loss_output_dtype_contract(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    empty_all_inf: bool,
) -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    loss = ArgumentReconstructionLoss()
    normal = loss(
        torch.tensor([[[1.0, 0.0]]], dtype=input_dtype),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        torch.tensor([[[True, True]]]),
    )
    empty_logits = torch.full(
        (1, 1, 2),
        float("-inf") if empty_all_inf else 1.0,
        dtype=input_dtype,
    )
    empty = loss(
        empty_logits,
        torch.tensor([[-1]]),
        torch.tensor([[False]]),
        torch.tensor([[[False, False]]]),
    )

    for output in (normal, empty):
        for value in (
            output.total,
            output.role_accuracy,
            output.competitive_role_accuracy,
            output.mean_target_margin,
        ):
            assert value.shape == ()
            assert value.dtype == output_dtype
            assert torch.isfinite(value)


def test_argument_reconstruction_loss_float16_extremes_remain_finite() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    maximum = torch.finfo(torch.float16).max
    output = ArgumentReconstructionLoss()(
        torch.tensor([[[-maximum, maximum]]], dtype=torch.float16),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        torch.tensor([[[True, True]]]),
    )

    assert output.total.item() == pytest.approx(131008.0)
    assert output.mean_target_margin.item() == pytest.approx(-131008.0)
    assert output.total.dtype == torch.float32
    assert torch.isfinite(output.total)
    assert torch.isfinite(output.mean_target_margin)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_argument_reconstruction_loss_rejects_nonfinite_derived_outputs(
    dtype: torch.dtype,
) -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    maximum = torch.finfo(dtype).max
    with pytest.raises(ValueError, match="derived"):
        ArgumentReconstructionLoss()(
            torch.tensor([[[-maximum, maximum]]], dtype=dtype),
            torch.tensor([[0]]),
            torch.tensor([[True]]),
            torch.tensor([[[True, True]]]),
        )


def test_argument_reconstruction_loss_handles_mixed_variable_arity() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    negative_infinity = float("-inf")
    logits = torch.tensor(
        [
            [[3.0, 1.0, 0.0], [0.0, 2.0, -1.0], [negative_infinity] * 3],
            [[-1.0, 0.0, 2.0], [negative_infinity] * 3, [negative_infinity] * 3],
        ]
    )
    output = ArgumentReconstructionLoss()(
        logits,
        torch.tensor([[0, 1, -1], [2, -1, -1]]),
        torch.tensor([[True, True, False], [True, False, False]]),
        torch.tensor(
            [
                [[True, True, True], [True, True, False], [False] * 3],
                [[False, True, True], [False] * 3, [False] * 3],
            ]
        ),
    )

    first = math.log(math.exp(3.0) + math.exp(1.0) + math.exp(0.0)) - 3.0
    second = math.log(math.exp(2.0) + math.exp(0.0)) - 2.0
    assert output.total.item() == pytest.approx((first + second + second) / 3.0)
    assert output.role_accuracy.item() == pytest.approx(1.0)
    assert output.competitive_role_accuracy.item() == pytest.approx(1.0)
    assert output.mean_target_margin.item() == pytest.approx(2.0)
    assert output.num_active_roles == 3
    assert output.num_competitive_roles == 3


def test_argument_reconstruction_loss_masking_is_role_specific() -> None:
    from acs_jepa.losses import ArgumentReconstructionLoss

    logits = torch.tensor(
        [[[2.0, 100.0, 0.0], [0.0, 2.0, -1.0]]], requires_grad=True
    )
    output = ArgumentReconstructionLoss()(
        logits,
        torch.tensor([[0, 1]]),
        torch.tensor([[True, True]]),
        torch.tensor([[[True, False, True], [True, True, False]]]),
    )
    output.total.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0, 1].item() == 0.0
    assert logits.grad[0, 1, 1].item() != 0.0
    assert logits.grad[0, 1, 2].item() == 0.0


def test_argument_reconstruction_loss_is_publicly_exported() -> None:
    from acs_jepa import ArgumentReconstructionLoss, ArgumentReconstructionLossOutput

    assert ArgumentReconstructionLoss.__name__ == "ArgumentReconstructionLoss"
    assert ArgumentReconstructionLossOutput.__name__ == "ArgumentReconstructionLossOutput"
