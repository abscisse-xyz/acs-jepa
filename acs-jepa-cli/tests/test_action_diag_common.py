from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = ROOT / "script" / "action_diag_common.py"


def _common_module():
    spec = importlib.util.spec_from_file_location("action_diag_common", COMMON_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(*, optional: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        jepa=nn.Linear(2, 2),
        goal_head=nn.Linear(2, 1) if optional else None,
        action_contrastive_anchor=nn.Linear(2, 2) if optional else None,
        argument_reconstruction_head=nn.Linear(2, 2) if optional else None,
        applicability_head=nn.Linear(2, 1) if optional else None,
    )


def _checkpoint(bundle: SimpleNamespace) -> dict[str, object]:
    return {
        "model_state_dict": bundle.jepa.state_dict(),
        "goal_head_state_dict": bundle.goal_head.state_dict(),
        "action_contrastive_anchor_state_dict": bundle.action_contrastive_anchor.state_dict(),
        "argument_reconstruction_head_state_dict": bundle.argument_reconstruction_head.state_dict(),
        "applicability_head_state_dict": bundle.applicability_head.state_dict(),
    }


def test_strict_restoration_loads_every_constructed_module_and_sets_eval() -> None:
    common = _common_module()
    source = _bundle()
    restored = _bundle()
    for module in vars(restored).values():
        module.train()

    metadata = common.restore_diagnostic_checkpoint_modules(restored, _checkpoint(source))

    assert set(metadata) == {
        "jepa",
        "goal_head",
        "action_contrastive_anchor",
        "argument_reconstruction_head",
        "applicability_head",
    }
    assert all(entry["status"] == "restored" for entry in metadata.values())
    assert all(module.training is False for module in vars(restored).values())
    for name, source_module in vars(source).items():
        restored_module = getattr(restored, name)
        for key, value in source_module.state_dict().items():
            assert torch.equal(value, restored_module.state_dict()[key])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda checkpoint: checkpoint.pop("applicability_head_state_dict"), "missing"),
        (lambda checkpoint: checkpoint.__setitem__("applicability_head_state_dict", None), "null"),
    ],
)
def test_strict_restoration_rejects_missing_or_null_configured_state(mutation, message: str) -> None:
    common = _common_module()
    bundle = _bundle()
    checkpoint = _checkpoint(bundle)
    mutation(checkpoint)

    with pytest.raises(ValueError, match=rf"applicability_head_state_dict.*{message}"):
        common.restore_diagnostic_checkpoint_modules(bundle, checkpoint)


def test_strict_restoration_rejects_incompatible_configured_state() -> None:
    common = _common_module()
    bundle = _bundle()
    checkpoint = _checkpoint(bundle)
    checkpoint["argument_reconstruction_head_state_dict"] = nn.Linear(3, 2).state_dict()

    with pytest.raises(ValueError, match="argument_reconstruction_head_state_dict.*incompatible"):
        common.restore_diagnostic_checkpoint_modules(bundle, checkpoint)


def test_strict_restoration_allows_disabled_baseline_modules() -> None:
    common = _common_module()
    bundle = _bundle(optional=False)
    checkpoint = {"model_state_dict": bundle.jepa.state_dict()}

    metadata = common.restore_diagnostic_checkpoint_modules(bundle, checkpoint)

    assert metadata["jepa"]["status"] == "restored"
    assert all(metadata[name]["status"] == "disabled" for name in metadata if name != "jepa")
    assert bundle.jepa.training is False
