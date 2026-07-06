"""Learning-rate scheduler helpers for the ACS-JEPA CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@dataclass
class NoOpScheduler:
    """Scheduler-compatible object that leaves optimizer learning rates unchanged."""

    optimizer: torch.optim.Optimizer

    def step(self) -> None:
        return None

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "none"}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return None


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_ratio: float,
        min_lr: float,
        start_factor: float,
    ) -> None:
        if total_steps <= 1:
            raise ValueError("WarmupCosineScheduler requires total_steps > 1")
        if not 0.0 < warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if not 0.0 < start_factor <= 1.0:
            raise ValueError("start_factor must be between 0 and 1")
        if min_lr < 0.0:
            raise ValueError("min_lr must be non-negative")

        warmup_steps = max(1, int(total_steps * warmup_ratio))
        warmup_steps = min(warmup_steps, total_steps - 1)
        cosine_steps = max(1, total_steps - warmup_steps)
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)
        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def step(self) -> None:
        self.scheduler.step()

    def get_last_lr(self) -> list[float]:
        return [float(lr) for lr in self.scheduler.get_last_lr()]

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DictConfig,
    *,
    total_steps: int,
) -> NoOpScheduler | WarmupCosineScheduler:
    """Build the configured optimizer scheduler."""

    scheduler_cfg = config.optimizer.scheduler
    kind = str(scheduler_cfg.kind).lower()
    if kind == "none" or total_steps <= 1:
        return NoOpScheduler(optimizer)
    if kind == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            total_steps=int(total_steps),
            warmup_ratio=float(scheduler_cfg.warmup_ratio),
            min_lr=float(scheduler_cfg.min_lr),
            start_factor=float(scheduler_cfg.start_factor),
        )
    raise ValueError(f"Unknown scheduler kind: {kind}")
