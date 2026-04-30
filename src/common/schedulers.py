from __future__ import annotations

from typing import Any

import torch # type: ignore

from src.common.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("none")
def build_none(optimizer: torch.optim.Optimizer, **kwargs: Any) -> None:
    return None


@SCHEDULER_REGISTRY.register("step_lr")
def build_step_lr(
    optimizer: torch.optim.Optimizer, **kwargs: Any
) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)


@SCHEDULER_REGISTRY.register("cosine")
def build_cosine(
    optimizer: torch.optim.Optimizer, **kwargs: Any
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)


@SCHEDULER_REGISTRY.register("linear_warmup")
def build_linear_warmup(
    optimizer: torch.optim.Optimizer, **kwargs: Any
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(kwargs.pop("total_steps"))
    warmup_ratio = float(kwargs.pop("warmup_ratio", 0.0))
    warmup_steps = max(0, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_scheduler(scheduler_config: dict[str, Any] | None, optimizer: torch.optim.Optimizer):
    if not scheduler_config:
        return None
    config = dict(scheduler_config)
    name = config.pop("name", "none")
    factory = SCHEDULER_REGISTRY.get(name)
    return factory(optimizer, **config)