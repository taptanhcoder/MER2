from __future__ import annotations

from typing import Any, Iterable

import torch # type: ignore

from src.common.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register("adam")
def build_adam(parameters: Iterable[torch.nn.Parameter], **kwargs: Any) -> torch.optim.Optimizer:
    return torch.optim.Adam(parameters, **kwargs)


@OPTIMIZER_REGISTRY.register("adamw")
def build_adamw(parameters: Iterable[torch.nn.Parameter], **kwargs: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, **kwargs)


@OPTIMIZER_REGISTRY.register("sgd")
def build_sgd(parameters: Iterable[torch.nn.Parameter], **kwargs: Any) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, **kwargs)


def build_optimizer(
    optimizer_config: dict[str, Any],
    parameters: Iterable[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    config = dict(optimizer_config)
    name = config.pop("name")
    factory = OPTIMIZER_REGISTRY.get(name)
    return factory(parameters, **config)