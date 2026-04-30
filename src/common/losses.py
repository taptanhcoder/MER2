from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.registry import LOSS_REGISTRY


def _to_float_tensor(values: Sequence[float] | torch.Tensor | None) -> torch.Tensor | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.detach().float().clone()
    return torch.tensor(list(values), dtype=torch.float32)


def balanced_class_weights_from_counts(class_counts: Sequence[int] | None) -> list[float] | None:
    if class_counts is None:
        return None

    counts = [max(int(x), 0) for x in class_counts]
    if not counts or sum(counts) <= 0:
        return None

    num_classes = len(counts)
    total = float(sum(counts))
    weights = []
    for count in counts:
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * float(count)))

    weight_sum = sum(weights)
    if weight_sum > 0:
        scale = float(num_classes) / weight_sum
        weights = [w * scale for w in weights]

    return weights


def drw_class_weights_from_counts(
    class_counts: Sequence[int] | None,
    beta: float = 0.9999,
) -> list[float] | None:
    if class_counts is None:
        return None

    counts = [max(int(x), 0) for x in class_counts]
    if not counts or sum(counts) <= 0:
        return None

    effective_nums = []
    for count in counts:
        if count <= 0:
            effective_nums.append(0.0)
        else:
            effective_nums.append((1.0 - beta**count) / (1.0 - beta))

    weights = []
    for effective_num in effective_nums:
        if effective_num <= 0:
            weights.append(0.0)
        else:
            weights.append(1.0 / effective_num)

    weight_sum = sum(weights)
    if weight_sum > 0:
        scale = float(len(weights)) / weight_sum
        weights = [w * scale for w in weights]

    return weights


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = str(reduction)

        weight_tensor = _to_float_tensor(class_weights)
        if weight_tensor is None:
            self.register_buffer("class_weights", torch.empty(0), persistent=True)
        else:
            self.register_buffer("class_weights", weight_tensor, persistent=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.device)

        weight = self.class_weights
        if weight.numel() == 0:
            weight = None
        else:
            weight = weight.to(logits.device)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        raise ValueError(f"Unsupported reduction for FocalLoss: {self.reduction}")


class LDAMLoss(nn.Module):
    def __init__(
        self,
        class_counts: Sequence[int],
        max_m: float = 0.5,
        scale: float = 30.0,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        drw_enabled: bool = False,
        drw_start_epoch: int = 2,
        drw_beta: float = 0.9999,
    ) -> None:
        super().__init__()

        counts = [max(int(x), 1) for x in class_counts]
        margins = 1.0 / torch.pow(torch.tensor(counts, dtype=torch.float32), 0.25)
        margins = margins * (float(max_m) / float(margins.max()))
        self.register_buffer("margins", margins, persistent=True)

        base_class_weights = _to_float_tensor(class_weights)
        if base_class_weights is None:
            self.register_buffer("base_class_weights", torch.empty(0), persistent=True)
        else:
            self.register_buffer("base_class_weights", base_class_weights, persistent=True)

        drw_weights = _to_float_tensor(drw_class_weights_from_counts(counts, beta=drw_beta))
        if drw_weights is None:
            self.register_buffer("drw_class_weights", torch.empty(0), persistent=True)
        else:
            self.register_buffer("drw_class_weights", drw_weights, persistent=True)

        self.scale = float(scale)
        self.drw_enabled = bool(drw_enabled)
        self.drw_start_epoch = int(drw_start_epoch)
        self.current_epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _get_active_class_weights(self, logits: torch.Tensor) -> torch.Tensor | None:
        if self.drw_enabled and self.current_epoch >= self.drw_start_epoch:
            if self.drw_class_weights.numel() > 0:
                return self.drw_class_weights.to(logits.device)

        if self.base_class_weights.numel() > 0:
            return self.base_class_weights.to(logits.device)

        return None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.device)
        margins = self.margins.to(logits.device)

        adjusted_logits = logits.clone()
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_margins = margins[targets]
        adjusted_logits[batch_indices, targets] -= target_margins

        class_weights = self._get_active_class_weights(logits)
        loss = F.cross_entropy(
            self.scale * adjusted_logits,
            targets,
            weight=class_weights,
            reduction="mean",
        )
        return loss


@LOSS_REGISTRY.register("cross_entropy")
def build_cross_entropy_loss(loss_cfg: dict[str, Any]) -> nn.Module:
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


@LOSS_REGISTRY.register("weighted_cross_entropy")
def build_weighted_cross_entropy_loss(loss_cfg: dict[str, Any]) -> nn.Module:
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    class_weights = loss_cfg.get("class_weights")

    if class_weights == "balanced":
        class_weights = balanced_class_weights_from_counts(loss_cfg.get("class_counts"))

    weight_tensor = _to_float_tensor(class_weights)
    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=label_smoothing,
    )


@LOSS_REGISTRY.register("focal")
def build_focal_loss(loss_cfg: dict[str, Any]) -> nn.Module:
    gamma = float(loss_cfg.get("gamma", 2.0))
    class_weights = loss_cfg.get("class_weights")

    if class_weights == "balanced":
        class_weights = balanced_class_weights_from_counts(loss_cfg.get("class_counts"))

    return FocalLoss(
        gamma=gamma,
        class_weights=class_weights,
        reduction=str(loss_cfg.get("reduction", "mean")),
    )


@LOSS_REGISTRY.register("ldam")
@LOSS_REGISTRY.register("ldam_drw")
def build_ldam_loss(loss_cfg: dict[str, Any]) -> nn.Module:
    class_counts = loss_cfg.get("class_counts")
    if class_counts is None:
        raise ValueError("LDAMLoss requires `class_counts` in loss config.")

    class_weights = loss_cfg.get("class_weights")
    if class_weights == "balanced":
        class_weights = balanced_class_weights_from_counts(class_counts)

    drw_enabled = bool(loss_cfg.get("drw_enabled", True))
    return LDAMLoss(
        class_counts=class_counts,
        max_m=float(loss_cfg.get("max_m", 0.5)),
        scale=float(loss_cfg.get("scale", 30.0)),
        class_weights=class_weights,
        drw_enabled=drw_enabled,
        drw_start_epoch=int(loss_cfg.get("drw_start_epoch", 2)),
        drw_beta=float(loss_cfg.get("drw_beta", 0.9999)),
    )


def build_loss(loss_cfg: dict[str, Any]) -> nn.Module:
    if "name" not in loss_cfg:
        raise KeyError("Loss config must contain a `name` field.")
    loss_name = str(loss_cfg["name"])
    builder = LOSS_REGISTRY.get(loss_name)
    return builder(loss_cfg)