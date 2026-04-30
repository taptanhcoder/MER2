from __future__ import annotations

import torch

from src.common.losses import (
    LDAMLoss,
    FocalLoss,
    balanced_class_weights_from_counts,
    build_loss,
)


def test_balanced_class_weights_from_counts() -> None:
    weights = balanced_class_weights_from_counts([10, 20, 40])
    assert weights is not None
    assert len(weights) == 3
    assert weights[0] > weights[1] > weights[2]


def test_weighted_cross_entropy_builds_module() -> None:
    criterion = build_loss(
        {
            "name": "weighted_cross_entropy",
            "class_weights": [1.0, 2.0, 3.0],
        }
    )
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.weight is not None
    assert criterion.weight.shape[0] == 3


def test_focal_loss_forward_cpu() -> None:
    criterion = FocalLoss(gamma=2.0, class_weights=[1.0, 2.0, 1.5])
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    loss = criterion(logits, targets)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_ldam_loss_forward_cpu() -> None:
    criterion = LDAMLoss(
        class_counts=[10, 20, 30],
        max_m=0.5,
        scale=30.0,
        class_weights=[1.0, 1.0, 1.0],
        drw_enabled=True,
        drw_start_epoch=2,
    )
    criterion.set_epoch(3)

    logits = torch.randn(5, 3)
    targets = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

    loss = criterion(logits, targets)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_loss_modules_can_move_device() -> None:
    focal = FocalLoss(gamma=2.0, class_weights=[1.0, 2.0])
    ldam = LDAMLoss(class_counts=[10, 20], drw_enabled=True)

    focal.to(torch.device("cpu"))
    ldam.to(torch.device("cpu"))

    assert focal.class_weights.device.type == "cpu"
    assert ldam.margins.device.type == "cpu"