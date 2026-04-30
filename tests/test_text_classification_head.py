from __future__ import annotations

import torch

from src.text.heads import ClassificationHead


def test_classification_head_new_api_without_hidden_layer() -> None:
    head = ClassificationHead(
        input_dim=8,
        hidden_dim=None,
        output_dim=5,
        dropout=0.1,
        activation="tanh",
    )
    x = torch.randn(4, 8)
    y = head(x)
    assert y.shape == (4, 5)


def test_classification_head_new_api_with_hidden_layer() -> None:
    head = ClassificationHead(
        input_dim=8,
        hidden_dim=16,
        output_dim=5,
        dropout=0.1,
        activation="relu",
    )
    x = torch.randn(3, 8)
    y = head(x)
    assert y.shape == (3, 5)


def test_classification_head_backward_compatible_api() -> None:
    head = ClassificationHead(
        hidden_size=8,
        num_classes=5,
        dropout=0.1,
    )
    x = torch.randn(2, 8)
    y = head(x)
    assert y.shape == (2, 5)