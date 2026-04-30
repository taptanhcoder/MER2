from __future__ import annotations

import torch
import torch.nn as nn


def _build_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.GELU()


class ClassificationHead(nn.Module):
    """
    Backward-compatible speech classification head.

    - If hidden_dim is None: dropout + linear
    - If hidden_dim is provided: small MLP head
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Dropout(dropout)]
        if hidden_dim is None:
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(_build_activation(activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)