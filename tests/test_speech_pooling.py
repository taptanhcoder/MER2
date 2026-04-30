from __future__ import annotations

import torch

from src.speech.pooling import build_pooling


def test_attention_pooling_output_shapes() -> None:
    pooling = build_pooling(
        {"name": "attention", "hidden_size": 8, "dropout": 0.0},
        hidden_size=8,
    )

    hidden_states = torch.randn(2, 5, 8)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    pooled, weights = pooling(hidden_states, attention_mask=attention_mask)

    assert pooled.shape == (2, 8)
    assert weights.shape == (2, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_mean_pooling_output_shapes() -> None:
    pooling = build_pooling({"name": "mean"}, hidden_size=8)

    hidden_states = torch.randn(3, 7, 8)
    attention_mask = torch.ones(3, 7, dtype=torch.long)

    pooled, weights = pooling(hidden_states, attention_mask=attention_mask)

    assert pooled.shape == (3, 8)
    assert weights is None