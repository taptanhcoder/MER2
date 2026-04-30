from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from src.text.models.phobert import PhoBERTClassifier


class DummyHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, 4)
        return SimpleNamespace(last_hidden_state=hidden)


def test_phobert_resolves_num_classes_from_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.text.models.phobert.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(hidden_size=4),
    )
    monkeypatch.setattr(
        "src.text.models.phobert.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyHFModel(),
    )

    config = {
        "model": {
            "name": "phobert-base",
            "pretrained_name": "vinai/phobert-base-v2",
            "dropout": 0.1,
            "pooling": "cls",
        },
        "dataset": {
            "num_classes": 5,
        },
        "label_space": {
            "labels": ["anger", "fear", "happiness", "neutral", "sadness"],
        },
    }

    model = PhoBERTClassifier(config)
    assert model.num_classes == 5


def test_phobert_resolves_num_classes_from_label_space(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.text.models.phobert.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(hidden_size=4),
    )
    monkeypatch.setattr(
        "src.text.models.phobert.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyHFModel(),
    )

    config = {
        "model": {
            "name": "phobert-base",
            "pretrained_name": "vinai/phobert-base-v2",
            "dropout": 0.1,
            "pooling": "cls",
        },
        "label_space": {
            "labels": ["anger", "fear", "happiness", "neutral", "sadness"],
        },
    }

    model = PhoBERTClassifier(config)
    assert model.num_classes == 5