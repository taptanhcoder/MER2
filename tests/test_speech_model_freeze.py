from types import SimpleNamespace

import torch
from torch import nn

from src.speech.models.hubert import HuBERTClassifier


class DummyConfig:
    hidden_size = 8


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 8)

    def forward(self, input_values, attention_mask=None, return_dict=True):
        hidden_states = self.proj(input_values.unsqueeze(-1))
        return SimpleNamespace(last_hidden_state=hidden_states)

    def _get_feature_vector_attention_mask(self, sequence_length, attention_mask):
        if attention_mask is None:
            return None
        return attention_mask[:, :sequence_length]


def test_hubert_model_freezes_encoder(monkeypatch):
    monkeypatch.setattr(
        "src.speech.models.hubert.AutoConfig.from_pretrained",
        lambda *args, **kwargs: DummyConfig(),
    )
    monkeypatch.setattr(
        "src.speech.models.hubert.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyEncoder(),
    )

    model = HuBERTClassifier(
        {
            "name": "hubert-base",
            "pretrained_name": "dummy/hubert",
            "pooling": "attention",
            "dropout": 0.1,
            "num_classes": 5,
            "freeze_encoder": True,
        }
    )

    encoder_requires_grad = {param.requires_grad for param in model.encoder.parameters()}
    classifier_requires_grad = {param.requires_grad for param in model.classifier.parameters()}

    assert encoder_requires_grad == {False}
    assert classifier_requires_grad == {True}
    assert model.trainable_parameter_count > 0