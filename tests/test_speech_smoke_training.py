from types import SimpleNamespace

import torch
from torch import nn

from src.speech.models.hubert import HuBERTClassifier
from src.speech.trainer import SpeechTrainer, epoch_result_to_dataframe


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


def test_speech_smoke_forward_and_dataframe(monkeypatch):
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

    batch = {
        "ids": ["a", "b"],
        "paths": ["a.wav", "b.wav"],
        "filenames": ["a.wav", "b.wav"],
        "group_ids": ["g1", "g2"],
        "label_names": ["anger", "sadness"],
        "raw_lengths": torch.tensor([12, 10], dtype=torch.long),
        "durations_sec": torch.tensor([0.1, 0.08], dtype=torch.float32),
        "labels": torch.tensor([0, 3], dtype=torch.long),
        "input_values": torch.randn(2, 12),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        ),
    }

    trainer = SpeechTrainer.__new__(SpeechTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = model

    prepared = SpeechTrainer.prepare_batch(trainer, batch)
    output = SpeechTrainer.forward_step(trainer, prepared)

    assert output.logits.shape == (2, 5)
    assert output.probs.shape == (2, 5)
    assert output.preds.shape == (2,)

    result = {
        "ids": batch["ids"],
        "labels": prepared["labels"],
        "preds": output.preds,
        "probs": output.probs,
        "logits": output.logits,
        "metadata": {
            "paths": batch["paths"],
            "group_ids": batch["group_ids"],
            "raw_lengths": batch["raw_lengths"],
            "durations_sec": batch["durations_sec"],
        },
    }
    df = epoch_result_to_dataframe(
        result,
        split_name="test",
        label_names=["anger", "fear", "happiness", "sadness", "neutral"],
    )

    assert len(df) == 2
    assert "confidence" in df.columns
    assert "entropy" in df.columns
    assert "top2_margin" in df.columns
    assert "path" in df.columns