from __future__ import annotations

import torch

from src.fusion.model import FusionClassifier


def test_fusion_model_forward_shapes() -> None:
    config = {
        "dataset": {"num_classes": 5},
        "model": {
            "name": "light-bica-gate",
            "text_token_dim": 8,
            "speech_token_dim": 10,
            "text_embedding_dim": 8,
            "speech_embedding_dim": 10,
            "reliability_dim": 6,
            "fusion_dim": 12,
            "interaction_hidden_dim": 16,
            "interaction_dropout": 0.1,
            "gate_hidden_dim": 16,
            "gate_dropout": 0.1,
            "use_interaction": True,
            "gate_mode": "three_way_classwise",
            "use_residual_logit_correction": False,
        },
        "dominance": {
            "use_modality_dropout": False,
            "text_dropout_prob": 0.1,
            "speech_dropout_prob": 0.1,
        },
    }

    model = FusionClassifier(config)
    batch_size = 3

    text_embedding = torch.randn(batch_size, 8)
    speech_embedding = torch.randn(batch_size, 10)
    text_tokens = torch.randn(batch_size, 16, 8)
    speech_tokens = torch.randn(batch_size, 16, 10)
    text_mask = torch.ones(batch_size, 16, dtype=torch.long)
    speech_mask = torch.ones(batch_size, 16, dtype=torch.long)

    text_logits = torch.randn(batch_size, 5)
    speech_logits = torch.randn(batch_size, 5)
    text_probs = torch.softmax(text_logits, dim=-1)
    speech_probs = torch.softmax(speech_logits, dim=-1)
    reliability = torch.randn(batch_size, 6)

    output, aux = model(
        text_embedding=text_embedding,
        speech_embedding=speech_embedding,
        text_tokens=text_tokens,
        speech_tokens=speech_tokens,
        text_token_mask=text_mask,
        speech_token_mask=speech_mask,
        text_logits_raw=text_logits,
        speech_logits_raw=speech_logits,
        text_probs_cal=text_probs,
        speech_probs_cal=speech_probs,
        reliability=reliability,
        labels=torch.tensor([0, 1, 2]),
        return_aux=True,
    )

    assert output.logits.shape == (batch_size, 5)
    assert output.probs.shape == (batch_size, 5)
    assert aux["alpha_text"].shape == (batch_size, 5)
    assert aux["alpha_speech"].shape == (batch_size, 5)
    assert aux["alpha_interaction"].shape == (batch_size, 5)