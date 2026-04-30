from __future__ import annotations

import torch

from src.fusion.features import (
    apply_temperature_scaling,
    build_modality_priors_from_valid_artifact,
    build_reliability_vector,
    logits_to_probs,
)


def test_temperature_scaling_scalar() -> None:
    logits = torch.tensor([[2.0, 1.0]])
    scaled = apply_temperature_scaling(logits, {"temperature": 2.0})
    assert scaled.shape == logits.shape
    assert torch.allclose(scaled, logits / 2.0)


def test_build_reliability_vector_shape() -> None:
    text_probs = logits_to_probs(torch.randn(4, 5))
    speech_probs = logits_to_probs(torch.randn(4, 5))
    reliability = build_reliability_vector(
        text_probs=text_probs,
        speech_probs=speech_probs,
        use_confidence=True,
        use_entropy=True,
        use_margin=True,
        use_variance=False,
    )
    assert reliability.shape == (4, 6)


def test_build_modality_priors_from_valid_artifact() -> None:
    artifact = {
        "label_id": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "text_probs_cal": torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.6, 0.4],
            ],
            dtype=torch.float32,
        ),
        "speech_probs_cal": torch.tensor(
            [
                [0.6, 0.4],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
            ],
            dtype=torch.float32,
        ),
    }
    prior_text, prior_speech = build_modality_priors_from_valid_artifact(artifact)
    assert prior_text.shape == (2,)
    assert prior_speech.shape == (2,)
    assert torch.all(prior_text >= 0)
    assert torch.all(prior_speech >= 0)