from __future__ import annotations

import torch


def dominance_regularizer(
    alpha_text: torch.Tensor,
    alpha_speech: torch.Tensor,
    prior_text: torch.Tensor,
    prior_speech: torch.Tensor,
    lambda_text: float = 1.0,
    lambda_speech: float = 1.0,
) -> torch.Tensor:
    mean_alpha_text = alpha_text.mean(dim=0)
    mean_alpha_speech = alpha_speech.mean(dim=0)

    prior_text = prior_text.to(alpha_text.device, dtype=alpha_text.dtype)
    prior_speech = prior_speech.to(alpha_speech.device, dtype=alpha_speech.dtype)

    reg_text = torch.abs(mean_alpha_text - prior_text).mean()
    reg_speech = torch.abs(mean_alpha_speech - prior_speech).mean()

    return float(lambda_text) * reg_text + float(lambda_speech) * reg_speech