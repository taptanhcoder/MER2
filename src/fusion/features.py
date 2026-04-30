from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_calibration_state(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_temperature_scaling(
    logits: torch.Tensor,
    calibration_state: dict[str, Any] | None,
) -> torch.Tensor:
    if calibration_state is None:
        return logits

    if calibration_state.get("class_temperature") is not None:
        temperatures = torch.tensor(
            calibration_state["class_temperature"],
            dtype=logits.dtype,
            device=logits.device,
        ).clamp_min(1e-6)
        return logits / temperatures

    temperature = float(calibration_state.get("temperature", 1.0))
    temperature = max(temperature, 1e-6)
    return logits / temperature


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def confidence_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs.max(dim=-1).values.unsqueeze(-1)


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    entropy = -(probs * (probs.clamp_min(eps).log())).sum(dim=-1, keepdim=True)
    return entropy


def margin_from_probs(probs: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        return top2[:, :1]
    return (top2[:, 0] - top2[:, 1]).unsqueeze(-1)


def variance_feature(
    probs: torch.Tensor,
    mc_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    if mc_probs is None:
        return torch.zeros(probs.shape[0], 1, dtype=probs.dtype, device=probs.device)
    # mc_probs: [K, N, C]
    pred_variance = mc_probs.var(dim=0).mean(dim=-1, keepdim=True)
    return pred_variance


def build_reliability_vector(
    text_probs: torch.Tensor,
    speech_probs: torch.Tensor,
    use_confidence: bool = True,
    use_entropy: bool = True,
    use_margin: bool = True,
    use_variance: bool = False,
    text_mc_probs: torch.Tensor | None = None,
    speech_mc_probs: torch.Tensor | None = None,
    asr_features: torch.Tensor | None = None,
) -> torch.Tensor:
    features: list[torch.Tensor] = []

    if use_confidence:
        features.append(confidence_from_probs(text_probs))
        features.append(confidence_from_probs(speech_probs))

    if use_entropy:
        features.append(entropy_from_probs(text_probs))
        features.append(entropy_from_probs(speech_probs))

    if use_margin:
        features.append(margin_from_probs(text_probs))
        features.append(margin_from_probs(speech_probs))

    if use_variance:
        features.append(variance_feature(text_probs, text_mc_probs))
        features.append(variance_feature(speech_probs, speech_mc_probs))

    if asr_features is not None:
        features.append(asr_features)

    if not features:
        return torch.zeros(text_probs.shape[0], 0, dtype=text_probs.dtype, device=text_probs.device)

    return torch.cat(features, dim=-1)


def _per_class_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    values = []
    for class_idx in range(num_classes):
        pred_mask = preds == class_idx
        label_mask = labels == class_idx

        tp = torch.logical_and(pred_mask, label_mask).sum().item()
        fp = torch.logical_and(pred_mask, ~label_mask).sum().item()
        fn = torch.logical_and(~pred_mask, label_mask).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        values.append(f1)

    return torch.tensor(values, dtype=torch.float32)


def build_modality_priors_from_valid_artifact(valid_artifact: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = valid_artifact["label_id"].long()
    text_probs = valid_artifact["text_probs_cal"].float()
    speech_probs = valid_artifact["speech_probs_cal"].float()

    num_classes = int(text_probs.shape[-1])

    text_preds = text_probs.argmax(dim=-1)
    speech_preds = speech_probs.argmax(dim=-1)

    text_f1 = _per_class_f1(text_preds, labels, num_classes=num_classes)
    speech_f1 = _per_class_f1(speech_preds, labels, num_classes=num_classes)

    denom = (text_f1 + speech_f1).clamp_min(1e-6)
    prior_text = text_f1 / denom
    prior_speech = speech_f1 / denom
    return prior_text, prior_speech