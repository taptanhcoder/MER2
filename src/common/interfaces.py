from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch # type: ignore


@dataclass
class ClassifierOutput:


    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor
    labels: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionBundle:

    ids: list[str]
    labels: list[int]
    preds: list[int]
    probs: list[list[float]]
    logits: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochResult:

    split: str
    loss: float
    metrics: dict[str, Any]
    predictions: PredictionBundle