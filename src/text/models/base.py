from __future__ import annotations

from typing import Any

import torch.nn as nn


class BaseTextClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def resolve_num_classes(config: dict[str, Any]) -> int:
        model_cfg = dict(config.get("model", {}))
        dataset_cfg = dict(config.get("dataset", {}))
        label_space_cfg = dict(config.get("label_space", {}))

        if model_cfg.get("num_classes") is not None:
            return int(model_cfg["num_classes"])

        if dataset_cfg.get("num_classes") is not None:
            return int(dataset_cfg["num_classes"])

        labels = label_space_cfg.get("labels")
        if labels is not None:
            return int(len(labels))

        raise KeyError(
            "Could not resolve num_classes for text model. "
            "Expected one of: model.num_classes, dataset.num_classes, "
            "or label_space.labels."
        )

    @staticmethod
    def resolve_model_config(config: dict[str, Any]) -> dict[str, Any]:
        if "model" not in config:
            raise KeyError("Full experiment config must contain a 'model' section.")
        return dict(config["model"])