from __future__ import annotations

from typing import Any

from transformers import AutoFeatureExtractor


def build_speech_processor(model_config: dict[str, Any]):
    pretrained_name = str(model_config.get("processor_name") or model_config["pretrained_name"])
    return AutoFeatureExtractor.from_pretrained(pretrained_name)